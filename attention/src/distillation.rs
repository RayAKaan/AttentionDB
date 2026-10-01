//! C8 candidate-set softmax distillation.
//!
//! The residual correction is a *ranking correction*: `S_final = S_base +
//! lambda * dS_attention` must preserve the baseline's relative preference
//! structure across the candidate set while adding useful signal. Min-max
//! normalization is deliberately NOT used: it destroys the relative
//! distribution (e.g. `[0.91, 0.90, 0.40]` and `[0.91, 0.50, 0.40]` collapse to
//! the same normalized shape) even though they induce very different ranking
//! distributions.
//!
//! Instead both score vectors are turned into probability distributions over
//! the candidate set with a shared temperature and compared with KL divergence:
//!
//! ```text
//! T_i = exp(S_base_i / tau_d) / sum_j exp(S_base_j / tau_d)
//! P_i = exp(S_final_i / tau_d) / sum_j exp(S_final_j / tau_d)
//! L_distill = KL(T || P) = sum_i T_i * (ln T_i - ln P_i)
//! ```
//!
//! KL is asymmetric on purpose: the baseline is the *teacher* and must be
//! preserved, so mass that the student places where the teacher has none is
//! penalized (`T_i -> 0`, `P_i > 0` gives no forward term, but a positive
//! `P_i` where `T_i > 0` is punished through the teacher's expectation). The
//! loss is minimized (never negative) and equals zero iff the two distributions
//! coincide.
//!
//! Everything is deterministic: fixed loop order, max-shift stabilization, no
//! RNG, no parallelism.

use crate::errors::{AttentionError, Result};

/// Numerically stable softmax over an arbitrary score vector, divided by
/// `temperature`. Returns a probability vector summing to 1.
///
/// An empty input returns an empty vector. A non-finite temperature is rejected
/// because it would silently destroy the distribution.
pub fn candidate_softmax(scores: &[f32], temperature: f32) -> Result<Vec<f32>> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(AttentionError::Config(
            "distillation temperature must be finite and > 0".into(),
        ));
    }
    if scores.is_empty() {
        return Ok(Vec::new());
    }
    let mut max_score = f32::NEG_INFINITY;
    for &s in scores {
        if s.is_finite() && s > max_score {
            max_score = s;
        }
    }
    if !max_score.is_finite() {
        return Err(AttentionError::NonFinite(
            "distillation scores are all non-finite".into(),
        ));
    }
    let inv_t = 1.0 / temperature;
    let mut probs = vec![0.0f32; scores.len()];
    let mut sum = 0.0f32;
    for (i, &s) in scores.iter().enumerate() {
        let e = if s.is_finite() {
            (s * inv_t - max_score * inv_t).exp()
        } else {
            0.0
        };
        probs[i] = e;
        sum += e;
    }
    if sum <= 0.0 || !sum.is_finite() {
        // Degenerate spread: fall back to uniform rather than propagating NaN.
        let uniform = 1.0 / scores.len() as f32;
        for p in probs.iter_mut() {
            *p = uniform;
        }
        return Ok(probs);
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }
    Ok(probs)
}

/// Kullback-Leibler divergence `KL(target || predicted)`.
///
/// Returns a non-negative value; `0.0` iff the two distributions are equal.
/// Terms where `target_i == 0` contribute nothing (0 * log 0 := 0), which is
/// what makes the teacher-only direction well defined.
pub fn kl_divergence(target: &[f32], predicted: &[f32]) -> Result<f32> {
    if target.len() != predicted.len() {
        return Err(AttentionError::DimensionMismatch {
            expected: target.len(),
            found: predicted.len(),
        });
    }
    if target.is_empty() {
        return Ok(0.0);
    }
    let mut kl = 0.0f32;
    for i in 0..target.len() {
        let t = target[i];
        let p = predicted[i];
        if !t.is_finite() || !p.is_finite() {
            return Err(AttentionError::NonFinite(
                "KL received a non-finite probability".into(),
            ));
        }
        if t <= 0.0 {
            continue;
        }
        if p <= 0.0 {
            // Teacher assigns mass where the student assigns none: infinite
            // penalty. Clamped to a large finite value to keep training alive.
            return Ok(f32::MAX / 4.0);
        }
        kl += t * (t.ln() - p.ln());
    }
    if kl < 0.0 {
        // Only reachable through floating-point cancellation; KL is a divergence.
        return Ok(0.0);
    }
    Ok(kl)
}

/// C8 distillation loss for one query's candidate set.
///
/// `baseline_scores` is the frozen teacher (C7-B `S_base`), `final_scores` the
/// student (`S_final = S_base + lambda * dS`). Returns `0.0` for an empty
/// candidate set and for a single candidate (a one-element distribution is
/// uniform under any temperature, so there is nothing to distill).
pub fn distillation_loss(
    baseline_scores: &[f32],
    final_scores: &[f32],
    temperature: f32,
) -> Result<f32> {
    if baseline_scores.len() != final_scores.len() {
        return Err(AttentionError::DimensionMismatch {
            expected: baseline_scores.len(),
            found: final_scores.len(),
        });
    }
    if baseline_scores.len() <= 1 {
        return Ok(0.0);
    }
    let t = candidate_softmax(baseline_scores, temperature)?;
    let p = candidate_softmax(final_scores, temperature)?;
    kl_divergence(&t, &p)
}

/// Gradient of [`distillation_loss`] with respect to `final_scores`:
/// `dL/ds_i = (P_i - T_i) / (tau_d * sum_j P_j)` — uniform because the softmax
/// normalizer is constant w.r.t. the logit shift.
///
/// Returned in the same order as `final_scores`. The caller maps this onto the
/// attention weights: `S_final` depends on the QKV weights only through the
/// residual delta `lambda * dS_attention`.
pub fn distillation_gradient(
    baseline_scores: &[f32],
    final_scores: &[f32],
    temperature: f32,
) -> Result<Vec<f32>> {
    if baseline_scores.len() != final_scores.len() {
        return Err(AttentionError::DimensionMismatch {
            expected: baseline_scores.len(),
            found: final_scores.len(),
        });
    }
    if baseline_scores.len() <= 1 {
        return Ok(vec![0.0; final_scores.len()]);
    }
    let t = candidate_softmax(baseline_scores, temperature)?;
    let p = candidate_softmax(final_scores, temperature)?;
    let inv_t = 1.0 / temperature;
    let mut grad = Vec::with_capacity(final_scores.len());
    for i in 0..final_scores.len() {
        grad.push((p[i] - t[i]) * inv_t);
    }
    Ok(grad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_is_a_distribution() {
        let p = candidate_softmax(&[1.0, 2.0, 3.0], 1.0).unwrap();
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
        assert!(p[0] < p[1] && p[1] < p[2]);
    }

    #[test]
    fn softmax_is_shift_invariant_and_stable() {
        let a = candidate_softmax(&[1.0, 2.0, 3.0], 1.0).unwrap();
        let b = candidate_softmax(&[101.0, 102.0, 103.0], 1.0).unwrap();
        for i in 0..a.len() {
            assert!((a[i] - b[i]).abs() < 1e-5, "{i}: {} vs {}", a[i], b[i]);
        }
    }

    #[test]
    fn rejects_bad_temperature() {
        assert!(candidate_softmax(&[1.0, 2.0], 0.0).is_err());
        assert!(candidate_softmax(&[1.0, 2.0], -1.0).is_err());
        assert!(candidate_softmax(&[1.0, 2.0], f32::NAN).is_err());
    }

    #[test]
    fn kl_is_zero_for_identical_distributions_and_nonnegative() {
        let s = vec![0.9, 0.5, 0.4, 0.1];
        let t = candidate_softmax(&s, 0.7).unwrap();
        assert!(kl_divergence(&t, &t).unwrap().abs() < 1e-6);
        let other = candidate_softmax(&[0.1, 0.5, 0.9, 0.4], 0.7).unwrap();
        assert!(kl_divergence(&t, &other).unwrap() > 0.0);
    }

    #[test]
    fn kl_matches_the_explicit_formula_over_the_teacher_support() {
        let target: Vec<f32> = vec![0.5, 0.5, 0.0];
        let predicted: Vec<f32> = vec![0.4, 0.4, 0.2];
        // Only the teacher's support enters; index 2 has T = 0 and contributes
        // 0 * log(0.2) := 0 rather than NaN.
        let manual: f32 = (0..target.len())
            .filter(|&i| target[i] > 0.0)
            .map(|i| target[i] * (target[i].ln() - predicted[i].ln()))
            .sum();
        let got = kl_divergence(&target, &predicted).unwrap();
        assert!((got - manual).abs() < 1e-6, "{got} vs {manual}");
        assert!(got > 0.0);
    }

    #[test]
    fn kl_penalizes_dropping_teacher_mass_without_returning_inf() {
        let target = vec![1.0, 0.0];
        let predicted = vec![0.0, 1.0];
        let kl = kl_divergence(&target, &predicted).unwrap();
        assert!(kl.is_finite(), "must stay finite to keep training alive");
        assert!(kl > 1e6, "near-maximal penalty, got {kl}");
    }

    #[test]
    fn identical_scores_give_zero_distillation_loss() {
        let s = vec![0.91, 0.90, 0.40];
        assert!(distillation_loss(&s, &s, 0.5).unwrap().abs() < 1e-6);
    }

    #[test]
    fn uniform_scaling_of_scores_is_not_distilled_away() {
        // The reason C8 does NOT use min-max: these two candidate sets are
        // equivalent after min-max but very different distributions.
        let base = vec![0.91, 0.90, 0.40];
        let flat = vec![0.80, 0.79, 0.20];
        let spread = vec![0.80, 0.39, 0.20];
        let tau = 0.07;
        let l_flat = distillation_loss(&base, &flat, tau).unwrap();
        let l_spread = distillation_loss(&base, &spread, tau).unwrap();
        assert!(
            l_spread > l_flat,
            "spread={l_spread} should exceed flat={l_flat}"
        );
    }

    #[test]
    fn distillation_gradient_matches_finite_difference() {
        let base = vec![0.91, 0.50, 0.40, 0.10];
        let fin = vec![0.80, 0.39, 0.44, 0.12];
        let tau = 0.5;
        let g = distillation_gradient(&base, &fin, tau).unwrap();
        let eps = 1e-3f32;
        for i in 0..fin.len() {
            let mut up = fin.clone();
            let mut down = fin.clone();
            up[i] += eps;
            down[i] -= eps;
            let num = (distillation_loss(&base, &up, tau).unwrap()
                - distillation_loss(&base, &down, tau).unwrap())
                / (2.0 * eps);
            assert!(
                (g[i] - num).abs() < 1e-2 * (1.0 + g[i].abs()),
                "grad[{i}] analytic={} numeric={num}",
                g[i]
            );
        }
    }

    #[test]
    fn single_and_empty_candidates_are_free() {
        assert_eq!(distillation_loss(&[], &[], 1.0).unwrap(), 0.0);
        assert_eq!(distillation_loss(&[0.5], &[0.9], 1.0).unwrap(), 0.0);
    }
}
