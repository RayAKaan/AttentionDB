use statrs::distribution::{StudentsT, ContinuousCDF};
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WelchTestResult {
    pub label: String,
    pub t_statistic: f64,
    pub degrees_of_freedom: f64,
    pub p_value_raw: f64,
    pub bh_adjusted_q: Option<f64>,
    pub significant_bh_fdr05: Option<bool>,
    pub cohens_d: f64,
    pub mean_a: f64,
    pub mean_b: f64,
    pub mean_difference: f64,
    pub n_a: usize,
    pub n_b: usize,
}

pub fn welch_t_test(
    label: impl Into<String>,
    sample_a: &[f64],
    sample_b: &[f64],
) -> Result<WelchTestResult> {
    anyhow::ensure!(sample_a.len() >= 2, "Need at least 2 samples in group A");
    anyhow::ensure!(sample_b.len() >= 2, "Need at least 2 samples in group B");

    let n_a = sample_a.len() as f64;
    let n_b = sample_b.len() as f64;

    let mean_a = sample_a.iter().sum::<f64>() / n_a;
    let mean_b = sample_b.iter().sum::<f64>() / n_b;

    let var_a = sample_a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
    let var_b = sample_b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);

    let se = (var_a / n_a + var_b / n_b).sqrt();
    anyhow::ensure!(se > f64::EPSILON, "Zero standard error");

    let t_stat = (mean_a - mean_b) / se;

    let dof_num = (var_a / n_a + var_b / n_b).powi(2);
    let dof_den = (var_a / n_a).powi(2) / (n_a - 1.0)
                + (var_b / n_b).powi(2) / (n_b - 1.0);
    let df = dof_num / dof_den;

    let t_dist = StudentsT::new(0.0, 1.0, df)
        .map_err(|e| anyhow::anyhow!("t-distribution init failed: {}", e))?;
    let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));

    let pooled_std = ((var_a * (n_a - 1.0) + var_b * (n_b - 1.0))
        / (n_a + n_b - 2.0))
        .sqrt();
    let cohens_d = if pooled_std > f64::EPSILON {
        (mean_a - mean_b).abs() / pooled_std
    } else {
        0.0
    };

    Ok(WelchTestResult {
        label: label.into(),
        t_statistic: t_stat,
        degrees_of_freedom: df,
        p_value_raw: p_value,
        bh_adjusted_q: None,
        significant_bh_fdr05: None,
        cohens_d,
        mean_a,
        mean_b,
        mean_difference: mean_a - mean_b,
        n_a: sample_a.len(),
        n_b: sample_b.len(),
    })
}

pub fn apply_benjamini_hochberg(
    results: &mut Vec<WelchTestResult>,
    fdr_level: f64,
) {
    let m = results.len();
    if m == 0 { return; }

    let mut indexed: Vec<(usize, f64)> = results
        .iter()
        .enumerate()
        .map(|(i, r)| (i, r.p_value_raw))
        .collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut q_values = vec![0.0f64; m];
    for (rank, &(orig_idx, p)) in indexed.iter().enumerate() {
        q_values[orig_idx] = (p * m as f64 / (rank + 1) as f64).min(1.0);
    }

    let mut min_q = 1.0f64;
    for &(orig_idx, _) in indexed.iter().rev() {
        min_q = min_q.min(q_values[orig_idx]);
        q_values[orig_idx] = min_q;
    }

    let critical_rank = indexed.iter().enumerate()
        .filter(|(rank, &(_, p))| p <= (*rank as f64 + 1.0) / m as f64 * fdr_level)
        .map(|(rank, _)| rank)
        .max();

    for (orig_idx, _) in &indexed {
        results[*orig_idx].bh_adjusted_q = Some(q_values[*orig_idx]);
    }

    for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
        results[orig_idx].significant_bh_fdr05 = Some(
            critical_rank.map_or(false, |cr| rank <= cr)
        );
    }
}

pub fn compare_system_to_all_competitors(
    target_name: &str,
    target_samples: &[f64],
    competitors: &[(&str, &[f64])],
) -> Result<Vec<WelchTestResult>> {
    let mut tests = Vec::new();

    for (comp_name, comp_samples) in competitors {
        let label = format!("{} vs {}", target_name, comp_name);
        let test = welch_t_test(label, target_samples, comp_samples)?;
        tests.push(test);
    }

    apply_benjamini_hochberg(&mut tests, 0.05);

    Ok(tests)
}
