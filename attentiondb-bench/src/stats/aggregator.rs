use crate::metrics::quality::QualityMetrics;
use crate::stats::confidence::ConfidenceInterval;
use crate::stats::bootstrap::bootstrap_confidence_interval;
use crate::stats::hypothesis::{WelchTestResult, welch_t_test, apply_benjamini_hochberg};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetric {
    pub mean: f64,
    pub std_dev: f64,
    pub ci_lower_95: f64,
    pub ci_upper_95: f64,
    pub bootstrap_ci_lower: f64,
    pub bootstrap_ci_upper: f64,
    pub n: usize,
}

impl AggregatedMetric {
    pub fn from_samples(
        samples: &[f64],
        confidence_level: f64,
        bootstrap_resamples: usize,
        rng: &mut StdRng,
    ) -> Self {
        let ci = ConfidenceInterval::from_sample(samples, confidence_level);
        let boot = bootstrap_confidence_interval(samples, bootstrap_resamples, confidence_level, rng);

        Self {
            mean: ci.mean,
            std_dev: ci.std_dev,
            ci_lower_95: ci.ci_lower,
            ci_upper_95: ci.ci_upper,
            bootstrap_ci_lower: boot.ci_lower,
            bootstrap_ci_upper: boot.ci_upper,
            n: ci.n,
        }
    }

    pub fn margin_of_error_95(&self) -> f64 {
        (self.ci_upper_95 - self.ci_lower_95) / 2.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedQualityMetrics {
    pub recall_at_1: AggregatedMetric,
    pub recall_at_10: AggregatedMetric,
    pub recall_at_100: AggregatedMetric,
    pub mrr: AggregatedMetric,
    pub ndcg_at_10: AggregatedMetric,
    pub ndcg_at_100: AggregatedMetric,
    pub precision_at_10: AggregatedMetric,
}

pub fn aggregate_quality_metrics(
    run_metrics: &[QualityMetrics],
    confidence_level: f64,
    bootstrap_resamples: usize,
    rng: &mut StdRng,
) -> AggregatedQualityMetrics {
    let extract = |f: fn(&QualityMetrics) -> f64| -> Vec<f64> {
        run_metrics.iter().map(f).collect()
    };

    AggregatedQualityMetrics {
        recall_at_1: AggregatedMetric::from_samples(&extract(|m| m.recall_at_1), confidence_level, bootstrap_resamples, rng),
        recall_at_10: AggregatedMetric::from_samples(&extract(|m| m.recall_at_10), confidence_level, bootstrap_resamples, rng),
        recall_at_100: AggregatedMetric::from_samples(&extract(|m| m.recall_at_100), confidence_level, bootstrap_resamples, rng),
        mrr: AggregatedMetric::from_samples(&extract(|m| m.mrr), confidence_level, bootstrap_resamples, rng),
        ndcg_at_10: AggregatedMetric::from_samples(&extract(|m| m.ndcg_at_10), confidence_level, bootstrap_resamples, rng),
        ndcg_at_100: AggregatedMetric::from_samples(&extract(|m| m.ndcg_at_100), confidence_level, bootstrap_resamples, rng),
        precision_at_10: AggregatedMetric::from_samples(&extract(|m| m.precision_at_10), confidence_level, bootstrap_resamples, rng),
    }
}

pub fn compute_statistical_comparisons(
    target_label: &str,
    target_samples: &[f64],
    competitor_labels: &[&str],
    competitor_samples: &[&[f64]],
) -> Vec<WelchTestResult> {
    let competitors: Vec<(&str, &[f64])> = competitor_labels.iter()
        .zip(competitor_samples.iter())
        .map(|(&l, &s)| (l, s))
        .collect();

    let mut results = Vec::new();
    for (comp_name, comp_samples) in &competitors {
        if let Ok(test) = welch_t_test(
            format!("{} vs {}: NDCG@10", target_label, comp_name),
            target_samples,
            comp_samples,
        ) {
            results.push(test);
        }
    }

    apply_benjamini_hochberg(&mut results, 0.05);
    results
}
