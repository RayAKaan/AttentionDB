use statrs::distribution::{Normal, ContinuousCDF};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnalysisResult {
    pub target_effect_size_d: f64,
    pub desired_power: f64,
    pub alpha: f64,
    pub required_n: usize,
    pub power_at_n30: f64,
    pub n30_sufficient: bool,
}

impl PowerAnalysisResult {
    pub fn compute(
        target_effect_size_d: f64,
        desired_power: f64,
        alpha: f64,
    ) -> Self {
        let required_n = minimum_n_for_power(target_effect_size_d, desired_power, alpha);
        let power_at_30 = achieved_power(target_effect_size_d, 30, alpha);

        Self {
            target_effect_size_d,
            desired_power,
            alpha,
            required_n,
            power_at_n30: power_at_30,
            n30_sufficient: required_n <= 30,
        }
    }
}

fn minimum_n_for_power(
    effect_size_d: f64,
    desired_power: f64,
    alpha: f64,
) -> usize {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let z_alpha_2 = normal.inverse_cdf(1.0 - alpha / 2.0);
    let z_beta = normal.inverse_cdf(desired_power);

    if effect_size_d.abs() < f64::EPSILON { return 10000; }

    let n = 2.0 * ((z_alpha_2 + z_beta) / effect_size_d).powi(2);
    (n.ceil() as usize).max(2)
}

fn achieved_power(effect_size_d: f64, n: usize, alpha: f64) -> f64 {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let z_alpha_2 = normal.inverse_cdf(1.0 - alpha / 2.0);
    let ncp = effect_size_d * (n as f64 / 2.0).sqrt();

    let power = 1.0 - normal.cdf(z_alpha_2 - ncp) + normal.cdf(-z_alpha_2 - ncp);
    power.clamp(0.0, 1.0)
}

pub fn validate_n_for_benchmark(
    effect_sizes_to_detect: &[f64],
    num_tests_per_config: usize,
    _n_per_group: usize,
) -> Vec<PowerAnalysisResult> {
    let adjusted_alpha = 0.05 / num_tests_per_config as f64;

    effect_sizes_to_detect.iter().map(|&d| {
        PowerAnalysisResult::compute(d, 0.80, adjusted_alpha)
    }).collect()
}

pub fn generate_power_report(
    results: &[PowerAnalysisResult],
    actual_n: usize,
) -> String {
    let mut report = String::new();
    report.push_str("Statistical Power Analysis\n");
    report.push_str("=========================\n\n");

    for r in results {
        report.push_str(&format!(
            "  d={:.1}: required N={} (N={}{}), power@N30={:.2}\n",
            r.target_effect_size_d,
            r.required_n,
            actual_n,
            if r.n30_sufficient { " ✓" } else { " ✗" },
            r.power_at_n30,
        ));
    }

    report
}
