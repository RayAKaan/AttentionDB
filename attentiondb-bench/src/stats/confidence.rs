use statrs::distribution::{StudentsT, ContinuousCDF};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub mean: f64,
    pub std_dev: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub confidence_level: f64,
    pub n: usize,
}

impl ConfidenceInterval {
    pub fn from_sample(sample: &[f64], confidence_level: f64) -> Self {
        let n = sample.len();
        let mean = sample.iter().sum::<f64>() / n as f64;
        let variance = sample.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (n as f64 - 1.0);
        let std_dev = variance.sqrt();

        let alpha = 1.0 - confidence_level;
        let t_dist = StudentsT::new(0.0, 1.0, (n - 1) as f64)
            .expect("valid t-distribution");
        let t_critical = t_dist.inverse_cdf(1.0 - alpha / 2.0);

        let margin = t_critical * std_dev / (n as f64).sqrt();

        Self {
            mean,
            std_dev,
            ci_lower: mean - margin,
            ci_upper: mean + margin,
            confidence_level,
            n,
        }
    }

    pub fn margin_of_error(&self) -> f64 {
        (self.ci_upper - self.ci_lower) / 2.0
    }

    pub fn relative_error(&self) -> f64 {
        if self.mean.abs() < f64::EPSILON { return 1.0; }
        self.margin_of_error() / self.mean.abs()
    }
}

pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() { 0.0 } else { data.iter().sum::<f64>() / data.len() as f64 }
}

pub fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let m = mean(data);
    let variance = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    variance.sqrt()
}

pub fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() { return 0.0; }
    if sorted_data.len() == 1 { return sorted_data[0]; }

    let rank = p * (sorted_data.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;

    if lower == upper { return sorted_data[lower]; }

    let frac = rank - lower as f64;
    sorted_data[lower] * (1.0 - frac) + sorted_data[upper] * frac
}

pub fn median(sorted_data: &[f64]) -> f64 {
    percentile(sorted_data, 0.50)
}
