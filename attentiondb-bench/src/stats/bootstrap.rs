use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapCI {
    pub point_estimate: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub confidence_level: f64,
    pub num_resamples: usize,
    pub std_error: f64,
}

pub fn bootstrap_confidence_interval(
    sample: &[f64],
    num_resamples: usize,
    confidence_level: f64,
    rng: &mut StdRng,
) -> BootstrapCI {
    let n = sample.len();
    let point_estimate = sample.iter().sum::<f64>() / n as f64;

    let mut resampled_means = Vec::with_capacity(num_resamples);

    for _ in 0..num_resamples {
        let resample_mean: f64 = (0..n)
            .map(|_| *sample.choose(rng).unwrap())
            .sum::<f64>() / n as f64;
        resampled_means.push(resample_mean);
    }

    resampled_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = 1.0 - confidence_level;
    let lower_idx = ((num_resamples as f64) * (alpha / 2.0)).round() as usize;
    let upper_idx = ((num_resamples as f64) * (1.0 - alpha / 2.0)).round() as usize;

    let ci_lower = resampled_means[lower_idx.min(num_resamples - 1)];
    let ci_upper = resampled_means[upper_idx.min(num_resamples - 1)];

    let mean_of_means: f64 = resampled_means.iter().sum::<f64>() / num_resamples as f64;
    let std_error: f64 = (resampled_means.iter()
        .map(|x| (x - mean_of_means).powi(2))
        .sum::<f64>() / (num_resamples - 1) as f64)
        .sqrt();

    BootstrapCI {
        point_estimate,
        ci_lower,
        ci_upper,
        confidence_level,
        num_resamples,
        std_error,
    }
}

pub fn bootstrap_difference(
    sample_a: &[f64],
    sample_b: &[f64],
    num_resamples: usize,
    confidence_level: f64,
    rng: &mut StdRng,
) -> BootstrapCI {
    let mean_a = sample_a.iter().sum::<f64>() / sample_a.len() as f64;
    let mean_b = sample_b.iter().sum::<f64>() / sample_b.len() as f64;
    let point_estimate = mean_a - mean_b;

    let n_a = sample_a.len();
    let n_b = sample_b.len();
    let mut resampled_diffs = Vec::with_capacity(num_resamples);

    for _ in 0..num_resamples {
        let boot_mean_a: f64 = (0..n_a)
            .map(|_| *sample_a.choose(rng).unwrap())
            .sum::<f64>() / n_a as f64;
        let boot_mean_b: f64 = (0..n_b)
            .map(|_| *sample_b.choose(rng).unwrap())
            .sum::<f64>() / n_b as f64;
        resampled_diffs.push(boot_mean_a - boot_mean_b);
    }

    resampled_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = 1.0 - confidence_level;
    let lower_idx = ((num_resamples as f64) * (alpha / 2.0)).round() as usize;
    let upper_idx = ((num_resamples as f64) * (1.0 - alpha / 2.0)).round() as usize;

    let ci_lower = resampled_diffs[lower_idx.min(num_resamples - 1)];
    let ci_upper = resampled_diffs[upper_idx.min(num_resamples - 1)];

    let mean_of_diffs: f64 = resampled_diffs.iter().sum::<f64>() / num_resamples as f64;
    let std_error: f64 = (resampled_diffs.iter()
        .map(|x| (x - mean_of_diffs).powi(2))
        .sum::<f64>() / (num_resamples - 1) as f64)
        .sqrt();

    BootstrapCI {
        point_estimate,
        ci_lower,
        ci_upper,
        confidence_level,
        num_resamples,
        std_error,
    }
}
