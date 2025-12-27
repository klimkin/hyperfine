//! Bootstrap resampling for confidence interval estimation.

use rand::Rng;

/// Computes the percentile bootstrap confidence interval for a dataset.
///
/// Uses the percentile method: samples with replacement `resamples` times,
/// computes the mean of each resample, and returns the confidence interval
/// bounds from the empirical distribution of means.
///
/// # Arguments
///
/// * `data` - The input data (e.g., log-ratios)
/// * `confidence` - Confidence level (e.g., 0.95 for 95% CI)
/// * `resamples` - Number of bootstrap resamples
/// * `rng` - Random number generator
///
/// # Returns
///
/// A tuple `(lower, upper)` representing the confidence interval bounds.
pub fn bootstrap_percentile_ci<R: Rng>(
    data: &[f64],
    confidence: f64,
    resamples: usize,
    rng: &mut R,
) -> (f64, f64) {
    if data.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    if data.len() == 1 {
        return (data[0], data[0]);
    }

    let mut bootstrap_means = Vec::with_capacity(resamples);
    let n = data.len();

    for _ in 0..resamples {
        let mut sum = 0.0;
        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            sum += data[idx];
        }
        bootstrap_means.push(sum / n as f64);
    }

    // Sort to get percentiles
    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate percentile indices
    let alpha = 1.0 - confidence;
    let lower_idx = ((alpha / 2.0) * resamples as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * resamples as f64).ceil() as usize;

    // Clamp to valid indices
    let lower_idx = lower_idx.min(resamples.saturating_sub(1));
    let upper_idx = upper_idx.min(resamples.saturating_sub(1));

    (bootstrap_means[lower_idx], bootstrap_means[upper_idx])
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    #[test]
    fn test_bootstrap_ci_basic() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let (lower, upper) = bootstrap_percentile_ci(&data, 0.95, 10000, &mut rng);

        // Mean is 3.0, CI should be centered around it
        assert!(lower < 3.0);
        assert!(upper > 3.0);
        // CI should contain the mean
        assert!(lower < upper);
    }

    #[test]
    fn test_bootstrap_ci_single_value() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = vec![5.0];

        let (lower, upper) = bootstrap_percentile_ci(&data, 0.95, 1000, &mut rng);

        assert_eq!(lower, 5.0);
        assert_eq!(upper, 5.0);
    }

    #[test]
    fn test_bootstrap_ci_empty() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data: Vec<f64> = vec![];

        let (lower, upper) = bootstrap_percentile_ci(&data, 0.95, 1000, &mut rng);

        assert!(lower.is_nan());
        assert!(upper.is_nan());
    }

    #[test]
    fn test_bootstrap_ci_reproducible_with_seed() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let mut rng1 = ChaCha8Rng::seed_from_u64(123);
        let (lower1, upper1) = bootstrap_percentile_ci(&data, 0.95, 5000, &mut rng1);

        let mut rng2 = ChaCha8Rng::seed_from_u64(123);
        let (lower2, upper2) = bootstrap_percentile_ci(&data, 0.95, 5000, &mut rng2);

        assert_eq!(lower1, lower2);
        assert_eq!(upper1, upper2);
    }

    #[test]
    fn test_bootstrap_ci_wider_with_lower_confidence() {
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let (lower_99, upper_99) = bootstrap_percentile_ci(&data, 0.99, 10000, &mut rng1);
        let (lower_90, upper_90) = bootstrap_percentile_ci(&data, 0.90, 10000, &mut rng2);

        // 99% CI should be wider than 90% CI
        let width_99 = upper_99 - lower_99;
        let width_90 = upper_90 - lower_90;
        assert!(width_99 > width_90, "99% CI ({}) should be wider than 90% CI ({})", width_99, width_90);
    }
}
