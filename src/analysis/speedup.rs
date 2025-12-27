//! Log-ratio based speedup calculation.
//!
//! Uses the log-ratio method for computing geometric mean speedup:
//! 1. Compute log-ratios: r_i = log(t_cmd_i) - log(t_ref_i)
//! 2. Speedup = exp(-mean(r_i))
//!
//! This approach is preferred because:
//! - Geometric mean is appropriate for ratios
//! - Log-space analysis handles multiplicative effects naturally
//! - Paired differences cancel common factors (drift, load)

/// Computes the log-ratios between paired timing samples.
///
/// For each pair (cmd_time, ref_time), computes log(cmd_time) - log(ref_time).
/// The result can be used with bootstrap CI to get confidence intervals.
///
/// # Arguments
///
/// * `cmd_times` - Timing samples for the command being analyzed
/// * `ref_times` - Timing samples for the reference command
///
/// # Returns
///
/// A vector of log-ratios. If the vectors have different lengths,
/// only the overlapping portion is used.
///
/// # Panics
///
/// Panics if any timing value is non-positive (would produce invalid log).
pub fn compute_log_ratios(cmd_times: &[f64], ref_times: &[f64]) -> Vec<f64> {
    let n = cmd_times.len().min(ref_times.len());
    let mut log_ratios = Vec::with_capacity(n);

    for i in 0..n {
        let cmd_time = cmd_times[i];
        let ref_time = ref_times[i];

        debug_assert!(cmd_time > 0.0, "Command time must be positive");
        debug_assert!(ref_time > 0.0, "Reference time must be positive");

        log_ratios.push(cmd_time.ln() - ref_time.ln());
    }

    log_ratios
}

/// Computes the speedup from log-ratios.
///
/// Speedup = exp(-mean(log_ratios))
///
/// A speedup > 1.0 means the command is faster than the reference.
/// A speedup < 1.0 means the command is slower than the reference.
///
/// # Arguments
///
/// * `log_ratios` - The log-ratios computed by `compute_log_ratios`
///
/// # Returns
///
/// The geometric mean speedup. Returns NaN if log_ratios is empty.
pub fn speedup_from_log_ratios(log_ratios: &[f64]) -> f64 {
    if log_ratios.is_empty() {
        return f64::NAN;
    }

    let mean = log_ratios.iter().sum::<f64>() / log_ratios.len() as f64;
    (-mean).exp()
}

/// Computes the speedup with confidence interval from timing samples.
///
/// Combines `compute_log_ratios`, `speedup_from_log_ratios`, and bootstrap CI
/// to produce a speedup estimate with confidence bounds.
///
/// # Arguments
///
/// * `cmd_times` - Timing samples for the command being analyzed
/// * `ref_times` - Timing samples for the reference command
/// * `confidence` - Confidence level (e.g., 0.95)
/// * `resamples` - Number of bootstrap resamples
/// * `rng` - Random number generator
///
/// # Returns
///
/// A tuple `(speedup, ci_lower, ci_upper)` where ci_lower and ci_upper
/// are the confidence interval bounds (already converted from log-space).
pub fn compute_speedup_with_ci<R: rand::Rng>(
    cmd_times: &[f64],
    ref_times: &[f64],
    confidence: f64,
    resamples: usize,
    rng: &mut R,
) -> (f64, f64, f64) {
    let log_ratios = compute_log_ratios(cmd_times, ref_times);
    let speedup = speedup_from_log_ratios(&log_ratios);

    let (ci_lower_log, ci_upper_log) =
        super::bootstrap::bootstrap_percentile_ci(&log_ratios, confidence, resamples, rng);

    // Convert CI bounds from log-space to speedup
    // Note: because of the negation, lower log-ratio = higher speedup
    let ci_lower = (-ci_upper_log).exp();
    let ci_upper = (-ci_lower_log).exp();

    (speedup, ci_lower, ci_upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_compute_log_ratios_equal_times() {
        let cmd_times = vec![1.0, 2.0, 3.0];
        let ref_times = vec![1.0, 2.0, 3.0];

        let log_ratios = compute_log_ratios(&cmd_times, &ref_times);

        assert_eq!(log_ratios.len(), 3);
        for ratio in log_ratios {
            assert!((ratio - 0.0).abs() < 1e-10, "Expected 0.0, got {}", ratio);
        }
    }

    #[test]
    fn test_compute_log_ratios_cmd_faster() {
        // Command takes half the time of reference
        let cmd_times = vec![0.5, 0.5, 0.5];
        let ref_times = vec![1.0, 1.0, 1.0];

        let log_ratios = compute_log_ratios(&cmd_times, &ref_times);

        // log(0.5) - log(1.0) = -ln(2) ≈ -0.693
        for ratio in log_ratios {
            assert!((ratio - (-std::f64::consts::LN_2)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_compute_log_ratios_cmd_slower() {
        // Command takes double the time of reference
        let cmd_times = vec![2.0, 2.0, 2.0];
        let ref_times = vec![1.0, 1.0, 1.0];

        let log_ratios = compute_log_ratios(&cmd_times, &ref_times);

        // log(2) - log(1) = ln(2) ≈ 0.693
        for ratio in log_ratios {
            assert!((ratio - std::f64::consts::LN_2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_compute_log_ratios_different_lengths() {
        let cmd_times = vec![1.0, 2.0, 3.0, 4.0];
        let ref_times = vec![1.0, 2.0];

        let log_ratios = compute_log_ratios(&cmd_times, &ref_times);

        assert_eq!(log_ratios.len(), 2);
    }

    #[test]
    fn test_speedup_from_log_ratios_equal() {
        let log_ratios = vec![0.0, 0.0, 0.0];
        let speedup = speedup_from_log_ratios(&log_ratios);
        assert!((speedup - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_speedup_from_log_ratios_faster() {
        // Negative log-ratio means cmd is faster
        let log_ratios = vec![-std::f64::consts::LN_2; 3];
        let speedup = speedup_from_log_ratios(&log_ratios);
        // exp(-(-ln(2))) = exp(ln(2)) = 2.0
        assert!((speedup - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_speedup_from_log_ratios_slower() {
        // Positive log-ratio means cmd is slower
        let log_ratios = vec![std::f64::consts::LN_2; 3];
        let speedup = speedup_from_log_ratios(&log_ratios);
        // exp(-ln(2)) = 1/2 = 0.5
        assert!((speedup - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_speedup_from_log_ratios_empty() {
        let log_ratios: Vec<f64> = vec![];
        let speedup = speedup_from_log_ratios(&log_ratios);
        assert!(speedup.is_nan());
    }

    #[test]
    fn test_compute_speedup_with_ci() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Command is 2x faster
        let cmd_times = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let ref_times = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let (speedup, ci_lower, ci_upper) =
            compute_speedup_with_ci(&cmd_times, &ref_times, 0.95, 10000, &mut rng);

        // Speedup should be 2.0
        assert!((speedup - 2.0).abs() < 1e-10);
        // With constant data, CI should be exactly 2.0 (no variance)
        assert!((ci_lower - 2.0).abs() < 1e-10);
        assert!((ci_upper - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_speedup_with_ci_noisy() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Add some noise
        let cmd_times = vec![0.45, 0.50, 0.55, 0.48, 0.52, 0.47, 0.53, 0.49, 0.51, 0.50];
        let ref_times = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let (speedup, ci_lower, ci_upper) =
            compute_speedup_with_ci(&cmd_times, &ref_times, 0.95, 10000, &mut rng);

        // Speedup should be around 2.0
        assert!(speedup > 1.8 && speedup < 2.2);
        // CI should be wider due to noise
        assert!(ci_lower < ci_upper);
        // CI should contain the point estimate
        assert!(ci_lower <= speedup && speedup <= ci_upper);
    }
}
