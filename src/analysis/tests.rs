//! Integration tests for the analysis module.

use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn test_full_analysis_pipeline() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Simulate a command that's about 2x faster than reference
    let cmd_times = vec![0.50, 0.52, 0.48, 0.51, 0.49, 0.50, 0.52, 0.48, 0.51, 0.49];
    let ref_times = vec![1.00, 1.02, 0.98, 1.01, 0.99, 1.00, 1.02, 0.98, 1.01, 0.99];

    // Compute log-ratios
    let log_ratios = compute_log_ratios(&cmd_times, &ref_times);
    assert_eq!(log_ratios.len(), 10);

    // Compute speedup
    let speedup = speedup_from_log_ratios(&log_ratios);
    assert!(
        speedup > 1.9 && speedup < 2.1,
        "Speedup should be ~2.0, got {}",
        speedup
    );

    // Compute CI
    let (ci_lower, ci_upper) = bootstrap_percentile_ci(&log_ratios, 0.95, 10000, &mut rng);

    // Convert to speedup space
    let speedup_ci_lower = (-ci_upper).exp();
    let speedup_ci_upper = (-ci_lower).exp();

    // CI should contain the speedup
    assert!(speedup_ci_lower <= speedup && speedup <= speedup_ci_upper);

    // With 1% delta, this should be "faster"
    let verdict = determine_verdict(speedup_ci_lower, speedup_ci_upper, 0.01);
    assert_eq!(verdict, Verdict::Faster);
}

#[test]
fn test_equivalent_commands() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Commands with very similar timing
    let cmd_times = vec![1.00, 1.01, 0.99, 1.00, 1.01, 0.99, 1.00, 1.01, 0.99, 1.00];
    let ref_times = vec![1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00];

    let log_ratios = compute_log_ratios(&cmd_times, &ref_times);
    let speedup = speedup_from_log_ratios(&log_ratios);

    // Speedup should be very close to 1.0
    assert!((speedup - 1.0).abs() < 0.02);

    let (ci_lower, ci_upper) = bootstrap_percentile_ci(&log_ratios, 0.95, 10000, &mut rng);
    let speedup_ci_lower = (-ci_upper).exp();
    let speedup_ci_upper = (-ci_lower).exp();

    // With 1% delta, this should be "no clear difference"
    let verdict = determine_verdict(speedup_ci_lower, speedup_ci_upper, 0.01);
    assert_eq!(verdict, Verdict::NoClearDifference);
}

#[test]
fn test_slower_command() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Command is 2x slower than reference
    let cmd_times = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
    let ref_times = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    let log_ratios = compute_log_ratios(&cmd_times, &ref_times);
    let speedup = speedup_from_log_ratios(&log_ratios);

    // Speedup should be 0.5 (half as fast)
    assert!((speedup - 0.5).abs() < 1e-10);

    let (ci_lower, ci_upper) = bootstrap_percentile_ci(&log_ratios, 0.95, 10000, &mut rng);
    let speedup_ci_lower = (-ci_upper).exp();
    let speedup_ci_upper = (-ci_lower).exp();

    // With 1% delta, this should be "slower"
    let verdict = determine_verdict(speedup_ci_lower, speedup_ci_upper, 0.01);
    assert_eq!(verdict, Verdict::Slower);
}
