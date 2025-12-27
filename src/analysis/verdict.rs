//! Practical significance verdict determination.
//!
//! Determines whether a speedup is practically significant based on
//! the confidence interval and a user-defined threshold (delta).

use serde::Serialize;

/// Verdict on practical significance of a speedup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Command is faster than reference (CI entirely above 1+delta)
    Faster,

    /// Command is slower than reference (CI entirely below 1-delta)
    Slower,

    /// No practical difference (CI overlaps the [1-delta, 1+delta] range)
    NoClearDifference,
}

impl Verdict {
    /// Returns a human-readable description of the verdict.
    pub fn description(&self) -> &'static str {
        match self {
            Verdict::Faster => "faster",
            Verdict::Slower => "slower",
            Verdict::NoClearDifference => "no clear difference",
        }
    }
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.description())
    }
}

/// Determines the practical significance verdict.
///
/// The verdict is determined by comparing the confidence interval
/// [ci_lower, ci_upper] against the practical equivalence zone [1-delta, 1+delta]:
///
/// - "faster": entire CI is above 1+delta (lower bound > 1+delta)
/// - "slower": entire CI is below 1-delta (upper bound < 1-delta)
/// - "no clear difference": CI overlaps [1-delta, 1+delta]
///
/// # Arguments
///
/// * `ci_lower` - Lower bound of the speedup confidence interval
/// * `ci_upper` - Upper bound of the speedup confidence interval
/// * `delta` - Practical significance threshold (e.g., 0.01 for 1%)
///
/// # Returns
///
/// The verdict indicating practical significance.
pub fn determine_verdict(ci_lower: f64, ci_upper: f64, delta: f64) -> Verdict {
    let upper_threshold = 1.0 + delta;
    let lower_threshold = 1.0 - delta;

    if ci_lower > upper_threshold {
        // Entire CI is above 1+delta -> definitely faster
        Verdict::Faster
    } else if ci_upper < lower_threshold {
        // Entire CI is below 1-delta -> definitely slower
        Verdict::Slower
    } else {
        // CI overlaps the equivalence zone -> no clear difference
        Verdict::NoClearDifference
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verdict_faster() {
        // CI is [1.20, 1.30], delta = 0.01
        // Lower bound 1.20 > 1.01 (1 + delta)
        let verdict = determine_verdict(1.20, 1.30, 0.01);
        assert_eq!(verdict, Verdict::Faster);
    }

    #[test]
    fn test_verdict_slower() {
        // CI is [0.70, 0.80], delta = 0.01
        // Upper bound 0.80 < 0.99 (1 - delta)
        let verdict = determine_verdict(0.70, 0.80, 0.01);
        assert_eq!(verdict, Verdict::Slower);
    }

    #[test]
    fn test_verdict_no_clear_difference_overlaps() {
        // CI is [0.95, 1.05], delta = 0.01
        // CI overlaps [0.99, 1.01]
        let verdict = determine_verdict(0.95, 1.05, 0.01);
        assert_eq!(verdict, Verdict::NoClearDifference);
    }

    #[test]
    fn test_verdict_no_clear_difference_inside() {
        // CI is [0.995, 1.005], entirely within [0.99, 1.01]
        let verdict = determine_verdict(0.995, 1.005, 0.01);
        assert_eq!(verdict, Verdict::NoClearDifference);
    }

    #[test]
    fn test_verdict_edge_case_at_threshold() {
        // CI lower is exactly at 1+delta
        // 1.01 is not > 1.01, so this should be NoClearDifference
        let verdict = determine_verdict(1.01, 1.10, 0.01);
        assert_eq!(verdict, Verdict::NoClearDifference);

        // CI lower is just above 1+delta
        let verdict = determine_verdict(1.011, 1.10, 0.01);
        assert_eq!(verdict, Verdict::Faster);
    }

    #[test]
    fn test_verdict_with_larger_delta() {
        // With delta = 0.10 (10%), we need larger differences
        // CI is [1.05, 1.15], delta = 0.10
        // Lower bound 1.05 is not > 1.10 (1 + delta)
        let verdict = determine_verdict(1.05, 1.15, 0.10);
        assert_eq!(verdict, Verdict::NoClearDifference);

        // CI is [1.15, 1.25]
        // Lower bound 1.15 > 1.10
        let verdict = determine_verdict(1.15, 1.25, 0.10);
        assert_eq!(verdict, Verdict::Faster);
    }

    #[test]
    fn test_verdict_display() {
        assert_eq!(format!("{}", Verdict::Faster), "faster");
        assert_eq!(format!("{}", Verdict::Slower), "slower");
        assert_eq!(format!("{}", Verdict::NoClearDifference), "no clear difference");
    }

    #[test]
    fn test_verdict_description() {
        assert_eq!(Verdict::Faster.description(), "faster");
        assert_eq!(Verdict::Slower.description(), "slower");
        assert_eq!(Verdict::NoClearDifference.description(), "no clear difference");
    }
}
