//! Statistical analysis module for benchmark comparisons.
//!
//! This module provides:
//! - Bootstrap confidence intervals for speedup estimates
//! - Log-ratio based speedup calculation (geometric mean)
//! - Practical significance verdicts

mod bootstrap;
mod speedup;
mod verdict;

pub use bootstrap::bootstrap_percentile_ci;
pub use speedup::{compute_log_ratios, compute_speedup_with_ci, speedup_from_log_ratios};
pub use verdict::{determine_verdict, Verdict};

/// Result of statistical analysis comparing a command to the reference.
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Name of the command being analyzed
    pub command: String,

    /// Geometric mean speedup (>1 means faster than reference)
    pub speedup: f64,

    /// Lower bound of the confidence interval
    pub ci_lower: f64,

    /// Upper bound of the confidence interval
    pub ci_upper: f64,

    /// Confidence level used (e.g., 0.95)
    pub confidence: f64,

    /// Practical significance verdict
    pub verdict: Verdict,
}

#[cfg(test)]
mod tests;
