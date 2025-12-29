pub mod benchmark_result;
pub mod executor;
pub mod relative_speed;
pub mod scheduler;
pub mod timing_result;

use std::cmp;

use crate::benchmark::executor::BenchmarkIteration;
use crate::command::Command;
use crate::options::{
    CmdFailureAction, CommandOutputPolicy, ExecutorKind, Options, OutputStyleOption,
};
use crate::outlier_detection::{modified_zscores, OUTLIER_THRESHOLD};
use crate::output::format::{format_duration, format_duration_unit};
use crate::output::progress_bar::get_progress_bar;
use crate::output::warnings::{OutlierWarningOptions, Warnings};
use crate::parameter::ParameterNameAndValue;
use crate::util::exit_code::extract_exit_code;
use crate::util::min_max::{max, min};
use crate::util::units::Second;
use benchmark_result::BenchmarkResult;
use timing_result::TimingResult;

use anyhow::{anyhow, Result};
use colored::*;
use statistical::{mean, median, standard_deviation};

use self::executor::Executor;

/// Accumulator for collecting benchmark timing data during execution
pub struct CommandAccumulator {
    pub times_real: Vec<Second>,
    pub times_user: Vec<Second>,
    pub times_system: Vec<Second>,
    pub memory_usage_byte: Vec<u64>,
    pub exit_codes: Vec<Option<i32>>,
    pub all_succeeded: bool,
}

impl Default for CommandAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandAccumulator {
    pub fn new() -> Self {
        CommandAccumulator {
            times_real: vec![],
            times_user: vec![],
            times_system: vec![],
            memory_usage_byte: vec![],
            exit_codes: vec![],
            all_succeeded: true,
        }
    }

    /// Add a timing result to the accumulator
    pub fn add_result(&mut self, res: &TimingResult, exit_code: Option<i32>, success: bool) {
        self.times_real.push(res.time_real);
        self.times_user.push(res.time_user);
        self.times_system.push(res.time_system);
        self.memory_usage_byte.push(res.memory_usage_byte);
        self.exit_codes.push(exit_code);
        self.all_succeeded = self.all_succeeded && success;
    }

    /// Build a BenchmarkResult from accumulated data
    pub fn build_result(&self, command: &Command<'_>) -> BenchmarkResult {
        let stats =
            BenchmarkStatistics::from_times(&self.times_real, &self.times_user, &self.times_system);

        BenchmarkResult {
            command: command.get_name(),
            command_with_unused_parameters: command.get_name_with_unused_parameters(),
            mean: stats.mean,
            stddev: stats.stddev,
            median: stats.median,
            user: stats.user_mean,
            system: stats.system_mean,
            min: stats.min,
            max: stats.max,
            times: Some(self.times_real.clone()),
            memory_usage_byte: Some(self.memory_usage_byte.clone()),
            exit_codes: self.exit_codes.clone(),
            parameters: command
                .get_parameters()
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect(),
        }
    }

    /// Get statistics from accumulated times
    pub fn get_statistics(&self) -> BenchmarkStatistics {
        BenchmarkStatistics::from_times(&self.times_real, &self.times_user, &self.times_system)
    }

    /// Get number of runs
    pub fn run_count(&self) -> usize {
        self.times_real.len()
    }
}

/// Threshold for warning about fast execution time
pub const MIN_EXECUTION_TIME: Second = 5e-3;

/// Run an intermediate command (setup, cleanup, prepare, or conclude)
pub fn run_intermediate_command(
    executor: &dyn Executor,
    command: &Command<'_>,
    error_output: &'static str,
    output_policy: &CommandOutputPolicy,
) -> Result<TimingResult> {
    executor
        .run_command_and_measure(
            command,
            BenchmarkIteration::NonBenchmarkRun,
            Some(CmdFailureAction::RaiseError),
            output_policy,
        )
        .map(|r| r.0)
        .map_err(|_| anyhow!(error_output))
}

/// Build a preparation command for a benchmark
pub fn build_preparation_command<'a>(
    options: &'a Options,
    number: usize,
    parameters: impl Iterator<Item = ParameterNameAndValue<'a>>,
) -> Option<Command<'a>> {
    options.preparation_command.as_ref().map(|values| {
        let preparation_command = if values.len() == 1 {
            &values[0]
        } else {
            &values[number]
        };
        Command::new_parametrized(None, preparation_command, parameters)
    })
}

/// Build a conclusion command for a benchmark
pub fn build_conclusion_command<'a>(
    options: &'a Options,
    number: usize,
    parameters: impl Iterator<Item = ParameterNameAndValue<'a>>,
) -> Option<Command<'a>> {
    options.conclusion_command.as_ref().map(|values| {
        let conclusion_command = if values.len() == 1 {
            &values[0]
        } else {
            &values[number]
        };
        Command::new_parametrized(None, conclusion_command, parameters)
    })
}

/// Run a preparation command if present
pub fn run_preparation_command_optional(
    executor: &dyn Executor,
    command: Option<&Command<'_>>,
    output_policy: &CommandOutputPolicy,
) -> Result<Option<TimingResult>> {
    command
        .map(|cmd| {
            let error_output = "The preparation command terminated with a non-zero exit code. \
                                Append ' || true' to the command if you are sure that this can be ignored.";
            run_intermediate_command(executor, cmd, error_output, output_policy)
        })
        .transpose()
}

/// Run a conclusion command if present
pub fn run_conclusion_command_optional(
    executor: &dyn Executor,
    command: Option<&Command<'_>>,
    output_policy: &CommandOutputPolicy,
) -> Result<Option<TimingResult>> {
    command
        .map(|cmd| {
            let error_output = "The conclusion command terminated with a non-zero exit code. \
                                Append ' || true' to the command if you are sure that this can be ignored.";
            run_intermediate_command(executor, cmd, error_output, output_policy)
        })
        .transpose()
}

/// Run setup command for a benchmark
pub fn run_setup_command<'a>(
    executor: &dyn Executor,
    options: &'a Options,
    parameters: impl IntoIterator<Item = ParameterNameAndValue<'a>>,
    output_policy: &CommandOutputPolicy,
) -> Result<TimingResult> {
    let command = options
        .setup_command
        .as_ref()
        .map(|setup_command| Command::new_parametrized(None, setup_command, parameters));

    let error_output = "The setup command terminated with a non-zero exit code. \
                        Append ' || true' to the command if you are sure that this can be ignored.";

    Ok(command
        .map(|cmd| run_intermediate_command(executor, &cmd, error_output, output_policy))
        .transpose()?
        .unwrap_or_default())
}

/// Run cleanup command for a benchmark
pub fn run_cleanup_command<'a>(
    executor: &dyn Executor,
    options: &'a Options,
    parameters: impl IntoIterator<Item = ParameterNameAndValue<'a>>,
    output_policy: &CommandOutputPolicy,
) -> Result<TimingResult> {
    let command = options
        .cleanup_command
        .as_ref()
        .map(|cleanup_command| Command::new_parametrized(None, cleanup_command, parameters));

    let error_output = "The cleanup command terminated with a non-zero exit code. \
                        Append ' || true' to the command if you are sure that this can be ignored.";

    Ok(command
        .map(|cmd| run_intermediate_command(executor, &cmd, error_output, output_policy))
        .transpose()?
        .unwrap_or_default())
}

/// Calculate the number of benchmark runs based on timing
pub fn calculate_run_count(
    options: &Options,
    initial_time: Second,
    preparation_overhead: Second,
    conclusion_overhead: Second,
    executor_overhead: Second,
) -> u64 {
    let runs_in_min_time = (options.min_benchmarking_time
        / (initial_time + executor_overhead + preparation_overhead + conclusion_overhead))
        as u64;

    let min = cmp::max(runs_in_min_time, options.run_bounds.min);

    options
        .run_bounds
        .max
        .as_ref()
        .map(|max| cmp::min(min, *max))
        .unwrap_or(min)
}

/// Computed statistics from benchmark timing data
pub struct BenchmarkStatistics {
    pub mean: Second,
    pub stddev: Option<Second>,
    pub median: Second,
    pub min: Second,
    pub max: Second,
    pub user_mean: Second,
    pub system_mean: Second,
}

impl BenchmarkStatistics {
    /// Compute statistics from timing vectors
    pub fn from_times(
        times_real: &[Second],
        times_user: &[Second],
        times_system: &[Second],
    ) -> Self {
        let t_mean = mean(times_real);
        let t_stddev = if times_real.len() > 1 {
            Some(standard_deviation(times_real, Some(t_mean)))
        } else {
            None
        };
        let t_median = median(times_real);
        let t_min = min(times_real);
        let t_max = max(times_real);
        let user_mean = mean(times_user);
        let system_mean = mean(times_system);

        BenchmarkStatistics {
            mean: t_mean,
            stddev: t_stddev,
            median: t_median,
            min: t_min,
            max: t_max,
            user_mean,
            system_mean,
        }
    }
}

/// Print formatted benchmark summary to console
pub fn print_benchmark_summary(
    stats: &BenchmarkStatistics,
    run_count: usize,
    time_unit: Option<crate::util::units::Unit>,
) {
    let (mean_str, time_unit) = format_duration_unit(stats.mean, time_unit);
    let min_str = format_duration(stats.min, Some(time_unit));
    let max_str = format_duration(stats.max, Some(time_unit));
    let num_str = format!("{run_count} runs");

    let user_str = format_duration(stats.user_mean, Some(time_unit));
    let system_str = format_duration(stats.system_mean, Some(time_unit));

    if run_count == 1 {
        println!(
            "  Time ({} ≡):        {:>8}  {:>8}     [User: {}, System: {}]",
            "abs".green().bold(),
            mean_str.green().bold(),
            "        ", // alignment
            user_str.blue(),
            system_str.blue()
        );
    } else {
        let stddev_str = format_duration(stats.stddev.unwrap(), Some(time_unit));

        println!(
            "  Time ({} ± {}):     {:>8} ± {:>8}    [User: {}, System: {}]",
            "mean".green().bold(),
            "σ".green(),
            mean_str.green().bold(),
            stddev_str.green(),
            user_str.blue(),
            system_str.blue()
        );

        println!(
            "  Range ({} … {}):   {:>8} … {:>8}    {}",
            "min".cyan(),
            "max".purple(),
            min_str.cyan(),
            max_str.purple(),
            num_str.dimmed()
        );
    }
}

/// Generate warnings for benchmark results
pub fn generate_warnings(
    times_real: &[Second],
    all_succeeded: bool,
    executor_kind: &ExecutorKind,
    warmup_count: u64,
    has_prepare_command: bool,
) -> Vec<Warnings> {
    let mut warnings = vec![];

    // Check execution time
    if matches!(executor_kind, ExecutorKind::Shell(_))
        && times_real.iter().any(|&t| t < MIN_EXECUTION_TIME)
    {
        warnings.push(Warnings::FastExecutionTime);
    }

    // Check program exit codes
    if !all_succeeded {
        warnings.push(Warnings::NonZeroExitCode);
    }

    // Run outlier detection
    let scores = modified_zscores(times_real);

    let outlier_warning_options = OutlierWarningOptions {
        warmup_in_use: warmup_count > 0,
        prepare_in_use: has_prepare_command,
    };

    if scores[0] > OUTLIER_THRESHOLD {
        warnings.push(Warnings::SlowInitialRun(
            times_real[0],
            outlier_warning_options,
        ));
    } else if scores.iter().any(|&s| s.abs() > OUTLIER_THRESHOLD) {
        warnings.push(Warnings::OutliersDetected(outlier_warning_options));
    }

    warnings
}

/// Print warnings to stderr
pub fn print_warnings(warnings: &[Warnings]) {
    if !warnings.is_empty() {
        eprintln!(" ");
        for warning in warnings {
            eprintln!("  {}: {}", "Warning".yellow(), warning);
        }
    }
}

pub struct Benchmark<'a> {
    number: usize,
    command: &'a Command<'a>,
    options: &'a Options,
    executor: &'a dyn Executor,
}

impl<'a> Benchmark<'a> {
    pub fn new(
        number: usize,
        command: &'a Command<'a>,
        options: &'a Options,
        executor: &'a dyn Executor,
    ) -> Self {
        Benchmark {
            number,
            command,
            options,
            executor,
        }
    }

    /// Run the benchmark for a single command
    pub fn run(&self) -> Result<BenchmarkResult> {
        if self.options.output_style != OutputStyleOption::Disabled {
            println!(
                "{}{}: {}",
                "Benchmark ".bold(),
                (self.number + 1).to_string().bold(),
                self.command.get_name_with_unused_parameters(),
            );
        }

        let mut accumulator = CommandAccumulator::new();
        let output_policy = &self.options.command_output_policies[self.number];

        let preparation_command = build_preparation_command(
            self.options,
            self.number,
            self.command.get_parameters().iter().cloned(),
        );

        let conclusion_command = build_conclusion_command(
            self.options,
            self.number,
            self.command.get_parameters().iter().cloned(),
        );

        run_setup_command(
            self.executor,
            self.options,
            self.command.get_parameters().iter().cloned(),
            output_policy,
        )?;

        // Warmup phase
        if self.options.warmup_count > 0 {
            let progress_bar = if self.options.output_style != OutputStyleOption::Disabled {
                Some(get_progress_bar(
                    self.options.warmup_count,
                    "Performing warmup runs",
                    self.options.output_style,
                ))
            } else {
                None
            };

            for i in 0..self.options.warmup_count {
                let _ = run_preparation_command_optional(
                    self.executor,
                    preparation_command.as_ref(),
                    output_policy,
                )?;
                let _ = self.executor.run_command_and_measure(
                    self.command,
                    BenchmarkIteration::Warmup(i),
                    None,
                    output_policy,
                )?;
                let _ = run_conclusion_command_optional(
                    self.executor,
                    conclusion_command.as_ref(),
                    output_policy,
                )?;
                if let Some(bar) = progress_bar.as_ref() {
                    bar.inc(1)
                }
            }
            if let Some(bar) = progress_bar.as_ref() {
                bar.finish_and_clear()
            }
        }

        // Set up progress bar (and spinner for initial measurement)
        let progress_bar = if self.options.output_style != OutputStyleOption::Disabled {
            Some(get_progress_bar(
                self.options.run_bounds.min,
                "Initial time measurement",
                self.options.output_style,
            ))
        } else {
            None
        };

        let preparation_result = run_preparation_command_optional(
            self.executor,
            preparation_command.as_ref(),
            output_policy,
        )?;
        let preparation_overhead =
            preparation_result.map_or(0.0, |res| res.time_real + self.executor.time_overhead());

        // Initial timing run
        let (res, status) = self.executor.run_command_and_measure(
            self.command,
            BenchmarkIteration::Benchmark(0),
            None,
            output_policy,
        )?;

        let conclusion_result = run_conclusion_command_optional(
            self.executor,
            conclusion_command.as_ref(),
            output_policy,
        )?;
        let conclusion_overhead =
            conclusion_result.map_or(0.0, |res| res.time_real + self.executor.time_overhead());

        // Determine number of benchmark runs
        let count = calculate_run_count(
            self.options,
            res.time_real,
            preparation_overhead,
            conclusion_overhead,
            self.executor.time_overhead(),
        );

        let count_remaining = count - 1;

        // Save the first result
        accumulator.add_result(&res, extract_exit_code(status), status.success());

        // Re-configure the progress bar
        if let Some(bar) = progress_bar.as_ref() {
            bar.set_length(count)
        }
        if let Some(bar) = progress_bar.as_ref() {
            bar.inc(1)
        }

        // Gather statistics (perform the actual benchmark)
        for i in 0..count_remaining {
            run_preparation_command_optional(
                self.executor,
                preparation_command.as_ref(),
                output_policy,
            )?;

            let msg = {
                let mean = format_duration(mean(&accumulator.times_real), self.options.time_unit);
                format!("Current estimate: {}", mean.to_string().green())
            };

            if let Some(bar) = progress_bar.as_ref() {
                bar.set_message(msg.to_owned())
            }

            let (res, status) = self.executor.run_command_and_measure(
                self.command,
                BenchmarkIteration::Benchmark(i + 1),
                None,
                output_policy,
            )?;

            accumulator.add_result(&res, extract_exit_code(status), status.success());

            if let Some(bar) = progress_bar.as_ref() {
                bar.inc(1)
            }

            run_conclusion_command_optional(
                self.executor,
                conclusion_command.as_ref(),
                output_policy,
            )?;
        }

        if let Some(bar) = progress_bar.as_ref() {
            bar.finish_and_clear()
        }

        // Print summary
        if self.options.output_style != OutputStyleOption::Disabled {
            let stats = accumulator.get_statistics();
            print_benchmark_summary(&stats, accumulator.run_count(), self.options.time_unit);
        }

        // Generate and print warnings
        let has_prepare = self
            .options
            .preparation_command
            .as_ref()
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        let warnings = generate_warnings(
            &accumulator.times_real,
            accumulator.all_succeeded,
            &self.options.executor_kind,
            self.options.warmup_count,
            has_prepare,
        );
        print_warnings(&warnings);

        if self.options.output_style != OutputStyleOption::Disabled {
            println!(" ");
        }

        run_cleanup_command(
            self.executor,
            self.options,
            self.command.get_parameters().iter().cloned(),
            output_policy,
        )?;

        Ok(accumulator.build_result(self.command))
    }
}
