use super::benchmark_result::BenchmarkResult;
use super::executor::{BenchmarkIteration, Executor, MockExecutor, RawExecutor, ShellExecutor};
use super::timing_result::TimingResult;
use super::{relative_speed, Benchmark, MIN_EXECUTION_TIME};
use colored::*;
use std::cmp;
use std::cmp::Ordering;

use crate::command::{Command, Commands};
use crate::export::ExportManager;
use crate::options::{
    CmdFailureAction, CommandOutputPolicy, ExecutorKind, Options, OutputStyleOption, SortOrder,
};
use crate::outlier_detection::{modified_zscores, OUTLIER_THRESHOLD};
use crate::output::format::{format_duration, format_duration_unit};
use crate::output::progress_bar::get_progress_bar;
use crate::output::warnings::{OutlierWarningOptions, Warnings};
use crate::util::exit_code::extract_exit_code;
use crate::util::min_max::{max, min};
use crate::util::units::Second;

use anyhow::{anyhow, Result};
use statistical::{mean, median, standard_deviation};

/// Accumulator for collecting benchmark timing data during interleaved execution
struct CommandAccumulator {
    times_real: Vec<Second>,
    times_user: Vec<Second>,
    times_system: Vec<Second>,
    memory_usage_byte: Vec<u64>,
    exit_codes: Vec<Option<i32>>,
    all_succeeded: bool,
}

impl CommandAccumulator {
    fn new() -> Self {
        CommandAccumulator {
            times_real: vec![],
            times_user: vec![],
            times_system: vec![],
            memory_usage_byte: vec![],
            exit_codes: vec![],
            all_succeeded: true,
        }
    }
}

pub struct Scheduler<'a> {
    commands: &'a Commands<'a>,
    options: &'a Options,
    export_manager: &'a ExportManager,
    results: Vec<BenchmarkResult>,
}

impl<'a> Scheduler<'a> {
    pub fn new(
        commands: &'a Commands,
        options: &'a Options,
        export_manager: &'a ExportManager,
    ) -> Self {
        Self {
            commands,
            options,
            export_manager,
            results: vec![],
        }
    }

    pub fn run_benchmarks(&mut self) -> Result<()> {
        let mut executor: Box<dyn Executor> = match self.options.executor_kind {
            ExecutorKind::Raw => Box::new(RawExecutor::new(self.options)),
            ExecutorKind::Mock(ref shell) => Box::new(MockExecutor::new(shell.clone())),
            ExecutorKind::Shell(ref shell) => Box::new(ShellExecutor::new(shell, self.options)),
        };

        let reference = self
            .options
            .reference_command
            .as_ref()
            .map(|cmd| Command::new(self.options.reference_name.as_deref(), cmd));

        executor.calibrate()?;

        // Use interleaved mode if requested and we have at least 2 commands
        let all_commands: Vec<_> = reference.iter().chain(self.commands.iter()).collect();
        if self.options.interleave && all_commands.len() >= 2 {
            return self.run_interleaved_benchmarks(&*executor, &all_commands);
        }

        for (number, cmd) in all_commands.into_iter().enumerate() {
            self.results
                .push(Benchmark::new(number, cmd, self.options, &*executor).run()?);

            // We export results after each individual benchmark, because
            // we would risk losing them if a later benchmark fails.
            self.export_manager.write_results(&self.results, true)?;
        }

        Ok(())
    }

    /// Run benchmarks in interleaved mode: alternate between commands in rounds
    fn run_interleaved_benchmarks(
        &mut self,
        executor: &dyn Executor,
        commands: &[&Command<'_>],
    ) -> Result<()> {
        let num_commands = commands.len();

        // Initialize result accumulators for each command
        let mut accumulators: Vec<CommandAccumulator> = (0..num_commands)
            .map(|_| CommandAccumulator::new())
            .collect();

        // Build preparation and conclusion commands for each benchmark
        let preparation_commands: Vec<Option<Command<'_>>> = commands
            .iter()
            .enumerate()
            .map(|(number, cmd)| {
                self.options.preparation_command.as_ref().map(|values| {
                    let preparation_command = if values.len() == 1 {
                        &values[0]
                    } else {
                        &values[number]
                    };
                    Command::new_parametrized(
                        None,
                        preparation_command,
                        cmd.get_parameters().iter().cloned(),
                    )
                })
            })
            .collect();

        let conclusion_commands: Vec<Option<Command<'_>>> = commands
            .iter()
            .enumerate()
            .map(|(number, cmd)| {
                self.options.conclusion_command.as_ref().map(|values| {
                    let conclusion_command = if values.len() == 1 {
                        &values[0]
                    } else {
                        &values[number]
                    };
                    Command::new_parametrized(
                        None,
                        conclusion_command,
                        cmd.get_parameters().iter().cloned(),
                    )
                })
            })
            .collect();

        // Print header for interleaved mode
        if self.options.output_style != OutputStyleOption::Disabled {
            println!(
                "{} Interleaved benchmarking {} commands",
                "Running".bold(),
                num_commands
            );
            for (i, cmd) in commands.iter().enumerate() {
                println!(
                    "  {}: {}",
                    format!("Command {}", i + 1).bold(),
                    cmd.get_name_with_unused_parameters()
                );
            }
            println!();
        }

        // Run setup commands for all benchmarks
        for (number, cmd) in commands.iter().enumerate() {
            let output_policy = &self.options.command_output_policies[number];
            self.run_setup_command(executor, cmd, output_policy)?;
        }

        // Warmup phase: run warmups for each command before timed rounds
        if self.options.warmup_count > 0 {
            let total_warmups = self.options.warmup_count * num_commands as u64;
            let progress_bar = if self.options.output_style != OutputStyleOption::Disabled {
                Some(get_progress_bar(
                    total_warmups,
                    "Performing warmup runs",
                    self.options.output_style,
                ))
            } else {
                None
            };

            for warmup_idx in 0..self.options.warmup_count {
                for (number, cmd) in commands.iter().enumerate() {
                    let output_policy = &self.options.command_output_policies[number];

                    let _ = self.run_preparation_command_optional(
                        executor,
                        preparation_commands[number].as_ref(),
                        output_policy,
                    )?;

                    let _ = executor.run_command_and_measure(
                        cmd,
                        BenchmarkIteration::Warmup(warmup_idx),
                        None,
                        output_policy,
                    )?;

                    let _ = self.run_conclusion_command_optional(
                        executor,
                        conclusion_commands[number].as_ref(),
                        output_policy,
                    )?;

                    if let Some(bar) = progress_bar.as_ref() {
                        bar.inc(1);
                    }
                }
            }

            if let Some(bar) = progress_bar.as_ref() {
                bar.finish_and_clear();
            }
        }

        // Initial timing round to determine run count
        let mut max_initial_time = 0.0;

        for (number, cmd) in commands.iter().enumerate() {
            let output_policy = &self.options.command_output_policies[number];

            let preparation_result = self.run_preparation_command_optional(
                executor,
                preparation_commands[number].as_ref(),
                output_policy,
            )?;
            let preparation_overhead =
                preparation_result.map_or(0.0, |res| res.time_real + executor.time_overhead());

            let (res, status) = executor.run_command_and_measure(
                cmd,
                BenchmarkIteration::Benchmark(0),
                None,
                output_policy,
            )?;

            let conclusion_result = self.run_conclusion_command_optional(
                executor,
                conclusion_commands[number].as_ref(),
                output_policy,
            )?;
            let conclusion_overhead =
                conclusion_result.map_or(0.0, |res| res.time_real + executor.time_overhead());

            let initial_time = res.time_real
                + executor.time_overhead()
                + preparation_overhead
                + conclusion_overhead;
            if initial_time > max_initial_time {
                max_initial_time = initial_time;
            }

            accumulators[number].times_real.push(res.time_real);
            accumulators[number].times_user.push(res.time_user);
            accumulators[number].times_system.push(res.time_system);
            accumulators[number]
                .memory_usage_byte
                .push(res.memory_usage_byte);
            accumulators[number]
                .exit_codes
                .push(extract_exit_code(status));
            accumulators[number].all_succeeded =
                accumulators[number].all_succeeded && status.success();
        }

        let runs_in_min_time = (self.options.min_benchmarking_time / max_initial_time) as u64;

        let run_count = {
            let min = cmp::max(runs_in_min_time, self.options.run_bounds.min);
            self.options
                .run_bounds
                .max
                .as_ref()
                .map(|max| cmp::min(min, *max))
                .unwrap_or(min)
        };

        self.build_and_update_results(commands, &accumulators);
        self.export_manager.write_results(&self.results, true)?;

        // Set up progress bar for remaining rounds
        let remaining_rounds = run_count - 1;
        let total_remaining_runs = remaining_rounds * num_commands as u64;
        let progress_bar = if self.options.output_style != OutputStyleOption::Disabled {
            Some(get_progress_bar(
                total_remaining_runs,
                "Interleaved benchmarking",
                self.options.output_style,
            ))
        } else {
            None
        };

        // Run remaining rounds in interleaved fashion
        for round in 1..run_count {
            for (number, cmd) in commands.iter().enumerate() {
                let output_policy = &self.options.command_output_policies[number];

                let _ = self.run_preparation_command_optional(
                    executor,
                    preparation_commands[number].as_ref(),
                    output_policy,
                )?;

                // Update progress bar message with current estimate
                if let Some(bar) = progress_bar.as_ref() {
                    let cmd_name = cmd.get_name();
                    let mean_time = mean(&accumulators[number].times_real);
                    let mean_str = format_duration(mean_time, self.options.time_unit);
                    bar.set_message(format!(
                        "Round {}/{}: {} (est: {})",
                        round + 1,
                        run_count,
                        cmd_name,
                        mean_str.to_string().green()
                    ));
                }

                let (res, status) = executor.run_command_and_measure(
                    cmd,
                    BenchmarkIteration::Benchmark(round),
                    None,
                    output_policy,
                )?;

                let _ = self.run_conclusion_command_optional(
                    executor,
                    conclusion_commands[number].as_ref(),
                    output_policy,
                )?;

                accumulators[number].times_real.push(res.time_real);
                accumulators[number].times_user.push(res.time_user);
                accumulators[number].times_system.push(res.time_system);
                accumulators[number]
                    .memory_usage_byte
                    .push(res.memory_usage_byte);
                accumulators[number]
                    .exit_codes
                    .push(extract_exit_code(status));
                accumulators[number].all_succeeded =
                    accumulators[number].all_succeeded && status.success();

                if let Some(bar) = progress_bar.as_ref() {
                    bar.inc(1);
                }
            }

            // Export results after each round
            self.build_and_update_results(commands, &accumulators);
            self.export_manager.write_results(&self.results, true)?;
        }

        if let Some(bar) = progress_bar.as_ref() {
            bar.finish_and_clear();
        }

        // Run cleanup commands for all benchmarks
        for (number, cmd) in commands.iter().enumerate() {
            let output_policy = &self.options.command_output_policies[number];
            self.run_cleanup_command(executor, cmd, output_policy)?;
        }

        // Build final results and print summaries
        self.build_and_update_results(commands, &accumulators);
        self.print_interleaved_summaries(commands, &accumulators);

        Ok(())
    }

    /// Build BenchmarkResult structs from accumulated data
    fn build_and_update_results(
        &mut self,
        commands: &[&Command<'_>],
        accumulators: &[CommandAccumulator],
    ) {
        self.results.clear();
        for (number, cmd) in commands.iter().enumerate() {
            let acc = &accumulators[number];
            let times_real = &acc.times_real;
            let times_user = &acc.times_user;
            let times_system = &acc.times_system;

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

            self.results.push(BenchmarkResult {
                command: cmd.get_name(),
                command_with_unused_parameters: cmd.get_name_with_unused_parameters(),
                mean: t_mean,
                stddev: t_stddev,
                median: t_median,
                user: user_mean,
                system: system_mean,
                min: t_min,
                max: t_max,
                times: Some(times_real.clone()),
                memory_usage_byte: Some(acc.memory_usage_byte.clone()),
                exit_codes: acc.exit_codes.clone(),
                parameters: cmd
                    .get_parameters()
                    .iter()
                    .map(|(name, value)| (name.to_string(), value.to_string()))
                    .collect(),
            });
        }
    }

    /// Print summary statistics for interleaved benchmarks
    fn print_interleaved_summaries(
        &self,
        commands: &[&Command<'_>],
        accumulators: &[CommandAccumulator],
    ) {
        if self.options.output_style == OutputStyleOption::Disabled {
            return;
        }

        for (number, cmd) in commands.iter().enumerate() {
            let acc = &accumulators[number];
            let times_real = &acc.times_real;
            let times_user = &acc.times_user;
            let times_system = &acc.times_system;

            println!(
                "{}{}: {}",
                "Benchmark ".bold(),
                (number + 1).to_string().bold(),
                cmd.get_name_with_unused_parameters(),
            );

            let t_num = times_real.len();
            let t_mean = mean(times_real);
            let t_stddev = if times_real.len() > 1 {
                Some(standard_deviation(times_real, Some(t_mean)))
            } else {
                None
            };
            let t_min = min(times_real);
            let t_max = max(times_real);

            let user_mean = mean(times_user);
            let system_mean = mean(times_system);

            let (mean_str, time_unit) = format_duration_unit(t_mean, self.options.time_unit);
            let min_str = format_duration(t_min, Some(time_unit));
            let max_str = format_duration(t_max, Some(time_unit));
            let num_str = format!("{t_num} runs");

            let user_str = format_duration(user_mean, Some(time_unit));
            let system_str = format_duration(system_mean, Some(time_unit));

            if times_real.len() == 1 {
                println!(
                    "  Time ({} ≡):        {:>8}  {:>8}     [User: {}, System: {}]",
                    "abs".green().bold(),
                    mean_str.green().bold(),
                    "        ",
                    user_str.blue(),
                    system_str.blue()
                );
            } else {
                let stddev_str = format_duration(t_stddev.unwrap(), Some(time_unit));

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

            // Warnings
            let mut warnings = vec![];

            if matches!(self.options.executor_kind, ExecutorKind::Shell(_))
                && times_real.iter().any(|&t| t < MIN_EXECUTION_TIME)
            {
                warnings.push(Warnings::FastExecutionTime);
            }

            if !acc.all_succeeded {
                warnings.push(Warnings::NonZeroExitCode);
            }

            let scores = modified_zscores(times_real);
            let outlier_warning_options = OutlierWarningOptions {
                warmup_in_use: self.options.warmup_count > 0,
                prepare_in_use: self
                    .options
                    .preparation_command
                    .as_ref()
                    .map(|v| v.len())
                    .unwrap_or(0)
                    > 0,
            };

            if scores[0] > OUTLIER_THRESHOLD {
                warnings.push(Warnings::SlowInitialRun(
                    times_real[0],
                    outlier_warning_options,
                ));
            } else if scores.iter().any(|&s| s.abs() > OUTLIER_THRESHOLD) {
                warnings.push(Warnings::OutliersDetected(outlier_warning_options));
            }

            if !warnings.is_empty() {
                eprintln!(" ");
                for warning in &warnings {
                    eprintln!("  {}: {}", "Warning".yellow(), warning);
                }
            }

            println!(" ");
        }
    }

    /// Run setup command for a benchmark
    fn run_setup_command(
        &self,
        executor: &dyn Executor,
        cmd: &Command<'_>,
        output_policy: &CommandOutputPolicy,
    ) -> Result<TimingResult> {
        let command = self.options.setup_command.as_ref().map(|setup_command| {
            Command::new_parametrized(None, setup_command, cmd.get_parameters().iter().cloned())
        });

        let error_output = "The setup command terminated with a non-zero exit code. \
                            Append ' || true' to the command if you are sure that this can be ignored.";

        Ok(command
            .map(|c| self.run_intermediate_command(executor, &c, error_output, output_policy))
            .transpose()?
            .unwrap_or_default())
    }

    /// Run cleanup command for a benchmark
    fn run_cleanup_command(
        &self,
        executor: &dyn Executor,
        cmd: &Command<'_>,
        output_policy: &CommandOutputPolicy,
    ) -> Result<TimingResult> {
        let command = self
            .options
            .cleanup_command
            .as_ref()
            .map(|cleanup_command| {
                Command::new_parametrized(
                    None,
                    cleanup_command,
                    cmd.get_parameters().iter().cloned(),
                )
            });

        let error_output = "The cleanup command terminated with a non-zero exit code. \
                            Append ' || true' to the command if you are sure that this can be ignored.";

        Ok(command
            .map(|c| self.run_intermediate_command(executor, &c, error_output, output_policy))
            .transpose()?
            .unwrap_or_default())
    }

    /// Run preparation command
    fn run_preparation_command(
        &self,
        executor: &dyn Executor,
        command: &Command<'_>,
        output_policy: &CommandOutputPolicy,
    ) -> Result<TimingResult> {
        let error_output = "The preparation command terminated with a non-zero exit code. \
                            Append ' || true' to the command if you are sure that this can be ignored.";

        self.run_intermediate_command(executor, command, error_output, output_policy)
    }

    fn run_preparation_command_optional(
        &self,
        executor: &dyn Executor,
        command: Option<&Command<'_>>,
        output_policy: &CommandOutputPolicy,
    ) -> Result<Option<TimingResult>> {
        command
            .map(|cmd| self.run_preparation_command(executor, cmd, output_policy))
            .transpose()
    }

    /// Run conclusion command
    fn run_conclusion_command(
        &self,
        executor: &dyn Executor,
        command: &Command<'_>,
        output_policy: &CommandOutputPolicy,
    ) -> Result<TimingResult> {
        let error_output = "The conclusion command terminated with a non-zero exit code. \
                            Append ' || true' to the command if you are sure that this can be ignored.";

        self.run_intermediate_command(executor, command, error_output, output_policy)
    }

    fn run_conclusion_command_optional(
        &self,
        executor: &dyn Executor,
        command: Option<&Command<'_>>,
        output_policy: &CommandOutputPolicy,
    ) -> Result<Option<TimingResult>> {
        command
            .map(|cmd| self.run_conclusion_command(executor, cmd, output_policy))
            .transpose()
    }

    /// Run an intermediate command (setup, cleanup, prepare, or conclude)
    fn run_intermediate_command(
        &self,
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

    pub fn print_relative_speed_comparison(&self) {
        if self.options.output_style == OutputStyleOption::Disabled {
            return;
        }

        if self.results.len() < 2 {
            return;
        }

        let reference = self
            .options
            .reference_command
            .as_ref()
            .map(|_| &self.results[0])
            .unwrap_or_else(|| relative_speed::fastest_of(&self.results));

        if let Some(annotated_results) = relative_speed::compute_with_check_from_reference(
            &self.results,
            reference,
            self.options.sort_order_speed_comparison,
        ) {
            match self.options.sort_order_speed_comparison {
                SortOrder::MeanTime => {
                    println!("{}", "Summary".bold());

                    let reference = annotated_results.iter().find(|r| r.is_reference).unwrap();
                    let others = annotated_results.iter().filter(|r| !r.is_reference);

                    println!(
                        "  {} ran",
                        reference.result.command_with_unused_parameters.cyan()
                    );

                    for item in others {
                        let stddev = if let Some(stddev) = item.relative_speed_stddev {
                            format!(" ± {}", format!("{:.2}", stddev).green())
                        } else {
                            "".into()
                        };
                        let comparator = match item.relative_ordering {
                            Ordering::Less => format!(
                                "{}{} times slower than",
                                format!("{:8.2}", item.relative_speed).bold().green(),
                                stddev
                            ),
                            Ordering::Greater => format!(
                                "{}{} times faster than",
                                format!("{:8.2}", item.relative_speed).bold().green(),
                                stddev
                            ),
                            Ordering::Equal => format!(
                                "    As fast ({}{}) as",
                                format!("{:.2}", item.relative_speed).bold().green(),
                                stddev
                            ),
                        };
                        println!(
                            "{} {}",
                            comparator,
                            &item.result.command_with_unused_parameters.magenta()
                        );
                    }
                }
                SortOrder::Command => {
                    println!("{}", "Relative speed comparison".bold());

                    for item in annotated_results {
                        println!(
                            "  {}{}  {}",
                            format!("{:10.2}", item.relative_speed).bold().green(),
                            if item.is_reference {
                                "        ".into()
                            } else if let Some(stddev) = item.relative_speed_stddev {
                                format!(" ± {}", format!("{stddev:5.2}").green())
                            } else {
                                "        ".into()
                            },
                            &item.result.command_with_unused_parameters,
                        );
                    }
                }
            }
        } else {
            eprintln!(
                "{}: The benchmark comparison could not be computed as some benchmark times are zero. \
                 This could be caused by background interference during the initial calibration phase \
                 of hyperfine, in combination with very fast commands (faster than a few milliseconds). \
                 Try to re-run the benchmark on a quiet system. If you did not do so already, try the \
                 --shell=none/-N option. If it does not help either, you command is most likely too fast \
                 to be accurately benchmarked by hyperfine.",
                 "Note".bold().red()
            );
        }
    }

    pub fn final_export(&self) -> Result<()> {
        self.export_manager.write_results(&self.results, false)
    }
}

#[cfg(test)]
fn generate_results(args: &[&'static str]) -> Result<Vec<BenchmarkResult>> {
    generate_results_with_options(args, None)
}

#[cfg(test)]
fn generate_results_with_options(
    args: &[&'static str],
    verify_options: Option<&dyn Fn(&Options)>,
) -> Result<Vec<BenchmarkResult>> {
    use crate::cli::get_cli_arguments;

    let args = ["hyperfine", "--debug-mode", "--style=none"]
        .iter()
        .chain(args);
    let cli_arguments = get_cli_arguments(args);
    let mut options = Options::from_cli_arguments(&cli_arguments)?;

    assert_eq!(options.executor_kind, ExecutorKind::Mock(None));

    if let Some(verify) = verify_options {
        verify(&options);
    }

    let commands = Commands::from_cli_arguments(&cli_arguments)?;
    let export_manager = ExportManager::from_cli_arguments(
        &cli_arguments,
        options.time_unit,
        options.sort_order_exports,
    )?;

    options.validate_against_command_list(&commands)?;

    let mut scheduler = Scheduler::new(&commands, &options, &export_manager);

    scheduler.run_benchmarks()?;
    Ok(scheduler.results)
}

#[test]
fn scheduler_basic() -> Result<()> {
    insta::assert_yaml_snapshot!(generate_results(&["--runs=2", "sleep 0.123", "sleep 0.456"])?, @r#"
    - command: sleep 0.123
      mean: 0.123
      stddev: 0
      median: 0.123
      user: 0
      system: 0
      min: 0.123
      max: 0.123
      times:
        - 0.123
        - 0.123
      memory_usage_byte:
        - 0
        - 0
      exit_codes:
        - 0
        - 0
    - command: sleep 0.456
      mean: 0.456
      stddev: 0
      median: 0.456
      user: 0
      system: 0
      min: 0.456
      max: 0.456
      times:
        - 0.456
        - 0.456
      memory_usage_byte:
        - 0
        - 0
      exit_codes:
        - 0
        - 0
    "#);

    Ok(())
}

#[test]
fn scheduler_interleaved_basic() -> Result<()> {
    let results = generate_results(&["--runs=2", "--interleave", "sleep 0.123", "sleep 0.456"])?;

    // Should have 2 commands
    assert_eq!(results.len(), 2);

    // Each should have 2 runs
    assert_eq!(results[0].times.as_ref().unwrap().len(), 2);
    assert_eq!(results[1].times.as_ref().unwrap().len(), 2);

    // Verify timing values
    assert_eq!(results[0].command, "sleep 0.123");
    assert_eq!(results[1].command, "sleep 0.456");
    assert!((results[0].mean - 0.123).abs() < 0.001);
    assert!((results[1].mean - 0.456).abs() < 0.001);

    Ok(())
}

#[test]
fn scheduler_interleaved_single_command_falls_back() -> Result<()> {
    // Interleave with a single command should fall back to sequential mode
    let results = generate_results(&["--runs=2", "--interleave", "sleep 0.123"])?;

    assert_eq!(results.len(), 1);

    Ok(())
}

#[test]
fn scheduler_interleaved_with_warmup() -> Result<()> {
    let results = generate_results(&[
        "--runs=2",
        "--warmup=1",
        "--interleave",
        "sleep 0.1",
        "sleep 0.2",
    ])?;

    assert_eq!(results.len(), 2);

    // Should still have 2 timed runs each (warmup doesn't count)
    assert_eq!(results[0].times.as_ref().unwrap().len(), 2);
    assert_eq!(results[1].times.as_ref().unwrap().len(), 2);

    Ok(())
}

#[test]
fn scheduler_interleaved_with_prepare() -> Result<()> {
    let results = generate_results_with_options(
        &[
            "--runs=2",
            "--interleave",
            "-p",
            "sleep 0.01",
            "sleep 0.1",
            "sleep 0.2",
        ],
        Some(&|options| {
            assert!(
                options.preparation_command.is_some(),
                "preparation_command should be set"
            );
        }),
    )?;

    assert_eq!(results.len(), 2);

    // Should have 2 timed runs each
    assert_eq!(results[0].times.as_ref().unwrap().len(), 2);
    assert_eq!(results[1].times.as_ref().unwrap().len(), 2);

    Ok(())
}

#[test]
fn scheduler_interleaved_with_conclude() -> Result<()> {
    let results = generate_results(&[
        "--runs=2",
        "--interleave",
        "--conclude",
        "sleep 0.01",
        "sleep 0.1",
        "sleep 0.2",
    ])?;

    assert_eq!(results.len(), 2);

    Ok(())
}

#[test]
fn scheduler_interleaved_with_setup_cleanup() -> Result<()> {
    let results = generate_results(&[
        "--runs=2",
        "--interleave",
        "--setup",
        "sleep 0.01",
        "--cleanup",
        "sleep 0.01",
        "sleep 0.1",
        "sleep 0.2",
    ])?;

    assert_eq!(results.len(), 2);

    Ok(())
}

#[test]
fn scheduler_interleaved_three_commands() -> Result<()> {
    let results = generate_results(&[
        "--runs=2",
        "--interleave",
        "sleep 0.1",
        "sleep 0.2",
        "sleep 0.3",
    ])?;

    assert_eq!(results.len(), 3);

    // All should have 2 runs
    for result in &results {
        assert_eq!(result.times.as_ref().unwrap().len(), 2);
    }

    Ok(())
}

#[test]
fn scheduler_interleaved_single_run() -> Result<()> {
    // Test interleaved mode with exactly 1 run (edge case for stddev calculation)
    let results = generate_results(&["--runs=1", "--interleave", "sleep 0.1", "sleep 0.2"])?;

    assert_eq!(results.len(), 2);

    // With only 1 run, stddev should be None
    assert!(results[0].stddev.is_none());
    assert!(results[1].stddev.is_none());

    // Each should have exactly 1 timing
    assert_eq!(results[0].times.as_ref().unwrap().len(), 1);
    assert_eq!(results[1].times.as_ref().unwrap().len(), 1);

    Ok(())
}

#[test]
fn scheduler_interleaved_auto_run_count_uses_max_initial_time() -> Result<()> {
    let results = generate_results(&[
        "--min-runs=1",
        "--max-runs=10",
        "--min-benchmarking-time=1",
        "--interleave",
        "sleep 0.25",
        "sleep 0.5",
    ])?;

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].times.as_ref().unwrap().len(), 2);
    assert_eq!(results[1].times.as_ref().unwrap().len(), 2);

    Ok(())
}

#[test]
fn scheduler_interleaved_with_prepare_and_conclude() -> Result<()> {
    // Test with both prepare and conclude commands
    let results = generate_results(&[
        "--runs=2",
        "--interleave",
        "-p",
        "sleep 0.01",
        "-c",
        "sleep 0.01",
        "sleep 0.1",
        "sleep 0.2",
    ])?;

    assert_eq!(results.len(), 2);

    Ok(())
}

#[test]
fn scheduler_interleaved_with_per_command_prepare() -> Result<()> {
    // Test with multiple prepare commands (one per benchmark command)
    let results = generate_results(&[
        "--runs=2",
        "--interleave",
        "-p",
        "sleep 0.01",
        "-p",
        "sleep 0.02",
        "sleep 0.1",
        "sleep 0.2",
    ])?;

    assert_eq!(results.len(), 2);

    Ok(())
}
