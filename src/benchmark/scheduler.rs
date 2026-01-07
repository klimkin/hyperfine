use super::benchmark_result::BenchmarkResult;
use super::executor::{BenchmarkIteration, Executor, MockExecutor, RawExecutor, ShellExecutor};
use super::{
    build_conclusion_command, build_preparation_command, calculate_run_count, generate_warnings,
    print_benchmark_summary, print_warnings, relative_speed, run_cleanup_command,
    run_conclusion_command_optional, run_preparation_command_optional, run_setup_command,
    Benchmark, CommandAccumulator,
};
use colored::*;
use std::cmp::Ordering;

use crate::analysis::{compute_speedup_with_ci, determine_verdict, AnalysisResult, Verdict};
use crate::command::{Command, Commands};
use crate::export::ExportManager;
use crate::options::{ExecutorKind, Options, OutputStyleOption, SortOrder};
use crate::output::format::format_duration;
use crate::output::progress_bar::{get_multi_progress_bar, get_progress_bar};
use crate::util::exit_code::extract_exit_code;

use anyhow::Result;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use statistical::mean;

pub struct Scheduler<'a> {
    commands: &'a Commands<'a>,
    options: &'a Options,
    export_manager: &'a ExportManager,
    results: Vec<BenchmarkResult>,
    analysis_results: Option<Vec<AnalysisResult>>,
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
            analysis_results: None,
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
                build_preparation_command(
                    self.options,
                    number,
                    cmd.get_parameters().iter().cloned(),
                )
            })
            .collect();

        let conclusion_commands: Vec<Option<Command<'_>>> = commands
            .iter()
            .enumerate()
            .map(|(number, cmd)| {
                build_conclusion_command(self.options, number, cmd.get_parameters().iter().cloned())
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
            run_setup_command(
                executor,
                self.options,
                cmd.get_parameters().iter().cloned(),
                output_policy,
            )?;
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

                    let _ = run_preparation_command_optional(
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

                    let _ = run_conclusion_command_optional(
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

            let preparation_result = run_preparation_command_optional(
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

            let conclusion_result = run_conclusion_command_optional(
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

            accumulators[number].add_result(&res, extract_exit_code(status), status.success());
        }

        // Use the slowest command's time for run count calculation
        let run_count = calculate_run_count(
            self.options,
            max_initial_time,
            0.0, // Already included in max_initial_time
            0.0, // Already included in max_initial_time
            0.0, // Already included in max_initial_time
        );

        self.build_and_update_results(commands, &accumulators);
        self.export_manager.write_results(&self.results, true)?;

        // Set up multi-progress bar for remaining rounds (one bar per command)
        let remaining_rounds = run_count - 1;
        let multi_progress =
            get_multi_progress_bar(num_commands, remaining_rounds, self.options.output_style);

        // Update initial message with estimates from first round
        if let Some((_, ref bars)) = multi_progress {
            for (number, bar) in bars.iter().enumerate() {
                let mean_time = mean(&accumulators[number].times_real);
                let mean_str = format_duration(mean_time, self.options.time_unit);
                bar.set_message(format!(
                    "Command {} estimate: {}",
                    number + 1,
                    mean_str.green()
                ));
            }
        }

        // Run remaining rounds in interleaved fashion
        for round in 1..run_count {
            for (number, cmd) in commands.iter().enumerate() {
                let output_policy = &self.options.command_output_policies[number];

                let _ = run_preparation_command_optional(
                    executor,
                    preparation_commands[number].as_ref(),
                    output_policy,
                )?;

                let (res, status) = executor.run_command_and_measure(
                    cmd,
                    BenchmarkIteration::Benchmark(round),
                    None,
                    output_policy,
                )?;

                let _ = run_conclusion_command_optional(
                    executor,
                    conclusion_commands[number].as_ref(),
                    output_policy,
                )?;

                accumulators[number].add_result(&res, extract_exit_code(status), status.success());

                // Update this command's progress bar
                if let Some((_, ref bars)) = multi_progress {
                    let mean_time = mean(&accumulators[number].times_real);
                    let mean_str = format_duration(mean_time, self.options.time_unit);
                    bars[number].set_message(format!(
                        "Command {} estimate: {}",
                        number + 1,
                        mean_str.green()
                    ));
                    bars[number].inc(1);
                }
            }

            // Export results after each round
            self.build_and_update_results(commands, &accumulators);
            self.export_manager.write_results(&self.results, true)?;
        }

        if let Some((_, bars)) = multi_progress {
            for bar in bars {
                bar.finish_and_clear();
            }
        }

        // Run cleanup commands for all benchmarks
        for (number, cmd) in commands.iter().enumerate() {
            let output_policy = &self.options.command_output_policies[number];
            run_cleanup_command(
                executor,
                self.options,
                cmd.get_parameters().iter().cloned(),
                output_policy,
            )?;
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
            self.results.push(accumulators[number].build_result(cmd));
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

        let has_prepare = self
            .options
            .preparation_command
            .as_ref()
            .map(|v| !v.is_empty())
            .unwrap_or(false);

        for (number, cmd) in commands.iter().enumerate() {
            let acc = &accumulators[number];

            println!(
                "{}{}: {}",
                "Benchmark ".bold(),
                (number + 1).to_string().bold(),
                cmd.get_name_with_unused_parameters(),
            );

            let stats = acc.get_statistics();
            print_benchmark_summary(&stats, acc.run_count(), self.options.time_unit);

            let warnings = generate_warnings(
                &acc.times_real,
                acc.all_succeeded,
                &self.options.executor_kind,
                self.options.warmup_count,
                has_prepare,
            );
            print_warnings(&warnings);

            println!(" ");
        }
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

    /// Check if statistical analysis flags are being used
    fn has_analysis_flags(&self) -> bool {
        self.options.seed.is_some()
            || self.options.confidence != 0.95
            || self.options.practical_delta != 0.01
            || self.options.resamples != 10000
    }

    /// Run statistical analysis on benchmark results (requires interleaved mode)
    pub fn run_analysis(&mut self) {
        // Check if we have enough results and if they have timing data
        if self.results.len() < 2 {
            return;
        }

        // Warn if analysis flags used without interleave
        if self.has_analysis_flags()
            && !self.options.interleave
            && self.options.output_style != OutputStyleOption::Disabled
        {
            eprintln!(
                "{}: Statistical analysis flags (--confidence, --practical-delta, --resamples, --seed) \
                 are most effective with --interleave mode for paired sample analysis.",
                "Warning".yellow()
            );
        }

        // Only run analysis if we have paired data (interleaved mode)
        if !self.options.interleave {
            return;
        }

        // Check that all results have timing data
        if !self.results.iter().all(|r| r.times.is_some()) {
            return;
        }

        // Create RNG (seeded or random)
        let mut rng: ChaCha8Rng = match self.options.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::from_entropy(),
        };

        // Use first command as reference (or the explicit reference if set)
        let ref_times = self.results[0].times.as_ref().unwrap();

        let mut analysis_results = Vec::new();

        for (i, result) in self.results.iter().enumerate() {
            if i == 0 {
                // Skip reference command itself
                continue;
            }

            let cmd_times = result.times.as_ref().unwrap();

            // Compute speedup with confidence interval
            let (speedup, ci_lower, ci_upper) = compute_speedup_with_ci(
                cmd_times,
                ref_times,
                self.options.confidence,
                self.options.resamples,
                &mut rng,
            );

            // Skip if no valid data (speedup is NaN)
            if speedup.is_nan() {
                continue;
            }

            // Determine verdict
            let verdict = determine_verdict(ci_lower, ci_upper, self.options.practical_delta);

            analysis_results.push(AnalysisResult {
                command: result.command.clone(),
                speedup,
                ci_lower,
                ci_upper,
                confidence: self.options.confidence,
                verdict,
            });
        }

        self.analysis_results = Some(analysis_results);
    }

    /// Print statistical analysis results to console
    pub fn print_analysis_results(&self) {
        if self.options.output_style == OutputStyleOption::Disabled {
            return;
        }

        let analysis_results = match &self.analysis_results {
            Some(results) if !results.is_empty() => results,
            _ => return,
        };

        let reference_name = &self.results[0].command_with_unused_parameters;

        println!();
        println!("{}", "Statistical Analysis".bold());
        println!("  Reference: {}", reference_name.cyan());
        println!(
            "  Confidence: {:.0}%, Practical delta: {:.1}%",
            self.options.confidence * 100.0,
            self.options.practical_delta * 100.0
        );
        println!();

        for result in analysis_results {
            let verdict_colored = match result.verdict {
                Verdict::Faster => result.verdict.description().green(),
                Verdict::Slower => result.verdict.description().red(),
                Verdict::NoClearDifference => result.verdict.description().yellow(),
            };

            println!("  {} vs reference:", result.command.cyan());
            println!(
                "    Speedup: {:.2}x ({:.0}% CI: {:.2}-{:.2})",
                result.speedup,
                result.confidence * 100.0,
                result.ci_lower,
                result.ci_upper
            );
            println!("    Verdict: {}", verdict_colored.bold());
            println!();
        }
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

#[cfg(test)]
fn generate_results_with_analysis(
    args: &[&'static str],
) -> Result<(Vec<BenchmarkResult>, Option<Vec<AnalysisResult>>)> {
    use crate::cli::get_cli_arguments;

    let args = ["hyperfine", "--debug-mode", "--style=none"]
        .iter()
        .chain(args);
    let cli_arguments = get_cli_arguments(args);
    let mut options = Options::from_cli_arguments(&cli_arguments)?;

    assert_eq!(options.executor_kind, ExecutorKind::Mock(None));

    let commands = Commands::from_cli_arguments(&cli_arguments)?;
    let export_manager = ExportManager::from_cli_arguments(
        &cli_arguments,
        options.time_unit,
        options.sort_order_exports,
    )?;

    options.validate_against_command_list(&commands)?;

    let mut scheduler = Scheduler::new(&commands, &options, &export_manager);

    scheduler.run_benchmarks()?;
    scheduler.run_analysis();

    Ok((scheduler.results, scheduler.analysis_results))
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

#[test]
fn scheduler_analysis_with_interleaved() -> Result<()> {
    // Analysis should run with interleaved mode
    let (results, analysis) = generate_results_with_analysis(&[
        "--runs=10",
        "--interleave",
        "--seed=42",
        "sleep 0.1",
        "sleep 0.2",
    ])?;

    assert_eq!(results.len(), 2);
    assert!(analysis.is_some());

    let analysis = analysis.unwrap();
    // Should have one analysis result (comparing second command to first)
    assert_eq!(analysis.len(), 1);

    // Second command is 2x slower, so speedup should be around 0.5
    assert!(
        analysis[0].speedup > 0.4 && analysis[0].speedup < 0.6,
        "Speedup should be ~0.5, got {}",
        analysis[0].speedup
    );

    // CI should contain the speedup
    assert!(analysis[0].ci_lower <= analysis[0].speedup);
    assert!(analysis[0].ci_upper >= analysis[0].speedup);

    // With such a clear difference, verdict should be "slower"
    assert_eq!(analysis[0].verdict, Verdict::Slower);

    Ok(())
}

#[test]
fn scheduler_analysis_without_interleaved() -> Result<()> {
    // Analysis should NOT run without interleaved mode
    let (results, analysis) =
        generate_results_with_analysis(&["--runs=10", "sleep 0.1", "sleep 0.2"])?;

    assert_eq!(results.len(), 2);
    // Analysis should be None when not using interleaved mode
    assert!(analysis.is_none());

    Ok(())
}

#[test]
fn scheduler_analysis_reproducible_with_seed() -> Result<()> {
    // Same seed should produce same results
    let (_, analysis1) = generate_results_with_analysis(&[
        "--runs=10",
        "--interleave",
        "--seed=12345",
        "sleep 0.1",
        "sleep 0.2",
    ])?;

    let (_, analysis2) = generate_results_with_analysis(&[
        "--runs=10",
        "--interleave",
        "--seed=12345",
        "sleep 0.1",
        "sleep 0.2",
    ])?;

    let a1 = analysis1.unwrap();
    let a2 = analysis2.unwrap();

    assert_eq!(a1[0].speedup, a2[0].speedup);
    assert_eq!(a1[0].ci_lower, a2[0].ci_lower);
    assert_eq!(a1[0].ci_upper, a2[0].ci_upper);

    Ok(())
}

#[test]
fn scheduler_analysis_single_command_no_analysis() -> Result<()> {
    // Single command should not produce analysis
    let (results, analysis) =
        generate_results_with_analysis(&["--runs=10", "--interleave", "sleep 0.1"])?;

    assert_eq!(results.len(), 1);
    // With only one command, no analysis possible
    assert!(analysis.is_none());

    Ok(())
}

#[test]
fn scheduler_robust_mode_enables_interleave() -> Result<()> {
    // --robust should enable interleaved mode and analysis
    let results = generate_results_with_options(
        &["--robust", "sleep 0.1", "sleep 0.2"],
        Some(&|options| {
            assert!(options.robust, "robust should be true");
            assert!(
                options.interleave,
                "interleave should be enabled by robust mode"
            );
            assert_eq!(
                options.run_bounds.min, 20,
                "min runs should be 20 in robust mode"
            );
        }),
    )?;

    // Should have 2 commands
    assert_eq!(results.len(), 2);

    // Should have at least 20 runs each (robust mode sets min=20)
    assert!(results[0].times.as_ref().unwrap().len() >= 20);
    assert!(results[1].times.as_ref().unwrap().len() >= 20);

    Ok(())
}

#[test]
fn scheduler_robust_mode_with_analysis() -> Result<()> {
    // --robust should produce analysis results
    let (results, analysis) =
        generate_results_with_analysis(&["--robust", "--seed=42", "sleep 0.1", "sleep 0.2"])?;

    assert_eq!(results.len(), 2);
    assert!(analysis.is_some(), "robust mode should produce analysis");

    let analysis = analysis.unwrap();
    assert_eq!(analysis.len(), 1);

    // Second command is 2x slower, so speedup should be around 0.5
    assert!(
        analysis[0].speedup > 0.4 && analysis[0].speedup < 0.6,
        "Speedup should be ~0.5, got {}",
        analysis[0].speedup
    );

    Ok(())
}

#[test]
fn scheduler_robust_mode_allows_confidence_override() -> Result<()> {
    // --robust with explicit --confidence should use the override
    let (_, analysis) = generate_results_with_analysis(&[
        "--robust",
        "--confidence=0.99",
        "--seed=42",
        "sleep 0.1",
        "sleep 0.2",
    ])?;

    let analysis = analysis.unwrap();
    // The confidence level should be 0.99, not 0.95
    assert_eq!(analysis[0].confidence, 0.99);

    Ok(())
}
