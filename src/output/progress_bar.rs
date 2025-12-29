use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::time::Duration;

use crate::options::OutputStyleOption;

#[cfg(not(windows))]
const TICK_SETTINGS: (&str, u64) = ("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ ", 80);

#[cfg(windows)]
const TICK_SETTINGS: (&str, u64) = (r"+-x| ", 200);

/// Return a pre-configured progress bar
pub fn get_progress_bar(length: u64, msg: &str, option: OutputStyleOption) -> ProgressBar {
    let progressbar_style = match option {
        OutputStyleOption::Basic | OutputStyleOption::Color => ProgressStyle::default_bar(),
        _ => ProgressStyle::default_spinner()
            .tick_chars(TICK_SETTINGS.0)
            .template(" {spinner} {msg:<30} {wide_bar} ETA {eta_precise} ")
            .expect("no template error"),
    };

    let progress_bar = match option {
        OutputStyleOption::Basic | OutputStyleOption::Color => ProgressBar::hidden(),
        _ => ProgressBar::new(length),
    };
    progress_bar.set_style(progressbar_style);
    progress_bar.enable_steady_tick(Duration::from_millis(TICK_SETTINGS.1));
    progress_bar.set_message(msg.to_owned());

    progress_bar
}

/// Return a multi-progress bar setup for interleaved benchmarking
pub fn get_multi_progress_bar(
    num_commands: usize,
    length: u64,
    option: OutputStyleOption,
) -> Option<(MultiProgress, Vec<ProgressBar>)> {
    if matches!(
        option,
        OutputStyleOption::Basic | OutputStyleOption::Color | OutputStyleOption::Disabled
    ) {
        return None;
    }

    let multi = MultiProgress::new();
    let style = ProgressStyle::default_bar()
        .template("  {msg:<31} {wide_bar} {pos:>3}/{len:3}")
        .expect("no template error");

    let bars: Vec<ProgressBar> = (0..num_commands)
        .map(|i| {
            let bar = multi.add(ProgressBar::new(length));
            bar.set_style(style.clone());
            bar.set_message(format!("Command {}", i + 1));
            bar
        })
        .collect();

    Some((multi, bars))
}
