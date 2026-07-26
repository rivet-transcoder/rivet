//! The terminal progress line: rate, elapsed, ETA, and projected output size.
//!
//! [`rivet::progress::RungProgress`] carries counters, not time — every
//! front-end needs a different presentation of them. This is the CLI's.
//!
//! Two behaviours worth knowing:
//!
//! - **Rate is smoothed.** A rung's instantaneous fps swings wildly (session
//!   setup, a filter warming up, chunk workers spinning up mid-run), and an ETA
//!   computed from the raw last-tick rate jitters so much it's unreadable. The
//!   rate here is an exponential moving average.
//! - **Output is throttled, and rewritten in place on a terminal.** Progress
//!   ticks arrive several times a second; printing each one scrolls anything
//!   useful off the screen.

use std::io::{IsTerminal, Write};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rivet::progress::{ProgressSink, RungProgress, RungStatus};

/// Minimum gap between printed updates for a rung.
const PRINT_INTERVAL: Duration = Duration::from_millis(500);

/// EMA weight for the newest rate sample. Low enough to ride out a stalled
/// tick, high enough to track a real slowdown within a few seconds.
const RATE_SMOOTHING: f64 = 0.25;

/// Per-rung tracking state.
#[derive(Default)]
struct RungState {
    last_print: Option<Instant>,
    last_sample: Option<(Instant, u64)>,
    /// Smoothed frames/second, `None` until two samples have been seen.
    fps: Option<f64>,
    /// Whether this rung's line has been finished with a newline.
    line_open: bool,
}

/// A [`ProgressSink`] that renders to stderr.
pub(crate) struct ProgressPrinter {
    started: Instant,
    inner: Mutex<Vec<RungState>>,
    /// In-place rewrite is only legible for a single rung on a terminal;
    /// multi-rung ladders and piped output get plain throttled lines.
    inplace: bool,
}

impl ProgressPrinter {
    pub(crate) fn new(rungs: usize) -> Self {
        Self {
            started: Instant::now(),
            inner: Mutex::new((0..rungs.max(1)).map(|_| RungState::default()).collect()),
            inplace: rungs == 1 && std::io::stderr().is_terminal(),
        }
    }
}

impl ProgressSink for ProgressPrinter {
    fn on_rung(&self, p: RungProgress) {
        let now = Instant::now();
        let terminal = matches!(p.status, RungStatus::Completed | RungStatus::Failed);

        let Ok(mut states) = self.inner.lock() else { return };
        if p.rung_index >= states.len() {
            states.resize_with(p.rung_index + 1, RungState::default);
        }
        let st = &mut states[p.rung_index];

        // Update the smoothed rate from the gap since the last sample.
        if let Some((t0, f0)) = st.last_sample {
            let dt = now.duration_since(t0).as_secs_f64();
            if dt >= 0.2 && p.frames_done > f0 {
                let sample = (p.frames_done - f0) as f64 / dt;
                st.fps = Some(match st.fps {
                    Some(prev) => prev * (1.0 - RATE_SMOOTHING) + sample * RATE_SMOOTHING,
                    None => sample,
                });
                st.last_sample = Some((now, p.frames_done));
            }
        } else {
            st.last_sample = Some((now, p.frames_done));
        }

        // Throttle, but never swallow a terminal update.
        if !terminal {
            if let Some(last) = st.last_print {
                if now.duration_since(last) < PRINT_INTERVAL {
                    return;
                }
            }
        }
        st.last_print = Some(now);

        let line = self.render(&p, st.fps);
        let mut err = std::io::stderr().lock();
        if self.inplace && !terminal {
            // \r + clear-to-end-of-line, so a shorter line can't leave debris.
            let _ = write!(err, "\r\x1b[2K{line}");
            st.line_open = true;
        } else {
            if st.line_open {
                let _ = writeln!(err);
                st.line_open = false;
            }
            let _ = writeln!(err, "{line}");
        }
        let _ = err.flush();
    }
}

impl ProgressPrinter {
    fn render(&self, p: &RungProgress, fps: Option<f64>) -> String {
        let elapsed = self.started.elapsed();
        let mut s = format!(
            "  [{:>6}] {:<6} {:>5.1}%",
            p.label,
            super::status_str(p.status),
            p.percent
        );

        match p.frames_total {
            Some(total) if total > 0 => {
                s.push_str(&format!("  {}/{} frames", p.frames_done, total))
            }
            _ => s.push_str(&format!("  {} frames", p.frames_done)),
        }

        if let Some(r) = fps.filter(|r| *r > 0.0) {
            s.push_str(&format!("  {r:.0} fps"));
        }

        s.push_str(&format!("  elapsed {}", hms(elapsed)));

        // ETA needs a known total and a rate to divide by; without either,
        // print nothing rather than a fabricated number.
        if let (Some(total), Some(rate)) = (p.frames_total, fps) {
            if rate > 0.0 && total > p.frames_done {
                let remaining = Duration::from_secs_f64((total - p.frames_done) as f64 / rate);
                s.push_str(&format!("  eta {}", hms(remaining)));
                s.push_str(&format!("  total ~{}", hms(elapsed + remaining)));
            }
        }

        // Size to date, and where it's heading. `bytes_out` is 0 until the
        // first packets land (and for the whole run in HLS mode, which writes
        // segments to disk), so treat 0 as "not known yet" and say nothing.
        if p.bytes_out > 0 {
            s.push_str(&format!("  {}", size(p.bytes_out)));
            if let Some(total) = p.frames_total {
                if p.frames_done > 0 && total > p.frames_done {
                    let projected =
                        (p.bytes_out as f64 / p.frames_done as f64 * total as f64) as u64;
                    s.push_str(&format!(" → ~{}", size(projected)));
                }
            }
        }

        if let Some(m) = &p.message {
            s.push_str(&format!("  ({m})"));
        }
        s
    }
}

/// `h:mm:ss`, dropping the hours field when it's zero.
fn hms(d: Duration) -> String {
    let secs = d.as_secs();
    let (h, m, s) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}

/// Human byte size — binary units, matching the job summary.
fn size(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    let b = bytes as f64;
    if b >= KIB * KIB * KIB {
        format!("{:.2} GiB", b / (KIB * KIB * KIB))
    } else if b >= KIB * KIB {
        format!("{:.1} MiB", b / (KIB * KIB))
    } else if b >= KIB {
        format!("{:.0} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress(frames_done: u64, frames_total: Option<u64>, bytes_out: u64) -> RungProgress {
        RungProgress {
            rung_index: 0,
            label: "1080p".into(),
            width: 1920,
            height: 1080,
            status: RungStatus::Running,
            percent: frames_total
                .map(|t| frames_done as f32 / t as f32 * 100.0)
                .unwrap_or(0.0),
            frames_done,
            frames_total,
            segments_written: 0,
            bytes_out,
            message: None,
        }
    }

    #[test]
    fn hms_formats_and_drops_empty_hours() {
        assert_eq!(hms(Duration::from_secs(0)), "0:00");
        assert_eq!(hms(Duration::from_secs(9)), "0:09");
        assert_eq!(hms(Duration::from_secs(75)), "1:15");
        assert_eq!(hms(Duration::from_secs(3600)), "1:00:00");
        assert_eq!(hms(Duration::from_secs(3661)), "1:01:01");
        assert_eq!(hms(Duration::from_secs(37 * 3600 + 61)), "37:01:01");
    }

    #[test]
    fn size_picks_a_readable_unit() {
        assert_eq!(size(512), "512 B");
        assert_eq!(size(2048), "2 KiB");
        assert_eq!(size(5 * 1024 * 1024), "5.0 MiB");
        assert_eq!(size(3 * 1024 * 1024 * 1024), "3.00 GiB");
    }

    #[test]
    fn eta_and_projection_appear_once_a_rate_is_known() {
        let pr = ProgressPrinter::new(1);
        // 1000 of 4000 frames at 100 fps => 30 s remaining.
        let line = pr.render(&progress(1000, Some(4000), 10_000_000), Some(100.0));
        assert!(line.contains("1000/4000 frames"), "{line}");
        assert!(line.contains("100 fps"), "{line}");
        assert!(line.contains("eta 0:30"), "{line}");
        assert!(line.contains("total ~"), "{line}");
        // 10 MB for a quarter of the frames projects to ~40 MB.
        assert!(line.contains("9.5 MiB"), "size so far: {line}");
        assert!(line.contains("→ ~38.1 MiB"), "projection: {line}");
    }

    #[test]
    fn no_rate_means_no_invented_eta() {
        let pr = ProgressPrinter::new(1);
        let line = pr.render(&progress(0, Some(4000), 0), None);
        assert!(!line.contains("eta"), "{line}");
        assert!(!line.contains("total ~"), "{line}");
        assert!(line.contains("elapsed"), "elapsed is always known: {line}");
    }

    #[test]
    fn unknown_total_still_reports_rate_and_elapsed() {
        let pr = ProgressPrinter::new(1);
        let line = pr.render(&progress(500, None, 0), Some(50.0));
        assert!(line.contains("500 frames"), "{line}");
        assert!(line.contains("50 fps"), "{line}");
        assert!(!line.contains("eta"), "no total, so no ETA: {line}");
    }

    #[test]
    fn zero_bytes_prints_no_size_at_all() {
        // HLS reports 0 for the whole run; 0 B would read as a real measurement.
        let pr = ProgressPrinter::new(1);
        let line = pr.render(&progress(500, Some(1000), 0), Some(50.0));
        assert!(!line.contains(" B"), "{line}");
        assert!(!line.contains("MiB"), "{line}");
    }

    #[test]
    fn a_finished_rung_projects_nothing() {
        let pr = ProgressPrinter::new(1);
        let line = pr.render(&progress(4000, Some(4000), 40_000_000), Some(100.0));
        assert!(!line.contains("eta"), "nothing left to wait for: {line}");
        assert!(!line.contains("→"), "no projection at 100%: {line}");
        assert!(line.contains("38.1 MiB"), "final size still shown: {line}");
    }

    #[test]
    fn multi_rung_does_not_rewrite_in_place() {
        // Two rungs interleaving on one line would be unreadable.
        assert!(!ProgressPrinter::new(3).inplace);
    }
}
