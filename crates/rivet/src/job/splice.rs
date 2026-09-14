use bytes::Bytes;

use super::audio::PreparedAudio;

/// One clip of a [splice](super::run_splice_job): an input plus an optional
/// `[start, end)` trim window in seconds (either bound `None` = open).
#[derive(Clone)]
pub struct Clip {
    pub input: Bytes,
    pub start: Option<f64>,
    pub end: Option<f64>,
}

impl Clip {
    /// A whole clip, no trim.
    pub fn new(input: impl Into<Bytes>) -> Self {
        Self { input: input.into(), start: None, end: None }
    }

    /// A clip trimmed to `[start, end)` seconds (either bound `None` = open).
    pub fn trimmed(input: impl Into<Bytes>, start: Option<f64>, end: Option<f64>) -> Self {
        Self { input: input.into(), start, end }
    }
}

/// Convert a trim time (seconds) to a half-open source frame index at `fps`
/// (`ceil`, so `[start,end)` is exact for non-integer fps). `None` → `None`.
pub(super) fn trim_frame(sec: Option<f64>, fps: f64) -> Option<u64> {
    sec.map(|s| (s.max(0.0) * fps).ceil() as u64)
}

/// Trim a prepared audio track to the window `[start, end)` seconds, dropping
/// packets outside it. Kept packets retain their explicit durations, so the
/// muxer re-times them from zero — aligning with the trimmed, rebased video.
/// Cut points land on packet boundaries (≤ ~20 ms), which is fine for A/V sync.
/// `None`/`None` returns the track unchanged.
pub(super) fn trim_audio(
    audio: Option<&PreparedAudio>,
    start: Option<f64>,
    end: Option<f64>,
) -> Option<PreparedAudio> {
    let a = audio?;
    if start.is_none() && end.is_none() {
        return Some(a.clone());
    }
    let ticks_per_sec = a.info.timescale.max(1) as f64;
    let start_tick = (start.unwrap_or(0.0).max(0.0) * ticks_per_sec) as u64;
    let end_tick = end.map(|e| (e.max(0.0) * ticks_per_sec) as u64);
    if !a.edit.is_identity() {
        // The track carries an edit (priming, a source trim, a late start):
        // its samples sit on the presentation through it, so the trim window is
        // a window on that presentation — cut exactly, by the same arithmetic
        // that applied the source's edit.
        let durations: Vec<u32> = a.samples.iter().map(|(_, d)| *d).collect();
        let total: u64 = durations.iter().map(|&d| u64::from(d)).sum();
        let window = a.edit.window(total, start_tick, end_tick);
        let preroll = container::edit::AudioPreroll::for_codec(&a.info.codec, a.info.timescale);
        let cut = container::edit::cut_audio_packets(&durations, &window, preroll);
        return Some(PreparedAudio {
            info: a.info.clone(),
            samples: a.samples[cut.packets].to_vec(),
            handling: a.handling.clone(),
            edit: cut.edit,
        });
    }
    let mut acc: u64 = 0;
    let mut kept = Vec::new();
    for (payload, dur) in &a.samples {
        let sample_start = acc;
        acc += *dur as u64;
        if sample_start < start_tick {
            continue;
        }
        if end_tick.is_some_and(|et| sample_start >= et) {
            break;
        }
        kept.push((payload.clone(), *dur));
    }
    Some(PreparedAudio {
        info: a.info.clone(),
        samples: kept,
        handling: a.handling.clone(),
        edit: container::edit::TrackEdit::default(),
    })
}
