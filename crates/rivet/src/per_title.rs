//! Per-title quality: ask the content how many bits it needs, instead of
//! telling it.
//!
//! # Why
//!
//! Every rung in a ladder is encoded at a quality derived from two enums and
//! a per-rung offset, and none of that has ever looked at the video. Easy
//! content — flat animation, a screen recording, a locked-off shot — gets the
//! same quantizer as a handheld night scene, so one of them is always wrong:
//! either the easy clip is spending bits nobody can see, or the hard one is
//! being starved.
//!
//! The fix is the oldest idea in per-title encoding: encode a sample of *this*
//! clip at several settings, measure, and keep the cheapest that still looks
//! acceptable. [`codec::bench`] does the encoding and scoring; this module
//! decides **what to sample** ([`sample_frames`]), runs the candidates
//! **across the GPU pool** ([`sweep_on_pool`]), and turns the answer into a
//! ladder-wide [`EncodeOverrides`] ([`select_shift`]). A caller composes the
//! three — with its own reporting between them — or calls
//! [`choose_quality_shift`] for the whole thing.
//!
//! What it deliberately does not do: vary the ladder's *shape* — the rungs,
//! their resolutions, or the per-rung steps between them. Those are decisions
//! about what devices and networks exist, not about this clip. It shifts the
//! whole ladder up or down by one number, which is the part the content has
//! an opinion about. Netflix-style per-title builds a convex hull over
//! (resolution, rate, quality); this is a deliberate subset of that.
//!
//! # Which seconds get sampled
//!
//! One contiguous run the length of a segment, from the middle, checked
//! against the pixels rather than trusted. A clip that fades in from black
//! opens on frames that cost nothing at every candidate and score
//! near-perfectly against themselves; a sweep taken there concludes the
//! content is free and shifts the whole ladder cheaper on the strength of a
//! fade. Windows that read as blank are skipped — see [`sample_frames`].
//!
//! # Why SSIM and not VMAF
//!
//! VMAF is a trained model and its coefficient file; putting it in a worker
//! means vendoring libvmaf. For *ranking candidates on one clip* — which is
//! the only thing this does — SSIM orders them the same way, and the absolute
//! number never leaves this module.

use std::sync::Arc;

use anyhow::Result;
use bytes::Bytes;
use tokio::task::JoinSet;

use codec::bench::{self, Sample, Sweep};
use codec::encode::EncoderConfig;
use codec::encode::tuning::EncodeOverrides;
use codec::frame::VideoFrame;
use container::streaming::{self, DemuxHeader};

use crate::gpu_pool::GpuPool;

/// How the sample is drawn from the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleSpec {
    /// How many frames to sample. More is a better estimate and a longer wait.
    ///
    /// The default is one segment at 30 fps, sized to the ladder's own segment
    /// rather than to a convenient duration, because the sample's byte count
    /// is only meaningful if its GOP matches what ships. At 60 frames the
    /// sample was mostly keyframes: 3,366 bytes per frame against the
    /// delivered rung's 220, and a size that moved only 1.9% across the whole
    /// quality range while the real ladder moved 42%.
    pub frames: usize,
    /// How many separate places in the clip the sample is drawn from.
    ///
    /// One, and the reasoning went both ways before landing here. Three
    /// windows were tried, to stop a static scene standing in for a whole
    /// video. They fixed that and broke something worse: every window boundary
    /// is a scene cut, so a 60-frame sample carried three keyframes against
    /// the real encode's one per six seconds, and its bytes stopped responding
    /// to quality at all. Rate is the harder signal to get right, so the sample
    /// is one contiguous run the length of a segment — the same GOP the ladder
    /// uses — and the length is what keeps a single static moment from
    /// dominating it.
    pub windows: usize,
    /// How deep into the source the sample may be taken from, in frames.
    ///
    /// The window wants the middle; this is what it is allowed to spend
    /// getting there. Without an index every skipped frame is a decoded frame,
    /// so on a long source "the middle" would mean decoding half the clip
    /// before the sweep even starts. 900 frames is around 30 seconds at
    /// 30 fps: past any intro or fade, and a bounded cost on the critical path
    /// regardless of source length.
    pub max_skip_frames: usize,
}

impl Default for SampleSpec {
    fn default() -> Self {
        Self { frames: 180, windows: 1, max_skip_frames: 900 }
    }
}

/// The candidate deltas tried by default, in libaom-CQ-equivalent steps
/// around the configured base.
///
/// Asymmetric on purpose: the interesting direction is *cheaper*, because the
/// base is already tuned for hard content and most content is not hard. Two
/// steps of headroom upward cover the clips that genuinely need more.
pub const DEFAULT_CANDIDATES: [i16; 6] = [-2, 0, 2, 4, 6, 8];

/// Decode a representative sample of the source: `spec.frames` frames, in
/// `spec.windows` windows spread through the reachable span, skipping the
/// very start (where fades live) and any window that reads as blank.
///
/// # Why the seek is a decode
///
/// There is no seek. The demuxer has no index, so the only way to arrive at
/// frame *n* is to decode to it. The cost is bounded by
/// [`SampleSpec::max_skip_frames`] rather than by the clip, so a long source
/// samples as deep as the budget reaches instead of paying for half a decode
/// on the critical path.
///
/// Blank windows are dropped — see [`codec::quality::window_looks_blank`] —
/// so a clip that fades in from black does not have the fade averaged into
/// its measurement. If every window is blank the least blank one is kept,
/// because an empty sample falls back to the base quality and a genuinely
/// dark clip still deserves a number.
///
/// Frames come back the right way up: the container's rotation is applied,
/// as it is to every frame the ladder encodes, so the sample is measured at
/// the picture's real shape.
pub fn sample_frames(
    input: &Bytes,
    header: &DemuxHeader,
    spec: &SampleSpec,
) -> Result<Vec<VideoFrame>> {
    let total = header.info.total_frames as usize;
    let mut demuxer = streaming::demux_streaming(input)?;
    let decoder = codec::decode::create_decoder(&header.codec, header.info.clone())?;
    let mut decoder = codec::decode::RotatingDecoder::new(decoder, header.rotation_degrees);

    let wanted = spec.frames.max(1);
    let windows = spec.windows.max(1);
    let per_window = wanted.div_ceil(windows).max(1);

    // How far in the sample may reach: the clip, capped by the decode budget.
    // Beyond that the windows would be spread across frames this pass will
    // never reach.
    let reach = total.min(spec.max_skip_frames + wanted).max(per_window);

    // Spread across the reachable span, skipping the very start. The opening
    // is where fades live, and it is also the least representative part of
    // most clips.
    let mut starts: Vec<usize> = (0..windows)
        .map(|i| {
            let span = reach.saturating_sub(per_window);
            span * (i + 1) / (windows + 1)
        })
        .collect();
    starts.dedup();

    let mut kept: Vec<VideoFrame> = Vec::with_capacity(wanted);
    let mut fallback: Option<(f64, Vec<VideoFrame>)> = None;
    let mut current: Vec<VideoFrame> = Vec::with_capacity(per_window);
    let mut idx = 0usize;
    let mut window_i = 0usize;

    'demux: loop {
        let Some(sample) = demuxer.next_video_sample()? else { break };
        decoder.push_sample(&sample.data)?;

        while let Some(frame) = decoder.decode_next()? {
            let here = idx;
            idx += 1;

            let Some(&begin) = starts.get(window_i) else { break 'demux };
            if here < begin {
                continue;
            }

            current.push(frame);
            if current.len() < per_window {
                continue;
            }

            // A window is judged whole: a majority of blank frames means it is
            // a fade, not content, and averaging it in would tell the sweep
            // the clip is cheaper than it is.
            let blank = codec::quality::blank_fraction(&current);
            if codec::quality::window_looks_blank(&current) {
                tracing::debug!(begin, blank, "per-title: a sample window reads as blank; dropping it");
                if fallback.as_ref().is_none_or(|(worst, _)| blank < *worst) {
                    fallback = Some((blank, std::mem::take(&mut current)));
                }
            } else {
                kept.append(&mut current);
            }

            current.clear();
            window_i += 1;
            if window_i >= starts.len() {
                break 'demux;
            }
        }
    }

    // A tail shorter than a full window still carries content on a clip too
    // short to fill every slot.
    if !current.is_empty() && !codec::quality::window_looks_blank(&current) {
        kept.append(&mut current);
    }

    if kept.is_empty() {
        if let Some((blank, frames)) = fallback.filter(|(_, f)| !f.is_empty()) {
            tracing::warn!(blank, "per-title: every sample window looked blank; measuring the least blank one");
            return Ok(frames);
        }
    }

    tracing::debug!(windows = starts.len(), per_window, collected = kept.len(), "per-title: sample gathered");
    Ok(kept)
}

/// Encode every candidate at once, one GPU lease each, and return the sweep
/// ordered by delta.
///
/// The candidates are independent encodes of the same frames, which is the
/// same shape as the ladder itself: decode once, fan the work across every
/// GPU. So they take a lease each from the same pool the rungs use and run
/// concurrently, and a six-candidate sweep on a four-GPU host costs about
/// two encodes of wall-clock rather than six.
///
/// A lease is claimed *before* each task is spawned, and that ordering is the
/// backpressure: exactly as many candidates run as there are cards and the
/// rest queue here rather than in the driver. A candidate that fails is
/// dropped rather than failing the sweep — one setting the hardware refuses
/// should not cost the job its per-title pass; the remaining rows still
/// answer the question.
///
/// `on_candidate(done, total)` is called as each candidate lands (failed ones
/// count: the wait is over either way, and a bar that stalls on a rejected
/// setting is exactly what a progress callback exists to prevent).
pub async fn sweep_on_pool(
    base: &EncoderConfig,
    frames: Vec<VideoFrame>,
    deltas: &[i16],
    gpu_pool: &Arc<GpuPool>,
    mut on_candidate: impl FnMut(u32, u32),
) -> Result<Sweep> {
    let frames = Arc::new(frames);

    let mut tasks: JoinSet<Option<Sample>> = JoinSet::new();
    for &delta in deltas {
        let Some(lease) = Arc::clone(gpu_pool).claim().await else {
            tracing::warn!(delta, "per-title: no GPU available for a candidate; skipping it");
            continue;
        };

        let frames = Arc::clone(&frames);
        let mut config = base.clone();
        config.gpu_index = Some(lease.gpu_index);
        config.gpu_vendor = Some(lease.vendor);

        tasks.spawn_blocking(move || {
            // The lease moves in and is dropped when this returns, which is
            // what hands the card to the next candidate — or to the ladder,
            // once the sweep is done.
            let _lease = lease;
            match bench::measure_candidate(&config, &frames, delta) {
                Ok(sample) => Some(sample),
                Err(e) => {
                    tracing::warn!(delta, error = %e, "per-title: a candidate failed; the sweep continues without it");
                    None
                }
            }
        });
    }

    let total = tasks.len() as u32;
    let mut samples = Vec::new();
    let mut done = 0u32;
    while let Some(joined) = tasks.join_next().await {
        done += 1;
        on_candidate(done, total);
        if let Ok(Some(sample)) = joined {
            samples.push(sample);
        }
    }

    // Ordered by delta, because the selection reads adjacent rows and the
    // join order is whatever the cards finished in.
    samples.sort_by_key(|s| s.quality_delta);
    Ok(Sweep { samples })
}

/// What the sweep decided.
#[derive(Debug, Clone)]
pub enum Selection {
    /// The cheapest candidate reaching the floor. `capped` means it was also
    /// the cheapest *offered*: the range decided the answer, not the content,
    /// and there is more saving this sweep cannot see.
    Chosen { sample: Sample, capped: bool },
    /// No candidate reached the floor; the clip keeps its base quality — the
    /// honest answer for content that has nothing to give.
    KeptBase,
}

impl Selection {
    /// The ladder-wide overrides this selection amounts to. `None` for
    /// [`Selection::KeptBase`] and for a chosen delta of zero.
    pub fn overrides(&self) -> Option<EncodeOverrides> {
        match self {
            Selection::Chosen { sample, .. } if sample.quality_delta != 0 => {
                Some(EncodeOverrides { quality_delta: sample.quality_delta, ..Default::default() })
            }
            _ => None,
        }
    }
}

/// Pick from a sweep: the cheapest candidate whose sample SSIM reaches
/// `floor`, judged on quality alone.
///
/// Quality alone, because the rate signal from a hardware backend cannot be
/// trusted here — some hand back fixed-size packets, so every candidate's
/// byte count is the same buffer, and a rate-distortion choice made against
/// a constant rate minimises distortion alone and picks the dearest candidate
/// every time. The quality signal from the same encodes is sound.
///
/// `deltas` are the candidates that were *offered* (sorted, deduped), so a
/// winner sitting at the cheap end can be reported as capped: everything
/// offered cleared the floor and the cheapest on the list won because the
/// list stopped there. A capped result is indistinguishable from a considered
/// one otherwise, and a ladder quietly leaving savings on the table looks
/// exactly like a ladder that is done.
pub fn select_shift(sweep: &Sweep, floor: f64, deltas: &[i16]) -> Selection {
    match sweep.cheapest_reaching(floor) {
        Some(best) => {
            let capped = deltas.len() > 1 && deltas.last() == Some(&best.quality_delta);
            Selection::Chosen { sample: best.clone(), capped }
        }
        None => Selection::KeptBase,
    }
}

/// Everything a per-title pass needs to be told.
#[derive(Debug, Clone)]
pub struct PerTitleSpec {
    /// The sample SSIM the chosen candidate must reach; the cheapest reaching
    /// it wins. `0.9985` was measured on real encodes as the value that takes
    /// the saving flat content offers and refuses every candidate on noisy
    /// content that has nothing to give.
    pub floor: f64,
    /// Candidate deltas to sweep, in libaom-CQ-equivalent steps.
    pub candidates: Vec<i16>,
    /// How the sample is drawn.
    pub sample: SampleSpec,
}

/// A per-title pass, start to finish: sample, sweep, select. Returns the
/// ladder-wide shift, or `None` when the source yields no frames, nothing
/// clears the floor, or the winner is the base itself. Callers wanting to
/// report between the steps compose [`sample_frames`], [`sweep_on_pool`] and
/// [`select_shift`] themselves.
pub async fn choose_quality_shift(
    input: &Bytes,
    header: &DemuxHeader,
    base: &EncoderConfig,
    spec: &PerTitleSpec,
    gpu_pool: &Arc<GpuPool>,
) -> Result<Option<EncodeOverrides>> {
    let frames = sample_frames(input, header, &spec.sample)?;
    if frames.is_empty() {
        tracing::warn!("per-title: the source yielded no frames to sample; using the base quality");
        return Ok(None);
    }
    let mut deltas = spec.candidates.clone();
    deltas.sort_unstable();
    deltas.dedup();
    let sweep = sweep_on_pool(base, frames, &deltas, gpu_pool, |_, _| {}).await?;
    Ok(select_shift(&sweep, spec.floor, &deltas).overrides())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(delta: i16, ssim: f64) -> Sample {
        Sample {
            quality_delta: delta,
            bytes: 1000,
            trimmed_bytes: 1000,
            ssim,
            psnr: 40.0,
            packets: 10,
            largest_packet: 100,
            mean_other_packet: 90,
        }
    }

    #[test]
    fn the_cheapest_candidate_reaching_the_floor_wins() {
        let sweep = Sweep {
            samples: vec![sample(-2, 0.9999), sample(0, 0.9998), sample(2, 0.9990), sample(4, 0.9970)],
        };
        match select_shift(&sweep, 0.9985, &[-2, 0, 2, 4]) {
            Selection::Chosen { sample, capped } => {
                assert_eq!(sample.quality_delta, 2);
                assert!(!capped, "2 was not the cheapest offered");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_winner_at_the_edge_of_the_range_is_reported_as_capped() {
        // Everything cleared the floor, so the cheapest on the list won
        // because the list stopped there — the range decided, not the clip.
        let sweep = Sweep { samples: vec![sample(0, 0.9999), sample(4, 0.9995), sample(8, 0.9990)] };
        match select_shift(&sweep, 0.9985, &[0, 4, 8]) {
            Selection::Chosen { sample, capped } => {
                assert_eq!(sample.quality_delta, 8);
                assert!(capped);
                assert_eq!(
                    Selection::Chosen { sample: sample.clone(), capped }.overrides().unwrap().quality_delta,
                    8
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_clip_that_cannot_reach_the_floor_keeps_its_base() {
        let sweep = Sweep { samples: vec![sample(0, 0.98), sample(4, 0.97)] };
        let selection = select_shift(&sweep, 0.9985, &[0, 4]);
        assert!(matches!(selection, Selection::KeptBase));
        assert!(selection.overrides().is_none());
    }

    #[test]
    fn choosing_the_base_itself_is_no_override() {
        let sweep = Sweep { samples: vec![sample(0, 0.9999), sample(2, 0.9)] };
        let selection = select_shift(&sweep, 0.9985, &[0, 2]);
        assert!(matches!(selection, Selection::Chosen { .. }));
        assert!(selection.overrides().is_none(), "a zero shift is the base");
    }

    #[test]
    fn the_defaults_are_a_segments_worth_and_lean_cheaper() {
        let s = SampleSpec::default();
        assert!(s.frames >= 150, "a sample shorter than a segment cannot carry the ladder's GOP");
        assert_eq!(s.windows, 1, "extra windows mean extra keyframes");
        let cheaper = DEFAULT_CANDIDATES.iter().filter(|d| **d > 0).count();
        let dearer = DEFAULT_CANDIDATES.iter().filter(|d| **d < 0).count();
        assert!(cheaper > dearer);
        assert!(DEFAULT_CANDIDATES.contains(&0), "the base itself must be a candidate");
    }
}
