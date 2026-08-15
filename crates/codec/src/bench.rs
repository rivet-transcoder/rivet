//! Encode the same frames several ways and say which setting won.
//!
//! # What this is for
//!
//! Every quality decision in an ABR ladder is a guess until somebody encodes
//! the content both ways and looks. The guesses are usually drawn from
//! published BD-rate figures, which describe *some* corpus and not the clip in
//! front of you — and the gap between them is large enough to invert a
//! decision. A ladder policy tuned that way spent 23% of its largest rung
//! buying 0.62 VMAF, which nobody can see, because the reasoning was sound and
//! the content disagreed.
//!
//! So: take a slice of already-decoded frames, encode it once per candidate,
//! decode each result, score it against the frames it came from, and return
//! the table. The caller picks.
//!
//! # It measures the rung, not the encoder
//!
//! When `base` names dimensions smaller than the frames, each frame is scaled
//! down to encode and the decode is scaled back up before scoring. That is
//! deliberate and it is the whole difference between a useful number and a
//! misleading one: a 720p rung is not watched at 720p, it is watched stretched
//! to the screen, and scoring at the rung's own size hides the scaling loss —
//! which is the dominant loss for every rung below source and grows as the rung
//! shrinks.
//!
//! An earlier version scored at native size. It certified a quality shift at
//! SSIM 0.9755 against a 0.970 floor and the delivered rung measured 0.9619 —
//! *below* the floor the sweep had promised — because the number described an
//! encode the ladder never performs. A floor is only worth setting if the
//! measurement predicts what ships.
//!
//! # Why a slice rather than the whole thing
//!
//! A sweep of six candidates over a five-minute source is thirty minutes of
//! encoding to answer a question a few seconds of it usually settles. The
//! caller chooses the slice, because which seconds are representative is a
//! property of the content — the first two seconds of a fade-in are a bad
//! sample of anything, and this module has no way to know that.
//!
//! # What it does not do
//!
//! It does not choose for you. There is no "optimal" without a target, and the
//! target belongs to the caller: minimum bytes at a quality floor, maximum
//! quality under a byte ceiling, or the knee of the curve. [`Sweep::best_at_or_above`]
//! and [`Sweep::knee`] are the two that come up most; anything else reads the
//! rows directly.

use anyhow::{Context, Result};

use crate::decode;
use crate::encode::{self, EncoderConfig};
use crate::frame::VideoFrame;
use crate::quality::{self, Score};

/// One candidate's result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// The quality delta this candidate was encoded with, in
    /// libaom-CQ-equivalent steps — the same currency
    /// [`crate::encode::tuning::EncodeOverrides::quality_delta`] uses.
    pub quality_delta: i16,
    /// Encoded size of the slice, in bytes.
    pub bytes: u64,
    /// Mean luma PSNR across the slice, in dB.
    pub psnr: f64,
    /// Mean luma SSIM across the slice.
    pub ssim: f64,
    /// How many packets the encoder produced.
    pub packets: usize,
    /// Size of the largest packet — in a normal GOP this is the keyframe.
    pub largest_packet: u64,
    /// Mean size of every packet except the largest.
    ///
    /// Together with `largest_packet` this says what shape the encode was. A
    /// mean close to the largest means every frame cost about the same, which
    /// means intra-only — and an intra-only sample cannot represent a ladder
    /// whose rungs are mostly inter frames, however its config reads.
    pub mean_other_packet: u64,
}

impl Sample {
    /// Bytes per SSIM point above a floor — lower is better value.
    ///
    /// The floor exists because SSIM near 1.0 makes the denominator tiny and
    /// the ratio meaningless; comparing candidates by "bytes per unit of
    /// quality *worth having*" needs a baseline for worth having.
    pub fn bytes_per_ssim_above(&self, floor: f64) -> f64 {
        let headroom = (self.ssim - floor).max(f64::EPSILON);
        self.bytes as f64 / headroom
    }

    /// SSIM expressed in dB: `-10 · log10(1 − SSIM)`.
    ///
    /// # Why raw SSIM is the wrong scale to set a threshold in
    ///
    /// SSIM saturates. The interesting range is all crammed against 1.0, and
    /// the distance from 0.98 to 0.99 is not remotely the same amount of
    /// quality as 0.90 to 0.91 — it is roughly twice as much, because what
    /// halved was the *error*. A threshold in raw SSIM therefore means
    /// something different at every quality level, which is exactly how a
    /// single number ends up describing two clips wrongly at once.
    ///
    /// Working in dB fixes the scale: it is a log of the residual error, so a
    /// fixed drop is a fixed proportion of error added wherever you start
    /// from. Measured on two clips at the same encoder settings:
    ///
    /// | | SSIM | dB |
    /// |---|---|---|
    /// | flat animation | 0.9989 | 29.6 |
    /// | noisy source | 0.9804 | 17.1 |
    ///
    /// Twelve dB apart, and raw SSIM makes them look like neighbours.
    ///
    /// Capped at 60 dB so a perfect slice does not return infinity and poison
    /// a comparison — beyond that the residual is far below anything a display
    /// can show.
    pub fn ssim_db(&self) -> f64 {
        let error = (1.0 - self.ssim).max(1e-6);
        (-10.0 * error.log10()).min(60.0)
    }
}

/// Every candidate's result, in the order they were encoded.
#[derive(Debug, Clone, Default)]
pub struct Sweep {
    pub samples: Vec<Sample>,
}

impl Sweep {
    /// The smallest candidate whose SSIM is at or above `floor`.
    ///
    /// The usual question — "how few bytes can this clip take and still look
    /// acceptable" — with the caller naming what acceptable means. `None` when
    /// nothing cleared the floor, which is a real answer and not an error: it
    /// says this content needs more bits than any candidate offered.
    pub fn best_at_or_above(&self, floor: f64) -> Option<&Sample> {
        self.samples
            .iter()
            .filter(|s| s.ssim >= floor)
            .min_by_key(|s| s.bytes)
    }

    /// The candidate encoded at the configured base — the `delta == 0` row.
    ///
    /// This is the reference every relative judgement is made against: it is
    /// what the ladder would have shipped without a sweep at all.
    pub fn base(&self) -> Option<&Sample> {
        self.samples.iter().find(|s| s.quality_delta == 0)
    }

    /// The cheapest candidate that gives up no more than `max_drop_db` of
    /// quality against this clip's *own* base encode.
    ///
    /// # Why relative, and not a fixed target
    ///
    /// An absolute floor assumes every clip can reach it and that reaching it
    /// means the same thing everywhere. Neither holds. Measured at identical
    /// encoder settings, flat animation delivered SSIM 0.9989 / VMAF 99.4 and
    /// a noisy source delivered 0.9804 / VMAF 81.7 — so a floor of 0.970 was
    /// "plenty of room to spare" for one and "already worse than we ship" for
    /// the other. Set low enough for the hard clip to pass, it lets the easy
    /// clip be destroyed; set high enough to protect the easy clip, the hard
    /// one can never clear it and never gets measured at all.
    ///
    /// Asking instead "how much worse than this clip's own best is this?"
    /// removes the assumption. Every clip is judged against what it can
    /// actually achieve, the budget means the same thing at both ends of the
    /// range, and no calibration table is needed.
    ///
    /// `absolute_floor` is a safety net, not the criterion: a clip whose base
    /// is already poor should not be allowed to give up its budget on top.
    /// Pass `None` to judge purely on the drop.
    ///
    /// `None` when there is no base row to compare against, or when nothing —
    /// not even the base itself — satisfies the constraints.
    pub fn best_within_drop(
        &self,
        max_drop_db: f64,
        absolute_floor: Option<f64>,
    ) -> Option<&Sample> {
        let base_db = self.base()?.ssim_db();
        let allowed = base_db - max_drop_db.max(0.0);

        self.samples
            .iter()
            .filter(|s| s.ssim_db() >= allowed)
            .filter(|s| absolute_floor.is_none_or(|floor| s.ssim >= floor))
            .min_by_key(|s| s.bytes)
    }

    /// The rate-distortion optimal candidate for a given `lambda`.
    ///
    /// Minimises `D + λ·R`, where `D` is distortion (`1 − SSIM`) and `R` is
    /// bytes per pixel of the sample. This is the standard Lagrangian
    /// formulation: the chosen point is where the rate-quality curve's slope
    /// equals `−λ`, which is the "encode at a consistent bitrate-quality
    /// slope" strategy the per-title literature names alongside targeting an
    /// absolute VMAF.
    ///
    /// # Why this and not a threshold
    ///
    /// A threshold — absolute or relative to the clip's own base — has to
    /// answer "how good is good enough", and that question has no single
    /// answer across content. Two attempts at it failed here in exactly
    /// opposite ways: an absolute SSIM floor was loose for flat animation and
    /// unreachable for a noisy source, and a relative dB budget then
    /// over-penalised near-lossless samples because dB is unbounded as SSIM
    /// approaches 1.
    ///
    /// A slope asks a different question — "is the next byte worth it" — which
    /// does have one answer, and it is scale-free:
    ///
    /// * On content where quality barely moves between candidates, `D` is
    ///   near-constant, the expression collapses to `λ·R`, and the cheapest
    ///   candidate wins. That is the correct answer for trivially compressible
    ///   content and it is precisely what both thresholds got wrong.
    /// * On content where quality falls steeply, `D` dominates and the
    ///   expression stops paying for the saving early.
    ///
    /// One constant, opposite answers, for the right reason.
    ///
    /// # Rate is normalised per pixel
    ///
    /// So a `λ` calibrated on one sample size or resolution keeps its meaning
    /// on another. Without it, doubling the sample length halves the effective
    /// λ and silently changes every decision.
    ///
    /// `pixels_per_frame` is the encoded size of one frame of the sample;
    /// `frames` how many were encoded. `None` when the sweep is empty or those
    /// are zero, because dividing by them is the whole point.
    pub fn rd_optimal(&self, lambda: f64, pixels_per_frame: u64, frames: usize) -> Option<&Sample> {
        let pixels = (pixels_per_frame as f64) * (frames as f64);
        if self.samples.is_empty() || pixels <= 0.0 {
            return None;
        }

        self.samples
            .iter()
            .min_by(|a, b| {
                let cost = |s: &Sample| (1.0 - s.ssim) + lambda * (s.bytes as f64 / pixels);
                cost(a).total_cmp(&cost(b))
            })
    }

    /// Where spending more bytes stops buying much quality.
    ///
    /// The largest drop in bytes-per-SSIM between adjacent candidates, which
    /// on a normal rate-quality curve is its knee. Requires the sweep to be
    /// ordered by `quality_delta`, which [`sweep`] guarantees.
    ///
    /// A knee is a heuristic, not a fact: on flat content every candidate
    /// looks the same and the answer is arbitrary. Check the rows when it
    /// matters.
    pub fn knee(&self) -> Option<&Sample> {
        if self.samples.len() < 3 {
            return None;
        }
        let floor = self
            .samples
            .iter()
            .map(|s| s.ssim)
            .fold(f64::INFINITY, f64::min)
            - 0.01;

        self.samples
            .windows(2)
            .map(|pair| {
                let gain = pair[0].bytes_per_ssim_above(floor) - pair[1].bytes_per_ssim_above(floor);
                (gain, &pair[1])
            })
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, sample)| sample)
    }
}

/// Encode `frames` once per candidate delta and score each result, in series.
///
/// Convenient, and the wrong shape for a fleet: candidates are completely
/// independent encodes of the same frames, so a host with four GPUs should be
/// running four of them at once rather than queueing behind one. An
/// orchestrator that owns a lease pool should call [`measure_candidate`] per
/// candidate and collect the results — see the transcoder's per-title pass.
/// This remains for callers with one device and for tests.
///
/// `base` supplies everything except the quality: dimensions, codec, frame
/// rate, GPU choice. Its `overrides.quality_delta` is replaced per candidate,
/// so a caller can pin tiles, references or a speed tier across the sweep and
/// vary only the one axis.
///
/// Candidates are sorted before encoding so [`Sweep::knee`] can rely on the
/// ordering, and duplicates are dropped — encoding the same setting twice
/// costs real time and tells you nothing new.
pub fn sweep(base: &EncoderConfig, frames: &[VideoFrame], candidates: &[i16]) -> Result<Sweep> {
    if frames.is_empty() {
        return Ok(Sweep::default());
    }

    let mut deltas = candidates.to_vec();
    deltas.sort_unstable();
    deltas.dedup();

    let mut samples = Vec::with_capacity(deltas.len());
    for delta in deltas {
        samples.push(measure(base, frames, delta)?);
    }

    Ok(Sweep { samples })
}

/// One candidate: encode the slice, decode it back, score it.
///
/// Public because the parallel driver lives in the layer that owns GPUs. This
/// crate deliberately does not: it knows how to encode on a device it is
/// handed and nothing about how many there are or who else wants them.
///
/// Pin the device through `base.gpu_index` / `base.gpu_vendor` before calling,
/// or every candidate lands on whichever card the dispatch chain picks first
/// and the fan-out is a queue with extra steps.
pub fn measure_candidate(base: &EncoderConfig, frames: &[VideoFrame], delta: i16) -> Result<Sample> {
    measure(base, frames, delta)
}

fn measure(base: &EncoderConfig, frames: &[VideoFrame], delta: i16) -> Result<Sample> {
    let mut config = base.clone();
    config.overrides.quality_delta = delta;

    let mut encoder = encode::select_encoder(config.clone(), None)
        .with_context(|| format!("creating an encoder for delta {delta}"))?;

    let mut payload = Vec::new();
    let mut packet_sizes: Vec<u64> = Vec::new();
    for frame in frames {
        // Scaled to the configured size first, so a caller measuring a rung
        // measures the rung. Feeding source-sized frames to a rung-sized
        // config would either be refused or silently measure something the
        // ladder never encodes.
        let source = if frame.width == config.width && frame.height == config.height {
            frame.clone()
        } else {
            crate::colorspace::scale_frame(frame, config.width, config.height)
                .with_context(|| format!("scaling to {}x{}", config.width, config.height))?
        };

        encoder.send_frame(&source).with_context(|| format!("encoding at delta {delta}"))?;
        while let Some(packet) = encoder.receive_packet()? {
            packet_sizes.push(packet.data.len() as u64);
            payload.extend_from_slice(&packet.data);
        }
    }
    encoder.flush()?;
    while let Some(packet) = encoder.receive_packet()? {
        packet_sizes.push(packet.data.len() as u64);
        payload.extend_from_slice(&packet.data);
    }

    let bytes = payload.len() as u64;

    // Decode the bitstream back and score against what went in. Scoring the
    // encoder's own reconstruction would be easier and would measure nothing:
    // it is the decoded picture a viewer sees, and a bitstream that decodes
    // differently from what the encoder thought it wrote is precisely the
    // class of bug worth catching here.
    let info = crate::frame::StreamInfo {
        codec: codec_label(config.codec).to_string(),
        width: config.width,
        height: config.height,
        frame_rate: config.frame_rate,
        duration: 0.0,
        pixel_format: config.pixel_format,
        color_space: frames[0].color_space,
        total_frames: frames.len() as u64,
        bitrate: 0,
        color_metadata: config.color_metadata,
    };
    let mut decoder = decode::create_decoder(codec_label(config.codec), info)
        .with_context(|| format!("creating a decoder for delta {delta}"))?;
    decoder.push_sample(&payload)?;

    let (mut psnr, mut ssim, mut scored) = (0.0f64, 0.0f64, 0usize);
    while let Some(decoded) = decoder.decode_next()? {
        let Some(reference) = frames.get(scored) else { break };

        // Scaled back up to the reference before scoring, because that is what
        // a viewer sees: a 720p rung is not watched at 720p, it is watched
        // stretched to the screen. Scoring at the rung's own size measures only
        // the encoder and hides the scaling loss entirely — which is the
        // dominant loss for every rung below source, and grows as the rung
        // shrinks.
        //
        // Getting this wrong is not academic. A sweep that scored at native
        // size certified a quality shift at SSIM 0.9755 against a 0.970 floor
        // and delivered 0.9619 — below the floor it had promised — because the
        // number described an encode the ladder never performs.
        let shown = if decoded.width == reference.width && decoded.height == reference.height {
            decoded
        } else {
            crate::colorspace::scale_frame(&decoded, reference.width, reference.height)
                .with_context(|| format!("scaling back to {}x{}", reference.width, reference.height))?
        };

        if let Some(Score { psnr: p, ssim: s }) = quality::score_frame(reference, &shown) {
            // A perfect frame is infinite PSNR, and one of those turns the
            // mean into infinity — which says "this clip is lossless" on the
            // strength of a single flat frame. Capped at the value beyond
            // which the difference is not a difference anybody can see.
            psnr += p.min(100.0);
            ssim += s;
            scored += 1;
        }
    }

    if scored == 0 {
        anyhow::bail!("delta {delta} produced no decodable frames to score");
    }

    let packets = packet_sizes.len();
    let largest_packet = packet_sizes.iter().copied().max().unwrap_or(0);
    let mean_other_packet = if packets > 1 {
        (packet_sizes.iter().sum::<u64>() - largest_packet) / (packets as u64 - 1)
    } else {
        0
    };

    Ok(Sample {
        quality_delta: delta,
        bytes,
        packets,
        largest_packet,
        mean_other_packet,
        psnr: psnr / scored as f64,
        ssim: ssim / scored as f64,
    })
}

fn codec_label(codec: crate::frame::VideoCodec) -> &'static str {
    match codec {
        crate::frame::VideoCodec::Av1 => "av1",
        crate::frame::VideoCodec::H264 => "h264",
        crate::frame::VideoCodec::H265 => "hevc",
    }
}

#[cfg(test)]
mod tests;
