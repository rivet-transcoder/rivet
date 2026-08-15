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

/// Encode `frames` once per candidate delta and score each result.
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
fn measure(base: &EncoderConfig, frames: &[VideoFrame], delta: i16) -> Result<Sample> {
    let mut config = base.clone();
    config.overrides.quality_delta = delta;

    let mut encoder = encode::select_encoder(config.clone(), None)
        .with_context(|| format!("creating an encoder for delta {delta}"))?;

    let mut payload = Vec::new();
    for frame in frames {
        encoder.send_frame(frame).with_context(|| format!("encoding at delta {delta}"))?;
        while let Some(packet) = encoder.receive_packet()? {
            payload.extend_from_slice(&packet.data);
        }
    }
    encoder.flush()?;
    while let Some(packet) = encoder.receive_packet()? {
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
        if let Some(Score { psnr: p, ssim: s }) = quality::score_frame(reference, &decoded) {
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

    Ok(Sample {
        quality_delta: delta,
        bytes,
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
