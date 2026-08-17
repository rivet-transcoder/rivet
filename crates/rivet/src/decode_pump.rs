//! Shared source decode pump.
//!
//! One pump per job (not per rung): demux + decode the source **once**, run
//! the rung-agnostic per-frame work (4:4:4 → 4:2:0 downsample + HDR tonemap),
//! and fan the normalized frame out to N per-rung mpsc channels via cheap
//! `VideoFrame::clone()` (the inner `Bytes` is `Arc`-backed).
//!
//! Per-rung scaling + encoding consume from those channels. Eliminating the
//! redundant per-rung decode is the whole point — a 5-rung ladder decodes the
//! source once, not five times. The cost: the slowest rung backpressures the
//! pump (usually the largest rung, whose encoder is slowest).
//!
//! # Splitting the decode across GPUs
//!
//! One decoder for the whole ladder is one decoder, and on a multi-GPU host
//! every rung then waits on it while the other cards' decode engines sit
//! idle. [`plan_decode_ranges`] cuts the source into ranges that can each be
//! decoded from a keyframe on a segment boundary; the orchestrator runs one
//! pump per range ([`DecodePumpConfig::sample_range`]), each pinned to its own
//! card, so the cards decode different stretches of the source at the same
//! time and the ladder's segment numbering stays continuous across the join.

use std::time::Instant;

use anyhow::{Context, Result};
use bytes::Bytes;

use codec::frame::{ColorMetadata, PixelFormat, VideoFrame};
use codec::{colorspace, decode};
use container::streaming;

/// Configuration for one decode pump.
#[derive(Clone)]
pub struct DecodePumpConfig {
    /// Source video codec label (e.g. `"h264"`).
    pub codec_name: String,
    /// Stream info handed to the decoder.
    pub info_for_decoder: codec::frame::StreamInfo,
    /// Source color metadata (drives HDR-aware tonemap vs SDR passthrough).
    pub source_color_metadata: ColorMetadata,
    /// Source pixel format.
    pub source_pixel_format: PixelFormat,
    /// Whether to run the 4:4:4 → 4:2:0 downsample per frame.
    pub needs_downsample: bool,
    /// Tonemap policy (from the [`OutputSpec`](crate::spec::OutputSpec)): when
    /// `true`, HDR (PQ/HLG) sources are mapped down to 8-bit SDR BT.709; when
    /// `false`, the source color/transfer/bit-depth passes through unchanged.
    /// The pump does not decide this on its own — the caller sets it from the
    /// spec's [`ColorPolicy`](crate::spec::ColorPolicy).
    pub tonemap_to_sdr: bool,
    /// Pin the decoder to this physical GPU; `None` = first matching adapter.
    pub gpu_index: Option<u32>,
    /// Decode only `[start_sample, end_sample)` of the source, by demuxed
    /// sample index. `None` decodes everything, which is the whole-source
    /// pump.
    ///
    /// `start_sample` **must** be a sample [`plan_decode_ranges`] returned —
    /// that is, one carrying an IDR/IRAP. Starting anywhere else gives the
    /// decoder a picture whose references it never saw, and the output is
    /// wrong rather than absent.
    ///
    /// Samples before `start_sample` are still demuxed — they have to be, the
    /// demuxer is a pull API with no seek — but they are not handed to the
    /// decoder. Demuxing is parsing; decoding is the expensive half, and
    /// skipping it is the entire saving. Composes with a clip's trim window,
    /// which counts *decoded* frames: the range decides what is decoded, the
    /// trim decides what is kept.
    pub sample_range: Option<(u64, Option<u64>)>,
    /// Clockwise rotation the container declared, in degrees (0/90/180/270).
    ///
    /// Applied to every frame as it leaves the decoder, so nothing fed by this
    /// pump has to know the source was recorded on its side or upside down.
    /// See [`codec::decode::RotatingDecoder`]. Set it from
    /// [`DemuxHeader::rotation_degrees`](container::streaming::DemuxHeader) —
    /// and size the rungs from
    /// [`DemuxHeader::upright_dims`](container::streaming::DemuxHeader::upright_dims),
    /// because 90/270 swap the picture's width and height.
    pub rotation_degrees: u32,
    /// Prepared per-frame video filter chain (crop/pad/flip/rotate/grayscale/
    /// overlay/colour), applied after colorspace normalize and before the frame
    /// is fanned out to the per-rung scalers. Overlay images are loaded once at
    /// prepare time. `Arc` so the per-GPU pump configs clone it cheaply.
    pub filters: std::sync::Arc<codec::filter::FilterChain>,
}

/// One contiguous slice of the source, decodable without anything before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeRange {
    /// First sample of the range. Always a keyframe.
    pub start_sample: u64,
    /// One past the last sample, or `None` for "to the end of the source".
    pub end_sample: Option<u64>,
    /// Frames before this range — its first segment's index, given a
    /// `frames_per_chunk` that divides the boundary.
    pub start_frame: u64,
}

impl DecodeRange {
    /// The single range meaning "decode all of it" — what every path used
    /// before range-parallel decode existed, and the fallback whenever a
    /// source cannot be split safely.
    pub fn whole_source() -> Self {
        Self { start_sample: 0, end_sample: None, start_frame: 0 }
    }

    /// The `sample_range` a pump config takes for this range: `None` for the
    /// whole source (nothing to skip), the bounds otherwise.
    pub fn sample_range(&self) -> Option<(u64, Option<u64>)> {
        if self.start_sample == 0 && self.end_sample.is_none() {
            None
        } else {
            Some((self.start_sample, self.end_sample))
        }
    }
}

/// Split the source into at most `want` ranges that can be decoded in
/// parallel, one per GPU.
///
/// Returns `None` when the source cannot be split safely, and the caller
/// should decode it whole. That is the answer whenever:
///
/// - the codec is not one whose keyframes we can identify from the bitstream
///   (H.264 / H.265 today — see [`container::nal_mux::sample_is_keyframe`]),
/// - the source has too few keyframes to give every range one,
/// - or no keyframe lands on a multiple of `frames_per_chunk`.
///
/// # Why boundaries must land on a segment boundary
///
/// Each range's scaler groups `frames_per_chunk` frames into a segment and
/// numbers segments from a base. If a range began mid-segment, its first
/// segment would hold fewer frames than the rung's others, every rung would
/// have to make the same odd split for playback to stay aligned, and the base
/// index could no longer be computed as `start_frame / frames_per_chunk`.
/// Requiring the boundary to be both a keyframe *and* a multiple of the
/// segment length keeps segment numbering arithmetic and identical on every
/// rung.
///
/// One decoded frame per demuxed sample is assumed, which holds for the
/// progressive single-layer streams the pipeline accepts.
pub fn plan_decode_ranges(
    input_data: &Bytes,
    codec_name: &str,
    frames_per_chunk: u32,
    want: usize,
) -> Option<Vec<DecodeRange>> {
    if want <= 1 || frames_per_chunk == 0 {
        return None;
    }
    let codec = nal_codec_for(codec_name)?;

    // Index pass: demux only, no decode. Record which sample indices may start
    // a range and how many samples there are.
    let mut demuxer = streaming::demux_streaming(input_data).ok()?;
    let mut keyframes: Vec<u64> = Vec::new();
    let mut total: u64 = 0;
    while let Ok(Some(sample)) = demuxer.next_video_sample() {
        if container::nal_mux::sample_is_keyframe(&sample.data, codec) {
            keyframes.push(total);
        }
        total += 1;
    }
    if total == 0 {
        return None;
    }

    let per_chunk = u64::from(frames_per_chunk);
    // Candidate boundaries: a keyframe that is also a segment boundary. Index 0
    // is excluded because it is the start of the first range, not a split.
    let candidates: Vec<u64> =
        keyframes.iter().copied().filter(|k| *k > 0 && k % per_chunk == 0).collect();
    if candidates.is_empty() {
        return None;
    }

    // Aim for equal-length ranges and take the candidate nearest each target.
    // Duplicates collapse, so a source with few usable boundaries yields fewer
    // ranges rather than empty ones.
    let mut splits: Vec<u64> = Vec::new();
    for n in 1..want {
        let target = total * n as u64 / want as u64;
        if let Some(best) = candidates.iter().copied().min_by_key(|c| c.abs_diff(target)) {
            if !splits.contains(&best) {
                splits.push(best);
            }
        }
    }
    if splits.is_empty() {
        return None;
    }
    splits.sort_unstable();

    let mut ranges = Vec::with_capacity(splits.len() + 1);
    let mut start = 0u64;
    for split in splits {
        ranges.push(DecodeRange { start_sample: start, end_sample: Some(split), start_frame: start });
        start = split;
    }
    ranges.push(DecodeRange { start_sample: start, end_sample: None, start_frame: start });

    Some(ranges)
}

/// The NAL family for a codec label, for the codecs whose keyframes and
/// parameter sets can be read out of a sample.
fn nal_codec_for(codec_name: &str) -> Option<container::nal_mux::NalMuxCodec> {
    match codec_name.to_ascii_lowercase().as_str() {
        "h264" | "avc" | "avc1" => Some(container::nal_mux::NalMuxCodec::H264),
        "h265" | "hevc" | "hvc1" | "hev1" => Some(container::nal_mux::NalMuxCodec::H265),
        _ => None,
    }
}

/// One clip of a splice: a decode config, its source bytes, and the **source
/// frame range** to keep. The first `start_frame` decoded frames are dropped
/// (the trim in-point); decoding stops once the source index reaches
/// `end_frame` (exclusive — the trim out-point). `end_frame = None` keeps the
/// clip to its end. A single full-range clip (`start_frame = 0`,
/// `end_frame = None`) is a plain, un-spliced transcode.
#[derive(Clone)]
pub struct ClipSource {
    pub cfg: DecodePumpConfig,
    pub input: Bytes,
    pub start_frame: u64,
    pub end_frame: Option<u64>,
}

impl ClipSource {
    /// A whole clip, no trim.
    pub fn whole(cfg: DecodePumpConfig, input: Bytes) -> Self {
        Self { cfg, input, start_frame: 0, end_frame: None }
    }
}

/// Single-input decode pump (no trim, no concat) — the common case. A thin
/// wrapper over [`run_spliced_decode_pump_blocking`] with one whole clip.
pub fn run_shared_decode_pump_blocking(
    cfg: DecodePumpConfig,
    input_data: Bytes,
    senders: Vec<tokio::sync::mpsc::Sender<VideoFrame>>,
    rt: tokio::runtime::Handle,
) -> Result<u64> {
    run_spliced_decode_pump_blocking(vec![ClipSource::whole(cfg, input_data)], senders, rt)
}

/// Spliced decode pump, designed for `tokio::task::spawn_blocking`. Decodes
/// each clip in order, **drops** frames outside the clip's `[start_frame,
/// end_frame)` source range (trim), and fans the kept frames out to all
/// `senders` **continuously across clips** (concat). Because the muxer numbers
/// output frames by count — not by source PTS — the join is automatically
/// gap-free and the timeline is zero-based, with no PTS rewriting.
///
/// If a sender's channel is closed (its rung gave up) the pump keeps going with
/// the rest; it stops only when *every* sender is closed. `rt` bridges into the
/// async `send().await`. Returns the total number of frames emitted.
pub fn run_spliced_decode_pump_blocking(
    clips: Vec<ClipSource>,
    senders: Vec<tokio::sync::mpsc::Sender<VideoFrame>>,
    rt: tokio::runtime::Handle,
) -> Result<u64> {
    let mut total: u64 = 0;
    let result = (|| {
        for (clip_idx, clip) in clips.iter().enumerate() {
            match decode_clip(clip, &senders, &rt, &mut total)
                .with_context(|| format!("decoding splice clip {clip_idx}"))?
            {
                Flow::Continue => {}
                Flow::AllReceiversClosed => break,
            }
        }
        Ok(total)
    })();
    // Drop senders so receivers wake and exit.
    drop(senders);
    result
}

enum Flow {
    Continue,
    AllReceiversClosed,
}

/// Decode one clip, applying its trim range, fanning kept frames to `senders`
/// and advancing the shared output counter `total`.
fn decode_clip(
    clip: &ClipSource,
    senders: &[tokio::sync::mpsc::Sender<VideoFrame>],
    rt: &tokio::runtime::Handle,
    total: &mut u64,
) -> Result<Flow> {
    let cfg = &clip.cfg;
    let mut demuxer =
        streaming::demux_streaming_shared(clip.input.clone())
            .context("demuxing clip for decode pump")?;
    let decoder =
        decode::create_decoder_on(&cfg.codec_name, cfg.info_for_decoder.clone(), cfg.gpu_index)
            .context("creating decoder for decode pump")?;
    // Wrapped here rather than at each consumer: every rung fed by this pump
    // wants the picture the right way up. A rotation of 0 returns the decoder
    // itself, so the common case pays nothing.
    let mut decoder = decode::RotatingDecoder::new(decoder, cfg.rotation_degrees);

    // Source-frame index within THIS clip — drives the trim decision.
    let mut src_idx: u64 = 0;

    // The decode range, by demuxed sample index. Everything before it is
    // parsed and not decoded; the range ends with a flush of what the decoder
    // still holds, because those frames belong to this range.
    let (start_sample, end_sample) = cfg.sample_range.unwrap_or((0, None));
    let mut sample_idx: u64 = 0;

    // Parameter sets seen while skipping to the start of the range.
    //
    // mp4 keeps SPS/PPS in `avcC` extradata, so the demuxer emits them in-band
    // once at the top of the stream. A range starting anywhere else gets an IDR
    // with nothing to configure the decoder from, and a decoder in that state
    // does not fail — it returns zero frames, which then surfaces far away as a
    // rung missing two thirds of its segments.
    let nal_codec = nal_codec_for(&cfg.codec_name);
    let mut carried_param_sets: Vec<u8> = Vec::new();
    let mut param_sets_replayed = start_sample == 0;

    // Drain the decoder after `finish()`, at the end of the range or the clip.
    let drain = |decoder: &mut Box<dyn decode::Decoder>,
                     src_idx: &mut u64,
                     total: &mut u64|
     -> Result<Flow> {
        decoder.finish().context("decoder finish in decode pump")?;
        while let Some(frame) =
            decoder.decode_next().context("decoding frame after finish in decode pump")?
        {
            match handle_frame(clip, cfg, frame, senders, rt, src_idx, total)? {
                FrameAction::Continue => {}
                FrameAction::ClipDone => return Ok(Flow::Continue),
                FrameAction::StopAll => return Ok(Flow::AllReceiversClosed),
            }
        }
        Ok(Flow::Continue)
    };

    loop {
        match demuxer
            .next_video_sample()
            .context("demuxing next video sample in decode pump")?
        {
            Some(sample) => {
                let idx = sample_idx;
                sample_idx += 1;

                // Before our range: demux past it without decoding. The parse
                // is not wasted — it is also where the parameter sets are
                // picked up, so the decoder can be configured when the range
                // proper begins.
                if idx < start_sample {
                    if let Some(codec) = nal_codec {
                        let sets = container::nal_mux::extract_parameter_sets(&sample.data, codec);
                        if !sets.is_empty() {
                            carried_param_sets = sets;
                        }
                    }
                    continue;
                }
                // Past our range: flush what the decoder still holds and stop.
                if end_sample.is_some_and(|end| idx >= end) {
                    return drain(&mut decoder, &mut src_idx, total);
                }
                // First sample of a range that started mid-stream: hand the
                // decoder the parameter sets in force here, ahead of the IDR —
                // unless this sample carries its own.
                if !param_sets_replayed {
                    param_sets_replayed = true;
                    let sample_has_own = nal_codec.is_some_and(|codec| {
                        !container::nal_mux::extract_parameter_sets(&sample.data, codec).is_empty()
                    });
                    if !carried_param_sets.is_empty() && !sample_has_own {
                        tracing::debug!(
                            start_sample,
                            bytes = carried_param_sets.len(),
                            "replaying parameter sets at decode-range start",
                        );
                        decoder
                            .push_sample(&carried_param_sets)
                            .context("pushing carried parameter sets at decode-range start")?;
                    }
                }

                decoder
                    .push_sample(&sample.data)
                    .context("pushing sample to decode pump decoder")?;
                while let Some(frame) =
                    decoder.decode_next().context("decoding frame in decode pump")?
                {
                    match handle_frame(clip, cfg, frame, senders, rt, &mut src_idx, total)? {
                        FrameAction::Continue => {}
                        FrameAction::ClipDone => return Ok(Flow::Continue),
                        FrameAction::StopAll => return Ok(Flow::AllReceiversClosed),
                    }
                }
            }
            None => return drain(&mut decoder, &mut src_idx, total),
        }
    }
}

enum FrameAction {
    Continue,
    ClipDone,
    StopAll,
}

/// Apply the clip's trim range to one decoded frame: drop frames before the
/// in-point, signal `ClipDone` at the out-point, otherwise normalize + fan out.
fn handle_frame(
    clip: &ClipSource,
    cfg: &DecodePumpConfig,
    frame: VideoFrame,
    senders: &[tokio::sync::mpsc::Sender<VideoFrame>],
    rt: &tokio::runtime::Handle,
    src_idx: &mut u64,
    total: &mut u64,
) -> Result<FrameAction> {
    if clip.end_frame.is_some_and(|end| *src_idx >= end) {
        return Ok(FrameAction::ClipDone); // reached the out-point
    }
    if *src_idx >= clip.start_frame {
        let normalized = normalize_frame(cfg, frame)?;
        if !fan_out(senders, normalized, rt)? {
            return Ok(FrameAction::StopAll);
        }
        *total += 1;
    }
    *src_idx += 1;
    Ok(FrameAction::Continue)
}

/// Rung-agnostic per-frame work: 4:4:4 → 4:2:0 downsample (if needed) then,
/// when the spec's color policy asks for it (`tonemap_to_sdr`), an HDR-aware
/// colorspace convert (tonemap PQ/HLG → SDR BT.709, identity for SDR). When the
/// policy is passthrough/HDR, the downsampled source is forwarded unchanged.
/// Per-rung scaling is NOT done here.
fn normalize_frame(cfg: &DecodePumpConfig, frame: VideoFrame) -> Result<VideoFrame> {
    let downsampled = if cfg.needs_downsample {
        colorspace::downsample_444_to_420_frame(&frame)
            .context("shared decode pump 4:4:4 → 4:2:0 downsample")?
    } else {
        frame
    };
    let normalized = if !cfg.tonemap_to_sdr {
        // Passthrough / HDR output: preserve the source color + bit depth.
        downsampled
    } else {
        colorspace::convert_to_sdr_bt709(&downsampled, &cfg.source_color_metadata)
            .context("shared decode pump colorspace convert (HDR-aware)")?
    };
    // Video filters (crop/pad/flip/rotate/grayscale/overlay/colour) run on the
    // normalized 4:2:0 frame, before the per-rung scalers see it.
    if cfg.filters.is_empty() {
        Ok(normalized)
    } else {
        cfg.filters.apply(normalized).context("shared decode pump video filters")
    }
}

/// Number of frames timed per candidate when benchmarking decoders. Chosen so
/// the measurement amortises driver init yet stays well under a second per
/// candidate even on a modest GPU.
pub const DECODE_BENCH_FRAMES: usize = 120;

/// Benchmark each candidate GPU by decoding a short prefix of `input` on it and
/// return the fastest `gpu_index` (what `--decode-with-fastest` pins the pump
/// to). Construction + first-frame latency is excluded — the clock starts after
/// a small warmup — so the number reflects steady-state decode throughput, not
/// driver init. Candidates that fail to construct or decode are skipped;
/// returns `None` if no candidate produced frames or fewer than two candidates
/// were given (nothing to choose).
pub fn fastest_decode_gpu(
    codec_name: &str,
    info: &codec::frame::StreamInfo,
    input: &Bytes,
    candidates: &[u32],
    measure_frames: usize,
) -> Option<u32> {
    if candidates.len() < 2 {
        return candidates.first().copied();
    }
    let mut best: Option<(u32, f64)> = None;
    for &gpu in candidates {
        match bench_decode_gpu(codec_name, info, input, gpu, measure_frames) {
            Ok(Some(fps)) => {
                tracing::info!(
                    gpu_index = gpu,
                    fps = format!("{fps:.1}"),
                    "decode-with-fastest: benchmarked candidate"
                );
                if best.is_none_or(|(_, b)| fps > b) {
                    best = Some((gpu, fps));
                }
            }
            Ok(None) => {
                tracing::warn!(gpu_index = gpu, "decode-with-fastest: no frames; skipping candidate")
            }
            Err(e) => tracing::warn!(
                gpu_index = gpu,
                error = %e,
                "decode-with-fastest: bench failed; skipping candidate"
            ),
        }
    }
    if let Some((gpu, fps)) = best {
        tracing::info!(
            gpu_index = gpu,
            fps = format!("{fps:.1}"),
            "decode-with-fastest: selected fastest decode GPU"
        );
    }
    best.map(|(g, _)| g)
}

/// Decode up to `measure_frames` frames (after an 8-frame warmup) from `input`
/// on `gpu`, returning the measured fps — or `None` if it produced no frames.
fn bench_decode_gpu(
    codec_name: &str,
    info: &codec::frame::StreamInfo,
    input: &Bytes,
    gpu: u32,
    measure_frames: usize,
) -> Result<Option<f64>> {
    const WARMUP: usize = 8;
    let target = WARMUP + measure_frames;
    let mut demuxer = streaming::demux_streaming(input).context("demux for decode bench")?;
    let mut decoder = decode::create_decoder_on(codec_name, info.clone(), Some(gpu))
        .context("create decoder for bench")?;
    let mut decoded = 0usize;
    let mut clock: Option<Instant> = None;
    'outer: loop {
        match demuxer.next_video_sample().context("bench next sample")? {
            Some(s) => {
                decoder.push_sample(&s.data).context("bench push")?;
                while decoder.decode_next().context("bench decode")?.is_some() {
                    decoded += 1;
                    if decoded == WARMUP {
                        clock = Some(Instant::now());
                    }
                    if decoded >= target {
                        break 'outer;
                    }
                }
            }
            None => {
                decoder.finish().context("bench finish")?;
                while decoder.decode_next().context("bench drain")?.is_some() {
                    decoded += 1;
                    if decoded == WARMUP {
                        clock = Some(Instant::now());
                    }
                    if decoded >= target {
                        break 'outer;
                    }
                }
                break;
            }
        }
    }
    let measured = decoded.saturating_sub(WARMUP);
    Ok(match clock {
        Some(t) if measured > 0 => {
            let secs = t.elapsed().as_secs_f64();
            (secs > 0.0).then_some(measured as f64 / secs)
        }
        // Tiny clip (< WARMUP+1 frames): every candidate decodes the same few
        // frames, so return the count — equal across candidates, first wins.
        _ => (decoded > 0).then_some(decoded as f64),
    })
}

/// Fan one frame out to every sender. Cloning `VideoFrame` is cheap (inner
/// `Bytes` is `Arc`-backed). Returns `false` only if EVERY sender is closed.
fn fan_out(
    senders: &[tokio::sync::mpsc::Sender<VideoFrame>],
    frame: VideoFrame,
    rt: &tokio::runtime::Handle,
) -> Result<bool> {
    let mut any_alive = false;
    for (idx, sender) in senders.iter().enumerate() {
        let frame_clone = frame.clone();
        let sender = sender.clone();
        let accepted = rt.block_on(async move { sender.send(frame_clone).await });
        match accepted {
            Ok(()) => any_alive = true,
            Err(_) => {
                tracing::warn!(rung_idx = idx, "shared decode pump: rung dropped its receiver");
            }
        }
    }
    Ok(any_alive)
}

#[cfg(test)]
mod tests {
    use super::*;
    use codec::frame::VideoFrame;

    /// `RIVET_TEST_MEDIA` env override, else the workspace `test_media/` dir —
    /// the same lookup the integration tests use. The corpus is fetched on
    /// demand and never committed, so a missing file is a skip, not a failure.
    fn read_test_media(name: &str) -> Option<Bytes> {
        let dir = match std::env::var_os("RIVET_TEST_MEDIA") {
            Some(dir) => std::path::PathBuf::from(dir),
            None => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()?
                .parent()?
                .join("test_media"),
        };
        std::fs::read(dir.join(name)).ok().map(Bytes::from)
    }

    /// Demux only: which sample indices are keyframes, and how many there are.
    fn h264_keyframes(input: &Bytes) -> (Vec<u64>, u64) {
        let mut demuxer = streaming::demux_streaming(input).expect("demux");
        let mut keyframes = Vec::new();
        let mut total = 0u64;
        while let Some(s) = demuxer.next_video_sample().expect("sample") {
            if container::nal_mux::sample_is_keyframe(
                &s.data,
                container::nal_mux::NalMuxCodec::H264,
            ) {
                keyframes.push(total);
            }
            total += 1;
        }
        (keyframes, total)
    }

    #[test]
    fn a_whole_source_range_is_the_no_op_it_claims_to_be() {
        assert_eq!(DecodeRange::whole_source().sample_range(), None);
        assert_eq!(
            DecodeRange { start_sample: 120, end_sample: None, start_frame: 120 }.sample_range(),
            Some((120, None))
        );
    }

    #[test]
    fn a_single_range_or_an_unfamiliar_codec_is_not_split() {
        let input = Bytes::from_static(b"not a video");
        assert!(plan_decode_ranges(&input, "h264", 60, 1).is_none(), "want=1 is no split");
        assert!(plan_decode_ranges(&input, "av1", 60, 4).is_none(), "no keyframe test for av1");
        assert!(plan_decode_ranges(&input, "h264", 0, 4).is_none(), "a zero chunk is no grid");
    }

    #[test]
    fn ranges_start_on_keyframes_that_fall_on_segment_boundaries() {
        // The two properties everything downstream relies on: a range can be
        // decoded from its first sample alone, and its first segment index is
        // `start_frame / frames_per_chunk` exactly.
        let Some(input) = read_test_media("bbb_h264_360p_short.mp4") else {
            eprintln!("SKIP: test_media/bbb_h264_360p_short.mp4 not present");
            return;
        };
        let (keyframes, total) = h264_keyframes(&input);
        assert!(keyframes.len() > 1, "the sample needs several keyframes to split on");

        // Pick a chunk length that divides at least one keyframe past the
        // first, so the planner has a boundary to use.
        let per_chunk = keyframes
            .iter()
            .copied()
            .find(|&k| k > 0)
            .map(|k| k as u32)
            .expect("a second keyframe");

        let ranges = plan_decode_ranges(&input, "h264", per_chunk, 3)
            .expect("a splittable source with want=3 should split");
        assert!(ranges.len() >= 2 && ranges.len() <= 3, "ranges: {ranges:?}");

        // Contiguous and covering: each range starts where the last ended, the
        // first at 0, the last open-ended.
        assert_eq!(ranges[0].start_sample, 0);
        assert!(ranges.last().unwrap().end_sample.is_none());
        for pair in ranges.windows(2) {
            assert_eq!(pair[0].end_sample, Some(pair[1].start_sample), "gap or overlap: {ranges:?}");
        }
        for r in &ranges {
            assert!(keyframes.contains(&r.start_sample), "{r:?} does not start on a keyframe");
            assert_eq!(r.start_sample % u64::from(per_chunk), 0, "{r:?} is off the segment grid");
            assert_eq!(r.start_frame, r.start_sample, "one frame per sample");
            assert!(r.start_sample < total);
        }
    }

    /// Decode with the pump under `cfg`, collecting every frame it emits.
    fn pump_frames(cfg: DecodePumpConfig, input: Bytes) -> Result<Vec<VideoFrame>> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("runtime");
        let (tx, mut rx) = tokio::sync::mpsc::channel::<VideoFrame>(4);
        let handle = rt.handle().clone();
        let pump =
            std::thread::spawn(move || run_shared_decode_pump_blocking(cfg, input, vec![tx], handle));
        let frames = rt.block_on(async move {
            let mut out = Vec::new();
            while let Some(f) = rx.recv().await {
                out.push(f);
            }
            out
        });
        pump.join().expect("pump thread")?;
        Ok(frames)
    }

    #[test]
    fn decoding_in_ranges_yields_the_frames_of_decoding_whole() {
        // The claim range-parallel decode rests on: two pumps over two ranges
        // produce, between them, exactly the frames one pump over the whole
        // source produces — same count, same pixels, in order. This is what
        // the parameter-set replay is for: without it the second range decodes
        // to nothing, and the ladder is missing everything after the split.
        let Some(input) = read_test_media("bbb_h264_360p_short.mp4") else {
            eprintln!("SKIP: test_media/bbb_h264_360p_short.mp4 not present");
            return;
        };
        let header = streaming::demux_streaming(&input).expect("demux").header().clone();
        let base = DecodePumpConfig {
            codec_name: header.codec.clone(),
            info_for_decoder: header.info.clone(),
            source_color_metadata: header.info.color_metadata,
            source_pixel_format: header.info.pixel_format,
            needs_downsample: false,
            tonemap_to_sdr: true,
            gpu_index: None,
            sample_range: None,
            rotation_degrees: header.rotation_degrees,
            filters: std::sync::Arc::new(codec::filter::FilterChain::prepare(&[]).expect("empty chain")),
        };

        let whole = match pump_frames(base.clone(), input.clone()) {
            Ok(frames) => frames,
            Err(e) => {
                eprintln!("SKIP: no H.264 decoder on this host/build ({e:#})");
                return;
            }
        };
        assert!(!whole.is_empty());

        let (keyframes, _) = h264_keyframes(&input);
        let per_chunk =
            keyframes.iter().copied().find(|&k| k > 0).expect("a second keyframe") as u32;
        let ranges = plan_decode_ranges(&input, "h264", per_chunk, 2).expect("splits in two");
        assert_eq!(ranges.len(), 2, "{ranges:?}");

        let mut joined = Vec::new();
        for range in &ranges {
            let cfg = DecodePumpConfig { sample_range: range.sample_range(), ..base.clone() };
            let frames = pump_frames(cfg, input.clone()).expect("range decodes");
            assert!(
                !frames.is_empty(),
                "range {range:?} decoded nothing — parameter sets not replayed?"
            );
            joined.extend(frames);
        }

        assert_eq!(joined.len(), whole.len(), "frame count differs between whole and ranged decode");
        for (i, (a, b)) in whole.iter().zip(joined.iter()).enumerate() {
            assert_eq!(
                (a.width, a.height, a.format),
                (b.width, b.height, b.format),
                "frame {i} shape"
            );
            assert_eq!(a.data, b.data, "frame {i} pixels differ between whole and ranged decode");
        }
    }
}
