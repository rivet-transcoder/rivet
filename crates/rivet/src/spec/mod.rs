//! Output specification — *how* a job should be transcoded.
//!
//! A job is described by an [`OutputSpec`]: the [`OutputMode`] (single file
//! vs segmented HLS), the [`VideoCodec`] + [`AudioCodecPolicy`], the [`Container`]
//! + [`Muxer`], and the user-defined ladder of [`Rung`]s (each with its own
//! [`Quality`]). Nothing about the output is hard-coded — the caller decides
//! the shape, the codec, the quality, and the renditions.
//!
//! ```
//! use rivet::spec::{OutputSpec, Rung, Quality};
//!
//! // A 3-rung HLS ladder with 4-second segments.
//! let spec = OutputSpec::hls(
//!     vec![Rung::new(1920, 1080), Rung::new(1280, 720), Rung::new(640, 360)],
//!     4.0,
//! );
//! assert!(spec.validate().is_ok());
//! ```

use anyhow::{Result, bail};
use codec::frame::{ColorMetadata, PixelFormat, TransferFn};

pub use codec::encode::tuning::{QualityTarget as PerceptualTarget, SpeedTier as Speed};

/// The low-level codec identity used by the encoder + muxer, re-exported from
/// [`codec::frame::VideoCodec`]. Most callers pick the codec via
/// [`VideoCodecPolicy`] (the spec-level dimension) and never touch this directly;
/// `VideoCodecPolicy::codec` resolves to it.
pub use codec::frame::VideoCodec;

mod policy;
mod rung;
#[cfg(test)]
mod tests;

pub use policy::*;
pub use rung::*;

/// Full output specification for a transcode job.
#[derive(Debug, Clone)]
pub struct OutputSpec {
    /// Output shape.
    pub mode: OutputMode,
    /// Output video codec policy (`Av1` default, or `H264` / `H265`).
    pub video_codec: VideoCodecPolicy,
    /// Audio handling.
    pub audio: AudioCodecPolicy,
    /// Target Opus bitrate in **bits per second** for tracks that get
    /// transcoded. `None` lets the encoder pick from the channel layout —
    /// 64 kbps per uncoupled stream + 96 kbps per coupled (stereo) pair, i.e.
    /// 64k mono, 96k stereo, 320k for 5.1. Ignored for passthrough tracks,
    /// which keep whatever bitrate they were authored at.
    pub audio_bitrate: Option<u32>,
    /// Which of the source's text subtitle tracks to carry. See
    /// [`SubtitlePolicy`]. A single-file MP4 gets a `tx3g` track per language;
    /// an HLS package gets a segmented-WebVTT rendition per language.
    pub subtitles: SubtitlePolicy,
    /// Audio filters applied to decoded PCM **before** the Opus encoder — today
    /// `channelmap`. See [`codec::audio::filter`].
    ///
    /// A filter forces the track to be decoded and re-encoded, so a non-empty
    /// chain is incompatible with a passthrough track; the audio job reports
    /// that rather than silently ignoring the filter.
    pub audio_filters: Vec<codec::audio::filter::AudioFilter>,
    /// Container format.
    pub container: Container,
    /// Muxer.
    pub muxer: Muxer,
    /// The ladder. Order is preserved; the first rung is treated as the
    /// "primary" for single-file callers that only want one output.
    pub rungs: Vec<Rung>,
    /// Cap the output frame rate (the encoder's signalled fps is clamped to
    /// this; the source cadence is otherwise preserved). `None` = source fps.
    pub max_frame_rate: Option<f64>,
    /// Pin hardware encode/decode to this GPU index on multi-GPU hosts.
    /// Kept in sync with `encode_policy` (`SingleGpu(idx)` ⇒ `gpu_index = idx`).
    pub gpu_index: Option<u32>,
    /// How to spread encode work across GPUs. See [`EncodePolicy`].
    pub encode_policy: EncodePolicy,
    /// The decode plan — which card(s) decode and whether the decode is split
    /// into ranges across them. See [`DecodePolicy`]: `Auto` (split, one range
    /// per capable card), `Whole`, `SpecificGpu(i)`, `FastestGpu`, `Ranges(n)`.
    pub decode_policy: DecodePolicy,
    /// GOP length in frames for every rung, when set. `None` = the engine's
    /// default of two seconds at the output frame rate.
    ///
    /// What it governs depends on the output. **Single file:** the encoder's
    /// keyframe cadence, and — on the multi-GPU path — the chunk grid, since a
    /// chunk is a whole number of GOPs. **HLS:** the segment grid is set by
    /// `segment_seconds` and every segment opens on an IDR regardless; a GOP
    /// *shorter* than the segment adds keyframes inside it (for seeking), a
    /// GOP longer than the segment is silently the segment, because every
    /// segment is encoded from a fresh IDR anyway. A rung's own
    /// [`Quality::keyframe_interval`] wins over this for that rung.
    pub gop: Option<u32>,
    /// Per-rung encoder knobs by *position in the ladder* — softer quality
    /// going down, one tile below 4K, more reference frames, and so on. See
    /// [`RungPolicy`](codec::encode::tuning::RungPolicy): the engine resolves
    /// it against each rung and layers the rung's own
    /// [`Quality::overrides`] on top (the rung-specific knob wins; quality
    /// deltas accumulate). Empty by default, so nothing changes unless asked;
    /// [`RungPolicy::recommended`](codec::encode::tuning::RungPolicy::recommended)
    /// is the measured ladder recommendation, and
    /// [`RungPolicy::parse`](codec::encode::tuning::RungPolicy::parse) reads
    /// the text grammar (`qstep=2;short<=2159:tiles=1x1;any:refs=3`).
    pub rung_policy: codec::encode::tuning::RungPolicy,
    /// Output color / tonemap policy. See [`ColorPolicy`].
    pub color: ColorPolicy,
    /// 4:4:4 → 4:2:0 chroma filter for 4:4:4 sources (`box`, the default,
    /// keeps outputs byte-identical to earlier releases; `lanczos` is the
    /// siting-correct separable Lanczos-2). No effect on 4:2:0 / 4:2:2
    /// sources. Settings key `chroma-downsample`.
    pub chroma_downsample: codec::colorspace::ChromaDownsample,
    /// Output bit depth. See [`BitDepth`].
    pub bit_depth: BitDepth,
    /// How the multi-GPU **single-file** path keeps quality consistent across
    /// the chunk seams it stitches. See [`ChunkSeamMode`].
    pub chunk_seam_mode: ChunkSeamMode,
    /// Video filters applied per-frame **before** per-rung scaling (crop, pad,
    /// flip, rotate, grayscale). Empty = none. See [`codec::filter`].
    pub filters: Vec<codec::filter::VideoFilter>,
    /// Splice **trim in-point**, in seconds from the start of the (single)
    /// input. `None` starts at the beginning. Frames before this point are
    /// decoded-and-dropped; the output timeline is re-based to zero. For
    /// multi-clip concatenation use [`run_splice_job`](crate::run_splice_job)
    /// with a per-clip range instead. Trimmed jobs take the serial encode path.
    pub trim_start: Option<f64>,
    /// Splice **trim out-point**, in seconds. `None` keeps the clip to its end.
    /// The kept range is `[trim_start, trim_end)`.
    pub trim_end: Option<f64>,
}

impl Default for OutputSpec {
    fn default() -> Self {
        Self {
            mode: OutputMode::SingleFile,
            video_codec: VideoCodecPolicy::Av1,
            audio: AudioCodecPolicy::Auto,
            audio_bitrate: None,
            audio_filters: Vec::new(),
            subtitles: SubtitlePolicy::default(),
            container: Container::Mp4,
            muxer: Muxer::Mp4File,
            rungs: Vec::new(),
            max_frame_rate: None,
            gpu_index: None,
            encode_policy: EncodePolicy::default(),
            decode_policy: DecodePolicy::Auto,
            gop: None,
            rung_policy: codec::encode::tuning::RungPolicy::new(),
            color: ColorPolicy::default(),

            chroma_downsample: codec::colorspace::ChromaDownsample::Box,
            bit_depth: BitDepth::default(),
            chunk_seam_mode: ChunkSeamMode::default(),
            filters: Vec::new(),
            trim_start: None,
            trim_end: None,
        }
    }
}

impl OutputSpec {
    /// One self-contained MP4 per rung (AV1 + Opus/passthrough audio).
    pub fn single_file(rungs: Vec<Rung>) -> Self {
        Self {
            mode: OutputMode::SingleFile,
            container: Container::Mp4,
            muxer: Muxer::Mp4File,
            rungs,
            ..Default::default()
        }
    }

    /// A segmented CMAF + HLS package with the given rungs and segment length.
    pub fn hls(rungs: Vec<Rung>, segment_seconds: f32) -> Self {
        Self {
            mode: OutputMode::Hls { segment_seconds },
            container: Container::Cmaf,
            muxer: Muxer::CmafHls,
            rungs,
            ..Default::default()
        }
    }

    /// Set the audio policy.
    pub fn with_audio(mut self, audio: AudioCodecPolicy) -> Self {
        self.audio = audio;
        self
    }

    /// Set the target Opus bitrate in bits per second for transcoded audio.
    /// Omit to let the encoder derive it from the channel layout.
    pub fn with_audio_bitrate(mut self, bits_per_second: u32) -> Self {
        self.audio_bitrate = Some(bits_per_second);
        self
    }

    /// Set the subtitle policy — every text track, none, or a language list.
    /// See [`SubtitlePolicy`].
    pub fn with_subtitles(mut self, policy: SubtitlePolicy) -> Self {
        self.subtitles = policy;
        self
    }

    /// Set the audio filter chain (`channelmap`) applied before the encoder.
    /// See [`codec::audio::filter`].
    pub fn with_audio_filters(
        mut self,
        filters: Vec<codec::audio::filter::AudioFilter>,
    ) -> Self {
        self.audio_filters = filters;
        self
    }

    /// Cap output frame rate.
    pub fn with_max_frame_rate(mut self, fps: f64) -> Self {
        self.max_frame_rate = Some(fps);
        self
    }

    /// Pin to a GPU index. Implies `EncodePolicy::SingleGpu(Some(idx))`.
    pub fn with_gpu_index(mut self, idx: u32) -> Self {
        self.gpu_index = Some(idx);
        self.encode_policy = EncodePolicy::SingleGpu(Some(idx));
        self
    }

    /// Select the GPU encode policy: a single (optionally pinned) GPU, or all
    /// GPUs (the multi-GPU engine).
    ///
    /// ```no_run
    /// # use rivet::spec::{OutputSpec, EncodePolicy, Rung};
    /// # let rungs: Vec<Rung> = vec![];
    /// // chunk-encode across every GPU and stitch:
    /// let _ = OutputSpec::single_file(rungs.clone()).encode_policy(EncodePolicy::AllGpus);
    /// // serial encode, pinned to GPU 1:
    /// let _ = OutputSpec::single_file(rungs).encode_policy(EncodePolicy::SingleGpu(Some(1)));
    /// ```
    pub fn encode_policy(mut self, policy: EncodePolicy) -> Self {
        self.encode_policy = policy;
        if let EncodePolicy::SingleGpu(idx) = policy {
            self.gpu_index = idx;
        }
        self
    }

    /// Set the [`DecodePolicy`] — `Auto` (split across the capable cards),
    /// `Whole` (one decoder), `SpecificGpu(i)` (decode on an iGPU while dGPUs
    /// encode, say), `FastestGpu` (benchmark decoders up front and pick the
    /// quickest) or `Ranges(n)`.
    pub fn decode_policy(mut self, policy: DecodePolicy) -> Self {
        self.decode_policy = policy;
        self
    }

    /// Set the per-rung [`RungPolicy`](codec::encode::tuning::RungPolicy). See
    /// [`OutputSpec::rung_policy`].
    pub fn with_rung_policy(mut self, policy: codec::encode::tuning::RungPolicy) -> Self {
        self.rung_policy = policy;
        self
    }

    /// Set the GOP length in frames for every rung. See [`OutputSpec::gop`].
    pub fn with_gop(mut self, frames: Option<u32>) -> Self {
        self.gop = frames;
        self
    }

    /// The GOP the multi-GPU single-file path chunks on: `gop`, else two
    /// seconds at `frame_rate`.
    pub fn gop_frames(&self, frame_rate: f64) -> u32 {
        self.gop.unwrap_or_else(|| ((frame_rate * 2.0).round() as u32).max(1)).max(1)
    }

    /// The spec with `rung_policy` folded into every rung's
    /// [`Quality::overrides`] and the policy itself emptied — what the engine
    /// runs, so no worker has to know the ladder's shape. `rung_policy` is
    /// resolved against each rung's position (index 0 is the largest, as
    /// [`Rung`]s are ordered) and the rung's own overrides are layered on
    /// top: a rung-specific knob wins over the ladder-wide one, and quality
    /// deltas accumulate. A spec with an empty policy comes back unchanged.
    pub fn with_rung_policy_resolved(&self) -> OutputSpec {
        use codec::encode::tuning::RungContext;
        let mut resolved = self.clone();
        let policy_is_empty = self.rung_policy.rules.is_empty() && self.rung_policy.global.is_empty();
        if policy_is_empty && self.gop.is_none() {
            return resolved;
        }
        let rung_count = self.rungs.len();
        for (index, rung) in resolved.rungs.iter_mut().enumerate() {
            if !policy_is_empty {
                let ctx = RungContext { width: rung.width, height: rung.height, index, rung_count };
                let from_policy = self.rung_policy.resolve(&ctx);
                rung.quality.overrides = from_policy.merge(rung.quality.overrides);
            }
            // The spec-wide GOP reaches every rung two ways, because the two
            // paths read different fields: the serial path applies
            // `Quality::keyframe_interval`; the multi-GPU workers take the
            // chunk grid from the job and honour `overrides.keyframe_interval`
            // for the encoder's own cadence within it. A rung's own values win.
            if let Some(gop) = self.gop {
                if rung.quality.keyframe_interval.is_none() {
                    rung.quality.keyframe_interval = Some(gop);
                }
                if rung.quality.overrides.keyframe_interval.is_none() {
                    rung.quality.overrides.keyframe_interval = Some(gop);
                }
            }
        }
        resolved.rung_policy = codec::encode::tuning::RungPolicy::new();
        resolved
    }

    /// Set the output color / tonemap policy (SDR tonemap vs HDR passthrough).
    pub fn with_color(mut self, color: ColorPolicy) -> Self {
        self.color = color;
        self
    }

    /// Set the output **bit depth** (`Auto` / `EightBit` / `TenBit`). Sets bits
    /// per sample only — the gamut/SDR-HDR choice is [`Self::with_color`]. For
    /// HDR you usually don't need this (the HDR [`ColorPolicy`] implies 10-bit).
    /// Choose the 4:4:4 → 4:2:0 chroma filter (see
    /// [`codec::colorspace::ChromaDownsample`]).
    pub fn with_chroma_downsample(mut self, filter: codec::colorspace::ChromaDownsample) -> Self {
        self.chroma_downsample = filter;
        self
    }

    pub fn with_bit_depth(mut self, depth: BitDepth) -> Self {
        self.bit_depth = depth;
        self
    }

    // ── Color presets ──────────────────────────────────────────────
    // One-call intent shortcuts that bundle the color policy (and the bit depth
    // it implies). Equivalent to the `with_color` / `with_bit_depth` pairs in the
    // comments, but say what you mean. The low-level builders stay available.

    /// **Web-safe SDR** (the default): BT.709 8-bit, tonemapping any HDR source
    /// down. Plays everywhere. Same as `.with_color(TonemapToSdr)
    /// .with_bit_depth(EightBit)`.
    pub fn web_sdr(self) -> Self {
        self.with_color(ColorPolicy::TonemapToSdr)
            .with_bit_depth(BitDepth::EightBit)
    }

    /// **HDR10**: BT.2020 wide gamut + PQ transfer, 10-bit, no tonemap. Needs a
    /// 10-bit HDR encoder (`nvidia` / `amd` / `qsv` — the software fallback is
    /// 8-bit). Same as
    /// `.with_color(Hdr10)` — the policy already implies 10-bit.
    pub fn hdr10(self) -> Self {
        self.with_color(ColorPolicy::Hdr10)
    }

    /// **HLG**: BT.2020 wide gamut + HLG transfer, 10-bit, no tonemap. Same as
    /// `.with_color(Hlg)`.
    pub fn hlg(self) -> Self {
        self.with_color(ColorPolicy::Hlg)
    }

    /// **Passthrough**: keep the source's gamut, transfer, and bit depth
    /// verbatim. Same as `.with_color(Passthrough)`.
    pub fn passthrough(self) -> Self {
        self.with_color(ColorPolicy::Passthrough)
    }

    /// Set how the multi-GPU single-file path handles chunk seams
    /// (`Parallel` fastest / `ParallelConstQp` seam-flat; seam-free is an
    /// encode plan — [`EncodePolicy::SingleGpu`] — not a seam mode).
    pub fn chunk_seam_mode(mut self, mode: ChunkSeamMode) -> Self {
        self.chunk_seam_mode = mode;
        self
    }

    /// Set the per-frame video filter chain (crop / pad / flip / rotate /
    /// grayscale), applied before per-rung scaling. See [`codec::filter`].
    pub fn with_filters(mut self, filters: Vec<codec::filter::VideoFilter>) -> Self {
        self.filters = filters;
        self
    }

    /// **Trim** the single input to the time range `[start, end)` in seconds
    /// (either bound `None` = open). The output is re-based to zero. Trimmed
    /// jobs use the serial encode path. For joining multiple clips, see
    /// [`run_splice_job`](crate::run_splice_job).
    pub fn with_trim(mut self, start: Option<f64>, end: Option<f64>) -> Self {
        self.trim_start = start;
        self.trim_end = end;
        self
    }

    /// Set the output video codec ([`VideoCodecPolicy::Av1`] default, or `H264` /
    /// `H265`). All three work for single-file MP4 and CMAF/HLS.
    pub fn with_video_codec(mut self, codec: VideoCodecPolicy) -> Self {
        self.video_codec = codec;
        self
    }

    /// Whether the decode pump tonemaps HDR→SDR for this spec (policy-driven —
    /// the pump never decides on its own).
    pub fn tonemaps(&self) -> bool {
        self.color.tonemaps()
    }

    /// Resolve the encoder's input `(color_metadata, pixel_format)` for a given
    /// source. The default (`TonemapToSdr` + `Auto`) reproduces the legacy
    /// source-driven fold: HDR sources collapse to 8-bit SDR; SDR sources keep
    /// their own bit depth and color. `Hdr10`/`Hlg` force BT.2020 10-bit;
    /// `Passthrough` keeps the source; `pixel_format` overrides the bit depth.
    pub fn resolve_output(
        &self,
        source_color: ColorMetadata,
        source_pixel_format: PixelFormat,
    ) -> (ColorMetadata, PixelFormat) {
        let source_is_hdr = matches!(
            source_color.transfer,
            TransferFn::St2084 | TransferFn::AribStdB67
        );
        // The pump normalises every source onto 4:2:0 at 8 or 10 bits
        // (`colorspace::normalize_layout_to_420`), so that is what the
        // encoder is configured for — never the source's own 4:4:4 / 4:2:2
        // / 12-bit format, which no encoder in the tree accepts.
        let source_pixel_format = encoder_input_format(source_pixel_format);
        // The pump normalises every source onto 4:2:0 at 8 or 10 bits
        // (`colorspace::normalize_layout_to_420`), so that is what the
        // encoder is configured for — never the source's own 4:4:4 / 4:2:2
        // / 12-bit format, which no encoder in the tree accepts.
        let source_pixel_format = encoder_input_format(source_pixel_format);
        let (color, mut pix) = match self.color {
            ColorPolicy::TonemapToSdr => {
                if source_is_hdr {
                    (ColorMetadata::default(), PixelFormat::Yuv420p)
                } else {
                    (source_color, source_pixel_format)
                }
            }
            ColorPolicy::Passthrough => (source_color, source_pixel_format),
            ColorPolicy::Hdr10 => (hdr_metadata(TransferFn::St2084), PixelFormat::Yuv420p10le),
            ColorPolicy::Hlg => (hdr_metadata(TransferFn::AribStdB67), PixelFormat::Yuv420p10le),
        };
        match self.bit_depth {
            BitDepth::Auto => {}
            BitDepth::EightBit => pix = PixelFormat::Yuv420p,
            BitDepth::TenBit => pix = PixelFormat::Yuv420p10le,
        }
        (color, pix)
    }

    /// Reject incoherent specifications.
    pub fn validate(&self) -> Result<()> {
        if self.rungs.is_empty() {
            bail!("OutputSpec has no rungs — at least one rendition is required");
        }
        for r in &self.rungs {
            if r.width == 0 || r.height == 0 {
                bail!("rung '{}' has a zero dimension ({}x{})", r.label, r.width, r.height);
            }
            if r.width % 2 != 0 || r.height % 2 != 0 {
                bail!(
                    "rung '{}' has an odd dimension ({}x{}); 4:2:0 requires even dims",
                    r.label,
                    r.width,
                    r.height
                );
            }
        }
        // AV1, H.264, and H.265 are all valid for SingleFile MP4 and for
        // HLS/CMAF (the CMAF muxer builds av01 / avc3 / hev1 init segments and
        // the codec invariant handles all three across the multi-GPU path).
        // Container/muxer/mode coherence.
        match self.mode {
            OutputMode::SingleFile => {
                if self.muxer != Muxer::Mp4File || self.container != Container::Mp4 {
                    bail!("SingleFile mode requires Container::Mp4 + Muxer::Mp4File");
                }
            }
            OutputMode::Hls { segment_seconds } => {
                if self.muxer != Muxer::CmafHls || self.container != Container::Cmaf {
                    bail!("Hls mode requires Container::Cmaf + Muxer::CmafHls");
                }
                if !(segment_seconds > 0.0) {
                    bail!("Hls segment_seconds must be > 0 (got {segment_seconds})");
                }
            }
        }
        // Subtitles aren't validated against the source here: the spec can't
        // see which languages the source carries, so a requested language
        // with no track is reported by the job layer once it knows.

        // Audio coherence: both knobs only reach the Opus encoder, so pairing
        // either with "drop the audio" is a contradiction worth naming.
        if self.audio == AudioCodecPolicy::Drop {
            if !self.audio_filters.is_empty() {
                bail!(
                    "audio filters were given ({}) but the audio policy is `drop` — \
                     nothing would be filtered",
                    codec::audio::filter::chain_to_string(&self.audio_filters)
                );
            }
            if self.audio_bitrate.is_some() {
                bail!("an audio bitrate was given but the audio policy is `drop`");
            }
        }
        if let Some(bps) = self.audio_bitrate {
            // libopus clamps the aggregate to `500·ch ..= 300000·ch`, and the
            // channel count isn't known until the track is demuxed — so the
            // check here is the widest meaningful band (8 channels), enough to
            // catch a misplaced decimal without second-guessing the encoder.
            if !(500..=2_400_000).contains(&bps) {
                bail!(
                    "audio bitrate {bps} bps is outside Opus's meaningful range \
                     (500..=2400000; libopus further clamps to 500..=300000 per channel)"
                );
            }
        }
        // Output color / bit-depth coherence + what this build can produce.
        if self.color.is_hdr() && matches!(self.bit_depth, BitDepth::EightBit) {
            bail!(
                "color {:?} is HDR and requires 10-bit output, but bit_depth is forced to 8-bit",
                self.color
            );
        }
        let caps = codec::encode::build_output_caps();
        let needs_10bit = self.color.is_hdr() || matches!(self.bit_depth, BitDepth::TenBit);
        if needs_10bit && caps.max_bit_depth < 10 {
            bail!(
                "10-bit output requested (color={:?}, bit_depth={:?}) but this build has no \
                 10-bit encoder — build with `nvidia` (NVENC), `amd` (AMF), or `qsv` (oneVPL \
                 P010). The software fallbacks (rav1e, h26x) are 8-bit only.",
                self.color,
                self.bit_depth
            );
        }
        if self.color.is_hdr() && !caps.hdr {
            bail!(
                "HDR output ({:?}) requested but this build has no HDR-capable encoder — build \
                 with the `nvidia`, `amd`, or `qsv` feature",
                self.color
            );
        }
        Ok(())
    }
}

/// BT.2020 10-bit HDR color metadata for the given transfer (PQ or HLG).
/// The 4:2:0 format the decode pump hands the encoder for a source
/// `format`: `Yuv420p` for every 8-bit layout, `Yuv420p10le` for every
/// 10- and 12-bit one (12-bit is narrowed with rounding; 4:2:2 / 4:4:4 are
/// chroma-downsampled; RGB is matrixed). Mirrors
/// [`codec::colorspace::normalize_layout_to_420`] and
/// [`codec::colorspace::convert_bit_depth_frame`].
pub fn encoder_input_format(format: PixelFormat) -> PixelFormat {
    match codec::colorspace::planar_bit_depth(format) {
        Some(b) if b > 8 => PixelFormat::Yuv420p10le,
        Some(_) => PixelFormat::Yuv420p,
        // Yuva444p10le (10-bit + alpha) narrows to 10-bit 4:2:0; NV12 / NV21
        // and RGB are 8-bit layouts.
        None if format == PixelFormat::Yuva444p10le => PixelFormat::Yuv420p10le,
        None => PixelFormat::Yuv420p,
    }
}

fn hdr_metadata(transfer: TransferFn) -> ColorMetadata {
    ColorMetadata {
        transfer,
        matrix_coefficients: 9, // BT.2020 non-constant luminance
        colour_primaries: 9,    // BT.2020
        full_range: false,
        ..ColorMetadata::default()
    }
}
