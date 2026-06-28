//! Output specification — *how* a job should be transcoded.
//!
//! A job is described by an [`OutputSpec`]: the [`OutputMode`] (single file
//! vs segmented HLS), the [`VideoCodec`] + [`AudioPolicy`], the [`Container`]
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

use codec::encode::tuning::{QualityTarget, SpeedTier};
use codec::encode::{AUTO_FROM_TARGET, EncoderConfig};
use codec::frame::{ColorMetadata, PixelFormat, TransferFn};

pub use codec::encode::tuning::{QualityTarget as PerceptualTarget, SpeedTier as Speed};

/// Output video codec.
///
/// Output video codec — re-exported from [`codec::frame::VideoCodec`] so the
/// spec, encoder, and muxer share one type. `Av1` (default, royalty-clean
/// AV1 + Opus in MP4) plus `H264` / `H265` for legacy-player compatibility.
/// H.264 / H.265 carry patent-licensing obligations AV1 was chosen to avoid;
/// they are single-file MP4 only today (HLS/CMAF stays AV1).
pub use codec::frame::VideoCodec;

/// How the source audio track is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AudioPolicy {
    /// Passthrough AAC / Opus / AC-3 / E-AC-3 verbatim; transcode MP3 /
    /// Vorbis to Opus; drop anything else.
    #[default]
    Auto,
    /// Keep/produce Opus: passthrough Opus, transcode everything else to Opus.
    ForceOpus,
    /// Drop audio entirely (video-only output).
    Drop,
}

/// Output container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Container {
    /// Plain MP4 (ISO-BMFF), one self-contained file.
    #[default]
    Mp4,
    /// Fragmented MP4 (CMAF) — `moof`+`mdat` segments, for HLS/DASH.
    Cmaf,
}

/// Muxer — how the container bytes are assembled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Muxer {
    /// `Av1Mp4Muxer` — a single faststart MP4 with interleaved A/V.
    #[default]
    Mp4File,
    /// `CmafVideoMuxer` + `CmafAudioMuxer` + HLS playlists.
    CmafHls,
}

/// The high-level shape of the output.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputMode {
    /// One self-contained file per rung.
    SingleFile,
    /// Segmented CMAF + HLS: a media playlist per rung, a shared audio
    /// rendition, and a master playlist. `segment_seconds` is the target
    /// segment length (segments still break on keyframes).
    Hls { segment_seconds: f32 },
}

impl Default for OutputMode {
    fn default() -> Self {
        OutputMode::SingleFile
    }
}

/// Encoder quality knobs for a rung.
#[derive(Debug, Clone)]
pub struct Quality {
    /// Constant rate factor in the encoder-native scale (rav1e/NVENC 0..=255).
    /// `None` derives the quantizer from [`Quality::target`].
    pub crf: Option<u8>,
    /// Encoder-native speed preset. `None` derives it from [`Quality::tier`].
    pub speed_preset: Option<u8>,
    /// Perceptual quality target (used when `crf` is `None`).
    pub target: QualityTarget,
    /// Speed/efficiency tier (used when `speed_preset` is `None`).
    pub tier: SpeedTier,
    /// GOP length in frames. `None` → `2 × frame_rate` (a 2-second GOP).
    pub keyframe_interval: Option<u32>,
}

impl Default for Quality {
    fn default() -> Self {
        Self {
            crf: None,
            speed_preset: None,
            target: QualityTarget::Standard,
            tier: SpeedTier::Standard,
            keyframe_interval: None,
        }
    }
}

impl Quality {
    /// A constant-rate-factor quality.
    pub fn crf(crf: u8) -> Self {
        Self {
            crf: Some(crf),
            ..Default::default()
        }
    }

    /// A perceptual-target quality.
    pub fn target(target: QualityTarget) -> Self {
        Self {
            target,
            ..Default::default()
        }
    }

    /// Apply these knobs onto an [`EncoderConfig`] for a given frame rate.
    pub(crate) fn apply(&self, cfg: &mut EncoderConfig, frame_rate: f64) {
        cfg.target = self.target;
        cfg.tier = self.tier;
        cfg.quality = self.crf.unwrap_or(AUTO_FROM_TARGET);
        cfg.speed_preset = self.speed_preset.unwrap_or(AUTO_FROM_TARGET);
        cfg.keyframe_interval = self
            .keyframe_interval
            .unwrap_or_else(|| (frame_rate * 2.0).round().max(1.0) as u32);
    }
}

/// One rendition of the output ladder.
#[derive(Debug, Clone)]
pub struct Rung {
    /// Target width in pixels (even).
    pub width: u32,
    /// Target height in pixels (even).
    pub height: u32,
    /// Human label, e.g. `"720p"` (short side). Auto-derived by [`Rung::new`].
    pub label: String,
    /// Per-rung encoder quality.
    pub quality: Quality,
}

impl Rung {
    /// A rung at `width × height` with default quality and an auto label
    /// (`"<short-side>p"`).
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            label: format!("{}p", width.min(height)),
            quality: Quality::default(),
        }
    }

    /// Override the per-rung quality.
    pub fn with_quality(mut self, quality: Quality) -> Self {
        self.quality = quality;
        self
    }

    /// Override the label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Short side (the "p" number).
    pub fn short_side(&self) -> u32 {
        self.width.min(self.height)
    }
}

/// Full output specification for a transcode job.
#[derive(Debug, Clone)]
pub struct OutputSpec {
    /// Output shape.
    pub mode: OutputMode,
    /// Video codec (AV1 only today).
    pub video_codec: VideoCodec,
    /// Audio handling.
    pub audio: AudioPolicy,
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
    /// Decode-pump GPU override. `None` (default) pins the decode pump to a GPU
    /// consistent with `encode_policy` (the first device of the selected
    /// family/set, round-robin for per-rung pumps). `Some(i)` forces decode
    /// onto GPU `i` — e.g. decode on an iGPU while the dGPUs encode.
    pub decode_gpu: Option<u32>,
    /// Output color / tonemap policy. See [`ColorPolicy`].
    pub color: ColorPolicy,
    /// Output bit depth. See [`BitDepth`].
    pub bit_depth: BitDepth,
    /// How the multi-GPU **single-file** path keeps quality consistent across
    /// the chunk seams it stitches. See [`ChunkSeamMode`].
    pub chunk_seam_mode: ChunkSeamMode,
    /// Video filters applied per-frame **before** per-rung scaling (crop, pad,
    /// flip, rotate, grayscale). Empty = none. See [`codec::filter`].
    pub filters: Vec<codec::filter::VideoFilter>,
}

/// Selects how a job's encode work is distributed across the host's GPUs.
///
/// Applies to both the single-file and HLS paths: `AllGpus` runs the multi-GPU
/// engine (decode once, chunk each rung across every GPU, stitch); `SingleGpu`
/// constrains the GPU pool to one device and (for single-file) takes the serial
/// encode path with no chunk overhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncodePolicy {
    /// Use **all** available GPUs (the multi-GPU lease-pool engine). For
    /// single-file this chunk-encodes each rung across the GPUs and stitches
    /// the packets; it falls back to single-GPU serial encode when only one
    /// GPU is present or the frame count is unknown. This is the default.
    #[default]
    AllGpus,
    /// Use a **single** GPU. `None` picks the first available GPU; `Some(i)`
    /// pins to GPU index `i`. Single-file uses the serial encode path.
    SingleGpu(Option<u32>),
    /// Use every GPU of one **vendor family** (and only that family) — e.g.
    /// `Family(GpuFamily::Nvidia)` on a host with an NVIDIA discrete + an
    /// integrated AMD/Intel GPU uses just the NVIDIA cards. With more than one
    /// device in the family, single-file chunks across them like `AllGpus`.
    Family(GpuFamily),
}

/// A GPU vendor family, for constraining encode to one vendor's devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFamily {
    Nvidia,
    Amd,
    Intel,
}

/// How the multi-GPU **single-file** path keeps quality consistent across the
/// chunk seams it stitches into one continuous video.
///
/// Only relevant when more than one GPU encodes a single file (the `AllGpus` /
/// `Family` policies on a multi-GPU host); single-GPU hosts, `SingleGpu`, and
/// HLS (whose segments are independent by design) are unaffected. AMD (AMF) and
/// Intel (QSV) chunks are already constant-QP, so their seams are quality-flat
/// — this chiefly governs **NVENC**, which otherwise runs VBR per chunk and can
/// leave a mild quality step at the ~2 s boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChunkSeamMode {
    /// Default. Chunk across GPUs for throughput; each chunk uses its encoder's
    /// normal rate control (VBR on NVENC). Fastest; NVENC may show mild quality
    /// steps at the seams on complex content.
    #[default]
    Parallel,
    /// Chunk across GPUs but force **constant-QP** so the seams are
    /// quality-flat, keeping the multi-GPU speedup. The QP is derived from the
    /// `QualityTarget` (via the per-encoder tuning CQ), so quality still tracks
    /// the target — the hand-rolled NVENC sets a real const-QP rather than a
    /// preset default. AMD/QSV are unchanged (already constant-QP).
    ParallelConstQp,
    /// Encode the whole file with **one encoder** — seam-free and
    /// `QualityTarget`-accurate, at the cost of the multi-GPU single-file
    /// speedup. (Like `SingleGpu`, but leaves multi-GPU in place for HLS jobs.)
    Serial,
}

/// Output **color** policy — the gamut (which colors are representable) and the
/// transfer curve (SDR vs HDR), plus whether to tonemap an HDR source down. This
/// is the *color* half of the decision; bit depth is the separate [`BitDepth`]
/// half (though the HDR variants here imply 10-bit on their own).
///
/// The decode pump never tonemaps on its own — this policy decides.
///
/// Glossary (the jargon these variants use):
/// - **BT.709** — the standard HD / SDR color gamut. What the vast majority of
///   video uses; "SDR" output means BT.709.
/// - **BT.2020** — the *wide* gamut used by HDR: more saturated, deeper colors.
/// - **PQ** (SMPTE ST 2084) — the HDR10 transfer curve (absolute brightness, up
///   to 10,000 nits).
/// - **HLG** (ARIB STD-B67) — the broadcast-friendly HDR transfer curve
///   (relative brightness; degrades gracefully on SDR screens).
/// - **tonemap** — squeeze an HDR signal's brightness/gamut down into SDR so it
///   looks right on ordinary (BT.709, 8-bit) screens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorPolicy {
    /// **SDR out.** Tonemap HDR (PQ / HLG) sources down to 8-bit **BT.709** SDR;
    /// SDR sources pass through unchanged. The default — maximally web-compatible.
    /// (Convenience builder: [`OutputSpec::web_sdr`].)
    #[default]
    TonemapToSdr,
    /// **Verbatim.** Keep the source's gamut, transfer, and bit depth as-is — no
    /// tonemap, no re-signaling. An HDR source stays HDR (needs a 10-bit
    /// encoder); an SDR source stays SDR. (Builder: [`OutputSpec::passthrough`].)
    Passthrough,
    /// **HDR10 out.** Force **BT.2020** gamut + **PQ** transfer, 10-bit. Sets
    /// 10-bit on its own, so you do *not* also need [`BitDepth::TenBit`].
    /// (Builder: [`OutputSpec::hdr10`].)
    Hdr10,
    /// **HLG out.** Force **BT.2020** gamut + **HLG** transfer, 10-bit. Implies
    /// 10-bit. (Builder: [`OutputSpec::hlg`].)
    Hlg,
}

impl ColorPolicy {
    /// Whether the decode pump tonemaps HDR→SDR under this policy.
    pub fn tonemaps(self) -> bool {
        matches!(self, ColorPolicy::TonemapToSdr)
    }

    /// Whether this policy signals HDR (PQ/HLG) in the output bitstream.
    pub fn is_hdr(self) -> bool {
        matches!(self, ColorPolicy::Hdr10 | ColorPolicy::Hlg)
    }
}

/// Output **bit depth** — bits per sample. The on-disk pixel format is *derived*
/// from this (the encoder is always AV1 4:2:0, the web-safe chroma subsampling):
/// 8-bit → **`yuv420p`**, 10-bit → **`yuv420p10le`** (`le` = little-endian 16-bit
/// words holding 10 valid bits). Bit depth is one axis; gamut + SDR/HDR transfer
/// is the orthogonal [`ColorPolicy`] axis.
///
/// You rarely set this by hand: `Auto` derives it from the color policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BitDepth {
    /// Derive depth from the [`ColorPolicy`]: 8-bit for an SDR tonemap, 10-bit
    /// for HDR (`Hdr10` / `Hlg`), the source's own depth for `Passthrough`. The
    /// default — the right choice almost always.
    #[default]
    Auto,
    /// Force **8-bit** 4:2:0 (`yuv420p`) — universal web compatibility.
    EightBit,
    /// Force **10-bit** 4:2:0 (`yuv420p10le`) — higher precision (banding-free
    /// gradients), and required by the HDR policies. Needs a 10-bit-capable
    /// encoder: NVENC (`nvidia`), AMF (`amd`), QSV (`qsv`), or `ffmpeg`.
    TenBit,
}

impl Default for OutputSpec {
    fn default() -> Self {
        Self {
            mode: OutputMode::SingleFile,
            video_codec: VideoCodec::Av1,
            audio: AudioPolicy::Auto,
            container: Container::Mp4,
            muxer: Muxer::Mp4File,
            rungs: Vec::new(),
            max_frame_rate: None,
            gpu_index: None,
            encode_policy: EncodePolicy::default(),
            decode_gpu: None,
            color: ColorPolicy::default(),
            bit_depth: BitDepth::default(),
            chunk_seam_mode: ChunkSeamMode::default(),
            filters: Vec::new(),
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
    pub fn with_audio(mut self, audio: AudioPolicy) -> Self {
        self.audio = audio;
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

    /// Pin the decode pump to a specific GPU index, independent of the encode
    /// policy. `None` (the default) follows `encode_policy`. Useful to decode on
    /// an integrated GPU while discrete GPUs encode, or to keep decode on one
    /// device while encode chunks across several.
    pub fn decode_gpu(mut self, idx: Option<u32>) -> Self {
        self.decode_gpu = idx;
        self
    }

    /// Set the output color / tonemap policy (SDR tonemap vs HDR passthrough).
    pub fn with_color(mut self, color: ColorPolicy) -> Self {
        self.color = color;
        self
    }

    /// Set the output **bit depth** (`Auto` / `EightBit` / `TenBit`). Sets bits
    /// per sample only — the gamut/SDR-HDR choice is [`Self::with_color`]. For
    /// HDR you usually don't need this (the HDR [`ColorPolicy`] implies 10-bit).
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
    /// 10-bit HDR encoder (`nvidia` / `amd` / `qsv` / `ffmpeg`). Same as
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
    /// (`Parallel` fastest / `ParallelConstQp` seam-flat / `Serial` seam-free).
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

    /// Set the output video codec (`Av1` default, or `H264` / `H265`). H.264 /
    /// H.265 are single-file MP4 only — `validate()` rejects them with HLS.
    pub fn with_video_codec(mut self, codec: VideoCodec) -> Self {
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
        // H.264 / H.265 are single-file MP4 only today (HLS/CMAF + the
        // multi-GPU AV1-segment codec invariant are AV1-specific).
        if self.video_codec != VideoCodec::Av1 && !matches!(self.mode, OutputMode::SingleFile) {
            bail!(
                "video codec {:?} is single-file MP4 only; HLS/CMAF output is AV1-only",
                self.video_codec
            );
        }
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
                 10-bit AV1 encoder — build with `nvidia` (NVENC), `amd` (AMF), or `qsv` (oneVPL \
                 P010) for hardware 10-bit, or `ffmpeg` for software.",
                self.color,
                self.bit_depth
            );
        }
        if self.color.is_hdr() && !caps.hdr {
            bail!(
                "HDR output ({:?}) requested but this build has no HDR-capable encoder — build \
                 with the `nvidia`, `amd`, `qsv`, or `ffmpeg` feature",
                self.color
            );
        }
        Ok(())
    }
}

/// BT.2020 10-bit HDR color metadata for the given transfer (PQ or HLG).
fn hdr_metadata(transfer: TransferFn) -> ColorMetadata {
    ColorMetadata {
        transfer,
        matrix_coefficients: 9, // BT.2020 non-constant luminance
        colour_primaries: 9,    // BT.2020
        full_range: false,
        ..ColorMetadata::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_file_sets_coherent_fields() {
        let s = OutputSpec::single_file(vec![Rung::new(1280, 720)]);
        assert_eq!(s.mode, OutputMode::SingleFile);
        assert_eq!(s.container, Container::Mp4);
        assert_eq!(s.muxer, Muxer::Mp4File);
        assert!(s.validate().is_ok());
    }

    #[test]
    fn encode_policy_defaults_to_all_gpus() {
        let s = OutputSpec::single_file(vec![Rung::new(640, 360)]);
        assert_eq!(s.encode_policy, EncodePolicy::AllGpus);
        assert_eq!(s.gpu_index, None);
    }

    #[test]
    fn chunk_seam_mode_defaults_parallel_and_builder_sets_it() {
        let s = OutputSpec::single_file(vec![Rung::new(640, 360)]);
        assert_eq!(s.chunk_seam_mode, ChunkSeamMode::Parallel);
        let s = s.chunk_seam_mode(ChunkSeamMode::Serial);
        assert_eq!(s.chunk_seam_mode, ChunkSeamMode::Serial);
        let s = OutputSpec::single_file(vec![Rung::new(640, 360)])
            .chunk_seam_mode(ChunkSeamMode::ParallelConstQp);
        assert_eq!(s.chunk_seam_mode, ChunkSeamMode::ParallelConstQp);
        assert!(s.validate().is_ok());
    }

    #[test]
    fn encode_policy_single_gpu_syncs_gpu_index() {
        let s = OutputSpec::single_file(vec![Rung::new(640, 360)])
            .encode_policy(EncodePolicy::SingleGpu(Some(2)));
        assert_eq!(s.encode_policy, EncodePolicy::SingleGpu(Some(2)));
        assert_eq!(s.gpu_index, Some(2));
    }

    #[test]
    fn with_gpu_index_implies_single_gpu_policy() {
        let s = OutputSpec::single_file(vec![Rung::new(640, 360)]).with_gpu_index(1);
        assert_eq!(s.encode_policy, EncodePolicy::SingleGpu(Some(1)));
        assert_eq!(s.gpu_index, Some(1));
    }

    #[test]
    fn encode_policy_family_does_not_pin_gpu_index() {
        let s = OutputSpec::single_file(vec![Rung::new(640, 360)])
            .encode_policy(EncodePolicy::Family(GpuFamily::Nvidia));
        assert_eq!(s.encode_policy, EncodePolicy::Family(GpuFamily::Nvidia));
        // Family is multi-GPU within a vendor — no single-GPU pin.
        assert_eq!(s.gpu_index, None);
    }

    #[test]
    fn decode_gpu_defaults_to_none_and_is_settable() {
        let s = OutputSpec::single_file(vec![Rung::new(640, 360)]);
        assert_eq!(s.decode_gpu, None);
        let s = s.decode_gpu(Some(0));
        assert_eq!(s.decode_gpu, Some(0));
        // decode_gpu is independent of the encode policy.
        assert_eq!(s.encode_policy, EncodePolicy::AllGpus);
    }

    #[test]
    fn encode_policy_all_gpus_leaves_gpu_index_untouched() {
        let s = OutputSpec::single_file(vec![Rung::new(640, 360)])
            .with_gpu_index(3)
            .encode_policy(EncodePolicy::AllGpus);
        // AllGpus doesn't clear an explicit pin; it just won't single-pin.
        assert_eq!(s.encode_policy, EncodePolicy::AllGpus);
        assert_eq!(s.gpu_index, Some(3));
    }

    #[test]
    fn hls_sets_coherent_fields() {
        let s = OutputSpec::hls(vec![Rung::new(1920, 1080), Rung::new(640, 360)], 4.0);
        assert!(matches!(s.mode, OutputMode::Hls { .. }));
        assert_eq!(s.container, Container::Cmaf);
        assert_eq!(s.muxer, Muxer::CmafHls);
        assert!(s.validate().is_ok());
    }

    #[test]
    fn validate_rejects_empty_rungs() {
        assert!(OutputSpec::single_file(vec![]).validate().is_err());
    }

    #[test]
    fn validate_rejects_odd_dimensions() {
        assert!(OutputSpec::single_file(vec![Rung::new(1281, 720)]).validate().is_err());
    }

    #[test]
    fn validate_rejects_incoherent_mode_muxer() {
        let mut s = OutputSpec::single_file(vec![Rung::new(640, 360)]);
        s.muxer = Muxer::CmafHls; // mismatched with SingleFile mode
        assert!(s.validate().is_err());
    }

    #[test]
    fn rung_label_uses_short_side() {
        assert_eq!(Rung::new(1920, 1080).label, "1080p");
        assert_eq!(Rung::new(1080, 1920).label, "1080p");
        assert_eq!(Rung::new(640, 360).short_side(), 360);
    }

    #[test]
    fn color_and_pixel_format_default_to_sdr_8bit() {
        let s = OutputSpec::single_file(vec![Rung::new(640, 360)]);
        assert_eq!(s.color, ColorPolicy::TonemapToSdr);
        assert_eq!(s.bit_depth, BitDepth::Auto);
        assert!(s.tonemaps());
        assert!(s.validate().is_ok());
    }

    #[test]
    fn resolve_output_default_folds_hdr_source_to_sdr_8bit() {
        let s = OutputSpec::single_file(vec![Rung::new(640, 360)]);
        let hdr_src = hdr_metadata(TransferFn::St2084);
        let (color, pix) = s.resolve_output(hdr_src, PixelFormat::Yuv420p10le);
        // Default TonemapToSdr collapses an HDR 10-bit source to 8-bit SDR.
        assert_eq!(color.transfer, TransferFn::Bt709);
        assert_eq!(pix, PixelFormat::Yuv420p);
    }

    #[test]
    fn resolve_output_passthrough_keeps_source() {
        let s = OutputSpec::single_file(vec![Rung::new(640, 360)]).with_color(ColorPolicy::Passthrough);
        assert!(!s.tonemaps());
        let src = hdr_metadata(TransferFn::St2084);
        let (color, pix) = s.resolve_output(src, PixelFormat::Yuv420p10le);
        assert_eq!(color.transfer, TransferFn::St2084);
        assert_eq!(pix, PixelFormat::Yuv420p10le);
    }

    #[test]
    fn validate_rejects_hdr_without_10bit_or_ffmpeg() {
        // HDR10 implies 10-bit; without the `ffmpeg` feature the build is 8-bit,
        // so validation must reject it on a default build.
        let s = OutputSpec::single_file(vec![Rung::new(640, 360)]).with_color(ColorPolicy::Hdr10);
        let caps = codec::encode::build_output_caps();
        if caps.max_bit_depth < 10 {
            assert!(s.validate().is_err(), "HDR must be rejected on an 8-bit-only build");
        } else {
            assert!(s.validate().is_ok());
        }
    }

    #[test]
    fn validate_rejects_hdr_forced_8bit() {
        let s = OutputSpec::single_file(vec![Rung::new(640, 360)])
            .with_color(ColorPolicy::Hdr10)
            .with_bit_depth(BitDepth::EightBit);
        assert!(s.validate().is_err());
    }

    #[test]
    fn quality_crf_applies_to_encoder_config() {
        let q = Quality::crf(28);
        let mut cfg = EncoderConfig::default();
        q.apply(&mut cfg, 30.0);
        assert_eq!(cfg.quality, 28);
        assert_eq!(cfg.keyframe_interval, 60); // 2 * 30
    }
}
