//! Policy enums — how video/audio codec, container, muxer, output-mode, color,
//! bit-depth, encode/decode distribution, chunk-seam handling, and GPU family
//! are selected. All types are `pub` and re-exported from the parent `spec`
//! module so callers reach them as `rivet::spec::VideoCodecPolicy`, etc.

use codec::frame::VideoCodec;

/// Output **video** codec policy — the video analogue of [`AudioCodecPolicy`].
/// Selects which codec the encoder produces:
/// - `Av1` *(default)* — royalty-clean (AV1 + Opus in MP4 = zero royalty exposure).
/// - `H264` / `H265` — for legacy-player compatibility; they carry the
///   patent-licensing obligations AV1 was chosen to avoid.
///
/// All three work for single-file MP4 **and** CMAF/HLS. Resolve to the
/// encoder/muxer's [`VideoCodec`] with [`VideoCodecPolicy::codec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VideoCodecPolicy {
    #[default]
    Av1,
    H264,
    H265,
}

impl VideoCodecPolicy {
    /// Resolve to the low-level [`VideoCodec`] the encoder + muxer consume.
    pub fn codec(self) -> VideoCodec {
        match self {
            VideoCodecPolicy::Av1 => VideoCodec::Av1,
            VideoCodecPolicy::H264 => VideoCodec::H264,
            VideoCodecPolicy::H265 => VideoCodec::H265,
        }
    }
}

/// Output **audio** codec policy — how the source audio track is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AudioCodecPolicy {
    /// Passthrough AAC / Opus / AC-3 / E-AC-3 verbatim; transcode MP3 /
    /// Vorbis to Opus; drop anything else.
    #[default]
    Auto,
    /// Keep/produce Opus: passthrough Opus, transcode everything else to Opus.
    ForceOpus,
    /// Drop audio entirely (video-only output).
    Drop,
}

/// Output **subtitle** policy — what happens to the source's subtitle tracks.
///
/// The only supported output format is `tx3g` (3GPP timed text, ffmpeg's
/// `mov_text`), which is what MP4 carries natively. That constrains what
/// "copy" can mean: **text** subtitles (SRT / ASS / WebVTT out of Matroska)
/// convert into it, and **bitmap** ones (PGS, VobSub, DVB) have no `tx3g`
/// representation, so they're dropped with a warning under either policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SubtitlePolicy {
    /// Carry text subtitles into the output as a `tx3g` track. The default —
    /// it matches `ffmpeg -c:s copy` for the formats MP4 can hold.
    #[default]
    Copy,
    /// Emit no subtitle track.
    Drop,
}

/// Deprecated alias for [`AudioCodecPolicy`] (renamed for symmetry with
/// [`VideoCodecPolicy`]).
#[deprecated(since = "0.1.5", note = "renamed to AudioCodecPolicy")]
pub type AudioPolicy = AudioCodecPolicy;

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

/// How the decode pump selects its GPU — the decode-side counterpart to
/// [`EncodePolicy`]. A sum type so the modes stay mutually exclusive: you can't
/// accidentally ask for "a specific GPU **and** the fastest".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodePolicy {
    /// Follow the encode policy: the first device of the selected family/set,
    /// round-robin for per-rung pumps. The default.
    #[default]
    Auto,
    /// Pin decode to this physical GPU index (e.g. decode on an iGPU while the
    /// dGPUs encode).
    SpecificGpu(u32),
    /// Benchmark every decode-capable GPU on a short prefix of the input before
    /// the job and pin the pump to the fastest. The engine resolves this to
    /// `SpecificGpu` once the winner is known; a no-op on single-GPU hosts.
    FastestGpu,
}

impl DecodePolicy {
    /// The concrete pinned GPU index, if any. `Auto` and an unresolved
    /// `FastestGpu` both return `None`, so the engine follows the encode policy.
    pub fn gpu_index(self) -> Option<u32> {
        match self {
            DecodePolicy::SpecificGpu(i) => Some(i),
            DecodePolicy::Auto | DecodePolicy::FastestGpu => None,
        }
    }

    /// Whether the engine should benchmark decoders and resolve a fastest GPU.
    pub fn is_fastest(self) -> bool {
        matches!(self, DecodePolicy::FastestGpu)
    }
}

impl std::str::FromStr for DecodePolicy {
    type Err = String;

    /// Parse `auto` / `fastest` / a GPU index — the `--decode-gpu` value space.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(DecodePolicy::Auto),
            "fastest" => Ok(DecodePolicy::FastestGpu),
            other => other.parse::<u32>().map(DecodePolicy::SpecificGpu).map_err(|_| {
                format!("decode-gpu must be 'auto', 'fastest', or a GPU index; got '{other}'")
            }),
        }
    }
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
    /// (Convenience builder: [`super::OutputSpec::web_sdr`].)
    #[default]
    TonemapToSdr,
    /// **Verbatim.** Keep the source's gamut, transfer, and bit depth as-is — no
    /// tonemap, no re-signaling. An HDR source stays HDR (needs a 10-bit
    /// encoder); an SDR source stays SDR. (Builder: [`super::OutputSpec::passthrough`].)
    Passthrough,
    /// **HDR10 out.** Force **BT.2020** gamut + **PQ** transfer, 10-bit. Sets
    /// 10-bit on its own, so you do *not* also need [`BitDepth::TenBit`].
    /// (Builder: [`super::OutputSpec::hdr10`].)
    Hdr10,
    /// **HLG out.** Force **BT.2020** gamut + **HLG** transfer, 10-bit. Implies
    /// 10-bit. (Builder: [`super::OutputSpec::hlg`].)
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
