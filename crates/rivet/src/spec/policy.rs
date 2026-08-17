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

/// The decode plan — which card(s) decode, and whether the decode is one
/// pump or split into ranges across the cards. One enum, so the two halves
/// cannot contradict each other: "pin decode to card 2" and "split the decode
/// across every card" are not both sayable.
///
/// One decoder for the whole ladder is one decoder, and once the ladder is
/// wide enough it is the ceiling: every encoder waits on it and adding GPUs
/// changes nothing. When the bitstream allows — an un-spliced H.264 / H.265
/// input whose keyframes fall on chunk boundaries — the source can be cut
/// into ranges that are each decodable from their first sample, one pump per
/// card, so the cards decode different stretches at the same time. The
/// numbering stays continuous across the join and the output is
/// byte-identical to a whole-source decode. See
/// [`plan_decode_ranges`](crate::decode_pump::plan_decode_ranges).
/// Anything that cannot be split safely decodes whole under every variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodePolicy {
    /// Split the decode into one range per decode-capable card of the encode
    /// policy's set, where the source allows; whole otherwise. The default.
    #[default]
    Auto,
    /// One decoder for the whole source, on the first decode-capable card of
    /// the encode policy's set. What every job did before ranges existed; the
    /// control arm of any comparison, and the choice for a host whose decode
    /// engines are already saturated.
    Whole,
    /// One decoder, pinned to this physical GPU index (e.g. decode on an iGPU
    /// while the dGPUs encode). Never split: a split on one card is no split.
    SpecificGpu(u32),
    /// Benchmark every decode-capable GPU on a short prefix of the input
    /// before the job and pin one decoder to the fastest. The engine resolves
    /// this to `SpecificGpu` once the winner is known; a no-op on single-GPU
    /// hosts.
    FastestGpu,
    /// Split into up to this many ranges, round-robin over the decode-capable
    /// cards. More ranges than cards is legal — several pumps then share a
    /// card — and is how the split is exercised on a one-card host.
    Ranges(usize),
}

impl DecodePolicy {
    /// The concrete pinned GPU index, if any. Everything but `SpecificGpu`
    /// returns `None`, so the engine picks from the decode-capable cards.
    pub fn gpu_index(self) -> Option<u32> {
        match self {
            DecodePolicy::SpecificGpu(i) => Some(i),
            _ => None,
        }
    }

    /// Whether the engine should benchmark decoders and resolve a fastest GPU.
    pub fn is_fastest(self) -> bool {
        matches!(self, DecodePolicy::FastestGpu)
    }

    /// How many decode ranges to ask for against a pool of `capacity` cards.
    /// One for anything that pins or benchmarks a single card.
    pub fn ranges_for(self, capacity: usize) -> usize {
        match self {
            DecodePolicy::Auto => capacity.max(1),
            DecodePolicy::Whole | DecodePolicy::SpecificGpu(_) | DecodePolicy::FastestGpu => 1,
            DecodePolicy::Ranges(n) => n.max(1),
        }
    }
}

impl std::str::FromStr for DecodePolicy {
    type Err = String;

    /// Parse the `--decode` value space: `auto`, `whole`, `fastest`, `gpu:N`,
    /// `ranges:N` — and a bare `N`, which is a GPU index (the `--decode-gpu`
    /// spelling this flag grew out of).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim().to_ascii_lowercase();
        let bad = || {
            format!(
                "decode must be 'auto', 'whole', 'fastest', 'gpu:N', 'ranges:N' or a GPU index; got '{s}'"
            )
        };
        if let Some(n) = s.strip_prefix("gpu:").or_else(|| s.strip_prefix("gpu=")) {
            return n.trim().parse::<u32>().map(DecodePolicy::SpecificGpu).map_err(|_| bad());
        }
        if let Some(n) = s
            .strip_prefix("ranges:")
            .or_else(|| s.strip_prefix("ranges="))
            .or_else(|| s.strip_prefix("split:"))
            .or_else(|| s.strip_prefix("split="))
        {
            return n.trim().parse::<usize>().map(DecodePolicy::Ranges).map_err(|_| bad());
        }
        match s.as_str() {
            "" | "auto" | "split" => Ok(DecodePolicy::Auto),
            "whole" | "none" | "single" => Ok(DecodePolicy::Whole),
            "fastest" => Ok(DecodePolicy::FastestGpu),
            other => other.parse::<u32>().map(DecodePolicy::SpecificGpu).map_err(|_| bad()),
        }
    }
}

/// The encode plan — which cards encode, and how the work is laid across
/// them. One enum, so the halves cannot contradict each other: "one encoder"
/// and "every card" are not both sayable, and there is no second knob that
/// silently turns a multi-GPU job serial.
///
/// Applies to both the single-file and HLS paths. Every spreading variant
/// runs the ladder engine (decode once — split across the cards where the
/// source allows — chunk each rung, encode across the cards, stitch or write
/// segments); `SingleGpu` takes the serial encode path with no chunk overhead,
/// which for single-file output means one encoder per rung and no seams at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncodePolicy {
    /// Every capable card, **ladder-scheduled**: one worker per card, each
    /// serving every rung and taking the next chunk of whichever rung is
    /// furthest behind. A card idles only when the whole job is out of work,
    /// and a ladder deeper than the GPU count still costs one decode. Falls
    /// back to single-GPU serial encode when only one GPU is present or the
    /// frame count is unknown. The default; measured faster than the pinned
    /// shape.
    #[default]
    AllGpus,
    /// Every capable card, each worker **pinned to its own rungs** (rung `i`
    /// to worker `i mod workers`) — "one rung, one GPU" when the ladder fits
    /// the pool. Predictable placement, and a rung's chunks all come off one
    /// card, at the cost of cards idling when their rungs are blocked. For
    /// benchmarking the two shapes against each other, and for hosts where
    /// placement matters more than throughput.
    PerRung,
    /// A **single** card, one encoder per rung, serial. `None` picks the first
    /// available GPU; `Some(i)` pins to GPU index `i`. Single-file output is
    /// seam-free by construction (there are no chunks); HLS runs one worker.
    SingleGpu(Option<u32>),
    /// Every GPU of one **vendor family** (and only that family),
    /// ladder-scheduled — e.g. `Family(GpuFamily::Nvidia)` on a host with an
    /// NVIDIA discrete + an integrated AMD/Intel GPU uses just the NVIDIA
    /// cards.
    Family(GpuFamily),
}

impl EncodePolicy {
    /// Whether this policy spreads work across more than one card (and so
    /// runs the ladder engine rather than the serial path).
    pub fn spreads(self) -> bool {
        !matches!(self, EncodePolicy::SingleGpu(_))
    }

    /// Whether workers are pinned to their own rungs rather than serving the
    /// whole ladder.
    pub fn pins_rungs(self) -> bool {
        matches!(self, EncodePolicy::PerRung)
    }
}

impl std::str::FromStr for EncodePolicy {
    type Err = String;

    /// Parse the `--encode` value space: `all` (or `ladder`), `per-rung`,
    /// `single`, `gpu:N`, `family:nvidia|amd|intel`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim().to_ascii_lowercase();
        let bad = || {
            format!(
                "encode must be 'all', 'per-rung', 'single', 'gpu:N' or 'family:nvidia|amd|intel'; got '{s}'"
            )
        };
        if let Some(n) = s.strip_prefix("gpu:").or_else(|| s.strip_prefix("gpu=")) {
            return n.trim().parse::<u32>().map(|i| EncodePolicy::SingleGpu(Some(i))).map_err(|_| bad());
        }
        if let Some(f) = s.strip_prefix("family:").or_else(|| s.strip_prefix("family=")) {
            return match f.trim() {
                "nvidia" => Ok(EncodePolicy::Family(GpuFamily::Nvidia)),
                "amd" => Ok(EncodePolicy::Family(GpuFamily::Amd)),
                "intel" => Ok(EncodePolicy::Family(GpuFamily::Intel)),
                _ => Err(bad()),
            };
        }
        match s.as_str() {
            "" | "all" | "auto" | "ladder" => Ok(EncodePolicy::AllGpus),
            "per-rung" | "per_rung" | "perrung" | "pinned" => Ok(EncodePolicy::PerRung),
            "single" | "serial" => Ok(EncodePolicy::SingleGpu(None)),
            _ => Err(bad()),
        }
    }
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
/// Only relevant when more than one GPU encodes a single file (a spreading
/// [`EncodePolicy`] on a multi-GPU host); single-GPU hosts, `SingleGpu`, and
/// HLS (whose segments are independent by design) are unaffected. AMD (AMF) and
/// Intel (QSV) chunks are already constant-QP, so their seams are quality-flat
/// — this chiefly governs **NVENC**, which otherwise runs VBR per chunk and can
/// leave a mild quality step at the chunk boundaries.
///
/// This is a seam-*quality* choice and nothing else. Wanting no seams at all
/// is not a seam mode, it is an encode plan: [`EncodePolicy::SingleGpu`], one
/// encoder per rung. (There used to be a `Serial` variant here that quietly
/// turned a multi-GPU job serial; that was two knobs for one question.)
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
    /// encoder: NVENC (`nvidia`), AMF (`amd`), or QSV (`qsv`). The
    /// `rav1e-fallback` software encoder is 8-bit only.
    TenBit,
}
