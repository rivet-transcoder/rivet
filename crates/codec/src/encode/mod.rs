#[cfg(feature = "amd")]
pub mod amf;
#[cfg(not(feature = "amd"))]
#[path = "amf_stub.rs"]
pub mod amf;
#[cfg(feature = "nvidia")]
pub mod nvenc;
#[cfg(not(feature = "nvidia"))]
#[path = "nvenc_stub.rs"]
pub mod nvenc;
#[cfg(feature = "qsv")]
pub mod qsv;
#[cfg(not(feature = "qsv"))]
#[path = "qsv_stub.rs"]
pub mod qsv;
// Software AV1 encode on this workspace's own `av1` crate. Always compiled —
// the `av1-sw-fallback` feature decides whether the dispatch chain FALLS BACK
// to it, not whether it exists. A caller that wants software encoding can
// always ask for it by name.
pub mod av1_sw;
// Software H.264 / H.265 encode on this workspace's own `h26x` crate. Always
// compiled, like the decoders; the `h26x-fallback` feature decides whether the
// dispatch chain FALLS BACK to it.
pub mod h26x_sw;
// The workspace's own clean-room encoders for the codecs no hardware backend
// here encodes — ProRes, VP8, VP9, MPEG-2, MPEG-4 Part 2. Always compiled and
// always reachable for their codec: there is no faster tier for them to be a
// fallback from (see `native.rs`).
mod native;
pub mod mpeg2_sw;
pub mod mpeg4_sw;
pub mod prores_sw;
pub mod vp8_sw;
pub mod vp9_sw;
pub mod tuning;

use crate::frame::{ColorMetadata, PixelFormat, VideoCodec, VideoFrame};
use crate::gpu;
use anyhow::Result;

pub use tuning::{QualityTarget, SpeedTier};

/// Pick a GPU for a given vendor, honouring an explicit `gpu_index`
/// request when set. Returns `None` if no vendor GPU is present OR
/// the requested index belongs to a different vendor.
///
/// - `requested = Some(idx)`: look up the GPU with `GpuDevice.index == idx`.
///   If it exists AND matches `vendor`, return it. If it exists but is
///   a different vendor (e.g. caller pinned variant to NVIDIA slot 2
///   but we're evaluating the AMD fallback branch), return `None` so
///   dispatch falls through to the next tier — the other vendor tiers
///   will see this same `requested` index and match it there.
/// - `requested = None`: first-of-vendor (original pre-multi-GPU
///   behaviour, single-GPU hosts unaffected).
fn pick_vendor_device(
    gpus: &[gpu::GpuDevice],
    vendor: gpu::GpuVendor,
    requested: Option<u32>,
) -> Option<&gpu::GpuDevice> {
    match requested {
        Some(idx) => gpus.iter().find(|g| g.index == idx && g.vendor == vendor),
        None => gpus.iter().find(|g| g.vendor == vendor),
    }
}

pub trait Encoder: Send {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()>;
    fn flush(&mut self) -> Result<()>;
    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>>;

    /// Force the **next** frame to be an IDR — a self-contained random-access
    /// point — regardless of where the encoder's own GOP cadence would place
    /// one.
    ///
    /// The chunked multi-GPU path needs this. It feeds each worker a lead-in
    /// margin of frames that are encoded to warm up rate control and then
    /// discarded, so the first *kept* frame is not at the encoder's frame 0
    /// and must be promoted to an IDR explicitly or the chunk won't stitch.
    ///
    /// Defaults to unsupported so a backend that hasn't implemented it says so
    /// rather than silently producing a chunk that can't stand alone; callers
    /// fall back to encoding without a lead-in.
    fn force_keyframe_next(&mut self) -> Result<()> {
        anyhow::bail!("this encoder backend cannot force a keyframe")
    }

    /// Restart this session as if it had just been built, keeping the
    /// expensive part — the device context, the driver session, the surface
    /// and bitstream rings — and discarding everything a new stream must not
    /// inherit: reference pictures, rate-control and lookahead state, any
    /// packet not yet collected, and the GOP position, so the **next frame
    /// sent is an IDR** that opens a closed GOP.
    ///
    /// The chunked multi-GPU path is the caller. It encodes chunks out of
    /// order across cards and concatenates them, which is only correct when
    /// every chunk stands alone; a fresh encoder per chunk guaranteed that at
    /// the cost of ~1300 session constructions on a feature-length file.
    /// After `reset()` the session must give the same guarantee: the first
    /// packet out is a keyframe, and no packet of the previous chunk is
    /// still queued — `receive_packet` returns `None` until a frame is sent.
    ///
    /// Call it only after [`flush`](Self::flush) has been drained; a backend
    /// may refuse (or flush for itself) otherwise. The session may be used
    /// for an unlimited number of streams this way.
    ///
    /// Defaults to [`ResetUnsupported`] so a backend that hasn't implemented
    /// it says so by type, and the caller rebuilds instead — exactly the
    /// previous behaviour — rather than trusting a reset that did nothing.
    fn reset(&mut self) -> Result<()> {
        Err(ResetUnsupported.into())
    }
}

/// The error [`Encoder::reset`] returns when the backend has no reset path.
///
/// A type rather than a message so a caller can tell "rebuild, this backend
/// can't" (silent, expected) from "the reset failed" (worth a warning) with a
/// `downcast_ref`, and never by matching text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResetUnsupported;

impl std::fmt::Display for ResetUnsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("this encoder backend cannot reset a session; rebuild it instead")
    }
}

impl std::error::Error for ResetUnsupported {}

pub use ::frame::EncodedPacket;

/// Encoder configuration.
///
/// Prefer `target` + `tier` — `quality` and `speed_preset` are the
/// legacy per-encoder escape hatches and are kept so existing callers
/// compile. When `quality` is set to its sentinel (u8::MAX) the
/// adapter derives the quantizer from `target` instead. Same for
/// `speed_preset` (u8::MAX sentinel → derive from `tier`).
///
/// `PartialEq` is what lets a pooled session be matched to the next chunk:
/// two configs that compare equal describe the same stream, so a session
/// built for one can be reset and reused for the other.
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderConfig {
    pub width: u32,
    pub height: u32,
    pub frame_rate: f64,
    /// Legacy escape hatch. `u8::MAX` means "derive from `target`".
    /// Otherwise a CRF on the codec's own scale ([`crf_scale_max`]): the
    /// software AV1 encoder takes four times it as `base_q_idx`; NVENC
    /// scales it to its CQ range.
    pub quality: u8,
    /// Legacy escape hatch. `u8::MAX` means "derive from `tier`".
    pub speed_preset: u8,
    pub keyframe_interval: u32,
    /// Perceptual quality target. Defaults to `Standard` (VMAF ~90).
    pub target: QualityTarget,
    /// Speed tier (Draft / Standard / Archive). Defaults to `Standard`.
    pub tier: SpeedTier,
    /// What the caller asked for on top of `target`/`tier`, already resolved
    /// for this rung.
    ///
    /// Default is empty and empty is inert — see
    /// `tuning::EncodeOverrides`. The caller resolves an
    /// `EncodePolicy` against a `RungContext` and puts the answer here; the
    /// encoders read it rather than knowing anything about ladders.
    pub overrides: tuning::EncodeOverrides,
    /// Thread budget for this encoder instance. `0` means "use all cores".
    /// When the pipeline runs N variants in parallel it should set this to
    /// `num_cpus / N` to avoid oversubscribing the software encoders' worker
    /// pools (the h26x encoders; the software AV1 encoder is one thread).
    pub threads: usize,
    /// Input pixel format. Drives the encoder's bit-depth dispatch
    /// (the software AV1 encoder + NVENC/AMF/QSV, roadmap #5).
    /// `Yuv420p` → 8-bit AV1 Profile 0; `Yuv420p10le` → 10-bit AV1
    /// Profile 0 (10-bit 4:2:0 is allowed in Profile 0 per AV1 §5.5.2
    /// — `seq_profile=0`, `seq_color_config` emits `high_bitdepth=1`,
    /// `twelve_bit=0`). HW backends pick the matching surface fourcc:
    /// NVENC `YUV420_10BIT`, AMF `P010`, QSV `P010` + `BitDepthLuma=10`.
    /// Set once at encoder construction; flipping mid-session requires
    /// reinitialising. The muxer's `pixi`-equivalent + AV1 sequence
    /// header in `av1C` carry the bit depth so HDR-capable browsers
    /// see 10-bit signaling.
    pub pixel_format: PixelFormat,
    /// Source color metadata. Encoders write
    /// `color_primaries` / `transfer_characteristics` /
    /// `matrix_coefficients` / `color_range` into the AV1 sequence
    /// header so HDR-capable players see the correct PQ/HLG transfer
    /// and BT.2020 primaries straight off the bitstream — not just the
    /// container `colr` atom (the hardware backends; complements
    /// Squad-18's container-side colr nclx writer). Without bitstream
    /// signalling, players that prefer the OBU header over the box
    /// (e.g. Chromium video framework) would silently fall back to
    /// BT.709. Defaults to SDR BT.709.
    pub color_metadata: ColorMetadata,
    /// Explicit GPU device index for HW encoders on multi-GPU hosts.
    /// When `Some(idx)`, `select_encoder` binds NVENC / AMF / QSV /
    /// Vulkan AV1 / FFmpeg hwaccel encoders to the device with
    /// `GpuDevice.index == idx`. When `None` (default), the first
    /// GPU of each vendor is used — matches the original pre-multi-GPU
    /// behaviour.
    ///
    /// Pipeline `transcode::run` assigns `variant_idx % devices.len()`
    /// per variant so a multi-variant job on a multi-GPU host spreads
    /// work across devices, matching the Python original's
    /// `ThreadPoolExecutor(max_workers=device_count)` per-variant fan-out.
    pub gpu_index: Option<u32>,
    /// Explicit vendor pin for HW encoder dispatch. When `Some(v)`,
    /// `select_encoder` skips the NVIDIA → AMD → Intel preference
    /// chain and goes DIRECTLY to the encoder backend matching `v`
    /// (NVENC for Nvidia, AMF for Amd, QSV for Intel). Used by the
    /// CMAF orchestrator to honor the GpuPool's lease — when the
    /// pool hands out an Intel slot (because the NVIDIA card is
    /// already encoding), this field tells the factory to dispatch
    /// to QSV instead of falling back to NVENC and pinning every
    /// variant to the NVIDIA card.
    ///
    /// `None` (default) preserves the legacy NVIDIA-first chain so
    /// CPU-only paths + tests + non-pool callers behave unchanged.
    pub gpu_vendor: Option<gpu::GpuVendor>,
    /// Prefer **constant-QP** rate control over the bitrate/quality default.
    /// Set by the multi-GPU single-file path under `ChunkSeamMode::ParallelConstQp`
    /// so independently-encoded chunks have a flat quality across the stitched
    /// seams. On NVENC this selects `RateControlMode::ConstQp` (the wrapper then
    /// uses the preset's default QP — the `target` bitrate mapping is skipped).
    /// AMD/QSV already encode constant-quality, so this is a no-op for them.
    pub constant_qp: bool,
    /// Output video codec. `Av1` (default, royalty-clean) or `H264` / `H265`
    /// for legacy-player compatibility. The HW backends dispatch the codec
    /// id / profile on this; the muxer picks the matching sample entry.
    pub codec: VideoCodec,
}

/// Sentinel meaning "derive from `target` or `tier`".
pub const AUTO_FROM_TARGET: u8 = u8::MAX;

/// Refuse, by name, a rung that asks `backend` for a bitrate or a coded
/// picture buffer: an average rate is coded by the software encoders only —
/// the native H.264 / H.265 one (`h26x_sw`), the AV1 one (`av1_sw`), and
/// rivet's own VP9 / MPEG-2 / MPEG-4 encoders. Every other backend encodes to
/// its quality target, and one that took the rung anyway would hand back a stream at whatever rate
/// that target came to — the request dropped with nothing to say so. Each
/// backend calls this before it touches a driver.
///
/// A constant-rate rung (`RateMode::Constant`) is refused here too: a backend
/// that codes one calls [`constant_rate_request`] instead.
pub(crate) fn refuse_rate(backend: &str, config: &EncoderConfig) -> Result<()> {
    let o = &config.overrides;
    if o.rate_mode == Some(tuning::RateMode::Constant) {
        anyhow::bail!(
            "{backend} does not code a constant rate, and this rung asks for one (rate=cbr, bitrate={:?}): \
             run it on a GPU (QSV, NVENC or AMF code a constant rate), or drop rate=cbr",
            o.bitrate
        );
    }
    if o.bitrate.is_some() || o.buffer_ms.is_some_and(|ms| ms > 0) {
        anyhow::bail!(
            "{backend} encodes to a quality target, and this rung asks for a rate (bitrate={:?}, \
             buffer={:?}ms): an average rate is coded by the software encoders (`h26x` for H.264 / \
             H.265, `av1` for AV1). Run the rung on the software encoder, or drop the bitrate and \
             encode to a quality target",
            o.bitrate,
            o.buffer_ms
        );
    }
    Ok(())
}

/// The rate request of a rung for a backend that codes a constant rate (QSV,
/// NVENC, AMF): `Some` rate for a constant-rate rung it can code, `None` for
/// a rung coded to its quality target, and an error, by name, for anything
/// else — a constant-rate rung that cannot be coded as asked
/// ([`tuning::constant_rate_refusal`]), or an average rate, which only the
/// software tier codes ([`refuse_rate`]). Each such backend calls this before
/// it touches a driver.
#[cfg_attr(not(any(feature = "qsv", feature = "nvidia", feature = "amd")), allow(dead_code))]
pub(crate) fn constant_rate_request(backend: &str, config: &EncoderConfig) -> Result<Option<tuning::ConstantRate>> {
    let o = &config.overrides;
    if o.rate_mode != Some(tuning::RateMode::Constant) {
        refuse_rate(backend, config)?;
        return Ok(None);
    }
    let crf = (config.quality != AUTO_FROM_TARGET).then_some(config.quality);
    if let Some(why) = tuning::constant_rate_refusal(o, crf, config.constant_qp) {
        anyhow::bail!("{backend}: {why}");
    }
    Ok(tuning::ConstantRate::from_overrides(o))
}

/// Whether hardware backend `backend` encodes `codec` at all — the web set
/// (AV1, H.264, H.265) on all three, and VP9 on QSV: Intel's VDEnc VP9
/// encoder (profile 0 8-bit, profile 2 10-bit; Arc A-series / DG2 and Meteor
/// Lake — not Battlemage or Lunar Lake, whose media blocks decode VP9 only,
/// per Intel's media-driver feature tables; a card without it refuses at
/// `MFXVideoENCODE_Init` and the chain moves on). NVENC encodes no VP8 or
/// VP9 (NVIDIA's NVENC application note: H.264, HEVC, AV1), and AMF has no
/// VP8 / VP9 encoder component (AMF SDK: AVC, HEVC, AV1). No hardware
/// backend here encodes VP8, MPEG-2, MPEG-4 Part 2 or ProRes. The software
/// backends answer `false`: see [`native_backend_for`] for those.
pub fn hardware_encodes(backend: EncoderBackend, codec: VideoCodec) -> bool {
    match backend {
        EncoderBackend::Nvenc | EncoderBackend::Amf => codec.is_web_set(),
        EncoderBackend::Qsv => codec.is_web_set() || codec == VideoCodec::Vp9,
        _ => false,
    }
}

/// Whether any hardware backend compiled into this build encodes `codec`.
fn compiled_hardware_encodes(codec: VideoCodec) -> bool {
    (cfg!(feature = "nvidia") && hardware_encodes(EncoderBackend::Nvenc, codec))
        || (cfg!(feature = "amd") && hardware_encodes(EncoderBackend::Amf, codec))
        || (cfg!(feature = "qsv") && hardware_encodes(EncoderBackend::Qsv, codec))
}

/// The hardware backend of a GPU vendor.
fn vendor_backend(vendor: gpu::GpuVendor) -> EncoderBackend {
    match vendor {
        gpu::GpuVendor::Nvidia => EncoderBackend::Nvenc,
        gpu::GpuVendor::Amd => EncoderBackend::Amf,
        gpu::GpuVendor::Intel => EncoderBackend::Qsv,
    }
}

/// Refuse, by name, a codec hardware backend `backend` does not encode
/// ([`hardware_encodes`]). Each hardware backend calls this first, so
/// nothing below it sees such a codec.
#[cfg_attr(not(any(feature = "qsv", feature = "nvidia", feature = "amd")), allow(dead_code))]
pub(crate) fn refuse_unencoded_codec(backend: EncoderBackend, codec: VideoCodec) -> Result<()> {
    if !hardware_encodes(backend, codec) {
        let name = match backend {
            EncoderBackend::Nvenc => "NVENC",
            EncoderBackend::Amf => "AMF",
            _ => "QSV",
        };
        anyhow::bail!(
            "{name} does not encode {}; {} is encoded by rivet's own encoder (`{}`){}",
            codec.label(),
            codec.label(),
            codec.label(),
            if codec == VideoCodec::Vp9 { " or by QSV on an Intel card that has VP9 encode" } else { "" }
        );
    }
    Ok(())
}

/// Whether `backend` codes a constant rate (`RateMode::Constant`): every
/// hardware backend does, for every codec it encodes, and so does the
/// native software H.264 / H.265 tier (`h26x_sw::CODES_CONSTANT_RATE`);
/// the software AV1 encoder targets an average bitrate but not a constant one.
pub fn backend_codes_constant_rate(backend: EncoderBackend) -> bool {
    match backend {
        EncoderBackend::Qsv | EncoderBackend::Nvenc | EncoderBackend::Amf => true,
        EncoderBackend::H26x => h26x_sw::CODES_CONSTANT_RATE,
        // The software AV1, VP9, MPEG-2 and MPEG-4 encoders code an average
        // rate; ProRes and VP8 code no rate at all.
        EncoderBackend::Av1
        | EncoderBackend::ProRes
        | EncoderBackend::Vp8
        | EncoderBackend::Vp9
        | EncoderBackend::Mpeg2
        | EncoderBackend::Mpeg4 => false,
    }
}

/// The top of a codec's CRF scale.
///
/// Shared so a shifted CRF can be clamped without a backend's private copy —
/// and so it can never land on [`AUTO_FROM_TARGET`], which would turn "the
/// worst quality this codec has" into "ignore the caller's CRF entirely".
pub(crate) fn crf_scale_max(codec: VideoCodec) -> u8 {
    match codec {
        VideoCodec::Av1 => 63,
        VideoCodec::H264 | VideoCodec::H265 => 51,
        // VP8 / VP9 take libvpx's `cq-level` scale; MPEG-2 / MPEG-4 their own
        // quantiser codes (`encode::native::quantizer`). ProRes takes none.
        VideoCodec::Vp8 | VideoCodec::Vp9 | VideoCodec::ProRes(_) => 63,
        VideoCodec::Mpeg2 | VideoCodec::Mpeg4 => 31,
    }
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            frame_rate: 30.0,
            quality: AUTO_FROM_TARGET,
            speed_preset: AUTO_FROM_TARGET,
            keyframe_interval: 240,
            target: QualityTarget::Standard,
            tier: SpeedTier::Standard,
            overrides: tuning::EncodeOverrides::default(),
            threads: 0,
            // 8-bit SDR baseline — keeps every existing
            // `EncoderConfig { ..default() }` literal compiling and
            // behaving unchanged. 10-bit callers (the software AV1 encoder
            // or the HW backends) explicitly opt in by setting
            // `pixel_format = Yuv420p10le` and populating
            // `color_metadata` from the source.
            pixel_format: PixelFormat::Yuv420p,
            color_metadata: ColorMetadata::default(),
            gpu_index: None,
            gpu_vendor: None,
            constant_qp: false,
            codec: VideoCodec::Av1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderBackend {
    Nvenc,
    Amf,
    Qsv,
    /// The native software H.264 / H.265 encoders (`h26x_sw`). Asking for
    /// this by name works with or without the `h26x-fallback` feature — the
    /// feature gates only whether the chain reaches it unasked.
    H26x,
    /// rivet's own software AV1 encoder (`av1_sw`), by name; likewise
    /// independent of `av1-sw-fallback`.
    Av1,
    /// This workspace's own ProRes encoder (`prores_sw`): the only ProRes
    /// encoder, always reachable for ProRes.
    ProRes,
    /// This workspace's own VP8 encoder (`vp8_sw`).
    Vp8,
    /// This workspace's own VP9 encoder (`vp9_sw`), profiles 0-3.
    Vp9,
    /// This workspace's own MPEG-2 Video encoder (`mpeg2_sw`).
    Mpeg2,
    /// This workspace's own MPEG-4 Part 2 encoder (`mpeg4_sw`).
    Mpeg4,
}

/// The workspace's own encoder for a codec no hardware backend here
/// encodes — ProRes, VP8, VP9, MPEG-2, MPEG-4 Part 2 — or `None` for the web
/// set (AV1, H.264, H.265), which the dispatch chain serves.
pub fn native_backend_for(codec: VideoCodec) -> Option<EncoderBackend> {
    match codec {
        VideoCodec::ProRes(_) => Some(EncoderBackend::ProRes),
        VideoCodec::Vp8 => Some(EncoderBackend::Vp8),
        VideoCodec::Vp9 => Some(EncoderBackend::Vp9),
        VideoCodec::Mpeg2 => Some(EncoderBackend::Mpeg2),
        VideoCodec::Mpeg4 => Some(EncoderBackend::Mpeg4),
        VideoCodec::Av1 | VideoCodec::H264 | VideoCodec::H265 => None,
    }
}

/// What output formats an encoder path can produce. AV1 here is 4:2:0 only;
/// 10-bit output is the web-safe AV1 Main profile (4:2:0 10-bit), HDR-tagged at
/// the container level (`colr`/`mdcv`/`clli`), not the wide-gamut professional
/// profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputCaps {
    /// Highest luma bit depth the path can encode (8 or 10).
    pub max_bit_depth: u8,
    /// Can produce HDR (PQ/HLG + BT.2020) output — i.e. 10-bit AV1 + the muxer's
    /// HDR color atoms.
    pub hdr: bool,
}

/// Output capabilities of a specific hardware backend. All three do 10-bit AV1,
/// so they can produce HDR natively: NVENC via
/// `Yuv420_10bit`, AMF via `P010`, and QSV via the in-repo oneVPL P010 path
/// ([`qsv`]).
pub fn backend_output_caps(backend: EncoderBackend) -> OutputCaps {
    match backend {
        EncoderBackend::Nvenc | EncoderBackend::Amf | EncoderBackend::Qsv => OutputCaps {
            max_bit_depth: 10,
            hdr: true,
        },
        // The native h26x tier encodes H.265 Main 10 and H.264 High 10 and
        // writes the SPS VUI colour description from `color_metadata`
        // (`h26x_sw::colour_description`), so a BT.2020 PQ / HLG stream says
        // so in the bitstream as well as the box: HDR.
        EncoderBackend::H26x => OutputCaps {
            max_bit_depth: 10,
            hdr: true,
        },
        // The software AV1 encoder codes profile 0 at 8 or 10 bits, and
        // writes no colour description into its sequence header (the
        // container's `colr` carries it): 10-bit, no HDR.
        EncoderBackend::Av1 => OutputCaps {
            max_bit_depth: 10,
            hdr: false,
        },
        // ProRes is coded from the pipeline's 8- or 10-bit frames and writes
        // the H.273 colour codes into its frame header: 10-bit with HDR.
        EncoderBackend::ProRes => OutputCaps {
            max_bit_depth: 10,
            hdr: true,
        },
        // VP9 profiles 2 and 3 carry 10 and 12 bits; the uncompressed
        // header has a colour space but no transfer, so HDR is the
        // container's to say (`vpcC`, `colr`) and is not claimed here. The
        // pipeline's widest output is 10 bits.
        EncoderBackend::Vp9 => OutputCaps {
            max_bit_depth: 10,
            hdr: false,
        },
        // VP8, MPEG-2 Main and MPEG-4 Simple / Advanced Simple: 8-bit 4:2:0,
        // and none of their encoders writes an HDR transfer.
        EncoderBackend::Vp8 | EncoderBackend::Mpeg2 | EncoderBackend::Mpeg4 => EIGHT_BIT_SDR,
    }
}

/// Output capabilities of **this build** — the union over every compiled
/// encoder path. 10-bit + HDR comes from NVENC (`nvidia`), AMF (`amd`), QSV
/// (`qsv`, via the in-repo P010 path), or the software H.265 Main 10 tier
/// (`h26x-fallback`, with the VUI colour description); a build with no
/// encoder feature is 8-bit SDR (the software AV1 tier, `av1-sw-fallback`, adds
/// 10-bit without HDR). Callers (e.g. rivet's
/// `OutputSpec::validate`) use this to reject a format the build can't produce.
pub fn build_output_caps() -> OutputCaps {
    // The union over every backend the build can reach unasked, taken from
    // the per-backend answers so the two cannot disagree. (They did: this
    // used to claim 10-bit + HDR for a software-AV1-only build, whose
    // encoder was then 8-bit.)
    union_caps(compiled_backends().into_iter().map(backend_output_caps))
}

/// The 8-bit SDR floor every encode path meets.
const EIGHT_BIT_SDR: OutputCaps = OutputCaps {
    max_bit_depth: 8,
    hdr: false,
};

/// Output capabilities of `backend` for one output `codec` — the per-codec
/// answer [`backend_output_caps`], which is per backend, cannot give.
///
/// They differ for H.264: no hardware backend here has a High 10 encoder
/// (NVENC has no High 10 profile GUID, oneVPL no `AVC High 10`, AMF no
/// 10-bit `Profile`, and each refuses a 10-bit H.264 request), so H.264 on
/// NVENC / AMF / QSV is 8-bit SDR. The native `h26x` tier writes High 10
/// with the VUI colour description, so H.264 there is 10-bit with HDR. A
/// codec the backend does not serve at all (AV1 on `h26x`, H.264 / H.265 on
/// the software AV1 encoder) reports the 8-bit floor, which leaves a union
/// unchanged.
pub fn backend_output_caps_for(backend: EncoderBackend, codec: VideoCodec) -> OutputCaps {
    // A codec only its own encoder serves: that encoder's answer, and the
    // floor from every other backend — the hardware three included, which
    // would otherwise lend their 10-bit HDR to a VP9 job they never see.
    let native = native_backend_for(codec);
    if native.is_some() || is_native_backend(backend) {
        return if native == Some(backend) {
            backend_output_caps(backend)
        } else if hardware_encodes(backend, codec) {
            // VP9 on QSV: profile 0 and profile 2 (10-bit), its colour in the
            // container as for rivet's own VP9 encoder.
            OutputCaps { max_bit_depth: 10, hdr: false }
        } else {
            EIGHT_BIT_SDR
        };
    }
    match (backend, codec) {
        (EncoderBackend::Nvenc | EncoderBackend::Amf | EncoderBackend::Qsv, VideoCodec::H264) => {
            EIGHT_BIT_SDR
        }
        (EncoderBackend::H26x, VideoCodec::Av1) => EIGHT_BIT_SDR,
        (EncoderBackend::Av1, VideoCodec::H264 | VideoCodec::H265) => EIGHT_BIT_SDR,
        _ => backend_output_caps(backend),
    }
}

/// Whether `backend` is one of the encoders [`native_backend_for`] names.
fn is_native_backend(backend: EncoderBackend) -> bool {
    matches!(
        backend,
        EncoderBackend::ProRes | EncoderBackend::Vp8 | EncoderBackend::Vp9 | EncoderBackend::Mpeg2 | EncoderBackend::Mpeg4
    )
}

/// Output capabilities of **this build** for one output `codec`: the union
/// of [`backend_output_caps_for`] over the compiled paths. H.264 at 10 bits
/// is here only when `h26x-fallback` is.
pub fn build_output_caps_for(codec: VideoCodec) -> OutputCaps {
    union_caps(compiled_backends().into_iter().map(|b| backend_output_caps_for(b, codec)))
}

/// The best of each capability over `caps`, from the 8-bit SDR floor.
fn union_caps(caps: impl Iterator<Item = OutputCaps>) -> OutputCaps {
    caps.fold(EIGHT_BIT_SDR, |acc, c| OutputCaps {
        max_bit_depth: acc.max_bit_depth.max(c.max_bit_depth),
        hdr: acc.hdr || c.hdr,
    })
}

/// Every encode backend the build can reach unasked, in dispatch-preference
/// order — the set [`build_output_caps_for`] takes its union over, as the
/// enum ([`encode_backends`] is the same list by name). For a caller that
/// wants the per-backend answers behind the union, e.g. to say which backend
/// limits a refused output.
pub fn compiled_encode_backends() -> Vec<EncoderBackend> {
    compiled_backends()
}

/// Every encode backend the build can reach unasked.
fn compiled_backends() -> Vec<EncoderBackend> {
    let mut compiled: Vec<EncoderBackend> = Vec::new();
    if cfg!(feature = "nvidia") {
        compiled.push(EncoderBackend::Nvenc);
    }
    if cfg!(feature = "amd") {
        compiled.push(EncoderBackend::Amf);
    }
    if cfg!(feature = "qsv") {
        compiled.push(EncoderBackend::Qsv);
    }
    if cfg!(feature = "av1-sw-fallback") {
        compiled.push(EncoderBackend::Av1);
    }
    if cfg!(feature = "h26x-fallback") {
        compiled.push(EncoderBackend::H26x);
    }
    // The workspace's own encoders for the codecs no hardware backend takes:
    // always reachable, each for its codec alone.
    compiled.extend([
        EncoderBackend::ProRes,
        EncoderBackend::Vp8,
        EncoderBackend::Vp9,
        EncoderBackend::Mpeg2,
        EncoderBackend::Mpeg4,
    ]);
    compiled
}

/// Encode backends compiled into this build, in dispatch-preference order.
/// The hardware three serve every output codec; `av1` is software AV1 and
/// `h26x` is software H.264 / H.265, each listed only when its `-fallback`
/// feature lets the chain reach it unasked.
pub fn encode_backends() -> Vec<&'static str> {
    let mut v = Vec::new();
    if cfg!(feature = "nvidia") {
        v.push("nvenc");
    }
    if cfg!(feature = "amd") {
        v.push("amf");
    }
    if cfg!(feature = "qsv") {
        v.push("qsv");
    }
    if cfg!(feature = "av1-sw-fallback") {
        v.push("av1");
    }
    if cfg!(feature = "h26x-fallback") {
        v.push("h26x");
    }
    v.extend(["prores", "vp8", "vp9", "mpeg2", "mpeg4"]);
    v
}

/// The software backend [`select_encoder`] would fall back to for `codec`
/// **in this build**, or `None` when the build has no software tier for it.
///
/// Answered from the feature flags, not by constructing an encoder: a
/// software encoder spins up a worker pool sized to the machine just to be
/// asked, and the ladder wants to know before it hands out leases, once per
/// job, not once per rung. This is the same gate the bottom of
/// `select_encoder` applies — `av1-sw-fallback` for AV1, `h26x-fallback` for
/// H.264 / H.265 — so `Some` means the chain would reach software unasked, and
/// a caller may ask for it by name via `select_encoder(cfg, Some(backend))`
/// and skip the hardware probes it already knows will decline.
pub fn software_backend_for(codec: VideoCodec) -> Option<EncoderBackend> {
    match codec {
        VideoCodec::Av1 if cfg!(feature = "av1-sw-fallback") => Some(EncoderBackend::Av1),
        VideoCodec::H264 | VideoCodec::H265 if cfg!(feature = "h26x-fallback") => {
            Some(EncoderBackend::H26x)
        }
        // The only encoder these codecs have, in every build.
        c @ (VideoCodec::Vp8 | VideoCodec::Vp9 | VideoCodec::Mpeg2 | VideoCodec::Mpeg4 | VideoCodec::ProRes(_)) => {
            native_backend_for(c)
        }
        _ => None,
    }
}

/// Whether this build can encode `codec` with no encode silicon at all — see
/// [`software_backend_for`].
pub fn software_encode_available(codec: VideoCodec) -> bool {
    software_backend_for(codec).is_some()
}

/// The `--features` flag that would make [`software_encode_available`] true
/// for `codec`. For error messages that tell the operator what to rebuild.
pub fn software_feature_for(codec: VideoCodec) -> &'static str {
    match codec {
        VideoCodec::Av1 => "av1-sw-fallback",
        VideoCodec::H264 | VideoCodec::H265 => "h26x-fallback",
        // In every build: no feature to name. A message that asks for one is
        // never reached for these (`software_backend_for` is `Some`).
        VideoCodec::Vp8 | VideoCodec::Vp9 | VideoCodec::Mpeg2 | VideoCodec::Mpeg4 | VideoCodec::ProRes(_) => {
            "default"
        }
    }
}

/// Construct the QSV encoder. The hand-rolled oneVPL encoder (`qsv.rs`) handles
/// both 8-bit (NV12) and 10-bit (P010) AV1; under `not(qsv)` this hits the stub.
fn make_qsv_encoder(config: EncoderConfig, gpu_index: u32) -> Result<Box<dyn Encoder>> {
    Ok(Box::new(qsv::QsvEncoder::new(config, gpu_index)?))
}

/// Create the best available AV1 encoder.
///
/// Priority: NVENC (Ada+) → AMF (RDNA3+) → QSV (Arc / Meteor Lake+).
///
/// Then, when the build opts in (`av1-sw-fallback`, `h26x-fallback`), the
/// software tiers; otherwise a host without encode silicon hard-fails at
/// construction. All backends compiled in; availability checked at runtime.
/// The config `select_encoder` would hand a backend, without building one.
///
/// Backend construction needs hardware, so the folds below would otherwise
/// only be exercised on a machine with a GPU — which is not where a wrong
/// clamp gets noticed.
#[cfg(test)]
pub(crate) fn select_encoder_config_for_test(config: EncoderConfig) -> EncoderConfig {
    resolve_overrides(config)
}

/// Fold the overrides that name the same thing an `EncoderConfig` field does.
///
/// One place, because four backends each remembering to check is three chances
/// to forget, and the failures are silent in both directions.
fn resolve_overrides(config: EncoderConfig) -> EncoderConfig {
    // `overrides.keyframe_interval` names the same thing as the field, and
    // every backend already reads the field. Folding here means the two can
    // never disagree — the alternative is four backends each remembering to
    // check, and the one that forgets emits IDRs where the segmenter does not
    // expect them, which is a broken stream rather than a worse one.
    let config = match config.overrides.keyframe_interval {
        Some(interval) => EncoderConfig { keyframe_interval: interval, ..config },
        None => config,
    };

    // The quality delta has to apply to the CRF escape hatch too.
    //
    // `quality` is documented as "the caller's CRF, or `AUTO_FROM_TARGET` to
    // derive one from `target`", and the backends honour that by skipping the
    // whole `tuning` path when a real CRF is present — which is where the
    // per-rung delta is applied. So a caller that sets both a CRF and a policy
    // got the CRF and silently none of the policy.
    //
    // That is not hypothetical: a service passing an explicit CRF for every
    // rung shipped a ladder policy, watched the rung sizes barely move, and
    // had nothing in any log to say why. `target` and `tier` were equally
    // inert for it and had been all along.
    //
    // Both paths, one delta, and they cannot both apply: a real CRF means the
    // adapters were never consulted.
    match config.overrides.quality_delta {
        0 => config,
        delta if config.quality == AUTO_FROM_TARGET => {
            // Applied by the adapters, in each backend's own units.
            let _ = delta;
            config
        }
        delta => {
            // `quality` is a libaom-style CQ here — the same currency the
            // delta is denominated in — so it adds directly.
            let ceiling = i32::from(crf_scale_max(config.codec));
            let shifted = (i32::from(config.quality) + i32::from(delta)).clamp(0, ceiling);
            EncoderConfig { quality: shifted as u8, ..config }
        }
    }
}

pub fn select_encoder(
    config: EncoderConfig,
    preferred: Option<EncoderBackend>,
) -> Result<Box<dyn Encoder>> {
    let config = resolve_overrides(config);

    // ProRes, VP8, MPEG-2, MPEG-4 Part 2 — and VP9 in a build without QSV:
    // no hardware backend here encodes them, so their own encoder is the
    // encoder, built without looking for a GPU, whatever vendor the lease
    // named. VP9 with QSV compiled in goes down the chain below, which ends
    // in its own encoder.
    if preferred.is_none()
        && let Some(backend) = native_backend_for(config.codec)
        && !compiled_hardware_encodes(config.codec)
    {
        return create_backend(backend, config, &[]);
    }

    let gpus = gpu::detect_gpus();

    if let Some(backend) = preferred {
        return create_backend(backend, config, &gpus);
    }

    // No FFmpeg tier. It used to sit here, ahead of everything, probing
    // libavcodec's av1_nvenc / av1_amf / av1_qsv / av1_vaapi / libsvtav1 /
    // libaom-av1 / librav1e chain — one interface covering every vendor and
    // the CPU fallbacks at once.
    //
    // It was removed because of what it dragged in rather than what it did:
    // FFmpeg dev libraries on the build host, LLVM and libclang for bindgen,
    // shared objects on the runtime image, and an LGPL surface next to this
    // crate's own licence. A build either had all of that or silently lost its
    // software encoder.
    //
    // What it actually provided is covered by the tiers below without any of
    // that: hardware via the in-tree NVENC / AMF / QSV backends, which are
    // hand-rolled dlopen FFI and need no SDK at build time, and software via
    // this workspace's own encoders, which are pure Rust. See `encode/av1_sw.rs`.

    // Vendor-pin shortcut: when the caller has already chosen which
    // GPU to use (CMAF orchestrator does this via the GpuPool lease,
    // 2026-05-03), dispatch DIRECTLY to that vendor's backend
    // instead of running the NVIDIA-first preference chain.
    // Without this, a host with both NVIDIA + Intel GPUs always
    // routed every variant to NVENC because the chain hits
    // `pick_vendor_device(Nvidia, ...)` first; the Arc sat idle even
    // when NVENC sessions were saturated. The software tiers remain the
    // last resort if hardware init fails on the pinned vendor.
    if let Some(pinned) = config.gpu_vendor {
        // The leased card first, then its siblings of the same vendor.
        //
        // A pinned vendor says which silicon the lease bought, not which
        // *card* must serve it, and a host can hold several that differ in
        // what they can actually do. devbox carries an Arc A310, an A380 and
        // an A750; the A310 advertises AV1 encode and then answers
        // `MFXCreateSession: -9` — no hardware implementation for the codec —
        // so a job leased to index 0 failed outright while two cards that can
        // encode it sat idle beside it. Trying them is still GPU-only; it is
        // the difference between "this vendor" and "this one card".
        let mut candidates: Vec<&gpu::GpuDevice> = Vec::new();

        if let Some(dev) = pick_vendor_device(&gpus, pinned, config.gpu_index) {
            candidates.push(dev);
        }
        for dev in gpus.iter().filter(|d| d.vendor == pinned) {
            if !candidates.iter().any(|c| c.index == dev.index) {
                candidates.push(dev);
            }
        }

        if candidates.is_empty() {
            return Err(anyhow::anyhow!(
                "vendor-pinned encoder requested (vendor={pinned:?}, gpu_index={:?}) but no matching GPU found",
                config.gpu_index,
            ));
        }

        let mut refusals: Vec<String> = Vec::new();

        for dev in candidates {
            if !hardware_encodes(vendor_backend(dev.vendor), config.codec) {
                refusals.push(format!(
                    "{} (idx {}): {:?} has no {} encoder",
                    dev.name,
                    dev.index,
                    dev.vendor,
                    config.codec.label()
                ));
                continue;
            }
            if !gpu::supports_av1_encode(dev) {
                refusals.push(format!("{} (idx {}): no {:?} encode silicon", dev.name, dev.index, config.codec));
                continue;
            }

            let attempt = match pinned {
                gpu::GpuVendor::Nvidia => nvenc::NvencEncoder::new(config.clone(), dev.index)
                    .map(|e| Box::new(e) as Box<dyn Encoder>),
                gpu::GpuVendor::Amd => amf::AmfEncoder::new(config.clone(), dev.vendor_index)
                    .map(|e| Box::new(e) as Box<dyn Encoder>),
                gpu::GpuVendor::Intel => make_qsv_encoder(config.clone(), dev.index),
            };

            match attempt {
                Ok(enc) => {
                    tracing::debug!(
                        gpu_name = %dev.name,
                        gpu_index = dev.index,
                        vendor = ?pinned,
                        codec = ?config.codec,
                        "using vendor-pinned hardware encoder (lease-driven dispatch)"
                    );
                    return Ok(enc);
                }
                Err(e) => {
                    // A card that will not start is a card that declines, the
                    // same way a decoder does. What it must not do is end the
                    // job while a sibling could serve it.
                    tracing::warn!(
                        gpu_name = %dev.name,
                        gpu_index = dev.index,
                        vendor = ?pinned,
                        error = %e,
                        "this GPU could not start the encoder; trying the next of the same vendor"
                    );
                    refusals.push(format!("{} (idx {}): {e}", dev.name, dev.index));
                }
            }
        }

        // Every card refused the dispatcher. Before giving up, try the legacy
        // `MFXInit` path once.
        //
        // Some hosts enumerate nothing through the dispatcher while their
        // hardware is fine — devbox answers `-9` on every adapter index with
        // `vainfo` reporting AV1 encode on all three of its Arc cards, and its
        // decoder works because that path has always used `MFXInit`. Without
        // this, such a host contributes nothing but failed jobs.
        //
        // It cannot pin a card, so the runtime chooses and the job will not
        // spread. That is worth saying out loud, and worth doing only here —
        // after every pinned attempt has failed — rather than as a silent
        // per-card retry, which is the collapse this encoder stopped doing.
        if pinned == gpu::GpuVendor::Intel {
            match qsv::QsvEncoder::new_unpinned(config.clone()) {
                Ok(enc) => {
                    tracing::warn!(
                        vendor = ?pinned,
                        codec = ?config.codec,
                        refusals = %refusals.join("; "),
                        "no card accepted a pinned session; fell back to an unpinned one — \
                         this job will not spread across GPUs"
                    );
                    return Ok(Box::new(enc));
                }
                Err(e) => refusals.push(format!("unpinned MFXInit: {e}")),
            }
        }

        // GPU-only directive (2026-05-08): the caller pinned a vendor for a
        // reason (lease-driven GPU pool dispatch), so there is still no CPU
        // fallback here. Every card of that vendor has now refused, and the
        // error names each one so the failed-job event says which.
        return Err(anyhow::anyhow!(
            "no {:?} GPU on this host could start a {:?} encoder (vendor={pinned:?}): {}",
            pinned,
            config.codec,
            refusals.join("; "),
        ));
    }

    // Auto-select: NVIDIA NVENC (Ada+) first, then AMD AMF (RDNA3+),
    // then Intel QSV (Arc / Meteor Lake+). No CPU fallback; hosts
    // without any AV1 encode silicon hard-fail at the end of the chain.
    //
    // Per-vendor device resolution: when `config.gpu_index` is Some,
    // prefer the GPU with matching `.index` for that vendor so
    // multi-GPU hosts can pin variant N → device N. When None, fall
    // back to first-of-vendor (single-GPU behaviour preserved).
    if let Some(dev) = pick_vendor_device(&gpus, gpu::GpuVendor::Nvidia, config.gpu_index)
        && hardware_encodes(EncoderBackend::Nvenc, config.codec)
    {
        if gpu::supports_av1_encode(dev) {
            match nvenc::NvencEncoder::new(config.clone(), dev.index) {
                Ok(enc) => {
                    tracing::info!(
                        gpu_name = %dev.name,
                        gpu_index = dev.index,
                        codec = ?config.codec,
                        "using NVENC hardware encoder"
                    );
                    return Ok(Box::new(enc));
                }
                Err(e) => {
                    tracing::warn!(error = %e, "NVENC init failed, falling back to next backend");
                }
            }
        } else {
            // Capability gap, not an error: this NVIDIA GPU's NVENC silicon
            // predates AV1 encode (AV1 NVENC was added in Ada Lovelace
            // RTX 4000 and Ampere datacenter A10/A10G/L4/L40 — consumer
            // 30-series and older do NOT have it). The GPU can still
            // handle NVDEC decode; only the encode half falls through.
            tracing::info!(
                gpu = %dev.name,
                "NVIDIA GPU lacks AV1 NVENC silicon — trying other GPU backends"
            );
        }
    }

    if let Some(dev) = pick_vendor_device(&gpus, gpu::GpuVendor::Amd, config.gpu_index)
        && hardware_encodes(EncoderBackend::Amf, config.codec)
    {
        if gpu::supports_av1_encode(dev) {
            match amf::AmfEncoder::new(config.clone(), dev.vendor_index) {
                Ok(enc) => {
                    tracing::info!(
                        gpu_name = %dev.name,
                        gpu_index = dev.index,
                        codec = ?config.codec,
                        "using AMF hardware encoder"
                    );
                    return Ok(Box::new(enc));
                }
                Err(e) => {
                    tracing::warn!(error = %e, "AMF init failed, falling back to next backend");
                }
            }
        } else {
            tracing::info!(
                gpu = %dev.name,
                codec = ?config.codec,
                "AMD GPU has no AMF encode block for this codec; trying Intel / CPU"
            );
        }
    }

    if let Some(dev) = pick_vendor_device(&gpus, gpu::GpuVendor::Intel, config.gpu_index)
        && hardware_encodes(EncoderBackend::Qsv, config.codec)
    {
        if gpu::supports_av1_encode(dev) {
            match make_qsv_encoder(config.clone(), dev.index) {
                Ok(enc) => {
                    tracing::info!(
                        gpu_name = %dev.name,
                        gpu_index = dev.index,
                        codec = ?config.codec,
                        "using QSV hardware encoder"
                    );
                    return Ok(enc);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "QSV init failed; chain exhausted");
                }
            }
        } else {
            tracing::info!(
                gpu = %dev.name,
                "Intel GPU predates Arc/Meteor Lake — no AV1 QSV silicon"
            );
        }
    }

    // VP9 after QSV: rivet's own encoder, always — it is the codec's default
    // encoder, not a fallback a build opts into. A QSV that declined (no
    // Intel card, a card without VP9 encode, a rung only the software
    // encoder codes — an average rate, 4:4:4, 12 bits) lands here.
    if let Some(backend) = native_backend_for(config.codec) {
        tracing::info!(codec = ?config.codec, "no GPU took this codec; rivet's own encoder");
        return create_backend(backend, config, &[]);
    }

    // Last tier: software, when the build asks for it — rivet's own AV1
    // encoder for AV1, the native h26x encoders for H.264 / H.265.
    //
    // Off by default, and that default is the important half. A throughput
    // fleet degrading silently into an encoder one to two orders of magnitude
    // slower reads as a capacity problem rather than the missing driver it
    // actually is — so a host with no encode silicon still hard-fails here
    // unless somebody has said, at build time, that slow output beats no
    // output.
    #[cfg(feature = "av1-sw-fallback")]
    if config.codec == VideoCodec::Av1 {
        match av1_sw::Av1Encoder::new(config.clone()) {
            Ok(enc) => return Ok(Box::new(enc)),
            Err(e) => {
                tracing::warn!(error = %e, "software AV1 fallback failed to initialise");
            }
        }
    }
    #[cfg(feature = "h26x-fallback")]
    if h26x_sw::H26xEncoder::supports(config.codec) {
        match h26x_sw::H26xEncoder::new(config.clone()) {
            Ok(enc) => return Ok(Box::new(enc)),
            Err(e) => {
                tracing::warn!(error = %e, "h26x software fallback failed to initialise");
            }
        }
    }

    let feature = software_feature_for(config.codec);
    Err(anyhow::anyhow!(
        "no {:?} encoder available — this host has no NVIDIA / AMD / Intel encode silicon for \
         it, or every vendor path failed to initialise. Rebuild with `--features {feature}` to \
         allow software encoding on hosts like this.",
        config.codec
    ))
}

/// Whether an AV1 encoder can actually be constructed for this device — the
/// authoritative, build-aware capability check. It runs the **same**
/// [`select_encoder`] dispatch a per-chunk worker uses, pinned to the device's
/// vendor + index, so `true` means a worker leased to this GPU will encode
/// rather than hard-fail. Used to drop AV1-incapable cards (e.g. a pre-Ada
/// NVIDIA that decodes via NVDEC but has no AV1 encode silicon) from the
/// multi-GPU encode pool, so a mixed-vendor host encodes on the capable cards
/// instead of aborting when a chunk leases to an incapable one.
///
/// The probe constructs + immediately drops a real encoder, so the verdict is
/// cached per GPU index (queried once per process).
/// Whether `dev` can encode `codec` in hardware — probed by actually building
/// the encoder the worker would use (vendor-pinned to this GPU) and seeing if
/// init succeeds. Cached per `(gpu_index, codec)` since a GPU may encode H.264
/// but not AV1 (e.g. NVIDIA Ampere consumer: H.264/H.265 yes, AV1 no). A GPU
/// that fails is dropped from the *encode* pool for that codec but stays usable
/// for decode.
pub fn encode_capable(dev: &gpu::GpuDevice, codec: VideoCodec) -> bool {
    encode_capable_at(dev, codec, false)
}

/// [`encode_capable`] for an output of a given depth: `ten_bit` probes the
/// encoder at `Yuv420p10le`, the format a 10-bit rung configures, so a card
/// whose encoder takes the codec only at 8 bits — NVENC, AMF and QSV for
/// H.264, see [`backend_output_caps_for`] — answers no, and a pool of leases
/// for a 10-bit output leaves it out. Cached per `(gpu_index, codec, ten_bit)`.
pub fn encode_capable_at(dev: &gpu::GpuDevice, codec: VideoCodec, ten_bit: bool) -> bool {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<(u32, VideoCodec, bool), bool>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (dev.index, codec, ten_bit);
    if let Some(&cached) = cache.lock().unwrap().get(&key) {
        return cached;
    }
    // A card whose vendor has no encoder for the codec (VP8, MPEG-2, MPEG-4
    // Part 2, ProRes anywhere; VP9 off Intel) is not asked.
    if !hardware_encodes(vendor_backend(dev.vendor), codec) {
        cache.lock().unwrap().insert(key, false);
        return false;
    }
    // A representative, widely-accepted probe size; codec support does not
    // depend on resolution, so any valid dims answer the capability question.
    let probe = EncoderConfig {
        width: 640,
        height: 480,
        frame_rate: 30.0,
        gpu_index: Some(dev.index),
        gpu_vendor: Some(dev.vendor),
        codec,
        pixel_format: if ten_bit {
            crate::frame::PixelFormat::Yuv420p10le
        } else {
            crate::frame::PixelFormat::Yuv420p
        },
        ..Default::default()
    };
    let capable = match select_encoder(probe, None) {
        Ok(_enc) => true, // encoder is dropped here, releasing the session
        Err(e) => {
            tracing::info!(
                gpu_index = dev.index,
                gpu = %dev.name,
                vendor = ?dev.vendor,
                ?codec,
                ten_bit,
                error = %e,
                "GPU cannot encode this codec — excluding it from the encode pool (still usable for decode)"
            );
            false
        }
    };
    cache.lock().unwrap().insert(key, capable);
    capable
}

/// Back-compat shim: AV1 encode capability (the inventory's "AV1" column).
pub fn av1_encode_capable(dev: &gpu::GpuDevice) -> bool {
    encode_capable(dev, VideoCodec::Av1)
}

fn create_backend(
    backend: EncoderBackend,
    config: EncoderConfig,
    gpus: &[gpu::GpuDevice],
) -> Result<Box<dyn Encoder>> {
    match backend {
        EncoderBackend::Nvenc => {
            let dev = pick_vendor_device(gpus, gpu::GpuVendor::Nvidia, config.gpu_index)
                .ok_or_else(|| match config.gpu_index {
                    Some(idx) => anyhow::anyhow!(
                        "NVENC requested on GPU index {idx} but no NVIDIA GPU with that index found"
                    ),
                    None => anyhow::anyhow!("NVENC requested but no NVIDIA GPU found"),
                })?;
            Ok(Box::new(nvenc::NvencEncoder::new(config, dev.index)?))
        }
        EncoderBackend::Amf => {
            let dev = pick_vendor_device(gpus, gpu::GpuVendor::Amd, config.gpu_index).ok_or_else(
                || match config.gpu_index {
                    Some(idx) => anyhow::anyhow!(
                        "AMF requested on GPU index {idx} but no AMD GPU with that index found"
                    ),
                    None => anyhow::anyhow!("AMF requested but no AMD GPU found"),
                },
            )?;
            // AMF takes the vendor-local ordinal: it selects the DXGI adapter
            // the context binds to on Windows (see `amf::AmfEncoder::new`).
            Ok(Box::new(amf::AmfEncoder::new(config, dev.vendor_index)?))
        }
        EncoderBackend::Qsv => {
            let dev = pick_vendor_device(gpus, gpu::GpuVendor::Intel, config.gpu_index)
                .ok_or_else(|| match config.gpu_index {
                    Some(idx) => anyhow::anyhow!(
                        "QSV requested on GPU index {idx} but no Intel GPU with that index found"
                    ),
                    None => anyhow::anyhow!("QSV requested but no Intel GPU found"),
                })?;
            Ok(Box::new(qsv::QsvEncoder::new(config, dev.index)?))
        }
        // The two software tiers, by name. No feature check: the features
        // gate falling back unasked, and this caller asked.
        EncoderBackend::H26x => Ok(Box::new(h26x_sw::H26xEncoder::new(config)?)),
        EncoderBackend::Av1 => {
            if config.codec != VideoCodec::Av1 {
                anyhow::bail!("the software AV1 encoder was requested but the output codec is {:?}", config.codec);
            }
            Ok(Box::new(av1_sw::Av1Encoder::new(config)?))
        }
        EncoderBackend::ProRes => Ok(Box::new(prores_sw::ProresEncoder::new(config)?)),
        EncoderBackend::Vp8 => Ok(Box::new(vp8_sw::Vp8Encoder::new(config)?)),
        EncoderBackend::Vp9 => Ok(Box::new(vp9_sw::Vp9Encoder::new(config)?)),
        EncoderBackend::Mpeg2 => Ok(Box::new(mpeg2_sw::Mpeg2Encoder::new(config)?)),
        EncoderBackend::Mpeg4 => Ok(Box::new(mpeg4_sw::Mpeg4Encoder::new(config)?)),
    }
}

#[cfg(test)]
mod gpu_selection_tests {
    use super::*;
    use crate::gpu::{GpuDevice, GpuVendor};

    fn synth(index: u32, vendor: GpuVendor) -> GpuDevice {
        GpuDevice {
            index,
            vendor_index: index,
            vendor,
            name: format!("synthetic-{index}"),
            generation: String::new(),
            pci_id: String::new(),
            vram_mib: 0,
            serial: None,
            host_pci_address: String::new(),
            vendor_id_hex: String::new(),
        }
    }

    #[test]
    fn pick_vendor_device_defaults_to_first_of_vendor_when_no_request() {
        // requested=None → first matching vendor wins (pre-multi-GPU
        // behaviour preserved).
        let gpus = vec![
            synth(0, GpuVendor::Nvidia),
            synth(1, GpuVendor::Nvidia),
            synth(2, GpuVendor::Amd),
        ];
        let nv = pick_vendor_device(&gpus, GpuVendor::Nvidia, None).unwrap();
        assert_eq!(nv.index, 0);
        let amd = pick_vendor_device(&gpus, GpuVendor::Amd, None).unwrap();
        assert_eq!(amd.index, 2);
    }

    #[test]
    fn pick_vendor_device_honours_explicit_request() {
        // requested=Some(1) + vendor=Nvidia → must find GPU with
        // index==1 AND vendor==Nvidia, not just first Nvidia.
        let gpus = vec![
            synth(0, GpuVendor::Nvidia),
            synth(1, GpuVendor::Nvidia),
            synth(2, GpuVendor::Nvidia),
        ];
        let dev = pick_vendor_device(&gpus, GpuVendor::Nvidia, Some(1)).unwrap();
        assert_eq!(dev.index, 1);
        let dev2 = pick_vendor_device(&gpus, GpuVendor::Nvidia, Some(2)).unwrap();
        assert_eq!(dev2.index, 2);
    }

    #[test]
    fn pick_vendor_device_returns_none_when_index_vendor_mismatch() {
        // requested=Some(2) + vendor=Nvidia but GPU 2 is AMD → None.
        // select_encoder then falls through to the AMD tier which will
        // find GPU 2 on its own find() pass.
        let gpus = vec![synth(0, GpuVendor::Nvidia), synth(2, GpuVendor::Amd)];
        assert!(pick_vendor_device(&gpus, GpuVendor::Nvidia, Some(2)).is_none());
        // Confirm the AMD tier finds it correctly with the same request.
        let dev = pick_vendor_device(&gpus, GpuVendor::Amd, Some(2)).unwrap();
        assert_eq!(dev.index, 2);
    }

    #[test]
    fn pick_vendor_device_no_gpus_returns_none() {
        let gpus: Vec<GpuDevice> = vec![];
        assert!(pick_vendor_device(&gpus, GpuVendor::Nvidia, None).is_none());
        assert!(pick_vendor_device(&gpus, GpuVendor::Nvidia, Some(0)).is_none());
    }

    /// The software answer is the feature flag, per codec — and it is
    /// answered without building an encoder (nothing here touches a GPU or a
    /// thread pool; the test would take seconds if it did).
    #[test]
    fn software_backend_follows_the_fallback_features() {
        let av1 = software_backend_for(VideoCodec::Av1);
        let h264 = software_backend_for(VideoCodec::H264);
        let h265 = software_backend_for(VideoCodec::H265);
        if cfg!(feature = "av1-sw-fallback") {
            assert_eq!(av1, Some(EncoderBackend::Av1));
        } else {
            assert_eq!(av1, None);
        }
        if cfg!(feature = "h26x-fallback") {
            assert_eq!(h264, Some(EncoderBackend::H26x));
            assert_eq!(h265, Some(EncoderBackend::H26x));
        } else {
            assert_eq!(h264, None);
            assert_eq!(h265, None);
        }
        for c in [VideoCodec::Av1, VideoCodec::H264, VideoCodec::H265] {
            assert_eq!(software_encode_available(c), software_backend_for(c).is_some());
        }
        assert_eq!(software_feature_for(VideoCodec::Av1), "av1-sw-fallback");
        assert_eq!(software_feature_for(VideoCodec::H264), "h26x-fallback");
        assert_eq!(software_feature_for(VideoCodec::H265), "h26x-fallback");
    }

    /// H.264 at 10 bits is the software tier's alone: every hardware backend
    /// reports it 8-bit SDR (they refuse a High 10 request), the `h26x` tier
    /// reports 10-bit HDR, and every other (backend, codec) pair is what the
    /// per-backend answer already said — or the floor, for a codec the
    /// backend does not serve.
    #[test]
    fn ten_bit_h264_is_reported_for_the_software_tier_only() {
        let ten_hdr = OutputCaps { max_bit_depth: 10, hdr: true };
        for hw in [EncoderBackend::Nvenc, EncoderBackend::Amf, EncoderBackend::Qsv] {
            assert_eq!(backend_output_caps_for(hw, VideoCodec::H264), EIGHT_BIT_SDR, "{hw:?} H.264");
            assert_eq!(backend_output_caps_for(hw, VideoCodec::H265), ten_hdr, "{hw:?} H.265");
            assert_eq!(backend_output_caps_for(hw, VideoCodec::Av1), ten_hdr, "{hw:?} AV1");
        }
        assert_eq!(backend_output_caps_for(EncoderBackend::H26x, VideoCodec::H264), ten_hdr);
        assert_eq!(backend_output_caps_for(EncoderBackend::H26x, VideoCodec::H265), ten_hdr);
        assert_eq!(backend_output_caps_for(EncoderBackend::H26x, VideoCodec::Av1), EIGHT_BIT_SDR);
        let ten_sdr = OutputCaps { max_bit_depth: 10, hdr: false };
        assert_eq!(backend_output_caps_for(EncoderBackend::Av1, VideoCodec::Av1), ten_sdr);
        for c in [VideoCodec::H264, VideoCodec::H265] {
            assert_eq!(backend_output_caps_for(EncoderBackend::Av1, c), EIGHT_BIT_SDR, "software av1 {c:?}");
        }
        // The build answer for H.264 is 10-bit exactly when the software tier
        // is compiled in; the hardware features alone never make it so.
        let h264 = build_output_caps_for(VideoCodec::H264);
        assert_eq!(h264.max_bit_depth == 10, cfg!(feature = "h26x-fallback"), "{h264:?}");
        // And no per-codec build answer claims more than the codec-agnostic one.
        let all = build_output_caps();
        for c in [VideoCodec::Av1, VideoCodec::H264, VideoCodec::H265] {
            let per = build_output_caps_for(c);
            assert!(per.max_bit_depth <= all.max_bit_depth && (!per.hdr || all.hdr), "{c:?}: {per:?} vs {all:?}");
        }
    }

    #[test]
    fn encoder_config_default_has_no_gpu_pin() {
        // Default is None so existing callers using `EncoderConfig {
        // ..default() }` literals get the pre-multi-GPU first-of-vendor
        // behaviour unchanged.
        let cfg = EncoderConfig::default();
        assert_eq!(cfg.gpu_index, None);
    }
}
