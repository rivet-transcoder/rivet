//! GPU-only decode dispatch.
//!
//! Per the 2026-05-08 directive: every CPU decoder (openh264, libde265,
//! libvpx, rav1d, libmpeg2, libxvidcore, pure-Rust ProRes) was deleted
//! along with the legacy `FallbackDecoder` GPU→CPU fallover. The
//! production binary supports exactly two backends:
//!
//!   - NVDEC (NVIDIA, via libnvcuvid)
//!   - QSV   (Intel,  via libvpl + iHD)
//!
//! Hosts without one of those (no NVIDIA, no Intel Arc / Meteor Lake,
//! or a codec the local GPU can't decode) hard-fail at
//! [`create_decoder`]. There is no CPU decode path of any shape.

#[cfg(feature = "amd")]
pub mod amf_dec;
#[cfg(feature = "nvidia")]
pub mod nvdec;
#[cfg(feature = "qsv")]
pub mod qsv_dec;
// libavcodec, the broad software tier. Feature-gated because it is the one
// backend needing anything from the host at build time.
#[cfg(feature = "ffmpeg")]
pub mod ffmpeg;
// Native H.264 / HEVC: this workspace's own decoders (`crates/h26x`), pure
// Rust, always compiled — the software tier for those two codecs, ahead of
// libavcodec, which then only catches what they refuse.
pub mod h26x_sw;
// Software H.264, the narrow one below it.
#[cfg(feature = "openh264-fallback")]
pub mod openh264_sw;
// Software AV1 decode. Always compiled — the `rav1d` feature decides whether
// the dispatch chain FALLS BACK to it, not whether it exists.
pub mod rav1d_sw;

use crate::frame::{StreamInfo, VideoFrame};
use crate::gpu;

/// Deinterleave an NV12 frame (Y plane + interleaved UV plane, each with its
/// own row stride) into a tightly-packed `Yuv420p` buffer (Y, then U, then V).
/// A shared NV12 deinterleave helper for the GPU decode paths.
#[cfg(any(feature = "nvidia", feature = "amd", feature = "qsv"))]
#[allow(dead_code)]
pub(crate) fn nv12_planes_to_yuv420p(
    y: &[u8],
    y_stride: usize,
    uv: &[u8],
    uv_stride: usize,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let cw = width / 2;
    let ch = height / 2;
    let mut out = Vec::with_capacity(width * height + 2 * cw * ch);
    for row in 0..height {
        let off = row * y_stride;
        out.extend_from_slice(&y[off..off + width]);
    }
    // U then V, deinterleaved from the UV plane.
    let mut u_plane = Vec::with_capacity(cw * ch);
    let mut v_plane = Vec::with_capacity(cw * ch);
    for row in 0..ch {
        let off = row * uv_stride;
        let r = &uv[off..off + cw * 2];
        for c in 0..cw {
            u_plane.push(r[2 * c]);
            v_plane.push(r[2 * c + 1]);
        }
    }
    out.extend_from_slice(&u_plane);
    out.extend_from_slice(&v_plane);
    out
}

/// Deinterleave host **P010** planes (Y `u16` + interleaved UV `u16`, 10-bit in
/// the HIGH bits) into a packed `Yuv420p10le` buffer (Y, U, V planar, 10-bit in
/// the LOW bits — `>> 6`). Shared by the AMD/Intel GPU decode paths.
#[cfg(any(feature = "amd", feature = "qsv"))]
#[allow(dead_code)]
pub(crate) fn p010_planes_to_yuv420p10le(
    y: &[u8],
    y_stride: usize,
    uv: &[u8],
    uv_stride: usize,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let cw = width.div_ceil(2);
    let ch = height.div_ceil(2);
    let mut out = Vec::with_capacity((width * height + 2 * cw * ch) * 2);
    let rd = |buf: &[u8], off: usize| -> u16 {
        if off + 1 < buf.len() {
            u16::from_le_bytes([buf[off], buf[off + 1]]) >> 6
        } else {
            0
        }
    };
    for row in 0..height {
        let base = row * y_stride;
        for col in 0..width {
            out.extend_from_slice(&rd(y, base + col * 2).to_le_bytes());
        }
    }
    for row in 0..ch {
        let base = row * uv_stride;
        for col in 0..cw {
            out.extend_from_slice(&rd(uv, base + col * 4).to_le_bytes());
        }
    }
    for row in 0..ch {
        let base = row * uv_stride;
        for col in 0..cw {
            out.extend_from_slice(&rd(uv, base + col * 4 + 2).to_le_bytes());
        }
    }
    out
}
use anyhow::{Context, Result, bail};

/// A decoder whose frames arrive already rotated to how they should be seen.
///
/// # Why this wraps rather than being applied by callers
///
/// The rotation lives in the container, and everything downstream — the ladder,
/// the thumbnail, a per-title sample — wants the picture the right way up. Left
/// to callers it is a step each of them has to remember, and the one that
/// forgets produces output that is upside down while the others are fine.
/// Wrapping the decoder means a consumer cannot get this wrong, because it
/// never sees the unrotated frame.
///
/// `Rotation::None` hands frames straight through, so a source with no rotation
/// pays nothing for this existing.
pub struct RotatingDecoder {
    inner: Box<dyn Decoder>,
    degrees: u32,
    info: StreamInfo,
}

impl RotatingDecoder {
    /// Wrap `inner` so every frame is rotated `degrees` clockwise.
    ///
    /// Anything other than 90, 180 or 270 is a pass-through — including 0,
    /// which is the overwhelmingly common case.
    pub fn new(inner: Box<dyn Decoder>, degrees: u32) -> Box<dyn Decoder> {
        if !matches!(degrees, 90 | 180 | 270) {
            return inner;
        }

        // 90 and 270 turn the picture on its side, so everything downstream
        // that sizes itself from the stream — the ladder most of all — has to
        // be told the dimensions it will actually receive, not the ones the
        // container recorded.
        let mut info = inner.stream_info().clone();
        if matches!(degrees, 90 | 270) {
            std::mem::swap(&mut info.width, &mut info.height);
        }

        Box::new(Self { inner, degrees, info })
    }
}

impl Decoder for RotatingDecoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    fn push_sample(&mut self, data: &[u8]) -> Result<()> {
        self.inner.push_sample(data)
    }

    fn finish(&mut self) -> Result<()> {
        self.inner.finish()
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        let Some(frame) = self.inner.decode_next()? else { return Ok(None) };
        let rotated =
            crate::filter::apply(&frame, &crate::filter::VideoFilter::Rotate(self.degrees))
                .context("rotating a decoded frame")?;
        Ok(Some(rotated))
    }
}

pub trait Decoder: Send {
    fn stream_info(&self) -> &StreamInfo;

    /// Feed one Annex-B (or codec-native — AV1 OBU, VP9 superframe) sample
    /// into the decoder. Implementations may buffer internally until
    /// `finish` is called or may decode eagerly and buffer produced
    /// frames. Pull frames via `decode_next` at any point.
    fn push_sample(&mut self, data: &[u8]) -> Result<()>;

    /// Signal end-of-stream. After this, no more `push_sample` calls;
    /// `decode_next` drains remaining frames.
    fn finish(&mut self) -> Result<()>;

    fn decode_next(&mut self) -> Result<Option<VideoFrame>>;
}

/// Truthy-string parse for env-var opt-outs. `1` / `true` / `yes` / `on`
/// / `y` / `t` (case-insensitive) all resolve true; anything else is
/// false. Mirrors the encode-side helper for symmetry.
#[cfg(feature = "nvidia")]
fn env_flag_truthy(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => {
            let v = v.to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on" | "y" | "t")
        }
        Err(_) => false,
    }
}

/// Per-codec NVDEC opt-out check. Mirrors the previous-stack
/// `DISABLE_NVDEC_<CODEC>` granular knob: `DISABLE_NVDEC=1` blocks every
/// codec, `DISABLE_NVDEC_H264=1` blocks just one. Used as a debugging
/// escape hatch when a specific codec/driver combo is misbehaving on
/// the active host (e.g. Blackwell + 4K H.264 silent-stall).
#[cfg(feature = "nvidia")]
fn nvdec_disabled_for(codec_lower: &str) -> bool {
    if env_flag_truthy("DISABLE_NVDEC") {
        return true;
    }
    let codec_canonical = match codec_lower {
        "h264" | "avc1" | "avc" => "H264",
        "h265" | "hevc" | "hvc1" | "hev1" | "hvc2" | "hev2" => "HEVC",
        "vp8" => "VP8",
        "vp9" | "vp09" => "VP9",
        "av1" | "av01" => "AV1",
        "mpeg2" | "mpeg2video" => "MPEG2",
        "mpeg4" | "mp4v" => "MPEG4",
        _ => return false,
    };
    env_flag_truthy(&format!("DISABLE_NVDEC_{codec_canonical}"))
}

/// Codecs the NVDEC streaming dispatch supports.
#[cfg(feature = "nvidia")]
fn nvdec_supports(codec_lower: &str) -> bool {
    matches!(
        codec_lower,
        "h264"
            | "avc1"
            | "avc"
            | "h265"
            | "hevc"
            | "hvc1"
            | "hev1"
            | "hvc2"
            | "hev2"
            | "vp8"
            | "vp9"
            | "vp09"
            | "av1"
            | "av01"
            | "mpeg2"
            | "mpeg2video"
            | "mpeg4"
            | "mp4v"
    )
}

/// Decode backends compiled into this build, in dispatch-preference order.
pub fn decode_backends() -> Vec<&'static str> {
    let mut v = Vec::new();
    if cfg!(feature = "nvidia") {
        v.push("nvdec");
    }
    if cfg!(feature = "amd") {
        v.push("amf");
    }
    if cfg!(feature = "qsv") {
        v.push("qsv");
    }
    v.push("h26x");
    if cfg!(feature = "ffmpeg") {
        v.push("ffmpeg");
    }
    if cfg!(feature = "openh264-fallback") {
        v.push("openh264");
    }
    if cfg!(feature = "rav1d-fallback") {
        v.push("rav1d");
    }
    v
}

/// One codec's decode support across the compiled backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeSupport {
    /// Canonical codec label, e.g. `"h264"`.
    pub codec: &'static str,
    /// Backend names that can decode it in this build (`"nvdec"`, `"amf"`,
    /// `"qsv"`, `"rav1d"`). Empty = this build can't decode it.
    pub backends: Vec<&'static str>,
}

/// Which compiled backends decode each common codec, for `rivet capabilities`.
pub fn decode_capabilities() -> Vec<DecodeSupport> {
    const CODECS: &[&str] = &[
        "h264", "hevc", "vp8", "vp9", "av1", "mpeg2", "mpeg4", "prores",
    ];
    CODECS
        .iter()
        .map(|&codec| {
            let mut backends: Vec<&'static str> = Vec::new();
            #[cfg(feature = "nvidia")]
            if nvdec_supports(codec) {
                backends.push("nvdec");
            }
            #[cfg(feature = "amd")]
            if amf_dec::supports(codec) {
                backends.push("amf");
            }
            // QSV: ask the driver what this host's silicon can actually decode
            // (MFXVideoDECODE_Query), not just what the build handles — so the
            // report reflects the real adapter (e.g. an older iGPU without AV1
            // decode). Probed once + cached; empty on a non-Intel host.
            #[cfg(feature = "qsv")]
            if qsv_dec::probe_decode_caps().contains(&codec) {
                backends.push("qsv");
            }
            // The software tiers, in the order they are tried. Listed at all
            // because a report that omitted them would understate what this
            // build can do on a host with no decode silicon — and listed only
            // for codecs each one actually serves, because the opposite
            // mistake is what got the previous FFmpeg integration deleted:
            // eight codecs advertised through a decoder `create_decoder` never
            // constructed.
            if h26x_sw::supports(codec) && !h26x_disabled() {
                backends.push("h26x");
            }
            #[cfg(feature = "ffmpeg")]
            if matches!(
                codec,
                "h264" | "h265" | "hevc" | "vp8" | "vp9" | "av1" | "mpeg2" | "mpeg4" | "prores"
            ) {
                backends.push("ffmpeg");
            }
            #[cfg(feature = "openh264-fallback")]
            if codec == "h264" {
                backends.push("openh264");
            }
            #[cfg(feature = "rav1d-fallback")]
            if codec == "av1" {
                backends.push("rav1d");
            }
            DecodeSupport { codec, backends }
        })
        .collect()
}

/// Construct a hardware decoder for `codec`. NVIDIA GPUs win on tie
/// when both vendors are present (NVDEC is generally lower-latency on
/// the standard codec set + is what the production fleet has been
/// tuned against). When NVDEC is disabled per env-var or doesn't
/// support the codec, fall through to QSV. If neither fits, hard-fail
/// — there is no CPU fallback.
pub fn create_decoder(codec: &str, info: StreamInfo) -> Result<Box<dyn Decoder>> {
    create_decoder_on(codec, info, None)
}

/// Construct a decoder pinned to a specific `gpu_index` when one is
/// supplied. `None` preserves the legacy "pick the first matching
/// adapter" behaviour for one-shot callers (thumbnails, tests, benches)
/// that don't care about distributing work across physical GPUs.
///
/// The pipeline's per-rung decode pumps should ALWAYS pass `Some(idx)`
/// so each rung's decode session lands on a distinct adapter — without
/// this, every QSV session piles onto the first physical Intel card
/// regardless of what the GPU pool's lease said. See the project memo
/// on QSV multi-adapter session pinning.
pub fn create_decoder_on(
    codec: &str,
    info: StreamInfo,
    gpu_index: Option<u32>,
) -> Result<Box<dyn Decoder>> {
    let codec_lower = codec.to_ascii_lowercase();
    let gpus = gpu::detect_gpus();

    // Pick the device. If the caller specified gpu_index, honour it
    // (matching against `g.index`). Otherwise fall back to the first
    // of each vendor — the legacy behaviour for callers that don't
    // care about pinning.
    #[cfg(feature = "nvidia")]
    let nvidia = match gpu_index {
        Some(idx) => gpus
            .iter()
            .find(|g| matches!(g.vendor, gpu::GpuVendor::Nvidia) && g.index == idx),
        None => gpus
            .iter()
            .find(|g| matches!(g.vendor, gpu::GpuVendor::Nvidia)),
    };

    // NVIDIA / NVDEC first — our hand-rolled CUVID FFI (`nvidia` feature). One
    // portable decoder for everything NVDEC handles: H.264/HEVC/AV1/VP8/VP9,
    // MPEG-2/MPEG-4 Part 2, and 10-bit P016.
    #[cfg(feature = "nvidia")]
    if let Some(dev) = nvidia
        && nvdec_supports(&codec_lower)
        && !nvdec_disabled_for(&codec_lower)
    {
        tracing::info!(
            backend = "nvdec",
            codec = %codec_lower,
            gpu_index = dev.index,
            gpu_name = %dev.name,
            "NVDEC decoder engaged (hand-rolled CUVID FFI)"
        );
        // A tier that cannot start is a tier that declines, not a job that
        // fails. See the QSV arm below, which is where this cost a real
        // upload.
        return Ok(guarded(
            nvdec::NvdecDecoder::new(info.clone(), dev.vendor_index),
            &codec_lower,
            info,
        ));
    }

    // AMD / AMF hardware decode — hand-rolled AMF FFI (`amd` feature).
    #[cfg(feature = "amd")]
    {
        let amd = match gpu_index {
            Some(idx) => gpus
                .iter()
                .find(|g| matches!(g.vendor, gpu::GpuVendor::Amd) && g.index == idx),
            None => gpus
                .iter()
                .find(|g| matches!(g.vendor, gpu::GpuVendor::Amd)),
        };
        if let Some(dev) = amd
            && amf_dec::supports(&codec_lower)
        {
            tracing::info!(
                backend = "amf",
                codec = %codec_lower,
                gpu_index = dev.index,
                gpu_name = %dev.name,
                "AMF decoder engaged (hand-rolled AMF FFI)"
            );
            match amf_dec::AmfDecoder::new(info.clone(), dev.vendor_index) {
                Ok(decoder) => {
                    return Ok(guarded(Box::new(decoder), &codec_lower, info));
                }
                Err(e) => tracing::warn!(
                    error = %e,
                    codec = %codec_lower,
                    gpu_index = dev.index,
                    "AMF decode could not start; trying the next tier"
                ),
            }
        }
    }

    // Intel / QSV hardware decode — hand-rolled oneVPL FFI (`qsv` feature).
    #[cfg(feature = "qsv")]
    {
        let intel = match gpu_index {
            Some(idx) => gpus
                .iter()
                .find(|g| matches!(g.vendor, gpu::GpuVendor::Intel) && g.index == idx),
            None => gpus
                .iter()
                .find(|g| matches!(g.vendor, gpu::GpuVendor::Intel)),
        };
        if let Some(dev) = intel
            && qsv_dec::supports(&codec_lower)
        {
            tracing::info!(
                backend = "qsv",
                codec = %codec_lower,
                gpu_index = dev.index,
                gpu_name = %dev.name,
                "QSV decoder engaged (hand-rolled oneVPL FFI)"
            );
            // Declining, not failing.
            //
            // `MFXVideoDECODE_Init failed: -3` is MFX_ERR_UNSUPPORTED: the card
            // is there and oneVPL loaded, and it will not decode *this* stream
            // — a profile or a resolution outside what the fixed-function block
            // handles. Propagating that killed the job outright on a host with
            // a perfectly good software decoder compiled in and every other
            // tier untried. A real 1920x818 H.264 upload died this way while a
            // 640x360 clip through the same worker succeeded.
            match qsv_dec::QsvDecoder::new(info.clone(), dev.vendor_index) {
                Ok(decoder) => {
                    return Ok(guarded(Box::new(decoder), &codec_lower, info));
                }
                Err(e) => tracing::warn!(
                    error = %e,
                    codec = %codec_lower,
                    gpu_index = dev.index,
                    "QSV decode could not start; trying the next tier"
                ),
            }
        }
    }

    create_software_decoder(&codec_lower, info)
}

/// The tiers that need no hardware.
///
/// Split out of [`create_decoder_on`] so a hardware decoder that fails *after*
/// being chosen can still reach them — see [`HardwareThenSoftware`]. Inline,
/// they were reachable only by falling off the end of the tier list, which a
/// decoder that has already been returned can never do.
fn create_software_decoder(codec_lower: &str, info: StreamInfo) -> Result<Box<dyn Decoder>> {
    // The native H.264 / HEVC decoders first among the software tiers.
    //
    // Pure Rust, always compiled, bit-exact against the conformance suites,
    // threaded across the machine — see `h26x_sw`. Ahead of libavcodec because
    // this is the workspace's own decoder and needs nothing from the host;
    // libavcodec (when built) is the tier behind it for what it refuses:
    // interlaced H.264, 4:2:2, the odd profile. A refusal is said up front on
    // the parameter set, so the guard rebuilds the next tier and replays the
    // samples fed so far.
    //
    // `RIVET_DISABLE_H26X=1` skips it, for comparing against the tiers below.
    if h26x_sw::supports(codec_lower) && !h26x_disabled() {
        let mut native_info = info.clone();
        native_info.codec = codec_lower.to_string();
        match h26x_sw::H26xDecoder::new(native_info) {
            Ok(dec) => {
                tracing::info!(
                    backend = "h26x",
                    codec = %codec_lower,
                    "native software decode engaged (rivet's own H.264/HEVC decoders)"
                );
                let codec = codec_lower.to_string();
                return Ok(Box::new(HardwareThenSoftware {
                    primary: Box::new(dec),
                    fallback: Some(Box::new(move || {
                        create_software_decoder_below_native(&codec, info)
                    })),
                    replay: Vec::new(),
                }));
            }
            Err(e) => tracing::warn!(
                error = %e,
                codec = %codec_lower,
                "the native decoder could not start; trying the next software tier"
            ),
        }
    }
    create_software_decoder_below_native(codec_lower, info)
}

/// `RIVET_DISABLE_H26X=1` takes the native tier out of the chain.
fn h26x_disabled() -> bool {
    matches!(
        std::env::var("RIVET_DISABLE_H26X").as_deref().map(str::to_ascii_lowercase).as_deref(),
        Ok("1" | "true" | "yes" | "on" | "y" | "t")
    )
}

/// The software tiers behind the native one: libavcodec, openh264, rav1d.
fn create_software_decoder_below_native(
    codec_lower: &str,
    info: StreamInfo,
) -> Result<Box<dyn Decoder>> {
    // libavcodec first among the remaining software tiers, when the build has it.
    //
    // Below the hardware ones deliberately — NVDEC and QSV are faster and
    // proven here — and above the per-codec modules because when there is no
    // GPU, breadth matters. Those modules are narrow, and for H.264 only
    // dependable on the profiles openh264 handles well: a High-profile 1080p
    // upload decoded eleven of its 5,533 frames through openh264, every
    // rendition came out under half a second while the audio ran the full 221,
    // and openh264 reported `dsNoParamSets` on frame after frame that
    // libavcodec reads without complaint.
    #[cfg(feature = "ffmpeg")]
    {
        let mut info = info.clone();
        if info.codec.is_empty() {
            // `FfmpegDecoder` maps its codec id from `StreamInfo`, and callers
            // that resolved the label separately may not have set it.
            info.codec = codec_lower.to_string();
        }

        match ffmpeg::FfmpegDecoder::new(info) {
            Ok(dec) => {
                tracing::info!(
                    backend = "ffmpeg",
                    codec = %codec_lower,
                    "libavcodec software decode engaged"
                );
                return Ok(Box::new(dec));
            }
            Err(e) => tracing::warn!(
                error = %e,
                codec = %codec_lower,
                "libavcodec could not start; trying the narrower software tiers"
            ),
        }
    }

    // Software H.264, when the build asks for it.
    //
    // On a host with no GPU and no libavcodec this is the only one. H.264 is
    // what cameras, phones and every existing library produce, so without it a
    // GPU-less worker accepts a job, downloads it, probes it and then has
    // nothing to decode it with — while the encode side falls back to rav1e
    // quite happily and makes the host look capable.
    #[cfg(feature = "openh264-fallback")]
    if codec_lower == "h264" || codec_lower == "avc1" {
        match openh264_sw::OpenH264SwDecoder::new(info.clone()) {
            Ok(dec) => {
                tracing::warn!(
                    backend = "openh264",
                    codec = %codec_lower,
                    "software H.264 decode engaged; no hardware decoder was available"
                );
                return Ok(Box::new(dec));
            }
            Err(e) => {
                tracing::warn!(error = %e, "openh264 software fallback failed to initialise");
            }
        }
    }

    // Last tier: software AV1, when the build asks for it.
    //
    // AV1 only — rav1d decodes nothing else, and this is not the place to
    // pretend otherwise. It matters more here than on the encode side: NVDEC
    // gained AV1 in Ampere while NVENC only got it in Ada, so a host can encode
    // AV1 in hardware and still have no way to decode it.
    #[cfg(feature = "rav1d-fallback")]
    if codec_lower == "av1" {
        match rav1d_sw::Rav1dDecoder::new(info.clone()) {
            Ok(dec) => return Ok(Box::new(dec)),
            Err(e) => {
                tracing::warn!(error = %e, "rav1d software fallback failed to initialise");
            }
        }
    }

    bail!(
        "no decoder available for codec '{}' on this host \n         (NVIDIA GPUs cover h264/h265/vp8/vp9/av1/mpeg2/mpeg4; \n          Intel Arc/Meteor Lake+ covers h264/h265/vp9/av1; \n          the native software tier covers progressive 4:2:0 H.264 and HEVC). \n         Rebuild with `--features ffmpeg` for the rest of H.264/HEVC in software, or \n         `--features rav1d-fallback` for software AV1.",
        codec_lower
    )
}

/// Wrap a hardware decoder so a late refusal degrades instead of failing.
///
/// A hardware decoder can accept construction and then refuse the first real
/// sample, by which point every other tier has been passed over. A real
/// 1920x818 upload failed exactly there, on a host whose software decoder was
/// compiled in, enabled, and never reached.
///
/// This keeps the fallback available past that point: the first sample the
/// hardware refuses rebuilds the next tier and replays everything fed so far,
/// so the job continues instead of ending. After the first successful sample
/// the hardware has proved itself and the fallback is dropped — a decoder that
/// fails on sample nine thousand is a real failure, not a capability question,
/// and pretending otherwise would silently re-decode a whole video.
fn guarded(primary: Box<dyn Decoder>, codec_lower: &str, info: StreamInfo) -> Box<dyn Decoder> {
    let codec = codec_lower.to_string();

    Box::new(HardwareThenSoftware {
        primary,
        fallback: Some(Box::new(move || create_software_decoder(&codec, info))),
        replay: Vec::new(),
    })
}

struct HardwareThenSoftware {
    primary: Box<dyn Decoder>,
    /// Rebuilds the next tier down. `None` once the primary has decoded
    /// something, or once it has been used.
    fallback: Option<Box<dyn FnOnce() -> Result<Box<dyn Decoder>> + Send>>,
    /// Everything pushed before the primary proved itself, to replay.
    replay: Vec<Vec<u8>>,
}

impl HardwareThenSoftware {
    /// Swap in the fallback and replay what the primary was given.
    fn degrade(&mut self, why: &anyhow::Error) -> Result<()> {
        let Some(build) = self.fallback.take() else {
            anyhow::bail!("{why}");
        };

        tracing::warn!(
            error = %why,
            "the hardware decoder refused this stream; falling back to software"
        );

        let mut replacement = build()?;
        for sample in std::mem::take(&mut self.replay) {
            replacement.push_sample(&sample)?;
        }

        self.primary = replacement;
        Ok(())
    }
}

impl Decoder for HardwareThenSoftware {
    fn stream_info(&self) -> &StreamInfo {
        self.primary.stream_info()
    }

    fn push_sample(&mut self, data: &[u8]) -> Result<()> {
        if self.fallback.is_some() {
            self.replay.push(data.to_vec());
        }

        match self.primary.push_sample(data) {
            Ok(()) => {
                // Proved. Stop holding samples for a replay that will not
                // happen — on a long video that buffer is the whole file.
                self.fallback = None;
                self.replay = Vec::new();
                Ok(())
            }
            Err(e) => {
                self.degrade(&e)?;
                Ok(())
            }
        }
    }

    fn finish(&mut self) -> Result<()> {
        self.primary.finish()
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        self.primary.decode_next()
    }
}

/// GPU indices whose vendor decoder can handle `codec` in this build (honoring
/// the `DISABLE_NVDEC*` knobs). These are exactly the candidates
/// `create_decoder_on(.., Some(idx))` would dispatch a decoder for — the
/// `--decode-with-fastest` benchmark times each one and pins the pump to the
/// quickest. Order follows `detect_gpus()`.
pub fn decode_capable_gpu_indices(codec: &str) -> Vec<u32> {
    let codec_lower = codec.to_ascii_lowercase();
    gpu::detect_gpus()
        .iter()
        .filter(|g| match g.vendor {
            gpu::GpuVendor::Nvidia => nvidia_can_decode(&codec_lower),
            gpu::GpuVendor::Amd => amd_can_decode(&codec_lower),
            gpu::GpuVendor::Intel => intel_can_decode(&codec_lower),
        })
        .map(|g| g.index)
        .collect()
}

#[cfg(feature = "nvidia")]
fn nvidia_can_decode(c: &str) -> bool {
    nvdec_supports(c) && !nvdec_disabled_for(c)
}
#[cfg(not(feature = "nvidia"))]
fn nvidia_can_decode(_c: &str) -> bool {
    false
}

#[cfg(feature = "amd")]
fn amd_can_decode(c: &str) -> bool {
    amf_dec::supports(c)
}
#[cfg(not(feature = "amd"))]
fn amd_can_decode(_c: &str) -> bool {
    false
}

#[cfg(feature = "qsv")]
fn intel_can_decode(c: &str) -> bool {
    qsv_dec::supports(c)
}
#[cfg(not(feature = "qsv"))]
fn intel_can_decode(_c: &str) -> bool {
    false
}

#[cfg(test)]
mod rotating_decoder_tests {
    use super::*;
    use crate::frame::{ColorSpace, PixelFormat};

    /// A decoder that yields one frame with a distinctive top-left pixel.
    struct OneFrame {
        info: StreamInfo,
        yielded: bool,
    }

    impl OneFrame {
        fn boxed(w: u32, h: u32) -> Box<dyn Decoder> {
            let info = StreamInfo {
                codec: "h264".into(),
                width: w,
                height: h,
                frame_rate: 30.0,
                duration: 1.0,
                pixel_format: PixelFormat::Yuv420p,
                color_space: ColorSpace::Bt709,
                total_frames: 1,
                bitrate: 0,
                color_metadata: crate::frame::ColorMetadata::default(),
            };
            Box::new(Self { info, yielded: false })
        }
    }

    impl Decoder for OneFrame {
        fn stream_info(&self) -> &StreamInfo {
            &self.info
        }
        fn push_sample(&mut self, _: &[u8]) -> Result<()> {
            Ok(())
        }
        fn finish(&mut self) -> Result<()> {
            Ok(())
        }
        fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
            if self.yielded {
                return Ok(None);
            }
            self.yielded = true;
            let (w, h) = (self.info.width as usize, self.info.height as usize);
            let mut data = vec![0u8; w * h * 3 / 2];
            data[0] = 200; // top-left luma, the corner we track
            Ok(Some(VideoFrame::new(
                bytes::Bytes::from(data),
                self.info.width,
                self.info.height,
                PixelFormat::Yuv420p,
                ColorSpace::Bt709,
                0,
            )))
        }
    }

    #[test]
    fn a_180_rotation_moves_the_corner_to_the_opposite_corner() {
        // The production case. A marked top-left pixel must end up bottom-right
        // — which is what "upside down" means in pixels rather than in words.
        let (w, h) = (16u32, 8u32);
        let mut d = RotatingDecoder::new(OneFrame::boxed(w, h), 180);
        let frame = d.decode_next().unwrap().expect("a frame");

        assert_eq!((frame.width, frame.height), (w, h), "180 must not resize");
        let last = (w * h - 1) as usize;
        assert_eq!(frame.data[last], 200, "the marked corner did not move");
        assert_eq!(frame.data[0], 0, "the original corner still carries the mark");
    }

    #[test]
    fn ninety_degrees_swaps_the_reported_dimensions() {
        // Everything downstream sizes itself from `stream_info` — the ladder
        // above all. If it keeps reporting the container's dimensions, every
        // rung is computed for a picture the decoder will never hand over.
        let d = RotatingDecoder::new(OneFrame::boxed(1920, 1080), 90);
        assert_eq!((d.stream_info().width, d.stream_info().height), (1080, 1920));
    }

    #[test]
    fn no_rotation_is_the_decoder_itself() {
        // The overwhelmingly common case pays nothing: same dimensions, and no
        // per-frame copy in the path.
        let d = RotatingDecoder::new(OneFrame::boxed(1920, 1080), 0);
        assert_eq!((d.stream_info().width, d.stream_info().height), (1920, 1080));
    }
}
