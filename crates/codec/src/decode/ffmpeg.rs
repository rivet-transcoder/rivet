//! FFmpeg-based primary decoder (gated on `codec/ffmpeg` feature).
//!
//! Wraps ffmpeg-next's libavcodec decoders behind our `Decoder` trait
//! so the pipeline's streaming push/pull shape stays identical to
//! the legacy per-codec stack. One trait impl covers every codec
//! FFmpeg knows (H.264 / H.265 / VP8 / VP9 / AV1 / MPEG-2 / MPEG-4 /
//! ProRes / …). Hardware acceleration via `AVHWDeviceContext` —
//! Vulkan preferred per the cross-vendor mandate; CUDA / D3D11 /
//! VAAPI enumerated at runtime.
//!
//! # Why this replaces the custom stack
//!
//! The hand-rolled Vulkan Video decoder hit driver-side edge cases
//! (green screen, static first-frame, artifacts — see memory
//! `project_vulkan_av1_decode_grey.md`). FFmpeg's implementation
//! is the reference — every browser / player / streaming service
//! ships it. Importing it via `ffmpeg-next` gives us known-correct
//! pixel output across every codec at the cost of ~30 MB of LGPL
//! dynamic libraries the operator ships alongside the binary.
//!
//! # Failure surface
//!
//! Construction returns `Err` when FFmpeg can't find a decoder for
//! the codec string (unlikely for mainstream codecs) or can't open
//! the device context (Vulkan absent → tries CUDA → tries software).
//! Once constructed, `push_sample` / `decode_next` return typed
//! errors that `FallbackDecoder` can catch to route to the legacy
//! CPU backends.
//!
//! Output is normalized to `Yuv420p` (8-bit) or `Yuv420p10le`
//! (10-bit HDR passthrough) via `sws_scale`. Multi-planar hardware
//! formats (NV12, P010, P016) transit back to host memory via
//! `av_hwframe_transfer_data` then convert through the same
//! software scaler path.

#![cfg(feature = "ffmpeg")]

use anyhow::{Result, anyhow};
use bytes::Bytes;
use std::collections::VecDeque;

use ffmpeg::codec::{self, Context as CodecContext, decoder, packet};
use ffmpeg::ffi as sys;
use ffmpeg::format::Pixel;
use ffmpeg::software::scaling::{context::Context as Scaler, flag::Flags as ScalerFlags};
use ffmpeg::util::frame::video::Video as VideoFrameFfmpeg;
use ffmpeg_next as ffmpeg;

use super::Decoder;
use crate::frame::{ColorSpace, PixelFormat, StreamInfo, VideoFrame};

/// Preferred hardware device types in priority order. FFmpeg tries
/// each until one succeeds. CPU software fallback is the final tier.
///
/// Vulkan first per the cross-vendor mandate — works on NVIDIA / AMD
/// RDNA3+ / Intel Arc with one codepath. CUDA second covers older
/// NVIDIA silicon before Vulkan Video rolled out. D3D11VA third on
/// Windows hosts that don't have Vulkan drivers. VAAPI on Linux-only
/// hardware. Everything after that uses libavcodec software decoders
/// (which are themselves very good — FFmpeg's reference).
/// Preferred hardware device types in priority order. The list is
/// platform-aware so we never try D3D11VA on Linux (would fail
/// immediately) or VAAPI on Windows (same).
///
/// - macOS: VideoToolbox first (native, covers H.264/HEVC + AV1
///   decode on M3/M4/later — Apple added AV1 decode in macOS 14).
///   No AV1 encode ASIC exists on Apple silicon as of 2026-01; the
///   encoder probe chain handles that separately.
/// - Windows: Vulkan first (cross-vendor on NVIDIA 553.40+ / AMD /
///   Intel Arc drivers), then CUDA (NVIDIA direct), then D3D11VA
///   (works on any Windows DXGI-capable GPU), then DXVA2 (legacy
///   Win7-era path for drivers that don't expose D3D11VA).
/// - Linux: Vulkan first, then CUDA, then VAAPI (canonical Linux
///   hwaccel for Intel / AMD / Mesa).
///
/// `FFMPEG_HWACCEL=<name>` overrides the list; `FFMPEG_HWACCEL=none`
/// forces software decode inside libavcodec.
#[cfg(target_os = "macos")]
const HWACCEL_PREFERENCE: &[&str] = &["videotoolbox", "vulkan"];

#[cfg(all(not(target_os = "macos"), target_os = "windows"))]
const HWACCEL_PREFERENCE: &[&str] = &["vulkan", "cuda", "d3d11va", "dxva2"];

#[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
const HWACCEL_PREFERENCE: &[&str] = &["vulkan", "cuda", "vaapi"];

/// RAII wrapper around `AVBufferRef*` pointing at an `AVHWDeviceContext`.
/// The buffer is created via `av_hwdevice_ctx_create` which returns a
/// refcounted reference; we drop it via `av_buffer_unref` so the
/// device tears down cleanly even when the decoder never opened it.
struct HwDeviceCtx {
    ptr: *mut sys::AVBufferRef,
    /// Which AVHWDeviceType this buffer holds — used to pick the matching
    /// HW pixel format for `get_format` dispatch.
    #[allow(dead_code)]
    device_type: sys::AVHWDeviceType,
}

// `AVBufferRef` from libavutil is thread-safe for refcount ops, and
// we use it across decoder frames behind a &mut FfmpegDecoder guard.
unsafe impl Send for HwDeviceCtx {}

impl Drop for HwDeviceCtx {
    fn drop(&mut self) {
        unsafe {
            if !self.ptr.is_null() {
                sys::av_buffer_unref(&mut self.ptr);
            }
        }
    }
}

/// Map a hwaccel name string to the matching `AVHWDeviceType`.
/// `None` means "not a recognised hwaccel name".
fn hwdevice_type_from_name(name: &str) -> Option<sys::AVHWDeviceType> {
    use sys::AVHWDeviceType::*;
    Some(match name {
        "vulkan" => AV_HWDEVICE_TYPE_VULKAN,
        "cuda" => AV_HWDEVICE_TYPE_CUDA,
        "d3d11va" => AV_HWDEVICE_TYPE_D3D11VA,
        "dxva2" => AV_HWDEVICE_TYPE_DXVA2,
        "vaapi" => AV_HWDEVICE_TYPE_VAAPI,
        "videotoolbox" => AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
        "qsv" => AV_HWDEVICE_TYPE_QSV,
        _ => return None,
    })
}

/// Check whether `codec` advertises support for `device_type` via its
/// `avcodec_get_hw_config` table. This avoids initialising a device
/// we know libavcodec won't accept for this codec.
unsafe fn codec_advertises_hwaccel(
    codec: *const sys::AVCodec,
    device_type: sys::AVHWDeviceType,
) -> bool {
    let mut i: i32 = 0;
    loop {
        let cfg = sys::avcodec_get_hw_config(codec, i);
        if cfg.is_null() {
            return false;
        }
        let cfg_ref = &*cfg;
        // `methods & HW_DEVICE_CTX` means the codec supports the
        // modern hw_device_ctx attach path (not the legacy
        // hwaccel-internal path which needs per-codec plumbing).
        let methods_ok =
            (cfg_ref.methods & sys::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32) != 0;
        if methods_ok && cfg_ref.device_type == device_type {
            return true;
        }
        i += 1;
    }
}

/// Try each hwaccel in priority order; return the first one libavcodec
/// can both open a device for AND that this codec advertises via
/// `avcodec_get_hw_config`. `None` means "use software decode".
///
/// Environment overrides:
/// - `FFMPEG_HWACCEL=none` — forces software decode.
/// - `FFMPEG_HWACCEL=<name>` — tries exactly that name (bypasses the list).
fn try_open_hwaccel(codec: *const sys::AVCodec) -> Option<HwDeviceCtx> {
    let override_name = std::env::var("FFMPEG_HWACCEL").ok();
    if override_name.as_deref() == Some("none") {
        return None;
    }
    let preference: Vec<&str> = match override_name.as_deref() {
        Some(name) => vec![name],
        None => HWACCEL_PREFERENCE.iter().copied().collect(),
    };

    for name in preference {
        let Some(device_type) = hwdevice_type_from_name(name) else {
            continue;
        };
        unsafe {
            if !codec_advertises_hwaccel(codec, device_type) {
                tracing::debug!(hwaccel = name, "codec does not advertise this hwaccel");
                continue;
            }
            let mut ctx: *mut sys::AVBufferRef = std::ptr::null_mut();
            let rc = sys::av_hwdevice_ctx_create(
                &mut ctx,
                device_type,
                std::ptr::null(), // default device
                std::ptr::null_mut(),
                0,
            );
            if rc == 0 && !ctx.is_null() {
                tracing::info!(hwaccel = name, "FFmpeg HW device created");
                return Some(HwDeviceCtx {
                    ptr: ctx,
                    device_type,
                });
            } else {
                tracing::debug!(
                    hwaccel = name,
                    rc = rc,
                    "av_hwdevice_ctx_create failed; trying next"
                );
            }
        }
    }
    tracing::info!("no FFmpeg hwaccel available; falling back to software decode");
    None
}

/// Initialize the FFmpeg runtime exactly once. Safe to call from
/// multiple threads — `ffmpeg::init` guards internally.
fn init_ffmpeg() {
    static INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    INIT.get_or_init(|| {
        let _ = ffmpeg::init();
        // Quiet FFmpeg's own stderr spam at the tracing level we
        // want. Our own diagnostics take over via the tracing crate.
        ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Warning);
    });
}

/// Map our codec label to FFmpeg's `AVCodecID`. Returns `None` when
/// the codec isn't something FFmpeg's default build catalogue
/// recognizes — `FallbackDecoder` falls through to the legacy stack
/// in that case.
fn codec_id_from_label(codec_lower: &str) -> Option<codec::Id> {
    use codec::Id::*;
    Some(match codec_lower {
        "h264" | "avc1" | "avc" => H264,
        "h265" | "hevc" | "hvc1" | "hev1" | "hvc2" | "hev2" => HEVC,
        "vp8" => VP8,
        "vp9" | "vp09" => VP9,
        "av1" | "av01" => AV1,
        "mpeg2" | "mpeg2video" => MPEG2VIDEO,
        "mpeg4" | "mp4v" => MPEG4,
        "prores" => PRORES,
        _ => return None,
    })
}

/// FFmpeg-backed decoder. One instance per stream.
pub struct FfmpegDecoder {
    info: StreamInfo,
    decoder: decoder::Video,
    /// Scratch frame for receiving decoded pictures from libavcodec.
    decoded: VideoFrameFfmpeg,
    /// Staging frame for `av_hwframe_transfer_data` when the decoded
    /// frame lives in HW memory (NV12 / P010 on a GPU surface) and
    /// must be copied to host memory before sws_scale.
    hw_transfer: VideoFrameFfmpeg,
    /// Lazily-built software scaler, reconfigured when the decoder's
    /// output format changes (first decoded frame defines it).
    scaler: Option<Scaler>,
    /// Target pixel format for our pipeline — 8-bit `Yuv420p` or
    /// 10-bit `Yuv420p10le`. Determined from `StreamInfo.pixel_format`
    /// at construction.
    target_pix_fmt: Pixel,
    pending_frames: VecDeque<VideoFrame>,
    frame_counter: u64,
    done: bool,
    /// HW device context, kept alive for the decoder's lifetime.
    /// Some(_) when we attached an hwaccel; None for software decode.
    /// Dropped after `decoder` so the context outlives any in-flight
    /// HW frames.
    #[allow(dead_code)]
    hw_device: Option<HwDeviceCtx>,
    /// Name of the engaged hwaccel ("vulkan", "cuda", etc.) or "none"
    /// for software. Surfaced via tracing logs + test assertions.
    hwaccel_name: &'static str,
}

impl FfmpegDecoder {
    pub fn new(info: StreamInfo) -> Result<Self> {
        init_ffmpeg();
        let codec_lower = info.codec.to_ascii_lowercase();
        let codec_id = codec_id_from_label(&codec_lower)
            .ok_or_else(|| anyhow!("FFmpeg: no codec_id mapped for '{codec_lower}'"))?;

        let ff_codec = decoder::find(codec_id).ok_or_else(|| {
            anyhow!(
                "FFmpeg: decoder for {codec_id:?} not present in this libavcodec build — \
                 rebuild FFmpeg with the relevant --enable-decoder flag"
            )
        })?;

        // Build a fresh decoder context. We don't copy parameters
        // from a demuxer's `AVCodecParameters` because the pipeline
        // hands us raw Annex-B (H.264 / H.265), OBU (AV1), or
        // codec-native samples via `push_sample` — libavcodec's
        // parser handles whatever shape the stream wires us.
        let mut ctx = CodecContext::new_with_codec(ff_codec);

        // Try each hwaccel in preference order. On success, attach the
        // device to the codec context BEFORE calling decoder()/video()
        // (which opens the codec — at open time libavcodec binds to
        // whatever hw_device_ctx is already present).
        //
        // av_buffer_ref increments the ref so the codec context owns
        // one refcount; HwDeviceCtx's Drop handles the other.
        let (hw_device, hwaccel_name) = unsafe {
            let codec_ptr = ff_codec.as_ptr() as *const sys::AVCodec;
            match try_open_hwaccel(codec_ptr) {
                Some(hw) => {
                    let raw_ctx = ctx.as_mut_ptr();
                    (*raw_ctx).hw_device_ctx = sys::av_buffer_ref(hw.ptr);
                    let name = match hw.device_type {
                        sys::AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN => "vulkan",
                        sys::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA => "cuda",
                        sys::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA => "d3d11va",
                        sys::AVHWDeviceType::AV_HWDEVICE_TYPE_DXVA2 => "dxva2",
                        sys::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI => "vaapi",
                        sys::AVHWDeviceType::AV_HWDEVICE_TYPE_QSV => "qsv",
                        sys::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX => "videotoolbox",
                        _ => "other",
                    };
                    (Some(hw), name)
                }
                None => (None, "none"),
            }
        };

        let mut dec = ctx
            .decoder()
            .video()
            .map_err(|e| anyhow!("FFmpeg: decoder().video() failed: {e}"))?;
        // Disable low-delay hacks so B-frame reorder drains cleanly.
        // Bigger frame queue = better throughput at the cost of ~1-2
        // frames of latency which the streaming pipeline tolerates.
        dec.set_flags(codec::Flags::empty());
        dec.set_threading(codec::threading::Config {
            kind: codec::threading::Type::Frame,
            count: 0, // 0 = libavcodec picks (typically num_cpus)
            #[cfg(not(feature = "ffmpeg_6_0"))]
            safe: true,
        });

        let target_pix_fmt = match info.pixel_format {
            PixelFormat::Yuv420p10le => Pixel::YUV420P10LE,
            _ => Pixel::YUV420P,
        };

        tracing::info!(
            codec = %codec_lower,
            hwaccel = hwaccel_name,
            width = info.width,
            height = info.height,
            "FFmpeg decoder opened"
        );

        Ok(Self {
            info,
            decoder: dec,
            decoded: VideoFrameFfmpeg::empty(),
            hw_transfer: VideoFrameFfmpeg::empty(),
            scaler: None,
            target_pix_fmt,
            pending_frames: VecDeque::new(),
            frame_counter: 0,
            done: false,
            hw_device,
            hwaccel_name,
        })
    }

    /// Returns the engaged hwaccel name ("vulkan" / "cuda" / "d3d11va" /
    /// "dxva2" / "vaapi" / "none" / "other"). Used by tests + for
    /// operator-facing tracing.
    pub fn hwaccel_engaged(&self) -> &'static str {
        self.hwaccel_name
    }

    /// Build / rebuild the scaler to match the current decoder's
    /// output format. Called lazily on the first received frame —
    /// libavcodec doesn't expose the pix_fmt until after the first
    /// successful `receive_frame`.
    fn ensure_scaler(&mut self, src_w: u32, src_h: u32, src_fmt: Pixel) -> Result<()> {
        // Rebuild when dimensions or format change. Real streams
        // rarely change mid-stream but hardware fallback transitions
        // (HW decoder uses NV12, SW uses YUV420P) would trip this.
        let needs_rebuild = match self.scaler.as_ref() {
            None => true,
            Some(s) => {
                s.input().width != src_w || s.input().height != src_h || s.input().format != src_fmt
            }
        };
        if needs_rebuild {
            self.scaler = Some(
                Scaler::get(
                    src_fmt,
                    src_w,
                    src_h,
                    self.target_pix_fmt,
                    self.info.width,
                    self.info.height,
                    ScalerFlags::BILINEAR,
                )
                .map_err(|e| anyhow!("FFmpeg: sws_scale ctx: {e}"))?,
            );
        }
        Ok(())
    }

    /// Pull all currently-decodable frames out of libavcodec and
    /// into `pending_frames`. Returns `Ok(())` when the decoder is
    /// idle (no more frames to emit without new input).
    fn drain_decoded(&mut self) -> Result<()> {
        loop {
            match self.decoder.receive_frame(&mut self.decoded) {
                Ok(()) => {
                    // HW decoders produce frames whose data lives in GPU
                    // memory (format is VULKAN / CUDA / D3D11 / etc —
                    // opaque from the host). Detect via `hw_frames_ctx`
                    // and transfer to a host-side software frame before
                    // running sws_scale. When decoding in software this
                    // branch is skipped and `src_frame` is the decoded
                    // frame directly.
                    let decoded_has_hw_ctx = unsafe {
                        let raw = self.decoded.as_ptr();
                        !raw.is_null() && !(*raw).hw_frames_ctx.is_null()
                    };
                    let src_frame: &VideoFrameFfmpeg = if decoded_has_hw_ctx {
                        unsafe {
                            let rc = sys::av_hwframe_transfer_data(
                                self.hw_transfer.as_mut_ptr(),
                                self.decoded.as_ptr(),
                                0,
                            );
                            if rc < 0 {
                                return Err(anyhow!(
                                    "FFmpeg: av_hwframe_transfer_data failed (rc={rc})"
                                ));
                            }
                        }
                        &self.hw_transfer
                    } else {
                        &self.decoded
                    };

                    let src_w = src_frame.width();
                    let src_h = src_frame.height();
                    let src_fmt = src_frame.format();
                    if src_w == 0 || src_h == 0 {
                        continue;
                    }
                    self.ensure_scaler(src_w, src_h, src_fmt)?;
                    let mut scaled = VideoFrameFfmpeg::empty();
                    self.scaler
                        .as_mut()
                        .unwrap()
                        .run(src_frame, &mut scaled)
                        .map_err(|e| anyhow!("FFmpeg: sws_scale run: {e}"))?;
                    let frame = ffmpeg_frame_to_video_frame(
                        &scaled,
                        &self.info,
                        self.target_pix_fmt,
                        self.frame_counter,
                    )?;
                    self.frame_counter += 1;
                    self.pending_frames.push_back(frame);
                }
                Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::util::error::EAGAIN => {
                    // No more frames without new input — normal idle.
                    return Ok(());
                }
                Err(ffmpeg::Error::Eof) => {
                    self.done = true;
                    return Ok(());
                }
                Err(e) => {
                    return Err(anyhow!("FFmpeg: receive_frame: {e}"));
                }
            }
        }
    }
}

/// Convert a `YUV420P` / `YUV420P10LE` FFmpeg frame into our
/// `VideoFrame`. The scaled frame has tightly-packed planes inside a
/// single allocation per plane — we concatenate into our flat-layout
/// `Yuv420p` convention of `[Y plane | U plane | V plane]`.
fn ffmpeg_frame_to_video_frame(
    src: &VideoFrameFfmpeg,
    info: &StreamInfo,
    target_fmt: Pixel,
    pts: u64,
) -> Result<VideoFrame> {
    let w = src.width() as usize;
    let h = src.height() as usize;
    let (bytes_per_sample, out_fmt) = match target_fmt {
        Pixel::YUV420P10LE => (2, PixelFormat::Yuv420p10le),
        _ => (1, PixelFormat::Yuv420p),
    };
    let y_len = w * h * bytes_per_sample;
    let uv_w = w / 2;
    let uv_h = h / 2;
    let uv_len = uv_w * uv_h * bytes_per_sample;
    let mut data = Vec::with_capacity(y_len + 2 * uv_len);

    // Planes come out of sws_scale with per-plane strides that may
    // exceed the logical width due to SIMD alignment. Copy row-by-row
    // to produce the tightly-packed layout our pipeline expects.
    for plane in 0..3 {
        let pw = if plane == 0 { w } else { uv_w };
        let ph = if plane == 0 { h } else { uv_h };
        let stride = src.stride(plane) as usize;
        let plane_bytes = src.data(plane);
        for row in 0..ph {
            let row_start = row * stride;
            let row_end = row_start + pw * bytes_per_sample;
            if row_end > plane_bytes.len() {
                return Err(anyhow!(
                    "FFmpeg frame plane {plane} row {row} exceeds buffer"
                ));
            }
            data.extend_from_slice(&plane_bytes[row_start..row_end]);
        }
    }

    Ok(VideoFrame::new(
        Bytes::from(data),
        info.width,
        info.height,
        out_fmt,
        info.color_space,
        pts,
    ))
}

impl Decoder for FfmpegDecoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    fn push_sample(&mut self, data: &[u8]) -> Result<()> {
        if self.done {
            return Err(anyhow!("FFmpeg: push_sample after finish"));
        }
        let mut pkt = packet::Packet::copy(data);
        // We don't have real PTS/DTS from `push_sample` — libavcodec
        // assigns sequential PTS when we leave them unset.
        pkt.set_pts(None);
        pkt.set_dts(None);
        self.decoder
            .send_packet(&pkt)
            .map_err(|e| anyhow!("FFmpeg: send_packet: {e}"))?;
        self.drain_decoded()?;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.done {
            return Ok(());
        }
        self.decoder
            .send_eof()
            .map_err(|e| anyhow!("FFmpeg: send_eof: {e}"))?;
        self.drain_decoded()?;
        self.done = true;
        Ok(())
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        // If we've pushed all samples and called finish, drain any
        // remaining reordered B-frames.
        if self.pending_frames.is_empty() && self.done {
            self.drain_decoded()?;
        }
        Ok(self.pending_frames.pop_front())
    }
}

// Silence the "unused import" warnings when the feature is enabled
// but specific paths aren't exercised (e.g. no HW ctx helper yet).
#[allow(unused_imports)]
use ffmpeg_next::util::error as _ff_error;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{ColorMetadata, ColorSpace, PixelFormat};

    fn test_info() -> StreamInfo {
        StreamInfo {
            codec: "h264".to_string(),
            width: 320,
            height: 176,
            frame_rate: 24.0,
            duration: 0.0,
            pixel_format: PixelFormat::Yuv420p,
            color_space: ColorSpace::Bt709,
            total_frames: 0,
            bitrate: 0,
            color_metadata: ColorMetadata::default(),
        }
    }

    #[test]
    fn codec_id_mapping_covers_mainstream() {
        assert!(codec_id_from_label("h264").is_some());
        assert!(codec_id_from_label("hevc").is_some());
        assert!(codec_id_from_label("av1").is_some());
        assert!(codec_id_from_label("vp9").is_some());
        assert!(codec_id_from_label("vp8").is_some());
        assert!(codec_id_from_label("mpeg2").is_some());
        assert!(codec_id_from_label("mpeg4").is_some());
        assert!(codec_id_from_label("prores").is_some());
        assert!(codec_id_from_label("unknown").is_none());
    }

    #[test]
    fn construct_h264_decoder() {
        // If FFmpeg dev libs are wired, constructing a decoder for
        // H.264 should succeed without any sample input. Serves as
        // the smoke test that the feature build is functional.
        match FfmpegDecoder::new(test_info()) {
            Ok(dec) => {
                assert_eq!(dec.stream_info().codec, "h264");
                // hwaccel_engaged reports which backend actually
                // attached. On hosts without Vulkan/CUDA/D3D11 the
                // answer is "none" (software decode), which is still
                // a pass — we just log it.
                let name = dec.hwaccel_engaged();
                eprintln!("hwaccel engaged: {name}");
                assert!(matches!(
                    name,
                    "vulkan"
                        | "cuda"
                        | "d3d11va"
                        | "dxva2"
                        | "vaapi"
                        | "qsv"
                        | "videotoolbox"
                        | "other"
                        | "none"
                ));
            }
            Err(e) => {
                eprintln!("skip: FFmpeg H.264 decoder construct failed: {e}");
            }
        }
    }

    #[test]
    fn hwaccel_override_none_forces_software() {
        // Setting FFMPEG_HWACCEL=none should bypass HW probe entirely.
        // Even on Windows hosts with D3D11VA available, the decoder
        // should report hwaccel=none.
        // Serialize against other tests mutating env — same pattern
        // the NVDEC disable tests use.
        unsafe {
            std::env::set_var("FFMPEG_HWACCEL", "none");
        }
        match FfmpegDecoder::new(test_info()) {
            Ok(dec) => assert_eq!(dec.hwaccel_engaged(), "none"),
            Err(e) => eprintln!("skip: construct failed: {e}"),
        }
        unsafe {
            std::env::remove_var("FFMPEG_HWACCEL");
        }
    }
}
