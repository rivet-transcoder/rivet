//! AMD **AMF hardware decode** — hand-rolled FFI on the SDK-mirrored vtables
//! in `crate::amf_ffi`, the runtime / context lifecycle in
//! `crate::amf_runtime` (both shared with the AMF encoder). Decodes H.264 /
//! HEVC / VP9 / AV1 on AMD GPUs whose VCN the AMF runtime drives.
//!
//! Flow: [`AmfRuntime::open`] (dlopen → `AMFInit` → `CreateContext` →
//! `InitDX11` on the chosen AMD adapter, or `AMFContext1::InitVulkan`) →
//! `CreateComponent(<decoder id>)` → `Init(NV12 | P010, w, h)` → per
//! sample: `AllocBuffer(HOST)` → copy the Annex-B access unit → `SetPts` →
//! `SubmitInput` (with the `AMF_INPUT_FULL` drain-and-retry) → loop
//! `QueryOutput` → `QueryInterface(IID_AMFSurface)` → `Convert(HOST)` →
//! read the NV12 / P010 planes → `Yuv420p` / `Yuv420p10le`. `finish` is
//! `Drain` then `QueryOutput` polled to `AMF_EOF`.
//!
//! Samples arrive as Annex-B access units with the parameter sets in band
//! (the demuxers convert AVCC / HVCC), so `AMF_VIDEO_DECODER_EXTRADATA`
//! ("Optional if stream is Annex B", `VideoDecoderUVD.h:72`) is not set.
//! Timestamps are `AMF_TS_PRESENTATION` (`:66`): the decoder reorders and the
//! output arrives in display order, which this decoder numbers 0, 1, 2, …
//! like the software decoders do.
//!
//! **Verified on hardware** (2026-08-27, Ryzen 9 9950X iGPU, driver
//! 32.0.21045.5002): H.264 and HEVC 8-bit and HEVC Main 10 (P010) decode
//! byte-for-byte equal to ffmpeg and to the in-tree `h26x` software
//! decoders — `tests/amf_decode_pixels.rs`. What this GPU cannot decode is
//! learned from the runtime, not assumed: [`probe_decode_caps`] tries
//! `CreateComponent` for each decoder id once, and `rivet capabilities` /
//! the dispatch report exactly that.
#![cfg(feature = "amd")]

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr;
use std::sync::OnceLock;

use anyhow::{Result, bail};
use bytes::Bytes;

use super::{Decoder, StreamInfo, nv12_planes_to_yuv420p, p010_planes_to_yuv420p10le};
use crate::amf_ffi::*;
use crate::amf_runtime::*;
use crate::frame::{ColorSpace, PixelFormat, VideoFrame};

// ─── Component ids and properties (components/VideoDecoderUVD.h) ──

/// `AMFVideoDecoderUVD_H264_AVC` (`:47`).
const DECODER_H264: &str = "AMFVideoDecoderUVD_H264_AVC";
/// `AMFVideoDecoderHW_H265_HEVC` (`:51`; Main 10 goes through the same
/// component, `:52` deprecates the separate `_MAIN10` id).
const DECODER_HEVC: &str = "AMFVideoDecoderHW_H265_HEVC";
/// `AMFVideoDecoderHW_VP9` (`:53`).
const DECODER_VP9: &str = "AMFVideoDecoderHW_VP9";
/// `AMFVideoDecoderHW_AV1` (`:55`).
const DECODER_AV1: &str = "AMFVideoDecoderHW_AV1";
/// `AMF_TIMESTAMP_MODE` (`:74`), `AMF_TS_PRESENTATION = 0` (`:66`).
const TIMESTAMP_MODE: &str = "TimestampMode";
const TS_PRESENTATION: i64 = 0;
/// `AMF_VIDEO_DECODER_SURFACE_CPU` (`:129`): "hint to decoder that output
/// will be consumed on cpu" — which it is, through `Convert(HOST)`.
const SURFACE_CPU: &str = "SurfaceCpu";

/// The four codecs and their component ids, in `rivet capabilities` order.
const CODECS: &[(&str, &str)] = &[
    ("h264", DECODER_H264),
    ("hevc", DECODER_HEVC),
    ("vp9", DECODER_VP9),
    ("av1", DECODER_AV1),
];

/// AMF decoder component id for a codec string, or `None` if AMF has no
/// component for it.
fn amf_decoder_id(codec_lower: &str) -> Option<&'static str> {
    Some(match codec_lower {
        "h264" | "avc1" | "avc" => DECODER_H264,
        "h265" | "hevc" | "hvc1" | "hev1" | "hvc2" | "hev2" => DECODER_HEVC,
        "av1" | "av01" => DECODER_AV1,
        "vp9" | "vp09" => DECODER_VP9,
        _ => return None,
    })
}

/// Canonical label for a codec string, as `CODECS` names it.
fn canonical(codec_lower: &str) -> Option<&'static str> {
    let id = amf_decoder_id(codec_lower)?;
    CODECS.iter().find(|(_, i)| *i == id).map(|(c, _)| *c)
}

/// Whether AMF has a decoder component for this codec at all (the build's
/// view). Whether *this host's* GPU has it is [`probe_decode_caps`].
pub fn supports(codec_lower: &str) -> bool {
    amf_decoder_id(codec_lower).is_some()
}

/// The codecs the first AMD GPU on this host can decode through AMF,
/// learned by asking the runtime: one context, one `CreateComponent` per
/// decoder id (a GPU without the block answers `AMF_CODEC_NOT_SUPPORTED`
/// or `AMF_DECODER_NOT_PRESENT`). Probed once per process. Empty on a
/// host with no AMD GPU, no AMF runtime, or a GPU the runtime does not
/// drive.
pub fn probe_decode_caps() -> &'static [&'static str] {
    static CAPS: OnceLock<Vec<&'static str>> = OnceLock::new();
    CAPS.get_or_init(|| {
        let gpus = crate::gpu::detect_gpus();
        let Some(dev) = gpus.iter().find(|g| g.vendor == crate::gpu::GpuVendor::Amd) else {
            return Vec::new();
        };
        let runtime = match AmfRuntime::open(dev.vendor_index) {
            Ok(r) => r,
            Err(e) => {
                tracing::info!(gpu = %dev.name, error = %e, "AMF decode probe: runtime/context unavailable");
                return Vec::new();
            }
        };
        let mut caps = Vec::new();
        for (codec, id) in CODECS {
            match unsafe { runtime.create_component(id) } {
                Ok(component) => {
                    unsafe { release_component(component) };
                    caps.push(*codec);
                }
                Err(e) => {
                    tracing::info!(gpu = %dev.name, codec, error = %e, "AMF decode probe: no component");
                }
            }
        }
        tracing::info!(gpu = %dev.name, ?caps, "AMF decode probe");
        caps
    })
}

/// Whether this host's AMD GPU decodes `codec_lower` through AMF (probed).
pub fn host_supports(codec_lower: &str) -> bool {
    canonical(codec_lower).is_some_and(|c| probe_decode_caps().contains(&c))
}

// ─── Decoder ──────────────────────────────────────────────────────

/// How long `finish` waits for the frames still in flight after `Drain`.
const FLUSH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
/// Bounded `AMF_INPUT_FULL` retry, as in the encoder.
const INPUT_FULL_MAX_RETRIES: u32 = 64;

fn trace_env() -> bool {
    std::env::var("AMF_DEC_TRACE").is_ok()
}

/// What ended a drain pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DrainEnd {
    Repeat,
    Eof,
    NeedMoreInput,
}

/// EXPERIMENT: Annex-B -> 4-byte length-prefixed NALs.
fn annexb_to_avcc(sample: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(sample.len() + 16);
    for nal in h26x::nal::annexb_nals(sample) {
        out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
        out.extend_from_slice(nal);
    }
    out
}

/// EXPERIMENT: an avcC record from the in-band SPS/PPS of an H.264 sample.
fn avcc_from_sample(sample: &[u8]) -> Option<Vec<u8>> {
    let mut sps = None;
    let mut pps = None;
    for nal in h26x::nal::annexb_nals(sample) {
        match nal.first().map(|b| b & 0x1f) {
            Some(7) => sps = Some(nal.to_vec()),
            Some(8) => pps = Some(nal.to_vec()),
            _ => {}
        }
    }
    let (sps, pps) = (sps?, pps?);
    let mut v = vec![1, sps[1], sps[2], sps[3], 0xff, 0xe1];
    v.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    v.extend_from_slice(&sps);
    v.push(1);
    v.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    v.extend_from_slice(&pps);
    Some(v)
}

pub struct AmfDecoder {
    info: StreamInfo,
    /// EXPERIMENT
    avcc_mode: bool,
    decoder: *mut c_void,
    frames: VecDeque<VideoFrame>,
    /// Display-order frame counter — the output pts, as the software
    /// decoders number theirs.
    next_pts: u64,
    /// Samples submitted so far; their input pts, in 100-ns ticks.
    submitted: u64,
    pts_timescale: u64,
    /// Declared last: the context, device and library outlive the component.
    runtime: AmfRuntime,
}

// Single-threaded driver; raw AMF pointers are owned + released in Drop.
unsafe impl Send for AmfDecoder {}

impl AmfDecoder {
    /// Build a decoder for `info` on the `vendor_index`-th AMD adapter
    /// (`GpuDevice::vendor_index`).
    pub fn new(info: StreamInfo, vendor_index: u32) -> Result<Self> {
        let codec = info.codec.to_ascii_lowercase();
        let decoder_id =
            amf_decoder_id(&codec).ok_or_else(|| anyhow::anyhow!("AMF cannot decode codec {codec}"))?;
        let ten_bit = matches!(info.pixel_format, PixelFormat::Yuv420p10le);

        let runtime = AmfRuntime::open(vendor_index)?;
        unsafe {
            let decoder = runtime
                .create_component(decoder_id)
                .map_err(|e| e.context("this GPU has no such decode block the AMF runtime drives"))?;

            // Presentation-order timestamps (the default, said explicitly)
            // and the CPU-consumption hint. The hint is advisory: a runtime
            // that predates it answers AMF_NOT_FOUND, which is not a reason
            // to refuse the stream.
            let ts_mode = std::env::var("AMF_DEC_TS").ok().and_then(|v| v.parse().ok()).unwrap_or(TS_PRESENTATION);
            if let Err(e) = set_int_property(decoder, TIMESTAMP_MODE, ts_mode) {
                release_component(decoder);
                return Err(e);
            }
            if let Ok(m) = std::env::var("AMF_DEC_REORDER") {
                let rc = set_int_property(decoder, "ReorderMode", m.parse().unwrap());
                eprintln!("EXPERIMENT ReorderMode={m} -> {rc:?}");
            }
            if let Err(e) = set_bool_property(decoder, SURFACE_CPU, true) {
                tracing::debug!(error = %e, "AMF decoder: SurfaceCpu hint not taken");
            }

            if let Ok(n) = std::env::var("AMF_DEC_POOL") {
                eprintln!("EXPERIMENT SurfacePoolSize={n} -> {:?}", set_int_property(decoder, "SurfacePoolSize", n.parse().unwrap()));
            }
            if let Ok(n) = std::env::var("AMF_DEC_DPB") {
                eprintln!("EXPERIMENT DPBSize={n} -> {:?}", set_int_property(decoder, "DPBSize", n.parse().unwrap()));
            }
            let surface_fmt = if std::env::var("AMF_DEC_INIT_UNKNOWN").is_ok() { 0 } else if ten_bit { AMF_SURFACE_P010 } else { AMF_SURFACE_NV12 };
            let w = info.width.max(16) as i32;
            let h = info.height.max(16) as i32;
            let decoder_vt = &*(*(decoder as *mut AmfComponentObj)).vtbl;
            let rc = (decoder_vt.init)(decoder, surface_fmt, w, h);
            if rc != AMF_OK {
                release_component(decoder);
                bail!(
                    "AMFComponent::Init({decoder_id}, fmt={surface_fmt}, {w}x{h}) failed: {rc} ({})",
                    result_name(rc)
                );
            }

            let fps = if info.frame_rate > 0.0 { info.frame_rate } else { 30.0 };
            tracing::info!(
                codec = %codec,
                component = decoder_id,
                width = info.width,
                height = info.height,
                ten_bit,
                vendor_index,
                "AMF decoder ready"
            );
            Ok(Self {
                info,
                avcc_mode: std::env::var("AMF_DEC_AVCC").is_ok(),
                decoder,
                frames: VecDeque::new(),
                next_pts: 0,
                submitted: 0,
                pts_timescale: (10_000_000.0f64 / fps).round().max(1.0) as u64,
                runtime,
            })
        }
    }

    /// Drain whatever `QueryOutput` has ready into the frame queue.
    unsafe fn drain_outputs(&mut self) -> Result<DrainEnd> {
        unsafe {
            let decoder_vt = &*(*(self.decoder as *mut AmfComponentObj)).vtbl;
            loop {
                let mut data: *mut c_void = ptr::null_mut();
                let rc = (decoder_vt.query_output)(self.decoder, &mut data);
                match rc {
                    AMF_OK => {
                        // AMF_OK with no data is "nothing this instant" —
                        // looping on it spins a core (seen: the H.264
                        // decoder answers exactly that while its first
                        // frames are in flight).
                        if data.is_null() {
                            return Ok(DrainEnd::Repeat);
                        }
                        let frame = self.surface_to_frame(data);
                        // Drop the AMFData ref QueryOutput handed us.
                        release(data);
                        self.frames.push_back(frame?);
                    }
                    AMF_REPEAT => return Ok(DrainEnd::Repeat),
                    AMF_EOF => return Ok(DrainEnd::Eof),
                    AMF_NEED_MORE_INPUT => return Ok(DrainEnd::NeedMoreInput),
                    other => bail!("AMF QueryOutput (decode) failed: {other} ({})", result_name(other)),
                }
            }
        }
    }

    /// `AMFData` → `AMFSurface` → host memory → a planar `VideoFrame`.
    unsafe fn surface_to_frame(&mut self, data: *mut c_void) -> Result<VideoFrame> {
        unsafe {
            let data_vt = &*(*(data as *mut AmfObj)).vtbl;
            let mut surf: *mut c_void = ptr::null_mut();
            let rc = (data_vt.query_interface)(data, &AMF_IID_SURFACE, &mut surf);
            if rc != AMF_OK || surf.is_null() {
                bail!(
                    "AMFData::QueryInterface(AMFSurface) failed: {rc} ({})",
                    result_name(rc)
                );
            }
            let result = self.read_surface(surf);
            release(surf);
            result
        }
    }

    unsafe fn read_surface(&mut self, surf: *mut c_void) -> Result<VideoFrame> {
        unsafe {
            let surf_vt = &*(*(surf as *mut AmfSurfaceObj)).vtbl;
            // The decoder's output lives in GPU memory (DX11 / Vulkan);
            // Convert copies it into host memory ("optimal interop if
            // possible. Copy through host memory if needed", core/Data.h:152).
            let rc = (surf_vt.data.convert)(surf, AMF_MEMORY_HOST);
            if rc != AMF_OK {
                bail!("AMFSurface::Convert(HOST) failed: {rc} ({})", result_name(rc));
            }
            let format = (surf_vt.get_format)(surf);
            let (ten_bit, bytes_per_sample) = match format {
                AMF_SURFACE_NV12 => (false, 1usize),
                AMF_SURFACE_P010 => (true, 2usize),
                other => bail!("AMF decoder produced surface format {other}, expected NV12 (1) or P010 (10)"),
            };

            let plane = |which: i32| -> Result<(*const u8, usize, usize, usize)> {
                let p = (surf_vt.get_plane)(surf, which);
                if p.is_null() {
                    bail!("AMF output surface has no plane {which}");
                }
                let pvt = &*(*(p as *mut AmfPlaneObj)).vtbl;
                let native = (pvt.get_native)(p) as *const u8;
                if native.is_null() {
                    bail!("AMF output plane {which} is not host-mapped after Convert(HOST)");
                }
                Ok((
                    native,
                    (pvt.get_h_pitch)(p).max(0) as usize,
                    (pvt.get_width)(p).max(0) as usize,
                    (pvt.get_height)(p).max(0) as usize,
                ))
            };
            let (y_ptr, y_pitch, y_w, y_h) = plane(AMF_PLANE_Y)?;
            let (uv_ptr, uv_pitch, _uv_w, uv_h) = plane(AMF_PLANE_UV)?;

            // The surface may be allocated larger than the picture (macroblock
            // / CTB alignment); the stream's own size is the visible one.
            let w = (self.info.width as usize).min(y_w).max(1);
            let h = (self.info.height as usize).min(y_h).max(1);
            let ch = h.div_ceil(2).min(uv_h.max(1));
            if y_pitch < w * bytes_per_sample || uv_pitch < w.div_ceil(2) * 2 * bytes_per_sample {
                bail!("AMF output plane pitch smaller than the picture ({y_pitch} / {uv_pitch} for {w}x{h})");
            }
            let y = std::slice::from_raw_parts(y_ptr, y_pitch * h);
            let uv = std::slice::from_raw_parts(uv_ptr, uv_pitch * ch);
            let (pixel_format, packed) = if ten_bit {
                (PixelFormat::Yuv420p10le, p010_planes_to_yuv420p10le(y, y_pitch, uv, uv_pitch, w, h))
            } else {
                (PixelFormat::Yuv420p, nv12_planes_to_yuv420p(y, y_pitch, uv, uv_pitch, w, h))
            };
            let pts = self.next_pts;
            self.next_pts += 1;
            Ok(VideoFrame::new(
                Bytes::from(packed),
                w as u32,
                h as u32,
                pixel_format,
                ColorSpace::Bt709,
                pts,
            ))
        }
    }
}

impl Decoder for AmfDecoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    fn push_sample(&mut self, sample: &[u8]) -> Result<()> {
        if sample.is_empty() {
            return Ok(());
        }
        let converted;
        let sample: &[u8] = if self.avcc_mode {
            if self.submitted == 0 {
                if let Some(avcc) = avcc_from_sample(sample) {
                    unsafe {
                        let eb = self.runtime.alloc_host_buffer(avcc.len())?;
                        let eb_vt = &*(*(eb as *mut AmfBufferObj)).vtbl;
                        ptr::copy_nonoverlapping(avcc.as_ptr(), (eb_vt.get_native)(eb) as *mut u8, avcc.len());
                        let rc = set_property(self.decoder, "ExtraData", AmfVariant::interface(eb));
                        release(eb);
                        eprintln!("EXPERIMENT ExtraData(avcC {} B) -> {rc:?}", avcc.len());
                    }
                }
            }
            converted = annexb_to_avcc(sample);
            &converted
        } else {
            sample
        };
        unsafe {
            let pad = if std::env::var("AMF_DEC_PAD").is_ok() { 64 } else { 0 };
            let buf = self.runtime.alloc_host_buffer(sample.len() + pad)?;
            let buf_vt = &*(*(buf as *mut AmfBufferObj)).vtbl;
            let dst = (buf_vt.get_native)(buf) as *mut u8;
            if dst.is_null() {
                release(buf);
                bail!("AMFBuffer::GetNative returned null for a host buffer");
            }
            ptr::copy_nonoverlapping(sample.as_ptr(), dst, sample.len());
            if pad > 0 {
                ptr::write_bytes(dst.add(sample.len()), 0, pad);
                // SetSize is AMFBuffer slot 23 — not typed in amf_ffi; call through a local typed view.
                let set_size: unsafe extern "system" fn(*mut c_void, usize) -> AmfResult = std::mem::transmute(buf_vt.set_size);
                let rc = set_size(buf, sample.len());
                if trace_env() { eprintln!("TRACE SetSize -> {rc}"); }
            }
            if std::env::var("AMF_DEC_NOPTS").is_err() {
                (buf_vt.data.set_pts)(buf, (self.submitted * self.pts_timescale) as i64);
                (buf_vt.data.set_duration)(buf, self.pts_timescale as i64);
            }

            let decoder_vt = &*(*(self.decoder as *mut AmfComponentObj)).vtbl;
            let mut attempt = 0u32;
            let trace = std::env::var("AMF_DEC_TRACE").is_ok();
            loop {
                let rc = (decoder_vt.submit_input)(self.decoder, buf);
                if trace {
                    eprintln!("TRACE sample {} ({} B) SubmitInput -> {rc} ({}) attempt {attempt}, frames so far {}", self.submitted, sample.len(), result_name(rc), self.next_pts);
                }
                match rc {
                    AMF_OK | AMF_NEED_MORE_INPUT => break,
                    AMF_INPUT_FULL | AMF_REPEAT => {
                        // Transient: free a slot by draining, keep our ref
                        // on the buffer, retry the same pointer.
                        if attempt >= INPUT_FULL_MAX_RETRIES {
                            release(buf);
                            bail!("AMF SubmitInput (decode) stuck at AMF_INPUT_FULL after {attempt} attempts");
                        }
                        attempt += 1;
                        if let Err(e) = self.drain_outputs() {
                            release(buf);
                            return Err(e);
                        }
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    rc => {
                        release(buf);
                        bail!("AMFComponent::SubmitInput (decode) failed: {rc} ({})", result_name(rc));
                    }
                }
            }
            // The component took its own ref; ours is done.
            release(buf);
            self.submitted += 1;
            if std::env::var("AMF_DEC_ONE_QUERY").is_ok() {
                // EXPERIMENT: exactly one QueryOutput per submitted sample, as ffmpeg does.
                let mut data: *mut c_void = ptr::null_mut();
                let rc = (decoder_vt.query_output)(self.decoder, &mut data);
                if rc == AMF_OK && !data.is_null() {
                    let f = self.surface_to_frame(data);
                    release(data);
                    self.frames.push_back(f?);
                }
            } else {
                self.drain_outputs()?;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        unsafe {
            let decoder_vt = &*(*(self.decoder as *mut AmfComponentObj)).vtbl;
            // EXPERIMENT: let the input queue settle before Drain.
            if std::env::var("AMF_DEC_SETTLE").is_ok() {
                let mut quiet = std::time::Instant::now();
                loop {
                    let before = self.frames.len();
                    self.drain_outputs()?;
                    if self.frames.len() != before {
                        quiet = std::time::Instant::now();
                    }
                    if quiet.elapsed() > std::time::Duration::from_millis(200) {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            }
            if std::env::var("AMF_DEC_TAIL_NAL").is_ok() {
                // EXPERIMENT: an H.264 end-of-stream NAL (type 11) / HEVC EOB (37)
                let hevc = self.info.codec.to_ascii_lowercase().contains("265") || self.info.codec.to_ascii_lowercase().contains("hevc") || self.info.codec.to_ascii_lowercase().starts_with("hvc") || self.info.codec.to_ascii_lowercase().starts_with("hev");
                let tail: &[u8] = if hevc { &[0, 0, 0, 1, 0x4a, 0x01] } else { &[0, 0, 0, 1, 0x0b] };
                let buf = self.runtime.alloc_host_buffer(tail.len())?;
                let buf_vt = &*(*(buf as *mut AmfBufferObj)).vtbl;
                ptr::copy_nonoverlapping(tail.as_ptr(), (buf_vt.get_native)(buf) as *mut u8, tail.len());
                let rc = (decoder_vt.submit_input)(self.decoder, buf);
                release(buf);
                eprintln!("EXPERIMENT tail NAL -> {rc} ({})", result_name(rc));
                self.drain_outputs()?;
            }
            if std::env::var("AMF_DEC_EOS_NULL").is_ok() {
                let rc = (decoder_vt.submit_input)(self.decoder, ptr::null_mut());
                eprintln!("EXPERIMENT SubmitInput(null) -> {rc} ({})", result_name(rc));
            }
            let rc = (decoder_vt.drain)(self.decoder);
            if rc != AMF_OK && rc != AMF_REPEAT {
                bail!("AMF Drain (decode) failed: {rc} ({})", result_name(rc));
            }
            // The frames already submitted are still in flight; QueryOutput
            // answers AMF_REPEAT until each lands and AMF_EOF once the last
            // one is out (the encoder lost its tail by stopping earlier).
            let deadline = std::time::Instant::now() + FLUSH_TIMEOUT;
            loop {
                match self.drain_outputs()? {
                    DrainEnd::Eof => {
                        if std::env::var("AMF_DEC_PAST_EOF").is_ok() {
                            // EXPERIMENT: keep asking after EOF.
                            let before = self.frames.len();
                            let mut codes = Vec::new();
                            for _ in 0..50 {
                                let mut data: *mut c_void = ptr::null_mut();
                                let rc = (decoder_vt.query_output)(self.decoder, &mut data);
                                codes.push(rc);
                                if rc == AMF_OK && !data.is_null() {
                                    let f = self.surface_to_frame(data);
                                    release(data);
                                    self.frames.push_back(f?);
                                }
                                std::thread::sleep(std::time::Duration::from_millis(4));
                            }
                            eprintln!("EXPERIMENT past EOF: +{} frames, codes {:?}", self.frames.len() - before, &codes[..10]);
                        }
                        return Ok(());
                    }
                    DrainEnd::Repeat | DrainEnd::NeedMoreInput => {
                        if std::time::Instant::now() >= deadline {
                            bail!("AMF decoder never reached AMF_EOF within {:?} of Drain", FLUSH_TIMEOUT);
                        }
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
            }
        }
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        Ok(self.frames.pop_front())
    }
}

impl Drop for AmfDecoder {
    fn drop(&mut self) {
        // The component first; `runtime` (context, device, library) drops
        // after it by field order.
        unsafe { release_component(self.decoder) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_ids_match_header() {
        assert_eq!(amf_decoder_id("h264"), Some("AMFVideoDecoderUVD_H264_AVC"));
        assert_eq!(amf_decoder_id("avc1"), Some("AMFVideoDecoderUVD_H264_AVC"));
        assert_eq!(amf_decoder_id("hevc"), Some("AMFVideoDecoderHW_H265_HEVC"));
        assert_eq!(amf_decoder_id("hvc1"), Some("AMFVideoDecoderHW_H265_HEVC"));
        assert_eq!(amf_decoder_id("vp9"), Some("AMFVideoDecoderHW_VP9"));
        assert_eq!(amf_decoder_id("av1"), Some("AMFVideoDecoderHW_AV1"));
        assert_eq!(amf_decoder_id("vp8"), None);
        assert_eq!(amf_decoder_id("prores"), None);
        assert_eq!(canonical("hev1"), Some("hevc"));
        assert_eq!(canonical("av01"), Some("av1"));
        assert!(supports("h265") && !supports("mpeg2"));
    }

    /// The probe on this machine: prints the verdict, and on a host with an
    /// AMD GPU the runtime drives, H.264 and HEVC must be in it (every AMF
    /// generation decodes those); `host_supports` agrees with it.
    #[test]
    fn probe_on_this_machine() {
        let caps = probe_decode_caps();
        eprintln!("AMF decode probe on this machine: {caps:?}");
        let amd = crate::gpu::detect_gpus().iter().any(|g| g.vendor == crate::gpu::GpuVendor::Amd);
        if amd && AmfRuntime::open(0).is_ok() {
            assert!(caps.contains(&"h264") && caps.contains(&"hevc"), "{caps:?}");
        }
        for c in caps {
            assert!(host_supports(c));
        }
        assert!(!host_supports("mpeg2"));
    }
}
