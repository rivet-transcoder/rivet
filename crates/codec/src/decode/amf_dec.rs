//! AMD **AMF hardware decode** — hand-rolled FFI (our own SDK-mirror code, no
//! external wrapper crate). Decodes H.264 / HEVC / AV1 / VP9 on AMD GPUs by
//! driving the AMF runtime directly, mirroring the AMF SDK decoder API and the
//! in-tree AMF *encoder* (`encode/amf.rs`) for the shared context / surface FFI.
//!
//! Flow: dlopen `amfrt64.dll` / `libamfrt64.so.1` → `AMFInit` (factory) →
//! `CreateContext` + `InitDX11`/`InitVulkan` → `CreateComponent(<decoder id>)` →
//! per sample: wrap the encoded bytes in an `AMFBuffer` → `SubmitInput` → loop
//! `QueryOutput` → downcast the `AMFData` to `AMFSurface` → read the NV12/P010
//! planes → `Yuv420p`/`Yuv420p10le`. Drain on `finish()`.
//!
//! **Verified-by-review only** — no AMD RDNA-class card on the dev box. The
//! spots that need confirming on real hardware are flagged `// VERIFY:` and
//! tracked in TODO.md (notably the `AMF_IID_SURFACE` GUID and the host-memory
//! surface read-back). If a stream fails to decode, the `ffmpeg` feature is the
//! fallback for AMD hosts.
#![cfg(feature = "amd")]

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use anyhow::{Result, bail};

use bytes::Bytes;

use super::{Decoder, StreamInfo, nv12_planes_to_yuv420p, p010_planes_to_yuv420p10le};
use crate::frame::{ColorSpace, PixelFormat, VideoFrame};

// ─── AMF result codes + constants (mirror vendor/amd/AMFPlatform.h) ───
type AmfResult = i32;
const AMF_OK: AmfResult = 0;
const AMF_EOF: AmfResult = 2024;
const AMF_REPEAT: AmfResult = 2023;
const AMF_NEED_MORE_INPUT: AmfResult = 2022;
const AMF_INPUT_FULL: AmfResult = 2020;

const fn amf_make_version(major: u64, minor: u64, release: u64, build: u64) -> u64 {
    (major << 48) | (minor << 32) | (release << 16) | build
}
const AMF_VERSION: u64 = amf_make_version(1, 4, 30, 0);

const AMF_MEMORY_HOST: i32 = 1;
const AMF_SURFACE_NV12: i32 = 1;
const AMF_SURFACE_P010: i32 = 10;

// VERIFY: AMFSurface interface GUID (vendor/amd/core/Surface.h). Best guess —
// QueryOutput returns an AMFData; we QueryInterface() it to AMFSurface. A wrong
// IID fails every output. Confirm bytes against the installed AMF SDK header.
const AMF_IID_SURFACE: [u8; 16] = [
    0x6b, 0x0f, 0xb9, 0x3b, 0x3b, 0x60, 0x15, 0x4e, 0x80, 0x49, 0x96, 0xc9, 0x57, 0x9c, 0x7e, 0xc7,
];

// ─── Shared AMF FFI (same ABI as encode/amf.rs) ──────────────────────
type QueryInterfaceFn = unsafe extern "C" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i64;
type AcquireFn = unsafe extern "C" fn(*mut c_void) -> i64;
type ReleaseFn = unsafe extern "C" fn(*mut c_void) -> i64;
type FnAmfInit = unsafe extern "C" fn(u64, *mut *mut c_void) -> AmfResult;

#[repr(C)]
struct AmfVariant {
    ty: i32,
    _pad: i32,
    value: [u8; 24],
}

#[repr(C)]
struct AmfFactoryVtbl {
    create_context: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> AmfResult,
    create_component:
        unsafe extern "C" fn(*mut c_void, *mut c_void, *const u16, *mut *mut c_void) -> AmfResult,
    set_cache_folder: unsafe extern "C" fn(*mut c_void, *const u16) -> AmfResult,
    get_cache_folder: unsafe extern "C" fn(*mut c_void) -> *const u16,
    get_debug: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> AmfResult,
    get_trace: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> AmfResult,
    get_programs: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> AmfResult,
}
#[repr(C)]
struct AmfFactoryObj {
    vtbl: *const AmfFactoryVtbl,
}

#[repr(C)]
struct AmfContextVtbl {
    query_interface: QueryInterfaceFn,
    acquire: AcquireFn,
    release: ReleaseFn,
    terminate: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    init_dx11: unsafe extern "C" fn(*mut c_void, *mut c_void, i32) -> AmfResult,
    get_dx11_device: unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void,
    lock_dx11: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    unlock_dx11: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    init_opencl: unsafe extern "C" fn(*mut c_void, *mut c_void) -> AmfResult,
    get_opencl_context: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    get_opencl_command_queue: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    get_opencl_device_id: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    convert_to_opencl: unsafe extern "C" fn(*mut c_void, *mut c_void) -> AmfResult,
    lock_opencl: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    unlock_opencl: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    init_opengl:
        unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> AmfResult,
    get_opengl_context: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    get_opengl_drawable: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    convert_to_opengl: unsafe extern "C" fn(*mut c_void, *mut c_void) -> AmfResult,
    lock_opengl: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    unlock_opengl: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    init_vulkan: unsafe extern "C" fn(*mut c_void, *mut c_void) -> AmfResult,
    get_vulkan_device: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    lock_vulkan: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    unlock_vulkan: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    alloc_buffer: unsafe extern "C" fn(*mut c_void, i32, usize, *mut *mut c_void) -> AmfResult,
    alloc_surface:
        unsafe extern "C" fn(*mut c_void, i32, i32, i32, i32, *mut *mut c_void) -> AmfResult,
}
#[repr(C)]
struct AmfContextObj {
    vtbl: *const AmfContextVtbl,
}

#[repr(C)]
struct AmfComponentVtbl {
    query_interface: QueryInterfaceFn,
    acquire: AcquireFn,
    release: ReleaseFn,
    set_property: unsafe extern "C" fn(*mut c_void, *const u16, AmfVariant) -> AmfResult,
    get_property: unsafe extern "C" fn(*mut c_void, *const u16, *mut AmfVariant) -> AmfResult,
    init: unsafe extern "C" fn(*mut c_void, i32, i32, i32) -> AmfResult,
    reinit: unsafe extern "C" fn(*mut c_void, i32, i32) -> AmfResult,
    terminate: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    drain: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    flush: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    submit_input: unsafe extern "C" fn(*mut c_void, *mut c_void) -> AmfResult,
    query_output: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> AmfResult,
    get_context: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    set_output_data_allocator_cb: unsafe extern "C" fn(*mut c_void, *mut c_void) -> AmfResult,
    get_caps: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> AmfResult,
    optimize: unsafe extern "C" fn(*mut c_void, *mut c_void) -> AmfResult,
}
#[repr(C)]
struct AmfComponentObj {
    vtbl: *const AmfComponentVtbl,
}

// AMFSurface — base AMFData methods then surface-specific. VERIFY: the `convert`
// slot (host read-back). We request host output via a property at Init instead
// of per-surface Convert where possible; this slot is the fallback.
#[repr(C)]
struct AmfSurfaceVtbl {
    query_interface: QueryInterfaceFn,
    acquire: AcquireFn,
    release: ReleaseFn,
    set_property: unsafe extern "C" fn(*mut c_void, *const u16, AmfVariant) -> AmfResult,
    get_property: unsafe extern "C" fn(*mut c_void, *const u16, *mut AmfVariant) -> AmfResult,
    duplicate: unsafe extern "C" fn(*mut c_void, i32, *mut *mut c_void) -> AmfResult,
    get_pts: unsafe extern "C" fn(*mut c_void) -> i64,
    set_pts: unsafe extern "C" fn(*mut c_void, i64),
    get_duration: unsafe extern "C" fn(*mut c_void) -> i64,
    set_duration: unsafe extern "C" fn(*mut c_void, i64),
    convert: unsafe extern "C" fn(*mut c_void, i32) -> AmfResult,
    get_planes_count: unsafe extern "C" fn(*mut c_void) -> usize,
    get_plane_at: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void,
    get_plane: unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void,
}
#[repr(C)]
struct AmfSurfaceObj {
    vtbl: *const AmfSurfaceVtbl,
}

#[repr(C)]
struct AmfPlaneVtbl {
    query_interface: QueryInterfaceFn,
    acquire: AcquireFn,
    release: ReleaseFn,
    get_type: unsafe extern "C" fn(*mut c_void) -> i32,
    get_native: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    get_pixel_size_in_bytes: unsafe extern "C" fn(*mut c_void) -> i32,
    get_offset_x: unsafe extern "C" fn(*mut c_void) -> i32,
    get_offset_y: unsafe extern "C" fn(*mut c_void) -> i32,
    get_width: unsafe extern "C" fn(*mut c_void) -> i32,
    get_height: unsafe extern "C" fn(*mut c_void) -> i32,
    get_h_pitch: unsafe extern "C" fn(*mut c_void) -> i32,
    get_v_pitch: unsafe extern "C" fn(*mut c_void) -> i32,
}
#[repr(C)]
struct AmfPlaneObj {
    vtbl: *const AmfPlaneVtbl,
}

#[repr(C)]
struct AmfBufferVtbl {
    query_interface: QueryInterfaceFn,
    acquire: AcquireFn,
    release: ReleaseFn,
    set_property: unsafe extern "C" fn(*mut c_void, *const u16, AmfVariant) -> AmfResult,
    get_property: unsafe extern "C" fn(*mut c_void, *const u16, *mut AmfVariant) -> AmfResult,
    duplicate: unsafe extern "C" fn(*mut c_void, i32, *mut *mut c_void) -> AmfResult,
    get_pts: unsafe extern "C" fn(*mut c_void) -> i64,
    set_pts: unsafe extern "C" fn(*mut c_void, i64),
    get_duration: unsafe extern "C" fn(*mut c_void) -> i64,
    set_duration: unsafe extern "C" fn(*mut c_void, i64),
    get_native: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    get_size: unsafe extern "C" fn(*mut c_void) -> usize,
}
#[repr(C)]
struct AmfBufferObj {
    vtbl: *const AmfBufferVtbl,
}

fn wide(s: &str) -> Vec<u16> {
    let mut out: Vec<u16> = s.encode_utf16().collect();
    out.push(0);
    out
}

/// AMF decoder component ID for a codec string, or `None` if AMF can't decode it.
fn amf_decoder_id(codec_lower: &str) -> Option<&'static str> {
    Some(match codec_lower {
        "h264" | "avc1" | "avc" => "AMFVideoDecoderUVD_H264_AVC",
        "h265" | "hevc" | "hvc1" | "hev1" | "hvc2" | "hev2" => "AMFVideoDecoderHW_H265_HEVC",
        "av1" | "av01" => "AMFVideoDecoderHW_AV1",
        "vp9" | "vp09" => "AMFVideoDecoderHW_VP9",
        _ => return None,
    })
}

/// Whether AMF can hardware-decode this codec.
pub fn supports(codec_lower: &str) -> bool {
    amf_decoder_id(codec_lower).is_some()
}

pub struct AmfDecoder {
    info: StreamInfo,
    _lib: Arc<libloading::Library>,
    context: *mut c_void,
    decoder: *mut c_void,
    frames: VecDeque<VideoFrame>,
    ten_bit: bool,
    pts: u64,
}

// Single-threaded driver; raw AMF pointers are owned + released in Drop.
unsafe impl Send for AmfDecoder {}

impl AmfDecoder {
    pub fn new(info: StreamInfo, gpu_index: u32) -> Result<Self> {
        let codec = info.codec.to_ascii_lowercase();
        let decoder_id = amf_decoder_id(&codec)
            .ok_or_else(|| anyhow::anyhow!("AMF cannot decode codec {codec}"))?;
        let ten_bit = matches!(info.pixel_format, PixelFormat::Yuv420p10le);

        let lib = unsafe { libloading::Library::new("libamfrt64.so.1") }
            .or_else(|_| unsafe { libloading::Library::new("libamfrt64.so") })
            .or_else(|_| unsafe { libloading::Library::new("amfrt64.dll") })
            .map_err(|e| anyhow::anyhow!("loading AMF runtime (AMD driver present?): {e}"))?;

        unsafe {
            let amf_init: libloading::Symbol<FnAmfInit> = lib.get(b"AMFInit")?;
            let mut factory: *mut c_void = ptr::null_mut();
            if amf_init(AMF_VERSION, &mut factory) != AMF_OK || factory.is_null() {
                bail!("AMFInit failed");
            }
            let factory_vt = &*(*(factory as *mut AmfFactoryObj)).vtbl;

            let mut context: *mut c_void = ptr::null_mut();
            if (factory_vt.create_context)(factory, &mut context) != AMF_OK || context.is_null() {
                bail!("AMFFactory::CreateContext failed");
            }
            let context_vt = &*(*(context as *mut AmfContextObj)).vtbl;
            if gpu_index != 0 {
                tracing::warn!(gpu_index, "AMF decode init picks adapter 0 unconditionally");
            }
            let rc_dx11 = (context_vt.init_dx11)(context, ptr::null_mut(), 0);
            if rc_dx11 != AMF_OK && (context_vt.init_vulkan)(context, ptr::null_mut()) != AMF_OK {
                (context_vt.release)(context);
                bail!("AMFContext::InitDX11 + InitVulkan both failed");
            }

            let id = wide(decoder_id);
            let mut decoder: *mut c_void = ptr::null_mut();
            if (factory_vt.create_component)(factory, context, id.as_ptr(), &mut decoder) != AMF_OK
                || decoder.is_null()
            {
                (context_vt.terminate)(context);
                (context_vt.release)(context);
                bail!("AMFFactory::CreateComponent({decoder_id}) failed — AMD decode unsupported");
            }
            let decoder_vt = &*(*(decoder as *mut AmfComponentObj)).vtbl;

            // VERIFY: some AMF decoders want AMF_VIDEO_DECODER_EXTRADATA (the
            // out-of-band SPS/PPS from MP4) set before Init; we currently rely
            // on in-band Annex-B parameter sets. See TODO.md.
            let surface_fmt = if ten_bit { AMF_SURFACE_P010 } else { AMF_SURFACE_NV12 };
            let w = info.width.max(16) as i32;
            let h = info.height.max(16) as i32;
            if (decoder_vt.init)(decoder, surface_fmt, w, h) != AMF_OK {
                (decoder_vt.terminate)(decoder);
                (context_vt.terminate)(context);
                (context_vt.release)(context);
                bail!("AMFComponent::Init (decoder) failed");
            }

            Ok(Self {
                info,
                _lib: Arc::new(lib),
                context,
                decoder,
                frames: VecDeque::new(),
                ten_bit,
                pts: 0,
            })
        }
    }

    /// Drain whatever `QueryOutput` has ready into the frame queue.
    unsafe fn drain_outputs(&mut self) -> Result<()> {
        let decoder_vt = unsafe { &*(*(self.decoder as *mut AmfComponentObj)).vtbl };
        loop {
            let mut data: *mut c_void = ptr::null_mut();
            let rc = unsafe { (decoder_vt.query_output)(self.decoder, &mut data) };
            match rc {
                AMF_OK if !data.is_null() => {
                    if let Some(frame) = unsafe { self.surface_to_frame(data) } {
                        self.frames.push_back(frame);
                    }
                    // Release the AMFData ref QueryOutput handed us.
                    let data_release = unsafe { (*(data as *mut AmfSurfaceObj)).vtbl };
                    unsafe { ((*data_release).release)(data) };
                }
                AMF_REPEAT | AMF_NEED_MORE_INPUT | AMF_EOF => break,
                _ => break,
            }
        }
        Ok(())
    }

    /// Downcast the AMFData → AMFSurface, copy NV12/P010 planes to a VideoFrame.
    unsafe fn surface_to_frame(&mut self, data: *mut c_void) -> Option<VideoFrame> {
        unsafe {
            // QueryInterface(AMFSurface). VERIFY: AMF_IID_SURFACE bytes.
            let data_vt = &*(*(data as *mut AmfSurfaceObj)).vtbl;
            let mut surf: *mut c_void = ptr::null_mut();
            if (data_vt.query_interface)(
                data,
                AMF_IID_SURFACE.as_ptr() as *const c_void,
                &mut surf,
            ) != 0
                || surf.is_null()
            {
                return None;
            }
            let surf_vt = &*(*(surf as *mut AmfSurfaceObj)).vtbl;
            // Ensure host-readable. VERIFY: convert slot / whether decoders can
            // be told to output host memory directly at Init.
            let _ = (surf_vt.convert)(surf, AMF_MEMORY_HOST);

            let read_plane = |idx: usize| -> Option<(*const u8, usize, usize, usize)> {
                let plane = (surf_vt.get_plane_at)(surf, idx);
                if plane.is_null() {
                    return None;
                }
                let pvt = &*(*(plane as *mut AmfPlaneObj)).vtbl;
                let native = (pvt.get_native)(plane) as *const u8;
                if native.is_null() {
                    return None;
                }
                Some((
                    native,
                    (pvt.get_h_pitch)(plane).max(0) as usize,
                    (pvt.get_width)(plane).max(0) as usize,
                    (pvt.get_height)(plane).max(0) as usize,
                ))
            };

            let (y_ptr, y_pitch, w, h) = read_plane(0)?;
            let (uv_ptr, uv_pitch, _, ch) = read_plane(1)?;
            let (format, packed) = if self.ten_bit {
                // VERIFY: P010 host plane layout (u16, high bits) → Yuv420p10le.
                let y = std::slice::from_raw_parts(y_ptr, y_pitch * h);
                let uv = std::slice::from_raw_parts(uv_ptr, uv_pitch * ch);
                (
                    PixelFormat::Yuv420p10le,
                    p010_planes_to_yuv420p10le(y, y_pitch, uv, uv_pitch, w, h),
                )
            } else {
                let y = std::slice::from_raw_parts(y_ptr, y_pitch * h);
                let uv = std::slice::from_raw_parts(uv_ptr, uv_pitch * ch);
                (
                    PixelFormat::Yuv420p,
                    nv12_planes_to_yuv420p(y, y_pitch, uv, uv_pitch, w, h),
                )
            };
            let frame = VideoFrame::new(
                Bytes::from(packed),
                w as u32,
                h as u32,
                format,
                ColorSpace::Bt709,
                self.pts,
            );
            self.pts += 1;
            ((*surf_vt).release)(surf);
            Some(frame)
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
        unsafe {
            let context_vt = &*(*(self.context as *mut AmfContextObj)).vtbl;
            let decoder_vt = &*(*(self.decoder as *mut AmfComponentObj)).vtbl;

            let mut buf: *mut c_void = ptr::null_mut();
            if (context_vt.alloc_buffer)(self.context, AMF_MEMORY_HOST, sample.len(), &mut buf)
                != AMF_OK
                || buf.is_null()
            {
                bail!("AMFContext::AllocBuffer({}) failed", sample.len());
            }
            let buf_vt = &*(*(buf as *mut AmfBufferObj)).vtbl;
            let dst = (buf_vt.get_native)(buf) as *mut u8;
            ptr::copy_nonoverlapping(sample.as_ptr(), dst, sample.len());

            // SubmitInput; on INPUT_FULL drain output and retry (bounded).
            for _ in 0..64 {
                match (decoder_vt.submit_input)(self.decoder, buf) {
                    AMF_OK | AMF_NEED_MORE_INPUT => break,
                    AMF_INPUT_FULL | AMF_REPEAT => {
                        self.drain_outputs()?;
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    rc => {
                        (buf_vt.release)(buf);
                        bail!("AMFComponent::SubmitInput (decode) failed: {rc}");
                    }
                }
            }
            (buf_vt.release)(buf);
            self.drain_outputs()?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        unsafe {
            let decoder_vt = &*(*(self.decoder as *mut AmfComponentObj)).vtbl;
            let _ = (decoder_vt.drain)(self.decoder);
            // Drain remaining outputs.
            for _ in 0..4096 {
                let before = self.frames.len();
                self.drain_outputs()?;
                let mut data: *mut c_void = ptr::null_mut();
                if (decoder_vt.query_output)(self.decoder, &mut data) == AMF_EOF {
                    break;
                }
                if !data.is_null() {
                    if let Some(f) = self.surface_to_frame(data) {
                        self.frames.push_back(f);
                    }
                    let dv = (*(data as *mut AmfSurfaceObj)).vtbl;
                    ((*dv).release)(data);
                }
                if self.frames.len() == before {
                    break;
                }
            }
        }
        Ok(())
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        Ok(self.frames.pop_front())
    }
}

impl Drop for AmfDecoder {
    fn drop(&mut self) {
        unsafe {
            if !self.decoder.is_null() {
                let vt = &*(*(self.decoder as *mut AmfComponentObj)).vtbl;
                (vt.terminate)(self.decoder);
                (vt.release)(self.decoder);
            }
            if !self.context.is_null() {
                let vt = &*(*(self.context as *mut AmfContextObj)).vtbl;
                (vt.terminate)(self.context);
                (vt.release)(self.context);
            }
        }
    }
}
