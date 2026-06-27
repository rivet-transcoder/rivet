//! Intel **QSV / oneVPL hardware decode** — hand-rolled FFI (our own SDK-mirror
//! code, no external wrapper crate). Decodes H.264 / HEVC / AV1 / VP9 on Intel
//! Arc / Meteor Lake+ by driving libvpl directly, mirroring the oneVPL decode
//! API and reusing the exact mfx struct layouts from the in-tree QSV *encoder*
//! (`encode/qsv.rs`, against `vendor/intel/` oneVPL 2.10 headers).
//!
//! Flow: dlopen `libvpl` → `MFXInit(HW)` → feed the first sample(s) to
//! `MFXVideoDECODE_DecodeHeader` (→ `mfxVideoParam`) → `QueryIOSurf` (work-pool
//! size) → allocate system-memory NV12/P010 surfaces → `Init` → per sample:
//! `DecodeFrameAsync` over the bitstream, `SyncOperation`, read the surface →
//! `Yuv420p`/`Yuv420p10le`. Drain on `finish()`.
//!
//! **Verified-by-review only** — no Intel Arc on the dev box. Spots needing
//! confirmation on real hardware are flagged `// VERIFY:` and tracked in
//! TODO.md (DecodeHeader retry, work-surface pool sizing, P010 shift). If a
//! stream fails, the `ffmpeg` feature is the fallback for Intel hosts.
#![cfg(feature = "qsv")]

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr;

use anyhow::{Result, bail};
use bytes::Bytes;

use super::{Decoder, StreamInfo, nv12_planes_to_yuv420p, p010_planes_to_yuv420p10le};
use crate::frame::{ColorSpace, PixelFormat, VideoFrame};

// Shared mfx structs + constants + types live in one place now (`qsv_ffi`),
// so the encoder and decoder can't drift apart on layout again.
use crate::qsv_ffi::{
    MFX_CHROMAFORMAT_YUV420, MFX_CODEC_AV1, MFX_CODEC_AVC, MFX_CODEC_HEVC, MFX_CODEC_VP9,
    MFX_ERR_MORE_DATA, MFX_ERR_MORE_SURFACE, MFX_ERR_NONE, MFX_FOURCC_NV12, MFX_FOURCC_P010,
    MfxBitstream, MfxFrameData, MfxFrameInfo, MfxFrameSurface1, MfxInfoMfx, MfxSession, MfxStatus,
    MfxSyncPoint, MfxVersion, MfxVideoParam,
};

// decode-only constants
const MFX_IMPL_HARDWARE_ANY: u32 = 0x0100;
const MFX_IOPATTERN_OUT_SYSTEM_MEMORY: u16 = 0x10;

// Shared mfx structs (MfxFrameInfo / MfxInfoMfx / MfxVideoParam / MfxFrameData /
// MfxFrameSurface1 / MfxBitstream / MfxVersion) are imported from `qsv_ffi`.

/// `mfxFrameSurfaceInterface` vtable (112 bytes, offsetof-verified): the methods
/// on an internally-allocated decode surface. Map @40, Unmap @48, Release @24.
#[repr(C)]
struct MfxFrameSurfaceInterface {
    context: *mut c_void,
    _version_reserved: [u8; 8],
    add_ref: unsafe extern "C" fn(*mut MfxFrameSurface1) -> MfxStatus,
    release: unsafe extern "C" fn(*mut MfxFrameSurface1) -> MfxStatus,
    _get_ref_counter: *mut c_void,
    map: unsafe extern "C" fn(*mut MfxFrameSurface1, u32) -> MfxStatus,
    unmap: unsafe extern "C" fn(*mut MfxFrameSurface1) -> MfxStatus,
    _get_native_handle: *mut c_void,
    _get_device_handle: *mut c_void,
    _synchronize: unsafe extern "C" fn(*mut MfxFrameSurface1, u32) -> MfxStatus,
    _tail: [u8; 32],
}
const MFX_MAP_READ: u32 = 1;

/// The FrameInterface vtable pointer lives at offset 0 of an internally
/// allocated decode surface.
unsafe fn surface_iface(surf: *mut MfxFrameSurface1) -> *const MfxFrameSurfaceInterface {
    unsafe { *(surf as *const *const MfxFrameSurfaceInterface) }
}

type FnMfxInit = unsafe extern "C" fn(u32, *mut MfxVersion, *mut MfxSession) -> MfxStatus;
type FnMfxClose = unsafe extern "C" fn(MfxSession) -> MfxStatus;
type FnDecodeHeader =
    unsafe extern "C" fn(MfxSession, *mut MfxBitstream, *mut MfxVideoParam) -> MfxStatus;
type FnDecodeQueryIOSurf =
    unsafe extern "C" fn(MfxSession, *mut MfxVideoParam, *mut MfxFrameAllocRequest) -> MfxStatus;
type FnDecodeInit = unsafe extern "C" fn(MfxSession, *mut MfxVideoParam) -> MfxStatus;
type FnDecodeClose = unsafe extern "C" fn(MfxSession) -> MfxStatus;
type FnDecodeFrameAsync = unsafe extern "C" fn(
    MfxSession,
    *mut MfxBitstream,
    *mut MfxFrameSurface1,
    *mut *mut MfxFrameSurface1,
    *mut MfxSyncPoint,
) -> MfxStatus;
type FnSyncOperation = unsafe extern "C" fn(MfxSession, MfxSyncPoint, u32) -> MfxStatus;

#[repr(C)]
struct MfxFrameAllocRequest {
    // Real oneVPL mfxFrameAllocRequest — 92 bytes (AllocId union @0, Info @16).
    alloc_id: u32,
    reserved3: [u32; 3],
    info: MfxFrameInfo,
    mem_type: u16,
    num_frame_min: u16,
    num_frame_suggested: u16,
    reserved2: u16,
}

fn mfx_codec_for(codec_lower: &str) -> Option<u32> {
    Some(match codec_lower {
        "h264" | "avc1" | "avc" => MFX_CODEC_AVC,
        "h265" | "hevc" | "hvc1" | "hev1" | "hvc2" | "hev2" => MFX_CODEC_HEVC,
        "av1" | "av01" => MFX_CODEC_AV1,
        "vp9" | "vp09" => MFX_CODEC_VP9,
        _ => return None,
    })
}

/// Whether QSV can hardware-decode this codec.
pub fn supports(codec_lower: &str) -> bool {
    mfx_codec_for(codec_lower).is_some()
}

/// A system-memory work surface + its backing buffer.
struct WorkSurface {
    surf: Box<MfxFrameSurface1>,
    _backing: Vec<u8>,
}

pub struct QsvDecoder {
    info: StreamInfo,
    lib: libloading::Library,
    session: MfxSession,
    surfaces: Vec<WorkSurface>,
    frames: VecDeque<VideoFrame>,
    ten_bit: bool,
    width: usize,
    height: usize,
    pitch: usize,
    pending: Vec<u8>,
    inited: bool,
    codec_id: u32,
    pts: u64,
}

unsafe impl Send for QsvDecoder {}

impl QsvDecoder {
    pub fn new(info: StreamInfo, _gpu_index: u32) -> Result<Self> {
        let codec = info.codec.to_ascii_lowercase();
        let codec_id =
            mfx_codec_for(&codec).ok_or_else(|| anyhow::anyhow!("QSV cannot decode {codec}"))?;
        let ten_bit = matches!(info.pixel_format, PixelFormat::Yuv420p10le);

        let lib = unsafe { libloading::Library::new("libvpl.so.2") }
            .or_else(|_| unsafe { libloading::Library::new("libvpl.so") })
            .or_else(|_| unsafe { libloading::Library::new("libvpl.dll") })
            .or_else(|_| unsafe { libloading::Library::new("libmfxhw64.dll") })
            .map_err(|e| anyhow::anyhow!("loading libvpl (Intel runtime present?): {e}"))?;

        unsafe {
            // Legacy init path — request a hardware implementation. VERIFY: AV1
            // decode needs a oneVPL 2.x runtime; bump the requested version if
            // Init reports an old implementation.
            let mfx_init: libloading::Symbol<FnMfxInit> = lib.get(b"MFXInit")?;
            let mut version = MfxVersion { minor: 0, major: 2 };
            let mut session: MfxSession = ptr::null_mut();
            let rc = mfx_init(MFX_IMPL_HARDWARE_ANY, &mut version, &mut session);
            if rc != MFX_ERR_NONE || session.is_null() {
                bail!("MFXInit(HW) failed: {rc} (no Intel QSV implementation?)");
            }

            Ok(Self {
                info,
                lib,
                session,
                surfaces: Vec::new(),
                frames: VecDeque::new(),
                ten_bit,
                width: 0,
                height: 0,
                pitch: 0,
                pending: Vec::new(),
                inited: false,
                codec_id,
                pts: 0,
            })
        }
    }

    /// After enough bitstream is buffered, parse the header + init the decoder.
    unsafe fn try_init(&mut self) -> Result<()> {
        if self.inited || self.pending.is_empty() {
            return Ok(());
        }
        unsafe {
            let decode_header: libloading::Symbol<FnDecodeHeader> =
                self.lib.get(b"MFXVideoDECODE_DecodeHeader")?;
            let query_iosurf: libloading::Symbol<FnDecodeQueryIOSurf> =
                self.lib.get(b"MFXVideoDECODE_QueryIOSurf")?;
            let decode_init: libloading::Symbol<FnDecodeInit> =
                self.lib.get(b"MFXVideoDECODE_Init")?;

            let mut param: MfxVideoParam = std::mem::zeroed();
            param.mfx.codec_id = self.codec_id;
            param.io_pattern = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;

            let mut bs: MfxBitstream = std::mem::zeroed();
            bs.data = self.pending.as_mut_ptr();
            bs.data_length = self.pending.len() as u32;
            bs.max_length = self.pending.len() as u32;

            // VERIFY: DecodeHeader returns MORE_DATA until it has the full SPS/PPS
            // (or seq header). We feed all buffered data; if it still wants more,
            // wait for the next push_sample.
            let rc = decode_header(self.session, &mut bs, &mut param);
            if rc == MFX_ERR_MORE_DATA {
                return Ok(());
            }
            if rc < 0 {
                bail!("MFXVideoDECODE_DecodeHeader failed: {rc}");
            }

            // Force NV12 / P010 system-memory output.
            param.mfx.frame_info.fourcc = if self.ten_bit { MFX_FOURCC_P010 } else { MFX_FOURCC_NV12 };
            param.mfx.frame_info.chroma_format = MFX_CHROMAFORMAT_YUV420;
            if self.ten_bit {
                param.mfx.frame_info.bit_depth_luma = 10;
                param.mfx.frame_info.bit_depth_chroma = 10;
                param.mfx.frame_info.shift = 1;
            }
            param.io_pattern = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;

            let w = param.mfx.frame_info.width.max(16) as usize;
            let h = param.mfx.frame_info.height.max(16) as usize;

            // Work-surface pool size. VERIFY: QueryIOSurf's suggested count vs the
            // stream's actual DPB depth; we add a few for safety.
            let mut req: MfxFrameAllocRequest = std::mem::zeroed();
            let n = if query_iosurf(self.session, &mut param, &mut req) == MFX_ERR_NONE {
                (req.num_frame_suggested as usize).max(4) + 4
            } else {
                8
            };

            // Allocate system-memory surfaces. Each plane is `pitch * h`; pitch
            // is byte width (2× for 10-bit). NV12 has Y + interleaved UV.
            let bytes_per = if self.ten_bit { 2 } else { 1 };
            let pitch = w * bytes_per;
            self.width = w;
            self.height = h;
            self.pitch = pitch;
            let frame_bytes = pitch * h + pitch * h.div_ceil(2);
            for _ in 0..n {
                let mut backing = vec![0u8; frame_bytes];
                let y = backing.as_mut_ptr();
                let uv = backing.as_mut_ptr().add(pitch * h);
                let mut surf: Box<MfxFrameSurface1> = Box::new(std::mem::zeroed());
                surf.info = param.mfx.frame_info;
                surf.data.y = y;
                surf.data.u = uv;
                surf.data.v = uv.add(bytes_per); // V interleaved right after U
                surf.data.pitch = (pitch & 0xFFFF) as u16;
                surf.data.pitch_high = (pitch >> 16) as u16;
                self.surfaces.push(WorkSurface { surf, _backing: backing });
            }

            if decode_init(self.session, &mut param) < 0 {
                bail!("MFXVideoDECODE_Init failed");
            }
            // Drop the bytes DecodeHeader consumed.
            let consumed = bs.data_offset as usize;
            self.pending.drain(..consumed.min(self.pending.len()));
            self.inited = true;
        }
        Ok(())
    }

    /// A free (`Locked == 0`) work surface pointer, or null if the pool is dry.
    fn free_surface(&mut self) -> *mut MfxFrameSurface1 {
        for ws in &mut self.surfaces {
            if ws.surf.data.locked == 0 {
                return ws.surf.as_mut() as *mut MfxFrameSurface1;
            }
        }
        ptr::null_mut()
    }

    /// Pump `self.pending` through DecodeFrameAsync, collecting decoded frames.
    unsafe fn pump(&mut self, drain: bool) -> Result<()> {
        if !self.inited {
            return Ok(());
        }
        unsafe {
            // Deref the symbols to plain fn pointers so we don't hold a borrow
            // on `self.lib` across the `self.free_surface()` / `read_surface()`
            // calls inside the loop.
            let decode_async = *self
                .lib
                .get::<FnDecodeFrameAsync>(b"MFXVideoDECODE_DecodeFrameAsync")?;
            let sync_op = *self.lib.get::<FnSyncOperation>(b"MFXVideoCORE_SyncOperation")?;

            let mut bs: MfxBitstream = std::mem::zeroed();
            if !drain {
                bs.data = self.pending.as_mut_ptr();
                bs.data_length = self.pending.len() as u32;
                bs.max_length = self.pending.len() as u32;
            }
            let bs_ptr = if drain { ptr::null_mut() } else { &mut bs as *mut MfxBitstream };

            loop {
                let mut out: *mut MfxFrameSurface1 = ptr::null_mut();
                let mut syncp: MfxSyncPoint = ptr::null_mut();
                // surface_work = NULL → oneVPL 2.x internal surface allocation
                // (the path shiguredo_vpl uses; the external work-surface pool
                // never produced frames on the iHD 2.x runtime).
                let rc = decode_async(self.session, bs_ptr, ptr::null_mut(), &mut out, &mut syncp);
                match rc {
                    MFX_ERR_NONE if !out.is_null() => {
                        if !syncp.is_null() {
                            sync_op(self.session, syncp, 60_000);
                        }
                        if let Some(f) = self.read_surface(out) {
                            self.frames.push_back(f);
                        }
                        // Release the internally-allocated surface.
                        let iface = surface_iface(out);
                        if !iface.is_null() {
                            ((*iface).release)(out);
                        }
                    }
                    MFX_ERR_MORE_SURFACE => continue,
                    MFX_ERR_MORE_DATA => break,
                    rc if rc > 0 => continue, // warning (e.g. device busy / param changed)
                    _ => break,
                }
                if !drain && bs.data_length == 0 {
                    break;
                }
            }
            if !drain {
                let consumed = bs.data_offset as usize;
                self.pending.drain(..consumed.min(self.pending.len()));
            }
        }
        Ok(())
    }

    /// Map an internally-allocated decode surface and copy it to a VideoFrame.
    unsafe fn read_surface(&mut self, surf: *mut MfxFrameSurface1) -> Option<VideoFrame> {
        unsafe {
            // oneVPL 2.x: the plane pointers are only valid between Map/Unmap.
            let iface = surface_iface(surf);
            if iface.is_null() {
                return None;
            }
            if ((*iface).map)(surf, MFX_MAP_READ) != MFX_ERR_NONE {
                return None;
            }
            let s = &*surf;
            // Output the DISPLAY (crop) dims, not the coded dims. e.g. 1080p
            // codes as 1088 (16-aligned); emitting 1088-tall frames into a
            // 1080-configured encoder fails EncodeFrameAsync. Crop is the
            // displayable size; fall back to coded only if crop is unset.
            let w = if s.info.crop_w > 0 { s.info.crop_w } else { s.info.width } as usize;
            let h = if s.info.crop_h > 0 { s.info.crop_h } else { s.info.height } as usize;
            let pitch = s.data.pitch as usize | ((s.data.pitch_high as usize) << 16);
            let ch = h.div_ceil(2);
            let y_ptr = s.data.y;
            let uv_ptr = s.data.u;
            let result = if y_ptr.is_null() || uv_ptr.is_null() || pitch == 0 {
                None
            } else {
                let y = std::slice::from_raw_parts(y_ptr, pitch * h);
                let uv = std::slice::from_raw_parts(uv_ptr, pitch * ch);
                let (format, packed) = if self.ten_bit {
                    (
                        PixelFormat::Yuv420p10le,
                        p010_planes_to_yuv420p10le(y, pitch, uv, pitch, w, h),
                    )
                } else {
                    (PixelFormat::Yuv420p, nv12_planes_to_yuv420p(y, pitch, uv, pitch, w, h))
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
                Some(frame)
            };
            ((*iface).unmap)(surf);
            result
        }
    }
}

impl Decoder for QsvDecoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    fn push_sample(&mut self, sample: &[u8]) -> Result<()> {
        self.pending.extend_from_slice(sample);
        unsafe {
            if !self.inited {
                self.try_init()?;
            }
            self.pump(false)?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        unsafe {
            if !self.inited {
                self.try_init()?;
            }
            self.pump(false)?;
            self.pump(true)?; // drain
        }
        Ok(())
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        Ok(self.frames.pop_front())
    }
}

impl Drop for QsvDecoder {
    fn drop(&mut self) {
        unsafe {
            if let Ok(close) = self.lib.get::<FnDecodeClose>(b"MFXVideoDECODE_Close") {
                let _ = close(self.session);
            }
            if let Ok(mfx_close) = self.lib.get::<FnMfxClose>(b"MFXClose") {
                let _ = mfx_close(self.session);
            }
        }
    }
}
