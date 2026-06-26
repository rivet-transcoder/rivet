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

// ─── oneVPL constants (mirror vendor/intel/mfxdefs.h) ────────────────
type MfxStatus = i32;
const MFX_ERR_NONE: MfxStatus = 0;
const MFX_ERR_MORE_DATA: MfxStatus = -10;
const MFX_ERR_MORE_SURFACE: MfxStatus = -11;

const MFX_IMPL_HARDWARE_ANY: u32 = 0x0100;
const MFX_CODEC_AVC: u32 = 0x20435641; // 'A','V','C',' '
const MFX_CODEC_HEVC: u32 = 0x43564548; // 'H','E','V','C'
const MFX_CODEC_AV1: u32 = 0x20315641; // 'A','V','1',' '
const MFX_CODEC_VP9: u32 = 0x20395056; // 'V','P','9',' '
const MFX_FOURCC_NV12: u32 = 0x3231564e;
const MFX_FOURCC_P010: u32 = 0x30313050;
const MFX_CHROMAFORMAT_YUV420: u16 = 1;
const MFX_IOPATTERN_OUT_SYSTEM_MEMORY: u16 = 0x10;

// ─── mfx structs (exact layouts shared with encode/qsv.rs) ───────────
#[repr(C)]
#[derive(Clone, Copy)]
struct MfxFrameInfo {
    bit_depth_luma: u32,
    bit_depth_chroma: u32,
    shift: u16,
    reserved_fi: [u16; 7],
    frame_id: u64,
    fourcc: u32,
    width: u16,
    height: u16,
    crop_x: u16,
    crop_y: u16,
    crop_w: u16,
    crop_h: u16,
    _crop_pad: [u16; 6],
    frame_rate_ext_n: u32,
    frame_rate_ext_d: u32,
    reserved3: u16,
    aspect_ratio_w: u16,
    aspect_ratio_h: u16,
    pic_struct: u16,
    chroma_format: u16,
    reserved2: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MfxInfoMfx {
    reserved: [u32; 6],
    low_power: u32,
    brc_param_multiplier: u16,
    _pad0: u16,
    frame_info: MfxFrameInfo,
    codec_id: u32,
    codec_profile: u16,
    codec_level: u16,
    num_thread: u16,
    target_usage: u16,
    gop_pic_size: u16,
    gop_ref_dist: u16,
    gop_opt_flag: u16,
    idr_interval: u16,
    rate_control_method: u16,
    qpi_or_delay: u16,
    buffer_size_kb: u16,
    qpp_or_kbps_or_icq: u16,
    qpb_or_maxkbps: u16,
    num_slice: u16,
    num_ref_frame: u16,
    encoded_order: u16,
    _tail: [u32; 27],
}

#[repr(C)]
struct MfxVideoParam {
    reserved: [u32; 2],
    reserved3: u16,
    async_depth: u16,
    mfx: MfxInfoMfx,
    protected: u16,
    io_pattern: u16,
    num_ext_param: u16,
    _pad1: u16,
    ext_param: *mut *mut c_void,
    _tail: [u32; 4],
}

#[repr(C)]
struct MfxFrameData {
    mem_id_or_y: *mut u8,
    u: *mut u8,
    v: *mut u8,
    a: *mut u8,
    pitch: u32,
    time_stamp: u64,
    frame_order: u32,
    locked: u16,
    reserved: [u16; 4],
    corrupted: u16,
    data_flag: u16,
}

#[repr(C)]
struct MfxFrameSurface1 {
    reserved: [u32; 4],
    info: MfxFrameInfo,
    data: MfxFrameData,
}

#[repr(C)]
struct MfxBitstream {
    reserved: [u32; 6],
    decode_time_stamp: i64,
    time_stamp: u64,
    data: *mut u8,
    data_offset: u32,
    data_length: u32,
    max_length: u32,
    pic_struct: u16,
    frame_type: u16,
    data_flag: u16,
    reserved2: u16,
}

#[repr(C)]
struct MfxVersion {
    minor: u16,
    major: u16,
}

type MfxSession = *mut c_void;
type MfxSyncPoint = *mut c_void;

type FnMfxInit =
    unsafe extern "C" fn(u32, *mut MfxVersion, *mut MfxSession) -> MfxStatus;
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
    reserved: [u32; 1],
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
                surf.data.mem_id_or_y = y;
                surf.data.u = uv;
                surf.data.v = uv.add(bytes_per); // V interleaved right after U
                surf.data.pitch = pitch as u32;
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
                let work = self.free_surface();
                if work.is_null() {
                    break; // pool exhausted; caller drains frames then retries
                }
                let mut out: *mut MfxFrameSurface1 = ptr::null_mut();
                let mut syncp: MfxSyncPoint = ptr::null_mut();
                let rc = decode_async(self.session, bs_ptr, work, &mut out, &mut syncp);
                match rc {
                    MFX_ERR_NONE if !syncp.is_null() && !out.is_null() => {
                        if sync_op(self.session, syncp, 60_000) == MFX_ERR_NONE {
                            if let Some(f) = self.read_surface(out) {
                                self.frames.push_back(f);
                            }
                        }
                        (*out).data.locked = 0; // we copied it out; free the surface
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

    /// Copy a decoded system-memory surface into a packed VideoFrame.
    unsafe fn read_surface(&mut self, surf: *mut MfxFrameSurface1) -> Option<VideoFrame> {
        unsafe {
            let s = &*surf;
            let w = (s.info.crop_w.max(s.info.width)) as usize;
            let h = (s.info.crop_h.max(s.info.height)) as usize;
            let pitch = s.data.pitch as usize;
            let ch = h.div_ceil(2);
            let y_ptr = s.data.mem_id_or_y;
            let uv_ptr = s.data.u;
            if y_ptr.is_null() || uv_ptr.is_null() {
                return None;
            }
            let y = std::slice::from_raw_parts(y_ptr, pitch * h);
            let uv = std::slice::from_raw_parts(uv_ptr, pitch * ch);
            let (format, packed) = if self.ten_bit {
                // VERIFY: P010 Shift handling on read-back.
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
