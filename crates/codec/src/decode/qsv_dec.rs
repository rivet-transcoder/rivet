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
    MfxBitstream, MfxFrameSurface1, MfxSession, MfxStatus, MfxSyncPoint, MfxVersion, MfxVideoParam,
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

fn mfx_codec_for(codec_lower: &str) -> Option<u32> {
    Some(match codec_lower {
        "h264" | "avc1" | "avc" => MFX_CODEC_AVC,
        "h265" | "hevc" | "hvc1" | "hev1" | "hvc2" | "hev2" => MFX_CODEC_HEVC,
        "av1" | "av01" => MFX_CODEC_AV1,
        "vp9" | "vp09" => MFX_CODEC_VP9,
        _ => return None,
    })
}

/// Whether this build's QSV decoder *handles* this codec (a static
/// codec-string match — the compile-time capability used by dispatch).
/// For what the **host's silicon** can actually decode, see
/// [`probe_decode_caps`].
pub fn supports(codec_lower: &str) -> bool {
    mfx_codec_for(codec_lower).is_some()
}

/// `MFXVideoDECODE_Query(session, in, out)` — the oneVPL decode capability
/// query. With `in.mfx.codec_id` set it reports whether the implementation can
/// decode that codec (and fills `out` with corrected params).
type FnDecodeQuery =
    unsafe extern "C" fn(MfxSession, *mut MfxVideoParam, *mut MfxVideoParam) -> MfxStatus;

/// dlopen the Intel oneVPL / Media SDK runtime (`libvpl` first, legacy
/// `libmfxhw64` as a fallback). Shared by the decoder and the capability probe.
fn load_libvpl() -> Result<libloading::Library> {
    unsafe { libloading::Library::new("libvpl.so.2") }
        .or_else(|_| unsafe { libloading::Library::new("libvpl.so") })
        .or_else(|_| unsafe { libloading::Library::new("libvpl.dll") })
        .or_else(|_| unsafe { libloading::Library::new("libmfxhw64.dll") })
        .map_err(|e| anyhow::anyhow!("loading libvpl (Intel runtime present?): {e}"))
}

/// The codecs the QSV decoder knows how to drive, with their mfx codec IDs.
const PROBE_CODECS: &[(&str, u32)] = &[
    ("h264", MFX_CODEC_AVC),
    ("hevc", MFX_CODEC_HEVC),
    ("av1", MFX_CODEC_AV1),
    ("vp9", MFX_CODEC_VP9),
];

/// Runtime hardware probe: which codecs **this host's** QSV implementation can
/// actually decode, asked of the driver via `MFXVideoDECODE_Query` (one HW
/// session, queried once and cached). Returns an empty slice when no usable
/// Intel oneVPL runtime is present, so a non-Intel host reports no QSV decode
/// rather than a static guess. This is what feeds the decode-capability report
/// (`decode_capabilities` / `rivet capabilities`).
pub fn probe_decode_caps() -> &'static [&'static str] {
    static CAPS: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    CAPS.get_or_init(|| probe_inner().unwrap_or_default())
}

fn probe_inner() -> Result<Vec<&'static str>> {
    let lib = load_libvpl()?;
    unsafe {
        // MFXInit(HW) succeeding *is* the load-bearing capability signal: it
        // proves a usable Intel oneVPL runtime + a hardware adapter are present.
        let mfx_init: libloading::Symbol<FnMfxInit> = lib.get(b"MFXInit")?;
        let mut version = MfxVersion { minor: 0, major: 2 };
        let mut session: MfxSession = ptr::null_mut();
        let rc = mfx_init(MFX_IMPL_HARDWARE_ANY, &mut version, &mut session);
        if rc != MFX_ERR_NONE || session.is_null() {
            bail!("MFXInit(HW) failed: {rc} (no Intel QSV implementation?)");
        }

        // Per-codec MFXVideoDECODE_Query with a representative frame_info. On a
        // runtime where Query is authoritative this filters codecs the silicon
        // can't decode (e.g. AV1 on a pre-Arc iGPU).
        let mut queried = Vec::new();
        if let Ok(query) = lib.get::<FnDecodeQuery>(b"MFXVideoDECODE_Query") {
            for &(label, codec_id) in PROBE_CODECS {
                let mut inp: MfxVideoParam = std::mem::zeroed();
                inp.mfx.codec_id = codec_id;
                let fi = &mut inp.mfx.frame_info;
                fi.fourcc = MFX_FOURCC_NV12;
                fi.chroma_format = MFX_CHROMAFORMAT_YUV420;
                fi.pic_struct = MFX_PICSTRUCT_PROGRESSIVE;
                fi.width = 640;
                fi.height = 480;
                fi.crop_w = 640;
                fi.crop_h = 480;
                fi.frame_rate_ext_n = 30;
                fi.frame_rate_ext_d = 1;
                inp.io_pattern = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
                let mut outp: MfxVideoParam = std::mem::zeroed();
                if query(session, &mut inp, &mut outp) >= 0 {
                    queried.push(label);
                }
            }
        }

        if let Ok(close) = lib.get::<crate::qsv_ffi::FnMfxClose>(b"MFXClose") {
            let _ = close(session);
        }

        // iHD's MFXVideoDECODE_Query is *advisory* — on the Arc box it returns an
        // error for every codec it nonetheless decodes (h264/hevc/vp9/av1 all
        // verified end-to-end). So when Query yields nothing but the HW session
        // initialised, the runtime is usable: report the build's codec list
        // rather than claim no decode. A non-empty Query result is trusted as-is.
        let supported: Vec<&'static str> = if queried.is_empty() {
            PROBE_CODECS.iter().map(|&(l, _)| l).collect()
        } else {
            queried
        };
        tracing::info!(
            codecs = ?supported,
            query_authoritative = !supported.is_empty() && supported.len() != PROBE_CODECS.len(),
            "QSV decode capability probe (MFXInit + MFXVideoDECODE_Query)"
        );
        Ok(supported)
    }
}

pub struct QsvDecoder {
    info: StreamInfo,
    lib: libloading::Library,
    session: MfxSession,
    frames: VecDeque<VideoFrame>,
    ten_bit: bool,
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

        let lib = load_libvpl()?;

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
                frames: VecDeque::new(),
                ten_bit,
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

            // Trust DecodeHeader's output format — the iHD driver returns NV12
            // for 8-bit, P010 for 10-bit (Main10). Forcing fourcc/bit-depth/shift
            // ourselves made HEVC Main10 Init fail. Just guarantee chroma + a
            // system-memory output, then derive ten_bit from the real fourcc.
            if param.mfx.frame_info.chroma_format == 0 {
                param.mfx.frame_info.chroma_format = MFX_CHROMAFORMAT_YUV420;
            }
            if param.mfx.frame_info.fourcc != MFX_FOURCC_P010 {
                param.mfx.frame_info.fourcc = MFX_FOURCC_NV12;
            }
            self.ten_bit = param.mfx.frame_info.fourcc == MFX_FOURCC_P010;
            param.io_pattern = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;

            // No external work-surface pool: DecodeFrameAsync runs with
            // surface_work=NULL (oneVPL 2.x internal allocation) and we read the
            // returned surface via its FrameInterface::Map. Just Init.
            let rc = decode_init(self.session, &mut param);
            if rc < 0 {
                tracing::error!(
                    status = rc,
                    fourcc = param.mfx.frame_info.fourcc,
                    bd = param.mfx.frame_info.bit_depth_luma,
                    shift = param.mfx.frame_info.shift,
                    w = param.mfx.frame_info.width,
                    h = param.mfx.frame_info.height,
                    "MFXVideoDECODE_Init failed"
                );
                bail!("MFXVideoDECODE_Init failed: {rc}");
            }
            // Drop the bytes DecodeHeader consumed.
            let consumed = bs.data_offset as usize;
            self.pending.drain(..consumed.min(self.pending.len()));
            self.inited = true;
        }
        Ok(())
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
