//! Shared Intel oneVPL FFI — the mfx struct layouts used by BOTH the QSV
//! encoder (`encode/qsv.rs`) and the QSV decoder (`decode/qsv_dec.rs`).
//!
//! These were previously duplicated in both files, which is how the same struct
//! layout bug shipped in two places. They are defined ONCE here, verified by
//! `offsetof` against the installed oneVPL 2.16 headers on an Intel Arc box
//! (the dev box's vendored `mfxstructs.h` was a wrong hand-simplified copy).
//! Sizes: mfxFrameInfo=68, mfxInfoMFX=136, mfxVideoParam=208 (mfx union is
//! 168B), mfxFrameData=96 (Y/U/V planes @48/@56/@64), mfxFrameSurface1=184,
//! mfxBitstream=72.
#![cfg(feature = "qsv")]
#![allow(dead_code)]

use std::ffi::c_void;

pub(crate) type MfxStatus = i32;
pub(crate) type MfxSession = *mut c_void;
pub(crate) type MfxSyncPoint = *mut c_void;

// ─── Status codes (shared) ───────────────────────────────────────────
pub(crate) const MFX_ERR_NONE: MfxStatus = 0;
pub(crate) const MFX_ERR_MORE_DATA: MfxStatus = -10;
pub(crate) const MFX_ERR_MORE_SURFACE: MfxStatus = -11;

// ─── Codec / format / chroma (shared) ────────────────────────────────
pub(crate) const MFX_CODEC_AVC: u32 = 0x20435641; // 'A','V','C',' '
pub(crate) const MFX_CODEC_HEVC: u32 = 0x43564548; // 'H','E','V','C'
pub(crate) const MFX_CODEC_AV1: u32 = 0x20315641; // 'A','V','1',' '
pub(crate) const MFX_CODEC_VP9: u32 = 0x20395056; // 'V','P','9',' '
pub(crate) const MFX_FOURCC_NV12: u32 = 0x3231564e; // 'N','V','1','2'
pub(crate) const MFX_FOURCC_P010: u32 = 0x30313050; // 'P','0','1','0'
pub(crate) const MFX_CHROMAFORMAT_YUV420: u16 = 1;
pub(crate) const MFX_PICSTRUCT_PROGRESSIVE: u16 = 1;

// ─── mfxVersion ──────────────────────────────────────────────────────
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct MfxVersion {
    pub(crate) minor: u16,
    pub(crate) major: u16,
}

// ─── mfxFrameInfo — 68 bytes ─────────────────────────────────────────
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct MfxFrameInfo {
    pub(crate) reserved: [u32; 4],
    pub(crate) channel_id: u16,
    pub(crate) bit_depth_luma: u16,
    pub(crate) bit_depth_chroma: u16,
    pub(crate) shift: u16,
    pub(crate) frame_id: [u16; 4], // mfxFrameId — 8 bytes, 2-aligned (NOT u64)
    pub(crate) fourcc: u32,
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) crop_x: u16,
    pub(crate) crop_y: u16,
    pub(crate) crop_w: u16,
    pub(crate) crop_h: u16,
    pub(crate) frame_rate_ext_n: u32,
    pub(crate) frame_rate_ext_d: u32,
    pub(crate) reserved3: u16,
    pub(crate) aspect_ratio_w: u16,
    pub(crate) aspect_ratio_h: u16,
    pub(crate) pic_struct: u16,
    pub(crate) chroma_format: u16,
    pub(crate) reserved2: u16,
}

// ─── mfxInfoMFX — 136 bytes ──────────────────────────────────────────
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct MfxInfoMfx {
    pub(crate) reserved: [u32; 7],
    pub(crate) low_power: u16,
    pub(crate) brc_param_multiplier: u16,
    pub(crate) frame_info: MfxFrameInfo,
    pub(crate) codec_id: u32,
    pub(crate) codec_profile: u16,
    pub(crate) codec_level: u16,
    pub(crate) num_thread: u16,
    pub(crate) target_usage: u16,
    pub(crate) gop_pic_size: u16,
    pub(crate) gop_ref_dist: u16,
    pub(crate) gop_opt_flag: u16,
    pub(crate) idr_interval: u16,
    pub(crate) rate_control_method: u16,
    pub(crate) qpi_or_delay: u16,
    pub(crate) buffer_size_kb: u16,
    pub(crate) qpp_or_kbps_or_icq: u16,
    pub(crate) qpb_or_maxkbps: u16,
    pub(crate) num_slice: u16,
    pub(crate) num_ref_frame: u16,
    pub(crate) encoded_order: u16,
}

// ─── mfxExtBuffer — 8 bytes ──────────────────────────────────────────
#[repr(C)]
pub(crate) struct MfxExtBuffer {
    pub(crate) buffer_id: u32,
    pub(crate) buffer_sz: u32,
}

// ─── mfxVideoParam — 208 bytes (mfx union is 168B) ───────────────────
#[repr(C)]
pub(crate) struct MfxVideoParam {
    pub(crate) alloc_id: u32,
    pub(crate) reserved: [u32; 2],
    pub(crate) reserved3: u16,
    pub(crate) async_depth: u16,
    pub(crate) mfx: MfxInfoMfx,
    pub(crate) _mfx_union_pad: [u8; 32],
    pub(crate) protected: u16,
    pub(crate) io_pattern: u16,
    pub(crate) ext_param: *mut *mut MfxExtBuffer,
    pub(crate) num_ext_param: u16,
    pub(crate) reserved2: u16,
}

// ─── mfxFrameData — 96 bytes (Y/U/V @48/@56/@64) ─────────────────────
#[repr(C)]
pub(crate) struct MfxFrameData {
    pub(crate) ext_param_or_reserved2: u64,
    pub(crate) num_ext_param: u16,
    pub(crate) reserved: [u16; 9],
    pub(crate) mem_type: u16,
    pub(crate) pitch_high: u16,
    pub(crate) time_stamp: u64,
    pub(crate) frame_order: u32,
    pub(crate) locked: u16,
    pub(crate) pitch: u16,
    pub(crate) y: *mut u8,
    pub(crate) u: *mut u8,
    pub(crate) v: *mut u8,
    pub(crate) a: *mut u8,
    pub(crate) mem_id: *mut c_void,
    pub(crate) corrupted: u16,
    pub(crate) data_flag: u16,
}

// ─── mfxFrameSurface1 — 184 bytes ────────────────────────────────────
#[repr(C)]
pub(crate) struct MfxFrameSurface1 {
    pub(crate) reserved: [u32; 4],
    pub(crate) info: MfxFrameInfo,
    pub(crate) data: MfxFrameData,
}

// ─── mfxBitstream — 72 bytes ─────────────────────────────────────────
#[repr(C)]
pub(crate) struct MfxBitstream {
    pub(crate) reserved: [u32; 6],
    pub(crate) decode_time_stamp: i64,
    pub(crate) time_stamp: u64,
    pub(crate) data: *mut u8,
    pub(crate) data_offset: u32,
    pub(crate) data_length: u32,
    pub(crate) max_length: u32,
    pub(crate) pic_struct: u16,
    pub(crate) frame_type: u16,
    pub(crate) data_flag: u16,
    pub(crate) reserved2: u16,
}

// ─── Shared fn types ─────────────────────────────────────────────────
pub(crate) type FnMfxClose = unsafe extern "C" fn(MfxSession) -> MfxStatus;
pub(crate) type FnSyncOperation = unsafe extern "C" fn(MfxSession, MfxSyncPoint, u32) -> MfxStatus;

// ─── ABI guards (offsetof-verified on Arc / oneVPL 2.16) ─────────────
const _: () = assert!(std::mem::size_of::<MfxVersion>() == 4);
const _: () = assert!(std::mem::size_of::<MfxFrameInfo>() == 68);
const _: () = assert!(std::mem::size_of::<MfxInfoMfx>() == 136);
const _: () = assert!(std::mem::size_of::<MfxVideoParam>() == 208);
const _: () = assert!(std::mem::size_of::<MfxFrameData>() == 96);
const _: () = assert!(std::mem::size_of::<MfxFrameSurface1>() == 184);
const _: () = assert!(std::mem::size_of::<MfxBitstream>() == 72);
const _: () = assert!(std::mem::size_of::<MfxExtBuffer>() == 8);
