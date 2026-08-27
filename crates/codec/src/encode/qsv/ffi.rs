//! oneVPL ABI constants, function-pointer types, and encoder-only ext-buffer
//! struct definitions for the QSV encode path.
//!
//! Shared mfx struct layouts (MfxVersion / MfxFrameInfo / MfxInfoMfx /
//! MfxVideoParam / MfxExtBuffer / MfxFrameData / MfxFrameSurface1 /
//! MfxBitstream) live in `crate::qsv_ffi` so encode and decode share one
//! offsetof-verified definition.  Everything below is encode-only.

use std::ffi::c_void;

use crate::qsv_ffi::{MfxBitstream, MfxExtBuffer, MfxFrameSurface1, MfxVideoParam};

// ─── Local type aliases ───────────────────────────────────────────────────────
// These shadow the crate::qsv_ffi aliases inside the qsv module so every file
// here can write `MfxStatus` without a long path.
pub(super) type MfxStatus = i32;
/// Opaque oneVPL session handle.
pub(super) type MfxSession = *mut c_void;
pub(super) type MfxSyncPoint = *mut c_void;
/// Handle returned by `MFXLoad()`.
pub(super) type MfxLoader = *mut c_void;
/// Handle returned by `MFXCreateConfig()`.
pub(super) type MfxConfig = *mut c_void;

// ─── Status / warning codes ───────────────────────────────────────────────────
pub(super) const MFX_ERR_NONE: MfxStatus = 0;
/// `MFX_ERR_NOT_ENOUGH_BUFFER` — the supplied output bitstream has no room
/// for this frame. Recoverable: drain what's in flight (which empties the
/// buffer) and, if that isn't enough, enlarge it.
pub(super) const MFX_ERR_NOT_ENOUGH_BUFFER: MfxStatus = -5;

pub(super) const MFX_ERR_MORE_DATA: MfxStatus = -10;
/// Decode-path only. Named so an encode `match` can distinguish it.
#[allow(dead_code)]
pub(super) const MFX_ERR_MORE_SURFACE: MfxStatus = -11;
pub(super) const MFX_WRN_IN_EXECUTION: MfxStatus = 1;
/// `MFX_WRN_DEVICE_BUSY` — the hardware cannot take the submission yet. The
/// documented remedy is to wait and repeat the *same* call; it is not an error.
pub(super) const MFX_WRN_DEVICE_BUSY: MfxStatus = 2;
pub(super) const MFX_WRN_INCOMPATIBLE_VIDEO_PARAM: MfxStatus = 5;
pub(super) const MFX_WRN_VIDEO_PARAM_CHANGED: MfxStatus = 3;
pub(super) const MFX_WRN_PARTIAL_ACCELERATION: MfxStatus = 4;

// ─── Codec / format / IO constants ───────────────────────────────────────────
// Four-character codec codes (little-endian u32).
pub(super) const MFX_CODEC_AV1: u32  = 0x20315641; // 'A','V','1',' '
pub(super) const MFX_CODEC_AVC: u32  = 0x20435641; // 'A','V','C',' ' (H.264)
pub(super) const MFX_CODEC_HEVC: u32 = 0x43564548; // 'H','E','V','C' (H.265)
pub(super) const MFX_FOURCC_NV12: u32 = 0x3231564e; // 'N','V','1','2'
/// Microsoft P010 surface FourCC — 16-bit per sample, valid 10 bits in the
/// upper 10 bits (`sample_10bit << 6`). Same plane geometry as NV12.
/// vendor/intel/mfxdefs.h:71.
pub(super) const MFX_FOURCC_P010: u32 = 0x30313050; // 'P','0','1','0'
pub(super) const MFX_CHROMAFORMAT_YUV420: u16 = 1;
pub(super) const MFX_IOPATTERN_IN_SYSTEM_MEMORY: u16 = 0x02;
pub(super) const MFX_PICSTRUCT_PROGRESSIVE: u16 = 1;
// Frame-type flags on mfxBitstream. vendor/intel/mfxstructs.h:185-188.
pub(super) const MFX_FRAMETYPE_I: u16   = 0x0001;
/// `MFX_FRAMETYPE_REF` — the frame is a reference. Paired with `I`+`IDR` when
/// forcing a random-access point that later frames may predict from.
pub(super) const MFX_FRAMETYPE_REF: u16 = 0x0040;
pub(super) const MFX_FRAMETYPE_IDR: u16 = 0x8000;

// ─── Rate-control mode constants ──────────────────────────────────────────────
// Values from vendor/intel/mfxdefs.h:73-84.
// NB: 8 is MFX_RATECONTROL_LA (lookahead), 9 is ICQ — the original value (8)
// was wrong and made AV1/Arc reject Query with MFX_ERR_UNSUPPORTED.
pub(super) const MFX_RATECONTROL_CQP: u16 = 3;
pub(super) const MFX_RATECONTROL_ICQ: u16 = 9;

// ─── Codec-profile constants ───────────────────────────────────────────────────
// AV1 profile (MAIN = 1 per vendor/intel/mfxav1.h:24, 0 = "auto").
pub(super) const MFX_PROFILE_AV1_MAIN: u16 = 1;
// H.264 / H.265 profiles (vendor/intel/mfxstructures.h).
pub(super) const MFX_PROFILE_AVC_HIGH: u16 = 100;
pub(super) const MFX_PROFILE_HEVC_MAIN: u16 = 1;
pub(super) const MFX_PROFILE_HEVC_MAIN10: u16 = 2;

// ─── Ext-buffer FOURCC identifiers ────────────────────────────────────────────
// FOURCC for AV1-specific ext buffers. vendor/intel/mfxstructs.h:128-129.
pub(super) const MFX_EXTBUFF_AV1_TILE_PARAM: u32 = 0x4c543141; // MFX_MAKEFOURCC(A,1,T,L)
#[allow(dead_code)]
pub(super) const MFX_EXTBUFF_AV1_BITSTREAM_PARAM: u32 = 0x42315641; // 'A','V','1','B' LE-u32.
/// `mfxExtCodingOption3` buffer id.
pub(super) const MFX_EXTBUFF_CODING_OPTION3: u32 = 0x334f4443; // MFX_MAKEFOURCC(C,D,O,3)
/// `mfxExtVideoSignalInfo` buffer id.
pub(super) const MFX_EXTBUFF_VIDEO_SIGNAL_INFO: u32 = 0x4e495356; // 'V','S','I','N' LE-u32.
/// Per oneVPL: `TargetChromaFormatPlus1 = MFX_CHROMAFORMAT_YUV420 + 1 = 2` for AV1 4:2:0.
pub(super) const MFX_TARGET_CHROMAFORMAT_YUV420_PLUS1: u16 = 2;

// ─── Encoder-only ext-buffer structs ─────────────────────────────────────────

/// oneVPL `mfxExtAV1TileParam` — 136 bytes per vendor/intel/mfxstructs.h:135-141.
/// Header (8) + 3 × u16 (6) + reserved[61] (122) = 136.
/// Our Rust mirror only carries the fields we set; `buffer_sz` is set to
/// `size_of::<Self>()` = 24 bytes (the driver reads only up to that).
#[repr(C)]
pub(super) struct MfxExtAv1TileParam {
    pub(super) header: MfxExtBuffer,
    pub(super) num_tile_rows: u16,
    pub(super) num_tile_columns: u16,
    pub(super) num_tile_groups: u16,
    pub(super) reserved: [u16; 5],
}

/// oneVPL `mfxExtCodingOption3` — **512 bytes**, offsetof-verified on oneVPL 2.16.
/// The only fields we set for 10-bit AV1 are `TargetChromaFormatPlus1` @158,
/// `TargetBitDepthLuma` @160, `TargetBitDepthChroma` @162.  The earlier layout
/// put them at @58/60/62 (after 3 NumRef arrays) — wrong by 100 bytes.
#[repr(C)]
pub(super) struct MfxExtCodingOption3 {
    pub(super) header: MfxExtBuffer,             // @0 (8 bytes)
    pub(super) _pad_to_158: [u8; 150],          // @8 → @158
    pub(super) target_chroma_format_plus1: u16, // @158
    pub(super) target_bit_depth_luma: u16,      // @160
    pub(super) target_bit_depth_chroma: u16,    // @162
    pub(super) _tail: [u8; 348],                // @164 → @512
}

/// oneVPL `mfxExtVideoSignalInfo` — H.273 colour signalling carried into the
/// AV1 OBU sequence header `color_config`. 8-byte header + 6×u16 named = 20
/// bytes; mirrors the 24-byte public layout (runtime reads only named fields).
#[repr(C)]
pub(super) struct MfxExtVideoSignalInfo {
    pub(super) header: MfxExtBuffer,
    pub(super) video_format: u16,               /* 5 = unspecified */
    pub(super) video_full_range: u16,           /* 0 = studio, 1 = full */
    pub(super) colour_description_present: u16, /* 1 = next 3 fields valid */
    pub(super) colour_primaries: u16,           /* H.273 §8.1 */
    pub(super) transfer_characteristics: u16,   /* H.273 §8.2 */
    pub(super) matrix_coefficients: u16,        /* H.273 §8.3 */
}

/// oneVPL `mfxEncodeCtrl` — per-frame encode control, the second argument to
/// `MFXVideoENCODE_EncodeFrameAsync`. Passing null (the long-standing default
/// here) lets the encoder decide everything, including IDR placement; supplying
/// one is how a specific frame is forced to be a random-access point.
///
/// Field order is load-bearing and follows `vendor/intel/mfxstructs.h`:
///
/// ```text
///   mfxExtBuffer Header;
///   mfxU32       reserved[4];
///   mfxU16       reserved1;
///   mfxU16       MfxNalUnitType;
///   mfxU16       SkipFrame;
///   mfxU16       QP;
///   mfxU16       FrameType;
///   mfxU16       NumExtParam;
///   mfxU16       NumPayload;
///   mfxU16       reserved2;
///   mfxExtBuffer **ExtParam;
///   mfxPayload   **Payload;
/// ```
///
/// The previous definition of this struct was the right *size* but its names
/// were shifted one `mfxU16` slot from the real layout — writing `frame_type`
/// would have landed in `QP`. It was never exercised (the pointer was always
/// null), so the mistake stayed invisible until this became a live path.
#[repr(C)]
pub(super) struct MfxEncodeCtrl {
    pub(super) header: MfxExtBuffer,
    pub(super) reserved: [u32; 4],
    pub(super) reserved1: u16,
    pub(super) mfx_nal_unit_type: u16,
    pub(super) skip_frame: u16,
    pub(super) qp: u16,
    pub(super) frame_type: u16,
    pub(super) num_ext_param: u16,
    pub(super) num_payload: u16,
    pub(super) reserved2: u16,
    pub(super) ext_param: *mut *mut MfxExtBuffer,
    pub(super) payload: *mut c_void,
}

// 8-byte header + 16 reserved + 8x u16 + two pointers = 56 bytes on 64-bit.
const _: () = assert!(std::mem::size_of::<MfxEncodeCtrl>() == 56);

impl MfxEncodeCtrl {
    /// A control that forces this frame to be an IDR — a self-contained
    /// random-access point later frames may predict from.
    pub(super) fn force_idr() -> Self {
        Self {
            header: MfxExtBuffer { buffer_id: 0, buffer_sz: 0 },
            reserved: [0; 4],
            reserved1: 0,
            mfx_nal_unit_type: 0,
            skip_frame: 0,
            qp: 0, // 0 = "use the configured rate control", not "QP 0"
            frame_type: MFX_FRAMETYPE_I | MFX_FRAMETYPE_IDR | MFX_FRAMETYPE_REF,
            num_ext_param: 0,
            num_payload: 0,
            reserved2: 0,
            ext_param: std::ptr::null_mut(),
            payload: std::ptr::null_mut(),
        }
    }
}

// ─── Function-pointer types ────────────────────────────────────────────────────

pub(super) type FnMfxClose = unsafe extern "C" fn(MfxSession) -> MfxStatus;

pub(super) type FnMfxLoad   = unsafe extern "C" fn() -> MfxLoader;
pub(super) type FnMfxUnload = unsafe extern "C" fn(MfxLoader);
pub(super) type FnMfxCreateConfig = unsafe extern "C" fn(MfxLoader) -> MfxConfig;
/// `MFXInit` — the pre-dispatcher entry point. Takes an implementation mask
/// rather than an adapter index, so it cannot pin a card; kept for hosts whose
/// dispatcher enumerates nothing. See `QsvEncoder::new_unpinned`.
pub(super) type FnMfxInit =
    unsafe extern "C" fn(u32, *mut crate::qsv_ffi::MfxVersion, *mut MfxSession) -> MfxStatus;

/// `MFX_IMPL_HARDWARE_ANY` — any hardware implementation the runtime has.
pub(super) const MFX_IMPL_HARDWARE_ANY: u32 = 0x0100;

pub(super) type FnMfxSetConfigFilterProperty =
    unsafe extern "C" fn(MfxConfig, *const u8, MfxVariant) -> MfxStatus;
pub(super) type FnMfxCreateSession =
    unsafe extern "C" fn(MfxLoader, u32, *mut MfxSession) -> MfxStatus;

pub(super) type FnEncodeQuery =
    unsafe extern "C" fn(MfxSession, *mut MfxVideoParam, *mut MfxVideoParam) -> MfxStatus;
pub(super) type FnEncodeInit =
    unsafe extern "C" fn(MfxSession, *mut MfxVideoParam) -> MfxStatus;
pub(super) type FnEncodeClose = unsafe extern "C" fn(MfxSession) -> MfxStatus;
/// `MFXVideoENCODE_Reset(session, par)` — stop the current stream and start a
/// new one on the same session with `par`. Handed the *same* `mfxVideoParam`
/// the session was initialised with, it is the documented way to restart the
/// GOP and the rate control without `Close` + `Init`: the next frame is an
/// IDR and a new sequence begins. `Encoder::reset` is built on it.
pub(super) type FnEncodeReset =
    unsafe extern "C" fn(MfxSession, *mut MfxVideoParam) -> MfxStatus;
pub(super) type FnEncodeFrameAsync = unsafe extern "C" fn(
    MfxSession,
    *mut c_void,
    *mut MfxFrameSurface1,
    *mut MfxBitstream,
    *mut MfxSyncPoint,
) -> MfxStatus;
pub(super) type FnSyncOperation =
    unsafe extern "C" fn(MfxSession, MfxSyncPoint, u32) -> MfxStatus;

// ─── oneVPL dispatcher variant ────────────────────────────────────────────────

/// mfxVariant — Version(u16) + pad + Type(u32) + Data(union, 8B). 16 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct MfxVariant {
    pub(super) version: u16,
    pub(super) _pad: u16,
    pub(super) ty: u32,
    pub(super) data: u64, // union; write the U32 value into the low 4 bytes (LE)
}
const _: () = assert!(std::mem::size_of::<MfxVariant>() == 16);
pub(super) const MFX_VARIANT_TYPE_U32: u32 = 5;
pub(super) const MFX_IMPL_TYPE_HARDWARE: u32 = 2;

// ─── Compile-time struct-size assertions ──────────────────────────────────────
// Catches ABI drift — if a future pad-edit accidentally changes a struct size,
// the const_assert fires at compile time rather than letting a silent
// offset-shift produce corrupt encodes on real hardware.

// mfxExtAV1TileParam — our Rust mirror spans the used fields only → 24 bytes.
const _: () = assert!(std::mem::size_of::<MfxExtAv1TileParam>() == 24);

// mfxEncodeCtrl — 56 bytes (offsetof-verified). Passed as NULL anyway.
const _: () = assert!(std::mem::size_of::<MfxEncodeCtrl>() == 56);

// mfxExtCodingOption3 — 512 bytes (oneVPL 2.16 layout).
const _: () = assert!(std::mem::size_of::<MfxExtCodingOption3>() == 512);

// mfxExtVideoSignalInfo — at least 20 bytes (8-byte header + 6×u16 named).
const _: () = assert!(std::mem::size_of::<MfxExtVideoSignalInfo>() >= 20);

// FOURCC pin: P010 = 0x30313050, NV12 = 0x3231564e.
const _: () = assert!(MFX_FOURCC_P010 == 0x30313050);
const _: () = assert!(MFX_FOURCC_NV12 == 0x3231564e);

// Ext buffer IDs — pinned so a future SDK rename fails compilation.
const _: () = assert!(MFX_EXTBUFF_CODING_OPTION3 == 0x334f4443);
const _: () = assert!(MFX_EXTBUFF_VIDEO_SIGNAL_INFO == 0x4e495356);
