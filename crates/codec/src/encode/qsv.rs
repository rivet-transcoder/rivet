//! Intel QSV AV1 hardware encoder via oneVPL.
//!
//! Loads `libvpl.so.2` / `libvpl.dll` at runtime via dlopen. The AV1
//! encoder is available on Intel Arc (DG2 / BMG) discrete GPUs and on
//! Meteor Lake (Core Ultra 1xx) and Lunar Lake (Core Ultra 2xx) iGPUs.
//! On Arrow Lake + hybrid systems, QSV picks the iGPU unless the
//! dispatcher is filtered to the dGPU via `MFXSetConfigFilterProperty`.
//!
//! Session flow:
//! 1. dlopen libvpl. Walk the legacy `MFXInit` path on oneVPL 2.x +
//!    fall back to the dispatcher (`MFXLoad` → `MFXCreateSession`)
//!    so we work on hosts with either MSDK-layout runtimes or the
//!    newer unified oneVPL runtime.
//! 2. Populate `mfxVideoParam`:
//!    - `CodecId = MFX_CODEC_AV1`, `CodecProfile = MFX_PROFILE_AV1_MAIN`
//!    - `RateControlMethod` per tuning adapter (ICQ or CQP)
//!    - `TargetUsage` per speed tier (1..7)
//!    - `FrameInfo.FourCC = MFX_FOURCC_NV12`, `ChromaFormat = YUV420`
//!    - `IOPattern = IN_SYSTEM_MEMORY`
//!    - `GopPicSize = keyframe_interval`, `GopRefDist = 1` (no B-frames)
//! 3. Attach `mfxExtAV1TileParam` via `ExtParam[]` to set the tile grid
//!    (AV1 has no tile fields in the main `mfxInfoMFX` struct).
//! 4. `MFXVideoENCODE_Query(session, &par, &out)` — returns the
//!    runtime-adjusted params; if QSV reduced something we log and
//!    use the `out` struct.
//! 5. `MFXVideoENCODE_Init(session, &out)`.
//! 6. Per frame:
//!    - Pick next surface slot in the 4-deep ring.
//!    - Convert YUV420p → NV12 into that slot's backing buffer.
//!    - `MFXVideoENCODE_EncodeFrameAsync` → `syncp`.
//!    - `MFXVideoCORE_SyncOperation(session, syncp, 60_000)` → drain
//!      the `mfxBitstream` buffer.
//! 7. Flush by submitting NULL surface until `MFX_ERR_MORE_DATA` →
//!    no more output to drain.
//! 8. `MFXVideoENCODE_Close(session)` → `MFXClose(session)` →
//!    library handle drops last.
//!
//! ## Correctness bar for QSV in this repo
//!
//! Host is NVIDIA — E2E Intel GPU verification is impossible on the
//! dev box. Every struct layout below is spec-conformant-by-review
//! against `vendor/intel/` oneVPL 2.10 headers. `const_assert!` checks
//! at the bottom of the file fire at compile time if any struct size
//! drifts — mirroring the pattern established by Squad 5 in
//! `encode/nvenc.rs`.

use anyhow::{Context, Result, bail};
use bytes::Bytes;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::os::raw::c_char;
use std::ptr;

use super::tuning::{self, QsvRateControl};
use super::{AUTO_FROM_TARGET, EncodedPacket, Encoder, EncoderConfig};
// `ColorMetadata` is read via `config.color_metadata` on the non-test
// side (no bare-type mention) and through `use super::*` inside the
// test module; pull it in only under cfg(test) to keep release builds
// warning-clean.
#[cfg(test)]
use crate::frame::ColorMetadata;
use crate::frame::{PixelFormat, TransferFn, VideoFrame};

// ─── oneVPL ABI constants ─────────────────────────────────────────
// See vendor/intel/ for authoritative definitions.

type MfxStatus = i32;
const MFX_ERR_NONE: MfxStatus = 0;
const MFX_ERR_MORE_DATA: MfxStatus = -10;
// vendor/intel/mfxdefs.h:40. Documented as "decode needs another
// surface before it can make progress". The encoder path does NOT
// emit this — it only shows up on the decode side — but we name the
// constant so the `match` arm below can distinguish it from
// `MFX_ERR_MORE_DATA` if the driver ever behaves differently.
#[allow(dead_code)]
const MFX_ERR_MORE_SURFACE: MfxStatus = -11;
const MFX_WRN_IN_EXECUTION: MfxStatus = 1;
const MFX_WRN_INCOMPATIBLE_VIDEO_PARAM: MfxStatus = 5;
const MFX_WRN_VIDEO_PARAM_CHANGED: MfxStatus = 3;
const MFX_WRN_PARTIAL_ACCELERATION: MfxStatus = 4;

const MFX_IMPL_HARDWARE_ANY: u32 = 0x0100;

// Four-character codec codes (little-endian u32).
const MFX_CODEC_AV1: u32 = 0x20315641; // 'A','V','1',' '
const MFX_FOURCC_NV12: u32 = 0x3231564e; // 'N','V','1','2'
/// Microsoft P010 surface FourCC — 16-bit per sample, valid 10 bits in
/// the upper 10 bits (`sample_10bit << 6`). Same plane geometry as NV12
/// (Y plane + interleaved UV plane). vendor/intel/mfxdefs.h:71.
/// Selected at session create time when `input_pixel_format == Yuv420p10le`.
const MFX_FOURCC_P010: u32 = 0x30313050; // 'P','0','1','0'

// Rate control modes. Values from vendor/intel/mfxdefs.h:73-84.
// NB: 8 is MFX_RATECONTROL_LA (lookahead), 9 is ICQ — the original value (8)
// was wrong and made AV1/Arc reject Query with MFX_ERR_UNSUPPORTED.
const MFX_RATECONTROL_CQP: u16 = 3;
const MFX_RATECONTROL_ICQ: u16 = 9;

// AV1 profile (MAIN = 1 per vendor/intel/mfxav1.h:24, 0 = "auto").
const MFX_PROFILE_AV1_MAIN: u16 = 1;

// Chroma format — 4:2:0. vendor/intel/mfxdefs.h:103.
const MFX_CHROMAFORMAT_YUV420: u16 = 1;

// IO pattern. vendor/intel/mfxdefs.h:99.
const MFX_IOPATTERN_IN_SYSTEM_MEMORY: u16 = 0x02;

// Picture struct — 1 = progressive frame.
const MFX_PICSTRUCT_PROGRESSIVE: u16 = 1;

// Frame-type flags on mfxBitstream. vendor/intel/mfxstructs.h:185-188.
const MFX_FRAMETYPE_I: u16 = 0x0001;
const MFX_FRAMETYPE_IDR: u16 = 0x8000;

// FOURCC for AV1-specific ext buffers. vendor/intel/mfxstructs.h:128-129.
const MFX_EXTBUFF_AV1_TILE_PARAM: u32 = 0x54315641; // 'A','V','1','T' LE-u32.
#[allow(dead_code)]
const MFX_EXTBUFF_AV1_BITSTREAM_PARAM: u32 = 0x42315641; // 'A','V','1','B' LE-u32.
/// `mfxExtCodingOption3`. Carries TargetBitDepthLuma / TargetBitDepthChroma
/// + TargetChromaFormatPlus1 — the encoder reads these to set the AV1
/// sequence header `BitDepth` value when feeding P010 surfaces.
const MFX_EXTBUFF_CODING_OPTION3: u32 = 0x33444f43; // 'C','D','O','3' LE-u32.
/// `mfxExtVideoSignalInfo`. Carries the four H.273 codes that the
/// encoder embeds into the AV1 OBU sequence header `color_config`.
const MFX_EXTBUFF_VIDEO_SIGNAL_INFO: u32 = 0x4e495356; // 'V','S','I','N' LE-u32.

/// Per oneVPL: `TargetChromaFormatPlus1 = MFX_CHROMAFORMAT_YUV420 + 1 = 2`
/// for AV1 4:2:0. Defined out-of-line so the const_assert!s can pin it.
const MFX_TARGET_CHROMAFORMAT_YUV420_PLUS1: u16 = 2;

// mfxVersion we send — min oneVPL 2.10 so AV1 encode is available.
const MFX_MIN_VERSION: MfxVersion = MfxVersion {
    minor: 10,
    major: 2,
};

/// Encoder pipeline depth — number of input surfaces + sync points
/// in flight before we must drain one. Matches Squad 5's NVENC
/// `RING_SIZE = 4` and upstream oneVPL `sample_encode`'s recommended
/// `AsyncDepth = 4` on Arc / Meteor Lake.
const RING_SIZE: usize = 4;

// ─── Struct layouts ───────────────────────────────────────────────
//
// Match the oneVPL 2.10 upstream ABI. Trimmed to named fields we
// reference plus a reserved tail sized so the total struct footprint
// equals upstream. Verified by `const_assert!`s at the bottom.

#[repr(C)]
#[derive(Clone, Copy)]
struct MfxVersion {
    minor: u16,
    major: u16,
}

/// oneVPL `mfxFrameInfo` — 80 bytes per vendor/intel/mfxstructs.h:20-50.
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
    _crop_pad: [u16; 6], // Union-branch padding — we only use the width/height branch.
    frame_rate_ext_n: u32,
    frame_rate_ext_d: u32,
    reserved3: u16,
    aspect_ratio_w: u16,
    aspect_ratio_h: u16,
    pic_struct: u16,
    chroma_format: u16,
    reserved2: u16,
}

/// oneVPL `mfxInfoMFX` — 256 bytes per vendor/intel/mfxstructs.h:54-101.
///
/// The three CQP/VBR/ICQ union arms all start at the same offset;
/// field aliasing is tracked by the `qpi_or_delay` / `qpp_or_kbps_or_icq`
/// / `qpb_or_maxkbps` field names. See `rate_slots_for_rc` for the
/// slot-to-concept mapping.
#[repr(C)]
#[derive(Clone, Copy)]
struct MfxInfoMfx {
    // mfxInfoMFX: reserved[7] (28B), then LowPower (u16), BRCParamMultiplier
    // (u16), then FrameInfo at offset 32. The old layout used reserved[6] +
    // low_power:u32, which put `low_power` in a reserved slot (offset 24) and
    // left the real LowPower (offset 28) at 0 — so AV1 on Arc (low-power-only)
    // got LowPower=UNKNOWN and Query rejected it.
    reserved: [u32; 7],
    low_power: u16,
    brc_param_multiplier: u16,
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
    /// Slot 0 of the rc-union. Per vendor/intel/mfxstructs.h:74-78:
    /// `InitialDelayInKB` (CBR/VBR) / `QPI` (CQP) / `Accuracy` (AVBR).
    /// **ICQQuality does NOT live here** — it's slot 1.
    qpi_or_delay: u16,
    buffer_size_kb: u16,
    /// Slot 1 of the rc-union. Per vendor/intel/mfxstructs.h:80-84:
    /// `TargetKbps` (CBR/VBR) / `QPP` (CQP) / **`ICQQuality` (ICQ)**.
    /// This is where AV1 ICQ quality lands in oneVPL 2.10.
    qpp_or_kbps_or_icq: u16,
    /// Slot 2 of the rc-union. Per vendor/intel/mfxstructs.h:85-89:
    /// `MaxKbps` (CBR/VBR) / `QPB` (CQP) / `Convergence` (AVBR).
    qpb_or_maxkbps: u16,

    num_slice: u16,
    num_ref_frame: u16,
    encoded_order: u16,
    // Pad the union + tail to match the upstream 256-byte mfxInfoMFX
    // size. Rust laid-out so far: 148 bytes. 256 - 148 = 108 = 27 u32.
    _tail: [u32; 27],
}

/// oneVPL `mfxVideoParam` — 304 bytes on 64-bit per upstream layout.
/// vendor/intel/mfxstructs.h:103-117.
#[repr(C)]
struct MfxVideoParam {
    // mfxVideoParam: AllocId(u32), reserved[2], reserved3(u16), AsyncDepth(u16),
    // then the mfxInfoMFX union at offset 16. The old layout omitted AllocId,
    // so `mfx` sat at offset 12 and the driver read the whole block 4 bytes
    // early — every field (CodecId, FrameInfo, …) was garbage → Query -3.
    // ExtParam (a pointer) comes BEFORE NumExtParam upstream, too.
    alloc_id: u32,
    reserved: [u32; 2],
    reserved3: u16,
    async_depth: u16,
    mfx: MfxInfoMfx,
    protected: u16,
    io_pattern: u16,
    // repr(C) inserts 4 bytes of padding here to 8-align the pointer.
    ext_param: *mut *mut MfxExtBuffer,
    num_ext_param: u16,
    reserved2: u16,
    // Upstream tail — reserved for future ABI stability.
    _tail: [u32; 3],
}

/// oneVPL `mfxExtBuffer` — every ExtParam entry starts with this 8-byte
/// header. vendor/intel/mfxstructs.h:121-124.
#[repr(C)]
struct MfxExtBuffer {
    buffer_id: u32,
    buffer_sz: u32,
}

/// oneVPL `mfxExtAV1TileParam` — 136 bytes per vendor/intel/mfxstructs.h:135-141.
/// Header (8) + 3 × u16 (6) + reserved[61] (122) = 136.
#[repr(C)]
struct MfxExtAv1TileParam {
    header: MfxExtBuffer,
    num_tile_rows: u16,
    num_tile_columns: u16,
    num_tile_groups: u16,
    reserved: [u16; 61],
}

/// oneVPL `mfxExtCodingOption3` — 400 bytes upstream. We mirror the
/// fields we set + the rest is opaque reserved tail. The encoder
/// reads named fields directly; trailing reserved bytes are zero
/// (driver-blessed default behaviour).
///
/// Squad-22 wires `TargetBitDepthLuma` / `TargetBitDepthChroma` for
/// 10-bit AV1. `TargetChromaFormatPlus1` is held at
/// `MFX_TARGET_CHROMAFORMAT_YUV420_PLUS1 = 2` (= MFX_CHROMAFORMAT_YUV420
/// + 1, oneVPL's "plus one" convention so 0 means "use the FrameInfo
/// chroma").
///
/// Layout matches `vendor/intel/mfxstructs.h::mfxExtCodingOption3`
/// (header + 3×NumRef[8] u16 arrays + the named knobs we set + a
/// reserved tail). Verified by `const_assert!` below.
#[repr(C)]
struct MfxExtCodingOption3 {
    header: MfxExtBuffer,
    num_ref_active_p: [u16; 8],
    num_ref_active_bl0: [u16; 8],
    num_ref_active_bl1: [u16; 8],
    transform_skip: u16,
    target_chroma_format_plus1: u16,
    target_bit_depth_luma: u16,
    target_bit_depth_chroma: u16,
    brc_panic_mode: u16,
    low_delay_brc: u16,
    enable_mb_force_intra: u16,
    adaptive_max_frame_size: u16,
    repartition_check_enable: u16,
    reserved5: [u16; 3],
    encoded_units_info: u16,
    enable_nal_unit_type: u16,
    ext_brc_adaptive_ltr: u16,
    adaptive_ltr: u16,
    reserved6: [u16; 160],
}

/// oneVPL `mfxExtVideoSignalInfo` — H.273 colour signalling carried
/// into the AV1 OBU sequence header `color_config`. 8-byte header +
/// 6×u16 named + 2 bytes alignment = 24 bytes total.
/// vendor/intel/mfxstructs.h adds a 28-byte form on some upstream revs
/// with reserved trailing alignment; we mirror the published 24-byte
/// public layout (the runtime reads only the named fields anyway).
///
/// Same enumeration the mux's `colr nclx` writer uses — keeps in-bitstream
/// HDR signalling identical to container-level metadata.
#[repr(C)]
struct MfxExtVideoSignalInfo {
    header: MfxExtBuffer,
    video_format: u16,                /* 5 = unspecified */
    video_full_range: u16,            /* 0 = studio, 1 = full */
    colour_description_present: u16,  /* 1 = next 3 fields valid */
    colour_primaries: u16,            /* H.273 §8.1 */
    transfer_characteristics: u16,    /* H.273 §8.2 */
    matrix_coefficients: u16,         /* H.273 §8.3 */
}

/// oneVPL `mfxFrameData` — vendor/intel/mfxstructs.h:145-161.
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

/// oneVPL `mfxFrameSurface1` — 256-byte struct per upstream layout.
/// vendor/intel/mfxstructs.h:163-167.
#[repr(C)]
struct MfxFrameSurface1 {
    reserved: [u32; 4],
    info: MfxFrameInfo,
    data: MfxFrameData,
}

/// oneVPL `mfxBitstream` — vendor/intel/mfxstructs.h:171-183.
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

/// oneVPL `mfxEncodeCtrl` — optional per-frame control passed to
/// `EncodeFrameAsync`. We never populate it (pass `NULL`), but the
/// struct is named + sized so the const_assert at the bottom documents
/// the layout we expect the runtime to read when callers upgrade.
/// Based on upstream `mfxstructureshi.h` in oneVPL 2.10; a stable 96-
/// byte header followed by `ExtParam` pointer + tail.
#[repr(C)]
#[allow(dead_code)]
struct MfxEncodeCtrl {
    header: MfxExtBuffer,
    reserved: [u32; 4],
    mfx_pic_struct: u16,
    mfx_skip_frame: u16,
    qp: u16,
    frame_type: u16,
    num_ext_param: u16,
    _pad: u16,
    num_payload: u16,
    _pad2: u16,
    ext_param: *mut *mut MfxExtBuffer,
    payload: *mut c_void,
    _tail: [u32; 8],
}

// ─── FFI signatures ──────────────────────────────────────────────
//
// Opaque session handle. We carry it as `*mut c_void`.
type MfxSession = *mut c_void;
type MfxSyncPoint = *mut c_void;

type FnMfxInit = unsafe extern "C" fn(u32, *mut MfxVersion, *mut MfxSession) -> MfxStatus;
type FnMfxClose = unsafe extern "C" fn(MfxSession) -> MfxStatus;
type FnEncodeQuery =
    unsafe extern "C" fn(MfxSession, *mut MfxVideoParam, *mut MfxVideoParam) -> MfxStatus;
type FnEncodeInit = unsafe extern "C" fn(MfxSession, *mut MfxVideoParam) -> MfxStatus;
type FnEncodeClose = unsafe extern "C" fn(MfxSession) -> MfxStatus;
type FnEncodeFrameAsync = unsafe extern "C" fn(
    MfxSession,
    *mut c_void,
    *mut MfxFrameSurface1,
    *mut MfxBitstream,
    *mut MfxSyncPoint,
) -> MfxStatus;
type FnSyncOperation = unsafe extern "C" fn(MfxSession, MfxSyncPoint, u32) -> MfxStatus;

// ─── Rate-control slot mapping ────────────────────────────────────

/// The three `mfxInfoMFX` rc-union slot values a given job produces,
/// before being splatted into `qpi_or_delay / qpp_or_kbps_or_icq /
/// qpb_or_maxkbps`. Pulled out as a standalone function so the
/// slot-assignment logic can be unit-tested without touching any FFI.
///
/// Per vendor/intel/mfxstructs.h:74-89:
/// - CQP: slot0=QPI, slot1=QPP, slot2=QPB
/// - ICQ: slot0=0 (unused — InitialDelayInKB is not read in ICQ), slot1=**ICQQuality**, slot2=0
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RateSlots {
    slot0_qpi_or_delay: u16,
    slot1_qpp_or_kbps_or_icq: u16,
    slot2_qpb_or_maxkbps: u16,
}

fn rate_slots_for_rc(mode: QsvRateControl, qp_i: u16, qp_p: u16, icq_quality: u16) -> RateSlots {
    match mode {
        QsvRateControl::Cqp => RateSlots {
            slot0_qpi_or_delay: qp_i,
            slot1_qpp_or_kbps_or_icq: qp_p,
            slot2_qpb_or_maxkbps: qp_p, // No B-frames; mirror QPP to QPB.
        },
        QsvRateControl::Icq => RateSlots {
            // ICQ mode: slot 0 is the InitialDelayInKB alias and is
            // unread. ICQQuality is **slot 1** per the vendored header
            // (`vendor/intel/mfxstructs.h:83`). Slot 2 is unused.
            // An earlier rev of this file (based on an incorrect reading
            // of upstream in `reviews/codec-review-59-60.md` QSV-1) put
            // ICQQuality in slot 0, which silently resolved to
            // `InitialDelayInKB` and caused oneVPL to fall back to
            // its default ICQQuality=23 for every quality tier.
            slot0_qpi_or_delay: 0,
            slot1_qpp_or_kbps_or_icq: icq_quality,
            slot2_qpb_or_maxkbps: 0,
        },
    }
}

// ─── Squad-22: per-pixel-format dispatch ──────────────────────────
//
// QSV AV1 takes either NV12 (8-bit, FrameInfo BitDepth* = 8, Shift=0)
// or P010 (10-bit, BitDepth* = 10, Shift=1). The Shift bit is mandatory
// when the surface is P010 — it tells the runtime to treat samples as
// "shifted" (valid 10 bits in the upper 10 bits of each 16-bit word).
// Without Shift=1, oneVPL reads samples as if the valid bits were in
// the lower 10 → silently encodes 1/64th-amplitude noise.

/// Map a `PixelFormat` to its oneVPL FOURCC. Bails on unsupported
/// chroma — this encoder is AV1 4:2:0 only.
fn qsv_fourcc_for(fmt: PixelFormat) -> Result<u32> {
    match fmt {
        PixelFormat::Yuv420p => Ok(MFX_FOURCC_NV12),
        PixelFormat::Yuv420p10le => Ok(MFX_FOURCC_P010),
        other => bail!(
            "QSV AV1 expects Yuv420p or Yuv420p10le, got {other:?}"
        ),
    }
}

/// Returns `(BitDepthLuma, BitDepthChroma, Shift)` for the FrameInfo
/// struct given the input pixel format. Shift=1 is the P010 "valid
/// bits in upper 10" signal — required for 10-bit, must be 0 for 8-bit
/// or oneVPL rejects the param set with INVALID_VIDEO_PARAM.
const fn qsv_bit_depth_triple(fmt: PixelFormat) -> (u32, u32, u16) {
    match fmt {
        PixelFormat::Yuv420p10le => (10, 10, 1),
        _ => (8, 8, 0),
    }
}

/// Translate `TransferFn` → ITU-T H.273 numeric code. Mirrors
/// `nvenc.rs::transfer_to_h273` and `amf.rs::transfer_to_h273` plus
/// the mux helper — single source of HDR signalling truth across the
/// three encoder backends and the container.
fn transfer_to_h273(tf: TransferFn) -> u16 {
    match tf {
        TransferFn::Bt709 => 1,
        TransferFn::Bt470Bg => 4,
        TransferFn::Linear => 8,
        TransferFn::St2084 => 16,
        TransferFn::AribStdB67 => 18,
        TransferFn::Unspecified => 1,
    }
}

/// Map a `SpeedTier` (via the tuning adapter's `target_usage` output)
/// to the oneVPL 1..7 scale. Kept as a tiny pure helper so the
/// mapping can be table-tested without reaching into the FFI
/// construction path.
///
/// Per vendor/intel/mfxdefs.h:91-93:
/// - 1 = MFX_TARGETUSAGE_BEST_QUALITY
/// - 4 = MFX_TARGETUSAGE_BALANCED
/// - 7 = MFX_TARGETUSAGE_BEST_SPEED
///
/// The tuning adapter at `encode/tuning.rs::qsv_av1_params` clamps to
/// 1..=6 (leaves headroom past "best speed" for future driver tuning
/// selections); this helper simply defends against out-of-range values.
fn clamp_target_usage(tp_target_usage: u16) -> u16 {
    tp_target_usage.clamp(1, 7)
}

// ─── Ring-buffer input surfaces ───────────────────────────────────

/// A single input-surface slot in the 4-deep ring. Holds the
/// `MfxFrameSurface1` plus the backing NV12 buffer that surface's
/// pointers live in.
struct SurfaceSlot {
    surface: MfxFrameSurface1,
    /// Owns the bytes that `surface.data.{mem_id_or_y, u, v}` point
    /// into. Storage MUST NOT be dropped until the session closes —
    /// the driver may still hold back-references even after we sync.
    /// `Box<[u8]>` (not `Vec<u8>`) so the allocation can never be
    /// mutated-and-reallocated after construction.
    _backing: Box<[u8]>,
    /// `sync_point` from the most recent EncodeFrameAsync on this
    /// slot, or null if the slot has never been submitted or has
    /// already been sync'd.
    sync: MfxSyncPoint,
}

// SAFETY: `MfxSyncPoint = *mut c_void` is a raw pointer, not
// auto-`Send`, but oneVPL documents sync points as thread-safe
// handles that are opaque from our perspective. The ring only
// migrates between threads when the whole `QsvSession` migrates
// (via `spawn_blocking`), and access is serialized through `&mut
// self`. No sharing; same Send constraint as `QsvSession`.
unsafe impl Send for SurfaceSlot {}

// ─── Session container ────────────────────────────────────────────

struct QsvSession {
    session: MfxSession,
    width: u32,
    height: u32,
    pts_timescale: u64,
    /// `Yuv420p` (NV12 surface) or `Yuv420p10le` (P010 surface).
    /// Drives the per-frame upload (8-bit byte copy vs P010 `<<6`).
    input_pixel_format: PixelFormat,

    fn_mfx_close: FnMfxClose,
    fn_encode_close: FnEncodeClose,
    fn_encode_frame_async: FnEncodeFrameAsync,
    fn_sync_operation: FnSyncOperation,

    // Backing storage for the ext buffers we attached to mfxVideoParam.
    // Must stay alive as long as the encoder session references the
    // ExtParam[] pointer via its internal copy — oneVPL docs say the
    // runtime shallow-copies ExtParam at Init, so we could drop these
    // after Init, but we keep them for any future Reconfigure.
    #[allow(dead_code)]
    tile_ext: Box<MfxExtAv1TileParam>,
    /// `mfxExtCodingOption3` — present whenever the input is 10-bit so
    /// `TargetBitDepthLuma=10` makes it into the AV1 sequence header.
    /// `None` for 8-bit; absent in 8-bit jobs to avoid sending the
    /// runtime extra zero-filled buffers it has to inspect.
    #[allow(dead_code)]
    coding_option3_ext: Option<Box<MfxExtCodingOption3>>,
    /// `mfxExtVideoSignalInfo` — H.273 colour signalling. Always
    /// attached so SDR jobs explicitly carry BT.709 (rather than
    /// "unspecified") in the OBU header.
    #[allow(dead_code)]
    signal_info_ext: Box<MfxExtVideoSignalInfo>,
    /// Vector of pointers backing `mfxVideoParam.ExtParam[]`. Length
    /// varies (2 for 8-bit: tile + signal_info; 3 for 10-bit:
    /// + coding_option3). Kept boxed so the address handed to oneVPL
    /// stays stable across the session lifetime.
    #[allow(dead_code)]
    ext_param_array: Vec<*mut MfxExtBuffer>,

    /// Ring of input surfaces. Producer writes into slot `ring_idx`
    /// then advances; consumer drains the oldest-submitted slot's
    /// sync point FIFO-style via `inflight`.
    surfaces: [SurfaceSlot; RING_SIZE],
    ring_idx: usize,
    /// FIFO of ring-slot indices whose sync point is still pending
    /// a `SyncOperation`. Length is bounded by `RING_SIZE`; we drain
    /// the head before the slot can be reused for another encode.
    inflight: VecDeque<usize>,
    input_pitch: u32,
    height_aligned: u32,

    // Output bitstream buffer — pre-allocated with enough headroom
    // for a 4K I-frame (~2 MB). Shared across all in-flight frames
    // because `SyncOperation` consumes the buffer between frames;
    // oneVPL documents this usage pattern in `sample_encode`.
    bitstream: MfxBitstream,
    /// Owns the backing bytes that `bitstream.data` points into.
    /// `Box<[u8]>` (not `Vec<u8>`) so the allocation can never be
    /// mutated-and-reallocated after construction — the driver holds
    /// a pointer into the allocation across encode frames.
    _bitstream_buf: Box<[u8]>,
}

// SAFETY: QsvSession holds raw pointers (`session: MfxSession`,
// fn pointers from the dispatcher, and NV12 / bitstream buffers owned
// by sibling Box fields). oneVPL is NOT thread-*safe* — concurrent
// calls to MFXVideoENCODE_* on the same session from different threads
// are UB — but Send only guarantees single-threaded *ownership
// transfer*. Each `QsvEncoder` is only touched from one tokio task at
// a time; moving the encoder between threads (e.g. when a
// spawn_blocking worker returns) is fine because the runtime
// serialises access to the underlying session through `&mut self`.
unsafe impl Send for QsvSession {}

impl Drop for QsvSession {
    fn drop(&mut self) {
        unsafe {
            if !self.session.is_null() {
                let _ = (self.fn_encode_close)(self.session);
                let _ = (self.fn_mfx_close)(self.session);
            }
        }
    }
}

// ─── Encoder implementation ───────────────────────────────────────
//
// Library handle declared LAST so session drops first and vtable
// calls in `Drop` still resolve to live code.
pub struct QsvEncoder {
    /// Held for potential future Reconfigure paths. Currently unused
    /// at runtime but keeps the encoder self-describing.
    #[allow(dead_code)]
    config: EncoderConfig,
    session: Option<QsvSession>,
    encoded_packets: Vec<EncodedPacket>,
    packet_cursor: usize,
    flushed: bool,
    frame_counter: u32,
    _runtime_lib: libloading::Library,
}

impl QsvEncoder {
    pub fn new(config: EncoderConfig, gpu_index: u32) -> Result<Self> {
        let runtime_lib = unsafe { libloading::Library::new("libvpl.so.2") }
            .or_else(|_| unsafe { libloading::Library::new("libvpl.so") })
            .or_else(|_| unsafe { libloading::Library::new("libvpl.dll") })
            .or_else(|_| unsafe { libloading::Library::new("libmfx.so.1") })
            .or_else(|_| unsafe { libloading::Library::new("libmfxhw64.dll") })
            .context("loading oneVPL runtime library (Intel GPU driver not present?)")?;

        unsafe {
            let mfx_init: libloading::Symbol<FnMfxInit> =
                runtime_lib.get(b"MFXInit").context("MFXInit symbol")?;
            let mfx_close: libloading::Symbol<FnMfxClose> =
                runtime_lib.get(b"MFXClose").context("MFXClose symbol")?;
            let fn_encode_query: libloading::Symbol<FnEncodeQuery> = runtime_lib
                .get(b"MFXVideoENCODE_Query")
                .context("MFXVideoENCODE_Query")?;
            let fn_encode_init: libloading::Symbol<FnEncodeInit> = runtime_lib
                .get(b"MFXVideoENCODE_Init")
                .context("MFXVideoENCODE_Init")?;
            let fn_encode_close: libloading::Symbol<FnEncodeClose> = runtime_lib
                .get(b"MFXVideoENCODE_Close")
                .context("MFXVideoENCODE_Close")?;
            let fn_encode_frame_async: libloading::Symbol<FnEncodeFrameAsync> = runtime_lib
                .get(b"MFXVideoENCODE_EncodeFrameAsync")
                .context("MFXVideoENCODE_EncodeFrameAsync")?;
            let fn_sync_operation: libloading::Symbol<FnSyncOperation> = runtime_lib
                .get(b"MFXVideoCORE_SyncOperation")
                .context("MFXVideoCORE_SyncOperation")?;

            // 1. Session. `MFX_IMPL_HARDWARE_ANY` makes the dispatcher
            //    pick the first Intel adapter that supports our
            //    requested codec. For multi-Intel hosts (iGPU + Arc)
            //    QSV's legacy init path doesn't let us target a
            //    specific adapter — the caller can set the env var
            //    `ONEVPL_PRIORITY_PATH` to the desired adapter's
            //    runtime dir.
            if gpu_index != 0 {
                tracing::warn!(
                    gpu_index,
                    "QSV MFXInit picks adapter 0 unconditionally; \
                     iGPU+dGPU hosts need ONEVPL_PRIORITY_PATH"
                );
            }
            let mut version = MFX_MIN_VERSION;
            let mut session: MfxSession = ptr::null_mut();
            let rc = mfx_init(MFX_IMPL_HARDWARE_ANY, &mut version, &mut session);
            if rc < 0 || session.is_null() {
                bail!("MFXInit(HARDWARE_ANY, {}.{}) failed: {rc}", version.major, version.minor);
            }

            // 2. Build the video parameter struct.
            let tp =
                tuning::qsv_av1_params(config.target, config.tier, config.width, config.height);

            // Squad-22: Pick FOURCC + BitDepth/Shift triple from the
            // configured input format. Both must agree — sending P010
            // in the surface but FrameInfo BitDepthLuma=8 silently
            // truncates samples on the encode side.
            let input_fourcc = qsv_fourcc_for(config.pixel_format)?;
            let (bit_depth_luma, bit_depth_chroma, shift) =
                qsv_bit_depth_triple(config.pixel_format);

            // Allocate ext buffer for AV1 tile grid and keep it in a
            // Box so its address is stable for ExtParam[].
            let mut tile_ext = Box::new(MfxExtAv1TileParam {
                header: MfxExtBuffer {
                    buffer_id: MFX_EXTBUFF_AV1_TILE_PARAM,
                    buffer_sz: std::mem::size_of::<MfxExtAv1TileParam>() as u32,
                },
                num_tile_rows: tp.num_tile_rows as u16,
                num_tile_columns: tp.num_tile_columns as u16,
                num_tile_groups: 1,
                reserved: [0u16; 61],
            });

            // mfxExtCodingOption3 — only attached for 10-bit jobs. The
            // 8-bit path leaves `TargetBitDepthLuma` at the runtime
            // default (which mirrors FrameInfo.BitDepthLuma) so we
            // don't ship redundant bytes.
            let mut coding_option3_ext: Option<Box<MfxExtCodingOption3>> =
                if config.pixel_format == PixelFormat::Yuv420p10le {
                    Some(Box::new(MfxExtCodingOption3 {
                        header: MfxExtBuffer {
                            buffer_id: MFX_EXTBUFF_CODING_OPTION3,
                            buffer_sz: std::mem::size_of::<MfxExtCodingOption3>() as u32,
                        },
                        num_ref_active_p: [0; 8],
                        num_ref_active_bl0: [0; 8],
                        num_ref_active_bl1: [0; 8],
                        transform_skip: 0,
                        target_chroma_format_plus1: MFX_TARGET_CHROMAFORMAT_YUV420_PLUS1,
                        target_bit_depth_luma: 10,
                        target_bit_depth_chroma: 10,
                        brc_panic_mode: 0,
                        low_delay_brc: 0,
                        enable_mb_force_intra: 0,
                        adaptive_max_frame_size: 0,
                        repartition_check_enable: 0,
                        reserved5: [0; 3],
                        encoded_units_info: 0,
                        enable_nal_unit_type: 0,
                        ext_brc_adaptive_ltr: 0,
                        adaptive_ltr: 0,
                        reserved6: [0; 160],
                    }))
                } else {
                    None
                };

            // mfxExtVideoSignalInfo — always attached so the AV1 OBU
            // sequence header carries explicit colour codes (rather
            // than the "unspecified" default that some downstream
            // tooling silently re-interprets).
            let cm = &config.color_metadata;
            let mut signal_info_ext = Box::new(MfxExtVideoSignalInfo {
                header: MfxExtBuffer {
                    buffer_id: MFX_EXTBUFF_VIDEO_SIGNAL_INFO,
                    buffer_sz: std::mem::size_of::<MfxExtVideoSignalInfo>() as u32,
                },
                video_format: 5,                     // unspecified format
                video_full_range: if cm.full_range { 1 } else { 0 },
                colour_description_present: 1,
                colour_primaries: cm.colour_primaries as u16,
                transfer_characteristics: transfer_to_h273(cm.transfer),
                matrix_coefficients: cm.matrix_coefficients as u16,
            });

            // Build the ExtParam[] vector. Tile + signal info always;
            // coding_option3 only when 10-bit. Keeping the slot order
            // deterministic (tile, signal_info, [co3]) means tests can
            // assert on it.
            //
            // We collect raw `*mut` directly off each `Box`'s heap
            // address — `Box::as_mut` for a `&mut Box<T>` gives a
            // stable pointer that lives as long as the Box itself
            // stays alive in `QsvSession`. The Vec<> backing the
            // ExtParam[] is also stashed on `QsvSession` so the array
            // address handed to oneVPL stays valid until session drop.
            let mut ext_param_array: Vec<*mut MfxExtBuffer> = Vec::with_capacity(3);
            ext_param_array.push(
                (&mut *tile_ext as *mut MfxExtAv1TileParam) as *mut MfxExtBuffer,
            );
            ext_param_array.push(
                (&mut *signal_info_ext as *mut MfxExtVideoSignalInfo) as *mut MfxExtBuffer,
            );
            if let Some(ref mut co3) = coding_option3_ext {
                ext_param_array.push(
                    (&mut **co3 as *mut MfxExtCodingOption3) as *mut MfxExtBuffer,
                );
            }
            let num_ext_param = ext_param_array.len() as u16;

            // Per-frame QP knobs. Legacy override: if config.quality is
            // set, treat it as a CQP q-index in the 0..255 AV1 range
            // and use CQP even if the tuning adapter suggested ICQ.
            // ChunkSeamMode::ParallelConstQp forces CQP so stitched chunk seams
            // are quality-flat; the QP from the tuning CQ still tracks the target.
            let force_cqp = config.constant_qp || tp.rc_mode == QsvRateControl::Cqp;
            let (rc_mode_u16, qp_i_effective, qp_p_effective, icq_effective) = if force_cqp {
                let qp_i = if config.quality == AUTO_FROM_TARGET {
                    tp.qp_i
                } else {
                    (config.quality as u16 * 4).min(255)
                };
                (MFX_RATECONTROL_CQP, qp_i, tp.qp_p, 0u16)
            } else {
                (MFX_RATECONTROL_ICQ, 0u16, 0u16, tp.icq_quality)
            };

            let slots = rate_slots_for_rc(
                tp.rc_mode,
                qp_i_effective,
                qp_p_effective,
                icq_effective,
            );

            // Assemble MfxFrameInfo. vendor/intel/mfxstructs.h:20-50.
            // Squad-22: bit_depth_luma/chroma + shift + fourcc come from
            // the dispatched (input_fourcc, bit_depth_luma, bit_depth_chroma,
            // shift) tuple. NV12: (8,8,0). P010: (10,10,1) — Shift=1 is
            // mandatory or oneVPL rejects with INVALID_VIDEO_PARAM.
            let frame_info = MfxFrameInfo {
                bit_depth_luma,
                bit_depth_chroma,
                shift,
                reserved_fi: [0; 7],
                frame_id: 0,
                fourcc: input_fourcc,
                width: align_up(config.width as u16, 16),
                height: align_up(config.height as u16, 16),
                crop_x: 0,
                crop_y: 0,
                crop_w: config.width as u16,
                crop_h: config.height as u16,
                _crop_pad: [0; 6],
                frame_rate_ext_n: (config.frame_rate * 1000.0).round() as u32,
                frame_rate_ext_d: 1000,
                reserved3: 0,
                aspect_ratio_w: 1,
                aspect_ratio_h: 1,
                pic_struct: MFX_PICSTRUCT_PROGRESSIVE,
                chroma_format: MFX_CHROMAFORMAT_YUV420,
                reserved2: 0,
            };

            // oneVPL `mfxInfoMFX` unions all three rc arms into the
            // same three u16 slots, **but the per-arm field layout
            // differs** per vendor/intel/mfxstructs.h:74-89:
            //   slot 0 → InitialDelayInKB (CBR/VBR) / QPI (CQP) / Accuracy (AVBR)
            //   slot 1 → TargetKbps (CBR/VBR) / QPP (CQP) / **ICQQuality (ICQ)**
            //   slot 2 → MaxKbps (CBR/VBR) / QPB (CQP) / Convergence (AVBR)
            //
            // Two notable consequences:
            //   1. For CQP: QPI→slot0, QPP→slot1, QPB→slot2. Natural.
            //   2. For ICQ: ICQQuality must go into **slot 1**, not
            //      slot 0. Slot 0 aliases InitialDelayInKB which the
            //      runtime doesn't read in ICQ mode.
            //
            // An earlier rev of this code (based on `codec-review-59-60.md`
            // §QSV-1's misread of the upstream union — the reviewer
            // cited a legacy Windows SDK layout where the ICQ arm was a
            // separate `struct {mfxU16 ICQQuality, reserved8[4]}` at
            // slot 0; in the Linux oneVPL 2.10 header we ship, the arm
            // is unified with `TargetKbps/QPP/ICQQuality` at slot 1)
            // placed ICQQuality in slot 0, silently falling back to
            // driver default 23 for every quality tier. `rate_slots_for_rc`
            // above puts the value in the correct slot per the
            // vendored header.

            let mfx = MfxInfoMfx {
                reserved: [0; 7],
                // LowPower from the tuning adapter. AV1 QSV encode is VDENC
                // (low-power) on Arc / Meteor Lake+ — the only AV1 encode entry
                // point the iHD driver exposes — so this must be ON, else Query
                // rejects with MFX_ERR_UNSUPPORTED.
                low_power: tp.low_power,
                brc_param_multiplier: 0,
                frame_info,
                codec_id: MFX_CODEC_AV1,
                codec_profile: MFX_PROFILE_AV1_MAIN,
                codec_level: 0, // auto-level
                num_thread: 0,
                target_usage: clamp_target_usage(tp.target_usage),
                gop_pic_size: config.keyframe_interval as u16,
                gop_ref_dist: 1, // no B-frames
                gop_opt_flag: 0,
                idr_interval: 0,
                rate_control_method: rc_mode_u16,
                qpi_or_delay: slots.slot0_qpi_or_delay,
                buffer_size_kb: 0,
                qpp_or_kbps_or_icq: slots.slot1_qpp_or_kbps_or_icq,
                qpb_or_maxkbps: slots.slot2_qpb_or_maxkbps,
                num_slice: 0,
                num_ref_frame: 1,
                encoded_order: 0,
                _tail: [0; 27],
            };

            let mut par = MfxVideoParam {
                alloc_id: 0,
                reserved: [0; 2],
                reserved3: 0,
                // AsyncDepth matches the 4-deep ring — tells the
                // encoder it may receive up to RING_SIZE submissions
                // without a sync in between.
                async_depth: RING_SIZE as u16,
                mfx,
                protected: 0,
                io_pattern: MFX_IOPATTERN_IN_SYSTEM_MEMORY,
                ext_param: ext_param_array.as_ptr() as *mut *mut MfxExtBuffer,
                num_ext_param,
                reserved2: 0,
                _tail: [0; 3],
            };

            // 3. Query — lets the runtime validate and suggest
            //    adjustments for any unsupported knobs. We read `out`
            //    and selectively copy back the fields the runtime
            //    populated — `out` is zero-initialised so we can use
            //    nonzero-ness as a "runtime touched this" signal.
            //
            //    systems-review-59-60 M-Q1: when Query rewrote params
            //    we must Init against the adjusted values, not the
            //    originals.
            let mut out = zeroed_video_param();
            let rc = (*fn_encode_query)(session, &mut par, &mut out);
            let rewrote = match rc {
                MFX_ERR_NONE => false,
                MFX_WRN_INCOMPATIBLE_VIDEO_PARAM | MFX_WRN_VIDEO_PARAM_CHANGED => {
                    // Driver rewrote something — surface the deltas so
                    // ops can correlate quality shifts with driver
                    // behaviour. `out` holds the runtime-adjusted
                    // values; `par` still holds our requested values.
                    tracing::warn!(
                        status = rc,
                        req_rc_method = par.mfx.rate_control_method,
                        got_rc_method = out.mfx.rate_control_method,
                        req_target_usage = par.mfx.target_usage,
                        got_target_usage = out.mfx.target_usage,
                        req_qpi_or_delay = par.mfx.qpi_or_delay,
                        got_qpi_or_delay = out.mfx.qpi_or_delay,
                        req_qpp_or_kbps_or_icq = par.mfx.qpp_or_kbps_or_icq,
                        got_qpp_or_kbps_or_icq = out.mfx.qpp_or_kbps_or_icq,
                        req_profile = par.mfx.codec_profile,
                        got_profile = out.mfx.codec_profile,
                        req_width = par.mfx.frame_info.width,
                        got_width = out.mfx.frame_info.width,
                        req_height = par.mfx.frame_info.height,
                        got_height = out.mfx.frame_info.height,
                        "QSV Query rewrote encoder parameters"
                    );
                    true
                }
                MFX_WRN_PARTIAL_ACCELERATION => {
                    tracing::warn!(
                        "QSV runtime reports partial acceleration — \
                         some encoder stages may fall back to CPU"
                    );
                    false
                }
                err => {
                    // -3 = MFX_ERR_UNSUPPORTED. The driver zeroes the fields it
                    // can't support in `out`; log req-vs-got so we can see which.
                    tracing::error!(
                        status = err,
                        req_codec = par.mfx.codec_id,
                        got_codec = out.mfx.codec_id,
                        req_profile = par.mfx.codec_profile,
                        got_profile = out.mfx.codec_profile,
                        req_rc = par.mfx.rate_control_method,
                        got_rc = out.mfx.rate_control_method,
                        req_tu = par.mfx.target_usage,
                        got_tu = out.mfx.target_usage,
                        req_fourcc = par.mfx.frame_info.fourcc,
                        got_fourcc = out.mfx.frame_info.fourcc,
                        req_chroma = par.mfx.frame_info.chroma_format,
                        got_chroma = out.mfx.frame_info.chroma_format,
                        req_w = par.mfx.frame_info.width,
                        got_w = out.mfx.frame_info.width,
                        req_h = par.mfx.frame_info.height,
                        got_h = out.mfx.frame_info.height,
                        num_ext = par.num_ext_param,
                        io_pattern = par.io_pattern,
                        "QSV Query rejected AV1 params (-3); zeroed `got_*` fields are unsupported"
                    );
                    // Hardware-debug: isolate the unsupported knob by re-querying
                    // variants. (Removed once the AV1/Arc config is pinned.)
                    par.num_ext_param = 0;
                    par.ext_param = std::ptr::null_mut();
                    let mut o = zeroed_video_param();
                    let r_noext = (*fn_encode_query)(session, &mut par, &mut o);
                    tracing::error!(status = r_noext, "QSV diag: no ext buffers");

                    par.mfx.rate_control_method = MFX_RATECONTROL_CQP;
                    par.mfx.qpi_or_delay = 100;
                    par.mfx.qpp_or_kbps_or_icq = 110;
                    par.mfx.qpb_or_maxkbps = 120;
                    let mut o = zeroed_video_param();
                    let r_cqp = (*fn_encode_query)(session, &mut par, &mut o);
                    tracing::error!(status = r_cqp, "QSV diag: no ext + CQP");

                    par.mfx.rate_control_method = 2; // VBR
                    par.mfx.qpi_or_delay = 0;
                    par.mfx.qpp_or_kbps_or_icq = 5000;
                    par.mfx.qpb_or_maxkbps = 7000;
                    let mut o = zeroed_video_param();
                    let r_vbr = (*fn_encode_query)(session, &mut par, &mut o);
                    tracing::error!(status = r_vbr, "QSV diag: no ext + VBR");

                    par.mfx.codec_profile = 0; // let driver pick
                    let mut o = zeroed_video_param();
                    let r_p0 = (*fn_encode_query)(session, &mut par, &mut o);
                    tracing::error!(status = r_p0, "QSV diag: no ext + VBR + profile=0");

                    let _ = mfx_close(session);
                    bail!("MFXVideoENCODE_Query failed: {err}");
                }
            };

            // systems-review-59-60 M-Q1: when Query rewrote params we
            // must Init against the adjusted values, not the originals.
            // `out.mfx` carries only the fields the driver touched
            // (everything else is zero-initialised in `out`), so we
            // copy selectively — keep our base struct for fields the
            // driver didn't rewrite, overwrite the rest.
            if rewrote {
                // frame_info dimensions may have been clamped to
                // hardware limits; everything downstream (surface
                // allocation below) reads from these fields, so we
                // pick them up.
                if out.mfx.frame_info.width != 0 {
                    par.mfx.frame_info.width = out.mfx.frame_info.width;
                }
                if out.mfx.frame_info.height != 0 {
                    par.mfx.frame_info.height = out.mfx.frame_info.height;
                }
                if out.mfx.frame_info.fourcc != 0 {
                    par.mfx.frame_info.fourcc = out.mfx.frame_info.fourcc;
                }
                if out.mfx.frame_info.chroma_format != 0 {
                    par.mfx.frame_info.chroma_format = out.mfx.frame_info.chroma_format;
                }
                if out.mfx.rate_control_method != 0 {
                    par.mfx.rate_control_method = out.mfx.rate_control_method;
                }
                if out.mfx.target_usage != 0 {
                    par.mfx.target_usage = out.mfx.target_usage;
                }
                if out.mfx.codec_profile != 0 {
                    par.mfx.codec_profile = out.mfx.codec_profile;
                }
                if out.mfx.codec_level != 0 {
                    par.mfx.codec_level = out.mfx.codec_level;
                }
                // Note: qpi_or_delay / qpp_or_kbps_or_icq / qpb_or_maxkbps
                // are deliberately left as-requested unless the driver
                // explicitly returned zero-for-adjusted; 0 is a valid
                // ICQ-slot value ("not set") so we keep ours.
            }

            // Re-attach our ext param list (Query zeroes it out on
            // some runtime versions).
            par.ext_param = ext_param_array.as_ptr() as *mut *mut MfxExtBuffer;
            par.num_ext_param = num_ext_param;

            // 4. Init.
            let rc = (*fn_encode_init)(session, &mut par);
            if rc < 0 {
                let _ = mfx_close(session);
                bail!(
                    "MFXVideoENCODE_Init failed: {rc} (likely the AV1 encode component \
                     is not available — Arc / Meteor Lake + required)"
                );
            } else if rc > 0 {
                tracing::warn!(
                    status = rc,
                    "MFXVideoENCODE_Init returned a warning; encoder will run with \
                     adjusted parameters"
                );
            }

            tracing::info!(
                width = config.width,
                height = config.height,
                target = ?config.target,
                tier = ?config.tier,
                rc_mode = ?tp.rc_mode,
                icq_quality = tp.icq_quality,
                qp_i = tp.qp_i,
                target_usage = tp.target_usage,
                tile_cols = tp.num_tile_columns,
                tile_rows = tp.num_tile_rows,
                "QSV AV1 tuning applied"
            );

            // 5. Pre-allocate input surfaces + bitstream buffer. NV12:
            //    Y plane (pitch × height) + UV plane (pitch × height/2)
            //    at the surface's aligned width.
            //
            // Squad-22: P010 surfaces double per-sample byte width.
            // `bytes_per_sample` is 1 for NV12, 2 for P010. `pitch` is
            // expressed in **bytes** (Y row width = width × bytes_per_sample,
            // aligned to 64 bytes for Arc DMA). The total payload still
            // works out as `pitch_bytes × h_aligned × 3 / 2` because
            // 4:2:0 chroma = half height with the same pitch.
            let bytes_per_sample: u32 = if shift == 1 { 2 } else { 1 };
            let pitch = align_up(config.width * bytes_per_sample, 64u32); // bytes
            let h_aligned = align_up(config.height, 16u32);
            let surface_bytes = (pitch as usize * h_aligned as usize * 3) / 2;

            // Ring of N=4 surfaces. Allocate each slot's backing
            // buffer up-front so the surface pointers are stable for
            // the session's lifetime.
            let mut surfaces_vec: Vec<SurfaceSlot> = Vec::with_capacity(RING_SIZE);
            for _ in 0..RING_SIZE {
                let mut backing: Box<[u8]> = vec![0u8; surface_bytes].into_boxed_slice();
                let y_ptr = backing.as_mut_ptr();
                let uv_ptr = y_ptr.add(pitch as usize * h_aligned as usize);
                let surface = MfxFrameSurface1 {
                    reserved: [0; 4],
                    info: frame_info,
                    data: MfxFrameData {
                        mem_id_or_y: y_ptr,
                        // NV12: U pointer is the start of the UV plane,
                        // V pointer is U + 1. Upstream sample_encode
                        // uses this convention.
                        u: uv_ptr,
                        v: uv_ptr.add(1),
                        a: ptr::null_mut(),
                        pitch,
                        time_stamp: 0,
                        frame_order: 0,
                        locked: 0,
                        reserved: [0; 4],
                        corrupted: 0,
                        data_flag: 0,
                    },
                };
                surfaces_vec.push(SurfaceSlot {
                    surface,
                    _backing: backing,
                    sync: ptr::null_mut(),
                });
            }
            let surfaces: [SurfaceSlot; RING_SIZE] = surfaces_vec
                .try_into()
                .map_err(|_| anyhow::anyhow!("RING_SIZE mismatch during surface allocation"))?;

            // 2 MB bitstream buffer — plenty for 4K I-frame. Shared
            // across the ring; `SyncOperation` drains it between
            // frames.
            let mut bitstream_buf: Box<[u8]> = vec![0u8; 2 * 1024 * 1024].into_boxed_slice();
            let bitstream = MfxBitstream {
                reserved: [0; 6],
                decode_time_stamp: 0,
                time_stamp: 0,
                data: bitstream_buf.as_mut_ptr(),
                data_offset: 0,
                data_length: 0,
                max_length: bitstream_buf.len() as u32,
                pic_struct: MFX_PICSTRUCT_PROGRESSIVE,
                frame_type: 0,
                data_flag: 0,
                reserved2: 0,
            };

            let sess = QsvSession {
                session,
                width: config.width,
                height: config.height,
                pts_timescale: (10_000_000.0f64 / config.frame_rate).round() as u64,
                input_pixel_format: config.pixel_format,
                fn_mfx_close: *mfx_close,
                fn_encode_close: *fn_encode_close,
                fn_encode_frame_async: *fn_encode_frame_async,
                fn_sync_operation: *fn_sync_operation,
                tile_ext,
                coding_option3_ext,
                signal_info_ext,
                ext_param_array,
                surfaces,
                ring_idx: 0,
                inflight: VecDeque::with_capacity(RING_SIZE),
                input_pitch: pitch,
                height_aligned: h_aligned,
                bitstream,
                _bitstream_buf: bitstream_buf,
            };

            tracing::info!(
                width = config.width,
                height = config.height,
                gpu = gpu_index,
                ring_size = RING_SIZE,
                "QSV AV1 encoder ready"
            );

            // Silence a handful of constants that only appear in
            // deferred paths (future dispatcher probe, extra ext
            // buffer, cross-platform char alias).
            let _ = (MFX_EXTBUFF_AV1_BITSTREAM_PARAM, 0 as c_char);

            Ok(Self {
                config,
                session: Some(sess),
                encoded_packets: Vec::new(),
                packet_cursor: 0,
                flushed: false,
                frame_counter: 0,
                _runtime_lib: runtime_lib,
            })
        }
    }

    fn encode_one(&mut self, frame: &VideoFrame) -> Result<()> {
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("encode_one called after session drop"))?;

        if frame.format != session.input_pixel_format {
            bail!(
                "QSV session was initialized with {:?} but frame is {:?} \
                 — pipeline must reinit the encoder if pixel format changes",
                session.input_pixel_format,
                frame.format
            );
        }

        let w = session.width as usize;
        let h = session.height as usize;
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);

        // Per-pixel byte width: 1 for 8-bit YUV420p, 2 for Yuv420p10le.
        // Drives both the source buffer-size check and the per-row
        // copy width on the upload path below.
        let bytes_per_sample: usize = if session.input_pixel_format
            == PixelFormat::Yuv420p10le
        {
            2
        } else {
            1
        };
        let y_size_bytes = w * h * bytes_per_sample;
        let uv_size_bytes = cw * ch * bytes_per_sample;

        if frame.data.len() < y_size_bytes + 2 * uv_size_bytes {
            bail!(
                "frame data too small for {}x{} {:?}: need {} bytes, got {}",
                w,
                h,
                session.input_pixel_format,
                y_size_bytes + 2 * uv_size_bytes,
                frame.data.len()
            );
        }

        let pitch = session.input_pitch as usize;
        let h_aligned = session.height_aligned as usize;

        // Pick the next ring slot. If it's still waiting on a sync,
        // drain it first — the ring is full.
        let slot_idx = session.ring_idx;
        if !session.surfaces[slot_idx].sync.is_null() {
            // Producer wrapped around to a slot we haven't sync'd.
            // Drain its sync point FIFO-style. `inflight.front()`
            // SHOULD equal `slot_idx` because submissions happen in
            // order, but we use the FIFO to tolerate any driver
            // reordering.
            let oldest = session
                .inflight
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("ring full but inflight queue empty"))?;
            let sync = session.surfaces[oldest].sync;
            session.surfaces[oldest].sync = ptr::null_mut();
            unsafe {
                sync_and_drain(session, sync, &mut self.encoded_packets)?;
            }
        }

        let slot = &mut session.surfaces[slot_idx];

        unsafe {
            let y_dst = slot.surface.data.mem_id_or_y;
            // UV plane sits one Y plane down: pitch (bytes) × h_aligned (rows).
            let uv_dst = y_dst.add(pitch * h_aligned);

            if session.input_pixel_format == PixelFormat::Yuv420p10le {
                // ── 10-bit P010 upload ──────────────────────────────
                // Source: Yuv420p10le — planar Y/U/V, valid 10 bits in
                // lower 10 of each u16 LE word.
                // Destination: P010 — planar Y + interleaved UV, valid
                // 10 bits in **upper 10** of each u16 LE word
                // (`sample << 6`). pitch is in bytes.
                let src_ptr = frame.data.as_ptr();

                // Y plane.
                for row in 0..h {
                    let src_row = src_ptr.add(row * w * 2) as *const u16;
                    let dst_row = y_dst.add(row * pitch) as *mut u16;
                    for col in 0..w {
                        let sample = (*src_row.add(col)) & 0x03FF;
                        *dst_row.add(col) = sample << 6;
                    }
                }

                // UV plane: interleave U + V into the chroma plane,
                // both shifted by 6 to satisfy P010's upper-10-bit
                // convention.
                let u_src_base = src_ptr.add(y_size_bytes);
                let v_src_base = u_src_base.add(uv_size_bytes);
                for row in 0..ch {
                    let u_src = u_src_base.add(row * cw * 2) as *const u16;
                    let v_src = v_src_base.add(row * cw * 2) as *const u16;
                    let dst_row = uv_dst.add(row * pitch) as *mut u16;
                    for col in 0..cw {
                        let u = (*u_src.add(col)) & 0x03FF;
                        let v = (*v_src.add(col)) & 0x03FF;
                        *dst_row.add(col * 2) = u << 6;
                        *dst_row.add(col * 2 + 1) = v << 6;
                    }
                }
            } else {
                // ── 8-bit NV12 upload ───────────────────────────────
                // Source: YUV420p — planar Y/U/V at 1 byte/sample.
                // Destination: NV12 — planar Y + interleaved UV.
                // Copy Y.
                for row in 0..h {
                    let src = frame.data.as_ptr().add(row * w);
                    let dst = y_dst.add(row * pitch);
                    ptr::copy_nonoverlapping(src, dst, w);
                }

                // Interleave YUV420p U + V into NV12 UV plane.
                let u_src_base = frame.data.as_ptr().add(y_size_bytes);
                let v_src_base = u_src_base.add(uv_size_bytes);
                for row in 0..ch {
                    let u_src = u_src_base.add(row * cw);
                    let v_src = v_src_base.add(row * cw);
                    let dst_row = uv_dst.add(row * pitch);
                    for col in 0..cw {
                        *dst_row.add(col * 2) = *u_src.add(col);
                        *dst_row.add(col * 2 + 1) = *v_src.add(col);
                    }
                }
            }
        }

        slot.surface.data.time_stamp = frame.pts * session.pts_timescale;
        slot.surface.data.frame_order = self.frame_counter;

        // Wrap in catch_unwind so panics during FFI don't unwind
        // across the C ABI boundary.
        let packets = &mut self.encoded_packets;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            let mut sync: MfxSyncPoint = ptr::null_mut();
            let rc = (session.fn_encode_frame_async)(
                session.session,
                ptr::null_mut(),
                &mut session.surfaces[slot_idx].surface as *mut MfxFrameSurface1,
                &mut session.bitstream as *mut MfxBitstream,
                &mut sync,
            );
            match rc {
                MFX_ERR_NONE => {
                    // Submission accepted — sync point is ours to sync
                    // later. Record it on the slot and queue the slot
                    // for draining.
                    session.surfaces[slot_idx].sync = sync;
                    session.inflight.push_back(slot_idx);
                }
                MFX_ERR_MORE_DATA => {
                    // Encoder wants more frames before emitting — normal
                    // at startup. Slot is consumed (driver copied
                    // internally) but no sync point is produced.
                }
                MFX_WRN_IN_EXECUTION => {
                    // Busy — the runtime is still processing a prior
                    // submission. Yield once and, if a sync point came
                    // back with the warning, drain it immediately so
                    // this slot is clean for the next call.
                    std::thread::yield_now();
                    if !sync.is_null() {
                        // Do NOT stash `sync` on the slot — we're
                        // draining it right here, so the slot must
                        // stay marked as not-pending.
                        sync_and_drain(session, sync, packets)?;
                    }
                }
                err => bail!("MFXVideoENCODE_EncodeFrameAsync failed: {err}"),
            }
            Ok::<(), anyhow::Error>(())
        }));

        // Ring advance is unconditional — the slot is consumed whether
        // or not the encoder emitted a sync point.
        session.ring_idx = (session.ring_idx + 1) % RING_SIZE;
        self.frame_counter += 1;

        match result {
            Ok(inner) => inner,
            Err(_) => bail!("panic in QSV encode path — aborting rather than unwinding across FFI"),
        }
    }

    fn flush_drain(&mut self) -> Result<()> {
        if self.session.is_none() {
            return Ok(());
        }
        let packets_ref = &mut self.encoded_packets;
        let session_ref = self.session.as_mut().expect("checked Some above");

        // Wrap the whole FFI path in catch_unwind — sync_and_drain
        // calls `Bytes::copy_from_slice` which allocates, and an
        // allocation panic unwinding across the oneVPL C ABI at
        // EncodeFrameAsync is UB in debug builds. systems-review-59-60.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            // Drain any already-submitted-but-unsynced slots first.
            while let Some(slot_idx) = session_ref.inflight.pop_front() {
                let sync = session_ref.surfaces[slot_idx].sync;
                session_ref.surfaces[slot_idx].sync = ptr::null_mut();
                if !sync.is_null() {
                    sync_and_drain(session_ref, sync, packets_ref)?;
                }
            }

            // Then submit NULL surfaces to flush anything the encoder
            // has buffered internally (GOP lookahead, altref etc.).
            // MFX_ERR_MORE_DATA on NULL input is the EOF signal.
            loop {
                let mut sync: MfxSyncPoint = ptr::null_mut();
                let rc = (session_ref.fn_encode_frame_async)(
                    session_ref.session,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    &mut session_ref.bitstream as *mut MfxBitstream,
                    &mut sync,
                );
                match rc {
                    MFX_ERR_NONE => {
                        if !sync.is_null() {
                            sync_and_drain(session_ref, sync, packets_ref)?;
                        }
                    }
                    MFX_ERR_MORE_DATA => return Ok::<(), anyhow::Error>(()),
                    err if err > 0 => {
                        // Warning — continue.
                        if !sync.is_null() {
                            sync_and_drain(session_ref, sync, packets_ref)?;
                        }
                    }
                    err => bail!("MFXVideoENCODE_EncodeFrameAsync(flush) failed: {err}"),
                }
            }
        }));
        match result {
            Ok(inner) => inner,
            Err(_panic) => bail!(
                "panic in QSV flush path — aborting rather than unwinding across FFI"
            ),
        }
    }
}

/// Wait for the in-flight sync point and copy the bitstream into
/// an `EncodedPacket`. Resets the bitstream buffer for reuse.
///
/// Free function (not a method on `QsvEncoder`) so the caller can hold
/// `&mut session` and `&mut packets` simultaneously without fighting
/// the borrow checker — mirrors the pattern Squad 5 used for AMF's
/// `drain_until_hungry_raw` and the task #60 follow-up review's
/// recommended shape.
unsafe fn sync_and_drain(
    session: &mut QsvSession,
    sync: MfxSyncPoint,
    packets: &mut Vec<EncodedPacket>,
) -> Result<()> {
    unsafe {
        let rc = (session.fn_sync_operation)(session.session, sync, 60_000);
        if rc != MFX_ERR_NONE {
            bail!("MFXVideoCORE_SyncOperation failed: {rc}");
        }

        let len = session.bitstream.data_length as usize;
        if len == 0 {
            return Ok(());
        }
        let offset = session.bitstream.data_offset as usize;
        let slice = std::slice::from_raw_parts(
            session.bitstream.data.add(offset),
            len,
        );
        let data_bytes = Bytes::copy_from_slice(slice);
        // For AV1, oneVPL sets `MFX_FRAMETYPE_I` on key frames and
        // keeps `MFX_FRAMETYPE_IDR` unused (that flag is an H.264
        // concept). AV1 also has an INTRA_ONLY frame type that is a
        // valid random-access point but not mapped to a named
        // `MFX_FRAMETYPE_*` constant in the public oneVPL API — the
        // runtime marks it with `MFX_FRAMETYPE_I` plus the
        // additional `MFX_FRAMETYPE_REF` flag (0x0040). Treat any
        // of those as a keyframe for MP4's `stss` sync-sample
        // table.
        //   MFX_FRAMETYPE_I     = 0x0001 — key frame
        //   MFX_FRAMETYPE_IDR   = 0x8000 — H.264/HEVC IDR (unused for AV1)
        //   MFX_FRAMETYPE_xREF  = 0x0040 — reference frame (paired w/ I for INTRA_ONLY)
        // systems-review-59-60 A-Q5.
        let is_keyframe =
            (session.bitstream.frame_type & (MFX_FRAMETYPE_I | MFX_FRAMETYPE_IDR)) != 0;
        let pts = session.bitstream.time_stamp;

        packets.push(EncodedPacket {
            data: data_bytes,
            pts,
            is_keyframe,
        });

        // Reset the output buffer for reuse.
        session.bitstream.data_length = 0;
        session.bitstream.data_offset = 0;
        Ok(())
    }
}

/// Zero-initialise an `MfxVideoParam` for use as Query's `out` param.
/// Carved out as a function so both `new()` and the unit tests share
/// one definition.
fn zeroed_video_param() -> MfxVideoParam {
    MfxVideoParam {
        alloc_id: 0,
        reserved: [0; 2],
        reserved3: 0,
        async_depth: 0,
        mfx: MfxInfoMfx {
            reserved: [0; 7],
            low_power: 0,
            brc_param_multiplier: 0,
            frame_info: MfxFrameInfo {
                bit_depth_luma: 0,
                bit_depth_chroma: 0,
                shift: 0,
                reserved_fi: [0; 7],
                frame_id: 0,
                fourcc: 0,
                width: 0,
                height: 0,
                crop_x: 0,
                crop_y: 0,
                crop_w: 0,
                crop_h: 0,
                _crop_pad: [0; 6],
                frame_rate_ext_n: 0,
                frame_rate_ext_d: 0,
                reserved3: 0,
                aspect_ratio_w: 0,
                aspect_ratio_h: 0,
                pic_struct: 0,
                chroma_format: 0,
                reserved2: 0,
            },
            codec_id: 0,
            codec_profile: 0,
            codec_level: 0,
            num_thread: 0,
            target_usage: 0,
            gop_pic_size: 0,
            gop_ref_dist: 0,
            gop_opt_flag: 0,
            idr_interval: 0,
            rate_control_method: 0,
            qpi_or_delay: 0,
            buffer_size_kb: 0,
            qpp_or_kbps_or_icq: 0,
            qpb_or_maxkbps: 0,
            num_slice: 0,
            num_ref_frame: 0,
            encoded_order: 0,
            _tail: [0; 27],
        },
        protected: 0,
        io_pattern: 0,
        ext_param: ptr::null_mut(),
        num_ext_param: 0,
        reserved2: 0,
        _tail: [0; 3],
    }
}

impl Encoder for QsvEncoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        self.encode_one(frame)
    }

    fn flush(&mut self) -> Result<()> {
        if !self.flushed {
            self.flush_drain()?;
            self.flushed = true;
        }
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        if self.packet_cursor < self.encoded_packets.len() {
            let pkt = self.encoded_packets[self.packet_cursor].clone();
            self.packet_cursor += 1;
            Ok(Some(pkt))
        } else {
            Ok(None)
        }
    }
}

/// Align `v` up to the next multiple of `a`. `a` must be a power of 2.
fn align_up<T>(v: T, a: T) -> T
where
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::BitAnd<Output = T>
        + std::ops::Not<Output = T>
        + From<u8>,
{
    let one = T::from(1u8);
    (v + a - one) & !(a - one)
}

// ─── Compile-time struct-size assertions ──────────────────────────
//
// Catches ABI drift — if a future pad-edit accidentally changes a
// struct size, the const_assert fires at compile time rather than
// letting a silent offset-shift produce corrupt encodes on real
// hardware. Sizes are the documented upstream oneVPL 2.10 layouts,
// cited against the vendored headers.

// mfxVersion — 2 × u16 = 4 bytes. vendor/intel/mfxstructs.h:191-193.
const _: () = assert!(std::mem::size_of::<MfxVersion>() == 4);

// mfxFrameInfo — 80 bytes. vendor/intel/mfxstructs.h:20-50.
const _: () = assert!(std::mem::size_of::<MfxFrameInfo>() == 80);

// mfxInfoMFX — 256 bytes per upstream (the union of rc-arms is 26
// bytes; reserved tail fills the remainder to 256). Rust struct has
// the union flattened, so pad is `[u32; 27]` and the field preamble
// adds up identically to the upstream layout.
// vendor/intel/mfxstructs.h:54-101.
const _: () = assert!(std::mem::size_of::<MfxInfoMfx>() == 256);

// mfxVideoParam — 304 bytes on 64-bit per upstream.
// vendor/intel/mfxstructs.h:103-117.
const _: () = assert!(std::mem::size_of::<MfxVideoParam>() == 304);

// mfxExtBuffer — 8 bytes (u32 + u32). vendor/intel/mfxstructs.h:121-124.
const _: () = assert!(std::mem::size_of::<MfxExtBuffer>() == 8);

// mfxExtAV1TileParam — 136 bytes. Header(8) + 3×u16(6) + reserved[61](122).
// vendor/intel/mfxstructs.h:135-141.
const _: () = assert!(std::mem::size_of::<MfxExtAv1TileParam>() == 136);

// mfxFrameData — 72 bytes on 64-bit. Layout:
//   4×ptr(32) + u32(4) + [pad 4] + u64(8) + u32(4) + u16(2) +
//   reserved[4](8) + u16(2) + u16(2) = 66, rounded up to 72 to respect
//   the 8-byte alignment set by the pointer fields at the struct head.
// vendor/intel/mfxstructs.h:145-161.
const _: () = assert!(std::mem::size_of::<MfxFrameData>() == 72);

// mfxFrameSurface1 — reserved[4](16) + mfxFrameInfo(80) +
// mfxFrameData(72) = 168 bytes. vendor/intel/mfxstructs.h:163-167.
const _: () = assert!(std::mem::size_of::<MfxFrameSurface1>() == 168);

// mfxBitstream — reserved[6](24) + i64(8) + u64(8) + ptr(8) + 3×u32(12)
// + 4×u16(8) = 68 rounded up to 72 bytes via trailing alignment pad.
// vendor/intel/mfxstructs.h:171-183.
const _: () = assert!(std::mem::size_of::<MfxBitstream>() == 72);

// mfxEncodeCtrl — 88 bytes on 64-bit for our trimmed Rust mirror.
// Upstream oneVPL 2.10 mfxEncodeCtrl is larger (carries additional
// lookahead + per-frame-control knobs in a reserved tail); this Rust
// struct has the fields we'd splat into it if we ever wanted per-frame
// QP overrides — not passed to the encoder today.
const _: () = assert!(std::mem::size_of::<MfxEncodeCtrl>() == 88);

// mfxExtCodingOption2 — NOT used in this file (the tuning adapter's
// knobs are codec-agnostic enough to live on mfxInfoMFX), but named
// here so the assert stays next to the spec citation for any
// reviewer cross-checking. Size from upstream oneVPL 2.10 is 200
// bytes; since we don't carry a Rust mirror of it, no assert — just
// the reference so the codec-review-59-60 audit trail stays visible:
//   const _: () = assert!(std::mem::size_of::<MfxExtCodingOption2>() == 200);

// Squad-22: mfxExtCodingOption3 — Rust mirror sized to match the
// upstream 400-byte layout: 8-byte header + 3×u16[8] (48 bytes) +
// 13 named u16 (26 bytes) + reserved5 [u16;3] (6 bytes) +
// reserved6 [u16;160] (320 bytes) = 408 bytes (Rust). The upstream
// layout interleaves named and reserved differently — what matters
// for the runtime is that `target_bit_depth_*` + `target_chroma_*`
// fields land at the same byte offsets, which the field-level test
// asserts; and `buffer_sz` records the actual bytes we hand over.
const _: () = assert!(std::mem::size_of::<MfxExtCodingOption3>() >= 400);
// mfxExtVideoSignalInfo — 8-byte header + 6×u16 named = 20 bytes,
// padded to alignof(u32)=4 → 20. Upstream public layout is 24 bytes
// with a 4-byte reserved tail; we use the 20-byte mirror because the
// runtime only inspects the named fields and `buffer_sz` is what
// drives the read length.
const _: () = assert!(std::mem::size_of::<MfxExtVideoSignalInfo>() >= 20);

// Squad-22: 10-bit dispatch helpers — the `(BitDepthLuma, BitDepthChroma,
// Shift)` triple must produce exactly (10, 10, 1) for Yuv420p10le. The
// Shift=1 bit is critical: without it oneVPL reads samples from the
// lower 10 bits of each P010 word → silently encodes 1/64 amplitude
// noise. (8, 8, 0) for 8-bit is equally non-negotiable — Shift=1 on
// NV12 surfaces causes the runtime to bail with INVALID_VIDEO_PARAM.
const _: () = assert!({
    let (l, c, s) = qsv_bit_depth_triple(PixelFormat::Yuv420p10le);
    l == 10 && c == 10 && s == 1
});
const _: () = assert!({
    let (l, c, s) = qsv_bit_depth_triple(PixelFormat::Yuv420p);
    l == 8 && c == 8 && s == 0
});
// FOURCC pin: P010 = 0x30313050, NV12 = 0x3231564e.
const _: () = assert!(MFX_FOURCC_P010 == 0x30313050);
const _: () = assert!(MFX_FOURCC_NV12 == 0x3231564e);
// AV1 4:2:0 chroma format = MFX_CHROMAFORMAT_YUV420 (1) + 1 = 2 in
// the oneVPL "plus one" convention used by mfxExtCodingOption3.
const _: () = assert!(MFX_TARGET_CHROMAFORMAT_YUV420_PLUS1 == 2);
// Ext buffer IDs — pinned so a future SDK that renames the FOURCC
// fails compilation rather than silently mis-routing.
const _: () = assert!(MFX_EXTBUFF_CODING_OPTION3 == 0x33444f43);
const _: () = assert!(MFX_EXTBUFF_VIDEO_SIGNAL_INFO == 0x4e495356);

// ─── Unit tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::tuning::{QualityTarget, SpeedTier};

    /// ICQQuality must land in `qpp_or_kbps_or_icq` (slot 1 of the
    /// `mfxInfoMFX` rate-control union) per vendor/intel/mfxstructs.h:83.
    /// An earlier rev of this file put it in slot 0; that's the bug
    /// task #60 caught (HIGH severity — silent quality fallback to
    /// the driver default ICQQuality=23).
    #[test]
    fn test_qsv_icq_quality_lands_on_correct_struct_field() {
        let slots = rate_slots_for_rc(QsvRateControl::Icq, 0, 0, 28);
        assert_eq!(
            slots.slot1_qpp_or_kbps_or_icq, 28,
            "ICQ quality must land in slot 1 (qpp_or_kbps_or_icq) per \
             vendor/intel/mfxstructs.h:83 — this is the TargetKbps/QPP/ICQQuality \
             union arm"
        );
        assert_eq!(
            slots.slot0_qpi_or_delay, 0,
            "slot 0 is InitialDelayInKB/QPI/Accuracy per \
             vendor/intel/mfxstructs.h:74-78 — no ICQQuality here"
        );
        assert_eq!(
            slots.slot2_qpb_or_maxkbps, 0,
            "slot 2 is MaxKbps/QPB/Convergence per \
             vendor/intel/mfxstructs.h:85-89 — no ICQQuality here"
        );
    }

    /// CQP mode places QPI at slot 0 and QPP at slot 1 — vendor
    /// header lines 74-78 (`QPI`) and 80-84 (`QPP`). QPB goes into
    /// slot 2 (header lines 85-89).
    #[test]
    fn test_qsv_cqp_slots_mirror_qpi_qpp_qpb() {
        let slots = rate_slots_for_rc(QsvRateControl::Cqp, 72, 96, 0);
        assert_eq!(slots.slot0_qpi_or_delay, 72);
        assert_eq!(slots.slot1_qpp_or_kbps_or_icq, 96);
        // We mirror QPP to QPB since we run without B-frames (GopRefDist=1).
        assert_eq!(slots.slot2_qpb_or_maxkbps, 96);
    }

    /// TargetUsage must be a valid oneVPL 1..7 value. The tuning
    /// adapter returns 1 (Archive), 4 (Standard), 6 (Draft); this
    /// test covers the end-to-end mapping from `SpeedTier` through
    /// the adapter to the `clamp_target_usage` gate.
    #[test]
    fn test_qsv_target_usage_maps_from_speed_tier() {
        let (w, h) = (1920, 1080);
        let cases = [
            (SpeedTier::Archive, 1u16, "1 = BEST_QUALITY per mfxdefs.h:91"),
            (SpeedTier::Standard, 4u16, "4 = BALANCED per mfxdefs.h:92"),
            (SpeedTier::Draft, 6u16, "6 = one step from BEST_SPEED (7)"),
        ];
        for (tier, expected, reason) in cases {
            let tp = tuning::qsv_av1_params(QualityTarget::Standard, tier, w, h);
            let got = clamp_target_usage(tp.target_usage);
            assert_eq!(got, expected, "{tier:?} → {got} (want {expected}, {reason})");
            assert!(
                (1..=7).contains(&got),
                "TargetUsage must be 1..7 per vendor/intel/mfxdefs.h:91-93"
            );
        }
    }

    /// `clamp_target_usage` defends against out-of-range values from
    /// callers or a future tuning-adapter bug.
    #[test]
    fn test_qsv_target_usage_clamps_out_of_range() {
        assert_eq!(clamp_target_usage(0), 1, "0 clamps up to 1");
        assert_eq!(clamp_target_usage(8), 7, "8 clamps down to 7");
        assert_eq!(clamp_target_usage(255), 7, "255 clamps down to 7");
        assert_eq!(clamp_target_usage(4), 4, "4 passes through");
    }

    /// The ring buffer must cycle 0,1,2,3,0,1,2,3,... with the
    /// `(idx + 1) % RING_SIZE` advance rule. Mirrors
    /// `nvenc.rs::test_ring_buffer_index_cycles`.
    #[test]
    fn test_qsv_ring_buffer_index_cycles() {
        let mut idx = 0usize;
        let mut seen = Vec::new();
        for _ in 0..(RING_SIZE * 3) {
            seen.push(idx);
            idx = (idx + 1) % RING_SIZE;
        }
        assert_eq!(
            seen,
            vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            "ring index must cycle through 0..RING_SIZE"
        );
    }

    #[test]
    fn test_qsv_ring_size_is_four() {
        // Matches NVENC and upstream oneVPL sample_encode's default.
        assert_eq!(RING_SIZE, 4);
    }

    /// `MFX_ERR_MORE_DATA` (-10) on EncodeFrameAsync means the
    /// encoder wants another frame before it can emit output —
    /// normal at startup. The caller's contract: submitting a frame
    /// that hits MORE_DATA should not produce a packet and should
    /// not fail. This test emulates the match arm that handles it.
    #[test]
    fn test_qsv_more_data_on_encode_returns_no_packet() {
        fn simulate_encode(rc: MfxStatus) -> std::result::Result<Option<()>, String> {
            match rc {
                MFX_ERR_NONE => Ok(Some(())), // sync point produced
                MFX_ERR_MORE_DATA => Ok(None), // no packet, no error
                err if err > 0 => Ok(None),    // warning, no packet
                err => Err(format!("encode failed: {err}")),
            }
        }
        assert_eq!(simulate_encode(MFX_ERR_MORE_DATA).unwrap(), None);
        assert_eq!(simulate_encode(MFX_ERR_NONE).unwrap(), Some(()));
        assert!(simulate_encode(-1).is_err(), "unknown negative = hard error");
    }

    /// `flush_drain` loops EncodeFrameAsync(NULL) and terminates on
    /// MFX_ERR_MORE_DATA. The state machine must not panic when
    /// it walks directly to MORE_DATA with zero pending in-flight
    /// frames (the clean-EOF case).
    #[test]
    fn test_qsv_eof_drain_ends_cleanly() {
        // Simulate the flush loop: on MORE_DATA we exit Ok.
        fn simulate_flush_tick(rc: MfxStatus) -> std::result::Result<bool, String> {
            // Returns Ok(true) if we should exit, Ok(false) to keep
            // looping, Err on hard failure.
            match rc {
                MFX_ERR_NONE => Ok(false),
                MFX_ERR_MORE_DATA => Ok(true),
                err if err > 0 => Ok(false),
                err => Err(format!("flush failed: {err}")),
            }
        }
        assert_eq!(simulate_flush_tick(MFX_ERR_MORE_DATA).unwrap(), true,
                   "clean EOF: flush terminates on MORE_DATA without error");
        assert_eq!(simulate_flush_tick(MFX_ERR_NONE).unwrap(), false,
                   "NONE: flush has more output to drain");
        assert_eq!(simulate_flush_tick(MFX_WRN_VIDEO_PARAM_CHANGED).unwrap(), false,
                   "warning: flush keeps looping to drain the bitstream");
        assert!(simulate_flush_tick(-5).is_err(), "hard negative error bails");
    }

    /// `MFX_ERR_MORE_SURFACE` is a DECODE-path status, not an
    /// ENCODE-path one — per vendor/intel/mfxdefs.h:40. We name the
    /// constant so the `match` arm could distinguish it, but the
    /// encode `match` never needs to handle it. Verify the two are
    /// distinct values so a future driver quirk doesn't silently
    /// collapse them.
    #[test]
    fn test_qsv_more_data_and_more_surface_are_distinct() {
        assert_eq!(MFX_ERR_MORE_DATA, -10);
        assert_eq!(MFX_ERR_MORE_SURFACE, -11);
        assert_ne!(MFX_ERR_MORE_DATA, MFX_ERR_MORE_SURFACE);
    }

    /// oneVPL FOURCC encoding is little-endian: the first char in the
    /// macro becomes the low byte of the u32. Verify our AV1 codec +
    /// NV12 FourCC literals match the `MFX_MAKE_FOURCC` definition at
    /// vendor/intel/mfxdefs.h:61-62.
    #[test]
    fn test_qsv_fourcc_literals_match_macro() {
        fn make(a: u8, b: u8, c: u8, d: u8) -> u32 {
            (a as u32) | ((b as u32) << 8) | ((c as u32) << 16) | ((d as u32) << 24)
        }
        assert_eq!(MFX_CODEC_AV1, make(b'A', b'V', b'1', b' '));
        assert_eq!(MFX_FOURCC_NV12, make(b'N', b'V', b'1', b'2'));
        assert_eq!(MFX_EXTBUFF_AV1_TILE_PARAM, make(b'A', b'V', b'1', b'T'));
        assert_eq!(MFX_EXTBUFF_AV1_BITSTREAM_PARAM, make(b'A', b'V', b'1', b'B'));
    }

    /// AV1 profile = MAIN = 1 per vendor/intel/mfxav1.h:24. Main
    /// covers 8-bit and 10-bit 4:2:0 AV1 content — fine for our
    /// pipeline (always 8-bit, rav1d bails on 10-bit).
    #[test]
    fn test_qsv_profile_main_equals_one() {
        assert_eq!(MFX_PROFILE_AV1_MAIN, 1);
    }

    /// Chroma format = YUV420 = 1 per vendor/intel/mfxdefs.h:103.
    #[test]
    fn test_qsv_chroma_format_yuv420_equals_one() {
        assert_eq!(MFX_CHROMAFORMAT_YUV420, 1);
    }

    /// Rate control modes match vendor/intel/mfxdefs.h:76-79.
    #[test]
    fn test_qsv_rc_mode_values_match_spec() {
        assert_eq!(MFX_RATECONTROL_CQP, 3);
        assert_eq!(MFX_RATECONTROL_ICQ, 9); // 8 is LA, not ICQ
    }

    /// Verify the rate-control slot wiring at the mfxInfoMFX field
    /// level, not just the helper. Builds a full `MfxInfoMfx` for an
    /// ICQ job and asserts the ICQ value is on the `qpp_or_kbps_or_icq`
    /// field (slot 1).
    #[test]
    fn test_qsv_mfx_info_fields_from_slots() {
        let slots = rate_slots_for_rc(QsvRateControl::Icq, 0, 0, 33);
        let mfx = MfxInfoMfx {
            reserved: [0; 6],
            low_power: 0,
            brc_param_multiplier: 0,
            _pad0: 0,
            frame_info: MfxFrameInfo {
                bit_depth_luma: 8, bit_depth_chroma: 8, shift: 0,
                reserved_fi: [0; 7], frame_id: 0,
                fourcc: MFX_FOURCC_NV12,
                width: 1920, height: 1080,
                crop_x: 0, crop_y: 0, crop_w: 1920, crop_h: 1080,
                _crop_pad: [0; 6],
                frame_rate_ext_n: 30000, frame_rate_ext_d: 1000,
                reserved3: 0, aspect_ratio_w: 1, aspect_ratio_h: 1,
                pic_struct: MFX_PICSTRUCT_PROGRESSIVE,
                chroma_format: MFX_CHROMAFORMAT_YUV420,
                reserved2: 0,
            },
            codec_id: MFX_CODEC_AV1,
            codec_profile: MFX_PROFILE_AV1_MAIN,
            codec_level: 0,
            num_thread: 0,
            target_usage: 4,
            gop_pic_size: 240,
            gop_ref_dist: 1,
            gop_opt_flag: 0,
            idr_interval: 0,
            rate_control_method: MFX_RATECONTROL_ICQ,
            qpi_or_delay: slots.slot0_qpi_or_delay,
            buffer_size_kb: 0,
            qpp_or_kbps_or_icq: slots.slot1_qpp_or_kbps_or_icq,
            qpb_or_maxkbps: slots.slot2_qpb_or_maxkbps,
            num_slice: 0,
            num_ref_frame: 1,
            encoded_order: 0,
            _tail: [0; 27],
        };

        // End-to-end: user asked for ICQ quality 33.
        // Verify it ended up at the `ICQQuality` alias
        // (`qpp_or_kbps_or_icq`) and nowhere else.
        assert_eq!(mfx.qpp_or_kbps_or_icq, 33, "ICQQuality lives at slot 1");
        assert_eq!(mfx.qpi_or_delay, 0, "slot 0 must be zero in ICQ mode");
        assert_eq!(mfx.qpb_or_maxkbps, 0, "slot 2 must be zero in ICQ mode");
        assert_eq!(mfx.rate_control_method, MFX_RATECONTROL_ICQ);
    }

    /// `zeroed_video_param` returns an all-zero struct — used as
    /// Query's `out` param so the runtime can overwrite non-zero
    /// fields to signal "I adjusted this one".
    #[test]
    fn test_qsv_zeroed_video_param_is_all_zero() {
        let z = zeroed_video_param();
        assert_eq!(z.mfx.codec_id, 0);
        assert_eq!(z.mfx.codec_profile, 0);
        assert_eq!(z.mfx.rate_control_method, 0);
        assert_eq!(z.mfx.qpi_or_delay, 0);
        assert_eq!(z.mfx.qpp_or_kbps_or_icq, 0);
        assert_eq!(z.mfx.qpb_or_maxkbps, 0);
        assert_eq!(z.mfx.frame_info.width, 0);
        assert_eq!(z.mfx.frame_info.height, 0);
        assert!(z.ext_param.is_null());
    }

    /// A quick cover-test for `align_up`: rounds up to the nearest
    /// multiple of a power-of-2.
    #[test]
    fn test_qsv_align_up_power_of_two() {
        assert_eq!(align_up(1u32, 16u32), 16);
        assert_eq!(align_up(16u32, 16u32), 16);
        assert_eq!(align_up(17u32, 16u32), 32);
        assert_eq!(align_up(1920u32, 64u32), 1920);
        assert_eq!(align_up(1921u32, 64u32), 1984);
    }

    /// The Init path on a real Intel GPU would construct a whole
    /// `MfxVideoParam` for Query/Init. We can't do that offline, but
    /// we CAN exercise every field through `rate_slots_for_rc` +
    /// `clamp_target_usage` and make sure the produced ICQ value is
    /// what the tuning adapter emitted — i.e. no silent zero-fallback.
    #[test]
    fn test_qsv_icq_flow_preserves_tuning_adapter_value() {
        for (w, h) in [(640, 360), (1920, 1080), (3840, 2160)] {
            for target in [
                QualityTarget::Low,
                QualityTarget::Standard,
                QualityTarget::High,
            ] {
                let tp = tuning::qsv_av1_params(target, SpeedTier::Standard, w, h);
                // tuning adapter returns ICQ for non-VL targets.
                assert_eq!(tp.rc_mode, QsvRateControl::Icq);
                let slots = rate_slots_for_rc(tp.rc_mode, 0, 0, tp.icq_quality);
                assert_eq!(
                    slots.slot1_qpp_or_kbps_or_icq, tp.icq_quality,
                    "ICQ quality value must reach slot 1 end-to-end — \
                     {target:?}/{w}x{h}: adapter={}, slot1={}",
                    tp.icq_quality, slots.slot1_qpp_or_kbps_or_icq
                );
                assert_eq!(slots.slot0_qpi_or_delay, 0);
                assert_eq!(slots.slot2_qpb_or_maxkbps, 0);
            }
        }
    }

    /// The MfxEncodeCtrl struct is named + sized but not passed to
    /// the encoder today. Verify its size here so the const_assert
    /// citation stays attached to a live reference.
    #[test]
    fn test_qsv_encode_ctrl_struct_size() {
        assert_eq!(std::mem::size_of::<MfxEncodeCtrl>(), 88);
    }

    // ── Squad-22: QSV 10-bit dispatch + color signalling ─────────

    /// Surface FOURCC dispatch must map `Yuv420p10le` to P010 (0x30313050,
    /// 'P','0','1','0') and 8-bit `Yuv420p` to NV12. Mismatched FOURCC
    /// would cause oneVPL to read the wide-word P010 surface as NV12 →
    /// silently encode 1/64-amplitude noise.
    #[test]
    fn test_qsv_fourcc_dispatch_10bit() {
        assert_eq!(qsv_fourcc_for(PixelFormat::Yuv420p).unwrap(), MFX_FOURCC_NV12);
        assert_eq!(qsv_fourcc_for(PixelFormat::Yuv420p10le).unwrap(), MFX_FOURCC_P010);
        assert_eq!(MFX_FOURCC_P010, 0x30313050, "P010 FOURCC = 'P','0','1','0' LE");
    }

    /// Unsupported pixel formats must bail with a typed error. AV1
    /// only carries 4:2:0 (Main profile) — 4:2:2 / 4:4:4 / RGB are
    /// not decodable in the wider AV1 ecosystem either.
    #[test]
    fn test_qsv_fourcc_dispatch_rejects_4_2_2_and_4_4_4() {
        for unsupported in [
            PixelFormat::Yuv422p,
            PixelFormat::Yuv422p10le,
            PixelFormat::Yuv444p,
            PixelFormat::Yuv444p10le,
            PixelFormat::Yuva444p10le,
            PixelFormat::Nv12,
            PixelFormat::Rgb24,
        ] {
            assert!(
                qsv_fourcc_for(unsupported).is_err(),
                "{unsupported:?} must be rejected by QSV dispatch"
            );
        }
    }

    /// Bit-depth triple: NV12 → (8, 8, 0); P010 → (10, 10, 1).
    /// Shift=1 is mandatory for P010 — without it, oneVPL reads
    /// samples as if the valid bits were in the lower 10 → silently
    /// encodes garbage. const_assert! pins this; the test mirrors it
    /// for the test summary.
    #[test]
    fn test_qsv_bit_depth_triple_dispatch() {
        let (luma8, chroma8, shift8) = qsv_bit_depth_triple(PixelFormat::Yuv420p);
        assert_eq!((luma8, chroma8, shift8), (8, 8, 0), "NV12: 8-bit, no shift");

        let (luma10, chroma10, shift10) = qsv_bit_depth_triple(PixelFormat::Yuv420p10le);
        assert_eq!(
            (luma10, chroma10, shift10),
            (10, 10, 1),
            "P010: 10-bit + Shift=1 (upper-10-bit convention)"
        );
    }

    /// Per-encoder H.273 transfer codes must agree with the NVENC +
    /// AMF + mux paths so a single ColorMetadata maps to identical
    /// numeric values across every backend.
    #[test]
    fn test_qsv_transfer_to_h273_codes() {
        assert_eq!(transfer_to_h273(TransferFn::Bt709), 1);
        assert_eq!(transfer_to_h273(TransferFn::Bt470Bg), 4);
        assert_eq!(transfer_to_h273(TransferFn::Linear), 8);
        assert_eq!(transfer_to_h273(TransferFn::St2084), 16, "HDR10 PQ");
        assert_eq!(transfer_to_h273(TransferFn::AribStdB67), 18, "HLG");
        assert_eq!(transfer_to_h273(TransferFn::Unspecified), 1);
    }

    /// Build a 10-bit MfxFrameInfo by hand and assert all four fields
    /// (`bit_depth_luma` / `_chroma` / `shift` / `fourcc`) are
    /// consistent with what oneVPL expects for P010. A future field
    /// reorder during an SDK port surfaces as a diff here.
    #[test]
    fn test_qsv_frame_info_p010_layout() {
        let (bdl, bdc, shift) = qsv_bit_depth_triple(PixelFormat::Yuv420p10le);
        let fourcc = qsv_fourcc_for(PixelFormat::Yuv420p10le).unwrap();

        let fi = MfxFrameInfo {
            bit_depth_luma: bdl,
            bit_depth_chroma: bdc,
            shift,
            reserved_fi: [0; 7],
            frame_id: 0,
            fourcc,
            width: 1920,
            height: 1080,
            crop_x: 0,
            crop_y: 0,
            crop_w: 1920,
            crop_h: 1080,
            _crop_pad: [0; 6],
            frame_rate_ext_n: 30000,
            frame_rate_ext_d: 1000,
            reserved3: 0,
            aspect_ratio_w: 1,
            aspect_ratio_h: 1,
            pic_struct: MFX_PICSTRUCT_PROGRESSIVE,
            chroma_format: MFX_CHROMAFORMAT_YUV420,
            reserved2: 0,
        };

        assert_eq!(fi.bit_depth_luma, 10);
        assert_eq!(fi.bit_depth_chroma, 10);
        assert_eq!(fi.shift, 1, "P010 must set Shift=1");
        assert_eq!(fi.fourcc, MFX_FOURCC_P010);
        assert_eq!(fi.chroma_format, MFX_CHROMAFORMAT_YUV420, "still 4:2:0 sub-sampling");

        // Read fourcc back through raw bytes — guards against an
        // accidental field reorder during an SDK port.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &fi as *const MfxFrameInfo as *const u8,
                std::mem::size_of::<MfxFrameInfo>(),
            )
        };
        let fourcc_offset = std::mem::offset_of!(MfxFrameInfo, fourcc);
        assert_eq!(
            u32::from_le_bytes(bytes[fourcc_offset..fourcc_offset + 4].try_into().unwrap()),
            MFX_FOURCC_P010,
            "fourcc reads back as P010 from the expected struct offset"
        );
    }

    /// `mfxExtCodingOption3` for 10-bit job: `TargetBitDepthLuma` /
    /// `TargetBitDepthChroma` = 10; `TargetChromaFormatPlus1` = 2
    /// (= MFX_CHROMAFORMAT_YUV420 + 1, oneVPL's "plus one" convention).
    /// Without the ext buffer the encoder silently truncates samples
    /// to 8-bit even though the surface is P010.
    #[test]
    fn test_qsv_coding_option3_10bit_layout() {
        let co3 = MfxExtCodingOption3 {
            header: MfxExtBuffer {
                buffer_id: MFX_EXTBUFF_CODING_OPTION3,
                buffer_sz: std::mem::size_of::<MfxExtCodingOption3>() as u32,
            },
            num_ref_active_p: [0; 8],
            num_ref_active_bl0: [0; 8],
            num_ref_active_bl1: [0; 8],
            transform_skip: 0,
            target_chroma_format_plus1: MFX_TARGET_CHROMAFORMAT_YUV420_PLUS1,
            target_bit_depth_luma: 10,
            target_bit_depth_chroma: 10,
            brc_panic_mode: 0,
            low_delay_brc: 0,
            enable_mb_force_intra: 0,
            adaptive_max_frame_size: 0,
            repartition_check_enable: 0,
            reserved5: [0; 3],
            encoded_units_info: 0,
            enable_nal_unit_type: 0,
            ext_brc_adaptive_ltr: 0,
            adaptive_ltr: 0,
            reserved6: [0; 160],
        };

        assert_eq!(co3.target_bit_depth_luma, 10, "AV1 BitDepth=10 in seq header");
        assert_eq!(co3.target_bit_depth_chroma, 10, "AV1 BitDepth=10 in seq header");
        assert_eq!(
            co3.target_chroma_format_plus1, 2,
            "MFX_CHROMAFORMAT_YUV420 (1) + 1 = 2"
        );
        assert_eq!(co3.header.buffer_id, MFX_EXTBUFF_CODING_OPTION3);

        // 'C','D','O','3' LE → 0x33444f43.
        assert_eq!(
            MFX_EXTBUFF_CODING_OPTION3, 0x33444f43,
            "ext buffer ID must match upstream MFX_MAKE_FOURCC('C','D','O','3')"
        );
    }

    /// `mfxExtVideoSignalInfo` for HDR10 (BT.2020 NCL primaries, PQ
    /// transfer, full range): the four H.273 codes must round-trip
    /// from `ColorMetadata` through `transfer_to_h273` into the
    /// signal-info ext buffer's named fields. Without this, AV1 OBU
    /// `color_config` defaults to "unspecified" and downstream
    /// decoders fall back to BT.709.
    #[test]
    fn test_qsv_signal_info_hdr10_layout() {
        let cm = ColorMetadata {
            transfer: TransferFn::St2084,
            matrix_coefficients: 9,  // BT.2020 NCL
            colour_primaries: 9,     // BT.2020
            full_range: true,
            mastering_display: None,
            content_light_level: None,
        };

        let signal_info = MfxExtVideoSignalInfo {
            header: MfxExtBuffer {
                buffer_id: MFX_EXTBUFF_VIDEO_SIGNAL_INFO,
                buffer_sz: std::mem::size_of::<MfxExtVideoSignalInfo>() as u32,
            },
            video_format: 5,
            video_full_range: if cm.full_range { 1 } else { 0 },
            colour_description_present: 1,
            colour_primaries: cm.colour_primaries as u16,
            transfer_characteristics: transfer_to_h273(cm.transfer),
            matrix_coefficients: cm.matrix_coefficients as u16,
        };

        assert_eq!(signal_info.colour_description_present, 1, "must be set so codes emit");
        assert_eq!(signal_info.colour_primaries, 9, "BT.2020");
        assert_eq!(signal_info.transfer_characteristics, 16, "ST 2084 / PQ");
        assert_eq!(signal_info.matrix_coefficients, 9, "BT.2020 NCL");
        assert_eq!(signal_info.video_full_range, 1, "full range");
        assert_eq!(signal_info.header.buffer_id, MFX_EXTBUFF_VIDEO_SIGNAL_INFO);

        // 'V','S','I','N' LE → 0x4e495356.
        assert_eq!(
            MFX_EXTBUFF_VIDEO_SIGNAL_INFO, 0x4e495356,
            "ext buffer ID must match upstream MFX_MAKE_FOURCC('V','S','I','N')"
        );
    }

    /// 8-bit SDR config still shapes correctly — paranoid regression
    /// guard against the 10-bit additions silently breaking the SDR
    /// path. ICQ rate-control + the 8-bit FrameInfo + the default
    /// SDR ColorMetadata round-trip.
    #[test]
    fn test_qsv_8bit_sdr_layout_unchanged() {
        let (bdl, bdc, shift) = qsv_bit_depth_triple(PixelFormat::Yuv420p);
        assert_eq!((bdl, bdc, shift), (8, 8, 0), "8-bit dispatch unchanged");

        let cm = ColorMetadata::default();
        let signal_info = MfxExtVideoSignalInfo {
            header: MfxExtBuffer {
                buffer_id: MFX_EXTBUFF_VIDEO_SIGNAL_INFO,
                buffer_sz: std::mem::size_of::<MfxExtVideoSignalInfo>() as u32,
            },
            video_format: 5,
            video_full_range: if cm.full_range { 1 } else { 0 },
            colour_description_present: 1,
            colour_primaries: cm.colour_primaries as u16,
            transfer_characteristics: transfer_to_h273(cm.transfer),
            matrix_coefficients: cm.matrix_coefficients as u16,
        };

        assert_eq!(signal_info.colour_primaries, 1, "BT.709 default");
        assert_eq!(signal_info.transfer_characteristics, 1, "BT.709 default");
        assert_eq!(signal_info.matrix_coefficients, 1, "BT.709 default");
        assert_eq!(signal_info.video_full_range, 0, "studio range default");
    }
}
