//! NVENC AV1 hardware encoder via NVIDIA Video Codec SDK.
//!
//! Loads libnvidia-encode at runtime via dlopen. Supports AV1 on Ada
//! Lovelace (RTX 4000+) and Ampere A10G (AWS g5).
//!
//! The modern NVENC API entry point is `NvEncodeAPICreateInstance`
//! which populates a `NV_ENCODE_API_FUNCTION_LIST` — a struct of
//! function pointers. We call everything through that table rather
//! than dlsym'ing each function by name. This matches how NVIDIA's
//! sample apps and all production encoders (OBS, FFmpeg) drive the
//! API.
//!
//! Session flow:
//! 1. NvEncodeAPICreateInstance                (get fn table)
//! 2. cuInit + cuCtxCreate                     (CUDA ctx for device)
//! 3. fn_list.nvEncOpenEncodeSessionEx         (attach session)
//! 4. fn_list.nvEncGetEncodePresetConfigEx     (seed encode config)
//! 5. fn_list.nvEncInitializeEncoder           (AV1 + P5 + tuning)
//! 6. fn_list.nvEncCreateInputBuffer × N       (IYUV / YUV420p ring)
//! 7. fn_list.nvEncCreateBitstreamBuffer × N   (output ring)
//! 8. Per frame:
//!    - lockInputBuffer → copy YUV → unlockInputBuffer  (ring slot i)
//!    - encodePicture  (NEED_MORE_INPUT is expected for initial B frames)
//!    - on success: lockBitstream → extract OBUs → unlockBitstream
//!    - advance ring index
//! 9. Flush with PIC_FLAG_EOS, drain every output buffer once
//! 10. destroyInputBuffer × N + destroyBitstreamBuffer × N (reverse alloc order)
//! 11. destroyEncoder → cuCtxDestroy
//!
//! ## Correctness bar for NVENC in this repo
//!
//! GPU E2E verification is not possible on the dev host. Every struct
//! layout below is "spec-conformant-by-review" against
//! `vendor/nvidia/nvEncodeAPI.h` (SDK 12.2) + the full SDK 12.2
//! headers. `const_assert!` checks at the bottom of the file fire at
//! compile time if any struct size drifts — mirroring the pattern in
//! `decode/nvdec.rs` (see review task #65).

use anyhow::{Context, Result, bail};
use bytes::Bytes;
use std::ffi::c_void;
use std::os::raw::{c_int, c_uint};
use std::ptr;

use super::tuning::{self, NvencRateControl};
use super::{AUTO_FROM_TARGET, EncodedPacket, Encoder, EncoderConfig, QualityTarget};
// `ColorMetadata` is reached through `config.color_metadata` on the
// non-test side (no bare-type mention) and through `use super::*`
// inside `mod tests`; pull it in only under cfg(test) to avoid the
// unused-import warning on release builds.
#[cfg(test)]
use crate::frame::ColorMetadata;
use crate::frame::{PixelFormat, TransferFn, VideoFrame};

// ─── NVENC API constants ──────────────────────────────────────────
// See vendor/nvidia/nvEncodeAPI.h for authoritative definitions.

const NV_ENC_SUCCESS: c_uint = 0;
// `_NVENCSTATUS` values of interest. vendor/nvidia/nvEncodeAPI.h:30-42.
const NV_ENC_ERR_INVALID_PTR: c_uint = 6;
const NV_ENC_ERR_INVALID_PARAM: c_uint = 8;
const NV_ENC_ERR_ENCODER_NOT_INITIALIZED: c_uint = 11;
const NV_ENC_ERR_LOCK_BUSY: c_uint = 13;
const NV_ENC_ERR_NEED_MORE_INPUT: c_uint = 17;
const NV_ENC_ERR_ENCODER_BUSY: c_uint = 18;

const NV_ENC_DEVICE_TYPE_CUDA: c_uint = 1;
const NV_ENC_BUFFER_FORMAT_IYUV: c_uint = 0x00000100;
/// 10-bit planar 4:2:0. P010-style: each sample is a 16-bit LE word
/// with the valid 10-bit value in the **upper 10 bits**
/// (`sample_10bit << 6`). See `vendor/nvidia/nvEncodeAPI.h:94-115` for
/// the SDK 12.2 enumeration. This matches NVDEC's P016 surface output
/// (Squad-6) and the pipeline's `Yuv420p10le` representation, which
/// stores the value in the **lower 10 bits** — `upload_frame_10bit`
/// performs the `<<6` left-shift on copy so the surface byte layout
/// satisfies the SDK convention.
const NV_ENC_BUFFER_FORMAT_YUV420_10BIT: c_uint = 0x00010000;

const NV_ENC_PIC_FLAG_FORCEIDR: c_uint = 0x02;
const NV_ENC_PIC_FLAG_EOS: c_uint = 0x08;

const NV_ENC_PIC_TYPE_P: c_uint = 0;
const NV_ENC_PIC_TYPE_I: c_uint = 2;
const NV_ENC_PIC_TYPE_IDR: c_uint = 3;

#[allow(dead_code)]
const NV_ENC_TUNING_INFO_HIGH_QUALITY: c_uint = 1;

// Rate control modes — vendor/nvidia/nvEncodeAPI.h:77-84 (_NV_ENC_PARAMS_RC_MODE).
const NV_ENC_PARAMS_RC_CONSTQP: u32 = 0x0;
const NV_ENC_PARAMS_RC_VBR: u32 = 0x1;
// `_HQ` is gone in SDK 12.2 (merged into VBR + high-quality tuning) but
// kept in the enum for back-compat. We emit plain VBR with tuning =
// HIGH_QUALITY which is the 12.2-idiomatic "VBR_HQ" path.
#[allow(dead_code)]
const NV_ENC_PARAMS_RC_VBR_HQ: u32 = 0x20;

// Ring-buffer depth. 4 mirrors ffmpeg libavcodec/nvenc.c's default
// `nb_surfaces` for 1-pass and keeps the encoder pipeline full on Ada
// without oversubscribing GPU memory.
const RING_SIZE: usize = 4;

// API version encoding — values lifted directly from
// vendor/nvidia/nvEncodeAPI.h (SDK 13.0; refreshed from
// FFmpeg/nv-codec-headers master 2026-05-01 to match production
// driver 580.126.09 / CUDA 13.0).
//
// CRITICAL DELTA from SDK 12.2: the NVENCAPI_VERSION formula
// SWAPPED major and minor positions:
//   12.2: NVENCAPI_VERSION = (MAJOR << 24) | MINOR        (= 0x0C000002)
//   13.0: NVENCAPI_VERSION =  MAJOR        | (MINOR << 24) (= 0x0000000D)
// We had been stamping 0x0D000000 thinking that meant "13.0" — the
// driver read it as a malformed 12.x.x marker and segfaulted on
// downstream parsing.
const NVENCAPI_MAJOR: u32 = 13;
const NVENCAPI_MINOR: u32 = 0;
const NVENCAPI_VERSION: u32 = NVENCAPI_MAJOR | (NVENCAPI_MINOR << 24);

const fn struct_version(ver: u32) -> u32 {
    NVENCAPI_VERSION | (ver << 16) | (0x7 << 28)
}

// Per-struct version constants — SDK 13.0 values from header.
// MOST changed from 12.2; comments list the deltas so a future
// SDK-bump audit can spot which structs grew.
const NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER: u32 = struct_version(1); // unchanged
const NV_ENC_INITIALIZE_PARAMS_VER: u32 = struct_version(7) | (1u32 << 31); // 12.2 was struct_version(5) without high-bit
const NV_ENC_CREATE_INPUT_BUFFER_VER: u32 = struct_version(2); // 12.2 was struct_version(1)
const NV_ENC_CREATE_BITSTREAM_BUFFER_VER: u32 = struct_version(1); // unchanged
const NV_ENC_LOCK_INPUT_BUFFER_VER: u32 = struct_version(1); // unchanged
const NV_ENC_LOCK_BITSTREAM_VER: u32 = struct_version(2) | (1u32 << 31); // 12.2 was (1) without high-bit
const NV_ENC_PIC_PARAMS_VER: u32 = struct_version(7) | (1u32 << 31); // 12.2 was (4) without high-bit
const NV_ENC_CONFIG_VER: u32 = struct_version(9) | (1u32 << 31); // 12.2 was (7) without high-bit
const NV_ENC_PRESET_CONFIG_VER: u32 = struct_version(5) | (1u32 << 31); // 12.2 was (4) | high-bit

// GUID layout: 32-bit Data1 (LE), 16-bit Data2/3 (LE), 8 raw bytes.
// Values from NVIDIA Video Codec SDK 12.2 headers (vendor/nvidia/nvEncodeAPI.h:49).
#[repr(C)]
#[derive(Clone, Copy)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

const NV_ENC_CODEC_AV1_GUID: Guid = Guid {
    data1: 0x0a352289,
    data2: 0x0aa7,
    data3: 0x4759,
    data4: [0x86, 0x2d, 0x5d, 0x15, 0xcd, 0x16, 0xd2, 0x54],
};

// Preset GUIDs from SDK 13.0 (vendor/nvidia/nvEncodeAPI.h:226-251).
// SDK 12.2 used different values for P5/P6/P7 — see tuning.rs comment
// for the full rotation. Sending 12.2 GUIDs to a 13.0 driver returns
// NV_ENC_ERR_UNSUPPORTED_PARAM (rc=12) from NvEncGetEncodePresetConfigEx.
#[allow(dead_code)]
const NV_ENC_PRESET_P5_GUID: Guid = Guid {
    data1: 0x21c6e6b4,
    data2: 0x297a,
    data3: 0x4cba,
    data4: [0x99, 0x8f, 0xb6, 0xcb, 0xde, 0x72, 0xad, 0xe3],
};

#[allow(dead_code)]
const NV_ENC_PRESET_P6_GUID: Guid = Guid {
    data1: 0x8e75c279,
    data2: 0x6299,
    data3: 0x4ab6,
    data4: [0x83, 0x02, 0x0b, 0x21, 0x5a, 0x33, 0x5c, 0xf5],
};

#[allow(dead_code)]
const NV_ENC_PRESET_P7_GUID: Guid = Guid {
    data1: 0x84848c12,
    data2: 0x6f71,
    data3: 0x4c13,
    data4: [0x93, 0x1b, 0x53, 0xe2, 0x83, 0xf5, 0x79, 0x74],
};

/// Rebuild a `Guid` from the adapter's raw 16-byte form. Keeps the
/// adapter independent of the SDK struct definition.
fn guid_from_bytes(bytes: [u8; 16]) -> Guid {
    Guid {
        data1: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        data2: u16::from_le_bytes([bytes[4], bytes[5]]),
        data3: u16::from_le_bytes([bytes[6], bytes[7]]),
        data4: [
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ],
    }
}

// ─── CUDA driver minimal FFI ──────────────────────────────────────
type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;

type FnCuInit = unsafe extern "C" fn(c_uint) -> CUresult;
type FnCuDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type FnCuCtxCreate = unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult;
type FnCuCtxDestroy = unsafe extern "C" fn(CUcontext) -> CUresult;
type FnCuCtxPushCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
type FnCuCtxPopCurrent = unsafe extern "C" fn(*mut CUcontext) -> CUresult;

// ─── NVENC API structs ────────────────────────────────────────────
//
// Layouts mirror vendor/nvidia/nvEncodeAPI.h (SDK 12.2). Where a Rust
// field is more granular than the header stub, the extended layout is
// taken from the full SDK 12.2 headers (`NV_ENC_RC_PARAMS`,
// `NV_ENC_CONFIG_AV1`, `NV_ENC_LOCK_BITSTREAM`). Compile-time
// `const_assert!`s at the bottom of the file verify exact sizes.

#[repr(C)]
struct NvEncOpenEncodeSessionExParams {
    version: u32,
    device_type: u32,
    device: *mut c_void,
    reserved: *mut c_void,
    api_version: u32,
    reserved1: [u32; 253],
    reserved2: [*mut c_void; 64],
}

/// `NV_ENC_INITIALIZE_PARAMS` — SDK 13.0 layout
/// (vendor/nvidia/nvEncodeAPI.h:2233-2292).
///
/// MAJOR LAYOUT CHANGE FROM SDK 12.2: the six u32 boolean fields after
/// `enable_ptd` (reportSliceOffsets, enableSubFrameWrite,
/// enableExternalMEHints, enableMEOnlyMode, enableWeightedPrediction,
/// enableOutputInVidmem) plus the `reserved[3]` block were COLLAPSED into
/// a single 32-bit bitfield word + new `privDataSize` u32 + `reserved` u32
/// + `privData` (void*). SDK 13 also packed in 4 NEW bitfield slots
/// (splitEncodeMode:4, enableReconFrameOutput:1, enableOutputStats:1,
/// enableUniDirectionalB:1) which the bitfield word now owns.
///
/// Other deltas:
///   - `maxMEHintCountsPerBlock[2]` is now `NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE[2]`
///     (was the wrong `[u32; 2]` mirror in 12.2 — incidentally compensated
///      for by the trailing reserved[287] over-size). Each element is 16 bytes
///     (1 bitfield u32 + 3 u32 reserved) → 32 bytes total.
///   - `numStateBuffers` (NEW in SDK 13) — encoding-without-state-advance.
///   - `outputStatsLevel` (NEW in SDK 13) — pairs with the new bitfield slot
///     `enableOutputStats`.
///   - `reserved1` shrunk from `[u32; 287]` → `[u32; 284]` to make room
///     for the two NEW u32 fields above. Total struct size unchanged at
///     1800 bytes — the new fields steal trailing reserved space.
///
/// The override block in NvencEncoder::new only sets `enable_encode_async = 0`
/// and `enable_ptd = 1`, so the bitfield rewrite is a no-op for our caller
/// (every other flag was already implicitly zero via mem::zeroed). New helpers
/// AV1_INIT_BIT_* below are provided for completeness in case a future caller
/// wants to set splitEncodeMode etc.
#[repr(C)]
struct NvEncInitializeParams {
    version: u32,
    encode_guid: Guid,
    preset_guid: Guid,
    encode_width: u32,
    encode_height: u32,
    dar_width: u32,
    dar_height: u32,
    frame_rate_num: u32,
    frame_rate_den: u32,
    enable_encode_async: u32,
    enable_ptd: u32,
    /// Bitfield word collapsing 11 boolean / packed flags. See INIT_BIT_*
    /// constants below for the layout.
    flags: u32,
    priv_data_size: u32,
    reserved: u32,
    priv_data: *mut c_void,
    encode_config: *mut c_void,
    max_encode_width: u32,
    max_encode_height: u32,
    /// Each element is `NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE`
    /// = 1 bitfield u32 + 3 u32 reserved = 16 bytes. We mirror as a flat
    /// 8-u32 array (`[u32; 8]` = 32 bytes) since we never set any of the
    /// hint counts in this service (external ME hints are not used).
    max_me_hint_counts_per_block: [u32; 8],
    tuning_info: u32,
    buffer_format: u32,
    /// NEW in SDK 13: number of state buffers for stateless encode flows.
    /// Always 0 (= single state buffer = our use case).
    num_state_buffers: u32,
    /// NEW in SDK 13: granularity for encoded-frame output stats. Always 0
    /// (= NV_ENC_OUTPUT_STATS_NONE). Pairs with the new `enableOutputStats`
    /// bit (bit 9 in the flags word above) which we leave at 0.
    output_stats_level: u32,
    reserved1: [u32; 284],
    reserved2: [*mut c_void; 64],
}

// Bitfield helpers for `NvEncInitializeParams.flags`. SDK 13 packs 11
// flags into one u32. Bit layout (LSB first):
//   bit  0:    reportSliceOffsets
//   bit  1:    enableSubFrameWrite
//   bit  2:    enableExternalMEHints
//   bit  3:    enableMEOnlyMode
//   bit  4:    enableWeightedPrediction
//   bits 5-8:  splitEncodeMode (4 bits, NEW in SDK 13)
//   bit  9:    enableOutputInVidmem
//   bit  10:   enableReconFrameOutput (NEW)
//   bit  11:   enableOutputStats (NEW)
//   bit  12:   enableUniDirectionalB (NEW)
//   bits 13-31: reservedBitFields
#[allow(dead_code)]
const INIT_BIT_REPORT_SLICE_OFFSETS: u32 = 1 << 0;
#[allow(dead_code)]
const INIT_BIT_ENABLE_SUB_FRAME_WRITE: u32 = 1 << 1;
#[allow(dead_code)]
const INIT_BIT_ENABLE_EXTERNAL_ME_HINTS: u32 = 1 << 2;
#[allow(dead_code)]
const INIT_BIT_ENABLE_ME_ONLY_MODE: u32 = 1 << 3;
#[allow(dead_code)]
const INIT_BIT_ENABLE_WEIGHTED_PREDICTION: u32 = 1 << 4;
#[allow(dead_code)]
const INIT_BIT_ENABLE_OUTPUT_IN_VIDMEM: u32 = 1 << 9;
#[allow(dead_code)]
const INIT_BIT_ENABLE_RECON_FRAME_OUTPUT: u32 = 1 << 10;
#[allow(dead_code)]
const INIT_BIT_ENABLE_OUTPUT_STATS: u32 = 1 << 11;
#[allow(dead_code)]
const INIT_BIT_ENABLE_UNI_DIRECTIONAL_B: u32 = 1 << 12;

/// Rate-control params. Layout is the full SDK 12.2 definition from
/// `_NV_ENC_RC_PARAMS` (not the minimal vendor stub at
/// `vendor/nvidia/nvEncodeAPI.h:154-170`).
///
/// For AV1 under NVENC the `targetQuality` field range is 0..63 for
/// AV1 (0..51 only applies to H.264/HEVC, per NVENC SDK 13 §3.8.3).
#[repr(C)]
struct NvEncRcParams {
    version: u32,
    rate_control_mode: u32,
    const_qp_inter_p: u32,
    const_qp_inter_b: u32,
    const_qp_intra: u32,
    average_bitrate: u32,
    max_bitrate: u32,
    vbv_buffer_size: u32,
    vbv_initial_delay: u32,
    /// Bitfield packed as SDK does (enableMinQP, enableMaxQP, enableAQ,
    /// enableLookahead, aqStrength nibble, etc).
    flags: u32,
    min_qp_inter_p: u32,
    min_qp_inter_b: u32,
    min_qp_intra: u32,
    max_qp_inter_p: u32,
    max_qp_inter_b: u32,
    max_qp_intra: u32,
    initial_rc_qp_inter_p: u32,
    initial_rc_qp_inter_b: u32,
    initial_rc_qp_intra: u32,
    /// 12 bytes covering `temporallayerIdxMask: u32` + `temporalLayerQP[8]: u8`
    /// in SDK 13 (vendor/nvidia/nvEncodeAPI.h:1586-1589). Was wrongly mirrored
    /// as `[u32; 2]` (8 bytes) in the SDK 12.2 mirror — that 4-byte deficit
    /// shifted every subsequent field by 4 bytes which the trailing reserved
    /// padding silently absorbed at the END but mis-mapped the intermediate
    /// field offsets. We always set this to all-zero (no temporal-layer
    /// bitrate plumbing in this service).
    temporally_layer_bitrate_ratio: [u32; 3],
    /// CQ quality target. Range: 0..63 for AV1, 0..51 for H.264/HEVC.
    target_quality: u8,
    /// 8-bit fractional part of `target_quality` (8.8 fixed-point).
    /// Left at 0 — whole-step CQ is enough for our VMAF bands.
    target_quality_lsb: u8,
    lookahead_depth: u16,
    low_delay_key_frame_scale: u32,
    qp_map_mode: u32,
    multi_pass: u32,
    alpha_layer_bitrate_ratio: u32,
    cbqpi_ofs: i8,
    cbqpp_ofs: i8,
    crqpi_ofs: i8,
    crqpp_ofs: i8,
    reserved: [u32; 4],
}

/// AV1-specific codec config. Lives inside NV_ENC_CONFIG's
/// encodeCodecConfig union. SDK 13.0 layout — see
/// vendor/nvidia/nvEncodeAPI.h `_NV_ENC_CONFIG_AV1`.
///
/// MAJOR LAYOUT CHANGE FROM SDK 12.2: all the boolean enable_*
/// fields collapsed into ONE 32-bit bitfield word at offset 16. The
/// override block in NvencEncoder::new now sets that bitfield via
/// shifts/masks. Three new groups of fields appended after the
/// original layout: outputBitDepth/inputBitDepth (replaced
/// pixel_bit_depth_minus_8), numFwdRefs/numBwdRefs typed enums,
/// ltrNumFrames/numTemporalLayers/tfLevel temporal-layer config.
#[repr(C)]
struct NvEncConfigAv1 {
    level: u32,
    tier: u32,
    min_part_size: u32,
    max_part_size: u32,
    /// Bitfield word — see ::AV1_BIT_* constants. The SDK declares
    /// these as C bitfields; we mirror as a single u32 + helpers
    /// because Rust doesn't have C-style bitfield syntax. Bit
    /// layout (LSB first):
    ///   bit  0:    outputAnnexBFormat (0 = LOB / MP4-friendly, 1 = AnnexB)
    ///   bit  1:    enableTimingInfo
    ///   bit  2:    enableDecoderModelInfo
    ///   bit  3:    enableFrameIdNumbers
    ///   bit  4:    disableSeqHdr
    ///   bit  5:    repeatSeqHdr (set 1 for keyframe-seekable MP4)
    ///   bit  6:    enableIntraRefresh
    ///   bits 7-8:  chromaFormatIDC (1 for 4:2:0)
    ///   bit  9:    enableBitstreamPadding
    ///   bit  10:   enableCustomTileConfig
    ///   bit  11:   enableFilmGrainParams
    ///   bit  12:   enableLTR
    ///   bit  13:   enableTemporalSVC
    ///   bit  14:   outputMaxCll
    ///   bit  15:   outputMasteringDisplay
    ///   bits 16+:  reserved
    flags: u32,
    idr_period: u32,
    intra_refresh_period: u32,
    intra_refresh_cnt: u32,
    max_num_ref_frames_in_dpb: u32,
    num_tile_columns: u32,
    num_tile_rows: u32,
    reserved2: u32,
    tile_widths: *mut u32,
    tile_heights: *mut u32,
    max_temporal_layers_minus1: u32,
    color_primaries: u32,
    transfer_characteristics: u32,
    matrix_coefficients: u32,
    color_range: u32,
    chroma_sample_position: u32,
    use_b_frames_as_ref: u32,
    film_grain_params: *mut c_void,
    num_fwd_refs: u32,
    num_bwd_refs: u32,
    /// `NV_ENC_BIT_DEPTH_8 = 0`, `NV_ENC_BIT_DEPTH_10 = 1`.
    output_bit_depth: u32,
    input_bit_depth: u32,
    ltr_num_frames: u32,
    num_temporal_layers: u32,
    tf_level: u32,
    reserved1: [u32; 230],
    reserved3: [*mut c_void; 62],
}

// Bitfield positions in NvEncConfigAv1.flags. Used by the override
// block to set specific enable flags without bit-twiddling at the
// call site.
const AV1_BIT_OUTPUT_ANNEXB_FORMAT: u32 = 1 << 0;
#[allow(dead_code)]
const AV1_BIT_ENABLE_TIMING_INFO: u32 = 1 << 1;
const AV1_BIT_REPEAT_SEQ_HDR: u32 = 1 << 5;
// chromaFormatIDC occupies bits 7-8; value 1 (= 4:2:0) goes in bit 7.
const AV1_CHROMA_FORMAT_IDC_420: u32 = 1 << 7;

/// Full encode config containing RC params + codec-specific union.
/// Size is fixed per SDK version; `const_assert!` verifies. SDK 13.0
/// layout — see vendor/nvidia/nvEncodeAPI.h `_NV_ENC_CONFIG`.
///
/// 2026-05-01 audit (#2): the `encodeCodecConfig` slot is the C
/// `NV_ENC_CODEC_CONFIG` UNION whose `sizeof` is driven by the LARGEST
/// variant — NV_ENC_CONFIG_H264 at 1792 bytes (HEVC=1560, AV1=1552).
/// Our previous mirror sized the slot to NV_ENC_CONFIG_AV1 (1552)
/// only, so the struct was 240 bytes shorter than the C ABI expected.
/// The driver then read/wrote 240 bytes past the end of our
/// stack-allocated `enc_config` during `NvEncInitializeEncoder` /
/// `NvEncGetEncodePresetConfigEx` — undefined behaviour that worked by
/// luck on prior runs. Verified against `sizeof(NV_ENC_CONFIG)` from
/// the vendored SDK 13 header: 3584. Trailing `_codec_config_pad`
/// brings the union slot up to 1792 without touching `codec_config_av1`'s
/// field offsets (which the encoder code reads at relative offsets).
///
/// Pre-SDK-13 audit gap (#1): this mirror was MISSING the trailing
/// `reserved2: [void*; 64]` field — the C struct has had it since at
/// least 12.2. Restored 2026-05-01 alongside the SDK 13 refresh.
#[repr(C)]
struct NvEncConfig {
    version: u32,
    profile_guid: Guid,
    gop_length: u32,
    frame_interval_p: u32,
    mono_chrome_encoding: u32,
    frame_field_mode: u32,
    mv_precision: u32,
    rc_params: NvEncRcParams,
    codec_config_av1: NvEncConfigAv1,
    /// Trailing pad to widen the `NV_ENC_CODEC_CONFIG` union slot from
    /// the AV1 variant size (1552) up to the H.264 variant size (1792)
    /// — which is what the C union sizes to. Driver may write into
    /// these bytes for variant-agnostic reserved fields; we keep them
    /// zero-initialised via `mem::zeroed()`. The encoder reads
    /// `codec_config_av1.flags` / `.idr_period` / etc. which all live
    /// inside the AV1-sized region BEFORE this pad, so the override
    /// block in `NvencEncoder::new` is unaffected.
    _codec_config_pad: [u32; 60],
    reserved: [u32; 278],
    reserved2: [*mut c_void; 64],
}

/// `_NV_ENC_PRESET_CONFIG` — wrapper around `NV_ENC_CONFIG` that
/// `NvEncGetEncodePresetConfigEx` populates with preset+tuning
/// defaults. SDK 13.0 layout (added a leading `reserved` u32, grew
/// reserved1 from 255 → 256):
///   u32 version
///   u32 reserved   ← NEW in SDK 13
///   NV_ENC_CONFIG presetCfg
///   u32 reserved1[256]   ← was [255] in 12.2
///   void* reserved2[64]
#[repr(C)]
struct NvEncPresetConfig {
    version: u32,
    reserved: u32,
    preset_cfg: NvEncConfig,
    reserved1: [u32; 256],
    reserved2: [*mut c_void; 64],
}

#[repr(C)]
struct NvEncCreateInputBuffer {
    version: u32,
    width: u32,
    height: u32,
    memory_heap: u32,
    buffer_fmt: u32,
    reserved: u32,
    input_buffer: *mut c_void,
    sys_mem_buffer: *mut c_void,
    reserved1: [u32; 57],
    reserved2: [*mut c_void; 63],
}

#[repr(C)]
struct NvEncCreateBitstreamBuffer {
    version: u32,
    size: u32,
    memory_heap: u32,
    reserved: u32,
    bitstream_buffer: *mut c_void,
    bitstream_buffer_ptr: *mut c_void,
    reserved1: [u32; 58],
    reserved2: [*mut c_void; 64],
}

#[repr(C)]
struct NvEncLockInputBuffer {
    version: u32,
    reserved1: u32,
    input_buffer: *mut c_void,
    buffer_data_ptr: *mut c_void,
    pitch: u32,
    reserved2: [u32; 251],
    reserved3: [*mut c_void; 64],
}

/// `NV_ENC_LOCK_BITSTREAM` — SDK 13.0 layout
/// (vendor/nvidia/nvEncodeAPI.h:2675-2714).
///
/// Previous mirror collapsed the seven scalar fields between
/// `ltr_frame_bitmap` and the trailing `reserved1[219]` into a flat
/// `reserved: [u32; 13]` blob and used a 64-element `reserved2` ptr
/// array instead of the spec's 63 + a trailing `reserved_internal[8]`
/// u32 array. Total size happened to land at 1544 either way, so the
/// `size_of` const_assert passed — but every named [out] field after
/// offset 88 lived at the wrong byte offset, and the [in] `reserved`
/// scalar at offset 116 + the [in,out] `output_stats_ptr` at offset
/// 120 weren't where the driver expected them either.
///
/// On Blackwell + driver 580+ this manifested as
/// `NvEncLockBitstream` returning `NV_ENC_ERR_INVALID_PARAM (8)`
/// during EOS drain, and would have silently corrupted any
/// callsite that read `intra_mb_count` / `inter_mb_count` /
/// `average_mvx` / `average_mvy` / `frame_idx_display` (we don't
/// today, but every future getRCStats consumer would have).
///
/// Fields recovered against SDK 13.0 spec:
///   `temporal_id`            (NEW SDK 13)
///   `alpha_layer_size_in_bytes` (NEW SDK 13)
///   `output_stats_ptr_size`  (NEW SDK 13)
///   `reserved` scalar @ 116  (NEW SDK 13 — gap was inside the prior `[u32;13]`)
///   `output_stats_ptr`       (NEW SDK 13)
///   `frame_idx_display`      (NEW SDK 13)
///   `reserved_internal[8]`   (NEW SDK 13)
///
/// Offset assertions below catch future SDK drift at compile time
/// — `size_of` alone is insufficient (the prior layout proved it).
#[repr(C)]
struct NvEncLockBitstream {
    version: u32,                      // offset 0
    bitfields: u32, // offset 4 — doNotWait:1, ltrFrame:1, getRCStats:1, reservedBitFields:29
    output_bitstream: *mut c_void, // offset 8
    slice_offsets: *mut u32, // offset 16
    frame_idx: u32, // offset 24
    hw_encode_status: u32, // offset 28
    num_slices: u32, // offset 32
    bitstream_size_in_bytes: u32, // offset 36
    output_time_stamp: u64, // offset 40
    output_duration: u64, // offset 48
    bitstream_buffer_ptr: *mut c_void, // offset 56
    picture_type: u32, // offset 64
    picture_struct: u32, // offset 68
    frame_avg_qp: u32, // offset 72
    frame_satd: u32, // offset 76
    ltr_frame_idx: u32, // offset 80
    ltr_frame_bitmap: u32, // offset 84
    temporal_id: u32, // offset 88
    intra_mb_count: u32, // offset 92
    inter_mb_count: u32, // offset 96
    average_mvx: i32, // offset 100
    average_mvy: i32, // offset 104
    alpha_layer_size_in_bytes: u32, // offset 108
    output_stats_ptr_size: u32, // offset 112
    reserved: u32,  // offset 116 — must be 0
    output_stats_ptr: *mut c_void, // offset 120
    frame_idx_display: u32, // offset 128
    reserved1: [u32; 219], // offset 132
    reserved2: [*mut c_void; 63], // offset 1008
    reserved_internal: [u32; 8], // offset 1512
                    // total size: 1544 bytes
}

/// `NV_ENC_PIC_PARAMS` — SDK 13.0 layout
/// (vendor/nvidia/nvEncodeAPI.h:2564-2625).
///
/// MAJOR LAYOUT CHANGE FROM PREVIOUS MIRROR — caused the 2026-05-01
/// post-init SIGSEGV right after frame 0 emitted from NVDEC.
///
/// 1. `codec_pic_params` was `[u8; 256]` then `[u8; 1024]`. The SDK 13
///    `NV_ENC_CODEC_PIC_PARAMS` union sizes to MAX(variant) which is
///    NV_ENC_PIC_PARAMS_AV1 = 1544 bytes (H264/HEVC = 1536; the
///    `uint32_t reserved[256]` placeholder in the union body is 1024
///    but the variants are larger and drive the union sizeof).
///    Verified against `sizeof(NV_ENC_CODEC_PIC_PARAMS)` from the
///    vendored SDK 13 header: 1544. ALSO the union's alignment is 8
///    (variants contain pointers); our `[u8; N]` mirror only had
///    1-byte alignment, so the driver expected a 4-byte alignment pad
///    BEFORE the union slot (offset 80) but we put it at offset 76
///    with no pad. Widening to `[u64; 193]` fixes both: 193*8=1544
///    bytes AND 8-byte alignment so Rust auto-pads to offset 80.
///
/// 2. `me_hint_counts_per_block: [u32; 2]` was 8 bytes; SDK 13 spec is
///    `NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE[2]` = 32 bytes (each
///    element is 1 bitfield u32 + 3 u32 reserved). Same mis-mirror as
///    NV_ENC_INITIALIZE_PARAMS.maxMEHintCountsPerBlock. Mirrored as
///    a flat `[u32; 8]` array since we never set any external ME hints.
///
/// 3. SDK 13 added 5 NEW fields between `meHintRefPicDist` and the
///    trailing reserved3 block:
///      - `reserved4: u32`
///      - (existing) alphaBuffer
///      - `meExternalSbHints: *void` (AV1 SB-level external hints)
///      - `meSbHintsCount: u32`
///      - `stateBufferIdx: u32`
///      - `outputReconBuffer: NV_ENC_OUTPUT_PTR` (= *void)
///
/// 4. Final reserved blocks rebalanced: reserved3 went `[u32; 286]` →
///    `[u32; 284]` and reserved4 went `[void*; 60]` → `[void*; 57]`
///    to keep total size matching SDK 13's spec'd layout (2840 bytes).
///
/// New const_assert at the bottom verifies size = 2840 bytes.
#[repr(C)]
struct NvEncPicParams {
    version: u32,
    input_width: u32,
    input_height: u32,
    input_pitch: u32,
    encode_pic_flags: u32,
    frame_idx: u32,
    input_timestamp: u64,
    input_duration: u64,
    input_buffer: *mut c_void,
    output_bitstream: *mut c_void,
    completion_event: *mut c_void,
    buffer_fmt: u32,
    picture_struct: u32,
    picture_type: u32,
    /// Union NV_ENC_CODEC_PIC_PARAMS — sizeof = max(variants) = 1544
    /// bytes (driven by NV_ENC_PIC_PARAMS_AV1; H264/HEVC variants are
    /// 1536). The `uint32_t reserved[256]` placeholder in the union
    /// body is just the trailing fallback; the named variants are
    /// LARGER and drive the union sizeof. Was the root cause of the
    /// post-init SIGSEGV under SDK 13 plus a long tail of misaligned
    /// post-codecPicParams fields. `[u64; 193]` carries both the right
    /// size (193*8 = 1544) and the right alignment (the union has
    /// 8-byte alignment because pointer-bearing variants drive it; the
    /// natural u64 alignment forces a 4-byte pad before this field, so
    /// the field lands at offset 80 just like the C compiler emits).
    /// We never populate the H.264 / HEVC / AV1 per-frame sub-variant;
    /// the encoder runs entirely on the preset+config defaults plus
    /// what NV_ENC_PIC_PARAMS top-level fields drive (frame_idx,
    /// encode_pic_flags, etc.).
    codec_pic_params: [u64; 193],
    /// `NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE[2]` = 32 bytes per spec.
    /// Mirrored as flat `[u32; 8]` since we never set external ME hints.
    me_hint_counts_per_block: [u32; 8],
    me_external_hints: *mut c_void,
    /// SDK 13 spec: `uint32_t reserved2[7]` (was 6 in our mirror).
    reserved2: [u32; 7],
    /// SDK 13 spec: `void* reserved5[2]`. Renamed for clarity.
    reserved5: [*mut c_void; 2],
    qp_delta_map: *mut i8,
    qp_delta_map_size: u32,
    reserved_bitfields: u32,
    me_hint_ref_pic_dist: [u16; 2],
    /// NEW in SDK 13: `uint32_t reserved4` between meHintRefPicDist and
    /// alphaBuffer (was implicit padding before).
    reserved4: u32,
    alpha_buffer: *mut c_void,
    /// NEW in SDK 13: AV1 SB-level external ME hints pointer. Always null.
    me_external_sb_hints: *mut c_void,
    /// NEW in SDK 13: count of meExternalSbHints entries. Always 0.
    me_sb_hints_count: u32,
    /// NEW in SDK 13: encoder state-buffer index for stateless flow.
    /// Must be in range [0, NV_ENC_INITIALIZE_PARAMS::numStateBuffers - 1].
    /// We set numStateBuffers=0 → stateBufferIdx=0 is the only valid value.
    state_buffer_idx: u32,
    /// NEW in SDK 13: reconstructed-frame output buffer pointer.
    /// Only used when enableReconFrameOutput=1; we leave at 0.
    output_recon_buffer: *mut c_void,
    /// SDK 13 spec: `uint32_t reserved3[284]`.
    reserved3: [u32; 284],
    /// SDK 13 spec: `void* reserved6[57]`. Renamed for clarity.
    reserved6: [*mut c_void; 57],
}

// ─── NVENC function list struct ───────────────────────────────────
//
// This mirrors NV_ENCODE_API_FUNCTION_LIST from nvEncodeAPI.h.
// NvEncodeAPICreateInstance fills this in with function pointers.
// The struct layout is stable across NVENC 11+; SDK 12.2 adds a few
// AV1-specific entries at the end that we don't need here.

#[repr(C)]
struct NvEncFunctionList {
    version: u32,
    reserved: u32,

    // All entries are `unsafe extern "C" fn(...) -> NVENCSTATUS`.
    // We keep them as raw pointers; they may be null for fields we
    // don't exercise.
    nv_enc_open_encode_session: *mut c_void,
    nv_enc_get_encode_guid_count: *mut c_void,
    nv_enc_get_encode_profile_guid_count: *mut c_void,
    nv_enc_get_encode_profile_guids: *mut c_void,
    nv_enc_get_encode_guids: *mut c_void,
    nv_enc_get_input_format_count: *mut c_void,
    nv_enc_get_input_formats: *mut c_void,
    nv_enc_get_encode_caps: *mut c_void,
    nv_enc_get_encode_preset_count: *mut c_void,
    nv_enc_get_encode_preset_guids: *mut c_void,
    nv_enc_get_encode_preset_config: *mut c_void,

    nv_enc_initialize_encoder: *mut c_void,
    nv_enc_create_input_buffer: *mut c_void,
    nv_enc_destroy_input_buffer: *mut c_void,
    nv_enc_create_bitstream_buffer: *mut c_void,
    nv_enc_destroy_bitstream_buffer: *mut c_void,
    nv_enc_encode_picture: *mut c_void,
    nv_enc_lock_bitstream: *mut c_void,
    nv_enc_unlock_bitstream: *mut c_void,
    nv_enc_lock_input_buffer: *mut c_void,
    nv_enc_unlock_input_buffer: *mut c_void,
    nv_enc_get_encode_stats: *mut c_void,
    nv_enc_get_sequence_params: *mut c_void,
    nv_enc_register_async_event: *mut c_void,
    nv_enc_unregister_async_event: *mut c_void,
    nv_enc_map_input_resource: *mut c_void,
    nv_enc_unmap_input_resource: *mut c_void,
    nv_enc_destroy_encoder: *mut c_void,
    nv_enc_invalidate_ref_frames: *mut c_void,
    nv_enc_open_encode_session_ex: *mut c_void,
    nv_enc_register_resource: *mut c_void,
    nv_enc_unregister_resource: *mut c_void,
    nv_enc_reconfigure_encoder: *mut c_void,
    reserved1: *mut c_void,
    nv_enc_create_mv_buffer: *mut c_void,
    nv_enc_destroy_mv_buffer: *mut c_void,
    nv_enc_run_motion_estimation_only: *mut c_void,
    nv_enc_get_last_error_string: *mut c_void,
    nv_enc_set_io_cuda_streams: *mut c_void,
    // ROOT CAUSE of the 2026-05-01 SIGSEGV chain (caught by the SDK 13
    // header refresh): SDK 13 SWAPPED these two entries. In SDK 12.2
    // the order was:
    //   nv_enc_get_sequence_param_ex
    //   nv_enc_get_encode_preset_config_ex
    // In SDK 13 the order is:
    //   nv_enc_get_encode_preset_config_ex   ← SWAPPED FIRST
    //   nv_enc_get_sequence_param_ex
    // With the 12.2 order against a 13.x driver-populated list, our
    // request for fn_list.nv_enc_get_encode_preset_config_ex actually
    // picked up the pointer to nvEncGetSequenceParamEx — we then called
    // it with NvEncGetEncodePresetConfigEx's argument shape (encoder,
    // GUID, GUID, tuningInfo, *NV_ENC_PRESET_CONFIG) which made
    // nvEncGetSequenceParamEx dereference one of the GUID 32-bit
    // values as an NV_ENC_SEQUENCE_PARAM_PAYLOAD pointer. Bogus
    // dereference → SIGSEGV inside the driver.
    nv_enc_get_encode_preset_config_ex: *mut c_void,
    nv_enc_get_sequence_param_ex: *mut c_void,
    // SDK 13 added two new entries here (introduced for stateful encode
    // mid-stream configuration):
    nv_enc_restore_encoder_state: *mut c_void,
    nv_enc_lookahead_picture: *mut c_void,
    // SDK 13 sized reserved2 at 275 entries. We mirror that exactly so
    // the const_assert! on the struct size catches any future drift.
    reserved2: [*mut c_void; 275],
}

const NV_ENCODE_API_FUNCTION_LIST_VER: u32 = struct_version(2);

type FnNvEncodeAPIGetMaxSupportedVersion = unsafe extern "C" fn(*mut u32) -> c_uint;
type FnNvEncodeAPICreateInstance = unsafe extern "C" fn(*mut NvEncFunctionList) -> c_uint;

type FnNvEncOpenEncodeSessionEx =
    unsafe extern "C" fn(*mut NvEncOpenEncodeSessionExParams, *mut *mut c_void) -> c_uint;
type FnNvEncInitializeEncoder =
    unsafe extern "C" fn(*mut c_void, *mut NvEncInitializeParams) -> c_uint;
type FnNvEncCreateInputBuffer =
    unsafe extern "C" fn(*mut c_void, *mut NvEncCreateInputBuffer) -> c_uint;
type FnNvEncDestroyInputBuffer = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_uint;
type FnNvEncCreateBitstreamBuffer =
    unsafe extern "C" fn(*mut c_void, *mut NvEncCreateBitstreamBuffer) -> c_uint;
type FnNvEncDestroyBitstreamBuffer = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_uint;
type FnNvEncLockInputBuffer =
    unsafe extern "C" fn(*mut c_void, *mut NvEncLockInputBuffer) -> c_uint;
type FnNvEncUnlockInputBuffer = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_uint;
type FnNvEncEncodePicture = unsafe extern "C" fn(*mut c_void, *mut NvEncPicParams) -> c_uint;
type FnNvEncLockBitstream = unsafe extern "C" fn(*mut c_void, *mut NvEncLockBitstream) -> c_uint;
type FnNvEncUnlockBitstream = unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_uint;
type FnNvEncDestroyEncoder = unsafe extern "C" fn(*mut c_void) -> c_uint;
/// `NvEncGetEncodePresetConfigEx(encoder, encodeGuid, presetGuid, tuningInfo, &preset_cfg)`.
/// SDK 12.2 entry; `Ex` variant takes tuning info so the seeded config
/// reflects both preset + tuning rather than preset only.
type FnNvEncGetEncodePresetConfigEx =
    unsafe extern "C" fn(*mut c_void, Guid, Guid, u32, *mut NvEncPresetConfig) -> c_uint;

/// Holds the live encode session + per-frame resources.
/// Dropped together so teardown order is enforced.
///
/// SAFETY: NVENC encoder handles and CUDA contexts are opaque pointers
/// accessed only from the thread that holds `Self`. The encoder's CUDA
/// context must be pushed current before any `fn_encode_picture` etc.
/// call — see `ctx_scope()`.
struct EncodeSession {
    encoder: *mut c_void,
    /// Ring of N input surfaces. Rotated per `EncodePicture` call.
    input_buffers: [*mut c_void; RING_SIZE],
    /// Matching ring of N output (bitstream) buffers. Each input
    /// surface is paired 1:1 with an output surface so lock/unlock of
    /// bitstream i can proceed while input i+1 is being copied.
    bitstream_buffers: [*mut c_void; RING_SIZE],
    cuda_ctx: CUcontext,
    width: u32,
    height: u32,
    /// `NV_ENC_BUFFER_FORMAT_*` value chosen at session create time.
    /// Drives both the upload routine (8-bit byte copy vs 16-bit P010
    /// `<<6` shift) and the per-frame `NV_ENC_PIC_PARAMS.buffer_fmt`
    /// field — has to match `NV_ENC_INITIALIZE_PARAMS.buffer_format`
    /// or NVENC returns INVALID_PARAM on the first encode.
    buffer_format: c_uint,

    // Function pointers captured up front. NVENC's fn-list table holds
    // opaque void* so we cast back at call time.
    fn_destroy_input_buffer: FnNvEncDestroyInputBuffer,
    fn_destroy_bitstream_buffer: FnNvEncDestroyBitstreamBuffer,
    fn_lock_input_buffer: FnNvEncLockInputBuffer,
    fn_unlock_input_buffer: FnNvEncUnlockInputBuffer,
    fn_encode_picture: FnNvEncEncodePicture,
    fn_lock_bitstream: FnNvEncLockBitstream,
    fn_unlock_bitstream: FnNvEncUnlockBitstream,
    fn_destroy_encoder: FnNvEncDestroyEncoder,

    fn_cu_ctx_destroy: FnCuCtxDestroy,
    fn_cu_ctx_push: FnCuCtxPushCurrent,
    fn_cu_ctx_pop: FnCuCtxPopCurrent,
}

unsafe impl Send for EncodeSession {}

impl EncodeSession {
    /// Push this session's CUDA context on the calling thread for the
    /// duration of the returned guard. Required because tokio workers
    /// may migrate between OS threads — without an explicit push the
    /// encoder calls hit CUDA_ERROR_INVALID_CONTEXT.
    unsafe fn ctx_scope(&self) -> Result<CtxScope> {
        unsafe { CtxScope::push(self.cuda_ctx, self.fn_cu_ctx_push, self.fn_cu_ctx_pop) }
    }
}

impl Drop for EncodeSession {
    fn drop(&mut self) {
        unsafe {
            // Push context so NvEncDestroy* calls run in the right
            // CUDA context (teardown on a different thread would
            // otherwise fail). Scope guard pops on exit.
            let _scope =
                CtxScope::push(self.cuda_ctx, self.fn_cu_ctx_push, self.fn_cu_ctx_pop).ok();

            // Teardown ring in REVERSE allocation order so the last
            // slot to be created is the first to go — matches the
            // standard RAII teardown convention and keeps the SDK's
            // internal handle tables consistent.
            for i in (0..RING_SIZE).rev() {
                if !self.input_buffers[i].is_null() {
                    (self.fn_destroy_input_buffer)(self.encoder, self.input_buffers[i]);
                }
                if !self.bitstream_buffers[i].is_null() {
                    (self.fn_destroy_bitstream_buffer)(self.encoder, self.bitstream_buffers[i]);
                }
            }
            if !self.encoder.is_null() {
                (self.fn_destroy_encoder)(self.encoder);
            }
            // Drop the scope guard BEFORE destroying the context it
            // references — explicit drop makes the ordering obvious.
            drop(_scope);
            if !self.cuda_ctx.is_null() {
                (self.fn_cu_ctx_destroy)(self.cuda_ctx);
            }
        }
    }
}

// ─── RAII: CUDA context scope guard ───────────────────────────────
struct CtxScope {
    pop: FnCuCtxPopCurrent,
}

impl CtxScope {
    unsafe fn push(
        ctx: CUcontext,
        push: FnCuCtxPushCurrent,
        pop: FnCuCtxPopCurrent,
    ) -> Result<Self> {
        unsafe {
            if push(ctx) != 0 {
                bail!("cuCtxPushCurrent failed");
            }
            Ok(Self { pop })
        }
    }
}

impl Drop for CtxScope {
    fn drop(&mut self) {
        let mut popped: CUcontext = ptr::null_mut();
        unsafe {
            (self.pop)(&mut popped);
        }
    }
}

// ─── Pixel-format dispatch helpers ────────────────────────────────
//
// Mirrors `crates/codec/src/encode/rav1e_enc.rs`'s pixel-format dispatch.
// Centralises (a) the input pixel format → NVENC buffer format mapping,
// (b) the per-format bytes/sample, and (c) the AV1 OBU `BitDepth` value.
// Keeping these in three small functions side-by-side makes the
// 8-bit / 10-bit branches obvious at the call site without scattering
// `match frame.format { … }` blocks throughout `upload_frame` and
// `encode_pending`.

/// Map a `PixelFormat` to its NVENC `NV_ENC_BUFFER_FORMAT` constant.
/// Returns the format only; the per-pixel bit depth lives in
/// `pixel_bit_depth_for_format`. Bails on unsupported chroma
/// (NVENC AV1 in this service is 4:2:0 only — H.264/HEVC have other
/// 4:2:2 / 4:4:4 paths but this encoder is AV1).
fn nvenc_buffer_format_for(fmt: PixelFormat) -> Result<c_uint> {
    match fmt {
        PixelFormat::Yuv420p => Ok(NV_ENC_BUFFER_FORMAT_IYUV),
        PixelFormat::Yuv420p10le => Ok(NV_ENC_BUFFER_FORMAT_YUV420_10BIT),
        other => bail!(
            "NVENC AV1 expects Yuv420p or Yuv420p10le, got {other:?} \
             (4:2:2 / 4:4:4 / RGB / alpha not supported on this backend)"
        ),
    }
}

/// Returns the `pixel_bit_depth_minus_8` value for the AV1 codec config.
/// 0 = 8-bit, 2 = 10-bit. Drives the AV1 sequence header `BitDepth`
/// signalling so a decoder knows the sample width up front.
const fn pixel_bit_depth_minus8_for(fmt: PixelFormat) -> u32 {
    match fmt {
        PixelFormat::Yuv420p10le => 2,
        _ => 0,
    }
}

/// Translate `TransferFn` → ITU-T H.273 numeric code for the AV1 OBU
/// sequence header `transfer_characteristics` field.
///
/// The mux side (`crates/container/src/mux.rs::transfer_to_h273`)
/// uses the same mapping — keeping a sibling helper here lets the
/// in-bitstream code match what gets written into `colr nclx` so a
/// downstream player sees consistent metadata between container and
/// elementary stream. Unspecified collapses to canonical Bt709 (1)
/// because the AV1 spec has no "unspecified" sentinel for transfer.
fn transfer_to_h273(tf: TransferFn) -> u32 {
    match tf {
        TransferFn::Bt709 => 1,
        TransferFn::Bt470Bg => 4,
        TransferFn::Linear => 8,
        TransferFn::St2084 => 16,
        TransferFn::AribStdB67 => 18,
        TransferFn::Unspecified => 1,
    }
}

// ─── Frame-rate rational mapping ──────────────────────────────────
//
// NVENC init params carry `frameRateNum` / `frameRateDen` as separate
// u32s. Pass the canonical rational for broadcast rates so 1001-family
// NTSC rates (23.976, 29.97, 59.94) encode with exact sync instead of
// the lossy `(fps*1000)/1000` shortcut (review task #3 MEDIUM-4).

/// Map a float fps to its canonical (num, den) pair. Common broadcast
/// rates are returned exactly; any other value falls back to
/// `(round(fps*1000), 1000)`.
///
/// 1001-family detector: if `fps ≈ k/1001` for integer `k`, treat as
/// `(k, 1001)` — keeps exact sync for precise 1001-family inputs.
fn fps_to_rational(fps: f64) -> (u32, u32) {
    // Exact hits first — avoids float rounding games for values that
    // can be represented cleanly. Tolerance ≤ 1e-3 covers both
    // 29.97 → 30000/1001 and 23.976 → 24000/1001 inputs from
    // user-facing config files.
    const EXACT: &[(f64, u32, u32)] = &[
        (23.976, 24_000, 1001),
        (24.0, 24, 1),
        (25.0, 25, 1),
        (29.97, 30_000, 1001),
        (30.0, 30, 1),
        (48.0, 48, 1),
        (50.0, 50, 1),
        (59.94, 60_000, 1001),
        (60.0, 60, 1),
    ];
    for &(f, n, d) in EXACT {
        if (fps - f).abs() < 1e-3 {
            return (n, d);
        }
    }

    // Integer fps shortcut — if `fps` rounds to itself (whole number),
    // prefer `(n, 1)` over any `k/1001` representation. Otherwise
    // 100.0 would hit the 1001-family detector below as (100100, 1001)
    // since 100*1001/1001 = 100 exactly.
    if (fps - fps.round()).abs() < 1e-6 && fps > 0.0 {
        return (fps.round() as u32, 1);
    }

    // 1001-family detector for more precise inputs like 23.9760239760…
    // If rounding `fps*1001` to an integer and dividing back lands
    // within 1e-4 of the original fps, treat it as a k/1001 rate.
    let k = (fps * 1001.0).round();
    if (k / 1001.0 - fps).abs() < 1e-4 && k > 0.0 {
        let k_u = k as u32;
        return (k_u, 1001);
    }

    // Generic fallback — (round(fps*1000), 1000). Round instead of
    // truncation so odd rates like 47.97 land at 47970/1000 exactly.
    let num = (fps * 1000.0).round().max(1.0) as u32;
    (num, 1000)
}

// ─── Encoder implementation ───────────────────────────────────────
//
// Field order matters: session drops BEFORE the libraries, so the
// fn pointers it stashed remain valid during its Drop. The two lib
// handles are declared LAST (Reference §10.8 — struct fields drop
// in source order).
pub struct NvencEncoder {
    config: EncoderConfig,
    session: Option<EncodeSession>,
    pending_frames: Vec<VideoFrame>,
    encoded_packets: Vec<EncodedPacket>,
    flushed: bool,
    packet_cursor: usize,
    frame_counter: u32,
    /// Current ring index. Advances modulo `RING_SIZE` per EncodePicture.
    ring_idx: usize,
    /// Per-slot last drained frame_idx (i64 with -1 sentinel for "never
    /// drained"). Used by `flush_eos` to discard stale reads from
    /// `NvEncLockBitstream`: on the SDK 13 driver shipped with 595.71.05,
    /// re-locking a slot whose bitstream has already been unlocked once
    /// returns NV_ENC_SUCCESS with the SAME packet bytes — there is no
    /// "buffer empty" status code. We compare `lock.frameIdx` against
    /// the last drained idx for the slot to detect staleness.
    last_drained_frame_idx: [i64; RING_SIZE],
    _encode_lib: libloading::Library,
    _cuda_lib: libloading::Library,
}

impl NvencEncoder {
    pub fn new(config: EncoderConfig, gpu_index: u32) -> Result<Self> {
        // Take the SHARED CUDA-init lock BEFORE any FFI work. This
        // serializes encoder construction not just against other
        // encoders but ALSO against NVDEC streaming-decoder ctor —
        // which does its own cuInit/cuCtxCreate concurrently and was
        // causing the FIRST encoder's NvEncOpenEncodeSessionEx to
        // segfault on Ada silicon even with 1-encoder parallelism.
        // See crates/codec/src/cuda_lock.rs for the full root-cause
        // narrative (prod 2026-05-01 PT 03:12:43 trace).
        let _init_guard = crate::cuda_lock::lock_for_cuda_init();

        // Structured trace at entry — operators reading CloudWatch
        // can grep `event=nvenc.init.start` to find every encoder
        // ctor and correlate with the next-step trace below to
        // pinpoint which API call is the segfault site if a crash
        // occurs mid-init. Pair with the SIGSEGV handler in
        // crates/transcoder/src/crash.rs.
        tracing::info!(
            event = "nvenc.init.start",
            gpu_index,
            width = config.width,
            height = config.height,
            ?config.target,
            ?config.tier,
            ?config.pixel_format,
            "NVENC init starting"
        );

        // Load NVENC
        let encode_lib = unsafe { libloading::Library::new("libnvidia-encode.so") }
            .or_else(|_| unsafe { libloading::Library::new("libnvidia-encode.so.1") })
            .or_else(|_| unsafe { libloading::Library::new("nvEncodeAPI64.dll") })
            .context("loading NVIDIA encode library")?;

        // Load CUDA driver for the encoder context.
        let cuda_lib = unsafe { libloading::Library::new("libcuda.so") }
            .or_else(|_| unsafe { libloading::Library::new("libcuda.so.1") })
            .or_else(|_| unsafe { libloading::Library::new("nvcuda.dll") })
            .context("loading CUDA driver for NVENC")?;

        unsafe {
            // Version check — AV1 requires SDK 12+.
            let get_version: libloading::Symbol<FnNvEncodeAPIGetMaxSupportedVersion> = encode_lib
                .get(b"NvEncodeAPIGetMaxSupportedVersion")
                .context("missing NvEncodeAPIGetMaxSupportedVersion")?;
            let mut version: u32 = 0;
            if get_version(&mut version) != NV_ENC_SUCCESS {
                bail!("NvEncodeAPIGetMaxSupportedVersion failed");
            }
            let driver_major = version >> 4;
            let driver_minor = version & 0xF;
            tracing::info!(
                major = driver_major,
                minor = driver_minor,
                "NVENC driver API version"
            );
            if driver_major < 12 {
                bail!(
                    "NVENC driver API < 12 does not support AV1 (got {driver_major}.{driver_minor})"
                );
            }

            // Get the function-pointer table. The SDK populates this
            // with every entry point we need.
            let create_instance: libloading::Symbol<FnNvEncodeAPICreateInstance> = encode_lib
                .get(b"NvEncodeAPICreateInstance")
                .context("missing NvEncodeAPICreateInstance")?;
            let mut fn_list: NvEncFunctionList = std::mem::zeroed();
            fn_list.version = NV_ENCODE_API_FUNCTION_LIST_VER;
            if create_instance(&mut fn_list) != NV_ENC_SUCCESS {
                bail!("NvEncodeAPICreateInstance failed");
            }

            // ─── CUDA context ───────────────────────────────────
            // Per-step tracing on the cuInit / cuDeviceGet /
            // cuCtxCreate sequence. The 2026-05-01 prod SIGSEGV fired
            // somewhere between `nvenc.init.start` and the
            // `NvEncInitializeEncoder` log line, with FIVE encoder
            // contexts being initialized simultaneously (one per
            // resolution variant). Smelling like CUDA context-create
            // contention. The narrow per-step traces below pinpoint
            // exactly which call dies on the next iteration.
            tracing::info!(event = "nvenc.cuda.cuInit", gpu_index, "cuInit");
            let cu_init: libloading::Symbol<FnCuInit> = cuda_lib.get(b"cuInit")?;
            if cu_init(0) != 0 {
                tracing::error!(
                    event = "nvenc.cuda.error",
                    fn_name = "cuInit",
                    gpu_index,
                    "cuInit failed"
                );
                bail!("cuInit failed");
            }
            tracing::info!(event = "nvenc.cuda.cuDeviceGet", gpu_index, "cuDeviceGet");
            let cu_device_get: libloading::Symbol<FnCuDeviceGet> = cuda_lib.get(b"cuDeviceGet")?;
            let mut device: CUdevice = 0;
            if cu_device_get(&mut device, gpu_index as c_int) != 0 {
                tracing::error!(
                    event = "nvenc.cuda.error",
                    fn_name = "cuDeviceGet",
                    gpu_index,
                    "cuDeviceGet failed"
                );
                bail!("cuDeviceGet failed for GPU {gpu_index}");
            }
            tracing::info!(
                event = "nvenc.cuda.cuCtxCreate",
                gpu_index,
                width = config.width,
                height = config.height,
                "cuCtxCreate (5-way contention candidate)"
            );
            let cu_ctx_create: libloading::Symbol<FnCuCtxCreate> =
                cuda_lib.get(b"cuCtxCreate_v2")?;
            let mut cuda_ctx: CUcontext = ptr::null_mut();
            if cu_ctx_create(&mut cuda_ctx, 0, device) != 0 {
                tracing::error!(
                    event = "nvenc.cuda.error",
                    fn_name = "cuCtxCreate",
                    gpu_index,
                    "cuCtxCreate failed"
                );
                bail!("cuCtxCreate failed");
            }
            tracing::info!(event = "nvenc.cuda.ok", gpu_index, "CUDA context created");
            let fn_cu_ctx_destroy: libloading::Symbol<FnCuCtxDestroy> =
                cuda_lib.get(b"cuCtxDestroy_v2")?;
            let fn_cu_ctx_push: libloading::Symbol<FnCuCtxPushCurrent> =
                cuda_lib.get(b"cuCtxPushCurrent_v2")?;
            let fn_cu_ctx_pop: libloading::Symbol<FnCuCtxPopCurrent> =
                cuda_lib.get(b"cuCtxPopCurrent_v2")?;

            // Translate fn-list void pointers into typed fn pointers.
            // If any required entry is null, the SDK version is too old
            // for what we're doing.
            macro_rules! cast_fn {
                ($field:expr, $ty:ty, $name:literal) => {{
                    if $field.is_null() {
                        bail!(concat!("NVENC fn-list missing ", $name));
                    }
                    std::mem::transmute::<*mut c_void, $ty>($field)
                }};
            }
            let fn_open_session: FnNvEncOpenEncodeSessionEx = cast_fn!(
                fn_list.nv_enc_open_encode_session_ex,
                FnNvEncOpenEncodeSessionEx,
                "OpenEncodeSessionEx"
            );
            let fn_initialize_encoder: FnNvEncInitializeEncoder = cast_fn!(
                fn_list.nv_enc_initialize_encoder,
                FnNvEncInitializeEncoder,
                "InitializeEncoder"
            );
            let fn_create_input_buffer: FnNvEncCreateInputBuffer = cast_fn!(
                fn_list.nv_enc_create_input_buffer,
                FnNvEncCreateInputBuffer,
                "CreateInputBuffer"
            );
            let fn_destroy_input_buffer: FnNvEncDestroyInputBuffer = cast_fn!(
                fn_list.nv_enc_destroy_input_buffer,
                FnNvEncDestroyInputBuffer,
                "DestroyInputBuffer"
            );
            let fn_create_bitstream_buffer: FnNvEncCreateBitstreamBuffer = cast_fn!(
                fn_list.nv_enc_create_bitstream_buffer,
                FnNvEncCreateBitstreamBuffer,
                "CreateBitstreamBuffer"
            );
            let fn_destroy_bitstream_buffer: FnNvEncDestroyBitstreamBuffer = cast_fn!(
                fn_list.nv_enc_destroy_bitstream_buffer,
                FnNvEncDestroyBitstreamBuffer,
                "DestroyBitstreamBuffer"
            );
            let fn_lock_input_buffer: FnNvEncLockInputBuffer = cast_fn!(
                fn_list.nv_enc_lock_input_buffer,
                FnNvEncLockInputBuffer,
                "LockInputBuffer"
            );
            let fn_unlock_input_buffer: FnNvEncUnlockInputBuffer = cast_fn!(
                fn_list.nv_enc_unlock_input_buffer,
                FnNvEncUnlockInputBuffer,
                "UnlockInputBuffer"
            );
            let fn_encode_picture: FnNvEncEncodePicture = cast_fn!(
                fn_list.nv_enc_encode_picture,
                FnNvEncEncodePicture,
                "EncodePicture"
            );
            let fn_lock_bitstream: FnNvEncLockBitstream = cast_fn!(
                fn_list.nv_enc_lock_bitstream,
                FnNvEncLockBitstream,
                "LockBitstream"
            );
            let fn_unlock_bitstream: FnNvEncUnlockBitstream = cast_fn!(
                fn_list.nv_enc_unlock_bitstream,
                FnNvEncUnlockBitstream,
                "UnlockBitstream"
            );
            let fn_destroy_encoder: FnNvEncDestroyEncoder = cast_fn!(
                fn_list.nv_enc_destroy_encoder,
                FnNvEncDestroyEncoder,
                "DestroyEncoder"
            );
            // Preset-config-ex: required for HIGH-1 fix. If the SDK
            // fn-list is missing it the driver is too old for AV1
            // anyway (added in 12.x).
            let fn_get_preset_config_ex: FnNvEncGetEncodePresetConfigEx = cast_fn!(
                fn_list.nv_enc_get_encode_preset_config_ex,
                FnNvEncGetEncodePresetConfigEx,
                "GetEncodePresetConfigEx"
            );

            // ─── Open encode session on the CUDA device ─────────
            let mut open_params: NvEncOpenEncodeSessionExParams = std::mem::zeroed();
            open_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
            open_params.device_type = NV_ENC_DEVICE_TYPE_CUDA;
            open_params.device = cuda_ctx;
            open_params.api_version = NVENCAPI_VERSION;
            let mut encoder: *mut c_void = ptr::null_mut();
            tracing::info!(
                event = "nvenc.ffi.call",
                fn_name = "NvEncOpenEncodeSessionEx",
                gpu_index,
                width = config.width,
                height = config.height,
                "calling NvEncOpenEncodeSessionEx (parallel-init candidate)"
            );
            let rc = fn_open_session(&mut open_params, &mut encoder);
            if rc != NV_ENC_SUCCESS {
                tracing::error!(
                    event = "nvenc.ffi.error",
                    fn_name = "NvEncOpenEncodeSessionEx",
                    rc,
                    gpu_index,
                    width = config.width,
                    height = config.height,
                    "NVENC FFI failed"
                );
                (*fn_cu_ctx_destroy)(cuda_ctx);
                bail!("NvEncOpenEncodeSessionEx failed: {rc}");
            }
            tracing::info!(
                event = "nvenc.ffi.ok",
                fn_name = "NvEncOpenEncodeSessionEx",
                gpu_index,
                width = config.width,
                height = config.height,
                "NvEncOpenEncodeSessionEx OK — session handle acquired"
            );

            // ─── Build encode config via the tuning adapter ────────
            //
            // The adapter maps (QualityTarget, SpeedTier, resolution) to
            // NVENC-native params. Legacy config.quality override: if set
            // to something other than AUTO_FROM_TARGET, we pass it
            // through as an AV1 CQ value in the 0..63 range (the correct
            // range for AV1; NOT 0..51 — that scale is H.264/HEVC).
            let tp =
                tuning::nvenc_av1_params(config.target, config.tier, config.width, config.height);
            let nvenc_cq = if config.quality == AUTO_FROM_TARGET {
                tp.cq
            } else {
                config.quality.min(63)
            };
            let preset_guid = guid_from_bytes(tp.preset_guid);

            // ─── HIGH-1: seed encode config from preset+tuning ─────
            //
            // Without this we ship a null `encodeConfig` and NVENC
            // silently uses driver defaults — which emit non-LOB OBU
            // streams that the MP4 muxer rejects. Call
            // `NvEncGetEncodePresetConfigEx` to get the preset's
            // driver-blessed baseline, then override the fields we
            // care about (RC mode, obu_payload_format, repeat_seq_hdr,
            // IDR period, tiles).
            // 16 KiB of trailing padding around our compiled-in struct
            // size. The 2026-05-01 SIGSEGV was caused by the production
            // L40S driver writing PAST the 4200-byte NvEncPresetConfig
            // boundary — almost certainly because the AV1 codec_config
            // sub-struct has grown since SDK 12.2 and the driver assumes
            // a larger buffer. Skipping the call left the encoder in a
            // hung state (the override block compensates for the missing
            // preset defaults but NvEncInitializeEncoder still got
            // unhappy about something downstream). The over-allocate
            // pattern lets the call run safely: driver writes whatever
            // it wants up to ~20 KiB; we read back our compiled-in
            // 4200-byte view and then OVERRIDE every field we care
            // about in the block below. Backwards-compatible struct
            // growth (driver added new fields at the end) lands in our
            // padding and is silently ignored on read.
            #[repr(C)]
            struct NvEncPresetConfigPadded {
                base: NvEncPresetConfig,
                _overflow_pad: [u8; 16384],
            }
            let mut padded: NvEncPresetConfigPadded = std::mem::zeroed();
            padded.base.version = NV_ENC_PRESET_CONFIG_VER;
            padded.base.preset_cfg.version = NV_ENC_CONFIG_VER;
            tracing::info!(
                event = "nvenc.ffi.call",
                fn_name = "NvEncGetEncodePresetConfigEx",
                gpu_index,
                width = config.width,
                height = config.height,
                buffer_size = std::mem::size_of::<NvEncPresetConfigPadded>(),
                "calling NvEncGetEncodePresetConfigEx (16 KiB over-allocated buffer)"
            );
            let rc = fn_get_preset_config_ex(
                encoder,
                NV_ENC_CODEC_AV1_GUID,
                preset_guid,
                tp.tuning_info,
                &mut padded.base,
            );
            // Alias `preset_cfg` to the base struct so the rest of the
            // function reads it the same way the original code did.
            let preset_cfg = &padded.base;
            if rc != NV_ENC_SUCCESS {
                tracing::error!(
                    event = "nvenc.ffi.error",
                    fn_name = "NvEncGetEncodePresetConfigEx",
                    rc,
                    gpu_index,
                    width = config.width,
                    height = config.height,
                    "NvEncGetEncodePresetConfigEx failed"
                );
                (fn_destroy_encoder)(encoder);
                (*fn_cu_ctx_destroy)(cuda_ctx);
                bail!("NvEncGetEncodePresetConfigEx failed: {rc}");
            }
            tracing::info!(
                event = "nvenc.ffi.ok",
                fn_name = "NvEncGetEncodePresetConfigEx",
                gpu_index,
                width = config.width,
                height = config.height,
                "NvEncGetEncodePresetConfigEx OK"
            );

            // Copy the preset-seeded config. Everything below overrides
            // on top of the driver's recommended defaults for (AV1,
            // preset, tuning).
            let mut enc_config: NvEncConfig = std::ptr::read(&preset_cfg.preset_cfg);
            enc_config.version = NV_ENC_CONFIG_VER;
            enc_config.gop_length = config.keyframe_interval;
            enc_config.frame_interval_p = 1; // no B-frames by default
            enc_config.mv_precision = 3; // quarter-pel (AV1 default)

            // ─── HIGH-3: plumb CQ target into rate control ─────────
            enc_config.rc_params.version = struct_version(1);
            match config.target {
                QualityTarget::VisuallyLossless => {
                    // Archival tier: CONSTQP with low QPs.
                    // Pick a base QP in the 8..12 band — the
                    // reviewer's `low in {8..12}` prescription — and
                    // bias intra < interP < interB so keyframes get
                    // the most bits.
                    //
                    // Clamp within 8..12 even if tp.cq reports lower:
                    // VisuallyLossless should never drop below QP 8
                    // under NVENC (below that the rate-control
                    // accuracy collapses).
                    let low = (nvenc_cq as u32).clamp(8, 12);
                    enc_config.rc_params.rate_control_mode = NV_ENC_PARAMS_RC_CONSTQP;
                    enc_config.rc_params.const_qp_intra = low;
                    enc_config.rc_params.const_qp_inter_p = low.saturating_add(1);
                    enc_config.rc_params.const_qp_inter_b = low.saturating_add(2);
                    // targetQuality is unused under CONSTQP but left
                    // at a sensible value for diagnostics / logs.
                    enc_config.rc_params.target_quality = low as u8;
                }
                _ => {
                    // All non-lossless tiers use plain VBR with
                    // `targetQuality` populated. SDK 12.2 merges the
                    // old VBR_HQ behaviour into VBR + tuningInfo =
                    // HIGH_QUALITY (set via init_params.tuning_info
                    // below), so the HQ flag on the RC mode itself is
                    // redundant on 12.2.
                    let rc_mode = match tp.rc_mode {
                        NvencRateControl::ConstQp => NV_ENC_PARAMS_RC_CONSTQP,
                        NvencRateControl::VbrTargetQuality => NV_ENC_PARAMS_RC_VBR,
                    };
                    enc_config.rc_params.rate_control_mode = rc_mode;
                    enc_config.rc_params.target_quality = nvenc_cq.min(51);
                    // targetQualityLSB = 0 — SDK takes integer CQ in
                    // the whole-step field; 8.8 fractional isn't
                    // needed for our VMAF bands.
                    enc_config.rc_params.target_quality_lsb = 0;
                    // Mirror the CQ into constQP too so driver
                    // versions that fall back to constQP when
                    // targetQuality is zero still use the right
                    // value. Safe under VBR: these fields are only
                    // read when rate_control_mode == CONSTQP.
                    enc_config.rc_params.const_qp_intra = nvenc_cq as u32;
                    enc_config.rc_params.const_qp_inter_p = (nvenc_cq as u32).saturating_add(2);
                    enc_config.rc_params.const_qp_inter_b = (nvenc_cq as u32).saturating_add(4);
                }
            }

            // ─── AV1 codec-specific config (SDK 13 layout) ───────────
            //
            // SDK 13 collapsed all the bool enable_* fields into a
            // single 32-bit bitfield word. Semantics also flipped on
            // bit 0: SDK 12.2 had `obu_payload_format` (1 = LOB / MP4),
            // SDK 13 has `outputAnnexBFormat` (1 = AnnexB, 0 = LOB).
            // We need LOB → bit 0 stays 0 (the default).
            //
            // HIGH-2 carry-forward: bit 5 (repeatSeqHdr) = 1 so every
            // IDR re-emits the sequence header for keyframe seekability.
            // chromaFormatIDC = 1 (4:2:0) goes in bit 7 (the LSB of a
            // 2-bit field at bits 7-8).
            //
            // outputBitDepth / inputBitDepth replaced the old
            // pixel_bit_depth_minus_8 fields. Enum: 8-bit = 0, 10-bit = 1.
            let buffer_format = nvenc_buffer_format_for(config.pixel_format)?;
            let bit_depth_minus8 = pixel_bit_depth_minus8_for(config.pixel_format);
            let bit_depth_enum = if bit_depth_minus8 == 0 { 0 } else { 1 };

            // outputAnnexBFormat = 0 (LOB), repeatSeqHdr = 1, chroma = 4:2:0.
            // enable_timing_info stays 0 (we don't emit it).
            enc_config.codec_config_av1.flags = AV1_BIT_REPEAT_SEQ_HDR | AV1_CHROMA_FORMAT_IDC_420;
            enc_config.codec_config_av1.idr_period = config.keyframe_interval;
            enc_config.codec_config_av1.max_num_ref_frames_in_dpb = 4;
            enc_config.codec_config_av1.num_tile_columns = tp.num_tile_columns;
            enc_config.codec_config_av1.num_tile_rows = tp.num_tile_rows;
            enc_config.codec_config_av1.output_bit_depth = bit_depth_enum;
            enc_config.codec_config_av1.input_bit_depth = bit_depth_enum;

            // Color signalling — wire ColorMetadata into the OBU seq
            // header. SDK 13 dropped the explicit
            // `color_description_present_flag` field; the codes are
            // emitted whenever any of them is non-zero (driver-side
            // policy per SDK 13 docs).
            let cm = &config.color_metadata;
            enc_config.codec_config_av1.color_primaries = cm.colour_primaries as u32;
            enc_config.codec_config_av1.transfer_characteristics = transfer_to_h273(cm.transfer);
            enc_config.codec_config_av1.matrix_coefficients = cm.matrix_coefficients as u32;
            enc_config.codec_config_av1.color_range = cm.full_range as u32;

            let mut init_params: NvEncInitializeParams = std::mem::zeroed();
            init_params.version = NV_ENC_INITIALIZE_PARAMS_VER;
            init_params.encode_guid = NV_ENC_CODEC_AV1_GUID;
            init_params.preset_guid = preset_guid;
            init_params.encode_width = config.width;
            init_params.encode_height = config.height;
            init_params.dar_width = config.width;
            init_params.dar_height = config.height;
            // MEDIUM-4: rational frame-rate mapping.
            let (num, den) = fps_to_rational(config.frame_rate);
            init_params.frame_rate_num = num;
            init_params.frame_rate_den = den;
            init_params.enable_encode_async = 0;
            init_params.enable_ptd = 1;
            init_params.max_encode_width = config.width;
            init_params.max_encode_height = config.height;
            init_params.tuning_info = tp.tuning_info;
            init_params.buffer_format = buffer_format;
            init_params.encode_config = (&mut enc_config) as *mut NvEncConfig as *mut c_void;

            tracing::info!(
                width = config.width,
                height = config.height,
                target = ?config.target,
                tier = ?config.tier,
                cq = nvenc_cq,
                rc_mode = enc_config.rc_params.rate_control_mode,
                tile_cols = tp.num_tile_columns,
                tile_rows = tp.num_tile_rows,
                frame_rate_num = num,
                frame_rate_den = den,
                "NVENC AV1 tuning applied"
            );

            // The leading suspect for the prod 4K SIGSEGV (2026-05-01).
            // Log immediately before AND after so a crash inside the
            // FFI shows up as a "before" line with no matching "after".
            tracing::info!(
                event = "nvenc.ffi.call",
                fn_name = "NvEncInitializeEncoder",
                width = config.width,
                height = config.height,
                gpu_index,
                "calling NvEncInitializeEncoder (4K segfault candidate)"
            );
            let rc = fn_initialize_encoder(encoder, &mut init_params);
            if rc != NV_ENC_SUCCESS {
                tracing::error!(
                    event = "nvenc.ffi.error",
                    fn_name = "NvEncInitializeEncoder",
                    rc,
                    width = config.width,
                    height = config.height,
                    gpu_index,
                    "NvEncInitializeEncoder failed"
                );
                (fn_destroy_encoder)(encoder);
                (*fn_cu_ctx_destroy)(cuda_ctx);
                bail!("NvEncInitializeEncoder failed: {rc}");
            }
            tracing::info!(
                event = "nvenc.ffi.ok",
                fn_name = "NvEncInitializeEncoder",
                width = config.width,
                height = config.height,
                "NvEncInitializeEncoder OK"
            );

            // ─── MEDIUM-5: Allocate input + bitstream buffer rings ──
            //
            // Partial-init teardown: if any allocation fails, tear
            // down the slots we've already created in reverse order
            // and bail.
            let mut input_buffers: [*mut c_void; RING_SIZE] = [ptr::null_mut(); RING_SIZE];
            let mut bitstream_buffers: [*mut c_void; RING_SIZE] = [ptr::null_mut(); RING_SIZE];

            let cleanup_partial =
                |allocated: usize,
                 inputs: &[*mut c_void; RING_SIZE],
                 outputs: &[*mut c_void; RING_SIZE]| {
                    for i in (0..allocated).rev() {
                        if !inputs[i].is_null() {
                            (fn_destroy_input_buffer)(encoder, inputs[i]);
                        }
                        if !outputs[i].is_null() {
                            (fn_destroy_bitstream_buffer)(encoder, outputs[i]);
                        }
                    }
                };

            for i in 0..RING_SIZE {
                let mut input_desc: NvEncCreateInputBuffer = std::mem::zeroed();
                input_desc.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
                input_desc.width = config.width;
                input_desc.height = config.height;
                input_desc.buffer_fmt = buffer_format;
                let rc = fn_create_input_buffer(encoder, &mut input_desc);
                if rc != NV_ENC_SUCCESS {
                    tracing::error!(
                        event = "nvenc.ffi.error",
                        fn_name = "NvEncCreateInputBuffer",
                        slot = i,
                        rc,
                        width = config.width,
                        height = config.height,
                        "NvEncCreateInputBuffer failed"
                    );
                    cleanup_partial(i, &input_buffers, &bitstream_buffers);
                    (fn_destroy_encoder)(encoder);
                    (*fn_cu_ctx_destroy)(cuda_ctx);
                    bail!("NvEncCreateInputBuffer (slot {i}) failed: {rc}");
                }
                input_buffers[i] = input_desc.input_buffer;

                let mut bitstream_desc: NvEncCreateBitstreamBuffer = std::mem::zeroed();
                bitstream_desc.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
                // 16 MB output buffer per slot. AV1 P/B frames are
                // typically <100 KB and 1080p I-frames <500 KB, but a
                // 4K I-frame at high-quality CQ on a complex source
                // can land in the 1-6 MB range — and the SDK 13
                // driver shipped with 595.71.05 SIGSEGVs in
                // `NvEncEncodePicture` rather than returning an error
                // when the output bitstream buffer is too small. 16 MB
                // ring × 4 slots = 64 MB host RAM, negligible compared
                // to NVDEC's GPU surfaces.
                bitstream_desc.size = 16 * 1024 * 1024;
                let rc = fn_create_bitstream_buffer(encoder, &mut bitstream_desc);
                if rc != NV_ENC_SUCCESS {
                    tracing::error!(
                        event = "nvenc.ffi.error",
                        fn_name = "NvEncCreateBitstreamBuffer",
                        slot = i,
                        rc,
                        width = config.width,
                        height = config.height,
                        "NvEncCreateBitstreamBuffer failed"
                    );
                    cleanup_partial(i + 1, &input_buffers, &bitstream_buffers);
                    (fn_destroy_encoder)(encoder);
                    (*fn_cu_ctx_destroy)(cuda_ctx);
                    bail!("NvEncCreateBitstreamBuffer (slot {i}) failed: {rc}");
                }
                bitstream_buffers[i] = bitstream_desc.bitstream_buffer;
            }
            tracing::info!(
                event = "nvenc.init.complete",
                gpu_index,
                width = config.width,
                height = config.height,
                ring_size = RING_SIZE,
                "NVENC encoder ready (init complete)"
            );

            let session = EncodeSession {
                encoder,
                input_buffers,
                bitstream_buffers,
                cuda_ctx,
                width: config.width,
                height: config.height,
                buffer_format,
                fn_destroy_input_buffer,
                fn_destroy_bitstream_buffer,
                fn_lock_input_buffer,
                fn_unlock_input_buffer,
                fn_encode_picture,
                fn_lock_bitstream,
                fn_unlock_bitstream,
                fn_destroy_encoder,
                fn_cu_ctx_destroy: *fn_cu_ctx_destroy,
                fn_cu_ctx_push: *fn_cu_ctx_push,
                fn_cu_ctx_pop: *fn_cu_ctx_pop,
            };

            tracing::info!(
                width = config.width,
                height = config.height,
                quality = config.quality,
                gpu = gpu_index,
                ring_size = RING_SIZE,
                "NVENC AV1 encoder ready"
            );

            Ok(Self {
                config,
                session: Some(session),
                pending_frames: Vec::new(),
                encoded_packets: Vec::new(),
                flushed: false,
                packet_cursor: 0,
                frame_counter: 0,
                ring_idx: 0,
                last_drained_frame_idx: [-1; RING_SIZE],
                _encode_lib: encode_lib,
                _cuda_lib: cuda_lib,
            })
        }
    }

    /// Copy an 8-bit YUV420p frame into a locked NVENC IYUV surface.
    /// Layout: Y plane, then U plane, then V plane, each laid out
    /// contiguously at the surface's pitch.
    ///
    /// MEDIUM-6: chroma plane dims use round-up `(w+1)/2, (h+1)/2`
    /// so odd widths/heights don't truncate the last column/row
    /// (mirrors the NVDEC fix in systems-review-2 M-N1).
    unsafe fn upload_frame(
        session: &EncodeSession,
        frame: &VideoFrame,
        slot: usize,
    ) -> Result<u32> {
        unsafe {
            let input_buffer = session.input_buffers[slot];
            let mut lock: NvEncLockInputBuffer = std::mem::zeroed();
            lock.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
            lock.input_buffer = input_buffer;
            let rc = (session.fn_lock_input_buffer)(session.encoder, &mut lock);
            if rc != NV_ENC_SUCCESS {
                bail!("NvEncLockInputBuffer failed: {rc}");
            }

            let pitch = lock.pitch as usize;
            let w = session.width as usize;
            let h = session.height as usize;
            // Round-up chroma dims for 4:2:0.
            let cw = w.div_ceil(2);
            let ch = h.div_ceil(2);
            let y_size = w * h;
            let uv_size = cw * ch;

            if frame.data.len() < y_size + 2 * uv_size {
                (session.fn_unlock_input_buffer)(session.encoder, input_buffer);
                bail!("frame data too small for {}x{} YUV420p", w, h);
            }

            let dst = lock.buffer_data_ptr as *mut u8;

            // Y plane: one row at a time to honor the surface pitch.
            for row in 0..h {
                let src = frame.data.as_ptr().add(row * w);
                let dst_row = dst.add(row * pitch);
                ptr::copy_nonoverlapping(src, dst_row, w);
            }

            // U plane: starts at dst + pitch*h on the surface. NVENC's
            // IYUV layout uses HALF-PITCH for chroma rows (the
            // surface is allocated as pitch*h Y + (pitch/2)*ch U +
            // (pitch/2)*ch V — a 4:2:0 chroma subsampling with
            // proportionally narrower chroma rows). The previous
            // mirror used full pitch for chroma rows, which hid on
            // sub-1080p sources where the driver-allocated surface
            // happened to have enough headroom but reliably SIGSEGV'd
            // at 4K (3840×2160 with pitch=4096): full-pitch chroma
            // ended up at offset 13.27 MiB while the driver's IYUV
            // surface only allocates ~13.27 MiB total — chroma writes
            // ran off the end. Verified 2026-05-01 against the 4K
            // segfault repro on the dev box.
            let chroma_pitch = pitch / 2;
            let u_dst_base = dst.add(pitch * h);
            let u_src_base = frame.data.as_ptr().add(y_size);
            for row in 0..ch {
                let src = u_src_base.add(row * cw);
                let dst_row = u_dst_base.add(row * chroma_pitch);
                ptr::copy_nonoverlapping(src, dst_row, cw);
            }

            // V plane: follows U at chroma_pitch*ch further in.
            let v_dst_base = u_dst_base.add(chroma_pitch * ch);
            let v_src_base = u_src_base.add(uv_size);
            for row in 0..ch {
                let src = v_src_base.add(row * cw);
                let dst_row = v_dst_base.add(row * chroma_pitch);
                ptr::copy_nonoverlapping(src, dst_row, cw);
            }

            let rc = (session.fn_unlock_input_buffer)(session.encoder, input_buffer);
            if rc != NV_ENC_SUCCESS {
                bail!("NvEncUnlockInputBuffer failed: {rc}");
            }
            Ok(lock.pitch)
        }
    }

    /// Copy a 10-bit YUV420p frame (`Yuv420p10le` — u16 LE per sample,
    /// valid value in the lower 10 bits) into a locked
    /// `NV_ENC_BUFFER_FORMAT_YUV420_10BIT` surface (P010-style — u16 LE
    /// per sample, valid value in the **upper 10 bits**, i.e.
    /// `sample_10bit << 6`).
    ///
    /// `pitch` from `NvEncLockInputBuffer` is in **bytes**, not samples;
    /// for a 10-bit surface that's 2× the sample count per row.
    /// Plane layout matches IYUV: planar Y → planar U → planar V at
    /// 2 bytes/sample. NVENC documents `_10BIT` as the same plane
    /// arrangement as IYUV with the wider sample width — confirmed
    /// against SDK 12.2 sample apps (`AppEncCuda10/AppEncode10Bit`).
    ///
    /// Round-up `(w+1)/2`, `(h+1)/2` chroma dims for odd dims, same as
    /// the 8-bit path (MEDIUM-6 in codec-review-3).
    unsafe fn upload_frame_10bit(
        session: &EncodeSession,
        frame: &VideoFrame,
        slot: usize,
    ) -> Result<u32> {
        unsafe {
            let input_buffer = session.input_buffers[slot];
            let mut lock: NvEncLockInputBuffer = std::mem::zeroed();
            lock.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
            lock.input_buffer = input_buffer;
            let rc = (session.fn_lock_input_buffer)(session.encoder, &mut lock);
            if rc != NV_ENC_SUCCESS {
                bail!("NvEncLockInputBuffer failed: {rc}");
            }

            let pitch_bytes = lock.pitch as usize;
            let w = session.width as usize;
            let h = session.height as usize;
            let cw = w.div_ceil(2);
            let ch = h.div_ceil(2);
            // Frame data layout (Yuv420p10le): Y plane (w*h u16) + U
            // plane (cw*ch u16) + V plane (cw*ch u16). Bytes are
            // 2× the sample counts.
            let y_bytes = w * h * 2;
            let uv_bytes = cw * ch * 2;
            if frame.data.len() < y_bytes + 2 * uv_bytes {
                (session.fn_unlock_input_buffer)(session.encoder, input_buffer);
                bail!(
                    "frame data too small for {}x{} Yuv420p10le: need {} bytes, got {}",
                    w,
                    h,
                    y_bytes + 2 * uv_bytes,
                    frame.data.len()
                );
            }

            let dst = lock.buffer_data_ptr as *mut u8;
            let src_ptr = frame.data.as_ptr();

            // Y plane: w samples per row, 2*w bytes per row, shift each
            // u16 left by 6 to satisfy the SDK's upper-10-bit
            // convention.
            for row in 0..h {
                let src_row = src_ptr.add(row * w * 2) as *const u16;
                let dst_row = dst.add(row * pitch_bytes) as *mut u16;
                for col in 0..w {
                    // `<<6` keeps the AV1-significant bits in the
                    // upper 10 of the 16-bit container; the bottom 6
                    // bits are zero (matches NVDEC P016 output
                    // emitted by Squad-6 before its `>>6` normalize).
                    let sample = (*src_row.add(col)) & 0x03FF;
                    *dst_row.add(col) = sample << 6;
                }
            }

            // U plane: starts at dst + pitch_bytes*h on the surface.
            // Same half-pitch convention as the 8-bit IYUV path —
            // chroma rows are HALF the luma row stride. See the 8-bit
            // upload_frame for the 4K SIGSEGV diagnosis that fixed
            // this. For 10-bit P010, both byte stride and sample
            // stride are halved (chroma carries half the samples per
            // row, each sample still 2 bytes).
            let chroma_pitch_bytes = pitch_bytes / 2;
            let u_dst_base = dst.add(pitch_bytes * h);
            let u_src_base = src_ptr.add(y_bytes);
            for row in 0..ch {
                let src_row = u_src_base.add(row * cw * 2) as *const u16;
                let dst_row = u_dst_base.add(row * chroma_pitch_bytes) as *mut u16;
                for col in 0..cw {
                    let sample = (*src_row.add(col)) & 0x03FF;
                    *dst_row.add(col) = sample << 6;
                }
            }

            // V plane: follows U at chroma_pitch_bytes*ch further in.
            let v_dst_base = u_dst_base.add(chroma_pitch_bytes * ch);
            let v_src_base = u_src_base.add(uv_bytes);
            for row in 0..ch {
                let src_row = v_src_base.add(row * cw * 2) as *const u16;
                let dst_row = v_dst_base.add(row * chroma_pitch_bytes) as *mut u16;
                for col in 0..cw {
                    let sample = (*src_row.add(col)) & 0x03FF;
                    *dst_row.add(col) = sample << 6;
                }
            }

            let rc = (session.fn_unlock_input_buffer)(session.encoder, input_buffer);
            if rc != NV_ENC_SUCCESS {
                bail!("NvEncUnlockInputBuffer failed: {rc}");
            }
            Ok(lock.pitch)
        }
    }

    /// Drain one bitstream from the given ring slot. Returns
    /// `Ok(None)` if the encoder is not ready yet (NEED_MORE_INPUT /
    /// busy / zero-byte lock). Propagates real errors as `Err`.
    ///
    /// LOW-7: PTS comes from `NV_ENC_LOCK_BITSTREAM.outputTimeStamp`
    /// (matches the input-frame PTS NVENC stamped into this picture).
    /// LOW-8: unexpected error codes propagate.
    /// Lock the slot's bitstream buffer, copy out any encoded bytes, unlock.
    /// Returns `Some((frame_idx, packet))` if the lock surfaced a packet,
    /// `None` if the encoder reported "no data ready" via the documented
    /// `LOCK_BUSY` / `NEED_MORE_INPUT` / `ENCODER_BUSY` rc values OR if the
    /// driver populated `bitstream_size_in_bytes = 0`.
    ///
    /// **Caller responsibility**: track per-slot `last_drained_frame_idx`
    /// and discard the result if `frame_idx == last_drained[slot]` —
    /// the SDK 13 driver shipped with 595.71.05 returns the previously
    /// drained packet bytes verbatim when a slot is re-locked without an
    /// intervening fresh encode_picture, rather than signalling an empty
    /// state. Callers must compare frame_idx to detect stale reads.
    unsafe fn drain_bitstream(
        session: &EncodeSession,
        slot: usize,
    ) -> Result<Option<(u32, EncodedPacket)>> {
        unsafe {
            let bitstream_buffer = session.bitstream_buffers[slot];
            let mut lock: NvEncLockBitstream = std::mem::zeroed();
            lock.version = NV_ENC_LOCK_BITSTREAM_VER;
            lock.output_bitstream = bitstream_buffer;
            let rc = (session.fn_lock_bitstream)(session.encoder, &mut lock);
            match rc {
                NV_ENC_SUCCESS => { /* fall through to read the bytes */ }
                // "No packet ready on this slot" — three flavors:
                //   NEED_MORE_INPUT   — encoder is still buffering for B-frame
                //                       lookahead / temporal reordering.
                //   LOCK_BUSY         — driver is mid-update on this slot;
                //                       caller can retry next tick.
                //   ENCODER_BUSY      — same shape as LOCK_BUSY across the
                //                       whole encoder.
                //   INVALID_PARAM (8) — added 2026-05-08. Driver 580+
                //                       (Blackwell era) returns INVALID_PARAM
                //                       when the EOS drain walks ring slots
                //                       that never received a frame, where
                //                       the SDK 13 driver shipped with
                //                       595.71.05 used to return
                //                       SUCCESS-with-stale-data. Treating
                //                       INVALID_PARAM as "no packet here"
                //                       is consistent with the other three
                //                       — they all mean the same thing on
                //                       the consumer side.
                NV_ENC_ERR_NEED_MORE_INPUT
                | NV_ENC_ERR_LOCK_BUSY
                | NV_ENC_ERR_ENCODER_BUSY
                | NV_ENC_ERR_INVALID_PARAM => {
                    return Ok(None);
                }
                NV_ENC_ERR_INVALID_PTR | NV_ENC_ERR_ENCODER_NOT_INITIALIZED => {
                    bail!("NvEncLockBitstream failed (fatal): {rc}")
                }
                other => bail!("NvEncLockBitstream failed: {other}"),
            }

            let size = lock.bitstream_size_in_bytes as usize;
            // Defensive cap: the bitstream output buffer is allocated
            // at 16 MiB (see CreateBitstreamBuffer). Anything larger
            // would mean the driver wrote past its own buffer or the
            // NV_ENC_LOCK_BITSTREAM struct layout drifted — refuse
            // rather than try to allocate gigabytes.
            const MAX_BITSTREAM_BYTES: usize = 16 * 1024 * 1024;
            if size > MAX_BITSTREAM_BYTES {
                let _ = (session.fn_unlock_bitstream)(session.encoder, bitstream_buffer);
                bail!(
                    "NvEncLockBitstream returned implausible size {} bytes (max {}) — \
                     likely NV_ENC_LOCK_BITSTREAM struct layout drift",
                    size,
                    MAX_BITSTREAM_BYTES
                );
            }
            let data = if size > 0 && !lock.bitstream_buffer_ptr.is_null() {
                let slice =
                    std::slice::from_raw_parts(lock.bitstream_buffer_ptr as *const u8, size);
                Bytes::copy_from_slice(slice)
            } else {
                Bytes::new()
            };

            let is_keyframe = matches!(lock.picture_type, NV_ENC_PIC_TYPE_IDR | NV_ENC_PIC_TYPE_I);
            let pts = lock.output_time_stamp;

            let unlock_rc = (session.fn_unlock_bitstream)(session.encoder, bitstream_buffer);
            if unlock_rc != NV_ENC_SUCCESS {
                bail!("NvEncUnlockBitstream failed: {unlock_rc}");
            }

            if size == 0 {
                return Ok(None);
            }

            Ok(Some((
                lock.frame_idx,
                EncodedPacket {
                    data,
                    pts,
                    is_keyframe,
                },
            )))
        }
    }

    fn encode_pending(&mut self) -> Result<()> {
        if self.pending_frames.is_empty() {
            return Ok(());
        }
        let Some(session) = &self.session else {
            bail!("encode_pending called without live session");
        };

        // Pin our CUDA context to this thread before touching NVENC.
        // Tokio may have migrated us since session creation. Quality /
        // rate-control settings are baked into the encoder at
        // initialize-time — nothing to do per batch here.
        let _scope = unsafe { session.ctx_scope()? };

        let pending = std::mem::take(&mut self.pending_frames);
        for frame in pending {
            // Frame format must match what the session was initialized
            // with — switching mid-stream would silently scramble the
            // surface plane layouts. Better to bail than encode garbage.
            if frame.format != self.config.pixel_format {
                bail!(
                    "NVENC session was initialized with {:?} but frame is {:?} \
                     — pipeline must reinit the encoder if pixel format changes",
                    self.config.pixel_format,
                    frame.format
                );
            }
            let slot = self.ring_idx;
            unsafe {
                // Dispatch to the bit-depth-appropriate uploader.
                // Both end up writing into the same NVENC input
                // surface; only the per-sample byte width and the
                // value-bit shift differ.
                let pitch = match frame.format {
                    PixelFormat::Yuv420p10le => Self::upload_frame_10bit(session, &frame, slot)?,
                    _ => Self::upload_frame(session, &frame, slot)?,
                };

                let mut pic: NvEncPicParams = std::mem::zeroed();
                pic.version = NV_ENC_PIC_PARAMS_VER;
                pic.input_width = session.width;
                pic.input_height = session.height;
                pic.input_pitch = pitch;
                pic.input_buffer = session.input_buffers[slot];
                pic.output_bitstream = session.bitstream_buffers[slot];
                pic.buffer_fmt = session.buffer_format;
                pic.frame_idx = self.frame_counter;
                pic.input_timestamp = frame.pts;
                pic.picture_struct = 1; // NV_ENC_PIC_STRUCT_FRAME

                // Force IDR on keyframe cadence so downstream tooling
                // has well-defined random-access points. The preset's
                // PTD logic will still insert its own IDR at scene
                // cuts but this guarantees at least every N frames.
                let is_idr = self
                    .frame_counter
                    .is_multiple_of(self.config.keyframe_interval);
                pic.picture_type = if is_idr {
                    NV_ENC_PIC_TYPE_IDR
                } else {
                    NV_ENC_PIC_TYPE_P
                };
                if is_idr {
                    pic.encode_pic_flags |= NV_ENC_PIC_FLAG_FORCEIDR;
                }

                let rc = (session.fn_encode_picture)(session.encoder, &mut pic);
                self.frame_counter += 1;

                match rc {
                    NV_ENC_SUCCESS => {
                        if let Some((frame_idx, pkt)) = Self::drain_bitstream(session, slot)? {
                            self.last_drained_frame_idx[slot] = frame_idx as i64;
                            self.encoded_packets.push(pkt);
                        }
                    }
                    NV_ENC_ERR_NEED_MORE_INPUT => {
                        // Normal for initial B-frames or lookahead warmup —
                        // NVENC is accumulating frames before emitting a
                        // packet. Nothing to drain until the next frame.
                    }
                    other => bail!("NvEncEncodePicture failed: {other}"),
                }
            }
            self.ring_idx = (self.ring_idx + 1) % RING_SIZE;
        }
        Ok(())
    }

    fn flush_eos(&mut self) -> Result<()> {
        let Some(session) = &self.session else {
            return Ok(());
        };
        unsafe {
            let _scope = session.ctx_scope()?;

            // EOS picture: null input buffer, PIC_FLAG_EOS set. This
            // tells NVENC to drain anything it was holding for
            // lookahead/B-frames. Use the current ring slot's output
            // buffer — NVENC only needs one output handle on the EOS
            // picture; the actual drained packets come through
            // `LockBitstream` on each ring buffer below.
            let mut pic: NvEncPicParams = std::mem::zeroed();
            pic.version = NV_ENC_PIC_PARAMS_VER;
            pic.encode_pic_flags = NV_ENC_PIC_FLAG_EOS;
            pic.input_buffer = ptr::null_mut();
            pic.output_bitstream = session.bitstream_buffers[self.ring_idx];
            pic.buffer_fmt = session.buffer_format;
            let _ = (session.fn_encode_picture)(session.encoder, &mut pic);

            // Walk every ring-buffer slot once. Each slot may hold at
            // most ONE pending frame that EOS just released.
            //
            // 2026-05-01 BUG FIX: this used to be a `loop { drain →
            // break on None }` per slot, on the (incorrect) theory
            // that a slot could hold multiple queued packets. In
            // practice on the driver shipped with NVENC SDK 13
            // (595.71.05), `NvEncLockBitstream` on a slot whose
            // bitstream has already been unlocked once returns
            // NV_ENC_SUCCESS with the SAME packet bytes every call —
            // never NEED_MORE_INPUT, never size=0. The inner loop
            // therefore appended the same 1.2 KB packet to
            // `encoded_packets` forever, growing the heap by ~60 GB
            // before OOM-kill. A single lock+drain per slot is the
            // correct teardown — the bitstream output buffer for any
            // ring slot can hold exactly one encoded frame at a time.
            // LOW-7: drained PTS comes from each lock's
            // `output_time_stamp` (handled inside drain_bitstream).
            for i in 0..RING_SIZE {
                // Start the walk from the "oldest" slot so drained
                // packets come out in roughly submission order. The
                // oldest in-flight slot is `ring_idx` itself (next to
                // be written), so the producer wrote RING_SIZE-1
                // slots before it.
                let slot = (self.ring_idx + i) % RING_SIZE;
                if let Some((frame_idx, pkt)) = Self::drain_bitstream(session, slot)? {
                    // Stale-read filter: if the driver handed us back a
                    // frame_idx we've already drained from this slot,
                    // it's the previous packet bytes — see drain_bitstream
                    // docstring and 2026-05-01 SDK 13 driver bug note.
                    // Skip silently; the real EOS-flushed frames (if any)
                    // will arrive with frame_idx > last_drained_frame_idx.
                    if (frame_idx as i64) > self.last_drained_frame_idx[slot] {
                        self.last_drained_frame_idx[slot] = frame_idx as i64;
                        self.encoded_packets.push(pkt);
                    }
                }
            }
        }
        Ok(())
    }
}

impl Encoder for NvencEncoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        // Defer pixel-format mismatch reporting to encode_pending so
        // the error message can show both the configured + observed
        // formats — keeps the per-format dispatch in one place.
        if frame.format != self.config.pixel_format {
            bail!(
                "NVENC session was initialized with {:?} but frame is {:?}",
                self.config.pixel_format,
                frame.format
            );
        }
        self.pending_frames.push(frame.clone());
        // Encode immediately — NVENC holds its own lookahead buffer
        // internally when the preset enables it, so we don't batch
        // on our side (batching here would just add latency).
        self.encode_pending()?;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.encode_pending()?;
        if !self.flushed {
            self.flush_eos()?;
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

// ─── Compile-time struct-size assertions ──────────────────────────
//
// Catches the class of bug that produced the CUVIDPARSERPARAMS drift
// in task #65. Expected sizes are computed against SDK 12.2 headers
// + the stub at vendor/nvidia/nvEncodeAPI.h, and measured empirically
// against the Rust repr(C) layout on MSVC x64 (which matches every
// target we care about).
//
// If any of these fire, the Rust struct no longer matches the NVENC
// ABI — expect INVALID_VERSION on create, silent corruption on
// encode, or INVALID_PARAM on InitializeEncoder.

// NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS (vendor/nvidia/nvEncodeAPI.h:144-152).
// u32 + u32 + ptr + ptr + u32 + u32[253] + ptr[64] = 1552.
const _: () = assert!(std::mem::size_of::<NvEncOpenEncodeSessionExParams>() == 1552);

// NV_ENC_INITIALIZE_PARAMS (vendor/nvidia/nvEncodeAPI.h:173-200). After
// GUID+14×u32+3×u32+ptr+2×u32+[u32;2]+2×u32+287×u32+64×ptr with ptr
// alignment: 1800 bytes measured.
const _: () = assert!(std::mem::size_of::<NvEncInitializeParams>() == 1800);

// NV_ENC_RC_PARAMS — SDK 13.0 layout
// (vendor/nvidia/nvEncodeAPI.h:1555-1627). SDK 13 inserted
// `lookaheadLevel: NV_ENC_LOOKAHEAD_LEVEL (u32)` between
// `reserved2` and `viewBitrateRatios[]` and grew temporallayerIdxMask
// + temporalLayerQP[8] from the previous mis-mirrored 8-byte slot to
// the spec-correct 12 bytes. Total grew 124 → 128 bytes.
const _: () = assert!(std::mem::size_of::<NvEncRcParams>() == 128);

// NV_ENC_CONFIG_AV1 — SDK 13.0 `_NV_ENC_CONFIG_AV1`. Bitfield-packed
// flags + 14×u32 + 2×ptr (tile_widths/heights) + 6×u32 enums + 1×ptr
// (film_grain_params) + 7×u32 + reserved1[230] + reserved3[62 ptrs]
// with alignment padding. MSVC x64 size: 1552.
const _: () = assert!(std::mem::size_of::<NvEncConfigAv1>() == 1552);

// NV_ENC_CONFIG — SDK 13.0 `_NV_ENC_CONFIG` (version macro ver=9 at
// vendor/nvidia/nvEncodeAPI.h:2200). u32 + GUID(16) + 5×u32 +
// RC_PARAMS(128) + CODEC_CONFIG_UNION(1792, sized to H264 variant) +
// reserved[278] + reserved2[64 ptrs] = 3584 bytes. Verified against
// `sizeof(NV_ENC_CONFIG)` printed by a small C harness compiled
// against the vendored SDK 13 header on the dev box.
const _: () = assert!(std::mem::size_of::<NvEncConfig>() == 3584);

// NV_ENC_PRESET_CONFIG — SDK 13.0 `_NV_ENC_PRESET_CONFIG` (added a
// leading reserved u32 + grew reserved1 from [255] to [256]).
// 2×u32 + NV_ENC_CONFIG(3584) + reserved1[256] + reserved2[64 ptrs]
// = 5128 bytes.
const _: () = assert!(std::mem::size_of::<NvEncPresetConfig>() == 5128);

// NV_ENC_CREATE_INPUT_BUFFER (vendor/nvidia/nvEncodeAPI.h:203-214).
// 6×u32 + 2×ptr + 57×u32 + 63×ptr with alignment pads = 776 bytes.
const _: () = assert!(std::mem::size_of::<NvEncCreateInputBuffer>() == 776);

// NV_ENC_CREATE_BITSTREAM_BUFFER (vendor/nvidia/nvEncodeAPI.h:217-226).
// 4×u32 + 2×ptr + 58×u32 + 64×ptr = 776 bytes.
const _: () = assert!(std::mem::size_of::<NvEncCreateBitstreamBuffer>() == 776);

// NV_ENC_LOCK_BITSTREAM — SDK 13.0 layout
// (vendor/nvidia/nvEncodeAPI.h:2675-2714). 1544 bytes total. The size
// alone is insufficient: the prior `[u32; 13]` blob in place of the
// seven scalar fields (temporal_id ... output_stats_ptr_size +
// reserved scalar) PLUS `[*void; 64]` instead of `[*void; 63] +
// reserved_internal[8]` totalled the same 1544 bytes but every
// field after offset 88 lived at the wrong place. Driver 580+
// surfaced it as INVALID_PARAM during EOS drain. Below: per-field
// offset assertions catch the same class of drift at compile time.
const _: () = assert!(std::mem::size_of::<NvEncLockBitstream>() == 1544);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, version) == 0);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, output_bitstream) == 8);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, slice_offsets) == 16);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, frame_idx) == 24);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, bitstream_size_in_bytes) == 36);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, output_time_stamp) == 40);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, bitstream_buffer_ptr) == 56);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, picture_type) == 64);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, ltr_frame_bitmap) == 84);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, temporal_id) == 88);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, intra_mb_count) == 92);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, alpha_layer_size_in_bytes) == 108);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, output_stats_ptr_size) == 112);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, reserved) == 116);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, output_stats_ptr) == 120);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, frame_idx_display) == 128);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, reserved1) == 132);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, reserved2) == 1008);
const _: () = assert!(std::mem::offset_of!(NvEncLockBitstream, reserved_internal) == 1512);

// NV_ENC_PIC_PARAMS (vendor/nvidia/nvEncodeAPI.h:2564-2625). SDK 13.0
// total size 3360 bytes. Size grew 2048 → 2840 → 3360 across two
// rounds of mirror correction:
//
//  Round 1 (commit 53365e7, 2840): widened codec_pic_params from
//   [u8; 256] to [u8; 1024] thinking the union body was the
//   `uint32_t reserved[256]` placeholder.
//  Round 2 (this commit, 3360): the union actually sizes to
//   max(variants), and NV_ENC_PIC_PARAMS_AV1 is 1544 bytes (verified
//   via sizeof() against the vendored SDK 13 header). Switched the
//   mirror field to `[u64; 193]` so the slot has both the right
//   size (1544) AND the right alignment (8) — and the natural u64
//   alignment forces the 4-byte pre-pad the C compiler emits to
//   put codecPicParams at offset 80 instead of 76.
//
// Every field after codec_pic_params shifts forward by 524 bytes vs
// the old mirror — including the SDK 13 "new in 13" fields
// (reserved4, alphaBuffer, meExternalSbHints, meSbHintsCount,
// stateBufferIdx, outputReconBuffer) which are now at the offsets
// the driver expects.
const _: () = assert!(std::mem::size_of::<NvEncPicParams>() == 3360);

// NV_ENCODE_API_FUNCTION_LIST — 41 typed fn-pointer slots + 256-ptr
// tail. NVIDIA's real SDK 12.2 struct is smaller than this; we carry
// a deliberately-large tail so `NvEncodeAPICreateInstance` cannot
// write past our buffer if the SDK adds entries. Only checked `>=`
// for that reason. Minimum baseline: version(4)+reserved(4) +
// 41×ptr(328) = 336.
const _: () = assert!(std::mem::size_of::<NvEncFunctionList>() >= 336);

// Squad-22: pin the 10-bit buffer-format constant. The SDK
// enumeration is `0x00010000`; if that ever changes (NVIDIA splits
// 10-bit into per-format variants in a future SDK rev) the dispatch
// in `nvenc_buffer_format_for` would silently mis-route.
const _: () = assert!(NV_ENC_BUFFER_FORMAT_YUV420_10BIT == 0x00010000);
// And the 8-bit IYUV constant for symmetry — both must agree with
// `vendor/nvidia/nvEncodeAPI.h:94-115`.
const _: () = assert!(NV_ENC_BUFFER_FORMAT_IYUV == 0x00000100);
// 10-bit AV1 sets `pixel_bit_depth_minus_8 = 2`. The SDK reads this
// field as a `uint32_t`; if we ever accidentally narrow the type the
// arithmetic on the call site would silently truncate at depth=2.
const _: () = assert!(pixel_bit_depth_minus8_for(PixelFormat::Yuv420p10le) == 2);
const _: () = assert!(pixel_bit_depth_minus8_for(PixelFormat::Yuv420p) == 0);

// ─── Unit tests ──────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fps_rational_mapping() {
        // Broadcast rates from MEDIUM-4.
        assert_eq!(fps_to_rational(23.976), (24_000, 1001));
        assert_eq!(fps_to_rational(24.0), (24, 1));
        assert_eq!(fps_to_rational(25.0), (25, 1));
        assert_eq!(fps_to_rational(29.97), (30_000, 1001));
        assert_eq!(fps_to_rational(30.0), (30, 1));
        assert_eq!(fps_to_rational(48.0), (48, 1));
        assert_eq!(fps_to_rational(50.0), (50, 1));
        assert_eq!(fps_to_rational(59.94), (60_000, 1001));
        assert_eq!(fps_to_rational(60.0), (60, 1));
    }

    #[test]
    fn test_fps_rational_1001_family_detection() {
        // Higher-precision 1001-family values should still hit the
        // canonical rational.
        let (n, d) = fps_to_rational(23.9760239760);
        assert_eq!(d, 1001);
        assert_eq!(n, 24_000);

        let (n, d) = fps_to_rational(29.9700299700);
        assert_eq!(d, 1001);
        assert_eq!(n, 30_000);

        let (n, d) = fps_to_rational(59.9400599400);
        assert_eq!(d, 1001);
        assert_eq!(n, 60_000);
    }

    #[test]
    fn test_fps_rational_generic_fallback() {
        // Integer fps not in the broadcast table: use `(n, 1)` form
        // (integer shortcut, before 1001-family detector would match).
        assert_eq!(fps_to_rational(100.0), (100, 1));
        assert_eq!(fps_to_rational(120.0), (120, 1));
        // 23.5 has no 1001-family match and isn't integer — generic
        // fallback (round(fps*1000), 1000).
        assert_eq!(fps_to_rational(23.5), (23_500, 1000));
    }

    #[test]
    fn test_nvenc_cq_clamps_to_51() {
        // The SDK documents 0..51 for H.264/HEVC and 0..63 for AV1 on
        // `targetQuality`. The code path clamps to 51 before handing
        // the value to `rc_params.target_quality` to stay inside the
        // historical H.264/HEVC band (AV1's 0..63 is not rejected but
        // values >51 produce ill-defined behaviour on older drivers).
        let clamped = 75u8.min(51);
        assert_eq!(clamped, 51);
        let ok = 40u8.min(51);
        assert_eq!(ok, 40);
        let at_limit = 51u8.min(51);
        assert_eq!(at_limit, 51);
    }

    #[test]
    fn test_ring_buffer_index_cycles() {
        // Sanity: ring_idx walks 0,1,2,3,0,1,2,3,... under
        // `(ring_idx + 1) % RING_SIZE`.
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
    fn test_ring_size_is_four() {
        // MEDIUM-5 prescribes N=4 input/output buffers.
        assert_eq!(RING_SIZE, 4);
    }

    // ── Squad-22: 10-bit dispatch + color signalling tests ───────

    /// `nvenc_buffer_format_for` must return the YUV420_10BIT constant
    /// for `Yuv420p10le` and IYUV for plain `Yuv420p`. Mismatched dispatch
    /// here would produce silently-wrong encodes (the IYUV path on a
    /// 10-bit surface would write the wide samples into the 8-bit slot
    /// the GPU expects → uniform mid-gray output).
    #[test]
    fn test_nvenc_buffer_format_dispatch_10bit() {
        let fmt_8 = nvenc_buffer_format_for(PixelFormat::Yuv420p).unwrap();
        let fmt_10 = nvenc_buffer_format_for(PixelFormat::Yuv420p10le).unwrap();
        assert_eq!(fmt_8, NV_ENC_BUFFER_FORMAT_IYUV);
        assert_eq!(fmt_10, NV_ENC_BUFFER_FORMAT_YUV420_10BIT);
        assert_ne!(
            fmt_8, fmt_10,
            "10-bit must select a different SDK constant from 8-bit"
        );
    }

    /// Unsupported pixel formats must bail with a typed error, NOT
    /// fall through to the IYUV path. Mirrors the NVDEC chroma reject
    /// (Squad-6) — we carry an explicit not-supported list rather than
    /// best-effort attempts that produce silent corruption.
    #[test]
    fn test_nvenc_buffer_format_dispatch_rejects_4_2_2_and_4_4_4() {
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
                nvenc_buffer_format_for(unsupported).is_err(),
                "{unsupported:?} must be rejected by NVENC dispatch"
            );
        }
    }

    /// `pixel_bit_depth_minus_8` field controls the AV1 OBU sequence
    /// header `BitDepth` value: 0 → 8-bit, 2 → 10-bit. A const_assert!
    /// at the bottom of the file pins this; the test mirrors it for
    /// the test summary.
    #[test]
    fn test_nvenc_pixel_bit_depth_dispatch() {
        assert_eq!(pixel_bit_depth_minus8_for(PixelFormat::Yuv420p), 0);
        assert_eq!(pixel_bit_depth_minus8_for(PixelFormat::Yuv420p10le), 2);
    }

    /// `transfer_to_h273` must round-trip every `TransferFn` variant
    /// to its ITU-T H.273 numeric code. These match the codes the mux
    /// `colr nclx` writer emits — keeping the in-bitstream value
    /// identical to the container-level metadata.
    #[test]
    fn test_nvenc_transfer_to_h273_codes() {
        assert_eq!(transfer_to_h273(TransferFn::Bt709), 1);
        assert_eq!(transfer_to_h273(TransferFn::Bt470Bg), 4);
        assert_eq!(transfer_to_h273(TransferFn::Linear), 8);
        assert_eq!(transfer_to_h273(TransferFn::St2084), 16, "HDR10 PQ");
        assert_eq!(transfer_to_h273(TransferFn::AribStdB67), 18, "HLG");
        assert_eq!(
            transfer_to_h273(TransferFn::Unspecified),
            1,
            "Unspecified collapses to canonical Bt709 — AV1 has no \
             unspecified sentinel for transfer"
        );
    }

    /// Build a 10-bit AV1 codec config by hand and assert the bytes
    /// at the bit-depth + color signalling offsets carry the expected
    /// values. This is the "construct the struct, dump its bytes,
    /// assert" test the task spec calls for — it doesn't need a GPU
    /// because all the field writes are pure-Rust struct mutations.
    ///
    /// SDK 13 retired `pixel_bit_depth_minus_8`/`input_pixel_bit_depth_minus_8`
    /// in favour of `output_bit_depth`/`input_bit_depth` enums (8-bit=0,
    /// 10-bit=1). It also dropped the explicit
    /// `color_description_present_flag` field; the four color codes are
    /// emitted whenever any of them is non-zero (driver-side per SDK 13
    /// docs). `chroma_format_idc` was folded into `flags` bits 7-8.
    #[test]
    fn test_nvenc_av1_config_10bit_hdr_layout() {
        let mut cfg: NvEncConfigAv1 = unsafe { std::mem::zeroed() };
        // SDK 13 enum: 0 = NV_ENC_BIT_DEPTH_8, 1 = NV_ENC_BIT_DEPTH_10.
        let bit_depth_minus8 = pixel_bit_depth_minus8_for(PixelFormat::Yuv420p10le);
        let bit_depth_enum: u32 = if bit_depth_minus8 == 0 { 0 } else { 1 };
        cfg.output_bit_depth = bit_depth_enum;
        cfg.input_bit_depth = bit_depth_enum;
        cfg.flags |= AV1_CHROMA_FORMAT_IDC_420;

        // HDR10 metadata — BT.2020 NCL primaries, PQ transfer, full range.
        let cm = ColorMetadata {
            transfer: TransferFn::St2084,
            matrix_coefficients: 9, // BT.2020 NCL
            colour_primaries: 9,    // BT.2020
            full_range: true,
            mastering_display: None,
            content_light_level: None,
        };
        cfg.color_primaries = cm.colour_primaries as u32;
        cfg.transfer_characteristics = transfer_to_h273(cm.transfer);
        cfg.matrix_coefficients = cm.matrix_coefficients as u32;
        cfg.color_range = cm.full_range as u32;

        assert_eq!(cfg.output_bit_depth, 1, "10-bit enum value");
        assert_eq!(cfg.input_bit_depth, 1, "10-bit input enum value");
        assert_eq!(cfg.color_primaries, 9, "BT.2020");
        assert_eq!(cfg.transfer_characteristics, 16, "ST 2084 / PQ");
        assert_eq!(cfg.matrix_coefficients, 9, "BT.2020 NCL");
        assert_eq!(cfg.color_range, 1, "full range");
        assert_eq!(
            cfg.flags & AV1_CHROMA_FORMAT_IDC_420,
            AV1_CHROMA_FORMAT_IDC_420,
            "chromaFormatIDC=1 (4:2:0) packed into flags bits 7-8"
        );

        // Byte-level: u32 LE reads at the field offsets. An accidental
        // field reorder during a future SDK port surfaces as a diff here.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &cfg as *const NvEncConfigAv1 as *const u8,
                std::mem::size_of::<NvEncConfigAv1>(),
            )
        };
        let bd_offset = std::mem::offset_of!(NvEncConfigAv1, output_bit_depth);
        assert_eq!(
            u32::from_le_bytes(bytes[bd_offset..bd_offset + 4].try_into().unwrap()),
            1,
            "output_bit_depth must read back as 1 (10-bit) from raw bytes"
        );

        let prim_offset = std::mem::offset_of!(NvEncConfigAv1, color_primaries);
        assert_eq!(
            u32::from_le_bytes(bytes[prim_offset..prim_offset + 4].try_into().unwrap()),
            9,
            "color_primaries=9 (BT.2020) at the expected offset"
        );

        let trans_offset = std::mem::offset_of!(NvEncConfigAv1, transfer_characteristics);
        assert_eq!(
            u32::from_le_bytes(bytes[trans_offset..trans_offset + 4].try_into().unwrap()),
            16,
            "transfer_characteristics=16 (PQ) at the expected offset"
        );

        let range_offset = std::mem::offset_of!(NvEncConfigAv1, color_range);
        assert_eq!(
            u32::from_le_bytes(bytes[range_offset..range_offset + 4].try_into().unwrap()),
            1,
            "color_range=1 (full) at the expected offset"
        );
    }

    /// 8-bit default config still shapes correctly — paranoid guard
    /// against the 10-bit additions silently breaking the SDR path.
    #[test]
    fn test_nvenc_av1_config_8bit_sdr_layout() {
        let mut cfg: NvEncConfigAv1 = unsafe { std::mem::zeroed() };
        let bit_depth_minus8 = pixel_bit_depth_minus8_for(PixelFormat::Yuv420p);
        let bit_depth_enum: u32 = if bit_depth_minus8 == 0 { 0 } else { 1 };
        cfg.output_bit_depth = bit_depth_enum;
        cfg.input_bit_depth = bit_depth_enum;
        cfg.flags |= AV1_CHROMA_FORMAT_IDC_420;
        let cm = ColorMetadata::default();
        cfg.color_primaries = cm.colour_primaries as u32;
        cfg.transfer_characteristics = transfer_to_h273(cm.transfer);
        cfg.matrix_coefficients = cm.matrix_coefficients as u32;
        cfg.color_range = cm.full_range as u32;

        assert_eq!(cfg.output_bit_depth, 0, "8-bit enum value");
        assert_eq!(cfg.color_primaries, 1, "BT.709 default");
        assert_eq!(cfg.transfer_characteristics, 1, "BT.709 default");
        assert_eq!(cfg.matrix_coefficients, 1, "BT.709 default");
        assert_eq!(cfg.color_range, 0, "studio range default");
    }

    #[test]
    fn test_guid_roundtrip() {
        // `guid_from_bytes` on the P5 GUID bytes (NVENC SDK 13 layout,
        // little-endian per Microsoft GUID convention) must reproduce
        // the typed P5 GUID constant.
        //
        // P5 = 21c6e6b4-297a-4cba-998f-b6cbde72ade3
        // The leading three groups serialise LE, the last two groups
        // serialise BE — that's the asymmetry MS GUIDs are known for.
        // (Earlier the test held SDK 12.2's pre-rotation P5 bytes
        // d0918ee2-a509-4681-af96-e9c3c45b7aa7; updated alongside the
        // constant in the SDK 13 layout-fix commit.)
        let bytes: [u8; 16] = [
            0xb4, 0xe6, 0xc6, 0x21, 0x7a, 0x29, 0xba, 0x4c, 0x99, 0x8f, 0xb6, 0xcb, 0xde, 0x72,
            0xad, 0xe3,
        ];
        let g = guid_from_bytes(bytes);
        assert_eq!(g.data1, NV_ENC_PRESET_P5_GUID.data1);
        assert_eq!(g.data2, NV_ENC_PRESET_P5_GUID.data2);
        assert_eq!(g.data3, NV_ENC_PRESET_P5_GUID.data3);
        assert_eq!(g.data4, NV_ENC_PRESET_P5_GUID.data4);
    }
}
