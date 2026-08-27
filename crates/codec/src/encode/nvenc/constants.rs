//! NVENC API constants, GUIDs, CUDA FFI types, and function-pointer type aliases.

use std::ffi::c_void;
use std::os::raw::{c_int, c_uint};

// ─── NVENC API constants ──────────────────────────────────────────
// See vendor/nvidia/nvEncodeAPI.h for authoritative definitions.

pub(super) const NV_ENC_SUCCESS: c_uint = 0;
// `_NVENCSTATUS` values of interest. vendor/nvidia/nvEncodeAPI.h:30-42.
pub(super) const NV_ENC_ERR_INVALID_PTR: c_uint = 6;
pub(super) const NV_ENC_ERR_INVALID_PARAM: c_uint = 8;
pub(super) const NV_ENC_ERR_ENCODER_NOT_INITIALIZED: c_uint = 11;
pub(super) const NV_ENC_ERR_LOCK_BUSY: c_uint = 13;
pub(super) const NV_ENC_ERR_NEED_MORE_INPUT: c_uint = 17;
pub(super) const NV_ENC_ERR_ENCODER_BUSY: c_uint = 18;

pub(super) const NV_ENC_DEVICE_TYPE_CUDA: c_uint = 1;
pub(super) const NV_ENC_BUFFER_FORMAT_IYUV: c_uint = 0x00000100;
/// 10-bit planar 4:2:0. P010-style: each sample is a 16-bit LE word
/// with the valid 10-bit value in the **upper 10 bits**
/// (`sample_10bit << 6`). See `vendor/nvidia/nvEncodeAPI.h:94-115` for
/// the SDK 12.2 enumeration. This matches NVDEC's P016 surface output
/// (Squad-6) and the pipeline's `Yuv420p10le` representation, which
/// stores the value in the **lower 10 bits** — `upload_frame_10bit`
/// performs the `<<6` left-shift on copy so the surface byte layout
/// satisfies the SDK convention.
pub(super) const NV_ENC_BUFFER_FORMAT_YUV420_10BIT: c_uint = 0x00010000;

pub(super) const NV_ENC_PIC_FLAG_FORCEIDR: c_uint = 0x02;
pub(super) const NV_ENC_PIC_FLAG_EOS: c_uint = 0x08;

pub(super) const NV_ENC_PIC_TYPE_P: c_uint = 0;
pub(super) const NV_ENC_PIC_TYPE_I: c_uint = 2;
pub(super) const NV_ENC_PIC_TYPE_IDR: c_uint = 3;

#[allow(dead_code)]
pub(super) const NV_ENC_TUNING_INFO_HIGH_QUALITY: c_uint = 1;

// Rate control modes — vendor/nvidia/nvEncodeAPI.h:77-84 (_NV_ENC_PARAMS_RC_MODE).
pub(super) const NV_ENC_PARAMS_RC_CONSTQP: u32 = 0x0;
pub(super) const NV_ENC_PARAMS_RC_VBR: u32 = 0x1;

// NV_ENC_RC_PARAMS bitfield bits (nvEncodeAPI.h). `enableLookahead` (bit 5) and
// `zeroReorderDelay` (bit 9) control output buffering. Our ring-of-4 sync drain
// is strictly 1-in-1-out (one packet per EncodePicture), so for H.264/H.265 we
// CLEAR lookahead and SET zeroReorderDelay to force the encoder to emit each
// frame's packet immediately — otherwise lookahead/reorder buffering strands
// the tail frames (frame loss) or deadlocks the EOS drain (hang).
pub(super) const RC_FLAG_ENABLE_LOOKAHEAD: u32 = 1 << 5;
pub(super) const RC_FLAG_ZERO_REORDER_DELAY: u32 = 1 << 9;
// `_HQ` is gone in SDK 12.2 (merged into VBR + high-quality tuning) but
// kept in the enum for back-compat. We emit plain VBR with tuning =
// HIGH_QUALITY which is the 12.2-idiomatic "VBR_HQ" path.
#[allow(dead_code)]
pub(super) const NV_ENC_PARAMS_RC_VBR_HQ: u32 = 0x20;

// Ring-buffer depth. 4 mirrors ffmpeg libavcodec/nvenc.c's default
// `nb_surfaces` for 1-pass and keeps the encoder pipeline full on Ada
// without oversubscribing GPU memory.
/// Input surfaces (and matching bitstream buffers) this encoder owns.
///
/// Was four, which is only enough while every EncodePicture returns a packet
/// immediately. Lookahead and reordering make the encoder hold frames, and a
/// pool that shallow then hands the next frame a surface the encoder is still
/// reading — the per-segment wrong-frame bug. Sixteen is deeper than the
/// lookahead this service asks for, and slots are chosen by "not outstanding"
/// rather than in turn, so depth is headroom rather than the correctness
/// argument.
pub(super) const RING_SIZE: usize = 16;

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
pub(super) const NVENCAPI_MAJOR: u32 = 13;
pub(super) const NVENCAPI_MINOR: u32 = 0;
pub(super) const NVENCAPI_VERSION: u32 = NVENCAPI_MAJOR | (NVENCAPI_MINOR << 24);

pub(super) const fn struct_version(ver: u32) -> u32 {
    NVENCAPI_VERSION | (ver << 16) | (0x7 << 28)
}

// Per-struct version constants — SDK 13.0 values from header.
// MOST changed from 12.2; comments list the deltas so a future
// SDK-bump audit can spot which structs grew.
pub(super) const NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER: u32 = struct_version(1); // unchanged
pub(super) const NV_ENC_INITIALIZE_PARAMS_VER: u32 = struct_version(7) | (1u32 << 31); // 12.2 was struct_version(5) without high-bit
pub(super) const NV_ENC_CREATE_INPUT_BUFFER_VER: u32 = struct_version(2); // 12.2 was struct_version(1)
pub(super) const NV_ENC_CREATE_BITSTREAM_BUFFER_VER: u32 = struct_version(1); // unchanged
pub(super) const NV_ENC_LOCK_INPUT_BUFFER_VER: u32 = struct_version(1); // unchanged
pub(super) const NV_ENC_LOCK_BITSTREAM_VER: u32 = struct_version(2) | (1u32 << 31); // 12.2 was (1) without high-bit
pub(super) const NV_ENC_PIC_PARAMS_VER: u32 = struct_version(7) | (1u32 << 31); // 12.2 was (4) without high-bit
pub(super) const NV_ENC_CONFIG_VER: u32 = struct_version(9) | (1u32 << 31); // 12.2 was (7) without high-bit
pub(super) const NV_ENC_PRESET_CONFIG_VER: u32 = struct_version(5) | (1u32 << 31); // 12.2 was (4) | high-bit
/// `NV_ENC_RECONFIGURE_PARAMS_VER` — `NVENCAPI_STRUCT_VERSION(1) | (1 << 31)`,
/// unchanged from SDK 12.2 through 13.0.
pub(super) const NV_ENC_RECONFIGURE_PARAMS_VER: u32 = struct_version(1) | (1u32 << 31);

// GUID layout: 32-bit Data1 (LE), 16-bit Data2/3 (LE), 8 raw bytes.
// Values from NVIDIA Video Codec SDK 12.2 headers (vendor/nvidia/nvEncodeAPI.h:49).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Guid {
    pub(super) data1: u32,
    pub(super) data2: u16,
    pub(super) data3: u16,
    pub(super) data4: [u8; 8],
}

pub(super) const NV_ENC_CODEC_AV1_GUID: Guid = Guid {
    data1: 0x0a352289,
    data2: 0x0aa7,
    data3: 0x4759,
    data4: [0x86, 0x2d, 0x5d, 0x15, 0xcd, 0x16, 0xd2, 0x54],
};

// NV_ENC_CODEC_H264_GUID (nvEncodeAPI.h). Supported by every NVENC since
// Kepler — including the RTX 3090 (Ampere) which has no AV1 encode silicon.
pub(super) const NV_ENC_CODEC_H264_GUID: Guid = Guid {
    data1: 0x6bc82762,
    data2: 0x4e63,
    data3: 0x4ca4,
    data4: [0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf],
};

// NV_ENC_CODEC_HEVC_GUID (nvEncodeAPI.h). Supported since Maxwell 2nd-gen.
pub(super) const NV_ENC_CODEC_HEVC_GUID: Guid = Guid {
    data1: 0x790cdc88,
    data2: 0x4522,
    data3: 0x4d7b,
    data4: [0x94, 0x25, 0xbd, 0xa9, 0x97, 0x5f, 0x76, 0x03],
};

// NV_ENC_H264_PROFILE_HIGH_GUID — 4:2:0 8-bit High profile.
pub(super) const NV_ENC_H264_PROFILE_HIGH_GUID: Guid = Guid {
    data1: 0xe7cbc309,
    data2: 0x4f7a,
    data3: 0x4b89,
    data4: [0xaf, 0x2a, 0xd5, 0x37, 0xc9, 0x2b, 0xe3, 0x10],
};

// NV_ENC_HEVC_PROFILE_MAIN_GUID — 4:2:0 8-bit Main profile.
pub(super) const NV_ENC_HEVC_PROFILE_MAIN_GUID: Guid = Guid {
    data1: 0xb514c39a,
    data2: 0xb55b,
    data3: 0x40fa,
    data4: [0x87, 0x8f, 0xf1, 0x25, 0x3b, 0x4d, 0xfd, 0xec],
};

// NV_ENC_HEVC_PROFILE_MAIN10_GUID — 4:2:0 10-bit Main 10 profile (Pascal+).
pub(super) const NV_ENC_HEVC_PROFILE_MAIN10_GUID: Guid = Guid {
    data1: 0xfa4d2b6c,
    data2: 0x3a5b,
    data3: 0x411a,
    data4: [0x80, 0x18, 0x0a, 0x3f, 0x5e, 0x3c, 0x9b, 0xe5],
};

/// The NVENC codec GUID for our `VideoCodec`.
pub(super) fn nvenc_codec_guid(codec: crate::frame::VideoCodec) -> Guid {
    use crate::frame::VideoCodec;
    match codec {
        VideoCodec::Av1 => NV_ENC_CODEC_AV1_GUID,
        VideoCodec::H264 => NV_ENC_CODEC_H264_GUID,
        VideoCodec::H265 => NV_ENC_CODEC_HEVC_GUID,
    }
}

/// The profile GUID for H.264 / H.265 at the given bit depth. AV1 uses the
/// preset's default profile (a zero GUID lets the driver autoselect).
///
/// NVENC has **no H.264 High 10 profile** — H.264 10-bit (Hi10P) is not in the
/// hardware, so 10-bit H.264 falls through to High (8-bit) and the
/// `SUPPORT_10BIT_ENCODE` capability query rejects it before init. HEVC has a
/// real Main 10 profile (Pascal+), so 10-bit H.265 selects it.
pub(super) fn nvenc_profile_guid(codec: crate::frame::VideoCodec, ten_bit: bool) -> Guid {
    use crate::frame::VideoCodec;
    match codec {
        VideoCodec::H264 => NV_ENC_H264_PROFILE_HIGH_GUID,
        VideoCodec::H265 if ten_bit => NV_ENC_HEVC_PROFILE_MAIN10_GUID,
        VideoCodec::H265 => NV_ENC_HEVC_PROFILE_MAIN_GUID,
        VideoCodec::Av1 => Guid { data1: 0, data2: 0, data3: 0, data4: [0u8; 8] },
    }
}

// `NV_ENC_CAPS_PARAM` (nvEncodeAPI.h) — query one capability at a time.
// Layout: u32 version + NV_ENC_CAPS enum + reserved[62] = 256 bytes.
#[repr(C)]
pub(super) struct NvEncCapsParam {
    pub(super) version: u32,
    pub(super) caps_to_query: c_uint,
    pub(super) reserved: [u32; 62],
}
// NV_ENC_CAPS enum values (stable across SDK versions, nvEncodeAPI.h).
pub(super) const NV_ENC_CAPS_WIDTH_MAX: c_uint = 16;
pub(super) const NV_ENC_CAPS_HEIGHT_MAX: c_uint = 17;
pub(super) const NV_ENC_CAPS_SUPPORT_10BIT_ENCODE: c_uint = 39;

// Preset GUIDs from SDK 13.0 (vendor/nvidia/nvEncodeAPI.h:226-251).
// SDK 12.2 used different values for P5/P6/P7 — see tuning.rs comment
// for the full rotation. Sending 12.2 GUIDs to a 13.0 driver returns
// NV_ENC_ERR_UNSUPPORTED_PARAM (rc=12) from NvEncGetEncodePresetConfigEx.
#[allow(dead_code)]
pub(super) const NV_ENC_PRESET_P5_GUID: Guid = Guid {
    data1: 0x21c6e6b4,
    data2: 0x297a,
    data3: 0x4cba,
    data4: [0x99, 0x8f, 0xb6, 0xcb, 0xde, 0x72, 0xad, 0xe3],
};

#[allow(dead_code)]
pub(super) const NV_ENC_PRESET_P6_GUID: Guid = Guid {
    data1: 0x8e75c279,
    data2: 0x6299,
    data3: 0x4ab6,
    data4: [0x83, 0x02, 0x0b, 0x21, 0x5a, 0x33, 0x5c, 0xf5],
};

#[allow(dead_code)]
pub(super) const NV_ENC_PRESET_P7_GUID: Guid = Guid {
    data1: 0x84848c12,
    data2: 0x6f71,
    data3: 0x4c13,
    data4: [0x93, 0x1b, 0x53, 0xe2, 0x83, 0xf5, 0x79, 0x74],
};

/// Rebuild a `Guid` from the adapter's raw 16-byte form. Keeps the
/// adapter independent of the SDK struct definition.
pub(super) fn guid_from_bytes(bytes: [u8; 16]) -> Guid {
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
pub(super) type CUresult = c_int;
pub(super) type CUdevice = c_int;
pub(super) type CUcontext = *mut c_void;

pub(super) type FnCuInit = unsafe extern "C" fn(c_uint) -> CUresult;
pub(super) type FnCuDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
pub(super) type FnCuCtxCreate = unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult;
pub(super) type FnCuCtxDestroy = unsafe extern "C" fn(CUcontext) -> CUresult;
pub(super) type FnCuCtxPushCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
pub(super) type FnCuCtxPopCurrent = unsafe extern "C" fn(*mut CUcontext) -> CUresult;

// ─── NVENC function-list version + fn-pointer type aliases ───────

pub(super) const NV_ENCODE_API_FUNCTION_LIST_VER: u32 = struct_version(2);

pub(super) type FnNvEncodeAPIGetMaxSupportedVersion = unsafe extern "C" fn(*mut u32) -> c_uint;
pub(super) type FnNvEncodeAPICreateInstance =
    unsafe extern "C" fn(*mut super::buffers::NvEncFunctionList) -> c_uint;

pub(super) type FnNvEncGetEncodeGUIDCount =
    unsafe extern "C" fn(*mut c_void, *mut u32) -> c_uint;
pub(super) type FnNvEncGetEncodeGUIDs =
    unsafe extern "C" fn(*mut c_void, *mut Guid, u32, *mut u32) -> c_uint;
pub(super) type FnNvEncGetEncodeCaps =
    unsafe extern "C" fn(*mut c_void, Guid, *mut NvEncCapsParam, *mut c_int) -> c_uint;
pub(super) type FnNvEncOpenEncodeSessionEx =
    unsafe extern "C" fn(*mut super::ffi::NvEncOpenEncodeSessionExParams, *mut *mut c_void)
        -> c_uint;
pub(super) type FnNvEncInitializeEncoder =
    unsafe extern "C" fn(*mut c_void, *mut super::ffi::NvEncInitializeParams) -> c_uint;
pub(super) type FnNvEncCreateInputBuffer =
    unsafe extern "C" fn(*mut c_void, *mut super::buffers::NvEncCreateInputBuffer) -> c_uint;
pub(super) type FnNvEncDestroyInputBuffer =
    unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_uint;
pub(super) type FnNvEncCreateBitstreamBuffer =
    unsafe extern "C" fn(*mut c_void, *mut super::buffers::NvEncCreateBitstreamBuffer) -> c_uint;
pub(super) type FnNvEncDestroyBitstreamBuffer =
    unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_uint;
pub(super) type FnNvEncLockInputBuffer =
    unsafe extern "C" fn(*mut c_void, *mut super::buffers::NvEncLockInputBuffer) -> c_uint;
pub(super) type FnNvEncUnlockInputBuffer =
    unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_uint;
pub(super) type FnNvEncEncodePicture =
    unsafe extern "C" fn(*mut c_void, *mut super::buffers::NvEncPicParams) -> c_uint;
pub(super) type FnNvEncLockBitstream =
    unsafe extern "C" fn(*mut c_void, *mut super::buffers::NvEncLockBitstream) -> c_uint;
pub(super) type FnNvEncUnlockBitstream =
    unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_uint;
pub(super) type FnNvEncDestroyEncoder =
    unsafe extern "C" fn(*mut c_void) -> c_uint;
/// `NvEncReconfigureEncoder(encoder, &reconfigure_params)`. The one entry
/// point that restarts a session in place — with `resetEncoder` set it
/// clears the rate-control state and, with `forceIDR`, opens a new GOP on
/// the next picture. `Encoder::reset` is built on it.
pub(super) type FnNvEncReconfigureEncoder =
    unsafe extern "C" fn(*mut c_void, *mut super::ffi::NvEncReconfigureParams) -> c_uint;
/// `NvEncGetEncodePresetConfigEx(encoder, encodeGuid, presetGuid, tuningInfo, &preset_cfg)`.
/// SDK 12.2 entry; `Ex` variant takes tuning info so the seeded config
/// reflects both preset + tuning rather than preset only.
pub(super) type FnNvEncGetEncodePresetConfigEx =
    unsafe extern "C" fn(*mut c_void, Guid, Guid, u32, *mut super::ffi::NvEncPresetConfig)
        -> c_uint;

// Squad-22: pin the 10-bit buffer-format constant. The SDK
// enumeration is `0x00010000`; if that ever changes (NVIDIA splits
// 10-bit into per-format variants in a future SDK rev) the dispatch
// in `nvenc_buffer_format_for` would silently mis-route.
const _: () = assert!(NV_ENC_BUFFER_FORMAT_YUV420_10BIT == 0x00010000);
// And the 8-bit IYUV constant for symmetry — both must agree with
// `vendor/nvidia/nvEncodeAPI.h:94-115`.
const _: () = assert!(NV_ENC_BUFFER_FORMAT_IYUV == 0x00000100);
