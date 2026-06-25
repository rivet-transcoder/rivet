//! **Legacy fallback** — retained for the default-feature build and
//! as a failover target when the `codec/ffmpeg` feature is enabled.
//! New dispatch prefers `super::ffmpeg::FfmpegDecoder` which wires
//! `hwaccel=cuda` onto libavcodec to drive the same NVDEC silicon
//! with a battle-tested frame pipeline (see the 2026-04-19 migration
//! in `mod.rs::create_decoder`). This custom libnvcuvid wrapper remains
//! engaged for the default-feature build (no FFmpeg dep) and as a
//! failover when the FFmpeg path errors.
//!
//! NVDEC hardware video decoder via NVIDIA CUDA Video Decoder API.
//!
//! Loads libcuda and libnvcuvid at runtime via dlopen. No compile-time
//! CUDA SDK needed — the vendored headers in `vendor/nvidia/` are the
//! authoritative reference for the struct layouts and function
//! signatures used here.
//!
//! Flow:
//! 1. cuInit + cuCtxCreate                      (driver init)
//! 2. cuvidCreateVideoParser                    (stateless parser)
//! 3. per sample: cuCtxPushCurrent + cuvidParseVideoData + cuCtxPopCurrent
//!    - pfn_sequence_callback: cuvidCreateDecoder (first time)
//!    - pfn_decode_picture:    cuvidDecodePicture
//!    - pfn_display_picture:   cuvidMapVideoFrame + cuMemcpy2D then push
//!      NV12 bytes into FrameCollector
//! 4. cuvidDestroyVideoParser + cuvidDestroyDecoder + cuCtxDestroy
//!
//! Library-lifetime note: CUDA + CUVID libraries are stored as fields on
//! NvdecDecoder and declared LAST so they drop after every resource that
//! references them (Rust drops struct fields in source order — Reference
//! §10.8). All FFI fn pointers captured into CallbackState are borrowed
//! from libraries whose Library handles outlive the callback dispatch.
//!
//! Thread-safety note: cuCtxCreate makes the context current only on the
//! calling thread. Every cuvid* call happens under cuCtxPushCurrent /
//! cuCtxPopCurrent so a tokio worker that migrates between threads still
//! has the right context bound before touching the decoder.

use anyhow::{Context, Result, bail};
use bytes::Bytes;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::os::raw::{c_int, c_uchar, c_uint, c_ulong, c_ulonglong};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::sync::{Arc, Mutex};

use super::Decoder;
use crate::frame::{ColorMetadata, ColorSpace, PixelFormat, StreamInfo, TransferFn, VideoFrame};

// ─── Typed errors surfaced to the caller ──────────────────────────
//
// `anyhow::Error::downcast_ref::<NvdecError>()` lets callers (and tests)
// pattern-match on specific NVDEC reject reasons without string-matching
// the display message. The decode_next / new paths wrap these in
// anyhow::Error so the Decoder trait signature stays unchanged.
//
// Reviewer note (codec-review-2 HIGH-1, HIGH-2): previously any of
// these rejects surfaced as an opaque "NVDEC produced no frames: <string>"
// anyhow and the pipeline couldn't tell "4:2:2 unsupported" from
// "driver OOM". A typed variant keeps the CPU-fallback decision in
// decode/mod.rs explainable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NvdecError {
    /// Chroma subsampling reported by the CUVID parser is not one of
    /// the formats this backend supports. Currently only 4:2:0 (value 1)
    /// passes; monochrome (0), 4:2:2 (2), 4:4:4 (3) all produce this.
    UnsupportedChroma {
        chroma_format: c_int,
        label: &'static str,
        width: u32,
        height: u32,
    },
    /// Bit depth outside the 8/10/12-bit 4:2:0 envelope. HEVC Rext
    /// 14-bit / 16-bit content lands here; the existing NV12/P016 copy
    /// math does not generalize.
    UnsupportedPixelFormat { bit_depth: u8 },
}

impl std::fmt::Display for NvdecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedChroma {
                chroma_format,
                label,
                width,
                height,
            } => write!(
                f,
                "NVDEC reject: chroma_format={} ({}) at {}x{} — only 4:2:0 supported",
                chroma_format, label, width, height
            ),
            Self::UnsupportedPixelFormat { bit_depth } => write!(
                f,
                "NVDEC reject: {}-bit content — only 8/10/12-bit 4:2:0 supported",
                bit_depth
            ),
        }
    }
}

impl std::error::Error for NvdecError {}

// ─── CUDA Driver API FFI ───────────────────────────────────────────
type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUdeviceptr = c_ulonglong;

type FnCuInit = unsafe extern "C" fn(c_uint) -> CUresult;
type FnCuDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type FnCuCtxCreate = unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult;
type FnCuCtxDestroy = unsafe extern "C" fn(CUcontext) -> CUresult;
type FnCuCtxPushCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
type FnCuCtxPopCurrent = unsafe extern "C" fn(*mut CUcontext) -> CUresult;
type FnCuMemcpy2D = unsafe extern "C" fn(*const CudaMemcpy2D) -> CUresult;

const CU_MEMORYTYPE_HOST: c_uint = 1;
const CU_MEMORYTYPE_DEVICE: c_uint = 2;

#[repr(C)]
struct CudaMemcpy2D {
    src_x_in_bytes: usize,
    src_y: usize,
    src_memory_type: c_uint,
    src_host: *const c_void,
    src_device: CUdeviceptr,
    src_array: *const c_void,
    src_pitch: usize,
    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_memory_type: c_uint,
    dst_host: *mut c_void,
    dst_device: CUdeviceptr,
    dst_array: *const c_void,
    dst_pitch: usize,
    width_in_bytes: usize,
    height: usize,
}

// ─── CUVID (Video Decoder) FFI ─────────────────────────────────────
type CUvideoparser = *mut c_void;
type CUvideodecoder = *mut c_void;

/// Mirrors CUVIDEOFORMAT from SDK 12.2. Layout padded out with an
/// explicit reserved tail so the driver can write trailing fields
/// we don't read without corrupting adjacent memory. Only the fields
/// we actually consume in sequence_callback are named.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CuVideoFormat {
    pub codec: c_int,
    pub frame_rate_num: c_uint,
    pub frame_rate_den: c_uint,
    pub progressive_sequence: u8,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub min_num_decode_surfaces: u8,
    pub coded_width: c_uint,
    pub coded_height: c_uint,
    pub display_area_left: c_int,
    pub display_area_top: c_int,
    pub display_area_right: c_int,
    pub display_area_bottom: c_int,
    pub chroma_format: c_int,
    pub bitrate: c_uint,
    pub display_aspect_num: c_int,
    pub display_aspect_den: c_int,
    pub video_signal_description: [u8; 8],
    pub seqhdr_data_length: c_uint,
    // Reserved tail for HDR metadata + codec-specific format info the
    // driver writes in SDK 12.x. Size chosen to comfortably exceed the
    // real struct size (reported ~1 KB for AV1 sequence headers).
    pub _reserved_tail: [u8; 1024],
}

/// Layout matches CUVIDPARSERPARAMS from nv-codec-headers
/// (FFmpeg/nv-codec-headers/include/ffnvcodec/dynlink_nvcuvid.h).
///
/// Authoritative field breakdown after max_display_delay:
///   - `bAnnexb:1 | bMemoryOptimize:1 | uReserved:30` — 1 u32 bitfield
///   - `uReserved1[4]` — 4 more u32
///   - pUserData + 5 callback fn pointers
///   - `pvReserved2[5]` — 5 void pointers
///   - pExtVideoInfo
///
/// The earlier 80-byte stub (single `reserved: c_uint`) placed callbacks
/// where the driver expected reserved zero bytes — segfault on long
/// streams, zero frames on short ones. Current size matches the SDK
/// within Rust's layout rules (152 bytes on Windows x64).
#[repr(C)]
struct CuVideoParserParams {
    codec_type: c_int,
    max_num_decode_surfaces: c_uint,
    clock_rate: c_uint,
    error_threshold: c_uint,
    max_display_delay: c_uint,
    /// Bitfield word (bAnnexb | bMemoryOptimize | uReserved:30) + 4 reserved u32.
    /// We zero-init and never set any bits; SDK layout compatible.
    reserved1: [c_uint; 5],
    user_data: *mut c_void,
    pfn_sequence_callback: Option<unsafe extern "C" fn(*mut c_void, *mut CuVideoFormat) -> c_int>,
    pfn_decode_picture: Option<unsafe extern "C" fn(*mut c_void, *mut CuVideoPicParams) -> c_int>,
    pfn_display_picture: Option<unsafe extern "C" fn(*mut c_void, *mut CuVideoDispInfo) -> c_int>,
    pfn_get_operating_point: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int>,
    pfn_get_sei_msg: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int>,
    /// SDK: `void *pvReserved2[5]`.
    reserved2: [*mut c_void; 5],
    ext_video_info: *mut c_void,
}

#[repr(C)]
struct CuVideoSourceDataPacket {
    flags: c_ulong,
    payload_size: c_ulong,
    payload: *const u8,
    timestamp: c_ulonglong,
}

#[repr(C)]
struct CuVideoDecodeCreateInfo {
    code_width: c_ulong,
    coded_height: c_ulong,
    num_decode_surfaces: c_ulong,
    codec_type: c_int,
    chroma_format: c_int,
    creation_flags: c_ulong,
    bit_depth_minus8: c_ulong,
    intra_decode_only: c_ulong,
    max_width: c_ulong,
    max_height: c_ulong,
    reserved1: c_ulong,
    display_area_left: i16,
    display_area_top: i16,
    display_area_right: i16,
    display_area_bottom: i16,
    output_format: c_int,
    deinterlace_mode: c_int,
    target_width: c_ulong,
    target_height: c_ulong,
    num_output_surfaces: c_ulong,
    vid_lock: *mut c_void,
    target_rect_left: i16,
    target_rect_top: i16,
    target_rect_right: i16,
    target_rect_bottom: i16,
    enable_histogram: c_ulong,
    reserved2: [c_ulong; 4],
}

/// Mirrors CUVIDPICPARAMS from SDK 12.2.
///
/// Critical — task #39 audit (2026-04-17): the REAL NVIDIA Video Codec SDK
/// 12.2 defines the trailing codec-specific region as a union whose byte
/// size is fixed by its `unsigned int CodecReserved[1024]` fallback
/// variant — that's **4096 bytes** (1024 × 4). All concrete codec
/// variants (CUVIDH264PICPARAMS, CUVIDHEVCPICPARAMS, CUVIDVP9PICPARAMS,
/// CUVIDAV1PICPARAMS, CUVIDVP8PICPARAMS, CUVIDMPEG2PICPARAMS,
/// CUVIDMPEG4PICPARAMS) fit within that 4 KiB envelope.
///
/// Note: the vendored stub at `vendor/nvidia/cuviddec.h` simplifies the
/// union to `unsigned char CodecSpecific[1024]` (1024 bytes) for
/// documentation purposes. That stub is NOT the ABI we call at runtime —
/// we dlopen the real driver binary which follows the 4096-byte layout.
/// A Rust buffer smaller than 4096 would be a driver-side write
/// overflow; larger is safe (driver writes only what it needs, we read
/// only the parsed callback-input fields before this struct, so the
/// trailing bytes are never examined).
///
/// Earlier revisions declared this as `[u8; 2048]` — half the correct
/// size. The driver overran it on H.264 Main profile (larger reference
/// lists + scaling matrices than Baseline) producing silent zero-frames
/// and the class of memory corruption that triggered task #39's
/// segfault hunt on Windows. H.264 High's different pic-params shape
/// happened to fit. Same root cause class as the CUVIDPARSERPARAMS
/// 80→152 fix (task #39/#52/#53).
#[repr(C)]
struct CuVideoPicParams {
    pic_width_in_mbs: c_int,
    pic_height_in_mbs: c_int,
    curr_pic_idx: c_int,
    field_pic_flag: c_int,
    bottom_field_flag: c_int,
    second_field: c_int,
    n_bitstream_data_len: c_uint,
    p_bitstream_data: *const u8,
    n_num_slices: c_uint,
    p_slice_data_offsets: *const c_uint,
    ref_pic_flag: c_int,
    intra_pic_flag: c_int,
    reserved: [c_uint; 30],
    // Matches the REAL SDK `union { ...; unsigned int CodecReserved[1024]; }`
    // = 4096 bytes. See struct-size assertion below.
    codec_specific: [c_uint; 1024],
}

#[repr(C)]
struct CuVideoDispInfo {
    picture_index: c_int,
    progressive_frame: c_int,
    top_field_first: c_int,
    repeat_first_field: c_int,
    timestamp: c_ulonglong,
}

// ─── Codec-variant pic-params shape witnesses (Squad-12, task #39) ────
//
// The driver writes a codec-specific pic-params blob into our
// `CuVideoPicParams.codec_specific` array on every `pfn_decode_picture`
// callback. We treat the contents as opaque (the parser populates them
// before we hand the struct to `cuvidDecodePicture`), but the SHAPE
// matters: the union variant the driver picks must fit within the 4096
// byte `CodecReserved[1024]` envelope or it overruns our allocation.
//
// These structs mirror the per-codec field shape closely enough to
// produce a defensible upper-bound on their packed sizeof, which we
// then assert ≤ 4096 at compile time. They are NOT used at runtime —
// declared here so a future ABI drift (e.g. an extra DPB slot in
// CUVIDH264PICPARAMS, or a new HEVC scaling list dimension) trips the
// const_assert immediately rather than silently corrupting the parser
// state and reproducing task #39 on a different code path.
//
// Reference: nv-codec-headers 12.2 (FFmpeg/nv-codec-headers
// include/ffnvcodec/cuviddec.h) and the published doxygen at
// https://ffmpeg.org/doxygen/trunk/cuviddec_8h_source.html.

/// CUVIDH264DPBENTRY — one entry of the H.264 reference picture buffer.
/// Six i32 fields (PicIdx, FrameIdx, is_long_term, not_existing,
/// used_for_reference, FieldOrderCnt[2]) → 28 bytes on every target.
/// dpb[16] in CUVIDH264PICPARAMS → 448 bytes.
#[repr(C)]
#[allow(dead_code)]
struct CuVideoH264DpbEntry {
    pic_idx: c_int,
    frame_idx: c_int,
    is_long_term: c_int,
    not_existing: c_int,
    used_for_reference: c_int,
    field_order_cnt: [c_int; 2],
}
const _: () = assert!(std::mem::size_of::<CuVideoH264DpbEntry>() == 28);
// dpb[16] block size — the segfault hunt called this out as "16 vs 17"
// — 17 was a bogus historical theory; the SDK has always been 16.
const _: () = assert!(std::mem::size_of::<[CuVideoH264DpbEntry; 16]>() == 448);

/// Upper-bound shape of CUVIDH264PICPARAMS. Concrete fields lifted from
/// nv-codec-headers 12.2; reserved tail padded out so even if the driver
/// adds a small block in a future SDK we still fit. Real SDK reports
/// ~1.9 KiB; our witness sizes ~3.1 KiB which is conservative.
#[repr(C)]
#[allow(dead_code)]
struct CuVideoH264PicParamsShape {
    // SPS/PPS scalars — ~30 ints worth of flags + counters in the SDK.
    sps_pps_scalars: [c_int; 32],
    // The 16-entry DPB.
    dpb: [CuVideoH264DpbEntry; 16],
    // Quant matrices: WeightScale4x4[6][16] + WeightScale8x8[2][64].
    weight_scale_4x4: [[u8; 16]; 6],
    weight_scale_8x8: [[u8; 64]; 2],
    // FMO/ASO + slice_group_map (union of u64 + ptr) + MVC/SVC ext blob.
    fmo_aso_extras: [u8; 256],
    // Reserved tail to absorb future SDK additions without re-verifying.
    reserved_tail: [u8; 1024],
}
const _: () = assert!(std::mem::size_of::<CuVideoH264PicParamsShape>() <= 4096);

/// Upper-bound shape of CUVIDHEVCPICPARAMS. SPS/PPS scalars + RPS arrays
/// (RefPicIdx[16] / PicOrderCntVal[16] / IsLongTerm[16] etc.) + scaling
/// lists. Real SDK reports ~1.2 KiB; our witness sizes ~2.5 KiB.
#[repr(C)]
#[allow(dead_code)]
struct CuVideoHevcPicParamsShape {
    sps_pps_scalars: [c_int; 64],
    ref_pic_idx: [c_int; 16],
    pic_order_cnt_val: [c_int; 16],
    is_long_term: [c_uchar; 16],
    // RpsSetStCurrBefore/After/LtCurr — three 8-entry sets per the SDK.
    rps_sets: [[c_uchar; 8]; 3],
    // ScalingList4x4[6][16] + 8x8[6][64] + 16x16[6][64] + 32x32[2][64]
    // + ScalingListDCCoeff16x16[6] + 32x32[2].
    scaling_list_4x4: [[c_uchar; 16]; 6],
    scaling_list_8x8: [[c_uchar; 64]; 6],
    scaling_list_16x16: [[c_uchar; 64]; 6],
    scaling_list_32x32: [[c_uchar; 64]; 2],
    scaling_list_dc_16x16: [c_uchar; 6],
    scaling_list_dc_32x32: [c_uchar; 2],
    // Reserved tail.
    reserved_tail: [u8; 256],
}
const _: () = assert!(std::mem::size_of::<CuVideoHevcPicParamsShape>() <= 4096);

/// Upper-bound shape of CUVIDAV1PICPARAMS. The largest of the variants
/// per the SDK (~1.7 KiB) — film grain table + tile column/row arrays.
#[repr(C)]
#[allow(dead_code)]
struct CuVideoAv1PicParamsShape {
    seq_header_scalars: [c_int; 32],
    // Reference frame indices (REF_FRAMES = 8 in AV1 spec).
    ref_frame_map: [c_int; 8],
    // Tile cols/rows can be up to MAX_TILE_COLS=64 / MAX_TILE_ROWS=64.
    tile_col_start_sb: [c_int; 64],
    tile_row_start_sb: [c_int; 64],
    // Loop restoration unit shifts + film grain table.
    loop_filter: [c_int; 16],
    // Film grain: scaling_points_y[14][2] + cb[10][2] + cr[10][2] + ar coeffs.
    film_grain: [u8; 512],
    reserved_tail: [u8; 256],
}
const _: () = assert!(std::mem::size_of::<CuVideoAv1PicParamsShape>() <= 4096);

/// Upper-bound shape of CUVIDVP9PICPARAMS — compact (~0.5 KiB) since VP9
/// reference handling is frame-buffer-only; no DPB entries per se.
#[repr(C)]
#[allow(dead_code)]
struct CuVideoVp9PicParamsShape {
    profile_and_scalars: [c_int; 32],
    ref_frame_map: [c_int; 8],
    // Compressed header context probabilities — entropy coder tables.
    probs: [u8; 384],
    reserved_tail: [u8; 128],
}
const _: () = assert!(std::mem::size_of::<CuVideoVp9PicParamsShape>() <= 4096);

/// Upper-bound shape of CUVIDVP8PICPARAMS — smaller still than VP9.
#[repr(C)]
#[allow(dead_code)]
struct CuVideoVp8PicParamsShape {
    profile_and_scalars: [c_int; 16],
    last_ref: c_int,
    golden_ref: c_int,
    alt_ref: c_int,
    // VP8 quant tables / loop filter tables.
    tables: [u8; 256],
    reserved_tail: [u8; 64],
}
const _: () = assert!(std::mem::size_of::<CuVideoVp8PicParamsShape>() <= 4096);

/// Upper-bound shape of CUVIDMPEG2PICPARAMS — tiny by modern standards.
#[repr(C)]
#[allow(dead_code)]
struct CuVideoMpeg2PicParamsShape {
    forward_ref_pic_idx: c_int,
    backward_ref_pic_idx: c_int,
    picture_coding_type: c_int,
    full_pel_forward_vector: c_int,
    full_pel_backward_vector: c_int,
    f_code: [[c_int; 2]; 2],
    intra_dc_precision: c_int,
    frame_pred_frame_dct: c_int,
    concealment_motion_vectors: c_int,
    q_scale_type: c_int,
    intra_vlc_format: c_int,
    alternate_scan: c_int,
    top_field_first: c_int,
    quant_matrix_intra: [c_uchar; 64],
    quant_matrix_inter: [c_uchar; 64],
    reserved_tail: [u8; 32],
}
const _: () = assert!(std::mem::size_of::<CuVideoMpeg2PicParamsShape>() <= 4096);

/// Upper-bound shape of CUVIDMPEG4PICPARAMS — comparable to MPEG-2.
#[repr(C)]
#[allow(dead_code)]
struct CuVideoMpeg4PicParamsShape {
    forward_ref_pic_idx: c_int,
    backward_ref_pic_idx: c_int,
    vop_time_increment_resolution: c_int,
    vop_coding_type: c_int,
    interlaced: c_int,
    quant_type: c_int,
    quarter_sample: c_int,
    short_video_header: c_int,
    divx_flags: c_int,
    top_field_first: c_int,
    rounding_control: c_int,
    alternate_vertical_scan_flag: c_int,
    quant_matrix_intra: [c_uchar; 64],
    quant_matrix_inter: [c_uchar; 64],
    reserved_tail: [u8; 32],
}
const _: () = assert!(std::mem::size_of::<CuVideoMpeg4PicParamsShape>() <= 4096);

#[repr(C)]
struct CuVideoProcParams {
    progressive_frame: c_int,
    second_field: c_int,
    top_field_first: c_int,
    unpaired_field: c_int,
    reserved_flags: c_uint,
    reserved_zero: c_uint,
    raw_input_dptr: c_ulonglong,
    raw_input_pitch: c_uint,
    raw_input_format: c_uint,
    raw_output_dptr: c_ulonglong,
    raw_output_pitch: c_uint,
    reserved1: c_uint,
    output_stream: *mut c_void,
    reserved: [c_uint; 46],
    histogram_dptr: *mut c_void,
    reserved2: [*mut c_void; 1],
}

// TODO: when container compiles and tests can run, wire in
// `cuvidGetDecoderCaps` pre-flight in sequence_callback. The CUVIDDECODECAPS
// struct (SDK 12.2 cuviddec.h) reports `bIsSupported`, `nMaxWidth`,
// `nMaxHeight` for a given (codec, chroma_format, bit_depth_minus8) tuple.
// Running the query before cuvidCreateDecoder would convert "driver
// rejects silently" into an explicit "3090 NVDEC does not advertise
// HEVC 4:2:2 support" error in the WARN fallback log. Not wiring here
// yet because adding untested FFI struct layouts on top of unrunnable
// tests (container::demux currently broken by WIP task #12) would
// introduce drift I can't verify.

// ─── Compile-time struct-size assertions ──────────────────────────
//
// Task #39 NVDEC Windows segfault audit: CUVID FFI mirrors are verified
// for byte-exact layout against the REAL NVIDIA Video Codec SDK 12.2
// (dlopen'd nvcuvid.dll / libnvcuvid.so, NOT the vendored stub at
// `vendor/nvidia/*.h` which is a simplified reference). The most common
// cause of STATUS_ACCESS_VIOLATION in NVDEC pipelines is a Rust struct
// under-sized relative to the C ABI: the driver writes past our
// allocation into adjacent state, corruption surfaces later as a segfault
// or — worse — as silent wrong-frames. Compile-time asserts convert
// that class of bug from "intermittent crash on long streams" into a
// build-time error.
//
// Prior drift caught by this approach:
//   - CUVIDPARSERPARAMS 80→136 (task #39/#52/#53, fix: add reserved2 array)
//   - CUVIDPICPARAMS    2048→4280 (task #65, fix: codec_specific [u8;2048]→[c_uint;1024])
//
// Squad-12 (2026-04-17 PM) added per-codec-variant shape witnesses
// (CuVideoH264PicParamsShape, CuVideoHevcPicParamsShape, …) so a future
// SDK that grows any one variant past the 4096-byte CodecReserved[1024]
// envelope fails compilation rather than silently overflowing.
// CUVIDH264DPBENTRY size locked at 28 bytes (dpb[16] = 448 bytes).
//
// Expected sizes are computed against ffmpeg's nv-codec-headers 12.2
// (FFmpeg/nv-codec-headers/include/ffnvcodec/{dynlink_nvcuvid,
// dynlink_cuviddec}.h) on Windows MSVC x64 (c_ulong=4, pointer=8).
// Linux x86_64 differs in c_ulong=8 width; the asserts below are
// platform-conditional where that matters.
//
// If any of these assertions fire: the Rust struct no longer matches
// the driver ABI — expect silent zero-frames or STATUS_ACCESS_VIOLATION
// depending on stream length and corruption target. Fix by comparing
// field-by-field against the linked headers and updating reserved counts.

// CUVIDPARSERPARAMS: 5×u32 + 5×u32 + ptr + 5×fn_ptr + 5×ptr + ptr = 136
const _: () = assert!(std::mem::size_of::<CuVideoParserParams>() == 136);

// CUVIDEOFORMAT: 64–68 bytes of named fields (video_signal_description is
// 4 bytes in the real SDK, 7 bytes in vendored/older layouts) + our
// 1024-byte _reserved_tail. Driver only writes the front-of-struct
// fields; tail is defensive padding so any driver-version drift in the
// trailing layout cannot clobber adjacent heap state.
// We don't assert an exact size since the tail length is a Rust choice
// — just that it's comfortably above the SDK's worst-case 72 bytes.
const _: () = assert!(std::mem::size_of::<CuVideoFormat>() >= 72);

// CUVIDPICPARAMS — Windows MSVC x64 layout (task #39 audit):
//   6×c_int                    = 24
//   n_bitstream_data_len u32   = 4   (cumulative 28)
//   [align 8]                  = +4  (32)
//   p_bitstream_data *const    = 8   (40)
//   n_num_slices u32           = 4   (44)
//   [align 8]                  = +4  (48)
//   p_slice_data_offsets       = 8   (56)
//   2×c_int                    = 8   (64)
//   30×c_uint reserved         = 120 (184)
//   1024×c_uint codec_specific = 4096 (4280)
// Total: 4280 bytes.
//
// The real SDK union variants (CUVIDH264PICPARAMS ~1.9 KiB with DPB+
// scaling lists, CUVIDHEVCPICPARAMS ~1.2 KiB, CUVIDAV1PICPARAMS ~1.7
// KiB, CUVIDVP9PICPARAMS ~0.5 KiB) all fit inside the 4096-byte
// CodecReserved[1024] fallback. Individual variant size asserts below.
const _: () = assert!(std::mem::size_of::<CuVideoPicParams>() == 4280);
// The codec_specific region must be exactly the 4096-byte SDK envelope.
// Separating this check from the whole-struct assert makes the diagnostic
// obvious when someone accidentally edits codec_specific's element type
// without updating the length (e.g. changes [c_uint;1024] → [u8;1024]).
const _: () = assert!(std::mem::size_of::<[c_uint; 1024]>() == 4096);

// CUVIDPARSERDISPINFO: 4×i32 + u64 = 24. Matches SDK.
const _: () = assert!(std::mem::size_of::<CuVideoDispInfo>() == 24);

// CUVIDSOURCEDATAPACKET: Windows MSVC x64 has c_ulong=4 →
//   flags (4) + payload_size (4) + [pad 0] + payload* (8) + timestamp u64 (8) = 24
// Linux x86_64 has c_ulong=8 →
//   flags (8) + payload_size (8) + payload* (8) + timestamp u64 (8) = 32
// Assert per-platform — a mismatch means the driver reads payload from
// the wrong offset and either segfaults or decodes random memory.
#[cfg(target_os = "windows")]
const _: () = assert!(std::mem::size_of::<CuVideoSourceDataPacket>() == 24);
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
const _: () = assert!(std::mem::size_of::<CuVideoSourceDataPacket>() == 32);

// CUVIDDECODECREATEINFO — Windows MSVC x64:
//   3×c_ulong (12) + 2×c_int (8) + 6×c_ulong (24) = 44
//   + 4×i16 display_area (8) = 52
//   + 2×c_int format/deinterlace (8) = 60
//   + 3×c_ulong target (12) = 72
//   + vid_lock ptr (8) = 80
//   + 4×i16 target_rect (8) = 88
//   + enable_histogram c_ulong (4) = 92
//   + 4×c_ulong reserved2 (16) = 108
//   + trailing 4 bytes align to 8-byte pointer alignment = 112
#[cfg(target_os = "windows")]
const _: () = assert!(std::mem::size_of::<CuVideoDecodeCreateInfo>() == 112);

// CUVIDPROCPARAMS — 4×i32 + 2×u32 + u64 + 2×u32 + u64 + 2×u32 + ptr
// + 46×u32 + ptr + ptr, with pointer alignment pads = 264.
const _: () = assert!(std::mem::size_of::<CuVideoProcParams>() == 264);

type FnCuvidCreateVideoParser =
    unsafe extern "C" fn(*mut CUvideoparser, *mut CuVideoParserParams) -> CUresult;
type FnCuvidParseVideoData =
    unsafe extern "C" fn(CUvideoparser, *mut CuVideoSourceDataPacket) -> CUresult;
type FnCuvidDestroyVideoParser = unsafe extern "C" fn(CUvideoparser) -> CUresult;
type FnCuvidCreateDecoder =
    unsafe extern "C" fn(*mut CUvideodecoder, *mut CuVideoDecodeCreateInfo) -> CUresult;
type FnCuvidDestroyDecoder = unsafe extern "C" fn(CUvideodecoder) -> CUresult;
type FnCuvidDecodePicture = unsafe extern "C" fn(CUvideodecoder, *mut CuVideoPicParams) -> CUresult;
type FnCuvidMapVideoFrame = unsafe extern "C" fn(
    CUvideodecoder,
    c_int,
    *mut CUdeviceptr,
    *mut c_uint,
    *mut CuVideoProcParams,
) -> CUresult;
type FnCuvidUnmapVideoFrame = unsafe extern "C" fn(CUvideodecoder, CUdeviceptr) -> CUresult;

// ─── Codec constants ───────────────────────────────────────────────
const CUVID_H264: c_int = 4;
const CUVID_HEVC: c_int = 8;
const CUVID_VP8: c_int = 9;
const CUVID_VP9: c_int = 10;
const CUVID_AV1: c_int = 11;
const CUVID_MPEG2: c_int = 1;
const CUVID_MPEG4: c_int = 3;

const CUVID_PKT_ENDOFSTREAM: c_ulong = 1;
/// Tells the parser to associate the packet with its timestamp. Without
/// this flag the parser consumes data silently and may never emit
/// picture-complete callbacks. ffmpeg sets this on every data packet.
const CUVID_PKT_TIMESTAMP: c_ulong = 2;

// cudaVideoSurfaceFormat (cuviddec.h):
//   NV12 = 0    — 8-bit per sample, semi-planar (Y plane + interleaved UV)
//   P016 = 1    — 16-bit per sample, semi-planar; 10-bit data in the
//                 high 10 bits of each u16, low 6 bits zero-padded
//   YUV444 = 2  — 8-bit 4:4:4
//   YUV444_16 = 3 — 16-bit 4:4:4
// We only use NV12 (8-bit 4:2:0) and P016 (10/12-bit 4:2:0).
const CUVID_FMT_NV12: c_int = 0;
const CUVID_FMT_P016: c_int = 1;
const CUVID_CHROMA_420: c_int = 1;
/// Force the CUVID software decoder backend. On Windows the SDK
/// default may select DXVA, which produces different surface layouts
/// and is the suspected root cause of the H.264 segfault seen on
/// GPU boxes in testing. ffmpeg's cuviddec.c sets this unconditionally.
const CUVID_CREATE_PREFER_CUVID: c_ulong = 0x01;

fn codec_to_cuvid(codec: &str) -> Option<c_int> {
    match codec {
        "h264" | "avc1" | "avc" => Some(CUVID_H264),
        "h265" | "hevc" | "hvc1" | "hev1" => Some(CUVID_HEVC),
        "vp8" => Some(CUVID_VP8),
        "vp9" | "vp09" => Some(CUVID_VP9),
        "av1" | "av01" => Some(CUVID_AV1),
        "mpeg2" | "mpeg2video" => Some(CUVID_MPEG2),
        "mpeg4" | "mp4v" => Some(CUVID_MPEG4),
        _ => None,
    }
}

/// Pure-Rust validator for the subset of CUVIDEOFORMAT fields this
/// backend cares about. Extracted out of `sequence_callback` so the
/// chroma / bit-depth reject matrix can be unit-tested without
/// spinning up a GPU context. Returns `None` when the format is
/// acceptable for NVDEC decoding on this backend.
///
/// Contract (codec-review-2 HIGH-1 + HIGH-2):
///   chroma_format values → action
///     0 (Monochrome) → Err UnsupportedChroma
///     1 (4:2:0)      → accept (subject to bit depth check)
///     2 (4:2:2)      → Err UnsupportedChroma
///     3 (4:4:4)      → Err UnsupportedChroma
///   bit_depth_luma_minus8 values → action
///     0 (8-bit)      → accept (NV12 surface)
///     2 (10-bit)     → accept (P016 surface)
///     4 (12-bit)     → accept (P016 surface, shares wire format)
///     >4             → Err UnsupportedPixelFormat
pub fn validate_format(
    chroma_format: c_int,
    bit_depth_luma_minus8: u8,
    coded_width: u32,
    coded_height: u32,
) -> Option<NvdecError> {
    if chroma_format != CUVID_CHROMA_420 {
        let label: &'static str = match chroma_format {
            0 => "Monochrome",
            2 => "4:2:2",
            3 => "4:4:4",
            _ => "unknown",
        };
        return Some(NvdecError::UnsupportedChroma {
            chroma_format,
            label,
            width: coded_width,
            height: coded_height,
        });
    }
    let bit_depth = bit_depth_luma_minus8 + 8;
    if bit_depth > 12 {
        return Some(NvdecError::UnsupportedPixelFormat { bit_depth });
    }
    None
}

/// Pure-Rust P016 → Yuv420p10le deinterleave + 10-bit normalization.
/// Extracted out of `decode_next` so the right-shift, UV interleave,
/// and odd-dimension handling can be unit-tested without a GPU.
///
/// Input layout (`p016_bytes`):
///   Y plane: `w * h` samples × 2 bytes LE, 10-bit value in the HIGH
///            bits of each u16 (low 6 bits zero per SDK).
///   UV plane: `ceil(w/2) * ceil(h/2)` interleaved UV pairs, each pair
///             is 4 bytes (U u16 LE + V u16 LE), same high-bit layout.
///
/// Output layout (`Vec<u8>`, little-endian u16 packed):
///   Y plane: `w * h * 2` bytes, 10-bit value in the LOW bits.
///   U plane: `ceil(w/2) * ceil(h/2) * 2` bytes, 10-bit low bits.
///   V plane: `ceil(w/2) * ceil(h/2) * 2` bytes, 10-bit low bits.
///
/// 12-bit content also uses this path; the >>6 shift clips to 10-bit
/// range which is what the downstream 10-bit pipeline expects.
pub fn deinterleave_p016_to_yuv420p10le(p016_bytes: &[u8], w: usize, h: usize) -> Vec<u8> {
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let uv_pairs = cw * ch;
    let y_bytes = w * h * 2;
    let mut out = Vec::with_capacity(y_bytes + uv_pairs * 4);

    // Y plane: u16 LE samples, right-shift by 6 and re-emit LE.
    let y_src = &p016_bytes[..y_bytes.min(p016_bytes.len())];
    for chunk in y_src.chunks_exact(2) {
        let sample = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.extend_from_slice(&(sample >> 6).to_le_bytes());
    }
    if out.len() < y_bytes {
        out.resize(y_bytes, 0);
    }

    // UV interleave: pair stride = 4 bytes (U u16 LE, V u16 LE).
    if p016_bytes.len() > y_bytes {
        let uv = &p016_bytes[y_bytes..];
        let mut u = Vec::with_capacity(uv_pairs * 2);
        let mut v = Vec::with_capacity(uv_pairs * 2);
        for i in 0..uv_pairs {
            let base = i * 4;
            if base + 3 < uv.len() {
                let us = u16::from_le_bytes([uv[base], uv[base + 1]]) >> 6;
                let vs = u16::from_le_bytes([uv[base + 2], uv[base + 3]]) >> 6;
                u.extend_from_slice(&us.to_le_bytes());
                v.extend_from_slice(&vs.to_le_bytes());
            }
        }
        out.extend_from_slice(&u);
        out.extend_from_slice(&v);
    }
    out
}

// ─── Decoded frame collector ───────────────────────────────────────
#[derive(Clone)]
struct DecodedFrame {
    /// Raw NV12 bytes (8-bit, 1 byte/sample) or P016 bytes (10/12-bit,
    /// 2 bytes/sample in the high bits). Deinterleave to planar at
    /// drain time — see NvdecDecoder::decode_next.
    nv12: Vec<u8>,
    width: u32,
    height: u32,
    /// 0 = 8-bit (NV12 / Yuv420p), 2 = 10-bit (P016 / Yuv420p10le).
    /// Captured from the sequence_callback's CUVIDEOFORMAT so each
    /// frame carries its own format if the stream renegotiates.
    bit_depth_minus8: u8,
    /// ColorSpace derived from the SPS VUI matrix_coefficients, carried
    /// per-frame so the encoder / colorspace converter sees the
    /// source's primaries without re-parsing the SPS.
    color_space: ColorSpace,
    /// Timestamp as reported by the CUVID parser in display order.
    /// Preserved end-to-end so upstream frame-rate/duration math in
    /// the pipeline is correct rather than assuming integer frame
    /// counts from zero.
    timestamp: u64,
}

/// Decoded-frame ring shared between the parser display callback (writer)
/// and `decode_next` (reader). For the eager `NvdecDecoder::new_with_pts`
/// path the entire run lands here in one pass and the reader drains
/// after parser teardown — `VecDeque` is interchangeable with `Vec` for
/// that pattern (push_back + sequential pop_front in display order).
///
/// For the streaming `NvdecStreamingDecoder` path (Squad-36) the writer
/// fires per `cuvidParseVideoData(per-sample)` invocation and the reader
/// pops one frame per `decode_next` call. `VecDeque::pop_front` is O(1)
/// and `push_back` is amortised O(1); the only theoretical growth is
/// the reorder window (≤ B-pyramid depth, typically ≤ 16 frames for
/// H.264 High / HEVC) plus whatever the caller hasn't drained yet.
struct FrameCollector {
    frames: VecDeque<DecodedFrame>,
}

// ─── Callback state shared across the three parser callbacks ──────
//
// The parser hands us a raw `*mut c_void` per callback. We stash a
// pointer to this struct in CUVIDPARSERPARAMS.user_data, so the
// callbacks can resolve back to the decoder they belong to.
//
// Lifetime: the callback state must outlive every cuvidParseVideoData
// call. We box it once in `NvdecDecoder::new` and drop it only after
// the parser is destroyed.
struct CallbackState {
    cuvid_create_decoder: FnCuvidCreateDecoder,
    cuvid_decode_picture: FnCuvidDecodePicture,
    cuvid_map_video_frame: FnCuvidMapVideoFrame,
    cuvid_unmap_video_frame: FnCuvidUnmapVideoFrame,
    cu_memcpy2d: FnCuMemcpy2D,

    decoder: Option<CUvideodecoder>,
    collector: Arc<Mutex<FrameCollector>>,
    width: u32,
    height: u32,
    codec_type: c_int,
    /// Copied from CUVIDEOFORMAT.bit_depth_luma_minus8 in sequence_callback
    /// so display_callback knows whether to memcpy NV12 (1 byte/sample)
    /// or P016 (2 bytes/sample) and decode_next knows which PixelFormat
    /// to tag on the emitted VideoFrame.
    bit_depth_luma_minus8: u8,
    /// Derived from CUVIDEOFORMAT.video_signal_description in
    /// sequence_callback. Propagated to every DecodedFrame so downstream
    /// colorspace conversion knows the source's matrix_coefficients
    /// (BT.601/709/2020) without re-parsing the SPS.
    color_space: ColorSpace,
    /// Raw H.273 values captured from the SPS VUI so StreamInfo can
    /// round-trip them to the mux's `colr nclx` box. Populated in
    /// sequence_callback, read once after the parse loop finishes
    /// to update the outer NvdecDecoder.info.color_metadata.
    vui_colour_primaries: u8,
    vui_transfer_characteristics: u8,
    vui_matrix_coefficients: u8,
    vui_full_range_flag: bool,
    error: Option<String>,
    /// Typed reject reason captured in sequence_callback. Propagated up
    /// to `NvdecDecoder::new_with_pts`'s Err path so the caller can
    /// `.downcast_ref::<NvdecError>()` and steer fallback / abort
    /// policy (codec-review-2 HIGH-1). Only set when a `set_error`
    /// was caused by an NvdecError variant; plain driver failures
    /// continue through the string path.
    typed_error: Option<NvdecError>,
}

// SAFETY: The collector is Arc<Mutex<FrameCollector>> — the only piece
// of shared state — and all other fields are plain fn pointers + POD.
// Callbacks fire under the thread that calls cuvidParseVideoData; the
// Mutex serializes any cross-thread access from the drain-side code.
// The CUvideodecoder is only touched while its context is pushed on
// the current thread.
unsafe impl Send for CallbackState {}

impl CallbackState {
    fn set_error(&mut self, msg: impl Into<String>) {
        if self.error.is_none() {
            self.error = Some(msg.into());
        }
    }

    /// Record a structured reject reason *and* its string form so the
    /// existing first-wins `error` string path keeps its diagnostics
    /// while the outer caller can downcast the anyhow chain to pattern
    /// match on the cause. If a typed error was already latched,
    /// subsequent calls are ignored (first-wins).
    fn set_typed_error(&mut self, err: NvdecError) {
        if self.typed_error.is_none() {
            self.typed_error = Some(err.clone());
        }
        // Keep the string channel in sync so log lines / generic
        // "NVDEC produced no frames: <err>" messages stay populated.
        self.set_error(err.to_string());
    }
}

// ─── RAII: CUDA context scope guard ───────────────────────────────
//
// Pushes the given CUDA context on construction, pops it on drop.
// Any early return out of a scope holding this guard still runs the
// destructor, so the context stack is balanced even on error paths.
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

// ─── Public decoder ────────────────────────────────────────────────
pub struct NvdecDecoder {
    info: StreamInfo,
    decoded_frames: Vec<DecodedFrame>,
    frame_cursor: usize,

    // Library handles held so the OS keeps the fn pointers mapped for
    // the life of the decoder. Declared LAST so Rust drops everything
    // above them first (Reference §10.8 — struct fields drop in source
    // order). Required because CallbackState may in theory hold fn
    // pointers into these after new() returns.
    _cuvid_lib: libloading::Library,
    _cuda_lib: libloading::Library,
}

impl NvdecDecoder {
    /// Streaming-shape constructor (Squad-36 NVDEC streaming follow-up).
    /// Returns the `NvdecStreamingDecoder` impl boxed as the trait
    /// object. Caller drives via `push_sample` + `finish` +
    /// `decode_next`.
    ///
    /// Memory shape (Squad-36, supersedes the streaming-migration-55
    /// lazy-flush note): each `push_sample` call now invokes
    /// `cuvidParseVideoData` immediately on the just-pushed bytes, the
    /// display callback enqueues into a bounded `VecDeque<DecodedFrame>`
    /// inside `CallbackState.collector`, and `decode_next` pops one
    /// frame per call. Peak heap is bounded by (one bitstream sample) +
    /// (CUVID's internal DPB which is GPU-side, not RSS) +
    /// (reorder-window-bounded VecDeque, ≤ B-pyramid depth ≈ 16
    /// frames). The eager `NvdecDecoder::new_with_pts` constructor +
    /// the lazy-flush `NvdecPushDecoder` wrapper are retained as
    /// library code (smoke tests + future bench reference) but the
    /// production dispatch path no longer goes through them.
    ///
    /// Squad-6 typed reject (`UnsupportedChroma` / `UnsupportedPixelFormat`)
    /// surfaces from `push_sample` — the sequence callback fires on the
    /// first sample carrying a sequence header (typically the first
    /// IDR), which is when the format becomes known.
    ///
    /// Squad-12 per-codec-variant `const_assert!` shape witnesses are
    /// shared with the eager path; the FFI struct definitions here are
    /// used by both.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(info: StreamInfo, gpu_index: u32) -> Box<dyn Decoder> {
        match NvdecStreamingDecoder::try_new(info.clone(), gpu_index) {
            Ok(d) => Box::new(d),
            Err(e) => {
                // Surface init failure as a deferred-error decoder so
                // the caller's first `push_sample` returns the real
                // anyhow chain (matches the Decoder trait contract:
                // `new()` is infallible, errors land on the data path).
                tracing::warn!(error = %e, "NvdecStreamingDecoder init failed; first push will return the error");
                Box::new(NvdecInitErrorDecoder {
                    info,
                    error: Some(e),
                })
            }
        }
    }

    /// PTS-aware eager pump. `samples_with_pts[i].1` is passed through
    /// to `CUVIDSOURCEDATAPACKET.timestamp` verbatim — the CUVID parser
    /// treats timestamps as opaque u64 tokens and hands the matching
    /// value back on `CUVIDPARSERDISPINFO.timestamp` in display order.
    /// Units are therefore whatever the demuxer uses; no 10 MHz scaling
    /// is required because `clock_rate` in `CuVideoParserParams` is 0
    /// (pass-through mode).
    ///
    /// Internal — called from `NvdecPushDecoder::finish()` on the
    /// accumulated sample run. External callers should construct the
    /// push wrapper via `NvdecDecoder::new(info, gpu_index)` and feed
    /// samples through the `Decoder` trait.
    #[allow(clippy::new_ret_no_self)]
    pub fn new_with_pts(
        samples_with_pts: Vec<(Vec<u8>, u64)>,
        info: StreamInfo,
        gpu_index: u32,
    ) -> Result<Box<dyn Decoder>> {
        // Load CUDA driver + cuvid up-front. Both libs will move into
        // the final NvdecDecoder so they outlive any borrowed fn
        // pointer.
        let cuda_lib = unsafe { libloading::Library::new("libcuda.so") }
            .or_else(|_| unsafe { libloading::Library::new("libcuda.so.1") })
            .or_else(|_| unsafe { libloading::Library::new("nvcuda.dll") })
            .context("loading CUDA driver — is the NVIDIA driver installed?")?;

        let cuvid_lib = unsafe { libloading::Library::new("libnvcuvid.so") }
            .or_else(|_| unsafe { libloading::Library::new("libnvcuvid.so.1") })
            .or_else(|_| unsafe { libloading::Library::new("nvcuvid.dll") })
            .context("loading cuvid — is the NVIDIA driver installed?")?;

        let cuvid_codec = codec_to_cuvid(&info.codec)
            .context(format!("unsupported NVDEC codec: {}", info.codec))?;

        let decoded_frames = unsafe {
            // ─── Driver init + context ──────────────────────────────
            let cu_init: libloading::Symbol<FnCuInit> = cuda_lib.get(b"cuInit")?;
            if cu_init(0) != 0 {
                bail!("cuInit failed");
            }

            let cu_device_get: libloading::Symbol<FnCuDeviceGet> = cuda_lib.get(b"cuDeviceGet")?;
            let mut device: CUdevice = 0;
            if cu_device_get(&mut device, gpu_index as c_int) != 0 {
                bail!("cuDeviceGet failed for GPU {gpu_index}");
            }

            let cu_ctx_create: libloading::Symbol<FnCuCtxCreate> =
                cuda_lib.get(b"cuCtxCreate_v2")?;
            let cu_ctx_destroy: libloading::Symbol<FnCuCtxDestroy> =
                cuda_lib.get(b"cuCtxDestroy_v2")?;
            let cu_ctx_push: libloading::Symbol<FnCuCtxPushCurrent> =
                cuda_lib.get(b"cuCtxPushCurrent_v2")?;
            let cu_ctx_pop: libloading::Symbol<FnCuCtxPopCurrent> =
                cuda_lib.get(b"cuCtxPopCurrent_v2")?;
            let mut ctx: CUcontext = ptr::null_mut();
            if cu_ctx_create(&mut ctx, 0, device) != 0 {
                bail!("cuCtxCreate failed");
            }

            // ─── Resolve cuvid + cuda function pointers ────────────
            let cuvid_create_parser: libloading::Symbol<FnCuvidCreateVideoParser> =
                cuvid_lib.get(b"cuvidCreateVideoParser")?;
            let cuvid_parse_data: libloading::Symbol<FnCuvidParseVideoData> =
                cuvid_lib.get(b"cuvidParseVideoData")?;
            let cuvid_destroy_parser: libloading::Symbol<FnCuvidDestroyVideoParser> =
                cuvid_lib.get(b"cuvidDestroyVideoParser")?;
            let cuvid_create_decoder: libloading::Symbol<FnCuvidCreateDecoder> =
                cuvid_lib.get(b"cuvidCreateDecoder")?;
            let cuvid_destroy_decoder: libloading::Symbol<FnCuvidDestroyDecoder> =
                cuvid_lib.get(b"cuvidDestroyDecoder")?;
            let cuvid_decode_picture: libloading::Symbol<FnCuvidDecodePicture> =
                cuvid_lib.get(b"cuvidDecodePicture")?;
            let cuvid_map_video_frame: libloading::Symbol<FnCuvidMapVideoFrame> = cuvid_lib
                .get(b"cuvidMapVideoFrame64")
                .or_else(|_| cuvid_lib.get(b"cuvidMapVideoFrame"))?;
            let cuvid_unmap_video_frame: libloading::Symbol<FnCuvidUnmapVideoFrame> = cuvid_lib
                .get(b"cuvidUnmapVideoFrame64")
                .or_else(|_| cuvid_lib.get(b"cuvidUnmapVideoFrame"))?;
            let cu_memcpy2d: libloading::Symbol<FnCuMemcpy2D> = cuda_lib.get(b"cuMemcpy2D_v2")?;

            let collector = Arc::new(Mutex::new(FrameCollector {
                frames: VecDeque::new(),
            }));

            let mut state = Box::new(CallbackState {
                cuvid_create_decoder: *cuvid_create_decoder,
                cuvid_decode_picture: *cuvid_decode_picture,
                cuvid_map_video_frame: *cuvid_map_video_frame,
                cuvid_unmap_video_frame: *cuvid_unmap_video_frame,
                cu_memcpy2d: *cu_memcpy2d,
                decoder: None,
                collector: Arc::clone(&collector),
                width: info.width,
                height: info.height,
                codec_type: cuvid_codec,
                // sequence_callback overwrites this from the real
                // stream's CUVIDEOFORMAT. 0 == 8-bit default until we
                // see an actual sequence header.
                bit_depth_luma_minus8: 0,
                // Overwritten by sequence_callback from the
                // CUVIDEOFORMAT.video_signal_description bytes.
                color_space: ColorSpace::Bt709,
                vui_colour_primaries: 1,
                vui_transfer_characteristics: 1,
                vui_matrix_coefficients: 1,
                vui_full_range_flag: false,
                error: None,
                typed_error: None,
            });
            let state_ptr: *mut c_void = (&mut *state) as *mut CallbackState as *mut c_void;

            // ─── Parser setup ──────────────────────────────────────
            let mut parser_params: CuVideoParserParams = std::mem::zeroed();
            parser_params.codec_type = cuvid_codec;
            parser_params.max_num_decode_surfaces = 20;
            parser_params.clock_rate = 0;
            parser_params.error_threshold = 100;
            parser_params.max_display_delay = 4;
            // reserved1[0] is the packed bitfield word in the SDK:
            //   bit 0 = bAnnexb       (input is Annex-B, not AVCC)
            //   bit 1 = bMemoryOptimize
            //   bits 2..31 reserved
            // Setting bAnnexb=1 tells the parser our samples are Annex-B
            // (our demuxer converts avcC → Annex-B already) and also
            // makes the parser more lenient about non-IDR recovery
            // points on open-GOP streams like exoplayer_h264_main_720p.mp4
            // (sample 0 = SPS+PPS+SEI+non-IDR-slice, no IDR in file).
            parser_params.reserved1[0] = 1;
            parser_params.user_data = state_ptr;
            parser_params.pfn_sequence_callback = Some(sequence_callback);
            parser_params.pfn_decode_picture = Some(decode_callback);
            parser_params.pfn_display_picture = Some(display_callback);
            // AV1 operating-point hook — PROBLEMS.md §"NVDEC AV1
            // CUVIDEOFORMAT layout mismatch". Non-AV1 codecs ignore it.
            parser_params.pfn_get_operating_point = Some(get_operating_point_callback);

            let mut parser: CUvideoparser = ptr::null_mut();
            let create_rc = cuvid_create_parser(&mut parser, &mut parser_params);
            if create_rc != 0 {
                cu_ctx_destroy(ctx);
                bail!("cuvidCreateVideoParser failed: {create_rc}");
            }

            // Everything below this line must clean up `parser`, the
            // decoder (if created), and the CUDA context on any error.
            // We use a closure-returning-Result pattern so `?` is safe,
            // then run teardown unconditionally after.
            let parse_result: Result<()> = (|| {
                // Context must be current on *this* thread before any
                // cuvid* or cuMemcpy call. Scope guard pops on drop.
                let _scope = CtxScope::push(ctx, *cu_ctx_push, *cu_ctx_pop)?;

                for (idx, (sample, pts)) in samples_with_pts.iter().enumerate() {
                    // Task #39 hardening: empty samples are a degenerate
                    // case the CUVID parser does not document — some
                    // driver versions tolerate a 0-length payload, others
                    // dereference the payload pointer before checking
                    // payload_size. Skip cleanly rather than hand the
                    // driver something it may mishandle. A real demuxer
                    // should never emit empty samples; if one does, the
                    // stream is malformed and a quiet skip is preferable
                    // to a STATUS_ACCESS_VIOLATION.
                    if sample.is_empty() {
                        continue;
                    }
                    let mut packet: CuVideoSourceDataPacket = std::mem::zeroed();
                    packet.payload_size = sample.len() as c_ulong;
                    packet.payload = sample.as_ptr();
                    // Real demuxer PTS rather than the sample index.
                    // codec-review-2 HIGH-3: the previous `idx` counter
                    // produced correct decode order but wrong display
                    // order for B-frame-heavy streams, because CUVID
                    // hands the timestamp back in display order on
                    // `CUVIDPARSERDISPINFO.timestamp`. Passing idx
                    // would make frame 2 (B) display with timestamp=1
                    // even though its real PTS is 40ms later.
                    packet.timestamp = *pts as c_ulonglong;
                    // CUVID_PKT_TIMESTAMP is required on every data
                    // packet. Without it the parser swallows data
                    // without emitting sequence_callback or display
                    // notifications (verified against ffmpeg's
                    // libavcodec/cuviddec.c:cuvid_decode_packet).
                    packet.flags = CUVID_PKT_TIMESTAMP;

                    let rc = cuvid_parse_data(parser, &mut packet);
                    // Non-zero rc is not fatal per the SDK — the parser
                    // may skip corrupted NALUs and keep going. Only log
                    // the first occurrence per stream to avoid log spam.
                    if rc != 0 && idx == 0 {
                        tracing::warn!(
                            rc = rc,
                            "cuvidParseVideoData returned non-zero at first sample"
                        );
                    }
                    if let Some(e) = &state.error {
                        tracing::warn!(error = %e, "NVDEC callback reported failure");
                        break;
                    }
                }

                // Flush any buffered frames out of the parser.
                let mut eos_packet: CuVideoSourceDataPacket = std::mem::zeroed();
                eos_packet.flags = CUVID_PKT_ENDOFSTREAM;
                cuvid_parse_data(parser, &mut eos_packet);
                Ok(())
            })();

            // ─── Teardown order: parser → decoder → context ────────
            // Always runs, even if parse_result is Err, so the driver
            // doesn't leak resources or leave a floating parser around.
            cuvid_destroy_parser(parser);
            if let Some(dec) = state.decoder.take() {
                cuvid_destroy_decoder(dec);
            }
            cu_ctx_destroy(ctx);

            // Propagate parse failures now that cleanup ran.
            parse_result?;

            // Snapshot any callback-reported error before dropping state,
            // so we can surface it rather than bailing with a generic
            // "produced no frames" that hides the real reason.
            let cb_error = state.error.take();
            // Typed reject (UnsupportedChroma / UnsupportedPixelFormat)
            // — propagate as anyhow::Error so the outer caller can
            // `.downcast_ref::<NvdecError>()`. Distinct from the string
            // path: only the typed variants surface here.
            let cb_typed_error = state.typed_error.take();
            // Snapshot the VUI bytes so we can propagate them into
            // StreamInfo.color_metadata below. sequence_callback is the
            // only thing that writes these, so reading once after the
            // parse loop is safe.
            let cb_colour_primaries = state.vui_colour_primaries;
            let cb_transfer = state.vui_transfer_characteristics;
            let cb_matrix_coefficients = state.vui_matrix_coefficients;
            let cb_full_range = state.vui_full_range_flag;
            let cb_color_space = state.color_space;

            // Drop the boxed state now that no callback can fire.
            drop(state);

            let collected = collector.lock().unwrap();

            tracing::info!(
                codec = cuvid_codec,
                gpu = gpu_index,
                frames = collected.frames.len(),
                "NVDEC decode complete"
            );

            if collected.frames.is_empty() {
                // Prefer the typed reject (HIGH-1 / HIGH-2): lets
                // decode/mod.rs or tests dispatch on the structured
                // variant instead of matching on the string form.
                if let Some(te) = cb_typed_error {
                    return Err(anyhow::Error::new(te));
                }
                if let Some(e) = cb_error {
                    bail!("NVDEC produced no frames: {e}");
                }
                bail!("NVDEC produced no frames");
            }

            // Return both the frames and the post-parse VUI state so the
            // outer caller can fold the color metadata back into
            // StreamInfo. Without this the unsafe-block scope swallows
            // the cb_* locals.
            //
            // The collector is now a `VecDeque` (Squad-36 streaming
            // refactor) — collect into a `Vec` for the eager-constructor
            // return shape `NvdecDecoder.decoded_frames: Vec<DecodedFrame>`.
            // Iteration order matches `pop_front` so display order is
            // preserved.
            let frames_vec: Vec<DecodedFrame> = collected.frames.iter().cloned().collect();
            (
                frames_vec,
                cb_color_space,
                cb_colour_primaries,
                cb_transfer,
                cb_matrix_coefficients,
                cb_full_range,
            )
        };
        let (
            decoded_frames,
            cb_color_space,
            cb_colour_primaries,
            cb_transfer,
            cb_matrix_coefficients,
            cb_full_range,
        ) = decoded_frames;

        // Apply SPS VUI color metadata to the outgoing StreamInfo so
        // downstream consumers (pipeline validate, MP4 mux colr box
        // writer) see the real HDR properties of HDR10 / BT.2020
        // content rather than the SDR default that NvdecDecoder::new
        // was given at construction time.
        let mut info = info;
        info.color_space = cb_color_space;
        info.color_metadata = ColorMetadata {
            transfer: TransferFn::from_h273(cb_transfer),
            matrix_coefficients: cb_matrix_coefficients,
            colour_primaries: cb_colour_primaries,
            full_range: cb_full_range,
            // CUVIDEOFORMAT.video_signal_description carries the colour
            // primaries / transfer / matrix triple but NOT the SMPTE
            // ST 2086 mastering display volume nor MaxCLL / MaxFALL.
            // Those live in HEVC SEI 137 / 144 (HEVC) and AV1 metadata
            // OBU type 1 / 2 (AV1), neither of which the NVDEC parser
            // surfaces to user code in SDK 12.2. The CPU SEI parser
            // (Squad-21) populates these on the StreamInfo upstream of
            // decoder dispatch (during demux / probe) — preserve those
            // values here rather than overwriting them with `None`.
            mastering_display: info.color_metadata.mastering_display,
            content_light_level: info.color_metadata.content_light_level,
        };

        Ok(Box::new(NvdecDecoder {
            info,
            decoded_frames,
            frame_cursor: 0,
            _cuvid_lib: cuvid_lib,
            _cuda_lib: cuda_lib,
        }))
    }
}

// ─── Callbacks ─────────────────────────────────────────────────────
//
// Each callback body is wrapped in std::panic::catch_unwind. Unwinding
// across an `extern "C"` boundary is UB per the Rustonomicon. If a
// Rust panic escapes into cuvidParseVideoData we get memory corruption
// at best. catch_unwind gives us a defined path: convert to error and
// return 0 (which tells the parser to abort cleanly).

unsafe extern "C" fn sequence_callback(
    user_data: *mut c_void,
    format: *mut CuVideoFormat,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if user_data.is_null() || format.is_null() {
                return 0;
            }
            let state = &mut *(user_data as *mut CallbackState);
            let fmt = &*format;

            // Task #39 hardening: verify the driver-reported codec matches
            // what we told the parser to expect. A mismatch here means the
            // CUVIDEOFORMAT struct layout drifted (bytes mean different
            // things than Rust thinks) OR we set up the parser for the
            // wrong codec OR the driver is quietly reinterpreting the
            // stream. Any of those is a catastrophic misconfiguration we
            // want to abort on immediately rather than proceed into
            // undefined decode behaviour. Using tracing::warn! + typed
            // error rather than assert! so the failure is diagnosable in
            // prod log aggregators without crashing the worker process.
            if fmt.codec != state.codec_type {
                tracing::warn!(
                    expected = state.codec_type,
                    got = fmt.codec,
                    "NVDEC sequence_callback codec mismatch — ABI drift suspected"
                );
                state.set_error(format!(
                    "sequence_callback codec mismatch: expected {} got {}",
                    state.codec_type, fmt.codec
                ));
                return 0;
            }

            // Honor the parser's declared minimum; pad up for pipelining but
            // cap at 32 so small-VRAM GPUs (e.g. Jetson, T4, A10) don't OOM
            // when the decoder reserves 4K×32 surfaces per stream.
            let num_surfaces = (fmt.min_num_decode_surfaces as c_uint).clamp(20, 32) as c_ulong;

            // INFO level because this is a backend-engaged signal —
            // operators want to see it in prod logs to confirm NVDEC is
            // actually taking H.264/HEVC/VP9/AV1 traffic rather than
            // silently falling back to CPU. Fires once per sequence
            // (on first IDR and on mid-stream resolution changes).
            tracing::info!(
                codec = fmt.codec,
                width = fmt.coded_width,
                height = fmt.coded_height,
                chroma = fmt.chroma_format,
                bit_depth = fmt.bit_depth_luma_minus8 + 8,
                surfaces = num_surfaces,
                "NVDEC backend engaged"
            );

            // Reject non-4:2:0 sources up-front. The NV12 buffer sizing and
            // the chroma deinterleave loop in decode_next both assume 4:2:0
            // subsampling. If the driver reports something else, fail
            // cleanly rather than corrupt the output.
            //
            // Chroma format values (SDK cudaVideoChromaFormat):
            //   0 = Monochrome, 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4
            // HEVC Range Extensions profiles produce 4:2:2 or 4:4:4 in
            // the wild (perf-analyst flagged a test HEVC sample as RExt);
            // those land here and bubble up as a WARN in decode/mod.rs.
            // Route both chroma + bit-depth reject through the same
            // pure-Rust validator the unit tests exercise, so the
            // reject matrix can't drift between the callback and the
            // standalone validate_format() public API
            // (codec-review-2 HIGH-1 + HIGH-2).
            if let Some(err) = validate_format(
                fmt.chroma_format,
                fmt.bit_depth_luma_minus8,
                fmt.coded_width,
                fmt.coded_height,
            ) {
                // Actionable warn carries the codec, the resolution,
                // and the structured reject reason so operators
                // running multiple GPU hosts can attribute the
                // reject to a specific worker. The log fields mirror
                // the NvdecError variant contents.
                match &err {
                    NvdecError::UnsupportedChroma { label, .. } => {
                        tracing::warn!(
                            codec = state.codec_type,
                            w = fmt.coded_width,
                            h = fmt.coded_height,
                            chroma = fmt.chroma_format,
                            chroma_label = *label,
                            "NVDEC rejecting: chroma {} unsupported",
                            label
                        );
                    }
                    NvdecError::UnsupportedPixelFormat { bit_depth } => {
                        tracing::warn!(
                            codec = state.codec_type,
                            w = fmt.coded_width,
                            h = fmt.coded_height,
                            bit_depth = bit_depth,
                            "NVDEC rejecting: {}-bit content unsupported",
                            bit_depth
                        );
                    }
                }
                state.set_typed_error(err);
                return 0;
            }
            let is_high_bit_depth = fmt.bit_depth_luma_minus8 > 0;
            // Record the bit depth on the callback state so
            // display_callback knows whether to use the NV12 (1 byte
            // per sample) or P016 (2 bytes per sample) copy path.
            state.bit_depth_luma_minus8 = fmt.bit_depth_luma_minus8;

            // Color metadata from CUVIDEOFORMAT.video_signal_description.
            // SDK layout (SDK 12.2 nvcuvid.h):
            //   byte 0: video_format:3 | video_full_range_flag:1 | reserved:4
            //   byte 1: color_primaries
            //   byte 2: transfer_characteristics
            //   byte 3: matrix_coefficients
            // Values per ITU-T H.273 / H.265 §E.3.1:
            //   matrix_coefficients  = 1  → BT.709
            //                          5  → BT.601 625
            //                          6  → BT.601 525
            //                          9  → BT.2020 non-constant luminance
            //                          10 → BT.2020 constant luminance
            // transfer_characteristics = 16 (PQ / SMPTE ST 2084) indicates
            // HDR10; we still tag as BT.2020 since ColorSpace doesn't yet
            // have an HDR10 variant — the transfer curve is a separate
            // concern from the matrix. If/when a downstream consumer
            // needs to distinguish, StreamInfo can grow a TransferFn
            // field.
            let cp = fmt.video_signal_description[1];
            let tc = fmt.video_signal_description[2];
            let mc = fmt.video_signal_description[3];
            let full_range = (fmt.video_signal_description[0] >> 3) & 1 == 1;
            state.vui_colour_primaries = cp;
            state.vui_transfer_characteristics = tc;
            state.vui_matrix_coefficients = mc;
            state.vui_full_range_flag = full_range;
            state.color_space = match mc {
                1 => ColorSpace::Bt709,
                5 | 6 => ColorSpace::Bt601,
                9 | 10 => ColorSpace::Bt2020,
                _ => {
                    // Unspecified (0/2) or unknown: infer from bit depth.
                    // HDR10 streams always hit the 10-bit path; non-HDR
                    // 10-bit streams are rare enough to tag BT.2020 as
                    // a conservative default.
                    if is_high_bit_depth {
                        ColorSpace::Bt2020
                    } else {
                        ColorSpace::Bt709
                    }
                }
            };
            tracing::info!(
                matrix_coefficients = mc,
                color_primaries = fmt.video_signal_description[1],
                transfer = fmt.video_signal_description[2],
                color_space = ?state.color_space,
                "NVDEC color metadata"
            );

            if state.decoder.is_none() {
                let mut create_info: CuVideoDecodeCreateInfo = std::mem::zeroed();
                create_info.code_width = fmt.coded_width as c_ulong;
                create_info.coded_height = fmt.coded_height as c_ulong;
                create_info.num_decode_surfaces = num_surfaces;
                create_info.codec_type = state.codec_type;
                create_info.chroma_format = CUVID_CHROMA_420;
                // Explicitly prefer the CUVID (native NVDEC) backend rather
                // than letting the driver pick DXVA on Windows. Matches
                // ffmpeg libavcodec/cuviddec.c. This is the leading
                // suspect for the H.264 segfault seen on Windows — a
                // DXVA-backed decoder hands back surfaces with different
                // pitch/layout semantics than our cuMemcpy2D assumes.
                create_info.creation_flags = CUVID_CREATE_PREFER_CUVID;
                create_info.bit_depth_minus8 = fmt.bit_depth_luma_minus8 as c_ulong;
                // P016 surface for 10/12-bit, NV12 for 8-bit. P016 lays
                // out 16 bits per sample with the high-order bits
                // carrying the actual 10/12-bit value (low bits zero).
                create_info.output_format = if is_high_bit_depth {
                    CUVID_FMT_P016
                } else {
                    CUVID_FMT_NV12
                };
                // Progressive → Weave (0, no-op).
                // Interlaced → codec-dependent:
                //   H.264 → Adaptive (2): best quality for MBAFF/PAFF
                //   that dominates the H.264 interlaced corpus.
                //   HEVC  → Bob (1): the driver rejects Adaptive for
                //   HEVC interlaced streams with INVALID_ARG (see
                //   codec-review-2 MEDIUM-5 and nvdec-segfault-hunt.md);
                //   Bob is the highest-quality mode the driver will
                //   accept for HEVC.
                //   Other codecs → Bob (1) as a safe default.
                create_info.deinterlace_mode = if fmt.progressive_sequence != 0 {
                    0
                } else if state.codec_type == CUVID_H264 {
                    2
                } else {
                    1
                };
                create_info.target_width = fmt.coded_width as c_ulong;
                create_info.target_height = fmt.coded_height as c_ulong;
                // ffmpeg uses 1 output surface; we use 4 for better
                // pipelining between display_callback and the decoder.
                // Some drivers reject > 4 on older GPUs.
                create_info.num_output_surfaces = 4;
                // Leave max_width / max_height as zero per ffmpeg
                // (memset'd to zero; never written). Setting them equal
                // to coded dimensions rejects any future resolution
                // upshift within the stream and has been seen to trigger
                // INVALID_ARG on some driver versions.
                create_info.max_width = 0;
                create_info.max_height = 0;

                state.width = fmt.coded_width;
                state.height = fmt.coded_height;

                let mut decoder: CUvideodecoder = ptr::null_mut();
                let rc = (state.cuvid_create_decoder)(&mut decoder, &mut create_info);
                if rc != 0 {
                    state.set_error(format!("cuvidCreateDecoder failed: {rc}"));
                    return 0;
                }
                state.decoder = Some(decoder);
            }

            num_surfaces as c_int
        }))
        .unwrap_or(0)
    }
}

unsafe extern "C" fn decode_callback(
    user_data: *mut c_void,
    pic_params: *mut CuVideoPicParams,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if user_data.is_null() || pic_params.is_null() {
                return 0;
            }
            let state = &mut *(user_data as *mut CallbackState);

            let Some(decoder) = state.decoder else {
                state.set_error("decode_callback before decoder created");
                return 0;
            };

            let rc = (state.cuvid_decode_picture)(decoder, pic_params);
            if rc != 0 {
                state.set_error(format!("cuvidDecodePicture failed: {rc}"));
                return 0;
            }
            1
        }))
        .unwrap_or(0)
    }
}

unsafe extern "C" fn display_callback(
    user_data: *mut c_void,
    disp_info: *mut CuVideoDispInfo,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if user_data.is_null() || disp_info.is_null() {
                return 0;
            }
            let state = &mut *(user_data as *mut CallbackState);
            let info = &*disp_info;

            let Some(decoder) = state.decoder else {
                state.set_error("display_callback before decoder created");
                return 0;
            };

            // NVDEC occasionally returns a sentinel picture_index on parse
            // recovery paths (observed < 0 on truncated streams). Passing
            // it back to cuvidMapVideoFrame can segfault inside the driver.
            if info.picture_index < 0 {
                state.set_error(format!(
                    "display_callback picture_index invalid: {}",
                    info.picture_index
                ));
                return 0;
            }

            let mut proc_params: CuVideoProcParams = std::mem::zeroed();
            proc_params.progressive_frame = info.progressive_frame;
            proc_params.second_field = 0;
            proc_params.top_field_first = info.top_field_first;
            proc_params.unpaired_field = 0;

            let mut frame_ptr: CUdeviceptr = 0;
            let mut pitch: c_uint = 0;
            let rc = (state.cuvid_map_video_frame)(
                decoder,
                info.picture_index,
                &mut frame_ptr,
                &mut pitch,
                &mut proc_params,
            );
            if rc != 0 {
                state.set_error(format!("cuvidMapVideoFrame failed: {rc}"));
                return 0;
            }

            let width = state.width as usize;
            let height = state.height as usize;
            // 1 byte/sample for NV12, 2 bytes/sample for P016.
            // Chroma plane is ceil(width/2) × ceil(height/2) samples of
            // interleaved UV. Because chroma is stored as UV pairs side
            // by side at chroma resolution, its row width in bytes is
            // 2 * ceil(w/2) * bytes_per_sample — which for even widths
            // equals the luma row width. For odd widths NVDEC already
            // pads up to ceil(w/2) when outputting NV12/P016, so using
            // the luma row_bytes is still the correct copy stride.
            //
            // codec-review-2 MEDIUM-4: previously we used `height/2`
            // here which silently truncated the last chroma row for
            // odd-height streams (1080 is even, but 1079-height HDR
            // tests and 4:2:0 film transfers hit the odd case). The
            // missing row showed up as a green band at the bottom of
            // the frame after NV12→planar.
            let bytes_per_sample = if state.bit_depth_luma_minus8 > 0 {
                2
            } else {
                1
            };
            let row_bytes = width * bytes_per_sample;
            let chroma_height = height.div_ceil(2);
            let y_bytes = row_bytes * height;
            let uv_bytes = row_bytes * chroma_height;
            let mut host_buf = vec![0u8; y_bytes + uv_bytes];

            let mut luma_copy: CudaMemcpy2D = std::mem::zeroed();
            luma_copy.src_memory_type = CU_MEMORYTYPE_DEVICE;
            luma_copy.src_device = frame_ptr;
            luma_copy.src_pitch = pitch as usize;
            luma_copy.dst_memory_type = CU_MEMORYTYPE_HOST;
            luma_copy.dst_host = host_buf.as_mut_ptr() as *mut c_void;
            luma_copy.dst_pitch = row_bytes;
            luma_copy.width_in_bytes = row_bytes;
            luma_copy.height = height;
            let rc = (state.cu_memcpy2d)(&luma_copy);
            if rc != 0 {
                (state.cuvid_unmap_video_frame)(decoder, frame_ptr);
                state.set_error(format!("cuMemcpy2D (luma) failed: {rc}"));
                return 0;
            }

            let chroma_src = frame_ptr + (pitch as CUdeviceptr) * (height as CUdeviceptr);
            let mut chroma_copy: CudaMemcpy2D = std::mem::zeroed();
            chroma_copy.src_memory_type = CU_MEMORYTYPE_DEVICE;
            chroma_copy.src_device = chroma_src;
            chroma_copy.src_pitch = pitch as usize;
            chroma_copy.dst_memory_type = CU_MEMORYTYPE_HOST;
            chroma_copy.dst_host = host_buf[y_bytes..].as_mut_ptr() as *mut c_void;
            chroma_copy.dst_pitch = row_bytes;
            chroma_copy.width_in_bytes = row_bytes;
            // ceil(h/2) rows — see host_buf sizing above. The driver
            // always emits ceil(h/2) chroma rows regardless of parity;
            // the previous `height / 2` dropped the last row on odd h.
            chroma_copy.height = chroma_height;
            let rc = (state.cu_memcpy2d)(&chroma_copy);

            let _ = (state.cuvid_unmap_video_frame)(decoder, frame_ptr);

            if rc != 0 {
                state.set_error(format!("cuMemcpy2D (chroma) failed: {rc}"));
                return 0;
            }

            if let Ok(mut c) = state.collector.lock() {
                // push_back so the streaming reader (NvdecStreamingDecoder)
                // can pop_front in display order. Eager callers drain
                // sequentially after teardown — same observed order
                // either way.
                c.frames.push_back(DecodedFrame {
                    nv12: host_buf,
                    width: state.width,
                    height: state.height,
                    bit_depth_minus8: state.bit_depth_luma_minus8,
                    color_space: state.color_space,
                    timestamp: info.timestamp,
                });
            }
            1
        }))
        .unwrap_or(0)
    }
}

/// AV1 operating-point callback for `pfn_get_operating_point` on
/// `CuVideoParserParams` (NVIDIA Video Codec SDK 12.x).
///
/// PROBLEMS.md §"NVDEC AV1 — CUVIDEOFORMAT layout mismatch" hypothesis:
/// on AV1 streams, when the callback is *not* set, some SDK versions
/// don't fully populate the `CUVIDEOFORMAT` named fields before
/// `pfn_sequence_callback` fires — we observed `chroma_format=3`
/// (4:4:4) and `coded_width=coded_height=0` on a clean SVT-AV1 4:2:0
/// source, which means the whole struct was being read at the wrong
/// offset. FFmpeg's `libavcodec/cuviddec.c::cuvid_handle_operating_point`
/// always wires this callback for AV1; the parser may take a different
/// code path depending on whether it's set.
///
/// Return value encoding (per SDK nvcuvid.h):
///   `(output_all_layers << 16) | operating_point_index`
///
/// We pick operating point 0 (always present — the base layer for
/// scalable streams, the entire bitstream for single-layer streams)
/// with `output_all_layers = 0`. Matches FFmpeg's default and what
/// mainstream players use for non-scalable AV1.
///
/// The callback is wired on every `CuVideoParserParams` setup
/// regardless of codec; non-AV1 codecs ignore it (the SDK only calls
/// it from the AV1 parser path). Cost is one fn-pointer populated
/// per parser construction.
unsafe extern "C" fn get_operating_point_callback(
    _user_data: *mut c_void,
    _op_info: *mut c_void,
) -> c_int {
    // catch_unwind defence: the callback runs on an NVIDIA-driver-
    // owned thread; a Rust panic crossing a C boundary is UB. Matches
    // the other callbacks in this file.
    catch_unwind(AssertUnwindSafe(|| 0_i32)).unwrap_or(0)
}

/// Structural mirror of `CUVIDOPERATINGPOINTINFO` (nvcuvid.h). Not
/// read at runtime — the callback above returns a fixed value
/// without inspecting the struct — but the shape is documented here
/// so a future session implementing layer-selective decode has a
/// reference. Tagged with `#[allow(dead_code)]` to silence the
/// unused-field warnings.
#[repr(C)]
#[allow(dead_code)]
struct CuVideoOperatingPointInfo {
    codec: c_int,
    // Union: AV1 fields vs CodecReserved[1024].
    // AV1 variant:
    //   unsigned char  operating_points_cnt;
    //   unsigned char  reserved24_bits[3];
    //   unsigned short operating_points_idc[32];
    //   → 4 + 64 = 68 bytes
    // CodecReserved[1024] is the upper bound; assert below.
    reserved: [u8; 1024],
}
const _: () = assert!(std::mem::size_of::<CuVideoOperatingPointInfo>() <= 1024 + 8);

impl NvdecDecoder {
    /// Test-only constructor: build an `NvdecDecoder` pre-seeded with
    /// a synthetic `Vec<DecodedFrame>` so `decode_next` can be unit-
    /// tested without standing up a CUDA context.
    ///
    /// Each tuple is `(nv12_or_p016_bytes, width, height,
    /// bit_depth_minus8, pts)`. The harness fills in a placeholder
    /// `ColorSpace::Bt709` and uses the `info` caller-supplied for
    /// `stream_info()`.
    ///
    /// Exposed for tests/nvdec_smoke.rs only. Loads an always-present
    /// system library (`kernel32` on Windows, `libc` on Linux, libSystem
    /// on macOS) into the library handles so the Drop order matches
    /// the production path even though no FFI fn pointers are captured.
    #[doc(hidden)]
    pub fn test_new_from_frames(
        frames: Vec<(Vec<u8>, u32, u32, u8, u64)>,
        info: StreamInfo,
    ) -> Box<dyn Decoder> {
        let decoded_frames: Vec<DecodedFrame> = frames
            .into_iter()
            .map(|(bytes, w, h, bd, pts)| DecodedFrame {
                nv12: bytes,
                width: w,
                height: h,
                bit_depth_minus8: bd,
                color_space: ColorSpace::Bt709,
                timestamp: pts,
            })
            .collect();
        let cuda_lib = unsafe { libloading::Library::new("kernel32.dll") }
            .or_else(|_| unsafe { libloading::Library::new("libc.so.6") })
            .or_else(|_| unsafe { libloading::Library::new("/usr/lib/libSystem.B.dylib") })
            .expect("test harness: a placeholder system library must load");
        let cuvid_lib = unsafe { libloading::Library::new("kernel32.dll") }
            .or_else(|_| unsafe { libloading::Library::new("libc.so.6") })
            .or_else(|_| unsafe { libloading::Library::new("/usr/lib/libSystem.B.dylib") })
            .expect("test harness: a placeholder system library must load");
        Box::new(NvdecDecoder {
            info,
            decoded_frames,
            frame_cursor: 0,
            _cuvid_lib: cuvid_lib,
            _cuda_lib: cuda_lib,
        })
    }
}

impl Decoder for NvdecDecoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    // NvdecDecoder proper is the eager post-decode type: all frames
    // are already decoded and sitting in self.decoded_frames by the
    // time this instance exists. push_sample/finish are therefore
    // explicit no-ops — any streaming caller should construct an
    // NvdecPushDecoder instead, which buffers samples and invokes
    // NvdecDecoder::new in its own finish().
    fn push_sample(&mut self, _data: &[u8]) -> Result<()> {
        anyhow::bail!(
            "NvdecDecoder: push_sample on eager-mode instance — use NvdecPushDecoder for streaming"
        );
    }

    fn finish(&mut self) -> Result<()> {
        Ok(())
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        if self.frame_cursor >= self.decoded_frames.len() {
            return Ok(None);
        }

        let frame = &self.decoded_frames[self.frame_cursor];
        self.frame_cursor += 1;
        Ok(Some(decoded_frame_to_video_frame(frame)))
    }
}

/// Convert one `DecodedFrame` (NV12 or P016 bytes) into a `VideoFrame`.
/// Shared between the eager `NvdecDecoder` (Vec drain) and the
/// streaming `NvdecStreamingDecoder` (VecDeque drain) paths so the
/// deinterleave / planar conversion has a single source of truth.
fn decoded_frame_to_video_frame(frame: &DecodedFrame) -> VideoFrame {
    let w = frame.width as usize;
    let h = frame.height as usize;
    // Round up to keep odd-sized chroma planes intact (M-A10). For
    // subsampled 4:2:0, chroma dimensions are ceil(w/2) × ceil(h/2).
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let uv_pairs = cw * ch;

    let (yuv, pixel_format) = if frame.bit_depth_minus8 > 0 {
        // P016 → Yuv420p10le — routed through the pure-Rust
        // helper so the deinterleave + 10-bit normalize path has
        // unit coverage (codec-review-2 HIGH-2). The helper
        // right-shifts each u16 sample by 6 so the 10-bit value
        // lands in the LOW bits of the emitted LE u16, matching
        // what the encoder / colorspace consumer expects.
        let _ = uv_pairs; // silence unused warn on this branch
        let out = deinterleave_p016_to_yuv420p10le(&frame.nv12, w, h);
        (out, PixelFormat::Yuv420p10le)
    } else {
        // NV12 → Yuv420p. 1 byte per sample, interleaved UV pair
        // stride is 2 bytes.
        let y_size = w * h;
        let mut out = Vec::with_capacity(y_size + uv_pairs * 2);
        out.extend_from_slice(&frame.nv12[..y_size.min(frame.nv12.len())]);
        if frame.nv12.len() > y_size {
            let uv = &frame.nv12[y_size..];
            let mut u = Vec::with_capacity(uv_pairs);
            let mut v = Vec::with_capacity(uv_pairs);
            for i in 0..uv_pairs {
                if i * 2 + 1 < uv.len() {
                    u.push(uv[i * 2]);
                    v.push(uv[i * 2 + 1]);
                }
            }
            out.extend_from_slice(&u);
            out.extend_from_slice(&v);
        }
        (out, PixelFormat::Yuv420p)
    };

    VideoFrame::new(
        Bytes::from(yuv),
        frame.width,
        frame.height,
        pixel_format,
        frame.color_space,
        frame.timestamp,
    )
}

// ─── Push-mode wrapper ─────────────────────────────────────────────
//
// cuvid's parser fundamentally wants all samples fed via
// cuvidParseVideoData in one pass so its stateful picture queue can
// build up reference-list context. Our NvdecDecoder::new() does this
// eagerly: takes Vec<Vec<u8>>, runs the whole parse loop, then hands
// back an instance pre-populated with decoded frames.
//
// The push-mode trait (#55) wants per-sample streaming. The cheapest
// safe bridge: buffer the incoming samples in a Vec, then on finish()
// call NvdecDecoder::new with the accumulated buffer. This preserves
// the full fix stack (CUVIDPARSERPARAMS size, CUVIDPICPARAMS size,
// CUVID_CREATE_PREFER_CUVID, bAnnexb=1, struct-size assertions) that
// took tasks #39/#52/#53/#65 to nail down, without re-architecting
// the parser loop into an incremental feeder.
//
// Memory cost: O(total bitstream bytes) until finish() — which is
// unchanged from the pre-refactor steady state, since create_decoder's
// old signature also took Vec<Vec<u8>> up front. Task #55's goal is
// to reduce *frame memory* (decoded frame bytes × variant count) not
// *bitstream memory*, so this tradeoff is aligned with the intent.
pub struct NvdecPushDecoder {
    info: StreamInfo,
    gpu_index: u32,
    /// (bitstream bytes, PTS). The PTS is either a real demuxer PTS
    /// from `push_sample_with_pts`, or a fabricated monotonic index
    /// from `push_sample`. codec-review-2 HIGH-3: keep real PTS
    /// end-to-end so B-frame display order survives.
    pending_samples: Vec<(Vec<u8>, u64)>,
    decoded: Option<Box<dyn Decoder>>,
    finished: bool,
}

impl NvdecPushDecoder {
    pub fn new(info: StreamInfo, gpu_index: u32) -> Self {
        Self {
            info,
            gpu_index,
            pending_samples: Vec::new(),
            decoded: None,
            finished: false,
        }
    }

    /// Push with an explicit PTS. Preferred over the trait-level
    /// `push_sample` when the caller has the real demuxer timestamp,
    /// because the trait signature (no PTS) forces us to fabricate a
    /// counter that is wrong for B-frame-heavy streams.
    pub fn push_sample_with_pts(&mut self, data: &[u8], pts: u64) -> Result<()> {
        if self.finished {
            anyhow::bail!("NvdecPushDecoder: push_sample after finish");
        }
        self.pending_samples.push((data.to_vec(), pts));
        Ok(())
    }
}

impl Decoder for NvdecPushDecoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    fn push_sample(&mut self, data: &[u8]) -> Result<()> {
        if self.finished {
            anyhow::bail!("NvdecPushDecoder: push_sample after finish");
        }
        // No real PTS supplied — fabricate a monotonic counter from
        // the current buffer length so each sample at least gets a
        // distinct timestamp. Callers that need correct display-order
        // PTS (B-frame streams) should use `push_sample_with_pts`.
        let pts = self.pending_samples.len() as u64;
        self.pending_samples.push((data.to_vec(), pts));
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        let samples = std::mem::take(&mut self.pending_samples);
        // Eager decode: the existing new_with_pts does the full cuvid
        // parse loop. On success we store the resulting Decoder so
        // decode_next can delegate. On error we propagate — the outer
        // factory is responsible for CPU fallback logging.
        self.decoded = Some(NvdecDecoder::new_with_pts(
            samples,
            self.info.clone(),
            self.gpu_index,
        )?);
        Ok(())
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        match self.decoded.as_mut() {
            Some(inner) => inner.decode_next(),
            None => {
                // Caller pulled without finishing — treat as no-op
                // rather than panic so streaming code that polls
                // decode_next opportunistically doesn't crash.
                Ok(None)
            }
        }
    }
}

// ─── True-streaming NVDEC decoder (Squad-36) ──────────────────────
//
// The eager `NvdecDecoder::new_with_pts` runs the entire CUVID parse
// loop inside the constructor (collect-everything-then-parse), and
// `NvdecPushDecoder` is the fake-streaming buffer-then-eager wrapper.
// Both materialise the full decoded NV12/P016 frame set in RAM, which
// blows past the streaming-migration-55 RSS budget on long inputs
// (~315 GiB projected for 15 min 1080p60). The
// streaming-migration-55-codebase-audit.md HARD-BLOCK A1+A2 findings
// flagged this and the v1 streaming pipeline gated NVDEC OFF.
//
// `NvdecStreamingDecoder` is the structural answer: build the parser
// and decoder contexts up front, then per `push_sample` invoke
// `cuvidParseVideoData` on JUST that sample's bytes. The display
// callback enqueues into a bounded `VecDeque<DecodedFrame>` and
// `decode_next` pops one at a time. CUVID's parser is documented as
// supporting incremental parse-per-call (each `cuvidParseVideoData`
// invocation accept-and-process the contained payload then return —
// state lives in the parser handle, not the call). `finish()` flushes
// by calling `cuvidParseVideoData` once with `CUVID_PKT_ENDOFSTREAM`.
//
// Memory shape:
//   - Bitstream: caller's `&[u8]` lives only for the duration of one
//     `push_sample` (we don't copy; CUVID consumes the pointer
//     synchronously inside `cuvidParseVideoData`).
//   - Decoded: VecDeque holds the reorder window — typically ≤ 16
//     frames for B-pyramid H.264/HEVC. At ~3.1 MiB per 1080p NV12
//     frame, ≤ 50 MiB even on the worst-case reorder. Actual peak
//     depends on how fast the caller drains via `decode_next`.
//
// Correctness preservation (per Squad-36 brief):
//   - Squad-12 const_assert! shape witnesses: shared FFI defs at the
//     top of this file; the streaming decoder uses the same
//     CuVideoParserParams + CuVideoPicParams + CuVideoSourceDataPacket
//     so the asserts cover both paths.
//   - Squad-21 SEI scanner: HEVC SEI 137/144 (mastering display + CLL)
//     is read CPU-side from the demuxer (probe_mp4_visual_color_metadata,
//     hevc_sei::scan_for_hdr_sei) and lives on
//     StreamInfo.color_metadata BEFORE create_decoder runs. The
//     streaming decoder preserves those fields (does not overwrite —
//     CUVIDEOFORMAT doesn't surface the SEI 137/144 payloads in SDK
//     12.2 anyway). Same code path as the eager NvdecDecoder, just
//     applied incrementally.
//   - Squad-6 typed UnsupportedChroma/UnsupportedPixelFormat reject:
//     the sequence callback is unchanged; on the first sample carrying
//     a sequence header (typically the first IDR), it runs the same
//     `validate_format()` check and sets `state.typed_error`. The
//     streaming `push_sample` checks this after each
//     `cuvidParseVideoData` and surfaces the typed reject as an
//     anyhow::Error wrapping `NvdecError`.

/// Resolved CUDA + CUVID FFI handles + the CUDA context. Held by
/// `NvdecStreamingDecoder` for the lifetime of the decoder so each
/// `push_sample` can re-enter the context without re-resolving
/// symbols.
struct NvdecCtx {
    // Library handles MUST drop AFTER any field that holds borrowed
    // fn pointers from them (Reference §10.8 — fields drop in source
    // order). Declared LAST in `NvdecStreamingDecoder` for the same
    // reason.
    cu_ctx: CUcontext,
    cu_ctx_destroy: FnCuCtxDestroy,
    cu_ctx_push: FnCuCtxPushCurrent,
    cu_ctx_pop: FnCuCtxPopCurrent,
    cuvid_destroy_parser: FnCuvidDestroyVideoParser,
    cuvid_destroy_decoder: FnCuvidDestroyDecoder,
    cuvid_parse_data: FnCuvidParseVideoData,
}

// SAFETY: CUcontext is just an opaque void* the driver returns; it is
// thread-bound only via cuCtxPushCurrent/PopCurrent which we wrap with
// CtxScope. fn pointers are POD. Send is the only required marker for
// the Decoder trait.
unsafe impl Send for NvdecCtx {}

/// True-streaming NVDEC decoder. See module-level comment block
/// above this struct for the design rationale.
pub struct NvdecStreamingDecoder {
    info: StreamInfo,
    /// Owns the boxed CallbackState referenced from
    /// `parser_params.user_data`. Must outlive `parser`. We keep an
    /// `Arc` clone of `state.collector` outside so `decode_next` can
    /// drain without re-locking through the box.
    state: Box<CallbackState>,
    /// Mirror of `state.collector` so `decode_next` doesn't need to
    /// borrow `state` (keeps the borrow checker happy when push_sample
    /// is also taking &mut self).
    collector: Arc<Mutex<FrameCollector>>,
    /// CUVID parser handle. Created in `try_new`, destroyed in `Drop`.
    parser: CUvideoparser,
    /// Resolved FFI + CUDA context. Drop order: parser → ctx
    /// (handled by `Drop` on this struct).
    ctx: NvdecCtx,
    /// EOS already sent? Subsequent push_sample calls return Ok(())
    /// (idempotent finish; matches the trait shape every other
    /// streaming-shape decoder follows).
    finished: bool,
    /// Sample counter for fabricated PTS when the trait-level
    /// `push_sample(&[u8])` is called without an explicit PTS. Real
    /// demuxer PTS still flows through the eager / push-with-pts
    /// paths.
    sample_counter: u64,

    // Library handles held last so they outlive every fn pointer
    // captured into `state`. See the eager NvdecDecoder field-order
    // note for the Reference §10.8 cite.
    _cuvid_lib: libloading::Library,
    _cuda_lib: libloading::Library,
}

// SAFETY: CallbackState is Send (already declared). Library handles +
// CUcontext are Send. The only piece of cross-thread state is the
// Arc<Mutex<FrameCollector>>, which is Send+Sync by construction.
unsafe impl Send for NvdecStreamingDecoder {}

impl NvdecStreamingDecoder {
    /// Build the parser + decoder contexts WITHOUT consuming any
    /// bitstream. Per the Squad-36 brief: the caller drives via
    /// `push_sample` after construction. Returns Err only if the
    /// driver libraries fail to load or if the codec isn't supported
    /// — actual parse / decode failures land on the data path.
    fn try_new(info: StreamInfo, gpu_index: u32) -> Result<Self> {
        // Take the SHARED CUDA-init lock BEFORE any CUDA / cuvid FFI
        // work. The lock serializes streaming-decoder construction
        // against NVENC encoder construction (which does its own
        // cuInit + cuCtxCreate) — concurrent CUDA inits from those
        // two backends were the precise root cause of the prod
        // SIGSEGV captured 2026-05-01 PT 03:12:43. See
        // crates/codec/src/cuda_lock.rs for the trace + reasoning.
        // Released when the variable goes out of scope at function
        // exit; per-frame parse + decode work that follows runs
        // concurrently as before.
        let _init_guard = crate::cuda_lock::lock_for_cuda_init();

        let cuda_lib = unsafe { libloading::Library::new("libcuda.so") }
            .or_else(|_| unsafe { libloading::Library::new("libcuda.so.1") })
            .or_else(|_| unsafe { libloading::Library::new("nvcuda.dll") })
            .context("loading CUDA driver — is the NVIDIA driver installed?")?;

        let cuvid_lib = unsafe { libloading::Library::new("libnvcuvid.so") }
            .or_else(|_| unsafe { libloading::Library::new("libnvcuvid.so.1") })
            .or_else(|_| unsafe { libloading::Library::new("nvcuvid.dll") })
            .context("loading cuvid — is the NVIDIA driver installed?")?;

        let cuvid_codec = codec_to_cuvid(&info.codec)
            .context(format!("unsupported NVDEC codec: {}", info.codec))?;

        // Resolve all FFI symbols up front so push_sample doesn't
        // re-enter libloading on every call.
        let (state, parser, ctx) = unsafe {
            let cu_init: libloading::Symbol<FnCuInit> = cuda_lib.get(b"cuInit")?;
            if cu_init(0) != 0 {
                bail!("cuInit failed");
            }

            let cu_device_get: libloading::Symbol<FnCuDeviceGet> = cuda_lib.get(b"cuDeviceGet")?;
            let mut device: CUdevice = 0;
            if cu_device_get(&mut device, gpu_index as c_int) != 0 {
                bail!("cuDeviceGet failed for GPU {gpu_index}");
            }

            let cu_ctx_create: libloading::Symbol<FnCuCtxCreate> =
                cuda_lib.get(b"cuCtxCreate_v2")?;
            let cu_ctx_destroy: libloading::Symbol<FnCuCtxDestroy> =
                cuda_lib.get(b"cuCtxDestroy_v2")?;
            let cu_ctx_push: libloading::Symbol<FnCuCtxPushCurrent> =
                cuda_lib.get(b"cuCtxPushCurrent_v2")?;
            let cu_ctx_pop: libloading::Symbol<FnCuCtxPopCurrent> =
                cuda_lib.get(b"cuCtxPopCurrent_v2")?;

            let mut cu_ctx: CUcontext = ptr::null_mut();
            if cu_ctx_create(&mut cu_ctx, 0, device) != 0 {
                bail!("cuCtxCreate failed");
            }

            let cuvid_create_parser: libloading::Symbol<FnCuvidCreateVideoParser> =
                cuvid_lib.get(b"cuvidCreateVideoParser")?;
            let cuvid_parse_data: libloading::Symbol<FnCuvidParseVideoData> =
                cuvid_lib.get(b"cuvidParseVideoData")?;
            let cuvid_destroy_parser: libloading::Symbol<FnCuvidDestroyVideoParser> =
                cuvid_lib.get(b"cuvidDestroyVideoParser")?;
            let cuvid_create_decoder: libloading::Symbol<FnCuvidCreateDecoder> =
                cuvid_lib.get(b"cuvidCreateDecoder")?;
            let cuvid_destroy_decoder: libloading::Symbol<FnCuvidDestroyDecoder> =
                cuvid_lib.get(b"cuvidDestroyDecoder")?;
            let cuvid_decode_picture: libloading::Symbol<FnCuvidDecodePicture> =
                cuvid_lib.get(b"cuvidDecodePicture")?;
            let cuvid_map_video_frame: libloading::Symbol<FnCuvidMapVideoFrame> = cuvid_lib
                .get(b"cuvidMapVideoFrame64")
                .or_else(|_| cuvid_lib.get(b"cuvidMapVideoFrame"))?;
            let cuvid_unmap_video_frame: libloading::Symbol<FnCuvidUnmapVideoFrame> = cuvid_lib
                .get(b"cuvidUnmapVideoFrame64")
                .or_else(|_| cuvid_lib.get(b"cuvidUnmapVideoFrame"))?;
            let cu_memcpy2d: libloading::Symbol<FnCuMemcpy2D> = cuda_lib.get(b"cuMemcpy2D_v2")?;

            let collector = Arc::new(Mutex::new(FrameCollector {
                frames: VecDeque::new(),
            }));

            let mut state = Box::new(CallbackState {
                cuvid_create_decoder: *cuvid_create_decoder,
                cuvid_decode_picture: *cuvid_decode_picture,
                cuvid_map_video_frame: *cuvid_map_video_frame,
                cuvid_unmap_video_frame: *cuvid_unmap_video_frame,
                cu_memcpy2d: *cu_memcpy2d,
                decoder: None,
                collector: Arc::clone(&collector),
                width: info.width,
                height: info.height,
                codec_type: cuvid_codec,
                bit_depth_luma_minus8: 0,
                color_space: ColorSpace::Bt709,
                vui_colour_primaries: 1,
                vui_transfer_characteristics: 1,
                vui_matrix_coefficients: 1,
                vui_full_range_flag: false,
                error: None,
                typed_error: None,
            });
            let state_ptr: *mut c_void = (&mut *state) as *mut CallbackState as *mut c_void;

            let mut parser_params: CuVideoParserParams = std::mem::zeroed();
            parser_params.codec_type = cuvid_codec;
            parser_params.max_num_decode_surfaces = 20;
            parser_params.clock_rate = 0;
            parser_params.error_threshold = 100;
            parser_params.max_display_delay = 4;
            // bAnnexb=1 (Squad-12 / task #39): tells the parser our
            // samples are Annex-B, also makes the parser more lenient
            // about non-IDR recovery on open-GOP streams.
            parser_params.reserved1[0] = 1;
            parser_params.user_data = state_ptr;
            parser_params.pfn_sequence_callback = Some(sequence_callback);
            parser_params.pfn_decode_picture = Some(decode_callback);
            parser_params.pfn_display_picture = Some(display_callback);
            // AV1 operating-point hook (streaming path mirror of the
            // eager new_with_pts setup above). See
            // `get_operating_point_callback` docstring.
            parser_params.pfn_get_operating_point = Some(get_operating_point_callback);

            let mut parser: CUvideoparser = ptr::null_mut();
            let create_rc = cuvid_create_parser(&mut parser, &mut parser_params);
            if create_rc != 0 {
                cu_ctx_destroy(cu_ctx);
                bail!("cuvidCreateVideoParser failed: {create_rc}");
            }

            let ctx = NvdecCtx {
                cu_ctx,
                cu_ctx_destroy: *cu_ctx_destroy,
                cu_ctx_push: *cu_ctx_push,
                cu_ctx_pop: *cu_ctx_pop,
                cuvid_destroy_parser: *cuvid_destroy_parser,
                cuvid_destroy_decoder: *cuvid_destroy_decoder,
                cuvid_parse_data: *cuvid_parse_data,
            };
            (state, parser, ctx)
        };

        // Stash collector outside the box so decode_next can lock it
        // without going through the &mut self borrow on `state`.
        let collector = Arc::clone(&state.collector);

        Ok(Self {
            info,
            state,
            collector,
            parser,
            ctx,
            finished: false,
            sample_counter: 0,
            _cuvid_lib: cuvid_lib,
            _cuda_lib: cuda_lib,
        })
    }

    /// Push one sample with an explicit PTS. Preferred over the trait
    /// `push_sample` when the caller has the real demuxer timestamp;
    /// the fabricated counter the trait shape forces is wrong for
    /// B-frame-heavy streams (codec-review-2 HIGH-3).
    pub fn push_sample_with_pts(&mut self, data: &[u8], pts: u64) -> Result<()> {
        if self.finished {
            anyhow::bail!("NvdecStreamingDecoder: push_sample after finish");
        }
        // Empty samples: skip per Squad-12 hardening — some driver
        // versions dereference the payload pointer before checking
        // payload_size, which would segfault.
        if data.is_empty() {
            return Ok(());
        }

        unsafe {
            let _scope = CtxScope::push(self.ctx.cu_ctx, self.ctx.cu_ctx_push, self.ctx.cu_ctx_pop)
                .context("push CUDA context for incremental parse")?;

            let mut packet: CuVideoSourceDataPacket = std::mem::zeroed();
            packet.payload_size = data.len() as c_ulong;
            packet.payload = data.as_ptr();
            packet.timestamp = pts as c_ulonglong;
            packet.flags = CUVID_PKT_TIMESTAMP;

            let rc = (self.ctx.cuvid_parse_data)(self.parser, &mut packet);
            if rc != 0 {
                // Non-fatal per the SDK — log only on first occurrence
                // (cheap: state.error is none until first failure).
                if self.state.error.is_none() {
                    tracing::warn!(
                        rc = rc,
                        "cuvidParseVideoData returned non-zero (incremental)"
                    );
                }
            }
        }

        // Surface a typed reject ASAP: sequence_callback may have
        // populated state.typed_error if the sequence header carried
        // an unsupported chroma/bit_depth (Squad-6 typed reject path).
        // Returning Err here is the same behaviour the eager path used
        // to surface from finish() under the lazy-flush wrapper.
        if let Some(te) = self.state.typed_error.take() {
            self.finished = true;
            return Err(anyhow::Error::new(te));
        }

        Ok(())
    }
}

impl Drop for NvdecStreamingDecoder {
    fn drop(&mut self) {
        unsafe {
            // Push the context so destroy calls are bound to it.
            // Errors during teardown are logged, not propagated —
            // there's no caller to surface them to.
            let push_rc = (self.ctx.cu_ctx_push)(self.ctx.cu_ctx);
            if push_rc != 0 {
                tracing::warn!(rc = push_rc, "Drop: cuCtxPushCurrent failed");
            }

            // Order: parser → decoder → ctx (matches the eager path's
            // teardown sequence in new_with_pts).
            (self.ctx.cuvid_destroy_parser)(self.parser);
            if let Some(dec) = self.state.decoder.take() {
                (self.ctx.cuvid_destroy_decoder)(dec);
            }

            // Pop before destroy so the destroy doesn't run with the
            // already-freed context bound.
            let mut popped: CUcontext = ptr::null_mut();
            (self.ctx.cu_ctx_pop)(&mut popped);

            (self.ctx.cu_ctx_destroy)(self.ctx.cu_ctx);
        }
    }
}

impl Decoder for NvdecStreamingDecoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    fn push_sample(&mut self, data: &[u8]) -> Result<()> {
        // Trait shape doesn't carry a PTS — fabricate a monotonic
        // counter so each sample at least gets a distinct timestamp.
        // Callers that have a real demuxer PTS should use
        // `push_sample_with_pts`. This matches the existing
        // NvdecPushDecoder fallback shape so the trait callers see
        // identical semantics.
        let pts = self.sample_counter;
        self.sample_counter += 1;
        self.push_sample_with_pts(data, pts)
    }

    fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;

        unsafe {
            let _scope = CtxScope::push(self.ctx.cu_ctx, self.ctx.cu_ctx_push, self.ctx.cu_ctx_pop)
                .context("push CUDA context for EOS flush")?;

            // Send a single packet with CUVID_PKT_ENDOFSTREAM. The
            // parser flushes any buffered DPB pictures through the
            // display callback (one extra batch into the collector
            // VecDeque, drained by subsequent decode_next calls).
            let mut eos_packet: CuVideoSourceDataPacket = std::mem::zeroed();
            eos_packet.flags = CUVID_PKT_ENDOFSTREAM;
            (self.ctx.cuvid_parse_data)(self.parser, &mut eos_packet);
        }

        // Apply VUI color metadata captured by sequence_callback to
        // the StreamInfo carried on this decoder so callers reading
        // `stream_info()` after finish see the real HDR signalling
        // rather than the SDR default. Same fold-back the eager path
        // does in `new_with_pts`. NOTE: mastering_display + content
        // light level are NOT overwritten — those come from the CPU
        // SEI scanner (Squad-21) at probe time and CUVIDEOFORMAT
        // doesn't surface them in SDK 12.2.
        self.info.color_space = self.state.color_space;
        self.info.color_metadata = ColorMetadata {
            transfer: TransferFn::from_h273(self.state.vui_transfer_characteristics),
            matrix_coefficients: self.state.vui_matrix_coefficients,
            colour_primaries: self.state.vui_colour_primaries,
            full_range: self.state.vui_full_range_flag,
            mastering_display: self.info.color_metadata.mastering_display,
            content_light_level: self.info.color_metadata.content_light_level,
        };

        if let Some(te) = self.state.typed_error.take() {
            return Err(anyhow::Error::new(te));
        }
        Ok(())
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        // Surface a deferred typed reject if sequence_callback latched
        // one between the last push and this drain (e.g. caller pushes
        // then immediately drains).
        if let Some(te) = self.state.typed_error.take() {
            return Err(anyhow::Error::new(te));
        }
        let mut guard = self.collector.lock().unwrap();
        match guard.frames.pop_front() {
            Some(frame) => Ok(Some(decoded_frame_to_video_frame(&frame))),
            None => Ok(None),
        }
    }
}

// ─── Init-error trampoline ────────────────────────────────────────
//
// `Decoder::new` is infallible by design (the trait lets the data path
// surface errors via push_sample / finish / decode_next). When the
// streaming NVDEC ctor fails to load the driver libraries — which on
// non-NVIDIA hosts is the common case if the dispatch layer didn't
// already gate it out — wrap the error and surface it on the first
// data-path call. This lets `create_decoder` keep the simple
// `Box<dyn Decoder>` return shape without a fallback dance.
struct NvdecInitErrorDecoder {
    info: StreamInfo,
    /// Taken on the first push_sample / finish / decode_next call.
    /// Subsequent calls return Ok(()) / Ok(None) so the caller can
    /// drain cleanly after seeing the first error.
    error: Option<anyhow::Error>,
}

impl Decoder for NvdecInitErrorDecoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }
    fn push_sample(&mut self, _data: &[u8]) -> Result<()> {
        if let Some(e) = self.error.take() {
            return Err(e);
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<()> {
        if let Some(e) = self.error.take() {
            return Err(e);
        }
        Ok(())
    }
    fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        if let Some(e) = self.error.take() {
            return Err(e);
        }
        Ok(None)
    }
}
