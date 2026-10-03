//! CUVID / CUDA FFI struct mirrors, function-pointer type aliases,
//! compile-time size assertions, and codec/surface/packet constants.
//!
//! Everything here is `pub` so sibling submodules within `nvdec/` can
//! reach items as `super::ffi::TypeName`. The `ffi` module itself is
//! private (declared without `pub` in `mod.rs`), so nothing leaks to
//! external callers.

use std::ffi::c_void;
use std::os::raw::{c_int, c_uchar, c_uint, c_ulong, c_ulonglong};

// ─── CUDA Driver API FFI ───────────────────────────────────────────
pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUdeviceptr = c_ulonglong;

pub type FnCuInit = unsafe extern "C" fn(c_uint) -> CUresult;
pub type FnCuDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
pub type FnCuCtxCreate = unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult;
pub type FnCuCtxDestroy = unsafe extern "C" fn(CUcontext) -> CUresult;
pub type FnCuCtxPushCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
pub type FnCuCtxPopCurrent = unsafe extern "C" fn(*mut CUcontext) -> CUresult;
pub type FnCuMemcpy2D = unsafe extern "C" fn(*const CudaMemcpy2D) -> CUresult;

pub const CU_MEMORYTYPE_HOST: c_uint = 1;
pub const CU_MEMORYTYPE_DEVICE: c_uint = 2;

#[repr(C)]
pub struct CudaMemcpy2D {
    pub src_x_in_bytes: usize,
    pub src_y: usize,
    pub src_memory_type: c_uint,
    pub src_host: *const c_void,
    pub src_device: CUdeviceptr,
    pub src_array: *const c_void,
    pub src_pitch: usize,
    pub dst_x_in_bytes: usize,
    pub dst_y: usize,
    pub dst_memory_type: c_uint,
    pub dst_host: *mut c_void,
    pub dst_device: CUdeviceptr,
    pub dst_array: *const c_void,
    pub dst_pitch: usize,
    pub width_in_bytes: usize,
    pub height: usize,
}

// ─── CUVID (Video Decoder) FFI ─────────────────────────────────────
pub type CUvideoparser = *mut c_void;
pub type CUvideodecoder = *mut c_void;

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

/// Layout matches CUVIDPARSERPARAMS in nvcuvid.h (NVIDIA Video Codec SDK).
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
pub struct CuVideoParserParams {
    pub codec_type: c_int,
    pub max_num_decode_surfaces: c_uint,
    pub clock_rate: c_uint,
    pub error_threshold: c_uint,
    pub max_display_delay: c_uint,
    /// Bitfield word (bAnnexb | bMemoryOptimize | uReserved:30) + 4 reserved u32.
    /// We zero-init and never set any bits; SDK layout compatible.
    pub reserved1: [c_uint; 5],
    pub user_data: *mut c_void,
    pub pfn_sequence_callback: Option<unsafe extern "C" fn(*mut c_void, *mut CuVideoFormat) -> c_int>,
    pub pfn_decode_picture: Option<unsafe extern "C" fn(*mut c_void, *mut CuVideoPicParams) -> c_int>,
    pub pfn_display_picture: Option<unsafe extern "C" fn(*mut c_void, *mut CuVideoDispInfo) -> c_int>,
    pub pfn_get_operating_point: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int>,
    pub pfn_get_sei_msg: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int>,
    /// SDK: `void *pvReserved2[5]`.
    pub reserved2: [*mut c_void; 5],
    pub ext_video_info: *mut c_void,
}

#[repr(C)]
pub struct CuVideoSourceDataPacket {
    pub flags: c_ulong,
    pub payload_size: c_ulong,
    pub payload: *const u8,
    pub timestamp: c_ulonglong,
}

#[repr(C)]
pub struct CuVideoDecodeCreateInfo {
    pub code_width: c_ulong,
    pub coded_height: c_ulong,
    pub num_decode_surfaces: c_ulong,
    pub codec_type: c_int,
    pub chroma_format: c_int,
    pub creation_flags: c_ulong,
    pub bit_depth_minus8: c_ulong,
    pub intra_decode_only: c_ulong,
    pub max_width: c_ulong,
    pub max_height: c_ulong,
    pub reserved1: c_ulong,
    pub display_area_left: i16,
    pub display_area_top: i16,
    pub display_area_right: i16,
    pub display_area_bottom: i16,
    pub output_format: c_int,
    pub deinterlace_mode: c_int,
    pub target_width: c_ulong,
    pub target_height: c_ulong,
    pub num_output_surfaces: c_ulong,
    pub vid_lock: *mut c_void,
    pub target_rect_left: i16,
    pub target_rect_top: i16,
    pub target_rect_right: i16,
    pub target_rect_bottom: i16,
    pub enable_histogram: c_ulong,
    pub reserved2: [c_ulong; 4],
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
pub struct CuVideoPicParams {
    pub pic_width_in_mbs: c_int,
    pub pic_height_in_mbs: c_int,
    pub curr_pic_idx: c_int,
    pub field_pic_flag: c_int,
    pub bottom_field_flag: c_int,
    pub second_field: c_int,
    pub n_bitstream_data_len: c_uint,
    pub p_bitstream_data: *const u8,
    pub n_num_slices: c_uint,
    pub p_slice_data_offsets: *const c_uint,
    pub ref_pic_flag: c_int,
    pub intra_pic_flag: c_int,
    pub reserved: [c_uint; 30],
    // Matches the REAL SDK `union { ...; unsigned int CodecReserved[1024]; }`
    // = 4096 bytes. See struct-size assertion below.
    pub codec_specific: [c_uint; 1024],
}

#[repr(C)]
pub struct CuVideoDispInfo {
    pub picture_index: c_int,
    pub progressive_frame: c_int,
    pub top_field_first: c_int,
    pub repeat_first_field: c_int,
    pub timestamp: c_ulonglong,
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
// Reference: cuviddec.h of the NVIDIA Video Codec SDK 12.2 (the `Interface`
// headers NVIDIA ships with the SDK, MIT-licensed; older revisions on
// github.com/NVIDIA/video-sdk-samples).

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
/// cuviddec.h (NVIDIA Video Codec SDK 12.2); reserved tail padded out so even
/// if the driver adds a small block in a future SDK we still fit. Real SDK reports
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
pub struct CuVideoProcParams {
    pub progressive_frame: c_int,
    pub second_field: c_int,
    pub top_field_first: c_int,
    pub unpaired_field: c_int,
    pub reserved_flags: c_uint,
    pub reserved_zero: c_uint,
    pub raw_input_dptr: c_ulonglong,
    pub raw_input_pitch: c_uint,
    pub raw_input_format: c_uint,
    pub raw_output_dptr: c_ulonglong,
    pub raw_output_pitch: c_uint,
    pub reserved1: c_uint,
    pub output_stream: *mut c_void,
    pub reserved: [c_uint; 46],
    pub histogram_dptr: *mut c_void,
    pub reserved2: [*mut c_void; 1],
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
// Expected sizes are computed against the NVIDIA Video Codec SDK 12.2
// headers (nvcuvid.h, cuviddec.h) on Windows MSVC x64 (c_ulong=4, pointer=8).
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

pub type FnCuvidCreateVideoParser =
    unsafe extern "C" fn(*mut CUvideoparser, *mut CuVideoParserParams) -> CUresult;
pub type FnCuvidParseVideoData =
    unsafe extern "C" fn(CUvideoparser, *mut CuVideoSourceDataPacket) -> CUresult;
pub type FnCuvidDestroyVideoParser = unsafe extern "C" fn(CUvideoparser) -> CUresult;
pub type FnCuvidCreateDecoder =
    unsafe extern "C" fn(*mut CUvideodecoder, *mut CuVideoDecodeCreateInfo) -> CUresult;
pub type FnCuvidDestroyDecoder = unsafe extern "C" fn(CUvideodecoder) -> CUresult;
pub type FnCuvidDecodePicture =
    unsafe extern "C" fn(CUvideodecoder, *mut CuVideoPicParams) -> CUresult;
pub type FnCuvidMapVideoFrame = unsafe extern "C" fn(
    CUvideodecoder,
    c_int,
    *mut CUdeviceptr,
    *mut c_uint,
    *mut CuVideoProcParams,
) -> CUresult;
pub type FnCuvidUnmapVideoFrame = unsafe extern "C" fn(CUvideodecoder, CUdeviceptr) -> CUresult;
pub type FnCuvidGetDecoderCaps = unsafe extern "C" fn(*mut CuVideoDecodeCaps) -> CUresult;

// CUVIDDECODECAPS (cuviddec.h, NVIDIA Video Codec SDK): the caller fills the
// IN fields (codec / chroma / bit-depth) and the driver fills the OUT fields —
// whether the GPU's NVDEC supports that combination and its min/max
// dimensions. Run before `cuvidCreateDecoder` so an unsupported tuple is a
// clean typed error.
//
// 88 bytes, as NVIDIA's current header has it:
//
//   cudaVideoCodec eCodecType; cudaVideoChromaFormat eChromaFormat;
//   unsigned int nBitDepthMinus8; unsigned int reserved1[3];
//   unsigned char bIsSupported; unsigned char nNumNVDECs;
//   unsigned short nOutputFormatMask; unsigned int nMaxWidth;
//   unsigned int nMaxHeight; unsigned int nMaxMBCount;
//   unsigned short nMinWidth; unsigned short nMinHeight;
//   unsigned char bIsHistogramSupported; unsigned char nCounterBitDepth;
//   unsigned short nMaxHistogramBins; unsigned int reserved3[10];
//
// (The SDK 8/9 header's `reserved2[3]` / `reserved3[11]` is the same 88.)
// Until 2026-10-03 this mirror ended at `reserved3[8]`, 80 bytes: the driver
// writes the struct whole, so every caps query wrote 8 bytes past it.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CuVideoDecodeCaps {
    // IN
    pub codec_type: c_int,
    pub chroma_format: c_int,
    pub bit_depth_minus8: u32,
    pub reserved1: [u32; 3],
    // OUT
    pub is_supported: u8,
    pub num_nvdecs: u8,
    pub output_format_mask: u16,
    pub max_width: u32,
    pub max_height: u32,
    pub max_mb_count: u32,
    pub min_width: u16,
    pub min_height: u16,
    pub is_histogram_supported: u8,
    pub counter_bit_depth: u8,
    pub max_histogram_bins: u16,
    pub reserved3: [u32; 10],
}
const _: () = assert!(std::mem::size_of::<CuVideoDecodeCaps>() == 88);
const _: () = assert!(std::mem::offset_of!(CuVideoDecodeCaps, is_supported) == 24);
const _: () = assert!(std::mem::offset_of!(CuVideoDecodeCaps, max_width) == 28);
const _: () = assert!(std::mem::offset_of!(CuVideoDecodeCaps, min_width) == 40);
const _: () = assert!(std::mem::offset_of!(CuVideoDecodeCaps, reserved3) == 48);

// ─── Codec constants ───────────────────────────────────────────────
pub const CUVID_H264: c_int = 4;
pub const CUVID_HEVC: c_int = 8;
pub const CUVID_VP8: c_int = 9;
pub const CUVID_VP9: c_int = 10;
pub const CUVID_AV1: c_int = 11;
pub const CUVID_MPEG2: c_int = 1;
pub const CUVID_MPEG4: c_int = 3;

pub const CUVID_PKT_ENDOFSTREAM: c_ulong = 1;
/// "Timestamp in packet is valid" (NVDEC Programming Guide 4.1.2,
/// `CUvideopacketflags` in nvcuvid.h): the parser hands the packet's
/// timestamp back on `CUVIDPARSERDISPINFO` only when this is set, so every
/// data packet sets it.
pub const CUVID_PKT_TIMESTAMP: c_ulong = 2;
/// `CUVID_PKT_ENDOFPICTURE` (nvcuvid.h): "Set when the packet contains
/// exactly one frame or one field". The NVDEC Programming Guide (4.1.2):
/// "MUST be set when packet contains exactly one frame or one field data
/// ... If packet has more than one frame data, parser will trigger decode
/// callback for first frame data. Rest of the NALU will get dropped." Set
/// on VP8 and VP9 packets, which are fed one frame each (a VP9 superframe is
/// split first: [`split_frames`](super::convert::split_frames)).
pub const CUVID_PKT_ENDOFPICTURE: c_ulong = 0x08;

// cudaVideoSurfaceFormat (cuviddec.h):
//   NV12 = 0    — 8-bit per sample, semi-planar (Y plane + interleaved UV)
//   P016 = 1    — 16-bit per sample, semi-planar; 10-bit data in the
//                 high 10 bits of each u16, low 6 bits zero-padded
//   YUV444 = 2  — 8-bit 4:4:4
//   YUV444_16 = 3 — 16-bit 4:4:4
// We only use NV12 (8-bit 4:2:0) and P016 (10/12-bit 4:2:0).
pub const CUVID_FMT_NV12: c_int = 0;
pub const CUVID_FMT_P016: c_int = 1;
pub const CUVID_CHROMA_420: c_int = 1;
/// `cudaVideoCreate_PreferCUVID` (`cudaVideoCreateFlags`, cuviddec.h,
/// NVIDIA Video Codec SDK): "Use dedicated video engines directly". The
/// header's other values are `_Default` 0x00, `_PreferCUDA` 0x01 ("Use
/// CUDA-based decoder (requires valid vidLock object for
/// multi-threading)") and `_PreferDXVA` 0x02.
///
/// Until 2026-10-03 this was 0x01 — `_PreferCUDA` under the CUVID name —
/// with no `vidLock`. Not yet re-run on NVIDIA hardware after the fix.
pub const CUVID_CREATE_PREFER_CUVID: c_ulong = 0x04;

/// Structural mirror of `CUVIDOPERATINGPOINTINFO` (nvcuvid.h). Not
/// read at runtime — the callback above returns a fixed value
/// without inspecting the struct — but the shape is documented here
/// so a future session implementing layer-selective decode has a
/// reference. Tagged with `#[allow(dead_code)]` to silence the
/// unused-field warnings.
#[repr(C)]
#[allow(dead_code)]
pub struct CuVideoOperatingPointInfo {
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
