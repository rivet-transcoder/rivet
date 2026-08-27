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

mod buffers;
mod constants;
mod ffi;
mod helpers;
mod session;
mod upload;
#[cfg(test)]
mod tests;

use anyhow::{Context, Result, anyhow, bail};
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
use crate::frame::{PixelFormat, VideoFrame};

use self::buffers::{
    NvEncCreateBitstreamBuffer, NvEncCreateInputBuffer, NvEncFunctionList, NvEncLockBitstream,
    NvEncPicParams,
};
use self::constants::{
    CUcontext, CUdevice, FnCuCtxCreate, FnCuCtxDestroy, FnCuCtxPopCurrent, FnCuCtxPushCurrent,
    FnCuDeviceGet, FnCuInit, FnNvEncCreateBitstreamBuffer, FnNvEncCreateInputBuffer,
    FnNvEncDestroyBitstreamBuffer, FnNvEncDestroyEncoder, FnNvEncDestroyInputBuffer,
    FnNvEncEncodePicture, FnNvEncGetEncodeCaps, FnNvEncGetEncodeGUIDCount, FnNvEncGetEncodeGUIDs,
    FnNvEncGetEncodePresetConfigEx, FnNvEncInitializeEncoder, FnNvEncLockBitstream,
    FnNvEncGetSequenceParams, FnNvEncLockInputBuffer, FnNvEncOpenEncodeSessionEx,
    FnNvEncReconfigureEncoder, FnNvEncUnlockBitstream,
    FnNvEncUnlockInputBuffer, FnNvEncodeAPICreateInstance, FnNvEncodeAPIGetMaxSupportedVersion,
    Guid, NV_ENC_CAPS_HEIGHT_MAX, NV_ENC_CAPS_SUPPORT_10BIT_ENCODE, NV_ENC_CAPS_WIDTH_MAX,
    NV_ENC_CONFIG_VER, NV_ENC_CREATE_BITSTREAM_BUFFER_VER, NV_ENC_CREATE_INPUT_BUFFER_VER,
    NV_ENC_DEVICE_TYPE_CUDA, NV_ENC_ERR_ENCODER_BUSY, NV_ENC_ERR_ENCODER_NOT_INITIALIZED,
    NV_ENC_ERR_INVALID_PARAM, NV_ENC_ERR_INVALID_PTR, NV_ENC_ERR_LOCK_BUSY,
    NV_ENC_ERR_NEED_MORE_INPUT, NV_ENC_INITIALIZE_PARAMS_VER, NV_ENC_LOCK_BITSTREAM_VER,
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER, NV_ENC_PARAMS_RC_CONSTQP, NV_ENC_PARAMS_RC_VBR,
    NV_ENC_PIC_FLAG_EOS, NV_ENC_PIC_FLAG_FORCEIDR, NV_ENC_PIC_PARAMS_VER, NV_ENC_PIC_TYPE_I,
    NV_ENC_PIC_TYPE_IDR, NV_ENC_PIC_TYPE_P, NV_ENC_PRESET_CONFIG_VER,
    NV_ENCODE_API_FUNCTION_LIST_VER, NVENCAPI_VERSION, NvEncCapsParam, RC_FLAG_ENABLE_LOOKAHEAD,
    RC_FLAG_ZERO_REORDER_DELAY, RING_SIZE, struct_version, nvenc_codec_guid, nvenc_profile_guid,
    guid_from_bytes, NV_ENC_SUCCESS,
};
use self::ffi::{
    AV1_BIT_REPEAT_SEQ_HDR, AV1_CHROMA_FORMAT_IDC_420, NV_ENC_BIT_DEPTH_8, NV_ENC_BIT_DEPTH_10,
    NvEncConfig, NvEncConfigAv1, NvEncConfigHevcBitDepth, NvEncInitializeParams,
    NvEncOpenEncodeSessionExParams, NvEncPresetConfig,
};
use self::helpers::{fps_to_rational, nvenc_buffer_format_for, pixel_bit_depth_minus8_for,
    transfer_to_h273};
use self::session::EncodeSession;
use self::upload::{upload_frame, upload_frame_10bit};

pub struct NvencEncoder {
    config: EncoderConfig,
    session: Option<EncodeSession>,
    pending_frames: Vec<VideoFrame>,
    encoded_packets: Vec<EncodedPacket>,
    flushed: bool,
    packet_cursor: usize,
    /// Pictures submitted over the whole life of the session — the driver's
    /// `frameIdx`. Never rewound, not even by `reset`: the stale-read filter
    /// on the EOS drain (`last_drained_frame_idx`) only works while this is
    /// monotonic, so a packet left in a slot by the previous stream can
    /// never be mistaken for one of the current stream's.
    frame_counter: u32,
    /// Pictures submitted since the session was built or last reset — the
    /// position in the *current* stream. Drives the IDR cadence (so the
    /// first picture after a reset is an IDR) and the "was anything
    /// buffered" test in `flush_eos`.
    chunk_frames: u32,
    /// Parameter sets to prepend to the first packet of a reset stream —
    /// see `EncodeSession::sequence_params`. Set by `reset` for H.264/H.265,
    /// taken by the first packet queued after it. AV1 never needs it: its
    /// sequence header is repeated on every keyframe by config.
    pending_headers: Option<Bytes>,
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
    /// Slots NVENC has taken a frame for but not yet returned a packet for.
    ///
    /// Set when EncodePicture answers NEED_MORE_INPUT, which means the encoder
    /// is still holding that input surface. Uploading the next frame into a
    /// slot in this state overwrites a picture the encoder has not read yet —
    /// silently, producing an output frame with the wrong pixels. That is a
    /// real bug that shipped: see the ring note in `encode_pending`.
    slot_in_flight: [bool; RING_SIZE],
    /// Slots with a submission outstanding, oldest first — so "block on the one
    /// that has been waiting longest" is answerable when every slot is busy.
    inflight: std::collections::VecDeque<usize>,
    _encode_lib: libloading::Library,
    _cuda_lib: libloading::Library,
}

impl NvencEncoder {
    pub fn new(config: EncoderConfig, gpu_index: u32) -> Result<Self> {
        // The codec GUID drives capability validation, preset selection, and
        // session init. AV1 (Ada+ / Ampere datacenter), H.264 (Kepler+), and
        // H.265 (Maxwell+) all dispatch through the same path; codec-specific
        // config is branched below. H.264/H.265 emit Annex-B (the muxer's
        // nal_mux repackages); only the AV1 union is hand-overridden.
        let codec_guid = nvenc_codec_guid(config.codec);
        let is_av1 = config.codec == crate::frame::VideoCodec::Av1;
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
            // Optional rather than required: a table without it still
            // encodes, it just cannot `reset` — the caller rebuilds, as it
            // always did.
            let fn_reconfigure_encoder: Option<FnNvEncReconfigureEncoder> =
                if fn_list.nv_enc_reconfigure_encoder.is_null() {
                    None
                } else {
                    Some(std::mem::transmute::<*mut c_void, FnNvEncReconfigureEncoder>(
                        fn_list.nv_enc_reconfigure_encoder,
                    ))
                };
            let fn_get_sequence_params: Option<FnNvEncGetSequenceParams> =
                if fn_list.nv_enc_get_sequence_params.is_null() {
                    None
                } else {
                    Some(std::mem::transmute::<*mut c_void, FnNvEncGetSequenceParams>(
                        fn_list.nv_enc_get_sequence_params,
                    ))
                };
            // Preset-config-ex: required for HIGH-1 fix. If the SDK
            // fn-list is missing it the driver is too old for AV1
            // anyway (added in 12.x).
            let fn_get_preset_config_ex: FnNvEncGetEncodePresetConfigEx = cast_fn!(
                fn_list.nv_enc_get_encode_preset_config_ex,
                FnNvEncGetEncodePresetConfigEx,
                "GetEncodePresetConfigEx"
            );
            // Capability-query fns — used to confirm this GPU's NVENC actually
            // does AV1 (and the requested resolution / bit depth) instead of
            // guessing from the board name.
            let fn_get_guid_count: FnNvEncGetEncodeGUIDCount = cast_fn!(
                fn_list.nv_enc_get_encode_guid_count,
                FnNvEncGetEncodeGUIDCount,
                "GetEncodeGUIDCount"
            );
            let fn_get_guids: FnNvEncGetEncodeGUIDs = cast_fn!(
                fn_list.nv_enc_get_encode_guids,
                FnNvEncGetEncodeGUIDs,
                "GetEncodeGUIDs"
            );
            let fn_get_encode_caps: FnNvEncGetEncodeCaps = cast_fn!(
                fn_list.nv_enc_get_encode_caps,
                FnNvEncGetEncodeCaps,
                "GetEncodeCaps"
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

            // ─── Capability validation (real driver query) ──────────
            // Ask the driver what this GPU's NVENC actually supports instead of
            // guessing from the board name: enumerate the encode codecs and bail
            // cleanly if AV1 isn't among them (Ampere consumer / Turing / Pascal),
            // then check AV1 max resolution + 10-bit against the requested config.
            let cap_err: Option<String> = {
                let mut guid_count: u32 = 0;
                if fn_get_guid_count(encoder, &mut guid_count) != NV_ENC_SUCCESS {
                    Some(format!("NvEncGetEncodeGUIDCount failed on GPU {gpu_index}"))
                } else {
                    let mut guids = vec![
                        Guid { data1: 0, data2: 0, data3: 0, data4: [0u8; 8] };
                        guid_count.max(1) as usize
                    ];
                    let mut returned: u32 = 0;
                    if fn_get_guids(encoder, guids.as_mut_ptr(), guid_count, &mut returned)
                        != NV_ENC_SUCCESS
                    {
                        Some(format!("NvEncGetEncodeGUIDs failed on GPU {gpu_index}"))
                    } else if !guids[..returned as usize]
                        .iter()
                        .any(|g| *g == codec_guid)
                    {
                        Some(format!(
                            "NVENC on GPU {gpu_index} does not support {:?} encode \
                             ({returned} codec(s) advertised, none matched) — AV1 needs \
                             NVIDIA Ada+ / an Ampere datacenter SKU; H.264 needs Kepler+; \
                             H.265 needs Maxwell+",
                            config.codec
                        ))
                    } else {
                        // Codec supported — validate resolution + 10-bit caps for
                        // the SELECTED codec's GUID.
                        let query = |cap: c_uint| -> i32 {
                            let mut p: NvEncCapsParam = std::mem::zeroed();
                            p.version = struct_version(1);
                            p.caps_to_query = cap;
                            let mut val: c_int = 0;
                            let rc = fn_get_encode_caps(encoder, codec_guid, &mut p, &mut val);
                            if rc != NV_ENC_SUCCESS { -1 } else { val }
                        };
                        let w_max = query(NV_ENC_CAPS_WIDTH_MAX);
                        let h_max = query(NV_ENC_CAPS_HEIGHT_MAX);
                        if w_max > 0
                            && h_max > 0
                            && ((config.width as i32) > w_max || (config.height as i32) > h_max)
                        {
                            Some(format!(
                                "NVENC {:?} on GPU {gpu_index} maxes at {w_max}x{h_max}, \
                                 requested {}x{}",
                                config.codec, config.width, config.height
                            ))
                        } else if config.pixel_format == PixelFormat::Yuv420p10le
                            && query(NV_ENC_CAPS_SUPPORT_10BIT_ENCODE) == 0
                        {
                            Some(format!(
                                "NVENC on GPU {gpu_index} does not support 10-bit {:?} encode",
                                config.codec
                            ))
                        } else {
                            tracing::info!(
                                gpu_index,
                                codec = ?config.codec,
                                w_max,
                                h_max,
                                ten_bit = config.pixel_format == PixelFormat::Yuv420p10le,
                                "NVENC capability validated"
                            );
                            None
                        }
                    }
                }
            };
            if let Some(msg) = cap_err {
                fn_destroy_encoder(encoder);
                (*fn_cu_ctx_destroy)(cuda_ctx);
                bail!("{msg}");
            }

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
                codec_guid,
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

            // ChunkSeamMode::ParallelConstQp — force real constant-QP so
            // independently-encoded chunks have flat quality across the stitched
            // seams. Unlike the (now-retired) shiguredo wrapper, which exposed no
            // QP value and fell back to the preset default, we set the QP from
            // the tuning CQ. intra < interP < interB so keyframes get more bits.
            if config.constant_qp {
                let q = nvenc_cq as u32;
                enc_config.rc_params.rate_control_mode = NV_ENC_PARAMS_RC_CONSTQP;
                enc_config.rc_params.const_qp_intra = q;
                enc_config.rc_params.const_qp_inter_p = q.saturating_add(1);
                enc_config.rc_params.const_qp_inter_b = q.saturating_add(2);
                enc_config.rc_params.target_quality = q.min(255) as u8;
            }

            // Force strictly 1-in-1-out — for every codec, not just H.26x.
            //
            // The input-surface ring is `RING_SIZE` (4) deep and `encode_pending`
            // advances it after every EncodePicture, including the ones that
            // answer NEED_MORE_INPUT — where NVENC has *not* released the
            // surface. Four frames later that surface is overwritten while the
            // encoder still owes a packet for it, so the picture it finally
            // emits carries frame N+4's pixels and frame N's are lost.
            //
            // These four lines used to live in the H.264/H.265 arm below, so AV1
            // kept the driver preset's lookahead and hit exactly that: because a
            // fresh encoder is built per segment, the warmup — and the swap —
            // repeated at every segment, and each segment's first frame showed
            // the frame four later before snapping back. Confirmed on a 25 fps
            // AV1 ladder by matching decoded output against the source: output
            // frames 0, 100 and 200 — every segment start — each carried the
            // source frame four ahead, and the frame they replaced was gone.
            //
            // They are rate-control fields, not codec-union fields, so applying
            // them uniformly is safe.
            // Lookahead and reordering are now a caller decision rather than a
            // blanket off.
            //
            // They were disabled for every codec because the four-slot ring
            // could not survive the encoder holding frames. The ring now picks
            // a slot with nothing outstanding and blocks on the oldest when all
            // are busy, so buffering is safe and this can be asked for.
            //
            // Still off by default: an empty override has to leave behaviour
            // exactly as it was.
            let lookahead = config.overrides.lookahead_frames.unwrap_or(0);
            let bframes = config.overrides.bframes.unwrap_or(0);

            if lookahead > 0 {
                // Bounded by the pool: the encoder may hold at most what we can
                // avoid overwriting, with slack for the frame being submitted.
                let cap = (RING_SIZE as u32).saturating_sub(4);
                enc_config.rc_params.lookahead_depth = lookahead.min(cap) as u16;
                enc_config.rc_params.flags |= RC_FLAG_ENABLE_LOOKAHEAD;
                enc_config.rc_params.flags &= !RC_FLAG_ZERO_REORDER_DELAY;
            } else {
                enc_config.rc_params.flags &= !RC_FLAG_ENABLE_LOOKAHEAD;
                enc_config.rc_params.flags |= RC_FLAG_ZERO_REORDER_DELAY;
                enc_config.rc_params.lookahead_depth = 0;
            }
            enc_config.frame_interval_p = u32::from(bframes) + 1;
            enc_config.rc_params.multi_pass =
                if config.overrides.multi_pass.unwrap_or(false) { 1 } else { 0 };

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
            // pixel_bit_depth_minus_8 fields. The `NV_ENC_BIT_DEPTH` enum is
            // the *literal* bit depth: 8-bit = 8, 10-bit = 10 (NOT 0/1 — the
            // driver rejects 0/1, which mis-sized the input surface and tripped
            // `NvEncCreateInputBuffer` INVALID_PARAM on the first 10-bit run).
            let buffer_format = nvenc_buffer_format_for(config.pixel_format)?;
            let bit_depth_minus8 = pixel_bit_depth_minus8_for(config.pixel_format);
            let bit_depth_enum = if bit_depth_minus8 == 0 {
                NV_ENC_BIT_DEPTH_8
            } else {
                NV_ENC_BIT_DEPTH_10
            };

            if is_av1 {
                // ─── AV1 codec-specific config (SDK 13 union view) ───────
                // outputAnnexBFormat = 0 (LOB), repeatSeqHdr = 1, chroma 4:2:0.
                enc_config.codec_config_av1.flags =
                    AV1_BIT_REPEAT_SEQ_HDR | AV1_CHROMA_FORMAT_IDC_420;
                enc_config.codec_config_av1.idr_period = config.keyframe_interval;

                // Reference frames. Left at the driver's own choice unless the
                // caller asks — `num_fwd_refs = 0` means autoselect, which is
                // what every previous run did, so an empty override still
                // produces the identical bitstream.
                //
                // AV1 allows seven; the DPB has to hold them plus the picture
                // being coded, and NVENC rejects a DPB smaller than the
                // reference count it was asked for.
                match config.overrides.reference_frames {
                    Some(refs) => {
                        let refs = u32::from(refs).clamp(1, 7);
                        enc_config.codec_config_av1.num_fwd_refs = refs;
                        enc_config.codec_config_av1.max_num_ref_frames_in_dpb = (refs + 1).max(4);
                    }
                    None => {
                        enc_config.codec_config_av1.max_num_ref_frames_in_dpb = 4;
                    }
                }
                enc_config.codec_config_av1.num_tile_columns = tp.num_tile_columns;
                enc_config.codec_config_av1.num_tile_rows = tp.num_tile_rows;
                enc_config.codec_config_av1.output_bit_depth = bit_depth_enum;
                enc_config.codec_config_av1.input_bit_depth = bit_depth_enum;

                // Color signalling — wire ColorMetadata into the OBU seq header.
                let cm = &config.color_metadata;
                enc_config.codec_config_av1.color_primaries = cm.colour_primaries as u32;
                enc_config.codec_config_av1.transfer_characteristics =
                    transfer_to_h273(cm.transfer);
                enc_config.codec_config_av1.matrix_coefficients = cm.matrix_coefficients as u32;
                enc_config.codec_config_av1.color_range = cm.full_range as u32;
            } else {
                // H.264 / H.265: the preset's GetEncodePresetConfigEx already
                // seeded the codec config union (idrPeriod, entropy mode, VUI,
                // SPS/PPS-at-start) with driver-blessed defaults for the codec
                // GUID. We deliberately do NOT hand-write the H264/HEVC union
                // (its exact SDK-13 layout would have to be mirrored byte-exact
                // and the union slot is already filled). Output is Annex-B with
                // SPS/PPS on the first IRAP — exactly what the muxer's nal_mux
                // captures into avcC/hvcC. The shared gop_length above drives
                // periodic IDRs; profile is pinned via init_params below.
                //
                // The 1-in-1-out rate-control settings this arm used to carry are
                // now applied to every codec above. They mattered most here —
                // the H.264/H.265 presets enable RC lookahead (~16 frames),
                // which the single-pass EOS drain can't recover, observed as
                // 84/96 frame loss (H.264) and an EOS deadlock (H.265) — but
                // AV1 needed them too.

                // H.265 Main 10: the encoder validates the input surface format
                // against its configured inputBitDepth at NvEncCreateInputBuffer
                // time, so a 10-bit buffer on the 8-bit-default preset config
                // fails INVALID_PARAM. Overlay the SDK-13 NV_ENC_CONFIG_HEVC
                // bit-depth fields (offsets 200/204) on the union and set them
                // to 10. Only the two u32s are written — every preset-seeded
                // field (level/tier/VUI/chroma) is left intact. H.264 has no
                // 10-bit hardware path (capability-rejected upstream), so the
                // poke is gated to H.265 — its layout differs from H.264's.
                if config.codec == crate::frame::VideoCodec::H265 && bit_depth_minus8 != 0 {
                    let hevc = &mut *(&mut enc_config.codec_config_av1 as *mut NvEncConfigAv1
                        as *mut NvEncConfigHevcBitDepth);
                    hevc.output_bit_depth = bit_depth_enum;
                    hevc.input_bit_depth = bit_depth_enum;
                }
                tracing::info!(
                    codec = ?config.codec,
                    ten_bit = bit_depth_minus8 != 0,
                    "NVENC H.264/H.265 using preset-seeded codec config (Annex-B, 1-in-1-out)"
                );
            }

            let mut init_params: NvEncInitializeParams = std::mem::zeroed();
            init_params.version = NV_ENC_INITIALIZE_PARAMS_VER;
            init_params.encode_guid = codec_guid;
            // Pin H.264 High / H.265 Main; AV1 keeps the preset's auto profile.
            enc_config.profile_guid = nvenc_profile_guid(config.codec, bit_depth_minus8 != 0);
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

            // Keep the stream description for `reset`. Boxed so the config
            // pointer inside the init params has a stable target; the pointer
            // is refreshed on every reconfigure anyway.
            let mut enc_config = Box::new(enc_config);
            let mut init_params = Box::new(init_params);
            init_params.encode_config = &mut *enc_config as *mut NvEncConfig as *mut c_void;

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
                fn_reconfigure_encoder,
                fn_get_sequence_params,
                init_params,
                enc_config,
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
                chunk_frames: 0,
                pending_headers: None,
                ring_idx: 0,
                last_drained_frame_idx: [-1; RING_SIZE],
                slot_in_flight: [false; RING_SIZE],
                inflight: std::collections::VecDeque::with_capacity(RING_SIZE),
                _encode_lib: encode_lib,
                _cuda_lib: cuda_lib,
            })
        }
    }

    unsafe fn drain_bitstream(
        session: &EncodeSession,
        slot: usize,
        do_not_wait: bool,
    ) -> Result<Option<(u32, EncodedPacket)>> {
        unsafe {
            let bitstream_buffer = session.bitstream_buffers[slot];
            let mut lock: NvEncLockBitstream = std::mem::zeroed();
            lock.version = NV_ENC_LOCK_BITSTREAM_VER;
            // doNotWait (bit 0 of the bitfield word): non-blocking lock. The EOS
            // flush walks ring slots that may have no pending output — a blocking
            // lock there busy-waits forever (the encoder produced all packets
            // 1-in-1-out during encode, so there's nothing to flush). Per-frame
            // drains pass false: in sync mode the output is ready right after a
            // SUCCESS EncodePicture, so blocking returns immediately.
            lock.bitfields = if do_not_wait { 1 } else { 0 };
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

    /// Queue a drained packet, giving the first packet of a reset stream the
    /// parameter sets the driver did not write in-band. A free function over
    /// the two fields rather than a method: the callers hold `self.session`
    /// borrowed across the ring loop.
    fn queue_packet(
        packets: &mut Vec<EncodedPacket>,
        pending_headers: &mut Option<Bytes>,
        mut pkt: EncodedPacket,
    ) {
        if let Some(headers) = pending_headers.take() {
            let mut data = Vec::with_capacity(headers.len() + pkt.data.len());
            data.extend_from_slice(&headers);
            data.extend_from_slice(&pkt.data);
            pkt.data = Bytes::from(data);
        }
        packets.push(pkt);
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
            // Take a slot the encoder has finished with, not the next in turn.
            //
            // Round-robin over four slots is only correct while every
            // EncodePicture returns a packet immediately. The moment the
            // encoder buffers — lookahead, reordering — it keeps the input
            // surface, and the slot four submissions later is that same memory:
            // the picture it eventually emits carries the newer frame's pixels.
            // That is a silently wrong frame, not an error, and it is what the
            // per-segment jump was.
            //
            // So: find a slot with nothing outstanding; if every slot is busy,
            // block on the oldest until it yields, which is the only thing that
            // can free one. That makes buffering safe rather than forbidden,
            // which is what lets lookahead be turned back on.
            let slot = match (0..RING_SIZE).find(|&i| !self.slot_in_flight[i]) {
                Some(free) => free,
                None => {
                    let oldest = self
                        .inflight
                        .pop_front()
                        .ok_or_else(|| anyhow!("every NVENC slot busy but none recorded"))?;

                    // Blocking: the encoder owes us this one, and the wait is
                    // bounded by it finishing the frame it already has.
                    match unsafe { Self::drain_bitstream(session, oldest, false)? } {
                        Some((frame_idx, pkt)) => {
                            self.last_drained_frame_idx[oldest] = frame_idx as i64;
                            Self::queue_packet(&mut self.encoded_packets, &mut self.pending_headers, pkt);
                        }
                        None => bail!(
                            "NVENC slot {oldest} yielded no packet on a blocking lock. The                              encoder is holding {RING_SIZE} frames and will not release one;                              overwriting a held surface would emit the wrong picture."
                        ),
                    }
                    self.slot_in_flight[oldest] = false;
                    oldest
                }
            };

            unsafe {
                // Dispatch to the bit-depth-appropriate uploader.
                // Both end up writing into the same NVENC input
                // surface; only the per-sample byte width and the
                // value-bit shift differ.
                let pitch = match frame.format {
                    PixelFormat::Yuv420p10le => upload_frame_10bit(session, &frame, slot)?,
                    _ => upload_frame(session, &frame, slot)?,
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
                // Counted from the start of the *current* stream, so the first
                // picture after a `reset` is an IDR by the same rule as the
                // first picture of a new session.
                let is_idr = self
                    .chunk_frames
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
                self.chunk_frames += 1;

                match rc {
                    NV_ENC_SUCCESS => {
                        let got = Self::drain_bitstream(session, slot, false)?;
                        let drained = got.is_some();
                        if let Some((frame_idx, pkt)) = got {
                            self.last_drained_frame_idx[slot] = frame_idx as i64;
                            Self::queue_packet(&mut self.encoded_packets, &mut self.pending_headers, pkt);
                        }
                        // SUCCESS without a packet still means the encoder is
                        // sitting on this surface.
                        self.slot_in_flight[slot] = !drained;
                        if !drained {
                            self.inflight.push_back(slot);
                        }
                        tracing::debug!(
                            target: "nvenc_drain",
                            frame = self.frame_counter - 1,
                            slot,
                            rc = "SUCCESS",
                            drained,
                            total_packets = self.encoded_packets.len(),
                            "encode_picture"
                        );
                    }
                    NV_ENC_ERR_NEED_MORE_INPUT => {
                        // NVENC is accumulating frames before emitting a
                        // packet, and is still holding this input surface.
                        // Nothing to drain until the next frame — and nothing
                        // may be written into this slot until it comes back.
                        self.slot_in_flight[slot] = true;
                        self.inflight.push_back(slot);
                        tracing::debug!(
                            target: "nvenc_drain",
                            frame = self.frame_counter - 1,
                            slot,
                            rc = "NEED_MORE_INPUT",
                            "encode_picture (buffering)"
                        );
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
        // If every submitted frame was already drained during encode (the
        // 1-in-1-out case, forced for every codec via zeroReorderDelay + no
        // lookahead), there is nothing buffered to flush. Sending the EOS
        // picture and locking the empty ring buffers busy-waits forever on the
        // SDK 13 driver — skip it entirely.
        if self.encoded_packets.len() >= self.chunk_frames as usize {
            tracing::info!(
                target: "nvenc_drain",
                packets = self.encoded_packets.len(),
                frames = self.chunk_frames,
                "flush_eos: nothing buffered, skipping EOS drain"
            );
            return Ok(());
        }
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
            let eos_rc = (session.fn_encode_picture)(session.encoder, &mut pic);
            tracing::info!(
                target: "nvenc_drain",
                eos_rc,
                packets_before = self.encoded_packets.len(),
                "flush_eos: EOS picture sent"
            );

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
                if let Some((frame_idx, pkt)) = Self::drain_bitstream(session, slot, true)? {
                    // Stale-read filter: if the driver handed us back a
                    // frame_idx we've already drained from this slot,
                    // it's the previous packet bytes — see drain_bitstream
                    // docstring and 2026-05-01 SDK 13 driver bug note.
                    // Skip silently; the real EOS-flushed frames (if any)
                    // will arrive with frame_idx > last_drained_frame_idx.
                    if (frame_idx as i64) > self.last_drained_frame_idx[slot] {
                        self.last_drained_frame_idx[slot] = frame_idx as i64;
                        Self::queue_packet(&mut self.encoded_packets, &mut self.pending_headers, pkt);
                    }
                }
            }
            tracing::info!(
                target: "nvenc_drain",
                total_packets = self.encoded_packets.len(),
                "flush_eos: drain complete"
            );
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

    /// `NvEncReconfigureEncoder` with the session's own parameters,
    /// `resetEncoder=1` and `forceIDR=1`, then a clean slate on this side:
    /// the packet queue, the cursor, the ring's in-flight bookkeeping and the
    /// in-stream picture count all start over. The session handle, the CUDA
    /// context and both buffer rings are kept — that is the whole saving.
    ///
    /// A stream that was never flushed is flushed first, so no picture is
    /// left in the driver to surface in the next stream. `frame_counter`
    /// deliberately keeps counting — see its field note.
    fn reset(&mut self) -> Result<()> {
        let Some(session) = self.session.as_ref() else {
            bail!("reset called without live session");
        };
        let is_av1 = self.config.codec == crate::frame::VideoCodec::Av1;
        if session.fn_reconfigure_encoder.is_none()
            || (!is_av1 && session.fn_get_sequence_params.is_none())
        {
            return Err(super::ResetUnsupported.into());
        }
        // Everything the driver still owes us is collected and discarded
        // with the old stream; the ring is idle after this.
        self.encode_pending()?;
        if !self.flushed {
            self.flush_eos()?;
        }
        let session = self.session.as_mut().expect("checked above");
        unsafe { session.reconfigure_reset()? };
        // The new stream's first IDR comes without parameter sets (the
        // driver writes them once per session); fetch them now, after the
        // reconfigure, so they describe the stream about to start.
        let headers = if is_av1 { None } else { Some(Bytes::from(unsafe { session.sequence_params()? })) };

        let discarded = self.encoded_packets.len() - self.packet_cursor;
        self.pending_frames.clear();
        self.encoded_packets.clear();
        self.packet_cursor = 0;
        self.flushed = false;
        self.chunk_frames = 0;
        self.pending_headers = headers;
        self.slot_in_flight = [false; RING_SIZE];
        self.inflight.clear();
        tracing::debug!(
            event = "nvenc.reset",
            frames_so_far = self.frame_counter,
            discarded_packets = discarded,
            header_bytes = self.pending_headers.as_ref().map_or(0, |h| h.len()),
            "NVENC session reset in place (NvEncReconfigureEncoder resetEncoder=1 forceIDR=1; \
             parameter sets re-fetched for the first packet)"
        );
        Ok(())
    }
}
