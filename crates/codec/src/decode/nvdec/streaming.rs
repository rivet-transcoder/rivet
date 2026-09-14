//! True-streaming NVDEC decoder (Squad-36) and init-error trampoline.
//!
//! `NvdecStreamingDecoder` drives `cuvidParseVideoData` once per
//! `push_sample` (no per-stream accumulation), the display callback
//! enqueues into a `VecDeque<DecodedFrame>`, and `decode_next` pops
//! one frame per call. `finish()` sends `CUVID_PKT_ENDOFSTREAM`.
//!
//! See the long design-rationale comment at the top of the original
//! `nvdec.rs` around line 2116 for the full memory-shape analysis.

use anyhow::{Context, Result, bail};
use std::collections::VecDeque;
use std::ffi::c_void;
use std::os::raw::{c_int, c_ulong, c_ulonglong};
use std::ptr;
use std::sync::{Arc, Mutex};

use crate::decode::Decoder;
use crate::frame::{ColorMetadata, ColorSpace, StreamInfo, TransferFn, VideoFrame};
use super::callbacks::{
    decode_callback, display_callback, get_operating_point_callback, sequence_callback,
};
use super::convert::{codec_to_cuvid, decoded_frame_to_video_frame};
use super::ffi::{
    CUcontext, CUdevice,
    CuVideoParserParams, CuVideoSourceDataPacket, CUvideoparser,
    FnCuCtxCreate, FnCuCtxDestroy, FnCuCtxPopCurrent, FnCuCtxPushCurrent, FnCuDeviceGet,
    FnCuInit, FnCuMemcpy2D,
    FnCuvidCreateDecoder, FnCuvidCreateVideoParser, FnCuvidDecodePicture,
    FnCuvidDestroyDecoder, FnCuvidDestroyVideoParser, FnCuvidGetDecoderCaps,
    FnCuvidMapVideoFrame, FnCuvidParseVideoData, FnCuvidUnmapVideoFrame,
    CUVID_AV1, CUVID_PKT_ENDOFSTREAM, CUVID_PKT_TIMESTAMP,
};
use super::state::{CallbackState, CtxScope, FrameCollector};

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
pub(super) struct NvdecCtx {
    // Library handles MUST drop AFTER any field that holds borrowed
    // fn pointers from them (Reference §10.8 — fields drop in source
    // order). Declared LAST in `NvdecStreamingDecoder` for the same
    // reason.
    pub(super) cu_ctx: CUcontext,
    pub(super) cu_ctx_destroy: FnCuCtxDestroy,
    pub(super) cu_ctx_push: FnCuCtxPushCurrent,
    pub(super) cu_ctx_pop: FnCuCtxPopCurrent,
    pub(super) cuvid_destroy_parser: FnCuvidDestroyVideoParser,
    pub(super) cuvid_destroy_decoder: FnCuvidDestroyDecoder,
    pub(super) cuvid_parse_data: FnCuvidParseVideoData,
}

// SAFETY: CUcontext is just an opaque void* the driver returns; it is
// thread-bound only via cuCtxPushCurrent/PopCurrent which we wrap with
// CtxScope. fn pointers are POD. Send is the only required marker for
// the Decoder trait.
unsafe impl Send for NvdecCtx {}

/// True-streaming NVDEC decoder. See module-level comment block
/// above this struct for the design rationale.
pub struct NvdecStreamingDecoder {
    pub(super) info: StreamInfo,
    /// Owns the boxed CallbackState referenced from
    /// `parser_params.user_data`. Must outlive `parser`. We keep an
    /// `Arc` clone of `state.collector` outside so `decode_next` can
    /// drain without re-locking through the box.
    pub(super) state: Box<CallbackState>,
    /// Mirror of `state.collector` so `decode_next` doesn't need to
    /// borrow `state` (keeps the borrow checker happy when push_sample
    /// is also taking &mut self).
    pub(super) collector: Arc<Mutex<FrameCollector>>,
    /// CUVID parser handle. Created in `try_new`, destroyed in `Drop`.
    pub(super) parser: CUvideoparser,
    /// Resolved FFI + CUDA context. Drop order: parser → ctx
    /// (handled by `Drop` on this struct).
    pub(super) ctx: NvdecCtx,
    /// EOS already sent? Subsequent push_sample calls return Ok(())
    /// (idempotent finish; matches the trait shape every other
    /// streaming-shape decoder follows).
    pub(super) finished: bool,
    /// Sample counter for fabricated PTS when the trait-level
    /// `push_sample(&[u8])` is called without an explicit PTS. Real
    /// demuxer PTS still flows through the eager / push-with-pts
    /// paths.
    pub(super) sample_counter: u64,

    // Library handles held last so they outlive every fn pointer
    // captured into `state`. See the eager NvdecDecoder field-order
    // note for the Reference §10.8 cite.
    pub(super) _cuvid_lib: libloading::Library,
    pub(super) _cuda_lib: libloading::Library,
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
    pub(super) fn try_new(info: StreamInfo, gpu_index: u32) -> Result<Self> {
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
            let cuvid_get_decoder_caps: libloading::Symbol<FnCuvidGetDecoderCaps> =
                cuvid_lib.get(b"cuvidGetDecoderCaps")?;
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
                cuvid_get_decoder_caps: *cuvid_get_decoder_caps,
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
            // Bit 0 of this word is `bAnnexb`, which the SDK header
            // (`CUVIDPARSERPARAMS`) documents as "IN: AV1 annexB stream":
            // length-delimited Annex B temporal units. MP4 / MKV / WebM carry
            // AV1 as low-overhead (Section 5) OBUs, so with the bit set every
            // AV1 `cuvidParseVideoData` returned 999, the sequence callback
            // never ran, and the decoder yielded no frames and no error. The
            // bit is AV1-only; H.264 / HEVC keep it as they always had it
            // (Squad-12 / task #39), so their parse is unchanged.
            if cuvid_codec != CUVID_AV1 {
                parser_params.reserved1[0] = 1;
            }
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
pub(super) struct NvdecInitErrorDecoder {
    pub(super) info: StreamInfo,
    /// Taken on the first push_sample / finish / decode_next call.
    /// Subsequent calls return Ok(()) / Ok(None) so the caller can
    /// drain cleanly after seeing the first error.
    pub(super) error: Option<anyhow::Error>,
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
