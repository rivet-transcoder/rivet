//! **Legacy eager decoder** — retained as library code and for
//! `NvdecPushDecoder::finish()` (which calls `new_with_pts` after
//! buffering all samples). The production dispatch path goes through
//! `NvdecStreamingDecoder` (Squad-36).

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
use super::state::{CallbackState, CtxScope, DecodedFrame, FrameCollector};
use super::streaming::{NvdecInitErrorDecoder, NvdecStreamingDecoder};

// ─── Public decoder ────────────────────────────────────────────────
pub struct NvdecDecoder {
    pub(super) info: StreamInfo,
    pub(super) decoded_frames: Vec<DecodedFrame>,
    pub(super) frame_cursor: usize,

    // Library handles held so the OS keeps the fn pointers mapped for
    // the life of the decoder. Declared LAST so Rust drops everything
    // above them first (Reference §10.8 — struct fields drop in source
    // order). Required because CallbackState may in theory hold fn
    // pointers into these after new() returns.
    pub(super) _cuvid_lib: libloading::Library,
    pub(super) _cuda_lib: libloading::Library,
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
            //   bit 0 = bAnnexb         "IN: AV1 annexB stream"
            //   bit 1 = bMemoryOptimize
            //   bits 2..31 reserved
            // bAnnexb is AV1-only: it declares length-delimited Annex B
            // temporal units, and MP4 / MKV / WebM carry low-overhead
            // (Section 5) OBUs, so setting it for AV1 made every parse call
            // fail with no frames. H.264 / HEVC keep the bit they always had
            // (it was set here in the belief it meant H.264 Annex-B input and
            // eased open-GOP recovery on exoplayer_h264_main_720p.mp4).
            if cuvid_codec != CUVID_AV1 {
                parser_params.reserved1[0] = 1;
            }
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

