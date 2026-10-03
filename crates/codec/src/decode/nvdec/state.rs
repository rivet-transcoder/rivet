//! Internal runtime state shared across the CUVID parser callbacks,
//! the eager decoder, and the streaming decoder.

use anyhow::{Result, bail};
use std::collections::VecDeque;
use std::os::raw::c_int;
use std::ptr;
use std::sync::{Arc, Mutex};

use crate::frame::ColorSpace;
use super::NvdecError;
use super::ffi::{
    CUcontext, CUvideodecoder,
    FnCuCtxPopCurrent, FnCuCtxPushCurrent,
    FnCuMemcpy2D,
    FnCuvidCreateDecoder, FnCuvidDecodePicture, FnCuvidGetDecoderCaps,
    FnCuvidMapVideoFrame, FnCuvidUnmapVideoFrame,
};

// ─── Decoded frame collector ───────────────────────────────────────
#[derive(Clone)]
pub struct DecodedFrame {
    /// Raw NV12 bytes (8-bit, 1 byte/sample) or P016 bytes (10/12-bit,
    /// 2 bytes/sample in the high bits). Deinterleave to planar at
    /// drain time — see NvdecDecoder::decode_next.
    pub nv12: Vec<u8>,
    pub width: u32,
    pub height: u32,
    /// The picture inside the `width` x `height` surface (odd where the
    /// stream's is): what the frame is cropped to on conversion.
    pub picture_width: u32,
    pub picture_height: u32,
    /// 0 = 8-bit (NV12 / Yuv420p), 2 = 10-bit (P016 / Yuv420p10le).
    /// Captured from the sequence_callback's CUVIDEOFORMAT so each
    /// frame carries its own format if the stream renegotiates.
    pub bit_depth_minus8: u8,
    /// ColorSpace derived from the SPS VUI matrix_coefficients, carried
    /// per-frame so the encoder / colorspace converter sees the
    /// source's primaries without re-parsing the SPS.
    pub color_space: ColorSpace,
    /// Timestamp as reported by the CUVID parser in display order.
    /// Preserved end-to-end so upstream frame-rate/duration math in
    /// the pipeline is correct rather than assuming integer frame
    /// counts from zero.
    pub timestamp: u64,
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
pub struct FrameCollector {
    pub frames: VecDeque<DecodedFrame>,
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
pub struct CallbackState {
    pub cuvid_create_decoder: FnCuvidCreateDecoder,
    pub cuvid_get_decoder_caps: FnCuvidGetDecoderCaps,
    pub cuvid_decode_picture: FnCuvidDecodePicture,
    pub cuvid_map_video_frame: FnCuvidMapVideoFrame,
    pub cuvid_unmap_video_frame: FnCuvidUnmapVideoFrame,
    pub cu_memcpy2d: FnCuMemcpy2D,

    pub decoder: Option<CUvideodecoder>,
    pub collector: Arc<Mutex<FrameCollector>>,
    pub width: u32,
    pub height: u32,
    /// The picture inside the output surface (`OutputGeometry::picture_*`).
    pub picture_width: u32,
    pub picture_height: u32,
    pub codec_type: c_int,
    /// Copied from CUVIDEOFORMAT.bit_depth_luma_minus8 in sequence_callback
    /// so display_callback knows whether to memcpy NV12 (1 byte/sample)
    /// or P016 (2 bytes/sample) and decode_next knows which PixelFormat
    /// to tag on the emitted VideoFrame.
    pub bit_depth_luma_minus8: u8,
    /// Derived from CUVIDEOFORMAT.video_signal_description in
    /// sequence_callback. Propagated to every DecodedFrame so downstream
    /// colorspace conversion knows the source's matrix_coefficients
    /// (BT.601/709/2020) without re-parsing the SPS.
    pub color_space: ColorSpace,
    /// Raw H.273 values captured from the SPS VUI so StreamInfo can
    /// round-trip them to the mux's `colr nclx` box. Populated in
    /// sequence_callback, read once after the parse loop finishes
    /// to update the outer NvdecDecoder.info.color_metadata.
    pub vui_colour_primaries: u8,
    pub vui_transfer_characteristics: u8,
    pub vui_matrix_coefficients: u8,
    pub vui_full_range_flag: bool,
    pub error: Option<String>,
    /// Typed reject reason captured in sequence_callback. Propagated up
    /// to `NvdecDecoder::new_with_pts`'s Err path so the caller can
    /// `.downcast_ref::<NvdecError>()` and steer fallback / abort
    /// policy (codec-review-2 HIGH-1). Only set when a `set_error`
    /// was caused by an NvdecError variant; plain driver failures
    /// continue through the string path.
    pub typed_error: Option<NvdecError>,
}

// SAFETY: The collector is Arc<Mutex<FrameCollector>> — the only piece
// of shared state — and all other fields are plain fn pointers + POD.
// Callbacks fire under the thread that calls cuvidParseVideoData; the
// Mutex serializes any cross-thread access from the drain-side code.
// The CUvideodecoder is only touched while its context is pushed on
// the current thread.
unsafe impl Send for CallbackState {}

impl CallbackState {
    pub fn set_error(&mut self, msg: impl Into<String>) {
        if self.error.is_none() {
            self.error = Some(msg.into());
        }
    }

    /// Record a structured reject reason *and* its string form so the
    /// existing first-wins `error` string path keeps its diagnostics
    /// while the outer caller can downcast the anyhow chain to pattern
    /// match on the cause. If a typed error was already latched,
    /// subsequent calls are ignored (first-wins).
    pub fn set_typed_error(&mut self, err: NvdecError) {
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
pub struct CtxScope {
    pub pop: FnCuCtxPopCurrent,
}

impl CtxScope {
    pub unsafe fn push(
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
