//! Live encode session + CUDA context RAII guard.

use anyhow::{bail, Result};
use std::ffi::c_void;
use std::os::raw::c_uint;
use std::ptr;

use super::constants::{
    CUcontext, FnCuCtxDestroy, FnCuCtxPopCurrent, FnCuCtxPushCurrent, FnNvEncDestroyBitstreamBuffer,
    FnNvEncDestroyEncoder, FnNvEncDestroyInputBuffer, FnNvEncEncodePicture, FnNvEncLockBitstream,
    FnNvEncLockInputBuffer, FnNvEncReconfigureEncoder, FnNvEncUnlockBitstream,
    FnNvEncUnlockInputBuffer, NV_ENC_RECONFIGURE_PARAMS_VER, NV_ENC_SUCCESS, RING_SIZE,
};
use super::ffi::{
    NvEncConfig, NvEncInitializeParams, NvEncReconfigureParams, RECONFIGURE_BIT_FORCE_IDR,
    RECONFIGURE_BIT_RESET_ENCODER,
};

/// Holds the live encode session + per-frame resources.
/// Dropped together so teardown order is enforced.
///
/// SAFETY: NVENC encoder handles and CUDA contexts are opaque pointers
/// accessed only from the thread that holds `Self`. The encoder's CUDA
/// context must be pushed current before any `fn_encode_picture` etc.
/// call — see `ctx_scope()`.
pub(super) struct EncodeSession {
    pub(super) encoder: *mut c_void,
    /// Ring of N input surfaces. Rotated per `EncodePicture` call.
    pub(super) input_buffers: [*mut c_void; RING_SIZE],
    /// Matching ring of N output (bitstream) buffers. Each input
    /// surface is paired 1:1 with an output surface so lock/unlock of
    /// bitstream i can proceed while input i+1 is being copied.
    pub(super) bitstream_buffers: [*mut c_void; RING_SIZE],
    pub(super) cuda_ctx: CUcontext,
    pub(super) width: u32,
    pub(super) height: u32,
    /// `NV_ENC_BUFFER_FORMAT_*` value chosen at session create time.
    /// Drives both the upload routine (8-bit byte copy vs 16-bit P010
    /// `<<6` shift) and the per-frame `NV_ENC_PIC_PARAMS.buffer_fmt`
    /// field — has to match `NV_ENC_INITIALIZE_PARAMS.buffer_format`
    /// or NVENC returns INVALID_PARAM on the first encode.
    pub(super) buffer_format: c_uint,

    // Function pointers captured up front. NVENC's fn-list table holds
    // opaque void* so we cast back at call time.
    pub(super) fn_destroy_input_buffer: FnNvEncDestroyInputBuffer,
    pub(super) fn_destroy_bitstream_buffer: FnNvEncDestroyBitstreamBuffer,
    pub(super) fn_lock_input_buffer: FnNvEncLockInputBuffer,
    pub(super) fn_unlock_input_buffer: FnNvEncUnlockInputBuffer,
    pub(super) fn_encode_picture: FnNvEncEncodePicture,
    pub(super) fn_lock_bitstream: FnNvEncLockBitstream,
    pub(super) fn_unlock_bitstream: FnNvEncUnlockBitstream,
    pub(super) fn_destroy_encoder: FnNvEncDestroyEncoder,

    pub(super) fn_cu_ctx_destroy: FnCuCtxDestroy,
    pub(super) fn_cu_ctx_push: FnCuCtxPushCurrent,
    pub(super) fn_cu_ctx_pop: FnCuCtxPopCurrent,

    /// `NvEncReconfigureEncoder`, when the driver's function list has it
    /// (every SDK this crate targets does; `None` only if a table came back
    /// with the slot empty, in which case `reset` is refused by type).
    pub(super) fn_reconfigure_encoder: Option<FnNvEncReconfigureEncoder>,
    /// The exact parameters `NvEncInitializeEncoder` was given, kept so a
    /// reset can hand the driver the *same* stream description again.
    /// Boxed: `init_params.encode_config` must point at `enc_config`, and a
    /// heap address survives the session being moved.
    pub(super) init_params: Box<NvEncInitializeParams>,
    pub(super) enc_config: Box<NvEncConfig>,
}

unsafe impl Send for EncodeSession {}

impl EncodeSession {
    /// Restart the driver-side stream in place: `NvEncReconfigureEncoder`
    /// with the parameters the session was initialised with, `resetEncoder`
    /// (rate control and lookahead state discarded) and `forceIDR` (the next
    /// picture opens a new GOP). Nothing about the stream description
    /// changes, which is what makes the call legal — the API refuses a
    /// reconfigure that alters PTD, bit depth or the maximum dimensions.
    ///
    /// The caller must have drained every outstanding picture first: the
    /// driver documents the call as valid only between frames.
    pub(super) unsafe fn reconfigure_reset(&mut self) -> Result<()> {
        let Some(reconfigure) = self.fn_reconfigure_encoder else {
            return Err(crate::encode::ResetUnsupported.into());
        };
        unsafe {
            let _scope = self.ctx_scope()?;
            let mut params: NvEncReconfigureParams = std::mem::zeroed();
            params.version = NV_ENC_RECONFIGURE_PARAMS_VER;
            // A bytewise copy of the init params; the config pointer is
            // refreshed rather than trusted, in case the box was ever moved.
            ptr::copy_nonoverlapping(
                &*self.init_params as *const NvEncInitializeParams,
                &mut params.re_init_encode_params as *mut NvEncInitializeParams,
                1,
            );
            params.re_init_encode_params.encode_config =
                &mut *self.enc_config as *mut NvEncConfig as *mut c_void;
            params.flags = RECONFIGURE_BIT_RESET_ENCODER | RECONFIGURE_BIT_FORCE_IDR;
            let rc = reconfigure(self.encoder, &mut params);
            if rc != NV_ENC_SUCCESS {
                bail!("NvEncReconfigureEncoder (resetEncoder=1, forceIDR=1) failed: {rc}");
            }
            // The driver may have rewritten fields of its copy; ours is the
            // one that worked, so keep it as is for the next reset.
            Ok(())
        }
    }

    /// Push this session's CUDA context on the calling thread for the
    /// duration of the returned guard. Required because tokio workers
    /// may migrate between OS threads — without an explicit push the
    /// encoder calls hit CUDA_ERROR_INVALID_CONTEXT.
    pub(super) unsafe fn ctx_scope(&self) -> Result<CtxScope> {
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
pub(super) struct CtxScope {
    pop: FnCuCtxPopCurrent,
}

impl CtxScope {
    pub(super) unsafe fn push(
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
