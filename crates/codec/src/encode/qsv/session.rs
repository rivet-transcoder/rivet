//! `QsvSession` — owns the live oneVPL session handle, all function pointers,
//! surface ring, ext-buffer backing, and the output bitstream buffer.
//!
//! On `Drop` the session calls `MFXVideoENCODE_Close`, then `MFXClose`,
//! then `MFXUnload` — in that order, matching the teardown sequence
//! specified in the oneVPL API reference.

use std::collections::VecDeque;

use crate::frame::PixelFormat;
use crate::qsv_ffi::{MfxBitstream, MfxExtBuffer};

use super::ffi::{
    FnEncodeClose, FnEncodeFrameAsync, FnMfxClose, FnMfxUnload, FnSyncOperation, MfxLoader,
    MfxSession,
};
use super::ffi::{
    MfxExtAv1TileParam, MfxExtCodingOption3, MfxExtVideoSignalInfo,
};
use super::surface::{POOL_SIZE, SurfaceSlot};

/// All state that outlives the constructor and must be accessed from
/// `encode_one` / `flush_drain` / `sync_and_drain`.
pub(super) struct QsvSession {
    pub(super) session: MfxSession,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) pts_timescale: u64,
    /// `Yuv420p` (NV12 surface) or `Yuv420p10le` (P010 surface).
    /// Drives the per-frame upload (8-bit byte copy vs P010 `<<6`).
    pub(super) input_pixel_format: PixelFormat,

    pub(super) fn_mfx_close: FnMfxClose,
    pub(super) fn_encode_close: FnEncodeClose,
    pub(super) fn_encode_frame_async: FnEncodeFrameAsync,
    pub(super) fn_sync_operation: FnSyncOperation,
    /// oneVPL dispatcher loader — kept alive for the session's lifetime,
    /// `MFXUnload`'d after `MFXClose` in Drop.
    pub(super) loader: MfxLoader,
    pub(super) fn_unload: FnMfxUnload,

    /// Backing storage for the ext buffers we attached to `mfxVideoParam`.
    /// Must stay alive as long as the encoder session references the
    /// `ExtParam[]` pointer via its internal copy — oneVPL docs say the
    /// runtime shallow-copies `ExtParam` at `Init`, so we could drop these
    /// after Init, but we keep them for any future `Reconfigure`.
    #[allow(dead_code)]
    pub(super) tile_ext: Box<MfxExtAv1TileParam>,
    /// `mfxExtCodingOption3` — present whenever the input is 10-bit so
    /// `TargetBitDepthLuma=10` makes it into the AV1 sequence header.
    /// `None` for 8-bit; absent in 8-bit jobs to avoid sending the
    /// runtime extra zero-filled buffers it has to inspect.
    #[allow(dead_code)]
    pub(super) coding_option3_ext: Option<Box<MfxExtCodingOption3>>,
    /// `mfxExtVideoSignalInfo` — H.273 colour signalling.  Always
    /// attached so SDR jobs explicitly carry BT.709 (rather than
    /// "unspecified") in the OBU header.
    #[allow(dead_code)]
    pub(super) signal_info_ext: Box<MfxExtVideoSignalInfo>,
    /// Vector of pointers backing `mfxVideoParam.ExtParam[]`.  Length
    /// varies (2 for 8-bit: tile + signal_info; 3 for 10-bit:
    /// + coding_option3).  Kept boxed so the address handed to oneVPL
    /// stays stable across the session lifetime.
    #[allow(dead_code)]
    pub(super) ext_param_array: Vec<*mut MfxExtBuffer>,

    /// Pool of input surfaces.  The producer writes into whichever slot the
    /// runtime has released — no sync point outstanding and `Data.Locked == 0`
    /// — rather than taking them in turn; the consumer drains the
    /// oldest-submitted slot's sync point FIFO-style via `inflight`.
    ///
    /// Choosing by turn is what let a frame be written over a surface the
    /// runtime was still holding, so slot choice is now a search, not an index.
    pub(super) surfaces: [SurfaceSlot; POOL_SIZE],
    /// FIFO of slot indices whose sync point is still pending
    /// a `SyncOperation`.  Length is bounded by `POOL_SIZE`; we drain
    /// the head to free a slot when none is available.
    pub(super) inflight: VecDeque<usize>,
    pub(super) input_pitch: u32,
    pub(super) height_aligned: u32,

    /// Output bitstream buffer — pre-allocated with enough headroom
    /// for a 4K I-frame (~2 MB).  Shared across all in-flight frames
    /// because `SyncOperation` consumes the buffer between frames;
    /// oneVPL documents this usage pattern in `sample_encode`.
    pub(super) bitstream: MfxBitstream,
    /// Owns the backing bytes that `bitstream.data` points into.
    /// `Box<[u8]>` (not `Vec<u8>`) so the allocation can never be
    /// mutated-and-reallocated after construction — the driver holds
    /// a pointer into the allocation across encode frames.
    pub(super) _bitstream_buf: Box<[u8]>,
}

// SAFETY: `QsvSession` holds raw pointers (`session: MfxSession`,
// fn pointers from the dispatcher, and NV12 / bitstream buffers owned
// by sibling `Box` fields).  oneVPL is NOT thread-*safe* — concurrent
// calls to `MFXVideoENCODE_*` on the same session from different threads
// are UB — but `Send` only guarantees single-threaded *ownership transfer*.
// Each `QsvEncoder` is only touched from one tokio task at a time; moving
// the encoder between threads (e.g. when a `spawn_blocking` worker returns)
// is fine because the runtime serialises access to the underlying session
// through `&mut self`.
unsafe impl Send for QsvSession {}

impl Drop for QsvSession {
    fn drop(&mut self) {
        unsafe {
            if !self.session.is_null() {
                let _ = (self.fn_encode_close)(self.session);
                let _ = (self.fn_mfx_close)(self.session);
            }
            if !self.loader.is_null() {
                (self.fn_unload)(self.loader);
            }
        }
    }
}
