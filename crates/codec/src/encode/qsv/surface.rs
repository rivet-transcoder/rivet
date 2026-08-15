//! Input surface pool for QSV encode.
//!
//! A single `SurfaceSlot` pairs an `MfxFrameSurface1` with its backing
//! allocation and the last sync point produced by `EncodeFrameAsync` on
//! that slot.

use crate::qsv_ffi::MfxFrameSurface1;
use super::ffi::MfxSyncPoint;

/// What we advertise to the runtime as `mfxVideoParam.AsyncDepth` — how many
/// submissions it may have in flight at once.  Upstream oneVPL
/// `sample_encode` recommends 4 on Arc / Meteor Lake.
pub(super) const ASYNC_DEPTH: usize = 4;

/// How many input surfaces we own.
///
/// This must be **larger** than `ASYNC_DEPTH`, and it is not the same number.
/// The runtime holds a surface for as long as it needs it — through the async
/// queue and for as long as the frame is a prediction reference — and signals
/// that by keeping `mfxFrameData.Locked` above zero. A pool sized to the async
/// depth alone leaves no surface free to write the next frame into.
///
/// This used to be a 4-slot ring indexed round-robin, which meant frame N and
/// frame N+4 shared a surface. On a fresh encoder, `EncodeFrameAsync` answers
/// `MFX_ERR_MORE_DATA` for the first few frames: the runtime has taken the
/// surface but produced no sync point. The reuse guard only waited on slots
/// that had a sync point, so those slots looked free — and frame 4 overwrote
/// frame 0's surface while the runtime was still holding it. The picture the
/// encoder eventually emitted for frame 0 carried frame 4's pixels.
///
/// Because a fresh encoder is built per CMAF segment, that happened at every
/// segment: the first frame of each showed the frame four later, then snapped
/// back. Slots are now chosen by looking for one the runtime has released, so
/// the size is headroom rather than a correctness argument.
pub(super) const POOL_SIZE: usize = 16;

/// A single input-surface slot in the 4-deep ring.  Holds the
/// `MfxFrameSurface1` plus the backing NV12/P010 buffer that surface's
/// pointers live in.
pub(super) struct SurfaceSlot {
    pub(super) surface: MfxFrameSurface1,
    /// Owns the bytes that `surface.data.{mem_id_or_y, u, v}` point
    /// into.  Storage MUST NOT be dropped until the session closes —
    /// the driver may still hold back-references even after we sync.
    /// `Box<[u8]>` (not `Vec<u8>`) so the allocation can never be
    /// mutated-and-reallocated after construction.
    pub(super) _backing: Box<[u8]>,
    /// `sync_point` from the most recent `EncodeFrameAsync` on this
    /// slot, or null if the slot has never been submitted or has
    /// already been synced.
    pub(super) sync: MfxSyncPoint,
    /// This slot's **own** output bitstream, and the buffer behind it.
    ///
    /// oneVPL's contract is one `mfxBitstream` per in-flight submission. The
    /// ring shared a single one, so under sustained pressure two frames could
    /// land in it before either was synced: the first sync then drained both
    /// as one packet and the second saw `data_length == 0` and produced none.
    /// The video survived (the access units were still there, concatenated)
    /// but the packet count no longer matched the frame count, which the MP4
    /// sample table is built from. Cost 2 frames in 480 on a long chunk.
    pub(super) bitstream: crate::qsv_ffi::MfxBitstream,
    /// Owns the bytes `bitstream.data` points into. Same rule as `_backing`.
    pub(super) bitstream_buf: Box<[u8]>,
    /// Per-frame encode control for this slot's in-flight submission.
    ///
    /// Lives here, not on the caller's stack, for the same reason
    /// `_backing` does: `EncodeFrameAsync` is **asynchronous**, so the
    /// runtime reads the control after the call returns. A stack local
    /// would be gone by then — which is precisely why the first attempt at
    /// forcing an IDR did nothing: the request was written to memory that
    /// had already been reclaimed.
    pub(super) ctrl: super::ffi::MfxEncodeCtrl,
}

// SAFETY: `MfxSyncPoint = *mut c_void` is a raw pointer, not
// auto-`Send`, but oneVPL documents sync points as thread-safe
// handles that are opaque from our perspective.  The ring only
// migrates between threads when the whole `QsvSession` migrates
// (via `spawn_blocking`), and access is serialized through `&mut
// self`.  No sharing; same Send constraint as `QsvSession`.
unsafe impl Send for SurfaceSlot {}
