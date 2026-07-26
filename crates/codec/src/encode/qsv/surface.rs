//! Ring-buffer input surface pool for QSV encode.
//!
//! A single `SurfaceSlot` pairs an `MfxFrameSurface1` with its backing
//! allocation and the last sync point produced by `EncodeFrameAsync` on
//! that slot.  `RING_SIZE = 4` matches upstream `sample_encode`'s
//! recommended `AsyncDepth = 4` on Arc / Meteor Lake.

use crate::qsv_ffi::MfxFrameSurface1;
use super::ffi::MfxSyncPoint;

/// Encoder pipeline depth — number of input surfaces + sync points
/// in flight before we must drain one.  Matches NVENC's `RING_SIZE = 4`
/// and upstream oneVPL `sample_encode`'s recommended `AsyncDepth = 4`
/// on Arc / Meteor Lake.
pub(super) const RING_SIZE: usize = 4;

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
