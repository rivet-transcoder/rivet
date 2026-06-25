//! Process-wide CUDA driver init mutex.
//!
//! 2026-05-01 prod root cause refinement: when 5 NVENC encoder
//! constructions ran in parallel, the NVIDIA driver segfaulted
//! inside `NvEncOpenEncodeSessionEx`. Adding a mutex to NVENC
//! construction reduced 5-way parallelism to 1, but the FIRST
//! encoder still segfaulted — because NVDEC's `NvdecStreamingDecoder`
//! construction was running in PARALLEL on a sibling thread, doing
//! its OWN `cuInit` + `cuCtxCreate` + cuvid parser create. The
//! NVIDIA driver's session table can't handle simultaneous CUDA
//! context creation from different code paths on the same GPU,
//! even when each path's logic is single-threaded.
//!
//! Captured in /ecs/transcoder-production at 10:12:43.159..220
//! (PT 03:12:43): NVENC starts cuCtxCreate, NVDEC engages 2ms
//! later, both finish their CUDA setup ~60ms later, NVENC's
//! NvEncOpenEncodeSessionEx fires, FATAL SIGSEGV.
//!
//! This mutex serializes the brief CUDA-init + first-FFI-call
//! window across BOTH NVENC and NVDEC. Once each backend has its
//! context + decoder/encoder handle, it releases the lock and
//! per-frame work runs concurrently as before. Cold-start latency
//! adds ~50–200 ms total per pipeline run; FRAME throughput is
//! unchanged.
//!
//! Lock poisoning is treated as recoverable: the only invariant
//! we protect is "no two CUDA inits happening at the same time",
//! and a panic during a previous init carries no state we'd
//! corrupt by re-entering.

use std::sync::Mutex;

/// Global mutex serializing CUDA-driver init across NVENC + NVDEC.
/// Acquire at the START of any code path that calls cuInit /
/// cuCtxCreate / cuvidCreateDecoder / NvEncOpenEncodeSessionEx /
/// NvEncInitializeEncoder, hold until the construction window
/// closes (caller stores the GPU handle and is ready for parallel
/// per-frame work).
pub(crate) static CUDA_INIT_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the lock, treating poisoning as recoverable. See module
/// docstring — the lock protects no in-memory invariant we'd corrupt.
pub(crate) fn lock_for_cuda_init() -> std::sync::MutexGuard<'static, ()> {
    CUDA_INIT_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
