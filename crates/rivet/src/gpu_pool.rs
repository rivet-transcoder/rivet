//! Process-wide GPU reservation pool.
//!
//! Each detected GPU is a slot. Callers `claim()` an available slot
//! and hold the returned `GpuLease` for the duration of their work;
//! `Drop` releases the slot back to the pool. The lease's
//! `gpu_index` field is the device index the work should run on.
//!
//! Concurrency model: one variant per GPU at any time. With N GPUs
//! and M waiters, the first N waiters get leases immediately and the
//! remaining M−N park on the semaphore until a lease drops. This is
//! the deliberate design decision from 2026-05-02 — concurrent
//! NVENC sessions on the same CUDA context deadlocked at session
//! ~5/5 init, GPU went idle, no frames encoded. One-encoder-per-GPU
//! is the load-bearing invariant; the pool's role is to enforce it
//! while still letting variants run in parallel ACROSS GPUs.
//!
//! CPU-only hosts (no GPUs detected): `claim()` returns `None`
//! immediately — callers fall back to CPU encode without queuing.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use codec::gpu::{GpuDevice, GpuVendor};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

pub struct GpuPool {
    /// Per-slot GPU device index (`GpuDevice.index`, not vec position
    /// — accommodates sparse `CUDA_VISIBLE_DEVICES` setups).
    gpu_indices: Vec<u32>,
    /// Per-slot vendor — load-bearing for the encoder factory's
    /// vendor-aware dispatch. Without it, multi-vendor hosts (NVIDIA
    /// + Intel Arc) ALWAYS picked NVENC because the factory tries
    /// NVIDIA first and both vendors expose index 0; the Arc sat
    /// idle even when the NVIDIA card was busy.
    gpu_vendors: Vec<GpuVendor>,
    /// Per-slot human-readable device name. Used by `snapshot_leases`
    /// (Phase 2 worker_load reporting) so the backend's admin view
    /// can label each GPU lease badge with the same string the hello
    /// frame already advertised. Stays in lockstep with the hello
    /// frame's `WsGpuInfo.name`.
    gpu_names: Vec<String>,
    /// Per-slot free flag. `true` = available; `false` = leased.
    /// Atomic so the CAS-find-free-slot path under `claim()` is
    /// lock-free; correctness is enforced by the semaphore counting
    /// (see `claim`).
    free: Arc<Vec<AtomicBool>>,
    /// Semaphore with N permits (= number of GPUs). Acquiring a
    /// permit guarantees at least one `free` slot exists, so the
    /// CAS loop in `claim()` always succeeds without retry.
    permits: Arc<Semaphore>,
    /// Count of variant tasks currently blocked inside `claim()`'s
    /// `acquire_owned().await`. Used by the LeaseArbiter (planned
    /// 2026-05-10) to decide whether to dispatch a helper task: if
    /// any variant is already waiting for a permit, that variant
    /// must claim before the arbiter steals a permit for a helper.
    /// Incremented immediately before `acquire_owned().await`;
    /// decremented as soon as the await returns (success or
    /// cancellation) via the `PendingClaimGuard` RAII helper.
    ///
    /// `try_claim()` does NOT touch this — helpers are not blocked
    /// claimers in the spare-capacity sense.
    pending_claimers: Arc<AtomicUsize>,
}

/// RAII guard that increments `pending_claimers` on construction
/// and decrements on drop. Used inside `claim()` to bracket the
/// `acquire_owned().await` so the counter stays accurate even when
/// the awaiting task is cancelled mid-await (the future is dropped,
/// guard drops, counter decrements).
struct PendingClaimGuard {
    counter: Arc<AtomicUsize>,
}

impl PendingClaimGuard {
    fn new(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self { counter }
    }
}

impl Drop for PendingClaimGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Snapshot of one GPU slot's lease state at a moment in time.
/// Returned by [`GpuPool::snapshot_leases`] for Phase 2 worker_load
/// reporting. Field shape matches `queue::WsGpuLeaseEntry` so the
/// caller can map across without a wire-format-aware translation.
#[derive(Debug, Clone)]
pub struct GpuLeaseEntry {
    pub vendor: GpuVendor,
    pub name: String,
    pub index: u32,
    pub leased: bool,
}

/// RAII guard returned by `GpuPool::claim`. The slot is released
/// (and the underlying semaphore permit dropped) when this value
/// is dropped — typically at the end of the variant's encode task.
pub struct GpuLease {
    pub gpu_index: u32,
    pub vendor: GpuVendor,
    slot_idx: usize,
    free: Arc<Vec<AtomicBool>>,
    _permit: OwnedSemaphorePermit,
}

impl Drop for GpuLease {
    fn drop(&mut self) {
        self.free[self.slot_idx].store(true, Ordering::Release);
    }
}

impl GpuPool {
    /// Build a pool from the host's detected GPU inventory. An empty
    /// inventory is permitted — the resulting pool always returns
    /// `None` from `claim()` so CPU-only hosts work without
    /// special-casing at the call site.
    pub fn new(devices: &[GpuDevice]) -> Self {
        let n = devices.len();
        Self {
            gpu_indices: devices.iter().map(|d| d.index).collect(),
            gpu_vendors: devices.iter().map(|d| d.vendor).collect(),
            gpu_names: devices.iter().map(|d| d.name.clone()).collect(),
            free: Arc::new((0..n).map(|_| AtomicBool::new(true)).collect()),
            // Semaphore::new(0) is valid but `acquire` would deadlock.
            // We never acquire on the empty path because `claim()`
            // returns `None` early on CPU-only hosts.
            permits: Arc::new(Semaphore::new(n)),
            pending_claimers: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// How many variant tasks are currently parked inside `claim()`
    /// waiting for a permit. The LeaseArbiter consults this to decide
    /// whether to dispatch a helper: when `pending_claimers() > 0`,
    /// at least one variant task wants a GPU and the arbiter must
    /// step back so the variant claims first (FIFO fairness).
    ///
    /// Reads with `Ordering::Acquire`. The result is momentary — by
    /// the time the caller observes it, a claim may have resolved or
    /// a new one parked. That's expected; the arbiter re-checks
    /// before each dispatch decision.
    pub fn pending_claimers(&self) -> usize {
        self.pending_claimers.load(Ordering::Acquire)
    }

    /// How many GPUs this pool manages. Useful for pre-spawning
    /// variants when fewer variants exist than GPUs (no point
    /// over-claiming).
    pub fn capacity(&self) -> usize {
        self.gpu_indices.len()
    }

    /// Snapshot per-GPU lease state. Result preserves slot order
    /// (matches the order [`GpuPool::new`] saw devices), so callers
    /// stitching the result against the hello frame's `gpu_pool`
    /// see consistent indices across both reports.
    ///
    /// Reads `free` slots with `Ordering::Acquire`. The result is a
    /// momentary snapshot — by the time the caller observes it, a
    /// claim or drop may have flipped any slot. That's expected;
    /// load reporting is best-effort observability, not a
    /// transactional view.
    ///
    /// Used by the worker's Phase 2 (2026-05-07) load-tick task to
    /// build the `worker_load` frame's `gpu_pool` field.
    pub fn snapshot_leases(&self) -> Vec<GpuLeaseEntry> {
        self.gpu_indices
            .iter()
            .zip(self.gpu_vendors.iter())
            .zip(self.gpu_names.iter())
            .enumerate()
            .map(|(slot_idx, ((index, vendor), name))| GpuLeaseEntry {
                vendor: *vendor,
                name: name.clone(),
                index: *index,
                leased: !self.free[slot_idx].load(Ordering::Acquire),
            })
            .collect()
    }

    /// Claim an available GPU. Awaits if every GPU is currently
    /// leased. Returns `None` immediately on CPU-only hosts — the
    /// caller should fall back to CPU encode.
    pub async fn claim(self: &Arc<Self>) -> Option<GpuLease> {
        if self.gpu_indices.is_empty() {
            return None;
        }
        // Track "blocked waiting for a permit" for the LeaseArbiter's
        // fairness check. Guard is scoped to the await: on success the
        // guard drops at end-of-block (decrement); on cancellation the
        // future is dropped mid-await, the guard drops, and the
        // counter still decrements. Either way the count stays
        // accurate.
        let permit = {
            let _pending = PendingClaimGuard::new(Arc::clone(&self.pending_claimers));
            Arc::clone(&self.permits)
                .acquire_owned()
                .await
                .expect("GpuPool semaphore should never be closed")
        };
        // The permit guarantees ≥1 free slot. None here means the
        // semaphore count and free-flag count drifted apart — a bug
        // (RAII Drop bypassed, wrong atomic ordering, etc.).
        match self.assign_free_slot(permit) {
            Some(lease) => Some(lease),
            None => unreachable!(
                "GpuPool: permit acquired but no free slot found — \
                 semaphore count and free-flag count drifted apart"
            ),
        }
    }

    /// Try to claim a GPU without blocking. Returns `None` if every
    /// GPU is currently leased OR if the host has no GPUs.
    ///
    /// Used by the LeaseArbiter (planned 2026-05-10) to grab a helper
    /// lease without contending with blocked variant tasks. Tokio's
    /// Semaphore preserves FIFO ordering for queued waiters — a
    /// permit released while a variant task is parked in
    /// `acquire_owned().await` is reserved for that waiter and is NOT
    /// visible to `try_acquire_owned()`, so this method cannot steal
    /// a permit out from under a queued variant.
    ///
    /// Does NOT increment `pending_claimers`; helpers are not blocked
    /// claimers in the spare-capacity sense.
    pub fn try_claim(self: &Arc<Self>) -> Option<GpuLease> {
        if self.gpu_indices.is_empty() {
            return None;
        }
        let permit = Arc::clone(&self.permits).try_acquire_owned().ok()?;
        Some(self.assign_free_slot(permit).expect(
            "GpuPool: try_acquire_owned succeeded but no free slot found — \
             this would mean semaphore and free-flag counts drifted apart",
        ))
    }

    /// Permit → lease conversion shared by `claim()` and `try_claim()`.
    /// The permit guarantees ≥1 free slot exists; the CAS loop finds
    /// the first slot we win the race for. With N ≤ 16 GPUs in
    /// realistic deployments the linear scan is faster than any
    /// index-tracking scheme.
    ///
    /// Returns `Some(lease)` on the only correct path. Returns `None`
    /// only if the semaphore and free-flag counts drifted apart,
    /// which the pool's invariants forbid (`claim` panics via
    /// `unreachable!` in that case; `try_claim` propagates as a
    /// distinguishable "this would never happen" via `expect`).
    fn assign_free_slot(&self, permit: OwnedSemaphorePermit) -> Option<GpuLease> {
        for (slot_idx, slot) in self.free.iter().enumerate() {
            if slot
                .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Some(GpuLease {
                    gpu_index: self.gpu_indices[slot_idx],
                    vendor: self.gpu_vendors[slot_idx],
                    slot_idx,
                    free: Arc::clone(&self.free),
                    _permit: permit,
                });
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codec::gpu::GpuVendor;

    fn synth(index: u32) -> GpuDevice {
        GpuDevice {
            index,
            vendor_index: index,
            vendor: GpuVendor::Nvidia,
            name: format!("synth-{index}"),
            generation: "Synth".into(),
            pci_id: String::new(),
            vram_mib: 0,
            serial: None,
            host_pci_address: String::new(),
            vendor_id_hex: String::new(),
        }
    }

    fn synth_intel(index: u32) -> GpuDevice {
        GpuDevice {
            index,
            vendor_index: index,
            vendor: GpuVendor::Intel,
            name: format!("intel-{index}"),
            generation: "Synth".into(),
            pci_id: String::new(),
            vram_mib: 0,
            serial: None,
            host_pci_address: String::new(),
            vendor_id_hex: String::new(),
        }
    }

    #[tokio::test]
    async fn empty_pool_returns_none() {
        let pool = Arc::new(GpuPool::new(&[]));
        assert!(pool.claim().await.is_none());
        assert_eq!(pool.capacity(), 0);
    }

    #[tokio::test]
    async fn single_gpu_serializes_claims() {
        let pool = Arc::new(GpuPool::new(&[synth(0)]));
        let lease1 = pool.claim().await.unwrap();
        assert_eq!(lease1.gpu_index, 0);

        // Second claim must wait — race it against a short timeout to
        // assert it does NOT resolve while lease1 is held.
        let pool_clone = Arc::clone(&pool);
        let claim2 = tokio::spawn(async move { pool_clone.claim().await.unwrap() });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(
            !claim2.is_finished(),
            "second claim resolved while lease held"
        );

        drop(lease1);
        let lease2 = claim2.await.unwrap();
        assert_eq!(lease2.gpu_index, 0);
    }

    #[tokio::test]
    async fn two_gpus_concurrent_leases_distinct_indices() {
        let pool = Arc::new(GpuPool::new(&[synth(0), synth(1)]));
        let lease_a = pool.claim().await.unwrap();
        let lease_b = pool.claim().await.unwrap();
        assert_ne!(lease_a.gpu_index, lease_b.gpu_index);
    }

    #[tokio::test]
    async fn third_claim_waits_until_one_drops() {
        let pool = Arc::new(GpuPool::new(&[synth(0), synth(1)]));
        let lease_a = pool.claim().await.unwrap();
        let _lease_b = pool.claim().await.unwrap();

        let pool_clone = Arc::clone(&pool);
        let claim_c = tokio::spawn(async move { pool_clone.claim().await.unwrap() });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(!claim_c.is_finished());

        let dropped_idx = lease_a.gpu_index;
        drop(lease_a);

        let lease_c = claim_c.await.unwrap();
        assert_eq!(lease_c.gpu_index, dropped_idx);
    }

    #[tokio::test]
    async fn two_intel_arc_cards_both_get_intel_leases() {
        // 2× Arc, 0× NVIDIA. Each card has its own per-vendor index.
        // Both leases come back vendor=Intel and the indices are
        // distinct so the encoder factory's pick_vendor_device(Intel,
        // Some(0/1)) finds the right physical card per lease.
        let pool = Arc::new(GpuPool::new(&[synth_intel(0), synth_intel(1)]));
        let l1 = pool.claim().await.unwrap();
        let l2 = pool.claim().await.unwrap();
        assert_eq!(l1.vendor, GpuVendor::Intel);
        assert_eq!(l2.vendor, GpuVendor::Intel);
        let mut indices: Vec<u32> = vec![l1.gpu_index, l2.gpu_index];
        indices.sort();
        assert_eq!(indices, vec![0, 1]);
    }

    #[tokio::test]
    async fn two_nvidia_cards_both_get_nvidia_leases() {
        // 2× NVIDIA, 0× Arc. Same shape as the Intel-Intel case.
        let pool = Arc::new(GpuPool::new(&[synth(0), synth(1)]));
        let l1 = pool.claim().await.unwrap();
        let l2 = pool.claim().await.unwrap();
        assert_eq!(l1.vendor, GpuVendor::Nvidia);
        assert_eq!(l2.vendor, GpuVendor::Nvidia);
        let mut indices: Vec<u32> = vec![l1.gpu_index, l2.gpu_index];
        indices.sort();
        assert_eq!(indices, vec![0, 1]);
    }

    #[tokio::test]
    async fn lease_carries_vendor_for_dispatch() {
        // Multi-vendor host: NVIDIA at index 0 + Intel at index 0.
        // Without vendor on the lease, the encoder factory's NVIDIA-
        // first dispatch would have always picked NVENC. With vendor,
        // each lease tells the factory which backend to use.
        let pool = Arc::new(GpuPool::new(&[synth(0), synth_intel(0)]));
        let l1 = pool.claim().await.unwrap();
        let l2 = pool.claim().await.unwrap();
        let mut vendors: Vec<GpuVendor> = vec![l1.vendor, l2.vendor];
        // Order is non-deterministic between the two slots; both
        // vendors must appear exactly once.
        vendors.sort_by_key(|v| match v {
            GpuVendor::Nvidia => 0,
            GpuVendor::Amd => 1,
            GpuVendor::Intel => 2,
        });
        assert_eq!(vendors, vec![GpuVendor::Nvidia, GpuVendor::Intel]);
    }

    #[tokio::test]
    async fn snapshot_leases_reflects_current_state() {
        // Phase 2 contract: snapshot returns one entry per slot in
        // construction order; `leased` mirrors the live free-flag.
        let pool = Arc::new(GpuPool::new(&[synth(0), synth_intel(1)]));

        let snap0 = pool.snapshot_leases();
        assert_eq!(snap0.len(), 2);
        assert_eq!(snap0[0].index, 0);
        assert_eq!(snap0[0].vendor, GpuVendor::Nvidia);
        assert!(!snap0[0].leased);
        assert_eq!(snap0[1].index, 1);
        assert_eq!(snap0[1].vendor, GpuVendor::Intel);
        assert!(!snap0[1].leased);

        // Claim the NVIDIA slot → snapshot reflects it.
        let lease = pool.claim().await.unwrap();
        // Order in which slots get claimed isn't strictly tied to
        // vec position, but with N=2 and one outstanding lease the
        // snapshot must show exactly one `leased=true`.
        let snap1 = pool.snapshot_leases();
        let leased_count = snap1.iter().filter(|e| e.leased).count();
        assert_eq!(leased_count, 1);

        drop(lease);
        let snap2 = pool.snapshot_leases();
        assert!(snap2.iter().all(|e| !e.leased));
    }

    #[tokio::test]
    async fn snapshot_leases_empty_for_cpu_host() {
        let pool = Arc::new(GpuPool::new(&[]));
        let snap = pool.snapshot_leases();
        assert!(snap.is_empty());
    }

    #[tokio::test]
    async fn snapshot_leases_carries_device_name() {
        // The Phase 2 load-tick task reads .name straight off the
        // snapshot to build the worker_load frame's gpu_pool entry,
        // so the lookup must hit the real GpuDevice.name (not the
        // stringified vendor).
        let pool = Arc::new(GpuPool::new(&[synth(0)]));
        let snap = pool.snapshot_leases();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].name, "synth-0");
    }

    // ---- pending_claimers + try_claim (2026-05-10) ----

    #[tokio::test]
    async fn pending_claimers_starts_at_zero() {
        let pool = Arc::new(GpuPool::new(&[synth(0), synth(1)]));
        assert_eq!(pool.pending_claimers(), 0);
    }

    #[tokio::test]
    async fn pending_claimers_zero_after_unblocked_claim() {
        // Single GPU, single immediate claim — never blocks, so the
        // count should observe 0 both before AND after the claim.
        let pool = Arc::new(GpuPool::new(&[synth(0)]));
        assert_eq!(pool.pending_claimers(), 0);
        let _lease = pool.claim().await.unwrap();
        assert_eq!(pool.pending_claimers(), 0);
    }

    #[tokio::test]
    async fn pending_claimers_increments_during_blocked_claim() {
        // 1 GPU, take it; spawn a second claim → that task parks in
        // `acquire_owned().await`; pending_claimers should observe 1.
        let pool = Arc::new(GpuPool::new(&[synth(0)]));
        let lease1 = pool.claim().await.unwrap();
        assert_eq!(pool.pending_claimers(), 0);

        let pool_clone = Arc::clone(&pool);
        let claim2 = tokio::spawn(async move { pool_clone.claim().await.unwrap() });

        // Give the spawned task a moment to enter the await.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(
            pool.pending_claimers(),
            1,
            "blocked claimer should be counted",
        );

        // Release the lease → blocked claimer resumes.
        drop(lease1);
        let _lease2 = claim2.await.unwrap();
        assert_eq!(
            pool.pending_claimers(),
            0,
            "after resume, blocked count returns to 0",
        );
    }

    #[tokio::test]
    async fn pending_claimers_increments_for_multiple_blockers() {
        // 1 GPU, 3 concurrent claimers (1 immediate, 2 blocked) →
        // pending observes 2 while both are parked.
        let pool = Arc::new(GpuPool::new(&[synth(0)]));
        let lease1 = pool.claim().await.unwrap();

        let pool_a = Arc::clone(&pool);
        let _a = tokio::spawn(async move { pool_a.claim().await.unwrap() });
        let pool_b = Arc::clone(&pool);
        let _b = tokio::spawn(async move { pool_b.claim().await.unwrap() });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(pool.pending_claimers(), 2);

        drop(lease1);
        // First waiter resumes; second still parked → count goes 2→1.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(pool.pending_claimers(), 1);
    }

    #[tokio::test]
    async fn pending_claimers_decrements_under_cancellation() {
        // Park a claim, then abort the task before the await
        // resolves. The PendingClaimGuard's Drop must still run and
        // bring the count back to 0.
        let pool = Arc::new(GpuPool::new(&[synth(0)]));
        let _lease1 = pool.claim().await.unwrap();

        let pool_clone = Arc::clone(&pool);
        let task = tokio::spawn(async move { pool_clone.claim().await });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(pool.pending_claimers(), 1);

        task.abort();
        // Abort drops the future, which drops the PendingClaimGuard
        // inside the await scope. Allow a scheduler tick to observe.
        let _ = task.await; // resolves with JoinError(Cancelled)
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert_eq!(
            pool.pending_claimers(),
            0,
            "cancelled claim must still decrement pending_claimers",
        );
    }

    #[tokio::test]
    async fn try_claim_returns_none_when_pool_full() {
        // All permits taken → try_claim is None.
        let pool = Arc::new(GpuPool::new(&[synth(0)]));
        let _lease = pool.claim().await.unwrap();
        assert!(pool.try_claim().is_none());
    }

    #[tokio::test]
    async fn try_claim_returns_lease_when_capacity_available() {
        let pool = Arc::new(GpuPool::new(&[synth(0), synth(1)]));
        let lease1 = pool.try_claim().unwrap();
        let lease2 = pool.try_claim().unwrap();
        assert_ne!(lease1.gpu_index, lease2.gpu_index);
        assert!(
            pool.try_claim().is_none(),
            "after both GPUs leased, third try_claim must be None",
        );
    }

    #[tokio::test]
    async fn try_claim_returns_none_on_cpu_only_host() {
        let pool = Arc::new(GpuPool::new(&[]));
        assert!(pool.try_claim().is_none());
    }

    #[tokio::test]
    async fn try_claim_does_not_steal_from_blocked_claimer() {
        // The contract the LeaseArbiter relies on: when a variant
        // task is parked in `claim()`'s `acquire_owned().await` and a
        // permit becomes available, that permit goes to the parked
        // variant FIRST. A racing `try_claim()` must return None.
        //
        // Tokio's Semaphore is documented as FIFO for `acquire_owned`;
        // released permits are reserved for queued waiters and are
        // NOT visible to `try_acquire_owned()`. This test guards
        // against an accidental regression (e.g. someone swapping in
        // a non-fair semaphore) by verifying the behaviour
        // empirically.
        let pool = Arc::new(GpuPool::new(&[synth(0)]));
        let lease1 = pool.claim().await.unwrap();

        // Park a blocked claimer.
        let pool_clone = Arc::clone(&pool);
        let blocked = tokio::spawn(async move { pool_clone.claim().await.unwrap() });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(pool.pending_claimers(), 1);

        // Release the lease. The released permit is now reserved for
        // the parked claimer per Tokio's FIFO contract.
        drop(lease1);

        // Try to steal it from the blocked claimer — must fail.
        assert!(
            pool.try_claim().is_none(),
            "try_claim must not steal a permit reserved for a queued claimer",
        );

        // The blocked claimer should still resolve.
        let _lease2 = blocked.await.unwrap();
    }

    #[tokio::test]
    async fn try_claim_lease_drop_releases_permit() {
        // try_claim leases use the same RAII Drop path; verify the
        // permit returns to the pool when the lease drops.
        let pool = Arc::new(GpuPool::new(&[synth(0)]));
        let lease = pool.try_claim().unwrap();
        assert!(pool.try_claim().is_none());
        drop(lease);
        assert!(pool.try_claim().is_some(), "permit returned to pool after lease drop");
    }

    #[tokio::test]
    async fn try_claim_does_not_affect_pending_claimers() {
        // try_claim must not touch pending_claimers — helpers are
        // opportunistic, not blocked claimers.
        let pool = Arc::new(GpuPool::new(&[synth(0)]));
        let _l1 = pool.try_claim().unwrap();
        assert_eq!(pool.pending_claimers(), 0);
        assert!(pool.try_claim().is_none());
        assert_eq!(pool.pending_claimers(), 0);
    }

    #[tokio::test]
    async fn sparse_indices_preserved() {
        // CUDA_VISIBLE_DEVICES could expose only [0, 2, 5].
        let pool = Arc::new(GpuPool::new(&[synth(0), synth(2), synth(5)]));
        let l0 = pool.claim().await.unwrap();
        let l1 = pool.claim().await.unwrap();
        let l2 = pool.claim().await.unwrap();
        let mut got: Vec<u32> = vec![l0.gpu_index, l1.gpu_index, l2.gpu_index];
        got.sort();
        assert_eq!(got, vec![0, 2, 5]);
    }
}
