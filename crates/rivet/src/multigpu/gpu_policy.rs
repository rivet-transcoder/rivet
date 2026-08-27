//! GPU-pool construction helpers derived from [`crate::spec::EncodePolicy`].
//!
//! The one decision here that is not a filter: what the ladder gets when the
//! policy selects **no card that can encode the job's codec** in this build.
//! If the build carries a software encoder for the codec, that is a pool of
//! software slots ([`GpuPool::software`]) sized by [`software_pool_plan`];
//! otherwise it is an empty pool and the run fails with a message that names
//! the missing feature. A policy that *pins* a card or a vendor never falls to
//! software — it asked for that silicon by name, and quietly encoding on the
//! CPU instead is exactly the silent narrowing the software tiers are gated
//! against.

use std::sync::Arc;

use codec::frame::VideoCodec;
use codec::gpu::GpuDevice;

use crate::gpu_pool::GpuPool;
use crate::spec::{EncodePolicy, GpuFamily};

/// Build a [`GpuPool`] from the host's detected GPU inventory.
pub fn detect_gpu_pool() -> Arc<GpuPool> {
    Arc::new(GpuPool::new(&codec::gpu::detect_gpus()))
}

fn policy_vendor(fam: GpuFamily) -> codec::gpu::GpuVendor {
    match fam {
        GpuFamily::Nvidia => codec::gpu::GpuVendor::Nvidia,
        GpuFamily::Amd => codec::gpu::GpuVendor::Amd,
        GpuFamily::Intel => codec::gpu::GpuVendor::Intel,
    }
}

/// The host GPUs selected by an [`EncodePolicy`]: all of them for `AllGpus` /
/// `PerRung`, the first / pinned index for `SingleGpu`, every device of one
/// vendor for `Family`.
fn select_gpus_for_policy(policy: EncodePolicy) -> Vec<GpuDevice> {
    let gpus = codec::gpu::detect_gpus();
    match policy {
        EncodePolicy::AllGpus | EncodePolicy::PerRung => gpus,
        EncodePolicy::SingleGpu(None) => gpus.into_iter().take(1).collect(),
        EncodePolicy::SingleGpu(Some(idx)) => gpus.into_iter().filter(|g| g.index == idx).collect(),
        EncodePolicy::Family(fam) => {
            let v = policy_vendor(fam);
            gpus.into_iter().filter(|g| g.vendor == v).collect()
        }
    }
}

/// Environment override for the number of software slots
/// ([`host_software_pool_plan`]). The derived default is right for a
/// dedicated box; an operator sharing one, or measuring the ladder against
/// a single encoder, sets this. Clamped to `1..=available_parallelism`.
pub const SOFTWARE_SLOTS_ENV: &str = "RIVET_SOFTWARE_SLOTS";

/// Threads each software encoder is aimed at when deriving the slot count.
///
/// Independent chunks scale with the number of encoders almost linearly; an
/// encoder's own worker pool does not — a 360p rung has about twenty
/// macroblock rows to hand out, and the last cores of a wide pool wait on the
/// first. So the machine is divided into several modest encoders rather than
/// one wide one, and the split is bounded below by what keeps an encoder's
/// pool useful and above by [`MAX_SOFTWARE_SLOTS`].
const SOFTWARE_THREADS_PER_SLOT: usize = 4;

/// Ceiling on derived software slots. Every ladder worker holds one chunk
/// while it encodes it — on the single-file path that is a lead-in plus ten
/// GOPs of frames, about 1.6 GiB at 1080p — so the slot count is bounded by
/// memory before it is bounded by cores. Eight is four times a typical
/// multi-GPU host's worker count and the point where a 64-core box gets wider
/// encoders instead of more of them.
const MAX_SOFTWARE_SLOTS: usize = 8;

/// How a CPU-only host is divided among software encoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftwarePoolPlan {
    /// Software slots in the pool — ladder workers running at once.
    pub slots: usize,
    /// Thread budget handed to each slot's encoder.
    pub threads: usize,
    /// The parallelism the plan divided (`available_parallelism`, which
    /// honours a container CPU quota where a core count does not).
    pub parallelism: usize,
}

/// Divide `parallelism` threads among software encoder slots so that
/// `slots × threads` covers the machine about once: `parallelism /
/// SOFTWARE_THREADS_PER_SLOT` slots, clamped to `1..=MAX_SOFTWARE_SLOTS`, each
/// with `parallelism / slots` threads. `slots_override` (the operator's
/// [`SOFTWARE_SLOTS_ENV`]) replaces the derived slot count and is clamped to
/// `1..=parallelism` so no slot is left with nothing.
///
/// Pure, so the arithmetic is testable on any machine.
pub fn software_pool_plan(parallelism: usize, slots_override: Option<usize>) -> SoftwarePoolPlan {
    let parallelism = parallelism.max(1);
    let slots = match slots_override {
        Some(n) => n.clamp(1, parallelism),
        None => (parallelism / SOFTWARE_THREADS_PER_SLOT).clamp(1, MAX_SOFTWARE_SLOTS),
    };
    let threads = (parallelism / slots).max(1);
    SoftwarePoolPlan { slots, threads, parallelism }
}

/// [`software_pool_plan`] for this host: the runtime's parallelism and the
/// [`SOFTWARE_SLOTS_ENV`] override, if set to a number.
pub fn host_software_pool_plan() -> SoftwarePoolPlan {
    let parallelism = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let over = std::env::var(SOFTWARE_SLOTS_ENV).ok().and_then(|v| v.trim().parse::<usize>().ok());
    software_pool_plan(parallelism, over)
}

/// Whether the policy asked for particular silicon by name. Such a policy
/// gets an empty pool rather than software slots when nothing it named can
/// encode the codec.
fn pins_silicon(policy: EncodePolicy) -> bool {
    matches!(policy, EncodePolicy::SingleGpu(Some(_)) | EncodePolicy::Family(_))
}

/// Why a policy ended up with an empty pool for `codec` — the operator-facing
/// sentence. `software` is whether this build could have encoded it on the
/// CPU.
pub(crate) fn empty_pool_reason(policy: EncodePolicy, codec: VideoCodec, software: bool) -> String {
    let where_ = match policy {
        EncodePolicy::AllGpus | EncodePolicy::PerRung | EncodePolicy::SingleGpu(None) => {
            "no GPU on this host".to_string()
        }
        EncodePolicy::SingleGpu(Some(idx)) => format!("GPU {idx} (pinned by the encode policy) cannot"),
        EncodePolicy::Family(fam) => format!("no {fam:?} GPU (the encode policy's family)"),
    };
    let verb = if where_.ends_with("cannot") { "" } else { " can" };
    if software {
        format!(
            "{where_}{verb} encode {codec:?} in this build; software {codec:?} encoding is compiled \
             in, but the encode policy {policy:?} names a card, so it was not used — use `--encode \
             all` (or `single`) to run the ladder on the CPU"
        )
    } else {
        format!(
            "{where_}{verb} encode {codec:?} in this build, and the build has no software {codec:?} \
             encoder either — rebuild with `--features {}` to allow encoding on the CPU, or build \
             with the vendor feature (`nvidia` / `amd` / `qsv`) for the silicon that is present",
            codec::encode::software_feature_for(codec)
        )
    }
}

/// The pool a policy gets for `codec` on a host whose policy-selected,
/// encode-capable cards are `capable`, given `software` — the CPU plan when
/// this build has a software encoder for the codec, `None` when it does not.
///
/// - Any capable card: a pool of exactly those cards.
/// - None, software available, policy not pinning silicon: a pool of
///   software slots — `plan.slots` of them for a spreading policy, **one**
///   with the whole machine for `SingleGpu(None)`, whose meaning ("one
///   encoder at a time") survives the move to the CPU.
/// - Otherwise: an empty pool. The caller's `claim()` finds nothing and the
///   run fails with [`empty_pool_reason`].
///
/// Pure — no detection, no probing — so the zero-GPU cases are unit-testable.
pub(crate) fn pool_for(
    policy: EncodePolicy,
    codec: VideoCodec,
    capable: Vec<GpuDevice>,
    software: Option<SoftwarePoolPlan>,
) -> GpuPool {
    if !capable.is_empty() {
        return GpuPool::new(&capable);
    }
    match software {
        Some(plan) if !pins_silicon(policy) => {
            let (slots, threads) = if policy.spreads() {
                (plan.slots, plan.threads)
            } else {
                (1, plan.parallelism)
            };
            tracing::info!(
                ?codec,
                encode = ?policy,
                slots,
                threads_per_slot = threads,
                parallelism = plan.parallelism,
                "no GPU can encode this codec in this build — the ladder runs on software \
                 leases: each slot is a CPU share, one software encoder at a time per slot",
            );
            GpuPool::software(slots, threads)
        }
        _ => {
            tracing::warn!(
                ?codec,
                encode = ?policy,
                reason = %empty_pool_reason(policy, codec, software.is_some()),
                "the encode pool is empty",
            );
            GpuPool::new(&[])
        }
    }
}

/// Build a [`GpuPool`] constrained to the given [`EncodePolicy`] for `codec`.
///
/// Cards that can't actually encode the REQUESTED `codec` (e.g. a pre-Ada
/// NVIDIA that decodes via NVDEC but has no AV1 encode silicon — yet can
/// still encode H.264/H.265; or, in a build without the vendor feature, every
/// card) are dropped from the **encode** pool, so a worker never leases an
/// incapable card and hard-fails the run. Dropped cards stay available for
/// the decode pump ([`policy_gpu_indices`] is intentionally NOT filtered).
///
/// When nothing capable is left, the pool is what [`pool_for`] says: software
/// slots if this build has a software encoder for the codec and the policy
/// did not pin silicon, else empty (capacity 0), so the orchestrator's lease
/// claim surfaces a clear error.
pub fn gpu_pool_for_policy(policy: EncodePolicy, codec: VideoCodec) -> Arc<GpuPool> {
    let capable: Vec<GpuDevice> = select_gpus_for_policy(policy)
        .into_iter()
        .filter(|g| codec::encode::encode_capable(g, codec))
        .collect();
    let software = codec::encode::software_encode_available(codec).then(host_software_pool_plan);
    Arc::new(pool_for(policy, codec, capable, software))
}

/// The GPU indices an [`EncodePolicy`] selects, in detection order. Used to pin
/// the decode pump to a device consistent with the policy (so decode honors a
/// `Family` / `SingleGpu` constraint, not just encode).
pub fn policy_gpu_indices(policy: EncodePolicy) -> Vec<u32> {
    select_gpus_for_policy(policy).into_iter().map(|g| g.index).collect()
}

/// The GPU index to pin a *serial* (single-GPU) encode/decode to under a
/// policy: `None` (auto/first-available) for `AllGpus`, the pinned index for
/// `SingleGpu`, the first device of the vendor for `Family`.
pub fn serial_gpu_for_policy(policy: EncodePolicy) -> Option<u32> {
    match policy {
        EncodePolicy::AllGpus | EncodePolicy::PerRung => None,
        EncodePolicy::SingleGpu(idx) => idx,
        EncodePolicy::Family(_) => select_gpus_for_policy(policy).first().map(|g| g.index),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use codec::gpu::GpuVendor;

    fn synth(index: u32, vendor: GpuVendor) -> GpuDevice {
        GpuDevice {
            index,
            vendor_index: index,
            vendor,
            name: format!("synth-{index}"),
            generation: "Synth".into(),
            pci_id: String::new(),
            vram_mib: 0,
            serial: None,
            host_pci_address: String::new(),
            vendor_id_hex: String::new(),
        }
    }

    fn plan32() -> SoftwarePoolPlan {
        software_pool_plan(32, None)
    }

    // ---- the arithmetic ----

    #[test]
    fn plan_divides_the_machine_once() {
        // 32 cores → 8 slots × 4 threads: covers the machine exactly once.
        assert_eq!(plan32(), SoftwarePoolPlan { slots: 8, threads: 4, parallelism: 32 });
        assert_eq!(software_pool_plan(16, None), SoftwarePoolPlan { slots: 4, threads: 4, parallelism: 16 });
        // A wider box gets wider encoders, not more of them.
        assert_eq!(software_pool_plan(64, None), SoftwarePoolPlan { slots: 8, threads: 8, parallelism: 64 });
        for p in [1usize, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128] {
            let plan = software_pool_plan(p, None);
            assert!(plan.slots >= 1 && plan.threads >= 1, "{p}: {plan:?}");
            assert!(plan.slots * plan.threads <= p, "{p}: oversubscribed: {plan:?}");
            assert!(plan.slots <= MAX_SOFTWARE_SLOTS, "{p}: {plan:?}");
        }
    }

    #[test]
    fn small_machines_get_one_slot_with_every_core() {
        assert_eq!(software_pool_plan(1, None), SoftwarePoolPlan { slots: 1, threads: 1, parallelism: 1 });
        assert_eq!(software_pool_plan(4, None), SoftwarePoolPlan { slots: 1, threads: 4, parallelism: 4 });
        assert_eq!(software_pool_plan(6, None), SoftwarePoolPlan { slots: 1, threads: 6, parallelism: 6 });
        // Zero parallelism is nonsense; treated as one core.
        assert_eq!(software_pool_plan(0, None).slots, 1);
    }

    #[test]
    fn the_override_replaces_the_slot_count_and_is_clamped() {
        assert_eq!(software_pool_plan(32, Some(1)), SoftwarePoolPlan { slots: 1, threads: 32, parallelism: 32 });
        assert_eq!(software_pool_plan(32, Some(16)), SoftwarePoolPlan { slots: 16, threads: 2, parallelism: 32 });
        // More slots than cores: clamped so every slot keeps a thread.
        assert_eq!(software_pool_plan(32, Some(500)), SoftwarePoolPlan { slots: 32, threads: 1, parallelism: 32 });
        // Zero: clamped up to one.
        assert_eq!(software_pool_plan(32, Some(0)).slots, 1);
    }

    // ---- the decision ----

    #[test]
    fn zero_gpus_with_software_available_gets_a_software_pool() {
        let pool = pool_for(EncodePolicy::AllGpus, VideoCodec::H264, Vec::new(), Some(plan32()));
        assert!(pool.is_software(), "the ladder must get software slots on a CPU-only host");
        assert_eq!(pool.capacity(), 8);
        assert_eq!(pool.software_threads(), Some(4));
        // `PerRung` spreads too.
        let pool = pool_for(EncodePolicy::PerRung, VideoCodec::Av1, Vec::new(), Some(plan32()));
        assert!(pool.is_software());
        assert_eq!(pool.capacity(), 8);
    }

    #[test]
    fn zero_gpus_without_software_gets_an_empty_pool() {
        let pool = pool_for(EncodePolicy::AllGpus, VideoCodec::H264, Vec::new(), None);
        assert!(!pool.is_software());
        assert_eq!(pool.capacity(), 0, "nothing to hand out: the run must fail, by name");
    }

    #[test]
    fn single_unpinned_keeps_its_meaning_on_the_cpu() {
        // `--encode single`: one encoder at a time, so one slot — with the
        // whole machine, since nothing else is running.
        let pool = pool_for(EncodePolicy::SingleGpu(None), VideoCodec::H265, Vec::new(), Some(plan32()));
        assert!(pool.is_software());
        assert_eq!(pool.capacity(), 1);
        assert_eq!(pool.software_threads(), Some(32));
    }

    #[test]
    fn a_policy_that_pins_silicon_never_falls_to_software() {
        for policy in [EncodePolicy::SingleGpu(Some(0)), EncodePolicy::Family(GpuFamily::Nvidia)] {
            let pool = pool_for(policy, VideoCodec::H264, Vec::new(), Some(plan32()));
            assert!(!pool.is_software(), "{policy:?} asked for silicon by name");
            assert_eq!(pool.capacity(), 0, "{policy:?}");
        }
    }

    #[test]
    fn capable_cards_win_over_software_whatever_the_policy() {
        let cards = vec![synth(0, GpuVendor::Nvidia), synth(1, GpuVendor::Intel)];
        for policy in [
            EncodePolicy::AllGpus,
            EncodePolicy::PerRung,
            EncodePolicy::SingleGpu(None),
            EncodePolicy::SingleGpu(Some(0)),
            EncodePolicy::Family(GpuFamily::Nvidia),
        ] {
            let pool = pool_for(policy, VideoCodec::Av1, cards.clone(), Some(plan32()));
            assert!(!pool.is_software(), "{policy:?}");
            assert_eq!(pool.capacity(), 2, "{policy:?}");
        }
    }

    #[test]
    fn the_empty_pool_reason_names_the_fix() {
        let s = empty_pool_reason(EncodePolicy::AllGpus, VideoCodec::H264, false);
        assert!(s.contains("no GPU on this host can encode H264"), "{s}");
        assert!(s.contains("--features h26x-fallback"), "{s}");

        let s = empty_pool_reason(EncodePolicy::AllGpus, VideoCodec::Av1, false);
        assert!(s.contains("--features rav1e-fallback"), "{s}");

        let s = empty_pool_reason(EncodePolicy::SingleGpu(Some(1)), VideoCodec::H265, true);
        assert!(s.contains("GPU 1 (pinned by the encode policy) cannot encode H265"), "{s}");
        assert!(s.contains("software H265 encoding is compiled in"), "{s}");
        assert!(s.contains("--encode all"), "{s}");

        let s = empty_pool_reason(EncodePolicy::Family(GpuFamily::Amd), VideoCodec::Av1, true);
        assert!(s.contains("no Amd GPU (the encode policy's family) can encode Av1"), "{s}");
    }
}
