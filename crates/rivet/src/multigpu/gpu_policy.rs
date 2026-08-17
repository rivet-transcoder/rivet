//! GPU-pool construction helpers derived from [`crate::spec::EncodePolicy`].

use std::sync::Arc;

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
fn select_gpus_for_policy(policy: EncodePolicy) -> Vec<codec::gpu::GpuDevice> {
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

/// Build a [`GpuPool`] constrained to the given [`EncodePolicy`]. An empty pool
/// (e.g. a pinned index or vendor family that isn't present) yields capacity 0,
/// so the orchestrator's pre-flight probe / lease claim surfaces a clear error.
///
/// When more than one GPU is selected, cards that can't actually encode the
/// REQUESTED `codec` (e.g. a pre-Ada NVIDIA that decodes via NVDEC but has no
/// AV1 encode silicon — yet can still encode H.264/H.265) are dropped from the
/// **encode** pool, so a worker never leases an incapable card and hard-fails
/// the run; the capable cards do the encoding. A single selected GPU is left
/// as-is, since the serial path's non-pinned encoder dispatch already falls
/// through vendors. Dropped cards stay available for the decode pump
/// ([`policy_gpu_indices`] is intentionally NOT filtered).
pub fn gpu_pool_for_policy(policy: EncodePolicy, codec: codec::frame::VideoCodec) -> Arc<GpuPool> {
    let selected = select_gpus_for_policy(policy);
    let pool_gpus = if selected.len() > 1 {
        selected.into_iter().filter(|g| codec::encode::encode_capable(g, codec)).collect()
    } else {
        selected
    };
    Arc::new(GpuPool::new(&pool_gpus))
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
