//! GPU-pool construction helpers derived from [`crate::spec::EncodePolicy`].
//!
//! The one decision here that is not a filter: what the ladder gets when the
//! policy selects **no card that can encode the job's codec** in this build.
//! If the build carries a software encoder for the codec, that is a pool of
//! software slots ([`GpuPool::software`]) sized by [`software_pool_plan`];
//! otherwise there is nothing to encode on and [`gpu_pool_for_policy`]
//! **refuses**, by name, before a frame is decoded. A policy that *pins* a
//! card or a vendor never falls to software — it asked for that silicon by
//! name, and quietly encoding on the CPU instead is exactly the silent
//! narrowing the software tiers are gated against.
//!
//! An empty pool is never handed to a caller. It used to be — the ladder's
//! first lease claim was meant to catch it — but by then the decode pumps and
//! scalers were already running in blocking threads, the error path did not
//! stop them, and a run with more chunks than the queues hold sat at `0/N
//! frames` forever: the scaler blocked on a full queue nobody would drain,
//! the pump behind it, and the runtime could not shut down. The refusal now
//! lives where the emptiness is decided.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::Result;
use codec::frame::{PixelFormat, VideoCodec};
use codec::gpu::{GpuDevice, GpuVendor};

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

/// The host's GPUs, detected once per process: detection walks CUDA, WMI and
/// sysfs, which takes seconds on a loaded machine, and the cards installed do
/// not change during a run.
fn host_cards() -> &'static [GpuDevice] {
    codec::gpu::detect_gpus_cached()
}

/// The host GPUs selected by an [`EncodePolicy`]: all of them for `AllGpus` /
/// `PerRung`, the first / pinned index for `SingleGpu`, every device of one
/// vendor for `Family`.
fn select_gpus_for_policy(policy: EncodePolicy) -> Vec<GpuDevice> {
    let gpus = host_cards().to_vec();
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
/// gets nothing rather than software slots when nothing it named can encode
/// the codec.
pub(crate) fn pins_silicon(policy: EncodePolicy) -> bool {
    matches!(policy, EncodePolicy::SingleGpu(Some(_)) | EncodePolicy::Family(_))
}

/// The `--encode` spelling of a policy, for a refusal that quotes the flag
/// the operator typed.
fn policy_flag(policy: EncodePolicy) -> String {
    match policy {
        EncodePolicy::AllGpus => "all".into(),
        EncodePolicy::PerRung => "per-rung".into(),
        EncodePolicy::SingleGpu(None) => "single".into(),
        EncodePolicy::SingleGpu(Some(idx)) => format!("gpu:{idx}"),
        EncodePolicy::Family(fam) => format!("family:{}", family_flag(fam)),
    }
}

fn family_flag(fam: GpuFamily) -> &'static str {
    match fam {
        GpuFamily::Nvidia => "nvidia",
        GpuFamily::Amd => "amd",
        GpuFamily::Intel => "intel",
    }
}

fn vendor_flag(v: GpuVendor) -> &'static str {
    match v {
        GpuVendor::Nvidia => "nvidia",
        GpuVendor::Amd => "amd",
        GpuVendor::Intel => "intel",
    }
}

/// The codec as the operator reads it.
fn codec_name(codec: VideoCodec) -> &'static str {
    match codec {
        VideoCodec::Av1 => "AV1",
        VideoCodec::H264 => "H.264",
        VideoCodec::H265 => "H.265",
    }
}

/// One detected card and whether it can encode the job's codec in this
/// build — the host as a refusal describes it.
#[derive(Debug, Clone)]
pub struct CardVerdict {
    pub device: GpuDevice,
    pub capable: bool,
}

/// Every detected card with its verdict for `codec` at the output's depth.
/// Answered once per process for each codec and depth: the first caller
/// detects the host and builds one encoder per card; a caller asking the same
/// question meanwhile waits for that answer instead of probing the cards
/// again, and every later caller reads it.
pub(crate) fn host_verdicts(codec: VideoCodec, ten_bit: bool) -> Vec<CardVerdict> {
    type Answer = Arc<OnceLock<Vec<CardVerdict>>>;
    static ANSWERS: OnceLock<Mutex<HashMap<(VideoCodec, bool), Answer>>> = OnceLock::new();
    let answer = {
        let mut answers = ANSWERS.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
        Arc::clone(answers.entry((codec, ten_bit)).or_default())
    };
    answer
        .get_or_init(|| {
            host_cards()
                .iter()
                .map(|device| CardVerdict {
                    capable: codec::encode::encode_capable_at(device, codec, ten_bit),
                    device: device.clone(),
                })
                .collect()
        })
        .clone()
}

/// The host an empty-pool refusal describes: which cards are present and
/// whether each can encode the job's codec at the output's depth.
///
/// A refusal names the host so the operator can see what would serve, and
/// that costs a detection and one encoder construction per card — seconds on
/// a loaded machine. [`HostCards::Detected`] pays it once per process
/// (`host_verdicts`); [`HostCards::Fixed`] is a given inventory, for a
/// caller that must not wait on the hardware to be told why a pool it built
/// is empty — a unit test of the ladder's refusal, whose time bound is about
/// the refusal and not about the machine.
#[derive(Debug, Clone, Default)]
pub enum HostCards {
    /// This machine's cards, each judged for the job's codec and depth.
    #[default]
    Detected,
    /// These cards with these verdicts, whatever the codec and depth.
    Fixed(Vec<CardVerdict>),
}

impl HostCards {
    fn verdicts(&self, codec: VideoCodec, ten_bit: bool) -> Vec<CardVerdict> {
        match self {
            HostCards::Detected => host_verdicts(codec, ten_bit),
            HostCards::Fixed(cards) => cards.clone(),
        }
    }
}

/// Why `policy` has nothing to encode `codec` on — the operator-facing
/// message, built from what is actually on the host so it names the families
/// present, which of them could serve, and how to reach the software pool.
/// `ten_bit` is whether the output is 10-bit; the message then names the
/// format ("10-bit H.264") wherever it names the codec. `software_depth` is
/// the bit depth this build's software encoder for the codec reaches, `None`
/// when the build has none: a software tier short of the output's depth is
/// named as such, never as a feature to build with.
///
/// Pure, so every shape of host is unit-testable.
pub(crate) fn empty_pool_reason(
    policy: EncodePolicy,
    codec: VideoCodec,
    ten_bit: bool,
    cards: &[CardVerdict],
    software_depth: Option<u8>,
) -> String {
    let name = codec_name(codec);
    let codec_s = if ten_bit { format!("10-bit {name}") } else { name.to_string() };
    let software = software_depth.is_some_and(|bits| !ten_bit || bits >= 10);
    let flag = policy_flag(policy);

    // Why nothing matched — the half that depends on the policy.
    let why = match policy {
        EncodePolicy::Family(fam) => {
            let vendor = policy_vendor(fam);
            let label = codec::gpu::manufacturer_label(vendor);
            if cards.iter().any(|c| c.device.vendor == vendor) {
                format!("the {label} GPU(s) present cannot encode {codec_s} in this build")
            } else {
                format!("no {label} GPU is present")
            }
        }
        EncodePolicy::SingleGpu(Some(idx)) => match cards.iter().find(|c| c.device.index == idx) {
            Some(c) => format!("gpu {idx} ({}) cannot encode {codec_s} in this build", c.device.name),
            None => format!("there is no gpu {idx}"),
        },
        EncodePolicy::AllGpus | EncodePolicy::PerRung | EncodePolicy::SingleGpu(None) => {
            if software {
                // Unreachable by construction (an unpinned policy with
                // software available gets a software pool), kept honest.
                format!("no GPU on this host can encode {codec_s} in this build")
            } else if let Some(bits) = software_depth {
                format!(
                    "no GPU on this host can encode {codec_s} in this build, and the build's \
                     software {name} encoder is {bits}-bit"
                )
            } else {
                format!(
                    "no GPU on this host can encode {codec_s} in this build, and the build has no \
                     software {codec_s} encoder either"
                )
            }
        }
    };

    // What is on the host.
    let present = if cards.is_empty() {
        "No GPU was detected.".to_string()
    } else {
        let list: Vec<String> = cards
            .iter()
            .map(|c| {
                format!(
                    "{} (gpu {}, {}, {})",
                    c.device.name,
                    c.device.index,
                    codec::gpu::manufacturer_label(c.device.vendor),
                    if c.capable { format!("encodes {codec_s}") } else { format!("cannot encode {codec_s} in this build") }
                )
            })
            .collect();
        format!("Present: {}.", list.join("; "))
    };

    // What to do about it.
    let capable: Vec<&CardVerdict> = cards.iter().filter(|c| c.capable).collect();
    let mut fixes: Vec<String> = Vec::new();
    if let Some(first) = capable.first() {
        let mut families: Vec<&'static str> = capable.iter().map(|c| vendor_flag(c.device.vendor)).collect();
        families.dedup();
        fixes.push(format!(
            "pin a card that can (`--encode {}` or `--encode gpu:{}`) or drop the pin (`--encode all`, the default) to use them",
            families.iter().map(|f| format!("family:{f}")).collect::<Vec<_>>().join(" / "),
            first.device.index
        ));
    }
    if software {
        let feature = codec::encode::software_feature_for(codec);
        if capable.is_empty() {
            if pins_silicon(policy) {
                fixes.push(format!(
                    "the software {codec_s} encoder (`{feature}`) is compiled in and takes the job when no card is pinned: drop the pin (`--encode all`, the default) to run on the software pool"
                ));
            }
        } else {
            fixes.push(format!(
                "to run on the software {codec_s} encoder (`{feature}`) instead, drop the pin and hide the cards (`CUDA_VISIBLE_DEVICES=-1` hides NVIDIA), or build without the vendor features — the software pool takes the job only when no card can encode {codec_s} and none is pinned"
            ));
        }
    } else if let Some(bits) = software_depth {
        // The software tier is compiled in but short of the output's depth:
        // building with its feature would change nothing.
        fixes.push(format!(
            "this build's software {name} encoder (`{}`) is {bits}-bit: `--pixel-format 8bit` encodes the job at 8 bits, which it takes",
            codec::encode::software_feature_for(codec)
        ));
    } else {
        fixes.push(format!(
            "rebuild with `--features {}` for a software {codec_s} encoder, or with the vendor feature (`nvidia` / `amd` / `qsv`) for the silicon that is present",
            codec::encode::software_feature_for(codec)
        ));
    }

    format!("no encoder matches `--encode {flag}` for {codec_s} on this host: {why}. {present} Fix: {}.", fixes.join("; "))
}

/// The refusal for a policy that has nothing to encode `codec` on, naming
/// what is on `host`. Every path that finds the pool empty — the builder, the
/// ladder's preflight, its lease claim — raises this one error, so the
/// operator reads the same sentence whichever of them spoke.
pub(crate) fn empty_pool_error(
    host: &HostCards,
    policy: EncodePolicy,
    codec: VideoCodec,
    output_pixel_format: PixelFormat,
) -> anyhow::Error {
    empty_pool_error_at(host, policy, codec, is_ten_bit(output_pixel_format))
}

/// [`empty_pool_error`] for an output known only by whether it is 10-bit.
fn empty_pool_error_at(host: &HostCards, policy: EncodePolicy, codec: VideoCodec, ten_bit: bool) -> anyhow::Error {
    let cards = host.verdicts(codec, ten_bit);
    let reason = empty_pool_reason(policy, codec, ten_bit, &cards, software_depth(codec));
    tracing::warn!(?codec, ten_bit, encode = ?policy, reason = %reason, "the encode pool is empty; refusing");
    anyhow::anyhow!(reason)
}

/// [`software_reaches`] for an encoder configured for `output_pixel_format`.
pub(crate) fn software_reaches_output(codec: VideoCodec, output_pixel_format: PixelFormat) -> bool {
    software_reaches(codec, is_ten_bit(output_pixel_format))
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
/// - Otherwise: an empty pool (capacity 0). [`gpu_pool_for_policy`] turns
///   that into a refusal ([`empty_pool_error`]) rather than handing it out.
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
        _ => GpuPool::new(&[]),
    }
}

/// Build a [`GpuPool`] constrained to the given [`EncodePolicy`] for `codec`,
/// or refuse — by name — when the policy leaves nothing to encode on.
///
/// Cards that can't actually encode the REQUESTED `codec` (e.g. a pre-Ada
/// NVIDIA that decodes via NVDEC but has no AV1 encode silicon — yet can
/// still encode H.264/H.265; or, in a build without the vendor feature, every
/// card) are dropped from the **encode** pool, so a worker never leases an
/// incapable card and hard-fails the run. Dropped cards stay available for
/// the decode pump ([`policy_gpu_indices`] is intentionally NOT filtered).
///
/// When nothing capable is left, the pool is what `pool_for` says: software
/// slots if this build has a software encoder for the codec and the policy
/// did not pin silicon. Otherwise this is `Err` — `empty_pool_error`, which
/// names the pin, the families present and how to reach the software pool —
/// and the caller has not decoded a frame yet. The pool returned always has
/// at least one slot.
///
/// Capability is judged at the job's `output_pixel_format`, because every
/// worker that leases from this pool builds its encoder for that format on the
/// card it leased and nothing falls back from there. A card that takes the
/// codec only at 8 bits — NVENC for H.264 — is left out of a pool for a 10-bit
/// output, and the software slots take its place when the software tier
/// reaches 10 bits (`h26x` does, `rav1e` does not). Judging at the codec
/// alone handed the HLS ladder an RTX 3090 for 10-bit H.264 on a build with
/// `h26x-fallback`, and the ladder failed building its first encoder.
pub fn gpu_pool_for_policy(
    policy: EncodePolicy,
    codec: VideoCodec,
    output_pixel_format: PixelFormat,
) -> Result<Arc<GpuPool>> {
    pool_at(policy, codec, is_ten_bit(output_pixel_format))
}

/// The pool for the **serial** single-file encoder, which [`serial_target`]
/// reads. An unpinned policy is judged at the codec alone, as it always was:
/// the serial encoder is built by the dispatcher for the job's own format, and
/// the dispatcher falls back across backends (NVENC declining 10-bit H.264
/// hands it to `h26x`) and builds a backend pinned by name
/// (`TRANSCODE_ENCODER_BACKEND`) with or without its feature, so a pool at the
/// output's depth would refuse jobs that encode today. A policy that pins
/// silicon gets no fallback — its card is pinned by vendor — so that one is
/// judged at the output's depth like a lease, and refused before decoding.
pub fn gpu_pool_for_serial(
    policy: EncodePolicy,
    codec: VideoCodec,
    output_pixel_format: PixelFormat,
) -> Result<Arc<GpuPool>> {
    pool_at(policy, codec, serial_probe_is_ten_bit(policy, output_pixel_format))
}

/// Whether the serial pool judges the cards at 10 bits: only for a 10-bit
/// output under a policy that pins silicon (see [`gpu_pool_for_serial`]).
fn serial_probe_is_ten_bit(policy: EncodePolicy, output_pixel_format: PixelFormat) -> bool {
    pins_silicon(policy) && is_ten_bit(output_pixel_format)
}

/// Whether an encoder configured for `pixel_format` encodes more than 8 bits.
fn is_ten_bit(pixel_format: PixelFormat) -> bool {
    codec::colorspace::planar_bit_depth(pixel_format).is_some_and(|bits| bits > 8)
        || pixel_format == PixelFormat::Yuv420p10le
}

/// Whether this build's software encoder for `codec` produces the output:
/// there is one, and for a `ten_bit` output it is 10-bit (`h26x` for H.264 /
/// H.265 is; `rav1e` for AV1 is not).
pub(crate) fn software_reaches(codec: VideoCodec, ten_bit: bool) -> bool {
    software_depth(codec).is_some_and(|bits| !ten_bit || bits >= 10)
}

/// The bit depth this build's software encoder for `codec` reaches, or `None`
/// when the build has none.
fn software_depth(codec: VideoCodec) -> Option<u8> {
    codec::encode::software_backend_for(codec)
        .map(|backend| codec::encode::backend_output_caps_for(backend, codec).max_bit_depth)
}

fn pool_at(policy: EncodePolicy, codec: VideoCodec, ten_bit: bool) -> Result<Arc<GpuPool>> {
    let capable: Vec<GpuDevice> = select_gpus_for_policy(policy)
        .into_iter()
        .filter(|g| codec::encode::encode_capable_at(g, codec, ten_bit))
        .collect();
    let software = software_reaches(codec, ten_bit).then(host_software_pool_plan);
    let pool = pool_for(policy, codec, capable, software);
    if pool.capacity() == 0 {
        return Err(empty_pool_error_at(&HostCards::Detected, policy, codec, ten_bit));
    }
    Ok(Arc::new(pool))
}

/// Where a **serial** (one encoder per rung) job encodes under `policy`,
/// given the pool the policy produced: `(gpu_index, gpu_vendor)` for the
/// encoder config.
///
/// A policy that pins silicon gets the pool's first slot — the first card
/// the policy named that can encode the codec — as *both* an index and a
/// vendor, so the dispatcher's vendor-pinned branch runs and, should that
/// card fail to start, the job fails naming the vendor rather than sliding
/// down the NVIDIA-first chain to another vendor or to software. Before this
/// the serial path carried only an index, which the chain treats as a
/// preference: `--encode family:intel` on a host with no Intel card encoded
/// on NVENC, and said so only at `info`.
///
/// An unpinned policy keeps the index it always had (the policy's first
/// card, or none) and no vendor pin: the chain may still fall to software
/// for it, which is the documented meaning of "no pin". A software pool pins
/// nothing either way.
pub fn serial_target(policy: EncodePolicy, pool: &GpuPool) -> (Option<u32>, Option<GpuVendor>) {
    if pool.is_software() {
        return (None, None);
    }
    if pins_silicon(policy) {
        return match pool.snapshot_leases().first() {
            Some(slot) => (Some(slot.index), Some(slot.vendor)),
            None => (None, None),
        };
    }
    (serial_gpu_for_policy(policy), None)
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

    // ---- the refusal ----

    fn verdict(index: u32, vendor: GpuVendor, capable: bool) -> CardVerdict {
        CardVerdict { device: synth(index, vendor), capable }
    }

    /// This host, as the bug was found on it: an NVIDIA card that encodes
    /// H.264, an AMD iGPU the build cannot drive, no Intel anywhere.
    fn nvidia_plus_amd() -> Vec<CardVerdict> {
        vec![verdict(0, GpuVendor::Nvidia, true), verdict(1, GpuVendor::Amd, false)]
    }

    /// The bug's own shape: a family pin that names silicon the host does
    /// not have. The refusal quotes the flag, says the family is absent,
    /// lists what IS there with its verdict, and names both ways out — the
    /// card that could serve, and the software pool.
    #[test]
    fn a_family_that_is_absent_is_refused_by_name() {
        let s = empty_pool_reason(EncodePolicy::Family(GpuFamily::Intel), VideoCodec::H264, false, &nvidia_plus_amd(), Some(10));
        assert!(s.starts_with("no encoder matches `--encode family:intel` for H.264 on this host: no Intel GPU is present."), "{s}");
        assert!(s.contains("Present: synth-0 (gpu 0, NVIDIA, encodes H.264); synth-1 (gpu 1, AMD, cannot encode H.264 in this build)."), "{s}");
        assert!(s.contains("`--encode family:nvidia` or `--encode gpu:0`"), "{s}");
        assert!(s.contains("`--encode all`, the default"), "{s}");
        assert!(s.contains("software H.264 encoder (`h26x-fallback`)"), "{s}");
        assert!(s.contains("CUDA_VISIBLE_DEVICES=-1"), "{s}");
    }

    /// The family is there but this build cannot drive it for the codec
    /// (an AMD iGPU without the `amd` feature; an Ampere card asked for AV1).
    #[test]
    fn a_family_that_is_present_but_incapable_says_so() {
        let s = empty_pool_reason(EncodePolicy::Family(GpuFamily::Amd), VideoCodec::Av1, false, &nvidia_plus_amd(), None);
        assert!(s.contains("`--encode family:amd` for AV1"), "{s}");
        assert!(s.contains("the AMD GPU(s) present cannot encode AV1 in this build"), "{s}");
        assert!(s.contains("rebuild with `--features rav1e-fallback`"), "{s}");
        // Only the NVIDIA card is offered, and it is offered once.
        assert!(s.contains("`--encode family:nvidia` or `--encode gpu:0`"), "{s}");
        assert_eq!(s.matches("family:nvidia").count(), 1, "{s}");
    }

    #[test]
    fn a_pinned_index_that_is_absent_or_incapable_is_named() {
        let s = empty_pool_reason(EncodePolicy::SingleGpu(Some(7)), VideoCodec::H265, false, &nvidia_plus_amd(), Some(10));
        assert!(s.contains("`--encode gpu:7` for H.265 on this host: there is no gpu 7."), "{s}");
        let s = empty_pool_reason(EncodePolicy::SingleGpu(Some(1)), VideoCodec::H265, false, &nvidia_plus_amd(), Some(10));
        assert!(s.contains("gpu 1 (synth-1) cannot encode H.265 in this build."), "{s}");
    }

    /// No cards at all and no software tier: the only fix is a build.
    #[test]
    fn a_bare_host_without_software_names_the_feature() {
        let s = empty_pool_reason(EncodePolicy::AllGpus, VideoCodec::H264, false, &[], None);
        assert!(s.contains("`--encode all` for H.264"), "{s}");
        assert!(s.contains("no GPU on this host can encode H.264 in this build, and the build has no software H.264 encoder either"), "{s}");
        assert!(s.contains("No GPU was detected."), "{s}");
        assert!(s.contains("--features h26x-fallback"), "{s}");
        assert!(!s.contains("Present:"), "{s}");
    }

    /// A pin on a host where nothing can encode the codec, with software
    /// compiled in: dropping the pin is the whole fix, and the message says
    /// exactly that rather than telling the operator to hide cards that
    /// were never going to serve.
    #[test]
    fn a_pin_with_nothing_capable_and_software_available_says_drop_the_pin() {
        let cards = vec![verdict(0, GpuVendor::Nvidia, false)];
        let s = empty_pool_reason(EncodePolicy::Family(GpuFamily::Nvidia), VideoCodec::Av1, false, &cards, Some(8));
        assert!(s.contains("the NVIDIA GPU(s) present cannot encode AV1 in this build"), "{s}");
        assert!(s.contains("takes the job when no card is pinned: drop the pin (`--encode all`, the default)"), "{s}");
        assert!(!s.contains("CUDA_VISIBLE_DEVICES"), "{s}");
        assert!(!s.contains("pin a card that can"), "{s}");
    }

    #[test]
    fn the_flag_spelling_round_trips_the_parser() {
        for policy in [
            EncodePolicy::AllGpus,
            EncodePolicy::PerRung,
            EncodePolicy::SingleGpu(None),
            EncodePolicy::SingleGpu(Some(3)),
            EncodePolicy::Family(GpuFamily::Nvidia),
            EncodePolicy::Family(GpuFamily::Amd),
            EncodePolicy::Family(GpuFamily::Intel),
        ] {
            let flag = policy_flag(policy);
            assert_eq!(flag.parse::<EncodePolicy>(), Ok(policy), "{flag}");
        }
    }

    /// The builder on THIS host: a family pin the host cannot satisfy is an
    /// error, not an empty pool. Guarded on the host's inventory — a box
    /// with an Intel card that encodes H.264 is asked about AMD instead,
    /// and a box with every vendor present skips (says so).
    #[test]
    fn the_builder_refuses_a_family_this_host_lacks() {
        let present: Vec<GpuVendor> = codec::gpu::detect_gpus().iter().map(|g| g.vendor).collect();
        let absent = [GpuFamily::Intel, GpuFamily::Amd, GpuFamily::Nvidia]
            .into_iter()
            .find(|f| !present.contains(&policy_vendor(*f)));
        let Some(fam) = absent else {
            eprintln!("every vendor is present on this host; nothing to refuse");
            return;
        };
        let err = match gpu_pool_for_policy(EncodePolicy::Family(fam), VideoCodec::H264, PixelFormat::Yuv420p) {
            Ok(pool) => panic!("family {fam:?} is absent yet the builder handed out a pool of {}", pool.capacity()),
            Err(e) => format!("{e:#}"),
        };
        assert!(err.contains(&format!("no encoder matches `--encode family:{}` for H.264", family_flag(fam))), "{err}");
        assert!(err.contains(&format!("no {} GPU is present", codec::gpu::manufacturer_label(policy_vendor(fam)))), "{err}");
    }

    // ---- the output's depth ----

    /// A pool for a 10-bit output is judged at 10 bits, and its refusal says
    /// so: the card that encodes H.264 only at 8 bits (NVENC) is named as
    /// unable to encode 10-bit H.264, and the compiled-in 10-bit software tier
    /// is offered by dropping the pin, never by building a feature the build
    /// already has.
    #[test]
    fn a_ten_bit_output_is_named_and_the_compiled_in_software_tier_offered() {
        let cards = vec![verdict(0, GpuVendor::Nvidia, false)];
        let s = empty_pool_reason(EncodePolicy::Family(GpuFamily::Nvidia), VideoCodec::H264, true, &cards, Some(10));
        assert!(s.starts_with("no encoder matches `--encode family:nvidia` for 10-bit H.264 on this host: the NVIDIA GPU(s) present cannot encode 10-bit H.264 in this build."), "{s}");
        assert!(s.contains("Present: synth-0 (gpu 0, NVIDIA, cannot encode 10-bit H.264 in this build)."), "{s}");
        assert!(s.contains("the software 10-bit H.264 encoder (`h26x-fallback`) is compiled in and takes the job when no card is pinned"), "{s}");
        assert!(!s.contains("rebuild with"), "{s}");
    }

    /// A software tier that is compiled in but short of the output's depth
    /// (rav1e is 8-bit AV1) is named as such, with the setting that brings the
    /// job within its reach; only a build with no software tier at all is told
    /// to build one.
    #[test]
    fn a_software_tier_short_of_the_depth_is_named_not_rebuilt() {
        let cards = vec![verdict(0, GpuVendor::Nvidia, false)];
        let s = empty_pool_reason(EncodePolicy::AllGpus, VideoCodec::Av1, true, &cards, Some(8));
        assert!(s.contains("no GPU on this host can encode 10-bit AV1 in this build, and the build's software AV1 encoder is 8-bit."), "{s}");
        assert!(s.contains("this build's software AV1 encoder (`rav1e-fallback`) is 8-bit: `--pixel-format 8bit` encodes the job at 8 bits"), "{s}");
        assert!(!s.contains("rebuild with"), "{s}");
        let s = empty_pool_reason(EncodePolicy::AllGpus, VideoCodec::Av1, true, &cards, None);
        assert!(s.contains("rebuild with `--features rav1e-fallback` for a software 10-bit AV1 encoder"), "{s}");
    }

    /// The serial pool is judged at 10 bits only for a policy that pins
    /// silicon: an unpinned serial encoder is built by the dispatcher, which
    /// falls back from a card that declines the format and builds a backend
    /// pinned by name.
    #[test]
    fn only_a_pinned_serial_job_is_judged_at_ten_bits() {
        assert!(is_ten_bit(PixelFormat::Yuv420p10le));
        assert!(!is_ten_bit(PixelFormat::Yuv420p));
        for policy in [EncodePolicy::AllGpus, EncodePolicy::PerRung, EncodePolicy::SingleGpu(None)] {
            assert!(!serial_probe_is_ten_bit(policy, PixelFormat::Yuv420p10le), "{policy:?}");
        }
        for policy in [EncodePolicy::SingleGpu(Some(0)), EncodePolicy::Family(GpuFamily::Nvidia)] {
            assert!(serial_probe_is_ten_bit(policy, PixelFormat::Yuv420p10le), "{policy:?}");
            assert!(!serial_probe_is_ten_bit(policy, PixelFormat::Yuv420p), "{policy:?}");
        }
    }

    /// Software slots stand in for a 10-bit output only when the software
    /// tier is 10-bit: h26x for H.264 / H.265 is, rav1e for AV1 is not.
    #[test]
    fn the_software_tier_reaches_ten_bits_only_where_it_is_ten_bit() {
        for c in [VideoCodec::H264, VideoCodec::H265] {
            assert_eq!(software_reaches(c, false), codec::encode::software_encode_available(c), "{c:?}");
            assert_eq!(software_reaches(c, true), codec::encode::software_encode_available(c), "{c:?}");
        }
        assert_eq!(software_reaches(VideoCodec::Av1, false), codec::encode::software_encode_available(VideoCodec::Av1));
        assert!(!software_reaches(VideoCodec::Av1, true), "rav1e is 8-bit");
    }

    /// This host, on a build with NVENC and the software H.26x tier: the
    /// RTX 3090 encodes H.264 at 8 bits and not at 10, so a lease pool for a
    /// 10-bit H.264 output is software slots (the HLS ladder leased the card
    /// and failed building its first encoder), while an 8-bit pool keeps the
    /// card, the unpinned serial pool keeps it (the dispatcher falls back),
    /// and a family pin is refused by name at 10 bits. Skips, saying so, on a
    /// host with no NVIDIA card that encodes 8-bit H.264.
    #[cfg(all(feature = "nvidia", feature = "h26x-fallback"))]
    #[test]
    fn on_this_host_nvenc_is_no_lease_for_ten_bit_h264() {
        let Some(card) = codec::gpu::detect_gpus()
            .into_iter()
            .find(|g| g.vendor == GpuVendor::Nvidia && codec::encode::encode_capable_at(g, VideoCodec::H264, false))
        else {
            eprintln!("SKIP: no NVIDIA card here encodes 8-bit H.264");
            return;
        };
        assert!(!codec::encode::encode_capable_at(&card, VideoCodec::H264, true), "{} took 10-bit H.264", card.name);
        let ten = gpu_pool_for_policy(EncodePolicy::AllGpus, VideoCodec::H264, PixelFormat::Yuv420p10le).expect("software slots");
        assert!(ten.is_software(), "a 10-bit H.264 lease pool must be software here, got {} card slot(s)", ten.capacity());
        let eight = gpu_pool_for_policy(EncodePolicy::AllGpus, VideoCodec::H264, PixelFormat::Yuv420p).expect("the card");
        assert!(!eight.is_software());
        let serial = gpu_pool_for_serial(EncodePolicy::AllGpus, VideoCodec::H264, PixelFormat::Yuv420p10le).expect("the card");
        assert!(!serial.is_software(), "the unpinned serial pool is judged at the codec");
        let err = match gpu_pool_for_serial(EncodePolicy::Family(GpuFamily::Nvidia), VideoCodec::H264, PixelFormat::Yuv420p10le) {
            Ok(pool) => panic!("a family pin at 10 bits got a pool of {}", pool.capacity()),
            Err(e) => format!("{e:#}"),
        };
        assert!(err.contains("`--encode family:nvidia` for 10-bit H.264"), "{err}");
        assert!(err.contains("drop the pin"), "{err}");
    }

    // ---- the serial target ----

    /// A pinning policy encodes on the pool's first card and pins its
    /// vendor, so the dispatcher cannot slide to another vendor or to
    /// software if that card declines.
    #[test]
    fn a_pinned_policy_pins_the_pools_first_card_and_vendor() {
        let pool = GpuPool::new(&[synth(2, GpuVendor::Amd), synth(3, GpuVendor::Amd)]);
        assert_eq!(serial_target(EncodePolicy::Family(GpuFamily::Amd), &pool), (Some(2), Some(GpuVendor::Amd)));
        let pool = GpuPool::new(&[synth(1, GpuVendor::Intel)]);
        assert_eq!(serial_target(EncodePolicy::SingleGpu(Some(1)), &pool), (Some(1), Some(GpuVendor::Intel)));
    }

    /// An unpinned policy is unchanged: no vendor pin, and no card index
    /// either (the chain picks), which is what it did before.
    #[test]
    fn an_unpinned_policy_pins_nothing() {
        let pool = GpuPool::new(&[synth(0, GpuVendor::Nvidia)]);
        for policy in [EncodePolicy::AllGpus, EncodePolicy::PerRung, EncodePolicy::SingleGpu(None)] {
            assert_eq!(serial_target(policy, &pool), (None, None), "{policy:?}");
        }
    }

    /// A software pool has no card to name; the chain reaches software on
    /// its own.
    #[test]
    fn a_software_pool_pins_nothing() {
        let pool = GpuPool::software(1, 32);
        assert_eq!(serial_target(EncodePolicy::SingleGpu(None), &pool), (None, None));
        assert_eq!(serial_target(EncodePolicy::AllGpus, &pool), (None, None));
    }
}
