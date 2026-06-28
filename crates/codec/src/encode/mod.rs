#[cfg(feature = "amd")]
pub mod amf;
#[cfg(not(feature = "amd"))]
#[path = "amf_stub.rs"]
pub mod amf;
#[cfg(feature = "ffmpeg")]
pub mod ffmpeg_enc;
#[cfg(feature = "nvidia")]
pub mod nvenc;
#[cfg(not(feature = "nvidia"))]
#[path = "nvenc_stub.rs"]
pub mod nvenc;
#[cfg(feature = "qsv")]
pub mod qsv;
#[cfg(not(feature = "qsv"))]
#[path = "qsv_stub.rs"]
pub mod qsv;
pub mod tuning;
// rav1e CPU encoder + Vulkan video encoder were deleted 2026-05-08
// per the GPU-only encoding directive. Production hosts must have
// AV1 silicon (NVIDIA Ada+ / AMD RDNA3+ / Intel Arc); jobs that
// land on a host without one of those vendor-native paths now
// hard-fail at encoder construction.

use crate::frame::{ColorMetadata, PixelFormat, VideoCodec, VideoFrame};
use crate::gpu;
use anyhow::Result;
use bytes::Bytes;

pub use tuning::{QualityTarget, SpeedTier};

/// Pick a GPU for a given vendor, honouring an explicit `gpu_index`
/// request when set. Returns `None` if no vendor GPU is present OR
/// the requested index belongs to a different vendor.
///
/// - `requested = Some(idx)`: look up the GPU with `GpuDevice.index == idx`.
///   If it exists AND matches `vendor`, return it. If it exists but is
///   a different vendor (e.g. caller pinned variant to NVIDIA slot 2
///   but we're evaluating the AMD fallback branch), return `None` so
///   dispatch falls through to the next tier — the other vendor tiers
///   will see this same `requested` index and match it there.
/// - `requested = None`: first-of-vendor (original pre-multi-GPU
///   behaviour, single-GPU hosts unaffected).
fn pick_vendor_device(
    gpus: &[gpu::GpuDevice],
    vendor: gpu::GpuVendor,
    requested: Option<u32>,
) -> Option<&gpu::GpuDevice> {
    match requested {
        Some(idx) => gpus.iter().find(|g| g.index == idx && g.vendor == vendor),
        None => gpus.iter().find(|g| g.vendor == vendor),
    }
}

/// Shared truthy-string parse for env flags — mirrors the decode-side
/// `env_flag_truthy` so `DISABLE_FFMPEG=1` / `true` / `yes` / `on`
/// all work identically across decode + encode dispatch.
#[cfg(feature = "ffmpeg")]
fn ffmpeg_disable_flag() -> bool {
    match std::env::var("DISABLE_FFMPEG") {
        Ok(v) => {
            let v = v.to_ascii_lowercase();
            matches!(v.as_str(), "1" | "true" | "yes" | "on" | "y" | "t")
        }
        Err(_) => false,
    }
}

pub trait Encoder: Send {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()>;
    fn flush(&mut self) -> Result<()>;
    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>>;
}

#[derive(Debug, Clone)]
pub struct EncodedPacket {
    pub data: Bytes,
    pub pts: u64,
    pub is_keyframe: bool,
}

/// Encoder configuration.
///
/// Prefer `target` + `tier` — `quality` and `speed_preset` are the
/// legacy per-encoder escape hatches and are kept so existing callers
/// compile. When `quality` is set to its sentinel (u8::MAX) the
/// adapter derives the quantizer from `target` instead. Same for
/// `speed_preset` (u8::MAX sentinel → derive from `tier`).
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub width: u32,
    pub height: u32,
    pub frame_rate: f64,
    /// Legacy escape hatch. `u8::MAX` means "derive from `target`".
    /// Otherwise: rav1e → used as quantizer 0-255; NVENC → scaled to
    /// its CQ range.
    pub quality: u8,
    /// Legacy escape hatch. `u8::MAX` means "derive from `tier`".
    pub speed_preset: u8,
    pub keyframe_interval: u32,
    /// Perceptual quality target. Defaults to `Standard` (VMAF ~90).
    pub target: QualityTarget,
    /// Speed tier (Draft / Standard / Archive). Defaults to `Standard`.
    pub tier: SpeedTier,
    /// Thread budget for this encoder instance. `0` means "use all cores"
    /// (rav1e default). When the pipeline runs N variants in parallel it
    /// should set this to `num_cpus / N` to avoid oversubscribing rayon
    /// workers across concurrent rav1e encoders.
    pub threads: usize,
    /// Input pixel format. Drives the encoder's bit-depth dispatch
    /// (Squad-19 rav1e CPU + Squad-22 NVENC/AMF/QSV, roadmap #5).
    /// `Yuv420p` → 8-bit AV1 Profile 0; `Yuv420p10le` → 10-bit AV1
    /// Profile 0 (10-bit 4:2:0 is allowed in Profile 0 per AV1 §5.5.2
    /// — `seq_profile=0`, `seq_color_config` emits `high_bitdepth=1`,
    /// `twelve_bit=0`). HW backends pick the matching surface fourcc:
    /// NVENC `YUV420_10BIT`, AMF `P010`, QSV `P010` + `BitDepthLuma=10`.
    /// Set once at encoder construction; flipping mid-session requires
    /// reinitialising. The muxer's `pixi`-equivalent + AV1 sequence
    /// header in `av1C` carry the bit depth so HDR-capable browsers
    /// see 10-bit signaling.
    pub pixel_format: PixelFormat,
    /// Source color metadata. Encoders write
    /// `color_primaries` / `transfer_characteristics` /
    /// `matrix_coefficients` / `color_range` into the AV1 sequence
    /// header so HDR-capable players see the correct PQ/HLG transfer
    /// + BT.2020 primaries straight off the bitstream — not just the
    /// container `colr` atom (Squad-19 rav1e + Squad-22 HW; complements
    /// Squad-18's container-side colr nclx writer). Without bitstream
    /// signalling, players that prefer the OBU header over the box
    /// (e.g. Chromium video framework) would silently fall back to
    /// BT.709. Defaults to SDR BT.709.
    pub color_metadata: ColorMetadata,
    /// Explicit GPU device index for HW encoders on multi-GPU hosts.
    /// When `Some(idx)`, `select_encoder` binds NVENC / AMF / QSV /
    /// Vulkan AV1 / FFmpeg hwaccel encoders to the device with
    /// `GpuDevice.index == idx`. When `None` (default), the first
    /// GPU of each vendor is used — matches the original pre-multi-GPU
    /// behaviour.
    ///
    /// Pipeline `transcode::run` assigns `variant_idx % devices.len()`
    /// per variant so a multi-variant job on a multi-GPU host spreads
    /// work across devices, matching the Python original's
    /// `ThreadPoolExecutor(max_workers=device_count)` per-variant fan-out.
    pub gpu_index: Option<u32>,
    /// Explicit vendor pin for HW encoder dispatch. When `Some(v)`,
    /// `select_encoder` skips the NVIDIA → AMD → Intel preference
    /// chain and goes DIRECTLY to the encoder backend matching `v`
    /// (NVENC for Nvidia, AMF for Amd, QSV for Intel). Used by the
    /// CMAF orchestrator to honor the GpuPool's lease — when the
    /// pool hands out an Intel slot (because the NVIDIA card is
    /// already encoding), this field tells the factory to dispatch
    /// to QSV instead of falling back to NVENC and pinning every
    /// variant to the NVIDIA card.
    ///
    /// `None` (default) preserves the legacy NVIDIA-first chain so
    /// CPU-only paths + tests + non-pool callers behave unchanged.
    pub gpu_vendor: Option<gpu::GpuVendor>,
    /// Prefer **constant-QP** rate control over the bitrate/quality default.
    /// Set by the multi-GPU single-file path under `ChunkSeamMode::ParallelConstQp`
    /// so independently-encoded chunks have a flat quality across the stitched
    /// seams. On NVENC this selects `RateControlMode::ConstQp` (the wrapper then
    /// uses the preset's default QP — the `target` bitrate mapping is skipped).
    /// AMD/QSV already encode constant-quality, so this is a no-op for them.
    pub constant_qp: bool,
    /// Output video codec. `Av1` (default, royalty-clean) or `H264` / `H265`
    /// for legacy-player compatibility. The HW backends dispatch the codec
    /// id / profile on this; the muxer picks the matching sample entry.
    pub codec: VideoCodec,
}

/// Sentinel meaning "derive from `target` or `tier`".
pub const AUTO_FROM_TARGET: u8 = u8::MAX;

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            frame_rate: 30.0,
            quality: AUTO_FROM_TARGET,
            speed_preset: AUTO_FROM_TARGET,
            keyframe_interval: 240,
            target: QualityTarget::Standard,
            tier: SpeedTier::Standard,
            threads: 0,
            // 8-bit SDR baseline — keeps every existing
            // `EncoderConfig { ..default() }` literal compiling and
            // behaving unchanged. 10-bit callers (Squad-19 rav1e or
            // Squad-22 HW backends) explicitly opt in by setting
            // `pixel_format = Yuv420p10le` and populating
            // `color_metadata` from the source.
            pixel_format: PixelFormat::Yuv420p,
            color_metadata: ColorMetadata::default(),
            gpu_index: None,
            gpu_vendor: None,
            constant_qp: false,
            codec: VideoCodec::Av1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderBackend {
    Nvenc,
    Amf,
    Qsv,
}

/// What output formats an encoder path can produce. AV1 here is 4:2:0 only;
/// 10-bit output is the web-safe AV1 Main profile (4:2:0 10-bit), HDR-tagged at
/// the container level (`colr`/`mdcv`/`clli`), not the wide-gamut professional
/// profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputCaps {
    /// Highest luma bit depth the path can encode (8 or 10).
    pub max_bit_depth: u8,
    /// Can produce HDR (PQ/HLG + BT.2020) output — i.e. 10-bit AV1 + the muxer's
    /// HDR color atoms.
    pub hdr: bool,
}

/// Output capabilities of a specific hardware backend. All three do 10-bit AV1,
/// so they can produce HDR without the `ffmpeg` feature: NVENC via
/// `Yuv420_10bit`, AMF via `P010`, and QSV via the in-repo oneVPL P010 path
/// ([`qsv_p010`]).
pub fn backend_output_caps(backend: EncoderBackend) -> OutputCaps {
    match backend {
        EncoderBackend::Nvenc | EncoderBackend::Amf | EncoderBackend::Qsv => {
            OutputCaps { max_bit_depth: 10, hdr: true }
        }
    }
}

/// Output capabilities of **this build** — the union over every compiled
/// encoder path. 10-bit + HDR comes from NVENC (`nvidia`), AMF (`amd`), QSV
/// (`qsv`, via the in-repo P010 path), or the `ffmpeg` software/hwaccel
/// encoders; a build with no encoder feature is 8-bit. Callers (e.g. rivet's
/// `OutputSpec::validate`) use this to reject a format the build can't produce.
pub fn build_output_caps() -> OutputCaps {
    #[cfg(any(
        feature = "ffmpeg",
        feature = "nvidia",
        feature = "amd",
        feature = "qsv"
    ))]
    {
        return OutputCaps { max_bit_depth: 10, hdr: true };
    }
    #[allow(unreachable_code)]
    OutputCaps { max_bit_depth: 8, hdr: false }
}

/// AV1-encode backends compiled into this build, in dispatch-preference order.
pub fn encode_backends() -> Vec<&'static str> {
    let mut v = Vec::new();
    if cfg!(feature = "nvidia") {
        v.push("nvenc");
    }
    if cfg!(feature = "amd") {
        v.push("amf");
    }
    if cfg!(feature = "qsv") {
        v.push("qsv");
    }
    if cfg!(feature = "ffmpeg") {
        v.push("ffmpeg");
    }
    v
}

/// Construct the QSV encoder. The hand-rolled oneVPL encoder (`qsv.rs`) handles
/// both 8-bit (NV12) and 10-bit (P010) AV1; under `not(qsv)` this hits the stub.
fn make_qsv_encoder(config: EncoderConfig, gpu_index: u32) -> Result<Box<dyn Encoder>> {
    Ok(Box::new(qsv::QsvEncoder::new(config, gpu_index)?))
}

/// Create the best available AV1 encoder.
///
/// Priority: NVENC (Ada+) → AMF (RDNA3+) → QSV (Arc / Meteor Lake+).
///
/// GPU-only — there is no CPU fallback. Hosts without AV1-encode
/// silicon hard-fail at construction. The previous rav1e CPU and
/// Vulkan Video tiers were removed 2026-05-08: rav1e on Archive
/// preset doesn't keep up with real-time throughput at 4K and the
/// Vulkan-encode binding never made it past scaffolding.
/// All backends compiled in; availability checked at runtime.
pub fn select_encoder(
    config: EncoderConfig,
    preferred: Option<EncoderBackend>,
) -> Result<Box<dyn Encoder>> {
    let gpus = gpu::detect_gpus();

    if let Some(backend) = preferred {
        return create_backend(backend, config, &gpus);
    }

    // Tier 0 (feature-gated): FFmpeg AV1 encoder (libavcodec's
    // av1_nvenc / av1_amf / av1_qsv / av1_vaapi / libsvtav1 /
    // libaom-av1 / librav1e probe chain). When the `ffmpeg` feature
    // is built and DISABLE_FFMPEG is not set, FFmpeg is the first
    // encoder tried for every host — one interface covers every GPU
    // vendor AND the CPU fallbacks. The native NVENC / AMF / QSV /
    // Vulkan AV1 / rav1e paths below remain as failover when the
    // FFmpeg probe chain errors. See `docs/hw-matrix.md`.
    #[cfg(feature = "ffmpeg")]
    {
        if !ffmpeg_disable_flag() {
            match ffmpeg_enc::FfmpegEncoder::new(config.clone()) {
                Ok(enc) => {
                    tracing::info!(
                        backend = "ffmpeg",
                        av1_encoder = enc.engaged(),
                        "FFmpeg primary encoder dispatch engaged"
                    );
                    return Ok(Box::new(enc));
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "FFmpeg AV1 encoder chain exhausted; falling through to native backends"
                    );
                }
            }
        } else {
            tracing::debug!("DISABLE_FFMPEG set; skipping FFmpeg encoder dispatch");
        }
    }

    // Vendor-pin shortcut: when the caller has already chosen which
    // GPU to use (CMAF orchestrator does this via the GpuPool lease,
    // 2026-05-03), dispatch DIRECTLY to that vendor's backend
    // instead of running the NVIDIA-first preference chain.
    // Without this, a host with both NVIDIA + Intel GPUs always
    // routed every variant to NVENC because the chain hits
    // `pick_vendor_device(Nvidia, ...)` first; the Arc sat idle even
    // when NVENC sessions were saturated. CPU rav1e remains the
    // last-resort if hardware init fails on the pinned vendor.
    if let Some(pinned) = config.gpu_vendor {
        if let Some(dev) = pick_vendor_device(&gpus, pinned, config.gpu_index) {
            if gpu::supports_av1_encode(dev) {
                let attempt = match pinned {
                    gpu::GpuVendor::Nvidia => nvenc::NvencEncoder::new(config.clone(), dev.index)
                        .map(|e| Box::new(e) as Box<dyn Encoder>),
                    gpu::GpuVendor::Amd => amf::AmfEncoder::new(config.clone(), dev.index)
                        .map(|e| Box::new(e) as Box<dyn Encoder>),
                    gpu::GpuVendor::Intel => make_qsv_encoder(config.clone(), dev.index),
                };
                return match attempt {
                    Ok(enc) => {
                        tracing::info!(
                            gpu_name = %dev.name,
                            gpu_index = dev.index,
                            vendor = ?pinned,
                            "using vendor-pinned AV1 hardware encoder (lease-driven dispatch)"
                        );
                        Ok(enc)
                    }
                    Err(e) => {
                        // GPU-only directive (2026-05-08): the caller
                        // pinned a vendor for a reason (lease-driven
                        // GPU pool dispatch). Init failure is a hard
                        // error — there is no CPU fallback. Surface
                        // the underlying driver error so the worker
                        // can report it on the failed-job event.
                        Err(anyhow::anyhow!(
                            "vendor-pinned AV1 encoder init failed (vendor={pinned:?}, gpu={}, idx={}): {e}",
                            dev.name,
                            dev.index,
                        ))
                    }
                };
            }
            return Err(anyhow::anyhow!(
                "vendor-pinned GPU lacks AV1 encode silicon (vendor={pinned:?}, gpu={}); \
                 GPU-only encode policy has no CPU fallback",
                dev.name,
            ));
        }
        return Err(anyhow::anyhow!(
            "vendor-pinned encoder requested (vendor={pinned:?}, gpu_index={:?}) but no matching GPU found",
            config.gpu_index,
        ));
    }

    // Auto-select: NVIDIA NVENC (Ada+) first, then AMD AMF (RDNA3+),
    // then Intel QSV (Arc / Meteor Lake+). No CPU fallback; hosts
    // without any AV1 encode silicon hard-fail at the end of the chain.
    //
    // Per-vendor device resolution: when `config.gpu_index` is Some,
    // prefer the GPU with matching `.index` for that vendor so
    // multi-GPU hosts can pin variant N → device N. When None, fall
    // back to first-of-vendor (single-GPU behaviour preserved).
    if let Some(dev) = pick_vendor_device(&gpus, gpu::GpuVendor::Nvidia, config.gpu_index) {
        if gpu::supports_av1_encode(dev) {
            match nvenc::NvencEncoder::new(config.clone(), dev.index) {
                Ok(enc) => {
                    tracing::info!(
                        gpu_name = %dev.name,
                        gpu_index = dev.index,
                        "using NVENC AV1 hardware encoder"
                    );
                    return Ok(Box::new(enc));
                }
                Err(e) => {
                    tracing::warn!(error = %e, "NVENC init failed, falling back to next backend");
                }
            }
        } else {
            // Capability gap, not an error: this NVIDIA GPU's NVENC silicon
            // predates AV1 encode (AV1 NVENC was added in Ada Lovelace
            // RTX 4000 and Ampere datacenter A10/A10G/L4/L40 — consumer
            // 30-series and older do NOT have it). The GPU can still
            // handle NVDEC decode; only the encode half falls through.
            tracing::info!(
                gpu = %dev.name,
                "NVIDIA GPU lacks AV1 NVENC silicon — trying other GPU backends"
            );
        }
    }

    if let Some(dev) = pick_vendor_device(&gpus, gpu::GpuVendor::Amd, config.gpu_index) {
        if gpu::supports_av1_encode(dev) {
            match amf::AmfEncoder::new(config.clone(), dev.index) {
                Ok(enc) => {
                    tracing::info!(
                        gpu_name = %dev.name,
                        gpu_index = dev.index,
                        "using AMF AV1 hardware encoder"
                    );
                    return Ok(Box::new(enc));
                }
                Err(e) => {
                    tracing::warn!(error = %e, "AMF init failed, falling back to next backend");
                }
            }
        } else {
            tracing::info!(
                gpu = %dev.name,
                "AMD GPU predates RDNA3 — no AV1 AMF silicon; trying Intel / CPU"
            );
        }
    }

    if let Some(dev) = pick_vendor_device(&gpus, gpu::GpuVendor::Intel, config.gpu_index) {
        if gpu::supports_av1_encode(dev) {
            match make_qsv_encoder(config.clone(), dev.index) {
                Ok(enc) => {
                    tracing::info!(
                        gpu_name = %dev.name,
                        gpu_index = dev.index,
                        "using QSV AV1 hardware encoder"
                    );
                    return Ok(enc);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "QSV init failed; chain exhausted");
                }
            }
        } else {
            tracing::info!(
                gpu = %dev.name,
                "Intel GPU predates Arc/Meteor Lake — no AV1 QSV silicon"
            );
        }
    }

    // GPU-only encode (2026-05-08): no CPU fallback. A host that
    // reaches this point has no AV1 encode silicon (or every vendor
    // path failed init) and must be reprovisioned.
    Err(anyhow::anyhow!(
        "no AV1 GPU encoder available — the host needs NVIDIA Ada+ / AMD RDNA3+ / Intel Arc \
         for AV1 hardware encoding. CPU encoding (rav1e) was removed per the GPU-only directive."
    ))
}

/// Whether an AV1 encoder can actually be constructed for this device — the
/// authoritative, build-aware capability check. It runs the **same**
/// [`select_encoder`] dispatch a per-chunk worker uses, pinned to the device's
/// vendor + index, so `true` means a worker leased to this GPU will encode
/// rather than hard-fail. Used to drop AV1-incapable cards (e.g. a pre-Ada
/// NVIDIA that decodes via NVDEC but has no AV1 encode silicon) from the
/// multi-GPU encode pool, so a mixed-vendor host encodes on the capable cards
/// instead of aborting when a chunk leases to an incapable one.
///
/// The probe constructs + immediately drops a real encoder, so the verdict is
/// cached per GPU index (queried once per process).
pub fn av1_encode_capable(dev: &gpu::GpuDevice) -> bool {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<u32, bool>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&cached) = cache.lock().unwrap().get(&dev.index) {
        return cached;
    }
    // A representative, widely-accepted probe size; AV1 codec support does not
    // depend on resolution, so any valid dims answer the capability question.
    let probe = EncoderConfig {
        width: 640,
        height: 480,
        frame_rate: 30.0,
        gpu_index: Some(dev.index),
        gpu_vendor: Some(dev.vendor),
        ..Default::default()
    };
    let capable = match select_encoder(probe, None) {
        Ok(_enc) => true, // encoder is dropped here, releasing the session
        Err(e) => {
            tracing::info!(
                gpu_index = dev.index,
                gpu = %dev.name,
                vendor = ?dev.vendor,
                error = %e,
                "GPU cannot encode AV1 — excluding it from the encode pool (still usable for decode)"
            );
            false
        }
    };
    cache.lock().unwrap().insert(dev.index, capable);
    capable
}

fn create_backend(
    backend: EncoderBackend,
    config: EncoderConfig,
    gpus: &[gpu::GpuDevice],
) -> Result<Box<dyn Encoder>> {
    match backend {
        EncoderBackend::Nvenc => {
            let dev = pick_vendor_device(gpus, gpu::GpuVendor::Nvidia, config.gpu_index)
                .ok_or_else(|| match config.gpu_index {
                    Some(idx) => anyhow::anyhow!(
                        "NVENC requested on GPU index {idx} but no NVIDIA GPU with that index found"
                    ),
                    None => anyhow::anyhow!("NVENC requested but no NVIDIA GPU found"),
                })?;
            Ok(Box::new(nvenc::NvencEncoder::new(config, dev.index)?))
        }
        EncoderBackend::Amf => {
            let dev = pick_vendor_device(gpus, gpu::GpuVendor::Amd, config.gpu_index).ok_or_else(
                || match config.gpu_index {
                    Some(idx) => anyhow::anyhow!(
                        "AMF requested on GPU index {idx} but no AMD GPU with that index found"
                    ),
                    None => anyhow::anyhow!("AMF requested but no AMD GPU found"),
                },
            )?;
            Ok(Box::new(amf::AmfEncoder::new(config, dev.index)?))
        }
        EncoderBackend::Qsv => {
            let dev = pick_vendor_device(gpus, gpu::GpuVendor::Intel, config.gpu_index)
                .ok_or_else(|| match config.gpu_index {
                    Some(idx) => anyhow::anyhow!(
                        "QSV requested on GPU index {idx} but no Intel GPU with that index found"
                    ),
                    None => anyhow::anyhow!("QSV requested but no Intel GPU found"),
                })?;
            Ok(Box::new(qsv::QsvEncoder::new(config, dev.index)?))
        }
    }
}

#[cfg(test)]
mod gpu_selection_tests {
    use super::*;
    use crate::gpu::{GpuDevice, GpuVendor};

    fn synth(index: u32, vendor: GpuVendor) -> GpuDevice {
        GpuDevice {
            index,
            vendor,
            name: format!("synthetic-{index}"),
            generation: String::new(),
            pci_id: String::new(),
            vram_mib: 0,
            serial: None,
            host_pci_address: String::new(),
            vendor_id_hex: String::new(),
        }
    }

    #[test]
    fn pick_vendor_device_defaults_to_first_of_vendor_when_no_request() {
        // requested=None → first matching vendor wins (pre-multi-GPU
        // behaviour preserved).
        let gpus = vec![
            synth(0, GpuVendor::Nvidia),
            synth(1, GpuVendor::Nvidia),
            synth(2, GpuVendor::Amd),
        ];
        let nv = pick_vendor_device(&gpus, GpuVendor::Nvidia, None).unwrap();
        assert_eq!(nv.index, 0);
        let amd = pick_vendor_device(&gpus, GpuVendor::Amd, None).unwrap();
        assert_eq!(amd.index, 2);
    }

    #[test]
    fn pick_vendor_device_honours_explicit_request() {
        // requested=Some(1) + vendor=Nvidia → must find GPU with
        // index==1 AND vendor==Nvidia, not just first Nvidia.
        let gpus = vec![
            synth(0, GpuVendor::Nvidia),
            synth(1, GpuVendor::Nvidia),
            synth(2, GpuVendor::Nvidia),
        ];
        let dev = pick_vendor_device(&gpus, GpuVendor::Nvidia, Some(1)).unwrap();
        assert_eq!(dev.index, 1);
        let dev2 = pick_vendor_device(&gpus, GpuVendor::Nvidia, Some(2)).unwrap();
        assert_eq!(dev2.index, 2);
    }

    #[test]
    fn pick_vendor_device_returns_none_when_index_vendor_mismatch() {
        // requested=Some(2) + vendor=Nvidia but GPU 2 is AMD → None.
        // select_encoder then falls through to the AMD tier which will
        // find GPU 2 on its own find() pass.
        let gpus = vec![synth(0, GpuVendor::Nvidia), synth(2, GpuVendor::Amd)];
        assert!(pick_vendor_device(&gpus, GpuVendor::Nvidia, Some(2)).is_none());
        // Confirm the AMD tier finds it correctly with the same request.
        let dev = pick_vendor_device(&gpus, GpuVendor::Amd, Some(2)).unwrap();
        assert_eq!(dev.index, 2);
    }

    #[test]
    fn pick_vendor_device_no_gpus_returns_none() {
        let gpus: Vec<GpuDevice> = vec![];
        assert!(pick_vendor_device(&gpus, GpuVendor::Nvidia, None).is_none());
        assert!(pick_vendor_device(&gpus, GpuVendor::Nvidia, Some(0)).is_none());
    }

    #[test]
    fn encoder_config_default_has_no_gpu_pin() {
        // Default is None so existing callers using `EncoderConfig {
        // ..default() }` literals get the pre-multi-GPU first-of-vendor
        // behaviour unchanged.
        let cfg = EncoderConfig::default();
        assert_eq!(cfg.gpu_index, None);
    }
}
