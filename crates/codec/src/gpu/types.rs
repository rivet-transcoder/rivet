//! Core GPU type definitions: `GpuDevice`, `GpuVendor`, `GpuUtilization`.

#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub vendor: GpuVendor,
    pub name: String,
    /// Global device index across ALL vendors (0-based, in `detect_gpus()`
    /// order) — what the user addresses via `--decode-gpu N` / the GPU policy.
    pub index: u32,
    /// Index of this device WITHIN its own vendor's set (0-based). On a
    /// single-vendor host this equals `index`; on a mixed host (e.g. NVIDIA +
    /// AMD iGPU) they differ. The per-vendor hardware decoder/encoder uses THIS
    /// to pick the physical adapter (CUDA ordinal, QSV/AMF adapter), since those
    /// SDKs enumerate only their own vendor's devices.
    pub vendor_index: u32,
    /// Architecture / generation label, e.g. "Blackwell" (RTX 5060),
    /// "Ada Lovelace" (RTX 4000-series), "Ampere" (RTX 3000), "Alchemist DG2"
    /// (Arc A-series), "Battlemage BMG" (Arc B-series), "RDNA3" (RX 7000).
    /// Phase 2 (2026-05-07) inventory page surface — derived from the PCI
    /// device id at detect time so the inventory aggregations don't have
    /// to re-derive it. "Unknown" when the device id falls outside the
    /// per-vendor known-id table; preserved verbatim to the admin UI so
    /// operators can spot fleet rows that need a label update.
    pub generation: String,
    /// Lowercase `vendor:device` PCI tuple, e.g. `"0x10de:0x2d05"`. Stable
    /// identifier across driver / kernel versions. Empty string when the
    /// platform path doesn't expose a device id (NVIDIA via CUDA on
    /// Windows: cuda doesn't surface PCI; the field stays empty rather
    /// than synthesise something misleading).
    pub pci_id: String,
    /// Total VRAM in MiB. NVIDIA via NVML `memory_info().total`; Intel via
    /// `/sys/class/drm/cardN/device/mem_info_vram_total` when present;
    /// AMD same. 0 when the platform path can't read it — admin UI shows
    /// "—" for that case rather than "0 MiB".
    pub vram_mib: u64,
    /// Vendor-reported serial number of the physical card. NVIDIA via
    /// NVML `Device::serial()` (returns the manufacturer's serial sticker
    /// for cards that have one — datacenter Tesla / A10G / consumer Pro
    /// cards expose it; consumer GeForce typically doesn't). Intel /
    /// AMD: try `/sys/class/drm/cardN/device/serial[_number]` paths;
    /// usually `None`. Stable identifier for warranty tracking + the
    /// `transcoder_gpus` asset table — when present, the same card
    /// across host moves dedups to a single row.
    pub serial: Option<String>,
    /// PCI host slot address, e.g. `"04:00.0"`. Used as the dedupe
    /// fallback when `serial` is absent — assumes the card stays in
    /// the same slot of the same host (the dev-box reality).
    /// Empty when the platform path doesn't expose it.
    pub host_pci_address: String,
    /// Vendor portion of the PCI tuple as a standalone hex string,
    /// e.g. `"0x10de"`. Already implicit in `pci_id` but exposed
    /// separately so the SQL inventory query can index on it
    /// without parsing.
    pub vendor_id_hex: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
}

/// Per-GPU live utilisation snapshot. Read on every load tick (5 s
/// cadence) by the Phase 2 (2026-05-07) `worker_load` reporter and
/// folded into the `WsGpuLeaseEntry` for the wire. NVIDIA values come
/// from NVML; Intel values come from sysfs `gt_cur_freq_mhz` /
/// `gt_max_freq_mhz` for a coarse "busy" proxy + `mem_info_vram_*`
/// for memory; AMD is currently a no-op (returns all zeros) — radeontop
/// / `amdsmi` integration is the proper fix and is deferred per the
/// brief's "Phase 1 stand-in for Intel; AMD skipped" guidance.
#[derive(Debug, Clone, Default)]
pub struct GpuUtilization {
    /// 0..=100 compute / overall GPU busy.
    pub util_percent: u8,
    /// 0..=100 NVENC ASIC busy (encoder pipeline).
    pub encoder_percent: u8,
    /// 0..=100 NVDEC ASIC busy (decoder pipeline).
    pub decoder_percent: u8,
    /// VRAM in use (MiB).
    pub mem_used_mib: u32,
    /// VRAM total (MiB) — duplicated from the static device record so
    /// the wire entry is self-contained for the FE bar render.
    pub mem_total_mib: u32,
    /// Core temperature in °C; `None` when the platform path doesn't
    /// expose it.
    pub temperature_c: Option<u8>,
}
