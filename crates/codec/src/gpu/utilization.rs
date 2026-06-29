//! `GpuUtilizationReader` — live per-GPU utilisation snapshots via NVML / sysfs.

use super::types::{GpuDevice, GpuVendor, GpuUtilization};

/// One-shot accumulator that opens NVML once and reads per-GPU
/// utilisation for every NVIDIA device on each load tick. Holding
/// the NVML handle across reads avoids the init cost
/// (microseconds) on every tick and is the documented pattern.
pub struct GpuUtilizationReader {
    nvml: Option<nvml_wrapper::Nvml>,
}

impl GpuUtilizationReader {
    /// Build a reader. NVML init failure is non-fatal — the reader
    /// folds to "all zeroes" on every NVIDIA device and the rest of
    /// the load-tick path stays alive. Logged once at startup so
    /// operators can tell "no NVIDIA card" from "NVIDIA card but
    /// driver missing".
    pub fn new() -> Self {
        let nvml = match super::nvidia::init_nvml_with_fallback() {
            Ok(n) => Some(n),
            Err(e) => {
                // info-level: many production hosts are AMD/Intel-only
                // and this isn't a problem. Operators looking at the
                // dev box logs see this once at boot.
                tracing::info!(error = %e, "nvml not available; NVIDIA GPU utilisation will be 0");
                None
            }
        };
        Self { nvml }
    }

    /// Read the per-tick snapshot for one device. Cheap when NVML is
    /// available (handful of FFI calls); free when it's not (returns
    /// the zero-initialised default).
    pub fn read(&self, device: &GpuDevice) -> GpuUtilization {
        match device.vendor {
            GpuVendor::Nvidia => self.read_nvidia(device).unwrap_or_default(),
            GpuVendor::Intel => self.read_intel(device).unwrap_or_default(),
            GpuVendor::Amd => GpuUtilization::default(),
        }
    }

    fn read_nvidia(&self, device: &GpuDevice) -> Option<GpuUtilization> {
        let nvml = self.nvml.as_ref()?;
        let dev = nvml.device_by_index(device.index).ok()?;
        let util = dev.utilization_rates().ok();
        // EncoderUtilizationInfo / DecoderUtilizationInfo have a
        // `utilization` field (0..=100) plus a sampling period; we
        // surface only the percentage.
        let enc = dev.encoder_utilization().ok();
        let dec = dev.decoder_utilization().ok();
        let mem = dev.memory_info().ok();
        let temp = dev
            .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
            .ok()
            .and_then(|t| u8::try_from(t).ok());
        Some(GpuUtilization {
            util_percent: util.as_ref().map(|u| u.gpu.min(100) as u8).unwrap_or(0),
            encoder_percent: enc
                .as_ref()
                .map(|e| e.utilization.min(100) as u8)
                .unwrap_or(0),
            decoder_percent: dec
                .as_ref()
                .map(|d| d.utilization.min(100) as u8)
                .unwrap_or(0),
            mem_used_mib: mem
                .as_ref()
                .map(|m| (m.used / 1024 / 1024) as u32)
                .unwrap_or(0),
            mem_total_mib: mem
                .as_ref()
                .map(|m| (m.total / 1024 / 1024) as u32)
                .unwrap_or(device.vram_mib as u32),
            temperature_c: temp,
        })
    }

    /// Intel stand-in via sysfs `gt_cur_freq_mhz` / `gt_max_freq_mhz`
    /// for a coarse "busy" proxy and `mem_info_vram_used` for memory.
    /// The i915 driver doesn't expose per-engine busy% via sysfs
    /// cleanly — `intel_gpu_top -J` is the proper source but the
    /// fork+capture cost on every 5 s tick is heavy. Phase 1: leave
    /// encoder/decoder at 0 and let `util_percent` be the freq-ratio
    /// proxy; real fix is the perf event interface (`i915_pmu`)
    /// which deserves its own task.
    #[cfg(target_os = "linux")]
    fn read_intel(&self, _device: &GpuDevice) -> Option<GpuUtilization> {
        // We don't have the bdf here, so walk /sys/class/drm/cardN
        // for an Intel card. Index 0 returns the first one that
        // matches; multi-Intel hosts (rare today) get the same
        // utilisation reported across both — acceptable until the
        // proper i915_pmu integration lands.
        let mut out = GpuUtilization::default();
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let Some(name_str) = name.to_str() else {
                    continue;
                };
                if !name_str.starts_with("card") || name_str.contains('-') {
                    continue;
                }
                // Confirm Intel via vendor file under device link.
                let device_link = entry.path().join("device").join("vendor");
                let vendor = std::fs::read_to_string(&device_link).unwrap_or_default();
                if vendor.trim() != "0x8086" {
                    continue;
                }
                let cur = std::fs::read_to_string(entry.path().join("gt_cur_freq_mhz"))
                    .ok()
                    .and_then(|s| s.trim().parse::<u32>().ok());
                let max = std::fs::read_to_string(entry.path().join("gt_max_freq_mhz"))
                    .ok()
                    .and_then(|s| s.trim().parse::<u32>().ok());
                if let (Some(cur), Some(max)) = (cur, max) {
                    if max > 0 {
                        out.util_percent = ((cur as u64 * 100 / max as u64).min(100)) as u8;
                    }
                }
                let used = std::fs::read_to_string(
                    entry.path().join("device").join("mem_info_vram_used"),
                )
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok());
                let total = std::fs::read_to_string(
                    entry.path().join("device").join("mem_info_vram_total"),
                )
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok());
                if let Some(u) = used {
                    out.mem_used_mib = (u / 1024 / 1024) as u32;
                }
                if let Some(t) = total {
                    out.mem_total_mib = (t / 1024 / 1024) as u32;
                }
                // Fall back to the catalog VRAM total stored on the
                // device record when sysfs didn't expose it. The dev
                // box's kernel doesn't have mem_info_vram_total, so
                // without this Intel cards report 0 / 0 forever.
                if out.mem_total_mib == 0 && _device.vram_mib > 0 {
                    out.mem_total_mib = _device.vram_mib as u32;
                }
                // Fall back to DRM fdinfo aggregation when sysfs didn't
                // expose `mem_info_vram_used` (older kernels). Filtered
                // to this card's PCI BDF so multi-Intel hosts report
                // per-device used memory, not the cross-card total.
                // This is the same source `intel_gpu_top -J` and `nvtop`
                // use, available since kernel ~5.19 (i915) / ~6.8 (xe).
                if out.mem_used_mib == 0 {
                    let bdf = super::sysfs::read_pci_bdf_from_drm_card(&entry.path());
                    if let Some(bytes) =
                        super::sysfs::read_intel_vram_resident_bytes(bdf.as_deref())
                    {
                        out.mem_used_mib = (bytes / 1024 / 1024) as u32;
                    }
                }
                return Some(out);
            }
        }
        if out.mem_total_mib == 0 && _device.vram_mib > 0 {
            out.mem_total_mib = _device.vram_mib as u32;
        }
        Some(out)
    }

    #[cfg(not(target_os = "linux"))]
    fn read_intel(&self, _device: &GpuDevice) -> Option<GpuUtilization> {
        // Windows path for Intel hosts is performance-counter via
        // the WMI `Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine`
        // surface — same fork-cost concern as `intel_gpu_top` on
        // Linux, deferred. Returns all zeroes.
        Some(GpuUtilization::default())
    }
}

impl Default for GpuUtilizationReader {
    fn default() -> Self {
        Self::new()
    }
}
