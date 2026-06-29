//! Linux sysfs helpers for reading GPU attributes from `/sys` and `/proc/*/fdinfo`.

/// Read VRAM total (MiB) from sysfs for a DRM device. AMD's amdgpu
/// driver and Intel's i915 driver both expose `mem_info_vram_total`
/// inside the device dir for discrete cards; integrated SKUs (Intel
/// iGPU sharing system memory, AMD APUs) generally don't, in which
/// case we return 0 and the inventory page renders "—". Best-effort:
/// any read failure returns 0 silently.
#[cfg(target_os = "linux")]
pub(super) fn read_drm_vram_mib(device_path: &std::path::Path) -> u64 {
    // Path patterns (try in order):
    //   /sys/bus/pci/devices/<bdf>/mem_info_vram_total  (amdgpu)
    //   /sys/bus/pci/devices/<bdf>/drm/cardN/device/mem_info_vram_total
    //   /sys/bus/pci/devices/<bdf>/i915_capabilities (Intel; not VRAM)
    let direct = device_path.join("mem_info_vram_total");
    if let Ok(s) = std::fs::read_to_string(&direct) {
        if let Ok(bytes) = s.trim().parse::<u64>() {
            return bytes / 1024 / 1024;
        }
    }
    // Walk drm/cardN/device/mem_info_vram_total (one extra hop on
    // some kernel versions).
    let drm_dir = device_path.join("drm");
    if let Ok(entries) = std::fs::read_dir(&drm_dir) {
        for entry in entries.flatten() {
            let candidate = entry.path().join("device").join("mem_info_vram_total");
            if let Ok(s) = std::fs::read_to_string(&candidate) {
                if let Ok(bytes) = s.trim().parse::<u64>() {
                    return bytes / 1024 / 1024;
                }
            }
        }
    }
    0
}

#[cfg(not(target_os = "linux"))]
pub(super) fn read_drm_vram_mib(_device_path: &std::path::Path) -> u64 {
    0
}

/// Extract the host-readable PCI bus address (e.g. `04:00.0`) from
/// a sysfs device path. The sysfs path is normally
/// `/sys/bus/pci/devices/0000:04:00.0`; we want the last path
/// component minus the domain prefix, since the abbreviated form is
/// what `lspci` shows and what admins recognise. Empty string on
/// non-matching shapes (defensive).
#[cfg(target_os = "linux")]
pub(super) fn host_pci_address_from_sysfs(device_path: &std::path::Path) -> String {
    let Some(name) = device_path.file_name().and_then(|n| n.to_str()) else {
        return String::new();
    };
    // Sysfs PCI BDF format: <domain>:<bus>:<device>.<function>
    // e.g. "0000:04:00.0". Strip the leading "0000:" prefix when
    // present so the result matches the conventional 7-char form.
    if let Some(rest) = name.strip_prefix("0000:") {
        return rest.to_string();
    }
    name.to_string()
}

#[cfg(not(target_os = "linux"))]
pub(super) fn host_pci_address_from_sysfs(_device_path: &std::path::Path) -> String {
    String::new()
}

/// Best-effort serial-number read from sysfs. AMD / Intel cards
/// occasionally expose `serial_number` or `serial` under the device
/// dir; consumer cards usually don't. Empty result → `None`.
///
/// Treat the literal "0" sentinel the same as None (matching the NVML
/// behaviour documented in `nvmlDeviceGetSerial`: "0 is not a valid
/// serial for a nvidia card"). Some i915 / amdgpu code paths return
/// "0" when the hardware doesn't have a real serial fuse, and we
/// don't want to create asset rows keyed on that sentinel.
#[cfg(target_os = "linux")]
pub(super) fn read_drm_serial(device_path: &std::path::Path) -> Option<String> {
    for fname in &["serial_number", "serial"] {
        let path = device_path.join(fname);
        if let Ok(s) = std::fs::read_to_string(&path) {
            let trimmed = s.trim().to_string();
            if !trimmed.is_empty() && trimmed != "0" {
                return Some(trimmed);
            }
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
pub(super) fn read_drm_serial(_device_path: &std::path::Path) -> Option<String> {
    None
}

/// Resolve the PCI BDF (e.g. `0000:03:00.0`) backing a
/// `/sys/class/drm/cardN` entry. The `device` symlink under the card
/// dir always points to the PCI device node — the file_name segment
/// of the resolved path IS the BDF. Returns None on read_link failure
/// (non-PCI virtual GPUs etc.).
#[cfg(target_os = "linux")]
pub(super) fn read_pci_bdf_from_drm_card(card_dir: &std::path::Path) -> Option<String> {
    let target = std::fs::read_link(card_dir.join("device")).ok()?;
    target
        .file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.to_string())
}

/// Aggregate Intel VRAM bytes resident across every DRM client by
/// walking `/proc/*/fdinfo/*`. The kernel exposes per-fd accounting
/// in DRM fdinfo (i915 since ~5.19, xe driver since ~6.8); summing
/// `drm-resident-local0` across all clients gives the same number
/// `intel_gpu_top -J` reports for "VRAM used".
///
/// When `bdf_filter` is `Some(...)`, only fdinfo entries whose
/// `drm-pdev:` matches that BDF are counted — the multi-Intel case
/// (the dev box has Arc A750 + Arc A310 today) gets per-card
/// accounting instead of a cross-card total. When `None`, every
/// Intel client is summed.
///
/// Returns `None` when no Intel DRM clients are visible (rather than
/// `Some(0)`) so the caller can distinguish "no usage right now"
/// from "fdinfo path not available on this kernel" — the former
/// shouldn't trigger a different fallback, the latter could.
#[cfg(target_os = "linux")]
pub(super) fn read_intel_vram_resident_bytes(bdf_filter: Option<&str>) -> Option<u64> {
    let proc_dir = std::fs::read_dir("/proc").ok()?;
    let mut total_bytes: u64 = 0;
    let mut found_any_intel_client = false;

    for proc_entry in proc_dir.flatten() {
        let pid_name = proc_entry.file_name();
        let Some(pid_str) = pid_name.to_str() else {
            continue;
        };
        if !pid_str.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }
        let fdinfo_dir = proc_entry.path().join("fdinfo");
        let Ok(fd_entries) = std::fs::read_dir(&fdinfo_dir) else {
            continue;
        };

        for fd_entry in fd_entries.flatten() {
            // fdinfo files for non-DRM fds are short and have no
            // "drm-driver" key — read_to_string is cheap on those.
            // Permission errors on other-user processes also fall
            // through silently (the transcoder runs as root in our
            // production container, so this is rare in practice).
            let Ok(content) = std::fs::read_to_string(fd_entry.path()) else {
                continue;
            };
            if !content.contains("drm-driver:") {
                continue;
            }
            // Match either i915 (mainline Intel driver) or xe (newer
            // Intel driver shipping with kernel 6.8+; takes over Arc
            // discrete cards). Whitespace between key and value is a
            // single tab in i915's emitter and a single space in xe's
            // — accept both.
            let is_intel = content
                .lines()
                .filter_map(|l| l.strip_prefix("drm-driver:"))
                .any(|v| {
                    let v = v.trim();
                    v == "i915" || v == "xe"
                });
            if !is_intel {
                continue;
            }
            // Optional BDF filter — only count clients on the card we
            // care about. drm-pdev format is `drm-pdev: 0000:03:00.0`.
            if let Some(want_bdf) = bdf_filter {
                let matches = content
                    .lines()
                    .filter_map(|l| l.strip_prefix("drm-pdev:"))
                    .any(|v| v.trim() == want_bdf);
                if !matches {
                    continue;
                }
            }
            found_any_intel_client = true;
            // Sum drm-resident-local0 across the client. "local0" is
            // the i915/xe naming for the on-card VRAM region; values
            // are formatted as "<num> <unit>" with unit ∈ {B, KiB,
            // MiB, GiB} per drm-fdinfo.rst.
            for line in content.lines() {
                if let Some(rest) = line.strip_prefix("drm-resident-local0:") {
                    if let Some(bytes) = parse_drm_size(rest) {
                        total_bytes = total_bytes.saturating_add(bytes);
                    }
                }
            }
        }
    }

    if found_any_intel_client {
        Some(total_bytes)
    } else {
        None
    }
}

/// Parse a DRM fdinfo size value: `<number> <unit>` where unit is
/// one of B / KiB / MiB / GiB. Bare numbers are treated as bytes.
/// Returns None on garbage input.
#[cfg(target_os = "linux")]
fn parse_drm_size(s: &str) -> Option<u64> {
    let trimmed = s.trim();
    let mut parts = trimmed.split_whitespace();
    let num: u64 = parts.next()?.parse().ok()?;
    let unit = parts.next().unwrap_or("B");
    let multiplier: u64 = match unit {
        "B" | "" => 1,
        "KiB" => 1024,
        "MiB" => 1024 * 1024,
        "GiB" => 1024 * 1024 * 1024,
        _ => return None,
    };
    Some(num.saturating_mul(multiplier))
}
