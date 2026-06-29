//! Intel GPU detection — Windows via WMI, Linux via `/sys/bus/pci/devices`.

use super::types::{GpuDevice, GpuVendor};

pub(super) fn detect_intel() -> Vec<GpuDevice> {
    // Windows: enumerate Intel (PCI vendor 0x8086) via WMI.
    #[cfg(windows)]
    {
        return super::detect_windows_vendor(GpuVendor::Intel, 0x8086);
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = std::fs::read_dir("/sys/bus/pci/devices") {
            let mut idx = 0u32;
            return entries
                .filter_map(|e| e.ok())
                .filter_map(|entry| {
                    let vendor_path = entry.path().join("vendor");
                    let class_path = entry.path().join("class");
                    let device_path = entry.path().join("device");
                    let vendor = std::fs::read_to_string(&vendor_path).ok()?;
                    let class = std::fs::read_to_string(&class_path).ok()?;
                    if vendor.trim() == "0x8086" && class.trim().starts_with("0x0300") {
                        // Read the PCI device ID so we can label the GPU
                        // by family. Without this every Intel device was
                        // tagged "Intel Integrated GPU" — which made
                        // `supports_av1_encode`'s `contains("arc")`
                        // substring match miss the discrete Arc cards
                        // and silently route every job to rav1e CPU.
                        let device_id_str = std::fs::read_to_string(&device_path)
                            .ok()
                            .map(|s| s.trim().to_string())
                            .unwrap_or_default();
                        let name = intel_label_from_device_id(&device_id_str);
                        let pci_id = if device_id_str.starts_with("0x") {
                            format!("0x8086:{device_id_str}")
                        } else {
                            String::new()
                        };
                        // Prefer the live sysfs read (newer i915
                        // exposes total VRAM via mem_info_vram_total).
                        // Fall back to the static SKU catalog when the
                        // sysfs path is missing — the dev box's kernel
                        // is one of the older versions that doesn't
                        // export the field.
                        let live_vram = super::sysfs::read_drm_vram_mib(&entry.path());
                        let vram_mib = if live_vram > 0 {
                            live_vram
                        } else {
                            intel_vram_mib_from_device_id(&device_id_str)
                                .map(u64::from)
                                .unwrap_or(0)
                        };
                        let generation = intel_generation_from_device_id(&device_id_str);
                        let host_pci_address =
                            super::sysfs::host_pci_address_from_sysfs(&entry.path());
                        let serial = super::sysfs::read_drm_serial(&entry.path());
                        let dev = GpuDevice {
                            vendor: GpuVendor::Intel,
                            name,
                            index: idx,
                            vendor_index: idx,
                            generation,
                            pci_id,
                            vram_mib,
                            serial,
                            host_pci_address,
                            vendor_id_hex: "0x8086".into(),
                        };
                        idx += 1;
                        Some(dev)
                    } else {
                        None
                    }
                })
                .collect();
        }
    }
    Vec::new()
}

/// Intel generation lookup. Mirrors `intel_label_from_device_id` —
/// stays in lockstep so the inventory page's manufacturer / generation
/// rollup agrees with the per-row name shown elsewhere.
pub(super) fn intel_generation_from_device_id(device_id: &str) -> String {
    let id_u16 = device_id
        .strip_prefix("0x")
        .and_then(|s| u16::from_str_radix(s, 16).ok());
    match id_u16 {
        // Alchemist DG2 — entire 0x56xx range.
        Some(id) if (0x5690..=0x56af).contains(&id) => "Alchemist DG2".into(),
        // Battlemage BMG-G21 — 0xe200..=0xe21f.
        Some(id) if (0xe200..=0xe21f).contains(&id) => "Battlemage BMG".into(),
        // Lunar Lake Xe2 iGPU.
        Some(id) if (0x6420..=0x643f).contains(&id) => "Lunar Lake".into(),
        // Meteor Lake Xe-LP iGPU.
        Some(id) if (0x7d40..=0x7d6f).contains(&id) => "Meteor Lake".into(),
        // Older iGPU families surface in the inventory but have no
        // AV1 encode silicon — labelled by family for fleet visibility.
        Some(id) if (0xa780..=0xa7ff).contains(&id) => "Raptor Lake".into(),
        Some(id) if (0x4680..=0x46ff).contains(&id) => "Alder Lake".into(),
        Some(id) if (0x9a00..=0x9aff).contains(&id) => "Tiger Lake".into(),
        _ => "Unknown".into(),
    }
}

/// Map an Intel PCI device id (`0xNNNN`) to a human-readable label.
/// Discrete Arc GPUs (Alchemist DG2, Battlemage BMG) are SKU-specific
/// where the device id is well-known so admins can tell A310 from A750
/// in the inventory log; family-level for unknown variants. Meteor Lake
/// / Lunar Lake / Arrow Lake iGPUs are family-level only (the AV1 QSV
/// silicon is a property of the family, not the SKU).
///
/// Device-id table cross-checked against
/// `i915_pci_ids.h` / `xe_pci.c` in upstream kernel
/// (`drivers/gpu/drm/i915/i915_pciids.h` for DG2 + BMG entries).
/// Catalog VRAM total in MiB for known Intel discrete SKUs. The
/// i915 driver on the dev box's kernel doesn't expose
/// `/sys/class/drm/card*/device/mem_info_vram_total` — that path was
/// added later — so the live read returns zero for both Arc cards.
/// Fall back to a static SKU table so the inventory page can at least
/// display "this is a 4 GB card vs an 8 GB card" without depending on
/// kernel introspection. Live `mem_used_mib` stays 0 until i915_pmu /
/// intel_gpu_top wiring lands; that's a separate task.
///
/// A770 has both 8 GB and 16 GB Limited Edition variants under the
/// same PCI device id (0x56a0). Discriminating requires the subsystem
/// device id; for our inventory display we report the more common
/// 8 GB SKU and accept the LE undercount as a known limitation.
fn intel_vram_mib_from_device_id(device_id: &str) -> Option<u32> {
    let id_u16 = device_id
        .strip_prefix("0x")
        .and_then(|s| u16::from_str_radix(s, 16).ok())?;
    Some(match id_u16 {
        // Alchemist DG2-128 (small die)
        0x56a5 => 6 * 1024, // A380
        0x56a6 => 4 * 1024, // A310
        0x5693 => 4 * 1024, // A350M
        // Alchemist DG2-512 (full die)
        0x56a0 => 8 * 1024,  // A770 (8 GB; 16 GB LE shares this id)
        0x56a1 => 8 * 1024,  // A750
        0x56a2 => 8 * 1024,  // A580
        0x5690 => 16 * 1024, // A770M (16 GB common spec)
        0x5691 => 12 * 1024, // A730M
        0x5692 => 8 * 1024,  // A550M
        // Battlemage
        0xe20b => 12 * 1024, // B580
        0xe20c => 10 * 1024, // B570
        // Unknown DG2 / BMG SKUs — the catalog doesn't help here, return None
        _ => return None,
    })
}

fn intel_label_from_device_id(device_id: &str) -> String {
    let id_u16 = device_id
        .strip_prefix("0x")
        .and_then(|s| u16::from_str_radix(s, 16).ok());
    match id_u16 {
        // Alchemist / DG2 discrete — per-SKU mapping.
        // DG2-128 (small die): A310 / A380 / A350M.
        Some(0x56a5) => "Intel Arc A380".into(),
        Some(0x56a6) => "Intel Arc A310".into(),
        Some(0x5693) => "Intel Arc A350M".into(),
        // DG2-512 (full die): A580 / A750 / A770 + mobile A550M..A770M.
        Some(0x56a0) => "Intel Arc A770".into(),
        Some(0x56a1) => "Intel Arc A750".into(),
        Some(0x56a2) => "Intel Arc A580".into(),
        Some(0x5690) => "Intel Arc A770M".into(),
        Some(0x5691) => "Intel Arc A730M".into(),
        Some(0x5692) => "Intel Arc A550M".into(),
        // Any other device id in the DG2-reserved 0x56xx range — likely
        // a future SKU or a workstation Pro variant we haven't tagged.
        // Family-level fallback so AV1 dispatch still picks it up via
        // the `contains("alchemist")` substring match.
        Some(id) if (0x5690..=0x56af).contains(&id) => {
            format!("Intel Arc Alchemist (DG2 0x{id:04x})")
        }
        // Battlemage BMG-G21 discrete — per-SKU.
        Some(0xe20b) => "Intel Arc B580".into(),
        Some(0xe20c) => "Intel Arc B570".into(),
        Some(id) if (0xe200..=0xe21f).contains(&id) => {
            format!("Intel Arc Battlemage (BMG 0x{id:04x})")
        }
        // Lunar Lake Xe2 iGPU (Core Ultra 2xx mobile) — has AV1 encode.
        Some(id) if (0x6420..=0x643f).contains(&id) => "Intel Lunar Lake iGPU".into(),
        // Meteor Lake Xe-LP iGPU (Core Ultra 1xx mobile) — has AV1 encode.
        Some(id) if (0x7d40..=0x7d6f).contains(&id) => "Intel Meteor Lake iGPU".into(),
        // Anything else is some flavour of older iGPU (Coffee Lake → DG1
        // → Tiger Lake → Alder Lake → Raptor Lake) that decodes plenty
        // of formats but doesn't have AV1 QSV.
        Some(id) => format!("Intel iGPU 0x{id:04x}"),
        None => "Intel GPU".into(),
    }
}
