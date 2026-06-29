//! AMD GPU detection — Windows via WMI, Linux via `/sys/bus/pci/devices`.

use super::types::{GpuDevice, GpuVendor};

pub(super) fn detect_amd() -> Vec<GpuDevice> {
    // Windows: enumerate AMD (PCI vendor 0x1002) via WMI.
    #[cfg(windows)]
    {
        return super::detect_windows_vendor(GpuVendor::Amd, 0x1002);
    }
    // Linux: check /sys/bus/pci/devices for AMD GPU (vendor 1002)
    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = std::fs::read_dir("/sys/bus/pci/devices") {
            let mut idx = 0u32;
            return entries
                .filter_map(|e| e.ok())
                .filter_map(|entry| {
                    let vendor_path = entry.path().join("vendor");
                    let class_path = entry.path().join("class");
                    let vendor = std::fs::read_to_string(&vendor_path).ok()?;
                    let class = std::fs::read_to_string(&class_path).ok()?;
                    // VGA (0x030000) or 3D controller (0x030200)
                    if vendor.trim() == "0x1002" && class.trim().starts_with("0x0302") {
                        let device_path = entry.path().join("device");
                        let device = std::fs::read_to_string(&device_path)
                            .unwrap_or_default()
                            .trim()
                            .to_string();
                        let after = device.trim_start_matches("0x");
                        let pci_id = format!("0x1002:0x{after}");
                        let vram_mib = super::sysfs::read_drm_vram_mib(&entry.path());
                        let generation = amd_generation_from_device_id(&device);
                        let host_pci_address =
                            super::sysfs::host_pci_address_from_sysfs(&entry.path());
                        let serial = super::sysfs::read_drm_serial(&entry.path());
                        let dev = GpuDevice {
                            vendor: GpuVendor::Amd,
                            name: format!("AMD GPU {device}"),
                            index: idx,
                            vendor_index: idx,
                            generation,
                            pci_id,
                            vram_mib,
                            serial,
                            host_pci_address,
                            vendor_id_hex: "0x1002".into(),
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

/// AMD generation lookup. RDNA3 (RX 7000) is the only generation we
/// have AV1 encode silicon on today; earlier (RDNA1/2/Polaris/Vega) +
/// later (RDNA4 announced) all surface in the inventory page so
/// operators know the lay of the fleet. PCI device ids cross-checked
/// against the upstream amdgpu driver's `pci_table` (drivers/gpu/drm/
/// amd/amdgpu/amdgpu_drv.c) for the families we expect to see.
pub(super) fn amd_generation_from_device_id(device_id: &str) -> String {
    let id_u16 = device_id
        .strip_prefix("0x")
        .and_then(|s| u16::from_str_radix(s, 16).ok());
    match id_u16 {
        // Navi 31 / 32 / 33 (RDNA3) — RX 7000 series.
        Some(id) if (0x7400..=0x74ff).contains(&id) => "RDNA3".into(),
        // Navi 21 / 22 / 23 / 24 (RDNA2) — RX 6000 series.
        Some(id) if (0x73a0..=0x73ff).contains(&id) => "RDNA2".into(),
        Some(id) if (0x7300..=0x73a0).contains(&id) => "RDNA2".into(),
        // Navi 10 / 14 (RDNA1) — RX 5000 series.
        Some(id) if (0x7310..=0x7350).contains(&id) => "RDNA1".into(),
        // Vega 10 / 20 (GCN5) — Vega 56/64, MI50/60.
        Some(id) if (0x6860..=0x687f).contains(&id) => "Vega".into(),
        // Polaris 10/11/12 (GCN4) — RX 400 / 500.
        Some(id) if (0x67c0..=0x67ff).contains(&id) => "Polaris".into(),
        Some(id) if (0x6980..=0x69ff).contains(&id) => "Polaris".into(),
        _ => "Unknown".into(),
    }
}
