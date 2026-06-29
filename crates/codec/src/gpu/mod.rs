//! GPU device enumeration for NVDEC/NVENC scheduling.
//!
//! NVIDIA detection loads libcuda via dlopen, calls cuInit +
//! cuDeviceGetCount + cuDeviceGetName. This works on minimal container
//! images where the `nvidia-smi` binary may be absent but the driver's
//! user-mode libraries are bind-mounted by the NVIDIA Container Toolkit.
//! AMD/Intel detection scans /sys/bus/pci/devices on Linux.

mod amd;
mod intel;
mod nvidia;
mod sysfs;
mod types;
mod utilization;

pub use types::{GpuDevice, GpuUtilization, GpuVendor};
pub use utilization::GpuUtilizationReader;

pub fn detect_gpus() -> Vec<GpuDevice> {
    let mut devices = Vec::new();
    devices.extend(nvidia::detect_nvidia());
    devices.extend(amd::detect_amd());
    devices.extend(intel::detect_intel());
    // Each detect_* numbers its own vendor from 0 (kept as `vendor_index`).
    // Assign the GLOBAL `index` here so a mixed host (e.g. NVIDIA + AMD iGPU)
    // gets unique, user-addressable indices instead of colliding 0s.
    for (i, d) in devices.iter_mut().enumerate() {
        d.index = i as u32;
    }
    devices
}

/// Human-readable manufacturer label. Used by the WS hello frame's
/// `WsGpuInfo.manufacturer` field and by the admin inventory page's
/// "by manufacturer" rollup. Stays in lockstep with `vendor_label` in
/// `transcoder/src/capabilities.rs` so the registration POST + the
/// hello frame agree on the spelling.
pub fn manufacturer_label(v: GpuVendor) -> &'static str {
    match v {
        GpuVendor::Nvidia => "NVIDIA",
        GpuVendor::Amd => "AMD",
        GpuVendor::Intel => "Intel",
    }
}

pub fn has_nvidia() -> bool {
    !nvidia::detect_nvidia().is_empty()
}

pub fn supports_av1_encode(device: &GpuDevice) -> bool {
    match device.vendor {
        // NVIDIA: defer to the **real driver capability query**, not a
        // board-name list. The substring list this used to carry was brittle —
        // every new SKU had to be added by hand, and a missed one (e.g. the
        // RTX 5060 once was) now *hard-fails* the job since there's no CPU
        // fallback. NVENC AV1 support is authoritatively validated by
        // `nvEncGetEncodeCaps` / `GetEncodeGUIDs` in `NvencEncoder::new`, which
        // enumerates the GPU's actual encode codecs and bails cleanly if AV1
        // isn't among them (verified on an RTX 3090: "2 codec(s), none AV1").
        // So admit every NVIDIA GPU here and let the real query be the gate.
        GpuVendor::Nvidia => true,
        // AMD: defer to the real path. AV1 VCN encode is RDNA3+ (RX 7000+), but
        // rather than a brittle SKU list, `AmfEncoder::new` is authoritative —
        // AMF `CreateComponent(AMFVideoEncoderVCN_AV1)` fails on a pre-RDNA3 GPU
        // and we bail cleanly ("RDNA3+ GPU required"). Admit every AMD GPU here
        // and let that decide (matches the NVIDIA policy above).
        GpuVendor::Amd => true,
        // Intel: defer to the real path. AV1 QSV is Arc / Meteor Lake+, but
        // rather than a brittle family-name list, `QsvEncoder::new` is
        // authoritative — `MFXVideoENCODE_Query` (+ Init) reports whether the
        // GPU's oneVPL implementation supports AV1, and we bail cleanly if not.
        // Admit every Intel GPU here and let that decide (matches NVIDIA/AMD).
        GpuVendor::Intel => true,
    }
}

// ─── Windows helpers ─────────────────────────────────────────────────────────

/// Enumerate the host's video controllers on Windows via WMI
/// (`Get-CimInstance Win32_VideoController`). Cached for the process — the query
/// spawns PowerShell (~hundreds of ms) and the GPU set is stable per run.
/// Returns `(name, vendor_id, device_id)` per controller; empty if PowerShell
/// is unavailable. The Linux paths read `/sys` directly and don't use this.
#[cfg(windows)]
fn windows_video_controllers() -> &'static [(String, u16, u16)] {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Vec<(String, u16, u16)>> = OnceLock::new();
    CACHE.get_or_init(|| {
        let output = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_VideoController | \
                 ForEach-Object { \"$($_.Name)|$($_.PNPDeviceID)\" }",
            ])
            .output();
        let Ok(output) = output else {
            return Vec::new();
        };
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| {
                let (name, pnp) = line.split_once('|')?;
                let vendor_id = win_hex_after(pnp, "VEN_")?;
                let device_id = win_hex_after(pnp, "DEV_").unwrap_or(0);
                Some((name.trim().to_string(), vendor_id, device_id))
            })
            .collect()
    })
}

/// Extract the 4 hex digits following `marker` (e.g. `VEN_1002`) from a Windows
/// PNPDeviceID, as a `u16`.
#[cfg(windows)]
fn win_hex_after(s: &str, marker: &str) -> Option<u16> {
    let start = s.find(marker)? + marker.len();
    let hex: String = s[start..].chars().take(4).collect();
    u16::from_str_radix(&hex, 16).ok()
}

/// Build a `GpuDevice` list from the Windows controllers of one PCI vendor.
/// `vendor_index`/`index` are vendor-local here; `detect_gpus()` reassigns the
/// global `index`.
#[cfg(windows)]
fn detect_windows_vendor(vendor: GpuVendor, vendor_id: u16) -> Vec<GpuDevice> {
    let vendor_hex = format!("0x{vendor_id:04x}");
    windows_video_controllers()
        .iter()
        .filter(|(_, vid, _)| *vid == vendor_id)
        .enumerate()
        .map(|(idx, (name, _vid, did))| {
            let device = format!("0x{did:04x}");
            let generation = match vendor {
                GpuVendor::Amd => amd::amd_generation_from_device_id(&device),
                GpuVendor::Intel => intel::intel_generation_from_device_id(&device),
                GpuVendor::Nvidia => "Unknown".into(),
            };
            GpuDevice {
                vendor,
                name: name.clone(),
                index: idx as u32,
                vendor_index: idx as u32,
                generation,
                pci_id: format!("{vendor_hex}:{device}"),
                vram_mib: 0, // WMI AdapterRAM is u32-capped + unreliable
                serial: None,
                host_pci_address: String::new(),
                vendor_id_hex: vendor_hex.clone(),
            }
        })
        .collect()
}
