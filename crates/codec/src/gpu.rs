//! GPU device enumeration for NVDEC/NVENC scheduling.
//!
//! NVIDIA detection loads libcuda via dlopen, calls cuInit +
//! cuDeviceGetCount + cuDeviceGetName. This works on minimal container
//! images where the `nvidia-smi` binary may be absent but the driver's
//! user-mode libraries are bind-mounted by the NVIDIA Container Toolkit.
//! AMD/Intel detection scans /sys/bus/pci/devices on Linux.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::ptr;

#[derive(Debug, Clone)]
pub struct GpuDevice {
    pub vendor: GpuVendor,
    pub name: String,
    pub index: u32,
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

pub fn detect_gpus() -> Vec<GpuDevice> {
    let mut devices = Vec::new();
    devices.extend(detect_nvidia());
    devices.extend(detect_amd());
    devices.extend(detect_intel());
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
    !detect_nvidia().is_empty()
}

// ─── NVIDIA via libcuda dlopen ─────────────────────────────────────
type CUresult = c_int;
type CUdevice = c_int;

type FnCuInit = unsafe extern "C" fn(c_uint) -> CUresult;
type FnCuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CUresult;
type FnCuDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type FnCuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;

fn detect_nvidia() -> Vec<GpuDevice> {
    // Try the usual driver library names across Linux / Windows.
    let lib = unsafe { libloading::Library::new("libcuda.so") }
        .or_else(|_| unsafe { libloading::Library::new("libcuda.so.1") })
        .or_else(|_| unsafe { libloading::Library::new("nvcuda.dll") });

    let Ok(lib) = lib else { return Vec::new() };

    unsafe {
        let cu_init: libloading::Symbol<FnCuInit> = match lib.get(b"cuInit") {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        // Initialization flag is reserved — must be zero.
        if cu_init(0) != 0 {
            return Vec::new();
        }

        let cu_device_get_count: libloading::Symbol<FnCuDeviceGetCount> =
            match lib.get(b"cuDeviceGetCount") {
                Ok(f) => f,
                Err(_) => return Vec::new(),
            };
        let mut count: c_int = 0;
        if cu_device_get_count(&mut count) != 0 || count <= 0 {
            return Vec::new();
        }

        let cu_device_get: libloading::Symbol<FnCuDeviceGet> = match lib.get(b"cuDeviceGet") {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        let cu_device_get_name: libloading::Symbol<FnCuDeviceGetName> =
            match lib.get(b"cuDeviceGetName") {
                Ok(f) => f,
                Err(_) => return Vec::new(),
            };

        let mut devices = Vec::with_capacity(count as usize);
        for ordinal in 0..count {
            let mut dev: CUdevice = 0;
            if cu_device_get(&mut dev, ordinal) != 0 {
                continue;
            }
            let mut name_buf = [0i8; 256];
            let name = if cu_device_get_name(
                name_buf.as_mut_ptr() as *mut c_char,
                name_buf.len() as c_int,
                dev,
            ) == 0
            {
                CStr::from_ptr(name_buf.as_ptr() as *const c_char)
                    .to_string_lossy()
                    .into_owned()
            } else {
                format!("NVIDIA GPU {ordinal}")
            };
            // Phase 2 (2026-05-07) richer inventory: try to enrich
            // via NVML for VRAM total + PCI id + serial + bus address +
            // generation. NVML failure (driver missing, NVML so/dll
            // absent) leaves those fields empty/zero; the
            // cuda-reported `name` is still authoritative for the
            // substring-based AV1 dispatch in supports_av1_encode.
            let nvml_lookup = nvidia_nvml_lookup(ordinal as u32);
            let generation = nvidia_generation_from_name(&name);
            devices.push(GpuDevice {
                vendor: GpuVendor::Nvidia,
                name,
                index: ordinal as u32,
                generation,
                pci_id: nvml_lookup.pci_id,
                vram_mib: nvml_lookup.vram_mib,
                serial: nvml_lookup.serial,
                host_pci_address: nvml_lookup.host_pci_address,
                vendor_id_hex: "0x10de".into(),
            });
        }
        // Silence unused-import warnings from the libloading bounds checks
        let _ = ptr::null::<c_void>();
        devices
    }
}

/// Initialize NVML, trying both the unversioned and SONAME-versioned
/// library names. The default `Nvml::init()` dlopens `libnvidia-ml.so`
/// (no suffix) — but the NVIDIA Container Toolkit only mounts
/// `libnvidia-ml.so.1` into containers, with no unversioned alias.
/// On the dev box we observed the bare `init()` failing with
/// "cannot open shared object file" while the `.so.1` was present.
/// Fall back to the explicit SONAME path; if both fail, the caller
/// folds to "no NVML available" same as before.
fn init_nvml_with_fallback() -> Result<nvml_wrapper::Nvml, nvml_wrapper::error::NvmlError> {
    match nvml_wrapper::Nvml::init() {
        Ok(n) => Ok(n),
        Err(_) => nvml_wrapper::Nvml::builder()
            .lib_path(std::ffi::OsStr::new("libnvidia-ml.so.1"))
            .init(),
    }
}

/// NVML lookup result. Bundled into a struct so the call site stays
/// self-documenting as we add fields for the Phase 2 (2026-05-07)
/// inventory + asset-table extension.
#[derive(Debug, Clone, Default)]
struct NvmlLookup {
    pci_id: String,
    vram_mib: u64,
    serial: Option<String>,
    host_pci_address: String,
}

/// NVML lookup helper. Returns enrichment fields for the given CUDA
/// ordinal. NVML's device handle indexing matches CUDA's ordinal in
/// the common case (`CUDA_VISIBLE_DEVICES` empty / unset); mismatches
/// are tolerated by returning all defaults on `device_by_index`
/// errors, which the caller folds to empty/None.
///
/// NVML init is performed inside this function and torn down on return —
/// repeated lookups during device enumeration share the same NVML
/// process across the loop body via the `Nvml::init` call cost
/// (microseconds) rather than holding a long-lived handle in static
/// storage. Cross-platform: the `nvml-wrapper` crate dlopens
/// `libnvidia-ml.so.1` on Linux and `nvml.dll` on Windows, same shape
/// as our existing `libcuda` libloading path.
fn nvidia_nvml_lookup(ordinal: u32) -> NvmlLookup {
    let nvml = match init_nvml_with_fallback() {
        Ok(n) => n,
        Err(_) => return NvmlLookup::default(),
    };
    let device = match nvml.device_by_index(ordinal) {
        Ok(d) => d,
        Err(_) => return NvmlLookup::default(),
    };
    let (pci_id, host_pci_address) = match device.pci_info() {
        Ok(p) => {
            let id = format!(
                "0x{:04x}:0x{:04x}",
                p.pci_device_id >> 16,
                p.pci_device_id & 0xFFFF
            );
            // bus_id format: "00000000:04:00.0". Strip the leading
            // 0000-domain so the abbreviation matches the lspci /
            // /sys/bus/pci/devices/<bdf> form admins recognise.
            let bus = p
                .bus_id
                .trim_start_matches('0')
                .trim_start_matches(':')
                .to_string();
            // If trimming above ate too much (single-domain "0000:..."),
            // fall back to the raw bus_id; defensive against leading-zero
            // pathological cases.
            let host_pci = if bus.is_empty() {
                p.bus_id.clone()
            } else {
                bus
            };
            (id, host_pci)
        }
        Err(_) => (String::new(), String::new()),
    };
    let vram_mib = match device.memory_info() {
        Ok(m) => m.total / 1024 / 1024,
        Err(_) => 0,
    };
    // `serial()` returns Err for cards without a serial sticker
    // (consumer GeForce typically; datacenter Tesla / A10G expose it).
    // Don't fail the worker — debug-log + None per coordinator's
    // "graceful failure" guidance.
    //
    // Per NVML docs: "0 is not a valid serial for a nvidia card."
    // Some consumer cards / driver-fallback paths return literal "0"
    // instead of erroring. Treat that as None too so we don't
    // mistakenly create asset rows keyed on a sentinel value.
    let serial = match device.serial() {
        Ok(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() || trimmed == "0" {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Err(e) => {
            tracing::debug!(error = %e, ordinal, "nvml serial unavailable");
            None
        }
    };
    NvmlLookup {
        pci_id,
        vram_mib,
        serial,
        host_pci_address,
    }
}

/// NVIDIA generation lookup by marketing name substring. Matches
/// the same convention `supports_av1_encode` uses (lowercase substring
/// match) so the two stay in lockstep. Order matters: the more
/// specific datacenter SKUs (B100/B200) are matched before the
/// looser consumer family (5xxx) to avoid "B5060" — not a real SKU
/// today, but defensive.
fn nvidia_generation_from_name(name: &str) -> String {
    let n = name.to_lowercase();
    // Blackwell consumer (RTX 50xx) + datacenter (B100/B200/GB200).
    if n.contains("rtx 50")
        || n.contains("5050")
        || n.contains("5060")
        || n.contains("5070")
        || n.contains("5080")
        || n.contains("5090")
        || n.contains("b100")
        || n.contains("b200")
        || n.contains("gb200")
    {
        return "Blackwell".into();
    }
    // Hopper datacenter (H100/H200). No NVENC silicon — surfaces in
    // the inventory page so operators don't try to schedule encodes.
    if n.contains("h100") || n.contains("h200") {
        return "Hopper".into();
    }
    // Ada Lovelace: RTX 40xx + L4/L40 datacenter.
    if n.contains("rtx 40")
        || n.contains("4060")
        || n.contains("4070")
        || n.contains("4080")
        || n.contains("4090")
        || n.contains("ada")
        || n.contains("l4")
        || n.contains("l40")
    {
        return "Ada Lovelace".into();
    }
    // Ampere: RTX 30xx + A10/A10G/A100.
    if n.contains("rtx 30")
        || n.contains("3050")
        || n.contains("3060")
        || n.contains("3070")
        || n.contains("3080")
        || n.contains("3090")
        || n.contains("a10")
        || n.contains("a100")
        || n.contains("ampere")
    {
        return "Ampere".into();
    }
    // Turing: RTX 20xx + T4.
    if n.contains("rtx 20")
        || n.contains("2060")
        || n.contains("2070")
        || n.contains("2080")
        || n.contains(" t4")
        || n.contains("turing")
    {
        return "Turing".into();
    }
    // Pascal: GTX 10xx + P100/P40/P4.
    if n.contains("gtx 10")
        || n.contains("1050")
        || n.contains("1060")
        || n.contains("1070")
        || n.contains("1080")
        || n.contains("p100")
        || n.contains("p40")
        || n.contains("pascal")
    {
        return "Pascal".into();
    }
    "Unknown".into()
}

fn detect_amd() -> Vec<GpuDevice> {
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
                        let vram_mib = read_drm_vram_mib(&entry.path());
                        let generation = amd_generation_from_device_id(&device);
                        let host_pci_address = host_pci_address_from_sysfs(&entry.path());
                        let serial = read_drm_serial(&entry.path());
                        let dev = GpuDevice {
                            vendor: GpuVendor::Amd,
                            name: format!("AMD GPU {device}"),
                            index: idx,
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

fn detect_intel() -> Vec<GpuDevice> {
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
                        let live_vram = read_drm_vram_mib(&entry.path());
                        let vram_mib = if live_vram > 0 {
                            live_vram
                        } else {
                            intel_vram_mib_from_device_id(&device_id_str)
                                .map(u64::from)
                                .unwrap_or(0)
                        };
                        let generation = intel_generation_from_device_id(&device_id_str);
                        let host_pci_address = host_pci_address_from_sysfs(&entry.path());
                        let serial = read_drm_serial(&entry.path());
                        let dev = GpuDevice {
                            vendor: GpuVendor::Intel,
                            name,
                            index: idx,
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

/// Read VRAM total (MiB) from sysfs for a DRM device. AMD's amdgpu
/// driver and Intel's i915 driver both expose `mem_info_vram_total`
/// inside the device dir for discrete cards; integrated SKUs (Intel
/// iGPU sharing system memory, AMD APUs) generally don't, in which
/// case we return 0 and the inventory page renders "—". Best-effort:
/// any read failure returns 0 silently.
#[cfg(target_os = "linux")]
fn read_drm_vram_mib(device_path: &std::path::Path) -> u64 {
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
fn read_drm_vram_mib(_device_path: &std::path::Path) -> u64 {
    0
}

/// Extract the host-readable PCI bus address (e.g. `04:00.0`) from
/// a sysfs device path. The sysfs path is normally
/// `/sys/bus/pci/devices/0000:04:00.0`; we want the last path
/// component minus the domain prefix, since the abbreviated form is
/// what `lspci` shows and what admins recognise. Empty string on
/// non-matching shapes (defensive).
#[cfg(target_os = "linux")]
fn host_pci_address_from_sysfs(device_path: &std::path::Path) -> String {
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
fn host_pci_address_from_sysfs(_device_path: &std::path::Path) -> String {
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
fn read_drm_serial(device_path: &std::path::Path) -> Option<String> {
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
fn read_drm_serial(_device_path: &std::path::Path) -> Option<String> {
    None
}

/// AMD generation lookup. RDNA3 (RX 7000) is the only generation we
/// have AV1 encode silicon on today; earlier (RDNA1/2/Polaris/Vega) +
/// later (RDNA4 announced) all surface in the inventory page so
/// operators know the lay of the fleet. PCI device ids cross-checked
/// against the upstream amdgpu driver's `pci_table` (drivers/gpu/drm/
/// amd/amdgpu/amdgpu_drv.c) for the families we expect to see.
fn amd_generation_from_device_id(device_id: &str) -> String {
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

/// Intel generation lookup. Mirrors `intel_label_from_device_id` —
/// stays in lockstep so the inventory page's manufacturer / generation
/// rollup agrees with the per-row name shown elsewhere.
fn intel_generation_from_device_id(device_id: &str) -> String {
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
        let nvml = match init_nvml_with_fallback() {
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
                let used =
                    std::fs::read_to_string(entry.path().join("device").join("mem_info_vram_used"))
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
                    let bdf = read_pci_bdf_from_drm_card(&entry.path());
                    if let Some(bytes) = read_intel_vram_resident_bytes(bdf.as_deref()) {
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

/// Resolve the PCI BDF (e.g. `0000:03:00.0`) backing a
/// `/sys/class/drm/cardN` entry. The `device` symlink under the card
/// dir always points to the PCI device node — the file_name segment
/// of the resolved path IS the BDF. Returns None on read_link failure
/// (non-PCI virtual GPUs etc.).
#[cfg(target_os = "linux")]
fn read_pci_bdf_from_drm_card(card_dir: &std::path::Path) -> Option<String> {
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
fn read_intel_vram_resident_bytes(bdf_filter: Option<&str>) -> Option<u64> {
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

impl Default for GpuUtilizationReader {
    fn default() -> Self {
        Self::new()
    }
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

