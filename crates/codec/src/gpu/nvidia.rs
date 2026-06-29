//! NVIDIA GPU detection via libcuda dlopen + NVML enrichment.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::ptr;

use super::types::{GpuDevice, GpuVendor};

// ─── NVIDIA via libcuda dlopen ─────────────────────────────────────
type CUresult = c_int;
type CUdevice = c_int;

type FnCuInit = unsafe extern "C" fn(c_uint) -> CUresult;
type FnCuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CUresult;
type FnCuDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type FnCuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;

pub(super) fn detect_nvidia() -> Vec<GpuDevice> {
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
                vendor_index: ordinal as u32,
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
pub(super) fn init_nvml_with_fallback(
) -> Result<nvml_wrapper::Nvml, nvml_wrapper::error::NvmlError> {
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
