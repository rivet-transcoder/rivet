//! Create a Direct3D 11 device bound to a **specific AMD adapter**, so AMF's
//! `InitDX11` targets the right GPU on a multi-adapter host.
//!
//! AMF's `InitDX11(null)` lets the runtime create its device on DXGI adapter 0.
//! On a box with an NVIDIA card in slot 0 and an AMD iGPU elsewhere, that's the
//! wrong (non-AMD) adapter and AMF init fails outright. We enumerate adapters
//! via DXGI, find the `vendor_index`-th AMD one (PCI vendor `0x1002`), and make
//! a D3D11 device on it with `D3D11CreateDevice`; the caller hands that device
//! to `AMFContext::InitDX11`.
//!
//! Hand-rolled dlopen FFI (dxgi.dll + d3d11.dll), matching the in-tree AMF FFI
//! style — no `windows` / `winapi` crate. Windows-only.

use anyhow::{Result, bail};
use std::ffi::c_void;
use std::os::raw::c_int;
use std::ptr;

type Hresult = i32;
const S_OK: Hresult = 0;

#[repr(C)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

// IID_IDXGIFactory1 = 770aae78-f26f-4dba-a829-253c83d1b387
const IID_IDXGIFACTORY1: Guid = Guid {
    data1: 0x770a_ae78,
    data2: 0xf26f,
    data3: 0x4dba,
    data4: [0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87],
};

// DXGI_ADAPTER_DESC1 (x64 layout). We only read `vendor_id`.
#[repr(C)]
struct DxgiAdapterDesc1 {
    description: [u16; 128],
    vendor_id: u32,
    device_id: u32,
    sub_sys_id: u32,
    revision: u32,
    dedicated_video_memory: usize,
    dedicated_system_memory: usize,
    shared_system_memory: usize,
    adapter_luid: i64,
    flags: u32,
}

// --- COM vtables: only the slots we call are typed; the rest are opaque
// pointers laid out in exact interface order so the offsets line up. ---

#[repr(C)]
struct FactoryVtbl {
    query_interface: *const c_void,
    add_ref: *const c_void,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDXGIObject
    set_private_data: *const c_void,
    set_private_data_interface: *const c_void,
    get_private_data: *const c_void,
    get_parent: *const c_void,
    // IDXGIFactory
    enum_adapters: *const c_void,
    make_window_association: *const c_void,
    get_window_association: *const c_void,
    create_swap_chain: *const c_void,
    create_software_adapter: *const c_void,
    // IDXGIFactory1
    enum_adapters1: unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> Hresult,
    is_current: *const c_void,
}
#[repr(C)]
struct FactoryObj {
    vtbl: *const FactoryVtbl,
}

#[repr(C)]
struct AdapterVtbl {
    query_interface: *const c_void,
    add_ref: *const c_void,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
    // IDXGIObject
    set_private_data: *const c_void,
    set_private_data_interface: *const c_void,
    get_private_data: *const c_void,
    get_parent: *const c_void,
    // IDXGIAdapter
    enum_outputs: *const c_void,
    get_desc: *const c_void,
    check_interface_support: *const c_void,
    // IDXGIAdapter1
    get_desc1: unsafe extern "system" fn(*mut c_void, *mut DxgiAdapterDesc1) -> Hresult,
}
#[repr(C)]
struct AdapterObj {
    vtbl: *const AdapterVtbl,
}

// Minimal IUnknown view (Release at slot 2) for the created device.
#[repr(C)]
struct ReleaseVtbl {
    query_interface: *const c_void,
    add_ref: *const c_void,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
}
#[repr(C)]
struct ComObj {
    vtbl: *const ReleaseVtbl,
}

unsafe fn com_release(obj: *mut c_void) {
    if !obj.is_null() {
        unsafe {
            let vt = &*(*(obj as *mut ComObj)).vtbl;
            (vt.release)(obj);
        }
    }
}

type FnCreateDxgiFactory1 = unsafe extern "system" fn(*const Guid, *mut *mut c_void) -> Hresult;
#[allow(clippy::type_complexity)]
type FnD3d11CreateDevice = unsafe extern "system" fn(
    *mut c_void,      // pAdapter
    c_int,            // DriverType
    *mut c_void,      // Software
    u32,              // Flags
    *const c_void,    // pFeatureLevels
    u32,              // FeatureLevels
    u32,              // SDKVersion
    *mut *mut c_void, // ppDevice
    *mut c_int,       // pFeatureLevel
    *mut *mut c_void, // ppImmediateContext
) -> Hresult;

const D3D_DRIVER_TYPE_UNKNOWN: c_int = 0;
const D3D11_SDK_VERSION: u32 = 7;
/// `D3D11_CREATE_DEVICE_VIDEO_SUPPORT` — required for a device AMF will use for
/// hardware video decode; without it AMF's `InitDX11` rejects the device.
const D3D11_CREATE_DEVICE_VIDEO_SUPPORT: u32 = 0x800;

/// A D3D11 device created on a specific AMD adapter, ready to hand to AMF's
/// `InitDX11`. Holds the dlopen'd `dxgi.dll` / `d3d11.dll` so they outlive the
/// device, and releases the device on drop.
pub struct AmdD3d11Device {
    device: *mut c_void,
    _dxgi: libloading::Library,
    _d3d11: libloading::Library,
}

impl AmdD3d11Device {
    /// The `ID3D11Device*` to pass as AMF `InitDX11`'s device argument.
    pub fn as_ptr(&self) -> *mut c_void {
        self.device
    }
}

impl Drop for AmdD3d11Device {
    fn drop(&mut self) {
        unsafe { com_release(self.device) };
    }
}

/// Create a D3D11 device on the `vendor_index`-th AMD (PCI vendor `0x1002`)
/// adapter so AMF binds to that GPU instead of DXGI adapter 0.
pub fn create_amd_d3d11_device(vendor_index: u32) -> Result<AmdD3d11Device> {
    unsafe {
        let dxgi = libloading::Library::new("dxgi.dll")
            .map_err(|e| anyhow::anyhow!("loading dxgi.dll: {e}"))?;
        let d3d11 = libloading::Library::new("d3d11.dll")
            .map_err(|e| anyhow::anyhow!("loading d3d11.dll: {e}"))?;
        let create_factory = *dxgi.get::<FnCreateDxgiFactory1>(b"CreateDXGIFactory1")?;
        let create_device = *d3d11.get::<FnD3d11CreateDevice>(b"D3D11CreateDevice")?;

        let mut factory: *mut c_void = ptr::null_mut();
        if create_factory(&IID_IDXGIFACTORY1, &mut factory) != S_OK || factory.is_null() {
            bail!("CreateDXGIFactory1 failed");
        }
        let factory_vt = &*(*(factory as *mut FactoryObj)).vtbl;

        // Walk adapters; pick the vendor_index-th AMD one.
        let mut amd_seen = 0u32;
        let mut chosen: *mut c_void = ptr::null_mut();
        let mut i = 0u32;
        loop {
            let mut adapter: *mut c_void = ptr::null_mut();
            // Enumeration ends with DXGI_ERROR_NOT_FOUND (non-S_OK).
            if (factory_vt.enum_adapters1)(factory, i, &mut adapter) != S_OK || adapter.is_null() {
                break;
            }
            let adapter_vt = &*(*(adapter as *mut AdapterObj)).vtbl;
            let mut desc: DxgiAdapterDesc1 = std::mem::zeroed();
            if (adapter_vt.get_desc1)(adapter, &mut desc) == S_OK && desc.vendor_id == 0x1002 {
                if amd_seen == vendor_index {
                    chosen = adapter; // keep this ref; released after device create
                    break;
                }
                amd_seen += 1;
            }
            (adapter_vt.release)(adapter);
            i += 1;
        }
        (factory_vt.release)(factory);

        if chosen.is_null() {
            bail!("no AMD (0x1002) DXGI adapter at vendor index {vendor_index}");
        }

        // DriverType MUST be UNKNOWN when an explicit adapter is supplied.
        let mut device: *mut c_void = ptr::null_mut();
        let hr = create_device(
            chosen,
            D3D_DRIVER_TYPE_UNKNOWN,
            ptr::null_mut(),
            D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
            ptr::null(),
            0,
            D3D11_SDK_VERSION,
            &mut device,
            ptr::null_mut(),
            ptr::null_mut(),
        );
        com_release(chosen); // the device holds its own adapter ref

        if hr != S_OK || device.is_null() {
            bail!("D3D11CreateDevice on AMD adapter {vendor_index} failed (hr=0x{hr:08x})");
        }
        Ok(AmdD3d11Device { device, _dxgi: dxgi, _d3d11: d3d11 })
    }
}

#[cfg(test)]
mod tests {
    /// Exercises the DXGI enumeration + D3D11CreateDevice + the `Drop`
    /// `com_release` path. A wrong vtable offset would segfault here. Prints +
    /// passes when no AMD adapter is present (non-AMD CI), so it's safe to run
    /// anywhere; on the foxbox it must construct + drop a real device cleanly.
    #[test]
    fn create_and_drop_amd_d3d11_device() {
        match super::create_amd_d3d11_device(0) {
            Ok(dev) => {
                assert!(!dev.as_ptr().is_null(), "device pointer is null");
                drop(dev); // exercises com_release
                eprintln!("create_amd_d3d11_device(0): OK + dropped cleanly");
            }
            Err(e) => eprintln!("create_amd_d3d11_device(0): no device ({e})"),
        }
    }
}
