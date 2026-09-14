//! The AMF runtime lifecycle shared by the encoder and the decoder: dlopen,
//! `AMFInit`, a context bound to the chosen AMD GPU, and the property
//! helpers every `SetProperty` / `GetProperty` goes through.
//!
//! Context binding:
//! - **Windows**: a D3D11 device made on the `vendor_index`-th AMD adapter
//!   ([`crate::amf_device`]) is handed to `InitDX11(dev, AMF_DX11_1)` —
//!   `InitDX11(null)` would bind DXGI adapter 0, which on a mixed host is the
//!   wrong (non-AMD) card.
//! - **elsewhere**: `QueryInterface(IID_AMFContext1)` → `InitVulkan(null)`
//!   (`core/Context.h:371`; AMF picks the first AMD GPU).
//!
//! Drop order inside [`AmfRuntime`] is context (Terminate + Release), then
//! the D3D11 device, then the library handle that provides the code behind
//! every vtable pointer just called. A component created on the context must
//! be released before the runtime is dropped; the encoder and decoder keep
//! the runtime as their last field for that reason.

use anyhow::{Context, Result, bail};
use std::ffi::c_void;
use std::ptr;

use crate::amf_ffi::*;

// ─── Wide strings ─────────────────────────────────────────────────

/// Encode a null-terminated `wchar_t` string the way the SDK's property
/// names are declared (`const wchar_t*`): UTF-16 on Windows, UTF-32 on
/// Linux — see [`AmfWchar`]. Every property name in the headers is ASCII,
/// so both encodings are the code points themselves.
pub(crate) fn wide(s: &str) -> Vec<AmfWchar> {
    let mut out: Vec<AmfWchar> = s.chars().map(|c| c as u32 as AmfWchar).collect();
    out.push(0);
    out
}

/// Decode a null-terminated `wchar_t` string back to UTF-8 (tests and logs).
#[cfg(test)]
pub(crate) unsafe fn from_wide(p: *const AmfWchar) -> String {
    unsafe {
        let mut len = 0usize;
        while *p.add(len) != 0 {
            len += 1;
        }
        std::slice::from_raw_parts(p, len)
            .iter()
            .map(|&c| char::from_u32(c as u32).unwrap_or('\u{fffd}'))
            .collect()
    }
}

// ─── Property helpers ─────────────────────────────────────────────

/// Set one property on any AMF object (component, surface, buffer, context)
/// through its `AMFPropertyStorage` prefix. Returns the `AMF_RESULT` as a
/// Rust `Result` so the call site can bail cleanly when the driver rejects a
/// knob value.
pub(crate) unsafe fn set_property(obj: *mut c_void, name: &str, value: AmfVariant) -> Result<()> {
    unsafe {
        let vt = &*(*(obj as *mut AmfObj)).vtbl;
        let wname = wide(name);
        let rc = (vt.set_property)(obj, wname.as_ptr(), value);
        if rc != AMF_OK {
            bail!("AMF SetProperty({name}) failed: {rc} ({})", result_name(rc));
        }
        Ok(())
    }
}

/// `SetProperty(name, amf_int64)`.
pub(crate) unsafe fn set_int_property(obj: *mut c_void, name: &str, value: i64) -> Result<()> {
    unsafe {
        set_property(obj, name, AmfVariant::int64(value))
            .map_err(|e| anyhow::anyhow!("{e} (value {value})"))
    }
}

/// `SetProperty(name, amf_bool)`.
pub(crate) unsafe fn set_bool_property(obj: *mut c_void, name: &str, value: bool) -> Result<()> {
    unsafe {
        set_property(obj, name, AmfVariant::bool_(value))
            .map_err(|e| anyhow::anyhow!("{e} (value {value})"))
    }
}

/// `SetProperty(name, AMFRate { num, den })`.
pub(crate) unsafe fn set_rate_property(
    obj: *mut c_void,
    name: &str,
    num: u32,
    den: u32,
) -> Result<()> {
    unsafe {
        set_property(obj, name, AmfVariant::rate(num, den))
            .map_err(|e| anyhow::anyhow!("{e} (value {num}/{den})"))
    }
}

/// `GetProperty(name)` as an `amf_int64`, or `None` when the object does not
/// carry the property or it is not int-typed.
#[allow(dead_code)]
pub(crate) unsafe fn get_int_property(obj: *mut c_void, name: &str) -> Option<i64> {
    unsafe {
        let vt = &*(*(obj as *mut AmfObj)).vtbl;
        let wname = wide(name);
        let mut var = AmfVariant::empty();
        if (vt.get_property)(obj, wname.as_ptr(), &mut var) != AMF_OK {
            return None;
        }
        var.as_int64()
    }
}

/// `Release` any AMF object through its `AMFInterface` prefix.
pub(crate) unsafe fn release(obj: *mut c_void) {
    unsafe {
        if !obj.is_null() {
            let vt = &*(*(obj as *mut AmfObj)).vtbl;
            let _ = (vt.release)(obj);
        }
    }
}

// ─── Runtime + context ────────────────────────────────────────────

/// Load the AMF runtime library: `libamfrt64.so.1` / `libamfrt64.so` on
/// Linux, `amfrt64.dll` on Windows (both ship with the Adrenalin and Pro
/// driver bundles).
pub(crate) fn load_library() -> Result<libloading::Library> {
    unsafe { libloading::Library::new("libamfrt64.so.1") }
        .or_else(|_| unsafe { libloading::Library::new("libamfrt64.so") })
        .or_else(|_| unsafe { libloading::Library::new("amfrt64.dll") })
        .context("loading AMF runtime library (AMD driver not present?)")
}

/// A loaded AMF runtime with a factory and a context bound to one AMD GPU.
pub(crate) struct AmfRuntime {
    /// The factory is a runtime singleton, not reference-counted.
    pub(crate) factory: *mut c_void,
    pub(crate) context: *mut c_void,
    /// Keeps the AMD-adapter D3D11 device alive for the context's lifetime.
    #[cfg(windows)]
    _amd_device: Option<crate::amf_device::AmdD3d11Device>,
    /// Declared last: it provides the code behind every vtable pointer.
    _lib: libloading::Library,
}

// The context is thread-safe per the SDK's "Thread Safety" appendix; the
// D3D11 device is a free-threaded COM object only this owner releases.
unsafe impl Send for AmfRuntime {}

impl AmfRuntime {
    /// `AMFInit` + `CreateContext` + bind to the `vendor_index`-th AMD GPU
    /// (`GpuDevice::vendor_index`, the vendor-local ordinal — not the global
    /// `index`). Fails cleanly, with the context released, when the GPU is
    /// not one the runtime drives.
    pub(crate) fn open(vendor_index: u32) -> Result<Self> {
        let lib = load_library()?;
        unsafe {
            let amf_init: libloading::Symbol<FnAmfInit> =
                lib.get(b"AMFInit").context("AMFInit symbol")?;
            let mut factory: *mut c_void = ptr::null_mut();
            let rc = amf_init(AMF_VERSION, &mut factory);
            if rc != AMF_OK || factory.is_null() {
                bail!("AMFInit failed: {rc} ({})", result_name(rc));
            }
            let factory_vt = &*(*(factory as *mut AmfFactoryObj)).vtbl;

            let mut context: *mut c_void = ptr::null_mut();
            let rc = (factory_vt.create_context)(factory, &mut context);
            if rc != AMF_OK || context.is_null() {
                bail!("AMFFactory::CreateContext failed: {rc} ({})", result_name(rc));
            }
            let context_vt = &*(*(context as *mut AmfContextObj)).vtbl;

            #[cfg(windows)]
            let amd_device = {
                let dev = match crate::amf_device::create_amd_d3d11_device(vendor_index) {
                    Ok(dev) => dev,
                    Err(e) => {
                        release_context(context);
                        return Err(e.context("creating a D3D11 device on the AMD adapter for AMF"));
                    }
                };
                let rc = (context_vt.init_dx11)(context, dev.as_ptr(), AMF_DX11_1);
                if rc != AMF_OK {
                    release_context(context);
                    bail!(
                        "AMFContext::InitDX11 on AMD adapter {vendor_index} failed: {rc} ({}) — this \
                         GPU is not one the AMF runtime drives",
                        result_name(rc)
                    );
                }
                Some(dev)
            };
            #[cfg(not(windows))]
            {
                if vendor_index != 0 {
                    tracing::warn!(
                        vendor_index,
                        "AMF InitVulkan(null) picks the first AMD GPU; multi-AMD hosts may need \
                         external adapter routing"
                    );
                }
                let mut context1: *mut c_void = ptr::null_mut();
                let rc = (context_vt.ps.query_interface)(context, &AMF_IID_CONTEXT1, &mut context1);
                if rc != AMF_OK || context1.is_null() {
                    release_context(context);
                    bail!(
                        "AMFContext::QueryInterface(AMFContext1) failed: {rc} ({}) — runtime older \
                         than the Vulkan-capable 1.4.x?",
                        result_name(rc)
                    );
                }
                let context1_vt = &*(*(context1 as *mut AmfContext1Obj)).vtbl;
                let rc = (context1_vt.init_vulkan)(context1, ptr::null_mut());
                // QueryInterface handed us a second ref on the same object;
                // the base handle keeps it alive.
                let _ = (context1_vt.base.ps.release)(context1);
                if rc != AMF_OK {
                    release_context(context);
                    bail!(
                        "AMFContext1::InitVulkan failed: {rc} ({}) — no AMF-capable AMD GPU",
                        result_name(rc)
                    );
                }
            }

            Ok(Self {
                factory,
                context,
                #[cfg(windows)]
                _amd_device: amd_device,
                _lib: lib,
            })
        }
    }

    /// `AMFFactory::CreateComponent(context, id)`. The caller owns the
    /// returned component and must `Terminate` + `Release` it before this
    /// runtime drops.
    pub(crate) unsafe fn create_component(&self, id: &str) -> Result<*mut c_void> {
        unsafe {
            let factory_vt = &*(*(self.factory as *mut AmfFactoryObj)).vtbl;
            let wid = wide(id);
            let mut component: *mut c_void = ptr::null_mut();
            let rc =
                (factory_vt.create_component)(self.factory, self.context, wid.as_ptr(), &mut component);
            if rc != AMF_OK || component.is_null() {
                bail!(
                    "AMFFactory::CreateComponent({id}) failed: {rc} ({})",
                    result_name(rc)
                );
            }
            Ok(component)
        }
    }

    /// `AMFContext::AllocBuffer(HOST, size)`; the caller releases.
    pub(crate) unsafe fn alloc_host_buffer(&self, size: usize) -> Result<*mut c_void> {
        unsafe {
            let context_vt = &*(*(self.context as *mut AmfContextObj)).vtbl;
            let mut buf: *mut c_void = ptr::null_mut();
            let rc = (context_vt.alloc_buffer)(self.context, AMF_MEMORY_HOST, size, &mut buf);
            if rc != AMF_OK || buf.is_null() {
                bail!("AMFContext::AllocBuffer({size}) failed: {rc} ({})", result_name(rc));
            }
            Ok(buf)
        }
    }
}

impl Drop for AmfRuntime {
    fn drop(&mut self) {
        unsafe {
            if !self.context.is_null() {
                release_context(self.context);
            }
        }
    }
}

/// `Terminate` + `Release` a context we created.
pub(crate) unsafe fn release_context(context: *mut c_void) {
    unsafe {
        let vt = &*(*(context as *mut AmfContextObj)).vtbl;
        let _ = (vt.terminate)(context);
        let _ = (vt.ps.release)(context);
    }
}

/// `Terminate` + `Release` a component.
pub(crate) unsafe fn release_component(component: *mut c_void) {
    unsafe {
        if !component.is_null() {
            let vt = &*(*(component as *mut AmfComponentObj)).vtbl;
            let _ = (vt.terminate)(component);
            let _ = (vt.ps.release)(component);
        }
    }
}
