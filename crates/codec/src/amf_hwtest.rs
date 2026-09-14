//! One lock for every on-hardware AMF test (the box has a single AMF-capable
//! iGPU, and two AMF sessions decoding/encoding on it at once are flaky). The
//! AMF encoder and decoder hardware tests live in different test binaries —
//! the `rivet-codec` lib-test binary and the `amf_decode_pixels` integration
//! binary — which `cargo test` runs as separate processes in parallel, so a
//! plain `static Mutex` cannot serialise them. This holds a process-wide
//! `Mutex` **and**, on Windows, a named kernel mutex so the guard is exclusive
//! across every test process on the machine.
//!
//! Test-support only: nothing outside `#[cfg(test)]` / the AMF test binaries
//! calls it, and it does no I/O when there is no AMD GPU. Every test that
//! loads the AMF runtime or touches the AMD adapter takes it: the encoder
//! round-trips and `AmfEncoder::new` / property-storage ABI tests
//! (`encode/amf/tests*.rs`), the decode probe (`decode/amf_dec.rs`), the
//! D3D11 device test (`amf_device.rs`), and the `amf_decode_pixels` and
//! `gpu_vendor_matrix` binaries. The guard is not re-entrant on the process
//! lock, so a test takes it once, at the top.
#![cfg(feature = "amd")]
#![allow(dead_code)]

use std::sync::{Mutex, MutexGuard, OnceLock};

/// Held for the duration of one on-hardware AMF test. Drops both the
/// in-process lock and (on Windows) the named cross-process one.
pub struct HwGuard {
    _process: MutexGuard<'static, ()>,
    #[cfg(windows)]
    named: *mut core::ffi::c_void,
}

// The guard is created and dropped on one thread; the raw handle is only
// released, by this owner, in `Drop`.
unsafe impl Send for HwGuard {}

/// Acquire the shared hardware-test lock. Blocks until every other test
/// process that took it has released it.
pub fn hw_lock() -> HwGuard {
    static PROCESS: OnceLock<Mutex<()>> = OnceLock::new();
    // A poisoned lock is fine here — it only guards hardware access, not data.
    let process = PROCESS
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    #[cfg(windows)]
    let named = unsafe { named_lock_acquire() };

    HwGuard {
        _process: process,
        #[cfg(windows)]
        named,
    }
}

#[cfg(windows)]
impl Drop for HwGuard {
    fn drop(&mut self) {
        unsafe {
            if !self.named.is_null() {
                ReleaseMutex(self.named);
                CloseHandle(self.named);
            }
        }
    }
}

#[cfg(windows)]
unsafe extern "system" {
    fn CreateMutexW(
        attrs: *mut core::ffi::c_void,
        initial_owner: i32,
        name: *const u16,
    ) -> *mut core::ffi::c_void;
    fn WaitForSingleObject(handle: *mut core::ffi::c_void, ms: u32) -> u32;
    fn ReleaseMutex(handle: *mut core::ffi::c_void) -> i32;
    fn CloseHandle(handle: *mut core::ffi::c_void) -> i32;
}

/// `CreateMutexW` a machine-wide named mutex and wait for it. Returns the
/// handle to release + close on drop, or null if the runtime refused (in
/// which case the process `Mutex` alone still serialises within this binary).
#[cfg(windows)]
unsafe fn named_lock_acquire() -> *mut core::ffi::c_void {
    unsafe {
        const INFINITE: u32 = 0xFFFF_FFFF;
        // A local (non-"Global\") name is per-session, which is all the test
        // runners share; letters only, so no path/permission surprises.
        let name: Vec<u16> = "rivet_amf_hw_test_lock\0".encode_utf16().collect();
        let h = CreateMutexW(std::ptr::null_mut(), 0, name.as_ptr());
        if !h.is_null() {
            WaitForSingleObject(h, INFINITE);
        }
        h
    }
}
