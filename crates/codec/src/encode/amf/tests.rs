// ─── Tests: shared session driver, FFI layout, runtime ABI ────────
//
// Most of these tests need no silicon (the dev box is an RTX 3090 + the
// Ryzen 9 9950X iGPU, whose VCN the AMF runtime drives for H.264 / H.265 but
// not AV1). They exercise:
//
// - the retry driver, the drain helper's status mapping, the ring index
//   cycling and the variant layout, against a mock component whose vtable
//   is laid out exactly as `ffi.rs` (and therefore the header) says;
// - the property-storage ABI against the **installed AMF runtime**
//   (`amfrt64.dll` ships with the Adrenalin driver even where no VCN is
//   usable): `AMFInit`, `CreateContext`, `SetProperty` / `GetProperty` /
//   `HasProperty` / `GetPropertyCount` / `QueryInterface(AMFContext1)` /
//   `Acquire` / `Release` / `Terminate` on a real context — a wrong slot
//   order or variant size fails these, so they are real evidence for the
//   layout every other call goes through. Skipped (loudly) where the runtime
//   cannot be loaded;
// - `AmfEncoder::new` for each codec on this machine, which must either
//   succeed or fail with a clear message, and must tear down cleanly
//   either way. The end-to-end encode on the iGPU lives in `tests_h26x.rs`.

use super::{
    // ffi.rs items (brought into amf via private `use self::ffi::*;`)
    AMF_EOF, AMF_FAIL, AMF_IID_BUFFER, AMF_IID_CONTEXT1, AMF_INPUT_FULL, AMF_NEED_MORE_INPUT,
    AMF_NOT_FOUND, AMF_OK, AMF_REPEAT, AMF_SURFACE_NV12, AMF_SURFACE_P010, AMF_VARIANT_BOOL,
    AMF_VARIANT_INT64, AMF_VARIANT_RATE, AmfComponentObj, AmfComponentVtbl, AmfDataVtbl,
    AmfGuid, AmfLong, AmfPropertyStorageVtbl, AmfResult, AmfSurfaceObj, AmfSurfaceVtbl,
    AmfVariant, AmfWchar, INPUT_FULL_MAX_RETRIES, RING_SIZE, Slot,
    // surface.rs items
    SurfaceGuard,
    // config.rs items
    amf_color_bit_depth_for, amf_color_profile_for, amf_surface_format_for, frame_rate_rational,
    from_wide, set_int_property, transfer_to_h273, wide,
    // codec plans
    AV1_PLAN, AVC_PLAN, CodecPlan, HEVC_PLAN,
    // private free functions in mod.rs
    drain_until_hungry_raw, submit_with_backpressure,
};
use crate::frame::{PixelFormat, TransferFn};

use std::cell::RefCell;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

// ── Mock AMF component ────────────────────────────────────────
//
// Minimal fake AMF component built to match the vtable layout our
// production code calls through. Each test configures a canned sequence of
// AMF_RESULT values for SubmitInput / QueryOutput; the mock returns them in
// order and tracks Release counts so we can assert no UAF or leak occurred.

thread_local! {
    static MOCK_SUBMIT_RESULTS: RefCell<Vec<AmfResult>> = const { RefCell::new(Vec::new()) };
    static MOCK_QUERY_RESULTS: RefCell<Vec<AmfResult>> = const { RefCell::new(Vec::new()) };
    static MOCK_SUBMIT_CALLS: AtomicUsize = const { AtomicUsize::new(0) };
    static MOCK_QUERY_CALLS: AtomicUsize = const { AtomicUsize::new(0) };
    static MOCK_SURFACE_REFCOUNT: AtomicI64 = const { AtomicI64::new(0) };
    /// Records the surface pointer passed to each SubmitInput call so we
    /// can assert the driver retries with the SAME pointer.
    static MOCK_SUBMIT_POINTERS: RefCell<Vec<*mut c_void>> = const { RefCell::new(Vec::new()) };
    /// Every `(name, int64 value)` SetProperty recorded on any mock object.
    pub(super) static RECORDED: RefCell<Vec<(String, AmfVariant)>> = const { RefCell::new(Vec::new()) };
}

fn mock_reset() {
    MOCK_SUBMIT_RESULTS.with(|v| v.borrow_mut().clear());
    MOCK_QUERY_RESULTS.with(|v| v.borrow_mut().clear());
    MOCK_SUBMIT_POINTERS.with(|v| v.borrow_mut().clear());
    MOCK_SUBMIT_CALLS.with(|c| c.store(0, Ordering::SeqCst));
    MOCK_QUERY_CALLS.with(|c| c.store(0, Ordering::SeqCst));
    MOCK_SURFACE_REFCOUNT.with(|c| c.store(1, Ordering::SeqCst));
    RECORDED.with(|r| r.borrow_mut().clear());
}

fn set_submit_sequence(results: &[AmfResult]) {
    MOCK_SUBMIT_RESULTS.with(|v| *v.borrow_mut() = results.to_vec());
}

fn set_query_sequence(results: &[AmfResult]) {
    MOCK_QUERY_RESULTS.with(|v| *v.borrow_mut() = results.to_vec());
}

fn submit_call_count() -> usize {
    MOCK_SUBMIT_CALLS.with(|c| c.load(Ordering::SeqCst))
}

fn query_call_count() -> usize {
    MOCK_QUERY_CALLS.with(|c| c.load(Ordering::SeqCst))
}

fn surface_refcount() -> i64 {
    MOCK_SURFACE_REFCOUNT.with(|c| c.load(Ordering::SeqCst))
}

fn submit_pointer_at(idx: usize) -> Option<*mut c_void> {
    MOCK_SUBMIT_POINTERS.with(|v| v.borrow().get(idx).copied())
}

/// Everything the mock recorded, as `(name, variant)`.
pub(super) fn recorded() -> Vec<(String, AmfVariant)> {
    RECORDED.with(|r| r.borrow().clone())
}

// ── Mock vtable functions ─────────────────────────────────────

unsafe extern "system" fn mock_qi(_: *mut c_void, _: *const AmfGuid, _: *mut *mut c_void) -> AmfResult {
    AMF_OK
}
unsafe extern "system" fn mock_acquire(_: *mut c_void) -> AmfLong {
    1
}
unsafe extern "system" fn mock_release_component(_: *mut c_void) -> AmfLong {
    1
}
/// Records every SetProperty (component and surface alike).
unsafe extern "system" fn mock_set_property(_: *mut c_void, name: *const AmfWchar, v: AmfVariant) -> AmfResult {
    let s = unsafe { from_wide(name) };
    RECORDED.with(|r| r.borrow_mut().push((s, v)));
    AMF_OK
}
unsafe extern "system" fn mock_get_property(_: *mut c_void, _: *const AmfWchar, _: *mut AmfVariant) -> AmfResult {
    AMF_NOT_FOUND
}
unsafe extern "system" fn mock_has_property(_: *mut c_void, _: *const AmfWchar) -> u8 {
    0
}
unsafe extern "system" fn mock_get_property_count(_: *mut c_void) -> usize {
    0
}
unsafe extern "system" fn mock_init(_: *mut c_void, _: i32, _: i32, _: i32) -> AmfResult {
    AMF_OK
}
unsafe extern "system" fn mock_reinit(_: *mut c_void, _: i32, _: i32) -> AmfResult {
    AMF_OK
}
unsafe extern "system" fn mock_result(_: *mut c_void) -> AmfResult {
    AMF_OK
}

unsafe extern "system" fn mock_submit_input(_: *mut c_void, surface: *mut c_void) -> AmfResult {
    MOCK_SUBMIT_POINTERS.with(|v| v.borrow_mut().push(surface));
    let idx = MOCK_SUBMIT_CALLS.with(|c| c.fetch_add(1, Ordering::SeqCst));
    MOCK_SUBMIT_RESULTS.with(|v| v.borrow().get(idx).copied().unwrap_or(AMF_OK))
}

unsafe extern "system" fn mock_query_output(_: *mut c_void, data: *mut *mut c_void) -> AmfResult {
    let idx = MOCK_QUERY_CALLS.with(|c| c.fetch_add(1, Ordering::SeqCst));
    let rc = MOCK_QUERY_RESULTS.with(|v| v.borrow().get(idx).copied().unwrap_or(AMF_REPEAT));
    if rc == AMF_OK {
        // Null data — the drain helper treats that as "no packet this
        // round" and keeps looping.
        unsafe {
            *data = ptr::null_mut();
        }
    }
    rc
}

unsafe extern "system" fn mock_surface_release(_: *mut c_void) -> AmfLong {
    let prev = MOCK_SURFACE_REFCOUNT.with(|c| c.fetch_sub(1, Ordering::SeqCst));
    assert!(prev > 0, "surface Release when refcount already zero (UAF indicator)");
    (prev - 1) as AmfLong
}
unsafe extern "system" fn mock_convert(_: *mut c_void, _: i32) -> AmfResult {
    AMF_OK
}
unsafe extern "system" fn mock_get_i64(_: *mut c_void) -> i64 {
    0
}
unsafe extern "system" fn mock_set_i64(_: *mut c_void, _: i64) {}
unsafe extern "system" fn mock_get_format(_: *mut c_void) -> i32 {
    AMF_SURFACE_NV12
}
unsafe extern "system" fn mock_get_planes_count(_: *mut c_void) -> usize {
    2
}
unsafe extern "system" fn mock_get_plane_at(_: *mut c_void, _: usize) -> *mut c_void {
    ptr::null_mut()
}
unsafe extern "system" fn mock_get_plane(_: *mut c_void, _: i32) -> *mut c_void {
    ptr::null_mut()
}

const fn mock_ps(release: unsafe extern "system" fn(*mut c_void) -> AmfLong) -> AmfPropertyStorageVtbl {
    AmfPropertyStorageVtbl {
        acquire: mock_acquire,
        release,
        query_interface: mock_qi,
        set_property: mock_set_property,
        get_property: mock_get_property,
        has_property: mock_has_property,
        get_property_count: mock_get_property_count,
        get_property_at: Slot::NULL,
        clear: Slot::NULL,
        add_to: Slot::NULL,
        copy_to: Slot::NULL,
        add_observer: Slot::NULL,
        remove_observer: Slot::NULL,
    }
}

pub(super) static MOCK_SURFACE_VTBL: AmfSurfaceVtbl = AmfSurfaceVtbl {
    data: AmfDataVtbl {
        ps: mock_ps(mock_surface_release),
        get_memory_type: Slot::NULL,
        duplicate: Slot::NULL,
        convert: mock_convert,
        interop: Slot::NULL,
        get_data_type: Slot::NULL,
        is_reusable: Slot::NULL,
        set_pts: mock_set_i64,
        get_pts: mock_get_i64,
        set_duration: mock_set_i64,
        get_duration: mock_get_i64,
    },
    get_format: mock_get_format,
    get_planes_count: mock_get_planes_count,
    get_plane_at: mock_get_plane_at,
    get_plane: mock_get_plane,
    get_frame_type: Slot::NULL,
    set_frame_type: Slot::NULL,
    set_crop: Slot::NULL,
    copy_surface_region: Slot::NULL,
    add_observer_surface: Slot::NULL,
    remove_observer_surface: Slot::NULL,
};

pub(super) static MOCK_COMPONENT_VTBL: AmfComponentVtbl = AmfComponentVtbl {
    ps: mock_ps(mock_release_component),
    get_properties_info_count: Slot::NULL,
    get_property_info_at: Slot::NULL,
    get_property_info: Slot::NULL,
    validate_property: Slot::NULL,
    init: mock_init,
    reinit: mock_reinit,
    terminate: mock_result,
    drain: mock_result,
    flush: mock_result,
    submit_input: mock_submit_input,
    query_output: mock_query_output,
    get_context: Slot::NULL,
    set_output_data_allocator_cb: Slot::NULL,
    get_caps: Slot::NULL,
    optimize: Slot::NULL,
};

/// A fake surface + component that resolve to the mock vtables.
pub(super) fn make_mock_pair() -> (Box<AmfSurfaceObj>, Box<AmfComponentObj>) {
    let surface = Box::new(AmfSurfaceObj { vtbl: &MOCK_SURFACE_VTBL });
    let component = Box::new(AmfComponentObj { vtbl: &MOCK_COMPONENT_VTBL });
    (surface, component)
}

// ── Retry driver ──────────────────────────────────────────────

/// An AMF_INPUT_FULL return from SubmitInput must NOT release the surface
/// before the retry: the retry must pass the SAME pointer, the refcount must
/// stay ≥1 across it, and reach exactly 0 after the success path releases.
#[test]
fn test_amf_input_full_does_not_release_surface_before_retry() {
    mock_reset();
    set_submit_sequence(&[AMF_INPUT_FULL, AMF_OK]);
    set_query_sequence(&[AMF_REPEAT]);

    let (mut surface, mut component) = make_mock_pair();
    let surface_ptr: *mut c_void = surface.as_mut() as *mut _ as *mut c_void;
    let component_ptr: *mut c_void = component.as_mut() as *mut _ as *mut c_void;

    let mut guard = SurfaceGuard::new(surface_ptr);
    let mut packets = Vec::new();

    let result = unsafe { submit_with_backpressure(&mut packets, component_ptr, &mut guard, &AVC_PLAN, 333_333) };
    assert!(result.is_ok(), "submit_with_backpressure failed: {result:?}");

    assert_eq!(submit_call_count(), 2, "SubmitInput must retry exactly once on INPUT_FULL");
    assert_eq!(submit_pointer_at(0), Some(surface_ptr));
    assert_eq!(submit_pointer_at(1), Some(surface_ptr), "retry must pass the SAME surface pointer");
    assert_eq!(surface_refcount(), 0, "exactly one release after success (no leak, no double-release)");
    drop(guard);
    assert_eq!(surface_refcount(), 0, "Drop after transfer must be a no-op");
}

/// AMF_NEED_MORE_INPUT on QueryOutput is "no packet yet", not an error.
#[test]
fn test_amf_need_more_input_returns_no_packet() {
    mock_reset();
    set_query_sequence(&[AMF_NEED_MORE_INPUT]);
    let (_, mut component) = make_mock_pair();
    let component_ptr: *mut c_void = component.as_mut() as *mut _ as *mut c_void;
    let mut packets = Vec::new();
    let result = unsafe { drain_until_hungry_raw(&mut packets, component_ptr, &AVC_PLAN, 333_333) };
    assert_eq!(result.unwrap(), super::DrainEnd::NeedMoreInput);
    assert_eq!(packets.len(), 0);
    assert_eq!(query_call_count(), 1);
}

/// AMF_EOF after Drain() ends the flush loop cleanly.
#[test]
fn test_amf_eof_ends_drain_cleanly() {
    mock_reset();
    set_query_sequence(&[AMF_EOF]);
    let (_, mut component) = make_mock_pair();
    let component_ptr: *mut c_void = component.as_mut() as *mut _ as *mut c_void;
    let mut packets = Vec::new();
    let result = unsafe { drain_until_hungry_raw(&mut packets, component_ptr, &HEVC_PLAN, 333_333) };
    assert_eq!(result.unwrap(), super::DrainEnd::Eof, "EOF is reported so the flush loop can stop");
    assert_eq!(packets.len(), 0);
    assert_eq!(query_call_count(), 1);
}

/// AMF_OK with a null buffer keeps draining until a "hungry" status.
#[test]
fn test_amf_ok_null_data_keeps_draining() {
    mock_reset();
    set_query_sequence(&[AMF_OK, AMF_OK, AMF_REPEAT]);
    let (_, mut component) = make_mock_pair();
    let component_ptr: *mut c_void = component.as_mut() as *mut _ as *mut c_void;
    let mut packets = Vec::new();
    let end = unsafe { drain_until_hungry_raw(&mut packets, component_ptr, &AV1_PLAN, 333_333) }.unwrap();
    assert_eq!(end, super::DrainEnd::Repeat);
    assert_eq!(query_call_count(), 3);
    assert!(packets.is_empty());
}

#[test]
fn test_amf_ring_buffer_index_cycles() {
    let mut idx = 0usize;
    let mut seen = Vec::new();
    for _ in 0..(RING_SIZE * 3) {
        seen.push(idx);
        idx = (idx + 1) % RING_SIZE;
    }
    assert_eq!(seen, vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
}

#[test]
fn test_amf_ring_size_is_four() {
    assert_eq!(RING_SIZE, 4, "RING_SIZE must match Squad-5's NVENC default of 4");
}

/// AMF_REPEAT on SubmitInput has the same "retry same surface" semantics.
#[test]
fn test_amf_repeat_on_submit_retries_same_surface() {
    mock_reset();
    set_submit_sequence(&[AMF_REPEAT, AMF_OK]);
    set_query_sequence(&[AMF_REPEAT]);
    let (mut surface, mut component) = make_mock_pair();
    let surface_ptr: *mut c_void = surface.as_mut() as *mut _ as *mut c_void;
    let component_ptr: *mut c_void = component.as_mut() as *mut _ as *mut c_void;
    let mut guard = SurfaceGuard::new(surface_ptr);
    let mut packets = Vec::new();
    let result = unsafe { submit_with_backpressure(&mut packets, component_ptr, &mut guard, &HEVC_PLAN, 333_333) };
    assert!(result.is_ok());
    assert_eq!(submit_call_count(), 2);
    assert_eq!(submit_pointer_at(1), Some(surface_ptr));
    assert_eq!(surface_refcount(), 0);
    drop(guard);
}

/// A hard SubmitInput error surfaces as Err and the guard releases the
/// caller-held ref exactly once.
#[test]
fn test_amf_submit_hard_error_releases_through_guard() {
    mock_reset();
    set_submit_sequence(&[AMF_FAIL]);
    set_query_sequence(&[AMF_REPEAT]);
    let (mut surface, mut component) = make_mock_pair();
    let surface_ptr: *mut c_void = surface.as_mut() as *mut _ as *mut c_void;
    let component_ptr: *mut c_void = component.as_mut() as *mut _ as *mut c_void;
    let mut packets = Vec::new();
    {
        let mut guard = SurfaceGuard::new(surface_ptr);
        let result = unsafe { submit_with_backpressure(&mut packets, component_ptr, &mut guard, &AVC_PLAN, 333_333) };
        assert!(result.is_err(), "hard error must propagate as Err");
        let msg = format!("{:#}", result.unwrap_err());
        assert!(msg.contains("AMF_FAIL"), "error names the result code: {msg}");
    }
    assert_eq!(surface_refcount(), 0, "hard-error path must release exactly once via the guard");
}

/// Saturated forever → bail after INPUT_FULL_MAX_RETRIES + 1 attempts.
#[test]
fn test_amf_submit_bounded_retry_budget() {
    mock_reset();
    let n = INPUT_FULL_MAX_RETRIES as usize + 2;
    set_submit_sequence(&vec![AMF_INPUT_FULL; n]);
    set_query_sequence(&vec![AMF_REPEAT; n]);
    let (mut surface, mut component) = make_mock_pair();
    let surface_ptr: *mut c_void = surface.as_mut() as *mut _ as *mut c_void;
    let component_ptr: *mut c_void = component.as_mut() as *mut _ as *mut c_void;
    let mut packets = Vec::new();
    {
        let mut guard = SurfaceGuard::new(surface_ptr);
        let result = unsafe { submit_with_backpressure(&mut packets, component_ptr, &mut guard, &AVC_PLAN, 333_333) };
        assert!(result.is_err(), "stuck backpressure must eventually bail (not spin)");
        assert_eq!(submit_call_count() as u32, INPUT_FULL_MAX_RETRIES + 1);
    }
    assert_eq!(surface_refcount(), 0);
}

// ── FFI layout ────────────────────────────────────────────────

/// `AMFVariantStruct` (core/Variant.h:80-103): 24 bytes, type at 0, union
/// at 8, each constructor tagging the right arm.
#[test]
fn test_amf_variant_layout_and_arms() {
    assert_eq!(std::mem::size_of::<AmfVariant>(), 24);
    assert_eq!(std::mem::align_of::<AmfVariant>(), 8);
    assert_eq!(std::mem::offset_of!(AmfVariant, value), 8);

    let v = AmfVariant::int64(0x0123_4567_89ab_cdef);
    assert_eq!(v.ty, AMF_VARIANT_INT64);
    assert_eq!(v.as_int64(), Some(0x0123_4567_89ab_cdef));
    // Byte view: the LE int64 sits at bytes 8..16.
    let bytes: [u8; 24] = unsafe { std::mem::transmute(v) };
    assert_eq!(&bytes[8..16], &0x0123_4567_89ab_cdefi64.to_le_bytes());
    assert_eq!(&bytes[16..24], &[0u8; 8], "unused tail of the union stays zero");

    let b = AmfVariant::bool_(true);
    assert_eq!(b.ty, AMF_VARIANT_BOOL);
    assert_eq!(b.as_bool(), Some(true));
    assert_eq!(b.as_int64(), None, "tag mismatch reads as None");
    let bytes: [u8; 24] = unsafe { std::mem::transmute(b) };
    assert_eq!(bytes[8], 1, "amf_bool is one byte at the union start");

    let r = AmfVariant::rate(30000, 1001);
    assert_eq!(r.ty, AMF_VARIANT_RATE);
    assert_eq!(r.as_rate(), Some((30000, 1001)));
    let bytes: [u8; 24] = unsafe { std::mem::transmute(r) };
    assert_eq!(&bytes[8..12], &30000u32.to_le_bytes(), "AMFRate.num at 8");
    assert_eq!(&bytes[12..16], &1001u32.to_le_bytes(), "AMFRate.den at 12");

    let e = AmfVariant::empty();
    assert_eq!(e.ty, 0);
    assert_eq!(e.as_int64(), None);
}

/// The IIDs as the runtime sees them in memory: `AMFGuid` is
/// `{u32, u16, u16, u8[8]}` (core/Platform.h:508-521), so the first three
/// fields are little-endian on x86-64 and the tail is raw.
#[test]
fn test_amf_iid_byte_layout() {
    let bytes: [u8; 16] = unsafe { std::mem::transmute(AMF_IID_BUFFER) };
    assert_eq!(&bytes[0..4], &0xb04b_7248u32.to_le_bytes(), "IID_AMFBuffer data1 (Buffer.h:135)");
    assert_eq!(&bytes[4..6], &0xb6f0u16.to_le_bytes());
    assert_eq!(&bytes[6..8], &0x4321u16.to_le_bytes());
    assert_eq!(&bytes[8..16], &[0xb6, 0x91, 0xba, 0xa4, 0x74, 0x0f, 0x9f, 0xcb]);

    let bytes: [u8; 16] = unsafe { std::mem::transmute(AMF_IID_CONTEXT1) };
    assert_eq!(&bytes[0..4], &0xd9e9_f868u32.to_le_bytes(), "IID_AMFContext1 data1 (Context.h:278)");
    assert_eq!(&bytes[4..6], &0x6220u16.to_le_bytes());
    assert_eq!(&bytes[6..8], &0x44c6u16.to_le_bytes());
    assert_eq!(&bytes[8..16], &[0xa2, 0x2f, 0x7c, 0xd6, 0xda, 0xc6, 0x86, 0x46]);
}

/// Property names are `wchar_t` strings: 2-byte code units on Windows,
/// 4-byte on Linux, always null-terminated.
#[test]
fn test_amf_wide_encoding() {
    let w = wide("HevcQP_I");
    assert_eq!(w.len(), "HevcQP_I".len() + 1);
    assert_eq!(*w.last().unwrap(), 0);
    assert_eq!(std::mem::size_of::<AmfWchar>(), if cfg!(windows) { 2 } else { 4 });
    assert_eq!(unsafe { from_wide(w.as_ptr()) }, "HevcQP_I");
}

// ── Config helpers ────────────────────────────────────────────

#[test]
fn test_amf_surface_format_dispatch() {
    assert_eq!(amf_surface_format_for(PixelFormat::Yuv420p).unwrap(), AMF_SURFACE_NV12);
    assert_eq!(amf_surface_format_for(PixelFormat::Yuv420p10le).unwrap(), AMF_SURFACE_P010);
    assert!(amf_surface_format_for(PixelFormat::Yuv422p).is_err());
    assert!(amf_surface_format_for(PixelFormat::Rgb24).is_err());
    assert!(amf_surface_format_for(PixelFormat::Yuv444p10le).is_err());
}

/// `AMF_COLOR_BIT_DEPTH_ENUM` is the literal depth (ColorSpace.h:106-107),
/// not an ordinal — `10`, not `2`.
#[test]
fn test_amf_color_bit_depth_is_literal_depth() {
    assert_eq!(amf_color_bit_depth_for(PixelFormat::Yuv420p), 8);
    assert_eq!(amf_color_bit_depth_for(PixelFormat::Yuv420p10le), 10);
}

#[test]
fn test_amf_transfer_to_h273_codes() {
    assert_eq!(transfer_to_h273(TransferFn::Bt709), 1);
    assert_eq!(transfer_to_h273(TransferFn::St2084), 16);
    assert_eq!(transfer_to_h273(TransferFn::AribStdB67), 18);
    assert_eq!(transfer_to_h273(TransferFn::Linear), 8);
    assert_eq!(transfer_to_h273(TransferFn::Bt470Bg), 4);
    assert_eq!(transfer_to_h273(TransferFn::Unspecified), 1);
}

/// Colour profile enum (ColorSpace.h:46-57): 709 = 1, 2020 = 2, FULL_709 =
/// 7, FULL_2020 = 8; BT.2020 is H.273 matrix 9 or 10.
#[test]
fn test_amf_color_profile_mapping() {
    assert_eq!(amf_color_profile_for(1, false), 1);
    assert_eq!(amf_color_profile_for(1, true), 7);
    assert_eq!(amf_color_profile_for(9, false), 2);
    assert_eq!(amf_color_profile_for(10, true), 8);
    assert_eq!(amf_color_profile_for(6, false), 1, "BT.601 has no AMF profile of its own; 709");
}

#[test]
fn test_amf_frame_rate_rational() {
    assert_eq!(frame_rate_rational(30.0), (30, 1));
    assert_eq!(frame_rate_rational(60.0), (60, 1));
    assert_eq!(frame_rate_rational(29.97), (29970, 1000));
    assert_eq!(frame_rate_rational(23.976), (23976, 1000));
    assert_eq!(frame_rate_rational(0.0), (30, 1), "nonsense falls back to 30");
    assert_eq!(frame_rate_rational(f64::NAN), (30, 1));
}

/// `set_int_property` goes through the property-storage prefix of whatever
/// object it is handed: a surface here, a component elsewhere.
#[test]
fn test_amf_set_int_property_reaches_surface_slot() {
    mock_reset();
    let (mut surface, _) = make_mock_pair();
    let surface_ptr: *mut c_void = surface.as_mut() as *mut _ as *mut c_void;
    unsafe { set_int_property(surface_ptr, "ForcePictureType", 2) }.unwrap();
    let rec = recorded();
    assert_eq!(rec.len(), 1);
    assert_eq!(rec[0].0, "ForcePictureType");
    assert_eq!(rec[0].1.as_int64(), Some(2));
}

// ── Codec plans ───────────────────────────────────────────────

fn plan_names(p: &CodecPlan) -> (&'static str, &'static str, &'static str) {
    (p.component_id, p.force_key.0, p.output_type)
}

/// Component ids and per-surface / per-buffer property names, verbatim from
/// the headers (VideoEncoderVCE.h:45,287,303; VideoEncoderHEVC.h:35,253,267;
/// VideoEncoderAV1.h:35,302,313), and the keyframe predicates against the
/// enum values (IDR = 0 for both H.26x output types, KEY = 0 for AV1).
#[test]
fn test_codec_plans_match_headers() {
    assert_eq!(plan_names(&AVC_PLAN), ("AMFVideoEncoderVCE_AVC", "ForcePictureType", "OutputDataType"));
    assert_eq!(AVC_PLAN.force_key.1, 2, "AMF_VIDEO_ENCODER_PICTURE_TYPE_IDR");
    assert_eq!(AVC_PLAN.key_extras, &["InsertSPS", "InsertPPS"]);
    assert!((AVC_PLAN.is_keyframe)(0) && !(AVC_PLAN.is_keyframe)(1) && !(AVC_PLAN.is_keyframe)(2));

    assert_eq!(plan_names(&HEVC_PLAN), ("AMFVideoEncoderHW_HEVC", "HevcForcePictureType", "HevcOutputDataType"));
    assert_eq!(HEVC_PLAN.force_key.1, 2, "AMF_VIDEO_ENCODER_HEVC_PICTURE_TYPE_IDR");
    assert_eq!(HEVC_PLAN.key_extras, &["HevcInsertHeader"]);
    assert!((HEVC_PLAN.is_keyframe)(0) && !(HEVC_PLAN.is_keyframe)(1));

    assert_eq!(plan_names(&AV1_PLAN), ("AMFVideoEncoderHW_AV1", "Av1ForceFrameType", "Av1OutputFrameType"));
    assert_eq!(AV1_PLAN.force_key.1, 1, "AMF_VIDEO_ENCODER_AV1_FORCE_FRAME_TYPE_KEY");
    assert!(AV1_PLAN.key_extras.is_empty());
    assert!((AV1_PLAN.is_keyframe)(0) && !(AV1_PLAN.is_keyframe)(1) && !(AV1_PLAN.is_keyframe)(2));
}

/// `mark_key_frame` sets the force-key int and every extra bool on the
/// surface, in that order.
#[test]
fn test_mark_key_frame_sets_plan_properties() {
    mock_reset();
    let (mut surface, _) = make_mock_pair();
    let surface_ptr: *mut c_void = surface.as_mut() as *mut _ as *mut c_void;
    unsafe { super::mark_key_frame(surface_ptr, &AVC_PLAN) }.unwrap();
    let rec = recorded();
    let names: Vec<&str> = rec.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["ForcePictureType", "InsertSPS", "InsertPPS"]);
    assert_eq!(rec[0].1.as_int64(), Some(2));
    assert_eq!(rec[1].1.as_bool(), Some(true));
    assert_eq!(rec[2].1.as_bool(), Some(true));

    mock_reset();
    unsafe { super::mark_key_frame(surface_ptr, &HEVC_PLAN) }.unwrap();
    let rec = recorded();
    let names: Vec<&str> = rec.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["HevcForcePictureType", "HevcInsertHeader"]);

    mock_reset();
    unsafe { super::mark_key_frame(surface_ptr, &AV1_PLAN) }.unwrap();
    let rec = recorded();
    assert_eq!(rec.len(), 1);
    assert_eq!(rec[0].0, "Av1ForceFrameType");
    assert_eq!(rec[0].1.as_int64(), Some(1));
}

// ── The installed runtime ─────────────────────────────────────

/// Load the AMF runtime if this machine has one. `None` (with a printed
/// reason) where it does not; the tests that need it then pass trivially
/// and say so — a test that cannot run is not evidence, and it prints that.
fn load_runtime() -> Option<libloading::Library> {
    let attempt = unsafe { libloading::Library::new("libamfrt64.so.1") }
        .or_else(|_| unsafe { libloading::Library::new("libamfrt64.so") })
        .or_else(|_| unsafe { libloading::Library::new("amfrt64.dll") });
    match attempt {
        Ok(lib) => Some(lib),
        Err(e) => {
            eprintln!("SKIPPED: AMF runtime not loadable on this machine: {e}");
            None
        }
    }
}

/// The property-storage ABI against the real runtime. Needs no GPU: an
/// `AMFContext` is a property bag before any device is bound to it. If any
/// slot in the 13-slot prefix were misplaced, or `AMFVariantStruct` the
/// wrong size, this would return garbage or crash — so it is real evidence
/// for the layout the encoder's every `SetProperty` goes through.
#[test]
fn test_amf_runtime_property_storage_abi() {
    let Some(lib) = load_runtime() else { return };
    unsafe {
        let amf_init: libloading::Symbol<super::FnAmfInit> = lib.get(b"AMFInit").expect("AMFInit export");
        let mut factory: *mut c_void = ptr::null_mut();
        let rc = amf_init(super::AMF_VERSION, &mut factory);
        assert_eq!(rc, AMF_OK, "AMFInit(1.4.30)");
        assert!(!factory.is_null());
        let factory_vt = &*(*(factory as *mut super::AmfFactoryObj)).vtbl;

        let mut ctx: *mut c_void = ptr::null_mut();
        assert_eq!((factory_vt.create_context)(factory, &mut ctx), AMF_OK, "CreateContext");
        assert!(!ctx.is_null());
        let ctx_vt = &*(*(ctx as *mut super::AmfContextObj)).vtbl;
        let ps = &ctx_vt.ps;

        // SetProperty / GetProperty round trip through the by-value variant.
        set_int_property(ctx, "RivetAbiProbe", 0x1234_5678_9abc).expect("SetProperty on a context");
        assert_eq!(super::get_int_property(ctx, "RivetAbiProbe"), Some(0x1234_5678_9abc));
        // HasProperty returns amf_bool; GetPropertyCount counts what we set.
        let name = wide("RivetAbiProbe");
        assert_eq!((ps.has_property)(ctx, name.as_ptr()), 1, "HasProperty(set) == true");
        let missing = wide("RivetAbiMissing");
        assert_eq!((ps.has_property)(ctx, missing.as_ptr()), 0, "HasProperty(unset) == false");
        assert!((ps.get_property_count)(ctx) >= 1, "GetPropertyCount");
        // GetProperty on a missing name is AMF_NOT_FOUND (= 11), the value
        // the old decoder misread as "not AMF-capable".
        let mut var = AmfVariant::empty();
        assert_eq!((ps.get_property)(ctx, missing.as_ptr(), &mut var), AMF_NOT_FOUND);
        // A bool round-trips as a bool.
        super::set_bool_property(ctx, "RivetAbiBool", true).unwrap();
        let bname = wide("RivetAbiBool");
        let mut var = AmfVariant::empty();
        assert_eq!((ps.get_property)(ctx, bname.as_ptr(), &mut var), AMF_OK);
        assert_eq!(var.as_bool(), Some(true));
        // And an AMFRate.
        super::set_rate_property(ctx, "RivetAbiRate", 30000, 1001).unwrap();
        let rname = wide("RivetAbiRate");
        let mut var = AmfVariant::empty();
        assert_eq!((ps.get_property)(ctx, rname.as_ptr(), &mut var), AMF_OK);
        assert_eq!(var.as_rate(), Some((30000, 1001)));

        // QueryInterface(IID_AMFContext1): the IID bytes and slot 2.
        let mut ctx1: *mut c_void = ptr::null_mut();
        assert_eq!((ps.query_interface)(ctx, &AMF_IID_CONTEXT1, &mut ctx1), AMF_OK, "QI(AMFContext1)");
        assert!(!ctx1.is_null());
        // Acquire / Release (slots 0 / 1) return the new count.
        let after_acquire = (ps.acquire)(ctx);
        let after_release = (ps.release)(ctx);
        assert_eq!(after_acquire - 1, after_release, "Acquire then Release nets to zero");
        let ctx1_vt = &*(*(ctx1 as *mut super::AmfContext1Obj)).vtbl;
        let _ = (ctx1_vt.base.ps.release)(ctx1);

        // Teardown through slots 13 and 1.
        assert_eq!((ctx_vt.terminate)(ctx), AMF_OK, "Terminate on an unbound context");
        assert_eq!((ps.release)(ctx), 0, "final Release returns 0");
    }
    eprintln!("AMF runtime property-storage ABI: verified against the installed runtime");
}

/// The whole `AmfEncoder::new` path on this machine, for each codec. On an
/// AMF-capable GPU it succeeds (and the session is torn down cleanly); on
/// this box it must fail with a message that says why, and — the point —
/// must not crash while tearing the half-built session down.
#[test]
fn test_amf_encoder_new_on_this_machine_fails_or_succeeds_cleanly() {
    use crate::encode::EncoderConfig;
    use crate::frame::VideoCodec;
    if load_runtime().is_none() {
        return;
    }
    for codec in [VideoCodec::H264, VideoCodec::H265, VideoCodec::Av1] {
        let cfg = EncoderConfig {
            width: 640,
            height: 480,
            frame_rate: 30.0,
            codec,
            ..Default::default()
        };
        match super::AmfEncoder::new(cfg, 0) {
            Ok(enc) => {
                eprintln!("{codec:?}: AmfEncoder::new succeeded on this machine (AMF-capable GPU present)");
                drop(enc);
            }
            Err(e) => {
                let msg = format!("{e:#}");
                eprintln!("{codec:?}: AmfEncoder::new failed cleanly: {msg}");
                assert!(msg.contains("AMF"), "the error names AMF: {msg}");
            }
        }
    }
}
