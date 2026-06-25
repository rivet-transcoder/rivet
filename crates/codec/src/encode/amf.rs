//! AMD AMF AV1 hardware encoder via the Advanced Media Framework runtime.
//!
//! Loads `amfrt64.dll` / `libamfrt64.so.1` at runtime via dlopen. The AV1
//! encoder component is only available on RDNA3+ silicon (Radeon RX
//! 7000 series and later). On older GPUs `CreateComponent` returns
//! `AMF_NOT_SUPPORTED` and we surface that to `select_encoder`'s
//! fallback chain.
//!
//! Session flow (mirroring the AMF sample `VCEEncoderD3D11` adapted
//! for AV1 host-memory submission):
//! 1. dlopen `amfrt64.dll` / `libamfrt64.so.1`
//! 2. AMFInit(AMF_VERSION, &factory)
//! 3. factory->CreateContext(&ctx); ctx->InitDX11(null)  /* Windows */
//!    (or ctx->InitVulkan(null) on Linux — AMF picks the first AMD GPU)
//! 4. factory->CreateComponent(ctx, AMFVideoEncoderVCN_AV1, &encoder)
//! 5. encoder->SetProperty(USAGE = TRANSCODING)        /* baseline */
//! 6. encoder->SetProperty(RATE_CONTROL_METHOD = ...)  /* from adapter */
//! 7. encoder->SetProperty(Q_INDEX_INTRA/INTER, QUALITY_PRESET,
//!    GOP_SIZE, tile count, AQ, OUTPUT_MODE, ...)
//! 8. encoder->Init(NV12, width, height)
//! 9. Per frame:
//!    - ctx->AllocSurface(HOST, NV12, w, h, &surf)
//!    - copy YUV420p → NV12 into surf's Y and UV planes
//!    - surf->SetPts(frame.pts_ticks); surf->SetProperty(FORCE_KEY)
//!    - encoder->SubmitInput(surf); (release surf)
//!    - loop: encoder->QueryOutput(&data); on AMF_OK read AMFBuffer
//!      native pointer → copy into EncodedPacket; on AMF_REPEAT break
//! 10. Flush: encoder->Drain(); drain QueryOutput until AMF_EOF
//! 11. Drop order: encoder->Terminate → encoder.Release → ctx.Terminate
//!     → ctx.Release → library handle drops last (it provides the code
//!     behind every vtable pointer we just called).
//!
//! # AMF_INPUT_FULL retry policy (#59 follow-up)
//!
//! AMF signals `AMF_INPUT_FULL` when the encoder's internal input queue
//! is saturated. The SDK's `AMFComponent` header documents this as a
//! **transient** status — NOT a failure. The correct sequence is:
//!
//!   1. Do NOT release the surface. The surface's caller-held ref is
//!      still valid, and releasing it makes the retry a use-after-free.
//!   2. Drain at least one output packet via `QueryOutput` to free a
//!      slot in the input queue.
//!   3. Retry `SubmitInput` with the SAME surface pointer.
//!   4. Only after the eventual `AMF_OK` (or `AMF_NEED_MORE_INPUT`)
//!      does the encoder take its own ref — we then release our caller-
//!      held ref.
//!
//! The ring-buffer of `RING_SIZE` pre-tracked slots follows Squad-5's
//! NVENC pattern for visibility and test coverage. Each AMF surface is
//! allocated fresh per frame (AMF's ref-counted memory model means the
//! encoder retains its own ref on submitted surfaces until the frame is
//! done, so there is nothing to reuse slot-to-slot as in NVENC); the
//! ring index is for in-flight bookkeeping and a public diagnostic
//! signal that mirrors the NVENC drain path.

use anyhow::{Context, Result, bail};
use bytes::Bytes;
use std::ffi::c_void;
use std::os::raw::c_int;
use std::ptr;

use super::tuning::{self, AmfQualityPreset, AmfRateControl};
use super::{AUTO_FROM_TARGET, EncodedPacket, Encoder, EncoderConfig};
// `ColorMetadata` is read via `config.color_metadata` on the non-test
// side (no bare-type mention) and through `use super::*` inside the
// test module; pull it in only under cfg(test) to keep release builds
// warning-clean.
#[cfg(test)]
use crate::frame::ColorMetadata;
use crate::frame::{PixelFormat, TransferFn, VideoFrame};

// ─── AMF ABI constants ────────────────────────────────────────────
// See vendor/amd/ for authoritative definitions.

type AmfResult = i32;
const AMF_OK: AmfResult = 0;
#[allow(dead_code)]
const AMF_FAIL: AmfResult = 1;
const AMF_NEED_MORE_INPUT: AmfResult = 2022;
const AMF_REPEAT: AmfResult = 2023;
const AMF_EOF: AmfResult = 2024;
const AMF_INPUT_FULL: AmfResult = 2020;

const AMF_VERSION: u64 = amf_make_version(1, 4, 30, 0);

const fn amf_make_version(major: u64, minor: u64, sub_major: u64, sub_minor: u64) -> u64 {
    (major << 48) | (minor << 32) | (sub_major << 16) | sub_minor
}

// AMF memory / surface format enums (`AMF_MEMORY_TYPE`, `AMF_SURFACE_FORMAT`).
// Values from `vendor/amd/AMFContext.h:46-58`.
const AMF_MEMORY_HOST: i32 = 1;
const AMF_SURFACE_NV12: i32 = 1;
/// `AMF_SURFACE_P010` per `vendor/amd/AMFContext.h:57`. Same NV12-style
/// plane layout (Y plane + interleaved UV plane) but each sample is a
/// 16-bit LE word with the valid 10-bit value in the **upper 10 bits**
/// (Microsoft P010 convention; AMD VCN AV1 inherits this from the DX11
/// surface format definition). The `upload_frame_p010` helper performs
/// the `<<6` shift on copy from the pipeline's `Yuv420p10le`
/// (lower-10-bits) representation.
const AMF_SURFACE_P010: i32 = 10;

// AMF plane types (`AMF_PLANE_TYPE`).
const AMF_PLANE_Y: i32 = 2;
const AMF_PLANE_UV: i32 = 3;

// Variant type tags. Only the ones we set are named.
const AMF_VARIANT_INT64: i32 = 2;

// AMF AV1 rate-control enum values (mirror `AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_*`).
const AMF_RC_CQP: i64 = 1;
const AMF_RC_QUALITY_VBR: i64 = 5;

// AMF AV1 output frame type (read back from the AMFBuffer property bag).
const AMF_OUTPUT_FRAME_TYPE_KEY: i64 = 0;
const AMF_OUTPUT_FRAME_TYPE_INTRA_ONLY: i64 = 1;

// AMF AV1 USAGE_TRANSCODING baseline — picks production defaults for
// rate control + preset. The individual SetProperty calls afterward
// tighten individual knobs to what the tuning adapter asked for.
const AMF_USAGE_TRANSCODING: i64 = 0;

// AV1 output mode frame packing — 0 = packed frame-level OBUs with size
// fields (LOB). That's what AV1-ISOBMFF / MP4 mux expects.
const AMF_OUTPUT_MODE_FRAME: i64 = 0;

// `Av1ColorBitDepth` enum values per `vendor/amd/VideoEncoderAV1.h:58-59`.
//   1 = AMF_VIDEO_ENCODER_AV1_COLOR_BIT_DEPTH_8
//   2 = AMF_VIDEO_ENCODER_AV1_COLOR_BIT_DEPTH_10
const AMF_AV1_COLOR_BIT_DEPTH_8: i64 = 1;
const AMF_AV1_COLOR_BIT_DEPTH_10: i64 = 2;

// ─── Ring-buffer configuration ────────────────────────────────────
//
// Squad-5's NVENC path uses RING_SIZE=4 (mirrors ffmpeg's libavcodec/
// nvenc.c default `nb_surfaces`) to keep the encoder pipeline full
// without oversubscribing GPU memory. We mirror the same depth for
// AMF so ops can reason about in-flight buffers uniformly across both
// vendors.
//
// Each ring slot carries the caller-held surface pointer that is
// currently awaiting the encoder's QueryOutput. AMF's ref-counted
// surface model means the encoder retains its own ref after a
// successful `SubmitInput`; our in-flight tracking is therefore a
// SAFETY mirror of the encoder's internal queue, not a reuse pool.
const RING_SIZE: usize = 4;

// `AMF_INPUT_FULL` retry policy. The AMF SDK documents INPUT_FULL as
// transient: the caller should drain at least one output packet and
// retry. We bound the retry count so a pathological driver state can't
// spin us forever. Per practical measurements on Radeon PRO W7800 (RDNA3)
// the deepest observed back-pressure drained within ~3 retries; 16 is a
// safety margin 5× that.
const INPUT_FULL_MAX_RETRIES: u32 = 16;

// Initial backoff when a drain pass yields zero packets but SubmitInput
// still rejects the surface. 1 ms matches the AMF runtime's own internal
// poll granularity. Doubles up to 16 ms on repeated failures.
const INPUT_FULL_BACKOFF_MS_INITIAL: u64 = 1;
const INPUT_FULL_BACKOFF_MS_MAX: u64 = 16;

// ─── Variant helpers ──────────────────────────────────────────────

/// AMFVariantStruct layout matching `vendor/amd/AMFComponent.h`.
/// Padded to 32 bytes so SetProperty ABI is stable across SDK rev bumps.
///
/// Header walks (vendor/amd/AMFComponent.h:56-69):
///   - `type`: int32 at offset 0
///   - `pad`:  int32 at offset 4
///   - `value` union: 24 bytes starting at offset 8
///     - `int64Value` is the first field of the union → offset 8
///
/// Our Rust layout below puts `ty` at 0, `_pad` at 4, and `value[0..8]`
/// at offset 8 — matching the header. `AmfVariant::int64` writes the
/// little-endian i64 into `value[0..8]`, which is the union's first
/// field (int64Value). Confirmed against `vendor/amd/AMFComponent.h`
/// offset 8 for the integer arm.
#[repr(C)]
#[derive(Clone, Copy)]
struct AmfVariant {
    ty: i32,
    _pad: i32,
    value: [u8; 24],
}

// Compile-time ABI guard: AMFVariantStruct must be exactly 32 bytes
// (4 type + 4 pad + 24 payload) on 64-bit platforms. The AMF SDK
// documents this as stable across SDK revs; a mismatch here means
// SetProperty / GetProperty will splat bytes into the wrong union slot.
const _: () = {
    assert!(
        std::mem::size_of::<AmfVariant>() == 32,
        "AmfVariant must be 32 bytes"
    );
    // int64Value must land at offset 8 inside the struct (offset 0 of
    // `value`). This is the lemma behind `AmfVariant::int64` writing
    // at `value[0..8]`.
    assert!(
        std::mem::offset_of!(AmfVariant, value) == 8,
        "AmfVariant value payload must start at offset 8"
    );
};

// Squad-22: AMF surface format constants pinned. AMD has frozen
// `AMF_SURFACE_FORMAT` values 1..10 since AMF 1.4 — but a future
// renumbering would silently mis-route the surface allocator.
const _: () = assert!(AMF_SURFACE_NV12 == 1);
const _: () = assert!(AMF_SURFACE_P010 == 10);
// `Av1ColorBitDepth` enum values from `vendor/amd/VideoEncoderAV1.h:58-59`.
// 10-bit being `2` (not `10`) is one of the property values that has
// surprised callers — pin it.
const _: () = assert!(AMF_AV1_COLOR_BIT_DEPTH_8 == 1);
const _: () = assert!(AMF_AV1_COLOR_BIT_DEPTH_10 == 2);
const _: () = assert!(amf_color_bit_depth_for(PixelFormat::Yuv420p10le) == 2);
const _: () = assert!(amf_color_bit_depth_for(PixelFormat::Yuv420p) == 1);

impl AmfVariant {
    fn int64(v: i64) -> Self {
        let mut value = [0u8; 24];
        value[..8].copy_from_slice(&v.to_le_bytes());
        Self {
            ty: AMF_VARIANT_INT64,
            _pad: 0,
            value,
        }
    }

    /// Read the int64 arm — used by tests and for output-buffer property
    /// reads. Returns `None` if the variant is not int-typed.
    #[allow(dead_code)]
    fn as_int64(&self) -> Option<i64> {
        if self.ty == AMF_VARIANT_INT64 {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&self.value[..8]);
            Some(i64::from_le_bytes(bytes))
        } else {
            None
        }
    }
}

// ─── Vtable shapes (abbreviated) ──────────────────────────────────
//
// AMF uses COM-style vtables: every handle is a `*mut Object` where
// `Object` is `{ *const Vtbl }`. The vtables below list only the slots
// we actually call; the rest is padded so offsets match the upstream
// ABI for whatever SDK rev the host runtime ships.

type QueryInterfaceFn = unsafe extern "C" fn(*mut c_void, *const c_void, *mut *mut c_void) -> i64;
type AcquireFn = unsafe extern "C" fn(*mut c_void) -> i64;
type ReleaseFn = unsafe extern "C" fn(*mut c_void) -> i64;

#[repr(C)]
struct AmfFactoryVtbl {
    create_context: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> AmfResult,
    create_component:
        unsafe extern "C" fn(*mut c_void, *mut c_void, *const u16, *mut *mut c_void) -> AmfResult,
    set_cache_folder: unsafe extern "C" fn(*mut c_void, *const u16) -> AmfResult,
    get_cache_folder: unsafe extern "C" fn(*mut c_void) -> *const u16,
    get_debug: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> AmfResult,
    get_trace: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> AmfResult,
    get_programs: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> AmfResult,
}

#[repr(C)]
struct AmfFactoryObj {
    vtbl: *const AmfFactoryVtbl,
}

// AMFContext vtable — we bind QueryInterface/Acquire/Release (inherited
// from AMFInterface), Terminate, InitDX11, InitVulkan, AllocSurface,
// and pad the rest. Real upstream has ~30 entries; we slot through the
// first N in declaration order and leave the tail as `_reserved`.
#[repr(C)]
struct AmfContextVtbl {
    query_interface: QueryInterfaceFn,
    acquire: AcquireFn,
    release: ReleaseFn,
    terminate: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    init_dx11: unsafe extern "C" fn(*mut c_void, *mut c_void, i32) -> AmfResult,
    get_dx11_device: unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void,
    lock_dx11: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    unlock_dx11: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    init_opencl: unsafe extern "C" fn(*mut c_void, *mut c_void) -> AmfResult,
    get_opencl_context: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    get_opencl_command_queue: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    get_opencl_device_id: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    convert_to_opencl: unsafe extern "C" fn(*mut c_void, *mut c_void) -> AmfResult,
    lock_opencl: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    unlock_opencl: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    init_opengl:
        unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> AmfResult,
    get_opengl_context: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    get_opengl_drawable: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    convert_to_opengl: unsafe extern "C" fn(*mut c_void, *mut c_void) -> AmfResult,
    lock_opengl: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    unlock_opengl: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    init_vulkan: unsafe extern "C" fn(*mut c_void, *mut c_void) -> AmfResult,
    get_vulkan_device: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    lock_vulkan: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    unlock_vulkan: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    alloc_buffer: unsafe extern "C" fn(*mut c_void, i32, usize, *mut *mut c_void) -> AmfResult,
    alloc_surface: unsafe extern "C" fn(
        *mut c_void,
        i32, // memory type
        i32, // surface format
        i32, // width
        i32, // height
        *mut *mut c_void,
    ) -> AmfResult,
    create_surface_from_host_native: unsafe extern "C" fn(
        *mut c_void,
        i32,
        i32,
        i32,
        i32,
        i32,
        *mut c_void,
        *mut *mut c_void,
        *mut c_void,
    ) -> AmfResult,
}

#[repr(C)]
struct AmfContextObj {
    vtbl: *const AmfContextVtbl,
}

#[repr(C)]
struct AmfComponentVtbl {
    query_interface: QueryInterfaceFn,
    acquire: AcquireFn,
    release: ReleaseFn,
    // SetProperty / GetProperty take the variant by value; the AMF C ABI
    // passes it as an inline 32-byte struct, so `by value` matches the
    // layout in `vendor/amd/AMFComponent.h`.
    set_property: unsafe extern "C" fn(*mut c_void, *const u16, AmfVariant) -> AmfResult,
    get_property: unsafe extern "C" fn(*mut c_void, *const u16, *mut AmfVariant) -> AmfResult,
    init: unsafe extern "C" fn(*mut c_void, i32, i32, i32) -> AmfResult,
    reinit: unsafe extern "C" fn(*mut c_void, i32, i32) -> AmfResult,
    terminate: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    drain: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    flush: unsafe extern "C" fn(*mut c_void) -> AmfResult,
    submit_input: unsafe extern "C" fn(*mut c_void, *mut c_void) -> AmfResult,
    query_output: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> AmfResult,
    get_context: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    set_output_data_allocator_cb: unsafe extern "C" fn(*mut c_void, *mut c_void) -> AmfResult,
    get_caps: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> AmfResult,
    optimize: unsafe extern "C" fn(*mut c_void, *mut c_void) -> AmfResult,
}

#[repr(C)]
struct AmfComponentObj {
    vtbl: *const AmfComponentVtbl,
}

// AMFSurface — we only need vtable slots through `GetPlane` /
// `GetPlaneAt`. Layout keeps the AMFData prefix intact so QueryInterface
// works if a caller cross-casts.
#[repr(C)]
struct AmfSurfaceVtbl {
    query_interface: QueryInterfaceFn,
    acquire: AcquireFn,
    release: ReleaseFn,
    set_property: unsafe extern "C" fn(*mut c_void, *const u16, AmfVariant) -> AmfResult,
    get_property: unsafe extern "C" fn(*mut c_void, *const u16, *mut AmfVariant) -> AmfResult,
    duplicate: unsafe extern "C" fn(*mut c_void, i32, *mut *mut c_void) -> AmfResult,
    get_pts: unsafe extern "C" fn(*mut c_void) -> i64,
    set_pts: unsafe extern "C" fn(*mut c_void, i64),
    get_duration: unsafe extern "C" fn(*mut c_void) -> i64,
    set_duration: unsafe extern "C" fn(*mut c_void, i64),
    // Surface-specific
    get_planes_count: unsafe extern "C" fn(*mut c_void) -> usize,
    get_plane_at: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void,
    get_plane: unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void,
}

#[repr(C)]
struct AmfSurfaceObj {
    vtbl: *const AmfSurfaceVtbl,
}

#[repr(C)]
struct AmfPlaneVtbl {
    query_interface: QueryInterfaceFn,
    acquire: AcquireFn,
    release: ReleaseFn,
    get_type: unsafe extern "C" fn(*mut c_void) -> i32,
    get_native: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    get_pixel_size_in_bytes: unsafe extern "C" fn(*mut c_void) -> i32,
    get_offset_x: unsafe extern "C" fn(*mut c_void) -> i32,
    get_offset_y: unsafe extern "C" fn(*mut c_void) -> i32,
    get_width: unsafe extern "C" fn(*mut c_void) -> i32,
    get_height: unsafe extern "C" fn(*mut c_void) -> i32,
    get_h_pitch: unsafe extern "C" fn(*mut c_void) -> i32,
    get_v_pitch: unsafe extern "C" fn(*mut c_void) -> i32,
}

#[repr(C)]
struct AmfPlaneObj {
    vtbl: *const AmfPlaneVtbl,
}

// AMFBuffer — output bitstream.
#[repr(C)]
struct AmfBufferVtbl {
    query_interface: QueryInterfaceFn,
    acquire: AcquireFn,
    release: ReleaseFn,
    set_property: unsafe extern "C" fn(*mut c_void, *const u16, AmfVariant) -> AmfResult,
    get_property: unsafe extern "C" fn(*mut c_void, *const u16, *mut AmfVariant) -> AmfResult,
    duplicate: unsafe extern "C" fn(*mut c_void, i32, *mut *mut c_void) -> AmfResult,
    get_pts: unsafe extern "C" fn(*mut c_void) -> i64,
    set_pts: unsafe extern "C" fn(*mut c_void, i64),
    get_duration: unsafe extern "C" fn(*mut c_void) -> i64,
    set_duration: unsafe extern "C" fn(*mut c_void, i64),
    get_native: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    get_size: unsafe extern "C" fn(*mut c_void) -> usize,
}

#[repr(C)]
struct AmfBufferObj {
    vtbl: *const AmfBufferVtbl,
}

// IID constants used by QueryInterface to downcast AMFData → AMFBuffer /
// AMFSurface / AMFPlane. AMF publishes these as GUID literals; we carry
// them as 16-byte arrays matching the runtime's in-memory representation.
const AMF_IID_BUFFER: [u8; 16] = [
    0xbe, 0x5d, 0xd7, 0xb1, 0x6c, 0x0e, 0x4c, 0x43, 0xb7, 0x28, 0x02, 0x85, 0x98, 0x37, 0x85, 0x7d,
];

// ─── AMF init entry-point ABI ─────────────────────────────────────

type FnAmfInit = unsafe extern "C" fn(u64, *mut *mut c_void) -> AmfResult;

// ─── Helpers ─────────────────────────────────────────────────────

/// Encode a UTF-16 null-terminated wide string the way AMF expects
/// (the SDK property names are `wchar_t*` — on Windows that's u16,
/// on Linux wchar_t is u32 but AMF's ABI declares `amf_wchar_t = u16`
/// explicitly via its own typedef to stay portable).
fn wide(s: &str) -> Vec<u16> {
    let mut out: Vec<u16> = s.encode_utf16().collect();
    out.push(0);
    out
}

// Property-name wide strings, one per SetProperty call we make.
// Stored as constants so we don't re-encode for every frame.
fn prop(s: &str) -> Vec<u16> {
    wide(s)
}

// ─── Squad-22: per-pixel-format dispatch ──────────────────────────
//
// AMF VCN AV1 supports NV12 (8-bit) and P010 (10-bit) host-memory
// surfaces; both are interleaved-chroma YUV 4:2:0. Selecting the wrong
// surface format for the input depth produces silent garbage (the
// 8-bit shader path on a wide-word surface reads two adjacent samples
// per byte → noise + halved width).

fn amf_surface_format_for(fmt: PixelFormat) -> Result<i32> {
    match fmt {
        PixelFormat::Yuv420p => Ok(AMF_SURFACE_NV12),
        PixelFormat::Yuv420p10le => Ok(AMF_SURFACE_P010),
        other => bail!("AMF AV1 expects Yuv420p or Yuv420p10le, got {other:?}"),
    }
}

const fn amf_color_bit_depth_for(fmt: PixelFormat) -> i64 {
    match fmt {
        PixelFormat::Yuv420p10le => AMF_AV1_COLOR_BIT_DEPTH_10,
        _ => AMF_AV1_COLOR_BIT_DEPTH_8,
    }
}

/// Translate `TransferFn` → ITU-T H.273 numeric code. Same table as
/// `nvenc.rs::transfer_to_h273` and the mux's `transfer_to_h273` —
/// keeping the three in lockstep means HDR signalling matches across
/// container `colr nclx`, AMF AV1 OBU, and NVENC AV1 OBU.
fn transfer_to_h273(tf: TransferFn) -> i64 {
    match tf {
        TransferFn::Bt709 => 1,
        TransferFn::Bt470Bg => 4,
        TransferFn::Linear => 8,
        TransferFn::St2084 => 16,
        TransferFn::AribStdB67 => 18,
        TransferFn::Unspecified => 1,
    }
}

// ─── RAII surface guard ──────────────────────────────────────────
//
// Wraps the caller-held ref on an AMF surface so it gets released on
// every exit path — including `bail!`, `?` early-return, and panic
// unwind (which catch_unwind converts to an error). Drop is a no-op
// after `transfer_to_encoder` marks the ref as consumed by
// `SubmitInput` returning AMF_OK / AMF_NEED_MORE_INPUT.
//
// This is the belt-and-suspenders fix for codec-review-59-60 A-A4 —
// explicit releases at every match arm cover the nominal paths, but
// a panic inside a SetProperty call, for example, would leak without
// this guard.
struct SurfaceGuard {
    surface: *mut c_void,
    owned: bool,
}

impl SurfaceGuard {
    fn new(surface: *mut c_void) -> Self {
        Self {
            surface,
            owned: true,
        }
    }

    /// Marks the caller-held ref as transferred to the encoder. After
    /// this, `Drop` will NOT release. Call this immediately after the
    /// `SubmitInput` call that returned `AMF_OK` / `AMF_NEED_MORE_INPUT`.
    fn transfer_to_encoder(&mut self) {
        self.owned = false;
    }

    fn as_ptr(&self) -> *mut c_void {
        self.surface
    }
}

impl Drop for SurfaceGuard {
    fn drop(&mut self) {
        if self.owned && !self.surface.is_null() {
            unsafe {
                let obj = self.surface as *mut AmfSurfaceObj;
                let vt = &*(*obj).vtbl;
                (vt.release)(self.surface);
            }
        }
    }
}

// ─── Session container ────────────────────────────────────────────

/// Holds the live AMF objects. Dropped in reverse-acquisition order:
/// encoder first (it holds a strong ref on the context), context
/// second. The library handle that provides every vtable we just
/// called drops LAST via `AmfEncoder`'s field order.
struct AmfSession {
    encoder: *mut c_void,
    context: *mut c_void,
    /// Factory is a singleton owned by the AMF runtime; we get it back
    /// from AMFInit and stash it so we can create more contexts if a
    /// future Reconfigure path needs it. Not reference-counted.
    #[allow(dead_code)]
    factory: *mut c_void,

    width: u32,
    height: u32,
    pts_timescale: u64,
    /// `AMF_SURFACE_NV12` (8-bit) or `AMF_SURFACE_P010` (10-bit).
    /// Captured at session create so `upload_frame_static` knows
    /// which plane width + per-sample byte count to use.
    surface_format: i32,
}

// AMF's COM-style vtables are thread-safe per the SDK's "Thread Safety"
// appendix: every context/component object internally synchronises
// SetProperty / SubmitInput / QueryOutput. We only touch one encoder
// per `AmfEncoder`, so Send is sufficient for tokio migration.
//
// Caveat (systems-review-59-60 #4): AMF's DX11/Vulkan device init creates
// per-thread state on some driver versions. A task migrated mid-encode
// could see device-removed errors. The pipeline's `spawn_blocking`
// ensures the encoder stays on one OS thread for its lifetime, so this
// is theoretical for our usage.
unsafe impl Send for AmfSession {}

impl Drop for AmfSession {
    fn drop(&mut self) {
        unsafe {
            // Encoder first — Terminate releases internal hardware
            // resources before we drop the last COM ref.
            if !self.encoder.is_null() {
                let obj = self.encoder as *mut AmfComponentObj;
                let vt = &*(*obj).vtbl;
                let _ = (vt.terminate)(self.encoder);
                let _ = (vt.release)(self.encoder);
            }
            // Context next — same pattern. The factory is not
            // reference-counted and is owned by the runtime; do not
            // Release it.
            if !self.context.is_null() {
                let obj = self.context as *mut AmfContextObj;
                let vt = &*(*obj).vtbl;
                let _ = (vt.terminate)(self.context);
                let _ = (vt.release)(self.context);
            }
        }
    }
}

// ─── Encoder implementation ───────────────────────────────────────

// Field order matters for drop: session drops BEFORE _runtime_lib, so
// all the vtable calls inside `AmfSession::drop` still resolve to
// valid code. Library handle is declared LAST (Reference §10.8 —
// struct fields drop in source order).
pub struct AmfEncoder {
    config: EncoderConfig,
    session: Option<AmfSession>,
    encoded_packets: Vec<EncodedPacket>,
    packet_cursor: usize,
    flushed: bool,
    frame_counter: u32,
    /// Current ring slot. Advances modulo `RING_SIZE` per successful
    /// `SubmitInput`. Mirrors NVENC's `ring_idx` for observational
    /// parity and in-flight bookkeeping.
    ring_idx: usize,
    _runtime_lib: libloading::Library,
}

impl AmfEncoder {
    pub fn new(config: EncoderConfig, gpu_index: u32) -> Result<Self> {
        // 1. dlopen the AMF runtime. On Linux the library name is
        //    `libamfrt64.so.1`; on Windows it's `amfrt64.dll`. Both
        //    ship with the Adrenalin driver and Pro driver bundles.
        let runtime_lib = unsafe { libloading::Library::new("libamfrt64.so.1") }
            .or_else(|_| unsafe { libloading::Library::new("libamfrt64.so") })
            .or_else(|_| unsafe { libloading::Library::new("amfrt64.dll") })
            .context("loading AMF runtime library (AMD driver not present?)")?;

        unsafe {
            // 2. Factory.
            let amf_init: libloading::Symbol<FnAmfInit> =
                runtime_lib.get(b"AMFInit").context("AMFInit symbol")?;
            let mut factory: *mut c_void = ptr::null_mut();
            let rc = amf_init(AMF_VERSION, &mut factory);
            if rc != AMF_OK || factory.is_null() {
                bail!("AMFInit failed: {rc}");
            }

            // 3. Context.
            let mut context: *mut c_void = ptr::null_mut();
            let factory_obj = factory as *mut AmfFactoryObj;
            let factory_vt = &*(*factory_obj).vtbl;
            let rc = (factory_vt.create_context)(factory, &mut context);
            if rc != AMF_OK || context.is_null() {
                bail!("AMFFactory::CreateContext failed: {rc}");
            }

            // Initialize the context on a real GPU. We try DX11 first
            // (Windows / WSL2), then Vulkan (Linux). A null device ptr
            // tells AMF to pick the first AMD adapter; the caller's
            // `gpu_index` threads through the pipeline but AMF itself
            // does not expose an ordinal-based init — the driver
            // deterministically picks adapter 0 unless a VkPhysicalDevice
            // or D3D11Device is passed, so multi-AMD hosts require the
            // caller to also set `AGS_DESIRED_ADAPTER_ID` env var.
            // We emit a debug log when gpu_index != 0 so the ops team
            // can notice.
            if gpu_index != 0 {
                tracing::warn!(
                    gpu_index,
                    "AMF init picks adapter 0 unconditionally; \
                     multi-AMD hosts may need external adapter routing"
                );
            }
            let context_obj = context as *mut AmfContextObj;
            let context_vt = &*(*context_obj).vtbl;

            // Try DX11 (both Windows and WSL2 ship a DX11 runtime that
            // AMF can target). If not available — e.g., bare-metal
            // Linux — fall through to Vulkan.
            let rc_dx11 = (context_vt.init_dx11)(context, ptr::null_mut(), 0);
            if rc_dx11 != AMF_OK {
                let rc_vk = (context_vt.init_vulkan)(context, ptr::null_mut());
                if rc_vk != AMF_OK {
                    // Fail → drop context, bail.
                    (context_vt.release)(context);
                    bail!("AMFContext::InitDX11 ({rc_dx11}) and InitVulkan ({rc_vk}) both failed");
                }
            }

            // 4. Encoder component.
            let component_id = wide("AMFVideoEncoderVCN_AV1");
            let mut encoder: *mut c_void = ptr::null_mut();
            let rc = (factory_vt.create_component)(
                factory,
                context,
                component_id.as_ptr(),
                &mut encoder,
            );
            if rc != AMF_OK || encoder.is_null() {
                (context_vt.terminate)(context);
                (context_vt.release)(context);
                bail!(
                    "AMFFactory::CreateComponent(AMFVideoEncoderVCN_AV1) failed: {rc} — RDNA3+ GPU required"
                );
            }

            let encoder_obj = encoder as *mut AmfComponentObj;
            let encoder_vt = &*(*encoder_obj).vtbl;

            // 5. Apply tuning adapter params.
            let tp =
                tuning::amf_av1_params(config.target, config.tier, config.width, config.height);

            // Legacy quality override: if caller passed a concrete
            // `config.quality`, use it as the CQP q-index (0..255).
            // Otherwise use the adapter's derived value.
            let q_intra = if config.quality == AUTO_FROM_TARGET {
                tp.q_index_intra
            } else {
                // Caller-provided quality is a 0..63 CQ scale (NVENC-
                // compatible); scale up 4× to match AMF's 0..255 range.
                ((config.quality as u32 * 4).min(255)) as u8
            };
            let q_inter = q_intra.saturating_add(8);

            // Baseline: USAGE_TRANSCODING picks driver-tuned defaults,
            // then override every knob we care about so the behaviour
            // does not drift when AMD ships a new driver that tweaks
            // the USAGE preset internals.
            set_int_property(encoder, encoder_vt, "Av1Usage", AMF_USAGE_TRANSCODING)?;
            set_int_property(
                encoder,
                encoder_vt,
                "Av1RateControlMethod",
                match tp.rc_mode {
                    AmfRateControl::Cqp => AMF_RC_CQP,
                    AmfRateControl::QualityVbr => AMF_RC_QUALITY_VBR,
                },
            )?;
            set_int_property(
                encoder,
                encoder_vt,
                "Av1QualityPreset",
                amf_quality_preset_i64(tp.quality_preset),
            )?;
            set_int_property(encoder, encoder_vt, "Av1QIndexIntra", q_intra as i64)?;
            set_int_property(encoder, encoder_vt, "Av1QIndexInter", q_inter as i64)?;
            if tp.rc_mode == AmfRateControl::QualityVbr {
                set_int_property(
                    encoder,
                    encoder_vt,
                    "Av1QvbrQualityLevel",
                    tp.qvbr_quality as i64,
                )?;
            }
            set_int_property(
                encoder,
                encoder_vt,
                "Av1GOPSize",
                config.keyframe_interval as i64,
            )?;
            set_int_property(encoder, encoder_vt, "Av1AQMode", tp.aq_mode as i64)?;
            set_int_property(
                encoder,
                encoder_vt,
                "Av1TilesPerFrame",
                tp.tiles_per_frame as i64,
            )?;
            // Frame-level LOB output — mandatory for MP4 muxing so
            // every OBU carries `obu_has_size_field = 1`.
            set_int_property(encoder, encoder_vt, "Av1OutputMode", AMF_OUTPUT_MODE_FRAME)?;

            // Squad-22: bit-depth + color signalling dispatch. The bit
            // depth property tells AMF to write `BitDepth=10` into the
            // AV1 sequence header; the color-* properties write the
            // four H.273 codes into the same header. AMF infers
            // `color_description_present_flag = 1` when any of the
            // three primaries/transfer/matrix codes is non-zero, so
            // setting them is sufficient — we don't have a separate
            // present-flag knob to toggle (unlike NVENC).
            let surface_fmt = amf_surface_format_for(config.pixel_format)?;
            let color_bit_depth = amf_color_bit_depth_for(config.pixel_format);
            set_int_property(encoder, encoder_vt, "Av1ColorBitDepth", color_bit_depth)?;
            // Color signalling — wire ColorMetadata. Even SDR jobs go
            // through this block so the BT.709 codes land in the OBU
            // header explicitly (rather than via "unspecified" which
            // some ABR client libraries treat as "must guess from
            // resolution + transfer", producing inconsistent gamma).
            let cm = &config.color_metadata;
            set_int_property(
                encoder,
                encoder_vt,
                "Av1OutColorPrimaries",
                cm.colour_primaries as i64,
            )?;
            set_int_property(
                encoder,
                encoder_vt,
                "Av1OutColorTransferChar",
                transfer_to_h273(cm.transfer),
            )?;
            set_int_property(
                encoder,
                encoder_vt,
                "Av1OutColorMatrixCoeff",
                cm.matrix_coefficients as i64,
            )?;
            set_int_property(
                encoder,
                encoder_vt,
                "Av1OutColorRange",
                if cm.full_range { 1 } else { 0 },
            )?;

            tracing::info!(
                width = config.width,
                height = config.height,
                target = ?config.target,
                tier = ?config.tier,
                q_index_intra = q_intra,
                q_index_inter = q_inter,
                qvbr_quality = tp.qvbr_quality,
                rc_mode = ?tp.rc_mode,
                quality_preset = ?tp.quality_preset,
                tiles_per_frame = tp.tiles_per_frame,
                ring_size = RING_SIZE,
                "AMF AV1 tuning applied"
            );

            // 6. Init the encoder on the dispatched input format. AV1
            // VCN consumes NV12 (8-bit) or P010 (10-bit) — same
            // interleaved-chroma plane layout, different sample width.
            let rc = (encoder_vt.init)(
                encoder,
                surface_fmt,
                config.width as i32,
                config.height as i32,
            );
            if rc != AMF_OK {
                (encoder_vt.release)(encoder);
                (context_vt.terminate)(context);
                (context_vt.release)(context);
                bail!(
                    "AMFComponent::Init(AV1, {fmt}, {w}x{h}) failed: {rc} \
                     (surface format dispatched for {pf:?})",
                    fmt = surface_fmt,
                    w = config.width,
                    h = config.height,
                    pf = config.pixel_format,
                );
            }

            let session = AmfSession {
                encoder,
                context,
                factory,
                width: config.width,
                height: config.height,
                // AMF uses 100-ns ticks for PTS. We receive PTS in u64
                // "sample counts" from the decoder, and convert by
                // multiplying by (10_000_000 / frame_rate).
                pts_timescale: (10_000_000.0f64 / config.frame_rate).round() as u64,
                surface_format: surface_fmt,
            };

            tracing::info!(
                width = config.width,
                height = config.height,
                gpu = gpu_index,
                "AMF AV1 encoder ready"
            );

            Ok(Self {
                config,
                session: Some(session),
                encoded_packets: Vec::new(),
                packet_cursor: 0,
                flushed: false,
                frame_counter: 0,
                ring_idx: 0,
                _runtime_lib: runtime_lib,
            })
        }
    }

    // Surface upload is a free function (`upload_frame_static`) so it
    // doesn't need `&AmfSession` and can be called without interfering
    // with `&mut self` borrows on `AmfEncoder`.

    fn encode_one(&mut self, frame: &VideoFrame) -> Result<()> {
        // Borrow the session through encode_one. The encoder/context
        // raw pointers are read from `&self.session` once and *not*
        // snapshotted into a plain-data copy. This way, a future
        // refactor that calls `self.session.take()` inside the
        // unsafe block is a compile error rather than a silent UAF.
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("encode_one called after session drop"))?;
        let encoder_ptr = session.encoder;
        let snap = SessionSnapshot {
            encoder: session.encoder,
            context: session.context,
            width: session.width,
            height: session.height,
            pts_timescale: session.pts_timescale,
            surface_format: session.surface_format,
        };
        let force_key = self
            .frame_counter
            .is_multiple_of(self.effective_keyframe_interval());
        let packets = &mut self.encoded_packets;
        let ring_slot = self.ring_idx;

        let outcome = unsafe {
            // Wrap the whole unsafe block in catch_unwind so a panic
            // in our FFI path never unwinds across the AMF C ABI (UB).
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let raw_surface = upload_frame_static(&snap, frame)?;
                // RAII guard: surface is released on every exit path
                // unless `transfer_to_encoder` is called after a
                // successful SubmitInput. This is the safety net for
                // panics partway through property sets / retries —
                // catch_unwind itself stops FFI unwinds, but inside
                // the closure any `?` or `bail!` after alloc would
                // otherwise leak the caller-held ref (codec-review-
                // 59-60 A-A4).
                let mut guard = SurfaceGuard::new(raw_surface);

                if force_key {
                    let surface_obj = guard.as_ptr() as *mut AmfSurfaceObj;
                    let surface_vt = &*(*surface_obj).vtbl;
                    let key = AmfVariant::int64(1);
                    let name = prop("Av1ForceKeyFrame");
                    (surface_vt.set_property)(guard.as_ptr(), name.as_ptr(), key);
                }

                // Submit with bounded retry on AMF_INPUT_FULL / AMF_REPEAT.
                // Both statuses are transient per AMF SDK: the caller
                // must drain output (freeing a slot in the encoder's
                // input queue) and retry with the SAME surface pointer.
                // Releasing the surface BEFORE the successful retry
                // would UAF the second SubmitInput — that's the bug
                // this task is fixing (codec-review-59-60 AMF-5).
                submit_with_backpressure(packets, encoder_ptr, &mut guard)?;

                // Drain whatever's ready now. AMF sometimes produces a
                // packet per SubmitInput, sometimes not.
                drain_until_hungry_raw(packets, encoder_ptr)?;
                Ok::<(), anyhow::Error>(())
            }));

            match result {
                Ok(inner) => inner,
                Err(_panic) => {
                    bail!("panic in AMF encode path — aborting rather than unwinding across FFI")
                }
            }
        };

        outcome?;
        self.frame_counter += 1;
        self.ring_idx = (ring_slot + 1) % RING_SIZE;
        Ok(())
    }

    fn effective_keyframe_interval(&self) -> u32 {
        if self.config.keyframe_interval == 0 {
            240
        } else {
            self.config.keyframe_interval
        }
    }

    // drain_until_hungry is a free function (see end of file) so it
    // operates on `&mut packets` rather than `&mut self`. This keeps
    // `&self.session` alive across the call and prevents a future
    // `self.session.take()` introduction from silently turning the
    // raw encoder pointer into a UAF.

    fn flush_drain(&mut self) -> Result<()> {
        let encoder_ptr = match &self.session {
            Some(s) => s.encoder,
            None => return Ok(()),
        };
        let packets = &mut self.encoded_packets;
        // Wrap the whole FFI path in catch_unwind for the same reason
        // as encode_one — Drain + QueryOutput + buffer_to_packet all
        // allocate (Bytes::copy_from_slice) and a panic unwinding
        // across the AMF C ABI is UB in debug/test builds.
        // systems-review-59-60.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            let encoder_obj = encoder_ptr as *mut AmfComponentObj;
            let encoder_vt = &*(*encoder_obj).vtbl;
            // AMF Drain() marks the pipeline as "no more input will
            // ever arrive" — after this, QueryOutput drains the
            // internal reorder buffer until AMF_EOF.
            let rc = (encoder_vt.drain)(encoder_ptr);
            if rc != AMF_OK && rc != AMF_REPEAT {
                bail!("AMF Drain failed: {rc}");
            }
            drain_until_hungry_raw(packets, encoder_ptr)?;
            Ok::<(), anyhow::Error>(())
        }));
        match result {
            Ok(inner) => inner,
            Err(_panic) => {
                bail!("panic in AMF flush path — aborting rather than unwinding across FFI")
            }
        }
    }

    /// Suppress unused warning — `c_int` type is here for future
    /// NV_ENC-style rc tables where we need to pass a C `int` through.
    #[allow(dead_code)]
    fn _suppress_unused_c_int() -> c_int {
        0
    }
}

impl Encoder for AmfEncoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        if frame.format != self.config.pixel_format {
            bail!(
                "AMF session was initialized with {:?} input but frame is {:?}",
                self.config.pixel_format,
                frame.format
            );
        }
        self.encode_one(frame)
    }

    fn flush(&mut self) -> Result<()> {
        if !self.flushed {
            self.flush_drain()?;
            self.flushed = true;
        }
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        if self.packet_cursor < self.encoded_packets.len() {
            let pkt = self.encoded_packets[self.packet_cursor].clone();
            self.packet_cursor += 1;
            Ok(Some(pkt))
        } else {
            Ok(None)
        }
    }
}

/// Submit `guard.as_ptr()` to the encoder, retrying on transient
/// back-pressure statuses. On success the guard is marked as
/// transferred and its `Drop` becomes a no-op (the encoder's internal
/// ref now owns the surface lifetime). On hard failure the guard's
/// `Drop` releases our caller-held ref exactly once.
///
/// The #59 follow-up bug: previously the caller released the surface
/// BEFORE the retry on `AMF_INPUT_FULL`. That made the retry a
/// use-after-free because AMF rejected the frame (no ownership taken)
/// and we had just dropped our only ref. The fix is to keep the
/// caller-held ref alive across the retry loop — exactly what the
/// `SurfaceGuard` + `transfer_to_encoder` pattern encodes.
///
/// Retry policy: bounded at `INPUT_FULL_MAX_RETRIES` attempts with
/// exponential backoff starting at `INPUT_FULL_BACKOFF_MS_INITIAL` ms
/// and capped at `INPUT_FULL_BACKOFF_MS_MAX` ms. A drain pass between
/// each retry attempts to free an input slot. This is not unbounded
/// so a stuck driver can't spin us forever.
unsafe fn submit_with_backpressure(
    packets: &mut Vec<EncodedPacket>,
    encoder: *mut c_void,
    guard: &mut SurfaceGuard,
) -> Result<()> {
    unsafe {
        let encoder_obj = encoder as *mut AmfComponentObj;
        let encoder_vt = &*(*encoder_obj).vtbl;

        let mut backoff_ms = INPUT_FULL_BACKOFF_MS_INITIAL;
        for attempt in 0..=INPUT_FULL_MAX_RETRIES {
            let rc = (encoder_vt.submit_input)(encoder, guard.as_ptr());
            match rc {
                AMF_OK | AMF_NEED_MORE_INPUT => {
                    // Per AMF SDK "Reference Counting" appendix:
                    // SubmitInput takes a fresh internal ref on
                    // AMF_OK / AMF_NEED_MORE_INPUT. Our caller-held
                    // ref is now redundant — release it exactly once
                    // and mark the guard so Drop is a no-op at
                    // scope exit.
                    let surface_obj = guard.as_ptr() as *mut AmfSurfaceObj;
                    let surface_vt = &*(*surface_obj).vtbl;
                    (surface_vt.release)(guard.as_ptr());
                    guard.transfer_to_encoder();
                    return Ok(());
                }
                AMF_INPUT_FULL | AMF_REPEAT => {
                    // Transient — drain output to free an input slot,
                    // then retry. Critically: the surface is NOT
                    // released here; the guard still owns the caller-
                    // held ref and the same pointer is handed back
                    // to the retry.
                    if attempt == INPUT_FULL_MAX_RETRIES {
                        tracing::warn!(
                            status = rc,
                            attempts = attempt + 1,
                            "AMF SubmitInput backpressure exceeded retry budget — \
                             surface still caller-owned, releasing via guard"
                        );
                        bail!(
                            "AMF SubmitInput stuck at {rc} after {} attempts",
                            attempt + 1
                        );
                    }
                    // Drain first; in steady state one drain frees
                    // exactly one input slot.
                    drain_until_hungry_raw(packets, encoder)?;
                    // If drain returned without any output (encoder
                    // still warming up or mid-reorder), spin the
                    // current OS thread for `backoff_ms` so we don't
                    // busy-loop the driver. Yields on Windows and
                    // Linux — not a blocking syscall.
                    if attempt > 0 {
                        std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
                        backoff_ms = (backoff_ms * 2).min(INPUT_FULL_BACKOFF_MS_MAX);
                    }
                    continue;
                }
                other => {
                    // Hard error: surface still caller-owned. Guard's
                    // Drop will release our ref on return from bail.
                    tracing::warn!(
                        status = other,
                        "AMF SubmitInput hard failure — surface still caller-owned, \
                         releasing via guard"
                    );
                    bail!("AMF SubmitInput failed: {other}");
                }
            }
        }
        // Unreachable — loop exit always via return/bail above.
        unreachable!("submit_with_backpressure loop invariant violated")
    }
}

/// Drain `QueryOutput` into `packets` until the encoder returns
/// `AMF_REPEAT` (no more data available yet), `AMF_EOF`, or
/// `AMF_NEED_MORE_INPUT`. Free function (not a method on AmfEncoder)
/// so it takes `&mut Vec<EncodedPacket>` rather than `&mut self`.
/// This keeps `&self.session` alive through the call and makes a
/// future `self.session.take()` inside the unsafe block a compile
/// error rather than a silent UAF. systems-review-59-60.
unsafe fn drain_until_hungry_raw(
    packets: &mut Vec<EncodedPacket>,
    encoder: *mut c_void,
) -> Result<()> {
    unsafe {
        loop {
            let encoder_obj = encoder as *mut AmfComponentObj;
            let encoder_vt = &*(*encoder_obj).vtbl;
            let mut data: *mut c_void = ptr::null_mut();
            let rc = (encoder_vt.query_output)(encoder, &mut data);
            match rc {
                AMF_OK => {
                    if data.is_null() {
                        continue;
                    }
                    if let Some(pkt) = buffer_to_packet(data)? {
                        packets.push(pkt);
                    }
                    // buffer_to_packet released any QueryInterface ref
                    // it took; drop the AMFData ref here.
                    let obj = data as *mut AmfBufferObj;
                    ((*(*obj).vtbl).release)(data);
                }
                // AMF_REPEAT on QueryOutput means "no more data this
                // round but more may appear later" — normal hungry
                // return for the drain loop.
                AMF_REPEAT => return Ok(()),
                // AMF_EOF is the expected terminator after `Drain()`
                // has been called — signals the encoder has flushed
                // its reorder buffer and no further output will come.
                // Treated as a clean empty return.
                AMF_EOF => return Ok(()),
                // AMF_NEED_MORE_INPUT on QueryOutput means the encoder
                // requires more frames before it can emit anything
                // (typical for initial lookahead warmup / reorder).
                // Equivalent to "no packet yet"; clean empty return.
                AMF_NEED_MORE_INPUT => return Ok(()),
                other => bail!("AMF QueryOutput failed: {other}"),
            }
        }
    }
}

/// Cross-cast an AMFData* to AMFBuffer* via QueryInterface and copy
/// its native bytes into an EncodedPacket. Free function for the same
/// reason as `drain_until_hungry_raw` — no `&self` aliasing concerns.
///
/// SAFETY precondition (codec-review-59-60 M-A1): we rely on AMFData
/// and AMFBuffer sharing the first three vtable slots (QueryInterface,
/// Acquire, Release — COM IUnknown). This is guaranteed by the AMF
/// SDK's AMFInterface inheritance chain. If QueryInterface fails we
/// bail rather than fall through to `treat AMFData as AMFBuffer` — a
/// future SDK rev that reorders AMFData vtable entries past slot 3
/// would otherwise call `GetSize` at the wrong offset and read garbage.
unsafe fn buffer_to_packet(data: *mut c_void) -> Result<Option<EncodedPacket>> {
    unsafe {
        let data_obj = data as *mut AmfBufferObj;
        let data_vt = &*(*data_obj).vtbl;

        let mut buffer: *mut c_void = ptr::null_mut();
        let qi_rc =
            (data_vt.query_interface)(data, AMF_IID_BUFFER.as_ptr() as *const c_void, &mut buffer);
        if qi_rc != 0 || buffer.is_null() {
            // Fail loudly rather than splatting bytes through a
            // possibly-shifted vtable layout.
            bail!("AMFData::QueryInterface(AMFBuffer) failed: {qi_rc}");
        }
        let buffer_obj = buffer as *mut AmfBufferObj;
        let buffer_vt = &*(*buffer_obj).vtbl;

        let size = (buffer_vt.get_size)(buffer_obj as *mut c_void);
        let native = (buffer_vt.get_native)(buffer_obj as *mut c_void) as *const u8;
        if size == 0 || native.is_null() {
            (buffer_vt.release)(buffer_obj as *mut c_void);
            return Ok(None);
        }

        let slice = std::slice::from_raw_parts(native, size);
        let data_bytes = Bytes::copy_from_slice(slice);

        let pts_ticks = (buffer_vt.get_pts)(buffer_obj as *mut c_void) as u64;

        // Read the frame-type property so we can tag keyframes in
        // the EncodedPacket. Bailing on the Get is fine — we just
        // fall back to "not a keyframe".
        let prop_name = prop("Av1OutputFrameType");
        let mut var: AmfVariant = AmfVariant {
            ty: 0,
            _pad: 0,
            value: [0; 24],
        };
        let is_keyframe =
            if (buffer_vt.get_property)(buffer_obj as *mut c_void, prop_name.as_ptr(), &mut var)
                == AMF_OK
                && var.ty == AMF_VARIANT_INT64
            {
                let mut v_bytes = [0u8; 8];
                v_bytes.copy_from_slice(&var.value[..8]);
                let v = i64::from_le_bytes(v_bytes);
                v == AMF_OUTPUT_FRAME_TYPE_KEY || v == AMF_OUTPUT_FRAME_TYPE_INTRA_ONLY
            } else {
                false
            };

        (buffer_vt.release)(buffer_obj as *mut c_void);

        Ok(Some(EncodedPacket {
            data: data_bytes,
            pts: pts_ticks,
            is_keyframe,
        }))
    }
}

/// Map `AmfQualityPreset` variants to the i64 values the AMF SetProperty
/// ABI expects. The enum's `#[repr(i64)]` makes this effectively a
/// discriminant read, but going through a match keeps the translation
/// explicit and audit-able against the AMD AMF header constants.
fn amf_quality_preset_i64(preset: AmfQualityPreset) -> i64 {
    match preset {
        AmfQualityPreset::HighQuality => 10,
        AmfQualityPreset::Quality => 30,
        AmfQualityPreset::Balanced => 50,
        AmfQualityPreset::Speed => 70,
    }
}

/// Set a single i64-valued property on an AMF component, wide-string
/// encoded. Returns the AMF_RESULT as a Rust `Result` so the call
/// site can bail cleanly when the driver rejects a knob value.
unsafe fn set_int_property(
    obj: *mut c_void,
    vt: &AmfComponentVtbl,
    name: &str,
    value: i64,
) -> Result<()> {
    unsafe {
        let wname = wide(name);
        let rc = (vt.set_property)(obj, wname.as_ptr(), AmfVariant::int64(value));
        if rc != AMF_OK {
            bail!("AMF SetProperty({}, {}) failed: {rc}", name, value);
        }
        Ok(())
    }
}

/// Plain-data snapshot of the fields `upload_frame_static` needs. Used
/// so we can hold session pointers across a self-mutating call without
/// fighting the borrow checker.
#[derive(Clone, Copy)]
struct SessionSnapshot {
    encoder: *mut c_void,
    context: *mut c_void,
    width: u32,
    height: u32,
    pts_timescale: u64,
    /// `AMF_SURFACE_NV12` or `AMF_SURFACE_P010`.
    surface_format: i32,
}

/// Copy a YUV420p (8-bit) or Yuv420p10le (10-bit) frame into a freshly-
/// allocated AMF surface. The surface format must already have been
/// captured into `snap.surface_format` at session-create time —
/// dispatching here per-frame would silently mismatch the encoder
/// component's Init format.
///
/// Returns an AMF-owned surface pointer; caller must Release when
/// done (SubmitInput keeps its own internal ref, so one Release
/// balances one AllocSurface regardless of SubmitInput outcome).
///
/// The `encoder` field in the snapshot is unused here but kept so
/// future extensions (e.g. encoder-owned surface recycling via the
/// AMFComponent::SubmitInput variant that accepts a hint pool) have
/// it handy.
unsafe fn upload_frame_static(snap: &SessionSnapshot, frame: &VideoFrame) -> Result<*mut c_void> {
    let _ = snap.encoder; // reserved for future recycling path
    unsafe {
        let context_obj = snap.context as *mut AmfContextObj;
        let context_vt = &*(*context_obj).vtbl;

        let mut surface: *mut c_void = ptr::null_mut();
        let rc = (context_vt.alloc_surface)(
            snap.context,
            AMF_MEMORY_HOST,
            snap.surface_format,
            snap.width as i32,
            snap.height as i32,
            &mut surface,
        );
        if rc != AMF_OK || surface.is_null() {
            bail!(
                "AMFContext::AllocSurface({}x{} fmt={}) failed: {rc}",
                snap.width,
                snap.height,
                snap.surface_format,
            );
        }

        let surface_obj = surface as *mut AmfSurfaceObj;
        let surface_vt = &*(*surface_obj).vtbl;

        let y_plane = (surface_vt.get_plane)(surface, AMF_PLANE_Y);
        let uv_plane = (surface_vt.get_plane)(surface, AMF_PLANE_UV);
        if y_plane.is_null() || uv_plane.is_null() {
            (surface_vt.release)(surface);
            bail!(
                "AMF surface (fmt={}) missing Y or UV plane",
                snap.surface_format
            );
        }

        // Per-format upload. Both branches share the plane geometry +
        // PTS write; they only differ in per-sample byte width and the
        // P010 `<<6` shift.
        let upload_result = match snap.surface_format {
            AMF_SURFACE_NV12 => copy_yuv420p_to_nv12_surface(
                surface,
                surface_vt,
                y_plane,
                uv_plane,
                snap.width,
                snap.height,
                frame,
            ),
            AMF_SURFACE_P010 => copy_yuv420p10le_to_p010_surface(
                surface,
                surface_vt,
                y_plane,
                uv_plane,
                snap.width,
                snap.height,
                frame,
            ),
            other => {
                (surface_vt.release)(surface);
                bail!("AMF surface format {other} not supported by uploader");
            }
        };
        upload_result?;

        (surface_vt.set_pts)(surface, (frame.pts * snap.pts_timescale) as i64);

        Ok(surface)
    }
}

/// 8-bit YUV420p → AMF NV12 surface. Y plane: byte copy at
/// surface pitch. UV plane: interleave U + V from separate planes
/// into the single NV12 chroma plane.
unsafe fn copy_yuv420p_to_nv12_surface(
    surface: *mut c_void,
    surface_vt: &AmfSurfaceVtbl,
    y_plane: *mut c_void,
    uv_plane: *mut c_void,
    width: u32,
    height: u32,
    frame: &VideoFrame,
) -> Result<()> {
    unsafe {
        let w = width as usize;
        let h = height as usize;
        let y_size = w * h;
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let uv_size = cw * ch;

        if frame.data.len() < y_size + 2 * uv_size {
            (surface_vt.release)(surface);
            bail!(
                "frame data too small for {}x{} YUV420p: need {} bytes, got {}",
                w,
                h,
                y_size + 2 * uv_size,
                frame.data.len()
            );
        }

        let y_plane_obj = y_plane as *mut AmfPlaneObj;
        let y_vt = &*(*y_plane_obj).vtbl;
        let y_dst = (y_vt.get_native)(y_plane) as *mut u8;
        let y_pitch = (y_vt.get_h_pitch)(y_plane) as usize;
        if y_dst.is_null() {
            (surface_vt.release)(surface);
            bail!("AMF Y plane native pointer is null — surface not host-mapped?");
        }
        for row in 0..h {
            let src = frame.data.as_ptr().add(row * w);
            let dst = y_dst.add(row * y_pitch);
            ptr::copy_nonoverlapping(src, dst, w);
        }

        let uv_plane_obj = uv_plane as *mut AmfPlaneObj;
        let uv_vt = &*(*uv_plane_obj).vtbl;
        let uv_dst = (uv_vt.get_native)(uv_plane) as *mut u8;
        let uv_pitch = (uv_vt.get_h_pitch)(uv_plane) as usize;
        if uv_dst.is_null() {
            (surface_vt.release)(surface);
            bail!("AMF UV plane native pointer is null — surface not host-mapped?");
        }
        let u_src_base = frame.data.as_ptr().add(y_size);
        let v_src_base = u_src_base.add(uv_size);
        for row in 0..ch {
            let u_src = u_src_base.add(row * cw);
            let v_src = v_src_base.add(row * cw);
            let dst_row = uv_dst.add(row * uv_pitch);
            for col in 0..cw {
                *dst_row.add(col * 2) = *u_src.add(col);
                *dst_row.add(col * 2 + 1) = *v_src.add(col);
            }
        }
        Ok(())
    }
}

/// 10-bit `Yuv420p10le` → AMF P010 surface. Same plane geometry as
/// NV12 but each sample is 2 bytes; P010 stores the valid 10-bit
/// value in the **upper 10 bits** of the 16-bit word, so we shift
/// each source sample left by 6 on the way in. Source format keeps
/// the value in the **lower 10 bits** (matches NVDEC `>>6`-normalized
/// surface output from Squad-6).
unsafe fn copy_yuv420p10le_to_p010_surface(
    surface: *mut c_void,
    surface_vt: &AmfSurfaceVtbl,
    y_plane: *mut c_void,
    uv_plane: *mut c_void,
    width: u32,
    height: u32,
    frame: &VideoFrame,
) -> Result<()> {
    unsafe {
        let w = width as usize;
        let h = height as usize;
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        // 2 bytes per sample.
        let y_bytes = w * h * 2;
        let uv_bytes = cw * ch * 2;

        if frame.data.len() < y_bytes + 2 * uv_bytes {
            (surface_vt.release)(surface);
            bail!(
                "frame data too small for {}x{} Yuv420p10le: need {} bytes, got {}",
                w,
                h,
                y_bytes + 2 * uv_bytes,
                frame.data.len()
            );
        }

        let y_plane_obj = y_plane as *mut AmfPlaneObj;
        let y_vt = &*(*y_plane_obj).vtbl;
        let y_dst = (y_vt.get_native)(y_plane) as *mut u8;
        let y_pitch_bytes = (y_vt.get_h_pitch)(y_plane) as usize;
        if y_dst.is_null() {
            (surface_vt.release)(surface);
            bail!("AMF P010 Y plane native pointer is null");
        }

        let src_ptr = frame.data.as_ptr();

        // Y plane: w samples per row.
        for row in 0..h {
            let src_row = src_ptr.add(row * w * 2) as *const u16;
            let dst_row = y_dst.add(row * y_pitch_bytes) as *mut u16;
            for col in 0..w {
                let sample = (*src_row.add(col)) & 0x03FF;
                *dst_row.add(col) = sample << 6;
            }
        }

        let uv_plane_obj = uv_plane as *mut AmfPlaneObj;
        let uv_vt = &*(*uv_plane_obj).vtbl;
        let uv_dst = (uv_vt.get_native)(uv_plane) as *mut u8;
        let uv_pitch_bytes = (uv_vt.get_h_pitch)(uv_plane) as usize;
        if uv_dst.is_null() {
            (surface_vt.release)(surface);
            bail!("AMF P010 UV plane native pointer is null");
        }
        let u_src_base = src_ptr.add(y_bytes);
        let v_src_base = u_src_base.add(uv_bytes);
        // UV plane: cw samples (cw*2 bytes) per row interleaved as
        // U,V,U,V… (pitch is in bytes).
        for row in 0..ch {
            let u_src = u_src_base.add(row * cw * 2) as *const u16;
            let v_src = v_src_base.add(row * cw * 2) as *const u16;
            let dst_row = uv_dst.add(row * uv_pitch_bytes) as *mut u16;
            for col in 0..cw {
                let u = (*u_src.add(col)) & 0x03FF;
                let v = (*v_src.add(col)) & 0x03FF;
                *dst_row.add(col * 2) = u << 6;
                *dst_row.add(col * 2 + 1) = v << 6;
            }
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────
//
// GPU E2E is impossible on a non-AMD host (our dev box is RTX 3090).
// These tests exercise the FFI-agnostic invariants: the AMF retry
// driver, the drain helper's status mapping, the ring index cycling,
// and the variant layout. Each test builds a mock component vtable
// that returns a canned status sequence, drives it through the same
// functions the real path uses, and asserts the observable behaviour.
#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

    // ── Mock AMF component ────────────────────────────────────────
    //
    // Minimal fake AMF component built to match the vtable layout our
    // production code calls through. Each test configures a canned
    // sequence of AMF_RESULT values for SubmitInput / QueryOutput;
    // the mock returns them in order and tracks Acquire/Release
    // counts so we can assert no UAF or leak occurred.
    //
    // All fields are thread_local so the mock state is accessible from
    // the `extern "C"` vtable functions (which cannot close over
    // captures).

    thread_local! {
        static MOCK_SUBMIT_RESULTS: RefCell<Vec<AmfResult>> = const { RefCell::new(Vec::new()) };
        static MOCK_QUERY_RESULTS: RefCell<Vec<AmfResult>> = const { RefCell::new(Vec::new()) };
        static MOCK_SUBMIT_CALLS: AtomicUsize = const { AtomicUsize::new(0) };
        static MOCK_QUERY_CALLS: AtomicUsize = const { AtomicUsize::new(0) };
        static MOCK_SURFACE_REFCOUNT: AtomicI64 = const { AtomicI64::new(0) };
        /// Records the surface pointer passed to each SubmitInput call
        /// so we can assert the driver retries with the SAME pointer
        /// (no UAF, no substitution).
        static MOCK_SUBMIT_POINTERS: RefCell<Vec<*mut c_void>> = const { RefCell::new(Vec::new()) };
    }

    fn mock_reset() {
        MOCK_SUBMIT_RESULTS.with(|v| v.borrow_mut().clear());
        MOCK_QUERY_RESULTS.with(|v| v.borrow_mut().clear());
        MOCK_SUBMIT_POINTERS.with(|v| v.borrow_mut().clear());
        MOCK_SUBMIT_CALLS.with(|c| c.store(0, Ordering::SeqCst));
        MOCK_QUERY_CALLS.with(|c| c.store(0, Ordering::SeqCst));
        MOCK_SURFACE_REFCOUNT.with(|c| c.store(1, Ordering::SeqCst));
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

    // ── Mock component vtable funcs ───────────────────────────────

    unsafe extern "C" fn mock_qi(_: *mut c_void, _: *const c_void, _: *mut *mut c_void) -> i64 {
        0
    }
    unsafe extern "C" fn mock_acquire(_: *mut c_void) -> i64 {
        1
    }
    unsafe extern "C" fn mock_release_component(_: *mut c_void) -> i64 {
        1
    }
    unsafe extern "C" fn mock_set_property(
        _: *mut c_void,
        _: *const u16,
        _: AmfVariant,
    ) -> AmfResult {
        AMF_OK
    }
    unsafe extern "C" fn mock_get_property(
        _: *mut c_void,
        _: *const u16,
        _: *mut AmfVariant,
    ) -> AmfResult {
        AMF_OK
    }
    unsafe extern "C" fn mock_init(_: *mut c_void, _: i32, _: i32, _: i32) -> AmfResult {
        AMF_OK
    }
    unsafe extern "C" fn mock_reinit(_: *mut c_void, _: i32, _: i32) -> AmfResult {
        AMF_OK
    }
    unsafe extern "C" fn mock_terminate(_: *mut c_void) -> AmfResult {
        AMF_OK
    }
    unsafe extern "C" fn mock_drain(_: *mut c_void) -> AmfResult {
        AMF_OK
    }
    unsafe extern "C" fn mock_flush(_: *mut c_void) -> AmfResult {
        AMF_OK
    }

    unsafe extern "C" fn mock_submit_input(_: *mut c_void, surface: *mut c_void) -> AmfResult {
        MOCK_SUBMIT_POINTERS.with(|v| v.borrow_mut().push(surface));
        let idx = MOCK_SUBMIT_CALLS.with(|c| c.fetch_add(1, Ordering::SeqCst));
        MOCK_SUBMIT_RESULTS.with(|v| {
            let v = v.borrow();
            v.get(idx).copied().unwrap_or(AMF_OK)
        })
    }

    unsafe extern "C" fn mock_query_output(_: *mut c_void, data: *mut *mut c_void) -> AmfResult {
        let idx = MOCK_QUERY_CALLS.with(|c| c.fetch_add(1, Ordering::SeqCst));
        let rc = MOCK_QUERY_RESULTS.with(|v| {
            let v = v.borrow();
            v.get(idx).copied().unwrap_or(AMF_REPEAT)
        });
        if rc == AMF_OK {
            // Return null data — drain helper treats that as "no
            // packet produced this round" and continues looping.
            unsafe {
                *data = ptr::null_mut();
            }
        }
        rc
    }

    unsafe extern "C" fn mock_get_context(_: *mut c_void) -> *mut c_void {
        ptr::null_mut()
    }
    unsafe extern "C" fn mock_set_output_cb(_: *mut c_void, _: *mut c_void) -> AmfResult {
        AMF_OK
    }
    unsafe extern "C" fn mock_get_caps(_: *mut c_void, _: *mut *mut c_void) -> AmfResult {
        AMF_OK
    }
    unsafe extern "C" fn mock_optimize(_: *mut c_void, _: *mut c_void) -> AmfResult {
        AMF_OK
    }

    // ── Mock surface vtable funcs ─────────────────────────────────
    //
    // The driver only calls Release on the surface (directly via the
    // guard) and never touches any other surface slot in these tests.
    // The full vtable is populated so the Rust struct layout matches
    // what production code expects to walk.

    unsafe extern "C" fn mock_surface_release(_: *mut c_void) -> i64 {
        let prev = MOCK_SURFACE_REFCOUNT.with(|c| c.fetch_sub(1, Ordering::SeqCst));
        assert!(
            prev > 0,
            "surface Release when refcount already zero (UAF indicator)"
        );
        prev - 1
    }

    unsafe extern "C" fn mock_surface_set_property(
        _: *mut c_void,
        _: *const u16,
        _: AmfVariant,
    ) -> AmfResult {
        AMF_OK
    }
    unsafe extern "C" fn mock_surface_get_property(
        _: *mut c_void,
        _: *const u16,
        _: *mut AmfVariant,
    ) -> AmfResult {
        AMF_OK
    }
    unsafe extern "C" fn mock_surface_duplicate(
        _: *mut c_void,
        _: i32,
        _: *mut *mut c_void,
    ) -> AmfResult {
        AMF_OK
    }
    unsafe extern "C" fn mock_surface_get_pts(_: *mut c_void) -> i64 {
        0
    }
    unsafe extern "C" fn mock_surface_set_pts(_: *mut c_void, _: i64) {}
    unsafe extern "C" fn mock_surface_get_duration(_: *mut c_void) -> i64 {
        0
    }
    unsafe extern "C" fn mock_surface_set_duration(_: *mut c_void, _: i64) {}
    unsafe extern "C" fn mock_surface_get_planes_count(_: *mut c_void) -> usize {
        2
    }
    unsafe extern "C" fn mock_surface_get_plane_at(_: *mut c_void, _: usize) -> *mut c_void {
        ptr::null_mut()
    }
    unsafe extern "C" fn mock_surface_get_plane(_: *mut c_void, _: i32) -> *mut c_void {
        ptr::null_mut()
    }

    static MOCK_SURFACE_VTBL: AmfSurfaceVtbl = AmfSurfaceVtbl {
        query_interface: mock_qi,
        acquire: mock_acquire,
        release: mock_surface_release,
        set_property: mock_surface_set_property,
        get_property: mock_surface_get_property,
        duplicate: mock_surface_duplicate,
        get_pts: mock_surface_get_pts,
        set_pts: mock_surface_set_pts,
        get_duration: mock_surface_get_duration,
        set_duration: mock_surface_set_duration,
        get_planes_count: mock_surface_get_planes_count,
        get_plane_at: mock_surface_get_plane_at,
        get_plane: mock_surface_get_plane,
    };

    static MOCK_COMPONENT_VTBL: AmfComponentVtbl = AmfComponentVtbl {
        query_interface: mock_qi,
        acquire: mock_acquire,
        release: mock_release_component,
        set_property: mock_set_property,
        get_property: mock_get_property,
        init: mock_init,
        reinit: mock_reinit,
        terminate: mock_terminate,
        drain: mock_drain,
        flush: mock_flush,
        submit_input: mock_submit_input,
        query_output: mock_query_output,
        get_context: mock_get_context,
        set_output_data_allocator_cb: mock_set_output_cb,
        get_caps: mock_get_caps,
        optimize: mock_optimize,
    };

    /// Build a fake surface + component on the stack that resolve to
    /// the mock vtables. Returns pointers the driver can hand through
    /// its FFI signatures. Caller owns the backing storage via the
    /// returned tuple — pointers are only valid for the lifetime of
    /// the stack frame that owns them.
    fn make_mock_pair() -> (Box<AmfSurfaceObj>, Box<AmfComponentObj>) {
        let surface = Box::new(AmfSurfaceObj {
            vtbl: &MOCK_SURFACE_VTBL,
        });
        let component = Box::new(AmfComponentObj {
            vtbl: &MOCK_COMPONENT_VTBL,
        });
        (surface, component)
    }

    // ── Tests ─────────────────────────────────────────────────────

    /// The core #59 regression test: an AMF_INPUT_FULL return from
    /// SubmitInput must NOT release the surface before the retry.
    /// If the driver releases prematurely, the retry submit would
    /// be dereferencing a zero-refcount surface — a UAF. This test
    /// runs through the real `submit_with_backpressure` function
    /// against a mock that returns `INPUT_FULL, OK` and asserts:
    ///   1. SubmitInput was called twice with the SAME surface ptr.
    ///   2. The surface refcount stayed at ≥1 across the retry.
    ///   3. Final refcount is 0 after the success path releases.
    #[test]
    fn test_amf_input_full_does_not_release_surface_before_retry() {
        mock_reset();
        set_submit_sequence(&[AMF_INPUT_FULL, AMF_OK]);
        // Drain between retries returns REPEAT immediately (no output
        // available) — the driver's backoff then wakes up and retries.
        set_query_sequence(&[AMF_REPEAT]);

        let (mut surface, mut component) = make_mock_pair();
        let surface_ptr: *mut c_void = surface.as_mut() as *mut _ as *mut c_void;
        let component_ptr: *mut c_void = component.as_mut() as *mut _ as *mut c_void;

        let mut guard = SurfaceGuard::new(surface_ptr);
        let mut packets = Vec::new();

        let result = unsafe { submit_with_backpressure(&mut packets, component_ptr, &mut guard) };
        assert!(
            result.is_ok(),
            "submit_with_backpressure failed: {result:?}"
        );

        assert_eq!(
            submit_call_count(),
            2,
            "SubmitInput must retry exactly once on INPUT_FULL before success"
        );
        assert_eq!(
            submit_pointer_at(0),
            Some(surface_ptr),
            "first submit must pass the original surface pointer"
        );
        assert_eq!(
            submit_pointer_at(1),
            Some(surface_ptr),
            "retry submit must pass the SAME surface pointer — anything else would be a UAF tell"
        );
        // After the success path, the success-arm's explicit release
        // has dropped our caller-held ref from 1 → 0. No double-free.
        assert_eq!(
            surface_refcount(),
            0,
            "surface refcount must reach exactly 0 after success (no leak, no double-release)"
        );
        // Guard's owned flag must be cleared (transfer_to_encoder was
        // called) so Drop is a no-op at end of scope.
        // Sanity-check by letting the guard drop and verifying the
        // refcount doesn't go negative (the mock panics if it does).
        drop(guard);
        assert_eq!(surface_refcount(), 0, "Drop after transfer must be a no-op");
    }

    /// AMF_NEED_MORE_INPUT on QueryOutput is the driver's signal that
    /// the encoder needs more frames before it can emit anything
    /// (typical for lookahead warm-up). The drain helper must treat
    /// this as a clean "no packet available" return, NOT an error.
    #[test]
    fn test_amf_need_more_input_returns_no_packet() {
        mock_reset();
        set_query_sequence(&[AMF_NEED_MORE_INPUT]);

        let (_, mut component) = make_mock_pair();
        let component_ptr: *mut c_void = component.as_mut() as *mut _ as *mut c_void;
        let mut packets = Vec::new();

        let result = unsafe { drain_until_hungry_raw(&mut packets, component_ptr) };
        assert!(
            result.is_ok(),
            "AMF_NEED_MORE_INPUT on drain must be Ok (no packet yet), got {result:?}"
        );
        assert_eq!(packets.len(), 0, "no packets should be emitted");
        assert_eq!(
            query_call_count(),
            1,
            "drain should have returned after the single NEED_MORE_INPUT"
        );
    }

    /// AMF_EOF on QueryOutput signals end-of-stream after Drain() was
    /// called. The drain helper must return cleanly (not bail) so
    /// flush_drain can complete. No packets should be appended for
    /// the EOF return itself.
    #[test]
    fn test_amf_eof_ends_drain_cleanly() {
        mock_reset();
        set_query_sequence(&[AMF_EOF]);

        let (_, mut component) = make_mock_pair();
        let component_ptr: *mut c_void = component.as_mut() as *mut _ as *mut c_void;
        let mut packets = Vec::new();

        let result = unsafe { drain_until_hungry_raw(&mut packets, component_ptr) };
        assert!(
            result.is_ok(),
            "AMF_EOF on drain must end the flush loop cleanly, got {result:?}"
        );
        assert_eq!(packets.len(), 0, "no packets at EOF");
        assert_eq!(
            query_call_count(),
            1,
            "drain should return on the first EOF"
        );
    }

    /// The ring index must cycle 0, 1, 2, 3, 0, 1, 2, 3 ... under the
    /// `(ring_idx + 1) % RING_SIZE` advancement rule that `encode_one`
    /// uses on every successful SubmitInput. Mirrors NVENC's parallel
    /// test so both backends are validated identically.
    #[test]
    fn test_amf_ring_buffer_index_cycles() {
        let mut idx = 0usize;
        let mut seen = Vec::new();
        for _ in 0..(RING_SIZE * 3) {
            seen.push(idx);
            idx = (idx + 1) % RING_SIZE;
        }
        assert_eq!(
            seen,
            vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            "ring index must cycle through 0..RING_SIZE"
        );
    }

    /// Ring size is the NVENC-parity constant.
    #[test]
    fn test_amf_ring_size_is_four() {
        assert_eq!(
            RING_SIZE, 4,
            "RING_SIZE must match Squad-5's NVENC default of 4"
        );
    }

    /// AMF_REPEAT on SubmitInput is documented as a transient "retry
    /// same surface" status, identical semantics to AMF_INPUT_FULL.
    /// The driver must handle it the same way.
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

        let result = unsafe { submit_with_backpressure(&mut packets, component_ptr, &mut guard) };
        assert!(result.is_ok(), "AMF_REPEAT retry must succeed");
        assert_eq!(submit_call_count(), 2);
        assert_eq!(submit_pointer_at(1), Some(surface_ptr));
        assert_eq!(surface_refcount(), 0);
        drop(guard);
    }

    /// A hard error from SubmitInput (anything other than OK,
    /// NEED_MORE_INPUT, INPUT_FULL, REPEAT) must surface as Err and
    /// the guard's Drop must release the caller-held ref exactly once
    /// — not zero times (leak) and not twice (double-free).
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
            let result =
                unsafe { submit_with_backpressure(&mut packets, component_ptr, &mut guard) };
            assert!(result.is_err(), "hard error must propagate as Err");
            // Guard goes out of scope here → Drop releases our ref.
        }
        assert_eq!(
            surface_refcount(),
            0,
            "hard-error path must release exactly once via the guard's Drop"
        );
    }

    /// Bounded retry budget: if both SubmitInput AND QueryOutput stay
    /// saturated indefinitely, the driver must eventually bail rather
    /// than spin forever. This simulates a stuck GPU queue.
    #[test]
    fn test_amf_submit_bounded_retry_budget() {
        mock_reset();
        // Fill submit with INPUT_FULL responses exceeding the retry
        // budget. Every QueryOutput returns REPEAT (no output), so
        // backoff + retry proceeds without clearing space.
        let saturated: Vec<AmfResult> = (0..(INPUT_FULL_MAX_RETRIES as usize + 2))
            .map(|_| AMF_INPUT_FULL)
            .collect();
        set_submit_sequence(&saturated);
        let drains: Vec<AmfResult> = (0..(INPUT_FULL_MAX_RETRIES as usize + 2))
            .map(|_| AMF_REPEAT)
            .collect();
        set_query_sequence(&drains);

        let (mut surface, mut component) = make_mock_pair();
        let surface_ptr: *mut c_void = surface.as_mut() as *mut _ as *mut c_void;
        let component_ptr: *mut c_void = component.as_mut() as *mut _ as *mut c_void;

        let mut packets = Vec::new();
        {
            let mut guard = SurfaceGuard::new(surface_ptr);
            let result =
                unsafe { submit_with_backpressure(&mut packets, component_ptr, &mut guard) };
            assert!(
                result.is_err(),
                "stuck backpressure must eventually bail (not spin)"
            );
            // Ring-buffer state does NOT advance here (caller
            // responsibility); this test just checks the retry ceiling.
            assert_eq!(
                submit_call_count() as u32,
                INPUT_FULL_MAX_RETRIES + 1,
                "retry count must match INPUT_FULL_MAX_RETRIES + 1 (initial + retries)"
            );
        }
        // Guard drop releases the single caller ref once.
        assert_eq!(
            surface_refcount(),
            0,
            "bounded-retry failure must still release cleanly via guard"
        );
    }

    /// Variant layout ABI guard: the `int64` arm must live at offset
    /// 8 of the struct so the C ABI's tagged-union write lands in the
    /// right byte range. codec-review-59-60 M-A2 follow-up.
    #[test]
    fn test_amf_variant_int64_layout() {
        let v = AmfVariant::int64(0x0123_4567_89ab_cdef);
        assert_eq!(v.ty, AMF_VARIANT_INT64);
        assert_eq!(v._pad, 0);
        assert_eq!(
            v.as_int64(),
            Some(0x0123_4567_89ab_cdef),
            "int64 round-trip must match"
        );
        // Byte-level check: little-endian bytes of the payload in
        // value[0..8].
        let expected = 0x0123_4567_89ab_cdefi64.to_le_bytes();
        assert_eq!(
            &v.value[..8],
            &expected,
            "int64 payload must be LE-encoded into value[0..8]"
        );
        // Size invariant held at compile time by the const_assert
        // above; belt-and-suspenders runtime check here.
        assert_eq!(std::mem::size_of::<AmfVariant>(), 32);
        assert_eq!(std::mem::offset_of!(AmfVariant, value), 8);
    }

    /// Verify AMF_IID_BUFFER matches the expected little-endian GUID
    /// layout of `{0xb1d75dbe, 0x0e6c, 0x434c, {0xb7, 0x28, 0x02,
    /// 0x85, 0x98, 0x37, 0x85, 0x7d}}` (AMFBuffer.h). codec-review-
    /// 59-60 AMF-7 follow-up.
    #[test]
    fn test_amf_iid_buffer_byte_order() {
        // First 4 bytes = LE u32 of 0xb1d75dbe
        assert_eq!(&AMF_IID_BUFFER[0..4], &0xb1d75dbeu32.to_le_bytes());
        // Next 2 bytes = LE u16 of 0x0e6c
        assert_eq!(&AMF_IID_BUFFER[4..6], &0x0e6cu16.to_le_bytes());
        // Next 2 bytes = LE u16 of 0x434c
        assert_eq!(&AMF_IID_BUFFER[6..8], &0x434cu16.to_le_bytes());
        // Trailing 8 bytes are raw.
        assert_eq!(
            &AMF_IID_BUFFER[8..16],
            &[0xb7, 0x28, 0x02, 0x85, 0x98, 0x37, 0x85, 0x7d]
        );
    }

    /// Quality-preset mapping must cover all four documented AMF enum
    /// values, not an arbitrary scale — codec-review-59-60 AMF-3.
    #[test]
    fn test_amf_quality_preset_mapping_exhaustive() {
        assert_eq!(amf_quality_preset_i64(AmfQualityPreset::HighQuality), 10);
        assert_eq!(amf_quality_preset_i64(AmfQualityPreset::Quality), 30);
        assert_eq!(amf_quality_preset_i64(AmfQualityPreset::Balanced), 50);
        assert_eq!(amf_quality_preset_i64(AmfQualityPreset::Speed), 70);
    }

    // ── Squad-22: AMF 10-bit dispatch + color signalling ─────────

    /// Surface-format dispatch must map `Yuv420p10le` to P010 and
    /// `Yuv420p` to NV12 — anything else must bail. Same correctness-
    /// by-review story as NVENC: a wide-word surface allocated as NV12
    /// would receive byte-truncated samples → silent black frames.
    #[test]
    fn test_amf_surface_format_dispatch() {
        assert_eq!(
            amf_surface_format_for(PixelFormat::Yuv420p).unwrap(),
            AMF_SURFACE_NV12,
            "8-bit → NV12"
        );
        assert_eq!(
            amf_surface_format_for(PixelFormat::Yuv420p10le).unwrap(),
            AMF_SURFACE_P010,
            "10-bit → P010"
        );
        assert!(amf_surface_format_for(PixelFormat::Yuv422p).is_err());
        assert!(amf_surface_format_for(PixelFormat::Rgb24).is_err());
        assert!(amf_surface_format_for(PixelFormat::Yuv444p10le).is_err());
    }

    /// `Av1ColorBitDepth` SetProperty value must be 2 for 10-bit (NOT
    /// 10 — easy mis-set; the property is an enum, not a literal bit
    /// depth). vendor/amd/VideoEncoderAV1.h:58-59.
    #[test]
    fn test_amf_color_bit_depth_dispatch() {
        assert_eq!(amf_color_bit_depth_for(PixelFormat::Yuv420p), 1);
        assert_eq!(amf_color_bit_depth_for(PixelFormat::Yuv420p10le), 2);
    }

    /// HDR transfer codes round-trip to their H.273 numeric values,
    /// matching the NVENC + mux paths so a single `ColorMetadata`
    /// goes through three independent code paths to the same number.
    #[test]
    fn test_amf_transfer_to_h273_codes() {
        assert_eq!(transfer_to_h273(TransferFn::Bt709), 1);
        assert_eq!(transfer_to_h273(TransferFn::St2084), 16);
        assert_eq!(transfer_to_h273(TransferFn::AribStdB67), 18);
        assert_eq!(transfer_to_h273(TransferFn::Linear), 8);
        assert_eq!(transfer_to_h273(TransferFn::Bt470Bg), 4);
        assert_eq!(transfer_to_h273(TransferFn::Unspecified), 1);
    }

    /// End-to-end SetProperty sequence for an HDR10 10-bit job using a
    /// mock component — verifies the four color SetProperty calls all
    /// land with the expected numeric values, and the bit-depth
    /// property carries the AMF enum value `2` (not `10`).
    ///
    /// Records every property name+value the driver writes and asserts
    /// the HDR-related ones in declaration order. The mock component
    /// vtable from the existing tests is reused.
    #[test]
    fn test_amf_hdr10_set_property_sequence() {
        thread_local! {
            static RECORDED: std::cell::RefCell<Vec<(String, i64)>> =
                const { std::cell::RefCell::new(Vec::new()) };
        }
        unsafe extern "C" fn record_set_property(
            _: *mut c_void,
            name: *const u16,
            v: AmfVariant,
        ) -> AmfResult {
            // Decode the wide string back to UTF-8.
            unsafe {
                let mut len = 0usize;
                while *name.add(len) != 0 {
                    len += 1;
                }
                let slice = std::slice::from_raw_parts(name, len);
                let s = String::from_utf16_lossy(slice);
                let value = v.as_int64().unwrap_or(0);
                RECORDED.with(|r| r.borrow_mut().push((s, value)));
            }
            AMF_OK
        }

        static REC_VTBL: AmfComponentVtbl = AmfComponentVtbl {
            query_interface: mock_qi,
            acquire: mock_acquire,
            release: mock_release_component,
            set_property: record_set_property,
            get_property: mock_get_property,
            init: mock_init,
            reinit: mock_reinit,
            terminate: mock_terminate,
            drain: mock_drain,
            flush: mock_flush,
            submit_input: mock_submit_input,
            query_output: mock_query_output,
            get_context: mock_get_context,
            set_output_data_allocator_cb: mock_set_output_cb,
            get_caps: mock_get_caps,
            optimize: mock_optimize,
        };

        let mut component = Box::new(AmfComponentObj { vtbl: &REC_VTBL });
        let component_ptr: *mut c_void = component.as_mut() as *mut _ as *mut c_void;
        let vt: &AmfComponentVtbl = unsafe { &*(*(component_ptr as *mut AmfComponentObj)).vtbl };

        // 10-bit + HDR10 metadata.
        let cm = ColorMetadata {
            transfer: TransferFn::St2084,
            matrix_coefficients: 9, // BT.2020 NCL
            colour_primaries: 9,    // BT.2020
            full_range: true,
            mastering_display: None,
            content_light_level: None,
        };

        // Drive the same SetProperty sequence the production new() path
        // uses for 10-bit + HDR10.
        unsafe {
            set_int_property(
                component_ptr,
                vt,
                "Av1ColorBitDepth",
                amf_color_bit_depth_for(PixelFormat::Yuv420p10le),
            )
            .unwrap();
            set_int_property(
                component_ptr,
                vt,
                "Av1OutColorPrimaries",
                cm.colour_primaries as i64,
            )
            .unwrap();
            set_int_property(
                component_ptr,
                vt,
                "Av1OutColorTransferChar",
                transfer_to_h273(cm.transfer),
            )
            .unwrap();
            set_int_property(
                component_ptr,
                vt,
                "Av1OutColorMatrixCoeff",
                cm.matrix_coefficients as i64,
            )
            .unwrap();
            set_int_property(component_ptr, vt, "Av1OutColorRange", cm.full_range as i64).unwrap();
        }

        let recorded: Vec<(String, i64)> = RECORDED.with(|r| r.borrow().clone());
        // Find each property by name to be order-tolerant — the test
        // asserts the values, not the call order.
        let lookup = |name: &str| -> i64 {
            recorded
                .iter()
                .find(|(n, _)| n == name)
                .expect("property recorded")
                .1
        };
        assert_eq!(
            lookup("Av1ColorBitDepth"),
            2,
            "10-bit enum is value 2, not 10"
        );
        assert_eq!(lookup("Av1OutColorPrimaries"), 9, "BT.2020");
        assert_eq!(lookup("Av1OutColorTransferChar"), 16, "ST 2084 / PQ");
        assert_eq!(lookup("Av1OutColorMatrixCoeff"), 9, "BT.2020 NCL");
        assert_eq!(lookup("Av1OutColorRange"), 1, "full range");
    }
}
