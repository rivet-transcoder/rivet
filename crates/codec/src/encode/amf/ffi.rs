//! COM-vtable FFI bindings for the AMF runtime, mirrored slot-for-slot from
//! the AMD AMF SDK **v1.4.36** C headers (`amf/public/include/…`).
//!
//! Every `#[repr(C)]` vtable below lists **every** slot of its C `…Vtbl`
//! typedef, in header order, and a `const` block at the end of the file pins
//! the byte offset of every slot we call (and the size of every vtable) to
//! the value the header implies (`slot index × 8`). Header references are
//! `File.h:line` in the v1.4.36 tag.
//!
//! Layout facts that are easy to get wrong, all taken from the headers:
//!
//! - `AMFInterface` is **Acquire, Release, QueryInterface** in that order
//!   (`core/Interface.h:79-81`) — not COM's `IUnknown` order.
//! - Every property-bearing interface then carries the ten
//!   `AMFPropertyStorage` slots (`core/PropertyStorage.h:117-126`) before its
//!   own methods; `AMFComponent` adds the four `AMFPropertyStorageEx` slots
//!   on top (`core/PropertyStorageEx.h:189-192`).
//! - `InitVulkan` is **not** on `AMFContext`; it is on `AMFContext1`, reached
//!   through `QueryInterface(IID_AMFContext1)` (`core/Context.h:278, 371`).
//! - `AMFVariantStruct` is 24 bytes: a 4-byte enum, 4 bytes of padding and a
//!   16-byte, 8-aligned union (`core/Variant.h:80-103`, `core/Platform.h:240`
//!   for the 16-byte `AMFRect` member).
//! - `AMF_RESULT` is a plain sequential enum (`core/Result.h:45-128`):
//!   `AMF_EOF = 23`, `AMF_REPEAT = 24`, `AMF_INPUT_FULL = 25`,
//!   `AMF_NEED_MORE_INPUT = 44`.
//! - `AMF_STD_CALL` is `__stdcall` (`core/Platform.h:120`), spelled
//!   `extern "system"` here; `amf_long` is a C `long`
//!   (`core/Platform.h:213`), which is 32-bit on Windows and 64-bit on Linux.
//! - Property names are `const wchar_t*` with the platform's `wchar_t`
//!   (there is no AMF-specific wide-char typedef in `core/Platform.h`): 16-bit
//!   on Windows, 32-bit on Linux. See [`AmfWchar`].
//!
//! Slots we never call are typed as [`Slot`] (an opaque pointer of the same
//! width) but keep their header name so the offset table reads 1:1 against
//! the header.

use std::ffi::c_void;
use std::os::raw::c_long;

// ─── Scalar typedefs (core/Platform.h) ───────────────────────────────

/// `AMF_RESULT` (`core/Result.h:43`), a C enum → `int`.
pub(super) type AmfResult = i32;

/// The platform `wchar_t` the SDK's `const wchar_t*` property names use.
/// `core/Platform.h` defines no AMF-specific wide-char type, so this is
/// 16-bit on Windows and 32-bit everywhere else.
#[cfg(windows)]
pub(super) type AmfWchar = u16;
/// See the Windows definition.
#[cfg(not(windows))]
pub(super) type AmfWchar = u32;

/// `amf_long` (`core/Platform.h:213`): the return type of `Acquire` / `Release`.
pub(super) type AmfLong = c_long;

// ─── AMF_RESULT values (core/Result.h:45-128) ────────────────────────
//
// The enum has no explicit values after `AMF_OK = 0`; the numbers below are
// the members' positions in the header, comments and blank lines skipped.

pub(super) const AMF_OK: AmfResult = 0;
#[allow(dead_code)]
pub(super) const AMF_FAIL: AmfResult = 1;
#[allow(dead_code)]
pub(super) const AMF_INVALID_ARG: AmfResult = 4;
#[allow(dead_code)]
pub(super) const AMF_NOT_SUPPORTED: AmfResult = 10;
/// Also what `GetProperty` returns for a name the object does not carry.
#[allow(dead_code)]
pub(super) const AMF_NOT_FOUND: AmfResult = 11;
#[allow(dead_code)]
pub(super) const AMF_NO_DEVICE: AmfResult = 17;
pub(super) const AMF_EOF: AmfResult = 23;
pub(super) const AMF_REPEAT: AmfResult = 24;
pub(super) const AMF_INPUT_FULL: AmfResult = 25;
#[allow(dead_code)]
pub(super) const AMF_CODEC_NOT_SUPPORTED: AmfResult = 30;
#[allow(dead_code)]
pub(super) const AMF_SURFACE_FORMAT_NOT_SUPPORTED: AmfResult = 31;
#[allow(dead_code)]
pub(super) const AMF_ENCODER_NOT_PRESENT: AmfResult = 36;
pub(super) const AMF_NEED_MORE_INPUT: AmfResult = 44;

/// Human-readable name for the result codes this module handles, for logs.
pub(super) fn result_name(rc: AmfResult) -> &'static str {
    match rc {
        AMF_OK => "AMF_OK",
        AMF_FAIL => "AMF_FAIL",
        AMF_INVALID_ARG => "AMF_INVALID_ARG",
        AMF_NOT_SUPPORTED => "AMF_NOT_SUPPORTED",
        AMF_NOT_FOUND => "AMF_NOT_FOUND",
        AMF_NO_DEVICE => "AMF_NO_DEVICE",
        AMF_EOF => "AMF_EOF",
        AMF_REPEAT => "AMF_REPEAT",
        AMF_INPUT_FULL => "AMF_INPUT_FULL",
        AMF_CODEC_NOT_SUPPORTED => "AMF_CODEC_NOT_SUPPORTED",
        AMF_SURFACE_FORMAT_NOT_SUPPORTED => "AMF_SURFACE_FORMAT_NOT_SUPPORTED",
        AMF_ENCODER_NOT_PRESENT => "AMF_ENCODER_NOT_PRESENT",
        AMF_NEED_MORE_INPUT => "AMF_NEED_MORE_INPUT",
        _ => "AMF_RESULT(other)",
    }
}

// ─── Version (core/Version.h:45-57) ──────────────────────────────────

/// `AMF_MAKE_FULL_VERSION` (`core/Version.h:45`).
const fn amf_make_version(major: u64, minor: u64, release: u64, build: u64) -> u64 {
    (major << 48) | (minor << 32) | (release << 16) | build
}

/// The version handed to `AMFInit`. 1.4.30 rather than the header's 1.4.36
/// so an older Adrenalin runtime still accepts us; every slot and property
/// this module uses exists at 1.4.30 (the C vtables are append-only across
/// 1.4.x). Same value as `decode/amf_dec.rs`.
pub(super) const AMF_VERSION: u64 = amf_make_version(1, 4, 30, 0);

// ─── Enums ───────────────────────────────────────────────────────────

/// `AMF_MEMORY_HOST` (`core/Data.h:57`).
pub(super) const AMF_MEMORY_HOST: i32 = 1;

/// `AMF_DX11_0` (`core/Data.h:75`): the default `dxVersionRequired` for
/// `InitDX11` (the C++ default argument, `core/Context.h:67`).
#[allow(dead_code)]
pub(super) const AMF_DX11_0: i32 = 110;
/// `AMF_DX11_1` (`core/Data.h:76`): what we ask for when handing AMF the
/// D3D11.1 device made by `crate::amf_device`.
#[allow(dead_code)]
pub(super) const AMF_DX11_1: i32 = 111;

/// `AMF_SURFACE_NV12` (`core/Surface.h:53`).
pub(super) const AMF_SURFACE_NV12: i32 = 1;
/// `AMF_SURFACE_P010` (`core/Surface.h:62`): "16 allocated, upper 10 bits
/// are used" — the `<<6` in `surface.rs` comes from that line.
pub(super) const AMF_SURFACE_P010: i32 = 10;

/// `AMF_PLANE_Y` / `AMF_PLANE_UV` (`core/Plane.h:48-49`).
pub(super) const AMF_PLANE_Y: i32 = 2;
pub(super) const AMF_PLANE_UV: i32 = 3;

/// `AMF_VARIANT_TYPE` tags we read or write (`core/Variant.h:56-63`).
pub(super) const AMF_VARIANT_BOOL: i32 = 1;
pub(super) const AMF_VARIANT_INT64: i32 = 2;
pub(super) const AMF_VARIANT_RATE: i32 = 7;

/// `AMF_COLOR_BIT_DEPTH_ENUM` (`components/ColorSpace.h:105-107`): the
/// values are the literal bit depths, `8` and `10`.
pub(super) const AMF_COLOR_BIT_DEPTH_8: i64 = 8;
pub(super) const AMF_COLOR_BIT_DEPTH_10: i64 = 10;

/// `AMF_VIDEO_CONVERTER_COLOR_PROFILE_ENUM` (`components/ColorSpace.h:46-57`).
pub(super) const AMF_COLOR_PROFILE_709: i64 = 1;
pub(super) const AMF_COLOR_PROFILE_2020: i64 = 2;
pub(super) const AMF_COLOR_PROFILE_FULL_709: i64 = 7;
pub(super) const AMF_COLOR_PROFILE_FULL_2020: i64 = 8;

// ─── Ring-buffer / back-pressure configuration ───────────────────────
//
// Squad-5's NVENC path uses RING_SIZE=4 (mirrors ffmpeg's libavcodec/
// nvenc.c default `nb_surfaces`). We mirror the same depth for AMF so ops
// can reason about in-flight buffers uniformly across both vendors. Each
// AMF surface is allocated fresh per frame (the encoder keeps its own ref
// after SubmitInput); the ring index is in-flight bookkeeping, not a pool.
pub(super) const RING_SIZE: usize = 4;

// `AMF_INPUT_FULL` retry policy. The SDK documents INPUT_FULL as transient
// ("returned by AMFComponent::SubmitInput if input queue is full",
// core/Result.h:90): the caller should drain at least one output packet and
// retry. Bounded so a stuck driver cannot spin us forever.
pub(super) const INPUT_FULL_MAX_RETRIES: u32 = 16;
pub(super) const INPUT_FULL_BACKOFF_MS_INITIAL: u64 = 1;
pub(super) const INPUT_FULL_BACKOFF_MS_MAX: u64 = 16;

// ─── AMFGuid (core/Platform.h:508-521) ───────────────────────────────

/// `AMFGuid`: `data1: u32, data2: u16, data3: u16, data41..data48: u8`.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct AmfGuid {
    pub(super) data1: u32,
    pub(super) data2: u16,
    pub(super) data3: u16,
    pub(super) data4: [u8; 8],
}

/// `IID_AMFBuffer` (`core/Buffer.h:102` / `:135`):
/// `{0xb04b7248, 0xb6f0, 0x4321, {0xb6, 0x91, 0xba, 0xa4, 0x74, 0x0f, 0x9f, 0xcb}}`.
pub(super) const AMF_IID_BUFFER: AmfGuid = AmfGuid {
    data1: 0xb04b_7248,
    data2: 0xb6f0,
    data3: 0x4321,
    data4: [0xb6, 0x91, 0xba, 0xa4, 0x74, 0x0f, 0x9f, 0xcb],
};

/// `IID_AMFContext1` (`core/Context.h:140` / `:278`):
/// `{0xd9e9f868, 0x6220, 0x44c6, {0xa2, 0x2f, 0x7c, 0xd6, 0xda, 0xc6, 0x86, 0x46}}`.
#[cfg_attr(windows, allow(dead_code))]
pub(super) const AMF_IID_CONTEXT1: AmfGuid = AmfGuid {
    data1: 0xd9e9_f868,
    data2: 0x6220,
    data3: 0x44c6,
    data4: [0xa2, 0x2f, 0x7c, 0xd6, 0xda, 0xc6, 0x86, 0x46],
};

// ─── AMFVariantStruct (core/Variant.h:80-103) ────────────────────────

/// The union arm of `AMFVariantStruct`. Its widest members are the 16-byte
/// `AMFRect` (`core/Platform.h:240-246`) and `AMFFloatVector4D`
/// (`core/Platform.h:359-365`); its strictest alignment is 8 (`amf_int64`,
/// `amf_double`, the pointers).
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) union AmfVariantValue {
    /// `amf_bool boolValue` — `amf_uint8` in C (`core/Platform.h:208`), a
    /// one-byte `bool` in C++ (`:206`); the same one byte either way.
    pub(super) bool_: u8,
    /// `amf_int64 int64Value`.
    pub(super) int64: i64,
    /// `amf_double doubleValue`.
    pub(super) double: f64,
    /// `char* stringValue` / `wchar_t* wstringValue` / `AMFInterface* pInterface`.
    pub(super) pointer: *mut c_void,
    /// `AMFRect rectValue` (`left, top, right, bottom`).
    pub(super) rect: [i32; 4],
    /// `AMFRate rateValue` (`num, den`; `core/Platform.h:381-384`) — also
    /// `AMFRatio`, `AMFSize`, `AMFPoint`.
    pub(super) rate: [u32; 2],
    /// `AMFFloatVector4D floatVector4DValue` — also the 16-byte zero
    /// initialiser.
    pub(super) float4: [f32; 4],
}

/// `AMFVariantStruct`: `AMF_VARIANT_TYPE type` at 0, the union at 8.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct AmfVariant {
    pub(super) ty: i32,
    pub(super) value: AmfVariantValue,
}

impl AmfVariant {
    /// An `AMF_VARIANT_EMPTY` variant with every payload byte zero.
    pub(super) const fn empty() -> Self {
        Self {
            ty: 0,
            value: AmfVariantValue { float4: [0.0; 4] },
        }
    }

    /// `AMFVariantAssignInt64` (`core/Variant.h:176`).
    pub(super) const fn int64(v: i64) -> Self {
        let mut out = Self::empty();
        out.ty = AMF_VARIANT_INT64;
        out.value = AmfVariantValue { int64: v };
        out
    }

    /// `AMFVariantAssignBool` (`core/Variant.h:175`).
    pub(super) const fn bool_(v: bool) -> Self {
        let mut out = Self::empty();
        out.ty = AMF_VARIANT_BOOL;
        out.value = AmfVariantValue {
            bool_: if v { 1 } else { 0 },
        };
        out
    }

    /// `AMFVariantAssignRate` — an `AMFRate { num, den }`.
    pub(super) const fn rate(num: u32, den: u32) -> Self {
        let mut out = Self::empty();
        out.ty = AMF_VARIANT_RATE;
        out.value = AmfVariantValue { rate: [num, den] };
        out
    }

    /// Read the int64 arm; `None` if the variant is not int-typed.
    pub(super) fn as_int64(&self) -> Option<i64> {
        if self.ty == AMF_VARIANT_INT64 {
            // SAFETY: the tag says the int64 arm is the live one, and every
            // arm is plain data, so reading it is defined either way.
            Some(unsafe { self.value.int64 })
        } else {
            None
        }
    }

    /// Read the bool arm; `None` if the variant is not bool-typed.
    #[allow(dead_code)]
    pub(super) fn as_bool(&self) -> Option<bool> {
        if self.ty == AMF_VARIANT_BOOL {
            // SAFETY: as above.
            Some(unsafe { self.value.bool_ } != 0)
        } else {
            None
        }
    }

    /// Read the rate arm as `(num, den)`; `None` if not rate-typed.
    #[allow(dead_code)]
    pub(super) fn as_rate(&self) -> Option<(u32, u32)> {
        if self.ty == AMF_VARIANT_RATE {
            // SAFETY: as above.
            let r = unsafe { self.value.rate };
            Some((r[0], r[1]))
        } else {
            None
        }
    }
}

// ─── Opaque slot ─────────────────────────────────────────────────────

/// A vtable slot we never call. Same width as a function pointer, so the
/// slots after it land where the header puts them; `Sync` so mock vtables in
/// tests can live in `static`s.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub(super) struct Slot(*const c_void);

// SAFETY: a `Slot` is never dereferenced by this crate; it only occupies
// space in a vtable that the runtime (or a test mock) owns.
unsafe impl Sync for Slot {}

impl Slot {
    /// Only mock vtables in tests fill slots by hand.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) const NULL: Slot = Slot(std::ptr::null());
}

// ─── Function-pointer shapes ─────────────────────────────────────────

/// `amf_long Acquire(pThis)` / `amf_long Release(pThis)`.
pub(super) type RefCountFn = unsafe extern "system" fn(*mut c_void) -> AmfLong;
/// `AMF_RESULT QueryInterface(pThis, const AMFGuid*, void**)`.
pub(super) type QueryInterfaceFn =
    unsafe extern "system" fn(*mut c_void, *const AmfGuid, *mut *mut c_void) -> AmfResult;
/// `AMF_RESULT SetProperty(pThis, const wchar_t* name, AMFVariantStruct value)`
/// — the variant goes **by value** (`core/PropertyStorage.h:117`).
pub(super) type SetPropertyFn =
    unsafe extern "system" fn(*mut c_void, *const AmfWchar, AmfVariant) -> AmfResult;
/// `AMF_RESULT GetProperty(pThis, const wchar_t* name, AMFVariantStruct* pValue)`.
pub(super) type GetPropertyFn =
    unsafe extern "system" fn(*mut c_void, *const AmfWchar, *mut AmfVariant) -> AmfResult;
/// `amf_bool HasProperty(pThis, const wchar_t* name)`.
pub(super) type HasPropertyFn = unsafe extern "system" fn(*mut c_void, *const AmfWchar) -> u8;
/// `amf_size GetPropertyCount(pThis)`.
pub(super) type GetPropertyCountFn = unsafe extern "system" fn(*mut c_void) -> usize;
/// `AMF_RESULT f(pThis)`.
pub(super) type ResultFn = unsafe extern "system" fn(*mut c_void) -> AmfResult;

// ─── AMFPropertyStorage prefix (core/Interface.h:79-81 + PropertyStorage.h:117-126) ──

/// The 13 slots every property-bearing AMF interface starts with: the three
/// `AMFInterface` slots then the ten `AMFPropertyStorage` slots. `AMFData`,
/// `AMFBuffer`, `AMFSurface`, `AMFComponent`, `AMFContext` and `AMFContext1`
/// all begin with exactly this block, so a `*mut c_void` handle to any of
/// them can be viewed through it for ref-counting and property access.
#[repr(C)]
pub(super) struct AmfPropertyStorageVtbl {
    /// `core/Interface.h:79`
    pub(super) acquire: RefCountFn,
    /// `core/Interface.h:80`
    pub(super) release: RefCountFn,
    /// `core/Interface.h:81`
    pub(super) query_interface: QueryInterfaceFn,
    /// `core/PropertyStorage.h:117`
    pub(super) set_property: SetPropertyFn,
    /// `core/PropertyStorage.h:118`
    pub(super) get_property: GetPropertyFn,
    /// `core/PropertyStorage.h:119`
    pub(super) has_property: HasPropertyFn,
    /// `core/PropertyStorage.h:120`
    pub(super) get_property_count: GetPropertyCountFn,
    /// `core/PropertyStorage.h:121`
    pub(super) get_property_at: Slot,
    /// `core/PropertyStorage.h:122`
    pub(super) clear: Slot,
    /// `core/PropertyStorage.h:123`
    pub(super) add_to: Slot,
    /// `core/PropertyStorage.h:124`
    pub(super) copy_to: Slot,
    /// `core/PropertyStorage.h:125`
    pub(super) add_observer: Slot,
    /// `core/PropertyStorage.h:126`
    pub(super) remove_observer: Slot,
}

/// Any AMF object viewed through its property-storage prefix.
#[repr(C)]
pub(super) struct AmfObj {
    pub(super) vtbl: *const AmfPropertyStorageVtbl,
}

// ─── AMFFactory (core/Factory.h:70-79) ───────────────────────────────
//
// The factory is the one interface with no `AMFInterface` prefix: it is a
// runtime singleton, not reference-counted.

#[repr(C)]
pub(super) struct AmfFactoryVtbl {
    /// `core/Factory.h:72`
    pub(super) create_context: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> AmfResult,
    /// `core/Factory.h:73`
    pub(super) create_component: unsafe extern "system" fn(
        *mut c_void,
        *mut c_void,
        *const AmfWchar,
        *mut *mut c_void,
    ) -> AmfResult,
    /// `core/Factory.h:74`
    pub(super) set_cache_folder: Slot,
    /// `core/Factory.h:75`
    pub(super) get_cache_folder: Slot,
    /// `core/Factory.h:76`
    pub(super) get_debug: Slot,
    /// `core/Factory.h:77`
    pub(super) get_trace: Slot,
    /// `core/Factory.h:78`
    pub(super) get_programs: Slot,
}

#[repr(C)]
pub(super) struct AmfFactoryObj {
    pub(super) vtbl: *const AmfFactoryVtbl,
}

// ─── AMFContext (core/Context.h:185-269) ─────────────────────────────

#[repr(C)]
pub(super) struct AmfContextVtbl {
    /// `core/Interface.h:79-81` + `core/Context.h:194-203`
    pub(super) ps: AmfPropertyStorageVtbl,
    /// `core/Context.h:208`
    pub(super) terminate: ResultFn,
    /// `core/Context.h:211`
    pub(super) init_dx9: Slot,
    /// `core/Context.h:212`
    pub(super) get_dx9_device: Slot,
    /// `core/Context.h:213`
    pub(super) lock_dx9: Slot,
    /// `core/Context.h:214`
    pub(super) unlock_dx9: Slot,
    /// `core/Context.h:216`: `InitDX11(pThis, void* pDX11Device, AMF_DX_VERSION dxVersionRequired)`
    pub(super) init_dx11: unsafe extern "system" fn(*mut c_void, *mut c_void, i32) -> AmfResult,
    /// `core/Context.h:217`
    pub(super) get_dx11_device: Slot,
    /// `core/Context.h:218`
    pub(super) lock_dx11: Slot,
    /// `core/Context.h:219`
    pub(super) unlock_dx11: Slot,
    /// `core/Context.h:222`
    pub(super) init_opencl: Slot,
    /// `core/Context.h:223`
    pub(super) get_opencl_context: Slot,
    /// `core/Context.h:224`
    pub(super) get_opencl_command_queue: Slot,
    /// `core/Context.h:225`
    pub(super) get_opencl_device_id: Slot,
    /// `core/Context.h:226`
    pub(super) get_opencl_compute_factory: Slot,
    /// `core/Context.h:227`
    pub(super) init_opencl_ex: Slot,
    /// `core/Context.h:228`
    pub(super) lock_opencl: Slot,
    /// `core/Context.h:229`
    pub(super) unlock_opencl: Slot,
    /// `core/Context.h:232`
    pub(super) init_opengl: Slot,
    /// `core/Context.h:233`
    pub(super) get_opengl_context: Slot,
    /// `core/Context.h:234`
    pub(super) get_opengl_drawable: Slot,
    /// `core/Context.h:235`
    pub(super) lock_opengl: Slot,
    /// `core/Context.h:236`
    pub(super) unlock_opengl: Slot,
    /// `core/Context.h:238`
    pub(super) init_xv: Slot,
    /// `core/Context.h:239`
    pub(super) get_xv_device: Slot,
    /// `core/Context.h:240`
    pub(super) lock_xv: Slot,
    /// `core/Context.h:241`
    pub(super) unlock_xv: Slot,
    /// `core/Context.h:244`
    pub(super) init_gralloc: Slot,
    /// `core/Context.h:245`
    pub(super) get_gralloc_device: Slot,
    /// `core/Context.h:246`
    pub(super) lock_gralloc: Slot,
    /// `core/Context.h:247`
    pub(super) unlock_gralloc: Slot,
    /// `core/Context.h:249`: `AllocBuffer(pThis, AMF_MEMORY_TYPE, amf_size, AMFBuffer**)`
    pub(super) alloc_buffer:
        unsafe extern "system" fn(*mut c_void, i32, usize, *mut *mut c_void) -> AmfResult,
    /// `core/Context.h:250`: `AllocSurface(pThis, AMF_MEMORY_TYPE, AMF_SURFACE_FORMAT, width, height, AMFSurface**)`
    pub(super) alloc_surface:
        unsafe extern "system" fn(*mut c_void, i32, i32, i32, i32, *mut *mut c_void) -> AmfResult,
    /// `core/Context.h:251`
    pub(super) alloc_audio_buffer: Slot,
    /// `core/Context.h:255`
    pub(super) create_buffer_from_host_native: Slot,
    /// `core/Context.h:256`
    pub(super) create_surface_from_host_native: Slot,
    /// `core/Context.h:258`
    pub(super) create_surface_from_dx9_native: Slot,
    /// `core/Context.h:259`
    pub(super) create_surface_from_dx11_native: Slot,
    /// `core/Context.h:260`
    pub(super) create_surface_from_opengl_native: Slot,
    /// `core/Context.h:261`
    pub(super) create_surface_from_gralloc_native: Slot,
    /// `core/Context.h:262`
    pub(super) create_surface_from_opencl_native: Slot,
    /// `core/Context.h:264`
    pub(super) create_buffer_from_opencl_native: Slot,
    /// `core/Context.h:267`
    pub(super) get_compute: Slot,
}

#[repr(C)]
pub(super) struct AmfContextObj {
    pub(super) vtbl: *const AmfContextVtbl,
}

// ─── AMFContext1 (core/Context.h:280-380) ────────────────────────────
//
// Obtained with `QueryInterface(IID_AMFContext1)`; extends `AMFContext` with
// the DX11-native buffer, the `…Ex` allocators and the Vulkan entry points.

#[repr(C)]
#[cfg_attr(windows, allow(dead_code))]
pub(super) struct AmfContext1Vtbl {
    /// `core/Context.h:283-362` — identical to `AMFContextVtbl`.
    pub(super) base: AmfContextVtbl,
    /// `core/Context.h:366`
    pub(super) create_buffer_from_dx11_native: Slot,
    /// `core/Context.h:367`
    pub(super) alloc_buffer_ex: Slot,
    /// `core/Context.h:368`
    pub(super) alloc_surface_ex: Slot,
    /// `core/Context.h:371`: `InitVulkan(pThis, void* pVulkanDevice)`
    pub(super) init_vulkan: unsafe extern "system" fn(*mut c_void, *mut c_void) -> AmfResult,
    /// `core/Context.h:372`
    pub(super) get_vulkan_device: Slot,
    /// `core/Context.h:373`
    pub(super) lock_vulkan: Slot,
    /// `core/Context.h:374`
    pub(super) unlock_vulkan: Slot,
    /// `core/Context.h:376`
    pub(super) create_surface_from_vulkan_native: Slot,
    /// `core/Context.h:377`
    pub(super) create_buffer_from_vulkan_native: Slot,
    /// `core/Context.h:378`
    pub(super) get_vulkan_device_extensions: Slot,
}

#[repr(C)]
#[cfg_attr(windows, allow(dead_code))]
pub(super) struct AmfContext1Obj {
    pub(super) vtbl: *const AmfContext1Vtbl,
}

// ─── AMFComponent (components/Component.h:144-185) ───────────────────

#[repr(C)]
pub(super) struct AmfComponentVtbl {
    /// `core/Interface.h:79-81` + `components/Component.h:152-161`
    pub(super) ps: AmfPropertyStorageVtbl,
    /// `components/Component.h:165` (`AMFPropertyStorageEx`)
    pub(super) get_properties_info_count: Slot,
    /// `components/Component.h:166`
    pub(super) get_property_info_at: Slot,
    /// `components/Component.h:167`
    pub(super) get_property_info: Slot,
    /// `components/Component.h:168`
    pub(super) validate_property: Slot,
    /// `components/Component.h:172`: `Init(pThis, AMF_SURFACE_FORMAT, width, height)`
    pub(super) init: unsafe extern "system" fn(*mut c_void, i32, i32, i32) -> AmfResult,
    /// `components/Component.h:173`: `ReInit(pThis, width, height)`
    pub(super) reinit: unsafe extern "system" fn(*mut c_void, i32, i32) -> AmfResult,
    /// `components/Component.h:174`
    pub(super) terminate: ResultFn,
    /// `components/Component.h:175`
    pub(super) drain: ResultFn,
    /// `components/Component.h:176`
    pub(super) flush: ResultFn,
    /// `components/Component.h:178`: `SubmitInput(pThis, AMFData*)`
    pub(super) submit_input: unsafe extern "system" fn(*mut c_void, *mut c_void) -> AmfResult,
    /// `components/Component.h:179`: `QueryOutput(pThis, AMFData**)`
    pub(super) query_output: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> AmfResult,
    /// `components/Component.h:180`
    pub(super) get_context: Slot,
    /// `components/Component.h:181`
    pub(super) set_output_data_allocator_cb: Slot,
    /// `components/Component.h:183`
    pub(super) get_caps: Slot,
    /// `components/Component.h:184`
    pub(super) optimize: Slot,
}

#[repr(C)]
pub(super) struct AmfComponentObj {
    pub(super) vtbl: *const AmfComponentVtbl,
}

// ─── AMFData (core/Data.h:128-164) ───────────────────────────────────

/// The `AMFData` block shared by `AMFBuffer` and `AMFSurface`: property
/// storage, then the ten `AMFData` methods.
#[repr(C)]
pub(super) struct AmfDataVtbl {
    /// `core/Interface.h:79-81` + `core/Data.h:136-145`
    pub(super) ps: AmfPropertyStorageVtbl,
    /// `core/Data.h:149`
    pub(super) get_memory_type: Slot,
    /// `core/Data.h:151`
    pub(super) duplicate: Slot,
    /// `core/Data.h:152`: `Convert(pThis, AMF_MEMORY_TYPE)`
    pub(super) convert: unsafe extern "system" fn(*mut c_void, i32) -> AmfResult,
    /// `core/Data.h:153`
    pub(super) interop: Slot,
    /// `core/Data.h:155`
    pub(super) get_data_type: Slot,
    /// `core/Data.h:157`
    pub(super) is_reusable: Slot,
    /// `core/Data.h:159`: `SetPts(pThis, amf_pts)` — **before** `GetPts`.
    pub(super) set_pts: unsafe extern "system" fn(*mut c_void, i64),
    /// `core/Data.h:160`
    pub(super) get_pts: unsafe extern "system" fn(*mut c_void) -> i64,
    /// `core/Data.h:161`
    pub(super) set_duration: unsafe extern "system" fn(*mut c_void, i64),
    /// `core/Data.h:162`
    pub(super) get_duration: unsafe extern "system" fn(*mut c_void) -> i64,
}

// ─── AMFBuffer (core/Buffer.h:137-183) ───────────────────────────────

#[repr(C)]
pub(super) struct AmfBufferVtbl {
    /// `core/Buffer.h:140-171` — identical to `AMFDataVtbl`.
    pub(super) data: AmfDataVtbl,
    /// `core/Buffer.h:175`
    pub(super) set_size: Slot,
    /// `core/Buffer.h:176`: `amf_size GetSize(pThis)`
    pub(super) get_size: unsafe extern "system" fn(*mut c_void) -> usize,
    /// `core/Buffer.h:177`: `void* GetNative(pThis)`
    pub(super) get_native: unsafe extern "system" fn(*mut c_void) -> *mut c_void,
    /// `core/Buffer.h:180`
    pub(super) add_observer_buffer: Slot,
    /// `core/Buffer.h:181`
    pub(super) remove_observer_buffer: Slot,
}

#[repr(C)]
pub(super) struct AmfBufferObj {
    pub(super) vtbl: *const AmfBufferVtbl,
}

// ─── AMFSurface (core/Surface.h:223-279) ─────────────────────────────

#[repr(C)]
pub(super) struct AmfSurfaceVtbl {
    /// `core/Surface.h:226-257` — identical to `AMFDataVtbl`.
    pub(super) data: AmfDataVtbl,
    /// `core/Surface.h:261`
    pub(super) get_format: unsafe extern "system" fn(*mut c_void) -> i32,
    /// `core/Surface.h:264`: `amf_size GetPlanesCount(pThis)`
    pub(super) get_planes_count: unsafe extern "system" fn(*mut c_void) -> usize,
    /// `core/Surface.h:265`: `AMFPlane* GetPlaneAt(pThis, amf_size)`
    pub(super) get_plane_at: unsafe extern "system" fn(*mut c_void, usize) -> *mut c_void,
    /// `core/Surface.h:266`: `AMFPlane* GetPlane(pThis, AMF_PLANE_TYPE)`
    pub(super) get_plane: unsafe extern "system" fn(*mut c_void, i32) -> *mut c_void,
    /// `core/Surface.h:268`
    pub(super) get_frame_type: Slot,
    /// `core/Surface.h:269`
    pub(super) set_frame_type: Slot,
    /// `core/Surface.h:271`
    pub(super) set_crop: Slot,
    /// `core/Surface.h:272`
    pub(super) copy_surface_region: Slot,
    /// `core/Surface.h:276`
    pub(super) add_observer_surface: Slot,
    /// `core/Surface.h:277`
    pub(super) remove_observer_surface: Slot,
}

#[repr(C)]
pub(super) struct AmfSurfaceObj {
    pub(super) vtbl: *const AmfSurfaceVtbl,
}

// ─── AMFPlane (core/Plane.h:81-100) ──────────────────────────────────
//
// `AMFPlane` is an `AMFInterface` only — no property storage.

#[repr(C)]
pub(super) struct AmfPlaneVtbl {
    /// `core/Plane.h:84`
    pub(super) acquire: RefCountFn,
    /// `core/Plane.h:85`
    pub(super) release: RefCountFn,
    /// `core/Plane.h:86`
    pub(super) query_interface: QueryInterfaceFn,
    /// `core/Plane.h:89`
    pub(super) get_type: unsafe extern "system" fn(*mut c_void) -> i32,
    /// `core/Plane.h:90`
    pub(super) get_native: unsafe extern "system" fn(*mut c_void) -> *mut c_void,
    /// `core/Plane.h:91`
    pub(super) get_pixel_size_in_bytes: unsafe extern "system" fn(*mut c_void) -> i32,
    /// `core/Plane.h:92`
    pub(super) get_offset_x: unsafe extern "system" fn(*mut c_void) -> i32,
    /// `core/Plane.h:93`
    pub(super) get_offset_y: unsafe extern "system" fn(*mut c_void) -> i32,
    /// `core/Plane.h:94`
    pub(super) get_width: unsafe extern "system" fn(*mut c_void) -> i32,
    /// `core/Plane.h:95`
    pub(super) get_height: unsafe extern "system" fn(*mut c_void) -> i32,
    /// `core/Plane.h:96`
    pub(super) get_h_pitch: unsafe extern "system" fn(*mut c_void) -> i32,
    /// `core/Plane.h:97`
    pub(super) get_v_pitch: unsafe extern "system" fn(*mut c_void) -> i32,
    /// `core/Plane.h:98`
    pub(super) is_tiled: Slot,
}

#[repr(C)]
pub(super) struct AmfPlaneObj {
    pub(super) vtbl: *const AmfPlaneVtbl,
}

// ─── AMFInit entry point (core/Factory.h, exported by amfrt64) ───────

/// `AMF_RESULT AMF_CDECL_CALL AMFInit(amf_uint64 version, AMFFactory** ppFactory)`
/// (`core/Factory.h:95` names the export, `:105` the C typedef). `AMF_CDECL_CALL`
/// is `__cdecl` (`core/Platform.h:121`).
pub(super) type FnAmfInit = unsafe extern "C" fn(u64, *mut *mut c_void) -> AmfResult;

// ─── Compile-time layout proof ───────────────────────────────────────
//
// Every slot this module calls, pinned to `header slot index × 8`. The slot
// indices are the positions of the members in the C `…Vtbl` typedefs cited
// above (comments and blank lines skipped). A mismatch here is a compile
// error, not a crash on a customer's GPU.

const PTR: usize = std::mem::size_of::<usize>();

macro_rules! slot_is {
    ($ty:ty, $field:ident, $index:expr) => {
        assert!(
            std::mem::offset_of!($ty, $field) == $index * PTR,
            concat!(stringify!($ty), "::", stringify!($field), " is not at header slot ", stringify!($index))
        );
    };
}

const _: () = {
    // 64-bit only: every offset below assumes 8-byte function pointers, and
    // `amfrt64` is the only runtime we load.
    assert!(PTR == 8, "AMF FFI layout is verified for 64-bit targets only");

    // AMFGuid (core/Platform.h:508-521): 4 + 2 + 2 + 8 = 16, no padding.
    assert!(std::mem::size_of::<AmfGuid>() == 16);
    assert!(std::mem::offset_of!(AmfGuid, data2) == 4);
    assert!(std::mem::offset_of!(AmfGuid, data3) == 6);
    assert!(std::mem::offset_of!(AmfGuid, data4) == 8);

    // AMFVariantStruct (core/Variant.h:80-103): enum at 0, 16-byte union at 8.
    assert!(std::mem::size_of::<AmfVariantValue>() == 16);
    assert!(std::mem::align_of::<AmfVariantValue>() == 8);
    assert!(std::mem::size_of::<AmfVariant>() == 24);
    assert!(std::mem::align_of::<AmfVariant>() == 8);
    assert!(std::mem::offset_of!(AmfVariant, ty) == 0);
    assert!(std::mem::offset_of!(AmfVariant, value) == 8);

    // AMFInterface + AMFPropertyStorage prefix: 13 slots.
    assert!(std::mem::size_of::<AmfPropertyStorageVtbl>() == 13 * PTR);
    slot_is!(AmfPropertyStorageVtbl, acquire, 0); // Interface.h:79
    slot_is!(AmfPropertyStorageVtbl, release, 1); // Interface.h:80
    slot_is!(AmfPropertyStorageVtbl, query_interface, 2); // Interface.h:81
    slot_is!(AmfPropertyStorageVtbl, set_property, 3); // PropertyStorage.h:117
    slot_is!(AmfPropertyStorageVtbl, get_property, 4); // PropertyStorage.h:118
    slot_is!(AmfPropertyStorageVtbl, has_property, 5); // PropertyStorage.h:119
    slot_is!(AmfPropertyStorageVtbl, get_property_count, 6); // PropertyStorage.h:120
    slot_is!(AmfPropertyStorageVtbl, remove_observer, 12); // PropertyStorage.h:126

    // AMFFactory (Factory.h:72-78): 7 slots, no prefix.
    assert!(std::mem::size_of::<AmfFactoryVtbl>() == 7 * PTR);
    slot_is!(AmfFactoryVtbl, create_context, 0);
    slot_is!(AmfFactoryVtbl, create_component, 1);

    // AMFContext (Context.h:188-267): 55 slots.
    assert!(std::mem::size_of::<AmfContextVtbl>() == 55 * PTR);
    slot_is!(AmfContextVtbl, terminate, 13); // Context.h:208
    slot_is!(AmfContextVtbl, init_dx11, 18); // Context.h:216
    slot_is!(AmfContextVtbl, alloc_buffer, 43); // Context.h:249
    slot_is!(AmfContextVtbl, alloc_surface, 44); // Context.h:250
    slot_is!(AmfContextVtbl, get_compute, 54); // Context.h:267

    // AMFContext1 (Context.h:283-378): 65 slots.
    assert!(std::mem::size_of::<AmfContext1Vtbl>() == 65 * PTR);
    slot_is!(AmfContext1Vtbl, create_buffer_from_dx11_native, 55); // Context.h:366
    slot_is!(AmfContext1Vtbl, init_vulkan, 58); // Context.h:371
    slot_is!(AmfContext1Vtbl, get_vulkan_device_extensions, 64); // Context.h:378

    // AMFComponent (Component.h:147-184): 28 slots.
    assert!(std::mem::size_of::<AmfComponentVtbl>() == 28 * PTR);
    slot_is!(AmfComponentVtbl, get_properties_info_count, 13); // Component.h:165
    slot_is!(AmfComponentVtbl, init, 17); // Component.h:172
    slot_is!(AmfComponentVtbl, reinit, 18); // Component.h:173
    slot_is!(AmfComponentVtbl, terminate, 19); // Component.h:174
    slot_is!(AmfComponentVtbl, drain, 20); // Component.h:175
    slot_is!(AmfComponentVtbl, flush, 21); // Component.h:176
    slot_is!(AmfComponentVtbl, submit_input, 22); // Component.h:178
    slot_is!(AmfComponentVtbl, query_output, 23); // Component.h:179
    slot_is!(AmfComponentVtbl, optimize, 27); // Component.h:184

    // AMFData (Data.h:131-162): 23 slots.
    assert!(std::mem::size_of::<AmfDataVtbl>() == 23 * PTR);
    slot_is!(AmfDataVtbl, get_memory_type, 13); // Data.h:149
    slot_is!(AmfDataVtbl, convert, 15); // Data.h:152
    slot_is!(AmfDataVtbl, set_pts, 19); // Data.h:159
    slot_is!(AmfDataVtbl, get_pts, 20); // Data.h:160
    slot_is!(AmfDataVtbl, get_duration, 22); // Data.h:162

    // AMFBuffer (Buffer.h:140-181): 28 slots.
    assert!(std::mem::size_of::<AmfBufferVtbl>() == 28 * PTR);
    slot_is!(AmfBufferVtbl, set_size, 23); // Buffer.h:175
    slot_is!(AmfBufferVtbl, get_size, 24); // Buffer.h:176
    slot_is!(AmfBufferVtbl, get_native, 25); // Buffer.h:177
    slot_is!(AmfBufferVtbl, remove_observer_buffer, 27); // Buffer.h:181

    // AMFSurface (Surface.h:226-277): 33 slots.
    assert!(std::mem::size_of::<AmfSurfaceVtbl>() == 33 * PTR);
    slot_is!(AmfSurfaceVtbl, get_format, 23); // Surface.h:261
    slot_is!(AmfSurfaceVtbl, get_planes_count, 24); // Surface.h:264
    slot_is!(AmfSurfaceVtbl, get_plane_at, 25); // Surface.h:265
    slot_is!(AmfSurfaceVtbl, get_plane, 26); // Surface.h:266
    slot_is!(AmfSurfaceVtbl, remove_observer_surface, 32); // Surface.h:277

    // AMFPlane (Plane.h:84-98): 13 slots.
    assert!(std::mem::size_of::<AmfPlaneVtbl>() == 13 * PTR);
    slot_is!(AmfPlaneVtbl, get_type, 3); // Plane.h:89
    slot_is!(AmfPlaneVtbl, get_native, 4); // Plane.h:90
    slot_is!(AmfPlaneVtbl, get_pixel_size_in_bytes, 5); // Plane.h:91
    slot_is!(AmfPlaneVtbl, get_width, 8); // Plane.h:94
    slot_is!(AmfPlaneVtbl, get_height, 9); // Plane.h:95
    slot_is!(AmfPlaneVtbl, get_h_pitch, 10); // Plane.h:96
    slot_is!(AmfPlaneVtbl, get_v_pitch, 11); // Plane.h:97
    slot_is!(AmfPlaneVtbl, is_tiled, 12); // Plane.h:98

    // Enum values pinned to the header lines cited at their definitions.
    assert!(AMF_SURFACE_NV12 == 1); // Surface.h:53
    assert!(AMF_SURFACE_P010 == 10); // Surface.h:62
    assert!(AMF_MEMORY_HOST == 1); // Data.h:57
    assert!(AMF_DX11_0 == 110 && AMF_DX11_1 == 111); // Data.h:75-76
    assert!(AMF_PLANE_Y == 2 && AMF_PLANE_UV == 3); // Plane.h:48-49
    assert!(AMF_COLOR_BIT_DEPTH_8 == 8 && AMF_COLOR_BIT_DEPTH_10 == 10); // ColorSpace.h:106-107
    assert!(AMF_EOF == 23 && AMF_REPEAT == 24 && AMF_INPUT_FULL == 25); // Result.h:88-90
    assert!(AMF_NEED_MORE_INPUT == 44); // Result.h:124
    assert!(AMF_NOT_FOUND == 11); // Result.h:61
};
