//! AMD AMF hardware encoders — AV1, H.264 and H.265 — via the Advanced Media
//! Framework runtime.
//!
//! Loads `amfrt64.dll` / `libamfrt64.so.1` at runtime via dlopen and drives
//! the C vtables mirrored in [`ffi`] (slot-for-slot from the AMF SDK v1.4.36
//! headers, with compile-time offset proofs). Three components sit behind
//! one session flow:
//!
//! | Codec | Component id (`components/VideoEncoder*.h`) | Silicon |
//! |-------|---------------------------------------------|---------|
//! | AV1   | `AMFVideoEncoderHW_AV1`  (`av1.rs`)          | RDNA3+ (RX 7000+) |
//! | H.264 | `AMFVideoEncoderVCE_AVC` (`h26x.rs`)         | every AMF-capable GPU |
//! | H.265 | `AMFVideoEncoderHW_HEVC` (`h26x.rs`)         | Polaris+ (8-bit), Vega+ (Main 10) |
//!
//! On a GPU without the component, `CreateComponent` fails and the error is
//! surfaced to `select_encoder`'s fallback chain.
//!
//! Session flow (mirroring the AMF `SimpleEncoder` sample for host-memory
//! submission):
//! 1. dlopen `amfrt64.dll` / `libamfrt64.so.1`
//! 2. `AMFInit(AMF_VERSION, &factory)`
//! 3. `factory->CreateContext(&ctx)`; then
//!    - Windows: a D3D11 device made on the chosen AMD adapter
//!      (`crate::amf_device`) handed to `ctx->InitDX11(dev, AMF_DX11_1)` —
//!      `InitDX11(null)` would bind DXGI adapter 0, which on a mixed host is
//!      the wrong (non-AMD) card;
//!    - elsewhere: `ctx->QueryInterface(IID_AMFContext1)` →
//!      `ctx1->InitVulkan(null)` (AMF picks the first AMD GPU).
//! 4. `factory->CreateComponent(ctx, <component id>, &encoder)`
//! 5. codec-specific `SetProperty` sequence (`av1.rs` / `h26x.rs`)
//! 6. `encoder->Init(NV12 | P010, width, height)`
//! 7. Per frame:
//!    - `ctx->AllocSurface(HOST, fmt, w, h, &surf)`
//!    - copy YUV420p → NV12 (or 10-bit → P010) into the surface's planes
//!    - `surf->SetPts(pts)`; on an IDR/key frame the plan's force-key
//!      property (+ SPS/PPS insertion for H.26x)
//!    - `encoder->SubmitInput(surf)` (with the back-pressure retry below)
//!    - loop `encoder->QueryOutput(&data)`: `AMF_OK` → `QueryInterface`
//!      to `AMFBuffer`, copy the bytes into an `EncodedPacket`; `AMF_REPEAT`
//!      → break
//! 8. Flush: `encoder->Drain()`; drain `QueryOutput` until `AMF_EOF`
//! 9. Drop order: `encoder->Terminate` → `encoder.Release` → `ctx.Terminate`
//!    → `ctx.Release` → (Windows) the D3D11 device → the library handle
//!    last (it provides the code behind every vtable pointer just called).
//!
//! # `AMF_INPUT_FULL` retry policy
//!
//! AMF signals `AMF_INPUT_FULL` when the encoder's input queue is saturated
//! (`core/Result.h:90`). It is a **transient** status, not a failure:
//!
//!   1. Do NOT release the surface. The caller-held ref is still valid, and
//!      releasing it makes the retry a use-after-free.
//!   2. Drain at least one output packet via `QueryOutput` to free a slot.
//!   3. Retry `SubmitInput` with the SAME surface pointer.
//!   4. Only after the eventual `AMF_OK` (or `AMF_NEED_MORE_INPUT`) does
//!      the encoder take its own ref — we then release ours.
//!
//! The ring index (`RING_SIZE` slots) follows the NVENC pattern for
//! visibility; AMF surfaces are allocated fresh per frame, so it is in-flight
//! bookkeeping, not a reuse pool.
//!
//! # Verification status
//!
//! The development box has no AMF-capable silicon (its Ryzen desktop iGPU
//! answers `AMF_NOT_FOUND` at `InitDX11`), so no component here has encoded
//! a frame on hardware. What is proven, and how, is in the module docs of
//! `ffi.rs` (layout, by compile-time offset assertion) and `h26x.rs`
//! (names and values, by header citation; ABI, against the installed
//! runtime; fall-through, by running a job on this box).

use anyhow::{Context, Result, bail};
use bytes::Bytes;
use std::ffi::c_void;
use std::ptr;

use super::{EncodedPacket, Encoder, EncoderConfig};
use crate::frame::{VideoCodec, VideoFrame};

mod av1;
mod config;
mod ffi;
mod h26x;
mod surface;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_h26x;

// Bring all sub-module items into the amf module namespace so sibling
// sub-modules can access them via `super::ItemName` and so the encoder
// code in this file can use them unqualified.
use self::av1::*;
use self::config::*;
use self::ffi::*;
use self::h26x::*;
use self::surface::*;

// ─── Codec plan ───────────────────────────────────────────────────

/// What differs between the three components once the session is up: the
/// component id, how to force a random-access point on an input surface,
/// and how to recognise one on an output buffer. The pre-`Init` property
/// sequence is the codec's `apply_*_properties` function.
#[derive(Clone, Copy)]
pub(super) struct CodecPlan {
    /// The wide string handed to `AMFFactory::CreateComponent`.
    pub(super) component_id: &'static str,
    /// `(property, value)` set on an input surface to make it an IDR / key
    /// frame (`ForcePictureType = IDR`, `HevcForcePictureType = IDR`,
    /// `Av1ForceFrameType = KEY`).
    pub(super) force_key: (&'static str, i64),
    /// Bool properties set to `true` on the same surface — the in-band
    /// parameter-set insertion the H.26x muxer path relies on.
    pub(super) key_extras: &'static [&'static str],
    /// The output buffer's frame-type property.
    pub(super) output_type: &'static str,
    /// Whether that property's value marks a random-access point.
    pub(super) is_keyframe: fn(i64) -> bool,
}

/// The plan for an output codec.
pub(super) fn plan_for(codec: VideoCodec) -> CodecPlan {
    match codec {
        VideoCodec::Av1 => AV1_PLAN,
        VideoCodec::H264 => AVC_PLAN,
        VideoCodec::H265 => HEVC_PLAN,
    }
}

/// `keyframe_interval == 0` means "the caller left it unset", not "every
/// frame an IDR": use the same 240-frame default as the other backends.
pub(super) fn effective_keyframe_interval(keyframe_interval: u32) -> u32 {
    if keyframe_interval == 0 { 240 } else { keyframe_interval }
}

// ─── Session container ────────────────────────────────────────────

/// Holds the live AMF objects. Dropped in reverse-acquisition order:
/// encoder first (it holds a strong ref on the context), context second.
/// The library handle that provides every vtable we just called drops LAST
/// via `AmfEncoder`'s field order.
struct AmfSession {
    encoder: *mut c_void,
    context: *mut c_void,
    width: u32,
    height: u32,
    pts_timescale: u64,
    /// `AMF_SURFACE_NV12` (8-bit) or `AMF_SURFACE_P010` (10-bit). Captured
    /// at session create so `upload_frame_static` knows which plane width +
    /// per-sample byte count to use.
    surface_format: i32,
}

// AMF's COM-style vtables are thread-safe per the SDK's "Thread Safety"
// appendix: every context/component object internally synchronises
// SetProperty / SubmitInput / QueryOutput. We only touch one encoder per
// `AmfEncoder`, so Send is sufficient for tokio migration; the pipeline's
// `spawn_blocking` keeps the encoder on one OS thread for its lifetime.
unsafe impl Send for AmfSession {}

impl Drop for AmfSession {
    fn drop(&mut self) {
        unsafe {
            // Encoder first — Terminate releases internal hardware
            // resources before we drop the last ref.
            if !self.encoder.is_null() {
                let vt = &*(*(self.encoder as *mut AmfComponentObj)).vtbl;
                let _ = (vt.terminate)(self.encoder);
                let _ = (vt.ps.release)(self.encoder);
            }
            // Context next — same pattern. The factory is a runtime
            // singleton and is not reference-counted; nothing to release.
            if !self.context.is_null() {
                release_context(self.context);
            }
        }
    }
}

/// `Terminate` + `Release` a context we created.
unsafe fn release_context(context: *mut c_void) {
    unsafe {
        let vt = &*(*(context as *mut AmfContextObj)).vtbl;
        let _ = (vt.terminate)(context);
        let _ = (vt.ps.release)(context);
    }
}

// ─── Encoder implementation ───────────────────────────────────────

// Field order matters for drop: `session` drops BEFORE `_amd_device` and
// `_runtime_lib`, so all the vtable calls inside `AmfSession::drop` still
// resolve to valid code and a live device. The library handle is declared
// LAST (struct fields drop in declaration order).
pub struct AmfEncoder {
    config: EncoderConfig,
    plan: CodecPlan,
    session: Option<AmfSession>,
    encoded_packets: Vec<EncodedPacket>,
    packet_cursor: usize,
    flushed: bool,
    frame_counter: u32,
    /// Set by [`Encoder::force_keyframe_next`]; consumed by the next
    /// `encode_one`, which promotes that frame to an IDR / key frame.
    force_idr_pending: bool,
    /// Current ring slot. Advances modulo `RING_SIZE` per successful
    /// `SubmitInput`. Mirrors NVENC's `ring_idx` for observational parity.
    ring_idx: usize,
    /// Keeps the AMD-adapter D3D11 device alive for the AMF context's
    /// lifetime (Windows multi-adapter routing).
    #[cfg(windows)]
    _amd_device: Option<crate::amf_device::AmdD3d11Device>,
    _runtime_lib: libloading::Library,
}

// The session is `Send` (above); the Windows D3D11 device handle is a
// free-threaded COM object that only this encoder releases, and the library
// handle is `Send` already. The pipeline moves the whole encoder to one
// blocking thread and drives it there.
unsafe impl Send for AmfEncoder {}

impl AmfEncoder {
    /// Build an encoder for `config` on the `gpu_vendor_index`-th AMD adapter
    /// (`GpuDevice::vendor_index`, the vendor-local ordinal — not the global
    /// `index`). On Windows that ordinal selects the DXGI adapter the AMF
    /// context binds to; on Linux AMF picks the first AMD GPU itself and the
    /// ordinal is logged when it is not zero.
    pub fn new(config: EncoderConfig, gpu_vendor_index: u32) -> Result<Self> {
        let plan = plan_for(config.codec);
        // Refuse the (codec, format) pairs no component takes before any
        // runtime call — a clear error beats a driver's generic
        // AMF_INVALID_ARG from Init.
        let surface_fmt = amf_surface_format_for(config.pixel_format)?;
        if config.codec != VideoCodec::Av1 {
            check_h26x_format(config.codec, config.pixel_format)?;
        }

        // 1. dlopen the AMF runtime. On Linux the library name is
        //    `libamfrt64.so.1`; on Windows it's `amfrt64.dll`. Both ship
        //    with the Adrenalin driver and Pro driver bundles.
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
                bail!("AMFInit failed: {rc} ({})", result_name(rc));
            }
            let factory_vt = &*(*(factory as *mut AmfFactoryObj)).vtbl;

            // 3. Context, bound to a GPU.
            let mut context: *mut c_void = ptr::null_mut();
            let rc = (factory_vt.create_context)(factory, &mut context);
            if rc != AMF_OK || context.is_null() {
                bail!("AMFFactory::CreateContext failed: {rc} ({})", result_name(rc));
            }
            let context_vt = &*(*(context as *mut AmfContextObj)).vtbl;

            #[cfg(windows)]
            let amd_device = {
                // A D3D11 device on the chosen AMD adapter, handed to AMF.
                // A GPU whose VCN the runtime does not drive (the desktop
                // AM5 iGPU) answers AMF_NOT_FOUND here; that is the clean
                // "not capable" exit `select_encoder` falls through on.
                let dev = match crate::amf_device::create_amd_d3d11_device(gpu_vendor_index) {
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
                        "AMFContext::InitDX11 on AMD adapter {gpu_vendor_index} failed: {rc} ({}) — \
                         this GPU is not AMF-capable",
                        result_name(rc)
                    );
                }
                Some(dev)
            };
            #[cfg(not(windows))]
            {
                if gpu_vendor_index != 0 {
                    tracing::warn!(
                        gpu_vendor_index,
                        "AMF InitVulkan(null) picks the first AMD GPU; multi-AMD hosts may need \
                         external adapter routing"
                    );
                }
                // InitVulkan lives on AMFContext1 (core/Context.h:371).
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
                // give it back now — the base `context` handle keeps it alive.
                let _ = (context1_vt.base.ps.release)(context1);
                if rc != AMF_OK {
                    release_context(context);
                    bail!(
                        "AMFContext1::InitVulkan failed: {rc} ({}) — no AMF-capable AMD GPU",
                        result_name(rc)
                    );
                }
            }

            // 4. Encoder component.
            let component_id = wide(plan.component_id);
            let mut encoder: *mut c_void = ptr::null_mut();
            let rc = (factory_vt.create_component)(factory, context, component_id.as_ptr(), &mut encoder);
            if rc != AMF_OK || encoder.is_null() {
                release_context(context);
                bail!(
                    "AMFFactory::CreateComponent({}) failed: {rc} ({}) — this GPU has no {:?} \
                     encode block the AMF runtime drives",
                    plan.component_id,
                    result_name(rc),
                    config.codec
                );
            }
            let encoder_vt = &*(*(encoder as *mut AmfComponentObj)).vtbl;

            // 5. The codec's property sequence.
            let applied = match config.codec {
                VideoCodec::Av1 => apply_av1_properties(encoder, &config),
                VideoCodec::H264 => apply_avc_properties(encoder, &config),
                VideoCodec::H265 => apply_hevc_properties(encoder, &config),
            };
            let summary = match applied {
                Ok(s) => s,
                Err(e) => {
                    let _ = (encoder_vt.ps.release)(encoder);
                    release_context(context);
                    return Err(e.context(format!("configuring the AMF {:?} encoder", config.codec)));
                }
            };

            tracing::info!(
                codec = ?config.codec,
                width = config.width,
                height = config.height,
                target = ?config.target,
                tier = ?config.tier,
                %summary,
                ring_size = RING_SIZE,
                "AMF tuning applied"
            );

            // 6. Init on the dispatched input format.
            let rc = (encoder_vt.init)(encoder, surface_fmt, config.width as i32, config.height as i32);
            if rc != AMF_OK {
                let _ = (encoder_vt.terminate)(encoder);
                let _ = (encoder_vt.ps.release)(encoder);
                release_context(context);
                bail!(
                    "AMFComponent::Init({:?}, fmt={surface_fmt}, {}x{}) failed: {rc} ({}) (surface \
                     format dispatched for {:?})",
                    config.codec,
                    config.width,
                    config.height,
                    result_name(rc),
                    config.pixel_format,
                );
            }

            let session = AmfSession {
                encoder,
                context,
                width: config.width,
                height: config.height,
                // AMF uses 100-ns ticks for PTS (`amf_pts`, core/Platform.h:218).
                // Frame PTS arrive as sample counts; convert by
                // (10_000_000 / frame_rate).
                pts_timescale: (10_000_000.0f64 / config.frame_rate).round() as u64,
                surface_format: surface_fmt,
            };

            tracing::info!(
                codec = ?config.codec,
                component = plan.component_id,
                width = config.width,
                height = config.height,
                gpu_vendor_index,
                "AMF encoder ready"
            );

            Ok(Self {
                config,
                plan,
                session: Some(session),
                encoded_packets: Vec::new(),
                packet_cursor: 0,
                flushed: false,
                frame_counter: 0,
                force_idr_pending: false,
                ring_idx: 0,
                #[cfg(windows)]
                _amd_device: amd_device,
                _runtime_lib: runtime_lib,
            })
        }
    }

    fn encode_one(&mut self, frame: &VideoFrame) -> Result<()> {
        // The encoder/context raw pointers are read from `&self.session`
        // and copied into a plain-data snapshot for the unsafe block, so a
        // future refactor that calls `self.session.take()` inside it is a
        // compile error rather than a silent UAF.
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("encode_one called after session drop"))?;
        let encoder_ptr = session.encoder;
        let snap = SessionSnapshot {
            context: session.context,
            width: session.width,
            height: session.height,
            pts_timescale: session.pts_timescale,
            surface_format: session.surface_format,
        };
        // Every GOP boundary is driven from here as well as by the
        // component's own IDR period, so the two can never disagree about
        // where a segment may start; plus whatever the chunked path asked
        // for through `force_keyframe_next`.
        let force_key = self.force_idr_pending
            || self
                .frame_counter
                .is_multiple_of(effective_keyframe_interval(self.config.keyframe_interval));
        let plan = self.plan;
        let packets = &mut self.encoded_packets;
        let ring_slot = self.ring_idx;

        let outcome = unsafe {
            // catch_unwind so a panic in our FFI path never unwinds across
            // the AMF C ABI (UB).
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let raw_surface = upload_frame_static(&snap, frame)?;
                // RAII guard: the surface is released on every exit path
                // unless `transfer_to_encoder` is called after a successful
                // SubmitInput.
                let mut guard = SurfaceGuard::new(raw_surface);

                if force_key {
                    mark_key_frame(guard.as_ptr(), &plan)?;
                }

                submit_with_backpressure(packets, encoder_ptr, &mut guard, &plan)?;

                // Drain whatever is ready now. AMF sometimes produces a
                // packet per SubmitInput, sometimes not.
                drain_until_hungry_raw(packets, encoder_ptr, &plan)?;
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
        self.force_idr_pending = false;
        self.frame_counter += 1;
        self.ring_idx = (ring_slot + 1) % RING_SIZE;
        Ok(())
    }

    fn flush_drain(&mut self) -> Result<()> {
        let encoder_ptr = match &self.session {
            Some(s) => s.encoder,
            None => return Ok(()),
        };
        let plan = self.plan;
        let packets = &mut self.encoded_packets;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            let encoder_vt = &*(*(encoder_ptr as *mut AmfComponentObj)).vtbl;
            // Drain() marks the pipeline as "no more input will ever
            // arrive"; QueryOutput then empties the reorder buffer until
            // AMF_EOF.
            let rc = (encoder_vt.drain)(encoder_ptr);
            if rc != AMF_OK && rc != AMF_REPEAT {
                bail!("AMF Drain failed: {rc} ({})", result_name(rc));
            }
            drain_until_hungry_raw(packets, encoder_ptr, &plan)?;
            Ok::<(), anyhow::Error>(())
        }));
        match result {
            Ok(inner) => inner,
            Err(_panic) => {
                bail!("panic in AMF flush path — aborting rather than unwinding across FFI")
            }
        }
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

    fn force_keyframe_next(&mut self) -> Result<()> {
        // Supported for all three codecs: the next surface gets the plan's
        // force-key property (IDR for H.26x, KEY for AV1), with the
        // parameter sets re-inserted for H.26x so the chunk stands alone.
        self.force_idr_pending = true;
        Ok(())
    }
}

/// Set the plan's force-key property (and its extras) on an input surface.
unsafe fn mark_key_frame(surface: *mut c_void, plan: &CodecPlan) -> Result<()> {
    unsafe {
        let (name, value) = plan.force_key;
        set_int_property(surface, name, value)?;
        for extra in plan.key_extras {
            set_bool_property(surface, extra, true)?;
        }
        Ok(())
    }
}

/// Submit `guard.as_ptr()` to the encoder, retrying on transient
/// back-pressure statuses. On success the guard is marked as transferred
/// and its `Drop` becomes a no-op (the encoder's internal ref now owns the
/// surface lifetime). On hard failure the guard's `Drop` releases our
/// caller-held ref exactly once.
///
/// Retry policy: bounded at `INPUT_FULL_MAX_RETRIES` attempts with
/// exponential backoff from `INPUT_FULL_BACKOFF_MS_INITIAL` ms capped at
/// `INPUT_FULL_BACKOFF_MS_MAX` ms, with a drain pass between attempts.
unsafe fn submit_with_backpressure(
    packets: &mut Vec<EncodedPacket>,
    encoder: *mut c_void,
    guard: &mut SurfaceGuard,
    plan: &CodecPlan,
) -> Result<()> {
    unsafe {
        let encoder_vt = &*(*(encoder as *mut AmfComponentObj)).vtbl;
        let mut backoff_ms = INPUT_FULL_BACKOFF_MS_INITIAL;
        for attempt in 0..=INPUT_FULL_MAX_RETRIES {
            let rc = (encoder_vt.submit_input)(encoder, guard.as_ptr());
            match rc {
                AMF_OK | AMF_NEED_MORE_INPUT => {
                    // SubmitInput took its own ref; ours is now redundant —
                    // release it exactly once and mark the guard so Drop is
                    // a no-op at scope exit.
                    let surface_vt = &*(*(guard.as_ptr() as *mut AmfSurfaceObj)).vtbl;
                    (surface_vt.data.ps.release)(guard.as_ptr());
                    guard.transfer_to_encoder();
                    return Ok(());
                }
                AMF_INPUT_FULL | AMF_REPEAT => {
                    // Transient — drain output to free an input slot, then
                    // retry. The surface is NOT released here; the guard
                    // still owns the caller-held ref and the same pointer
                    // is handed back to the retry.
                    if attempt == INPUT_FULL_MAX_RETRIES {
                        tracing::warn!(
                            status = rc,
                            attempts = attempt + 1,
                            "AMF SubmitInput backpressure exceeded retry budget — \
                             surface still caller-owned, releasing via guard"
                        );
                        bail!(
                            "AMF SubmitInput stuck at {rc} ({}) after {} attempts",
                            result_name(rc),
                            attempt + 1
                        );
                    }
                    drain_until_hungry_raw(packets, encoder, plan)?;
                    if attempt > 0 {
                        std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
                        backoff_ms = (backoff_ms * 2).min(INPUT_FULL_BACKOFF_MS_MAX);
                    }
                    continue;
                }
                other => {
                    // Hard error: surface still caller-owned. Guard's Drop
                    // releases our ref on return from bail.
                    tracing::warn!(
                        status = other,
                        "AMF SubmitInput hard failure — surface still caller-owned, \
                         releasing via guard"
                    );
                    bail!("AMF SubmitInput failed: {other} ({})", result_name(other));
                }
            }
        }
        unreachable!("submit_with_backpressure loop invariant violated")
    }
}

/// Drain `QueryOutput` into `packets` until the encoder returns
/// `AMF_REPEAT` (no more data yet), `AMF_EOF`, or `AMF_NEED_MORE_INPUT`.
/// Free function (not a method on `AmfEncoder`) so it takes
/// `&mut Vec<EncodedPacket>` rather than `&mut self`; this keeps
/// `&self.session` alive through the call.
unsafe fn drain_until_hungry_raw(
    packets: &mut Vec<EncodedPacket>,
    encoder: *mut c_void,
    plan: &CodecPlan,
) -> Result<()> {
    unsafe {
        let encoder_vt = &*(*(encoder as *mut AmfComponentObj)).vtbl;
        loop {
            let mut data: *mut c_void = ptr::null_mut();
            let rc = (encoder_vt.query_output)(encoder, &mut data);
            match rc {
                AMF_OK => {
                    if data.is_null() {
                        continue;
                    }
                    let converted = buffer_to_packet(data, plan);
                    // Drop the AMFData ref QueryOutput handed us, whatever
                    // `buffer_to_packet` did.
                    let data_vt = &*(*(data as *mut AmfObj)).vtbl;
                    (data_vt.release)(data);
                    if let Some(pkt) = converted? {
                        packets.push(pkt);
                    }
                }
                // "no more data this round but more may appear later".
                AMF_REPEAT => return Ok(()),
                // Expected terminator after `Drain()`.
                AMF_EOF => return Ok(()),
                // The encoder wants more frames before it can emit (lookahead
                // warm-up); equivalent to "no packet yet".
                AMF_NEED_MORE_INPUT => return Ok(()),
                other => bail!("AMF QueryOutput failed: {other} ({})", result_name(other)),
            }
        }
    }
}

/// Cross-cast an `AMFData*` to `AMFBuffer*` via `QueryInterface` and copy
/// its bytes into an `EncodedPacket`, tagging keyframes from the plan's
/// output frame-type property.
///
/// SAFETY precondition: `AMFData` and `AMFBuffer` share the 13-slot
/// property-storage prefix (`ffi.rs`), so the `QueryInterface` call is made
/// through that prefix. If `QueryInterface` fails we bail rather than treat
/// the `AMFData` as an `AMFBuffer`.
unsafe fn buffer_to_packet(data: *mut c_void, plan: &CodecPlan) -> Result<Option<EncodedPacket>> {
    unsafe {
        let data_vt = &*(*(data as *mut AmfObj)).vtbl;

        let mut buffer: *mut c_void = ptr::null_mut();
        let qi_rc = (data_vt.query_interface)(data, &AMF_IID_BUFFER, &mut buffer);
        if qi_rc != AMF_OK || buffer.is_null() {
            bail!(
                "AMFData::QueryInterface(AMFBuffer) failed: {qi_rc} ({})",
                result_name(qi_rc)
            );
        }
        let buffer_vt = &*(*(buffer as *mut AmfBufferObj)).vtbl;

        let size = (buffer_vt.get_size)(buffer);
        let native = (buffer_vt.get_native)(buffer) as *const u8;
        if size == 0 || native.is_null() {
            (buffer_vt.data.ps.release)(buffer);
            return Ok(None);
        }

        let data_bytes = Bytes::copy_from_slice(std::slice::from_raw_parts(native, size));
        let pts_ticks = (buffer_vt.data.get_pts)(buffer) as u64;

        // The frame-type property tags keyframes. A missing property reads
        // as "not a keyframe" — the muxer then has no sync sample to cut at,
        // which is visible, rather than a wrong one, which is not.
        let is_keyframe = get_int_property(buffer, plan.output_type).is_some_and(plan.is_keyframe);

        (buffer_vt.data.ps.release)(buffer);

        Ok(Some(EncodedPacket {
            data: data_bytes,
            pts: pts_ticks,
            is_keyframe,
        }))
    }
}
