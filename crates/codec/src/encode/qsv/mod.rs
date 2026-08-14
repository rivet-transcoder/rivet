//! Intel QSV AV1 hardware encoder via oneVPL.
//!
//! Loads `libvpl.so.2` / `libvpl.dll` at runtime via dlopen. The AV1
//! encoder is available on Intel Arc (DG2 / BMG) discrete GPUs and on
//! Meteor Lake (Core Ultra 1xx) and Lunar Lake (Core Ultra 2xx) iGPUs.
//! On Arrow Lake + hybrid systems, QSV picks the iGPU unless the
//! dispatcher is filtered to the dGPU via `MFXSetConfigFilterProperty`.
//!
//! Session flow:
//! 1. dlopen libvpl. Walk the legacy `MFXInit` path on oneVPL 2.x +
//!    fall back to the dispatcher (`MFXLoad` → `MFXCreateSession`)
//!    so we work on hosts with either MSDK-layout runtimes or the
//!    newer unified oneVPL runtime.
//! 2. Populate `mfxVideoParam`:
//!    - `CodecId = MFX_CODEC_AV1`, `CodecProfile = MFX_PROFILE_AV1_MAIN`
//!    - `RateControlMethod` per tuning adapter (ICQ or CQP)
//!    - `TargetUsage` per speed tier (1..7)
//!    - `FrameInfo.FourCC = MFX_FOURCC_NV12`, `ChromaFormat = YUV420`
//!    - `IOPattern = IN_SYSTEM_MEMORY`
//!    - `GopPicSize = keyframe_interval`, `GopRefDist = 1` (no B-frames)
//! 3. Attach `mfxExtAV1TileParam` via `ExtParam[]` to set the tile grid
//!    (AV1 has no tile fields in the main `mfxInfoMFX` struct).
//! 4. `MFXVideoENCODE_Query(session, &par, &out)` — returns the
//!    runtime-adjusted params; if QSV reduced something we log and
//!    use the `out` struct.
//! 5. `MFXVideoENCODE_Init(session, &out)`.
//! 6. Per frame:
//!    - Pick next surface slot in the 4-deep ring.
//!    - Convert YUV420p → NV12 into that slot's backing buffer.
//!    - `MFXVideoENCODE_EncodeFrameAsync` → `syncp`.
//!    - `MFXVideoCORE_SyncOperation(session, syncp, 60_000)` → drain
//!      the `mfxBitstream` buffer.
//! 7. Flush by submitting NULL surface until `MFX_ERR_MORE_DATA` →
//!    no more output to drain.
//! 8. `MFXVideoENCODE_Close(session)` → `MFXClose(session)` →
//!    library handle drops last.
//!
//! ## Correctness bar for QSV in this repo
//!
//! Host is NVIDIA — E2E Intel GPU verification is impossible on the
//! dev box. Every struct layout below is spec-conformant-by-review
//! against `vendor/intel/` oneVPL 2.10 headers. `const_assert!` checks
//! at the bottom of the file fire at compile time if any struct size
//! drifts — mirroring the pattern established by Squad 5 in
//! `encode/nvenc.rs`.

use anyhow::{Context, Result, bail};
use bytes::Bytes;
use std::collections::VecDeque;
use std::os::raw::c_char;
use std::ptr;

use super::tuning::{self, QsvRateControl};
use super::{AUTO_FROM_TARGET, EncodedPacket, Encoder, EncoderConfig};
// `ColorMetadata` is read via `config.color_metadata` on the non-test
// side (no bare-type mention) and through `use super::*` inside the
// test module; pull it in only under cfg(test) to keep release builds
// warning-clean.
#[cfg(test)]
use crate::frame::ColorMetadata;
use crate::frame::{PixelFormat, VideoFrame};
// Shared mfx struct layouts live in one place (`qsv_ffi`) so encode + decode
// can't drift apart on layout again.
use crate::qsv_ffi::{
    MfxBitstream, MfxExtBuffer, MfxFrameData, MfxFrameInfo, MfxFrameSurface1, MfxInfoMfx,
    MfxVideoParam,
};

mod ffi;
mod config;
mod surface;
mod session;
#[cfg(test)]
mod tests;

use self::ffi::*;
use self::config::*;
use self::surface::*;
use self::session::*;

/// Ceiling for the output bitstream buffer, which grows on
/// `MFX_ERR_NOT_ENOUGH_BUFFER`. 64 MiB is far past any single frame at the
/// resolutions this encoder handles; hitting it means something is wrong, and
/// a bounded error beats growing until the allocator gives up.
const MAX_BITSTREAM_BYTES: usize = 64 * 1024 * 1024;

// ─── Encoder implementation ───────────────────────────────────────────────────
//
// Library handle declared LAST so session drops first and vtable
// calls in `Drop` still resolve to live code.

pub struct QsvEncoder {
    /// Held for potential future Reconfigure paths.  Currently unused
    /// at runtime but keeps the encoder self-describing.
    #[allow(dead_code)]
    config: EncoderConfig,
    session: Option<QsvSession>,
    encoded_packets: Vec<EncodedPacket>,
    packet_cursor: usize,
    flushed: bool,
    frame_counter: u32,
    /// Set by [`Encoder::force_keyframe_next`]; consumed by the next
    /// `send_frame`, which passes an `mfxEncodeCtrl` instead of null.
    force_idr_next: bool,
    _runtime_lib: libloading::Library,
}

/// True the first time a given `(codec, rate_control, low_power)` triple is
/// rejected by `MFXVideoENCODE_Query`, false every time after.
///
/// The rejection is expected on iHD — Query reports `MFX_ERR_UNSUPPORTED` for
/// parameter sets `Init` then accepts, so Init is the authority — but it's
/// worth saying once, because a *real* incompatibility looks identical right
/// up until Init fails.
fn first_query_rejection(codec: u32, rate_control: u16, low_power: u16) -> bool {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashSet<(u32, u16, u16)>>> = OnceLock::new();
    SEEN.get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .map(|mut s| s.insert((codec, rate_control, low_power)))
        .unwrap_or(false)
}

impl QsvEncoder {
    /// Build on a specific card, through the oneVPL 2.x dispatcher.
    pub fn new(config: EncoderConfig, gpu_index: u32) -> Result<Self> {
        Self::build(config, Some(gpu_index))
    }

    /// Build on whatever card the runtime hands out, through the legacy
    /// `MFXInit` path.
    ///
    /// # When this is the only thing that works
    ///
    /// The dispatcher (`MFXLoad` + `Impl=HARDWARE` + `MFXCreateSession`) can
    /// find no implementation on a host where the hardware is demonstrably
    /// fine. devbox is one: `vainfo` reports `VAProfileAV1Profile0 /
    /// VAEntrypointEncSliceLP` on all three of its Arc cards, GuC and HuC are
    /// loaded and authenticated, and every adapter index still answers
    /// `MFXCreateSession: -9`. The decoder on that same host works, because it
    /// has always used `MFXInit`.
    ///
    /// # Why it is a last resort and not the default
    ///
    /// `MFXInit` takes no adapter, so the runtime chooses. On a multi-GPU host
    /// every rung would land on whichever card it picks, which is the spread
    /// collapse this encoder deliberately stopped doing silently. So this is
    /// reached only after every card of the vendor has refused the dispatcher,
    /// and the caller says so in the log rather than letting it pass as normal.
    pub fn new_unpinned(config: EncoderConfig) -> Result<Self> {
        Self::build(config, None)
    }

    fn build(config: EncoderConfig, gpu_index: Option<u32>) -> Result<Self> {
        let runtime_lib = unsafe { libloading::Library::new("libvpl.so.2") }
            .or_else(|_| unsafe { libloading::Library::new("libvpl.so") })
            .or_else(|_| unsafe { libloading::Library::new("libvpl.dll") })
            .or_else(|_| unsafe { libloading::Library::new("libmfx.so.1") })
            .or_else(|_| unsafe { libloading::Library::new("libmfxhw64.dll") })
            .context("loading oneVPL runtime library (Intel GPU driver not present?)")?;

        unsafe {
            let fn_load: libloading::Symbol<FnMfxLoad> =
                runtime_lib.get(b"MFXLoad").context("MFXLoad symbol")?;
            let fn_create_config: libloading::Symbol<FnMfxCreateConfig> = runtime_lib
                .get(b"MFXCreateConfig")
                .context("MFXCreateConfig symbol")?;
            let fn_set_filter: libloading::Symbol<FnMfxSetConfigFilterProperty> = runtime_lib
                .get(b"MFXSetConfigFilterProperty")
                .context("MFXSetConfigFilterProperty symbol")?;
            let fn_create_session: libloading::Symbol<FnMfxCreateSession> = runtime_lib
                .get(b"MFXCreateSession")
                .context("MFXCreateSession symbol")?;
            let fn_unload: libloading::Symbol<FnMfxUnload> =
                runtime_lib.get(b"MFXUnload").context("MFXUnload symbol")?;
            let mfx_close: libloading::Symbol<FnMfxClose> =
                runtime_lib.get(b"MFXClose").context("MFXClose symbol")?;
            let fn_encode_query: libloading::Symbol<FnEncodeQuery> = runtime_lib
                .get(b"MFXVideoENCODE_Query")
                .context("MFXVideoENCODE_Query")?;
            let fn_encode_init: libloading::Symbol<FnEncodeInit> = runtime_lib
                .get(b"MFXVideoENCODE_Init")
                .context("MFXVideoENCODE_Init")?;
            let fn_encode_close: libloading::Symbol<FnEncodeClose> = runtime_lib
                .get(b"MFXVideoENCODE_Close")
                .context("MFXVideoENCODE_Close")?;
            let fn_encode_frame_async: libloading::Symbol<FnEncodeFrameAsync> = runtime_lib
                .get(b"MFXVideoENCODE_EncodeFrameAsync")
                .context("MFXVideoENCODE_EncodeFrameAsync")?;
            let fn_sync_operation: libloading::Symbol<FnSyncOperation> = runtime_lib
                .get(b"MFXVideoCORE_SyncOperation")
                .context("MFXVideoCORE_SyncOperation")?;

            // 1. Session. oneVPL 2.x dispatcher: load → require a HARDWARE
            //    implementation (selects the gen runtime — the legacy MFXInit
            //    path loads the 1.x MSDK runtime, which has no AV1) → create
            //    the session **on the requested adapter**.
            //
            //    That last part is the whole multi-GPU story. `MFXCreateSession`'s
            //    second argument is an index into the list of implementations
            //    surviving the filters set on the loader, and it used to be
            //    hardcoded to 0 — so every encoder, whichever GPU the scheduler
            //    had leased, was created on the first Intel adapter. On the 3x
            //    Arc box that meant `intel_gpu_top` showed card0 at 73% while
            //    cards 1 and 2 sat at 0.00% on every engine, with the chunk
            //    workers for those GPUs quietly sharing card0's encoder.
            let loader = fn_load();
            if loader.is_null() {
                bail!("MFXLoad returned a null loader (oneVPL dispatcher unavailable)");
            }
            let cfg = fn_create_config(loader);
            if cfg.is_null() {
                fn_unload(loader);
                bail!("MFXCreateConfig returned null");
            }
            let impl_var = MfxVariant {
                version: 0,
                _pad: 0,
                ty: MFX_VARIANT_TYPE_U32,
                data: MFX_IMPL_TYPE_HARDWARE as u64,
            };
            let rc = fn_set_filter(cfg, b"mfxImplDescription.Impl\0".as_ptr(), impl_var);
            if rc < 0 {
                fn_unload(loader);
                bail!("MFXSetConfigFilterProperty(Impl=HARDWARE) failed: {rc}");
            }
            // The implementation list is Intel-adapter-ordered, so the
            // vendor-local index is the one to use — `gpu_index` is global and
            // would overshoot on a host with a non-Intel card ahead of the Arcs.
            let mut session: MfxSession = ptr::null_mut();

            let rc = match gpu_index {
                Some(gpu_index) => {
                    let adapter =
                        crate::gpu::vendor_index_of(gpu_index).unwrap_or(gpu_index);
                    fn_create_session(loader, adapter, &mut session)
                }
                // Legacy path. The dispatcher is still loaded — it owns the
                // library handle and the unload below — but the session comes
                // from `MFXInit`, which is what some hosts answer to.
                None => {
                    let mfx_init: libloading::Symbol<FnMfxInit> =
                        runtime_lib.get(b"MFXInit").context("MFXInit symbol")?;
                    let mut version = crate::qsv_ffi::MfxVersion { minor: 0, major: 1 };
                    mfx_init(MFX_IMPL_HARDWARE_ANY, &mut version, &mut session)
                }
            };

            // No quiet retry on adapter 0.
            //
            // This used to fall back there whenever the requested adapter
            // failed, on the reasoning that a working encode beats a spread
            // one. The cost was hidden and larger than it looks: every rung
            // whose adapter refused landed on the same card while still
            // reporting the one it was leased, so the pool believed the work
            // was spread, `intel_gpu_top` showed one card saturated and the
            // others idle, and the log line admitting it was a warning nobody
            // reads. It also could not help the case that actually happens —
            // adapter 0 itself refusing — because it excluded that index.
            //
            // Failing here instead lets `encode::select_encoder` try the
            // vendor's other cards explicitly. That path knows which device it
            // moved to, says so, and reports the card it really used.
            if rc < 0 || session.is_null() {
                fn_unload(loader);
                match gpu_index {
                    Some(idx) => bail!(
                        "MFXCreateSession failed on adapter {}: {rc} \
                         (no Intel HW implementation for this codec on that card?)",
                        crate::gpu::vendor_index_of(idx).unwrap_or(idx)
                    ),
                    None => bail!(
                        "MFXInit(HARDWARE_ANY) failed: {rc} \
                         (no Intel HW implementation for this codec on any card?)"
                    ),
                }
            }

            // 2. Build the video parameter struct.
            let tp = tuning::qsv_params(
                config.codec,
                config.target,
                config.tier,
                config.width,
                config.height,
            );

            // Squad-22: Pick FOURCC + BitDepth/Shift triple from the
            // configured input format. Both must agree — sending P010
            // in the surface but FrameInfo BitDepthLuma=8 silently
            // truncates samples on the encode side.
            let input_fourcc = qsv_fourcc_for(config.pixel_format)?;
            let (bit_depth_luma, bit_depth_chroma, shift) =
                qsv_bit_depth_triple(config.pixel_format);

            // Allocate ext buffer for AV1 tile grid and keep it in a
            // Box so its address is stable for ExtParam[].
            let mut tile_ext = Box::new(MfxExtAv1TileParam {
                header: MfxExtBuffer {
                    buffer_id: MFX_EXTBUFF_AV1_TILE_PARAM,
                    buffer_sz: std::mem::size_of::<MfxExtAv1TileParam>() as u32,
                },
                num_tile_rows: tp.num_tile_rows as u16,
                num_tile_columns: tp.num_tile_columns as u16,
                num_tile_groups: 1,
                reserved: [0u16; 5],
            });

            // mfxExtCodingOption3 — only attached for 10-bit jobs. The
            // 8-bit path leaves `TargetBitDepthLuma` at the runtime
            // default (which mirrors FrameInfo.BitDepthLuma) so we
            // don't ship redundant bytes.
            let mut coding_option3_ext: Option<Box<MfxExtCodingOption3>> =
                if config.pixel_format == PixelFormat::Yuv420p10le {
                    Some(Box::new(MfxExtCodingOption3 {
                        header: MfxExtBuffer {
                            buffer_id: MFX_EXTBUFF_CODING_OPTION3,
                            buffer_sz: std::mem::size_of::<MfxExtCodingOption3>() as u32,
                        },
                        _pad_to_158: [0; 150],
                        target_chroma_format_plus1: MFX_TARGET_CHROMAFORMAT_YUV420_PLUS1,
                        target_bit_depth_luma: 10,
                        target_bit_depth_chroma: 10,
                        _tail: [0; 348],
                    }))
                } else {
                    None
                };

            // mfxExtVideoSignalInfo — always attached so the AV1 OBU
            // sequence header carries explicit colour codes (rather
            // than the "unspecified" default that some downstream
            // tooling silently re-interprets).
            let cm = &config.color_metadata;
            let mut signal_info_ext = Box::new(MfxExtVideoSignalInfo {
                header: MfxExtBuffer {
                    buffer_id: MFX_EXTBUFF_VIDEO_SIGNAL_INFO,
                    buffer_sz: std::mem::size_of::<MfxExtVideoSignalInfo>() as u32,
                },
                video_format: 5,                     // unspecified format
                video_full_range: if cm.full_range { 1 } else { 0 },
                colour_description_present: 1,
                colour_primaries: cm.colour_primaries as u16,
                transfer_characteristics: transfer_to_h273(cm.transfer),
                matrix_coefficients: cm.matrix_coefficients as u16,
            });

            // Build the ExtParam[] vector. Tile + signal info always;
            // coding_option3 only when 10-bit. Keeping the slot order
            // deterministic (tile, signal_info, [co3]) means tests can
            // assert on it.
            //
            // We collect raw `*mut` directly off each `Box`'s heap
            // address — `Box::as_mut` for a `&mut Box<T>` gives a
            // stable pointer that lives as long as the Box itself
            // stays alive in `QsvSession`. The Vec<> backing the
            // ExtParam[] is also stashed on `QsvSession` so the array
            // address handed to oneVPL stays valid until session drop.
            let mut ext_param_array: Vec<*mut MfxExtBuffer> = Vec::with_capacity(3);
            // The AV1 tile-param ext buffer is codec-specific — H.264 / H.265
            // Query/Init reject an unknown ext buffer, so attach it for AV1 only.
            if config.codec == crate::frame::VideoCodec::Av1 {
                ext_param_array.push(
                    (&mut *tile_ext as *mut MfxExtAv1TileParam) as *mut MfxExtBuffer,
                );
            }
            ext_param_array.push(
                (&mut *signal_info_ext as *mut MfxExtVideoSignalInfo) as *mut MfxExtBuffer,
            );
            if let Some(ref mut co3) = coding_option3_ext {
                ext_param_array.push(
                    (&mut **co3 as *mut MfxExtCodingOption3) as *mut MfxExtBuffer,
                );
            }
            let num_ext_param = ext_param_array.len() as u16;

            // Per-frame quality knobs. `config.quality` is the caller's CRF
            // escape hatch (`AUTO_FROM_TARGET` = derive from `config.target`);
            // it has to be honoured in **both** rate-control modes, not just
            // CQP, or `--crf` silently does nothing on the common ICQ path.
            //
            // CRF is read on the codec's own native quality scale — 0..51 for
            // H.264/HEVC (the familiar x264/x265 CRF range) and 0..63 for AV1
            // (the libaom/SVT cq-level range) — and converted to whatever the
            // target rate-control slot wants. See `crf_to_icq` / `crf_to_qp`.
            let force_cqp = config.constant_qp || tp.rc_mode == QsvRateControl::Cqp;
            let rc_effective = if force_cqp {
                QsvRateControl::Cqp
            } else {
                QsvRateControl::Icq
            };
            let (rc_mode_u16, qp_i_effective, qp_p_effective, icq_effective) = match rc_effective {
                QsvRateControl::Cqp => {
                    let (qp_i, qp_p) = if config.quality == AUTO_FROM_TARGET {
                        (tp.qp_i, tp.qp_p)
                    } else {
                        let qp = crf_to_qp(config.codec, config.quality);
                        (qp, inter_qp(config.codec, qp))
                    };
                    (MFX_RATECONTROL_CQP, qp_i, qp_p, 0u16)
                }
                QsvRateControl::Icq => {
                    let icq = if config.quality == AUTO_FROM_TARGET {
                        tp.icq_quality
                    } else {
                        crf_to_icq(config.codec, config.quality)
                    };
                    (MFX_RATECONTROL_ICQ, 0u16, 0u16, icq)
                }
            };

            // Must be the *effective* mode, not `tp.rc_mode`: when
            // `constant_qp` overrides the table's ICQ suggestion, laying the
            // slots out for ICQ writes the QP into a field CQP never reads,
            // so the driver silently falls back to its own default.
            let slots = rate_slots_for_rc(
                rc_effective,
                qp_i_effective,
                qp_p_effective,
                icq_effective,
            );

            // Assemble MfxFrameInfo. vendor/intel/mfxstructs.h:20-50.
            // Squad-22: bit_depth_luma/chroma + shift + fourcc come from
            // the dispatched (input_fourcc, bit_depth_luma, bit_depth_chroma,
            // shift) tuple. NV12: (8,8,0). P010: (10,10,1) — Shift=1 is
            // mandatory or oneVPL rejects with INVALID_VIDEO_PARAM.
            let frame_info = MfxFrameInfo {
                reserved: [0; 4],
                channel_id: 0,
                bit_depth_luma,
                bit_depth_chroma,
                shift,
                frame_id: [0; 4],
                fourcc: input_fourcc,
                width: align_up(config.width as u16, 16),
                height: align_up(config.height as u16, 16),
                crop_x: 0,
                crop_y: 0,
                crop_w: config.width as u16,
                crop_h: config.height as u16,
                frame_rate_ext_n: (config.frame_rate * 1000.0).round() as u32,
                frame_rate_ext_d: 1000,
                reserved3: 0,
                aspect_ratio_w: 1,
                aspect_ratio_h: 1,
                pic_struct: MFX_PICSTRUCT_PROGRESSIVE,
                chroma_format: MFX_CHROMAFORMAT_YUV420,
                reserved2: 0,
            };

            // oneVPL `mfxInfoMFX` unions all three rc arms into the
            // same three u16 slots, **but the per-arm field layout
            // differs** per vendor/intel/mfxstructs.h:74-89:
            //   slot 0 → InitialDelayInKB (CBR/VBR) / QPI (CQP) / Accuracy (AVBR)
            //   slot 1 → TargetKbps (CBR/VBR) / QPP (CQP) / **ICQQuality (ICQ)**
            //   slot 2 → MaxKbps (CBR/VBR) / QPB (CQP) / Convergence (AVBR)
            //
            // Two notable consequences:
            //   1. For CQP: QPI→slot0, QPP→slot1, QPB→slot2. Natural.
            //   2. For ICQ: ICQQuality must go into **slot 1**, not
            //      slot 0. Slot 0 aliases InitialDelayInKB which the
            //      runtime doesn't read in ICQ mode.
            //
            // An earlier rev of this code (based on `codec-review-59-60.md`
            // §QSV-1's misread of the upstream union — the reviewer
            // cited a legacy Windows SDK layout where the ICQ arm was a
            // separate `struct {mfxU16 ICQQuality, reserved8[4]}` at
            // slot 0; in the Linux oneVPL 2.10 header we ship, the arm
            // is unified with `TargetKbps/QPP/ICQQuality` at slot 1)
            // placed ICQQuality in slot 0, silently falling back to
            // driver default 23 for every quality tier. `rate_slots_for_rc`
            // above puts the value in the correct slot per the
            // vendored header.

            let (codec_id, codec_profile) = qsv_codec_ids(config.codec, config.pixel_format);
            let mfx = MfxInfoMfx {
                reserved: [0; 7],
                // LowPower from the tuning adapter. AV1 QSV encode is VDENC
                // (low-power) on Arc / Meteor Lake+ — the only AV1 encode entry
                // point the iHD driver exposes — so this must be ON, else Query
                // rejects with MFX_ERR_UNSUPPORTED.
                low_power: tp.low_power,
                brc_param_multiplier: 0,
                frame_info,
                codec_id,
                codec_profile,
                codec_level: 0, // auto-level
                num_thread: 0,
                target_usage: clamp_target_usage(tp.target_usage),
                gop_pic_size: config.keyframe_interval as u16,
                gop_ref_dist: 1, // no B-frames
                gop_opt_flag: 0,
                idr_interval: 0,
                rate_control_method: rc_mode_u16,
                qpi_or_delay: slots.slot0_qpi_or_delay,
                buffer_size_kb: 0,
                qpp_or_kbps_or_icq: slots.slot1_qpp_or_kbps_or_icq,
                qpb_or_maxkbps: slots.slot2_qpb_or_maxkbps,
                num_slice: 0,
                num_ref_frame: 1,
                encoded_order: 0,
            };

            let mut par = MfxVideoParam {
                alloc_id: 0,
                reserved: [0; 2],
                reserved3: 0,
                // AsyncDepth matches the 4-deep ring — tells the
                // encoder it may receive up to RING_SIZE submissions
                // without a sync in between.
                async_depth: RING_SIZE as u16,
                mfx,
                _mfx_union_pad: [0; 32],
                protected: 0,
                io_pattern: MFX_IOPATTERN_IN_SYSTEM_MEMORY,
                ext_param: ext_param_array.as_ptr() as *mut *mut MfxExtBuffer,
                num_ext_param,
                reserved2: 0,
            };

            // 3. Query — lets the runtime validate and suggest
            //    adjustments for any unsupported knobs. We read `out`
            //    and selectively copy back the fields the runtime
            //    populated — `out` is zero-initialised so we can use
            //    nonzero-ness as a "runtime touched this" signal.
            //
            //    systems-review-59-60 M-Q1: when Query rewrote params
            //    we must Init against the adjusted values, not the
            //    originals.
            let mut out = zeroed_video_param();
            let rc = (*fn_encode_query)(session, &mut par, &mut out);
            let rewrote = match rc {
                MFX_ERR_NONE => false,
                MFX_WRN_INCOMPATIBLE_VIDEO_PARAM | MFX_WRN_VIDEO_PARAM_CHANGED => {
                    // Driver rewrote something — surface the deltas so
                    // ops can correlate quality shifts with driver
                    // behaviour. `out` holds the runtime-adjusted
                    // values; `par` still holds our requested values.
                    tracing::warn!(
                        status = rc,
                        req_rc_method = par.mfx.rate_control_method,
                        got_rc_method = out.mfx.rate_control_method,
                        req_target_usage = par.mfx.target_usage,
                        got_target_usage = out.mfx.target_usage,
                        req_qpi_or_delay = par.mfx.qpi_or_delay,
                        got_qpi_or_delay = out.mfx.qpi_or_delay,
                        req_qpp_or_kbps_or_icq = par.mfx.qpp_or_kbps_or_icq,
                        got_qpp_or_kbps_or_icq = out.mfx.qpp_or_kbps_or_icq,
                        req_profile = par.mfx.codec_profile,
                        got_profile = out.mfx.codec_profile,
                        req_width = par.mfx.frame_info.width,
                        got_width = out.mfx.frame_info.width,
                        req_height = par.mfx.frame_info.height,
                        got_height = out.mfx.frame_info.height,
                        "QSV Query rewrote encoder parameters"
                    );
                    true
                }
                MFX_WRN_PARTIAL_ACCELERATION => {
                    tracing::warn!(
                        "QSV runtime reports partial acceleration — \
                         some encoder stages may fall back to CPU"
                    );
                    false
                }
                err => {
                    // MFXVideoENCODE_Query is ADVISORY, not authoritative — on the
                    // iHD AV1 implementation it returns MFX_ERR_UNSUPPORTED (-3)
                    // even for a param that MFXVideoENCODE_Init then accepts
                    // (verified in C against the real oneVPL headers: Query=-3 but
                    // Init=0 for the same param). So we do NOT bail here; we log
                    // and proceed to Init with our requested params. If the config
                    // is truly unsupported, Init fails and we bail there.
                    // Once per distinct param set, not once per encoder: the
                    // chunk worker builds ~1300 of these on a feature-length
                    // file, and 1300 identical warnings drown everything else.
                    if first_query_rejection(
                        par.mfx.codec_id,
                        par.mfx.rate_control_method,
                        par.mfx.low_power,
                    ) {
                        tracing::warn!(
                            status = err,
                            codec = par.mfx.codec_id,
                            rate_control = par.mfx.rate_control_method,
                            low_power = par.mfx.low_power,
                            "MFXVideoENCODE_Query rejected these params; proceeding to Init, \
                             which is authoritative on this runtime. Reported once per param \
                             set — a genuine incompatibility fails at Init with a hard error."
                        );
                    } else {
                        tracing::debug!(status = err, "MFXVideoENCODE_Query error (already reported)");
                    }
                    false
                }
            };

            // systems-review-59-60 M-Q1: when Query rewrote params we
            // must Init against the adjusted values, not the originals.
            // `out.mfx` carries only the fields the driver touched
            // (everything else is zero-initialised in `out`), so we
            // copy selectively — keep our base struct for fields the
            // driver didn't rewrite, overwrite the rest.
            if rewrote {
                // frame_info dimensions may have been clamped to
                // hardware limits; everything downstream (surface
                // allocation below) reads from these fields, so we
                // pick them up.
                if out.mfx.frame_info.width != 0 {
                    par.mfx.frame_info.width = out.mfx.frame_info.width;
                }
                if out.mfx.frame_info.height != 0 {
                    par.mfx.frame_info.height = out.mfx.frame_info.height;
                }
                if out.mfx.frame_info.fourcc != 0 {
                    par.mfx.frame_info.fourcc = out.mfx.frame_info.fourcc;
                }
                if out.mfx.frame_info.chroma_format != 0 {
                    par.mfx.frame_info.chroma_format = out.mfx.frame_info.chroma_format;
                }
                if out.mfx.rate_control_method != 0 {
                    par.mfx.rate_control_method = out.mfx.rate_control_method;
                }
                if out.mfx.target_usage != 0 {
                    par.mfx.target_usage = out.mfx.target_usage;
                }
                if out.mfx.codec_profile != 0 {
                    par.mfx.codec_profile = out.mfx.codec_profile;
                }
                if out.mfx.codec_level != 0 {
                    par.mfx.codec_level = out.mfx.codec_level;
                }
                // Note: qpi_or_delay / qpp_or_kbps_or_icq / qpb_or_maxkbps
                // are deliberately left as-requested unless the driver
                // explicitly returned zero-for-adjusted; 0 is a valid
                // ICQ-slot value ("not set") so we keep ours.
            }

            // Re-attach our ext param list (Query zeroes it out on
            // some runtime versions).
            par.ext_param = ext_param_array.as_ptr() as *mut *mut MfxExtBuffer;
            par.num_ext_param = num_ext_param;

            // 4. Init.
            let rc = (*fn_encode_init)(session, &mut par);
            if rc < 0 {
                let _ = mfx_close(session);
                bail!(
                    "MFXVideoENCODE_Init failed: {rc} (likely the AV1 encode component \
                     is not available — Arc / Meteor Lake + required)"
                );
            } else if rc > 0 {
                tracing::warn!(
                    status = rc,
                    "MFXVideoENCODE_Init returned a warning; encoder will run with \
                     adjusted parameters"
                );
            }

            tracing::debug!(
                width = config.width,
                height = config.height,
                target = ?config.target,
                tier = ?config.tier,
                rc_mode = ?tp.rc_mode,
                icq_quality = tp.icq_quality,
                qp_i = tp.qp_i,
                target_usage = tp.target_usage,
                tile_cols = tp.num_tile_columns,
                tile_rows = tp.num_tile_rows,
                codec = ?config.codec,
                "QSV tuning applied"
            );

            // 5. Pre-allocate input surfaces + bitstream buffer. NV12:
            //    Y plane (pitch × height) + UV plane (pitch × height/2)
            //    at the surface's aligned width.
            //
            // Squad-22: P010 surfaces double per-sample byte width.
            // `bytes_per_sample` is 1 for NV12, 2 for P010. `pitch` is
            // expressed in **bytes** (Y row width = width × bytes_per_sample,
            // aligned to 64 bytes for Arc DMA). The total payload still
            // works out as `pitch_bytes × h_aligned × 3 / 2` because
            // 4:2:0 chroma = half height with the same pitch.
            let bytes_per_sample: u32 = if shift == 1 { 2 } else { 1 };
            let pitch = align_up(config.width * bytes_per_sample, 64u32); // bytes
            let h_aligned = align_up(config.height, 16u32);
            let surface_bytes = (pitch as usize * h_aligned as usize * 3) / 2;

            // Ring of N=4 surfaces. Allocate each slot's backing
            // buffer up-front so the surface pointers are stable for
            // the session's lifetime.
            let mut surfaces_vec: Vec<SurfaceSlot> = Vec::with_capacity(RING_SIZE);
            let y_plane_bytes = pitch as usize * h_aligned as usize;
            for _ in 0..RING_SIZE {
                // Pre-fill the NV12 scratch with neutral black (Y=16, Cb/Cr=128
                // for 8-bit BT.709 limited; the 10-bit equivalents <<6). AV1
                // requires 16-multiple coded dims, so e.g. 572x240 encodes at
                // 576x240 and 1080 at 1088 — the padding rows/cols that the
                // per-frame upload never touches would otherwise be 0, which a
                // browser decodes through BT.709 as the distinctive GREEN bars.
                let (y_fill, c_fill): (u8, u8) = (16, 128);
                let mut backing: Box<[u8]> = if config.pixel_format == PixelFormat::Yuv420p10le {
                    // P010: 16<<6 and 128<<6 as LE u16.
                    let mut v = vec![0u8; surface_bytes].into_boxed_slice();
                    let (yb, cb) = ((16u16 << 6).to_le_bytes(), (128u16 << 6).to_le_bytes());
                    for i in (0..y_plane_bytes).step_by(2) {
                        v[i] = yb[0];
                        v[i + 1] = yb[1];
                    }
                    for i in (y_plane_bytes..surface_bytes).step_by(2) {
                        v[i] = cb[0];
                        v[i + 1] = cb[1];
                    }
                    v
                } else {
                    let mut v = vec![c_fill; surface_bytes].into_boxed_slice();
                    v[..y_plane_bytes].fill(y_fill);
                    v
                };
                let y_ptr = backing.as_mut_ptr();
                let uv_ptr = y_ptr.add(pitch as usize * h_aligned as usize);
                let surface = MfxFrameSurface1 {
                    reserved: [0; 4],
                    info: frame_info,
                    data: MfxFrameData {
                        ext_param_or_reserved2: 0,
                        num_ext_param: 0,
                        reserved: [0; 9],
                        mem_type: 0,
                        pitch_high: (pitch >> 16) as u16,
                        time_stamp: 0,
                        frame_order: 0,
                        locked: 0,
                        pitch: (pitch & 0xFFFF) as u16,
                        y: y_ptr,
                        // NV12: U pointer is the start of the UV plane,
                        // V pointer is U + 1. Upstream sample_encode
                        // uses this convention.
                        u: uv_ptr,
                        v: uv_ptr.add(1),
                        a: ptr::null_mut(),
                        mem_id: ptr::null_mut(),
                        corrupted: 0,
                        data_flag: 0,
                    },
                };
                let cap = surface_bytes.max(2 * 1024 * 1024);
                let mut bs_buf: Box<[u8]> = vec![0u8; cap].into_boxed_slice();
                let bs = MfxBitstream {
                    reserved: [0; 6],
                    decode_time_stamp: 0,
                    time_stamp: 0,
                    data: bs_buf.as_mut_ptr(),
                    data_offset: 0,
                    data_length: 0,
                    max_length: cap as u32,
                    pic_struct: MFX_PICSTRUCT_PROGRESSIVE,
                    frame_type: 0,
                    data_flag: 0,
                    reserved2: 0,
                };
                surfaces_vec.push(SurfaceSlot {
                    surface,
                    _backing: backing,
                    sync: ptr::null_mut(),
                    bitstream: bs,
                    bitstream_buf: bs_buf,
                    ctrl: MfxEncodeCtrl::force_idr(),
                });
            }
            let surfaces: [SurfaceSlot; RING_SIZE] = surfaces_vec
                .try_into()
                .map_err(|_| anyhow::anyhow!("RING_SIZE mismatch during surface allocation"))?;

            // Session-level bitstream, used only by the flush path's NULL
            // submissions. Per-frame output goes to the submitting slot's own
            // bitstream — see `SurfaceSlot::bitstream`.
            // Size the output bitstream buffer to the raw frame size (an encoded
            // AV1 frame is always smaller than raw), floored at 2 MiB. A fixed
            // 2 MiB overflowed a 1080p IDR → MFX_ERR_NOT_ENOUGH_BUFFER (-5).
            let bitstream_capacity = surface_bytes.max(2 * 1024 * 1024);
            debug_assert!(bitstream_capacity <= MAX_BITSTREAM_BYTES);
            let mut bitstream_buf: Box<[u8]> = vec![0u8; bitstream_capacity].into_boxed_slice();
            let bitstream = MfxBitstream {
                reserved: [0; 6],
                decode_time_stamp: 0,
                time_stamp: 0,
                data: bitstream_buf.as_mut_ptr(),
                data_offset: 0,
                data_length: 0,
                max_length: bitstream_buf.len() as u32,
                pic_struct: MFX_PICSTRUCT_PROGRESSIVE,
                frame_type: 0,
                data_flag: 0,
                reserved2: 0,
            };

            let sess = QsvSession {
                session,
                width: config.width,
                height: config.height,
                pts_timescale: (10_000_000.0f64 / config.frame_rate).round() as u64,
                input_pixel_format: config.pixel_format,
                fn_mfx_close: *mfx_close,
                fn_encode_close: *fn_encode_close,
                fn_encode_frame_async: *fn_encode_frame_async,
                fn_sync_operation: *fn_sync_operation,
                loader,
                fn_unload: *fn_unload,
                tile_ext,
                coding_option3_ext,
                signal_info_ext,
                ext_param_array,
                surfaces,
                ring_idx: 0,
                inflight: VecDeque::with_capacity(RING_SIZE),
                input_pitch: pitch,
                height_aligned: h_aligned,
                bitstream,
                _bitstream_buf: bitstream_buf,
            };

            tracing::debug!(
                width = config.width,
                height = config.height,
                gpu = gpu_index,
                ring_size = RING_SIZE,
                codec = ?config.codec,
                "QSV encoder ready"
            );

            // Silence a handful of constants that only appear in
            // deferred paths (future dispatcher probe, extra ext
            // buffer, cross-platform char alias).
            let _ = (MFX_EXTBUFF_AV1_BITSTREAM_PARAM, 0 as c_char);

            Ok(Self {
                config,
                session: Some(sess),
                encoded_packets: Vec::new(),
                packet_cursor: 0,
                flushed: false,
                frame_counter: 0,
                force_idr_next: false,
                _runtime_lib: runtime_lib,
            })
        }
    }

    fn encode_one(&mut self, frame: &VideoFrame) -> Result<()> {
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("encode_one called after session drop"))?;

        if frame.format != session.input_pixel_format {
            bail!(
                "QSV session was initialized with {:?} but frame is {:?} \
                 — pipeline must reinit the encoder if pixel format changes",
                session.input_pixel_format,
                frame.format
            );
        }

        let w = session.width as usize;
        let h = session.height as usize;
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);

        // Per-pixel byte width: 1 for 8-bit YUV420p, 2 for Yuv420p10le.
        // Drives both the source buffer-size check and the per-row
        // copy width on the upload path below.
        let bytes_per_sample: usize = if session.input_pixel_format
            == PixelFormat::Yuv420p10le
        {
            2
        } else {
            1
        };
        let y_size_bytes = w * h * bytes_per_sample;
        let uv_size_bytes = cw * ch * bytes_per_sample;

        if frame.data.len() < y_size_bytes + 2 * uv_size_bytes {
            bail!(
                "frame data too small for {}x{} {:?}: need {} bytes, got {}",
                w,
                h,
                session.input_pixel_format,
                y_size_bytes + 2 * uv_size_bytes,
                frame.data.len()
            );
        }

        let pitch = session.input_pitch as usize;
        let h_aligned = session.height_aligned as usize;

        // Pick the next ring slot. If it's still waiting on a sync,
        // drain it first — the ring is full.
        let slot_idx = session.ring_idx;
        // Drain FIFO until *this* slot is free — `while`, not `if`.
        //
        // The FIFO head is only the same slot as `ring_idx` while the two stay
        // in lockstep, and they don't: `MFX_ERR_MORE_DATA` and
        // `MFX_WRN_IN_EXECUTION` advance the ring without pushing to
        // `inflight`, so the two indices drift apart. Draining a single
        // arbitrary oldest entry then left `slot_idx` still pending, and the
        // code below overwrote it — dropping that submission's sync point, and
        // with it a frame. It cost 3 of 480 frames on a long chunk, silently,
        // because nothing downstream checked packet count against frame count.
        while !session.surfaces[slot_idx].sync.is_null() {
            let oldest = session
                .inflight
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("ring full but inflight queue empty"))?;
            let sync = session.surfaces[oldest].sync;
            session.surfaces[oldest].sync = ptr::null_mut();
            if !sync.is_null() {
                let (f, s) = (session.fn_sync_operation, session.session);
                unsafe {
                    sync_and_drain_bs(
                        f,
                        s,
                        &mut session.surfaces[oldest].bitstream,
                        sync,
                        &mut self.encoded_packets,
                    )?;
                }
            }
        }

        let slot = &mut session.surfaces[slot_idx];

        unsafe {
            let y_dst = slot.surface.data.y;
            // UV plane sits one Y plane down: pitch (bytes) × h_aligned (rows).
            let uv_dst = y_dst.add(pitch * h_aligned);

            if session.input_pixel_format == PixelFormat::Yuv420p10le {
                // ── 10-bit P010 upload ──────────────────────────────
                // Source: Yuv420p10le — planar Y/U/V, valid 10 bits in
                // lower 10 of each u16 LE word.
                // Destination: P010 — planar Y + interleaved UV, valid
                // 10 bits in **upper 10** of each u16 LE word
                // (`sample << 6`). pitch is in bytes.
                let src_ptr = frame.data.as_ptr();

                // Y plane.
                for row in 0..h {
                    let src_row = src_ptr.add(row * w * 2) as *const u16;
                    let dst_row = y_dst.add(row * pitch) as *mut u16;
                    for col in 0..w {
                        let sample = (*src_row.add(col)) & 0x03FF;
                        *dst_row.add(col) = sample << 6;
                    }
                }

                // UV plane: interleave U + V into the chroma plane,
                // both shifted by 6 to satisfy P010's upper-10-bit
                // convention.
                let u_src_base = src_ptr.add(y_size_bytes);
                let v_src_base = u_src_base.add(uv_size_bytes);
                for row in 0..ch {
                    let u_src = u_src_base.add(row * cw * 2) as *const u16;
                    let v_src = v_src_base.add(row * cw * 2) as *const u16;
                    let dst_row = uv_dst.add(row * pitch) as *mut u16;
                    for col in 0..cw {
                        let u = (*u_src.add(col)) & 0x03FF;
                        let v = (*v_src.add(col)) & 0x03FF;
                        *dst_row.add(col * 2) = u << 6;
                        *dst_row.add(col * 2 + 1) = v << 6;
                    }
                }
            } else {
                // ── 8-bit NV12 upload ───────────────────────────────
                // Source: YUV420p — planar Y/U/V at 1 byte/sample.
                // Destination: NV12 — planar Y + interleaved UV.
                // Copy Y.
                for row in 0..h {
                    let src = frame.data.as_ptr().add(row * w);
                    let dst = y_dst.add(row * pitch);
                    ptr::copy_nonoverlapping(src, dst, w);
                }

                // Interleave YUV420p U + V into NV12 UV plane.
                let u_src_base = frame.data.as_ptr().add(y_size_bytes);
                let v_src_base = u_src_base.add(uv_size_bytes);
                for row in 0..ch {
                    let u_src = u_src_base.add(row * cw);
                    let v_src = v_src_base.add(row * cw);
                    let dst_row = uv_dst.add(row * pitch);
                    for col in 0..cw {
                        *dst_row.add(col * 2) = *u_src.add(col);
                        *dst_row.add(col * 2 + 1) = *v_src.add(col);
                    }
                }
            }
        }

        slot.surface.data.time_stamp = frame.pts * session.pts_timescale;
        slot.surface.data.frame_order = self.frame_counter;

        // Wrap in catch_unwind so panics during FFI don't unwind
        // across the C ABI boundary.
        let packets = &mut self.encoded_packets;
        // The control must outlive the *async* submit — the runtime reads it
        // after `EncodeFrameAsync` returns — so it lives in the ring slot,
        // alongside the surface backing store, and stays valid until this
        // slot's sync point is drained. A stack local here silently did
        // nothing: by the time the runtime looked, the frame was gone.
        let ctrl_ptr: *mut std::ffi::c_void = if self.force_idr_next {
            session.surfaces[slot_idx].ctrl = MfxEncodeCtrl::force_idr();
            &mut session.surfaces[slot_idx].ctrl as *mut MfxEncodeCtrl as *mut std::ffi::c_void
        } else {
            ptr::null_mut()
        };
        self.force_idr_next = false;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            let mut sync: MfxSyncPoint = ptr::null_mut();
            let mut rc = (session.fn_encode_frame_async)(
                session.session,
                ctrl_ptr,
                &mut session.surfaces[slot_idx].surface as *mut MfxFrameSurface1,
                &mut session.surfaces[slot_idx].bitstream as *mut MfxBitstream,
                &mut sync,
            );
            // The output bitstream is shared across the whole ring, so a run of
            // large frames can fill it before their sync points are drained.
            // Recoverable, and it has to be recovered from: a long single-encoder
            // run (`--seam-mode serial`) died on this at 1080p around frame 420,
            // while the chunked path survived only because each chunk gets a
            // fresh buffer every 48 frames.
            //
            // Draining everything in flight empties the buffer; if one frame
            // genuinely doesn't fit, grow it. Both are safe here — after the
            // drain the runtime holds no reference to the old allocation.
            if rc == MFX_ERR_NOT_ENOUGH_BUFFER {
                while let Some(oldest) = session.inflight.pop_front() {
                    let s = session.surfaces[oldest].sync;
                    session.surfaces[oldest].sync = ptr::null_mut();
                    if !s.is_null() {
                        let (f, sh) = (session.fn_sync_operation, session.session);
                        sync_and_drain_bs(
                            f,
                            sh,
                            &mut session.surfaces[oldest].bitstream,
                            s,
                            packets,
                        )?;
                    }
                }
                rc = (session.fn_encode_frame_async)(
                    session.session,
                    ctrl_ptr,
                    &mut session.surfaces[slot_idx].surface as *mut MfxFrameSurface1,
                    &mut session.surfaces[slot_idx].bitstream as *mut MfxBitstream,
                    &mut sync,
                );
                if rc == MFX_ERR_NOT_ENOUGH_BUFFER {
                    let old = session.surfaces[slot_idx].bitstream_buf.len();
                    let new = (old * 2).min(MAX_BITSTREAM_BYTES);
                    if new <= old {
                        bail!(
                            "MFXVideoENCODE_EncodeFrameAsync: one frame exceeds the {} MiB \
                             bitstream ceiling at {w}x{h}",
                            MAX_BITSTREAM_BYTES / (1024 * 1024)
                        );
                    }
                    tracing::debug!(old, new, "growing the QSV output bitstream buffer");
                    let mut buf: Box<[u8]> = vec![0u8; new].into_boxed_slice();
                    let slot = &mut session.surfaces[slot_idx];
                    slot.bitstream.data = buf.as_mut_ptr();
                    slot.bitstream.max_length = new as u32;
                    slot.bitstream.data_offset = 0;
                    slot.bitstream.data_length = 0;
                    slot.bitstream_buf = buf;
                    rc = (session.fn_encode_frame_async)(
                        session.session,
                        ctrl_ptr,
                        &mut session.surfaces[slot_idx].surface as *mut MfxFrameSurface1,
                        &mut session.surfaces[slot_idx].bitstream as *mut MfxBitstream,
                        &mut sync,
                    );
                }
            }
            match rc {
                MFX_ERR_NONE => {
                    // Submission accepted — sync point is ours to sync
                    // later. Record it on the slot and queue the slot
                    // for draining.
                    session.surfaces[slot_idx].sync = sync;
                    session.inflight.push_back(slot_idx);
                }
                MFX_ERR_MORE_DATA => {
                    // Encoder wants more frames before emitting — normal
                    // at startup. Slot is consumed (driver copied
                    // internally) but no sync point is produced.
                }
                MFX_WRN_IN_EXECUTION => {
                    // Busy — the runtime is still processing a prior
                    // submission. Yield once and, if a sync point came
                    // back with the warning, drain it immediately so
                    // this slot is clean for the next call.
                    std::thread::yield_now();
                    if !sync.is_null() {
                        // Do NOT stash `sync` on the slot — we're
                        // draining it right here, so the slot must
                        // stay marked as not-pending.
                        let (f, sh) = (session.fn_sync_operation, session.session);
                        sync_and_drain_bs(
                            f,
                            sh,
                            &mut session.surfaces[slot_idx].bitstream,
                            sync,
                            packets,
                        )?;
                    }
                }
                err => {
                    tracing::error!(
                        status = err,
                        w,
                        h,
                        pitch,
                        h_aligned,
                        "MFXVideoENCODE_EncodeFrameAsync failed"
                    );
                    bail!("MFXVideoENCODE_EncodeFrameAsync failed: {err}");
                }
            }
            Ok::<(), anyhow::Error>(())
        }));

        // Ring advance is unconditional — the slot is consumed whether
        // or not the encoder emitted a sync point.
        session.ring_idx = (session.ring_idx + 1) % RING_SIZE;
        self.frame_counter += 1;

        match result {
            Ok(inner) => inner,
            Err(_) => bail!("panic in QSV encode path — aborting rather than unwinding across FFI"),
        }
    }

    fn flush_drain(&mut self) -> Result<()> {
        if self.session.is_none() {
            return Ok(());
        }
        let packets_ref = &mut self.encoded_packets;
        let session_ref = self.session.as_mut().expect("checked Some above");

        // Wrap the whole FFI path in catch_unwind — sync_and_drain
        // calls `Bytes::copy_from_slice` which allocates, and an
        // allocation panic unwinding across the oneVPL C ABI at
        // EncodeFrameAsync is UB in debug builds. systems-review-59-60.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            // Drain any already-submitted-but-unsynced slots first.
            while let Some(slot_idx) = session_ref.inflight.pop_front() {
                let sync = session_ref.surfaces[slot_idx].sync;
                session_ref.surfaces[slot_idx].sync = ptr::null_mut();
                if !sync.is_null() {
                    let (f, s) = (session_ref.fn_sync_operation, session_ref.session);
                    sync_and_drain_bs(
                        f,
                        s,
                        &mut session_ref.surfaces[slot_idx].bitstream,
                        sync,
                        packets_ref,
                    )?;
                }
            }

            // Then submit NULL surfaces to flush anything the encoder
            // has buffered internally (GOP lookahead, altref etc.).
            // MFX_ERR_MORE_DATA on NULL input is the EOF signal.
            loop {
                let mut sync: MfxSyncPoint = ptr::null_mut();
                let rc = (session_ref.fn_encode_frame_async)(
                    session_ref.session,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    &mut session_ref.bitstream as *mut MfxBitstream,
                    &mut sync,
                );
                match rc {
                    MFX_ERR_NONE => {
                        if !sync.is_null() {
                            let (f, s) = (session_ref.fn_sync_operation, session_ref.session);
                            sync_and_drain_bs(
                                f,
                                s,
                                &mut session_ref.bitstream,
                                sync,
                                packets_ref,
                            )?;
                        }
                    }
                    MFX_ERR_MORE_DATA => return Ok::<(), anyhow::Error>(()),
                    err if err > 0 => {
                        // Warning — continue.
                        if !sync.is_null() {
                            let (f, s) = (session_ref.fn_sync_operation, session_ref.session);
                            sync_and_drain_bs(
                                f,
                                s,
                                &mut session_ref.bitstream,
                                sync,
                                packets_ref,
                            )?;
                        }
                    }
                    err => bail!("MFXVideoENCODE_EncodeFrameAsync(flush) failed: {err}"),
                }
            }
        }));
        match result {
            Ok(inner) => inner,
            Err(_panic) => bail!(
                "panic in QSV flush path — aborting rather than unwinding across FFI"
            ),
        }
    }
}

/// Wait for the in-flight sync point and copy the bitstream into
/// an `EncodedPacket`. Resets the bitstream buffer for reuse.
///
/// Free function (not a method on `QsvEncoder`) so the caller can hold
/// `&mut session` and `&mut packets` simultaneously without fighting
/// the borrow checker — mirrors the pattern Squad 5 used for AMF's
/// `drain_until_hungry_raw` and the task #60 follow-up review's
/// recommended shape.
unsafe fn sync_and_drain_bs(
    fn_sync: FnSyncOperation,
    sess: MfxSession,
    session_bs: &mut MfxBitstream,
    sync: MfxSyncPoint,
    packets: &mut Vec<EncodedPacket>,
) -> Result<()> {
    // Aliased so the body below reads the same as before the split.
    let session = Bs { bitstream: session_bs };
    struct Bs<'a> { bitstream: &'a mut MfxBitstream }
    unsafe {
        let rc = fn_sync(sess, sync, 60_000);
        if rc != MFX_ERR_NONE {
            bail!("MFXVideoCORE_SyncOperation failed: {rc}");
        }

        let len = session.bitstream.data_length as usize;
        if len == 0 {
            return Ok(());
        }
        let offset = session.bitstream.data_offset as usize;
        let slice = std::slice::from_raw_parts(
            session.bitstream.data.add(offset),
            len,
        );
        let data_bytes = Bytes::copy_from_slice(slice);
        // For AV1, oneVPL sets `MFX_FRAMETYPE_I` on key frames and
        // keeps `MFX_FRAMETYPE_IDR` unused (that flag is an H.264
        // concept). AV1 also has an INTRA_ONLY frame type that is a
        // valid random-access point but not mapped to a named
        // `MFX_FRAMETYPE_*` constant in the public oneVPL API — the
        // runtime marks it with `MFX_FRAMETYPE_I` plus the
        // additional `MFX_FRAMETYPE_REF` flag (0x0040). Treat any
        // of those as a keyframe for MP4's `stss` sync-sample
        // table.
        //   MFX_FRAMETYPE_I     = 0x0001 — key frame
        //   MFX_FRAMETYPE_IDR   = 0x8000 — H.264/HEVC IDR (unused for AV1)
        //   MFX_FRAMETYPE_xREF  = 0x0040 — reference frame (paired w/ I for INTRA_ONLY)
        // systems-review-59-60 A-Q5.
        //
        // For AV1 the field is left at zero by the iHD runtime, so the flag
        // above is always false and the bitstream is the only source that
        // knows. Read from it rather than trusting the encoder: in CMAF a
        // segment must open on a sync sample, and a stream whose packets all
        // claim to be delta frames can never be cut into segments at all —
        // everything accumulates in one pending buffer and the trailing flush
        // fails with "first pending sample is not a keyframe".
        // This encoder is AV1-only — the tile params, the sequence-header
        // signalling and the dispatch above all say so — so the bitstream is
        // consulted for every packet rather than behind a codec test.
        let is_keyframe =
            (session.bitstream.frame_type & (MFX_FRAMETYPE_I | MFX_FRAMETYPE_IDR)) != 0
                || crate::pixel_format::av1_packet_is_keyframe(&data_bytes);
        let pts = session.bitstream.time_stamp;

        packets.push(EncodedPacket {
            data: data_bytes,
            pts,
            is_keyframe,
        });

        // Reset the output buffer for reuse.
        session.bitstream.data_length = 0;
        session.bitstream.data_offset = 0;
        Ok(())
    }
}

impl Encoder for QsvEncoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        self.encode_one(frame)
    }

    fn force_keyframe_next(&mut self) -> Result<()> {
        self.force_idr_next = true;
        Ok(())
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

/// Align `v` up to the next multiple of `a`. `a` must be a power of 2.
fn align_up<T>(v: T, a: T) -> T
where
    T: Copy
        + std::ops::Add<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::BitAnd<Output = T>
        + std::ops::Not<Output = T>
        + From<u8>,
{
    let one = T::from(1u8);
    (v + a - one) & !(a - one)
}
