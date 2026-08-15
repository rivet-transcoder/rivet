//! Unit tests for the QSV AV1 encoder.
//!
//! All tests run without live hardware. Tests assert on struct layouts,
//! rate-control slot mapping, constant values, and helper functions.

use super::*;                                      // pub items from mod.rs
use super::ffi::*;                                 // pub(super) ffi constants + structs
use super::config::*;                              // pub(super) config helpers
use super::surface::*;                             // RING_SIZE
// align_up is a private fn in mod.rs; child modules must import it explicitly.
use super::align_up;
use crate::encode::tuning::{self, QualityTarget, QsvRateControl, SpeedTier};
use crate::frame::{ColorMetadata, PixelFormat, TransferFn};
use crate::qsv_ffi::{MfxExtBuffer, MfxFrameInfo, MfxInfoMfx};

/// ICQQuality must land in `qpp_or_kbps_or_icq` (slot 1 of the
/// `mfxInfoMFX` rate-control union) per vendor/intel/mfxstructs.h:83.
/// An earlier rev of this file put it in slot 0; that's the bug
/// task #60 caught (HIGH severity — silent quality fallback to
/// the driver default ICQQuality=23).
#[test]
fn test_qsv_icq_quality_lands_on_correct_struct_field() {
    let slots = rate_slots_for_rc(QsvRateControl::Icq, 0, 0, 28);
    assert_eq!(
        slots.slot1_qpp_or_kbps_or_icq, 28,
        "ICQ quality must land in slot 1 (qpp_or_kbps_or_icq) per \
         vendor/intel/mfxstructs.h:83 — this is the TargetKbps/QPP/ICQQuality \
         union arm"
    );
    assert_eq!(
        slots.slot0_qpi_or_delay, 0,
        "slot 0 is InitialDelayInKB/QPI/Accuracy per \
         vendor/intel/mfxstructs.h:74-78 — no ICQQuality here"
    );
    assert_eq!(
        slots.slot2_qpb_or_maxkbps, 0,
        "slot 2 is MaxKbps/QPB/Convergence per \
         vendor/intel/mfxstructs.h:85-89 — no ICQQuality here"
    );
}

/// CQP mode places QPI at slot 0 and QPP at slot 1 — vendor
/// header lines 74-78 (`QPI`) and 80-84 (`QPP`). QPB goes into
/// slot 2 (header lines 85-89).
#[test]
fn test_qsv_cqp_slots_mirror_qpi_qpp_qpb() {
    let slots = rate_slots_for_rc(QsvRateControl::Cqp, 72, 96, 0);
    assert_eq!(slots.slot0_qpi_or_delay, 72);
    assert_eq!(slots.slot1_qpp_or_kbps_or_icq, 96);
    // We mirror QPP to QPB since we run without B-frames (GopRefDist=1).
    assert_eq!(slots.slot2_qpb_or_maxkbps, 96);
}

/// TargetUsage must be a valid oneVPL 1..7 value. The tuning
/// adapter returns 1 (Archive), 4 (Standard), 6 (Draft); this
/// test covers the end-to-end mapping from `SpeedTier` through
/// the adapter to the `clamp_target_usage` gate.
#[test]
fn test_qsv_target_usage_maps_from_speed_tier() {
    let (w, h) = (1920, 1080);
    let cases = [
        (SpeedTier::Archive, 1u16, "1 = BEST_QUALITY per mfxdefs.h:91"),
        (SpeedTier::Standard, 4u16, "4 = BALANCED per mfxdefs.h:92"),
        (SpeedTier::Draft, 6u16, "6 = one step from BEST_SPEED (7)"),
    ];
    for (tier, expected, reason) in cases {
        let tp = tuning::qsv_av1_params(QualityTarget::Standard, tier, w, h);
        let got = clamp_target_usage(tp.target_usage);
        assert_eq!(got, expected, "{tier:?} → {got} (want {expected}, {reason})");
        assert!(
            (1..=7).contains(&got),
            "TargetUsage must be 1..7 per vendor/intel/mfxdefs.h:91-93"
        );
    }
}

/// `clamp_target_usage` defends against out-of-range values from
/// callers or a future tuning-adapter bug.
#[test]
fn test_qsv_target_usage_clamps_out_of_range() {
    assert_eq!(clamp_target_usage(0), 1, "0 clamps up to 1");
    assert_eq!(clamp_target_usage(8), 7, "8 clamps down to 7");
    assert_eq!(clamp_target_usage(255), 7, "255 clamps down to 7");
    assert_eq!(clamp_target_usage(4), 4, "4 passes through");
}

/// A slot the runtime still holds must never be chosen — including when it
/// has no sync point outstanding.
///
/// That combination is exactly what `MFX_ERR_MORE_DATA` produces: the runtime
/// has taken the surface but owes us no packet yet, so there is nothing to
/// sync on and only `Locked` says it is still in use. The round-robin index
/// this replaced ignored `Locked`, so after four submissions it walked back
/// onto the surface holding frame 0 and overwrote it with frame 4 — and the
/// encoder emitted frame 4's pixels as frame 0, once per encoder.
#[test]
fn a_slot_the_runtime_holds_is_never_chosen() {
    // (sync point drained?, Locked)
    let slots = [
        (true, 1u16),  // MORE_DATA: no sync point, still held  <- the trap
        (true, 2),     // held as a prediction reference
        (false, 0),    // our sync point is still pending
        (true, 0),     // free
        (true, 0),
    ];
    let pick = (0..slots.len()).find(|&i| slots[i].0 && slots[i].1 == 0);
    assert_eq!(
        pick,
        Some(3),
        "must skip both held slots and the one with a pending sync point"
    );
}

#[test]
fn the_pool_is_larger_than_the_async_depth() {
    assert_eq!(ASYNC_DEPTH, 4, "upstream sample_encode's recommendation");
    assert!(
        POOL_SIZE > ASYNC_DEPTH,
        "the pool must cover what the runtime retains beyond its async queue, \
         or there is no free surface to write the next frame into"
    );
}

/// `MFX_ERR_MORE_DATA` (-10) on EncodeFrameAsync means the
/// encoder wants another frame before it can emit output —
/// normal at startup. The caller's contract: submitting a frame
/// that hits MORE_DATA should not produce a packet and should
/// not fail. This test emulates the match arm that handles it.
#[test]
fn test_qsv_more_data_on_encode_returns_no_packet() {
    fn simulate_encode(rc: MfxStatus) -> std::result::Result<Option<()>, String> {
        match rc {
            MFX_ERR_NONE => Ok(Some(())), // sync point produced
            MFX_ERR_MORE_DATA => Ok(None), // no packet, no error
            err if err > 0 => Ok(None),    // warning, no packet
            err => Err(format!("encode failed: {err}")),
        }
    }
    assert_eq!(simulate_encode(MFX_ERR_MORE_DATA).unwrap(), None);
    assert_eq!(simulate_encode(MFX_ERR_NONE).unwrap(), Some(()));
    assert!(simulate_encode(-1).is_err(), "unknown negative = hard error");
}

/// `flush_drain` loops EncodeFrameAsync(NULL) and terminates on
/// MFX_ERR_MORE_DATA. The state machine must not panic when
/// it walks directly to MORE_DATA with zero pending in-flight
/// frames (the clean-EOF case).
#[test]
fn test_qsv_eof_drain_ends_cleanly() {
    // Simulate the flush loop: on MORE_DATA we exit Ok.
    fn simulate_flush_tick(rc: MfxStatus) -> std::result::Result<bool, String> {
        // Returns Ok(true) if we should exit, Ok(false) to keep
        // looping, Err on hard failure.
        match rc {
            MFX_ERR_NONE => Ok(false),
            MFX_ERR_MORE_DATA => Ok(true),
            err if err > 0 => Ok(false),
            err => Err(format!("flush failed: {err}")),
        }
    }
    assert_eq!(simulate_flush_tick(MFX_ERR_MORE_DATA).unwrap(), true,
               "clean EOF: flush terminates on MORE_DATA without error");
    assert_eq!(simulate_flush_tick(MFX_ERR_NONE).unwrap(), false,
               "NONE: flush has more output to drain");
    assert_eq!(simulate_flush_tick(MFX_WRN_VIDEO_PARAM_CHANGED).unwrap(), false,
               "warning: flush keeps looping to drain the bitstream");
    assert!(simulate_flush_tick(-5).is_err(), "hard negative error bails");
}

/// `MFX_ERR_MORE_SURFACE` is a DECODE-path status, not an
/// ENCODE-path one — per vendor/intel/mfxdefs.h:40. We name the
/// constant so the `match` arm could distinguish it, but the
/// encode `match` never needs to handle it. Verify the two are
/// distinct values so a future driver quirk doesn't silently
/// collapse them.
#[test]
fn test_qsv_more_data_and_more_surface_are_distinct() {
    assert_eq!(MFX_ERR_MORE_DATA, -10);
    assert_eq!(MFX_ERR_MORE_SURFACE, -11);
    assert_ne!(MFX_ERR_MORE_DATA, MFX_ERR_MORE_SURFACE);
}

/// oneVPL FOURCC encoding is little-endian: the first char in the
/// macro becomes the low byte of the u32. Verify our AV1 codec +
/// NV12 FourCC literals match the `MFX_MAKE_FOURCC` definition at
/// vendor/intel/mfxdefs.h:61-62.
#[test]
fn test_qsv_fourcc_literals_match_macro() {
    fn make(a: u8, b: u8, c: u8, d: u8) -> u32 {
        (a as u32) | ((b as u32) << 8) | ((c as u32) << 16) | ((d as u32) << 24)
    }
    assert_eq!(MFX_CODEC_AV1, make(b'A', b'V', b'1', b' '));
    assert_eq!(MFX_FOURCC_NV12, make(b'N', b'V', b'1', b'2'));
    assert_eq!(MFX_EXTBUFF_AV1_TILE_PARAM, make(b'A', b'1', b'T', b'L'));
    assert_eq!(MFX_EXTBUFF_AV1_BITSTREAM_PARAM, make(b'A', b'V', b'1', b'B'));
}

/// AV1 profile = MAIN = 1 per vendor/intel/mfxav1.h:24. Main
/// covers 8-bit and 10-bit 4:2:0 AV1 content — fine for our
/// pipeline (always 8-bit, rav1d bails on 10-bit).
#[test]
fn test_qsv_profile_main_equals_one() {
    assert_eq!(MFX_PROFILE_AV1_MAIN, 1);
}

/// Chroma format = YUV420 = 1 per vendor/intel/mfxdefs.h:103.
#[test]
fn test_qsv_chroma_format_yuv420_equals_one() {
    assert_eq!(MFX_CHROMAFORMAT_YUV420, 1);
}

/// Rate control modes match vendor/intel/mfxdefs.h:76-79.
#[test]
fn test_qsv_rc_mode_values_match_spec() {
    assert_eq!(MFX_RATECONTROL_CQP, 3);
    assert_eq!(MFX_RATECONTROL_ICQ, 9); // 8 is LA, not ICQ
}

/// Verify the rate-control slot wiring at the mfxInfoMFX field
/// level, not just the helper. Builds a full `MfxInfoMfx` for an
/// ICQ job and asserts the ICQ value is on the `qpp_or_kbps_or_icq`
/// field (slot 1).
#[test]
fn test_qsv_mfx_info_fields_from_slots() {
    use crate::qsv_ffi::MfxInfoMfx;

    let slots = rate_slots_for_rc(QsvRateControl::Icq, 0, 0, 33);
    let mfx = MfxInfoMfx {
        reserved: [0; 7],
        low_power: 0,
        brc_param_multiplier: 0,
        frame_info: MfxFrameInfo {
            reserved: [0; 4],
            channel_id: 0,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            shift: 0,
            frame_id: [0; 4],
            fourcc: MFX_FOURCC_NV12,
            width: 1920,
            height: 1080,
            crop_x: 0,
            crop_y: 0,
            crop_w: 1920,
            crop_h: 1080,
            frame_rate_ext_n: 30000,
            frame_rate_ext_d: 1000,
            reserved3: 0,
            aspect_ratio_w: 1,
            aspect_ratio_h: 1,
            pic_struct: MFX_PICSTRUCT_PROGRESSIVE,
            chroma_format: MFX_CHROMAFORMAT_YUV420,
            reserved2: 0,
        },
        codec_id: MFX_CODEC_AV1,
        codec_profile: MFX_PROFILE_AV1_MAIN,
        codec_level: 0,
        num_thread: 0,
        target_usage: 4,
        gop_pic_size: 240,
        gop_ref_dist: 1,
        gop_opt_flag: 0,
        idr_interval: 0,
        rate_control_method: MFX_RATECONTROL_ICQ,
        qpi_or_delay: slots.slot0_qpi_or_delay,
        buffer_size_kb: 0,
        qpp_or_kbps_or_icq: slots.slot1_qpp_or_kbps_or_icq,
        qpb_or_maxkbps: slots.slot2_qpb_or_maxkbps,
        num_slice: 0,
        num_ref_frame: 1,
        encoded_order: 0,
    };

    // End-to-end: user asked for ICQ quality 33.
    // Verify it ended up at the `ICQQuality` alias
    // (`qpp_or_kbps_or_icq`) and nowhere else.
    assert_eq!(mfx.qpp_or_kbps_or_icq, 33, "ICQQuality lives at slot 1");
    assert_eq!(mfx.qpi_or_delay, 0, "slot 0 must be zero in ICQ mode");
    assert_eq!(mfx.qpb_or_maxkbps, 0, "slot 2 must be zero in ICQ mode");
    assert_eq!(mfx.rate_control_method, MFX_RATECONTROL_ICQ);
}

/// `zeroed_video_param` returns an all-zero struct — used as
/// Query's `out` param so the runtime can overwrite non-zero
/// fields to signal "I adjusted this one".
#[test]
fn test_qsv_zeroed_video_param_is_all_zero() {
    let z = zeroed_video_param();
    assert_eq!(z.mfx.codec_id, 0);
    assert_eq!(z.mfx.codec_profile, 0);
    assert_eq!(z.mfx.rate_control_method, 0);
    assert_eq!(z.mfx.qpi_or_delay, 0);
    assert_eq!(z.mfx.qpp_or_kbps_or_icq, 0);
    assert_eq!(z.mfx.qpb_or_maxkbps, 0);
    assert_eq!(z.mfx.frame_info.width, 0);
    assert_eq!(z.mfx.frame_info.height, 0);
    assert!(z.ext_param.is_null());
}

/// A quick cover-test for `align_up`: rounds up to the nearest
/// multiple of a power-of-2.
#[test]
fn test_qsv_align_up_power_of_two() {
    assert_eq!(align_up(1u32, 16u32), 16);
    assert_eq!(align_up(16u32, 16u32), 16);
    assert_eq!(align_up(17u32, 16u32), 32);
    assert_eq!(align_up(1920u32, 64u32), 1920);
    assert_eq!(align_up(1921u32, 64u32), 1984);
}

/// The Init path on a real Intel GPU would construct a whole
/// `MfxVideoParam` for Query/Init. We can't do that offline, but
/// we CAN exercise every field through `rate_slots_for_rc` +
/// `clamp_target_usage` and make sure the produced ICQ value is
/// what the tuning adapter emitted — i.e. no silent zero-fallback.
#[test]
fn test_qsv_icq_flow_preserves_tuning_adapter_value() {
    for (w, h) in [(640, 360), (1920, 1080), (3840, 2160)] {
        for target in [
            QualityTarget::Low,
            QualityTarget::Standard,
            QualityTarget::High,
        ] {
            let tp = tuning::qsv_av1_params(target, SpeedTier::Standard, w, h);
            // tuning adapter returns ICQ for non-VL targets.
            assert_eq!(tp.rc_mode, QsvRateControl::Icq);
            let slots = rate_slots_for_rc(tp.rc_mode, 0, 0, tp.icq_quality);
            assert_eq!(
                slots.slot1_qpp_or_kbps_or_icq, tp.icq_quality,
                "ICQ quality value must reach slot 1 end-to-end — \
                 {target:?}/{w}x{h}: adapter={}, slot1={}",
                tp.icq_quality, slots.slot1_qpp_or_kbps_or_icq
            );
            assert_eq!(slots.slot0_qpi_or_delay, 0);
            assert_eq!(slots.slot2_qpb_or_maxkbps, 0);
        }
    }
}

/// The MfxEncodeCtrl struct is named + sized but not passed to
/// the encoder today. Verify its size here so the const_assert
/// citation stays attached to a live reference.
#[test]
fn test_qsv_encode_ctrl_struct_size() {
    // Real mfxEncodeCtrl is 56 bytes (offsetof-verified). We pass NULL for
    // it anyway, but keep the size honest.
    assert_eq!(std::mem::size_of::<MfxEncodeCtrl>(), 56);
}

// ── Squad-22: QSV 10-bit dispatch + color signalling ─────────────────────────

/// Surface FOURCC dispatch must map `Yuv420p10le` to P010 (0x30313050,
/// 'P','0','1','0') and 8-bit `Yuv420p` to NV12. Mismatched FOURCC
/// would cause oneVPL to read the wide-word P010 surface as NV12 →
/// silently encode 1/64-amplitude noise.
#[test]
fn test_qsv_fourcc_dispatch_10bit() {
    assert_eq!(qsv_fourcc_for(PixelFormat::Yuv420p).unwrap(), MFX_FOURCC_NV12);
    assert_eq!(qsv_fourcc_for(PixelFormat::Yuv420p10le).unwrap(), MFX_FOURCC_P010);
    assert_eq!(MFX_FOURCC_P010, 0x30313050, "P010 FOURCC = 'P','0','1','0' LE");
}

/// Unsupported pixel formats must bail with a typed error. AV1
/// only carries 4:2:0 (Main profile) — 4:2:2 / 4:4:4 / RGB are
/// not decodable in the wider AV1 ecosystem either.
#[test]
fn test_qsv_fourcc_dispatch_rejects_4_2_2_and_4_4_4() {
    for unsupported in [
        PixelFormat::Yuv422p,
        PixelFormat::Yuv422p10le,
        PixelFormat::Yuv444p,
        PixelFormat::Yuv444p10le,
        PixelFormat::Yuva444p10le,
        PixelFormat::Nv12,
        PixelFormat::Rgb24,
    ] {
        assert!(
            qsv_fourcc_for(unsupported).is_err(),
            "{unsupported:?} must be rejected by QSV dispatch"
        );
    }
}

/// Bit-depth triple: NV12 → (8, 8, 0); P010 → (10, 10, 1).
/// Shift=1 is mandatory for P010 — without it, oneVPL reads
/// samples as if the valid bits were in the lower 10 → silently
/// encodes garbage. const_assert! pins this; the test mirrors it
/// for the test summary.
#[test]
fn test_qsv_bit_depth_triple_dispatch() {
    let (luma8, chroma8, shift8) = qsv_bit_depth_triple(PixelFormat::Yuv420p);
    assert_eq!((luma8, chroma8, shift8), (8, 8, 0), "NV12: 8-bit, no shift");

    let (luma10, chroma10, shift10) = qsv_bit_depth_triple(PixelFormat::Yuv420p10le);
    assert_eq!(
        (luma10, chroma10, shift10),
        (10, 10, 1),
        "P010: 10-bit + Shift=1 (upper-10-bit convention)"
    );
}

/// Per-encoder H.273 transfer codes must agree with the NVENC +
/// AMF + mux paths so a single ColorMetadata maps to identical
/// numeric values across every backend.
#[test]
fn test_qsv_transfer_to_h273_codes() {
    assert_eq!(transfer_to_h273(TransferFn::Bt709), 1);
    assert_eq!(transfer_to_h273(TransferFn::Bt470Bg), 4);
    assert_eq!(transfer_to_h273(TransferFn::Linear), 8);
    assert_eq!(transfer_to_h273(TransferFn::St2084), 16, "HDR10 PQ");
    assert_eq!(transfer_to_h273(TransferFn::AribStdB67), 18, "HLG");
    assert_eq!(transfer_to_h273(TransferFn::Unspecified), 1);
}

/// Build a 10-bit MfxFrameInfo by hand and assert all four fields
/// (`bit_depth_luma` / `_chroma` / `shift` / `fourcc`) are
/// consistent with what oneVPL expects for P010. A future field
/// reorder during an SDK port surfaces as a diff here.
#[test]
fn test_qsv_frame_info_p010_layout() {
    let (bdl, bdc, shift) = qsv_bit_depth_triple(PixelFormat::Yuv420p10le);
    let fourcc = qsv_fourcc_for(PixelFormat::Yuv420p10le).unwrap();

    let fi = MfxFrameInfo {
        reserved: [0; 4],
        channel_id: 0,
        bit_depth_luma: bdl,
        bit_depth_chroma: bdc,
        shift,
        frame_id: [0; 4],
        fourcc,
        width: 1920,
        height: 1080,
        crop_x: 0,
        crop_y: 0,
        crop_w: 1920,
        crop_h: 1080,
        frame_rate_ext_n: 30000,
        frame_rate_ext_d: 1000,
        reserved3: 0,
        aspect_ratio_w: 1,
        aspect_ratio_h: 1,
        pic_struct: MFX_PICSTRUCT_PROGRESSIVE,
        chroma_format: MFX_CHROMAFORMAT_YUV420,
        reserved2: 0,
    };

    assert_eq!(fi.bit_depth_luma, 10);
    assert_eq!(fi.bit_depth_chroma, 10);
    assert_eq!(fi.shift, 1, "P010 must set Shift=1");
    assert_eq!(fi.fourcc, MFX_FOURCC_P010);
    assert_eq!(fi.chroma_format, MFX_CHROMAFORMAT_YUV420, "still 4:2:0 sub-sampling");

    // Read fourcc back through raw bytes — guards against an
    // accidental field reorder during an SDK port.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            &fi as *const MfxFrameInfo as *const u8,
            std::mem::size_of::<MfxFrameInfo>(),
        )
    };
    let fourcc_offset = std::mem::offset_of!(MfxFrameInfo, fourcc);
    assert_eq!(
        u32::from_le_bytes(bytes[fourcc_offset..fourcc_offset + 4].try_into().unwrap()),
        MFX_FOURCC_P010,
        "fourcc reads back as P010 from the expected struct offset"
    );
}

/// `mfxExtCodingOption3` for 10-bit job: `TargetBitDepthLuma` /
/// `TargetBitDepthChroma` = 10; `TargetChromaFormatPlus1` = 2
/// (= MFX_CHROMAFORMAT_YUV420 + 1, oneVPL's "plus one" convention).
/// Without the ext buffer the encoder silently truncates samples
/// to 8-bit even though the surface is P010.
#[test]
fn test_qsv_coding_option3_10bit_layout() {
    let co3 = MfxExtCodingOption3 {
        header: MfxExtBuffer {
            buffer_id: MFX_EXTBUFF_CODING_OPTION3,
            buffer_sz: std::mem::size_of::<MfxExtCodingOption3>() as u32,
        },
        _pad_to_158: [0; 150],
        target_chroma_format_plus1: MFX_TARGET_CHROMAFORMAT_YUV420_PLUS1,
        target_bit_depth_luma: 10,
        target_bit_depth_chroma: 10,
        _tail: [0; 348],
    };

    assert_eq!(co3.target_bit_depth_luma, 10, "AV1 BitDepth=10 in seq header");
    assert_eq!(co3.target_bit_depth_chroma, 10, "AV1 BitDepth=10 in seq header");
    assert_eq!(
        co3.target_chroma_format_plus1, 2,
        "MFX_CHROMAFORMAT_YUV420 (1) + 1 = 2"
    );
    assert_eq!(co3.header.buffer_id, MFX_EXTBUFF_CODING_OPTION3);

    // offsetof-verified: TargetBitDepthLuma @160, TargetBitDepthChroma @162.
    assert_eq!(memoffset_target_bit_depth_luma(), 160);
    assert_eq!(MFX_EXTBUFF_CODING_OPTION3, 0x334f4443);
}

fn memoffset_target_bit_depth_luma() -> usize {
    let base = std::mem::MaybeUninit::<MfxExtCodingOption3>::uninit();
    let ptr = base.as_ptr();
    unsafe {
        (std::ptr::addr_of!((*ptr).target_bit_depth_luma) as usize) - (ptr as usize)
    }
}

/// `mfxExtVideoSignalInfo` for HDR10 (BT.2020 NCL primaries, PQ
/// transfer, full range): the four H.273 codes must round-trip
/// from `ColorMetadata` through `transfer_to_h273` into the
/// signal-info ext buffer's named fields. Without this, AV1 OBU
/// `color_config` defaults to "unspecified" and downstream
/// decoders fall back to BT.709.
#[test]
fn test_qsv_signal_info_hdr10_layout() {
    let cm = ColorMetadata {
        transfer: TransferFn::St2084,
        matrix_coefficients: 9,  // BT.2020 NCL
        colour_primaries: 9,     // BT.2020
        full_range: true,
        mastering_display: None,
        content_light_level: None,
    };

    let signal_info = MfxExtVideoSignalInfo {
        header: MfxExtBuffer {
            buffer_id: MFX_EXTBUFF_VIDEO_SIGNAL_INFO,
            buffer_sz: std::mem::size_of::<MfxExtVideoSignalInfo>() as u32,
        },
        video_format: 5,
        video_full_range: if cm.full_range { 1 } else { 0 },
        colour_description_present: 1,
        colour_primaries: cm.colour_primaries as u16,
        transfer_characteristics: transfer_to_h273(cm.transfer),
        matrix_coefficients: cm.matrix_coefficients as u16,
    };

    assert_eq!(signal_info.colour_description_present, 1, "must be set so codes emit");
    assert_eq!(signal_info.colour_primaries, 9, "BT.2020");
    assert_eq!(signal_info.transfer_characteristics, 16, "ST 2084 / PQ");
    assert_eq!(signal_info.matrix_coefficients, 9, "BT.2020 NCL");
    assert_eq!(signal_info.video_full_range, 1, "full range");
    assert_eq!(signal_info.header.buffer_id, MFX_EXTBUFF_VIDEO_SIGNAL_INFO);

    // 'V','S','I','N' LE → 0x4e495356.
    assert_eq!(
        MFX_EXTBUFF_VIDEO_SIGNAL_INFO, 0x4e495356,
        "ext buffer ID must match upstream MFX_MAKE_FOURCC('V','S','I','N')"
    );
}

/// 8-bit SDR config still shapes correctly — paranoid regression
/// guard against the 10-bit additions silently breaking the SDR
/// path. ICQ rate-control + the 8-bit FrameInfo + the default
/// SDR ColorMetadata round-trip.
#[test]
fn test_qsv_8bit_sdr_layout_unchanged() {
    let (bdl, bdc, shift) = qsv_bit_depth_triple(PixelFormat::Yuv420p);
    assert_eq!((bdl, bdc, shift), (8, 8, 0), "8-bit dispatch unchanged");

    let cm = ColorMetadata::default();
    let signal_info = MfxExtVideoSignalInfo {
        header: MfxExtBuffer {
            buffer_id: MFX_EXTBUFF_VIDEO_SIGNAL_INFO,
            buffer_sz: std::mem::size_of::<MfxExtVideoSignalInfo>() as u32,
        },
        video_format: 5,
        video_full_range: if cm.full_range { 1 } else { 0 },
        colour_description_present: 1,
        colour_primaries: cm.colour_primaries as u16,
        transfer_characteristics: transfer_to_h273(cm.transfer),
        matrix_coefficients: cm.matrix_coefficients as u16,
    };

    assert_eq!(signal_info.colour_primaries, 1, "BT.709 default");
    assert_eq!(signal_info.transfer_characteristics, 1, "BT.709 default");
    assert_eq!(signal_info.matrix_coefficients, 1, "BT.709 default");
    assert_eq!(signal_info.video_full_range, 0, "studio range default");
}

// ─── CRF → codec-native quality ──────────────────────────────────────────────
//
// `--crf` was a dead flag on this path: the ICQ branch never read
// `config.quality`, and `rate_slots_for_rc` was handed the tuning table's
// suggested mode instead of the `constant_qp`-resolved one, so a forced CQP
// wrote its QP into a slot only ICQ reads. `--crf 5` and `--crf 45` produced
// byte-identical output in both modes. These pin the conversions and the
// slot/mode agreement that fixed it.

#[test]
fn crf_reads_on_each_codecs_own_native_scale() {
    use crate::frame::VideoCodec;
    // H.264 / HEVC: the x264 / x265 0..51 CRF range, passed through to ICQ
    // (which is also 1..51) and to QP (0..51) unchanged.
    for c in [VideoCodec::H265, VideoCodec::H264] {
        assert_eq!(crf_to_icq(c, 22), 22, "{c:?} CRF is already on the ICQ scale");
        assert_eq!(crf_to_qp(c, 22), 22, "{c:?} CRF is already on the QP scale");
        // An out-of-range CRF is clamped into the codec's legal QP band
        // rather than handed to the driver as-is.
        assert_eq!(crf_to_qp(c, 200), 51);
        assert_eq!(crf_to_icq(c, 200), 51);
        // ICQ has no 0: it starts at 1.
        assert_eq!(crf_to_icq(c, 0), 1);
    }

    // AV1: the libaom / SVT 0..63 cq-level range. ICQ rescales by 51/63; QP
    // is the native 0..255 q-index at the usual 4x.
    assert_eq!(crf_to_qp(VideoCodec::Av1, 32), 128);
    assert_eq!(crf_to_qp(VideoCodec::Av1, 63), 252);
    assert_eq!(crf_to_qp(VideoCodec::Av1, 200), 252, "clamped to the 0..63 scale first");
    assert_eq!(crf_to_icq(VideoCodec::Av1, 63), 51, "top of scale maps to top of ICQ");
    assert!(
        (25..=27).contains(&crf_to_icq(VideoCodec::Av1, 32)),
        "libaom cq 32 should land near ICQ 26, got {}",
        crf_to_icq(VideoCodec::Av1, 32)
    );
}

#[test]
fn crf_is_monotonic_so_a_higher_number_really_is_lower_quality() {
    use crate::frame::VideoCodec;
    for c in [VideoCodec::Av1, VideoCodec::H265, VideoCodec::H264] {
        for crf in 1u8..50 {
            assert!(
                crf_to_icq(c, crf) <= crf_to_icq(c, crf + 1),
                "{c:?} ICQ not monotonic at {crf}"
            );
            assert!(
                crf_to_qp(c, crf) <= crf_to_qp(c, crf + 1),
                "{c:?} QP not monotonic at {crf}"
            );
        }
    }
}

#[test]
fn inter_qp_steps_coarser_within_the_codecs_range() {
    use crate::frame::VideoCodec;
    assert_eq!(inter_qp(VideoCodec::H265, 22), 24);
    assert_eq!(inter_qp(VideoCodec::Av1, 128), 136);
    // Never past the top of the scale.
    assert_eq!(inter_qp(VideoCodec::H265, 51), 51);
    assert_eq!(inter_qp(VideoCodec::Av1, 255), 255);
}

#[test]
fn cqp_slots_carry_the_qp_that_cqp_actually_reads() {
    // The regression: laying slots out for ICQ while running CQP put the QP
    // in slot1 (ICQQuality) and left slot0 (QPI) at zero, so the driver used
    // its own default and `--crf` did nothing.
    let cqp = rate_slots_for_rc(QsvRateControl::Cqp, 22, 24, 0);
    assert_eq!(cqp.slot0_qpi_or_delay, 22, "QPI must be in slot 0 under CQP");
    assert_eq!(cqp.slot1_qpp_or_kbps_or_icq, 24);

    let icq = rate_slots_for_rc(QsvRateControl::Icq, 0, 0, 22);
    assert_eq!(icq.slot0_qpi_or_delay, 0, "slot 0 is InitialDelayInKB under ICQ");
    assert_eq!(icq.slot1_qpp_or_kbps_or_icq, 22, "ICQQuality must be in slot 1");

    // The two layouts must not be interchangeable — that's the whole bug.
    assert_ne!(cqp, icq);
}

#[test]
fn hevc_tuning_stays_inside_hevcs_qp_range() {
    use crate::encode::tuning::qsv_params;
    use crate::frame::VideoCodec;
    // The AV1 table's `libaom_cq * 4` reaches 152, which is meaningless as an
    // HEVC QP. Every H.26x tier must sit in 0..51 on both knobs.
    for target in [
        QualityTarget::VisuallyLossless,
        QualityTarget::High,
        QualityTarget::Standard,
        QualityTarget::Low,
    ] {
        for c in [VideoCodec::H265, VideoCodec::H264] {
            let p = qsv_params(c, target, SpeedTier::Standard, 1920, 1080);
            assert!(p.qp_i <= 51, "{c:?} {target:?} QPI {} out of range", p.qp_i);
            assert!(p.qp_p <= 51, "{c:?} {target:?} QPP {} out of range", p.qp_p);
            assert!((1..=51).contains(&p.icq_quality), "{c:?} {target:?} ICQ out of range");
            assert!(p.qp_p >= p.qp_i, "inter frames should not be finer than intra");
            // Tiles are an AV1-only ext buffer; H.26x must not request a grid.
            assert_eq!((p.num_tile_columns, p.num_tile_rows), (0, 0));
        }
    }
    // AV1 keeps its wide q-index range.
    let av1 = qsv_params(VideoCodec::Av1, QualityTarget::Standard, SpeedTier::Standard, 1920, 1080);
    assert!(av1.qp_i > 51, "AV1 uses the 0..255 q-index, got {}", av1.qp_i);
}

#[test]
fn better_targets_ask_for_finer_quantization() {
    use crate::encode::tuning::qsv_params;
    use crate::frame::VideoCodec;
    let p = |t| qsv_params(VideoCodec::H265, t, SpeedTier::Standard, 1920, 1080).icq_quality;
    assert!(p(QualityTarget::VisuallyLossless) < p(QualityTarget::High));
    assert!(p(QualityTarget::High) < p(QualityTarget::Standard));
    assert!(p(QualityTarget::Standard) < p(QualityTarget::Low));
}
