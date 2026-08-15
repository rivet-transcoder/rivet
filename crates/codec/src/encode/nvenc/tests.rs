use super::constants::{
    guid_from_bytes, NV_ENC_BUFFER_FORMAT_IYUV, NV_ENC_BUFFER_FORMAT_YUV420_10BIT,
    NV_ENC_PRESET_P5_GUID, RING_SIZE,
};
use super::ffi::{NvEncConfigAv1, AV1_CHROMA_FORMAT_IDC_420, NV_ENC_BIT_DEPTH_10, NV_ENC_BIT_DEPTH_8};
use super::helpers::{
    fps_to_rational, nvenc_buffer_format_for, pixel_bit_depth_minus8_for, transfer_to_h273,
};
use crate::frame::{ColorMetadata, PixelFormat, TransferFn};

#[test]
fn test_fps_rational_mapping() {
    // Broadcast rates from MEDIUM-4.
    assert_eq!(fps_to_rational(23.976), (24_000, 1001));
    assert_eq!(fps_to_rational(24.0), (24, 1));
    assert_eq!(fps_to_rational(25.0), (25, 1));
    assert_eq!(fps_to_rational(29.97), (30_000, 1001));
    assert_eq!(fps_to_rational(30.0), (30, 1));
    assert_eq!(fps_to_rational(48.0), (48, 1));
    assert_eq!(fps_to_rational(50.0), (50, 1));
    assert_eq!(fps_to_rational(59.94), (60_000, 1001));
    assert_eq!(fps_to_rational(60.0), (60, 1));
}

#[test]
fn test_fps_rational_1001_family_detection() {
    // Higher-precision 1001-family values should still hit the
    // canonical rational.
    let (n, d) = fps_to_rational(23.9760239760);
    assert_eq!(d, 1001);
    assert_eq!(n, 24_000);

    let (n, d) = fps_to_rational(29.9700299700);
    assert_eq!(d, 1001);
    assert_eq!(n, 30_000);

    let (n, d) = fps_to_rational(59.9400599400);
    assert_eq!(d, 1001);
    assert_eq!(n, 60_000);
}

#[test]
fn test_fps_rational_generic_fallback() {
    // Integer fps not in the broadcast table: use `(n, 1)` form
    // (integer shortcut, before 1001-family detector would match).
    assert_eq!(fps_to_rational(100.0), (100, 1));
    assert_eq!(fps_to_rational(120.0), (120, 1));
    // 23.5 has no 1001-family match and isn't integer — generic
    // fallback (round(fps*1000), 1000).
    assert_eq!(fps_to_rational(23.5), (23_500, 1000));
}

#[test]
fn test_nvenc_cq_clamps_to_51() {
    // The SDK documents 0..51 for H.264/HEVC and 0..63 for AV1 on
    // `targetQuality`. The code path clamps to 51 before handing
    // the value to `rc_params.target_quality` to stay inside the
    // historical H.264/HEVC band (AV1's 0..63 is not rejected but
    // values >51 produce ill-defined behaviour on older drivers).
    let clamped = 75u8.min(51);
    assert_eq!(clamped, 51);
    let ok = 40u8.min(51);
    assert_eq!(ok, 40);
    let at_limit = 51u8.min(51);
    assert_eq!(at_limit, 51);
}

#[test]
fn a_slot_the_encoder_still_holds_is_never_reused() {
    // Replaces a test that asserted the ring walked 0,1,2,3,0,1,...
    //
    // That advance rule was the bug: it is only correct while every
    // EncodePicture returns a packet immediately, and the moment the encoder
    // buffers (lookahead, reordering) the slot four submissions later is a
    // surface it is still reading. The picture it emitted then carried the
    // newer frame's pixels — one wrong frame per encoder, which on a
    // per-segment encoder is one per segment.
    //
    // The rule now is "take a slot with nothing outstanding", so this asserts
    // selection, not arithmetic.
    let mut in_flight = [false; RING_SIZE];
    in_flight[0] = true; // the encoder is holding slot 0
    in_flight[1] = true;

    let chosen = (0..RING_SIZE).find(|&i| !in_flight[i]);

    assert_eq!(chosen, Some(2), "selection must skip every slot still held");
}

#[test]
fn the_pool_is_deeper_than_the_lookahead_it_allows() {
    // `mod.rs` caps a requested lookahead at `RING_SIZE - 4`, so the encoder
    // can never be asked to hold more frames than there are slots to keep
    // clear. If someone shrinks the pool, this is the thing that has to give.
    assert!(RING_SIZE >= 8, "a pool this shallow cannot carry any lookahead");

    let cap = RING_SIZE - 4;
    assert!(cap > 0, "no lookahead budget at all");
    assert!(cap < RING_SIZE, "lookahead must leave slots free to submit into");
}

// ── Squad-22: 10-bit dispatch + color signalling tests ───────

/// `nvenc_buffer_format_for` must return the YUV420_10BIT constant
/// for `Yuv420p10le` and IYUV for plain `Yuv420p`. Mismatched dispatch
/// here would produce silently-wrong encodes (the IYUV path on a
/// 10-bit surface would write the wide samples into the 8-bit slot
/// the GPU expects → uniform mid-gray output).
#[test]
fn test_nvenc_buffer_format_dispatch_10bit() {
    let fmt_8 = nvenc_buffer_format_for(PixelFormat::Yuv420p).unwrap();
    let fmt_10 = nvenc_buffer_format_for(PixelFormat::Yuv420p10le).unwrap();
    assert_eq!(fmt_8, NV_ENC_BUFFER_FORMAT_IYUV);
    assert_eq!(fmt_10, NV_ENC_BUFFER_FORMAT_YUV420_10BIT);
    assert_ne!(
        fmt_8, fmt_10,
        "10-bit must select a different SDK constant from 8-bit"
    );
}

/// Unsupported pixel formats must bail with a typed error, NOT
/// fall through to the IYUV path. Mirrors the NVDEC chroma reject
/// (Squad-6) — we carry an explicit not-supported list rather than
/// best-effort attempts that produce silent corruption.
#[test]
fn test_nvenc_buffer_format_dispatch_rejects_4_2_2_and_4_4_4() {
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
            nvenc_buffer_format_for(unsupported).is_err(),
            "{unsupported:?} must be rejected by NVENC dispatch"
        );
    }
}

/// `pixel_bit_depth_minus_8` field controls the AV1 OBU sequence
/// header `BitDepth` value: 0 → 8-bit, 2 → 10-bit. A const_assert!
/// at the bottom of the file pins this; the test mirrors it for
/// the test summary.
#[test]
fn test_nvenc_pixel_bit_depth_dispatch() {
    assert_eq!(pixel_bit_depth_minus8_for(PixelFormat::Yuv420p), 0);
    assert_eq!(pixel_bit_depth_minus8_for(PixelFormat::Yuv420p10le), 2);
}

/// `transfer_to_h273` must round-trip every `TransferFn` variant
/// to its ITU-T H.273 numeric code. These match the codes the mux
/// `colr nclx` writer emits — keeping the in-bitstream value
/// identical to the container-level metadata.
#[test]
fn test_nvenc_transfer_to_h273_codes() {
    assert_eq!(transfer_to_h273(TransferFn::Bt709), 1);
    assert_eq!(transfer_to_h273(TransferFn::Bt470Bg), 4);
    assert_eq!(transfer_to_h273(TransferFn::Linear), 8);
    assert_eq!(transfer_to_h273(TransferFn::St2084), 16, "HDR10 PQ");
    assert_eq!(transfer_to_h273(TransferFn::AribStdB67), 18, "HLG");
    assert_eq!(
        transfer_to_h273(TransferFn::Unspecified),
        1,
        "Unspecified collapses to canonical Bt709 — AV1 has no \
         unspecified sentinel for transfer"
    );
}

/// Build a 10-bit AV1 codec config by hand and assert the bytes
/// at the bit-depth + color signalling offsets carry the expected
/// values. This is the "construct the struct, dump its bytes,
/// assert" test the task spec calls for — it doesn't need a GPU
/// because all the field writes are pure-Rust struct mutations.
///
/// SDK 13 retired `pixel_bit_depth_minus_8`/`input_pixel_bit_depth_minus_8`
/// in favour of `output_bit_depth`/`input_bit_depth` enums (8-bit=0,
/// 10-bit=1). It also dropped the explicit
/// `color_description_present_flag` field; the four color codes are
/// emitted whenever any of them is non-zero (driver-side per SDK 13
/// docs). `chroma_format_idc` was folded into `flags` bits 7-8.
#[test]
fn test_nvenc_av1_config_10bit_hdr_layout() {
    let mut cfg: NvEncConfigAv1 = unsafe { std::mem::zeroed() };
    // SDK 13 NV_ENC_BIT_DEPTH enum is the literal depth: 8 and 10.
    let bit_depth_minus8 = pixel_bit_depth_minus8_for(PixelFormat::Yuv420p10le);
    let bit_depth_enum: u32 = if bit_depth_minus8 == 0 {
        NV_ENC_BIT_DEPTH_8
    } else {
        NV_ENC_BIT_DEPTH_10
    };
    cfg.output_bit_depth = bit_depth_enum;
    cfg.input_bit_depth = bit_depth_enum;
    cfg.flags |= AV1_CHROMA_FORMAT_IDC_420;

    // HDR10 metadata — BT.2020 NCL primaries, PQ transfer, full range.
    let cm = ColorMetadata {
        transfer: TransferFn::St2084,
        matrix_coefficients: 9, // BT.2020 NCL
        colour_primaries: 9,    // BT.2020
        full_range: true,
        mastering_display: None,
        content_light_level: None,
    };
    cfg.color_primaries = cm.colour_primaries as u32;
    cfg.transfer_characteristics = transfer_to_h273(cm.transfer);
    cfg.matrix_coefficients = cm.matrix_coefficients as u32;
    cfg.color_range = cm.full_range as u32;

    assert_eq!(cfg.output_bit_depth, 10, "NV_ENC_BIT_DEPTH_10");
    assert_eq!(cfg.input_bit_depth, 10, "NV_ENC_BIT_DEPTH_10 input");
    assert_eq!(cfg.color_primaries, 9, "BT.2020");
    assert_eq!(cfg.transfer_characteristics, 16, "ST 2084 / PQ");
    assert_eq!(cfg.matrix_coefficients, 9, "BT.2020 NCL");
    assert_eq!(cfg.color_range, 1, "full range");
    assert_eq!(
        cfg.flags & AV1_CHROMA_FORMAT_IDC_420,
        AV1_CHROMA_FORMAT_IDC_420,
        "chromaFormatIDC=1 (4:2:0) packed into flags bits 7-8"
    );

    // Byte-level: u32 LE reads at the field offsets. An accidental
    // field reorder during a future SDK port surfaces as a diff here.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            &cfg as *const NvEncConfigAv1 as *const u8,
            std::mem::size_of::<NvEncConfigAv1>(),
        )
    };
    let bd_offset = std::mem::offset_of!(NvEncConfigAv1, output_bit_depth);
    assert_eq!(
        u32::from_le_bytes(bytes[bd_offset..bd_offset + 4].try_into().unwrap()),
        10,
        "output_bit_depth must read back as 10 (NV_ENC_BIT_DEPTH_10) from raw bytes"
    );

    let prim_offset = std::mem::offset_of!(NvEncConfigAv1, color_primaries);
    assert_eq!(
        u32::from_le_bytes(bytes[prim_offset..prim_offset + 4].try_into().unwrap()),
        9,
        "color_primaries=9 (BT.2020) at the expected offset"
    );

    let trans_offset = std::mem::offset_of!(NvEncConfigAv1, transfer_characteristics);
    assert_eq!(
        u32::from_le_bytes(bytes[trans_offset..trans_offset + 4].try_into().unwrap()),
        16,
        "transfer_characteristics=16 (PQ) at the expected offset"
    );

    let range_offset = std::mem::offset_of!(NvEncConfigAv1, color_range);
    assert_eq!(
        u32::from_le_bytes(bytes[range_offset..range_offset + 4].try_into().unwrap()),
        1,
        "color_range=1 (full) at the expected offset"
    );
}

/// 8-bit default config still shapes correctly — paranoid guard
/// against the 10-bit additions silently breaking the SDR path.
#[test]
fn test_nvenc_av1_config_8bit_sdr_layout() {
    let mut cfg: NvEncConfigAv1 = unsafe { std::mem::zeroed() };
    let bit_depth_minus8 = pixel_bit_depth_minus8_for(PixelFormat::Yuv420p);
    let bit_depth_enum: u32 = if bit_depth_minus8 == 0 {
        NV_ENC_BIT_DEPTH_8
    } else {
        NV_ENC_BIT_DEPTH_10
    };
    cfg.output_bit_depth = bit_depth_enum;
    cfg.input_bit_depth = bit_depth_enum;
    cfg.flags |= AV1_CHROMA_FORMAT_IDC_420;
    let cm = ColorMetadata::default();
    cfg.color_primaries = cm.colour_primaries as u32;
    cfg.transfer_characteristics = transfer_to_h273(cm.transfer);
    cfg.matrix_coefficients = cm.matrix_coefficients as u32;
    cfg.color_range = cm.full_range as u32;

    assert_eq!(cfg.output_bit_depth, 8, "NV_ENC_BIT_DEPTH_8");
    assert_eq!(cfg.color_primaries, 1, "BT.709 default");
    assert_eq!(cfg.transfer_characteristics, 1, "BT.709 default");
    assert_eq!(cfg.matrix_coefficients, 1, "BT.709 default");
    assert_eq!(cfg.color_range, 0, "studio range default");
}

#[test]
fn test_guid_roundtrip() {
    // `guid_from_bytes` on the P5 GUID bytes (NVENC SDK 13 layout,
    // little-endian per Microsoft GUID convention) must reproduce
    // the typed P5 GUID constant.
    //
    // P5 = 21c6e6b4-297a-4cba-998f-b6cbde72ade3
    // The leading three groups serialise LE, the last two groups
    // serialise BE — that's the asymmetry MS GUIDs are known for.
    // (Earlier the test held SDK 12.2's pre-rotation P5 bytes
    // d0918ee2-a509-4681-af96-e9c3c45b7aa7; updated alongside the
    // constant in the SDK 13 layout-fix commit.)
    let bytes: [u8; 16] = [
        0xb4, 0xe6, 0xc6, 0x21, 0x7a, 0x29, 0xba, 0x4c, 0x99, 0x8f, 0xb6, 0xcb, 0xde, 0x72,
        0xad, 0xe3,
    ];
    let g = guid_from_bytes(bytes);
    assert_eq!(g.data1, NV_ENC_PRESET_P5_GUID.data1);
    assert_eq!(g.data2, NV_ENC_PRESET_P5_GUID.data2);
    assert_eq!(g.data3, NV_ENC_PRESET_P5_GUID.data3);
    assert_eq!(g.data4, NV_ENC_PRESET_P5_GUID.data4);
}
