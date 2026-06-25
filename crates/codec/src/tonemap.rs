//! HDR → SDR tonemap.
//!
//! Pipeline: 10-bit BT.2020 PQ/HLG Y'CbCr → linear scene-referred RGB →
//! BT.709 gamut → Hable filmic curve → BT.709 gamma → 8-bit BT.709
//! limited-range Y'CbCr.
//!
//! Single-output policy: every HDR upload gets tonemapped to SDR at
//! transcode time and the encoded ABR ladder is 8-bit BT.709. Every
//! viewer sees a correctly-mapped image regardless of display capability.
//! HDR-fidelity-for-HDR-viewers is a future dual-rendition path that
//! will reuse the same primitives for the SDR rungs.
//!
//! Reference standards:
//!   - ITU-R BT.2020 (matrix + primaries)
//!   - SMPTE ST.2084 (PQ EOTF)
//!   - ARIB STD-B67 (HLG EOTF)
//!   - ITU-R BT.709 (output matrix + transfer)
//!   - "Filmic Tonemapping for Real-time Rendering" — John Hable, 2010
//!
//! Implementation is pure-Rust scalar f32. AVX2 vectorisation is a
//! follow-up — the kernel is hot (per-pixel two matrix multiplies + one
//! transcendental on each channel + the filmic curve), but it's
//! single-threaded with a per-frame fan-out, so the per-thread budget
//! lands well inside a 1080p60 real-time window even scalar.

use anyhow::{Result, bail};
use bytes::Bytes;

use crate::frame::{ColorSpace, PixelFormat, TransferFn, VideoFrame};

// ── transfer (EOTF inverse: encoded → scene linear) ───────────────────

/// PQ inverse EOTF (SMPTE ST.2084).
///
/// Returns scene-linear in units where `1.0 = 100 cd/m² SDR diffuse white`,
/// so `100.0 = 10,000 nits PQ peak`. Tonemap operates in the same scene-
/// linear frame.
#[inline(always)]
fn pq_to_linear(n: f32) -> f32 {
    const M1_INV: f32 = 1.0 / 0.159_301_76;
    const M2_INV: f32 = 1.0 / 78.84375;
    const C1: f32 = 0.8359375;
    const C2: f32 = 18.851_563;
    const C3: f32 = 18.6875;
    let np = n.max(0.0).powf(M2_INV);
    let num = (np - C1).max(0.0);
    let den = C2 - C3 * np;
    if den <= 0.0 {
        return 0.0;
    }
    let lin01 = (num / den).powf(M1_INV); // 0..1, 1.0 = 10,000 nits
    lin01 * 100.0 // rescale so 1.0 = SDR diffuse white (~100 nits)
}

/// HLG inverse OETF (ARIB STD-B67) followed by the SDR-target OOTF
/// (γ=1.2) — the Apple-published / BBC-R&D recipe for HLG → BT.709
/// conversion.
///
/// The OOTF is the load-bearing piece: HLG signals are SCENE-referred
/// (the encoded value is the camera's view of light, not a display
/// luminance). Without applying the OOTF, the tonemap operates on
/// raw scene values and midtones land in the wrong place — iPhone
/// HLG content famously reads as ~1 stop too bright on every
/// generic HDR-passthrough or naive-tonemap pipeline because their
/// camera assumes Apple's downstream tonemapper handles the
/// scene→display transform.
///
/// Apple's documented gamma for SDR target: γ=1.2 (per "HDR Editing
/// Best Practices in iOS / macOS", WWDC 2020 + ARIB STD-B67 §3.3).
/// We apply per-channel for simplicity (the "constant luminance"
/// version uses Y_s = max(R,G,B) as the base; per-channel is what
/// most consumer HLG decoders ship and is accurate enough for
/// social-media playback).
///
/// Returns scene-linear-OOTF'd in the same 1.0=100-nit-SDR-white
/// frame as PQ so downstream tonemap math is uniform.
#[inline(always)]
fn hlg_to_linear(e: f32) -> f32 {
    const A: f32 = 0.17883277;
    const B: f32 = 1.0 - 4.0 * A;
    // c = 0.5 - a * ln(4a). Hardcoded so we don't pay for a runtime ln().
    const C: f32 = 0.559_910_7;
    /// SDR-target system gamma per BBC R&D / Apple HLG → BT.709 spec.
    const HLG_OOTF_GAMMA: f32 = 1.2;

    let e = e.max(0.0);
    // Step 1: inverse OETF — encoded HLG value → scene-linear (0..1
    // where 1.0 is the HLG peak, typically interpreted as 1000 nits
    // on a reference display).
    let scene_lin = if e <= 0.5 {
        (e * e) / 3.0
    } else {
        ((((e - C) / A).exp()) + B) / 12.0
    };
    // Step 2: OOTF — scene-linear → display-linear with γ=1.2 for
    // SDR target. Per-channel approximation. Naturally compresses
    // the iPhone "1-stop bright" overshoot since values >1 raised
    // to 1.2 expand and then get clipped by Hable's max_white.
    let display_lin = scene_lin.powf(HLG_OOTF_GAMMA);
    // Step 3: rescale to the 1.0=100-nit-SDR-white frame the tonemap
    // expects. HLG peak (1.0 → after OOTF still 1.0) maps to 10.0
    // here, same as PQ's 1000-nit reference.
    display_lin * 10.0
}

#[inline(always)]
fn dispatch_eotf(transfer: TransferFn, encoded: f32) -> f32 {
    match transfer {
        TransferFn::St2084 => pq_to_linear(encoded),
        TransferFn::AribStdB67 => hlg_to_linear(encoded),
        // Defensive: a non-HDR transfer reaching this path is a caller
        // bug — we've gated dispatch on `is_hdr` upstream. Treat as
        // identity rather than panicking so partial bugs don't take
        // out playback.
        _ => encoded.max(0.0),
    }
}

// ── tonemap (Hable filmic) ────────────────────────────────────────────

/// Uncharted 2 partial — the building block of Hable's filmic curve.
/// Numbers are Hable's published coefficients verbatim.
#[inline(always)]
fn hable_partial(x: f32) -> f32 {
    const A: f32 = 0.15;
    const B: f32 = 0.50;
    const C: f32 = 0.10;
    const D: f32 = 0.20;
    const E: f32 = 0.02;
    const F: f32 = 0.30;
    ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
}

/// Hable filmic tonemap. Input is scene-linear (1.0 = SDR diffuse
/// white reference). `max_white` is the scene-linear value that should
/// map to display white (1.0 SDR-linear out) — typically the source's
/// MaxCLL or the master display max luminance, divided by 100.
#[inline(always)]
fn hable_tonemap(x: f32, max_white: f32) -> f32 {
    // 2.0 exposure bias (Hable's recommended default — gives the toe
    // a film-stock feel and lifts midtones slightly).
    const EXPOSURE: f32 = 2.0;
    let curr = hable_partial(x * EXPOSURE);
    let scale = 1.0 / hable_partial(max_white * EXPOSURE);
    (curr * scale).clamp(0.0, 1.0)
}

// ── BT.709 OETF (linear → gamma-encoded) ──────────────────────────────

#[inline(always)]
fn bt709_oetf(l: f32) -> f32 {
    let l = l.clamp(0.0, 1.0);
    if l < 0.018 {
        4.5 * l
    } else {
        1.099 * l.powf(0.45) - 0.099
    }
}

// ── matrix coefficients ───────────────────────────────────────────────

/// BT.2020 NCL Y'CbCr → R'G'B' (still in encoded transfer function
/// domain). Cb / Cr inputs are normalised to [-0.5, 0.5].
#[inline(always)]
fn yuv2020ncl_to_rgb(y: f32, cb: f32, cr: f32) -> (f32, f32, f32) {
    // Kr = 0.2627, Kb = 0.0593, Kg = 1 - Kr - Kb = 0.6780.
    let r = y + 1.4746 * cr;
    let g = y - 0.16455 * cb - 0.57135 * cr;
    let b = y + 1.8814 * cb;
    (r, g, b)
}

/// Linear RGB BT.2020 → Linear RGB BT.709 (D65 white-point matched).
/// Negative coefficients are intentional — gamut conversion can produce
/// out-of-gamut values which we clip on the OETF input side.
#[inline(always)]
fn rgb2020_to_rgb709_linear(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let r_out = 1.66049 * r - 0.58764 * g - 0.07285 * b;
    let g_out = -0.12455 * r + 1.13290 * g - 0.01006 * b;
    let b_out = -0.01815 * r - 0.10058 * g + 1.11873 * b;
    (r_out, g_out, b_out)
}

/// R'G'B' BT.709 (gamma) → Y'CbCr 8-bit limited range.
/// Output triplet is (y, cb, cr) ∈ [16..235], [16..240], [16..240].
#[inline(always)]
fn rgb709_to_yuv709_limited(r: f32, g: f32, b: f32) -> (u8, u8, u8) {
    // Kr = 0.2126, Kb = 0.0722.
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let cb = (b - y) / 1.8556;
    let cr = (r - y) / 1.5748;
    let y8 = (y * 219.0 + 16.0).round().clamp(16.0, 235.0) as u8;
    let cb8 = (cb * 224.0 + 128.0).round().clamp(16.0, 240.0) as u8;
    let cr8 = (cr * 224.0 + 128.0).round().clamp(16.0, 240.0) as u8;
    (y8, cb8, cr8)
}

// ── chroma desub: 10-bit Y'CbCr code → normalised float ───────────────

const Y_BLACK_10: f32 = 64.0; // 16 << 2
const Y_RANGE_10: f32 = 876.0; // (235 - 16) << 2
const C_NEUTRAL_10: f32 = 512.0; // 128 << 2
const C_HALFRANGE_10: f32 = 448.0; // 224/2 << 2

#[inline(always)]
fn y10_to_normalised(y: u16) -> f32 {
    (y as f32 - Y_BLACK_10) / Y_RANGE_10
}

#[inline(always)]
fn c10_to_normalised(c: u16) -> f32 {
    (c as f32 - C_NEUTRAL_10) / (C_HALFRANGE_10 * 2.0)
}

// ── public entry ──────────────────────────────────────────────────────

/// Default scene-linear white point when the source carries no
/// mastering display metadata. Picked to match a typical HDR10 master
/// at 1000-nit peak — most consumer HDR content. Sources tagged with
/// `mastering_display.max_luminance` use that exact value instead.
const DEFAULT_MAX_WHITE_NITS: f32 = 1000.0;

/// HDR → SDR tonemap.
///
/// Input must be `Yuv420p10le` (BT.2020 NCL is assumed; CL would need
/// a different matrix). Output is `Yuv420p` (8-bit, BT.709 limited).
///
/// `transfer` selects the EOTF (PQ vs HLG). `max_white_nits` is the
/// scene-linear white point used to scale the Hable curve — pass the
/// source's mastering-display `max_luminance` (in cd/m²) when present;
/// otherwise `DEFAULT_MAX_WHITE_NITS`.
///
/// Implementation: per-pixel Y conversion at full resolution; chroma
/// downsampled by averaging the 2×2 luma-area RGB output back into a
/// single (cb, cr) per chroma sample. This is more expensive than a
/// "tonemap once per chroma block" approach but avoids the hue shifts
/// that can show up at high luminance on subsampled-tonemap output.
pub fn tonemap_yuv420p10le_bt2020_to_yuv420p_bt709(
    src: &VideoFrame,
    transfer: TransferFn,
    max_white_nits: Option<f32>,
) -> Result<VideoFrame> {
    if !matches!(src.format, PixelFormat::Yuv420p10le) {
        bail!(
            "tonemap_yuv420p10le_bt2020_to_yuv420p_bt709 expects Yuv420p10le; got {:?}",
            src.format
        );
    }
    let w = src.width as usize;
    let h = src.height as usize;
    if w == 0 || h == 0 || (w & 1) != 0 || (h & 1) != 0 {
        bail!("tonemap requires non-zero even dimensions; got {}x{}", w, h);
    }

    let max_white = (max_white_nits.unwrap_or(DEFAULT_MAX_WHITE_NITS) / 100.0).max(1.0);

    let y_plane_bytes = w * h * 2;
    let c_plane_bytes = (w / 2) * (h / 2) * 2;
    if src.data.len() < y_plane_bytes + 2 * c_plane_bytes {
        bail!(
            "Yuv420p10le frame too small for {}x{}: need {} bytes, got {}",
            w,
            h,
            y_plane_bytes + 2 * c_plane_bytes,
            src.data.len()
        );
    }

    // Reinterpret the byte slice as u16 LE planes. Endianness assumed
    // little — every host we ship to is x86_64 / aarch64 LE; a future
    // BE platform would need byteswap helpers here.
    let bytes = src.data.as_ref();
    let y_plane: &[u16] =
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u16, w * h) };
    let cb_plane: &[u16] = unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr().add(y_plane_bytes) as *const u16,
            (w / 2) * (h / 2),
        )
    };
    let cr_plane: &[u16] = unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr().add(y_plane_bytes + c_plane_bytes) as *const u16,
            (w / 2) * (h / 2),
        )
    };

    let mut out_y = vec![0u8; w * h];
    let mut out_cb = vec![0u8; (w / 2) * (h / 2)];
    let mut out_cr = vec![0u8; (w / 2) * (h / 2)];

    // Walk in 2x2 blocks so we can downsample the chroma in lockstep.
    for by in 0..(h / 2) {
        for bx in 0..(w / 2) {
            let cb_n = c10_to_normalised(cb_plane[by * (w / 2) + bx]);
            let cr_n = c10_to_normalised(cr_plane[by * (w / 2) + bx]);

            let mut acc_cb = 0.0_f32;
            let mut acc_cr = 0.0_f32;

            for dy in 0..2 {
                for dx in 0..2 {
                    let yi = by * 2 + dy;
                    let xi = bx * 2 + dx;
                    let y_n = y10_to_normalised(y_plane[yi * w + xi]);

                    // 1. BT.2020 NCL Y'CbCr → R'G'B' (still gamma).
                    let (r_g, g_g, b_g) = yuv2020ncl_to_rgb(y_n, cb_n, cr_n);

                    // 2. EOTF inverse: gamma → scene linear (1.0 = SDR diffuse).
                    let r_lin = dispatch_eotf(transfer, r_g);
                    let g_lin = dispatch_eotf(transfer, g_g);
                    let b_lin = dispatch_eotf(transfer, b_g);

                    // 3. Gamut convert: linear BT.2020 → linear BT.709.
                    let (r709, g709, b709) = rgb2020_to_rgb709_linear(r_lin, g_lin, b_lin);

                    // 4. Tonemap each channel (per-channel preserves
                    //    saturation better than luminance-only at the
                    //    cost of slightly less perceptually uniform
                    //    response — Hable's published recipe uses
                    //    per-channel).
                    let r_tm = hable_tonemap(r709, max_white);
                    let g_tm = hable_tonemap(g709, max_white);
                    let b_tm = hable_tonemap(b709, max_white);

                    // 5. OETF: linear → BT.709 gamma encoded.
                    let r_o = bt709_oetf(r_tm);
                    let g_o = bt709_oetf(g_tm);
                    let b_o = bt709_oetf(b_tm);

                    // 6. RGB → Y'CbCr 8-bit BT.709 limited.
                    let (y8, cb8, cr8) = rgb709_to_yuv709_limited(r_o, g_o, b_o);
                    out_y[yi * w + xi] = y8;
                    acc_cb += cb8 as f32;
                    acc_cr += cr8 as f32;
                }
            }

            // Downsample chroma: average the 4 per-pixel chroma values
            // back to one sample per 2x2 block (4:2:0 layout).
            out_cb[by * (w / 2) + bx] = (acc_cb * 0.25).round() as u8;
            out_cr[by * (w / 2) + bx] = (acc_cr * 0.25).round() as u8;
        }
    }

    let mut out = Vec::with_capacity(w * h + 2 * (w / 2) * (h / 2));
    out.extend_from_slice(&out_y);
    out.extend_from_slice(&out_cb);
    out.extend_from_slice(&out_cr);

    Ok(VideoFrame::new(
        Bytes::from(out),
        src.width,
        src.height,
        PixelFormat::Yuv420p,
        ColorSpace::Bt709,
        src.pts,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_solid_yuv420p10le(w: u32, h: u32, y10: u16, cb10: u16, cr10: u16) -> VideoFrame {
        let mut bytes = Vec::with_capacity((w * h * 2 + 2 * (w / 2) * (h / 2) * 2) as usize);
        for _ in 0..(w * h) {
            bytes.extend_from_slice(&y10.to_le_bytes());
        }
        for _ in 0..((w / 2) * (h / 2)) {
            bytes.extend_from_slice(&cb10.to_le_bytes());
        }
        for _ in 0..((w / 2) * (h / 2)) {
            bytes.extend_from_slice(&cr10.to_le_bytes());
        }
        VideoFrame::new(
            Bytes::from(bytes),
            w,
            h,
            PixelFormat::Yuv420p10le,
            ColorSpace::Bt2020,
            0,
        )
    }

    #[test]
    fn tonemap_solid_pq_black_yields_sdr_black() {
        // 10-bit limited-range black: Y=64, Cb=Cr=512.
        let src = make_solid_yuv420p10le(16, 16, 64, 512, 512);
        let out = tonemap_yuv420p10le_bt2020_to_yuv420p_bt709(&src, TransferFn::St2084, None)
            .expect("tonemap");
        assert_eq!(out.format, PixelFormat::Yuv420p);
        assert_eq!(out.color_space, ColorSpace::Bt709);
        let y = out.data[0];
        let cb = out.data[16 * 16];
        let cr = out.data[16 * 16 + 8 * 8];
        // Black should map to BT.709 limited black: Y≈16, Cb≈Cr≈128.
        assert!((y as i32 - 16).abs() <= 1, "Y near 16, got {}", y);
        assert!((cb as i32 - 128).abs() <= 1, "Cb near 128, got {}", cb);
        assert!((cr as i32 - 128).abs() <= 1, "Cr near 128, got {}", cr);
    }

    #[test]
    fn tonemap_solid_pq_white_clipped_under_one() {
        // 10-bit PQ peak: Y=940 (limited-range white).
        let src = make_solid_yuv420p10le(16, 16, 940, 512, 512);
        let out =
            tonemap_yuv420p10le_bt2020_to_yuv420p_bt709(&src, TransferFn::St2084, Some(1000.0))
                .expect("tonemap");
        let y = out.data[0];
        // PQ "white" code corresponds to 10,000 nits absolute. At
        // max_white=1000 nits, that's 10x overrange — Hable curve
        // saturates near 1.0, OETF gives ~235 limited-range. Allow
        // a small numerical margin.
        assert!(y >= 200, "PQ peak should map near limited-white; got {}", y);
        assert!(y <= 235, "limited-range upper bound 235, got {}", y);
    }

    #[test]
    fn tonemap_solid_pq_midgrey_yields_lifted_midgrey() {
        // PQ encoded ~50% (midpoint code 0.5 → ~92 nits → ~1.0 in
        // SDR-linear-1.0=100-nits frame). Hable with exposure=2 lifts
        // this above linear 0.5 → BT.709 OETF gives a code well above
        // the limited-range mid (Y≈126).
        let y10 = ((0.5 * Y_RANGE_10) + Y_BLACK_10) as u16;
        let src = make_solid_yuv420p10le(16, 16, y10, 512, 512);
        let out = tonemap_yuv420p10le_bt2020_to_yuv420p_bt709(&src, TransferFn::St2084, None)
            .expect("tonemap");
        let y = out.data[0];
        assert!(
            (130..=210).contains(&y),
            "PQ ~92 nits should land in upper-mid limited range, got {}",
            y
        );
    }

    #[test]
    fn tonemap_hlg_path_runs() {
        // Smoke: HLG black should map near limited-range black.
        let src = make_solid_yuv420p10le(8, 8, 64, 512, 512);
        let out = tonemap_yuv420p10le_bt2020_to_yuv420p_bt709(&src, TransferFn::AribStdB67, None)
            .expect("tonemap HLG");
        assert!((out.data[0] as i32 - 16).abs() <= 1);
    }

    #[test]
    fn tonemap_rejects_wrong_format() {
        let src = VideoFrame::new(
            Bytes::from(vec![0u8; 96]),
            8,
            8,
            PixelFormat::Yuv420p,
            ColorSpace::Bt709,
            0,
        );
        let err = tonemap_yuv420p10le_bt2020_to_yuv420p_bt709(&src, TransferFn::St2084, None)
            .expect_err("must reject 8-bit input");
        assert!(format!("{:?}", err).contains("Yuv420p10le"));
    }

    #[test]
    fn pq_eotf_monotonic() {
        // Sanity: EOTF must be monotonically increasing.
        let mut last = -1.0;
        for i in 0..=100 {
            let v = pq_to_linear(i as f32 / 100.0);
            assert!(v >= last, "non-monotonic at {}: {} < {}", i, v, last);
            last = v;
        }
    }

    #[test]
    fn hable_tonemap_clamps_to_unit() {
        // Inputs above max_white should clamp to <= 1.0.
        for x in [0.0, 1.0, 5.0, 50.0, 500.0_f32] {
            let v = hable_tonemap(x, 10.0);
            assert!(v >= 0.0 && v <= 1.0, "out of range at x={}: {}", x, v);
        }
    }

    #[test]
    fn bt709_oetf_inverts_neutral_grey() {
        // Reference values from ITU-R BT.709 §1.2:
        //   E' = 4.5 * E                       for 0 ≤ E < 0.018
        //   E' = 1.099 * E^0.45 - 0.099        for 0.018 ≤ E ≤ 1
        // At E = 0.5: 1.099 * 0.5^0.45 - 0.099 ≈ 0.7055.
        // At E = 1.0: 1.099 * 1.0 - 0.099 = 1.000.
        // (Earlier this test asserted 0.7398, which is the sRGB EOTF
        // value — different transfer function, different formula. The
        // BT.709 number is materially lower at mid-grey.)
        assert!((bt709_oetf(0.5) - 0.7055).abs() < 0.01);
        assert!((bt709_oetf(1.0) - 1.0).abs() < 0.01);
    }
}
