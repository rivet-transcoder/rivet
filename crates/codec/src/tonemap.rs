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
//! Two implementations of one per-pixel function. The scalar f32 path is
//! the reference: it is what the numbers in this file's comments were
//! derived against and what the tests hand-check. The AVX2 + FMA path
//! ([`tonemap_yuv420p10le_bt2020_to_yuv420p_bt709_avx2`]) does the same
//! arithmetic eight pixels at a time; its transcendentals (`powf`, `exp`)
//! are the Cephes single-precision `log` / `exp` polynomials ([`simd`]),
//! good to a few ulp, so the two paths can disagree by one 8-bit code on a
//! pixel whose value sits at a rounding boundary and never by more — the
//! tolerance is **≤ 1 LSB per output sample**, checked over the whole
//! 10-bit ramp for PQ and HLG and reported for a real clip by
//! `examples/tonemap_ab.rs`. The dispatcher picks AVX2 when the CPU has it;
//! `RIVET_TONEMAP_SCALAR=1` forces the reference, which is how the two are
//! timed against each other in one binary.

use anyhow::{Result, bail};
use bytes::Bytes;

use crate::frame::{ColorSpace, PixelFormat, TransferFn, VideoFrame};

// ── transfer (EOTF inverse: encoded → scene linear) ───────────────────

/// PQ constants (SMPTE ST.2084 §5.1), shared by both paths.
const PQ_M1_INV: f32 = 1.0 / 0.159_301_76;
const PQ_M2_INV: f32 = 1.0 / 78.84375;
const PQ_C1: f32 = 0.8359375;
const PQ_C2: f32 = 18.851_563;
const PQ_C3: f32 = 18.6875;

/// PQ inverse EOTF (SMPTE ST.2084).
///
/// Returns scene-linear in units where `1.0 = 100 cd/m² SDR diffuse white`,
/// so `100.0 = 10,000 nits PQ peak`. Tonemap operates in the same scene-
/// linear frame.
#[inline(always)]
fn pq_to_linear(n: f32) -> f32 {
    // The EOTF is defined on [0, 1]. A value above 1 is not a brighter
    // pixel, it is matrix overshoot from an out-of-gamut Y'CbCr triple
    // (near-peak luma with chroma at the range end), and just past 1 the
    // curve's denominator crosses zero: the reference used to return a
    // value that overflowed the Hable curve to NaN, which the final
    // `as u8` turned into Y = 0 — below limited-range black. Clamp to the
    // domain and such a pixel maps to peak white, as it should.
    let np = n.clamp(0.0, 1.0).powf(PQ_M2_INV);
    let num = (np - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * np;
    if den <= 0.0 {
        return 0.0;
    }
    let lin01 = (num / den).powf(PQ_M1_INV); // 0..1, 1.0 = 10,000 nits
    lin01 * 100.0 // rescale so 1.0 = SDR diffuse white (~100 nits)
}

/// HLG constants (ARIB STD-B67), shared by both paths.
const HLG_A: f32 = 0.17883277;
const HLG_B: f32 = 1.0 - 4.0 * HLG_A;
// c = 0.5 - a * ln(4a). Hardcoded so we don't pay for a runtime ln().
const HLG_C: f32 = 0.559_910_7;
/// SDR-target system gamma per BBC R&D / Apple HLG → BT.709 spec.
const HLG_OOTF_GAMMA: f32 = 1.2;

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
    // Signal domain is [0, 1] (see `pq_to_linear` for why overshoot is
    // clamped rather than extrapolated).
    let e = e.clamp(0.0, 1.0);
    // Step 1: inverse OETF — encoded HLG value → scene-linear (0..1
    // where 1.0 is the HLG peak, typically interpreted as 1000 nits
    // on a reference display).
    let scene_lin = if e <= 0.5 {
        (e * e) / 3.0
    } else {
        ((((e - HLG_C) / HLG_A).exp()) + HLG_B) / 12.0
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

/// Hable's published coefficients verbatim, shared by both paths.
const HABLE_A: f32 = 0.15;
const HABLE_B: f32 = 0.50;
const HABLE_C: f32 = 0.10;
const HABLE_D: f32 = 0.20;
const HABLE_E: f32 = 0.02;
const HABLE_F: f32 = 0.30;
/// 2.0 exposure bias (Hable's recommended default — gives the toe
/// a film-stock feel and lifts midtones slightly).
const HABLE_EXPOSURE: f32 = 2.0;

/// Uncharted 2 partial — the building block of Hable's filmic curve.
#[inline(always)]
fn hable_partial(x: f32) -> f32 {
    ((x * (HABLE_A * x + HABLE_C * HABLE_B) + HABLE_D * HABLE_E)
        / (x * (HABLE_A * x + HABLE_B) + HABLE_D * HABLE_F))
        - HABLE_E / HABLE_F
}

/// Hable filmic tonemap. Input is scene-linear (1.0 = SDR diffuse
/// white reference). `max_white` is the scene-linear value that should
/// map to display white (1.0 SDR-linear out) — typically the source's
/// MaxCLL or the master display max luminance, divided by 100.
#[inline(always)]
fn hable_tonemap(x: f32, max_white: f32) -> f32 {
    // Gamut clip first. The BT.2020 → BT.709 matrix sends an out-of-gamut
    // colour to a negative channel, and Hable's rational curve has a pole
    // just below zero (its denominator `x(Ax + B) + DF` vanishes at
    // x ≈ −0.062): a negative input lands anywhere from black to white
    // depending on the last bit of rounding. The reference used to clip
    // only at the OETF, after the curve — a pixel with B'709 < 0 came out
    // *white*. Clipping here makes such a channel black, which is what
    // "out of gamut, clip" means, and leaves the curve well-conditioned
    // (denominator ≥ DF) for the SIMD path to match.
    let x = x.max(0.0);
    let curr = hable_partial(x * HABLE_EXPOSURE);
    let scale = 1.0 / hable_partial(max_white * HABLE_EXPOSURE);
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
/// out-of-gamut (negative) values, which `hable_tonemap` clips to zero
/// before the curve.
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

/// The whole per-pixel chain, scalar: normalised Y'CbCr in, 8-bit BT.709
/// limited Y'CbCr out. Steps 1–6 of the module doc.
#[inline(always)]
fn tonemap_pixel_scalar(
    y_n: f32,
    cb_n: f32,
    cr_n: f32,
    transfer: TransferFn,
    max_white: f32,
) -> (u8, u8, u8) {
    // 1. BT.2020 NCL Y'CbCr → R'G'B' (still gamma).
    let (r_g, g_g, b_g) = yuv2020ncl_to_rgb(y_n, cb_n, cr_n);

    // 2. EOTF inverse: gamma → scene linear (1.0 = SDR diffuse).
    let r_lin = dispatch_eotf(transfer, r_g);
    let g_lin = dispatch_eotf(transfer, g_g);
    let b_lin = dispatch_eotf(transfer, b_g);

    // 3. Gamut convert: linear BT.2020 → linear BT.709.
    let (r709, g709, b709) = rgb2020_to_rgb709_linear(r_lin, g_lin, b_lin);

    // 4. Tonemap each channel (per-channel preserves saturation better
    //    than luminance-only at the cost of slightly less perceptually
    //    uniform response — Hable's published recipe uses per-channel).
    let r_tm = hable_tonemap(r709, max_white);
    let g_tm = hable_tonemap(g709, max_white);
    let b_tm = hable_tonemap(b709, max_white);

    // 5. OETF: linear → BT.709 gamma encoded.
    let r_o = bt709_oetf(r_tm);
    let g_o = bt709_oetf(g_tm);
    let b_o = bt709_oetf(b_tm);

    // 6. RGB → Y'CbCr 8-bit BT.709 limited.
    rgb709_to_yuv709_limited(r_o, g_o, b_o)
}

// ── public entry ──────────────────────────────────────────────────────

/// Default scene-linear white point when the source carries no
/// mastering display metadata. Picked to match a typical HDR10 master
/// at 1000-nit peak — most consumer HDR content. Sources tagged with
/// `mastering_display.max_luminance` use that exact value instead.
const DEFAULT_MAX_WHITE_NITS: f32 = 1000.0;

/// Validated, borrowed view of a `Yuv420p10le` frame's three planes.
struct Planes10<'a> {
    w: usize,
    h: usize,
    y: &'a [u16],
    cb: &'a [u16],
    cr: &'a [u16],
}

fn planes_10(src: &VideoFrame) -> Result<Planes10<'_>> {
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
    // BE platform would need byteswap helpers here. `Bytes` gives no
    // alignment promise, so this is only sound as unaligned reads; the
    // scalar path indexes (the compiler emits unaligned loads for u16
    // on every target we build) and the AVX2 path uses `loadu`.
    let bytes = src.data.as_ref();
    let y: &[u16] = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u16, w * h) };
    let cb: &[u16] = unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr().add(y_plane_bytes) as *const u16,
            (w / 2) * (h / 2),
        )
    };
    let cr: &[u16] = unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr().add(y_plane_bytes + c_plane_bytes) as *const u16,
            (w / 2) * (h / 2),
        )
    };
    Ok(Planes10 { w, h, y, cb, cr })
}

fn max_white_for(max_white_nits: Option<f32>) -> f32 {
    (max_white_nits.unwrap_or(DEFAULT_MAX_WHITE_NITS) / 100.0).max(1.0)
}

fn pack_frame(src: &VideoFrame, out_y: Vec<u8>, out_cb: Vec<u8>, out_cr: Vec<u8>) -> VideoFrame {
    let mut out = Vec::with_capacity(out_y.len() + out_cb.len() + out_cr.len());
    out.extend_from_slice(&out_y);
    out.extend_from_slice(&out_cb);
    out.extend_from_slice(&out_cr);
    VideoFrame::new(
        Bytes::from(out),
        src.width,
        src.height,
        PixelFormat::Yuv420p,
        ColorSpace::Bt709,
        src.pts,
    )
}

/// Whether `RIVET_TONEMAP_SCALAR` asks for the reference path. Read once.
fn scalar_forced() -> bool {
    static FORCED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCED.get_or_init(|| {
        std::env::var("RIVET_TONEMAP_SCALAR")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
            .unwrap_or(false)
    })
}

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
///
/// Runtime-dispatched: AVX2 + FMA when the CPU has them (and
/// `RIVET_TONEMAP_SCALAR` is unset), else the scalar reference.
pub fn tonemap_yuv420p10le_bt2020_to_yuv420p_bt709(
    src: &VideoFrame,
    transfer: TransferFn,
    max_white_nits: Option<f32>,
) -> Result<VideoFrame> {
    let use_avx2 = cfg!(any(target_arch = "x86", target_arch = "x86_64"))
        && !scalar_forced()
        && avx2_fma_available();
    static ANNOUNCED: std::sync::Once = std::sync::Once::new();
    ANNOUNCED.call_once(|| {
        tracing::info!(
            path = if use_avx2 { "avx2+fma" } else { "scalar" },
            forced_scalar = scalar_forced(),
            "HDR → SDR tonemap kernel selected"
        );
    });
    if use_avx2 {
        return tonemap_yuv420p10le_bt2020_to_yuv420p_bt709_avx2(src, transfer, max_white_nits);
    }
    tonemap_yuv420p10le_bt2020_to_yuv420p_bt709_scalar(src, transfer, max_white_nits)
}

fn avx2_fma_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

/// The scalar f32 reference. Same contract as
/// [`tonemap_yuv420p10le_bt2020_to_yuv420p_bt709`].
pub fn tonemap_yuv420p10le_bt2020_to_yuv420p_bt709_scalar(
    src: &VideoFrame,
    transfer: TransferFn,
    max_white_nits: Option<f32>,
) -> Result<VideoFrame> {
    let p = planes_10(src)?;
    let (w, h) = (p.w, p.h);
    let max_white = max_white_for(max_white_nits);

    let mut out_y = vec![0u8; w * h];
    let mut out_cb = vec![0u8; (w / 2) * (h / 2)];
    let mut out_cr = vec![0u8; (w / 2) * (h / 2)];

    // Walk in 2x2 blocks so we can downsample the chroma in lockstep.
    for by in 0..(h / 2) {
        for bx in 0..(w / 2) {
            let cb_n = c10_to_normalised(p.cb[by * (w / 2) + bx]);
            let cr_n = c10_to_normalised(p.cr[by * (w / 2) + bx]);

            let mut acc_cb = 0.0_f32;
            let mut acc_cr = 0.0_f32;

            for dy in 0..2 {
                for dx in 0..2 {
                    let yi = by * 2 + dy;
                    let xi = bx * 2 + dx;
                    let y_n = y10_to_normalised(p.y[yi * w + xi]);
                    let (y8, cb8, cr8) = tonemap_pixel_scalar(y_n, cb_n, cr_n, transfer, max_white);
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

    Ok(pack_frame(src, out_y, out_cb, out_cr))
}

/// The AVX2 + FMA path. Same contract as
/// [`tonemap_yuv420p10le_bt2020_to_yuv420p_bt709`]; agrees with the scalar
/// reference to within one 8-bit code per sample (see the module doc).
/// Falls back to the scalar reference on a CPU without AVX2 + FMA.
pub fn tonemap_yuv420p10le_bt2020_to_yuv420p_bt709_avx2(
    src: &VideoFrame,
    transfer: TransferFn,
    max_white_nits: Option<f32>,
) -> Result<VideoFrame> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let p = planes_10(src)?;
            let max_white = max_white_for(max_white_nits);
            let (out_y, out_cb, out_cr) =
                // SAFETY: avx2 + fma runtime-detected above; the planes were
                // bounds-checked by `planes_10`.
                unsafe { simd::tonemap_planes_avx2(&p, transfer, max_white) };
            return Ok(pack_frame(src, out_y, out_cb, out_cr));
        }
    }
    tonemap_yuv420p10le_bt2020_to_yuv420p_bt709_scalar(src, transfer, max_white_nits)
}

// ── AVX2 + FMA kernel ─────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod simd {
    //! Eight pixels per vector, one 2-row × 16-column strip per iteration
    //! (8 chroma sites), scalar tail for the right edge.
    //!
    //! `exp` / `log` are the Cephes `expf` / `logf` polynomials as
    //! vectorised in Pommier's `sse_mathfun` / Bloch's `avx_mathfun`
    //! (zlib licence, rewritten here in Rust intrinsics): ~1–2 ulp over
    //! the ranges this kernel uses. `pow(x, p) = exp(p · log(x))` with
    //! `x ≤ 0 → 0`, which is what every `powf` call in the scalar path
    //! sees (inputs are clamped non-negative first).

    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use super::{
        C_HALFRANGE_10, C_NEUTRAL_10, HABLE_A, HABLE_B, HABLE_C, HABLE_D, HABLE_E, HABLE_EXPOSURE,
        HABLE_F, HLG_A, HLG_B, HLG_C, HLG_OOTF_GAMMA, PQ_C1, PQ_C2, PQ_C3, PQ_M1_INV, PQ_M2_INV,
        Planes10, Y_BLACK_10, Y_RANGE_10, c10_to_normalised, hable_partial, tonemap_pixel_scalar,
        y10_to_normalised,
    };
    use crate::frame::TransferFn;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn exp_ps(x: __m256) -> __m256 {
        {
            let x = _mm256_min_ps(_mm256_set1_ps(88.376_26), x);
            let x = _mm256_max_ps(_mm256_set1_ps(-88.376_26), x);
            // fx = round(x · log2 e)
            let fx = _mm256_fmadd_ps(x, _mm256_set1_ps(1.442_695), _mm256_set1_ps(0.5));
            let fx = _mm256_floor_ps(fx);
            // x -= fx · ln 2 (split in two for precision)
            let x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693_359_4), x);
            let x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.121_944_4e-4), x);
            let z = _mm256_mul_ps(x, x);
            let mut y = _mm256_set1_ps(1.987_569_2e-4);
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.398_2e-3));
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.333_452e-3));
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.166_579_6e-2));
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.666_666_5e-1));
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.000_000_1e-1));
            y = _mm256_fmadd_ps(y, z, x);
            y = _mm256_add_ps(y, _mm256_set1_ps(1.0));
            // 2^fx by building the exponent field.
            let emm0 = _mm256_cvttps_epi32(fx);
            let emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
            let emm0 = _mm256_slli_epi32(emm0, 23);
            _mm256_mul_ps(y, _mm256_castsi256_ps(emm0))
        }
    }

    /// Natural log for x > 0. x ≤ 0 yields NaN (Cephes semantics); the
    /// caller masks those lanes.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn log_ps(x: __m256) -> __m256 {
        {
            let invalid = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);
            // Smallest normal, so denormals do not break the exponent split.
            let x = _mm256_max_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x0080_0000)));
            let emm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
            // Keep the mantissa, force the exponent to that of 0.5.
            let x = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(!0x7f80_0000u32 as i32)));
            let x = _mm256_or_ps(x, _mm256_set1_ps(0.5));
            let emm0 = _mm256_sub_epi32(emm0, _mm256_set1_epi32(0x7f));
            let e = _mm256_add_ps(_mm256_cvtepi32_ps(emm0), _mm256_set1_ps(1.0));
            // If x < 1/√2: e -= 1, x = 2x - 1; else x = x - 1.
            let mask = _mm256_cmp_ps(x, _mm256_set1_ps(0.707_106_77), _CMP_LT_OS);
            let tmp = _mm256_and_ps(x, mask);
            let x = _mm256_sub_ps(x, _mm256_set1_ps(1.0));
            let e = _mm256_sub_ps(e, _mm256_and_ps(_mm256_set1_ps(1.0), mask));
            let x = _mm256_add_ps(x, tmp);
            let z = _mm256_mul_ps(x, x);
            let mut y = _mm256_set1_ps(7.037_683_6e-2);
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-1.151_461e-1));
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.167_699_9e-1));
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-1.242_014_1e-1));
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.424_932_3e-1));
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-1.666_805_8e-1));
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(2.000_071_5e-1));
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(-2.499_999_4e-1));
            y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(3.333_333_1e-1));
            y = _mm256_mul_ps(y, x);
            y = _mm256_mul_ps(y, z);
            y = _mm256_fmadd_ps(e, _mm256_set1_ps(-2.121_944_4e-4), y);
            y = _mm256_fnmadd_ps(z, _mm256_set1_ps(0.5), y);
            let x = _mm256_add_ps(x, y);
            let x = _mm256_fmadd_ps(e, _mm256_set1_ps(0.693_359_4), x);
            _mm256_or_ps(x, invalid)
        }
    }

    /// `x^p` for x ≥ 0 (x ≤ 0 → 0), p a broadcast constant.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn pow_ps(x: __m256, p: __m256) -> __m256 {
        unsafe {
            let positive = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_GT_OS);
            let r = exp_ps(_mm256_mul_ps(p, log_ps(x)));
            _mm256_and_ps(r, positive)
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn pq_to_linear_ps(n: __m256) -> __m256 {
        unsafe {
            let n = _mm256_min_ps(_mm256_max_ps(n, _mm256_setzero_ps()), _mm256_set1_ps(1.0));
            let np = pow_ps(n, _mm256_set1_ps(PQ_M2_INV));
            let num = _mm256_max_ps(_mm256_sub_ps(np, _mm256_set1_ps(PQ_C1)), _mm256_setzero_ps());
            let den = _mm256_fnmadd_ps(_mm256_set1_ps(PQ_C3), np, _mm256_set1_ps(PQ_C2));
            // den ≤ 0 → 0 (the scalar early return). Divide anyway, then mask.
            let den_ok = _mm256_cmp_ps(den, _mm256_setzero_ps(), _CMP_GT_OS);
            let lin01 = pow_ps(_mm256_div_ps(num, den), _mm256_set1_ps(PQ_M1_INV));
            let lin01 = _mm256_and_ps(lin01, den_ok);
            _mm256_mul_ps(lin01, _mm256_set1_ps(100.0))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hlg_to_linear_ps(e: __m256) -> __m256 {
        unsafe {
            let e = _mm256_min_ps(_mm256_max_ps(e, _mm256_setzero_ps()), _mm256_set1_ps(1.0));
            let low = _mm256_div_ps(_mm256_mul_ps(e, e), _mm256_set1_ps(3.0));
            let t = _mm256_div_ps(_mm256_sub_ps(e, _mm256_set1_ps(HLG_C)), _mm256_set1_ps(HLG_A));
            let high = _mm256_div_ps(_mm256_add_ps(exp_ps(t), _mm256_set1_ps(HLG_B)), _mm256_set1_ps(12.0));
            let use_low = _mm256_cmp_ps(e, _mm256_set1_ps(0.5), _CMP_LE_OS);
            let scene = _mm256_blendv_ps(high, low, use_low);
            let display = pow_ps(scene, _mm256_set1_ps(HLG_OOTF_GAMMA));
            _mm256_mul_ps(display, _mm256_set1_ps(10.0))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn eotf_ps(transfer: TransferFn, v: __m256) -> __m256 {
        unsafe {
            match transfer {
                TransferFn::St2084 => pq_to_linear_ps(v),
                TransferFn::AribStdB67 => hlg_to_linear_ps(v),
                _ => _mm256_max_ps(v, _mm256_setzero_ps()),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hable_partial_ps(x: __m256) -> __m256 {
        {
            // ((x·(A·x + C·B) + D·E) / (x·(A·x + B) + D·F)) − E/F
            let ax = _mm256_mul_ps(_mm256_set1_ps(HABLE_A), x);
            let num = _mm256_fmadd_ps(
                x,
                _mm256_add_ps(ax, _mm256_set1_ps(HABLE_C * HABLE_B)),
                _mm256_set1_ps(HABLE_D * HABLE_E),
            );
            let den = _mm256_fmadd_ps(
                x,
                _mm256_add_ps(ax, _mm256_set1_ps(HABLE_B)),
                _mm256_set1_ps(HABLE_D * HABLE_F),
            );
            _mm256_sub_ps(_mm256_div_ps(num, den), _mm256_set1_ps(HABLE_E / HABLE_F))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hable_tonemap_ps(x: __m256, scale: __m256) -> __m256 {
        unsafe {
            // Gamut clip before the curve — see the scalar `hable_tonemap`.
            let x = _mm256_max_ps(x, _mm256_setzero_ps());
            let curr = hable_partial_ps(_mm256_mul_ps(x, _mm256_set1_ps(HABLE_EXPOSURE)));
            let v = _mm256_mul_ps(curr, scale);
            _mm256_min_ps(_mm256_max_ps(v, _mm256_setzero_ps()), _mm256_set1_ps(1.0))
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn bt709_oetf_ps(l: __m256) -> __m256 {
        unsafe {
            let l = _mm256_min_ps(_mm256_max_ps(l, _mm256_setzero_ps()), _mm256_set1_ps(1.0));
            let lin = _mm256_mul_ps(_mm256_set1_ps(4.5), l);
            let gam = _mm256_fmsub_ps(
                _mm256_set1_ps(1.099),
                pow_ps(l, _mm256_set1_ps(0.45)),
                _mm256_set1_ps(0.099),
            );
            let use_lin = _mm256_cmp_ps(l, _mm256_set1_ps(0.018), _CMP_LT_OS);
            _mm256_blendv_ps(gam, lin, use_lin)
        }
    }

    /// Eight pixels: normalised (y, cb, cr) → rounded, clamped 8-bit
    /// (y8, cb8, cr8) as f32 lanes.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tonemap_8px(
        y_n: __m256,
        cb_n: __m256,
        cr_n: __m256,
        transfer: TransferFn,
        scale: __m256,
    ) -> (__m256, __m256, __m256) {
        unsafe {
            // 1. BT.2020 NCL Y'CbCr → R'G'B'.
            let r_g = _mm256_fmadd_ps(_mm256_set1_ps(1.4746), cr_n, y_n);
            let g_g = _mm256_fnmadd_ps(
                _mm256_set1_ps(0.57135),
                cr_n,
                _mm256_fnmadd_ps(_mm256_set1_ps(0.16455), cb_n, y_n),
            );
            let b_g = _mm256_fmadd_ps(_mm256_set1_ps(1.8814), cb_n, y_n);
            // 2. EOTF.
            let r_l = eotf_ps(transfer, r_g);
            let g_l = eotf_ps(transfer, g_g);
            let b_l = eotf_ps(transfer, b_g);
            // 3. Gamut BT.2020 → BT.709 (linear).
            let r709 = _mm256_fnmadd_ps(
                _mm256_set1_ps(0.07285),
                b_l,
                _mm256_fnmadd_ps(_mm256_set1_ps(0.58764), g_l, _mm256_mul_ps(_mm256_set1_ps(1.66049), r_l)),
            );
            let g709 = _mm256_fnmadd_ps(
                _mm256_set1_ps(0.01006),
                b_l,
                _mm256_fmadd_ps(_mm256_set1_ps(1.13290), g_l, _mm256_mul_ps(_mm256_set1_ps(-0.12455), r_l)),
            );
            let b709 = _mm256_fmadd_ps(
                _mm256_set1_ps(1.11873),
                b_l,
                _mm256_fnmadd_ps(_mm256_set1_ps(0.10058), g_l, _mm256_mul_ps(_mm256_set1_ps(-0.01815), r_l)),
            );
            // 4. Hable. 5. OETF.
            let r_o = bt709_oetf_ps(hable_tonemap_ps(r709, scale));
            let g_o = bt709_oetf_ps(hable_tonemap_ps(g709, scale));
            let b_o = bt709_oetf_ps(hable_tonemap_ps(b709, scale));
            // 6. RGB → Y'CbCr BT.709 limited.
            let y = _mm256_fmadd_ps(
                _mm256_set1_ps(0.0722),
                b_o,
                _mm256_fmadd_ps(_mm256_set1_ps(0.7152), g_o, _mm256_mul_ps(_mm256_set1_ps(0.2126), r_o)),
            );
            let cb = _mm256_div_ps(_mm256_sub_ps(b_o, y), _mm256_set1_ps(1.8556));
            let cr = _mm256_div_ps(_mm256_sub_ps(r_o, y), _mm256_set1_ps(1.5748));
            let round = |v: __m256, lo: f32, hi: f32| {
                let v = _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                _mm256_min_ps(_mm256_max_ps(v, _mm256_set1_ps(lo)), _mm256_set1_ps(hi))
            };
            let y8 = round(_mm256_fmadd_ps(y, _mm256_set1_ps(219.0), _mm256_set1_ps(16.0)), 16.0, 235.0);
            let cb8 = round(_mm256_fmadd_ps(cb, _mm256_set1_ps(224.0), _mm256_set1_ps(128.0)), 16.0, 240.0);
            let cr8 = round(_mm256_fmadd_ps(cr, _mm256_set1_ps(224.0), _mm256_set1_ps(128.0)), 16.0, 240.0);
            (y8, cb8, cr8)
        }
    }

    /// Eight f32 lanes holding integers 0..=255 → eight bytes at `dst`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn store_8_bytes(dst: *mut u8, v: __m256) {
        unsafe {
            let i = _mm256_cvtps_epi32(v);
            let w = _mm256_packus_epi32(i, i); // [a0..a3,a0..a3 | a4..a7,a4..a7] as i16
            let b = _mm256_packus_epi16(w, w); // [a0..a3 ×4 | a4..a7 ×4] as u8
            let lo = _mm256_castsi256_si128(b);
            let hi = _mm256_extracti128_si256(b, 1);
            let lo32 = _mm_cvtsi128_si32(lo) as u32;
            let hi32 = _mm_cvtsi128_si32(hi) as u32;
            std::ptr::write_unaligned(dst as *mut u32, lo32);
            std::ptr::write_unaligned(dst.add(4) as *mut u32, hi32);
        }
    }

    /// Eight u16 LE samples at `src` → eight f32 lanes.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn load_8_u16_ps(src: *const u16) -> __m256 {
        unsafe {
            let v = _mm_loadu_si128(src as *const __m128i);
            _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(v))
        }
    }

    /// The full frame. Returns (Y, Cb, Cr) 8-bit planes.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn tonemap_planes_avx2(
        p: &Planes10<'_>,
        transfer: TransferFn,
        max_white: f32,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        unsafe {
            let (w, h) = (p.w, p.h);
            let cw = w / 2;
            let mut out_y = vec![0u8; w * h];
            let mut out_cb = vec![0u8; cw * (h / 2)];
            let mut out_cr = vec![0u8; cw * (h / 2)];

            let scale_s = 1.0 / hable_partial(max_white * HABLE_EXPOSURE);
            let scale = _mm256_set1_ps(scale_s);
            let v_y_black = _mm256_set1_ps(Y_BLACK_10);
            let v_y_range_inv = _mm256_set1_ps(1.0 / Y_RANGE_10);
            let v_c_neutral = _mm256_set1_ps(C_NEUTRAL_10);
            let v_c_range_inv = _mm256_set1_ps(1.0 / (C_HALFRANGE_10 * 2.0));
            // Chroma site i feeds luma columns 2i and 2i+1.
            let dup_lo = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
            let dup_hi = _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7);
            // hadd pairs within 128-bit lanes; this puts the 8 sums back in
            // column order.
            let unshuffle = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
            let quarter = _mm256_set1_ps(0.25);
            let half = _mm256_set1_ps(0.5);

            let vec_sites = cw & !7;
            for by in 0..(h / 2) {
                let c_row = by * cw;
                let y_row0 = by * 2 * w;
                let y_row1 = y_row0 + w;
                let mut bx = 0usize;
                while bx < vec_sites {
                    let cb = load_8_u16_ps(p.cb.as_ptr().add(c_row + bx));
                    let cr = load_8_u16_ps(p.cr.as_ptr().add(c_row + bx));
                    let cb_n = _mm256_mul_ps(_mm256_sub_ps(cb, v_c_neutral), v_c_range_inv);
                    let cr_n = _mm256_mul_ps(_mm256_sub_ps(cr, v_c_neutral), v_c_range_inv);
                    let cb_lo = _mm256_permutevar8x32_ps(cb_n, dup_lo);
                    let cb_hi = _mm256_permutevar8x32_ps(cb_n, dup_hi);
                    let cr_lo = _mm256_permutevar8x32_ps(cr_n, dup_lo);
                    let cr_hi = _mm256_permutevar8x32_ps(cr_n, dup_hi);

                    let mut acc_cb = _mm256_setzero_ps();
                    let mut acc_cr = _mm256_setzero_ps();
                    for row in [y_row0, y_row1] {
                        let base = row + bx * 2;
                        let y_lo = load_8_u16_ps(p.y.as_ptr().add(base));
                        let y_hi = load_8_u16_ps(p.y.as_ptr().add(base + 8));
                        let y_lo = _mm256_mul_ps(_mm256_sub_ps(y_lo, v_y_black), v_y_range_inv);
                        let y_hi = _mm256_mul_ps(_mm256_sub_ps(y_hi, v_y_black), v_y_range_inv);
                        let (y8a, cb8a, cr8a) = tonemap_8px(y_lo, cb_lo, cr_lo, transfer, scale);
                        let (y8b, cb8b, cr8b) = tonemap_8px(y_hi, cb_hi, cr_hi, transfer, scale);
                        store_8_bytes(out_y.as_mut_ptr().add(base), y8a);
                        store_8_bytes(out_y.as_mut_ptr().add(base + 8), y8b);
                        // Pair sums: [a01 a23 b01 b23 | a45 a67 b45 b67].
                        acc_cb = _mm256_add_ps(acc_cb, _mm256_hadd_ps(cb8a, cb8b));
                        acc_cr = _mm256_add_ps(acc_cr, _mm256_hadd_ps(cr8a, cr8b));
                    }
                    // The sum of four codes × 0.25 is an exact quarter, so a
                    // tie at .5 is systematic (one block in four); the scalar
                    // `.round()` is half-away-from-zero, which for these
                    // non-negative exact values is `floor(x + 0.5)` — not the
                    // half-to-even of `_mm256_round_ps`.
                    let avg_cb = _mm256_floor_ps(_mm256_fmadd_ps(
                        _mm256_permutevar8x32_ps(acc_cb, unshuffle),
                        quarter,
                        half,
                    ));
                    let avg_cr = _mm256_floor_ps(_mm256_fmadd_ps(
                        _mm256_permutevar8x32_ps(acc_cr, unshuffle),
                        quarter,
                        half,
                    ));
                    store_8_bytes(out_cb.as_mut_ptr().add(c_row + bx), avg_cb);
                    store_8_bytes(out_cr.as_mut_ptr().add(c_row + bx), avg_cr);
                    bx += 8;
                }
                // Scalar tail: the last cw % 8 chroma sites of the row.
                while bx < cw {
                    let cb_n = c10_to_normalised(p.cb[c_row + bx]);
                    let cr_n = c10_to_normalised(p.cr[c_row + bx]);
                    let mut acc_cb = 0.0_f32;
                    let mut acc_cr = 0.0_f32;
                    for row in [y_row0, y_row1] {
                        for dx in 0..2 {
                            let idx = row + bx * 2 + dx;
                            let y_n = y10_to_normalised(p.y[idx]);
                            let (y8, cb8, cr8) =
                                tonemap_pixel_scalar(y_n, cb_n, cr_n, transfer, max_white);
                            out_y[idx] = y8;
                            acc_cb += cb8 as f32;
                            acc_cr += cr8 as f32;
                        }
                    }
                    out_cb[c_row + bx] = (acc_cb * 0.25).round() as u8;
                    out_cr[c_row + bx] = (acc_cr * 0.25).round() as u8;
                    bx += 1;
                }
            }
            (out_y, out_cb, out_cr)
        }
    }
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

    // ── scalar vs AVX2 ────────────────────────────────────────────────

    /// A frame holding every 10-bit luma code (0..=1023, including the
    /// out-of-range ends) against a grid of chroma values. `w` = 1024
    /// luma columns → 512 chroma sites per row, so the vector body runs
    /// 64 times per row; the odd extra columns of the `tail` variant
    /// exercise the scalar tail.
    fn ramp_frame(chroma_grid: &[u16], tail: usize) -> VideoFrame {
        let w = 1024 + tail;
        let h = 2 * chroma_grid.len() * chroma_grid.len();
        let mut y = Vec::with_capacity(w * h * 2);
        let mut cb = Vec::with_capacity((w / 2) * (h / 2) * 2);
        let mut cr = Vec::with_capacity((w / 2) * (h / 2) * 2);
        for (row_pair, (cbv, crv)) in chroma_grid
            .iter()
            .flat_map(|a| chroma_grid.iter().map(move |b| (*a, *b)))
            .enumerate()
        {
            for dy in 0..2 {
                for x in 0..w {
                    // Row 0 of the pair ramps up, row 1 ramps down, so the
                    // 2×2 block mixes two luma codes.
                    let code = if dy == 0 { x % 1024 } else { 1023 - (x % 1024) };
                    y.extend_from_slice(&(code as u16).to_le_bytes());
                }
            }
            let _ = row_pair;
            for _ in 0..(w / 2) {
                cb.extend_from_slice(&cbv.to_le_bytes());
                cr.extend_from_slice(&crv.to_le_bytes());
            }
        }
        let mut data = y;
        data.extend_from_slice(&cb);
        data.extend_from_slice(&cr);
        VideoFrame::new(
            Bytes::from(data),
            w as u32,
            h as u32,
            PixelFormat::Yuv420p10le,
            ColorSpace::Bt2020,
            0,
        )
    }

    /// Max |a − b| over the frame and the number of samples that differ.
    fn max_abs_diff(a: &[u8], b: &[u8]) -> (u8, usize) {
        assert_eq!(a.len(), b.len());
        let mut max = 0u8;
        let mut n = 0usize;
        for (x, y) in a.iter().zip(b) {
            let d = x.abs_diff(*y);
            if d > 0 {
                n += 1;
            }
            max = max.max(d);
        }
        (max, n)
    }

    fn avx2_available() -> bool {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }

    #[test]
    fn avx2_matches_scalar_within_one_lsb_over_the_full_pq_and_hlg_ramp() {
        if !avx2_available() {
            eprintln!("SKIP: no AVX2+FMA on this host");
            return;
        }
        // Chroma at the neutral, the limited-range ends and beyond them,
        // and two in-between values — every luma code against each pair.
        let grid = [0u16, 64, 256, 512, 768, 960, 1023];
        for tail in [0usize, 6] {
            let frame = ramp_frame(&grid, tail);
            for (transfer, nits) in [
                (TransferFn::St2084, None),
                (TransferFn::St2084, Some(4000.0)),
                (TransferFn::AribStdB67, None),
            ] {
                let s = tonemap_yuv420p10le_bt2020_to_yuv420p_bt709_scalar(&frame, transfer, nits)
                    .expect("scalar");
                let v = tonemap_yuv420p10le_bt2020_to_yuv420p_bt709_avx2(&frame, transfer, nits)
                    .expect("avx2");
                assert_eq!(s.data.len(), v.data.len());
                let (max, n) = max_abs_diff(&s.data, &v.data);
                let total = s.data.len();
                assert!(
                    max <= 1,
                    "{transfer:?} nits={nits:?} tail={tail}: max diff {max} > 1 LSB ({n}/{total} samples differ)"
                );
                // A vector path that rounds everything the other way would
                // still pass ≤ 1 LSB; it should also agree on almost all
                // samples (the boundary cases are rare).
                assert!(
                    n * 100 < total,
                    "{transfer:?} nits={nits:?} tail={tail}: {n}/{total} samples differ (> 1%)"
                );
            }
        }
    }

    #[test]
    fn dispatcher_agrees_with_both_paths_and_honours_the_scalar_switch() {
        let frame = ramp_frame(&[64u16, 512, 960], 2);
        let s = tonemap_yuv420p10le_bt2020_to_yuv420p_bt709_scalar(&frame, TransferFn::St2084, None)
            .unwrap();
        let d = tonemap_yuv420p10le_bt2020_to_yuv420p_bt709(&frame, TransferFn::St2084, None).unwrap();
        let (max, _) = max_abs_diff(&s.data, &d.data);
        assert!(max <= 1);
        // The `_avx2` entry falls back to scalar where AVX2 is missing, so it
        // is always callable.
        let v = tonemap_yuv420p10le_bt2020_to_yuv420p_bt709_avx2(&frame, TransferFn::St2084, None)
            .unwrap();
        assert_eq!(v.data.len(), s.data.len());
        if !avx2_available() {
            assert_eq!(v.data, s.data);
        }
    }

    #[test]
    fn simd_exp_log_pow_match_libm_to_a_few_ulp() {
        if !avx2_available() {
            eprintln!("SKIP: no AVX2+FMA on this host");
            return;
        }
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[target_feature(enable = "avx2,fma")]
        unsafe fn run() {
            unsafe {
                let mut worst_rel = 0.0f32;
                // log over 1e-6..1e3, exp over -20..20, pow over the exponents
                // the kernel uses, on 8-lane batches.
                for i in (0..8000).step_by(8) {
                    let xs: [f32; 8] = std::array::from_fn(|k| 1e-6 * 1.002_5f32.powi((i + k) as i32));
                    let x = _mm256_loadu_ps(xs.as_ptr());
                    let mut got = [0f32; 8];
                    _mm256_storeu_ps(got.as_mut_ptr(), super::simd::log_ps(x));
                    for k in 0..8 {
                        let want = xs[k].ln();
                        let rel = ((got[k] - want) / want.abs().max(1e-3)).abs();
                        worst_rel = worst_rel.max(rel);
                    }
                    let es: [f32; 8] = std::array::from_fn(|k| -20.0 + 40.0 * ((i + k) as f32 / 8000.0));
                    let e = _mm256_loadu_ps(es.as_ptr());
                    _mm256_storeu_ps(got.as_mut_ptr(), super::simd::exp_ps(e));
                    for k in 0..8 {
                        let want = es[k].exp();
                        worst_rel = worst_rel.max(((got[k] - want) / want).abs());
                    }
                    for p in [PQ_M2_INV, PQ_M1_INV, 0.45, HLG_OOTF_GAMMA] {
                        _mm256_storeu_ps(
                            got.as_mut_ptr(),
                            super::simd::pow_ps(x, _mm256_set1_ps(p)),
                        );
                        for k in 0..8 {
                            let want = xs[k].powf(p);
                            worst_rel = worst_rel.max(((got[k] - want) / want).abs());
                        }
                    }
                }
                assert!(worst_rel < 1.5e-5, "worst relative error {worst_rel}");
                // pow(0, p) = 0, log(0) masked.
                let mut got = [1f32; 8];
                _mm256_storeu_ps(
                    got.as_mut_ptr(),
                    super::simd::pow_ps(_mm256_setzero_ps(), _mm256_set1_ps(0.45)),
                );
                assert!(got.iter().all(|v| *v == 0.0));
            }
        }
        unsafe { run() }
    }
}
