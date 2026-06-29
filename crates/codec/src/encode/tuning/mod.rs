//! AV1 encoder tuning adapter.
//!
//! Translates a single backend-agnostic perceptual quality target into
//! per-encoder parameters so identical inputs yield visually consistent
//! output across rav1e, NVENC AV1, and future backends (SVT-AV1, AMF,
//! QSV).
//!
//! See `docs/av1-tuning-research.md` for the source tables and
//! `docs/av1-tuning-methodology.md` for how to re-calibrate when a new
//! encoder lands.
//!
//! # Design
//!
//! The user picks two things:
//! 1. A `QualityTarget` — perceptual goal expressed in VMAF/SSIMULACRA2
//!    bands, not encoder-native CRF. Every backend must reach roughly
//!    the same VMAF for a given target (±2 VMAF band).
//! 2. A `SpeedTier` — how much wallclock to spend getting there. Maps
//!    to encoder-native speed presets.
//!
//! The adapter also takes `(width, height)` because tile grid and
//! lookahead sizing depend on frame size.

mod adapters;
mod params;
#[cfg(test)]
mod tests;

// ─── Re-exports: param structs, enums, and constants ────────────────────────
pub use params::{
    AmfAv1Params, AmfQualityPreset, AmfRateControl, MFX_CODINGOPTION_OFF, MFX_CODINGOPTION_ON,
    NvencAv1Params, NvencRateControl, QsvAv1Params, QsvRateControl, Rav1eParams,
};

// ─── Re-exports: public adapter functions ───────────────────────────────────
pub use adapters::{amf_av1_params, nvenc_av1_params, qsv_av1_params, rav1e_params};

// ─── Public types ────────────────────────────────────────────────

/// A single perceptual quality target, backend-agnostic.
///
/// Maps to VMAF / SSIMULACRA2 bands, NOT encoder CRF values:
///
/// | Variant             | Target VMAF | Target SSIMULACRA2 | Use case                         |
/// |---------------------|:-----------:|:------------------:|----------------------------------|
/// | `VisuallyLossless`  | ~98         | ~90                | Archive, master                  |
/// | `High`              | ~95         | ~80                | Premium OTT / top ABR rung       |
/// | `Standard`          | ~90         | ~70                | Default web / streaming          |
/// | `Low`               | ~85         | ~60                | Mobile / bandwidth-constrained   |
/// | `Vmaf(u8)`          | explicit    | n/a                | A/B testing escape hatch         |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QualityTarget {
    VisuallyLossless,
    High,
    #[default]
    Standard,
    Low,
    Vmaf(u8),
}

/// User-facing speed tier — maps to encoder-native speed presets.
///
/// | Variant    | rav1e | NVENC preset | SVT-AV1 preset | libaom cpu-used |
/// |------------|:-----:|:------------:|:--------------:|:---------------:|
/// | `Draft`    | 8     | P5           | 12             | 8               |
/// | `Standard` | 6     | P6           | 8              | 6               |
/// | `Archive`  | 4     | P7           | 4              | 4               |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpeedTier {
    Draft,
    #[default]
    Standard,
    Archive,
}

// ─── SDK constant ────────────────────────────────────────────────

/// SDK constant `NV_ENC_TUNING_INFO_HIGH_QUALITY = 1`.
pub const NVENC_TUNING_HIGH_QUALITY: u32 = 1;

// NVENC SDK 13.0 preset GUIDs (vendor/nvidia/nvEncodeAPI.h:226-251).
//
// CRITICAL: SDK 12.2 had ENTIRELY DIFFERENT preset GUIDs for P5/P6/P7
// — when we vendored SDK 13's nvEncodeAPI.h on 2026-05-01 we updated
// the NvEncFunctionList ordering + struct layouts but missed that the
// preset-GUID values themselves were also reshuffled. Sending SDK
// 12.2 P5/P6/P7 GUIDs to a SDK 13 driver returned NV_ENC_ERR_UNSUPPORTED_PARAM
// (rc=12) from NvEncGetEncodePresetConfigEx (the driver doesn't
// recognise the old GUIDs and rejects the lookup). For reference, the
// 12.2 → 13 GUID rotation:
//   P5: d0918ee2-a509-4681-af96-e9c3c45b7aa7 → 21c6e6b4-297a-4cba-998f-b6cbde72ade3
//   P6: fc8ebf15-6e19-47b4-8ea7-b1917f379eed → 8e75c279-6299-4ab6-8302-0b215a335cf5
//   P7: 84bdda58-33cb-4895-a372-ddeddb013ac4 → 84848c12-6f71-4c13-931b-53e283f57974
pub(self) const NV_ENC_PRESET_P5_GUID_BYTES: [u8; 16] = [
    0xb4, 0xe6, 0xc6, 0x21, // data1 = 0x21c6e6b4
    0x7a, 0x29, // data2 = 0x297a
    0xba, 0x4c, // data3 = 0x4cba
    0x99, 0x8f, 0xb6, 0xcb, 0xde, 0x72, 0xad, 0xe3,
];

pub(self) const NV_ENC_PRESET_P6_GUID_BYTES: [u8; 16] = [
    0x79, 0xc2, 0x75, 0x8e, // data1 = 0x8e75c279
    0x99, 0x62, // data2 = 0x6299
    0xb6, 0x4a, // data3 = 0x4ab6
    0x83, 0x02, 0x0b, 0x21, 0x5a, 0x33, 0x5c, 0xf5,
];

pub(self) const NV_ENC_PRESET_P7_GUID_BYTES: [u8; 16] = [
    0x12, 0x8c, 0x84, 0x84, // data1 = 0x84848c12
    0x71, 0x6f, // data2 = 0x6f71
    0x13, 0x4c, // data3 = 0x4c13
    0x93, 0x1b, 0x53, 0xe2, 0x83, 0xf5, 0x79, 0x74,
];

// ─── Shared helpers (used by adapters and tests) ──────────────────

/// libaom `cq-level` that corresponds to a given QualityTarget. libaom
/// is the cross-encoder reference: we equalize other encoders *to*
/// libaom's VMAF at each CQ.
///
/// Exposed `pub` so the FFmpeg-wrapper encoder path
/// (`encode::ffmpeg_enc`) can route `libsvtav1` / `libaom-av1` through
/// the same adapter tables as the native encoders.
pub fn libaom_cq_for_target(target: QualityTarget) -> u8 {
    match target {
        QualityTarget::VisuallyLossless => 20,
        QualityTarget::High => 27,
        QualityTarget::Standard => 32,
        QualityTarget::Low => 38,
        QualityTarget::Vmaf(v) => vmaf_to_libaom_cq(v),
    }
}

/// NVENC CQ that hits the same VMAF as `libaom_cq_for_target`, per
/// the research doc §2.4.
fn nvenc_cq_for_target(target: QualityTarget) -> u8 {
    match target {
        QualityTarget::VisuallyLossless => 19,
        QualityTarget::High => 25,
        QualityTarget::Standard => 30,
        QualityTarget::Low => 36,
        QualityTarget::Vmaf(v) => vmaf_to_nvenc_cq(v),
    }
}

/// Anchor points for libaom VMAF↔cq-level (research §2.1). Must stay
/// in descending VMAF order; `piecewise_cq` below depends on it.
const LIBAOM_ANCHORS: &[(i32, i32)] = &[
    (100, 10), // asymptote beyond VisuallyLossless
    (98, 20),
    (95, 27),
    (90, 32),
    (85, 38),
    (70, 55), // low-quality extrapolation
];

/// Anchor points for NVENC AV1 VMAF↔CQ. Calibrated down from libaom
/// to compensate for NVENC's documented compression-efficiency gap
/// (research §2.4). Same VMAF → lower CQ than libaom.
const NVENC_ANCHORS: &[(i32, i32)] =
    &[(100, 10), (98, 19), (95, 25), (90, 30), (85, 36), (70, 52)];

/// Piecewise-linear interpolation between anchors. Anchors are
/// `(vmaf, cq)` pairs in descending VMAF order. Out-of-range VMAF
/// values clamp to the nearest anchor's CQ.
fn piecewise_cq(vmaf: u8, anchors: &[(i32, i32)]) -> u8 {
    let v = vmaf as i32;
    // Above the top anchor: return its CQ (asymptote).
    if v >= anchors[0].0 {
        return anchors[0].1.clamp(0, 63) as u8;
    }
    // Below the bottom anchor: return its CQ.
    let last = anchors.len() - 1;
    if v <= anchors[last].0 {
        return anchors[last].1.clamp(0, 63) as u8;
    }
    // Linear interpolation between surrounding anchors.
    for pair in anchors.windows(2) {
        let (v_hi, cq_hi) = pair[0];
        let (v_lo, cq_lo) = pair[1];
        if v <= v_hi && v >= v_lo {
            let span = v_hi - v_lo;
            if span == 0 {
                return cq_hi.clamp(0, 63) as u8;
            }
            let t = v_hi - v; // 0 at high anchor, span at low anchor
            let cq = cq_hi + (cq_lo - cq_hi) * t / span;
            return cq.clamp(0, 63) as u8;
        }
    }
    anchors[last].1.clamp(0, 63) as u8
}

/// Generic piecewise-linear interpolator for non-CQ scales. Mirrors
/// `piecewise_cq` but with configurable clamp bounds so the same logic
/// serves AMF's 1..100 and QSV's 1..51.
fn piecewise_quality(vmaf: u8, anchors: &[(i32, i32)], lo: i32, hi: i32) -> u8 {
    let v = vmaf as i32;
    if v >= anchors[0].0 {
        return anchors[0].1.clamp(lo, hi) as u8;
    }
    let last = anchors.len() - 1;
    if v <= anchors[last].0 {
        return anchors[last].1.clamp(lo, hi) as u8;
    }
    for pair in anchors.windows(2) {
        let (v_hi, q_hi) = pair[0];
        let (v_lo, q_lo) = pair[1];
        if v <= v_hi && v >= v_lo {
            let span = v_hi - v_lo;
            if span == 0 {
                return q_hi.clamp(lo, hi) as u8;
            }
            let t = v_hi - v;
            let q = q_hi + (q_lo - q_hi) * t / span;
            return q.clamp(lo, hi) as u8;
        }
    }
    anchors[last].1.clamp(lo, hi) as u8
}

fn vmaf_to_libaom_cq(vmaf: u8) -> u8 {
    piecewise_cq(vmaf, LIBAOM_ANCHORS)
}

fn vmaf_to_nvenc_cq(vmaf: u8) -> u8 {
    piecewise_cq(vmaf, NVENC_ANCHORS)
}

/// Tile grid for rav1e (CPU). Returns `(columns, rows)`, literal counts.
/// rav1e is memory-bandwidth-limited and benefits from aggressive tiling
/// even at the cost of a small quality hit, because tile parallelism is
/// most of its throughput story at 4K+.
fn tile_grid_rav1e(width: u32, height: u32) -> (usize, usize) {
    let max_dim = width.max(height);
    if max_dim >= 3840 {
        (4, 4) // 16 tiles at 4K — rav1e fans out across cores
    } else if max_dim >= 1920 {
        (2, 2)
    } else {
        (1, 1)
    }
}

/// Tile grid for NVENC AV1. Returns `(columns, rows)`. NVENC has enough
/// internal parallelism that it does not need large tile grids for
/// throughput — and its HIGH_QUALITY tuning is sensitive to the ~1%
/// quality cost per extra tile row/column (tile boundaries break loop
/// filter continuity, and AV1 tiles are entropy-coded independently).
/// Cap at 2×2 even at 4K.
fn tile_grid_nvenc(width: u32, height: u32) -> (usize, usize) {
    let max_dim = width.max(height);
    if max_dim >= 1920 { (2, 2) } else { (1, 1) }
}

/// Shared HW-encoder tile grid. Used by NVENC, AMF, and QSV — all
/// three are "HQ-equivalent hardware encoders" that don't need rav1e's
/// aggressive tiling for throughput and are sensitive to the ~1%
/// quality cost per extra tile row/column. Cap at 2×2 even at 4K.
///
/// This is an alias over `tile_grid_nvenc` so the shared rule is
/// explicit at call sites. Changing the shared cap is a one-line
/// change here.
fn tile_grid_hw(width: u32, height: u32) -> (usize, usize) {
    tile_grid_nvenc(width, height)
}
