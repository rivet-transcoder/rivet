//! `denoise=dpir` — **deep denoise** with DPIR's DRUNet CNN.
//!
//! [DPIR](https://github.com/cszn/DPIR) (Zhang et al., "Plug-and-Play Image
//! Restoration with Deep Denoiser Prior", MIT licence) ships **DRUNet**, a
//! residual U-Net trained as a Gaussian denoiser for noise levels σ ∈ [0, 50]
//! (on the 8-bit scale). It takes the image plus a constant channel holding σ,
//! so one network covers every strength. This module runs it on
//! [candle](https://github.com/huggingface/candle) — pure Rust on the CPU, CUDA
//! with the `dpir-cuda` feature — inside the existing **resource-filter**
//! pattern: [`PreparedDpir::prepare`] loads the weights once from
//! [`FilterChain::prepare`](super::FilterChain::prepare), then
//! [`PreparedDpir::apply`] runs per frame.
//!
//! ## The dial: `denoise=dpir[:SIGMA][:color]`
//!
//! `SIGMA` is the **noise level in 8-bit code values** (`0..=50`, default
//! [`DEFAULT_SIGMA`]) — not the classical methods' `0..=1` blend. It is written
//! into the network's noise-level channel as `sigma_channel` describes, so
//! `denoise=dpir:25` says "this footage carries σ≈25 Gaussian noise"; a σ
//! above the real noise over-smooths, below it under-denoises. `color` runs the
//! RGB model (`drunet_color`) on all three planes; the default runs the
//! grayscale model (`drunet_gray`) on luma alone and leaves chroma untouched.
//!
//! ## The model file
//!
//! The weights are the upstream release assets, read directly in their legacy
//! `torch.save` layout ([`pth`]) — nothing to convert. They are looked up in
//! `$RIVET_DPIR_MODEL` (a file, or a directory holding `drunet_gray.pth` /
//! `drunet_color.pth`), else in the per-user cache dir ([`cache_dir`]), and a
//! missing file is an error that quotes the download URL. A `.safetensors`
//! file is accepted too.
//!
//! ## Tiling
//!
//! Frames are cut into tiles ([`DEFAULT_TILE`] pixels on the CPU,
//! [`DEFAULT_TILE_GPU`] on a GPU) with [`TILE_OVERLAP`] of context on each
//! side, padded (edge-replicate) to a multiple of 8 for the network's three
//! stride-2 stages, and only each tile's own interior is kept. That bounds
//! memory at any frame size (`RIVET_DPIR_TILE=0` runs the whole frame at
//! once; the overlap is paid per tile, so bigger tiles are cheaper).
//!
//! Feature-gated: without `dpir` the filter still parses and displays, and
//! `prepare` explains what to build.

// Without the feature the tiling / level / colour helpers below are reached
// only by their tests; they are kept unconditional so those tests always run.
#![cfg_attr(not(feature = "dpir"), allow(dead_code))]

use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use super::VideoFilter;
use crate::frame::ColorSpace;

#[cfg(feature = "dpir")]
mod net;
#[cfg(feature = "dpir")]
mod pth;
#[cfg(feature = "dpir")]
mod run;
#[cfg(test)]
mod tests;

#[cfg(feature = "dpir")]
pub use run::PreparedDpir;

/// Noise levels DRUNet was trained for, in 8-bit code values.
pub const SIGMA_RANGE: RangeInclusive<f32> = 0.0..=50.0;
/// The σ `denoise=dpir` runs at when none is given: moderate, and about what
/// visible sensor / compression noise measures at.
pub const DEFAULT_SIGMA: f32 = 15.0;
/// Where the weights come from — the KAIR release that DPIR points at.
pub const MODEL_URL_BASE: &str = "https://github.com/cszn/KAIR/releases/download/v1.0/";
/// Env var: the model file, or a directory holding the model files.
pub const ENV_MODEL: &str = "RIVET_DPIR_MODEL";
/// Env var: `cpu`, `cuda` or `cuda:N`; default picks CUDA when built with
/// `dpir-cuda` and a device opens, else the CPU.
pub const ENV_DEVICE: &str = "RIVET_DPIR_DEVICE";
/// Env var: tile edge in pixels (`0` = whole frame); default [`DEFAULT_TILE`]
/// on the CPU, [`DEFAULT_TILE_GPU`] on a GPU.
pub const ENV_TILE: &str = "RIVET_DPIR_TILE";
/// Tile edge in pixels on the CPU. 512 keeps the largest activation
/// (64 ch × (512+2·32)² f32) at ~85 MB, whatever the frame size.
pub const DEFAULT_TILE: usize = 512;
/// Tile edge in pixels on a GPU: 720p and 1080p frames go through whole.
/// Measured on the RTX 3090 (cuDNN, `docs/filters/denoise.md`): 512-px tiles
/// feed the network ~2× the pixels (the overlap) and cost 1.5–1.6× the time
/// of the whole frame at the same PSNR; the largest activation at this edge
/// is 64 ch × (2048+2·32)² f32 ≈ 1.1 GB, so a 4K frame still tiles.
pub const DEFAULT_TILE_GPU: usize = 2048;
/// Context on each side of a tile, in pixels. Measured on the release model:
/// tiled vs whole-frame 1080p output differs by at most a few code values
/// (see `docs/filters/denoise.md`).
pub const TILE_OVERLAP: usize = 32;

/// Serde default for [`VideoFilter::Dpir::sigma`].
#[cfg(feature = "serde")]
pub(super) fn default_sigma() -> f32 {
    DEFAULT_SIGMA
}

/// Which DRUNet: the single-channel one on luma, or the RGB one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpirModel {
    /// `drunet_gray.pth`: 1 image channel + σ → 1 channel. Runs on luma only.
    Gray,
    /// `drunet_color.pth`: 3 image channels + σ → 3 channels. Runs on R'G'B'.
    Color,
}

impl DpirModel {
    pub(super) fn for_color(color: bool) -> Self {
        if color { Self::Color } else { Self::Gray }
    }

    /// The release asset's file name.
    pub fn file_name(self) -> &'static str {
        match self {
            Self::Gray => "drunet_gray.pth",
            Self::Color => "drunet_color.pth",
        }
    }

    /// Where to download it from.
    pub fn url(self) -> String {
        format!("{MODEL_URL_BASE}{}", self.file_name())
    }

    /// Image channels the network takes (its input is this + 1 for σ).
    pub(crate) fn image_channels(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Color => 3,
        }
    }
}

/// Reject a σ outside [`SIGMA_RANGE`] (the network was not trained past it,
/// and extrapolates badly).
pub(super) fn validate_sigma(sigma: f32) -> Result<()> {
    if !SIGMA_RANGE.contains(&sigma) {
        bail!("dpir sigma must be {:?}..={:?} (8-bit noise level), got {sigma}", SIGMA_RANGE.start(), SIGMA_RANGE.end());
    }
    Ok(())
}

/// The `denoise=…` arm for DPIR: the tokens after `denoise=` (one of them is
/// `dpir`), order-free — a number is σ, `gray`/`luma` or `color`/`rgb` picks
/// the model.
pub(super) fn parse(parts: &[&str], spec: &str) -> Result<VideoFilter> {
    let mut sigma = DEFAULT_SIGMA;
    let mut color = false;
    for &p in parts {
        if p.eq_ignore_ascii_case("dpir") {
            continue;
        }
        if let Ok(s) = p.parse::<f32>() {
            sigma = s;
            continue;
        }
        match p.to_ascii_lowercase().as_str() {
            "gray" | "grey" | "luma" => color = false,
            "color" | "colour" | "rgb" => color = true,
            o => bail!("unknown dpir option '{o}' in '{spec}' (want a sigma 0..=50, gray, or color)"),
        }
    }
    validate_sigma(sigma)?;
    Ok(VideoFilter::Dpir { sigma, color })
}

// ── model file resolution ────────────────────────────────────────────────────

/// The per-user cache directory the model files live in when
/// [`ENV_MODEL`] is unset: `%LOCALAPPDATA%\rivet\models` on Windows,
/// `$XDG_CACHE_HOME/rivet/models` or `~/.cache/rivet/models` elsewhere.
pub fn cache_dir() -> Option<PathBuf> {
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
    };
    base.map(|b| b.join("rivet").join("models"))
}

/// Where `model`'s weights are expected, given the [`ENV_MODEL`] override
/// (a file — anything ending in `.pth` / `.safetensors` — or a directory) and
/// the cache dir. Pure: does not touch the file system.
pub fn expected_model_path(model: DpirModel, override_: Option<&Path>, cache: Option<&Path>) -> Result<PathBuf> {
    if let Some(o) = override_ {
        let is_file = o.extension().is_some_and(|e| e.eq_ignore_ascii_case("pth") || e.eq_ignore_ascii_case("safetensors"));
        return Ok(if is_file { o.to_path_buf() } else { o.join(model.file_name()) });
    }
    match cache {
        Some(c) => Ok(c.join(model.file_name())),
        None => bail!("no cache directory (LOCALAPPDATA / XDG_CACHE_HOME / HOME unset); set {ENV_MODEL} to the model file"),
    }
}

/// The error for a model file that is not there: names the path and the one
/// command that fixes it.
pub fn missing_model_error(model: DpirModel, path: &Path) -> anyhow::Error {
    anyhow::anyhow!(
        "DPIR model file not found: {p}\n\
         Download it once (about 130 MB, MIT licence):\n  \
         curl -L --create-dirs -o \"{p}\" {url}\n\
         or point {ENV_MODEL} at the file (or a directory holding it).",
        p = path.display(),
        url = model.url()
    )
}

/// Resolve `model`'s file from the environment and check it exists.
pub fn model_path(model: DpirModel) -> Result<PathBuf> {
    let override_ = std::env::var_os(ENV_MODEL).map(PathBuf::from);
    let cache = cache_dir();
    let path = expected_model_path(model, override_.as_deref(), cache.as_deref())?;
    if !path.is_file() {
        return Err(missing_model_error(model, &path));
    }
    Ok(path)
}

// ── tiling ───────────────────────────────────────────────────────────────────

/// One tile: the region **fed** to the network (`x, y, w, h`, which includes
/// the overlap) and the interior **kept** from its output (`keep_*`), both in
/// frame coordinates. The keep regions of [`tiles`] partition the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Tile {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub keep_x: usize,
    pub keep_y: usize,
    pub keep_w: usize,
    pub keep_h: usize,
}

/// Cut a `w×h` frame into `tile×tile` keep regions, each fed with up to
/// `overlap` pixels of context on every side (clamped at the frame edge).
/// `tile == 0` means one tile for the whole frame.
pub(crate) fn tiles(w: usize, h: usize, tile: usize, overlap: usize) -> Vec<Tile> {
    let tile = if tile == 0 { w.max(h).max(1) } else { tile };
    let mut out = Vec::new();
    let mut ty = 0;
    while ty < h {
        let keep_h = tile.min(h - ty);
        let mut tx = 0;
        while tx < w {
            let keep_w = tile.min(w - tx);
            let (x, y) = (tx.saturating_sub(overlap), ty.saturating_sub(overlap));
            let (x1, y1) = ((tx + keep_w + overlap).min(w), (ty + keep_h + overlap).min(h));
            out.push(Tile { x, y, w: x1 - x, h: y1 - y, keep_x: tx, keep_y: ty, keep_w, keep_h });
            tx += tile;
        }
        ty += tile;
    }
    out
}

/// Round `n` up to a multiple of `a`.
pub(crate) fn align_up(n: usize, a: usize) -> usize {
    n.div_ceil(a) * a
}

// ── samples and colour ───────────────────────────────────────────────────────

/// Limited-range code-value levels at a bit depth (`bps` bytes per sample).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Levels {
    /// Largest code value: 255 or 1023.
    pub max: f32,
    /// Luma black (16 / 64) and range (219 / 876).
    pub black: f32,
    pub y_range: f32,
    /// Chroma neutral (128 / 512) and range (224 / 896).
    pub c_mid: f32,
    pub c_range: f32,
}

impl Levels {
    pub(crate) fn for_bps(bps: usize) -> Self {
        let (max, k) = if bps == 2 { (1023.0, 4.0) } else { (255.0, 1.0) };
        Self { max, black: 16.0 * k, y_range: 219.0 * k, c_mid: 128.0 * k, c_range: 224.0 * k }
    }
}

/// Plane bytes (`u8`, or `u16` little-endian for `bps == 2`) → code values.
pub(crate) fn plane_to_f32(plane: &[u8], bps: usize) -> Vec<f32> {
    if bps == 2 {
        plane.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as f32).collect()
    } else {
        plane.iter().map(|&v| v as f32).collect()
    }
}

/// Code values → plane bytes, rounded to nearest and clamped to `0..=max`.
pub(crate) fn f32_to_plane(v: &[f32], bps: usize, max: f32) -> Vec<u8> {
    if bps == 2 {
        v.iter().flat_map(|&x| (x.round().clamp(0.0, max) as u16).to_le_bytes()).collect()
    } else {
        v.iter().map(|&x| x.round().clamp(0.0, max) as u8).collect()
    }
}

/// The value written into DRUNet's noise-level channel for a σ given in 8-bit
/// code values. The network was trained with the map = σ/255 over images in
/// `[0, 1]`:
///
/// - the **gray** path feeds luma as `code / max` (full code range), so one
///   8-bit code value is 1/255 and the map is exactly `σ / 255`;
/// - the **colour** path feeds limited-range R'G'B', where one 8-bit luma code
///   value is 1/219 of the range, so the same noise measures `σ / 219`.
pub(crate) fn sigma_channel(sigma: f32, model: DpirModel) -> f32 {
    match model {
        DpirModel::Gray => sigma / 255.0,
        DpirModel::Color => sigma / 219.0,
    }
}

/// `(Kr, Kb)` of the frame's matrix (BT.601 / BT.709 / BT.2020 NCL).
pub(crate) fn kr_kb(cs: ColorSpace) -> (f32, f32) {
    match cs {
        ColorSpace::Bt601 => (0.299, 0.114),
        ColorSpace::Bt709 => (0.2126, 0.0722),
        ColorSpace::Bt2020 => (0.2627, 0.0593),
    }
}

/// 4:2:0 code-value planes → gamma-encoded R'G'B', nominally `[0, 1]`, with
/// chroma replicated 2×2. `w×h` luma; chroma is `w/2 × h/2` (an odd last
/// row / column reuses the nearest chroma sample).
///
/// **Not clamped.** The matrix is invertible, so an unclamped round trip is
/// lossless; a clamp is not. Values outside the cube are common — a source
/// whose real matrix is not the one it is tagged with (untagged files are
/// BT.709 to the pipeline while ffmpeg wrote them BT.601) puts *most* pixels
/// outside it, and clamping cost 12 dB on such a clip. DRUNet is
/// convolutional with no input bound, so it takes the overshoot in stride.
pub(crate) fn yuv420_to_rgb(y: &[f32], u: &[f32], v: &[f32], w: usize, h: usize, cs: ColorSpace, lv: Levels) -> [Vec<f32>; 3] {
    let (kr, kb) = kr_kb(cs);
    let kg = 1.0 - kr - kb;
    let (cw, ch) = ((w / 2).max(1), (h / 2).max(1));
    let mut r = vec![0f32; w * h];
    let mut g = vec![0f32; w * h];
    let mut b = vec![0f32; w * h];
    for row in 0..h {
        let cy = (row / 2).min(ch - 1);
        for col in 0..w {
            let cx = (col / 2).min(cw - 1);
            let i = row * w + col;
            let yy = (y[i] - lv.black) / lv.y_range;
            let cb = (u[cy * cw + cx] - lv.c_mid) / lv.c_range;
            let cr = (v[cy * cw + cx] - lv.c_mid) / lv.c_range;
            r[i] = yy + 2.0 * (1.0 - kr) * cr;
            g[i] = yy - 2.0 * (1.0 - kb) * kb / kg * cb - 2.0 * (1.0 - kr) * kr / kg * cr;
            b[i] = yy + 2.0 * (1.0 - kb) * cb;
        }
    }
    [r, g, b]
}

/// R'G'B' → 4:2:0 code-value planes: luma per pixel, chroma the 2×2 mean of
/// the full-resolution Cb'/Cr'. The inverse of [`yuv420_to_rgb`] (exactly, up
/// to float rounding, for chroma that was 2×2-replicated); the caller's
/// [`f32_to_plane`] clamps to the code range.
pub(crate) fn rgb_to_yuv420(rgb: &[Vec<f32>; 3], w: usize, h: usize, cs: ColorSpace, lv: Levels) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (kr, kb) = kr_kb(cs);
    let kg = 1.0 - kr - kb;
    let (cw, ch) = (w / 2, h / 2);
    let mut y = vec![0f32; w * h];
    let mut cb_full = vec![0f32; w * h];
    let mut cr_full = vec![0f32; w * h];
    for i in 0..w * h {
        let (r, g, b) = (rgb[0][i], rgb[1][i], rgb[2][i]);
        let yy = kr * r + kg * g + kb * b;
        y[i] = lv.black + yy * lv.y_range;
        cb_full[i] = (b - yy) / (2.0 * (1.0 - kb));
        cr_full[i] = (r - yy) / (2.0 * (1.0 - kr));
    }
    let mut u = vec![0f32; cw * ch];
    let mut v = vec![0f32; cw * ch];
    for cy in 0..ch {
        for cx in 0..cw {
            let (mut sb, mut sr) = (0f32, 0f32);
            for dy in 0..2 {
                for dx in 0..2 {
                    let i = (cy * 2 + dy) * w + cx * 2 + dx;
                    sb += cb_full[i];
                    sr += cr_full[i];
                }
            }
            u[cy * cw + cx] = lv.c_mid + sb / 4.0 * lv.c_range;
            v[cy * cw + cx] = lv.c_mid + sr / 4.0 * lv.c_range;
        }
    }
    (y, u, v)
}

// ── without the feature ──────────────────────────────────────────────────────

/// The `denoise=dpir` step in a build without the `dpir` feature: parses and
/// displays like any filter, and [`prepare`](Self::prepare) says what to build.
#[cfg(not(feature = "dpir"))]
pub struct PreparedDpir {
    _never: std::convert::Infallible,
}

#[cfg(not(feature = "dpir"))]
impl PreparedDpir {
    /// Always an error naming the feature: the network needs candle.
    pub fn prepare(sigma: f32, color: bool) -> Result<Self> {
        validate_sigma(sigma)?;
        let _ = color;
        bail!(
            "denoise=dpir needs the `dpir` feature (build with `--features dpir`, \
             or `dpir-cuda` / `dpir-cudnn` for the GPU); this binary has none of them"
        )
    }

    /// Unreachable: no value of this type can be constructed.
    pub fn apply(&self, _frame: &crate::frame::VideoFrame) -> Result<crate::frame::VideoFrame> {
        match self._never {}
    }
}
