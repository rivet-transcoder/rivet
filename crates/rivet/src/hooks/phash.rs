//! Perceptual hashes: 64-bit fingerprints of a picture that change little
//! when the picture is resized, re-encoded or slightly altered, compared by
//! Hamming distance.
//!
//! Three classic algorithms, computed on 8-bit luma ([`super::frame::luma8`]):
//!
//! - **aHash** (average): the picture shrunk to 8×8, each bit whether a cell
//!   is brighter than the mean. Fast, the least discriminating.
//! - **dHash** (difference): shrunk to 9×8, each bit whether a cell is
//!   brighter than its right-hand neighbour. Robust to brightness and
//!   contrast changes.
//! - **pHash** (DCT): shrunk to 32×32, the 2-D DCT-II taken, each bit of the
//!   8×8 lowest frequencies whether it is above their median. The most robust
//!   to re-encoding and scaling; the usual choice.
//!
//! Bits are row-major, first bit the most significant, written as 16
//! lowercase hex digits — the layout the common `imagehash` implementations
//! print, so hashes computed here compare against theirs.

use std::sync::OnceLock;

use anyhow::{Result, bail};

use codec::frame::VideoFrame;

/// A perceptual hash algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PerceptualAlgorithm {
    AHash,
    DHash,
    PHash,
}

impl PerceptualAlgorithm {
    pub const ALL: [PerceptualAlgorithm; 3] =
        [PerceptualAlgorithm::AHash, PerceptualAlgorithm::DHash, PerceptualAlgorithm::PHash];

    pub fn as_str(self) -> &'static str {
        match self {
            PerceptualAlgorithm::AHash => "ahash",
            PerceptualAlgorithm::DHash => "dhash",
            PerceptualAlgorithm::PHash => "phash",
        }
    }

    /// This algorithm's hash of 8-bit luma `luma` (`width * height`).
    pub fn hash_luma(self, luma: &[u8], width: usize, height: usize) -> Result<u64> {
        if width == 0 || height == 0 || luma.len() < width * height {
            bail!("luma plane is {} bytes for {width}x{height}", luma.len());
        }
        Ok(match self {
            PerceptualAlgorithm::AHash => ahash(luma, width, height),
            PerceptualAlgorithm::DHash => dhash(luma, width, height),
            PerceptualAlgorithm::PHash => phash(luma, width, height),
        })
    }

    /// This algorithm's hash of `frame`.
    pub fn hash_frame(self, frame: &VideoFrame) -> Result<u64> {
        let luma = super::frame::luma8(frame)?;
        self.hash_luma(&luma, frame.width as usize, frame.height as usize)
    }
}

impl std::fmt::Display for PerceptualAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for PerceptualAlgorithm {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "ahash" | "average" | "a" => PerceptualAlgorithm::AHash,
            "dhash" | "difference" | "d" => PerceptualAlgorithm::DHash,
            "phash" | "dct" | "p" => PerceptualAlgorithm::PHash,
            other => bail!("unknown perceptual hash `{other}` (ahash, dhash, phash)"),
        })
    }
}

/// The number of bits two hashes differ in.
pub fn hamming(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// A hash as 16 lowercase hex digits.
pub fn to_hex(hash: u64) -> String {
    format!("{hash:016x}")
}

/// 16 hex digits (an optional `0x`) → the hash.
pub fn from_hex(s: &str) -> Result<u64> {
    let s = s.trim().trim_start_matches("0x");
    if s.is_empty() || s.len() > 16 {
        bail!("a 64-bit perceptual hash is up to 16 hex digits (got `{s}`)");
    }
    u64::from_str_radix(s, 16).map_err(|e| anyhow::anyhow!("`{s}` is not hex: {e}"))
}

/// `luma` shrunk to `tw × th` by area averaging (each output cell the mean of
/// the source pixels it covers, fractional coverage weighted).
pub fn shrink(luma: &[u8], w: usize, h: usize, tw: usize, th: usize) -> Vec<f32> {
    let mut out = vec![0f32; tw * th];
    let (fx, fy) = (w as f64 / tw as f64, h as f64 / th as f64);
    for ty in 0..th {
        let (y0, y1) = (ty as f64 * fy, (ty + 1) as f64 * fy);
        for tx in 0..tw {
            let (x0, x1) = (tx as f64 * fx, (tx + 1) as f64 * fx);
            let (mut sum, mut area) = (0f64, 0f64);
            let mut y = y0.floor() as usize;
            while (y as f64) < y1 && y < h {
                let wy = (y1.min((y + 1) as f64) - y0.max(y as f64)).max(0.0);
                let mut x = x0.floor() as usize;
                while (x as f64) < x1 && x < w {
                    let wx = (x1.min((x + 1) as f64) - x0.max(x as f64)).max(0.0);
                    let a = wx * wy;
                    sum += luma[y * w + x] as f64 * a;
                    area += a;
                    x += 1;
                }
                y += 1;
            }
            out[ty * tw + tx] = if area > 0.0 { (sum / area) as f32 } else { 0.0 };
        }
    }
    out
}

fn bits_of(cells: impl Iterator<Item = bool>) -> u64 {
    cells.fold(0u64, |acc, b| (acc << 1) | b as u64)
}

fn ahash(luma: &[u8], w: usize, h: usize) -> u64 {
    let cells = shrink(luma, w, h, 8, 8);
    let mean = cells.iter().sum::<f32>() / 64.0;
    bits_of(cells.iter().map(|&c| c > mean))
}

fn dhash(luma: &[u8], w: usize, h: usize) -> u64 {
    let cells = shrink(luma, w, h, 9, 8);
    bits_of((0..8).flat_map(|row| {
        let cells = &cells;
        (0..8).map(move |col| cells[row * 9 + col + 1] > cells[row * 9 + col])
    }))
}

/// The 32-point DCT-II basis, `[k][n] = cos(π (2n + 1) k / 64)`.
fn dct_basis() -> &'static [[f64; 32]; 32] {
    static BASIS: OnceLock<[[f64; 32]; 32]> = OnceLock::new();
    BASIS.get_or_init(|| {
        let mut b = [[0f64; 32]; 32];
        for (k, row) in b.iter_mut().enumerate() {
            for (n, v) in row.iter_mut().enumerate() {
                *v = (std::f64::consts::PI * (2 * n + 1) as f64 * k as f64 / 64.0).cos();
            }
        }
        b
    })
}

fn phash(luma: &[u8], w: usize, h: usize) -> u64 {
    let cells = shrink(luma, w, h, 32, 32);
    let basis = dct_basis();
    // Only the 8×8 lowest frequencies are kept, so only those are computed:
    // rows first (32 rows × 8 frequencies), then columns (8 × 8).
    let mut rows = [[0f64; 8]; 32];
    for (y, row) in rows.iter_mut().enumerate() {
        for (k, v) in row.iter_mut().enumerate() {
            *v = (0..32).map(|x| cells[y * 32 + x] as f64 * basis[k][x]).sum();
        }
    }
    let mut low = [0f64; 64];
    for ky in 0..8 {
        for kx in 0..8 {
            low[ky * 8 + kx] = (0..32).map(|y| rows[y][kx] * basis[ky][y]).sum();
        }
    }
    let mut sorted = low;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = (sorted[31] + sorted[32]) / 2.0;
    bits_of(low.iter().map(|&c| c > median))
}
