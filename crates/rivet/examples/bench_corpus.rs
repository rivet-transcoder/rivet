//! Generates the quality bench's corpus (`bench/README.md`): four 1080p30
//! clips, one per content type a ladder has to cope with, each opening on a
//! three-second fade from black. Made entirely by this workspace's own H.264
//! encoder and MP4 muxer, from generators written here; no external program.
//!
//! ```text
//! cargo run --release --example bench_corpus -- OUT_DIR [--seconds 20] [--size 1920x1080]
//! ```
//!
//! - `grain.mp4`: the test pattern averaged with per-sample random luma at
//!   35% — detail under heavy sensor-like noise.
//! - `flat.mp4`: a solid background with two solid boxes — large flat
//!   regions, hard edges, no texture.
//! - `motion.mp4`: the test pattern turning (3 rad/s) and zooming
//!   (1.4 + 0.3·sin(n/8)) — high temporal energy.
//! - `dark.mp4`: the test pattern through a dark tone curve (0 → 0,
//!   ½ → 0.09, 1 → 0.22, minus 6%) — low luma with detail in it.
//!
//! Coded at a fixed quantiser of 12, so the sources are close to their
//! generators.

#[path = "../tests/common/synth.rs"]
#[allow(dead_code)]
mod synth;

use h26x::ChromaFormat;
use synth::{H264, Rng};

const FADE_SECONDS: f64 = 3.0;

/// BT.709 limited-range Y'CbCr of an sRGB-ish 8-bit colour.
fn yuv(rgb: u32) -> (u8, u8, u8) {
    let (r, g, b) = (f64::from((rgb >> 16) & 255) / 255.0, f64::from((rgb >> 8) & 255) / 255.0, f64::from(rgb & 255) / 255.0);
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let cb = (b - y) / 1.8556;
    let cr = (r - y) / 1.5748;
    ((16.0 + 219.0 * y).round() as u8, (128.0 + 224.0 * cb).round() as u8, (128.0 + 224.0 * cr).round() as u8)
}

/// The fade from black over the first three seconds: luma toward 16,
/// chroma toward 128.
fn fade(p: &mut [u8], w: usize, h: usize, t: f64) {
    if t >= FADE_SECONDS {
        return;
    }
    let k = t / FADE_SECONDS;
    let luma = w * h;
    for (i, s) in p.iter_mut().enumerate() {
        let base = if i < luma { 16.0 } else { 128.0 };
        *s = (base + (f64::from(*s) - base) * k).round() as u8;
    }
}

/// Bilinear sample of plane `src` (`w x h`) at `(x, y)`, clamped to the edge.
fn sample(src: &[u8], w: usize, h: usize, x: f64, y: f64) -> u8 {
    let x = x.clamp(0.0, (w - 1) as f64);
    let y = y.clamp(0.0, (h - 1) as f64);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let at = |xx: usize, yy: usize| f64::from(src[yy * w + xx]);
    let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
    let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
    (top * (1.0 - fy) + bottom * fy).round() as u8
}

/// Frame `n` of `kind`.
fn frame(kind: &str, w: u32, h: u32, n: u64, fps: u32, rng: &mut Rng, still: &[u8]) -> Vec<u8> {
    let (wu, hu) = (w as usize, h as usize);
    let (cw, ch) = (wu.div_ceil(2), hu.div_ceil(2));
    let t = n as f64 / f64::from(fps);
    let mut p = match kind {
        "grain" => {
            let mut p = synth::test_pattern(w, h, n, ChromaFormat::Yuv420);
            for s in &mut p[..wu * hu] {
                let noise = (rng.next_u64() >> 56) as f64;
                *s = (f64::from(*s) * 0.65 + noise * 0.35).round().clamp(16.0, 235.0) as u8;
            }
            p
        }
        "flat" => {
            let mut p = synth::blank(w, h, ChromaFormat::Yuv420);
            let paint = |p: &mut [u8], (x0, y0, bw, bh): (usize, usize, usize, usize), rgb: u32| {
                let (y, u, v) = yuv(rgb);
                for yy in y0..(y0 + bh).min(hu) {
                    for xx in x0..(x0 + bw).min(wu) {
                        p[yy * wu + xx] = y;
                    }
                }
                for yy in y0 / 2..((y0 + bh) / 2).min(ch) {
                    for xx in x0 / 2..((x0 + bw) / 2).min(cw) {
                        p[wu * hu + yy * cw + xx] = u;
                        p[wu * hu + cw * ch + yy * cw + xx] = v;
                    }
                }
            };
            // Proportions of the 1920x1080 layout.
            let sx = |v: usize| v * wu / 1920;
            let sy = |v: usize| v * hu / 1080;
            paint(&mut p, (0, 0, wu, hu), 0x1e3a5f);
            paint(&mut p, (sx(200), sy(150), sx(700), sy(500)), 0xf5d76e);
            paint(&mut p, (sx(1000), sy(400), sx(600), sy(400)), 0xe8543f);
            p
        }
        "motion" => {
            let angle = t * 3.0;
            let zoom = 1.4 + 0.3 * (n as f64 / 8.0).sin();
            let (c, s) = (angle.cos(), angle.sin());
            let mut p = synth::blank(w, h, ChromaFormat::Yuv420);
            for (plane, pw, ph, off) in [(0, wu, hu, 0), (1, cw, ch, wu * hu), (2, cw, ch, wu * hu + cw * ch)] {
                let src = &still[off..off + pw * ph];
                let (cx, cy) = (pw as f64 / 2.0, ph as f64 / 2.0);
                for y in 0..ph {
                    for x in 0..pw {
                        let (dx, dy) = ((x as f64 - cx) / zoom, (y as f64 - cy) / zoom);
                        let (u, v) = (c * dx + s * dy + cx, -s * dx + c * dy + cy);
                        let inside = (0.0..pw as f64).contains(&u) && (0.0..ph as f64).contains(&v);
                        p[off + y * pw + x] = if inside { sample(src, pw, ph, u, v) } else if plane == 0 { 16 } else { 128 };
                    }
                }
            }
            p
        }
        "dark" => {
            let mut p = synth::test_pattern(w, h, n, ChromaFormat::Yuv420);
            for s in &mut p[..wu * hu] {
                let x = (f64::from(*s) - 16.0) / 219.0;
                let curved = if x <= 0.5 { x / 0.5 * 0.09 } else { 0.09 + (x - 0.5) / 0.5 * 0.13 };
                *s = (16.0 + 219.0 * (curved - 0.06).max(0.0)).round() as u8;
            }
            p
        }
        other => panic!("no clip {other}"),
    };
    fade(&mut p, wu, hu, t);
    p
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = std::path::PathBuf::from(args.first().expect("usage: bench_corpus OUT_DIR [--seconds S] [--size WxH]"));
    let opt = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let seconds: f64 = opt("--seconds").map_or(20.0, |s| s.parse().expect("--seconds S"));
    let (w, h) = opt("--size").map_or((1920, 1080), |s| {
        let (a, b) = s.split_once('x').expect("--size WxH");
        (a.parse().expect("width"), b.parse().expect("height"))
    });
    const FPS: u32 = 30;
    std::fs::create_dir_all(&out).expect("the output directory");
    let still = synth::test_pattern(w, h, 0, ChromaFormat::Yuv420);
    for kind in ["grain", "flat", "motion", "dark"] {
        let n = (seconds * f64::from(FPS)).round() as u64;
        let mut rng = Rng::new(42);
        let pictures: Vec<Vec<u8>> = (0..n).map(|i| frame(kind, w, h, i, FPS, &mut rng, &still)).collect();
        let cfg = H264 { qp: 12, ..H264::new(w, h, FPS) };
        let coded = synth::encode_h264(&cfg, pictures);
        let mp4 = synth::mp4(&coded, w, h, FPS, None, None);
        let path = out.join(format!("{kind}.mp4"));
        std::fs::write(&path, &mp4).expect("write the clip");
        println!("{:<14} {:>11} bytes  {w}x{h}, {n} frames", format!("{kind}.mp4"), mp4.len());
    }
}
