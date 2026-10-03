//! Scores each encoded rung of a ladder against the source it came from,
//! with VMAF and SSIM — the quality bench's scorer (`bench/score-ladder.sh`
//! runs it; see `bench/README.md`).
//!
//! ```text
//! cargo run --release --example bench_score -- SOURCE.mp4 RUNGS_DIR [--window SECONDS]
//! ```
//!
//! `RUNGS_DIR` holds rivet's output as it is: an HLS package
//! (`video/<label>/init.mp4` + `seg-*.m4s`) or single-file rungs
//! (`<label>.mp4`). Decoding, the upscale and SSIM are rivet's own; VMAF is
//! Netflix's `vmaf` command-line tool (libvmaf's, from its release
//! binaries), run as a black box on two Y4M files — `VMAF` names it when it is
//! not on `PATH`. No FFmpeg.
//!
//! What it does that is easy to get wrong:
//! - **Every segment, in order**: a CMAF rung is its init segment and every
//!   media segment joined, as a player sees it.
//! - **The rung is upscaled to the source's size** (bicubic) before the
//!   comparison: a 240p rung is watched stretched to a screen.
//! - **A window from the middle that misses black**: frames that are 98%
//!   black (luma within 10% of black) are found first, and the window walks
//!   forward if the middle lands in them, so a fade from black does not hand
//!   every rung free score. A clip shorter than a window is scored whole.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use codec::frame::{PixelFormat, VideoFrame};

/// Frames in display order out of an MP4 (or joined CMAF) file.
struct Frames {
    demux: Box<dyn container::streaming::StreamingDemuxer>,
    dec: Box<dyn codec::decode::Decoder>,
    flushed: bool,
}

impl Frames {
    fn open(file: &[u8]) -> (Self, u32, u32, f64) {
        let demux = container::streaming::demux_streaming(file).expect("demux");
        let header = demux.header().clone();
        let dec = codec::decode::create_decoder(&header.codec, header.info.clone()).expect("a decoder");
        let (w, h, fps) = (header.info.width, header.info.height, header.info.frame_rate);
        (Self { demux, dec, flushed: false }, w, h, fps)
    }

    fn next(&mut self) -> Option<VideoFrame> {
        loop {
            if let Some(f) = self.dec.decode_next().expect("decode") {
                return Some(to_8bit(f));
            }
            if self.flushed {
                return None;
            }
            match self.demux.next_video_sample().expect("demux") {
                Some(s) => self.dec.push_sample(&s.data).expect("decode"),
                None => {
                    self.dec.finish().expect("flush");
                    self.flushed = true;
                }
            }
        }
    }
}

/// An 8-bit 4:2:0 frame (a 10-bit one is shifted down).
fn to_8bit(f: VideoFrame) -> VideoFrame {
    match f.format {
        PixelFormat::Yuv420p => f,
        PixelFormat::Yuv420p10le => {
            let data: Vec<u8> = f.data.chunks_exact(2).map(|b| (u16::from_le_bytes([b[0], b[1]]) >> 2) as u8).collect();
            VideoFrame::new(data.into(), f.width, f.height, PixelFormat::Yuv420p, f.color_space, f.pts)
        }
        other => panic!("bench_score handles 4:2:0 only, not {other:?}"),
    }
}

/// Whether 98% of the frame's luma is within 10% of black.
fn is_black(f: &VideoFrame) -> bool {
    let n = (f.width * f.height) as usize;
    let dark = f.data[..n].iter().filter(|&&y| y <= 16 + 22).count();
    dark as f64 >= 0.98 * n as f64
}

/// One plane resized with a separable bicubic (Keys, a = -0.5).
fn bicubic(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    fn weights(src_len: usize, dst_len: usize) -> Vec<(isize, [f32; 4])> {
        let scale = src_len as f64 / dst_len as f64;
        (0..dst_len)
            .map(|i| {
                let x = (i as f64 + 0.5) * scale - 0.5;
                let x0 = x.floor();
                let t = x - x0;
                let k = |d: f64| {
                    let d = d.abs();
                    let a = -0.5;
                    if d <= 1.0 {
                        (a + 2.0) * d.powi(3) - (a + 3.0) * d * d + 1.0
                    } else if d < 2.0 {
                        a * d.powi(3) - 5.0 * a * d * d + 8.0 * a * d - 4.0 * a
                    } else {
                        0.0
                    }
                };
                (x0 as isize - 1, [k(t + 1.0) as f32, k(t) as f32, k(1.0 - t) as f32, k(2.0 - t) as f32])
            })
            .collect()
    }
    let (wx, wy) = (weights(sw, dw), weights(sh, dh));
    let clamp = |v: isize, n: usize| v.clamp(0, n as isize - 1) as usize;
    let mut rows = vec![0f32; dw * sh];
    for y in 0..sh {
        for (x, (start, w)) in wx.iter().enumerate() {
            rows[y * dw + x] = (0..4).map(|k| w[k] * f32::from(src[y * sw + clamp(start + k as isize, sw)])).sum();
        }
    }
    let mut out = vec![0u8; dw * dh];
    for (y, (start, w)) in wy.iter().enumerate() {
        for x in 0..dw {
            let v: f32 = (0..4).map(|k| w[k] * rows[clamp(start + k as isize, sh) * dw + x]).sum();
            out[y * dw + x] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// `f` (4:2:0) at `w x h`, bicubic.
fn upscale(f: &VideoFrame, w: u32, h: u32) -> Vec<u8> {
    if (f.width, f.height) == (w, h) {
        return f.data[..(w * h * 3 / 2) as usize].to_vec();
    }
    let (sw, sh) = (f.width as usize, f.height as usize);
    let (scw, sch) = (sw.div_ceil(2), sh.div_ceil(2));
    let (dw, dh) = (w as usize, h as usize);
    let (dcw, dch) = (dw.div_ceil(2), dh.div_ceil(2));
    let y = &f.data[..sw * sh];
    let u = &f.data[sw * sh..sw * sh + scw * sch];
    let v = &f.data[sw * sh + scw * sch..sw * sh + 2 * scw * sch];
    [bicubic(y, sw, sh, dw, dh), bicubic(u, scw, sch, dcw, dch), bicubic(v, scw, sch, dcw, dch)].concat()
}

fn y4m_header(w: u32, h: u32, fps: f64) -> String {
    format!("YUV4MPEG2 W{w} H{h} F{}:1000 Ip A1:1 C420jpeg\n", (fps * 1000.0).round() as u64)
}

/// The rung's file: a single MP4, or a CMAF directory joined.
fn rung_file(entry: &Path) -> Option<(String, Vec<u8>, u64)> {
    if entry.is_dir() {
        let init = entry.join("init.mp4");
        let mut joined = std::fs::read(&init).ok()?;
        let mut segs: Vec<PathBuf> = std::fs::read_dir(entry)
            .ok()?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "m4s"))
            .collect();
        segs.sort();
        let mut bytes = 0;
        for s in &segs {
            let b = std::fs::read(s).ok()?;
            bytes += b.len() as u64;
            joined.extend(b);
        }
        Some((entry.file_name()?.to_string_lossy().into_owned(), joined, bytes))
    } else if entry.extension().is_some_and(|x| x == "mp4") {
        let b = std::fs::read(entry).ok()?;
        let n = b.len() as u64;
        Some((entry.file_stem()?.to_string_lossy().into_owned(), b, n))
    } else {
        None
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: bench_score SOURCE.mp4 RUNGS_DIR [--window SECONDS]";
    let src_path = args.first().expect(usage);
    let mut rungs_dir = PathBuf::from(args.get(1).expect(usage));
    let window: f64 = args.iter().position(|a| a == "--window").map_or(10.0, |i| args[i + 1].parse().expect("--window S"));
    let vmaf = std::env::var("VMAF").unwrap_or_else(|_| "vmaf".into());
    let source = std::fs::read(src_path).expect("the source");

    // The black stretches, frame by frame.
    let (mut frames, sw, sh, fps) = Frames::open(&source);
    let mut black = Vec::new();
    while let Some(f) = frames.next() {
        black.push(is_black(&f));
    }
    let total = black.len();
    let win = ((window * fps).round() as usize).min(total);
    let overlaps = |s: usize| black[s..s + win].iter().any(|&b| b);
    let mut start = (total - win) / 2;
    if win == total {
        println!("source {sw}x{sh}, {total} frames — shorter than a window, scoring all of it");
    } else {
        let mut tries = 0;
        while overlaps(start) && tries < 4 {
            println!("  window at {:.1}s is black; moving on", start as f64 / fps);
            start += win;
            tries += 1;
            if start + win > total {
                start = (total - win) / 2;
                println!("  every window tried was black; scoring the middle anyway");
                break;
            }
        }
        println!("source {sw}x{sh}, {:.1}s — scoring {window}s from {:.1}s", total as f64 / fps, start as f64 / fps);
    }

    let tmp = tempfile::tempdir().expect("temp dir");
    println!("{:<10} {:>10} {:>10} {:>8}", "rung", "bytes", "vmaf", "ssim");
    if rungs_dir.join("video").is_dir() {
        rungs_dir = rungs_dir.join("video");
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&rungs_dir).expect("the rungs").filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for entry in entries {
        let Some((name, file, bytes)) = rung_file(&entry) else { continue };
        let (ref_path, dist_path) = (tmp.path().join("ref.y4m"), tmp.path().join("dist.y4m"));
        let mut r = std::io::BufWriter::new(std::fs::File::create(&ref_path).unwrap());
        let mut d = std::io::BufWriter::new(std::fs::File::create(&dist_path).unwrap());
        r.write_all(y4m_header(sw, sh, fps).as_bytes()).unwrap();
        d.write_all(y4m_header(sw, sh, fps).as_bytes()).unwrap();
        let (mut a, ..) = Frames::open(&source);
        let (mut b, ..) = Frames::open(&file);
        let mut ssim = Vec::new();
        for i in 0..start + win {
            let (Some(fa), Some(fb)) = (a.next(), b.next()) else { break };
            if i < start {
                continue;
            }
            let ref_pic = &fa.data[..(sw * sh * 3 / 2) as usize];
            let dist_pic = upscale(&fb, sw, sh);
            let n = (sw * sh) as usize;
            ssim.push(codec::quality::ssim_8bit(&ref_pic[..n], &dist_pic[..n], sw as usize, sh as usize));
            r.write_all(b"FRAME\n").unwrap();
            r.write_all(ref_pic).unwrap();
            d.write_all(b"FRAME\n").unwrap();
            d.write_all(&dist_pic).unwrap();
        }
        drop((r, d));
        let json = tmp.path().join("vmaf.json");
        let out = Command::new(&vmaf)
            .args(["--json", "--threads", "4", "-q", "-r"])
            .arg(&ref_path)
            .arg("-d")
            .arg(&dist_path)
            .arg("-o")
            .arg(&json)
            .output();
        let score = match out {
            Ok(o) if o.status.success() => std::fs::read_to_string(&json)
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                .and_then(|v| v["pooled_metrics"]["vmaf"]["mean"].as_f64())
                .map_or("n/a".to_string(), |v| format!("{v:.2}")),
            Ok(o) => {
                eprintln!("{name}: vmaf failed: {}", String::from_utf8_lossy(&o.stderr));
                "n/a".into()
            }
            Err(e) => {
                eprintln!("{name}: cannot run `{vmaf}` ({e}); set VMAF");
                "n/a".into()
            }
        };
        let ssim = if ssim.is_empty() { "n/a".into() } else { format!("{:.4}", ssim.iter().sum::<f64>() / ssim.len() as f64) };
        println!("{name:<10} {bytes:>10} {score:>10} {ssim:>8}");
    }
}
