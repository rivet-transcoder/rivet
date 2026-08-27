//! AMF decode pixel verification — runs the live AMF decoder on the dev
//! box's AMD GPU and compares every frame **byte for byte** against two
//! independent decoders of the same stream: ffmpeg (`-f rawvideo`) and the
//! in-tree `h26x` software decoders (for H.264 / HEVC). Conformant decoders
//! of the same bitstream are bit-exact, so anything but equality is a bug
//! (a wrong plane pitch, a missed `>>6`, a lost frame at the flush …).
//!
//! Needs: the `amd` feature, an AMD GPU the AMF runtime drives, and an
//! `ffmpeg` (env `FFMPEG`, the scoop path, or `ffmpeg` on PATH) to make the
//! clips and the reference frames. Each of those it cannot find makes the
//! test print `SKIPPED` and pass — a test that cannot run is not evidence,
//! and it says so.
#![cfg(feature = "amd")]

use std::path::PathBuf;
use std::process::Command;

use codec::decode::Decoder;
use codec::frame::{PixelFormat, StreamInfo};

fn ffmpeg() -> Option<PathBuf> {
    let candidates = [
        std::env::var("FFMPEG").ok(),
        Some("C:/Users/elyci/scoop/apps/ffmpeg/current/bin/ffmpeg.exe".to_string()),
        Some("ffmpeg".to_string()),
    ];
    for c in candidates.into_iter().flatten() {
        if Command::new(&c).arg("-version").output().is_ok_and(|o| o.status.success()) {
            return Some(PathBuf::from(c));
        }
    }
    None
}

fn amd_present() -> bool {
    codec::gpu::detect_gpus().iter().any(|g| g.vendor == codec::gpu::GpuVendor::Amd)
}

struct Clip {
    name: &'static str,
    /// The codec label `container::demux` should report.
    codec: &'static str,
    pix_fmt: &'static str,
    encode_args: &'static [&'static str],
}

const CLIPS: &[Clip] = &[
    Clip {
        name: "h264_8bit_nob",
        codec: "h264",
        pix_fmt: "yuv420p",
        // No B-frames: decode order is display order, no DPB reordering.
        encode_args: &["-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20", "-g", "30", "-bf", "0"],
    },
    Clip {
        name: "h264_8bit",
        codec: "h264",
        pix_fmt: "yuv420p",
        // B-frames on, so display order != decode order and the decoder's
        // reordering is exercised.
        encode_args: &["-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "20", "-g", "30", "-bf", "3"],
    },
    Clip {
        name: "hevc_8bit",
        codec: "hevc",
        pix_fmt: "yuv420p",
        encode_args: &["-pix_fmt", "yuv420p", "-c:v", "libx265", "-crf", "20", "-g", "30", "-x265-params", "bframes=3", "-tag:v", "hvc1"],
    },
    Clip {
        name: "hevc_main10",
        codec: "hevc",
        pix_fmt: "yuv420p10le",
        encode_args: &["-pix_fmt", "yuv420p10le", "-c:v", "libx265", "-crf", "20", "-g", "30", "-x265-params", "bframes=3", "-tag:v", "hvc1"],
    },
    Clip {
        name: "av1_8bit",
        codec: "av1",
        pix_fmt: "yuv420p",
        encode_args: &["-pix_fmt", "yuv420p", "-c:v", "libsvtav1", "-crf", "30", "-g", "30"],
    },
];

/// Make the clip with ffmpeg (once; reused across runs) and return its bytes.
fn make_clip(ffmpeg: &PathBuf, clip: &Clip) -> Option<Vec<u8>> {
    let dir = std::env::temp_dir().join("rivet_amf_decode_pixels");
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join(format!("{}.mp4", clip.name));
    if !path.exists() {
        let mut cmd = Command::new(ffmpeg);
        cmd.args(["-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=2"])
            .args(clip.encode_args)
            .arg(&path);
        let out = cmd.output().ok()?;
        if !out.status.success() {
            eprintln!("  ffmpeg could not make {}: {}", clip.name, String::from_utf8_lossy(&out.stderr));
            return None;
        }
    }
    std::fs::read(&path).ok()
}

/// ffmpeg's decode of the clip, as raw planar frames.
fn reference_frames(ffmpeg: &PathBuf, data_path: &PathBuf, pix_fmt: &str, frame_bytes: usize) -> Vec<Vec<u8>> {
    let out = Command::new(ffmpeg)
        .args(["-v", "error", "-i"])
        .arg(data_path)
        .args(["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"])
        .output()
        .expect("ffmpeg rawvideo decode");
    assert!(out.status.success(), "ffmpeg decode failed: {}", String::from_utf8_lossy(&out.stderr));
    out.stdout.chunks_exact(frame_bytes).map(|c| c.to_vec()).collect()
}

fn run_decoder(mut dec: Box<dyn Decoder>, samples: &[Vec<u8>]) -> anyhow::Result<Vec<codec::frame::VideoFrame>> {
    for s in samples {
        dec.push_sample(s)?;
    }
    dec.finish()?;
    let mut frames = Vec::new();
    while let Some(f) = dec.decode_next()? {
        frames.push(f);
    }
    Ok(frames)
}

/// Per-plane worst absolute difference and luma PSNR, for the failure message.
fn describe_diff(a: &[u8], b: &[u8], w: usize, h: usize, ten_bit: bool) -> String {
    let luma = w * h * if ten_bit { 2 } else { 1 };
    let (mut max, mut se) = (0i64, 0f64);
    let n = luma.min(a.len()).min(b.len());
    if ten_bit {
        for i in (0..n).step_by(2) {
            let x = u16::from_le_bytes([a[i], a[i + 1]]) as i64;
            let y = u16::from_le_bytes([b[i], b[i + 1]]) as i64;
            max = max.max((x - y).abs());
            se += ((x - y) * (x - y)) as f64;
        }
        let mse = se / (n / 2) as f64;
        format!("luma max|diff|={max} psnr={:.2} dB", 10.0 * (1023f64 * 1023.0 / mse.max(1e-9)).log10())
    } else {
        for i in 0..n {
            let d = a[i] as i64 - b[i] as i64;
            max = max.max(d.abs());
            se += (d * d) as f64;
        }
        let mse = se / n as f64;
        format!("luma max|diff|={max} psnr={:.2} dB", 10.0 * (255f64 * 255.0 / mse.max(1e-9)).log10())
    }
}

#[test]
fn amf_decode_is_bit_exact_against_ffmpeg_and_h26x() {
    if !amd_present() {
        eprintln!("SKIPPED: no AMD GPU on this machine");
        return;
    }
    let Some(ffmpeg) = ffmpeg() else {
        eprintln!("SKIPPED: no ffmpeg found (set FFMPEG)");
        return;
    };
    let caps = codec::decode::amf_dec::probe_decode_caps();
    eprintln!("AMF decode probe: {caps:?}");
    if caps.is_empty() {
        eprintln!("SKIPPED: the AMF runtime drives no decoder on this GPU");
        return;
    }

    let mut verified = Vec::new();
    for clip in CLIPS {
        let Some(data) = make_clip(&ffmpeg, clip) else {
            eprintln!("  {}: SKIPPED (could not make the clip)", clip.name);
            continue;
        };
        let path = std::env::temp_dir().join("rivet_amf_decode_pixels").join(format!("{}.mp4", clip.name));
        let demuxed = container::demux::demux(&data).expect("demux");
        assert_eq!(demuxed.codec.to_ascii_lowercase(), clip.codec, "{}: demuxed codec", clip.name);
        let info: StreamInfo = demuxed.info.clone();
        let ten_bit = clip.pix_fmt == "yuv420p10le";
        let (w, h) = (info.width as usize, info.height as usize);
        let frame_bytes = w * h * 3 / 2 * if ten_bit { 2 } else { 1 };
        eprintln!(
            "  {}: {} {}x{} {:?} {} samples",
            clip.name, demuxed.codec, w, h, info.pixel_format, demuxed.samples.len()
        );

        // A codec this GPU has no decoder for must refuse at construction —
        // cleanly, and consistently with the probe.
        let amf = codec::decode::amf_dec::AmfDecoder::new(info.clone(), 0);
        if !caps.contains(&clip.codec) {
            let err = amf.err().map(|e| format!("{e:#}")).unwrap_or_else(|| "(constructed!)".into());
            eprintln!("  {}: this GPU has no AMF {} decoder; AmfDecoder::new -> {err}", clip.name, clip.codec);
            assert!(err.contains("AMF"), "{}: refusal names AMF: {err}", clip.name);
            continue;
        }
        let amf = amf.unwrap_or_else(|e| panic!("{}: AmfDecoder::new: {e:#}", clip.name));
        let frames = run_decoder(Box::new(amf), &demuxed.samples).unwrap_or_else(|e| panic!("{}: AMF decode: {e:#}", clip.name));

        let reference = reference_frames(&ffmpeg, &path, clip.pix_fmt, frame_bytes);
        if std::env::var("AMF_DEC_SW_FIRST").is_ok() && clip.codec != "av1" {
            let sw = codec::decode::h26x_sw::H26xDecoder::new(info.clone()).unwrap();
            let sw_frames = run_decoder(Box::new(sw), &demuxed.samples).unwrap();
            let m: Vec<String> = sw_frames.iter().map(|f| reference.iter().position(|r| r[..] == f.data[..]).map_or("?".into(), |i| i.to_string())).collect();
            eprintln!("EXPERIMENT h26x_sw on the same samples: {} frames, matched [{}]", sw_frames.len(), m.join(" "));
        }
        if frames.len() != reference.len() {
            // Which reference frames came back, in which order — tells a
            // dropped head from a lost tail from a reorder bug.
            let matches: Vec<String> = frames
                .iter()
                .map(|f| reference.iter().position(|r| r[..] == f.data[..]).map_or("?".into(), |i| i.to_string()))
                .collect();
            panic!(
                "{}: {} frames from AMF vs {} from ffmpeg; AMF frames matched reference indices [{}]",
                clip.name,
                frames.len(),
                reference.len(),
                matches.join(" ")
            );
        }
        for (i, (f, r)) in frames.iter().zip(&reference).enumerate() {
            assert_eq!(f.width as usize, w);
            assert_eq!(f.height as usize, h);
            assert_eq!(f.pts, i as u64, "{}: frame {i} pts", clip.name);
            assert_eq!(
                f.format,
                if ten_bit { PixelFormat::Yuv420p10le } else { PixelFormat::Yuv420p },
                "{}: frame {i} format",
                clip.name
            );
            assert_eq!(f.data.len(), frame_bytes, "{}: frame {i} size", clip.name);
            if f.data[..] != r[..] {
                panic!(
                    "{}: frame {i} differs from ffmpeg: {}",
                    clip.name,
                    describe_diff(&f.data, r, w, h, ten_bit)
                );
            }
        }
        eprintln!("  {}: {} frames bit-exact vs ffmpeg", clip.name, frames.len());

        if clip.codec != "av1" {
            let sw = codec::decode::h26x_sw::H26xDecoder::new(info.clone())
                .unwrap_or_else(|e| panic!("{}: H26xDecoder::new: {e:#}", clip.name));
            let sw_frames = run_decoder(Box::new(sw), &demuxed.samples)
                .unwrap_or_else(|e| panic!("{}: h26x decode: {e:#}", clip.name));
            assert_eq!(frames.len(), sw_frames.len(), "{}: frame count vs h26x", clip.name);
            for (i, (f, s)) in frames.iter().zip(&sw_frames).enumerate() {
                if f.data[..] != s.data[..] {
                    panic!(
                        "{}: frame {i} differs from the h26x software decoder: {}",
                        clip.name,
                        describe_diff(&f.data, &s.data, w, h, ten_bit)
                    );
                }
            }
            eprintln!("  {}: {} frames bit-exact vs the h26x software decoder", clip.name, frames.len());
        }
        verified.push(clip.name);
    }
    eprintln!("AMF decode verified bit-exact on this machine: {verified:?}");
    assert!(verified.contains(&"h264_8bit") && verified.contains(&"hevc_8bit"), "{verified:?}");
}
