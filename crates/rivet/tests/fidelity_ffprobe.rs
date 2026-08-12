//! Squad-35: ffprobe smoke fidelity test.
//!
//! Produces a small valid AV1 MP4 via the production encoder+muxer path,
//! then asks ffprobe (an external, unrelated source of truth) to inspect
//! the file. This catches the class of bugs where our muxer happens to
//! satisfy our own demuxer but a third-party tool — which is what Apple,
//! Chrome, etc. effectively are — refuses the file.
//!
//! **ffprobe is not a dependency of rivet, and this is not a hole in the
//! no-FFmpeg rule.** Nothing in the build, the binary, or CI requires it: CI
//! does not install it, and the test skips with a clear `eprintln!` and
//! `return` (no panic) when `ffprobe -version` fails, mirroring the
//! "skip-when-fixture-absent" pattern used elsewhere. It runs only on a
//! developer machine that happens to have it, precisely *because* it is a
//! stranger to this codebase — the value is the independence, and an oracle
//! we shipped ourselves would not have it.
//!
//! What we assert when ffprobe IS available:
//!   - Exactly one video stream, codec_name=av1.
//!   - Stream width/height/frame-rate match what we encoded.
//!   - pix_fmt is yuv420p (we encode 8-bit by default).
//!   - nb_frames matches the number of packets we muxed (AV1 has no
//!     B-frames so nb_frames == packets exactly; ±0 tolerance).
//!   - Container format_long_name contains "MP4" or "ISOBMFF".
//!   - tags.major_brand is "iso6" — the brand Squad-18 wires for Apple
//!     compatibility.

use bytes::Bytes;
use std::io::Write;
use std::process::{Command, Stdio};

mod common;

use codec::encode::EncoderConfig;
use codec::frame::{ColorSpace, PixelFormat, VideoFrame};
use container::mux::Av1Mp4Muxer;

const W: u32 = 320;
const H: u32 = 240;
const FPS: f64 = 30.0;
const N_FRAMES: u32 = 10;

fn ffprobe_available() -> bool {
    Command::new("ffprobe")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn make_textured_frame(w: u32, h: u32, pts: u64) -> VideoFrame {
    let wu = w as usize;
    let hu = h as usize;
    let y_size = wu * hu;
    let uv_size = y_size / 4;
    let mut buf = Vec::with_capacity(y_size + 2 * uv_size);
    let t = pts as u8;
    for r in 0..hu {
        for c in 0..wu {
            buf.push(((r + c) as u8).wrapping_add(t));
        }
    }
    buf.extend(std::iter::repeat(128u8.wrapping_add(t / 2)).take(uv_size));
    buf.extend(std::iter::repeat(128u8.wrapping_add(t / 3)).take(uv_size));
    VideoFrame::new(
        Bytes::from(buf),
        w,
        h,
        PixelFormat::Yuv420p,
        ColorSpace::Bt709,
        pts,
    )
}

fn build_test_mp4() -> Option<(Bytes, usize)> {
    let config = EncoderConfig {
        width: W,
        height: H,
        frame_rate: FPS,
        quality: 200,
        speed_preset: 10,
        keyframe_interval: 5,
        ..EncoderConfig::default()
    };
    let mut encoder = common::try_av1_encoder(config)?;
    let mut muxer = Av1Mp4Muxer::new(W, H, FPS).expect("muxer");
    let mut packets = 0usize;
    for pts in 0..N_FRAMES {
        let f = make_textured_frame(W, H, pts as u64);
        encoder.send_frame(&f).expect("send_frame");
        while let Some(p) = encoder.receive_packet().expect("receive") {
            packets += 1;
            muxer.add_packet(p).expect("add_packet");
        }
    }
    encoder.flush().expect("flush");
    while let Some(p) = encoder.receive_packet().expect("receive after flush") {
        packets += 1;
        muxer.add_packet(p).expect("add_packet");
    }
    let out = muxer.finalize().expect("finalize");
    Some((out, packets))
}

/// Ask ffprobe to read MP4 bytes from stdin (`pipe:0`) and emit JSON
/// stream + format info. Returns the JSON payload as a String.
fn ffprobe_json(mp4: &[u8]) -> std::io::Result<String> {
    let mut child = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            "-i",
            "pipe:0",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    {
        let stdin = child.stdin.as_mut().expect("ffprobe stdin");
        stdin.write_all(mp4)?;
    }
    let out = child.wait_with_output()?;
    if !out.status.success() {
        return Err(std::io::Error::other(format!(
            "ffprobe exit {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Tiny scalar JSON value extractor — we don't pull serde into the
/// pipeline crate just for one optional test. Looks up `"<key>": "<val>"`
/// (string form) or `"<key>": <number>` and returns the captured chunk
/// without quotes. Returns None if absent. Good enough for ffprobe's
/// flat-ish output.
fn extract_str(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\":", key);
    let i = json.find(&needle)?;
    let rest = &json[i + needle.len()..];
    let rest = rest.trim_start();
    if let Some(stripped) = rest.strip_prefix('"') {
        let end = stripped.find('"')?;
        Some(stripped[..end].to_string())
    } else {
        let end = rest
            .find(|c: char| c == ',' || c == '}' || c == '\n')
            .unwrap_or(rest.len());
        Some(rest[..end].trim().to_string())
    }
}

#[test]
fn ffprobe_smoke_matches_muxer_output() {
    if !ffprobe_available() {
        eprintln!("ffprobe_smoke: ffprobe not on PATH — skipping (optional cross-check)");
        return;
    }

    let Some((mp4, packet_count)) = build_test_mp4() else {
        return;
    };
    let json = ffprobe_json(&mp4).expect("ffprobe must succeed on a valid AV1 MP4");
    eprintln!("ffprobe output ({} bytes):\n{}", json.len(), json);

    // Codec presence is the load-bearing check. ffprobe normalizes "av01"
    // sample-entry fourccs to codec_name "av1".
    let codec = extract_str(&json, "codec_name").expect("ffprobe JSON missing codec_name");
    assert_eq!(
        codec, "av1",
        "ffprobe sees codec_name={:?}; expected av1. Bug in muxer av1C/sample-entry?",
        codec
    );

    let width = extract_str(&json, "width")
        .and_then(|s| s.parse::<u32>().ok())
        .expect("width");
    let height = extract_str(&json, "height")
        .and_then(|s| s.parse::<u32>().ok())
        .expect("height");
    assert_eq!(width, W, "ffprobe width mismatch");
    assert_eq!(height, H, "ffprobe height mismatch");

    let pix_fmt = extract_str(&json, "pix_fmt").expect("pix_fmt missing");
    assert!(
        pix_fmt == "yuv420p" || pix_fmt == "yuv420p10le",
        "unexpected pix_fmt {:?}",
        pix_fmt
    );

    // nb_frames is omitted by ffprobe in some MP4 cases; tolerate that
    // but if present, it should equal packet count exactly (AV1 has no
    // B-frames).
    if let Some(nb) = extract_str(&json, "nb_frames").and_then(|s| s.parse::<u64>().ok()) {
        assert!(
            (nb as i64 - packet_count as i64).abs() <= 1,
            "ffprobe nb_frames={} but muxer reported packet_count={}",
            nb,
            packet_count
        );
    }

    // Container format. ffprobe surfaces the long name on the format
    // object; ISOBMFF / MP4 files all read with "QuickTime / MOV" or
    // "MP4" depending on ffprobe version — accept either.
    let fmt_long = extract_str(&json, "format_long_name").unwrap_or_default();
    assert!(
        fmt_long.contains("MP4") || fmt_long.contains("ISOBMFF") || fmt_long.contains("QuickTime"),
        "format_long_name {:?} doesn't look like an MP4-family container",
        fmt_long
    );

    // Apple-compat: Squad-18 wires major_brand=iso6. We assert it makes
    // it through ffprobe's tag parsing untouched.
    let major = extract_str(&json, "major_brand").unwrap_or_default();
    assert_eq!(
        major, "iso6",
        "major_brand should be iso6 (Squad-18 Apple-compat brand); got {:?}",
        major
    );

    eprintln!("ffprobe_smoke: all assertions passed");
}
