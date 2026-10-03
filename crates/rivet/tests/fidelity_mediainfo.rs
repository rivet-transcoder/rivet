//! An independent reader's view of rivet's MP4s: MediaInfo smoke test.
//!
//! Produces small MP4s through the production encoder + muxer paths — AV1
//! (when this build/host has an AV1 encoder), H.264 and H.265 (this
//! workspace's software encoders) — and asks MediaInfo (MediaArea's
//! `mediainfo` command-line tool, a stranger to this codebase with its own
//! ISOBMFF and bitstream parsers) what it sees. This catches the class of
//! bugs where our muxer happens to satisfy our own demuxer but a
//! third-party reader — which is what Apple, Chrome, etc. effectively are —
//! reads the file differently or refuses it.
//!
//! MediaInfo is not a dependency of rivet: the test skips with a line saying
//! so when it is not found (`MEDIAINFO`, else `mediainfo` on PATH), unless
//! `RIVET_REQUIRE_MEDIAINFO=1` is set, as in CI, where a missing tool is a
//! failure.
//!
//! Asserted, per file:
//!   - General: an ISOBMFF file with major brand `iso6` (the Apple-compat brand
//!     the muxer writes).
//!   - Video: format AV1 / AVC / HEVC, codec ID `av01` / `avc1` / `hvc1`,
//!     width and height as encoded, square samples, 4:2:0 at 8 bits, frame rate, and a
//!     frame count equal to the packets muxed.

use bytes::Bytes;
use std::process::Command;

mod common;

use codec::encode::{EncoderBackend, EncoderConfig, select_encoder};
use codec::frame::{ColorSpace, PixelFormat, VideoCodec, VideoFrame};
use container::mux::Av1Mp4Muxer;

const W: u32 = 320;
const H: u32 = 240;
const FPS: f64 = 30.0;
const N_FRAMES: u32 = 10;

/// The MediaInfo binary, or `None` (said so) where there is none.
fn mediainfo() -> Option<String> {
    let bin = std::env::var("MEDIAINFO").unwrap_or_else(|_| "mediainfo".into());
    let ok = Command::new(&bin).arg("--Version").output().is_ok_and(|o| o.status.success());
    if ok {
        return Some(bin);
    }
    assert!(
        std::env::var_os("RIVET_REQUIRE_MEDIAINFO").is_none(),
        "RIVET_REQUIRE_MEDIAINFO is set but `{bin}` does not run"
    );
    eprintln!("SKIP: no mediainfo (set MEDIAINFO or put it on PATH)");
    None
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

/// An MP4 of `N_FRAMES` textured frames from `encoder` and the number of
/// packets muxed.
fn build_mp4(codec: VideoCodec, mut encoder: Box<dyn codec::encode::Encoder>) -> (Bytes, usize) {
    let mut muxer = Av1Mp4Muxer::new_with_codec(W, H, FPS, codec).expect("muxer");
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
    (muxer.finalize().expect("finalize"), packets)
}

fn config(codec: VideoCodec) -> EncoderConfig {
    EncoderConfig {
        width: W,
        height: H,
        frame_rate: FPS,
        quality: 200,
        speed_preset: 10,
        keyframe_interval: 5,
        codec,
        threads: 1,
        ..EncoderConfig::default()
    }
}

/// MediaInfo's JSON for `mp4`.
fn mediainfo_json(bin: &str, mp4: &[u8], name: &str) -> String {
    let path = std::env::temp_dir().join(format!("rivet-mediainfo-{}-{name}.mp4", std::process::id()));
    std::fs::write(&path, mp4).unwrap();
    let out = Command::new(bin).arg("--Output=JSON").arg(&path).output().expect("mediainfo runs");
    let _ = std::fs::remove_file(&path);
    assert!(out.status.success(), "mediainfo failed on {name}: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The fields of the track whose `@type` is `kind`, as a flat
/// `"key": "value"` lookup (MediaInfo writes every value as a string).
fn track<'a>(json: &'a str, kind: &str) -> &'a str {
    let at = json.find(&format!("\"@type\":\"{kind}\"")).unwrap_or_else(|| panic!("no {kind} track: {json}"));
    let rest = &json[at..];
    &rest[..rest.find('}').unwrap_or(rest.len())]
}

fn field(track: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":\"");
    let i = track.find(&needle)? + needle.len();
    Some(track[i..i + track[i..].find('"')?].to_string())
}

fn check(bin: &str, name: &str, mp4: &[u8], packets: usize, format: &str, codec_id: &str) {
    let json = mediainfo_json(bin, mp4, name);
    let general = track(&json, "General");
    // MediaInfo names an ISOBMFF file by its major brand when it has no
    // other name for it ("MPEG-4" for the older brands).
    let container = field(general, "Format").unwrap_or_default();
    assert!(container == "MPEG-4" || container == "iso6", "{name}: container {container:?}: {json}");
    assert_eq!(field(general, "CodecID").as_deref(), Some("iso6"), "{name}: major brand: {json}");
    assert_eq!(field(general, "VideoCount").as_deref(), Some("1"), "{name}: one video track");
    let video = track(&json, "Video");
    assert_eq!(field(video, "Format").as_deref(), Some(format), "{name}: video format: {json}");
    assert_eq!(field(video, "CodecID").as_deref(), Some(codec_id), "{name}: sample entry: {json}");
    assert_eq!(field(video, "Width").as_deref(), Some(W.to_string().as_str()), "{name}: width");
    assert_eq!(field(video, "Height").as_deref(), Some(H.to_string().as_str()), "{name}: height");
    assert_eq!(field(video, "ChromaSubsampling").as_deref(), Some("4:2:0"), "{name}: chroma: {json}");
    assert_eq!(field(video, "BitDepth").as_deref(), Some("8"), "{name}: depth: {json}");
    assert_eq!(field(video, "PixelAspectRatio").as_deref(), Some("1.000"), "{name}: square samples");
    let rate: f64 = field(video, "FrameRate").expect("a frame rate").parse().unwrap();
    assert!((rate - FPS).abs() < 0.01, "{name}: frame rate {rate}");
    assert_eq!(field(video, "FrameCount").as_deref(), Some(packets.to_string().as_str()), "{name}: frames: {json}");
}

#[test]
fn mediainfo_reads_what_the_muxer_wrote() {
    let Some(bin) = mediainfo() else { return };
    for (codec, format, id) in [(VideoCodec::H264, "AVC", "avc1"), (VideoCodec::H265, "HEVC", "hvc1")] {
        let enc = select_encoder(config(codec), Some(EncoderBackend::H26x)).expect("the software encoder, by name");
        let (mp4, packets) = build_mp4(codec, enc);
        check(&bin, &format!("{codec:?}"), &mp4, packets, format, id);
    }
    if let Some(enc) = common::try_av1_encoder(config(VideoCodec::Av1)) {
        let (mp4, packets) = build_mp4(VideoCodec::Av1, enc);
        check(&bin, "av1", &mp4, packets, "AV1", "av01");
    }
}
