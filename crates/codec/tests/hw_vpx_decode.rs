//! VP9 (and VP8) hardware decode against rivet's own decoders, byte for byte.
//!
//! Every hardware VP9 decoder this build has — AMF (`amd`), QSV (`qsv`),
//! NVDEC (`nvidia`) — is run on the streams below, and every frame it returns
//! is compared with what rivet's own decoder (`crates/vp9`, bit-exact on the
//! WebM project's test vectors) makes of the same stream. VP9 fixes the
//! reconstruction exactly, so a conformant decoder is bit-exact and anything
//! else is a bug: a wrong plane pitch, a missed `>> 6` on P010, a frame lost
//! at the flush, a superframe's hidden frame shown.
//!
//! The streams:
//!
//! - clips made here by rivet's own VP9 encoder, profile 0 (8-bit) and
//!   profile 2 (10-bit), handed to the decoder packet by packet as the
//!   encoder emitted them;
//! - the WebM project's VP9 test vectors committed in `crates/vp9/tests/data`
//!   that are 4:2:0 at 8 or 10 bits (the only formats a hardware tier here is
//!   asked to decode), read from their WebM / IVF files;
//! - with `RIVET_VP9_VECTORS=<dir>`, every 4:2:0 profile 0 / 2 vector in that
//!   directory (`crates/vp9/tools/fetch-vectors.sh` downloads the full set).
//!
//! VP8 (NVDEC only — no other vendor's runtime here decodes it) is checked the
//! same way against `crates/vp8`, on a clip from rivet's own VP8 encoder.
//!
//! `RIVET_VPX_STREAMS=<substring>,…` limits a run to the named streams and
//! `RIVET_VPX_FRAMES` sets the length of rivet's own clips (40): bring a
//! decoder up one small stream at a time, single-threaded
//! (`--test-threads=1`), before running the set.
//!
//! A tier whose hardware is absent prints `SKIPPED` and passes, unless
//! `RIVET_REQUIRE_QSV=1` / `RIVET_REQUIRE_AMF=1` / `RIVET_REQUIRE_NVDEC=1`
//! says this machine must have it (the Intel CI runner sets the first).
//! No other implementation is run.
#![cfg(any(feature = "amd", feature = "qsv", feature = "nvidia"))]

use std::path::{Path, PathBuf};

use bytes::Bytes;
use codec::decode::Decoder;
use codec::encode::{EncoderBackend, EncoderConfig, select_encoder};
use codec::decode::vp9_hw_guard::{self, Vp9HwPolicy};
use codec::frame::{ColorSpace, PixelFormat, StreamInfo, VideoCodec, VideoFrame};

const W: u32 = 352;
const H: u32 = 288;

/// Frames in each of rivet's own clips: 40, or `RIVET_VPX_FRAMES`.
fn frames() -> u64 {
    std::env::var("RIVET_VPX_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(40)
}

/// `RIVET_VPX_STREAMS=a,b`: run only the streams whose name contains one of
/// these — for bringing a decoder up one stream at a time.
fn selected(name: &str) -> bool {
    match std::env::var("RIVET_VPX_STREAMS") {
        Ok(list) if !list.is_empty() => list.split(',').any(|s| name.contains(s.trim())),
        _ => true,
    }
}

/// One stream to decode: its codec label, its `StreamInfo` and its samples
/// (one container packet each — a frame or a superframe).
struct Stream {
    name: String,
    info: StreamInfo,
    samples: Vec<Vec<u8>>,
}

fn info(codec: &str, w: u32, h: u32, format: PixelFormat) -> StreamInfo {
    StreamInfo {
        codec: codec.to_string(),
        width: w,
        height: h,
        frame_rate: 30.0,
        duration: 0.0,
        pixel_format: format,
        color_space: ColorSpace::Bt709,
        total_frames: 0,
        bitrate: 0,
        color_metadata: Default::default(),
    }
}

/// Frame `t` of a moving pattern: a scrolling diagonal ramp, a box crossing
/// it, chroma that varies over the picture.
fn pattern(t: u64, ten_bit: bool) -> VideoFrame {
    let (w, h) = (W as usize, H as usize);
    let mut planes: Vec<u16> = Vec::with_capacity(w * h * 3 / 2);
    for y in 0..h {
        for x in 0..w {
            let in_box = x.wrapping_sub(t as usize * 5) % w < 64 && (100..180).contains(&y);
            planes.push(if in_box { 235 } else { (16 + (x + 2 * y + 3 * t as usize) % 220) as u16 });
        }
    }
    for c in 0..2 {
        for y in 0..h / 2 {
            for x in 0..w / 2 {
                planes.push((64 + (x * (c + 1) + y + t as usize) % 128) as u16);
            }
        }
    }
    let (data, format) = if ten_bit {
        (planes.iter().flat_map(|v| (v << 2).to_le_bytes()).collect::<Vec<u8>>(), PixelFormat::Yuv420p10le)
    } else {
        (planes.iter().map(|&v| v as u8).collect(), PixelFormat::Yuv420p)
    };
    VideoFrame::new(Bytes::from(data), W, H, format, ColorSpace::Bt709, t)
}

/// A clip from rivet's own encoder for `codec`, as packets.
fn own_clip(codec: VideoCodec, ten_bit: bool) -> Stream {
    let backend = if codec == VideoCodec::Vp8 { EncoderBackend::Vp8 } else { EncoderBackend::Vp9 };
    let format = if ten_bit { PixelFormat::Yuv420p10le } else { PixelFormat::Yuv420p };
    let cfg = EncoderConfig {
        width: W,
        height: H,
        frame_rate: 30.0,
        codec,
        keyframe_interval: 15,
        quality: 24,
        pixel_format: format,
        ..Default::default()
    };
    let mut enc = select_encoder(cfg, Some(backend)).expect("rivet's own encoder");
    let mut samples = Vec::new();
    for t in 0..frames() {
        enc.send_frame(&pattern(t, ten_bit)).expect("encode");
        while let Some(p) = enc.receive_packet().expect("packet") {
            samples.push(p.data.to_vec());
        }
    }
    enc.flush().expect("flush");
    while let Some(p) = enc.receive_packet().expect("packet") {
        samples.push(p.data.to_vec());
    }
    let label = if codec == VideoCodec::Vp8 { "vp8" } else { "vp9" };
    Stream {
        name: format!("rivet-{label}-{}", if ten_bit { "10bit" } else { "8bit" }),
        info: info(label, W, H, format),
        samples,
    }
}

/// Whether a vector's name says it is 4:2:0 at 8 or 10 bits: profile 0
/// (`vp90-`) or profile 2 (`vp92-`), and not one of the odd-chroma streams.
fn hardware_shaped(name: &str) -> bool {
    (name.starts_with("vp90-") || name.starts_with("vp92-")) && !name.contains("yuv44") && !name.contains("yuv422")
}

/// A test vector file as a [`Stream`]: WebM through rivet's demuxer, IVF
/// through the VP9 crate's reader.
fn vector(path: &Path) -> Option<Stream> {
    let name = path.file_name()?.to_str()?.to_string();
    let data = std::fs::read(path).ok()?;
    if name.ends_with(".ivf") {
        let reader = vp9::ivf::IvfReader::new(&data).ok()?;
        let (w, h) = (u32::from(reader.header().width), u32::from(reader.header().height));
        let format = if name.starts_with("vp92-") { PixelFormat::Yuv420p10le } else { PixelFormat::Yuv420p };
        let samples: Vec<Vec<u8>> = reader.map_while(|f| f.ok().map(|f| f.data.to_vec())).collect();
        return Some(Stream { name, info: info("vp9", w, h, format), samples });
    }
    let demuxed = container::demux::demux(&data).ok()?;
    if !demuxed.codec.eq_ignore_ascii_case("vp9") {
        return None;
    }
    let mut info = demuxed.info.clone();
    if name.starts_with("vp92-") {
        info.pixel_format = PixelFormat::Yuv420p10le;
    }
    Some(Stream { name, info, samples: demuxed.samples.iter().map(|s| s.to_vec()).collect() })
}

/// The committed vectors, then any in `RIVET_VP9_VECTORS`.
fn vectors() -> Vec<Stream> {
    let mut dirs = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../vp9/tests/data")];
    if let Ok(dir) = std::env::var("RIVET_VP9_VECTORS") {
        dirs.push(PathBuf::from(dir));
    }
    let mut out: Vec<Stream> = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            eprintln!("  (no vectors at {})", dir.display());
            continue;
        };
        let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        paths.sort();
        for p in paths {
            let Some(name) = p.file_name().and_then(|n| n.to_str()) else { continue };
            if !(name.ends_with(".webm") || name.ends_with(".ivf")) || !hardware_shaped(name) {
                continue;
            }
            if out.iter().any(|s| s.name == name) {
                continue;
            }
            match vector(&p) {
                Some(s) => out.push(s),
                None => eprintln!("  {name}: not readable as VP9; left out"),
            }
        }
    }
    out
}

fn run(mut dec: Box<dyn Decoder>, samples: &[Vec<u8>]) -> anyhow::Result<Vec<VideoFrame>> {
    let mut frames = Vec::new();
    for s in samples {
        dec.push_sample(s)?;
        while let Some(f) = dec.decode_next()? {
            frames.push(f);
        }
    }
    dec.finish()?;
    while let Some(f) = dec.decode_next()? {
        frames.push(f);
    }
    Ok(frames)
}

/// rivet's own decoder's frames for `stream`.
fn reference(stream: &Stream) -> Vec<VideoFrame> {
    let dec: Box<dyn Decoder> = if stream.info.codec == "vp8" {
        Box::new(codec::decode::vp8_sw::Vp8Decoder::new(stream.info.clone()).expect("vp8"))
    } else {
        Box::new(codec::decode::vp9_sw::Vp9Decoder::new(stream.info.clone()).expect("vp9"))
    };
    run(dec, &stream.samples).unwrap_or_else(|e| panic!("{}: rivet's decoder: {e:#}", stream.name))
}

/// The top-left `w` x `h` of a planar 4:2:0 frame.
fn crop(f: &VideoFrame, w: u32, h: u32) -> VideoFrame {
    let b = if f.format == PixelFormat::Yuv420p10le { 2 } else { 1 };
    let (sw, sh, dw, dh) = (f.width as usize, f.height as usize, w as usize, h as usize);
    let (scw, sch, dcw, dch) = (sw.div_ceil(2), sh.div_ceil(2), dw.div_ceil(2), dh.div_ceil(2));
    let mut out = Vec::new();
    for row in 0..dh {
        out.extend_from_slice(&f.data[row * sw * b..(row * sw + dw) * b]);
    }
    for plane in 0..2 {
        let base = (sw * sh + plane * scw * sch) * b;
        for row in 0..dch {
            out.extend_from_slice(&f.data[base + row * scw * b..base + (row * scw + dcw) * b]);
        }
    }
    VideoFrame::new(out.into(), w, h, f.format, f.color_space, f.pts)
}

fn luma_diff(a: &VideoFrame, b: &VideoFrame) -> String {
    let ten = a.format == PixelFormat::Yuv420p10le;
    let n = (a.width * a.height) as usize;
    let (mut max, mut se) = (0i64, 0f64);
    for i in 0..n.min(a.data.len()).min(b.data.len()) {
        let (x, y) = if ten {
            if 2 * i + 1 >= a.data.len().min(b.data.len()) {
                break;
            }
            (
                i64::from(u16::from_le_bytes([a.data[2 * i], a.data[2 * i + 1]])),
                i64::from(u16::from_le_bytes([b.data[2 * i], b.data[2 * i + 1]])),
            )
        } else {
            (i64::from(a.data[i]), i64::from(b.data[i]))
        };
        max = max.max((x - y).abs());
        se += ((x - y) * (x - y)) as f64;
    }
    let peak = if ten { 1023.0 } else { 255.0 };
    format!("luma max|diff|={max} psnr={:.2} dB", 10.0 * (peak * peak / (se / n as f64).max(1e-9)).log10())
}

/// `None` when every frame matched; otherwise why not.
fn compare(stream: &Stream, got: &[VideoFrame], want: &[VideoFrame]) -> Option<String> {
    if got.len() != want.len() {
        let matched: Vec<String> = got
            .iter()
            .map(|f| want.iter().position(|r| r.data[..] == f.data[..]).map_or("?".into(), |i| i.to_string()))
            .collect();
        return Some(format!(
            "{} frames, rivet's decoder {}; hardware frames matched reference indices [{}]",
            got.len(),
            want.len(),
            matched.join(" ")
        ));
    }
    for (i, (g, r)) in got.iter().zip(want).enumerate() {
        // A surface rounds an odd size up to even (AMF returns 352x288 for a
        // 351x287 stream); the picture is its top-left, which is what the
        // guard crops to.
        let cropped;
        let g = if g.format == r.format
            && (g.width, g.height) != (r.width, r.height)
            && (g.width == r.width || g.width == r.width + 1)
            && (g.height == r.height || g.height == r.height + 1)
        {
            cropped = crop(g, r.width, r.height);
            &cropped
        } else {
            g
        };
        if (g.width, g.height, g.format) != (r.width, r.height, r.format) {
            return Some(format!(
                "frame {i}: {}x{} {:?}, rivet's decoder {}x{} {:?}",
                g.width, g.height, g.format, r.width, r.height, r.format
            ));
        }
        if g.data[..] != r.data[..] {
            return Some(format!("frame {i} differs: {}", luma_diff(g, r)));
        }
    }
    let _ = stream;
    None
}

fn required(var: &str) -> bool {
    std::env::var(var).is_ok_and(|v| v == "1")
}

/// The VP9 features a hardware decoder is trusted with only by policy
/// (`codec::decode::vp9_hw_guard`) that `stream` uses.
fn guarded_features(stream: &Stream) -> Vec<&'static str> {
    use codec::vp9_header::{FrameHeader, RefSizes, peek};
    let mut out = Vec::new();
    if stream.info.codec != "vp9" {
        return out;
    }
    let (mut refs, mut first) = (RefSizes::default(), None);
    for packet in &stream.samples {
        for frame in vp9::superframe::split(packet) {
            let Some(h) = peek(frame) else { continue };
            if matches!(h, FrameHeader::ShowExisting { .. }) && !out.contains(&"show_existing_frame") {
                out.push("show_existing_frame");
            }
            if let FrameHeader::Coded(c) = &h {
                for (on, name) in [
                    (c.error_resilient, "error_resilient_mode"),
                    (c.segmentation, "segmentation"),
                    (c.intra_only, "intra_only"),
                ] {
                    if on && !out.contains(&name) {
                        out.push(name);
                    }
                }
            }
            let size = refs.apply(&h);
            if first.is_none() {
                first = size;
            }
            if size.is_some() && size != first && !out.contains(&"size change") {
                out.push("size change");
            }
        }
    }
    out
}

type Make = fn(&StreamInfo) -> anyhow::Result<Box<dyn Decoder>>;

/// Runs the tier on every stream of `codec_label` and compares with rivet's
/// own decoder, two ways:
///
/// - **raw**, the hardware decoder alone: must be bit-exact on every stream
///   that uses neither feature the guard watches for; on a stream that does,
///   the result is reported (that is the evidence a policy is relaxed on),
///   and a mismatch fails only if the tier's policy claims the feature;
/// - **guarded** (VP9), the decoder behind `Vp9HardwareGuard` with the
///   tier's policy, as `create_decoder` builds it: bit-exact on everything.
fn check_tier(tier: &'static str, codec_label: &str, streams: &[Stream], make: Make, policy: Vp9HwPolicy) {
    let mut failures = Vec::new();
    let mut exact = Vec::new();
    for stream in streams.iter().filter(|s| s.info.codec == codec_label && selected(&s.name)) {
        let want = reference(stream);
        let features = guarded_features(stream);
        // The bare hardware sees only what the guard would hand it, unless
        // RIVET_VPX_RAW_FEATURES=1 asks for the evidence a policy is relaxed
        // on: feeding a decoder streams outside what it was set up for is
        // how this test once hung an AMD iGPU's video engine (TDR).
        if codec_label == "vp9" && !guard_passes(stream, policy) && !required("RIVET_VPX_RAW_FEATURES") {
            eprintln!("  {tier} {}: raw: not run (the guard keeps it from the hardware: {features:?})", stream.name);
        } else {
            raw_pass(tier, stream, make, policy, &features, &want, &mut failures, &mut exact, codec_label);
        }
        if codec_label == "vp9" {
            guarded_pass(tier, stream, make, policy, &want, &mut failures, &mut exact);
        }
    }
    eprintln!("{tier} {codec_label}: {} stream(s) bit-exact as dispatched: {exact:?}", exact.len());
    assert!(failures.is_empty(), "{tier} {codec_label} decode differs from rivet's decoder:
  {}", failures.join("
  "));
}

/// Whether the guard, with `policy`, would keep the whole stream on the
/// hardware — asked of a guard over rivet's own decoder, no GPU involved.
fn guard_passes(stream: &Stream, policy: Vp9HwPolicy) -> bool {
    let stand_in = Box::new(codec::decode::vp9_sw::Vp9Decoder::new(stream.info.clone()).expect("vp9"));
    let mut g = codec::decode::vp9_hw_guard::Vp9HardwareGuard::new("dry run", stand_in, stream.info.clone(), policy)
        .with_rebuild(Box::new(|i| Ok(Box::new(codec::decode::vp9_sw::Vp9Decoder::new(i.clone())?) as Box<dyn Decoder>)));
    for s in &stream.samples {
        if g.push_sample(s).is_err() || g.switched() {
            return false;
        }
        while let Ok(Some(_)) = g.decode_next() {}
    }
    !g.switched()
}

/// The bare hardware decoder on `stream`: see [`check_tier`].
#[allow(clippy::too_many_arguments)]
fn raw_pass(
    tier: &str,
    stream: &Stream,
    make: Make,
    policy: Vp9HwPolicy,
    features: &[&'static str],
    want: &[VideoFrame],
    failures: &mut Vec<String>,
    exact: &mut Vec<String>,
    codec_label: &str,
) {
    let raw = make(&stream.info).and_then(|d| run(d, &stream.samples));
    let raw_verdict = match &raw {
        Ok(frames) => compare(stream, frames, want),
        Err(e) => Some(format!("declined or failed: {e:#}")),
    };
    let claimed = features.iter().all(|f| match *f {
        "show_existing_frame" => policy.show_existing,
        "error_resilient_mode" => policy.error_resilient,
        "segmentation" => policy.segmentation,
        "intra_only" => policy.intra_only,
        _ => policy.size_changes,
    });
    match (&raw_verdict, features.is_empty()) {
        (None, _) => {
            eprintln!("  {tier} {}: raw: {} frames bit-exact (uses {features:?})", stream.name, want.len());
            if !features.is_empty() && !claimed {
                eprintln!("  {tier} {}: raw hardware handled {features:?} bit-exact: the policy could trust it", stream.name);
            }
        }
        (Some(why), true) => {
            eprintln!("  {tier} {}: raw: MISMATCH: {why}", stream.name);
            failures.push(format!("{} (raw): {why}", stream.name));
        }
        (Some(why), false) => {
            eprintln!("  {tier} {}: raw: differs on a stream using {features:?}: {why}", stream.name);
            if claimed {
                failures.push(format!("{} (raw, policy trusts {features:?}): {why}", stream.name));
            }
        }
    }
    if codec_label != "vp9" && raw_verdict.is_none() {
        exact.push(stream.name.clone());
    }
}

/// The hardware decoder behind the VP9 guard, as `create_decoder` builds it:
/// see [`check_tier`].
fn guarded_pass(
    tier: &'static str,
    stream: &Stream,
    make: Make,
    policy: Vp9HwPolicy,
    want: &[VideoFrame],
    failures: &mut Vec<String>,
    exact: &mut Vec<String>,
) {
    let guarded = make(&stream.info).and_then(|d| {
        let g = codec::decode::vp9_hw_guard::Vp9HardwareGuard::new(tier, d, stream.info.clone(), policy)
            .with_rebuild(Box::new(move |i| make(i)));
        run(Box::new(g), &stream.samples)
    });
    match guarded.map_err(|e| format!("{e:#}")).map(|f| compare(stream, &f, want)) {
        Ok(None) => {
            eprintln!("  {tier} {}: guarded: {} frames bit-exact", stream.name, want.len());
            exact.push(stream.name.clone());
        }
        Ok(Some(why)) | Err(why) => {
            eprintln!("  {tier} {}: guarded: MISMATCH: {why}", stream.name);
            failures.push(format!("{} (guarded): {why}", stream.name));
        }
    }
}

fn vp9_streams() -> Vec<Stream> {
    let mut s = vec![own_clip(VideoCodec::Vp9, false), own_clip(VideoCodec::Vp9, true)];
    s.extend(vectors());
    s
}

#[cfg(feature = "amd")]
#[test]
fn amf_vp9_decode_is_bit_exact_against_rivets_decoder() {
    let present = codec::gpu::detect_gpus().iter().any(|g| g.vendor == codec::gpu::GpuVendor::Amd);
    let caps = if present { codec::decode::amf_dec::probe_decode_caps() } else { &[] };
    if !caps.contains(&"vp9") {
        assert!(!required("RIVET_REQUIRE_AMF"), "RIVET_REQUIRE_AMF=1 and AMF decodes no VP9 here (caps {caps:?})");
        eprintln!("SKIPPED: no AMF VP9 decoder on this machine (caps {caps:?})");
        return;
    }
    let _hw = codec::amf_hwtest::hw_lock();
    check_tier(
        "AMF",
        "vp9",
        &vp9_streams(),
        |info| Ok(Box::new(codec::decode::amf_dec::AmfDecoder::new(info.clone(), 0)?) as Box<dyn Decoder>),
        vp9_hw_guard::AMF_POLICY,
    );
}

#[cfg(feature = "qsv")]
#[test]
fn qsv_vp9_decode_is_bit_exact_against_rivets_decoder() {
    let present = codec::gpu::detect_gpus().iter().any(|g| g.vendor == codec::gpu::GpuVendor::Intel);
    let caps = if present { codec::decode::qsv_dec::probe_decode_caps() } else { &[] };
    if !caps.contains(&"vp9") {
        assert!(!required("RIVET_REQUIRE_QSV"), "RIVET_REQUIRE_QSV=1 and QSV decodes no VP9 here (caps {caps:?})");
        eprintln!("SKIPPED: no QSV VP9 decoder on this machine (caps {caps:?})");
        return;
    }
    check_tier(
        "QSV",
        "vp9",
        &vp9_streams(),
        |info| Ok(Box::new(codec::decode::qsv_dec::QsvDecoder::new(info.clone(), 0)?) as Box<dyn Decoder>),
        vp9_hw_guard::QSV_POLICY,
    );
}

#[cfg(feature = "nvidia")]
fn nvidia_present() -> bool {
    let present = codec::gpu::detect_gpus().iter().any(|g| g.vendor == codec::gpu::GpuVendor::Nvidia);
    if !present {
        assert!(!required("RIVET_REQUIRE_NVDEC"), "RIVET_REQUIRE_NVDEC=1 and there is no NVIDIA GPU here");
        eprintln!("SKIPPED: no NVIDIA GPU on this machine");
    }
    present
}

#[cfg(feature = "nvidia")]
#[test]
fn nvdec_vp9_decode_is_bit_exact_against_rivets_decoder() {
    if !nvidia_present() {
        return;
    }
    check_tier(
        "NVDEC",
        "vp9",
        &vp9_streams(),
        |info| Ok(codec::decode::nvdec::NvdecDecoder::new(info.clone(), 0)),
        vp9_hw_guard::NVDEC_POLICY,
    );
}

#[cfg(feature = "nvidia")]
#[test]
fn nvdec_vp8_decode_is_bit_exact_against_rivets_decoder() {
    if !nvidia_present() {
        return;
    }
    let streams = vec![own_clip(VideoCodec::Vp8, false)];
    check_tier(
        "NVDEC",
        "vp8",
        &streams,
        |info| Ok(codec::decode::nvdec::NvdecDecoder::new(info.clone(), 0)),
        Vp9HwPolicy::BASELINE,
    );
}
