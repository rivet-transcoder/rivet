//! AMF decode pixel verification — runs the live AMF decoder on the dev
//! box's AMD GPU and compares every frame **byte for byte** against a
//! software decoder of the same stream: the in-tree `h26x` decoders for
//! H.264 / HEVC (bit-exact against the JVT / JCT-VC conformance suites) and
//! rivet's own AV1 decoder (bit-exact on the AOM and Argon vectors) for AV1. Conformant decoders of the same bitstream are bit-exact,
//! so anything but equality is a bug (a wrong plane pitch, a missed `>>6`, a
//! lost frame at the flush …).
//!
//! The clips are made here with this workspace's own encoders (`h26x`,
//! av1) and muxer; no external program is run.
//!
//! Needs: the `amd` feature and an AMD GPU the AMF runtime drives. Without
//! either the test prints `SKIPPED` and passes — a test that cannot run is
//! not evidence, and it says so.
#![cfg(feature = "amd")]

use bytes::Bytes;
use codec::decode::Decoder;
use codec::encode::tuning::EncodeOverrides;
use codec::encode::{EncoderBackend, EncoderConfig, select_encoder};
use codec::frame::{ColorSpace, PixelFormat, StreamInfo, VideoCodec, VideoFrame};

const W: u32 = 640;
const H: u32 = 360;
const FRAMES: u64 = 60;

fn amd_present() -> bool {
    codec::gpu::detect_gpus().iter().any(|g| g.vendor == codec::gpu::GpuVendor::Amd)
}

struct Clip {
    name: &'static str,
    /// The AMF/canonical codec label, as `probe_decode_caps` reports it and
    /// `AmfDecoder` dispatches on (`h264` / `hevc` / `av1`). The demuxer's own
    /// label differs for HEVC (it says `h265`), so [`demux_label`] maps it.
    codec: &'static str,
    ten_bit: bool,
    /// B pictures between anchors (H.264 / HEVC).
    bframes: u8,
}

/// The label `container::demux` reports for a clip whose AMF label is `codec`.
/// Only HEVC differs: the demuxer says `h265`, AMF says `hevc`.
fn demux_label(codec: &str) -> &str {
    match codec {
        "hevc" => "h265",
        other => other,
    }
}

const CLIPS: &[Clip] = &[
    // No B-frames: decode order is display order, no DPB reordering.
    Clip { name: "h264_8bit_nob", codec: "h264", ten_bit: false, bframes: 0 },
    // B-frames on, so display order != decode order and the decoder's
    // reordering is exercised.
    Clip { name: "h264_8bit", codec: "h264", ten_bit: false, bframes: 3 },
    Clip { name: "hevc_8bit", codec: "hevc", ten_bit: false, bframes: 3 },
    Clip { name: "hevc_main10", codec: "hevc", ten_bit: true, bframes: 3 },
    Clip { name: "av1_8bit", codec: "av1", ten_bit: false, bframes: 0 },
];

/// Frame `t` of a moving test pattern: a diagonal ramp scrolling one sample
/// per frame, a box crossing it, and chroma that varies across the picture.
fn pattern(t: u64, ten_bit: bool) -> VideoFrame {
    let (w, h) = (W as usize, H as usize);
    let mut planes: Vec<u16> = Vec::with_capacity(w * h * 3 / 2);
    for y in 0..h {
        for x in 0..w {
            let in_box = x.wrapping_sub(t as usize * 5) % w < 80 && (140..220).contains(&y);
            planes.push(if in_box { 235 } else { (16 + (x + 2 * y + t as usize) % 220) as u16 });
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

/// Encode the clip with this workspace's encoder and mux it; returns the MP4.
fn make_clip(clip: &Clip) -> Vec<u8> {
    let (codec, backend) = match clip.codec {
        "h264" => (VideoCodec::H264, EncoderBackend::H26x),
        "hevc" => (VideoCodec::H265, EncoderBackend::H26x),
        _ => (VideoCodec::Av1, EncoderBackend::Av1),
    };
    let cfg = EncoderConfig {
        width: W,
        height: H,
        frame_rate: 30.0,
        codec,
        keyframe_interval: 30,
        quality: 20,
        threads: 0,
        pixel_format: if clip.ten_bit { PixelFormat::Yuv420p10le } else { PixelFormat::Yuv420p },
        overrides: EncodeOverrides { bframes: Some(clip.bframes), ..Default::default() },
        ..Default::default()
    };
    let mut enc = select_encoder(cfg, Some(backend)).unwrap_or_else(|e| panic!("{}: the encoder: {e:#}", clip.name));
    let mut mux = container::mux::Av1Mp4Muxer::new_with_codec(W, H, 30.0, codec).expect("muxer");
    for t in 0..FRAMES {
        enc.send_frame(&pattern(t, clip.ten_bit)).expect("encode");
        while let Some(p) = enc.receive_packet().expect("packet") {
            mux.add_packet(p).expect("mux");
        }
    }
    enc.flush().expect("flush");
    while let Some(p) = enc.receive_packet().expect("packet") {
        mux.add_packet(p).expect("mux");
    }
    mux.finalize().expect("finalize").to_vec()
}

/// The software decoder's frames: `h26x` for H.264 / HEVC, rivet's own decoder for AV1.
fn reference_frames(clip: &Clip, info: &StreamInfo, samples: &[Vec<u8>]) -> Vec<VideoFrame> {
    let dec: Box<dyn Decoder> = if clip.codec == "av1" {
        Box::new(codec::decode::av1_sw::Av1Decoder::new(info.clone()).expect("av1"))
    } else {
        Box::new(codec::decode::h26x_sw::H26xDecoder::new(info.clone()).expect("h26x"))
    };
    run_decoder(dec, samples).unwrap_or_else(|e| panic!("{}: software decode: {e:#}", clip.name))
}

fn run_decoder(mut dec: Box<dyn Decoder>, samples: &[Vec<u8>]) -> anyhow::Result<Vec<VideoFrame>> {
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
fn amf_decode_is_bit_exact_against_the_software_decoders() {
    if !amd_present() {
        eprintln!("SKIPPED: no AMD GPU on this machine");
        return;
    }
    // One AMF-capable iGPU: serialise against the encoder's hardware tests.
    let _hw = codec::amf_hwtest::hw_lock();
    let caps = codec::decode::amf_dec::probe_decode_caps();
    eprintln!("AMF decode probe: {caps:?}");
    if caps.is_empty() {
        eprintln!("SKIPPED: the AMF runtime drives no decoder on this GPU");
        return;
    }

    // `RIVET_AMF_CLIPS=a,b`: only the clips whose name contains one of these
    // — for bringing the decoder up one clip at a time.
    let only: Option<Vec<String>> = std::env::var("RIVET_AMF_CLIPS")
        .ok()
        .filter(|v| !v.is_empty())
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect());
    let mut verified = Vec::new();
    for clip in CLIPS.iter().filter(|c| only.as_ref().is_none_or(|o| o.iter().any(|n| c.name.contains(n.as_str())))) {
        let data = make_clip(clip);
        let demuxed = container::demux::demux(&data).expect("demux");
        assert_eq!(demuxed.codec.to_ascii_lowercase(), demux_label(clip.codec), "{}: demuxed codec", clip.name);
        let info: StreamInfo = demuxed.info.clone();
        let ten_bit = clip.ten_bit;
        let (w, h) = (info.width as usize, info.height as usize);
        let frame_bytes = w * h * 3 / 2 * if ten_bit { 2 } else { 1 };
        eprintln!(
            "  {}: {} {}x{} {:?} {} samples",
            clip.name,
            demuxed.codec,
            w,
            h,
            info.pixel_format,
            demuxed.samples.len()
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
        let frames =
            run_decoder(Box::new(amf), &demuxed.samples).unwrap_or_else(|e| panic!("{}: AMF decode: {e:#}", clip.name));

        let reference = reference_frames(clip, &info, &demuxed.samples);
        assert_eq!(reference.len() as u64, FRAMES, "{}: the software decoder's frame count", clip.name);
        if frames.len() != reference.len() {
            // Which reference frames came back, in which order — tells a
            // dropped head from a lost tail from a reorder bug.
            let matches: Vec<String> = frames
                .iter()
                .map(|f| reference.iter().position(|r| r.data[..] == f.data[..]).map_or("?".into(), |i| i.to_string()))
                .collect();
            panic!(
                "{}: {} frames from AMF vs {} from the software decoder; AMF frames matched reference indices [{}]",
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
            if f.data[..] != r.data[..] {
                panic!(
                    "{}: frame {i} differs from the software decoder: {}",
                    clip.name,
                    describe_diff(&f.data, &r.data, w, h, ten_bit)
                );
            }
        }
        eprintln!("  {}: {} frames bit-exact vs the software decoder", clip.name, frames.len());
        verified.push(clip.name);
    }
    eprintln!("AMF decode verified bit-exact on this machine: {verified:?}");
    if only.is_none() {
        assert!(verified.contains(&"h264_8bit") && verified.contains(&"hevc_8bit"), "{verified:?}");
    }
}
