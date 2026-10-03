//! VP9, VP8, MPEG-2, MPEG-4 Part 2 and ProRes output, end to end through the
//! job engine: a source is transcoded by `run_job_blocking` to each codec in
//! each file it goes in, and the output is read back with rivet's own
//! demuxers and decoded with rivet's own decoders. No other implementation
//! is run.
//!
//! What is checked, per output: the file's kind (`ftyp` brand, WebM
//! DocType), the codec the demuxer reports, the size, one frame per source
//! frame, presentation timestamps one frame apart, the luma PSNR of every
//! frame against the source as rivet decodes it, and the audio track the
//! file carries (Opus in WebM, AAC copied into an MP4 / QuickTime movie).
//! VP9 is also packaged as HLS: its master playlist names a `vp09` codecs
//! string, and the init segment and media segments, joined, decode to every
//! frame.
//!
//! Sources: a synthetic H.264 clip made here with rivet's own H.264 encoder
//! (asked for by name, so this needs no feature), the committed container
//! fixtures (`video_first.ts`: H.264 with AAC; `mpeg2_601.ts`;
//! `vp9_601.webm`), and `test_media/bbb_h264_360p_short.mp4` when present.

mod common;

use std::sync::Arc;

use codec::encode::{EncoderBackend, EncoderConfig, select_encoder};
use codec::frame::{ColorSpace, PixelFormat, VideoCodec, VideoFrame};
use container::mux::Av1Mp4Muxer;
use container::streaming::demux_streaming;
use rivet::{Container, OutputSpec, Rung, RungArtifact, VideoCodecPolicy};
use rivet::spec::ProresProfile;

/// A decoded clip: its frames' luma planes, its size and frame rate.
struct Decoded {
    codec: String,
    dims: (u32, u32),
    frame_rate: f64,
    /// Presentation times of the samples, in seconds, in decode order.
    pts: Vec<f64>,
    luma: Vec<Vec<f64>>,
    audio_codec: Option<String>,
}

/// The luma plane of `f` as 8-bit-scale samples.
fn luma(f: &VideoFrame) -> Vec<f64> {
    let n = (f.width * f.height) as usize;
    let bits = codec::colorspace::planar_bit_depth(f.format).unwrap_or(8);
    if bits == 8 {
        f.data[..n].iter().map(|&v| f64::from(v)).collect()
    } else {
        let scale = f64::from(1u32 << (bits - 8));
        (0..n).map(|i| f64::from(u16::from_le_bytes([f.data[2 * i], f.data[2 * i + 1]])) / scale).collect()
    }
}

/// Demux `file` with rivet's streaming demuxer and decode it with the decoder
/// rivet picks.
fn decode(file: &[u8]) -> Decoded {
    let mut demux = demux_streaming(file).unwrap_or_else(|e| panic!("rivet demuxes the file: {e:#}"));
    let header = demux.header().clone();
    let audio_codec = demux.audio().map(|a| a.codec.clone());
    let mut dec = codec::decode::create_decoder(&header.codec, header.info.clone()).expect("a decoder");
    let mut pts = Vec::new();
    let mut luma_planes = Vec::new();
    while let Some(s) = demux.next_video_sample().unwrap() {
        pts.push(header.pts_seconds(s.pts_ticks));
        dec.push_sample(&s.data).expect("decode");
        while let Some(f) = dec.decode_next().unwrap() {
            luma_planes.push(luma(&f));
        }
    }
    dec.finish().unwrap();
    while let Some(f) = dec.decode_next().unwrap() {
        luma_planes.push(luma(&f));
    }
    Decoded {
        codec: header.codec.clone(),
        dims: (header.info.width, header.info.height),
        frame_rate: header.info.frame_rate,
        pts,
        luma: luma_planes,
        audio_codec,
    }
}

fn psnr(a: &[f64], b: &[f64]) -> f64 {
    let mse = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
}

/// The one file a single-rung job made.
fn run(source: &[u8], spec: &OutputSpec) -> Vec<u8> {
    spec.validate().expect("a valid spec");
    let out = rivet::run_job_blocking(source, spec, None, Arc::new(rivet::fn_sink(|_| {}))).expect("the job");
    assert_eq!(out.rungs.len(), 1);
    match &out.rungs[0].artifact {
        RungArtifact::File(b) => b.clone(),
        other => panic!("a single file, not {other:?}"),
    }
}

/// Transcode `source` to `codec` in `container` at the source's size, and
/// check the output against the source. Returns the worst frame's PSNR.
fn transcode_and_check(name: &str, source: &[u8], policy: VideoCodecPolicy, container: Container, floor: f64) -> f64 {
    transcode_and_check_with(name, source, policy, container, floor, |s| s)
}

/// [`transcode_and_check`] with the spec adjusted by `shape`. A trim
/// (`with_trim(None, Some(end))`) is checked against the source's first
/// frames.
fn transcode_and_check_with(
    name: &str,
    source: &[u8],
    policy: VideoCodecPolicy,
    container: Container,
    floor: f64,
    shape: impl FnOnce(OutputSpec) -> OutputSpec,
) -> f64 {
    let mut src = decode(source);
    let spec = shape(
        OutputSpec::single_file(vec![Rung::new(src.dims.0, src.dims.1)])
            .with_video_codec(policy)
            .with_container(container),
    );
    if let Some(end) = spec.trim_end {
        src.luma.truncate((end * src.frame_rate).ceil() as usize);
    }
    let file = run(source, &spec);
    match container {
        Container::WebM => assert_eq!(&file[..4], &[0x1A, 0x45, 0xDF, 0xA3], "{name}: an EBML file"),
        Container::Mov => assert_eq!(&file[4..12], b"ftypqt  ", "{name}: a QuickTime movie"),
        _ => assert_eq!(&file[4..12], b"ftypiso6", "{name}: an ISO MP4"),
    }
    let out = decode(&file);
    let label = match policy {
        VideoCodecPolicy::ProRes(_) => "prores".to_string(),
        p => p.codec().label().to_string(),
    };
    assert_eq!(out.codec, label, "{name}: codec");
    assert_eq!(out.dims, src.dims, "{name}: size");
    assert_eq!(out.luma.len(), src.luma.len(), "{name}: one frame per source frame");
    assert_eq!(out.pts.len(), src.luma.len(), "{name}: one sample per frame");
    let mut presented = out.pts.clone();
    presented.sort_by(f64::total_cmp);
    let step = 1.0 / src.frame_rate;
    for (i, t) in presented.iter().enumerate() {
        let want = presented[0] + i as f64 * step;
        assert!((t - want).abs() < 0.002, "{name}: frame {i} presented at {t:.4}s, want {want:.4}s");
    }
    let mut worst = f64::INFINITY;
    for (i, (a, b)) in out.luma.iter().zip(&src.luma).enumerate() {
        let q = psnr(a, b);
        assert!(q > floor, "{name}: frame {i} at {q:.2} dB (floor {floor})");
        worst = worst.min(q);
    }
    eprintln!(
        "{name}: {} frames {}x{}, worst luma PSNR {worst:.2} dB, {} bytes, audio {:?}",
        out.luma.len(),
        out.dims.0,
        out.dims.1,
        file.len(),
        out.audio_codec
    );
    worst
}

/// A synthetic H.264 MP4: 128x96, 24 frames at 24 fps, made by rivet's own
/// H.264 encoder (by name), with a ramp and a moving square.
fn synthetic_h264() -> Vec<u8> {
    let (w, h, n) = (128u32, 96u32, 24u64);
    let cfg = EncoderConfig {
        width: w,
        height: h,
        frame_rate: 24.0,
        codec: VideoCodec::H264,
        keyframe_interval: 24,
        quality: 18,
        threads: 1,
        ..Default::default()
    };
    let mut enc = select_encoder(cfg, Some(EncoderBackend::H26x)).expect("rivet's H.264 encoder, by name");
    let mut mux = Av1Mp4Muxer::new_with_codec(w, h, 24.0, VideoCodec::H264).unwrap();
    for t in 0..n {
        let (wu, hu) = (w as usize, h as usize);
        let mut data = vec![128u8; wu * hu * 3 / 2];
        for y in 0..hu {
            for x in 0..wu {
                let ramp = ((x + 2 * y + 3 * t as usize) % 200) as u8 + 25;
                let sq = (x as i64 - (10 + 4 * t as i64)).unsigned_abs() < 12 && (30..60).contains(&y);
                data[y * wu + x] = if sq { 230 } else { ramp };
            }
        }
        let f = VideoFrame::new(data.into(), w, h, PixelFormat::Yuv420p, ColorSpace::Bt709, t);
        enc.send_frame(&f).unwrap();
        while let Some(p) = enc.receive_packet().unwrap() {
            mux.add_packet(p).unwrap();
        }
    }
    enc.flush().unwrap();
    while let Some(p) = enc.receive_packet().unwrap() {
        mux.add_packet(p).unwrap();
    }
    mux.finalize().unwrap().to_vec()
}

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/../container/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))).expect("a committed fixture")
}

#[test]
fn vp9_to_webm_and_mp4() {
    let src = synthetic_h264();
    transcode_and_check("vp9 webm", &src, VideoCodecPolicy::Vp9, Container::WebM, 30.0);
    transcode_and_check("vp9 mp4", &src, VideoCodecPolicy::Vp9, Container::Mp4, 30.0);
}

#[test]
fn vp8_to_webm_and_mp4() {
    let src = synthetic_h264();
    transcode_and_check("vp8 webm", &src, VideoCodecPolicy::Vp8, Container::WebM, 30.0);
    transcode_and_check("vp8 mp4", &src, VideoCodecPolicy::Vp8, Container::Mp4, 30.0);
}

#[test]
fn mpeg2_to_mp4_and_mov() {
    let src = synthetic_h264();
    transcode_and_check("mpeg2 mp4", &src, VideoCodecPolicy::Mpeg2, Container::Mp4, 30.0);
    transcode_and_check("mpeg2 mov", &src, VideoCodecPolicy::Mpeg2, Container::Mov, 30.0);
}

#[test]
fn mpeg4_to_mp4_and_mov() {
    let src = synthetic_h264();
    transcode_and_check("mpeg4 mp4", &src, VideoCodecPolicy::Mpeg4, Container::Mp4, 30.0);
    transcode_and_check("mpeg4 mov", &src, VideoCodecPolicy::Mpeg4, Container::Mov, 30.0);
}

#[test]
fn prores_to_mov() {
    let src = synthetic_h264();
    for (p, floor) in [(ProresProfile::Standard, 35.0), (ProresProfile::Hq, 35.0), (ProresProfile::P4444, 35.0)] {
        transcode_and_check(&format!("prores {} mov", p.name()), &src, VideoCodecPolicy::ProRes(p), Container::Mov, floor);
    }
}

/// The codec alone picks the file: a `.mov` for ProRes, a `.webm` for VP9.
#[test]
fn the_codec_picks_its_file() {
    let src = synthetic_h264();
    let prores = OutputSpec::single_file(vec![Rung::new(128, 96)]).with_video_codec(VideoCodecPolicy::ProRes(ProresProfile::Lt));
    assert_eq!((prores.container, prores.file_extension()), (Container::Mov, "mov"));
    assert_eq!(&run(&src, &prores)[4..12], b"ftypqt  ");
    let vp9 = OutputSpec::single_file(vec![Rung::new(128, 96)]).with_video_codec(VideoCodecPolicy::Vp9);
    assert_eq!((vp9.container, vp9.file_extension()), (Container::WebM, "webm"));
}

/// Real H.264 + AAC (a committed transport stream): VP9 in WebM carries the
/// audio as Opus; MPEG-4 in a QuickTime movie copies the AAC.
#[test]
fn audio_goes_where_the_file_takes_it() {
    let src = fixture("timing/video_first.ts");
    // Colour kept as it is (`passthrough`): the default SDR path re-derives a
    // BT.601 matrix to BT.709, which changes the luma this compares.
    transcode_and_check_with("ts -> vp9 webm", &src, VideoCodecPolicy::Vp9, Container::WebM, 28.0, |s| s.passthrough());
    let dims = decode(&src).dims;
    let rung = || vec![Rung::new(dims.0, dims.1)];
    let webm = run(&src, &OutputSpec::single_file(rung()).with_video_codec(VideoCodecPolicy::Vp9));
    assert_eq!(decode(&webm).audio_codec.as_deref(), Some("opus"));
    let mov = run(
        &src,
        &OutputSpec::single_file(rung()).with_video_codec(VideoCodecPolicy::Mpeg4).with_container(Container::Mov),
    );
    assert_eq!(decode(&mov).audio_codec.as_deref(), Some("aac"));
}

/// Other codecs in: an MPEG-2 transport stream to VP9, a VP9 WebM to MPEG-4
/// Part 2 and to ProRes.
#[test]
fn mpeg2_and_vp9_sources() {
    // Colour kept (`passthrough`): these are BT.601 sources, which the
    // default SDR path re-derives to BT.709.
    let keep = |s: OutputSpec| s.passthrough();
    transcode_and_check_with("mpeg2 ts -> vp9 webm", &fixture("colour/mpeg2_601.ts"), VideoCodecPolicy::Vp9, Container::WebM, 28.0, keep);
    let vp9 = fixture("colour/vp9_601.webm");
    transcode_and_check_with("vp9 webm -> mpeg4 mp4", &vp9, VideoCodecPolicy::Mpeg4, Container::Mp4, 28.0, keep);
    transcode_and_check_with("vp9 webm -> prores mov", &vp9, VideoCodecPolicy::ProRes(ProresProfile::Hq), Container::Mov, 35.0, keep);
}

/// VP9 as HLS: the master playlist's `CODECS` names `vp09`, and the init
/// segment with the media segments, joined, demux and decode to every frame.
#[test]
fn vp9_hls() {
    let src = synthetic_h264();
    let dir = tempfile::tempdir().unwrap();
    let spec = OutputSpec::hls(vec![Rung::new(128, 96)], 0.5).with_video_codec(VideoCodecPolicy::Vp9);
    spec.validate().unwrap();
    let out = rivet::run_job_blocking(&src, &spec, Some(dir.path()), Arc::new(rivet::fn_sink(|_| {}))).expect("the HLS job");
    let master = std::fs::read_to_string(out.master_playlist.expect("a master playlist")).unwrap();
    assert!(master.contains("CODECS=\"vp09.00."), "{master}");
    let RungArtifact::HlsRendition { dir: rung_dir, .. } = &out.rungs[0].artifact else { panic!("an HLS rendition") };
    let mut joined = std::fs::read(rung_dir.join("init.mp4")).unwrap();
    let mut segs: Vec<_> = std::fs::read_dir(rung_dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "m4s"))
        .collect();
    segs.sort();
    assert!(segs.len() >= 2, "segments: {segs:?}");
    for s in &segs {
        joined.extend(std::fs::read(s).unwrap());
    }
    let got = decode(&joined);
    let want = decode(&src);
    assert_eq!(got.codec, "vp9");
    assert_eq!(got.luma.len(), want.luma.len());
    for (i, (a, b)) in got.luma.iter().zip(&want.luma).enumerate() {
        assert!(psnr(a, b) > 30.0, "frame {i}");
    }
}

/// The real clip, when it is there: VP9 in WebM with its audio as Opus.
#[test]
fn bbb_to_vp9_webm() {
    let Some(src) = common::read_test_media("bbb_h264_360p_short.mp4") else {
        eprintln!("SKIP: test_media/bbb_h264_360p_short.mp4 is absent");
        return;
    };
    // Its first half second, colour kept: VP9 is encoded in software here,
    // and a debug build is slow.
    transcode_and_check_with("bbb -> vp9 webm", &src, VideoCodecPolicy::Vp9, Container::WebM, 28.0, |s| {
        s.passthrough().with_trim(None, Some(0.5))
    });
}

/// An MPEG program stream (`.mpg`) in: the synthetic clip as MPEG-2 video
/// (rivet's own encoder) packed into packs and PES packets, transcoded to
/// MPEG-4 Part 2 through the program-stream demuxer.
#[test]
fn a_program_stream_source() {
    let src = synthetic_h264();
    let dims = decode(&src).dims;
    let mpeg2 = run(&src, &OutputSpec::single_file(vec![Rung::new(dims.0, dims.1)]).with_video_codec(VideoCodecPolicy::Mpeg2));
    let mut demux = demux_streaming(&mpeg2).unwrap();
    let pack = [0, 0, 1, 0xBA, 0x44, 0, 4, 0, 4, 1, 0x01, 0x89, 0xC3, 0xF8];
    let mut ps = Vec::new();
    while let Some(s) = demux.next_video_sample().unwrap() {
        ps.extend(pack);
        let len = 3 + s.data.len();
        ps.extend([0, 0, 1, 0xE0, (len >> 8) as u8, len as u8, 0x80, 0x00, 0x00]);
        ps.extend(&s.data);
    }
    ps.extend([0, 0, 1, 0xB9]);
    assert_eq!(container::sniff_container(&ps), container::ContainerKind::MpegPs);
    transcode_and_check("mpg -> mpeg4 mp4", &ps, VideoCodecPolicy::Mpeg4, Container::Mp4, 30.0);
}

/// The bit depth rivet's decoder gives for `file`'s video.
fn decoded_depth(file: &[u8]) -> u8 {
    let mut demux = demux_streaming(file).expect("rivet demuxes the file");
    let header = demux.header().clone();
    let mut dec = codec::decode::create_decoder(&header.codec, header.info.clone()).expect("a decoder");
    while let Some(s) = demux.next_video_sample().unwrap() {
        dec.push_sample(&s.data).unwrap();
        if let Some(f) = dec.decode_next().unwrap() {
            return codec::colorspace::planar_bit_depth(f.format).unwrap_or(8);
        }
    }
    dec.finish().unwrap();
    let f = dec.decode_next().unwrap().expect("a frame");
    codec::colorspace::planar_bit_depth(f.format).unwrap_or(8)
}

/// VP9 at 10 bits: profile 2, in WebM and in MP4, decoded back at 10 bits
/// by rivet's own decoder, near the source.
#[test]
fn vp9_ten_bit_profile_2() {
    let src = synthetic_h264();
    for container in [Container::WebM, Container::Mp4] {
        let name = format!("vp9 10-bit {container:?}");
        transcode_and_check_with(&name, &src, VideoCodecPolicy::Vp9, container, 30.0, |s| {
            s.with_bit_depth(rivet::BitDepth::TenBit)
        });
        let spec = OutputSpec::single_file(vec![Rung::new(128, 96)])
            .with_video_codec(VideoCodecPolicy::Vp9)
            .with_container(container)
            .with_bit_depth(rivet::BitDepth::TenBit);
        let file = run(&src, &spec);
        assert_eq!(decoded_depth(&file), 10, "{name}");
        // The first frame's header says profile 2.
        let mut demux = demux_streaming(&file).unwrap();
        let first = demux.next_video_sample().unwrap().unwrap();
        let info = container::vpx::vp9_frame_info(&first.data).expect("a VP9 frame");
        assert_eq!((info.profile, info.bit_depth), (2, 10), "{name}");
    }
}

/// A VP9 bitrate rung: coded by the encoder's rate controller to the rate.
#[test]
fn vp9_bitrate_rung() {
    let src = synthetic_h264();
    let bitrate = 400_000u32;
    let overrides = codec::encode::tuning::EncodeOverrides { bitrate: Some(bitrate), ..Default::default() };
    let rung = Rung::new(128, 96).with_quality(rivet::Quality::default().with_overrides(overrides));
    let spec = OutputSpec::single_file(vec![rung]).with_video_codec(VideoCodecPolicy::Vp9).with_container(Container::WebM);
    let file = run(&src, &spec);
    let out = decode(&file);
    assert_eq!(out.luma.len(), 24);
    // One second of video at 400 kb/s is about 50 KB; the container and the
    // first key frame are on top. Well inside a factor of two either way.
    let seconds = 24.0 / 24.0;
    let achieved = file.len() as f64 * 8.0 / seconds;
    assert!((0.5..2.0).contains(&(achieved / f64::from(bitrate))), "{achieved:.0} b/s for {bitrate}");
    eprintln!("vp9 bitrate rung: asked {bitrate} b/s, file {achieved:.0} b/s");
}

/// AV1 in software — rivet's own encoder, then rivet's own decoder — when
/// the build falls back to it (`av1-sw-fallback`, or its old name
/// `rav1e-fallback`): 8- and 10-bit, through the job engine, MP4 out,
/// demuxed and decoded by rivet, PSNR against the source.
#[test]
fn av1_in_software_8_and_10_bit() {
    if !cfg!(feature = "av1-sw-fallback") {
        eprintln!("SKIP: build without `av1-sw-fallback` (no software AV1 encode tier)");
        return;
    }
    if codec::gpu::detect_gpus().iter().any(|g| codec::encode::encode_capable_at(g, VideoCodec::Av1, false)) {
        eprintln!("SKIP: this host encodes AV1 on a GPU; the software tier is not what would run");
        return;
    }
    let src = synthetic_h264();
    let worst8 = transcode_and_check("av1 sw 8-bit mp4", &src, VideoCodecPolicy::Av1, Container::Mp4, 30.0);
    let worst10 = transcode_and_check_with("av1 sw 10-bit mp4", &src, VideoCodecPolicy::Av1, Container::Mp4, 30.0, |s| {
        s.with_bit_depth(rivet::BitDepth::TenBit)
    });
    let spec = OutputSpec::single_file(vec![Rung::new(128, 96)]).with_bit_depth(rivet::BitDepth::TenBit);
    assert_eq!(decoded_depth(&run(&src, &spec)), 10);
    eprintln!("av1 software: worst luma PSNR 8-bit {worst8:.2} dB, 10-bit {worst10:.2} dB");

    // A bitrate rung, coded by the encoder's rate controller to its rate.
    let bitrate = 300_000u32;
    let overrides = codec::encode::tuning::EncodeOverrides { bitrate: Some(bitrate), ..Default::default() };
    let rung = Rung::new(128, 96).with_quality(rivet::Quality::default().with_overrides(overrides));
    let file = run(&src, &OutputSpec::single_file(vec![rung]));
    let out = decode(&file);
    assert_eq!(out.luma.len(), 24);
    let achieved = file.len() as f64 * 8.0; // one second of video
    let ratio = achieved / f64::from(bitrate);
    eprintln!("av1 software bitrate rung: asked {bitrate} b/s, file {achieved:.0} b/s ({ratio:.2}x)");
    assert!((0.5..1.6).contains(&ratio), "{achieved:.0} b/s for {bitrate}");
}
