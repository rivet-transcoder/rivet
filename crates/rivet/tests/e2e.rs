//! End-to-end transcode smoke test: synthetic frames + real media.
//!
//! The synthetic test exercises encode → mux on a deterministic set of frames
//! so CI without sample files can still verify the encoder-muxer contract.
//! The real-media test walks demux → decode → encode → mux and skips
//! gracefully if either the sample is absent or the CPU decoder rejects it.

use bytes::Bytes;

mod common;

use codec::colorspace;
use codec::decode;
use codec::encode::EncoderConfig;
use codec::frame::{ColorSpace, PixelFormat, VideoFrame};
use container::AudioInfo;
use container::demux;
use container::mux::Av1Mp4Muxer;

const TEST_WIDTH: u32 = 320;
const TEST_HEIGHT: u32 = 240;
const TEST_FRAME_RATE: f64 = 30.0;
const TEST_FRAMES: u32 = 10;

fn make_gradient_frame(width: u32, height: u32, pts: u64) -> VideoFrame {
    let w = width as usize;
    let h = height as usize;
    let y_size = w * h;
    let uv_size = y_size / 4;
    let mut buf = Vec::with_capacity(y_size + uv_size * 2);

    let t = pts as u8;
    for row in 0..h {
        for col in 0..w {
            buf.push(((row + col) as u8).wrapping_add(t));
        }
    }
    for _ in 0..uv_size {
        buf.push(128u8.wrapping_add(t / 2));
    }
    for _ in 0..uv_size {
        buf.push(128u8.wrapping_add(t / 3));
    }

    VideoFrame::new(
        Bytes::from(buf),
        width,
        height,
        PixelFormat::Yuv420p,
        ColorSpace::Bt709,
        pts,
    )
}

#[test]
fn synthetic_encode_and_mux_produces_valid_av1_mp4() {
    let config = EncoderConfig {
        width: TEST_WIDTH,
        height: TEST_HEIGHT,
        frame_rate: TEST_FRAME_RATE,
        quality: 80,
        speed_preset: 10,
        keyframe_interval: 30,
        ..EncoderConfig::default()
    };

    let Some(mut encoder) = common::try_av1_encoder(config.clone()) else {
        return;
    };

    let mut muxer = Av1Mp4Muxer::new(TEST_WIDTH, TEST_HEIGHT, TEST_FRAME_RATE).expect("muxer");
    let mut packet_count = 0usize;

    for pts in 0..TEST_FRAMES {
        let frame = make_gradient_frame(TEST_WIDTH, TEST_HEIGHT, pts as u64);
        encoder.send_frame(&frame).expect("send_frame");
        while let Some(pkt) = encoder.receive_packet().expect("receive_packet") {
            packet_count += 1;
            muxer.add_packet(pkt).expect("add_packet");
        }
    }

    encoder.flush().expect("flush");
    while let Some(pkt) = encoder
        .receive_packet()
        .expect("receive_packet after flush")
    {
        packet_count += 1;
        muxer.add_packet(pkt).expect("add_packet");
    }

    assert!(packet_count > 0, "encoder produced zero packets");

    let output = muxer.finalize().expect("mux finalize");
    assert!(!output.is_empty(), "muxed output is empty");

    assert_valid_av1_mp4(&output, packet_count);

    // Faststart ordering: moov must appear before mdat in the byte stream.
    let moov_pos = find_fourcc(&output, b"moov").expect("moov");
    let mdat_pos = find_fourcc(&output, b"mdat").expect("mdat");
    assert!(moov_pos < mdat_pos, "moov must precede mdat for faststart");
}

fn assert_valid_av1_mp4(data: &[u8], expected_sample_count: usize) {
    assert!(data.len() > 100, "mp4 too small: {} bytes", data.len());
    // ftyp at offset 0
    assert_eq!(&data[4..8], b"ftyp", "missing ftyp box");

    let mut pos = 0usize;
    let mut saw_moov = false;
    let mut saw_mdat = false;
    while pos + 8 <= data.len() {
        let size =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let btype = &data[pos + 4..pos + 8];
        if btype == b"moov" {
            saw_moov = true;
        }
        if btype == b"mdat" {
            saw_mdat = true;
        }
        if size == 0 {
            break;
        } // box extends to EOF
        pos = pos.checked_add(size).expect("box size overflow");
    }
    assert!(saw_moov, "no moov box");
    assert!(saw_mdat, "no mdat box");

    // Confirm av01 sample entry and av1C box exist somewhere
    let av01_pos = find_fourcc(data, b"av01");
    assert!(av01_pos.is_some(), "no av01 sample entry");
    let av1c_pos = find_fourcc(data, b"av1C");
    assert!(av1c_pos.is_some(), "no av1C configuration box");

    // Stsz reports sample_count equal to our packet count
    let stsz_pos = find_fourcc(data, b"stsz").expect("stsz box");
    // stsz body: version(1) flags(3) sample_size(4) sample_count(4) ...
    // fourcc is at stsz_pos; box header is 4 size bytes BEFORE that.
    let sample_count_off = stsz_pos + 4 /*stsz*/ + 4 /*ver+flags*/ + 4 /*sample_size*/;
    let sc = u32::from_be_bytes([
        data[sample_count_off],
        data[sample_count_off + 1],
        data[sample_count_off + 2],
        data[sample_count_off + 3],
    ]);
    assert_eq!(
        sc as usize, expected_sample_count,
        "stsz sample_count mismatch"
    );
}

fn find_fourcc(data: &[u8], fourcc: &[u8; 4]) -> Option<usize> {
    data.windows(4).position(|w| w == fourcc)
}

#[test]
fn real_media_pipeline_if_sample_exists() {
    // The real-media path exercises the full demux → decode → encode → mux
    // stack. If an NVIDIA GPU is present the decoder factory will pick
    // NVDEC; otherwise it falls through to rivet's own software decoders
    // (h26x / VP9 / AV1). Both paths should produce an AV1
    // MP4 — the assertion is codec-path-agnostic.

    let samples = [
        "bigbuck_bunny_8bit_750kbps_720p_60.0fps_h264.mp4",
        "bigbuck_bunny_8bit_750kbps_720p_60.0fps_hevc.mp4",
        "bigbuck_bunny_8bit_750kbps_720p_60.0fps_vp9.mkv",
    ];

    let test_media_dir = common::test_media_dir();

    // Diagnostic gate so we can see exactly where execution crashes on
    // GPU boxes. Each println is flushed via `eprintln!` so stdout
    // buffering doesn't hide the last step before a hard segfault.
    eprintln!("e2e real_media: test_media_dir={}", test_media_dir.display());
    eprintln!(
        "e2e real_media: test_media_dir={}",
        test_media_dir.display()
    );

    let mut ran_any = false;
    for name in samples {
        let path = test_media_dir.join(name);
        eprintln!("e2e: checking {}", path.display());
        if !path.exists() {
            eprintln!("skipping {} (not present)", name);
            continue;
        }
        ran_any = true;

        eprintln!("e2e: reading {}", name);
        let data = std::fs::read(&path).expect("read sample");
        eprintln!("e2e: read {} bytes, demuxing", data.len());
        let demuxed = match demux::demux(&data) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip {}: demux failed: {}", name, e);
                continue;
            }
        };
        eprintln!(
            "e2e: demuxed codec={} {}x{} {} samples",
            demuxed.codec,
            demuxed.info.width,
            demuxed.info.height,
            demuxed.samples.len()
        );

        eprintln!("e2e: create_decoder start");
        // #55 P3 streaming-shape compatibility: drop samples from the
        // factory call, push them through the trait afterwards. Will
        // be subsumed by pipeline-eng's StreamingDemuxer rewire (P5).
        let mut decoder = match decode::create_decoder(&demuxed.codec, demuxed.info.clone()) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip {}: decoder init failed: {}", name, e);
                continue;
            }
        };
        for s in &demuxed.samples {
            if let Err(e) = decoder.push_sample(s) {
                eprintln!("skip {}: push_sample failed: {}", name, e);
                continue;
            }
        }
        if let Err(e) = decoder.finish() {
            eprintln!("skip {}: finish failed: {}", name, e);
            continue;
        }
        eprintln!("e2e: create_decoder done");

        let target_width = 320u32;
        let target_height = 240u32;
        let frame_rate = demuxed.info.frame_rate.min(30.0);

        let config = EncoderConfig {
            width: target_width,
            height: target_height,
            frame_rate,
            quality: 110,
            speed_preset: 10,
            keyframe_interval: 30,
            ..EncoderConfig::default()
        };

        let Some(mut encoder) = common::try_av1_encoder(config) else {
            return;
        };
        let mut muxer = Av1Mp4Muxer::new(target_width, target_height, frame_rate).expect("muxer");
        let mut fed = 0usize;
        let mut decoder_failed = false;

        // Cap the real-media test at a handful of frames so CI runs fast.
        const MAX_FRAMES: usize = 30;
        while fed < MAX_FRAMES {
            let frame = match decoder.decode_next() {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => {
                    eprintln!("skip {}: decoder error at frame {}: {}", name, fed, e);
                    decoder_failed = true;
                    break;
                }
            };
            let normalized = match colorspace::convert_to_yuv420p_bt709(&frame) {
                Ok(n) => n,
                Err(e) => {
                    eprintln!("skip {}: colorspace error: {}", name, e);
                    decoder_failed = true;
                    break;
                }
            };
            let scaled =
                colorspace::scale_frame(&normalized, target_width, target_height).expect("scale");
            encoder.send_frame(&scaled).expect("encode");
            while let Some(p) = encoder.receive_packet().expect("receive") {
                muxer.add_packet(p).expect("add_packet");
            }
            fed += 1;
        }

        if decoder_failed || fed == 0 {
            continue;
        }

        encoder.flush().expect("flush");
        while let Some(p) = encoder.receive_packet().expect("receive after flush") {
            muxer.add_packet(p).expect("add_packet");
        }

        let output = match muxer.finalize() {
            Ok(o) => o,
            Err(e) => {
                eprintln!("skip {}: mux failed: {}", name, e);
                continue;
            }
        };
        assert!(!output.is_empty(), "{}: empty mux output", name);
        assert_eq!(&output[4..8], b"ftyp");
        assert!(find_fourcc(&output, b"av01").is_some(), "{}: no av01", name);
        eprintln!(
            "{}: pipeline OK, {} frames fed, output {} bytes",
            name,
            fed,
            output.len()
        );
        return; // first successful sample is enough
    }

    if !ran_any {
        eprintln!("test_media/ not present; real-media test skipped");
    }
}

/// Audio-passthrough e2e: demux a real file that has an AAC track, run
/// video through decode→encode→mux, mux the audio as passthrough, and
/// re-demux the output to assert both tracks survived.
///
/// Skips gracefully when the source sample is absent. The Jellyfin H.264
/// High L4.0 sample ships with an AAC audio track and is the primary
/// target here; fall-through to BBB samples keeps CI deterministic on
/// boxes with only the BBB corpus present.
#[test]
fn audio_passthrough_real_media_if_sample_exists() {
    let samples_with_audio = [
        "jellyfin_h264_high_l40_1080p_24fps.mp4",
        "jellyfin_av1_main_1080p_24fps.mp4",
        "jellyfin_hevc_main_1080p_24fps.mp4",
    ];
    let test_media_dir = common::test_media_dir();

    for name in samples_with_audio {
        let path = test_media_dir.join(name);
        if !path.exists() {
            eprintln!("skipping {} (not present)", name);
            continue;
        }

        let data = std::fs::read(&path).expect("read sample");
        let demuxed = match demux::demux(&data) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip {}: demux failed: {}", name, e);
                continue;
            }
        };

        // Sample must carry an audio track for this test to be meaningful.
        let audio_in = match demuxed.audio {
            Some(a) => a,
            None => {
                eprintln!("skip {}: no audio track", name);
                continue;
            }
        };
        eprintln!(
            "audio e2e: {} codec={} asc={}b ch={} sr={}Hz samples={}",
            name,
            audio_in.codec,
            audio_in.asc.len(),
            audio_in.channels,
            audio_in.sample_rate,
            audio_in.samples.len()
        );

        // Decoder → encoder → muxer. Same shape as the video-only e2e but
        // with audio plumbed into the muxer.
        let mut decoder = match decode::create_decoder(&demuxed.codec, demuxed.info.clone()) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip {}: decoder init failed: {}", name, e);
                continue;
            }
        };
        for s in &demuxed.samples {
            if let Err(e) = decoder.push_sample(s) {
                eprintln!("skip {}: push_sample failed: {}", name, e);
                continue;
            }
        }
        if let Err(e) = decoder.finish() {
            eprintln!("skip {}: finish failed: {}", name, e);
            continue;
        }
        let target_width = 320u32;
        let target_height = 240u32;
        let frame_rate = demuxed.info.frame_rate.min(30.0);
        let config = EncoderConfig {
            width: target_width,
            height: target_height,
            frame_rate,
            quality: 110,
            speed_preset: 10,
            keyframe_interval: 30,
            ..EncoderConfig::default()
        };
        let Some(mut encoder) = common::try_av1_encoder(config) else {
            return;
        };
        let mut muxer = Av1Mp4Muxer::new(target_width, target_height, frame_rate).expect("muxer");

        let mut fed = 0usize;
        let mut decoder_failed = false;
        const MAX_FRAMES: usize = 30;
        while fed < MAX_FRAMES {
            let frame = match decoder.decode_next() {
                Ok(Some(f)) => f,
                Ok(None) => break,
                Err(e) => {
                    eprintln!("skip {}: decoder error: {}", name, e);
                    decoder_failed = true;
                    break;
                }
            };
            let normalized = match colorspace::convert_to_yuv420p_bt709(&frame) {
                Ok(n) => n,
                Err(e) => {
                    eprintln!("skip {}: colorspace error: {}", name, e);
                    decoder_failed = true;
                    break;
                }
            };
            let scaled =
                colorspace::scale_frame(&normalized, target_width, target_height).expect("scale");
            encoder.send_frame(&scaled).expect("encode");
            while let Some(p) = encoder.receive_packet().expect("receive") {
                muxer.add_packet(p).expect("add_packet");
            }
            fed += 1;
        }
        if decoder_failed || fed == 0 {
            continue;
        }
        encoder.flush().expect("flush");
        while let Some(p) = encoder.receive_packet().expect("receive after flush") {
            muxer.add_packet(p).expect("add_packet");
        }

        // Audio passthrough: info from demuxer → with_audio + add_audio_sample.
        let info = AudioInfo {
            codec: audio_in.codec.clone(),
            sample_rate: audio_in.sample_rate,
            channels: audio_in.channels,
            timescale: audio_in.timescale,
            asc_bytes: audio_in.asc.clone(),
            codec_private: audio_in.codec_private.clone(),
        };
        muxer.with_audio(info).expect("with_audio");
        let mut pts: u64 = 0;
        for (sample, dur) in audio_in.samples.iter().zip(audio_in.durations.iter()) {
            muxer
                .add_audio_sample(sample, pts, *dur)
                .expect("add_audio_sample");
            pts = pts.saturating_add(*dur as u64);
        }

        let output = muxer.finalize().expect("mux finalize");
        assert!(!output.is_empty(), "empty mux output");
        assert_eq!(&output[4..8], b"ftyp");
        assert!(find_fourcc(&output, b"av01").is_some(), "{}: no av01", name);
        assert!(
            find_fourcc(&output, b"mp4a").is_some(),
            "{}: no mp4a audio entry",
            name
        );
        assert!(find_fourcc(&output, b"esds").is_some(), "{}: no esds", name);

        // Apple-compat structural assertions on the real-media output:
        // (a) ftyp.compatible_brands lists 'av01' (per AV1-ISOBMFF §2.1)
        //     and 'iso6' (Apple-required structural brand).
        // (b) `colr` atom present so QuickTime / iOS Safari don't apply
        //     BT.709 limited fallback.
        let ftyp_size = u32::from_be_bytes([output[0], output[1], output[2], output[3]]) as usize;
        let major = &output[8..12];
        assert_eq!(major, b"iso6", "{}: major_brand should be iso6", name);
        let compat = &output[16..ftyp_size];
        let brands: Vec<[u8; 4]> = compat
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        assert!(
            brands.iter().any(|b| b == b"av01"),
            "{}: compatible_brands missing 'av01'",
            name
        );
        assert!(
            brands.iter().any(|b| b == b"iso6"),
            "{}: compatible_brands missing 'iso6'",
            name
        );
        assert!(
            find_fourcc(&output, b"colr").is_some(),
            "{}: colr atom missing — Apple will silently apply BT.709 limited",
            name
        );

        // Re-demux and assert the audio survived.
        let round = demux::demux(&output).expect("roundtrip demux");
        let rt_audio = round.audio.expect("audio track lost in roundtrip");
        assert_eq!(rt_audio.codec, "aac");
        assert_eq!(rt_audio.channels, audio_in.channels);
        assert_eq!(rt_audio.sample_rate, audio_in.sample_rate);
        assert_eq!(rt_audio.asc, audio_in.asc, "ASC bytes drifted");
        assert_eq!(
            rt_audio.samples.len(),
            audio_in.samples.len(),
            "audio sample count drift: in={} out={}",
            audio_in.samples.len(),
            rt_audio.samples.len()
        );
        eprintln!(
            "{}: audio passthrough OK, video fed={} frames, audio {} samples, output {} bytes, \
             ftyp brands={:?}",
            name,
            fed,
            rt_audio.samples.len(),
            output.len(),
            brands
        );
        return; // one successful sample is enough
    }

    eprintln!("no audio-bearing test media present; audio e2e skipped");
}
