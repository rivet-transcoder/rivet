//! Fitted rungs end to end: sources of every awkward shape, made here by
//! this workspace's own encoders (`common::synth`) with a disc drawn on them
//! that is round *as shown*, through the job engine, and the outputs read
//! back with rivet's demuxer (the size and sample aspect: the `tkhd`/sample
//! entry, `pasp` and the SPS VUI) and decoded with rivet's decoder (the
//! disc's shape, measured in the decoded picture).
//!
//! The reported defect this pins: explicit rungs were a straight resize to
//! `WxH`, so a 640x480 source through a 1280x720 rung came out stretched
//! sideways and upscaled, and a portrait phone video was squashed into
//! landscape.
//!
//! It needs an H.264 encoder: a GPU, or `TRANSCODE_ENCODER_BACKEND=h26x`
//! (or the `h26x-fallback` feature) for the software one.

mod common;

use std::sync::Arc;

use common::synth;
use h26x::ChromaFormat;
use rivet::progress::NullSink;
use rivet::{RungArtifact, TranscodeSettings};

/// The containers a source is written in.
#[derive(Clone, Copy, PartialEq)]
enum Wrap {
    /// MP4 with the aspect in a `pasp` box only.
    Mp4,
    /// Matroska with the aspect as `DisplayWidth` / `DisplayHeight` only.
    Mkv,
    /// MPEG-TS, the aspect in the SPS VUI alone.
    Ts,
    /// MPEG-2 video in MPEG-TS, the sequence header's display aspect.
    Mpeg2Ts,
}

/// A one-second source at 10 fps, `w x h` stored, with `sar` samples,
/// carrying a disc that is round on screen: in stored samples it is
/// `1 / sar` as wide.
fn make_source((w, h): (u32, u32), (sn, sd): (u32, u32), chroma: ChromaFormat, wrap: Wrap) -> Vec<u8> {
    const FPS: u32 = 10;
    let pictures = (0..FPS).map(|_| synth::disc(w, h, (sn, sd), chroma));
    let square = (sn, sd) == (1, 1);
    match wrap {
        Wrap::Mpeg2Ts => {
            // 720x576 at 64:45 is a 16:9 picture: aspect_ratio_information 3.
            let aspect = if square { 1 } else { 3 };
            let coded = synth::encode_mpeg2(w, h, 25, aspect, pictures);
            synth::ts(&coded, 0x02, 25)
        }
        Wrap::Ts => {
            let sar = (!square).then_some((sn as u16, sd as u16));
            let cfg = synth::H264 { chroma, qp: 12, sar, ..synth::H264::new(w, h, FPS) };
            synth::ts(&synth::encode_h264(&cfg, pictures), 0x1B, FPS)
        }
        Wrap::Mp4 | Wrap::Mkv => {
            let cfg = synth::H264 { chroma, qp: 12, ..synth::H264::new(w, h, FPS) };
            let coded = synth::encode_h264(&cfg, pictures);
            if wrap == Wrap::Mp4 {
                synth::mp4(&coded, w, h, FPS, None, (!square).then_some((sn, sd)))
            } else {
                let display = (u64::from(w) * u64::from(sn) / u64::from(sd)) as u32;
                synth::mkv(&coded, w, h, FPS, Some((display, h)), None)
            }
        }
    }
}

fn mp4(size: (u32, u32), sar: (u32, u32)) -> Vec<u8> {
    make_source(size, sar, ChromaFormat::Yuv420, Wrap::Mp4)
}

/// A produced rung: label, width, height, file.
type Produced = (String, u32, u32, Vec<u8>);

/// Run `settings` over `input`; returns each produced rung's file and the
/// job's report of what fitting did.
fn run(input: &[u8], settings: &str) -> (Vec<Produced>, Vec<rivet::fit::FittedRung>) {
    let probed = rivet::probe_bytes(input).expect("probe");
    let spec = TranscodeSettings::parse_kv_line(settings)
        .expect("settings")
        .into_spec_for(&probed)
        .expect("spec");
    let out = rivet::run_job_blocking(input, &spec, None, Arc::new(NullSink)).expect("the job runs");
    let rungs = out
        .rungs
        .into_iter()
        .map(|r| match r.artifact {
            RungArtifact::File(bytes) => (r.label, r.width, r.height, bytes),
            RungArtifact::HlsRendition { .. } => unreachable!("single-file job"),
        })
        .collect();
    (rungs, out.renditions)
}

/// The file's video as rivet reads it: stored width and height, sample
/// aspect, and the luma plane of frame 5 (the middle one) decoded.
fn read_back(name: &str, bytes: &[u8]) -> (u32, u32, (u32, u32), Vec<u8>) {
    let probed = rivet::probe_bytes(bytes).unwrap_or_else(|e| panic!("{name}: rivet probes its own output: {e:#}"));
    let mut demux = container::streaming::demux_streaming(bytes).unwrap_or_else(|e| panic!("{name}: demux: {e:#}"));
    let header = demux.header().clone();
    let (w, h) = (header.info.width, header.info.height);
    let mut dec = codec::decode::create_decoder(&header.codec, header.info.clone()).expect("a decoder");
    let mut frames = Vec::new();
    while let Some(s) = demux.next_video_sample().unwrap() {
        dec.push_sample(&s.data).unwrap_or_else(|e| panic!("{name}: decode: {e:#}"));
        while let Some(f) = dec.decode_next().unwrap() {
            frames.push(f);
        }
    }
    dec.finish().unwrap();
    while let Some(f) = dec.decode_next().unwrap() {
        frames.push(f);
    }
    assert!(frames.len() > 5, "{name}: {} frames decoded", frames.len());
    let f = &frames[5];
    assert_eq!((f.width, f.height), (w, h), "{name}: decoded frame size");
    let luma = f.data[..(w * h) as usize].to_vec();
    (probed.stored_width, probed.stored_height, probed.sample_aspect, luma)
}

/// The bounding box of the disc in `luma`.
fn disc_extent(name: &str, luma: &[u8], (w, h): (u32, u32)) -> (u32, u32) {
    let (mut x0, mut x1, mut y0, mut y1) = (u32::MAX, 0, u32::MAX, 0);
    for y in 0..h {
        for x in 0..w {
            if luma[(y * w + x) as usize] > 125 {
                (x0, x1, y0, y1) = (x0.min(x), x1.max(x), y0.min(y), y1.max(y));
            }
        }
    }
    assert!(x0 <= x1, "{name}: no disc in the output");
    (x1 + 1 - x0, y1 + 1 - y0)
}

fn assert_round(name: &str, bytes: &[u8], want: (u32, u32)) {
    let (w, h, sar, luma) = read_back(name, bytes);
    assert_eq!((w, h), want, "{name}: the output's size");
    assert_eq!(sar, (1, 1), "{name}: output samples are not square");
    let (dw, dh) = disc_extent(name, &luma, (w, h));
    let roundness = f64::from(dw) / f64::from(dh);
    assert!(
        (roundness - 1.0).abs() <= 0.05,
        "{name}: the disc came out {dw}x{dh} in {w}x{h} — the picture is distorted"
    );
}

#[test]
fn every_shape_keeps_its_shape_through_explicit_rungs() {
    // (source, settings, the sizes that must come out, in order)
    type Case<'a> = (&'a str, Vec<u8>, &'a str, Vec<(u32, u32)>);
    let cases: Vec<Case> = vec![
        // 4:3 through the 720p preset's rung: kept 4:3, not upscaled.
        ("4x3", mp4((640, 480), (1, 1)), "codec=h264 rungs=1280x720", vec![(640, 480)]),
        ("4x3-up", mp4((640, 480), (1, 1)), "codec=h264 rungs=1280x720 upscale=1", vec![(960, 720)]),
        // Portrait through a landscape box: the box turns.
        ("9x16", mp4((360, 640), (1, 1)), "codec=h264 rungs=1280x720", vec![(360, 640)]),
        ("9x16-up", mp4((360, 640), (1, 1)), "codec=h264 rungs=1280x720 upscale=1", vec![(720, 1280)]),
        // 21:9 and 1:1 into 16:9 boxes.
        ("21x9", mp4((1280, 548), (1, 1)), "codec=h264 rungs=854x480", vec![(854, 366)]),
        ("1x1", mp4((480, 480), (1, 1)), "codec=h264 rungs=854x480", vec![(480, 480)]),
        // Anamorphic PAL 16:9: 720x576 at 64:45 is shown 1024x576.
        ("pal", mp4((720, 576), (64, 45)), "codec=h264 rungs=1920x1080", vec![(1024, 576)]),
        // Each fit on the 4:3 source.
        ("cover", mp4((640, 480), (1, 1)), "codec=h264 rungs=640x360 fit=cover", vec![(640, 360)]),
        ("pad", mp4((640, 480), (1, 1)), "codec=h264 rungs=854x480 fit=pad", vec![(854, 480)]),
        // A vertical rung that crops a landscape source.
        (
            "vertical",
            mp4((1280, 720), (1, 1)),
            "codec=h264 rungs=1280x720,720x1280:cover:fixed",
            vec![(1280, 720), (406, 720)],
        ),
    ];
    for (name, input, settings, want) in cases {
        let (rungs, _) = run(&input, settings);
        let got: Vec<_> = rungs.iter().map(|r| (r.1, r.2)).collect();
        assert_eq!(got, want, "{name}: rung sizes");
        for (label, w, h, bytes) in &rungs {
            assert_round(&format!("{name}-{label}"), bytes, (*w, *h));
        }
    }

    // `stretch` is still there when asked for, and distorts as it always did.
    let (rungs, _) = run(&mp4((640, 480), (1, 1)), "codec=h264 rungs=1280x720 fit=stretch");
    let (_, w, h, bytes) = &rungs[0];
    assert_eq!((*w, *h), (1280, 720));
    let (_, _, _, luma) = read_back("stretch", bytes);
    let (dw, dh) = disc_extent("stretch", &luma, (*w, *h));
    assert!(f64::from(dw) / f64::from(dh) > 1.25, "stretch kept the disc round: {dw}x{dh}");
}

#[test]
fn the_sample_aspect_is_read_from_every_container() {
    // PAL 16:9 in MP4 (`pasp` alone), Matroska (DisplayWidth alone) and
    // MPEG-TS (the SPS VUI alone), and as MPEG-2 (the sequence header's
    // display ratio).
    for (name, wrap) in [("pal.mp4", Wrap::Mp4), ("pal.mkv", Wrap::Mkv), ("pal.ts", Wrap::Ts), ("pal-mpeg2.ts", Wrap::Mpeg2Ts)] {
        let input = make_source((720, 576), (64, 45), ChromaFormat::Yuv420, wrap);
        let probed = rivet::probe_bytes(&input).unwrap();
        assert_eq!(probed.sample_aspect, (64, 45), "{name}: sample aspect");
        assert_eq!(probed.display_dims(), (1024, 576), "{name}: display size");
        let (rungs, _) = run(&input, "codec=h264 rungs=1280x720");
        assert_eq!((rungs[0].1, rungs[0].2), (1024, 576), "{name}");
        assert_round(name, &rungs[0].3, (1024, 576));
    }
}

#[test]
fn an_odd_sized_source_is_evened_down_with_its_colour_in_place() {
    // An odd 4:4:4 picture: the pipeline brings it to 4:2:0 with the
    // rounded-up chroma planes the scaler reads.
    let input = make_source((853, 480), (1, 1), ChromaFormat::Yuv444, Wrap::Mp4);
    let (rungs, _) = run(&input, "codec=h264 rungs=1280x720");
    assert_eq!((rungs[0].1, rungs[0].2), (852, 480));
    assert_round("853", &rungs[0].3, (852, 480));
}

#[test]
fn rungs_a_small_source_collapses_are_merged_and_reported() {
    let input = mp4((640, 480), (1, 1));
    // The compat preset's ladder over a 640x480 source.
    let (rungs, report) = run(&input, "codec=h264 rungs=1920x1080,1280x720,854x480,640x360");
    let got: Vec<_> = rungs.iter().map(|r| (r.0.as_str(), r.1, r.2)).collect();
    assert_eq!(got, vec![("480p", 640, 480), ("360p", 480, 360)]);
    let merged: Vec<_> = report.iter().map(|r| (r.requested, r.output, r.duplicate_of)).collect();
    assert_eq!(
        merged,
        vec![
            ((1920, 1080), (640, 480), None),
            ((1280, 720), (640, 480), Some(0)),
            ((854, 480), (640, 480), Some(0)),
            ((640, 360), (480, 360), None),
        ]
    );
}

#[test]
fn an_hls_ladder_is_fitted_too() {
    let dir = tempfile::tempdir().expect("temp dir");
    let input = mp4((360, 640), (1, 1));
    let probed = rivet::probe_bytes(&input).unwrap();
    let spec = TranscodeSettings::parse_kv_line("mode=hls codec=h264 segment-seconds=1 rungs=1920x1080,1280x720,480x270")
        .unwrap()
        .into_spec_for(&probed)
        .unwrap();
    let root = dir.path().join("package");
    let out = match rivet::run_job_blocking(&input, &spec, Some(&root), Arc::new(NullSink)) {
        Ok(out) => out,
        Err(e) if format!("{e:#}").contains("encoder") => {
            eprintln!("SKIP: no encoder for the HLS path here: {e:#}");
            return;
        }
        Err(e) => panic!("the HLS job: {e:#}"),
    };
    let got: Vec<_> = out.rungs.iter().map(|r| (r.label.as_str(), r.width, r.height)).collect();
    // Portrait: 1920x1080 and 1280x720 turn and collapse onto the source;
    // 480x270 turns to 270x480.
    assert_eq!(got, vec![("360p", 360, 640), ("270p", 270, 480)]);
    assert_eq!(out.renditions.iter().filter(|r| r.duplicate_of.is_some()).count(), 1);
    let master = std::fs::read_to_string(out.master_playlist.unwrap()).unwrap();
    assert!(master.contains("RESOLUTION=360x640"), "{master}");
    assert!(!master.contains("RESOLUTION=1920x1080") && !master.contains("RESOLUTION=1080x1920"), "{master}");
    for r in &out.rungs {
        let (rel, media) = match &r.artifact {
            RungArtifact::HlsRendition { dir, relative_dir } => (relative_dir.clone(), dir.clone()),
            RungArtifact::File(_) => unreachable!(),
        };
        // The rendition as a player gets it: the init segment, then every
        // media segment the playlist lists, in order.
        let playlist = std::fs::read_dir(&media)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().is_some_and(|x| x == "m3u8"))
            .unwrap_or_else(|| panic!("no media playlist in {rel}"));
        let text = std::fs::read_to_string(&playlist).unwrap();
        let init = text
            .lines()
            .find_map(|l| l.strip_prefix("#EXT-X-MAP:URI=\"")?.split('"').next())
            .unwrap_or("init.mp4");
        let mut joined = std::fs::read(media.join(init)).unwrap();
        for seg in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
            joined.extend(std::fs::read(media.join(seg.trim())).unwrap());
        }
        let (w, h, sar, _) = read_back(&rel, &joined);
        assert_eq!((w, h), (r.width, r.height), "{rel}: the rendition's size");
        assert_eq!(sar, (1, 1), "{rel}: square samples");
    }
}
