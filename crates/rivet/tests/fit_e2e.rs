//! Fitted rungs end to end: sources of every awkward shape, made by ffmpeg
//! with a disc drawn on them that is round *as shown*, through the job
//! engine, and the outputs checked by ffprobe (the size) and by measuring the
//! disc in ffmpeg's decode of the output (the shape).
//!
//! The reported defect this pins: explicit rungs were a straight resize to
//! `WxH`, so a 640x480 source through a 1280x720 rung came out stretched
//! sideways and upscaled, and a portrait phone video was squashed into
//! landscape.
//!
//! Like the other ffprobe tests, this skips (with a line saying so) where
//! ffmpeg is not installed, and it needs an H.264 encoder: a GPU, or
//! `TRANSCODE_ENCODER_BACKEND=h26x` for the software one.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;

use rivet::progress::NullSink;
use rivet::{RungArtifact, TranscodeSettings};

fn tools_available() -> bool {
    ["ffmpeg", "ffprobe"].iter().all(|tool| {
        Command::new(tool)
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
}

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rivet-fit-e2e-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A one-second source, `w x h` stored, with `sar` samples, carrying a disc
/// that is round on screen: in stored samples it is `1 / sar` as wide.
/// `extra` goes before the output name (codec, pixel format, container).
fn make_source(dir: &Path, name: &str, (w, h): (u32, u32), (sn, sd): (u32, u32), extra: &[&str]) -> Vec<u8> {
    let path = dir.join(name);
    let r = w.min(h) as f64 * 0.2;
    let disc = format!(
        "color=c=black:s={w}x{h}:r=10:d=1,format=yuv420p,\
         geq=lum='if(lte(hypot((X+0.5-W/2)*{sn}/{sd},Y+0.5-H/2),{r}),235,16)':cb=128:cr=128,setsar={sn}/{sd}"
    );
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-v", "error", "-y", "-f", "lavfi", "-i", &disc]);
    cmd.args(extra);
    cmd.arg(&path);
    let status = cmd.status().expect("ffmpeg runs");
    assert!(status.success(), "ffmpeg could not make {name}");
    std::fs::read(&path).unwrap()
}

fn h264(pix_fmt: &str) -> Vec<&str> {
    vec!["-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", pix_fmt]
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

/// ffprobe's width, height and sample aspect ratio of the file's video.
fn probe(dir: &Path, name: &str, bytes: &[u8]) -> (u32, u32, String) {
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,sample_aspect_ratio"])
        .args(["-of", "csv=p=0"])
        .arg(&path)
        .output()
        .expect("ffprobe runs");
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    let mut parts = text.split(',');
    let w = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    let h = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
    let sar = parts.next().unwrap_or("").to_string();
    (w, h, sar)
}

/// The bounding box of the disc in the middle frame of the file, decoded by
/// ffmpeg to grey.
fn disc_extent(dir: &Path, name: &str, (w, h): (u32, u32)) -> (u32, u32) {
    let path = dir.join(name);
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(&path)
        .args(["-vf", "select=eq(n\\,5)", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
        .output()
        .expect("ffmpeg decodes");
    let luma = out.stdout;
    assert_eq!(luma.len(), (w * h) as usize, "{name}: decoded frame size");
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

fn assert_round(dir: &Path, name: &str, bytes: &[u8], want: (u32, u32)) {
    let (w, h, sar) = probe(dir, name, bytes);
    assert_eq!((w, h), want, "{name}: ffprobe's size");
    assert!(matches!(sar.as_str(), "" | "1:1" | "N/A" | "0:1"), "{name}: output samples are not square: {sar}");
    let (dw, dh) = disc_extent(dir, name, (w, h));
    let roundness = f64::from(dw) / f64::from(dh);
    assert!(
        (roundness - 1.0).abs() <= 0.05,
        "{name}: the disc came out {dw}x{dh} in {w}x{h} — the picture is distorted"
    );
}

#[test]
fn every_shape_keeps_its_shape_through_explicit_rungs() {
    if !tools_available() {
        eprintln!("SKIP: ffmpeg / ffprobe not installed");
        return;
    }
    let dir = scratch();
    let mp4 = |name: &str, size, sar| {
        let mut extra = h264("yuv420p");
        extra.extend(["-f", "mp4"]);
        make_source(&dir, name, size, sar, &extra)
    };
    // (source, settings, the sizes that must come out, in order)
    type Case<'a> = (&'a str, Vec<u8>, &'a str, Vec<(u32, u32)>);
    let cases: Vec<Case> = vec![
        // 4:3 through the 720p preset's rung: kept 4:3, not upscaled.
        ("4x3", mp4("4x3.mp4", (640, 480), (1, 1)), "codec=h264 rungs=1280x720", vec![(640, 480)]),
        ("4x3-up", mp4("4x3.mp4", (640, 480), (1, 1)), "codec=h264 rungs=1280x720 upscale=1", vec![(960, 720)]),
        // Portrait through a landscape box: the box turns.
        ("9x16", mp4("9x16.mp4", (360, 640), (1, 1)), "codec=h264 rungs=1280x720", vec![(360, 640)]),
        ("9x16-up", mp4("9x16.mp4", (360, 640), (1, 1)), "codec=h264 rungs=1280x720 upscale=1", vec![(720, 1280)]),
        // 21:9 and 1:1 into 16:9 boxes.
        ("21x9", mp4("21x9.mp4", (1280, 548), (1, 1)), "codec=h264 rungs=854x480", vec![(854, 366)]),
        ("1x1", mp4("1x1.mp4", (480, 480), (1, 1)), "codec=h264 rungs=854x480", vec![(480, 480)]),
        // Anamorphic PAL 16:9: 720x576 at 64:45 is shown 1024x576.
        ("pal", mp4("pal.mp4", (720, 576), (64, 45)), "codec=h264 rungs=1920x1080", vec![(1024, 576)]),
        // Each fit on the 4:3 source.
        ("cover", mp4("4x3.mp4", (640, 480), (1, 1)), "codec=h264 rungs=640x360 fit=cover", vec![(640, 360)]),
        ("pad", mp4("4x3.mp4", (640, 480), (1, 1)), "codec=h264 rungs=854x480 fit=pad", vec![(854, 480)]),
        // A vertical rung that crops a landscape source.
        (
            "vertical",
            mp4("16x9.mp4", (1280, 720), (1, 1)),
            "codec=h264 rungs=1280x720,720x1280:cover:fixed",
            vec![(1280, 720), (406, 720)],
        ),
    ];
    for (name, input, settings, want) in cases {
        let (rungs, _) = run(&input, settings);
        let got: Vec<_> = rungs.iter().map(|r| (r.1, r.2)).collect();
        assert_eq!(got, want, "{name}: rung sizes");
        for (label, w, h, bytes) in &rungs {
            assert_round(&dir, &format!("{name}-{label}.mp4"), bytes, (*w, *h));
        }
    }

    // `stretch` is still there when asked for, and distorts as it always did.
    let (rungs, _) = run(&mp4("4x3.mp4", (640, 480), (1, 1)), "codec=h264 rungs=1280x720 fit=stretch");
    let (_, w, h, bytes) = &rungs[0];
    assert_eq!((*w, *h), (1280, 720));
    std::fs::write(dir.join("stretch.mp4"), bytes).unwrap();
    let (dw, dh) = disc_extent(&dir, "stretch.mp4", (*w, *h));
    assert!(f64::from(dw) / f64::from(dh) > 1.25, "stretch kept the disc round: {dw}x{dh}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_sample_aspect_is_read_from_every_container() {
    if !tools_available() {
        eprintln!("SKIP: ffmpeg / ffprobe not installed");
        return;
    }
    let dir = scratch().join("containers");
    std::fs::create_dir_all(&dir).unwrap();
    // PAL 16:9 in MP4 (`pasp`), Matroska (DisplayWidth) and MPEG-TS (the SPS
    // VUI alone), and as MPEG-2 (the sequence header's display ratio).
    let mut mkv = h264("yuv420p");
    mkv.extend(["-f", "matroska"]);
    let mut ts = h264("yuv420p");
    ts.extend(["-f", "mpegts"]);
    let mpeg2 = ["-c:v", "mpeg2video", "-q:v", "2", "-aspect", "16:9", "-f", "mpegts"];
    for (name, extra) in [("pal.mkv", mkv), ("pal.ts", ts), ("pal-mpeg2.ts", mpeg2.to_vec())] {
        let input = make_source(&dir, name, (720, 576), (64, 45), &extra);
        let probed = rivet::probe_bytes(&input).unwrap();
        assert_eq!(probed.sample_aspect, (64, 45), "{name}: sample aspect");
        assert_eq!(probed.display_dims(), (1024, 576), "{name}: display size");
        // MPEG-2 decodes on NVDEC or with the `ffmpeg` feature only; the
        // probe is what reads the ratio.
        if probed.video_codec == "mpeg2" {
            let spec = TranscodeSettings::parse_kv_line("codec=h264").unwrap().into_spec_for(&probed).unwrap();
            if let Err(e) = rivet::run_job_blocking(&input, &spec, None, Arc::new(NullSink))
                && format!("{e:#}").contains("no decoder available")
            {
                eprintln!("{name}: no MPEG-2 decoder in this build; checked the probe only");
                continue;
            }
        }
        let (rungs, _) = run(&input, "codec=h264 rungs=1280x720");
        assert_eq!((rungs[0].1, rungs[0].2), (1024, 576), "{name}");
        assert_round(&dir, &format!("{name}.mp4"), &rungs[0].3, (1024, 576));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_odd_sized_source_is_evened_down_with_its_colour_in_place() {
    if !tools_available() {
        eprintln!("SKIP: ffmpeg / ffprobe not installed");
        return;
    }
    let dir = scratch().join("odd");
    std::fs::create_dir_all(&dir).unwrap();
    // x264 refuses an odd 4:2:0 picture; 4:4:4 takes one, and the pipeline
    // brings it to 4:2:0 with the rounded-up chroma planes the scaler reads.
    let input = make_source(&dir, "853x480.mp4", (853, 480), (1, 1), &h264("yuv444p"));
    let (rungs, _) = run(&input, "codec=h264 rungs=1280x720");
    assert_eq!((rungs[0].1, rungs[0].2), (852, 480));
    assert_round(&dir, "853.mp4", &rungs[0].3, (852, 480));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rungs_a_small_source_collapses_are_merged_and_reported() {
    if !tools_available() {
        eprintln!("SKIP: ffmpeg / ffprobe not installed");
        return;
    }
    let dir = scratch().join("ladder");
    std::fs::create_dir_all(&dir).unwrap();
    let mut extra = h264("yuv420p");
    extra.extend(["-f", "mp4"]);
    let input = make_source(&dir, "4x3.mp4", (640, 480), (1, 1), &extra);
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
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_hls_ladder_is_fitted_too() {
    if !tools_available() {
        eprintln!("SKIP: ffmpeg / ffprobe not installed");
        return;
    }
    let dir = scratch().join("hls");
    std::fs::create_dir_all(&dir).unwrap();
    let mut extra = h264("yuv420p");
    extra.extend(["-f", "mp4"]);
    let input = make_source(&dir, "9x16.mp4", (360, 640), (1, 1), &extra);
    let probed = rivet::probe_bytes(&input).unwrap();
    let spec = TranscodeSettings::parse_kv_line("mode=hls codec=h264 segment-seconds=1 rungs=1920x1080,1280x720,480x270")
        .unwrap()
        .into_spec_for(&probed)
        .unwrap();
    let root = dir.join("package");
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
        let playlist = std::fs::read_dir(&media)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().is_some_and(|x| x == "m3u8"))
            .unwrap_or_else(|| panic!("no media playlist in {rel}"));
        let probe = Command::new("ffprobe")
            .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0"])
            .arg(&playlist)
            .output()
            .unwrap();
        let want = format!("{},{}", r.width, r.height);
        let text = String::from_utf8_lossy(&probe.stdout);
        let sizes: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        assert!(!sizes.is_empty() && sizes.iter().all(|s| *s == want), "{rel}: ffprobe says {sizes:?}, want {want}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
