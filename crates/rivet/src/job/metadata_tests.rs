//! Identifying metadata through the job engine: none reaches an output
//! unless `metadata-keep` names it, and what it names arrives.

use std::sync::Arc;

use container::metadata::{self, Categories, Category, Location, Metadata};

use super::lossless_tests::{native_flac, signal};
use crate::progress::NullSink;
use crate::settings::TranscodeSettings;
use crate::{JobOutput, RungArtifact};

fn run(input: &[u8], line: &str) -> anyhow::Result<JobOutput> {
    let settings = TranscodeSettings::parse_kv_line(line)?;
    let spec = settings.into_spec(0, 0)?;
    crate::run_job_blocking(input, &spec, None, Arc::new(NullSink))
}

fn file(out: &JobOutput) -> &[u8] {
    match &out.rungs[0].artifact {
        RungArtifact::File(b) => b,
        other => panic!("expected a file, got {other:?}"),
    }
}

/// What a phone or a tagger leaves in a file.
fn identifying() -> Metadata {
    let mut m = Metadata::default();
    m.location = Some(Location::coordinates(37.3349, -122.009, Some(10.0)));
    m.device.make = Some("Apple".into());
    m.device.model = Some("iPhone 15 Pro".into());
    m.device.software = Some("17.4.1".into());
    m.capture_time = Some("2024-05-01T12:34:56+02:00".into());
    m.descriptive.insert("title".into(), "Harbour".into());
    m.descriptive.insert("artist".into(), "Someone".into());
    m
}

fn tagged_flac() -> Vec<u8> {
    let flac = native_flac(&signal(20_000, 2, 16), 2, 16);
    metadata::write::flac(&flac, &identifying()).unwrap()
}

/// An `.m4a` whose FLAC `dfLa` carries the stream's Vorbis comments, the way
/// a tagging tool that writes FLAC-in-MP4 leaves it.
fn tagged_flac_m4a() -> Vec<u8> {
    let native = tagged_flac();
    let track = container::streaming::demux_audio(bytes::Bytes::from(native.clone())).unwrap().unwrap().track;
    // The blocks as the native file has them, all of them.
    let blocks = native[4..native.len() - track.samples.iter().map(|s| s.len()).sum::<usize>()].to_vec();
    let info = container::AudioInfo::flac(48_000, 2, blocks);
    let frames: Vec<(Vec<u8>, u32)> = track.samples.iter().map(|s| (s.to_vec(), 4096)).collect();
    container::mux::write_audio_mp4(&info, &frames, Default::default()).unwrap()
}

#[test]
fn a_tagged_flac_source_gives_an_untagged_flac_by_default() {
    let src = tagged_flac();
    assert_eq!(metadata::read(&src).categories(), Categories::ALL.minus(Categories::NONE.with(Category::Device)));
    let out = run(&src, "mode=audio audio=flac").unwrap();
    let m = metadata::read(file(&out));
    assert!(m.is_empty(), "{m:?}");
}

#[test]
fn metadata_keep_descriptive_carries_the_tags_and_nothing_else() {
    let out = run(&tagged_flac(), "mode=audio audio=flac metadata-keep=descriptive").unwrap();
    let m = metadata::read(file(&out));
    assert_eq!(m.categories(), Categories::NONE.with(Category::Descriptive), "{m:?}");
    assert_eq!(m.descriptive.get("title").map(String::as_str), Some("Harbour"));
}

#[test]
fn every_kept_category_reaches_an_m4a() {
    let out = run(&tagged_flac(), "mode=audio audio=alac metadata-keep=all").unwrap();
    let m = metadata::read(file(&out));
    // A FLAC source says where (LOCATION) and when (DATE); it has no device
    // beyond its encoder, which is not the output's.
    assert!(m.location.as_ref().is_some_and(Location::has_coordinates), "{m:?}");
    assert_eq!(m.capture_time.as_deref(), Some("2024-05-01T12:34:56+02:00"));
    assert_eq!(m.descriptive.get("artist").map(String::as_str), Some("Someone"));
}

#[test]
fn a_flac_in_mp4_passthrough_drops_the_sources_comments() {
    let src = tagged_flac_m4a();
    assert!(metadata::read(&src).categories().contains(Category::Descriptive), "the fixture carries tags");
    let out = run(&src, "mode=audio audio=flac audio-container=mp4").unwrap();
    let m = metadata::read(file(&out));
    assert!(m.is_empty(), "{m:?}");
}

#[test]
fn hls_refuses_metadata_keep() {
    let settings = TranscodeSettings::parse_kv_line("mode=hls rung=320x240 metadata-keep=location").unwrap();
    let err = format!("{:#}", settings.into_spec(640, 480).unwrap_err());
    assert!(err.contains("metadata-keep is not available for HLS"), "{err}");
    assert!(TranscodeSettings::parse_kv_line("metadata-keep=gps").is_err());
}

/// A short H.264 clip at `RIVET_TEST_MEDIA/stills_clip.mp4`, dressed as a
/// phone's recording.
fn phone_clip() -> Option<Vec<u8>> {
    let dir = std::env::var_os("RIVET_TEST_MEDIA")?;
    let clip = std::fs::read(std::path::Path::new(&dir).join("stills_clip.mp4")).ok()?;
    Some(metadata::write::mp4(&clip, &identifying()).unwrap())
}

#[test]
fn a_phones_video_comes_out_clean_unless_asked() {
    let Some(src) = phone_clip() else {
        eprintln!("SKIP: RIVET_TEST_MEDIA/stills_clip.mp4 not present");
        return;
    };
    assert_eq!(metadata::read(&src).categories(), Categories::ALL);
    let out = match run(&src, "codec=h264 rung=160x120") {
        Ok(out) => out,
        Err(e) if format!("{e:#}").contains("encoder") => {
            eprintln!("SKIP: no H.264 encoder: {e:#}");
            return;
        }
        Err(e) => panic!("{e:#}"),
    };
    let m = metadata::read(file(&out));
    assert!(m.is_empty(), "default output carries {m:?}");

    let out = run(&src, "codec=h264 rung=160x120 metadata-keep=location,capture_time").unwrap();
    let m = metadata::read(file(&out));
    assert_eq!(m.categories(), Categories::NONE.with(Category::Location).with(Category::CaptureTime), "{m:?}");
    let loc = m.location.unwrap();
    assert_eq!((loc.latitude, loc.longitude), (Some(37.3349), Some(-122.009)));
    // Still a playable file: the samples are where the offsets say.
    let demuxed = container::streaming::demux_streaming_shared(bytes::Bytes::copy_from_slice(file(&out))).unwrap();
    assert_eq!((demuxed.header().info.width, demuxed.header().info.height), (160, 120));
}

/// Stills: a phone's JPEG, its EXIF gone from every format by default, and
/// what is kept written as a fresh EXIF block each format's readers find.
#[cfg(feature = "image")]
mod stills {
    use super::*;
    use crate::image::{ImageFormat, ImageSpec, run_image_job};

    fn phone_jpeg() -> bytes::Bytes {
        let img = image::RgbImage::from_fn(64, 48, |x, y| image::Rgb([(x * 4) as u8, (y * 5) as u8, 128]));
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut jpeg).encode_image(&img).unwrap();
        let mut phone = identifying();
        phone.device.serial = Some("F2LXK0Q1".into());
        let tiff = metadata::exif::build(&phone).unwrap();
        metadata::write::still(&jpeg, &tiff, 64, 48).unwrap().into()
    }

    const ALL: [ImageFormat; 4] = [ImageFormat::Jpeg, ImageFormat::Png, ImageFormat::Webp, ImageFormat::Avif];

    #[test]
    fn a_phones_photo_loses_its_exif_in_every_format() {
        let src = phone_jpeg();
        assert_eq!(metadata::read(&src).categories(), Categories::ALL);
        for lossless in [false, true] {
            let formats: Vec<_> = if lossless { vec![ImageFormat::Webp, ImageFormat::Png] } else { ALL.to_vec() };
            let out = run_image_job(&src, &ImageSpec { formats, lossless, ..ImageSpec::default() }).unwrap();
            for a in &out.artifacts {
                let m = metadata::read(&a.bytes);
                assert!(m.is_empty(), "{:?} lossless={lossless}: {m:?}", a.format);
            }
        }
    }

    #[test]
    fn kept_categories_arrive_in_every_format_and_the_files_still_decode() {
        let src = phone_jpeg();
        let keep = Categories::NONE.with(Category::Location).with(Category::Device);
        for lossless in [false, true] {
            let formats: Vec<_> = if lossless { vec![ImageFormat::Webp, ImageFormat::Png] } else { ALL.to_vec() };
            let spec = ImageSpec { formats, lossless, metadata_keep: keep, ..ImageSpec::default() };
            let out = run_image_job(&src, &spec).unwrap();
            for a in &out.artifacts {
                let m = metadata::read(&a.bytes);
                assert_eq!(m.categories(), keep, "{:?} lossless={lossless}: {m:?}", a.format);
                let loc = m.location.clone().unwrap();
                assert!((loc.latitude.unwrap() - 37.3349).abs() < 1e-4, "{:?}", a.format);
                assert_eq!(m.device.serial.as_deref(), Some("F2LXK0Q1"), "a still carries serials in EXIF");
                // Still a picture: decoded again, at its size.
                let again = run_image_job(&a.bytes.clone().into(), &ImageSpec { formats: vec![ImageFormat::Png], ..ImageSpec::default() })
                    .unwrap_or_else(|e| panic!("{:?} lossless={lossless} no longer decodes: {e:#}", a.format));
                assert_eq!((again.artifacts[0].width, again.artifacts[0].height), (a.width, a.height), "{:?}", a.format);
            }
        }
    }

    #[test]
    fn metadata_keep_reaches_an_image_spec() {
        let spec = TranscodeSettings::parse_kv_line("mode=image image-format=jpeg metadata-keep=capture_time").unwrap().into_image_spec().unwrap();
        assert_eq!(spec.metadata_keep, Categories::NONE.with(Category::CaptureTime));
    }
}
