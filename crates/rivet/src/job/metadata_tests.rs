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

/// A short synthetic H.264 clip (`crate::synth`), dressed as a phone's
/// recording.
fn phone_clip() -> Option<Vec<u8>> {
    let clip = crate::synth::clip(320, 240, 25, 4.0, 0, 0, false);
    Some(metadata::write::mp4(&clip, &identifying()).unwrap())
}

#[test]
fn a_phones_video_comes_out_clean_unless_asked() {
    let Some(src) = phone_clip() else { return };
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
        let policy = container::metadata::Keep::parse("location,device:all").unwrap();
        let keep = policy.categories();
        for lossless in [false, true] {
            let formats: Vec<_> = if lossless { vec![ImageFormat::Webp, ImageFormat::Png] } else { ALL.to_vec() };
            let spec = ImageSpec { formats, lossless, metadata_keep: policy, ..ImageSpec::default() };
            let out = run_image_job(&src, &spec).unwrap();
            for a in &out.artifacts {
                let m = metadata::read(&a.bytes);
                assert_eq!(m.categories(), keep, "{:?} lossless={lossless}: {m:?}", a.format);
                let loc = m.location.clone().unwrap();
                assert!((loc.latitude.unwrap() - 37.3349).abs() < 1e-4, "{:?}", a.format);
                assert_eq!(m.device.serial.as_deref(), Some("F2LXK0Q1"), "a still carries serials in EXIF");
                // Still a picture: decoded again, at its size (AVIF where this
                // build decodes AV1 in software).
                if a.format == ImageFormat::Avif && !cfg!(feature = "rav1d-fallback") {
                    continue;
                }
                let again = run_image_job(&a.bytes.clone().into(), &ImageSpec { formats: vec![ImageFormat::Png], ..ImageSpec::default() })
                    .unwrap_or_else(|e| panic!("{:?} lossless={lossless} no longer decodes: {e:#}", a.format));
                assert_eq!((again.artifacts[0].width, again.artifacts[0].height), (a.width, a.height), "{:?}", a.format);
            }
        }
    }

    /// A kept colour profile and kept EXIF together: WebP's extended header
    /// is already there for the profile, so the EXIF flag goes on it (one
    /// `VP8X`, first), and every format still carries both and still decodes
    /// with an independent reader.
    #[test]
    fn a_kept_profile_and_kept_exif_live_together() {
        use image::ImageEncoder;
        let p3 = moxcms::ColorProfile::new_display_p3().encode().unwrap();
        let img = image::RgbaImage::from_pixel(16, 16, image::Rgba([200, 60, 40, 255]));
        let mut png = Vec::new();
        let mut enc = image::codecs::png::PngEncoder::new(&mut png);
        enc.set_icc_profile(p3).unwrap();
        enc.write_image(img.as_raw(), 16, 16, image::ExtendedColorType::Rgba8).unwrap();
        let tiff = metadata::exif::build(&identifying()).unwrap();
        let src = bytes::Bytes::from(metadata::write::still(&png, &tiff, 16, 16).unwrap());
        let policy = container::metadata::Keep::parse("location").unwrap();
        for lossless in [false, true] {
            let formats = if lossless { vec![ImageFormat::Webp, ImageFormat::Png] } else { vec![ImageFormat::Webp, ImageFormat::Png, ImageFormat::Jpeg] };
            let spec = ImageSpec { formats, lossless, keep_icc: true, metadata_keep: policy, ..ImageSpec::default() };
            for a in run_image_job(&src, &spec).unwrap().artifacts {
                let has = |m: &[u8]| a.bytes.windows(m.len()).filter(|w| *w == m).count();
                let profile: &[u8] = match a.format {
                    ImageFormat::Png => b"iCCP",
                    ImageFormat::Jpeg => b"ICC_PROFILE",
                    _ => b"ICCP",
                };
                assert!(has(profile) >= 1, "{:?} lossless={lossless} lost its profile", a.format);
                if a.format == ImageFormat::Webp {
                    assert_eq!(has(b"VP8X"), 1, "one extended header");
                    assert_eq!(&a.bytes[12..16], b"VP8X", "and it comes first");
                    assert_eq!(a.bytes[20] & 0x28, 0x28, "ICC and EXIF flags both set");
                }
                assert_eq!(metadata::read(&a.bytes).categories(), policy.categories(), "{:?}", a.format);
                let back = image::load_from_memory(&a.bytes).unwrap_or_else(|e| panic!("{:?} lossless={lossless}: {e}", a.format));
                assert_eq!((back.width(), back.height()), (16, 16));
            }
        }
    }

    /// `device` keeps what made the picture, not whose camera it was:
    /// make, model, software and lens, and no body or lens serial number or
    /// owner name, in any format. `device:all` writes those too.
    #[test]
    fn device_keep_leaves_out_serials_and_owner_and_device_all_writes_them() {
        let img = image::RgbImage::from_fn(32, 24, |x, y| image::Rgb([(x * 8) as u8, (y * 10) as u8, 90]));
        let mut jpeg = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut jpeg).encode_image(&img).unwrap();
        let mut phone = identifying();
        phone.device.lens = Some("iPhone 15 Pro back camera 6.765mm f/1.78".into());
        phone.device.serial = Some("F2LXK0Q1".into());
        phone.device.owner = Some("Ada Lovelace".into());
        let src: bytes::Bytes = metadata::write::still(&jpeg, &metadata::exif::build(&phone).unwrap(), 32, 24).unwrap().into();
        let source = metadata::read(&src);
        assert_eq!((source.device.serial.as_deref(), source.device.owner.as_deref()), (Some("F2LXK0Q1"), Some("Ada Lovelace")));

        for (policy, whole) in [("device", false), ("device:all", true)] {
            let keep = container::metadata::Keep::parse(policy).unwrap();
            let spec = ImageSpec { formats: ALL.to_vec(), metadata_keep: keep, ..ImageSpec::default() };
            for a in run_image_job(&src, &spec).unwrap().artifacts {
                let m = metadata::read(&a.bytes);
                let what = format!("{:?} with {policy}", a.format);
                assert_eq!(m.device.make.as_deref(), Some("Apple"), "{what}");
                assert_eq!(m.device.model.as_deref(), Some("iPhone 15 Pro"), "{what}");
                assert_eq!(m.device.software.as_deref(), Some("17.4.1"), "{what}");
                assert!(m.device.lens.is_some(), "{what}");
                let has = |text: &[u8]| a.bytes.windows(text.len()).any(|w| w == text);
                if whole {
                    assert_eq!(m.device.serial.as_deref(), Some("F2LXK0Q1"), "{what}");
                    assert_eq!(m.device.owner.as_deref(), Some("Ada Lovelace"), "{what}");
                } else {
                    assert!(m.device.serial.is_none() && m.device.owner.is_none(), "{what}: {:?}", m.device);
                    assert!(!has(b"F2LXK0Q1") && !has(b"Ada Lovelace"), "{what}: the bytes are not in the file at all");
                }
                assert!(m.violations(keep, &[]).is_empty(), "{what}: {:?}", m.violations(keep, &[]));
            }
        }
    }

    #[test]
    fn metadata_keep_reaches_an_image_spec() {
        let spec = TranscodeSettings::parse_kv_line("mode=image image-format=jpeg metadata-keep=capture_time").unwrap().into_image_spec().unwrap();
        assert_eq!(spec.metadata_keep, container::metadata::Keep::parse("capture_time").unwrap());
    }
}

/// A copied stream decoded to PCM, and its encoder names.
fn pcm_and_idents(file: &[u8]) -> (Vec<f32>, Vec<String>) {
    let src = container::streaming::demux_audio(bytes::Bytes::copy_from_slice(file)).unwrap().unwrap();
    let t = &src.track;
    let private = if t.codec == "aac" { &t.asc } else { &t.codec_private };
    let extra = (!private.is_empty()).then_some(private.as_slice());
    let mut dec = codec::audio::create_decoder(&t.codec, extra, t.sample_rate, t.channels as u8).unwrap();
    let mut pcm = Vec::new();
    for p in &t.samples {
        for f in dec.decode(p, 0).unwrap() {
            pcm.extend_from_slice(&f.samples);
        }
    }
    (pcm, metadata::read(file).embedded_software)
}

/// The 5.1 AAC fixture as another encoder would have left it: every access
/// unit opened by a fill element (`ID_FIL`, `EXT_FILL`) carrying an encoder
/// name, the way ffmpeg's AAC encoder writes `Lavc…` (see
/// `container::metadata::scrub`). rivet's encoder, which made the fixture,
/// writes no name; this puts one where a third-party stream has it.
fn aac_fixture() -> Vec<u8> {
    const NAME: &[u8] = b"Lavc61.19.100";
    let file = std::fs::read(format!("{}/tests/data/audio/tones_51_aac.m4a", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let src = container::streaming::demux_audio(bytes::Bytes::from(file)).unwrap().unwrap();
    let t = &src.track;
    let named = |au: &[u8]| -> Vec<u8> {
        // ID_FIL (6), count 15 + escape, then the payload: EXT_FILL (0),
        // five zero bits, the name, zeros to the count; then the original
        // elements, shifted by the 7 bits the header leaves over a byte.
        let cnt = NAME.len() + 2;
        let mut bits: Vec<bool> = Vec::new();
        let put = |v: u32, n: usize, bits: &mut Vec<bool>| (0..n).rev().for_each(|i| bits.push((v >> i) & 1 == 1));
        put(6, 3, &mut bits);
        put(15, 4, &mut bits);
        put((cnt - 14) as u32, 8, &mut bits);
        let payload_end = bits.len() + 8 * cnt;
        put(0, 9, &mut bits);
        NAME.iter().for_each(|&b| put(u32::from(b), 8, &mut bits));
        while bits.len() < payload_end {
            bits.push(false);
        }
        au.iter().for_each(|&b| put(u32::from(b), 8, &mut bits));
        bits.chunks(8).map(|c| c.iter().enumerate().fold(0u8, |v, (i, &b)| v | (u8::from(b) << (7 - i)))).collect()
    };
    let samples: Vec<(Vec<u8>, u32)> = t.samples.iter().zip(&t.durations).map(|(s, &d)| (named(s), d)).collect();
    let edit = src.edit.map_or_else(container::edit::TrackEdit::default, |e| container::edit::TrackEdit {
        delay: e.delay,
        media_time: e.media_start,
        duration: e.media_end.map(|end| end - e.media_start),
    });
    let info = container::AudioInfo::aac_lc(t.sample_rate, t.channels, t.asc.clone());
    container::mux::write_audio_mp4(&info, &samples, edit).unwrap()
}

#[test]
fn an_aac_copy_loses_its_encoder_name_and_decodes_the_same() {
    let src = aac_fixture();
    let (pcm, idents) = pcm_and_idents(&src);
    assert!(idents.iter().any(|i| i.starts_with("Lavc")), "the fixture names its encoder: {idents:?}");
    let out = run(&src, "mode=audio audio=aac audio-container=mp4").unwrap();
    assert!(out.audio_handling.starts_with("aac passthrough"), "{}", out.audio_handling);
    let (after, idents) = pcm_and_idents(file(&out));
    assert!(idents.is_empty(), "{idents:?}");
    assert_eq!(after, pcm, "the same audio, bit for bit");
    assert!(metadata::read(file(&out)).violations(Default::default(), &[]).is_empty());

    // Kept with the device: the stream as it was.
    let out = run(&src, "mode=audio audio=aac audio-container=mp4 metadata-keep=device").unwrap();
    let (_, idents) = pcm_and_idents(file(&out));
    assert!(idents.iter().any(|i| i.starts_with("Lavc")), "{idents:?}");
}

#[test]
fn an_mp3_copy_loses_its_encoder_names_keeps_its_gapless_edit_and_decodes_the_same() {
    // An MP3 as another encoder leaves it: rivet's frames behind a tag frame
    // naming `LAME3.100`, with the gapless delay and padding of rivet's own.
    let flac = native_flac(&signal(96_000, 2, 16), 2, 16);
    let ours = file(&run(&flac, "mode=audio audio=mp3").unwrap()).to_vec();
    let (track, edit) = container::mp3::read_file(&ours).unwrap();
    let edit = edit.expect("rivet's tag states its delay");
    let gapless = container::mp3::Gapless {
        encoder_delay: (edit.media_start - 529) as u32,
        samples: edit.media_end.unwrap() - edit.media_start,
    };
    let src = container::mp3::write_file(&track.samples, Some(gapless), Some("LAME3.100")).unwrap();
    let (pcm, idents) = pcm_and_idents(&src);
    assert!(idents.contains(&"LAME3.100".to_string()), "{idents:?}");
    let src_edit = container::streaming::demux_audio(bytes::Bytes::from(src.clone())).unwrap().unwrap().edit;

    let out = run(&src, "mode=audio audio=mp3").unwrap();
    assert!(out.audio_handling.starts_with("mp3 passthrough"), "{}", out.audio_handling);
    let bytes = file(&out);
    let (after, idents) = pcm_and_idents(bytes);
    assert!(idents.is_empty(), "the source's encoder name is gone: {idents:?}");
    assert_eq!(after, pcm, "the same audio, bit for bit");
    let out_edit = container::streaming::demux_audio(bytes::Bytes::copy_from_slice(bytes)).unwrap().unwrap().edit;
    assert_eq!(out_edit, src_edit, "the gapless delay and padding survive, under rivet's own name");
    assert!(metadata::read(bytes).violations(Default::default(), &[]).is_empty());
}
