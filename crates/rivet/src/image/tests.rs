use std::io::Cursor;

use bytes::Bytes;
use image::{ImageEncoder, Rgba, RgbaImage};

use super::*;

const RED: [u8; 3] = [255, 0, 0];

/// A `w x h` gradient with a red block in its top-left corner, so a turn or a
/// flip shows where that corner went.
fn picture(w: u32, h: u32) -> RgbaImage {
    RgbaImage::from_fn(w, h, |x, y| {
        if x < w / 4 && y < h / 4 {
            Rgba([RED[0], RED[1], RED[2], 255])
        } else {
            Rgba([0, (x * 255 / w) as u8, (y * 255 / h) as u8, 255])
        }
    })
}

fn rgb_of(img: &RgbaImage) -> Vec<u8> {
    img.pixels().flat_map(|p| [p.0[0], p.0[1], p.0[2]]).collect()
}

fn png_bytes(img: &RgbaImage) -> Vec<u8> {
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out)
        .write_image(img.as_raw(), img.width(), img.height(), image::ExtendedColorType::Rgba8)
        .unwrap();
    out
}

fn jpeg_bytes(img: &RgbaImage, app1: Option<&[u8]>) -> Vec<u8> {
    let mut out = Vec::new();
    let mut enc = jpeg_encoder::Encoder::new(&mut out, 95);
    if let Some(app1) = app1 {
        enc.add_app_segment(1, app1.to_vec()).unwrap();
    }
    enc.encode(&rgb_of(img), img.width() as u16, img.height() as u16, jpeg_encoder::ColorType::Rgb).unwrap();
    out
}

/// Every raster input format this module reads, made by the `image` crate's
/// own encoders.
fn raster_inputs(img: &RgbaImage) -> Vec<(SourceFormat, Vec<u8>)> {
    let (w, h) = img.dimensions();
    let rgb = rgb_of(img);
    let mut gif = Vec::new();
    image::codecs::gif::GifEncoder::new(&mut gif).encode(img.as_raw(), w, h, image::ExtendedColorType::Rgba8).unwrap();
    let mut tiff = Cursor::new(Vec::new());
    image::codecs::tiff::TiffEncoder::new(&mut tiff).write_image(&rgb, w, h, image::ExtendedColorType::Rgb8).unwrap();
    let mut bmp = Vec::new();
    image::codecs::bmp::BmpEncoder::new(&mut bmp).write_image(&rgb, w, h, image::ExtendedColorType::Rgb8).unwrap();
    let mut webp = Vec::new();
    image::codecs::webp::WebPEncoder::new_lossless(&mut webp)
        .write_image(img.as_raw(), w, h, image::ExtendedColorType::Rgba8)
        .unwrap();
    vec![
        (SourceFormat::Png, png_bytes(img)),
        (SourceFormat::Jpeg, jpeg_bytes(img, None)),
        (SourceFormat::Gif, gif),
        (SourceFormat::Tiff, tiff.into_inner()),
        (SourceFormat::Bmp, bmp),
        (SourceFormat::Webp, webp),
    ]
}

fn near(a: [u8; 3], b: [u8; 3], tolerance: u8) -> bool {
    a.iter().zip(b).all(|(x, y)| x.abs_diff(y) <= tolerance)
}

fn px(img: &RgbaImage, x: u32, y: u32) -> [u8; 3] {
    let p = img.get_pixel(x, y).0;
    [p[0], p[1], p[2]]
}

fn run(input: Vec<u8>, spec: &ImageSpec) -> Result<ImageJobOutput> {
    run_image_job(&Bytes::from(input), spec)
}

fn spec_of(formats: &[ImageFormat]) -> ImageSpec {
    ImageSpec { formats: formats.to_vec(), ..ImageSpec::default() }
}

/// Decode an output with the `image` crate — an independent reader, not ours.
fn read_back(bytes: &[u8]) -> RgbaImage {
    image::load_from_memory(bytes).expect("the output decodes").to_rgba8()
}

#[test]
fn every_raster_input_is_sniffed_probed_and_decoded() {
    let img = picture(64, 48);
    for (format, bytes) in raster_inputs(&img) {
        assert_eq!(sniff(&bytes), Some(format), "{format}");
        let info = probe(&bytes).unwrap().expect("an image");
        assert_eq!((info.container.as_str(), info.width, info.height), (format.label(), 64, 48), "{format}");
        assert_eq!(info.duration, 0.0);
        let picture = decode::decode(&bytes, format).unwrap();
        assert_eq!(picture.rgba.dimensions(), (64, 48), "{format}");
        // GIF is palettised and JPEG lossy; both keep a red corner red.
        assert!(near(px(&picture.rgba, 2, 2), RED, 40), "{format}: {:?}", px(&picture.rgba, 2, 2));
        assert!(!picture.alpha, "{format}");
    }
}

#[test]
fn video_and_audio_are_not_images() {
    assert_eq!(sniff(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41"), None);
    assert_eq!(sniff(b"\x1a\x45\xdf\xa3\x00\x00\x00\x00\x00\x00\x00\x00"), None);
    assert_eq!(sniff(b"ID3\x04\x00\x00\x00\x00\x00\x00"), None);
    // `BM` alone is not a bitmap.
    assert_eq!(sniff(b"BMthis is plain text, not a picture"), None);
}

/// An EXIF APP1 segment holding one orientation tag, and some text a GPS tag
/// could be.
fn exif(orientation: u16) -> Vec<u8> {
    let mut e = b"Exif\0\0".to_vec();
    // Little-endian TIFF header, IFD at 8.
    e.extend_from_slice(b"II*\0\x08\0\0\0");
    // One entry: 0x0112 Orientation, SHORT, count 1, value.
    e.extend_from_slice(&1u16.to_le_bytes());
    e.extend_from_slice(&0x0112u16.to_le_bytes());
    e.extend_from_slice(&3u16.to_le_bytes());
    e.extend_from_slice(&1u32.to_le_bytes());
    e.extend_from_slice(&orientation.to_le_bytes());
    e.extend_from_slice(&[0, 0]);
    e.extend_from_slice(&0u32.to_le_bytes());
    e.extend_from_slice(b"GPS 51.5007N 0.1246W SECRET-SERIAL-123");
    e
}

#[test]
fn exif_orientation_is_applied_and_every_output_is_upright() {
    // 6: the camera was turned a quarter clockwise; shown upright, the stored
    // top-left corner is the top-right.
    let jpeg = jpeg_bytes(&picture(64, 48), Some(&exif(6)));
    let info = probe(&jpeg).unwrap().unwrap();
    assert_eq!((info.width, info.height, info.stored_width, info.stored_height), (48, 64, 64, 48));

    let out = run(jpeg, &spec_of(&[ImageFormat::Png])).unwrap();
    let back = read_back(&out.artifacts[0].bytes);
    assert_eq!(back.dimensions(), (48, 64));
    assert!(near(px(&back, 45, 2), RED, 40), "the red corner is top-right: {:?}", px(&back, 45, 2));
    assert!(!near(px(&back, 2, 2), RED, 60), "and no longer top-left");
}

#[test]
fn metadata_never_reaches_an_output() {
    let jpeg = jpeg_bytes(&picture(64, 48), Some(&exif(1)));
    assert!(jpeg.windows(6).any(|w| w == b"SECRET"), "the fixture carries it");
    let out = run(jpeg, &spec_of(&[ImageFormat::Jpeg, ImageFormat::Png, ImageFormat::Webp, ImageFormat::Avif])).unwrap();
    for a in &out.artifacts {
        assert!(!a.bytes.windows(6).any(|w| w == b"SECRET"), "{}", a.format);
        assert!(!a.bytes.windows(4).any(|w| w == b"Exif"), "{}: an EXIF segment", a.format);
    }
}

#[test]
fn every_output_format_is_what_it_says() {
    let out = run(png_bytes(&picture(64, 48)), &spec_of(&ImageFormat::ALL)).unwrap();
    assert_eq!(out.artifacts.len(), 4);
    for a in &out.artifacts {
        assert_eq!((a.width, a.height, a.label.as_str()), (64, 48, "64x48"));
        assert_eq!(a.file_name(false), format!("64x48.{}", a.format.extension()));
        match a.format {
            ImageFormat::Avif => {
                assert_eq!(sniff(&a.bytes), Some(SourceFormat::Avif));
                assert!(a.bytes.windows(4).any(|w| w == b"avif"), "an avif brand");
            }
            ImageFormat::Webp | ImageFormat::Jpeg | ImageFormat::Png => {
                let back = read_back(&a.bytes);
                assert_eq!(back.dimensions(), (64, 48), "{}", a.format);
                assert!(near(px(&back, 2, 2), RED, 40), "{}: {:?}", a.format, px(&back, 2, 2));
            }
        }
    }
    assert!(!out.from_video && !out.several_frames);
    assert_eq!(out.decoded, "png");
}

#[test]
fn lossless_webp_and_png_give_back_every_pixel() {
    let img = picture(37, 23);
    let spec = ImageSpec { lossless: true, ..spec_of(&[ImageFormat::Webp, ImageFormat::Png]) };
    let out = run(png_bytes(&img), &spec).unwrap();
    for a in &out.artifacts {
        assert_eq!(rgb_of(&read_back(&a.bytes)), rgb_of(&img), "{}", a.format);
    }
}

#[test]
fn quality_changes_the_lossy_formats() {
    let img = picture(256, 192);
    for format in [ImageFormat::Jpeg, ImageFormat::Webp, ImageFormat::Avif] {
        let low = run(png_bytes(&img), &ImageSpec { quality: Some(10), ..spec_of(&[format]) }).unwrap();
        let high = run(png_bytes(&img), &ImageSpec { quality: Some(95), ..spec_of(&[format]) }).unwrap();
        assert!(low.artifacts[0].bytes.len() < high.artifacts[0].bytes.len(), "{format}");
    }
}

#[test]
fn renditions_are_fitted_as_video_rungs_are_but_to_the_pixel() {
    let spec = ImageSpec {
        renditions: vec![
            ImageRendition::new(200, 200),
            ImageRendition { fit: Some(Fit::Cover), ..ImageRendition::new(100, 100) },
            ImageRendition { fit: Some(Fit::Pad), ..ImageRendition::new(300, 300) },
            ImageRendition { fit: Some(Fit::Stretch), ..ImageRendition::new(50, 70) },
            // Larger than the picture: its own size, odd sides and all.
            ImageRendition::new(1000, 1000),
            // A portrait box turns to the landscape picture.
            ImageRendition::new(100, 300),
        ],
        ..spec_of(&[ImageFormat::Png])
    };
    let out = run(png_bytes(&picture(401, 301)), &spec).unwrap();
    let sizes: Vec<(u32, u32)> = out.artifacts.iter().map(|a| (a.width, a.height)).collect();
    assert_eq!(sizes, vec![(200, 150), (100, 100), (300, 300), (50, 70), (401, 301), (133, 100)]);
    let labels: Vec<&str> = out.artifacts.iter().map(|a| a.label.as_str()).collect();
    assert_eq!(labels, vec!["200x150", "100x100", "300x300", "50x70", "401x301", "133x100"]);
    assert_eq!(out.artifacts.iter().map(|a| a.rendition).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 5]);

    // Pad: black bars above and below a 300x225 picture.
    let padded = read_back(&out.artifacts[2].bytes);
    assert_eq!(px(&padded, 150, 5), [0, 0, 0]);
    assert!(near(px(&padded, 5, 45), RED, 40), "the picture starts below the bar: {:?}", px(&padded, 5, 45));
}

#[test]
fn renditions_a_small_picture_collapses_are_made_once() {
    let spec = ImageSpec {
        renditions: vec![ImageRendition::new(1000, 1000), ImageRendition::new(2000, 2000), ImageRendition::new(64, 64)],
        ..spec_of(&[ImageFormat::Png])
    };
    let out = run(png_bytes(&picture(120, 90)), &spec).unwrap();
    assert_eq!(out.artifacts.iter().map(|a| a.rendition).collect::<Vec<_>>(), vec![0, 2]);
    assert_eq!(out.merged, vec![MergedRendition { rendition: 1, same_as: 0, output: (120, 90) }]);

    // Upscale makes both.
    let up = run(png_bytes(&picture(120, 90)), &ImageSpec { upscale: true, ..spec }).unwrap();
    assert_eq!(up.artifacts.len(), 3);
    assert_eq!((up.artifacts[1].width, up.artifacts[1].height), (2000, 1500));
}

#[test]
fn transparency_is_kept_where_the_format_holds_it_and_flattened_onto_white_where_not() {
    let mut img = picture(40, 30);
    for (x, _, p) in img.enumerate_pixels_mut() {
        if x >= 20 {
            p.0 = [0, 0, 0, 0];
        }
    }
    let out = run(png_bytes(&img), &spec_of(&[ImageFormat::Png, ImageFormat::Webp, ImageFormat::Jpeg])).unwrap();
    for a in &out.artifacts {
        let back = read_back(&a.bytes);
        let right = back.get_pixel(30, 15).0;
        match a.format {
            ImageFormat::Jpeg => assert!(near([right[0], right[1], right[2]], [255, 255, 255], 8), "{right:?}"),
            _ => assert_eq!(right[3], 0, "{}", a.format),
        }
    }
}

#[test]
fn a_tagged_picture_is_converted_to_srgb_unless_its_profile_is_kept() {
    // A Display P3 picture: the same numbers mean a more saturated colour
    // than in sRGB, so converting has to change them.
    let p3 = moxcms::ColorProfile::new_display_p3().encode().unwrap();
    let img = RgbaImage::from_pixel(16, 16, Rgba([200, 60, 40, 255]));
    let mut tagged = Vec::new();
    let mut enc = image::codecs::png::PngEncoder::new(&mut tagged);
    enc.set_icc_profile(p3.clone()).unwrap();
    enc.write_image(img.as_raw(), 16, 16, image::ExtendedColorType::Rgba8).unwrap();

    let converted = run(tagged.clone(), &spec_of(&[ImageFormat::Png])).unwrap();
    let bytes = &converted.artifacts[0].bytes;
    assert!(!bytes.windows(4).any(|w| w == b"iCCP"), "converted output is untagged sRGB");
    let c = px(&read_back(bytes), 8, 8);
    assert_ne!(c, [200, 60, 40], "the colour was converted");
    assert!(c[0] > 200, "P3 red is redder than the same numbers in sRGB: {c:?}");

    let kept = run(tagged, &ImageSpec { keep_icc: true, ..spec_of(&[ImageFormat::Png, ImageFormat::Jpeg, ImageFormat::Webp]) })
        .unwrap();
    for a in &kept.artifacts {
        let marker: &[u8] = match a.format {
            ImageFormat::Png => b"iCCP",
            ImageFormat::Jpeg => b"ICC_PROFILE",
            _ => b"ICCP",
        };
        assert!(a.bytes.windows(marker.len()).any(|w| w == marker), "{} carries the profile", a.format);
    }
    assert_eq!(px(&read_back(&kept.artifacts[0].bytes), 8, 8), [200, 60, 40], "and its pixels as they were");
}

#[test]
fn a_spec_that_cannot_mean_anything_is_refused() {
    let refused = [
        ImageSpec { formats: vec![], ..ImageSpec::default() },
        spec_of(&[ImageFormat::Png, ImageFormat::Png]),
        ImageSpec { quality: Some(80), ..spec_of(&[ImageFormat::Png]) },
        ImageSpec { quality: Some(0), ..spec_of(&[ImageFormat::Jpeg]) },
        ImageSpec { quality: Some(80), lossless: true, ..spec_of(&[ImageFormat::Webp]) },
        ImageSpec { lossless: true, ..spec_of(&[ImageFormat::Jpeg]) },
        ImageSpec { speed: 0, ..ImageSpec::default() },
        ImageSpec { renditions: vec![ImageRendition::new(0, 10)], ..ImageSpec::default() },
        ImageSpec { renditions: vec![ImageRendition::new(20_000, 10)], ..ImageSpec::default() },
        ImageSpec { frames: Some(FrameSelection::Count(0)), ..ImageSpec::default() },
        ImageSpec { frames: Some(FrameSelection::At(vec![-1.0])), ..ImageSpec::default() },
        ImageSpec { frames: Some(FrameSelection::At(vec![])), ..ImageSpec::default() },
    ];
    for spec in refused {
        let err = spec.validate().expect_err(&format!("{spec:?}"));
        assert!(err.to_string().starts_with("invalid output spec"), "{err}");
    }
}

#[test]
fn frames_on_a_still_image_are_refused() {
    let spec = ImageSpec { frames: Some(FrameSelection::Count(3)), ..ImageSpec::default() };
    let err = run(png_bytes(&picture(32, 32)), &spec).unwrap_err();
    assert!(err.to_string().contains("frames pick stills from a video"), "{err}");
}

#[test]
fn an_oversized_picture_is_refused_from_its_header() {
    // A PNG header declaring 20000x20000: over the limit before a byte of it
    // is decoded.
    let mut png = png_bytes(&picture(8, 8));
    png[16..20].copy_from_slice(&20_000u32.to_be_bytes());
    png[20..24].copy_from_slice(&20_000u32.to_be_bytes());
    // The IHDR's CRC covers its type and data.
    let crc = png[12..29].iter().fold(!0u32, |mut c, &b| {
        c ^= u32::from(b);
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 };
        }
        c
    });
    png[29..33].copy_from_slice(&(!crc).to_be_bytes());
    let err = run(png, &ImageSpec::default()).unwrap_err();
    assert!(format!("{err:#}").contains("megapixel limit"), "{err:#}");
}

#[test]
fn a_denied_format_is_refused_by_the_setting_s_name() {
    let heic = include_bytes!("testdata/small.heic").to_vec();
    let spec = ImageSpec { decode_deny: ImageDecodeDeny::parse("heic").unwrap(), ..spec_of(&[ImageFormat::Png]) };
    let err = run(heic, &spec).unwrap_err();
    assert_eq!(err.to_string(), "decoding heic images is denied by the image-decode-deny setting");
    // Only what is named.
    run(png_bytes(&picture(16, 16)), &spec).unwrap();
    assert!(ImageDecodeDeny::parse("heic,webm").is_err());
    assert_eq!(ImageDecodeDeny::parse("").unwrap(), ImageDecodeDeny::default());
    assert_eq!(ImageDecodeDeny::parse("heif,HEIC").unwrap().0, vec![SourceFormat::Heic]);
}

#[test]
fn heic_is_decoded_by_the_hevc_decoder() {
    let heic = include_bytes!("testdata/small.heic");
    assert_eq!(sniff(heic), Some(SourceFormat::Heic));
    let info = probe(heic).unwrap().unwrap();
    assert_eq!((info.container.as_str(), info.video_codec.as_str()), ("heic", "hevc"));
    assert_eq!((info.width, info.height), (64, 48));

    let out = run(heic.to_vec(), &spec_of(&[ImageFormat::Png])).unwrap();
    assert_eq!(out.decoded, "hevc");
    let back = read_back(&out.artifacts[0].bytes);
    assert_eq!(back.dimensions(), (64, 48));
    assert!(near(px(&back, 3, 3), RED, 48), "{:?}", px(&back, 3, 3));
    // The gradient's far corner is not red.
    assert!(!near(px(&back, 60, 44), RED, 80));
}

/// An AVIF test needs an AV1 decoder: a GPU's, or rav1d with
/// `rav1d-fallback`. A build with neither cannot run it.
fn av1_decodable(result: &Result<ImageJobOutput>) -> bool {
    match result {
        Err(e) if format!("{e:#}").contains("no decoder available") => {
            eprintln!("SKIP: no AV1 decoder in this build ({e:#})");
            false
        }
        _ => true,
    }
}

#[test]
fn avif_is_decoded_by_the_av1_decoder_at_every_layout_and_depth() {
    for (name, bytes, size) in [
        ("4:2:0 10-bit, odd-sized (avifenc)", &include_bytes!("testdata/odd_420_10bit.avif")[..], (65, 49)),
        ("4:4:4 8-bit (avifenc)", &include_bytes!("testdata/small_444.avif")[..], (64, 48)),
    ] {
        assert_eq!(sniff(bytes), Some(SourceFormat::Avif), "{name}");
        let info = probe(bytes).unwrap().unwrap();
        assert_eq!((info.video_codec.as_str(), info.width, info.height), ("av1", size.0, size.1), "{name}");
        let result = run(bytes.to_vec(), &spec_of(&[ImageFormat::Png]));
        if !av1_decodable(&result) {
            return;
        }
        let back = read_back(&result.unwrap_or_else(|e| panic!("{name}: {e:#}")).artifacts[0].bytes);
        assert_eq!(back.dimensions(), size, "{name}");
        assert!(near(px(&back, 3, 3), RED, 48), "{name}: {:?}", px(&back, 3, 3));
    }
}

#[test]
fn an_avif_we_wrote_comes_back_with_its_transparency() {
    let mut img = picture(48, 32);
    for (x, _, p) in img.enumerate_pixels_mut() {
        if x >= 24 {
            p.0[3] = 0;
        }
    }
    let avif = run(png_bytes(&img), &ImageSpec { quality: Some(95), ..spec_of(&[ImageFormat::Avif]) }).unwrap();
    let result = run(avif.artifacts[0].bytes.clone(), &spec_of(&[ImageFormat::Png]));
    if !av1_decodable(&result) {
        return;
    }
    let back = read_back(&result.unwrap().artifacts[0].bytes);
    assert_eq!(back.dimensions(), (48, 32));
    assert!(back.get_pixel(40, 16).0[3] < 16, "transparent stays transparent");
    assert!(back.get_pixel(4, 4).0[3] > 240, "opaque stays opaque");
    assert!(near(px(&back, 3, 3), RED, 40), "{:?}", px(&back, 3, 3));
}

/// `RIVET_TEST_MEDIA/stills_clip.mp4`: any short H.264 clip, e.g.
/// `ffmpeg -f lavfi -i testsrc2=size=320x240:rate=25 -t 4 -c:v libx264
/// -pix_fmt yuv420p stills_clip.mp4`. Skipped when absent.
fn stills_clip() -> Option<Vec<u8>> {
    let dir = std::env::var_os("RIVET_TEST_MEDIA")?;
    std::fs::read(std::path::Path::new(&dir).join("stills_clip.mp4")).ok()
}

#[test]
fn stills_are_taken_from_a_video() {
    let Some(clip) = stills_clip() else {
        eprintln!("SKIP: RIVET_TEST_MEDIA/stills_clip.mp4 not present");
        return;
    };
    let spec = ImageSpec {
        frames: Some(FrameSelection::Count(4)),
        renditions: vec![ImageRendition::new(160, 160)],
        ..spec_of(&[ImageFormat::Jpeg, ImageFormat::Avif])
    };
    let out = run(clip.clone(), &spec).unwrap();
    assert!(out.from_video && out.several_frames);
    assert_eq!(out.decoded, "h264");
    assert_eq!(out.artifacts.len(), 8);
    let times: Vec<f64> = out.artifacts.iter().step_by(2).map(|a| a.frame.unwrap().1).collect();
    assert!(times.windows(2).all(|w| w[0] < w[1]), "evenly spaced and in order: {times:?}");
    assert!(times[0] > 0.0 && *times.last().unwrap() < 4.0, "{times:?}");
    assert_eq!(out.artifacts[0].file_name(true), "160x120-001.jpg");
    assert_eq!(out.artifacts[7].file_name(true), "160x120-004.avif");
    assert_eq!(read_back(&out.artifacts[0].bytes).dimensions(), (160, 120));

    // The poster: one frame, 10% in.
    let poster = run(clip.clone(), &spec_of(&[ImageFormat::Webp])).unwrap();
    assert_eq!(poster.artifacts.len(), 1);
    assert!(!poster.several_frames);
    let t = poster.artifacts[0].frame.unwrap().1;
    assert!((0.3..0.5).contains(&t), "{t}");
    assert_eq!(poster.artifacts[0].file_name(false), "320x240.webp");

    let at = run(clip.clone(), &ImageSpec { frames: Some(FrameSelection::At(vec![1.0, 2.0])), ..spec_of(&[ImageFormat::Png]) })
        .unwrap();
    let times: Vec<f64> = at.artifacts.iter().map(|a| a.frame.unwrap().1).collect();
    assert_eq!(times, vec![1.0, 2.0]);

    let err = run(clip, &ImageSpec { frames: Some(FrameSelection::At(vec![60.0])), ..spec_of(&[ImageFormat::Png]) })
        .unwrap_err();
    assert!(err.to_string().contains("past the end of the video"), "{err}");
}

#[test]
fn settings_build_an_image_spec_and_keep_the_modes_apart() {
    let mut s = crate::TranscodeSettings::default();
    for (k, v) in [
        ("mode", "image"),
        ("image-format", "avif,jpg"),
        ("rung", "1920x1920,640x640:cover"),
        ("image-quality", "70"),
        ("frames-count", "6"),
        ("image-decode-deny", "heic"),
        ("audio-decode-deny", "aac"),
    ] {
        s.apply_kv(k, v).unwrap();
    }
    let spec = s.into_image_spec().unwrap();
    assert_eq!(spec.formats, vec![ImageFormat::Avif, ImageFormat::Jpeg]);
    assert_eq!(spec.renditions.len(), 2);
    assert_eq!(spec.renditions[1].fit, Some(Fit::Cover));
    assert_eq!(spec.quality, Some(70));
    assert_eq!(spec.frames, Some(FrameSelection::Count(6)));
    assert!(spec.decode_deny.denies(SourceFormat::Heic));

    // A video knob in an image job, and an image knob in a video job.
    let mut video_knob = crate::TranscodeSettings::default();
    video_knob.apply_kv("mode", "image").unwrap();
    video_knob.apply_kv("codec", "h264").unwrap();
    assert!(video_knob.into_image_spec().unwrap_err().to_string().contains("`codec`"));

    let mut image_knob = crate::TranscodeSettings::default();
    image_knob.apply_kv("image-format", "png").unwrap();
    assert!(image_knob.clone().into_spec(1280, 720).unwrap_err().to_string().contains("image-format"));

    // `image-decode-deny` rides along on a video job, as `audio-decode-deny` does.
    let mut deny = crate::TranscodeSettings::default();
    deny.apply_kv("image-decode-deny", "heic").unwrap();
    deny.into_spec(1280, 720).unwrap();

    let mut image_mode = crate::TranscodeSettings::default();
    image_mode.apply_kv("mode", "image").unwrap();
    assert!(image_mode.into_spec(1280, 720).is_err(), "run_job does not make images");
}

#[test]
fn place_aligned_to_one_pixel_keeps_odd_sizes_and_to_two_is_place() {
    use crate::fit::{SourceShape, place, place_aligned};
    let shape = SourceShape::square(641, 481);
    let one = place_aligned(shape, (1000, 1000), Fit::Contain, Orientation::Auto, false, 1);
    assert_eq!(one.canvas, (641, 481));
    let two = place_aligned(shape, (1000, 1000), Fit::Contain, Orientation::Auto, false, 2);
    assert_eq!(two, place(shape, (1000, 1000), Fit::Contain, Orientation::Auto, false));
    assert_eq!(two.canvas, (640, 480));
}
