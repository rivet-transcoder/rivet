//! The H.264 / H.265 sample entry each muxing path writes, on streams from the
//! software encoder: `avc1` / `hvc1` with every parameter set out of band in a
//! complete `avcC` / `hvcC`, the entry every player takes; `avc3` / `hev1`
//! only where the sets really change. Safari's `<video>` element on iOS
//! refuses `avc3` outright (`MEDIA_ERR_SRC_NOT_SUPPORTED`), so an `avc3` file
//! from one encoder is a file that does not play there.
//!
//! The hardware backends reach the same writers with the same Annex-B
//! packets; `container/tests/sample_entry.rs` covers the shapes they differ in
//! (delimiters, repeated headers).

use std::path::Path;

use codec::encode::{self, EncodedPacket, EncoderBackend, EncoderConfig, SpeedTier};
use codec::frame::{ColorSpace, PixelFormat, VideoCodec, VideoFrame};
use container::cmaf::{CmafVideoMuxer, CmafVideoMuxerOptions};
use container::nal_mux::split_annexb_nals;

use super::RungArtifact;
use super::pump::{build_video_variant_spec, settle_sample_entry};
use super::run::{encode_rung_single_file, mux_rung_packets_to_mp4};
use crate::multigpu::{RungManifest, RungPackets};
use crate::progress::NullSink;
use crate::spec::Rung;

const W: u32 = 128;
const H: u32 = 96;
/// Frames per chunk; one GOP each, as the chunked engine cuts them.
const CHUNK: u64 = 8;

/// A picture whose content moves with `pts`, at 8 or 10 bits.
fn frame(pts: u64, ten: bool) -> VideoFrame {
    let (w, h) = (W as usize, H as usize);
    let luma = |x: usize, y: usize| ((x + y + 3 * pts as usize) % 200 + 20) as u16;
    let samples: Vec<u16> = (0..h)
        .flat_map(|y| (0..w).map(move |x| luma(x, y)))
        .chain(std::iter::repeat_n(128, w * h / 2))
        .collect();
    let (data, format) = if ten {
        (samples.iter().flat_map(|s| (s << 2).to_le_bytes()).collect::<Vec<u8>>(), PixelFormat::Yuv420p10le)
    } else {
        (samples.iter().map(|&s| s as u8).collect(), PixelFormat::Yuv420p)
    };
    VideoFrame::new(data.into(), W, H, format, ColorSpace::Bt709, pts)
}

fn config(codec: VideoCodec, ten: bool) -> EncoderConfig {
    EncoderConfig {
        width: W,
        height: H,
        frame_rate: 30.0,
        keyframe_interval: CHUNK as u32,
        tier: SpeedTier::Draft,
        threads: 1,
        pixel_format: if ten { PixelFormat::Yuv420p10le } else { PixelFormat::Yuv420p },
        codec,
        ..EncoderConfig::default()
    }
}

/// One software-encoder session over `frames` — one chunk of a stitch.
fn software_chunk(codec: VideoCodec, frames: std::ops::Range<u64>, ten: bool) -> Vec<EncodedPacket> {
    let mut enc = encode::select_encoder(config(codec, ten), Some(EncoderBackend::H26x)).expect("the software encoder");
    let mut packets = Vec::new();
    for pts in frames {
        enc.send_frame(&frame(pts, ten)).unwrap();
        while let Some(p) = enc.receive_packet().unwrap() {
            packets.push(p);
        }
    }
    enc.flush().unwrap();
    while let Some(p) = enc.receive_packet().unwrap() {
        packets.push(p);
    }
    packets
}

fn is_param_set(nal: &[u8], codec: VideoCodec) -> bool {
    match codec {
        VideoCodec::H264 => matches!(nal[0] & 0x1F, 7 | 8),
        _ => matches!((nal[0] >> 1) & 0x3F, 32..=34),
    }
}

fn trimmed(nal: &[u8]) -> Vec<u8> {
    nal[..nal.len() - nal.iter().rev().take_while(|&&b| b == 0).count()].to_vec()
}

/// The distinct parameter sets the encoder wrote, sorted.
fn stream_sets(packets: &[EncodedPacket], codec: VideoCodec) -> Vec<Vec<u8>> {
    let mut sets: Vec<Vec<u8>> = packets
        .iter()
        .flat_map(|p| split_annexb_nals(&p.data).into_iter().map(trimmed).collect::<Vec<_>>())
        .filter(|n| !n.is_empty() && is_param_set(n, codec))
        .collect();
    sets.sort();
    sets.dedup();
    sets
}

/// The visual sample entry's fourcc and body, from `stsd`.
fn sample_entry(mp4: &[u8]) -> ([u8; 4], &[u8]) {
    let at = mp4.windows(4).position(|w| w == b"stsd").expect("an stsd box");
    // stsd type, version/flags, entry_count, then the entry's size + type.
    let entry = at + 4 + 8;
    let size = u32::from_be_bytes(mp4[entry..entry + 4].try_into().unwrap()) as usize;
    (mp4[entry + 4..entry + 8].try_into().unwrap(), &mp4[entry..entry + size])
}

/// The config box's parameter sets, sorted, and (for `hvcC`) every array's
/// `array_completeness` bit. Parsing it through is the check that it parses.
fn config_sets(mp4: &[u8], codec: VideoCodec) -> (Vec<Vec<u8>>, Vec<u8>) {
    let (_, entry) = sample_entry(mp4);
    let tag: &[u8; 4] = if codec == VideoCodec::H264 { b"avcC" } else { b"hvcC" };
    let at = entry.windows(4).position(|w| w == tag).expect("a config box");
    let size = u32::from_be_bytes(entry[at - 4..at].try_into().unwrap()) as usize;
    let body = &entry[at + 4..at - 4 + size];
    let mut sets = Vec::new();
    let mut complete = Vec::new();
    let mut take = |at: &mut usize| {
        let len = u16::from_be_bytes([body[*at], body[*at + 1]]) as usize;
        sets.push(trimmed(&body[*at + 2..*at + 2 + len]));
        *at += 2 + len;
    };
    let mut at = if codec == VideoCodec::H264 {
        assert_eq!(body[4] & 3, 3, "4-byte NAL lengths");
        let mut at = 6;
        for _ in 0..body[5] & 0x1F {
            take(&mut at);
        }
        let pps = body[at];
        at += 1;
        for _ in 0..pps {
            take(&mut at);
        }
        at
    } else {
        assert_eq!(body[21] & 3, 3, "4-byte NAL lengths");
        let mut at = 23;
        for _ in 0..body[22] {
            complete.push(body[at] >> 7);
            let n = u16::from_be_bytes([body[at + 1], body[at + 2]]);
            at += 3;
            for _ in 0..n {
                take(&mut at);
            }
        }
        at
    };
    if codec == VideoCodec::H264 && !matches!(body[1], 66 | 77 | 88) {
        at += 4; // High profiles: chroma_format, bit depths, numOfSequenceParameterSetExt
    }
    assert_eq!(at, body.len(), "the record ends where its fields do");
    sets.sort();
    (sets, complete)
}

/// Whether any sample in the file carries a parameter set in band.
fn samples_carry_sets(mp4: &[u8], codec: VideoCodec) -> bool {
    let mdat = mp4.windows(4).position(|w| w == b"mdat").expect("an mdat") + 4;
    let data = &mp4[mdat..];
    let mut at = 0;
    while at + 4 <= data.len() {
        let len = u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) as usize;
        if len == 0 || at + 4 + len > data.len() {
            break; // the next box (moov after mdat)
        }
        if is_param_set(&data[at + 4..at + 4 + len], codec) {
            return true;
        }
        at += 4 + len;
    }
    false
}

/// ffprobe's `codec_tag_string` for the video stream, and how many frames
/// ffprobe decoded from it — every one, only when the parameter sets the
/// sample entry describes are the ones the pictures were coded with. `None`
/// (said so) on a host without ffprobe.
fn ffprobe_tag(mp4: &[u8], name: &str) -> Option<(String, u64)> {
    let file = name.replace(' ', "-");
    let path = std::env::temp_dir().join(format!("rivet-sample-entry-{}-{file}.mp4", std::process::id()));
    std::fs::write(&path, mp4).unwrap();
    let out = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-count_frames", "-select_streams", "v:0"])
        .args(["-show_entries", "stream=codec_tag_string,nb_read_frames"])
        .args(["-of", "default=noprint_wrappers=1"])
        .arg(&path)
        .output();
    let _ = std::fs::remove_file(&path);
    match out {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout).to_string();
            let field = |key: &str| {
                text.lines()
                    .find_map(|l| l.strip_prefix(key)?.strip_prefix('='))
                    .unwrap_or_else(|| panic!("{name}: ffprobe gave no {key}: {text}"))
                    .trim()
                    .to_string()
            };
            Some((field("codec_tag_string"), field("nb_read_frames").parse().unwrap()))
        }
        Ok(o) => panic!("ffprobe rejected {name}: {}", String::from_utf8_lossy(&o.stderr)),
        Err(_) => {
            eprintln!("skipping the ffprobe check: ffprobe not on PATH");
            None
        }
    }
}

fn file_bytes(artifact: RungArtifact) -> Vec<u8> {
    match artifact {
        RungArtifact::File(bytes) => bytes,
        _ => panic!("a single-file rung writes a file"),
    }
}

/// Out of band: the entry, every set the encoder wrote (when the test holds
/// the packets) in the config box and none in the samples, and every frame
/// decoding from them.
fn assert_out_of_band(mp4: &[u8], packets: Option<&[EncodedPacket]>, codec: VideoCodec, name: &str) {
    let expect: &[u8; 4] = if codec == VideoCodec::H264 { b"avc1" } else { b"hvc1" };
    assert_eq!(&sample_entry(mp4).0, expect, "{name}: sample entry");
    let (sets, complete) = config_sets(mp4, codec);
    let kinds = if codec == VideoCodec::H264 { 2 } else { 3 };
    assert_eq!(sets.len(), kinds, "{name}: one set of each kind: {sets:02x?}");
    if let Some(packets) = packets {
        assert_eq!(sets, stream_sets(packets, codec), "{name}: the config box holds the stream's sets");
    }
    if codec == VideoCodec::H265 {
        assert_eq!(complete, vec![1, 1, 1], "{name}: hvc1 arrays are complete");
    }
    assert!(!samples_carry_sets(mp4, codec), "{name}: no set in band");
    if let Some((tag, frames)) = ffprobe_tag(mp4, name) {
        assert_eq!(tag.as_bytes(), expect, "{name}: ffprobe codec_tag_string");
        assert_eq!(frames, 2 * CHUNK, "{name}: every frame decodes");
    }
}

#[test]
fn a_serial_software_encode_writes_avc1_and_hvc1() {
    for codec in [VideoCodec::H264, VideoCodec::H265] {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        for pts in 0..2 * CHUNK {
            tx.try_send(frame(pts, false)).unwrap();
        }
        drop(tx);
        let cfg = EncoderConfig { codec, threads: 1, ..EncoderConfig::default() };
        let out = encode_rung_single_file(
            0,
            &Rung::new(W, H),
            rx,
            cfg,
            Some(EncoderBackend::H26x),
            30.0,
            None,
            None,
            &[],
            &NullSink,
            (0, 1),
            crate::spec::Container::Mp4,
        )
        .expect("the serial rung");
        let mp4 = file_bytes(out.artifact);
        assert_out_of_band(&mp4, None, codec, &format!("serial {codec:?}"));
    }
}

#[test]
fn stitched_chunks_of_one_encoder_write_avc1_and_hvc1() {
    for codec in [VideoCodec::H264, VideoCodec::H265] {
        let mut packets = software_chunk(codec, 0..CHUNK, false);
        packets.extend(software_chunk(codec, CHUNK..2 * CHUNK, false));
        let rp = RungPackets {
            rung_index: 0,
            codec,
            width: W,
            height: H,
            label: "96p".into(),
            packets: packets.clone(),
        };
        let out = mux_rung_packets_to_mp4(rp, 30.0, Default::default(), None, &[], (0, 1)).expect("the stitch");
        assert_out_of_band(&file_bytes(out.artifact), Some(&packets), codec, &format!("stitched {codec:?}"));
    }
}

#[test]
fn stitched_chunks_whose_sets_differ_keep_them_in_band() {
    for codec in [VideoCodec::H264, VideoCodec::H265] {
        // A second chunk encoded at another depth: another profile, so another
        // SPS under the same id.
        let mut packets = software_chunk(codec, 0..CHUNK, false);
        packets.extend(software_chunk(codec, CHUNK..2 * CHUNK, true));
        let rp = RungPackets { rung_index: 0, codec, width: W, height: H, label: "96p".into(), packets };
        let mp4 = file_bytes(mux_rung_packets_to_mp4(rp, 30.0, Default::default(), None, &[], (0, 1)).unwrap().artifact);
        let expect: &[u8; 4] = if codec == VideoCodec::H264 { b"avc3" } else { b"hev1" };
        let name = format!("mixed {codec:?}");
        assert_eq!(&sample_entry(&mp4).0, expect, "{name}");
        assert!(samples_carry_sets(&mp4, codec), "{name}: each chunk carries its own sets");
        if codec == VideoCodec::H265 {
            assert_eq!(config_sets(&mp4, codec).1, vec![0, 0, 0], "{name}: hev1 arrays are not complete");
        }
        if let Some((tag, frames)) = ffprobe_tag(&mp4, &name) {
            assert_eq!(tag.as_bytes(), expect, "{name}: ffprobe codec_tag_string");
            assert_eq!(frames, 2 * CHUNK, "{name}: every frame decodes, each chunk with its own sets");
        }
    }
}

/// An HLS rendition written by a primary muxer and, from `helper`'s packets,
/// a helper muxer — the multi-GPU split — then settled and described.
fn hls_codecs(codec: VideoCodec, primary: &[EncodedPacket], helper: &[EncodedPacket], dir: &Path) -> String {
    let open = |first: u32, init: bool| {
        CmafVideoMuxer::new_with_codec_options(
            dir,
            W,
            H,
            30_000,
            Default::default(),
            codec,
            CmafVideoMuxerOptions {
                first_segment_index: first,
                write_init_segment: init,
                first_segment_base_decode_time: if init { 0 } else { CHUNK * 1000 },
            },
        )
        .unwrap()
    };
    let mut segments = Vec::new();
    let mut manifest = None;
    for (first, packets) in [(1, primary), (2, helper)] {
        let mut m = open(first, first == 1);
        for p in packets {
            m.add_packet(p.data.to_vec(), 1000, p.is_keyframe, p.pts).unwrap();
        }
        m.flush_segment().unwrap();
        let done = m.finalize().unwrap();
        segments.extend(done.segments.clone());
        manifest.get_or_insert(done);
    }
    let mut manifest = manifest.unwrap();
    manifest.segments = segments;
    let rm = RungManifest { rung_index: 0, width: W, height: H, label: "96p".into(), relative_dir: "96p".into(), manifest };
    settle_sample_entry(&rm);
    build_video_variant_spec(&rm, 30.0, 0, None).codec_string
}

#[test]
fn hls_codecs_name_avc1_and_hvc1_unless_a_helper_wrote_other_sets() {
    for codec in [VideoCodec::H264, VideoCodec::H265] {
        let a = software_chunk(codec, 0..CHUNK, false);
        let b = software_chunk(codec, CHUNK..2 * CHUNK, false);
        let ten = software_chunk(codec, CHUNK..2 * CHUNK, true);

        let dir = tempfile::tempdir().unwrap();
        let codecs = hls_codecs(codec, &a, &b, dir.path());
        let init = std::fs::read(dir.path().join("init.mp4")).unwrap();
        let (entry, _) = sample_entry(&init);
        match codec {
            // High (100).
            VideoCodec::H264 => {
                assert_eq!(&entry, b"avc1");
                assert!(codecs.starts_with("avc1.64"), "{codecs}");
            }
            _ => {
                assert_eq!(&entry, b"hvc1");
                assert!(codecs.starts_with("hvc1.1."), "{codecs}");
            }
        }
        let (sets, _) = config_sets(&init, codec);
        assert_eq!(sets, stream_sets(&a, codec), "{codec:?}: init holds the stream's sets");

        let dir = tempfile::tempdir().unwrap();
        let codecs = hls_codecs(codec, &a, &ten, dir.path());
        let prefix = if codec == VideoCodec::H264 { "avc3." } else { "hev1." };
        assert!(codecs.starts_with(prefix), "{codec:?} with a 10-bit helper: {codecs}");
    }
}
