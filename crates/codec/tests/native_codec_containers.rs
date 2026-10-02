//! rivet's own ProRes, VP8, VP9, MPEG-2 and MPEG-4 Part 2 encoders through
//! rivet's muxers and demuxers and back through rivet's own decoders — the
//! codec-and-container half of the output path, below the job engine.
//!
//! For each codec and each file it goes in, a short synthetic clip is
//! encoded through the `Encoder` trait (what `select_encoder` builds for the
//! codec), muxed, demuxed by the streaming demuxer the pipeline reads with,
//! and decoded by the decoder `create_decoder` picks. Every check is against
//! the clip itself: the codec label and size the demuxer reports, the sample
//! entry or codec ID written, one frame back per frame in, presentation
//! timestamps one frame apart in order, and the luma PSNR of every frame
//! against its source.
//!
//! The demux gaps this work closed are checked the same way, on files built
//! here: VP8 in MP4 (`vp08`), MPEG-4 Part 2 whose VOL is only in the `esds`
//! or the Matroska `CodecPrivate`, MPEG-1 / MPEG-2 and ProRes in Matroska
//! (`V_MPEG1`, `V_MPEG2`, `V_PRORES` with its eight header bytes stripped as
//! Matroska stores them), and MPEG-1 video in a transport stream (stream
//! type 0x01).
//!
//! No other implementation is run: the only oracle is the source picture.

use codec::encode::{EncodedPacket, Encoder, EncoderConfig, select_encoder};
use codec::frame::{ColorSpace, PixelFormat, ProresProfile, VideoCodec, VideoFrame};
use container::mux::Av1Mp4Muxer;
use container::streaming::{StreamingDemuxer, demux_streaming};
use container::webm::WebmMuxer;

const W: u32 = 96;
const H: u32 = 64;
const FPS: f64 = 25.0;

/// Frame `n`: a diagonal ramp drifting two samples a frame, with a bright
/// square crossing it — every frame distinguishable, motion for the inter
/// coders, an edge for the transforms.
fn source(n: u64) -> VideoFrame {
    let (w, h) = (W as usize, H as usize);
    let mut data = vec![128u8; w * h * 3 / 2];
    for y in 0..h {
        for x in 0..w {
            let ramp = ((x + y + 2 * n as usize) * 2 % 180) as u8 + 30;
            let sq = (x as i64 - (8 + 3 * n as i64)).unsigned_abs() < 10 && (16..36).contains(&y);
            data[y * w + x] = if sq { 225 } else { ramp };
        }
    }
    // A gentle chroma gradient, so the chroma planes are not flat.
    let (cw, ch) = (w / 2, h / 2);
    for y in 0..ch {
        for x in 0..cw {
            data[w * h + y * cw + x] = 100 + (x * 2) as u8;
            data[w * h + cw * ch + y * cw + x] = 150 - y as u8;
        }
    }
    VideoFrame::new(data.into(), W, H, PixelFormat::Yuv420p, ColorSpace::Bt709, n)
}

/// `frames` frames encoded by the encoder `select_encoder` builds for `codec`.
fn encode(codec: VideoCodec, frames: u64, tweak: impl FnOnce(&mut EncoderConfig)) -> Vec<EncodedPacket> {
    let mut cfg = EncoderConfig { width: W, height: H, frame_rate: FPS, codec, keyframe_interval: 10, ..Default::default() };
    tweak(&mut cfg);
    let mut enc: Box<dyn Encoder> = select_encoder(cfg, None).expect("rivet's own encoder for the codec");
    let mut out = Vec::new();
    for n in 0..frames {
        enc.send_frame(&source(n)).expect("send_frame");
        while let Some(p) = enc.receive_packet().unwrap() {
            out.push(p);
        }
    }
    enc.flush().unwrap();
    while let Some(p) = enc.receive_packet().unwrap() {
        out.push(p);
    }
    out
}

fn mp4(codec: VideoCodec, packets: Vec<EncodedPacket>, quicktime: bool) -> Vec<u8> {
    let mut m = Av1Mp4Muxer::new_with_codec(W, H, FPS, codec).unwrap();
    m.set_quicktime(quicktime);
    for p in packets {
        m.add_packet(p).unwrap();
    }
    m.finalize().unwrap().to_vec()
}

fn webm(codec: VideoCodec, packets: Vec<EncodedPacket>) -> Vec<u8> {
    let mut m = WebmMuxer::new(W, H, FPS, codec).unwrap();
    for p in packets {
        m.add_packet(p).unwrap();
    }
    m.finalize().unwrap()
}

/// What came back out of a file.
struct Readback {
    codec: String,
    dims: (u32, u32),
    /// Presentation timestamps in seconds, in decode order.
    pts: Vec<f64>,
    frames: Vec<VideoFrame>,
}

fn read_back(file: &[u8]) -> Readback {
    let mut demux: Box<dyn StreamingDemuxer> = match demux_streaming(file) {
        Ok(d) => d,
        Err(e) => panic!("rivet demuxes the file: {e:#}"),
    };
    let header = demux.header().clone();
    let mut dec = codec::decode::create_decoder(&header.codec, header.info.clone()).expect("a decoder for the codec");
    let mut pts = Vec::new();
    let mut frames = Vec::new();
    while let Some(s) = demux.next_video_sample().unwrap() {
        pts.push(header.pts_seconds(s.pts_ticks));
        dec.push_sample(&s.data).expect("decode");
        while let Some(f) = dec.decode_next().unwrap() {
            frames.push(f);
        }
    }
    dec.finish().unwrap();
    while let Some(f) = dec.decode_next().unwrap() {
        frames.push(f);
    }
    Readback { codec: header.codec.clone(), dims: (header.info.width, header.info.height), pts, frames }
}

/// Luma PSNR of `frame` against source frame `n` (8-bit, or 10-bit scaled).
fn psnr(frame: &VideoFrame, n: u64) -> f64 {
    let src = source(n);
    let luma = (W * H) as usize;
    let bits = codec::colorspace::planar_bit_depth(frame.format).expect("a planar YUV frame");
    let sample = |i: usize| -> f64 {
        if bits == 8 {
            f64::from(frame.data[i])
        } else {
            f64::from(u16::from_le_bytes([frame.data[2 * i], frame.data[2 * i + 1]])) / f64::from(1u32 << (bits - 8))
        }
    };
    let mse: f64 = (0..luma).map(|i| (sample(i) - f64::from(src.data[i])).powi(2)).sum::<f64>() / luma as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-9)).log10()
}

/// The checks every file gets: codec, size, one frame per frame, timestamps
/// one frame apart in presentation order, PSNR over `floor` on every frame.
/// Returns the worst PSNR.
fn check(file: &[u8], label: &str, frames: u64, floor: f64) -> f64 {
    let r = read_back(file);
    assert_eq!(r.codec, label, "codec label");
    assert_eq!(r.dims, (W, H), "{label}: dimensions");
    assert_eq!(r.pts.len() as u64, frames, "{label}: one sample per frame");
    assert_eq!(r.frames.len() as u64, frames, "{label}: one decoded frame per frame");
    let mut presented = r.pts.clone();
    presented.sort_by(f64::total_cmp);
    for (i, t) in presented.iter().enumerate() {
        let want = presented[0] + i as f64 / FPS;
        assert!((t - want).abs() < 0.0015, "{label}: frame {i} presented at {t:.4}s, want {want:.4}s");
    }
    let mut worst = f64::INFINITY;
    for (n, f) in r.frames.iter().enumerate() {
        assert_eq!((f.width, f.height), (W, H), "{label}: frame {n} size");
        let q = psnr(f, n as u64);
        assert!(q > floor, "{label}: frame {n} at {q:.2} dB (floor {floor})");
        worst = worst.min(q);
    }
    eprintln!("{label}: {frames} frames, worst luma PSNR {worst:.2} dB, {} bytes", file.len());
    worst
}

fn contains(file: &[u8], needle: &[u8]) -> bool {
    file.windows(needle.len()).any(|w| w == needle)
}

#[test]
fn vp9_in_webm_and_mp4() {
    let packets = encode(VideoCodec::Vp9, 12, |_| {});
    assert!(packets[0].is_keyframe && packets[10].is_keyframe && !packets[1].is_keyframe);
    let w = webm(VideoCodec::Vp9, packets.clone());
    assert!(contains(&w, b"webm") && contains(&w, b"V_VP9"));
    check(&w, "vp9", 12, 30.0);
    let m = mp4(VideoCodec::Vp9, packets, false);
    assert!(contains(&m, b"vp09") && contains(&m, b"vpcC"));
    check(&m, "vp9", 12, 30.0);
}

#[test]
fn vp8_in_webm_and_mp4() {
    let packets = encode(VideoCodec::Vp8, 12, |_| {});
    let w = webm(VideoCodec::Vp8, packets.clone());
    assert!(contains(&w, b"V_VP8"));
    check(&w, "vp8", 12, 30.0);
    // `vp08`: the demux gap this closed — the mp4 crate reads vp09 only.
    let m = mp4(VideoCodec::Vp8, packets, false);
    assert!(contains(&m, b"vp08") && contains(&m, b"vpcC"));
    check(&m, "vp8", 12, 30.0);
}

#[test]
fn mpeg2_with_b_pictures_in_mp4_and_mov() {
    let packets = encode(VideoCodec::Mpeg2, 13, |_| {});
    // Reference-first coding order: the muxer is handed the pictures out of
    // display order, each with its own timestamp, and writes a ctts.
    let order: Vec<u64> = packets.iter().map(|p| p.pts).collect();
    assert_ne!(order, (0..13).collect::<Vec<_>>(), "B pictures reorder");
    let m = mp4(VideoCodec::Mpeg2, packets.clone(), false);
    assert!(contains(&m, b"mp4v") && contains(&m, b"esds") && contains(&m, b"ctts"));
    check(&m, "mpeg2", 13, 30.0);
    let q = mp4(VideoCodec::Mpeg2, packets, true);
    assert_eq!(&q[4..12], b"ftypqt  ");
    check(&q, "mpeg2", 13, 30.0);
}

#[test]
fn mpeg4_simple_and_advanced_simple_in_mp4() {
    let sp = mp4(VideoCodec::Mpeg4, encode(VideoCodec::Mpeg4, 12, |_| {}), false);
    check(&sp, "mpeg4", 12, 30.0);
    let asp_packets = encode(VideoCodec::Mpeg4, 12, |c| c.overrides.bframes = Some(2));
    let asp = mp4(VideoCodec::Mpeg4, asp_packets, true);
    assert!(contains(&asp, b"ctts"));
    check(&asp, "mpeg4", 12, 30.0);
}

/// The `esds` carries the VOL; a file whose first sample does not (as other
/// muxers write it) still decodes, because the demuxer puts the VOL back.
#[test]
fn mpeg4_from_mp4_with_the_vol_only_in_the_esds() {
    let packets = encode(VideoCodec::Mpeg4, 8, |_| {});
    let file = mp4(VideoCodec::Mpeg4, packets.clone(), false);
    // The muxer's esds holds the configuration headers verbatim, and the
    // first sample keeps its own copy.
    let vol = container::mpeg_es::mpeg4_config(&packets[0].data).unwrap().to_vec();
    let copies = |f: &[u8]| f.windows(vol.len()).filter(|w| *w == vol.as_slice()).count();
    assert_eq!(copies(&file), 2);
    // The file as other muxers write it: the VOL in the esds alone.
    let stripped = strip_first_sample_prefix(&file, &packets[0].data, vol.len());
    assert_eq!(copies(&stripped), 1);
    check(&stripped, "mpeg4", 8, 30.0);
}

/// `file` with the first `n` bytes of its first sample (which starts with
/// `first`) removed, its `stsz` entry and the chunk offsets after it fixed.
/// Rebuilt by moving the sample's tail over the cut and shifting the mdat:
/// simpler here than a second muxer, and only this test needs it.
fn strip_first_sample_prefix(file: &[u8], first: &[u8], n: usize) -> Vec<u8> {
    let at = file.windows(first.len()).position(|w| w == first).expect("the first sample in the mdat");
    let mut out = file.to_vec();
    out.drain(at..at + n);
    // stsz: the first entry_size follows sample_size (0) and sample_count.
    let stsz = out.windows(4).position(|w| w == b"stsz").unwrap();
    let e = stsz + 4 + 4 + 4 + 4;
    let size = u32::from_be_bytes(out[e..e + 4].try_into().unwrap()) - n as u32;
    out[e..e + 4].copy_from_slice(&size.to_be_bytes());
    // mdat size, and every chunk offset past the cut.
    let mdat = out.windows(4).position(|w| w == b"mdat").unwrap() - 4;
    let msize = u32::from_be_bytes(out[mdat..mdat + 4].try_into().unwrap()) - n as u32;
    out[mdat..mdat + 4].copy_from_slice(&msize.to_be_bytes());
    let stco = out.windows(4).position(|w| w == b"stco").unwrap();
    let count = u32::from_be_bytes(out[stco + 8..stco + 12].try_into().unwrap()) as usize;
    for i in 0..count {
        let o = stco + 12 + 4 * i;
        let off = u32::from_be_bytes(out[o..o + 4].try_into().unwrap()) as usize;
        if off > at {
            out[o..o + 4].copy_from_slice(&((off - n) as u32).to_be_bytes());
        }
    }
    out
}

#[test]
fn prores_every_profile_in_mov() {
    for (p, floor) in [
        (ProresProfile::Proxy, 25.0),
        (ProresProfile::Lt, 30.0),
        (ProresProfile::Standard, 35.0),
        (ProresProfile::Hq, 35.0),
        (ProresProfile::P4444, 35.0),
        (ProresProfile::P4444Xq, 35.0),
    ] {
        let codec = VideoCodec::ProRes(p);
        let packets = encode(codec, 4, |_| {});
        assert!(packets.iter().all(|p| p.is_keyframe), "intra-only");
        let mov = mp4(codec, packets, true);
        assert_eq!(&mov[4..12], b"ftypqt  ");
        assert!(contains(&mov, p.fourcc().as_bytes()), "{p:?} sample entry");
        check(&mov, "prores", 4, floor);
    }
    // ProRes is a QuickTime codec: an ISO MP4 is refused.
    let mut m = Av1Mp4Muxer::new_with_codec(W, H, FPS, VideoCodec::ProRes(ProresProfile::Hq)).unwrap();
    for p in encode(VideoCodec::ProRes(ProresProfile::Hq), 1, |_| {}) {
        m.add_packet(p).unwrap();
    }
    assert!(m.finalize().is_err());
}

// ---------------------------------------------------------------------------
// Matroska and MPEG-TS files built here, for the demux mappings
// ---------------------------------------------------------------------------

fn vint_size(n: usize) -> Vec<u8> {
    // Eight-byte sizes throughout: valid EBML, and no arithmetic to get wrong.
    let mut b = (n as u64 | (1u64 << 56)).to_be_bytes().to_vec();
    b[0] = 0x01;
    b
}

fn el(id: &[u8], body: &[u8]) -> Vec<u8> {
    let mut v = id.to_vec();
    v.extend(vint_size(body.len()));
    v.extend_from_slice(body);
    v
}

fn uint(id: &[u8], v: u64) -> Vec<u8> {
    el(id, &v.to_be_bytes())
}

/// A Matroska file with one video track, `codec_id` and `codec_private`,
/// holding `frames` as SimpleBlocks 40 ms apart, every one flagged key.
fn matroska(codec_id: &str, codec_private: Option<&[u8]>, frames: &[Vec<u8>]) -> Vec<u8> {
    let mut ebml = Vec::new();
    ebml.extend(uint(&[0x42, 0x86], 1));
    ebml.extend(uint(&[0x42, 0xF7], 1));
    ebml.extend(uint(&[0x42, 0xF2], 4));
    ebml.extend(uint(&[0x42, 0xF3], 8));
    ebml.extend(el(&[0x42, 0x82], b"matroska"));
    ebml.extend(uint(&[0x42, 0x87], 4));
    ebml.extend(uint(&[0x42, 0x85], 2));
    let mut info = uint(&[0x2A, 0xD7, 0xB1], 1_000_000);
    info.extend(el(&[0x44, 0x89], &(frames.len() as f64 * 40.0).to_be_bytes()));
    info.extend(el(&[0x4D, 0x80], b"test"));
    info.extend(el(&[0x57, 0x41], b"test"));
    let mut video = uint(&[0xB0], u64::from(W));
    video.extend(uint(&[0xBA], u64::from(H)));
    let mut track = uint(&[0xD7], 1);
    track.extend(uint(&[0x73, 0xC5], 1));
    track.extend(uint(&[0x83], 1));
    track.extend(el(&[0x86], codec_id.as_bytes()));
    track.extend(uint(&[0x23, 0xE3, 0x83], 40_000_000));
    if let Some(cp) = codec_private {
        track.extend(el(&[0x63, 0xA2], cp));
    }
    track.extend(el(&[0xE0], &video));
    let tracks = el(&[0x16, 0x54, 0xAE, 0x6B], &el(&[0xAE], &track));
    let mut cluster = uint(&[0xE7], 0);
    for (i, f) in frames.iter().enumerate() {
        let mut block = vec![0x81];
        block.extend_from_slice(&((i * 40) as i16).to_be_bytes());
        block.push(0x80);
        block.extend_from_slice(f);
        cluster.extend(el(&[0xA3], &block));
    }
    let mut segment = el(&[0x15, 0x49, 0xA9, 0x66], &info);
    segment.extend(tracks);
    segment.extend(el(&[0x1F, 0x43, 0xB6, 0x75], &cluster));
    let mut file = el(&[0x1A, 0x45, 0xDF, 0xA3], &ebml);
    file.extend(el(&[0x18, 0x53, 0x80, 0x67], &segment));
    file
}

/// Coded-order packets as Matroska frames: `pts` order is not needed by the
/// MPEG decoders, which reorder themselves.
fn frames_of(packets: &[EncodedPacket]) -> Vec<Vec<u8>> {
    packets.iter().map(|p| p.data.to_vec()).collect()
}

/// Decode a Matroska file's video and count the frames; check the label.
fn matroska_check(file: &[u8], label: &str, frames: u64, floor: f64) {
    let r = read_back(file);
    assert_eq!(r.codec, label);
    assert_eq!(r.frames.len() as u64, frames, "{label}: frames");
    for (n, f) in r.frames.iter().enumerate() {
        let q = psnr(f, n as u64);
        assert!(q > floor, "{label}: frame {n} at {q:.2} dB");
    }
}

#[test]
fn mpeg4_from_matroska_with_the_vol_in_codec_private() {
    let packets = encode(VideoCodec::Mpeg4, 6, |_| {});
    let vol = container::mpeg_es::mpeg4_config(&packets[0].data).unwrap().to_vec();
    let mut frames = frames_of(&packets);
    frames[0].drain(..vol.len());
    for id in ["V_MPEG4/ISO/SP", "V_MPEG4/ISO/ASP"] {
        matroska_check(&matroska(id, Some(&vol), &frames), "mpeg4", 6, 30.0);
    }
    // And the AVI-style mapping Xvid-in-Matroska files use: a
    // BITMAPINFOHEADER naming the FourCC, the VOL after its 40 bytes.
    let mut bih = vec![0u8; 40];
    bih[0..4].copy_from_slice(&40u32.to_le_bytes());
    bih[16..20].copy_from_slice(b"XVID");
    bih.extend_from_slice(&vol);
    matroska_check(&matroska("V_MS/VFW/FOURCC", Some(&bih), &frames), "mpeg4", 6, 30.0);
}

#[test]
fn mpeg1_and_mpeg2_from_matroska() {
    let packets = encode(VideoCodec::Mpeg2, 7, |_| {});
    let seq = container::mpeg_es::mpeg2_config(&packets[0].data).unwrap().to_vec();
    matroska_check(&matroska("V_MPEG2", Some(&seq), &frames_of(&packets)), "mpeg2", 7, 30.0);
    // V_MPEG1 takes the same path (the MPEG-2 decoder reads both).
    matroska_check(&matroska("V_MPEG1", None, &frames_of(&packets)), "mpeg1", 7, 30.0);
}

#[test]
fn prores_from_matroska_without_its_frame_header() {
    let codec = VideoCodec::ProRes(ProresProfile::Hq);
    let packets = encode(codec, 3, |_| {});
    // Matroska stores a ProRes frame without its first eight bytes (the frame
    // size and `icpf`); the demuxer puts them back.
    let frames: Vec<Vec<u8>> = packets.iter().map(|p| p.data[8..].to_vec()).collect();
    assert_eq!(&packets[0].data[4..8], b"icpf");
    matroska_check(&matroska("V_PRORES", Some(b"apch"), &frames), "prores", 3, 35.0);
}

/// One 188-byte TS packet stream for `payload` on `pid`, the first packet
/// starting a unit, the last padded by adaptation-field stuffing.
fn ts_packetize(pid: u16, payload: &[u8], cc: &mut u8) -> Vec<u8> {
    let mut out = Vec::new();
    let mut rest = payload;
    let mut first = true;
    while !rest.is_empty() {
        let mut pkt = vec![0x47, (if first { 0x40 } else { 0 }) | (pid >> 8) as u8, pid as u8];
        let take = rest.len().min(184);
        if take < 184 {
            let af_len = 184 - take - 1;
            pkt.push(0x30 | (*cc & 0xF));
            pkt.push(af_len as u8);
            if af_len > 0 {
                pkt.push(0x00);
                pkt.extend(std::iter::repeat_n(0xFF, af_len - 1));
            }
        } else {
            pkt.push(0x10 | (*cc & 0xF));
        }
        pkt.extend_from_slice(&rest[..take]);
        assert_eq!(pkt.len(), 188);
        out.extend(pkt);
        rest = &rest[take..];
        *cc = cc.wrapping_add(1);
        first = false;
    }
    out
}

fn section_packet(pid: u16, section: &[u8], cc: &mut u8) -> Vec<u8> {
    let mut payload = vec![0u8]; // pointer_field
    payload.extend_from_slice(section);
    payload.extend_from_slice(&[0u8; 4]); // CRC (not checked)
    let mut pkt = vec![0x47, 0x40 | (pid >> 8) as u8, pid as u8, 0x10 | (*cc & 0xF)];
    pkt.extend(payload);
    pkt.resize(188, 0xFF);
    *cc = cc.wrapping_add(1);
    pkt
}

fn pts_bytes(pts: u64) -> [u8; 5] {
    [
        0x21 | (((pts >> 30) & 0x7) << 1) as u8,
        (pts >> 22) as u8,
        0x01 | (((pts >> 15) & 0x7F) << 1) as u8,
        (pts >> 7) as u8,
        0x01 | ((pts & 0x7F) << 1) as u8,
    ]
}

/// MPEG-1 video (`stream_type` 0x01) in a transport stream: labelled
/// `mpeg1`, sized from its sequence header, decoded by the MPEG-2 decoder.
/// The elementary stream is the MPEG-2 encoder's (the decoder takes either
/// syntax; the mapping is what is under test), one PES per picture.
#[test]
fn mpeg1_video_from_mpeg_ts() {
    let packets = encode(VideoCodec::Mpeg2, 6, |c| c.overrides.bframes = Some(0));
    let (mut cc_pat, mut cc_pmt, mut cc_vid) = (0u8, 0u8, 0u8);
    let mut ts = Vec::new();
    // PAT: program 1 on PID 0x100.
    let pat = [0x00, 0xB0, 13, 0x00, 0x01, 0xC1, 0x00, 0x00, 0x00, 0x01, 0xE1, 0x00];
    ts.extend(section_packet(0, &pat, &mut cc_pat));
    // PMT: PCR on 0x200, one stream: type 0x01 on 0x200.
    let pmt = [0x02, 0xB0, 18, 0x00, 0x01, 0xC1, 0x00, 0x00, 0xE2, 0x00, 0xF0, 0x00, 0x01, 0xE2, 0x00, 0xF0, 0x00];
    ts.extend(section_packet(0x100, &pmt, &mut cc_pmt));
    for p in &packets {
        let pts = 90_000 + p.pts * 3600;
        let mut pes = vec![0, 0, 1, 0xE0, 0, 0, 0x80, 0x80, 5];
        pes.extend(pts_bytes(pts));
        pes.extend_from_slice(&p.data);
        ts.extend(ts_packetize(0x200, &pes, &mut cc_vid));
    }
    let r = read_back(&ts);
    assert_eq!(r.codec, "mpeg1");
    assert_eq!(r.dims, (W, H));
    assert_eq!(r.frames.len(), 6);
    for (n, f) in r.frames.iter().enumerate() {
        assert!(psnr(f, n as u64) > 30.0, "frame {n}");
    }
}
