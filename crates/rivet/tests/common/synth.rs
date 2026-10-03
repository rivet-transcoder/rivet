//! Synthetic test media, made with this workspace's own encoders and written
//! with its own (or, for Matroska and MPEG-TS, these few dozen lines of)
//! muxers: no external program makes any byte of a test input.
//!
//! - Pictures: [`test_pattern`] (colour bars over a moving ramp, with a
//!   moving box, so every frame differs and motion search has something to
//!   find), optionally with uniform per-sample noise ([`add_noise`]), and
//!   [`disc`] — a disc that is round *as displayed* for a given sample
//!   aspect ratio, for geometry tests.
//! - Video: H.264 through `h26x`'s encoder ([`encode_h264`]), optionally
//!   with the sample aspect ratio written into the SPS VUI
//!   ([`sps_with_sar`]); MPEG-2 through `mpeg2`'s ([`encode_mpeg2`]).
//! - Audio: AAC-LC through rivet's AAC encoder ([`aac_sine`]) and DTS core
//!   through `dts`'s encoder ([`dts_5_1`]).
//! - Files: MP4 through rivet's own muxer ([`mp4`], with an optional `pasp`
//!   box), Matroska ([`mkv`]) and MPEG-TS ([`ts`]) written here from the
//!   specifications (RFC 9559; ISO/IEC 13818-1).
//!
//! Shared by the integration tests (`mod common;`), the lib tests that need
//! a clip (`#[path]`), and `examples/synth_clip.rs`, which writes one to disk
//! for CI jobs that hand a file to the CLI.

#![allow(dead_code)]

use bytes::Bytes;
use codec::audio::encode::aac::{AacConfig, AacEncoder};
use codec::audio::{AudioEncoder, AudioFrame};
use codec::encode::EncodedPacket;
use codec::frame::VideoCodec;
use container::AudioInfo;
use container::mux::Av1Mp4Muxer;
use h26x::ChromaFormat;
use h26x::encode::h264::H264Encoder;
use h26x::encode::{Config, RateControl};

// ---- pictures --------------------------------------------------------------

/// A small deterministic generator (xorshift64*), so a noisy clip is the
/// same clip on every run and every host.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[-1, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

/// Chroma plane size for `chroma`.
fn chroma_dims(w: u32, h: u32, chroma: ChromaFormat) -> (usize, usize) {
    match chroma {
        ChromaFormat::Yuv444 => (w as usize, h as usize),
        ChromaFormat::Yuv422 => (w.div_ceil(2) as usize, h as usize),
        ChromaFormat::Monochrome => (0, 0),
        ChromaFormat::Yuv420 => (w.div_ceil(2) as usize, h.div_ceil(2) as usize),
    }
}

/// One planar 8-bit picture: Y, then Cb, then Cr.
pub fn blank(w: u32, h: u32, chroma: ChromaFormat) -> Vec<u8> {
    let (cw, ch) = chroma_dims(w, h, chroma);
    let mut p = vec![16u8; (w * h) as usize];
    p.resize(p.len() + 2 * cw * ch, 128);
    p
}

/// Frame `t` of the test pattern: eight colour bars over the top half, a
/// diagonal luma ramp below them that scrolls one sample per frame, and a
/// white box crossing the picture. 4:2:0 or 4:4:4.
pub fn test_pattern(w: u32, h: u32, t: u64, chroma: ChromaFormat) -> Vec<u8> {
    // 75% bars (BT.601 limited range): white, yellow, cyan, green, magenta,
    // red, blue, black.
    const BARS: [(u8, u8, u8); 8] = [
        (180, 128, 128),
        (162, 44, 142),
        (131, 156, 44),
        (112, 72, 58),
        (84, 184, 198),
        (65, 100, 212),
        (35, 212, 114),
        (16, 128, 128),
    ];
    let (wu, hu) = (w as usize, h as usize);
    let (cw, ch) = chroma_dims(w, h, chroma);
    let (sx, sy) = (wu.div_ceil(cw.max(1)), hu.div_ceil(ch.max(1)));
    let mut p = blank(w, h, chroma);
    let box_size = (wu.min(hu) / 6).max(4);
    let bx = (t as usize * 3) % wu.max(1);
    let by = hu / 2 + (hu / 2).saturating_sub(box_size) / 2;
    let bar_of = |x: usize| BARS[(x * 8 / wu.max(1)).min(7)];
    for y in 0..hu {
        for x in 0..wu {
            let in_box = x >= bx && x < bx + box_size && y >= by && y < by + box_size;
            p[y * wu + x] = if in_box {
                235
            } else if y < hu / 2 {
                bar_of(x).0
            } else {
                (16 + (x + y + t as usize) % 220) as u8
            };
        }
    }
    let (cb, cr) = p.split_at_mut(wu * hu).1.split_at_mut(cw * ch);
    for y in 0..ch {
        for x in 0..cw {
            let (lx, ly) = (x * sx, y * sy);
            let (u, v) = if ly < hu / 2 { (bar_of(lx).1, bar_of(lx).2) } else { (128, 128) };
            cb[y * cw + x] = u;
            cr[y * cw + x] = v;
        }
    }
    p
}

/// Adds uniform noise of up to `strength` to every sample of `pic` (all
/// planes), different on every call: temporal and uniform, like film grain
/// at its crudest. Clamped to the limited range.
pub fn add_noise(pic: &mut [u8], strength: u8, rng: &mut Rng) {
    if strength == 0 {
        return;
    }
    for s in pic.iter_mut() {
        let n = (rng.uniform() * f64::from(strength)).round() as i32;
        *s = (i32::from(*s) + n).clamp(16, 240) as u8;
    }
}

/// A `w x h` picture with a disc of radius `min(w, h) / 5` (in rows) that is
/// round as displayed with sample aspect `sn:sd`: in stored samples it is
/// `sd / sn` as wide. Luma 235 inside, 16 outside, chroma neutral.
pub fn disc(w: u32, h: u32, (sn, sd): (u32, u32), chroma: ChromaFormat) -> Vec<u8> {
    let mut p = blank(w, h, chroma);
    let r = f64::from(w.min(h)) * 0.2;
    let (cx, cy) = (f64::from(w) / 2.0, f64::from(h) / 2.0);
    let k = f64::from(sn) / f64::from(sd);
    for y in 0..h as usize {
        for x in 0..w as usize {
            let dx = (x as f64 + 0.5 - cx) * k;
            let dy = y as f64 + 0.5 - cy;
            if dx.hypot(dy) <= r {
                p[y * w as usize + x] = 235;
            }
        }
    }
    p
}

// ---- video -----------------------------------------------------------------

/// How to code a clip with [`encode_h264`].
#[derive(Clone, Debug)]
pub struct H264 {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub chroma: ChromaFormat,
    /// Fixed quantiser, or (with `bitrate`) ignored.
    pub qp: u8,
    /// Average bit rate in bits/s; 0 for the fixed quantiser.
    pub bitrate: u32,
    /// Pictures between IDRs.
    pub gop: u32,
    /// Sample aspect ratio to write into the SPS VUI; `None` writes none.
    pub sar: Option<(u16, u16)>,
}

impl H264 {
    pub fn new(width: u32, height: u32, fps: u32) -> Self {
        Self { width, height, fps, chroma: ChromaFormat::Yuv420, qp: 20, bitrate: 0, gop: fps.max(1), sar: None }
    }
}

/// One coded picture: Annex B bytes, its display index, and whether it is a
/// random access point.
#[derive(Clone, Debug)]
pub struct Coded {
    pub data: Vec<u8>,
    pub pts: u64,
    pub key: bool,
}

/// Codes `pictures` (planar 8-bit, display order) as H.264 with this
/// workspace's encoder: no B pictures, so decode order is display order.
pub fn encode_h264(cfg: &H264, pictures: impl IntoIterator<Item = Vec<u8>>) -> Vec<Coded> {
    let rate = if cfg.bitrate > 0 { RateControl::Bitrate { bps: cfg.bitrate } } else { RateControl::ConstantQp(cfg.qp) };
    let mut enc = H264Encoder::new(Config {
        width: cfg.width,
        height: cfg.height,
        chroma: cfg.chroma,
        gop: cfg.gop,
        rate,
        fps: cfg.fps,
        threads: 0,
        ..Config::default()
    })
    .expect("the h26x H.264 encoder takes the configuration");
    let mut out = Vec::new();
    let take = |units: Vec<h26x::encode::Access>, out: &mut Vec<Coded>| {
        for a in units {
            let data = match cfg.sar {
                Some(sar) => rewrite_sps(&a.data, sar),
                None => a.data,
            };
            out.push(Coded { data, pts: a.display, key: a.keyframe });
        }
    };
    for p in pictures {
        let units = enc.push(&p).expect("encode a picture");
        take(units, &mut out);
    }
    let units = enc.flush().expect("flush the encoder");
    take(units, &mut out);
    out
}

/// Codes `pictures` (planar 4:2:0) as MPEG-2 video with this workspace's
/// encoder, I and P pictures only, `aspect_ratio_information` as given (1
/// square, 2 4:3, 3 16:9). Returns one coded picture per frame; the first
/// carries the sequence header.
pub fn encode_mpeg2(w: u32, h: u32, fps: u32, aspect: u8, pictures: impl IntoIterator<Item = Vec<u8>>) -> Vec<Coded> {
    let mut cfg = mpeg2::EncoderConfig::new(w, h);
    cfg.frame_rate = (fps, 1);
    cfg.aspect_ratio_information = aspect;
    cfg.b_frames = 0;
    cfg.gop_size = fps.max(1);
    let mut enc = mpeg2::Encoder::new(cfg).expect("the MPEG-2 encoder takes the configuration");
    let mut es = Vec::new();
    for p in pictures {
        let mut f = mpeg2::Frame::new(w, h, mpeg2::ChromaFormat::Yuv420);
        let n = f.data.len();
        f.data.copy_from_slice(&p[..n]);
        es.extend(enc.encode(&f).expect("encode an MPEG-2 picture"));
    }
    es.extend(enc.finish().expect("finish the MPEG-2 stream"));
    let types = container::mpeg_es::mpeg2_picture_types(&es);
    container::mpeg_es::split_mpeg2_pictures(&es)
        .into_iter()
        .enumerate()
        .map(|(i, data)| Coded { data, pts: i as u64, key: types.get(i) == Some(&1) })
        .collect()
}

/// The NAL units of an Annex B buffer (without start codes).
pub fn nals(data: &[u8]) -> Vec<&[u8]> {
    h26x::nal::annexb_nals(data).filter(|n| !n.is_empty()).collect()
}

/// `data` with every H.264 SPS replaced by [`sps_with_sar`]'s.
fn rewrite_sps(data: &[u8], sar: (u16, u16)) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 8);
    for n in nals(data) {
        out.extend_from_slice(&[0, 0, 0, 1]);
        if n[0] & 0x1f == 7 {
            out.extend(sps_with_sar(n, sar));
        } else {
            out.extend_from_slice(n);
        }
    }
    out
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl BitReader<'_> {
    fn bit(&mut self) -> u32 {
        let b = (self.data[self.pos / 8] >> (7 - self.pos % 8)) & 1;
        self.pos += 1;
        u32::from(b)
    }

    fn bits(&mut self, n: u32) -> u32 {
        (0..n).fold(0, |v, _| (v << 1) | self.bit())
    }

    fn ue(&mut self) -> u32 {
        let mut zeros = 0;
        while self.bit() == 0 {
            zeros += 1;
        }
        (1 << zeros) - 1 + self.bits(zeros)
    }

    fn se(&mut self) -> i32 {
        let k = self.ue();
        if k % 2 == 1 { k.div_ceil(2) as i32 } else { -((k / 2) as i32) }
    }
}

#[derive(Default)]
struct BitWriter {
    out: Vec<u8>,
    n: usize,
}

impl BitWriter {
    fn bit(&mut self, b: u32) {
        if self.n % 8 == 0 {
            self.out.push(0);
        }
        if b != 0 {
            *self.out.last_mut().unwrap() |= 1 << (7 - self.n % 8);
        }
        self.n += 1;
    }

    fn bits(&mut self, v: u32, n: u32) {
        for i in (0..n).rev() {
            self.bit((v >> i) & 1);
        }
    }
}

/// The RBSP of a NAL unit payload: emulation prevention bytes removed.
fn unescape(ebsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ebsp.len());
    let mut zeros = 0;
    for &b in ebsp {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

/// Emulation prevention (H.264 7.4.1) applied to an RBSP.
fn escape(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + 4);
    let mut zeros = 0;
    for &b in rbsp {
        if zeros >= 2 && b <= 3 {
            out.push(3);
            zeros = 0;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        out.push(b);
    }
    out
}

/// An H.264 SPS NAL unit (header byte included) rewritten to signal the
/// sample aspect ratio `sar` as `aspect_ratio_idc` 255 (Extended_SAR) in
/// its VUI (H.264 7.3.2.1.1, E.1.1). Every other field is kept: the bits
/// before the VUI are copied, a VUI is added if there was none (every other
/// VUI flag 0), and an existing one keeps everything after its
/// `aspect_ratio_info_present_flag`. Refuses an SPS that already signals an
/// aspect ratio or carries scaling matrices (this encoder writes neither).
pub fn sps_with_sar(nal: &[u8], (sar_w, sar_h): (u16, u16)) -> Vec<u8> {
    let rbsp = unescape(&nal[1..]);
    // The stop bit: the last 1 in the payload.
    let last = rbsp.iter().rposition(|&b| b != 0).expect("an SPS with a stop bit");
    let stop = last * 8 + 7 - rbsp[last].trailing_zeros() as usize;
    let mut r = BitReader { data: &rbsp, pos: 0 };
    let profile = r.bits(8);
    r.bits(16); // constraint flags, level_idc
    r.ue(); // seq_parameter_set_id
    if matches!(profile, 100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135) {
        if r.ue() == 3 {
            r.bit(); // separate_colour_plane_flag
        }
        r.ue(); // bit_depth_luma_minus8
        r.ue(); // bit_depth_chroma_minus8
        r.bit(); // qpprime_y_zero_transform_bypass_flag
        assert_eq!(r.bit(), 0, "sps_with_sar: scaling matrices are not handled");
    }
    r.ue(); // log2_max_frame_num_minus4
    match r.ue() {
        0 => {
            r.ue();
        }
        1 => {
            r.bit();
            r.se();
            r.se();
            for _ in 0..r.ue() {
                r.se();
            }
        }
        _ => {}
    }
    r.ue(); // max_num_ref_frames
    r.bit(); // gaps_in_frame_num_value_allowed_flag
    r.ue(); // pic_width_in_mbs_minus1
    r.ue(); // pic_height_in_map_units_minus1
    if r.bit() == 0 {
        r.bit(); // mb_adaptive_frame_field_flag
    }
    r.bit(); // direct_8x8_inference_flag
    if r.bit() == 1 {
        for _ in 0..4 {
            r.ue();
        }
    }
    let vui_flag_at = r.pos;
    let had_vui = r.bit() == 1;

    let mut w = BitWriter::default();
    let mut copy = BitReader { data: &rbsp, pos: 0 };
    for _ in 0..vui_flag_at {
        w.bit(copy.bit());
    }
    w.bit(1); // vui_parameters_present_flag
    w.bit(1); // aspect_ratio_info_present_flag
    w.bits(255, 8); // aspect_ratio_idc: Extended_SAR
    w.bits(u32::from(sar_w), 16);
    w.bits(u32::from(sar_h), 16);
    if had_vui {
        assert_eq!(r.bit(), 0, "sps_with_sar: the SPS already signals an aspect ratio");
        for _ in r.pos..stop {
            w.bit(r.bit());
        }
    } else {
        // overscan, video_signal_type, chroma_loc, timing, nal_hrd, vcl_hrd,
        // pic_struct, bitstream_restriction: all absent.
        w.bits(0, 8);
    }
    w.bit(1); // rbsp_stop_one_bit, then alignment zeros
    let mut out = vec![nal[0]];
    out.extend(escape(&w.out));
    out
}

// ---- audio -----------------------------------------------------------------

/// An audio track ready to mux: its description and its access units with
/// their durations (in the track's timescale).
pub struct AudioTrack {
    pub info: AudioInfo,
    pub frames: Vec<(Vec<u8>, u32)>,
}

/// `seconds` of a sine at `freq` Hz on every one of `channels` channels,
/// 48 kHz, coded as AAC-LC by rivet's encoder at `bitrate`.
pub fn aac_sine(freq: f64, seconds: f64, channels: u8, bitrate: u32) -> AudioTrack {
    const RATE: u32 = 48_000;
    let mut enc =
        AacEncoder::new(AacConfig { sample_rate: RATE, channels, bitrate }).expect("rivet's AAC encoder");
    let total = (seconds * f64::from(RATE)) as usize;
    let mut packets = Vec::new();
    let mut at = 0;
    while at < total {
        let n = (total - at).min(1024);
        let mut samples = Vec::with_capacity(n * channels as usize);
        for i in at..at + n {
            let v = (0.4 * (std::f64::consts::TAU * freq * i as f64 / f64::from(RATE)).sin()) as f32;
            samples.extend(std::iter::repeat_n(v, channels as usize));
        }
        let pts = (at as i64 * 1_000_000) / i64::from(RATE);
        packets.extend(enc.encode(&AudioFrame { samples, sample_rate: RATE, channels, pts }).expect("AAC encode"));
        at += n;
    }
    packets.extend(enc.flush().expect("AAC flush"));
    let asc = enc.extra_data();
    AudioTrack {
        info: AudioInfo::aac_lc(RATE, u16::from(channels), asc),
        frames: packets.into_iter().map(|p| (p.data, p.duration as u32)).collect(),
    }
}

/// The committed `tests/data/audio/tones_51_aac.m4a`: half a second of 5.1
/// (FL FR FC LFE BL BR, channelConfiguration 6) at 48 kHz with one tone per
/// channel — 400, 600, 800, 50, 1000 and 1200 Hz, each at 0.25 — coded as
/// AAC-LC at 192 kbit/s by rivet's encoder, in an audio-only MP4 whose edit
/// list hides the encoder's priming. `examples/synth_clip.rs --tones51-aac`
/// writes it.
pub fn tones_51_aac_m4a() -> Vec<u8> {
    const RATE: u32 = 48_000;
    const TONES: [f64; 6] = [400.0, 600.0, 800.0, 50.0, 1000.0, 1200.0];
    let total = RATE as usize / 2;
    let mut enc = AacEncoder::new(AacConfig { sample_rate: RATE, channels: 6, bitrate: 192_000 }).expect("rivet's AAC encoder");
    let mut packets = Vec::new();
    for start in (0..total).step_by(1024) {
        let n = (total - start).min(1024);
        let mut samples = Vec::with_capacity(n * 6);
        for i in start..start + n {
            for f in TONES {
                samples.push((0.25 * (std::f64::consts::TAU * f * i as f64 / f64::from(RATE)).sin()) as f32);
            }
        }
        let pts = (start as i64 * 1_000_000) / i64::from(RATE);
        packets.extend(enc.encode(&AudioFrame { samples, sample_rate: RATE, channels: 6, pts }).expect("AAC encode"));
    }
    packets.extend(enc.flush().expect("AAC flush"));
    let info = AudioInfo::aac_lc(RATE, 6, enc.extra_data());
    let samples: Vec<(Vec<u8>, u32)> = packets.into_iter().map(|p| (p.data, p.duration as u32)).collect();
    let edit = container::edit::TrackEdit { delay: 0, media_time: u64::from(enc.pre_skip()), duration: Some(total as u64) };
    container::mux::write_audio_mp4(&info, &samples, edit).expect("the audio-only MP4")
}

/// `seconds` of 5.1 (FL FR FC LFE SL SR) at 48 kHz coded as DTS core by the
/// `dts` crate's encoder at 768 kbit/s: tones at 440, 660 and 880 Hz on the
/// fronts, 60 Hz on the LFE, and two independent noises on the surrounds.
pub fn dts_5_1(seconds: f64) -> AudioTrack {
    const RATE: u32 = 48_000;
    let layout = dts::Layout::Surround51Side;
    let mut enc = dts::Encoder::new(dts::EncoderConfig::new(RATE, layout, 768_000)).expect("the DTS encoder");
    let speakers = layout.speakers();
    let total = (seconds * f64::from(RATE)) as usize;
    let mut rng = Rng::new(7);
    let mut pcm = Vec::with_capacity(total * speakers.len());
    for i in 0..total {
        let t = i as f64 / f64::from(RATE);
        for s in speakers {
            let tone = |f: f64| 0.3 * (std::f64::consts::TAU * f * t).sin();
            let v = match s {
                dts::Speaker::FL => tone(440.0),
                dts::Speaker::FR => tone(660.0),
                dts::Speaker::FC => tone(880.0),
                dts::Speaker::LFE => tone(60.0),
                _ => 0.2 * rng.uniform(),
            };
            pcm.push(v as f32);
        }
    }
    let mut frames = enc.encode(&pcm).expect("DTS encode");
    frames.extend(enc.flush().expect("DTS flush"));
    let core = container::dts_sync::parse_core_sync(&frames[0]).expect("a DTS core frame");
    let samples = core.samples_per_frame;
    let info = AudioInfo {
        codec: "dts".into(),
        sample_rate: RATE,
        channels: speakers.len() as u16,
        timescale: RATE,
        asc_bytes: Vec::new(),
        codec_private: container::mux::ddts_body_from_sync(&core, false),
    };
    AudioTrack { info, frames: frames.into_iter().map(|f| (f, samples)).collect() }
}

// ---- files -----------------------------------------------------------------

/// An MP4 with the H.264 `video` (rivet's muxer, `avc1`), the optional
/// audio track, and, when `pasp` is given, a `pasp` box in the sample entry.
pub fn mp4(video: &[Coded], w: u32, h: u32, fps: u32, audio: Option<&AudioTrack>, pasp: Option<(u32, u32)>) -> Vec<u8> {
    mp4_of(VideoCodec::H264, video, w, h, fps, audio, pasp)
}

/// [`mp4`] for any codec rivet's MP4 muxer takes Annex B for.
pub fn mp4_of(
    codec: VideoCodec,
    video: &[Coded],
    w: u32,
    h: u32,
    fps: u32,
    audio: Option<&AudioTrack>,
    pasp: Option<(u32, u32)>,
) -> Vec<u8> {
    let mut mux = Av1Mp4Muxer::new_with_codec(w, h, f64::from(fps), codec).expect("the MP4 muxer");
    if let Some(a) = audio {
        mux.with_audio(a.info.clone()).expect("the audio track");
        let mut pts = 0u64;
        for (data, dur) in &a.frames {
            mux.add_audio_sample(data, pts, *dur).expect("an audio sample");
            pts += u64::from(*dur);
        }
    }
    for c in video {
        mux.add_packet(EncodedPacket { data: Bytes::from(c.data.clone()), pts: c.pts, is_keyframe: c.key })
            .expect("a video sample");
    }
    let file = mux.finalize().expect("finalize the MP4").to_vec();
    match pasp {
        Some((hs, vs)) => {
            let mut b = Vec::with_capacity(16);
            b.extend_from_slice(&16u32.to_be_bytes());
            b.extend_from_slice(b"pasp");
            b.extend_from_slice(&hs.to_be_bytes());
            b.extend_from_slice(&vs.to_be_bytes());
            let entry: &[u8; 4] = if codec == VideoCodec::H265 { b"hvc1" } else { b"avc1" };
            add_to_sample_entry(&file, entry, &b)
        }
        None => file,
    }
}

/// `file` (an MP4) with `child` appended to the body of its `entry` sample
/// entry: every enclosing box grows by `child.len()`, and when `moov` comes
/// before `mdat` every chunk offset moves by as much.
pub fn add_to_sample_entry(file: &[u8], entry: &[u8; 4], child: &[u8]) -> Vec<u8> {
    fn boxes(data: &[u8]) -> Vec<(&[u8], &[u8; 4], &[u8])> {
        // (whole box, type, body)
        let mut out = Vec::new();
        let mut at = 0;
        while at + 8 <= data.len() {
            let size = u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) as usize;
            let (size, head) = match size {
                1 => (u64::from_be_bytes(data[at + 8..at + 16].try_into().unwrap()) as usize, 16),
                0 => (data.len() - at, 8),
                s => (s, 8),
            };
            let kind: &[u8; 4] = data[at + 4..at + 8].try_into().unwrap();
            out.push((&data[at..at + size], kind, &data[at + head..at + size]));
            at += size;
        }
        out
    }
    fn wrap(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut b = Vec::with_capacity(body.len() + 8);
        b.extend_from_slice(&(body.len() as u32 + 8).to_be_bytes());
        b.extend_from_slice(kind);
        b.extend_from_slice(body);
        b
    }
    fn rebuild(body: &[u8], entry: &[u8; 4], child: &[u8], shift: u64) -> Vec<u8> {
        let mut out = Vec::new();
        for (whole, kind, inner) in boxes(body) {
            match kind {
                b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" => {
                    out.extend(wrap(kind, &rebuild(inner, entry, child, shift)));
                }
                b"stsd" => {
                    let mut b = inner[..8].to_vec();
                    for (e, ek, ebody) in boxes(&inner[8..]) {
                        if ek == entry {
                            let mut grown = ebody.to_vec();
                            grown.extend_from_slice(child);
                            b.extend(wrap(ek, &grown));
                        } else {
                            b.extend_from_slice(e);
                        }
                    }
                    out.extend(wrap(kind, &b));
                }
                b"stco" if shift > 0 => {
                    let mut b = inner[..8].to_vec();
                    for o in inner[8..].chunks_exact(4) {
                        b.extend_from_slice(&(u32::from_be_bytes(o.try_into().unwrap()) + shift as u32).to_be_bytes());
                    }
                    out.extend(wrap(kind, &b));
                }
                b"co64" if shift > 0 => {
                    let mut b = inner[..8].to_vec();
                    for o in inner[8..].chunks_exact(8) {
                        b.extend_from_slice(&(u64::from_be_bytes(o.try_into().unwrap()) + shift).to_be_bytes());
                    }
                    out.extend(wrap(kind, &b));
                }
                _ => out.extend_from_slice(whole),
            }
        }
        out
    }
    let top = boxes(file);
    let pos = |k: &[u8; 4]| top.iter().position(|(_, kind, _)| *kind == k);
    let moov_first = pos(b"moov") < pos(b"mdat");
    let shift = if moov_first { child.len() as u64 } else { 0 };
    let out = rebuild(file, entry, child, shift);
    assert_eq!(out.len(), file.len() + child.len(), "add_to_sample_entry: exactly one `{}` entry", String::from_utf8_lossy(entry));
    out
}

/// An `avcC` record (ISO/IEC 14496-15 5.3.3.1) for the first SPS and PPS
/// in `video`.
pub fn avcc(video: &[Coded]) -> Vec<u8> {
    let all: Vec<&[u8]> = video.iter().flat_map(|c| nals(&c.data)).collect();
    let sps = all.iter().find(|n| n[0] & 0x1f == 7).expect("an SPS");
    let pps = all.iter().find(|n| n[0] & 0x1f == 8).expect("a PPS");
    let mut b = vec![1, sps[1], sps[2], sps[3], 0xFF, 0xE1];
    b.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    b.extend_from_slice(sps);
    b.push(1);
    b.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    b.extend_from_slice(pps);
    b
}

/// Annex B to four-byte length-prefixed NAL units.
pub fn length_prefixed(annexb: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(annexb.len());
    for n in nals(annexb) {
        out.extend_from_slice(&(n.len() as u32).to_be_bytes());
        out.extend_from_slice(n);
    }
    out
}

/// An EBML element: ID (with its marker bits, as the specification writes
/// it), size, body.
fn ebml(id: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 12);
    let id_bytes = id.to_be_bytes();
    let skip = id_bytes.iter().position(|&b| b != 0).unwrap_or(3);
    out.extend_from_slice(&id_bytes[skip..]);
    // The shortest size: n bytes hold 7n bits, all ones reserved.
    let len = body.len() as u64;
    let n = (1..=8).find(|&n| len < (1u64 << (7 * n)) - 1).unwrap();
    let marked = len | (1u64 << (7 * n));
    out.extend_from_slice(&marked.to_be_bytes()[8 - n as usize..]);
    out.extend_from_slice(body);
    out
}

fn ebml_uint(id: u32, v: u64) -> Vec<u8> {
    let b = v.to_be_bytes();
    let skip = b.iter().position(|&x| x != 0).unwrap_or(7);
    ebml(id, &b[skip..])
}

fn ebml_float(id: u32, v: f64) -> Vec<u8> {
    ebml(id, &v.to_be_bytes())
}

/// Matroska audio for [`mkv`]: codec ID, rate, channels, private data, and
/// the frames with their durations in samples.
pub struct MkvAudio<'a> {
    pub codec_id: &'a str,
    pub track: &'a AudioTrack,
}

/// A Matroska file (RFC 9559) with the H.264 `video` as `V_MPEG4/ISO/AVC`
/// (length-prefixed samples, `avcC` as `CodecPrivate`), stored at `w x h`
/// and displayed at `display` (`DisplayWidth` / `DisplayHeight`, in pixels),
/// and the optional audio track. One cluster per second; every block a
/// `SimpleBlock`.
pub fn mkv(video: &[Coded], w: u32, h: u32, fps: u32, display: Option<(u32, u32)>, audio: Option<MkvAudio>) -> Vec<u8> {
    let header = [
        ebml_uint(0x4286, 1),
        ebml_uint(0x42F7, 1),
        ebml_uint(0x42F2, 4),
        ebml_uint(0x42F3, 8),
        ebml(0x4282, b"matroska"),
        ebml_uint(0x4287, 4),
        ebml_uint(0x4285, 2),
    ]
    .concat();
    let duration_ms = video.len() as f64 * 1000.0 / f64::from(fps);
    let info = [
        ebml_uint(0x2AD7B1, 1_000_000),
        ebml(0x4D80, b"rivet test synth"),
        ebml(0x5741, b"rivet test synth"),
        ebml_float(0x4489, duration_ms),
    ]
    .concat();
    let mut v = vec![ebml_uint(0xB0, u64::from(w)), ebml_uint(0xBA, u64::from(h))];
    if let Some((dw, dh)) = display {
        v.push(ebml_uint(0x54B0, u64::from(dw)));
        v.push(ebml_uint(0x54BA, u64::from(dh)));
    }
    let mut tracks = ebml(
        0xAE,
        &[
            ebml_uint(0xD7, 1),
            ebml_uint(0x73C5, 1),
            ebml_uint(0x83, 1),
            ebml_uint(0x9C, 0),
            ebml(0x86, b"V_MPEG4/ISO/AVC"),
            ebml(0x63A2, &avcc(video)),
            ebml_uint(0x23E383, 1_000_000_000 / u64::from(fps)),
            ebml(0xE0, &v.concat()),
        ]
        .concat(),
    );
    if let Some(a) = &audio {
        let mut entry = vec![
            ebml_uint(0xD7, 2),
            ebml_uint(0x73C5, 2),
            ebml_uint(0x83, 2),
            ebml_uint(0x9C, 0),
            ebml(0x86, a.codec_id.as_bytes()),
        ];
        if !a.track.info.codec_private.is_empty() && a.codec_id != "A_DTS" {
            entry.push(ebml(0x63A2, &a.track.info.codec_private));
        }
        entry.push(ebml(
            0xE1,
            &[ebml_float(0xB5, f64::from(a.track.info.sample_rate)), ebml_uint(0x9F, u64::from(a.track.info.channels))]
                .concat(),
        ));
        tracks.extend(ebml(0xAE, &entry.concat()));
    }
    // Blocks by time, in milliseconds: (ms, track, key, payload).
    let mut blocks: Vec<(u64, u8, bool, Vec<u8>)> = video
        .iter()
        .map(|c| (c.pts * 1000 / u64::from(fps), 1, c.key, length_prefixed(&c.data)))
        .collect();
    if let Some(a) = &audio {
        let mut t = 0u64;
        for (data, dur) in &a.track.frames {
            blocks.push((t * 1000 / u64::from(a.track.info.timescale), 2, true, data.clone()));
            t += u64::from(*dur);
        }
    }
    blocks.sort_by_key(|b| (b.0, b.1));
    let mut clusters = Vec::new();
    let mut i = 0;
    while i < blocks.len() {
        let start = blocks[i].0;
        let mut body = ebml_uint(0xE7, start);
        while i < blocks.len() && blocks[i].0 < start + 1000 {
            let (ms, track, key, data) = &blocks[i];
            let mut b = vec![0x80 | track];
            b.extend_from_slice(&((ms - start) as i16).to_be_bytes());
            b.push(if *key { 0x80 } else { 0 });
            b.extend_from_slice(data);
            body.extend(ebml(0xA3, &b));
            i += 1;
        }
        clusters.extend(ebml(0x1F43B675, &body));
    }
    let segment = [ebml(0x1549A966, &info), ebml(0x1654AE6B, &tracks), clusters].concat();
    [ebml(0x1A45DFA3, &header), ebml(0x18538067, &segment)].concat()
}

/// CRC-32/MPEG-2 (ISO/IEC 13818-1 Annex A).
fn crc32_mpeg(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= u32::from(b) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 { (crc << 1) ^ 0x04C1_1DB7 } else { crc << 1 };
        }
    }
    crc
}

/// Splits `payload` into 188-byte transport packets on `pid`, the first
/// with `payload_unit_start_indicator` (and a PCR when given), the last
/// filled out with adaptation-field stuffing.
fn ts_packets(out: &mut Vec<u8>, pid: u16, cc: &mut u8, payload: &[u8], pcr: Option<u64>) {
    let mut at = 0;
    let mut first = true;
    while at < payload.len() {
        // The adaptation field's body (after its length byte), if any.
        let mut af: Option<Vec<u8>> = match (first, pcr) {
            (true, Some(base)) => Some(vec![
                0x10, // PCR_flag
                (base >> 25) as u8,
                (base >> 17) as u8,
                (base >> 9) as u8,
                (base >> 1) as u8,
                (((base & 1) << 7) as u8) | 0x7E,
                0,
            ]),
            _ => None,
        };
        let room = |af: &Option<Vec<u8>>| 184 - af.as_ref().map_or(0, |b| 1 + b.len());
        let left = payload.len() - at;
        if left < room(&af) {
            let stuff = room(&af) - left;
            af = Some(match af {
                Some(mut b) => {
                    b.resize(b.len() + stuff, 0xFF);
                    b
                }
                // One byte of stuffing is an adaptation field of length 0.
                None if stuff == 1 => Vec::new(),
                None => {
                    let mut b = vec![0x00];
                    b.resize(stuff - 1, 0xFF);
                    b
                }
            });
        }
        let take = room(&af).min(left);
        let control = if af.is_some() { 0x30 } else { 0x10 };
        let pusi = if first { 0x40 } else { 0 };
        let mut pkt = vec![0x47, pusi | (pid >> 8) as u8, pid as u8, control | *cc];
        if let Some(b) = &af {
            pkt.push(b.len() as u8);
            pkt.extend_from_slice(b);
        }
        pkt.extend_from_slice(&payload[at..at + take]);
        assert_eq!(pkt.len(), 188, "a transport packet is 188 bytes");
        out.extend(pkt);
        *cc = (*cc + 1) & 0x0F;
        at += take;
        first = false;
    }
}

/// A PSI section (table_id .. CRC) in a packet of its own.
fn psi(out: &mut Vec<u8>, pid: u16, cc: &mut u8, table_id: u8, id: u16, body: &[u8]) {
    let mut s = vec![table_id];
    let len = 5 + body.len() + 4;
    s.extend_from_slice(&(0xB000u16 | len as u16).to_be_bytes());
    s.extend_from_slice(&id.to_be_bytes());
    s.extend_from_slice(&[0xC1, 0, 0]); // version 0, current, section 0 of 0
    s.extend_from_slice(body);
    let crc = crc32_mpeg(&s);
    s.extend_from_slice(&crc.to_be_bytes());
    let mut payload = vec![0]; // pointer_field
    payload.extend(s);
    payload.resize(184, 0xFF);
    let mut pkt = vec![0x47, 0x40 | (pid >> 8) as u8, pid as u8, 0x10 | *cc];
    pkt.extend(payload);
    out.extend(pkt);
    *cc = (*cc + 1) & 0x0F;
}

/// An MPEG transport stream (ISO/IEC 13818-1) carrying `video` on PID 0x100
/// as `stream_type` (0x1B H.264, 0x02 MPEG-2 video), one PES per picture
/// with its PTS, the PCR on the video PID, PAT and PMT before every key
/// picture. Pictures must be in display order (no B pictures).
pub fn ts(video: &[Coded], stream_type: u8, fps: u32) -> Vec<u8> {
    ts_av(video, stream_type, fps, None)
}

/// An audio elementary stream for [`ts_av`]: its PMT `stream_type`, PES
/// `stream_id`, sample rate, and access units with their durations in
/// samples, one PES each.
pub struct TsAudio<'a> {
    pub stream_type: u8,
    pub stream_id: u8,
    pub rate: u32,
    pub units: &'a [(Vec<u8>, u32)],
}

/// The PES header (with a PTS) for `stream_id`.
fn pes_with_pts(stream_id: u8, pts: u64) -> Vec<u8> {
    let mut pes = vec![0, 0, 1, stream_id, 0, 0, 0x80, 0x80, 5];
    pes.extend_from_slice(&[
        0x21 | ((pts >> 29) & 0x0E) as u8,
        (pts >> 22) as u8,
        0x01 | ((pts >> 14) & 0xFE) as u8,
        (pts >> 7) as u8,
        0x01 | ((pts << 1) & 0xFE) as u8,
    ]);
    pes
}

/// [`ts`] with an audio track on PID 0x101 as well, its PES packets
/// interleaved with the pictures by PTS (the audio starting with the video).
pub fn ts_av(video: &[Coded], stream_type: u8, fps: u32, audio: Option<TsAudio>) -> Vec<u8> {
    const PMT: u16 = 0x1000;
    const VIDEO: u16 = 0x100;
    const AUDIO: u16 = 0x101;
    let mut out = Vec::new();
    let (mut cc_pat, mut cc_pmt, mut cc_v, mut cc_a) = (0u8, 0u8, 0u8, 0u8);
    // The audio PES packets, by PTS.
    let mut audio_pes: Vec<(u64, Vec<u8>)> = Vec::new();
    if let Some(a) = &audio {
        let mut samples = 0u64;
        for (unit, duration) in a.units {
            let pts = 126_000 + samples * 90_000 / u64::from(a.rate);
            let mut pes = pes_with_pts(a.stream_id, pts);
            pes.extend_from_slice(unit);
            audio_pes.push((pts, pes));
            samples += u64::from(*duration);
        }
    }
    let mut next_audio = 0;
    for c in video {
        let pts = 126_000 + c.pts * 90_000 / u64::from(fps);
        if c.key || out.is_empty() {
            psi(&mut out, 0, &mut cc_pat, 0x00, 1, &[0, 1, 0xE0 | (PMT >> 8) as u8, PMT as u8]);
            let mut pmt = vec![
                0xE0 | (VIDEO >> 8) as u8,
                VIDEO as u8,
                0xF0,
                0x00, // program_info_length 0
                stream_type,
                0xE0 | (VIDEO >> 8) as u8,
                VIDEO as u8,
                0xF0,
                0x00, // ES_info_length 0
            ];
            if let Some(a) = &audio {
                pmt.extend_from_slice(&[a.stream_type, 0xE0 | (AUDIO >> 8) as u8, AUDIO as u8, 0xF0, 0x00]);
            }
            psi(&mut out, PMT, &mut cc_pmt, 0x02, 1, &pmt);
        }
        let mut pes = pes_with_pts(0xE0, pts);
        if stream_type == 0x1B {
            pes.extend_from_slice(&[0, 0, 0, 1, 0x09, 0xF0]); // access unit delimiter
        }
        pes.extend_from_slice(&c.data);
        ts_packets(&mut out, VIDEO, &mut cc_v, &pes, Some(pts - 9_000));
        // The audio up to the next picture.
        let until = 126_000 + (c.pts + 1) * 90_000 / u64::from(fps);
        while next_audio < audio_pes.len() && audio_pes[next_audio].0 < until {
            ts_packets(&mut out, AUDIO, &mut cc_a, &audio_pes[next_audio].1, None);
            next_audio += 1;
        }
    }
    for (_, pes) in &audio_pes[next_audio..] {
        ts_packets(&mut out, AUDIO, &mut cc_a, pes, None);
    }
    out
}

// ---- clips -----------------------------------------------------------------

/// The usual synthetic source: `seconds` of [`test_pattern`] at `w x h` and
/// `fps`, noise of `noise` (0 for none), H.264 at `bitrate` (0 for a fixed
/// quantiser of 20), with `audio` seconds of a 440 Hz stereo AAC track when
/// asked for. An MP4.
pub fn clip(w: u32, h: u32, fps: u32, seconds: f64, noise: u8, bitrate: u32, audio: bool) -> Vec<u8> {
    let n = (seconds * f64::from(fps)).round() as u64;
    let mut rng = Rng::new(1);
    let pictures = (0..n).map(|t| {
        let mut p = test_pattern(w, h, t, ChromaFormat::Yuv420);
        add_noise(&mut p, noise, &mut rng);
        p
    });
    let cfg = H264 { bitrate, ..H264::new(w, h, fps) };
    let video = encode_h264(&cfg, pictures);
    let track = audio.then(|| aac_sine(440.0, seconds, 2, 128_000));
    mp4(&video, w, h, fps, track.as_ref(), None)
}
