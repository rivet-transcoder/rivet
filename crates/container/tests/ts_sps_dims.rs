//! Integration tests for TS demux width/height recovery.
//!
//! TS carries no container-level sample-entry width/height — the
//! dimensions live only in the H.264 SPS / HEVC SPS / MPEG-2 sequence
//! header inside the first video sample. These tests build a synthetic
//! TS stream with a known-dims SPS and assert the demuxer recovers them
//! (both the legacy `demux::demux` path and the streaming
//! `streaming::demux_streaming` path). Regression guard for
//! `PROBLEMS.md`'s "MPEG-TS demux — width/height never populated"
//! entry (closed 2026-04-18).

use container::{demux, streaming};

const TS_PACKET: usize = 188;
const TS_SYNC: u8 = 0x47;
const STREAM_TYPE_MPEG2_VIDEO: u8 = 0x02;
const STREAM_TYPE_H264: u8 = 0x1B;
const STREAM_TYPE_HEVC: u8 = 0x24;

// ─── Local BitWriter helper (mirrors pixel_format.rs tests) ────────
// Duplicated here rather than exposed from codec as a public API —
// test scaffolding, not production surface. Matches BitReader's
// MSB-first layout so synthesised SPS bytes parse bitwise-identical.

struct BitWriter {
    bytes: Vec<u8>,
    bit_pos: usize,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_pos: 8,
        }
    }
    fn write_bit(&mut self, b: u8) {
        if self.bit_pos == 8 {
            self.bytes.push(0);
            self.bit_pos = 0;
        }
        if b != 0 {
            let idx = self.bytes.len() - 1;
            self.bytes[idx] |= 1 << (7 - self.bit_pos);
        }
        self.bit_pos += 1;
    }
    fn write_bits(&mut self, val: u64, n: usize) {
        for i in 0..n {
            let bit = ((val >> (n - 1 - i)) & 1) as u8;
            self.write_bit(bit);
        }
    }
    fn write_ue(&mut self, v: u32) {
        let z = if v == 0 { 0 } else { (v + 1).ilog2() as usize };
        for _ in 0..z {
            self.write_bit(0);
        }
        self.write_bit(1);
        if z > 0 {
            let suffix = ((v + 1) - (1u32 << z)) as u64;
            self.write_bits(suffix, z);
        }
    }
    fn bytes(self) -> Vec<u8> {
        self.bytes
    }
}

fn build_h264_baseline_sps_1280x720() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write_bits(66, 8); // profile_idc = Baseline
    w.write_bits(0, 8);
    w.write_bits(30, 8); // level_idc
    w.write_ue(0); // sps_id
    w.write_ue(0); // log2_max_frame_num_minus4
    w.write_ue(0); // pic_order_cnt_type
    w.write_ue(0); // log2_max_pic_order_cnt_lsb_minus4
    w.write_ue(1); // max_num_ref_frames
    w.write_bit(0); // gaps_in_frame_num
    w.write_ue(1280 / 16 - 1);
    w.write_ue(720 / 16 - 1);
    w.write_bit(1); // frame_mbs_only_flag
    w.write_bit(1); // direct_8x8_inference_flag
    w.write_bit(0); // frame_cropping_flag
    w.write_bit(0); // vui_parameters_present_flag
    w.write_bit(1); // stop bit
    let mut nal = vec![0x67u8]; // NAL header: forbidden=0, nal_ref_idc=3, type=7
    nal.extend_from_slice(&w.bytes());
    nal
}

fn build_hevc_sps_1920x1080() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write_bits(0, 4);
    w.write_bits(0, 3);
    w.write_bits(1, 1);
    // profile_tier_level (max_sub_layers_minus1=0)
    w.write_bits(0b0_0_00001, 8);
    w.write_bits(0x40000000, 32);
    w.write_bits(0, 48);
    w.write_bits(93, 8);
    w.write_ue(0); // sps_seq_parameter_set_id
    w.write_ue(1); // chroma_format_idc
    w.write_ue(1920);
    w.write_ue(1080);
    w.write_bit(0); // conformance_window_flag
    w.write_ue(0);
    w.write_ue(0); // bit depths
    w.write_bit(1); // stop bit
    let mut nal = vec![0x42u8, 0x01u8]; // HEVC NAL header: type=33 (SPS)
    nal.extend_from_slice(&w.bytes());
    nal
}

fn build_mpeg2_sequence_header_1920x1080() -> Vec<u8> {
    // sequence_header_code prefix + 3-byte 12+12 dims. 1920=0x780,
    // 1080=0x438: byte0=0x78, byte1=0x04, byte2=0x38. Then 4 bits
    // aspect_ratio_information + 4 bits frame_rate_code + 32 bits
    // bit_rate+marker+vbv_buffer_size+constrained — filler.
    vec![
        0x00, 0x00, 0x01, 0xB3, 0x78, 0x04, 0x38, 0x13, 0xFF, 0xFF, 0xFF,
        0xF0, // aspect/fps + bit_rate + marker
    ]
}

// ─── TS packet helpers ────────────────────────────────────────────

fn ts_pkt(pid: u16, pusi: bool, payload: &[u8]) -> [u8; TS_PACKET] {
    let mut p = [0xFFu8; TS_PACKET];
    p[0] = TS_SYNC;
    p[1] = if pusi { 0x40 } else { 0x00 } | ((pid >> 8) & 0x1F) as u8;
    p[2] = (pid & 0xFF) as u8;
    p[3] = 0x10;
    let pay_len = payload.len().min(TS_PACKET - 4);
    p[4..4 + pay_len].copy_from_slice(&payload[..pay_len]);
    p
}

fn make_pes(es: &[u8]) -> Vec<u8> {
    make_pes_with_pts(es, 0)
}

/// Encode a 33-bit PTS into the 5-byte TS/PES layout:
/// `0010 | PTS[32..30] | 1 | PTS[29..15] | 1 | PTS[14..0] | 1`
/// (ISO/IEC 13818-1 §2.4.3.7).
fn encode_pts(pts: u64) -> [u8; 5] {
    let p32_30 = ((pts >> 30) & 0x07) as u8;
    let p29_15 = ((pts >> 15) & 0x7FFF) as u16;
    let p14_0 = (pts & 0x7FFF) as u16;
    [
        0x20 | (p32_30 << 1) | 0x01,
        ((p29_15 >> 7) & 0xFF) as u8,
        (((p29_15 & 0x7F) as u8) << 1) | 0x01,
        ((p14_0 >> 7) & 0xFF) as u8,
        (((p14_0 & 0x7F) as u8) << 1) | 0x01,
    ]
}

fn make_pes_with_pts(es: &[u8], pts_ticks: u64) -> Vec<u8> {
    let mut pes = vec![0u8, 0u8, 1u8, 0xE0];
    pes.extend_from_slice(&[0u8, 0u8]);
    pes.push(0x80);
    pes.push(0x80);
    pes.push(5);
    pes.extend_from_slice(&encode_pts(pts_ticks));
    pes.extend_from_slice(es);
    pes
}

fn build_ts_with_video(stream_type: u8, es_bytes: Vec<u8>) -> Vec<u8> {
    let mut pat = Vec::new();
    pat.push(0x00);
    let pat_len: usize = 5 + 4 + 4;
    pat.push(0xB0 | ((pat_len >> 8) & 0x0F) as u8);
    pat.push((pat_len & 0xFF) as u8);
    pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pat.extend_from_slice(&[0x00, 0x01]);
    pat.extend_from_slice(&[0xE1, 0x00]);
    pat.extend_from_slice(&[0u8; 4]);
    let mut pat_payload = vec![0u8];
    pat_payload.extend_from_slice(&pat);
    let pat_pkt = ts_pkt(0x0000, true, &pat_payload);

    let mut pmt = Vec::new();
    pmt.push(0x02);
    let pmt_len: usize = 9 + 5 + 4;
    pmt.push(0xB0 | ((pmt_len >> 8) & 0x0F) as u8);
    pmt.push((pmt_len & 0xFF) as u8);
    pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pmt.extend_from_slice(&[0xE2, 0x00]);
    pmt.extend_from_slice(&[0xF0, 0x00]);
    pmt.extend_from_slice(&[stream_type, 0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[0u8; 4]);
    let mut pmt_payload = vec![0u8];
    pmt_payload.extend_from_slice(&pmt);
    let pmt_pkt = ts_pkt(0x0100, true, &pmt_payload);

    // First PES: ES bytes preceded by Annex-B start code so the SPS
    // is exactly where the parser expects to find it (TS carries H.264
    // / HEVC Annex-B in-band, not length-prefixed).
    let mut au_a = Vec::new();
    au_a.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    au_a.extend_from_slice(&es_bytes);
    let pes_a = ts_pkt(0x0200, true, &make_pes(&au_a));
    // Second PES closes the first AU so `scan_first_video_au` returns
    // on the second PUSI.
    let mut au_b = Vec::new();
    au_b.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x65]); // IDR slice (any non-SPS NAL)
    au_b.extend_from_slice(&[0u8; 16]);
    let pes_b = ts_pkt(0x0200, true, &make_pes(&au_b));

    let mut buf = Vec::new();
    buf.extend_from_slice(&pat_pkt);
    buf.extend_from_slice(&pmt_pkt);
    buf.extend_from_slice(&pes_a);
    buf.extend_from_slice(&pes_b);
    for _ in 0..3 {
        buf.extend_from_slice(&ts_pkt(0x1FFF, false, &[]));
    }
    buf
}

// ─── Tests ────────────────────────────────────────────────────────

#[test]
fn demux_ts_h264_recovers_sps_dims() {
    let sps = build_h264_baseline_sps_1280x720();
    let buf = build_ts_with_video(STREAM_TYPE_H264, sps);
    let result = demux::demux(&buf).expect("demux");
    assert_eq!(result.codec, "h264");
    assert_eq!(result.info.width, 1280, "width recovered from H.264 SPS");
    assert_eq!(result.info.height, 720, "height recovered from H.264 SPS");
}

#[test]
fn demux_ts_hevc_recovers_sps_dims() {
    let sps = build_hevc_sps_1920x1080();
    let buf = build_ts_with_video(STREAM_TYPE_HEVC, sps);
    let result = demux::demux(&buf).expect("demux");
    assert_eq!(result.codec, "h265");
    assert_eq!(result.info.width, 1920);
    assert_eq!(result.info.height, 1080);
}

#[test]
fn demux_ts_mpeg2_recovers_sequence_header_dims() {
    let seq = build_mpeg2_sequence_header_1920x1080();
    let buf = build_ts_with_video(STREAM_TYPE_MPEG2_VIDEO, seq);
    let result = demux::demux(&buf).expect("demux");
    assert_eq!(result.codec, "mpeg2");
    assert_eq!(result.info.width, 1920);
    assert_eq!(result.info.height, 1080);
}

#[test]
fn streaming_ts_h264_populates_header_dims_before_first_sample() {
    let sps = build_h264_baseline_sps_1280x720();
    let buf = build_ts_with_video(STREAM_TYPE_H264, sps);
    let demuxer = streaming::demux_streaming(&buf).expect("streaming init");
    let header = demuxer.header();
    assert_eq!(header.codec, "h264");
    assert_eq!(
        header.info.width, 1280,
        "streaming init populates width BEFORE any next_video_sample call"
    );
    assert_eq!(header.info.height, 720);
}

#[test]
fn streaming_ts_hevc_populates_header_dims_before_first_sample() {
    let sps = build_hevc_sps_1920x1080();
    let buf = build_ts_with_video(STREAM_TYPE_HEVC, sps);
    let demuxer = streaming::demux_streaming(&buf).expect("streaming init");
    let header = demuxer.header();
    assert_eq!(header.codec, "h265");
    assert_eq!(header.info.width, 1920);
    assert_eq!(header.info.height, 1080);
}

#[test]
fn streaming_ts_mpeg2_populates_header_dims_before_first_sample() {
    let seq = build_mpeg2_sequence_header_1920x1080();
    let buf = build_ts_with_video(STREAM_TYPE_MPEG2_VIDEO, seq);
    let demuxer = streaming::demux_streaming(&buf).expect("streaming init");
    let header = demuxer.header();
    assert_eq!(header.codec, "mpeg2");
    assert_eq!(header.info.width, 1920);
    assert_eq!(header.info.height, 1080);
}

/// Variant of `build_ts_with_video` that emits N PES packets with
/// consecutive PTSes spaced `pts_delta` apart. Lets the demuxer's
/// frame_rate estimator see a stable inter-PTS window.
fn build_ts_with_n_video_pes(
    stream_type: u8,
    first_es: Vec<u8>,
    pts_delta: u64,
    n: usize,
) -> Vec<u8> {
    let mut pat = Vec::new();
    pat.push(0x00);
    let pat_len: usize = 5 + 4 + 4;
    pat.push(0xB0 | ((pat_len >> 8) & 0x0F) as u8);
    pat.push((pat_len & 0xFF) as u8);
    pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pat.extend_from_slice(&[0x00, 0x01]);
    pat.extend_from_slice(&[0xE1, 0x00]);
    pat.extend_from_slice(&[0u8; 4]);
    let mut pat_payload = vec![0u8];
    pat_payload.extend_from_slice(&pat);
    let pat_pkt = ts_pkt(0x0000, true, &pat_payload);

    let mut pmt = Vec::new();
    pmt.push(0x02);
    let pmt_len: usize = 9 + 5 + 4;
    pmt.push(0xB0 | ((pmt_len >> 8) & 0x0F) as u8);
    pmt.push((pmt_len & 0xFF) as u8);
    pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pmt.extend_from_slice(&[0xE2, 0x00]);
    pmt.extend_from_slice(&[0xF0, 0x00]);
    pmt.extend_from_slice(&[stream_type, 0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[0u8; 4]);
    let mut pmt_payload = vec![0u8];
    pmt_payload.extend_from_slice(&pmt);
    let pmt_pkt = ts_pkt(0x0100, true, &pmt_payload);

    let mut buf = Vec::new();
    buf.extend_from_slice(&pat_pkt);
    buf.extend_from_slice(&pmt_pkt);

    for i in 0..n {
        let mut au = Vec::new();
        au.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        if i == 0 {
            au.extend_from_slice(&first_es);
        } else {
            au.push(0x41); // non-IDR slice (placeholder)
            au.extend_from_slice(&[0u8; 12]);
        }
        let pts = i as u64 * pts_delta;
        buf.extend_from_slice(&ts_pkt(0x0200, true, &make_pes_with_pts(&au, pts)));
    }
    for _ in 0..3 {
        buf.extend_from_slice(&ts_pkt(0x1FFF, false, &[]));
    }
    buf
}

#[test]
fn streaming_ts_h264_frame_rate_24fps_from_pts_window() {
    // 24 fps → 90000/24 = 3750 ticks per frame. The streaming
    // demuxer's scan collects up to 64 PTSes during init; three is
    // enough for median-of-deltas to land on 3750.
    let sps = build_h264_baseline_sps_1280x720();
    let buf = build_ts_with_n_video_pes(STREAM_TYPE_H264, sps, 3750, 8);
    let demuxer = streaming::demux_streaming(&buf).expect("streaming init");
    let fr = demuxer.header().info.frame_rate;
    assert!((fr - 24.0).abs() < 1e-6, "expected 24.0 fps, got {}", fr);
}

#[test]
fn streaming_ts_h264_frame_rate_30fps_from_pts_window() {
    // 30 fps → 90000/30 = 3000 ticks per frame.
    let sps = build_h264_baseline_sps_1280x720();
    let buf = build_ts_with_n_video_pes(STREAM_TYPE_H264, sps, 3000, 8);
    let demuxer = streaming::demux_streaming(&buf).expect("streaming init");
    let fr = demuxer.header().info.frame_rate;
    assert!((fr - 30.0).abs() < 1e-6, "expected 30.0 fps, got {}", fr);
}

#[test]
fn streaming_ts_h264_frame_rate_falls_back_to_30_when_only_one_pts() {
    // Only 2 PES packets → 2 PTSes. That's the minimum for a delta;
    // here we set delta=3750 so fps=24. Regression guard that the
    // "only one PUSI" corner case does NOT crash.
    let sps = build_h264_baseline_sps_1280x720();
    let buf = build_ts_with_n_video_pes(STREAM_TYPE_H264, sps, 3750, 2);
    let demuxer = streaming::demux_streaming(&buf).expect("streaming init");
    let fr = demuxer.header().info.frame_rate;
    assert!(
        (fr - 24.0).abs() < 1e-6,
        "two PTSes still derive a rate, got {}",
        fr
    );
}

#[test]
fn demux_ts_first_sample_without_sps_leaves_dims_at_zero_not_error() {
    // First video sample contains only a non-IDR slice NAL (type=1) —
    // no SPS anywhere in the AU. `find_h264_sps` returns None, our
    // `detect_dims` bails, and the demuxer contract says: don't
    // hard-fail — surface 0×0 so the encoder reports the miss instead
    // (matches the "parse-failure non-fatal" pattern the PROBLEMS.md
    // close-out documented).
    let mut no_sps = vec![0x41u8]; // NAL type=1 (non-IDR slice)
    no_sps.extend_from_slice(&[0u8; 32]);
    let buf = build_ts_with_video(STREAM_TYPE_H264, no_sps);
    let result = demux::demux(&buf).expect("demux should succeed even without an SPS");
    assert_eq!(result.info.width, 0);
    assert_eq!(result.info.height, 0);
}
