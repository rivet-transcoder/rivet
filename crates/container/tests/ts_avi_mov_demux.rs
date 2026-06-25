//! Integration tests for the three new container routes added under
//! task #71: MPEG-TS, AVI, and the MOV → ProRes fourcc plumbing.
//!
//! All three tests build synthetic byte streams in-process — no
//! `test_media/` dependency. The shape of the synthetic input is just
//! enough to exercise the public `demux::demux()` dispatcher, the
//! magic-byte detector, and the per-format demuxer's happy path.
//! Heavier real-media coverage lives in
//! `crates/codec/tests/decode_integration.rs` (gated on test_media/).

use container::demux;

// ---------------------------------------------------------------------------
// MPEG-TS
// ---------------------------------------------------------------------------

const TS_PACKET: usize = 188;
const TS_SYNC: u8 = 0x47;
const STREAM_TYPE_MPEG2_VIDEO: u8 = 0x02;

fn ts_pkt(pid: u16, pusi: bool, payload: &[u8]) -> [u8; TS_PACKET] {
    let mut p = [0xFFu8; TS_PACKET];
    p[0] = TS_SYNC;
    p[1] = if pusi { 0x40 } else { 0x00 } | ((pid >> 8) & 0x1F) as u8;
    p[2] = (pid & 0xFF) as u8;
    p[3] = 0x10; // adaptation = 01 (payload only), continuity = 0
    let pay_len = payload.len().min(TS_PACKET - 4);
    p[4..4 + pay_len].copy_from_slice(&payload[..pay_len]);
    p
}

/// Build a minimal valid TS file with PAT+PMT pointing at a video PID
/// carrying two PES packets so the dispatcher's reassembly path runs
/// to completion.
fn build_minimal_ts() -> Vec<u8> {
    // PAT: program 1 → PMT PID 0x100
    let mut pat = Vec::new();
    pat.push(0x00); // table_id
    let pat_section_len: usize = 5 + 4 + 4; // header + program + CRC
    pat.push(0xB0 | ((pat_section_len >> 8) & 0x0F) as u8);
    pat.push((pat_section_len & 0xFF) as u8);
    pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pat.extend_from_slice(&[0x00, 0x01]); // program_number
    pat.extend_from_slice(&[0xE1, 0x00]); // PMT PID = 0x100
    pat.extend_from_slice(&[0u8; 4]); // CRC placeholder
    let mut pat_payload = vec![0u8]; // pointer_field
    pat_payload.extend_from_slice(&pat);
    let pat_pkt = ts_pkt(0x0000, true, &pat_payload);

    // PMT: program 1 → MPEG-2 video on PID 0x200.
    let mut pmt = Vec::new();
    pmt.push(0x02);
    let pmt_section_len: usize = 9 + 5 + 4; // hdr + 1 stream + CRC
    pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
    pmt.push((pmt_section_len & 0xFF) as u8);
    pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pmt.extend_from_slice(&[0xE2, 0x00]); // PCR PID
    pmt.extend_from_slice(&[0xF0, 0x00]); // program_info_length
    pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[0u8; 4]); // CRC placeholder
    let mut pmt_payload = vec![0u8];
    pmt_payload.extend_from_slice(&pmt);
    let pmt_pkt = ts_pkt(0x0100, true, &pmt_payload);

    // Two PES packets, each 32 bytes of distinct ES.
    let make_pes = |fill: u8| {
        let mut pes = vec![0u8, 0u8, 1u8, 0xE0]; // start code + video stream_id
        pes.extend_from_slice(&[0u8, 0u8]); // PES_packet_length = 0 (unbounded)
        pes.push(0x80); // first PES flag byte
        pes.push(0x80); // PTS_DTS_flags = 10
        pes.push(5); // PES_header_data_length
        pes.extend_from_slice(&[0x21, 0x00, 0x01, 0x00, 0x01]); // PTS = 0
        pes.extend_from_slice(&[fill; 32]);
        pes
    };
    let pes_a = ts_pkt(0x0200, true, &make_pes(0xAA));
    let pes_b = ts_pkt(0x0200, true, &make_pes(0xBB));

    let mut buf = Vec::new();
    buf.extend_from_slice(&pat_pkt);
    buf.extend_from_slice(&pmt_pkt);
    buf.extend_from_slice(&pes_a);
    buf.extend_from_slice(&pes_b);
    // A pair of trailing null packets so the magic-byte detector sees
    // 0x47 at offsets 0, 188, 376, 564, 752 (well beyond its 376 ceiling).
    for _ in 0..3 {
        buf.extend_from_slice(&ts_pkt(0x1FFF, false, &[]));
    }
    buf
}

#[test]
fn dispatcher_routes_ts_through_magic_byte_detection() {
    let buf = build_minimal_ts();
    let result = demux::demux(&buf).expect("demux dispatcher must accept synthetic TS");
    assert_eq!(result.codec, "mpeg2", "PMT stream_type 0x02 → mpeg2 label");
    assert_eq!(result.samples.len(), 2, "two PES packets → two samples");
    assert!(
        result.samples[0].starts_with(&[0xAA; 16]),
        "first sample reassembled from PES_A payload"
    );
    assert!(
        result.samples[1].starts_with(&[0xBB; 16]),
        "second sample reassembled from PES_B payload"
    );
    // No audio passthrough when PMT advertises only video — Squad-27.
    // (When PMT carries AAC-ADTS too, audio is now surfaced — see
    // dispatcher_surfaces_aac_audio_track_from_ts below.)
    assert!(result.audio.is_none(), "video-only TS → no audio track");
}

// ---------------------------------------------------------------------------
// MPEG-TS — AAC-ADTS audio extraction (Squad-27)
// ---------------------------------------------------------------------------

const STREAM_TYPE_AAC_ADTS: u8 = 0x0F;

/// Build a 7-byte ADTS header (no CRC). Same shape as the in-module test
/// helper; duplicated here because `ts.rs` test helpers are private.
fn build_adts_header_7(profile: u8, sr_idx: u8, ch_cfg: u8, frame_length: usize) -> [u8; 7] {
    let mut h = [0u8; 7];
    h[0] = 0xFF;
    h[1] = 0xF0 | 0x01; // protection_absent = 1
    h[2] = ((profile & 0x03) << 6) | ((sr_idx & 0x0F) << 2) | ((ch_cfg >> 2) & 0x01);
    h[3] = ((ch_cfg & 0x03) << 6) | (((frame_length >> 11) & 0x03) as u8);
    h[4] = ((frame_length >> 3) & 0xFF) as u8;
    h[5] = (((frame_length & 0x07) << 5) | 0x1F) as u8;
    h[6] = 0xFC;
    h
}

#[test]
fn dispatcher_surfaces_aac_audio_track_from_ts() {
    // Build a TS with PAT → PMT(MPEG-2 video on 0x200, AAC-ADTS on 0x300),
    // a video PES (so the demuxer's mandatory video bail doesn't trip),
    // and an audio PES carrying three ADTS frames (LC, 48k, stereo, 64-byte
    // payloads). Verify the dispatcher hands back audio with the
    // synthesized ASC + stripped frames + per-frame durations.
    let mut pat = Vec::new();
    pat.push(0x00);
    let pat_section_len: usize = 5 + 4 + 4;
    pat.push(0xB0 | ((pat_section_len >> 8) & 0x0F) as u8);
    pat.push((pat_section_len & 0xFF) as u8);
    pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pat.extend_from_slice(&[0x00, 0x01, 0xE1, 0x00, 0u8, 0u8, 0u8, 0u8]);
    let mut pat_payload = vec![0u8];
    pat_payload.extend_from_slice(&pat);
    let pat_pkt = ts_pkt(0x0000, true, &pat_payload);

    let mut pmt = Vec::new();
    pmt.push(0x02);
    let pmt_section_len: usize = 9 + 5 + 5 + 4;
    pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
    pmt.push((pmt_section_len & 0xFF) as u8);
    pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pmt.extend_from_slice(&[0xE2, 0x00]);
    pmt.extend_from_slice(&[0xF0, 0x00]);
    pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[STREAM_TYPE_AAC_ADTS, 0xE3, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[0u8; 4]);
    let mut pmt_payload = vec![0u8];
    pmt_payload.extend_from_slice(&pmt);
    let pmt_pkt = ts_pkt(0x0100, true, &pmt_payload);

    let video_pes = {
        let mut pes = vec![
            0u8, 0u8, 1u8, 0xE0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
        ];
        pes.extend_from_slice(&[0xAAu8; 32]);
        pes
    };
    let video_pkt = ts_pkt(0x0200, true, &video_pes);

    // Three ADTS frames, each 7-byte header + 16-byte payload = 23 bytes.
    // The whole audio PES (header + 3 frames = 14 + 69 = 83 bytes) must
    // fit inside one TS packet's payload (~184 bytes after the 4-byte
    // header). Distinct fill bytes verify the strip preserves frame
    // boundaries.
    let mut adts_stream = Vec::new();
    for fill in [0xC1u8, 0xC2u8, 0xC3u8] {
        adts_stream.extend_from_slice(&build_adts_header_7(1, 3, 2, 23));
        adts_stream.extend_from_slice(&[fill; 16]);
    }
    let audio_pes = {
        let mut pes = vec![
            0u8, 0u8, 1u8, 0xC0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
        ];
        pes.extend_from_slice(&adts_stream);
        pes
    };
    let audio_pkt = ts_pkt(0x0300, true, &audio_pes);

    let mut buf = Vec::new();
    buf.extend_from_slice(&pat_pkt);
    buf.extend_from_slice(&pmt_pkt);
    buf.extend_from_slice(&video_pkt);
    buf.extend_from_slice(&audio_pkt);
    for _ in 0..3 {
        buf.extend_from_slice(&ts_pkt(0x1FFF, false, &[]));
    }

    let result = demux::demux(&buf).expect("dispatcher must accept TS with AAC audio");
    assert_eq!(result.codec, "mpeg2");

    let audio = result.audio.expect("AAC audio should be surfaced");
    assert_eq!(audio.codec, "aac");
    assert_eq!(audio.sample_rate, 48000);
    assert_eq!(audio.timescale, 48000);
    assert_eq!(audio.channels, 2);
    assert_eq!(
        audio.asc,
        vec![0x11, 0x90],
        "synthesized ASC for LC/48k/stereo"
    );
    assert_eq!(audio.samples.len(), 3, "three ADTS frames → three samples");
    for (i, fill) in [0xC1u8, 0xC2u8, 0xC3u8].iter().enumerate() {
        assert_eq!(
            audio.samples[i].len(),
            16,
            "frame {} payload after 7-byte ADTS strip",
            i
        );
        assert!(
            audio.samples[i].iter().all(|b| b == fill),
            "frame {} payload not preserved",
            i
        );
    }
    assert_eq!(
        audio.durations,
        vec![1024, 1024, 1024],
        "AAC-LC frame duration = 1024 ticks at sample-rate timescale"
    );
}

/// Real-media end-to-end TS+AAC demux test. Gracefully skips when the
/// asset isn't on the host (test_media/ is gitignored). When the file
/// IS present, asserts the AAC audio track is surfaced with a non-empty
/// ASC and a sensible sample count — proving the synthetic-bitstream
/// path matches a conformant encoder's wire format.
#[test]
fn real_media_ts_with_aac_yields_audio_track() {
    use std::path::Path;
    let candidates = [
        "test_media/aac_in_ts.ts",
        "test_media/sample_aac.ts",
        "test_media/h264_aac_sample.ts",
        "test_media/aac_h264.m2ts",
    ];
    let path = candidates.iter().map(Path::new).find(|p| p.exists());
    let Some(path) = path else {
        eprintln!(
            "real_media_ts_with_aac_yields_audio_track: no test_media TS asset present, skipping"
        );
        return;
    };
    let bytes = std::fs::read(path).expect("read TS asset");
    let result = demux::demux(&bytes).expect("demux real TS file");
    let Some(audio) = result.audio else {
        // Real assets vary — some TS files are video-only. That's not a
        // failure mode we want to gate on; just log and move on.
        eprintln!(
            "real TS asset {} has no AAC audio track in PMT — passthrough check skipped",
            path.display()
        );
        return;
    };
    assert_eq!(audio.codec, "aac");
    assert!(!audio.asc.is_empty(), "synthesized ASC must be non-empty");
    assert!(
        !audio.samples.is_empty(),
        "must surface at least one AAC frame"
    );
    assert_eq!(
        audio.samples.len(),
        audio.durations.len(),
        "samples and durations must be parallel"
    );
    assert_eq!(
        audio.timescale, audio.sample_rate,
        "AAC mdhd timescale = sample_rate"
    );
}

#[test]
fn dispatcher_rejects_non_ts_buffer_starting_with_0x47() {
    // A single 0x47 byte at offset 0 with no follow-up sync points must
    // NOT be misidentified as TS — the detector demands the sync at
    // offset 188 too.
    let mut buf = vec![0u8; 1024];
    buf[0] = 0x47;
    match demux::demux(&buf) {
        Ok(_) => panic!("1024 bytes of zeros must not parse as TS"),
        Err(err) => {
            let msg = format!("{err:#}");
            // Either "unsupported container" (preferred) or some specific
            // demuxer error — but absolutely not a successful TS parse.
            assert!(
                msg.contains("unsupported container") || msg.contains("TS:"),
                "unexpected error path: {msg}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// MPEG-TS — AC-3 audio routing (Squad-37)
// ---------------------------------------------------------------------------

const STREAM_TYPE_AC3: u8 = 0x81;

/// Build a minimal AC-3 syncframe: stereo 48 kHz @ 128 kbps, frame
/// size = 384 bytes per ETSI TS 102 366 Table F.7. Bit layout matches
/// the in-module `synth_ac3_frame_stereo_48k_128k` helper.
fn synth_ac3_frame_stereo_48k_128k() -> Vec<u8> {
    let mut bytes = vec![0u8; 384];
    // Build the BSI prefix bit-by-bit into a small temp buffer, then
    // copy into the start of the 384-byte frame.
    let mut bw = MsbWriter::new();
    bw.put(16, 0x0B77); // syncword
    bw.put(16, 0); // crc1
    bw.put(2, 0); // fscod=0 → 48k
    bw.put(6, 8 << 1); // frmsizecod = bit_rate_code(8) << 1
    bw.put(5, 8); // bsid
    bw.put(3, 0); // bsmod
    bw.put(3, 2); // acmod=2 stereo
    bw.put(2, 0); // dsurmod (acmod==2)
    bw.put(1, 0); // lfeon
    let prefix = bw.finish();
    bytes[..prefix.len()].copy_from_slice(&prefix);
    bytes
}

struct MsbWriter {
    bytes: Vec<u8>,
    bit_pos: usize,
}
impl MsbWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_pos: 0,
        }
    }
    fn put(&mut self, n: usize, v: u32) {
        for i in (0..n).rev() {
            let bit = ((v >> i) & 0x01) as u8;
            if self.bit_pos % 8 == 0 {
                self.bytes.push(0);
            }
            let byte_idx = self.bit_pos / 8;
            let bit_idx = 7 - (self.bit_pos % 8);
            self.bytes[byte_idx] |= bit << bit_idx;
            self.bit_pos += 1;
        }
    }
    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

#[test]
fn dispatcher_routes_ts_with_ac3_audio_to_passthrough_track() {
    // Build a TS file with PAT → PMT(MPEG-2 video on 0x200, AC-3 on
    // 0x300 via stream_type 0x81 — the ATSC A/53 form) plus a
    // single-frame audio PES. The audio PES is split across multiple
    // TS packets via PUSI=1 → PUSI=0 continuation since the AC-3 frame
    // is 384 bytes (oversized for a single packet payload).

    // PAT
    let mut pat = vec![0x00];
    let pat_section_len: usize = 5 + 4 + 4;
    pat.push(0xB0 | ((pat_section_len >> 8) & 0x0F) as u8);
    pat.push((pat_section_len & 0xFF) as u8);
    pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pat.extend_from_slice(&[0x00, 0x01, 0xE1, 0x00, 0u8, 0u8, 0u8, 0u8]);
    let mut pat_payload = vec![0u8];
    pat_payload.extend_from_slice(&pat);
    let pat_pkt = ts_pkt(0x0000, true, &pat_payload);

    // PMT: video on 0x200 + AC-3 on 0x300 (stream_type 0x81).
    let mut pmt = vec![0x02];
    let pmt_section_len: usize = 9 + 5 + 5 + 4;
    pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
    pmt.push((pmt_section_len & 0xFF) as u8);
    pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pmt.extend_from_slice(&[0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[STREAM_TYPE_AC3, 0xE3, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[0u8; 4]);
    let mut pmt_payload = vec![0u8];
    pmt_payload.extend_from_slice(&pmt);
    let pmt_pkt = ts_pkt(0x0100, true, &pmt_payload);

    let video_pes = {
        let mut pes = vec![
            0u8, 0u8, 1u8, 0xE0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
        ];
        pes.extend_from_slice(&[0xAAu8; 32]);
        pes
    };
    let video_pkt = ts_pkt(0x0200, true, &video_pes);

    // Audio PES — one AC-3 frame.
    let frame = synth_ac3_frame_stereo_48k_128k();
    let mut audio_pes = vec![
        0u8, 0u8, 1u8, 0xC0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
    ];
    audio_pes.extend_from_slice(&frame);

    // Split across TS packets.
    let chunk = TS_PACKET - 4;
    let first_len = audio_pes.len().min(chunk);
    let mut audio_pkts: Vec<[u8; TS_PACKET]> = Vec::new();
    audio_pkts.push(ts_pkt(0x0300, true, &audio_pes[..first_len]));
    let mut cursor = first_len;
    while cursor < audio_pes.len() {
        let end = (cursor + chunk).min(audio_pes.len());
        // PUSI=0 continuation packet
        let mut pkt = ts_pkt(0x0300, false, &audio_pes[cursor..end]);
        // ts_pkt sets PUSI based on the bool — we already passed false,
        // so the packet is correct as-is.
        let _ = &mut pkt;
        audio_pkts.push(pkt);
        cursor = end;
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(&pat_pkt);
    buf.extend_from_slice(&pmt_pkt);
    buf.extend_from_slice(&video_pkt);
    for pkt in &audio_pkts {
        buf.extend_from_slice(pkt);
    }
    for _ in 0..3 {
        buf.extend_from_slice(&ts_pkt(0x1FFF, false, &[]));
    }

    let result = demux::demux(&buf).expect("dispatcher must accept TS with AC-3 audio");
    let audio = result.audio.expect("AC-3 audio must surface");
    assert_eq!(audio.codec, "ac3", "AC-3 PMT entry → codec=ac3");
    assert_eq!(audio.sample_rate, 48_000);
    assert_eq!(audio.channels, 2);
    assert_eq!(
        audio.codec_private.len(),
        3,
        "dac3 body is 3 bytes per ETSI §F.4"
    );
    assert!(
        !audio.samples.is_empty(),
        "at least one AC-3 frame extracted"
    );
    assert_eq!(
        &audio.samples[0][..2],
        &[0x0B, 0x77],
        "AC-3 frame begins with 0x0B77 sync word verbatim — passthrough preserves frame"
    );
}

// ---------------------------------------------------------------------------
// AVI
// ---------------------------------------------------------------------------

fn riff_chunk(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + payload.len() + 1);
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if out.len() & 1 == 1 {
        out.push(0); // word-align
    }
    out
}

fn riff_list(list_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(4 + payload.len());
    body.extend_from_slice(list_type);
    body.extend_from_slice(payload);
    riff_chunk(b"LIST", &body)
}

fn video_strl(
    handler: &[u8; 4],
    compression: &[u8; 4],
    w: u32,
    h: u32,
    rate: u32,
    scale: u32,
) -> Vec<u8> {
    let mut strh = Vec::with_capacity(56);
    strh.extend_from_slice(b"vids");
    strh.extend_from_slice(handler);
    strh.extend_from_slice(&[0u8; 12]); // flags/priority/lang/initialFrames
    strh.extend_from_slice(&scale.to_le_bytes());
    strh.extend_from_slice(&rate.to_le_bytes());
    strh.extend_from_slice(&[0u8; 24]); // start..rect
    let strh_chunk = riff_chunk(b"strh", &strh);

    let mut strf = Vec::with_capacity(40);
    strf.extend_from_slice(&40u32.to_le_bytes());
    strf.extend_from_slice(&(w as i32).to_le_bytes());
    strf.extend_from_slice(&(h as i32).to_le_bytes());
    strf.extend_from_slice(&1u16.to_le_bytes());
    strf.extend_from_slice(&24u16.to_le_bytes());
    strf.extend_from_slice(compression);
    strf.extend_from_slice(&[0u8; 20]);
    let strf_chunk = riff_chunk(b"strf", &strf);

    let mut strl_body = Vec::new();
    strl_body.extend_from_slice(&strh_chunk);
    strl_body.extend_from_slice(&strf_chunk);
    riff_list(b"strl", &strl_body)
}

fn build_minimal_avi(handler: &[u8; 4], compression: &[u8; 4]) -> Vec<u8> {
    let mut hdrl_body = Vec::new();
    hdrl_body.extend_from_slice(&riff_chunk(b"avih", &[0u8; 56]));
    hdrl_body.extend_from_slice(&video_strl(handler, compression, 640, 360, 25, 1));
    let hdrl = riff_list(b"hdrl", &hdrl_body);

    let mut movi_body = Vec::new();
    movi_body.extend_from_slice(&riff_chunk(b"00dc", b"sample-frame-1"));
    movi_body.extend_from_slice(&riff_chunk(b"00dc", b"sample-frame-2-bytes"));
    movi_body.extend_from_slice(&riff_chunk(b"01wb", b"audio-payload-ignored"));
    movi_body.extend_from_slice(&riff_chunk(b"00dc", b"sample-frame-3"));
    let movi = riff_list(b"movi", &movi_body);

    let mut riff_body = Vec::new();
    riff_body.extend_from_slice(b"AVI ");
    riff_body.extend_from_slice(&hdrl);
    riff_body.extend_from_slice(&movi);

    let mut file = Vec::new();
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&(riff_body.len() as u32).to_le_bytes());
    file.extend_from_slice(&riff_body);
    file
}

#[test]
fn dispatcher_routes_avi_xvid_to_mpeg4_codec() {
    let avi = build_minimal_avi(b"XVID", b"XVID");
    let result = demux::demux(&avi).expect("demux must accept synthetic AVI");
    assert_eq!(result.codec, "mpeg4", "XVID handler → mpeg4 (Part 2)");
    assert_eq!(result.info.width, 640);
    assert_eq!(result.info.height, 360);
    assert_eq!(result.samples.len(), 3, "three 00dc chunks → three samples");
    assert!(result.samples[0].starts_with(b"sample-frame-1"));
    assert!(result.samples[2].starts_with(b"sample-frame-3"));
}

#[test]
fn dispatcher_routes_avi_h264_to_h264_codec() {
    // AVI with H.264 fourcc — rare in the wild but a real path for
    // legacy GoPro / hardware-encoder workflows.
    let avi = build_minimal_avi(b"H264", b"H264");
    let result = demux::demux(&avi).expect("demux H264-AVI");
    assert_eq!(result.codec, "h264", "H264 handler → h264 codec label");
    assert!(!result.samples.is_empty());
}

#[test]
fn avi_handles_divx_family_fourccs() {
    // Spot-check the most common DivX/Xvid descendants — all should
    // route to the unified `mpeg4` decoder label.
    for fcc in [b"DIVX", b"DX50", b"DIV3", b"XviD", b"MP4V", b"M4S2"] {
        let avi = build_minimal_avi(fcc, fcc);
        let result =
            demux::demux(&avi).unwrap_or_else(|e| panic!("demux failed for fourcc {fcc:?}: {e:#}"));
        assert_eq!(
            result.codec,
            "mpeg4",
            "fourcc {:?} did not map to mpeg4",
            std::str::from_utf8(fcc).unwrap_or("?")
        );
    }
}

// ---------------------------------------------------------------------------
// MOV — ProRes fourcc plumbing (#71 deliverable 3)
// ---------------------------------------------------------------------------
//
// Real ProRes-in-MOV decode is exercised in
// `crates/codec/tests/decode_integration.rs::test_decode_prores_422_720p`
// against `test_media/prores_422_720p.mov`. That test is gated on the
// presence of the asset; the unit-level prores_sample_entry_fourcc
// detector tests in `crates/container/src/demux.rs` cover the
// fourcc-to-codec mapping for all six Apple fourccs without media.
//
// What this integration test does is the in-between layer: confirm the
// prores codec label flows through `decode::create_decoder` so that a
// MOV demux result lands at the pure-Rust ProRes backend rather than
// the unsupported-codec error path.

#[test]
fn create_decoder_accepts_prores_codec_label() {
    use codec::frame::{ColorSpace, PixelFormat, StreamInfo};
    let info = StreamInfo {
        codec: "prores".into(),
        width: 1280,
        height: 720,
        frame_rate: 24.0,
        duration: 0.0,
        pixel_format: PixelFormat::Yuv422p10le,
        color_space: ColorSpace::Bt709,
        total_frames: 0,
        bitrate: 0,
        color_metadata: Default::default(),
    };
    // Streaming-shape API (#55 P3): the constructor must succeed; we
    // immediately call finish() with no samples and decode_next must
    // return None. We're verifying the dispatch table contains a
    // "prores" arm.
    let mut dec = codec::decode::create_decoder("prores", info)
        .expect("ProRes decoder must be wired in create_decoder dispatch");
    dec.finish().expect("finish");
    let frame = dec.decode_next().expect("decode_next on empty input");
    assert!(frame.is_none(), "no samples → no frame");
}
