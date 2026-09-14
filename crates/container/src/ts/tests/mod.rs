use super::*;

mod ac3_audio;
mod adts_audio;
mod encrypted_guard;
mod framerate;
mod multi_program;
mod packet_layout;

// ---------------------------------------------------------------------------
// Shared test helpers
// Private items here are visible to all child modules via `super::`.
// ---------------------------------------------------------------------------

/// Build a minimal 188-byte TS packet. Remaining bytes are 0xFF
/// (already initialised). adaptation=01 means payload-only.
fn ts_pkt(pid: u16, pusi: bool, adaptation: u8, payload: &[u8]) -> [u8; TS_PACKET] {
    let mut p = [0xFFu8; TS_PACKET];
    p[0] = TS_SYNC;
    // TEI=0, PUSI=pusi, transport_priority=0, PID(13)
    p[1] = if pusi { 0x40 } else { 0x00 } | ((pid >> 8) & 0x1F) as u8;
    p[2] = (pid & 0xFF) as u8;
    // scramble=00, adaptation=adaptation, continuity=0
    p[3] = (adaptation & 0x03) << 4;
    let mut off = 4;
    // For these tests we always use adaptation=01 (payload only).
    let pay_len = payload.len().min(TS_PACKET - off);
    p[off..off + pay_len].copy_from_slice(&payload[..pay_len]);
    off += pay_len;
    // Pad any remaining bytes with 0xFF (already initialised).
    let _ = off;
    p
}

// ---- ADTS header builders used by adts_audio sub-module ----

/// Build a 7-byte ADTS header (no CRC) with the given fields.
/// `frame_length` covers header + payload.
fn build_adts_header_7(profile: u8, sr_idx: u8, ch_cfg: u8, frame_length: usize) -> [u8; 7] {
    let mut h = [0u8; 7];
    // Bytes 0..1: 0xFFF sync + ID(1)=0 (MPEG-4) + layer(2)=0 +
    // protection_absent(1)=1.
    h[0] = 0xFF;
    h[1] = 0xF0 | 0x01; // protection_absent = 1
    // Byte 2: profile(2) | sr_idx(4) | private(1) | ch_cfg high bit(1).
    h[2] = ((profile & 0x03) << 6) | ((sr_idx & 0x0F) << 2) | ((ch_cfg >> 2) & 0x01);
    // Byte 3: ch_cfg low 2 bits(2) | original/copy(1) | home(1) |
    // copyright_id_bit(1) | copyright_id_start(1) | frame_length high 2.
    h[3] = ((ch_cfg & 0x03) << 6) | (((frame_length >> 11) & 0x03) as u8);
    h[4] = ((frame_length >> 3) & 0xFF) as u8;
    h[5] = (((frame_length & 0x07) << 5) | 0x1F) as u8;
    // Byte 6: low buffer_fullness bits + number_of_raw_data_blocks(2) = 0.
    h[6] = 0xFC;
    h
}

/// Build a 9-byte ADTS header (with CRC). CRC bytes are placeholders.
fn build_adts_header_9(profile: u8, sr_idx: u8, ch_cfg: u8, frame_length: usize) -> [u8; 9] {
    let mut h = [0u8; 9];
    h[0] = 0xFF;
    h[1] = 0xF0; // protection_absent = 0 → CRC present
    h[2] = ((profile & 0x03) << 6) | ((sr_idx & 0x0F) << 2) | ((ch_cfg >> 2) & 0x01);
    h[3] = ((ch_cfg & 0x03) << 6) | (((frame_length >> 11) & 0x03) as u8);
    h[4] = ((frame_length >> 3) & 0xFF) as u8;
    h[5] = (((frame_length & 0x07) << 5) | 0x1F) as u8;
    h[6] = 0xFC;
    // Bytes 7..8: CRC placeholder (not validated by the parser).
    h
}

// ---- AC-3 / E-AC-3 frame builders used by ac3_audio sub-module ----

/// Local copy of the BitWriter used by the existing AAC tests, kept
/// alongside the Squad-37 sync-frame builders for self-containment.
struct BitWriter {
    bytes: Vec<u8>,
    bit_pos: usize,
}
impl BitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_pos: 0,
        }
    }
    fn put(&mut self, n: usize, v: u32) {
        for i in (0..n).rev() {
            let bit = ((v >> i) & 0x01) as u8;
            if self.bit_pos.is_multiple_of(8) {
                self.bytes.push(0);
            }
            let byte_idx = self.bit_pos / 8;
            let bit_idx = 7 - (self.bit_pos % 8);
            self.bytes[byte_idx] |= bit << bit_idx;
            self.bit_pos += 1;
        }
    }
    fn flush(self) -> Vec<u8> {
        self.bytes
    }
}

/// Build a minimal AC-3 syncframe by hand with a valid frmsizecod:
/// fscod=0 (48k), bit_rate_code=8 (128 kbps) → frame_length = 384
/// bytes per Table F.7. acmod=2 stereo, lfeon=0, bsid=8, bsmod=0.
/// The body bytes after the BSI prefix are zero-padded — only the
/// first ~7 bytes participate in our parser.
fn synth_ac3_frame_stereo_48k_128k() -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.put(16, 0x0B77); // syncword
    bw.put(16, 0); // crc1
    bw.put(2, 0); // fscod=0 → 48k
    bw.put(6, 8 << 1); // frmsizecod = bit_rate_code(8) << 1 = 16
    bw.put(5, 8); // bsid
    bw.put(3, 0); // bsmod
    bw.put(3, 2); // acmod=2 stereo
    // acmod=2 → dsurmod (2 bits)
    bw.put(2, 0);
    bw.put(1, 0); // lfeon=0
    // Pad up to 384 bytes (the AC-3 frame size we just announced).
    while bw.bytes.len() < 384 {
        bw.put(8, 0);
    }
    bw.flush()
}

/// E-AC-3 stereo frame with 6 audio blocks (numblkscod=3) at 48k.
/// frmsiz chosen such that frame_size_bytes = 192 ((0x5F + 1) * 2).
fn synth_eac3_frame_stereo_48k_192bytes() -> Vec<u8> {
    let mut bw = BitWriter::new();
    bw.put(16, 0x0B77);
    bw.put(2, 0); // strmtyp = 0 (independent)
    bw.put(3, 0); // substreamid
    bw.put(11, 0x5F); // frmsiz = 95 → frame_size = 192 bytes
    bw.put(2, 0); // fscod=0 → 48k
    bw.put(2, 3); // numblkscod=3 → 6 blocks
    bw.put(3, 2); // acmod=2 stereo
    bw.put(1, 0); // lfeon
    bw.put(5, 16); // bsid=16
    bw.put(5, 0); // dialnorm
    bw.put(1, 0); // compre=0
    while bw.bytes.len() < 192 {
        bw.put(8, 0);
    }
    bw.flush()
}

/// Build a continuation TS packet (PUSI=0) on `pid` with raw
/// `payload` bytes. Used by `build_ts_with_audio` when an audio PES
/// payload doesn't fit in a single 188-byte packet — the PES header
/// rides on the PUSI=1 packet, and continuation packets carry the
/// rest of the elementary-stream bytes verbatim until the next PUSI.
fn ts_pkt_continuation(pid: u16, payload: &[u8]) -> [u8; TS_PACKET] {
    let mut p = [0xFFu8; TS_PACKET];
    p[0] = TS_SYNC;
    p[1] = ((pid >> 8) & 0x1F) as u8; // PUSI=0
    p[2] = (pid & 0xFF) as u8;
    p[3] = 0b01 << 4; // adaptation=01 (payload only), continuity=0
    let pay_len = payload.len().min(TS_PACKET - 4);
    p[4..4 + pay_len].copy_from_slice(&payload[..pay_len]);
    p
}

/// Helper to build a TS file with: PAT, PMT, video PES (so the
/// video gate doesn't bail), audio PES on `audio_pid` with a given
/// `stream_type` byte and `descriptor_loop` for the PMT entry.
/// `audio_es` is the elementary-stream payload (AC-3 frame, etc.)
/// inserted into the audio PES packet body. If `audio_es` is too
/// large to fit in a single TS packet's payload area (~184 bytes),
/// the helper emits one PUSI=1 packet with the PES header + the
/// first chunk and successive PUSI=0 continuation packets carrying
/// the rest.
fn build_ts_with_audio(
    audio_stream_type: u8,
    audio_descriptors: &[u8],
    audio_pid: u16,
    audio_es: &[u8],
) -> Vec<u8> {
    // PAT pointing at PMT 0x100.
    let mut pat = vec![0x00];
    let pat_section_len: usize = 5 + 4 + 4;
    pat.push(0xB0 | ((pat_section_len >> 8) & 0x0F) as u8);
    pat.push((pat_section_len & 0xFF) as u8);
    pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pat.extend_from_slice(&[0x00, 0x01, 0xE1, 0x00, 0u8, 0u8, 0u8, 0u8]);
    let mut pat_payload = vec![0u8];
    pat_payload.extend_from_slice(&pat);
    let pat_pkt = ts_pkt(0x0000, true, 0b01, &pat_payload);

    // PMT advertising MPEG-2 video on 0x200 + audio entry.
    let mut pmt = vec![0x02];
    let pmt_stream_entries = 5  // video stream entry
        + 5 + audio_descriptors.len(); // audio stream entry + descriptors
    let pmt_section_len: usize = 9 + pmt_stream_entries + 4;
    pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
    pmt.push((pmt_section_len & 0xFF) as u8);
    pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pmt.extend_from_slice(&[0xE2, 0x00]); // PCR PID = 0x200
    pmt.extend_from_slice(&[0xF0, 0x00]); // program_info_length=0
    // Stream 1: MPEG-2 video on 0x200, no descriptors.
    pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
    // Stream 2: audio_pid w/ given stream_type + descriptors.
    pmt.push(audio_stream_type);
    pmt.push(0xE0 | ((audio_pid >> 8) & 0x1F) as u8);
    pmt.push((audio_pid & 0xFF) as u8);
    let esi_len = audio_descriptors.len() as u16;
    pmt.push(0xF0 | ((esi_len >> 8) & 0x0F) as u8);
    pmt.push((esi_len & 0xFF) as u8);
    pmt.extend_from_slice(audio_descriptors);
    pmt.extend_from_slice(&[0u8; 4]); // CRC placeholder
    let mut pmt_payload = vec![0u8];
    pmt_payload.extend_from_slice(&pmt);
    let pmt_pkt = ts_pkt(0x0100, true, 0b01, &pmt_payload);

    // Video PES (just enough so the video path doesn't bail).
    let video_pes = {
        let mut pes = vec![
            0u8, 0u8, 1u8, 0xE0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
        ];
        pes.extend_from_slice(&[0xAAu8; 16]);
        pes
    };
    let video_pkt = ts_pkt(0x0200, true, 0b01, &video_pes);

    // Audio PES — a single PES packet carrying all of audio_es,
    // potentially split across multiple TS packets via continuation.
    // Stream_id 0xC0 is audio per ISO/IEC 13818-1 §2.4.3.7.
    // Note: for AC-3 / E-AC-3, ATSC A/53 PES uses stream_id 0xBD
    // (PES private) rather than 0xC0; our parse_pes_header_audio
    // accepts the 0xC0..=0xDF range so we use 0xC0 here for test
    // simplicity. In real-world bitstreams the parser would also
    // need 0xBD support — that's a separate uplift.
    let mut audio_pes = vec![
        0u8, 0u8, 1u8, 0xC0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
    ];
    audio_pes.extend_from_slice(audio_es);

    // Split audio_pes across one PUSI=1 packet plus continuation
    // packets so PES payloads larger than 184 bytes flow through.
    let first_chunk_max = TS_PACKET - 4; // 184 bytes per TS packet payload
    let mut audio_pkts: Vec<[u8; TS_PACKET]> = Vec::new();
    let first_len = audio_pes.len().min(first_chunk_max);
    audio_pkts.push(ts_pkt(audio_pid, true, 0b01, &audio_pes[..first_len]));
    let mut cursor = first_len;
    while cursor < audio_pes.len() {
        let end = (cursor + first_chunk_max).min(audio_pes.len());
        audio_pkts.push(ts_pkt_continuation(audio_pid, &audio_pes[cursor..end]));
        cursor = end;
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(&pat_pkt);
    buf.extend_from_slice(&pmt_pkt);
    buf.extend_from_slice(&video_pkt);
    for pkt in &audio_pkts {
        buf.extend_from_slice(pkt);
    }
    buf.extend_from_slice(&ts_pkt(0x1FFF, false, 0b01, &[]));
    buf
}

/// Build a TS file with two distinct programs (program_number 1 and
/// 2). Program 1 carries MPEG-2 video on 0x200; program 2 carries
/// H.264 video on 0x300. Both PMTs live in their own PIDs (0x100,
/// 0x101 respectively).
fn build_two_program_ts() -> Vec<u8> {
    // PAT with TWO program entries.
    let mut pat = vec![0x00];
    let pat_section_len: usize = 5 + 4 + 4 + 4; // 2 programs + CRC
    pat.push(0xB0 | ((pat_section_len >> 8) & 0x0F) as u8);
    pat.push((pat_section_len & 0xFF) as u8);
    pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pat.extend_from_slice(&[0x00, 0x01, 0xE1, 0x00]); // program 1 → PMT 0x100
    pat.extend_from_slice(&[0x00, 0x02, 0xE1, 0x01]); // program 2 → PMT 0x101
    pat.extend_from_slice(&[0u8; 4]);
    let mut pat_payload = vec![0u8];
    pat_payload.extend_from_slice(&pat);
    let pat_pkt = ts_pkt(0x0000, true, 0b01, &pat_payload);

    // PMT 1: MPEG-2 video on 0x200.
    let mut pmt1 = vec![0x02];
    let pmt1_section_len: usize = 9 + 5 + 4;
    pmt1.push(0xB0 | ((pmt1_section_len >> 8) & 0x0F) as u8);
    pmt1.push((pmt1_section_len & 0xFF) as u8);
    pmt1.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]); // program 1
    pmt1.extend_from_slice(&[0xE2, 0x00, 0xF0, 0x00]);
    pmt1.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
    pmt1.extend_from_slice(&[0u8; 4]);
    let mut pmt1_payload = vec![0u8];
    pmt1_payload.extend_from_slice(&pmt1);
    let pmt1_pkt = ts_pkt(0x0100, true, 0b01, &pmt1_payload);

    // PMT 2: H.264 video on 0x300.
    let mut pmt2 = vec![0x02];
    let pmt2_section_len: usize = 9 + 5 + 4;
    pmt2.push(0xB0 | ((pmt2_section_len >> 8) & 0x0F) as u8);
    pmt2.push((pmt2_section_len & 0xFF) as u8);
    pmt2.extend_from_slice(&[0x00, 0x02, 0xC1, 0x00, 0x00]); // program 2
    pmt2.extend_from_slice(&[0xE3, 0x00, 0xF0, 0x00]);
    pmt2.extend_from_slice(&[STREAM_TYPE_H264, 0xE3, 0x00, 0xF0, 0x00]);
    pmt2.extend_from_slice(&[0u8; 4]);
    let mut pmt2_payload = vec![0u8];
    pmt2_payload.extend_from_slice(&pmt2);
    let pmt2_pkt = ts_pkt(0x0101, true, 0b01, &pmt2_payload);

    // Distinct PES bytes so we can tell programs apart at sample
    // level. Program 1 → 0xAA; program 2 → 0xBB.
    let make_pes = |fill: u8| {
        let mut pes = vec![
            0u8, 0u8, 1u8, 0xE0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
        ];
        pes.extend_from_slice(&[fill; 16]);
        pes
    };
    let p1_pes = ts_pkt(0x0200, true, 0b01, &make_pes(0xAA));
    let p2_pes = ts_pkt(0x0300, true, 0b01, &make_pes(0xBB));

    let mut buf = Vec::new();
    buf.extend_from_slice(&pat_pkt);
    buf.extend_from_slice(&pmt1_pkt);
    buf.extend_from_slice(&pmt2_pkt);
    // Two PES per program so the streaming path's PUSI flush yields.
    buf.extend_from_slice(&p1_pes);
    buf.extend_from_slice(&p2_pes);
    buf.extend_from_slice(&ts_pkt(0x0200, true, 0b01, &make_pes(0xAA)));
    buf.extend_from_slice(&ts_pkt(0x0300, true, 0b01, &make_pes(0xBB)));
    buf.extend_from_slice(&ts_pkt(0x1FFF, false, 0b01, &[]));
    buf
}

/// Build a single-program TS where the video PID's packets carry
/// `transport_scrambling_control != 0` (TSC=01 = "user-defined,
/// reserved" in ISO/IEC 13818-1 — both this and 10/11 indicate the
/// payload is encrypted and we have no CA tables).
fn build_encrypted_ts() -> Vec<u8> {
    // Reuse the single-program PAT/PMT shape from the existing tests.
    let mut pat = vec![0x00];
    let pat_section_len: usize = 5 + 4 + 4;
    pat.push(0xB0 | ((pat_section_len >> 8) & 0x0F) as u8);
    pat.push((pat_section_len & 0xFF) as u8);
    pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pat.extend_from_slice(&[0x00, 0x01, 0xE1, 0x00, 0u8, 0u8, 0u8, 0u8]);
    let mut pat_payload = vec![0u8];
    pat_payload.extend_from_slice(&pat);
    let pat_pkt = ts_pkt(0x0000, true, 0b01, &pat_payload);

    let mut pmt = vec![0x02];
    let pmt_section_len: usize = 9 + 5 + 4;
    pmt.push(0xB0 | ((pmt_section_len >> 8) & 0x0F) as u8);
    pmt.push((pmt_section_len & 0xFF) as u8);
    pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pmt.extend_from_slice(&[0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[0u8; 4]);
    let mut pmt_payload = vec![0u8];
    pmt_payload.extend_from_slice(&pmt);
    let pmt_pkt = ts_pkt(0x0100, true, 0b01, &pmt_payload);

    // Encrypted video PES: build the packet as normal but flip
    // bits 6-7 of byte 3 to TSC=01.
    let video_pes = {
        let mut pes = vec![
            0u8, 0u8, 1u8, 0xE0, 0u8, 0u8, 0x80, 0x80, 5, 0x21, 0x00, 0x01, 0x00, 0x01,
        ];
        pes.extend_from_slice(&[0xAAu8; 16]);
        pes
    };
    let mut video_pkt = ts_pkt(0x0200, true, 0b01, &video_pes);
    // TSC = 01 (single-bit set in the top 2 bits of byte 3).
    video_pkt[3] = (video_pkt[3] & 0x3F) | (0x01 << 6);

    let mut buf = Vec::new();
    buf.extend_from_slice(&pat_pkt);
    buf.extend_from_slice(&pmt_pkt);
    buf.extend_from_slice(&video_pkt);
    buf.extend_from_slice(&ts_pkt(0x1FFF, false, 0b01, &[]));
    buf
}
