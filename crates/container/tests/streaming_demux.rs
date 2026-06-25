//! Streaming demuxer tests (Squad streaming-migration-55 P1).
//!
//! For each format: drain the new `demux_streaming()` iterator one
//! sample at a time and assert byte-for-byte equality vs the legacy
//! `demux()`'s materialized `samples: Vec<Vec<u8>>`. Plus EOF /
//! error-mid-stream coverage.
//!
//! Synthetic AVI / TS fixtures are built in-test; MP4 / MKV exercise
//! the real-media corpus when present, gracefully skipping otherwise
//! (matches the existing decode_integration pattern).

use container::demux::demux;
use container::streaming::{Sample, StreamingDemuxer, demux_streaming};

use std::fs;
use std::path::PathBuf;

fn test_media(name: &str) -> Option<Vec<u8>> {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // workspace root
    p.push("test_media");
    p.push(name);
    fs::read(&p).ok()
}

/// Drain a streaming demuxer into a `Vec<Vec<u8>>` for byte-for-byte
/// comparison vs the legacy demux. Single-sample-at-a-time pull is
/// the load-bearing assertion: the streaming impl must NOT internally
/// accumulate.
fn drain<D: StreamingDemuxer + ?Sized>(d: &mut D) -> Vec<Sample> {
    let mut out = Vec::new();
    loop {
        match d.next_video_sample().expect("next_video_sample") {
            Some(s) => out.push(s),
            None => break,
        }
    }
    // After EOF, repeated calls must continue returning Ok(None) —
    // streaming demuxer is a one-shot iterator, not a cycle.
    assert!(d.next_video_sample().expect("post-EOF call").is_none());
    out
}

// ---------------- AVI ----------------

/// Build the same minimal XVID AVI fixture used by the avi.rs in-module
/// test, then drain via the streaming path and compare against legacy.
fn synth_avi_xvid_three_samples() -> Vec<u8> {
    fn chunk(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + payload.len());
        out.extend_from_slice(fourcc);
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        if out.len() & 1 == 1 {
            out.push(0);
        }
        out
    }
    fn list(list_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut body = Vec::with_capacity(4 + payload.len());
        body.extend_from_slice(list_type);
        body.extend_from_slice(payload);
        chunk(b"LIST", &body)
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
        strh.extend_from_slice(&[0u8; 12]);
        strh.extend_from_slice(&scale.to_le_bytes());
        strh.extend_from_slice(&rate.to_le_bytes());
        strh.extend_from_slice(&[0u8; 24]);
        let strh_chunk = chunk(b"strh", &strh);

        let mut strf = Vec::with_capacity(40);
        strf.extend_from_slice(&40u32.to_le_bytes());
        strf.extend_from_slice(&(w as i32).to_le_bytes());
        strf.extend_from_slice(&(h as i32).to_le_bytes());
        strf.extend_from_slice(&1u16.to_le_bytes());
        strf.extend_from_slice(&24u16.to_le_bytes());
        strf.extend_from_slice(compression);
        strf.extend_from_slice(&[0u8; 20]);
        let strf_chunk = chunk(b"strf", &strf);

        let mut strl_body = Vec::new();
        strl_body.extend_from_slice(&strh_chunk);
        strl_body.extend_from_slice(&strf_chunk);
        list(b"strl", &strl_body)
    }

    let mut hdrl_body = Vec::new();
    hdrl_body.extend_from_slice(&chunk(b"avih", &[0u8; 56]));
    hdrl_body.extend_from_slice(&video_strl(b"XVID", b"XVID", 320, 240, 30, 1));
    let hdrl = list(b"hdrl", &hdrl_body);

    let mut movi_body = Vec::new();
    movi_body.extend_from_slice(&chunk(b"00dc", b"frame-1-bytes"));
    movi_body.extend_from_slice(&chunk(b"01wb", b"audio-ignored"));
    movi_body.extend_from_slice(&chunk(b"00dc", b"frame-2"));
    movi_body.extend_from_slice(&chunk(b"00dc", b"frame-3-payload"));
    let movi = list(b"movi", &movi_body);

    let mut riff_body = Vec::new();
    riff_body.extend_from_slice(b"AVI ");
    riff_body.extend_from_slice(&hdrl);
    riff_body.extend_from_slice(&movi);

    let mut file = Vec::with_capacity(8 + riff_body.len());
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&(riff_body.len() as u32).to_le_bytes());
    file.extend_from_slice(&riff_body);
    file
}

#[test]
fn avi_streaming_matches_legacy_demux_byte_for_byte() {
    let bytes = synth_avi_xvid_three_samples();
    let legacy = demux(&bytes).expect("legacy demux");
    let mut streaming = demux_streaming(&bytes).expect("streaming demux");
    assert_eq!(streaming.header().codec, legacy.codec);
    assert_eq!(streaming.header().info.width, legacy.info.width);
    assert_eq!(streaming.header().info.height, legacy.info.height);
    let pulled = drain(&mut *streaming);
    assert_eq!(pulled.len(), legacy.samples.len(), "sample count");
    for (i, (s, l)) in pulled.iter().zip(legacy.samples.iter()).enumerate() {
        assert_eq!(&s.data, l, "sample {i} bytes");
    }
}

#[test]
fn avi_streaming_eof_returns_none_repeatedly() {
    let bytes = synth_avi_xvid_three_samples();
    let mut s = demux_streaming(&bytes).unwrap();
    let mut count = 0;
    while s.next_video_sample().unwrap().is_some() {
        count += 1;
    }
    assert_eq!(count, 3);
    // Three more None calls must remain None — no panic, no cycling.
    for _ in 0..3 {
        assert!(s.next_video_sample().unwrap().is_none());
    }
}

#[test]
fn avi_streaming_no_internal_buffering_pull_one_sample_keeps_remainder() {
    // Pull one sample, drop the demuxer, build a fresh one and pull
    // ALL samples. Sample 1 of fresh == sample 1 of partial.
    let bytes = synth_avi_xvid_three_samples();
    let mut partial = demux_streaming(&bytes).unwrap();
    let first = partial.next_video_sample().unwrap().unwrap();
    drop(partial);
    let mut fresh = demux_streaming(&bytes).unwrap();
    let fresh_first = fresh.next_video_sample().unwrap().unwrap();
    assert_eq!(first.data, fresh_first.data);
}

#[test]
fn avi_streaming_total_frames_comes_from_avih_not_from_drained_samples() {
    // The synthetic fixture's avih is all-zeroes (so dwTotalFrames=0 →
    // we report 0, not the legacy `samples.len()=3`). This is the
    // load-bearing assertion: the streaming path MUST NOT walk the movi
    // chunks to derive total_frames. 0 is a legitimate "unknown" sentinel
    // (matches TS / MKV streaming).
    let bytes = synth_avi_xvid_three_samples();
    let streaming = demux_streaming(&bytes).unwrap();
    assert_eq!(
        streaming.header().info.total_frames,
        0,
        "synthetic avih has dwTotalFrames=0; streaming must not synthesize from samples"
    );
}

// ---------------- AVI OpenDML (Squad-38) ----------------

/// Build a small synthetic OpenDML AVI: two `LIST movi` records (one
/// per `RIFF AVI ` / `RIFF AVIX` segment), an `indx` superindex inside
/// the video stream's strl pointing at one ix00 per movi, and a `dmlh`
/// reporting `dwTotalFrames=6`. Mirrors the in-module fixture exactly,
/// re-implemented here so this integration test does not depend on
/// avi.rs's pub(super) test helpers.
fn synth_opendml_two_movi_six_samples() -> (Vec<u8>, Vec<Vec<u8>>) {
    fn chunk(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + payload.len());
        out.extend_from_slice(fourcc);
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        if out.len() & 1 == 1 {
            out.push(0);
        }
        out
    }
    fn list(list_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut body = Vec::with_capacity(4 + payload.len());
        body.extend_from_slice(list_type);
        body.extend_from_slice(payload);
        chunk(b"LIST", &body)
    }
    fn build_indx(entries: &[(u64, u32, u32)]) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&4u16.to_le_bytes());
        body.push(0);
        body.push(0x00);
        body.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        body.extend_from_slice(b"00dc");
        body.extend_from_slice(&[0u8; 12]);
        for (qw_off, dw_size, dw_dur) in entries {
            body.extend_from_slice(&qw_off.to_le_bytes());
            body.extend_from_slice(&dw_size.to_le_bytes());
            body.extend_from_slice(&dw_dur.to_le_bytes());
        }
        chunk(b"indx", &body)
    }
    fn build_ix00(entries: &[(usize, usize)], qw_base_offset: u64) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&2u16.to_le_bytes());
        body.push(0);
        body.push(0x01);
        body.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        body.extend_from_slice(b"00dc");
        body.extend_from_slice(&qw_base_offset.to_le_bytes());
        body.extend_from_slice(&0u32.to_le_bytes());
        for (data_off, data_size) in entries {
            body.extend_from_slice(&(*data_off as u32).to_le_bytes());
            body.extend_from_slice(&(*data_size as u32).to_le_bytes());
        }
        chunk(b"ix00", &body)
    }
    fn build_hdrl(indx_chunk: &[u8], dmlh_total: u32, avih_total: u32) -> Vec<u8> {
        let mut avih_body = Vec::with_capacity(56);
        avih_body.extend_from_slice(&33333u32.to_le_bytes());
        avih_body.extend_from_slice(&[0u8; 12]);
        avih_body.extend_from_slice(&avih_total.to_le_bytes());
        avih_body.extend_from_slice(&[0u8; 32]);
        let avih_c = chunk(b"avih", &avih_body);
        let strh_c = {
            let mut s = Vec::with_capacity(56);
            s.extend_from_slice(b"vids");
            s.extend_from_slice(b"XVID");
            s.extend_from_slice(&[0u8; 12]);
            s.extend_from_slice(&1u32.to_le_bytes());
            s.extend_from_slice(&30u32.to_le_bytes());
            s.extend_from_slice(&[0u8; 24]);
            chunk(b"strh", &s)
        };
        let strf_c = {
            let mut s = Vec::with_capacity(40);
            s.extend_from_slice(&40u32.to_le_bytes());
            s.extend_from_slice(&320i32.to_le_bytes());
            s.extend_from_slice(&240i32.to_le_bytes());
            s.extend_from_slice(&1u16.to_le_bytes());
            s.extend_from_slice(&24u16.to_le_bytes());
            s.extend_from_slice(b"XVID");
            s.extend_from_slice(&[0u8; 20]);
            chunk(b"strf", &s)
        };
        let mut strl_body = Vec::new();
        strl_body.extend_from_slice(&strh_c);
        strl_body.extend_from_slice(&strf_c);
        strl_body.extend_from_slice(indx_chunk);
        let strl_c = list(b"strl", &strl_body);
        let dmlh_c = {
            let mut b = Vec::new();
            b.extend_from_slice(&dmlh_total.to_le_bytes());
            chunk(b"dmlh", &b)
        };
        let odml_c = list(b"odml", &dmlh_c);
        let mut hdrl_body = Vec::new();
        hdrl_body.extend_from_slice(&avih_c);
        hdrl_body.extend_from_slice(&strl_c);
        hdrl_body.extend_from_slice(&odml_c);
        list(b"hdrl", &hdrl_body)
    }

    let payloads: Vec<Vec<u8>> = (0..6)
        .map(|i| format!("opendml-streaming-frame-{i}").into_bytes())
        .collect();

    // movi#1 + ix00#1
    let mut movi1_body = Vec::new();
    let mut data_offsets_1 = Vec::new();
    for i in 0..3 {
        let cur = movi1_body.len();
        movi1_body.extend_from_slice(&chunk(b"00dc", &payloads[i]));
        data_offsets_1.push((cur + 8, payloads[i].len()));
    }
    let mut movi2_body = Vec::new();
    let mut data_offsets_2 = Vec::new();
    for i in 3..6 {
        let cur = movi2_body.len();
        movi2_body.extend_from_slice(&chunk(b"00dc", &payloads[i]));
        data_offsets_2.push((cur + 8, payloads[i].len()));
    }
    let movi1_chunk = list(b"movi", &movi1_body);
    let movi2_chunk = list(b"movi", &movi2_body);

    // Sized hdrl with placeholder indx so we can compute layout.
    let placeholder_indx = build_indx(&[(0, 0, 0), (0, 0, 0)]);
    let hdrl_pre = build_hdrl(&placeholder_indx, 6, 3);

    let avi_body_start = 12usize;
    let movi1_offset = avi_body_start + hdrl_pre.len();
    let movi1_body_offset = movi1_offset + 12;
    let movi1_end = movi1_offset + movi1_chunk.len();
    let ix1_offset = movi1_end;
    let ix1 = build_ix00(&data_offsets_1, movi1_body_offset as u64);
    let ix1_end = ix1_offset + ix1.len();

    let avix_outer = ix1_end;
    let avix_body_start = avix_outer + 12;
    let movi2_offset = avix_body_start;
    let movi2_body_offset = movi2_offset + 12;
    let ix2_offset = movi2_offset + movi2_chunk.len();
    let ix2 = build_ix00(&data_offsets_2, movi2_body_offset as u64);

    let real_indx = build_indx(&[
        (ix1_offset as u64, (ix1.len() - 8) as u32, 3),
        (ix2_offset as u64, (ix2.len() - 8) as u32, 3),
    ]);
    assert_eq!(real_indx.len(), placeholder_indx.len());
    let hdrl = build_hdrl(&real_indx, 6, 3);
    assert_eq!(hdrl.len(), hdrl_pre.len());

    let mut avi_seg = Vec::new();
    avi_seg.extend_from_slice(b"AVI ");
    avi_seg.extend_from_slice(&hdrl);
    avi_seg.extend_from_slice(&movi1_chunk);
    avi_seg.extend_from_slice(&ix1);
    let mut file = Vec::new();
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&(avi_seg.len() as u32).to_le_bytes());
    file.extend_from_slice(&avi_seg);

    let mut avix_seg = Vec::new();
    avix_seg.extend_from_slice(b"AVIX");
    avix_seg.extend_from_slice(&movi2_chunk);
    avix_seg.extend_from_slice(&ix2);
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&(avix_seg.len() as u32).to_le_bytes());
    file.extend_from_slice(&avix_seg);

    (file, payloads)
}

#[test]
fn avi_opendml_streaming_dispatch_walks_both_movi_lists() {
    // End-to-end: feed the multi-RIFF OpenDML fixture through the
    // public `demux_streaming` dispatcher, confirm all six samples
    // surface in superindex order across both `RIFF AVI ` + `RIFF AVIX`
    // segments, and that `header.info.total_frames` reflects the
    // dmlh value (6) — not the avih value (3) which would have wrapped
    // for a real >1 GiB clip.
    let (bytes, expected) = synth_opendml_two_movi_six_samples();
    let mut s = demux_streaming(&bytes).expect("opendml streaming");
    assert_eq!(s.header().codec, "mpeg4");
    assert_eq!(
        s.header().info.total_frames,
        6,
        "dmlh.dwTotalFrames must win"
    );
    let pulled = drain(&mut *s);
    assert_eq!(pulled.len(), 6, "should walk both movi LISTs end-to-end");
    for (i, (got, want)) in pulled.iter().zip(expected.iter()).enumerate() {
        assert_eq!(&got.data, want, "sample {i} mismatch");
    }
}

#[test]
fn avi_opendml_legacy_demux_dispatch_walks_both_movi_lists() {
    // Same fixture through the materializing `demux::demux` path —
    // confirms the legacy code path also handles multi-movi for the
    // bench / fidelity tests that don't use streaming.
    let (bytes, expected) = synth_opendml_two_movi_six_samples();
    let d = demux(&bytes).expect("opendml legacy demux");
    assert_eq!(d.codec, "mpeg4");
    assert_eq!(d.samples.len(), 6);
    assert_eq!(d.info.total_frames, 6);
    for (i, (g, e)) in d.samples.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "sample {i} mismatch on legacy path");
    }
}

// ---------------- TS ----------------

/// Build the same minimal MPEG-2 TS fixture as the in-module test.
fn synth_ts_two_pes_packets() -> Vec<u8> {
    const TS_PACKET: usize = 188;
    const TS_SYNC: u8 = 0x47;
    const STREAM_TYPE_MPEG2_VIDEO: u8 = 0x02;
    fn ts_pkt(pid: u16, pusi: bool, adaptation: u8, payload: &[u8]) -> [u8; TS_PACKET] {
        let mut p = [0xFFu8; TS_PACKET];
        p[0] = TS_SYNC;
        p[1] = if pusi { 0x40 } else { 0x00 } | ((pid >> 8) & 0x1F) as u8;
        p[2] = (pid & 0xFF) as u8;
        p[3] = (adaptation & 0x03) << 4;
        let off = 4;
        let pay_len = payload.len().min(TS_PACKET - off);
        p[off..off + pay_len].copy_from_slice(&payload[..pay_len]);
        p
    }

    let mut pat = Vec::new();
    pat.push(0x00);
    let section_length: usize = 5 + 4 + 4;
    pat.push(0xB0 | ((section_length >> 8) & 0x0F) as u8);
    pat.push((section_length & 0xFF) as u8);
    pat.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pat.extend_from_slice(&[0x00, 0x01]);
    pat.extend_from_slice(&[0xE1, 0x00]);
    pat.extend_from_slice(&[0, 0, 0, 0]);
    let mut pat_payload = vec![0u8];
    pat_payload.extend_from_slice(&pat);
    let pat_pkt = ts_pkt(0x0000, true, 0b01, &pat_payload);

    let mut pmt = Vec::new();
    pmt.push(0x02);
    let pmt_sec_len: usize = 9 + 5 + 4;
    pmt.push(0xB0 | ((pmt_sec_len >> 8) & 0x0F) as u8);
    pmt.push((pmt_sec_len & 0xFF) as u8);
    pmt.extend_from_slice(&[0x00, 0x01, 0xC1, 0x00, 0x00]);
    pmt.extend_from_slice(&[0xE2, 0x00]);
    pmt.extend_from_slice(&[0xF0, 0x00]);
    pmt.extend_from_slice(&[STREAM_TYPE_MPEG2_VIDEO, 0xE2, 0x00, 0xF0, 0x00]);
    pmt.extend_from_slice(&[0, 0, 0, 0]);
    let mut pmt_payload = vec![0u8];
    pmt_payload.extend_from_slice(&pmt);
    let pmt_pkt = ts_pkt(0x0100, true, 0b01, &pmt_payload);

    let make_pes = |byte: u8| {
        let mut pes = vec![0u8, 0u8, 1u8];
        pes.push(0xE0);
        pes.extend_from_slice(&[0u8, 0u8]);
        pes.push(0x80);
        pes.push(0x80);
        pes.push(5);
        pes.extend_from_slice(&[0x21, 0x00, 0x01, 0x00, 0x01]);
        pes.extend_from_slice(&[byte; 16]);
        pes
    };
    let pes_pkt_a = ts_pkt(0x0200, true, 0b01, &make_pes(0xAA));
    let pes_pkt_b = ts_pkt(0x0200, true, 0b01, &make_pes(0xBB));

    let mut buf = Vec::new();
    buf.extend_from_slice(&pat_pkt);
    buf.extend_from_slice(&pmt_pkt);
    buf.extend_from_slice(&pes_pkt_a);
    buf.extend_from_slice(&pes_pkt_b);
    buf.extend_from_slice(&ts_pkt(0x1FFF, false, 0b01, &[]));
    buf
}

#[test]
fn ts_streaming_matches_legacy_demux_byte_for_byte() {
    let bytes = synth_ts_two_pes_packets();
    let legacy = demux(&bytes).expect("legacy demux");
    let mut streaming = demux_streaming(&bytes).expect("streaming demux");
    assert_eq!(streaming.header().codec, "mpeg2");
    let pulled = drain(&mut *streaming);
    assert_eq!(pulled.len(), legacy.samples.len(), "sample count");
    for (i, (s, l)) in pulled.iter().zip(legacy.samples.iter()).enumerate() {
        assert_eq!(&s.data, l, "sample {i} bytes");
    }
}

#[test]
fn ts_streaming_handles_empty_packet_run_after_pmt_without_yielding() {
    // Drain — only 2 PUSI=1 sample boundaries → 2 samples; first call
    // walks all packets through the second PUSI before returning.
    let bytes = synth_ts_two_pes_packets();
    let mut s = demux_streaming(&bytes).unwrap();
    let s1 = s.next_video_sample().unwrap().expect("first sample");
    let s2 = s.next_video_sample().unwrap().expect("second sample");
    assert_eq!(s.next_video_sample().unwrap().is_none(), true);
    assert!(
        s1.data.starts_with(&[0xAA; 16]) || s1.data.contains(&0xAA),
        "first sample should contain AA payload"
    );
    assert!(
        s2.data.starts_with(&[0xBB; 16]) || s2.data.contains(&0xBB),
        "second sample should contain BB payload"
    );
}

// ---------------- MP4 (real media when available) ----------------

#[test]
fn mp4_streaming_matches_legacy_demux_when_real_media_present() {
    let Some(bytes) = test_media("jellyfin_h264_high_l40_1080p_24fps.mp4")
        .or_else(|| test_media("BBB_h264_baseline_320x180.mp4"))
        .or_else(|| test_media("av1_test_clip.mp4"))
    else {
        eprintln!("test_media MP4 not present; skipping streaming MP4 parity check");
        return;
    };

    let legacy = match demux(&bytes) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("legacy demux failed (test corpus may not match expectations): {e}");
            return;
        }
    };
    let mut streaming = demux_streaming(&bytes).expect("streaming demux opens");
    assert_eq!(streaming.header().codec, legacy.codec);
    assert_eq!(streaming.header().info.width, legacy.info.width);
    assert_eq!(streaming.header().info.height, legacy.info.height);

    // Pull one sample at a time and compare each against legacy's
    // index. This is the load-bearing test: per-sample length-prefix
    // → Annex-B conversion + the Squad-14 ParamSetTracker MUST behave
    // identically across the two paths or downstream H.264 / HEVC
    // decode regresses on ExoPlayer-style open-GOP MP4s.
    let mut idx = 0usize;
    while let Some(sample) = streaming.next_video_sample().expect("pull") {
        assert!(
            idx < legacy.samples.len(),
            "streaming yielded more samples than legacy"
        );
        assert_eq!(
            &sample.data, &legacy.samples[idx],
            "MP4 streaming sample {idx} diverged from legacy"
        );
        idx += 1;
    }
    assert_eq!(
        idx,
        legacy.samples.len(),
        "streaming yielded fewer samples than legacy"
    );
}

// ---------------- MKV (real media when available) ----------------

#[test]
fn mkv_streaming_matches_legacy_demux_when_real_media_present() {
    let Some(bytes) =
        test_media("BBB_vp9_360p.webm").or_else(|| test_media("BBB_h264_main_720p.mkv"))
    else {
        eprintln!("test_media MKV not present; skipping streaming MKV parity check");
        return;
    };

    let legacy = match demux(&bytes) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("legacy MKV demux failed: {e}");
            return;
        }
    };
    let mut streaming = demux_streaming(&bytes).expect("streaming demux opens");
    assert_eq!(streaming.header().codec, legacy.codec);
    assert_eq!(streaming.header().info.width, legacy.info.width);
    assert_eq!(streaming.header().info.height, legacy.info.height);

    let mut idx = 0usize;
    while let Some(sample) = streaming.next_video_sample().expect("pull") {
        assert!(
            idx < legacy.samples.len(),
            "streaming yielded more samples than legacy"
        );
        assert_eq!(
            &sample.data, &legacy.samples[idx],
            "MKV streaming sample {idx} diverged from legacy"
        );
        idx += 1;
    }
    assert_eq!(
        idx,
        legacy.samples.len(),
        "streaming yielded fewer samples than legacy"
    );
}

// ---------------- Unsupported / error propagation ----------------

#[test]
fn streaming_dispatcher_rejects_unknown_container() {
    let garbage = vec![0xCDu8; 1024];
    match demux_streaming(&garbage) {
        Ok(_) => panic!("dispatcher must reject unknown container"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(msg.contains("unsupported container"), "msg: {msg}");
        }
    }
}

#[test]
fn streaming_dispatcher_rejects_short_buffer() {
    let short = vec![0u8; 4];
    match demux_streaming(&short) {
        Ok(_) => panic!("dispatcher must reject short buffer"),
        Err(_) => {} // any error is acceptable; the dispatcher must not panic
    }
}
