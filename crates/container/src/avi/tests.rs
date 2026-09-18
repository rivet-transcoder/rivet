//! Unit tests for the AVI demuxer.
//! Declared by `#[cfg(test)] mod tests;` in mod.rs — this file is the
//! inner content only (no outer `mod tests { }` wrapper needed).

use super::*;
use super::opendml::{parse_indx_body, read_dmlh_total_frames};
use super::streaming::Backend;
use crate::streaming::StreamingDemuxer;

/// Build a minimal RIFF chunk: little-endian 4-byte size header.
fn chunk(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + payload.len());
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if out.len() & 1 == 1 {
        out.push(0);
    } // word-align
    out
}

/// Wrap a payload as `LIST <type> <payload>`.
fn list(list_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(4 + payload.len());
    body.extend_from_slice(list_type);
    body.extend_from_slice(payload);
    chunk(b"LIST", &body)
}

/// Emit a strh + strf pair for one video stream using a given fcc.
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
    strh.extend_from_slice(&[0u8; 12]); // flags/priority/lang/initial
    strh.extend_from_slice(&scale.to_le_bytes());
    strh.extend_from_slice(&rate.to_le_bytes());
    strh.extend_from_slice(&[0u8; 24]); // start/length/buf/quality/samplesize/rect
    let strh_chunk = chunk(b"strh", &strh);

    let mut strf = Vec::with_capacity(40);
    strf.extend_from_slice(&40u32.to_le_bytes()); // biSize
    strf.extend_from_slice(&(w as i32).to_le_bytes()); // biWidth
    strf.extend_from_slice(&(h as i32).to_le_bytes()); // biHeight
    strf.extend_from_slice(&1u16.to_le_bytes()); // biPlanes
    strf.extend_from_slice(&24u16.to_le_bytes()); // biBitCount
    strf.extend_from_slice(compression); // biCompression
    strf.extend_from_slice(&[0u8; 20]); // remaining BIH fields
    let strf_chunk = chunk(b"strf", &strf);

    let mut strl_body = Vec::new();
    strl_body.extend_from_slice(&strh_chunk);
    strl_body.extend_from_slice(&strf_chunk);
    list(b"strl", &strl_body)
}

#[test]
fn demux_minimal_xvid_avi_emits_samples() {
    // hdrl LIST: dummy avih + one video strl with XVID fourcc.
    let mut hdrl_body = Vec::new();
    hdrl_body.extend_from_slice(&chunk(b"avih", &[0u8; 56])); // MainAVIHeader
    hdrl_body.extend_from_slice(&video_strl(b"XVID", b"XVID", 320, 240, 30, 1));
    let hdrl = list(b"hdrl", &hdrl_body);

    // movi LIST: three compressed DIB samples (00dc) of distinct payloads.
    let mut movi_body = Vec::new();
    movi_body.extend_from_slice(&chunk(b"00dc", b"frame-1-bytes"));
    movi_body.extend_from_slice(&chunk(b"01wb", b"audio-ignored"));
    movi_body.extend_from_slice(&chunk(b"00dc", b"frame-2"));
    movi_body.extend_from_slice(&chunk(b"00dc", b"frame-3-payload"));
    let movi = list(b"movi", &movi_body);

    // Outer RIFF.
    let mut riff_body = Vec::new();
    riff_body.extend_from_slice(b"AVI ");
    riff_body.extend_from_slice(&hdrl);
    riff_body.extend_from_slice(&movi);

    let mut file = Vec::with_capacity(8 + riff_body.len());
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&(riff_body.len() as u32).to_le_bytes());
    file.extend_from_slice(&riff_body);

    let d = demux_avi(&file).expect("demux");
    assert_eq!(d.codec, "mpeg4");
    assert_eq!(d.info.width, 320);
    assert_eq!(d.info.height, 240);
    assert_eq!(d.samples.len(), 3);
    assert_eq!(d.samples[0], b"frame-1-bytes");
    assert_eq!(d.samples[1], b"frame-2");
    assert_eq!(d.samples[2], b"frame-3-payload");
}

#[test]
fn demux_rejects_unknown_fourcc() {
    let mut hdrl_body = Vec::new();
    hdrl_body.extend_from_slice(&chunk(b"avih", &[0u8; 56]));
    hdrl_body.extend_from_slice(&video_strl(b"ZZZZ", b"ZZZZ", 100, 100, 30, 1));
    let hdrl = list(b"hdrl", &hdrl_body);
    let movi = list(b"movi", &chunk(b"00dc", b"x"));
    let mut body = Vec::new();
    body.extend_from_slice(b"AVI ");
    body.extend_from_slice(&hdrl);
    body.extend_from_slice(&movi);
    let mut file = Vec::new();
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&(body.len() as u32).to_le_bytes());
    file.extend_from_slice(&body);
    assert!(demux_avi(&file).is_err());
}

#[test]
fn demux_handles_divx_variants() {
    for fcc in [b"DIVX", b"DX50", b"DIV3", b"XviD"] {
        let mut hdrl_body = Vec::new();
        hdrl_body.extend_from_slice(&chunk(b"avih", &[0u8; 56]));
        hdrl_body.extend_from_slice(&video_strl(fcc, fcc, 640, 480, 25, 1));
        let hdrl = list(b"hdrl", &hdrl_body);
        let movi = list(b"movi", &chunk(b"00dc", b"sample"));
        let mut body = Vec::new();
        body.extend_from_slice(b"AVI ");
        body.extend_from_slice(&hdrl);
        body.extend_from_slice(&movi);
        let mut file = Vec::new();
        file.extend_from_slice(b"RIFF");
        file.extend_from_slice(&(body.len() as u32).to_le_bytes());
        file.extend_from_slice(&body);
        let d = demux_avi(&file).expect("should demux");
        assert_eq!(d.codec, "mpeg4", "fourcc {:?} did not map to mpeg4", fcc);
    }
}

// ----- OpenDML 1.0 super-index fixture tests (Squad-38) -----

/// Build a synthetic OpenDML AVI: 2 movi LISTs each with 3 video
/// chunks (XVID), an indx super-index pointing at 2 ix00 sub-indexes,
/// each ix00 listing the 3 chunks in its movi, and `dmlh` reporting
/// `dwTotalFrames=6`. Returns the assembled file bytes plus the six
/// expected sample payloads in order, so tests can assert offsets +
/// content.
///
/// Layout (sizes computed bottom-up so absolute offsets work out):
///   `RIFF AVI ` segment
///     `LIST hdrl`
///       `avih` (dwTotalFrames=3 — only counts the first segment;
///                we expect dmlh's 6 to win)
///       `LIST strl`
///         strh (XVID), strf (320×240),
///         indx superindex pointing at the two ix00 chunks
///       `LIST odml` { dmlh (dwTotalFrames=6) }
///     `LIST movi` { 00dc×3 }
///     ix00 (3 entries pointing into movi#1)
///   `RIFF AVIX` segment
///     `LIST movi` { 00dc×3 }
///     ix00 (3 entries pointing into movi#2)
fn build_opendml_two_movi_six_samples() -> (Vec<u8>, Vec<Vec<u8>>) {
    // The six sample payloads — distinct so we can assert ordering.
    let payloads: Vec<Vec<u8>> = (0..6)
        .map(|i| format!("opendml-frame-{i}").into_bytes())
        .collect();

    // ----- Inner movi bodies + ix00 stub layout planning -----
    // We build movi LISTs first, then plan ix00 chunks from the
    // resulting per-chunk offsets, then assemble outer RIFF segments
    // so we know the absolute file offsets of each ix00 chunk
    // (needed for the indx superindex entries).

    // movi#1 body: three 00dc chunks with payloads 0, 1, 2.
    // We'll record (offset_into_movi_body_of_chunk_data, size) for each.
    let mut movi1_body = Vec::new();
    let mut chunk_data_offsets_in_movi1 = Vec::new();
    for payload in &payloads[0..3] {
        let cur_off = movi1_body.len();
        // Chunk header is 8 bytes; data starts at cur_off + 8.
        let c = chunk(b"00dc", payload);
        movi1_body.extend_from_slice(&c);
        chunk_data_offsets_in_movi1.push((cur_off + 8, payload.len()));
    }

    // movi#2 body: three 00dc chunks with payloads 3, 4, 5.
    let mut movi2_body = Vec::new();
    let mut chunk_data_offsets_in_movi2 = Vec::new();
    for payload in &payloads[3..6] {
        let cur_off = movi2_body.len();
        let c = chunk(b"00dc", payload);
        movi2_body.extend_from_slice(&c);
        chunk_data_offsets_in_movi2.push((cur_off + 8, payload.len()));
    }

    // The movi LIST wraps a 4-byte type ("movi") + body. So the
    // body starts +12 from the LIST chunk's start (+8 chunk header
    // + 4 type fourcc).
    let movi1_chunk = list(b"movi", &movi1_body);
    let movi2_chunk = list(b"movi", &movi2_body);

    // Build the two ix00 chunks. Each ix## chunk body layout:
    //   wLongsPerEntry=2 (u16), bIndexSubType=0 (u8),
    //   bIndexType=0x01 (u8), nEntriesInUse=N (u32),
    //   dwChunkId="00dc" (u32), qwBaseOffset (u64),
    //   dwReserved=0 (u32), then per-entry (dwOffset, dwSize) u32×2.
    //
    // We point qwBaseOffset at the start of the corresponding movi
    // LIST's BODY (i.e. the byte right after `movi` type fourcc).
    // dwOffset for each entry is the offset of the chunk DATA from
    // qwBaseOffset, i.e. exactly `chunk_data_offsets_in_moviX[i].0`.
    let build_ix00 = |entries: &[(usize, usize)], qw_base_offset: u64| -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&2u16.to_le_bytes()); // wLongsPerEntry
        body.push(0); // bIndexSubType
        body.push(0x01); // bIndexType=AVI_INDEX_OF_CHUNKS
        body.extend_from_slice(&(entries.len() as u32).to_le_bytes()); // nEntriesInUse
        body.extend_from_slice(b"00dc"); // dwChunkId
        body.extend_from_slice(&qw_base_offset.to_le_bytes()); // qwBaseOffset
        body.extend_from_slice(&0u32.to_le_bytes()); // dwReserved
        for (data_off, data_size) in entries {
            body.extend_from_slice(&(*data_off as u32).to_le_bytes()); // dwOffset
            body.extend_from_slice(&(*data_size as u32).to_le_bytes()); // dwSize
        }
        chunk(b"ix00", &body)
    };

    // We need the absolute file offsets of the two movi BODIES and
    // the two ix00 CHUNK HEADERS to fill the indx superindex.
    // Layout of the outer file is:
    //   [0..8]      "RIFF" + size32 of the AVI  segment payload
    //   [8..12]     "AVI " form type
    //   [12..]      LIST hdrl ... (size depends on indx contents
    //                — chicken/egg, we resolve below)
    //               LIST movi#1 ... (movi1_chunk)
    //               ix00#1 ... (ix1)
    //   then        "RIFF" + size32 of the AVIX segment payload
    //               "AVIX" form type
    //               LIST movi#2 ... (movi2_chunk)
    //               ix00#2 ... (ix2)
    //
    // To break the cycle, build hdrl with placeholder indx values
    // first, measure the resulting byte sizes, compute final
    // offsets, then rewrite the indx body and reassemble.

    // Build hdrl first with a PLACEHOLDER indx (zeroed offsets) so
    // we know the hdrl size — which doesn't change when we patch
    // the placeholder qwOffset values (size stays constant).
    let placeholder_indx = build_indx_placeholder();
    let hdrl_with_placeholder = build_hdrl(
        &placeholder_indx,
        /*dmlh_total*/ 6,
        /*avih_total*/ 3,
    );

    // Compute the absolute offsets we need to know AHEAD of writing
    // the real indx: positions of movi#1 body, movi#2 body,
    // ix00#1 chunk header, ix00#2 chunk header.

    // Position 0 of the file = "RIFF" header start. The AVI  segment
    // body begins at byte 12 (after RIFF/size/AVI ).
    let avi_body_start = 12usize;
    let hdrl_offset = avi_body_start; // hdrl is the first record
    let hdrl_end = hdrl_offset + hdrl_with_placeholder.len();

    let movi1_offset = hdrl_end; // movi LIST chunk header start
    // movi LIST body starts at movi1_offset + 8 (LIST hdr) + 4 (type "movi") = +12
    let movi1_body_offset = movi1_offset + 12;
    let movi1_end = movi1_offset + movi1_chunk.len();

    let ix1_offset = movi1_end; // ix00 chunk header for movi#1
    // ix00 chunk size doesn't depend on placeholder vs real values —
    // build a real one with the right qwBaseOffset to measure its byte
    // length (constant for fixed entries).
    let ix1_chunk_real = build_ix00(&chunk_data_offsets_in_movi1, movi1_body_offset as u64);
    let ix1_end = ix1_offset + ix1_chunk_real.len();

    // Now the second `RIFF AVIX` segment starts.
    let avix_outer_start = ix1_end;
    // RIFF chunk header (8) + form type "AVIX" (4) = 12 bytes before body.
    let avix_body_start = avix_outer_start + 12;

    let movi2_offset = avix_body_start;
    let movi2_body_offset = movi2_offset + 12;
    let movi2_end = movi2_offset + movi2_chunk.len();

    let ix2_offset = movi2_end;
    let ix2_chunk_real = build_ix00(&chunk_data_offsets_in_movi2, movi2_body_offset as u64);

    // Real indx superindex pointing at the two ix00 chunks.
    let real_indx = build_indx_real(&[
        (
            ix1_offset as u64,
            (ix1_chunk_real.len() - 8) as u32,
            /*dur*/ 3,
        ),
        (
            ix2_offset as u64,
            (ix2_chunk_real.len() - 8) as u32,
            /*dur*/ 3,
        ),
    ]);
    // Sanity: real and placeholder indx must be byte-identical in length.
    assert_eq!(
        real_indx.len(),
        placeholder_indx.len(),
        "indx size sanity — placeholder and real must match for offsets to stay valid"
    );

    let hdrl_real = build_hdrl(&real_indx, 6, 3);
    assert_eq!(
        hdrl_real.len(),
        hdrl_with_placeholder.len(),
        "hdrl size sanity — must not depend on indx values, only sizes"
    );

    // Assemble AVI  segment body (after the RIFF "AVI " 12-byte header).
    let mut avi_seg_body = Vec::new();
    avi_seg_body.extend_from_slice(b"AVI ");
    avi_seg_body.extend_from_slice(&hdrl_real);
    avi_seg_body.extend_from_slice(&movi1_chunk);
    avi_seg_body.extend_from_slice(&ix1_chunk_real);
    // RIFF wrapper for the AVI segment.
    let mut file = Vec::new();
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&(avi_seg_body.len() as u32).to_le_bytes());
    file.extend_from_slice(&avi_seg_body);

    // Assemble AVIX segment body.
    let mut avix_seg_body = Vec::new();
    avix_seg_body.extend_from_slice(b"AVIX");
    avix_seg_body.extend_from_slice(&movi2_chunk);
    avix_seg_body.extend_from_slice(&ix2_chunk_real);
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&(avix_seg_body.len() as u32).to_le_bytes());
    file.extend_from_slice(&avix_seg_body);

    // Sanity: confirm the actual byte positions match what we
    // computed (catches any off-by-one in the layout planning).
    assert_eq!(
        &file[movi1_offset..movi1_offset + 4],
        b"LIST",
        "movi#1 should start with LIST at the planned offset"
    );
    assert_eq!(
        &file[movi1_body_offset - 4..movi1_body_offset],
        b"movi",
        "movi#1 type fourcc should sit just before the body"
    );
    assert_eq!(&file[ix1_offset..ix1_offset + 4], b"ix00");
    assert_eq!(&file[movi2_offset..movi2_offset + 4], b"LIST");
    assert_eq!(&file[movi2_body_offset - 4..movi2_body_offset], b"movi");
    assert_eq!(&file[ix2_offset..ix2_offset + 4], b"ix00");

    (file, payloads)
}

/// Build a placeholder indx chunk with the right byte size for two
/// AVI_INDEX_OF_INDEXES entries but zeroed qwOffset / dwSize so we
/// can measure the chunk's overall size before knowing the real
/// offsets of the ix00 chunks it points at.
fn build_indx_placeholder() -> Vec<u8> {
    build_indx_real(&[(0, 0, 0), (0, 0, 0)])
}

/// Build a real indx (AVI_INDEX_OF_INDEXES) referring to the given
/// `(qwOffset, dwSize, dwDuration)` triples.
fn build_indx_real(entries: &[(u64, u32, u32)]) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&4u16.to_le_bytes()); // wLongsPerEntry=4
    body.push(0); // bIndexSubType
    body.push(0x00); // bIndexType=AVI_INDEX_OF_INDEXES
    body.extend_from_slice(&(entries.len() as u32).to_le_bytes()); // nEntriesInUse
    body.extend_from_slice(b"00dc"); // dwChunkId
    body.extend_from_slice(&[0u8; 12]); // dwReserved[3]
    for (qw_off, dw_size, dw_duration) in entries {
        body.extend_from_slice(&qw_off.to_le_bytes());
        body.extend_from_slice(&dw_size.to_le_bytes());
        body.extend_from_slice(&dw_duration.to_le_bytes());
    }
    chunk(b"indx", &body)
}

/// Build hdrl LIST containing avih (dwTotalFrames=avih_total),
/// strl with XVID strh+strf+indx, and odml LIST with dmlh
/// (dwTotalFrames=dmlh_total).
fn build_hdrl(indx_chunk: &[u8], dmlh_total: u32, avih_total: u32) -> Vec<u8> {
    // avih: u32 dwMicroSecPerFrame, dwMaxBytesPerSec, dwPaddingGranularity,
    // dwFlags, dwTotalFrames, then enough zeros to fill 56 bytes.
    let mut avih_body = Vec::with_capacity(56);
    avih_body.extend_from_slice(&33333u32.to_le_bytes()); // ~30 fps
    avih_body.extend_from_slice(&[0u8; 12]); // bytes/sec, padding, flags
    avih_body.extend_from_slice(&avih_total.to_le_bytes());
    avih_body.extend_from_slice(&[0u8; 32]); // initial frames + remaining 7 fields
    let avih_chunk = chunk(b"avih", &avih_body);

    // strl with XVID + indx tacked on the end (lives inside strl per
    // the OpenDML spec).
    let strh_chunk = {
        let mut strh = Vec::with_capacity(56);
        strh.extend_from_slice(b"vids");
        strh.extend_from_slice(b"XVID");
        strh.extend_from_slice(&[0u8; 12]);
        strh.extend_from_slice(&1u32.to_le_bytes()); // dwScale
        strh.extend_from_slice(&30u32.to_le_bytes()); // dwRate
        strh.extend_from_slice(&[0u8; 24]);
        chunk(b"strh", &strh)
    };
    let strf_chunk = {
        let mut strf = Vec::with_capacity(40);
        strf.extend_from_slice(&40u32.to_le_bytes());
        strf.extend_from_slice(&320i32.to_le_bytes());
        strf.extend_from_slice(&240i32.to_le_bytes());
        strf.extend_from_slice(&1u16.to_le_bytes());
        strf.extend_from_slice(&24u16.to_le_bytes());
        strf.extend_from_slice(b"XVID");
        strf.extend_from_slice(&[0u8; 20]);
        chunk(b"strf", &strf)
    };
    let mut strl_body = Vec::new();
    strl_body.extend_from_slice(&strh_chunk);
    strl_body.extend_from_slice(&strf_chunk);
    strl_body.extend_from_slice(indx_chunk);
    let strl_chunk = list(b"strl", &strl_body);

    // odml LIST: contains dmlh chunk with the total frame count.
    let dmlh_chunk = {
        let mut body = Vec::new();
        body.extend_from_slice(&dmlh_total.to_le_bytes());
        // dmlh is allowed to contain more reserved fields; we keep
        // it minimal at 4 bytes — every parser only reads the first
        // u32.
        chunk(b"dmlh", &body)
    };
    let odml_chunk = list(b"odml", &dmlh_chunk);

    let mut hdrl_body = Vec::new();
    hdrl_body.extend_from_slice(&avih_chunk);
    hdrl_body.extend_from_slice(&strl_chunk);
    hdrl_body.extend_from_slice(&odml_chunk);
    list(b"hdrl", &hdrl_body)
}

#[test]
fn opendml_streaming_walks_both_movi_lists_in_order() {
    let (file, expected) = build_opendml_two_movi_six_samples();
    let mut d = demux_avi_streaming_init(bytes::Bytes::from(file.clone())).expect("OpenDML init");
    // dmlh.dwTotalFrames=6 should win over avih.dwTotalFrames=3.
    assert_eq!(d.header.info.total_frames, 6);
    // Drain — six samples, in superindex (file) order.
    let mut got = Vec::new();
    while let Some(s) = d.next_video_sample().expect("next") {
        got.push(s.data);
    }
    assert_eq!(
        got.len(),
        6,
        "should pull all six samples across both movi LISTs"
    );
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g, e,
            "sample {i} mismatch — OpenDML walk lost ordering or content"
        );
    }
}

#[test]
fn opendml_legacy_demux_also_walks_both_movi_lists() {
    // The legacy `demux_avi` (Vec materialization path) must also
    // pick up multi-movi for the bench / fidelity tests that don't
    // use streaming.
    let (file, expected) = build_opendml_two_movi_six_samples();
    let d = demux_avi(&file).expect("legacy demux");
    assert_eq!(d.samples.len(), 6);
    for (i, (g, e)) in d.samples.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "legacy sample {i} mismatch");
    }
    assert_eq!(
        d.info.total_frames, 6,
        "legacy total_frames should honor dmlh"
    );
}

#[test]
fn opendml_total_frames_prefers_dmlh_over_avih() {
    let (file, _) = build_opendml_two_movi_six_samples();
    let d = demux_avi_streaming_init(bytes::Bytes::from(file.clone())).expect("init");
    assert_eq!(
        d.header.info.total_frames, 6,
        "dmlh.dwTotalFrames (6) must win over avih.dwTotalFrames (3)"
    );
    // Duration sanity: 6 frames / 30 fps = 0.2s (frame_rate from strh).
    assert!(
        (d.header.info.duration - 0.2).abs() < 1e-6,
        "duration = total_frames / frame_rate, got {}",
        d.header.info.duration
    );
}

#[test]
fn opendml_picks_indx_path_not_cursor_walk() {
    // White-box: the demuxer's backend should be OpenDml when the
    // input has an indx superindex. Confirms the dispatch took the
    // intended path and we're not accidentally running the cursor
    // walk over both movi LISTs (which would also pass the sample-
    // count test but defeats the streaming RSS goal for >1 GiB
    // files because the cursor walk reads through every byte).
    let (file, _) = build_opendml_two_movi_six_samples();
    let d = demux_avi_streaming_init(bytes::Bytes::from(file.clone())).expect("init");
    assert!(
        matches!(d.backend, Backend::OpenDml { .. }),
        "fixture has indx — backend must be OpenDml"
    );
}

#[test]
fn legacy_single_movi_without_indx_uses_cursor_backend() {
    // Backward-compat: a single-movi AVI without indx must still
    // work via the legacy cursor path (Squad-13's contract).
    let mut hdrl_body = Vec::new();
    hdrl_body.extend_from_slice(&chunk(b"avih", &[0u8; 56]));
    hdrl_body.extend_from_slice(&video_strl(b"XVID", b"XVID", 320, 240, 30, 1));
    let hdrl = list(b"hdrl", &hdrl_body);
    let mut movi_body = Vec::new();
    movi_body.extend_from_slice(&chunk(b"00dc", b"f0"));
    movi_body.extend_from_slice(&chunk(b"00dc", b"f1"));
    let movi = list(b"movi", &movi_body);
    let mut riff_body = Vec::new();
    riff_body.extend_from_slice(b"AVI ");
    riff_body.extend_from_slice(&hdrl);
    riff_body.extend_from_slice(&movi);
    let mut file = Vec::new();
    file.extend_from_slice(b"RIFF");
    file.extend_from_slice(&(riff_body.len() as u32).to_le_bytes());
    file.extend_from_slice(&riff_body);

    let mut d = demux_avi_streaming_init(bytes::Bytes::from(file.clone())).expect("init");
    assert!(
        matches!(d.backend, Backend::Cursor(_)),
        "no indx → must take cursor backend (legacy path)"
    );
    let s0 = d.next_video_sample().unwrap().unwrap();
    let s1 = d.next_video_sample().unwrap().unwrap();
    assert_eq!(s0.data, b"f0");
    assert_eq!(s1.data, b"f1");
    assert!(d.next_video_sample().unwrap().is_none());
}

#[test]
fn parse_indx_body_decodes_two_index_of_indexes_entries() {
    // Direct test of the indx body parser — wire layout regression.
    let entries = [
        (0xDEAD_BEEFu64, 0x1234u32, 100u32),
        (0xCAFE_F00Du64, 0x5678u32, 200u32),
    ];
    let chunk_bytes = build_indx_real(&entries);
    // Skip the 8-byte chunk header to get the body.
    let body = &chunk_bytes[8..8 + (chunk_bytes.len() - 8 - (chunk_bytes.len() & 1))];
    let parsed = parse_indx_body(body).expect("parse");
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0], (0xDEAD_BEEFusize, 0x1234usize));
    assert_eq!(parsed[1], (0xCAFE_F00Dusize, 0x5678usize));
}

#[test]
fn read_dmlh_total_frames_finds_value_inside_odml_list() {
    let dmlh_chunk = {
        let mut body = Vec::new();
        body.extend_from_slice(&42u32.to_le_bytes());
        body.extend_from_slice(&[0u8; 244]); // pad to spec's 248-byte minimum
        chunk(b"dmlh", &body)
    };
    let odml = list(b"odml", &dmlh_chunk);
    let mut hdrl_body = Vec::new();
    hdrl_body.extend_from_slice(&chunk(b"avih", &[0u8; 56]));
    hdrl_body.extend_from_slice(&odml);
    // Strip the outer LIST header — read_dmlh_total_frames takes the
    // hdrl body (starts after `hdrl` type fourcc).
    assert_eq!(read_dmlh_total_frames(&hdrl_body), Some(42));
}

#[test]
fn read_dmlh_total_frames_returns_none_when_odml_absent() {
    let mut hdrl_body = Vec::new();
    hdrl_body.extend_from_slice(&chunk(b"avih", &[0u8; 56]));
    // No odml LIST → fall through to None.
    assert_eq!(read_dmlh_total_frames(&hdrl_body), None);
}

// ----- length-prefixed H.264 (an avcC record in strf) -----

/// The avcC record `ffmpeg -i clip.mp4 -c copy clip.avi` wrote into `strf`
/// (`biSize` 87 = 40 + these 47 bytes): High@3.0 640x360, 4-byte lengths,
/// one SPS, one PPS, the High-profile tail.
const CLIP_AVCC: &str = "0164001effe1001a6764001eacd940a02ff970110000030001000003003c0f162d9601000668ebe1b2c8b0fdf8f800";

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

/// Length-prefixed (4-byte) access unit from NAL units.
fn length_prefixed_au(nals: &[&[u8]]) -> Vec<u8> {
    nals.iter().flat_map(|n| [&(n.len() as u32).to_be_bytes()[..], *n].concat()).collect()
}

/// Annex-B access unit from NAL units.
fn annexb_au(nals: &[&[u8]]) -> Vec<u8> {
    nals.iter().flat_map(|n| [&[0u8, 0, 0, 1][..], *n].concat()).collect()
}

/// A one-stream H.264 AVI (fourcc `H264`, 640x360 at 30/1) whose `strf`
/// carries `extradata` after the BITMAPINFOHEADER, `biSize` covering it, as
/// ffmpeg writes it; one `00dc` chunk per sample.
fn h264_avi(extradata: &[u8], samples: &[Vec<u8>]) -> Vec<u8> {
    let mut strh = b"vids".to_vec();
    strh.extend_from_slice(b"H264");
    strh.extend_from_slice(&[0u8; 12]);
    strh.extend_from_slice(&1u32.to_le_bytes()); // dwScale
    strh.extend_from_slice(&30u32.to_le_bytes()); // dwRate
    strh.extend_from_slice(&[0u8; 24]);
    let mut strf = ((40 + extradata.len()) as u32).to_le_bytes().to_vec(); // biSize
    strf.extend_from_slice(&640i32.to_le_bytes());
    strf.extend_from_slice(&360i32.to_le_bytes());
    strf.extend_from_slice(&1u16.to_le_bytes());
    strf.extend_from_slice(&24u16.to_le_bytes());
    strf.extend_from_slice(b"H264");
    strf.extend_from_slice(&[0u8; 20]);
    strf.extend_from_slice(extradata);
    let mut strl = chunk(b"strh", &strh);
    strl.extend_from_slice(&chunk(b"strf", &strf));
    let mut hdrl_body = chunk(b"avih", &[0u8; 56]);
    hdrl_body.extend_from_slice(&list(b"strl", &strl));
    let movi_body: Vec<u8> = samples.iter().flat_map(|s| chunk(b"00dc", s)).collect();
    let mut riff_body = b"AVI ".to_vec();
    riff_body.extend_from_slice(&list(b"hdrl", &hdrl_body));
    riff_body.extend_from_slice(&list(b"movi", &movi_body));
    let mut file = b"RIFF".to_vec();
    file.extend_from_slice(&(riff_body.len() as u32).to_le_bytes());
    file.extend_from_slice(&riff_body);
    file
}

fn drain_samples(file: &[u8]) -> Vec<Vec<u8>> {
    let mut d = demux_avi_streaming_init(bytes::Bytes::from(file.to_vec())).expect("init");
    let mut out = Vec::new();
    while let Some(s) = d.next_video_sample().expect("next") {
        out.push(s.data);
    }
    out
}

/// `-c copy` from an MP4 stores H.264 in AVI length-prefixed with an avcC
/// record in `strf`; develop handed those samples to the decoder as they
/// were and it decoded nothing. Both demuxers now convert them to Annex-B
/// through the MP4 / MKV converter — the parameter sets from the record
/// ahead of the first IDR, after the NAL units that precede it — and hand
/// out exactly what that converter makes of them.
#[test]
fn length_prefixed_h264_in_avi_is_converted_to_annexb_like_mp4() {
    use crate::annexb::{NaluCodec, ParamSetTracker, length_prefixed_to_annexb_tracked, parse_avcc};
    let avcc = unhex(CLIP_AVCC);
    let (sps, pps) = (&avcc[8..34], &avcc[37..43]);
    assert_eq!((sps[0] & 0x1f, pps[0] & 0x1f), (7, 8), "fixture offsets");
    let aud: &[u8] = &[0x09, 0xf0];
    let idr: &[u8] = &[0x65, 0x88, 0x84, 0x00, 0x10];
    let p: &[u8] = &[0x41, 0x9a, 0x02, 0x03];
    let samples = vec![length_prefixed_au(&[aud, idr]), length_prefixed_au(&[p])];
    let file = h264_avi(&avcc, &samples);

    let want = vec![annexb_au(&[aud, sps, pps, idr]), annexb_au(&[p])];
    let got = drain_samples(&file);
    assert_eq!(got, want);
    let cfg = parse_avcc(&avcc).expect("avcC");
    let mut tracker = ParamSetTracker::new(NaluCodec::Avc);
    let mp4_way: Vec<Vec<u8>> = samples
        .iter()
        .map(|s| length_prefixed_to_annexb_tracked(s, cfg.length_size, &mut tracker, &cfg.parameter_sets))
        .collect();
    assert_eq!(got, mp4_way, "the MP4 path's converter, sample for sample");
    assert_eq!(demux_avi(&file).expect("legacy demux").samples, want);

    // The first sample, peeked for the header's colour and pixel format, is
    // the converted one: the SPS parses (8-bit 4:2:0).
    let d = demux_avi_streaming_init(bytes::Bytes::from(file.clone())).expect("init");
    assert_eq!(d.header.info.pixel_format, PixelFormat::Yuv420p);
    assert!(super::riff::length_prefixed("h264", &avcc).is_some());
    assert!(super::riff::length_prefixed("mpeg4", &avcc).is_none());
}

/// Annex-B H.264 in AVI — start-code parameter sets in `strf`, as
/// `-bsf:v h264_mp4toannexb` writes it, or no extradata at all — is handed
/// out byte for byte.
#[test]
fn annexb_h264_in_avi_is_left_untouched() {
    let avcc = unhex(CLIP_AVCC);
    let (sps, pps) = (&avcc[8..34], &avcc[37..43]);
    let idr: &[u8] = &[0x65, 0x88, 0x84, 0x00, 0x10];
    let p: &[u8] = &[0x41, 0x9a, 0x02, 0x03];
    let samples = vec![annexb_au(&[sps, pps, idr]), annexb_au(&[p])];
    for extradata in [annexb_au(&[sps, pps]), Vec::new()] {
        assert!(super::riff::length_prefixed("h264", &extradata).is_none());
        let file = h264_avi(&extradata, &samples);
        assert_eq!(drain_samples(&file), samples);
        assert_eq!(demux_avi(&file).expect("legacy demux").samples, samples);
    }
}

// ----- empty chunks (a time base finer than the frame rate) -----

/// A legacy (no `indx`) AVI the way ffmpeg writes a stream whose time base
/// is finer than its frame rate — `ffmpeg -i clip.mp4 -c copy clip.avi` puts
/// a 30 fps H.264 stream on `strh` rate 600 / scale 1, one frame every 20
/// chunks with the 19 between them empty. `frames` frames on a `scale/rate`
/// time base, each followed by an audio chunk and `fill` empty `00dc`
/// chunks; `avih.dwTotalFrames` is the chunk count, as ffmpeg writes it.
fn filler_avi(rate: u32, scale: u32, frames: usize, fill: usize) -> Vec<u8> {
    let mut avih = vec![0u8; 56];
    avih[16..20].copy_from_slice(&((frames * (1 + fill)) as u32).to_le_bytes());
    let mut hdrl_body = chunk(b"avih", &avih);
    hdrl_body.extend_from_slice(&video_strl(b"XVID", b"XVID", 320, 240, rate, scale));
    let hdrl = list(b"hdrl", &hdrl_body);

    let mut movi_body = Vec::new();
    for i in 0..frames {
        movi_body.extend_from_slice(&chunk(b"00dc", format!("frame-{i}").as_bytes()));
        movi_body.extend_from_slice(&chunk(b"01wb", b"audio"));
        for _ in 0..fill {
            movi_body.extend_from_slice(&chunk(b"00dc", b""));
        }
    }
    let movi = list(b"movi", &movi_body);

    let mut riff_body = b"AVI ".to_vec();
    riff_body.extend_from_slice(&hdrl);
    riff_body.extend_from_slice(&movi);
    let mut file = b"RIFF".to_vec();
    file.extend_from_slice(&(riff_body.len() as u32).to_le_bytes());
    file.extend_from_slice(&riff_body);
    file
}

/// Drain a streaming demuxer to `(pts_ticks, data)` pairs.
fn drain_timed(d: &mut AviStreamingDemuxer) -> Vec<(i64, Vec<u8>)> {
    let mut out = Vec::new();
    while let Some(s) = d.next_video_sample().expect("next") {
        out.push((s.pts_ticks, s.data));
    }
    out
}

/// Empty chunks are dropped or repeated frames' slots, not frames: both
/// demuxers count only the frames and take the frame rate from them, the
/// streaming one hands out only the frames, and a frame's `pts_ticks` is its
/// chunk position × `dwScale` on a `dwRate` timescale. Before, this file read
/// 120 frames at 600 fps (a 0.2 s output from a 4 s source; CLI progress
/// `120/2400 frames`).
#[test]
fn empty_chunks_are_slots_not_frames() {
    let file = filler_avi(600, 1, 6, 19); // 120 chunks, 6 frames: 30 fps, 0.2 s

    let mut d = demux_avi_streaming_init(bytes::Bytes::from(file.clone())).expect("init");
    assert_eq!(d.header.info.total_frames, 6);
    assert!((d.header.info.frame_rate - 30.0).abs() < 1e-9, "fps {}", d.header.info.frame_rate);
    assert!((d.header.info.duration - 0.2).abs() < 1e-9, "duration {}", d.header.info.duration);
    assert_eq!(d.header.timescale, 600);
    let want: Vec<(i64, Vec<u8>)> =
        (0..6).map(|i| (i * 20, format!("frame-{i}").into_bytes())).collect();
    assert_eq!(drain_timed(&mut d), want);

    let legacy = demux_avi(&file).expect("legacy demux");
    let want_samples: Vec<Vec<u8>> = want.into_iter().map(|(_, s)| s).collect();
    assert_eq!(legacy.samples, want_samples);
    assert_eq!(legacy.info.total_frames, 6);
    assert!((legacy.info.frame_rate - 30.0).abs() < 1e-9, "fps {}", legacy.info.frame_rate);
    assert!((legacy.info.duration - 0.2).abs() < 1e-9, "duration {}", legacy.info.duration);
}

/// A rate that is not a whole number of ticks a second keeps exact
/// timestamps: on 1001/60000 with a frame every other chunk, frame `i` is at
/// `2i × 1001` sixty-thousandths — 29.97 fps.
#[test]
fn a_fractional_time_base_keeps_exact_chunk_timestamps() {
    let file = filler_avi(60000, 1001, 5, 1);
    let mut d = demux_avi_streaming_init(bytes::Bytes::from(file)).expect("init");
    assert_eq!(d.header.timescale, 60000);
    assert_eq!(d.header.info.total_frames, 5);
    assert!((d.header.info.frame_rate - 30000.0 / 1001.0).abs() < 1e-9, "fps {}", d.header.info.frame_rate);
    let pts: Vec<i64> = drain_timed(&mut d).into_iter().map(|(p, _)| p).collect();
    assert_eq!(pts, [0, 2002, 4004, 6006, 8008]);
    assert!((d.header.pts_seconds(pts[1]) - 1001.0 / 30000.0).abs() < 1e-12);
}

/// With no empty chunk nothing changes: the header count, the `strh` rate,
/// every chunk a sample at consecutive ticks.
#[test]
fn a_stream_without_empty_chunks_keeps_its_header_count_and_rate() {
    let file = filler_avi(30, 1, 4, 0);
    let mut d = demux_avi_streaming_init(bytes::Bytes::from(file.clone())).expect("init");
    assert_eq!(d.header.info.total_frames, 4);
    assert_eq!(d.header.info.frame_rate, 30.0);
    assert_eq!(d.header.timescale, 30);
    let pts: Vec<i64> = drain_timed(&mut d).into_iter().map(|(p, _)| p).collect();
    assert_eq!(pts, [0, 1, 2, 3]);
    let legacy = demux_avi(&file).expect("legacy demux");
    assert_eq!((legacy.samples.len(), legacy.info.total_frames), (4, 4));
    assert_eq!(legacy.info.frame_rate, 30.0);
}

/// The header walk counts what the sample walks hand out, `rec ` LISTs
/// included, and stops where they stop on a truncated chunk.
#[test]
fn count_movi_video_chunks_matches_the_sample_walk() {
    use super::riff::{count_movi_video_chunks, frames_per_second};
    let mut rec_body = chunk(b"00dc", b"in-rec");
    rec_body.extend_from_slice(&chunk(b"00dc", b""));
    let mut movi_body = chunk(b"00dc", b"a");
    movi_body.extend_from_slice(&chunk(b"00db", b""));
    movi_body.extend_from_slice(&chunk(b"01wb", b"audio"));
    movi_body.extend_from_slice(&chunk(b"00dd", b"keyframe-index"));
    movi_body.extend_from_slice(&list(b"rec ", &rec_body));
    let whole = movi_body.len();
    // A truncated last chunk: its header claims more bytes than remain.
    movi_body.extend_from_slice(b"00dc");
    movi_body.extend_from_slice(&100u32.to_le_bytes());
    movi_body.extend_from_slice(b"short");

    assert_eq!(count_movi_video_chunks(&movi_body, &[(0, movi_body.len())], b"00"), (4, 2));
    let mut samples = Vec::new();
    let chunks = collect_movi_samples(&movi_body, "00", &mut samples).expect("walk");
    assert_eq!((chunks, samples.len() as u64), (4, 2));
    assert_eq!(samples, [b"a".to_vec(), b"in-rec".to_vec()]);
    assert_eq!(count_movi_video_chunks(&movi_body, &[(0, whole)], b"00"), (4, 2));
    // A stream with no chunks counts nothing. (Like both sample walks, the
    // count matches `##dc` / `##db` by prefix and last byte only, so it is
    // only ever asked about the video stream's prefix: `01wb` ends in `b`.)
    assert_eq!(count_movi_video_chunks(&movi_body, &[(0, whole)], b"02"), (0, 0));

    assert_eq!(frames_per_second(600.0, 2400, 120), 30.0);
    assert_eq!(frames_per_second(30.0, 120, 120), 30.0);
    assert_eq!(frames_per_second(25.0, 0, 0), 25.0);
}

// ----- audio -----

/// An `auds` strl: strh (`dwScale`, `dwRate`, `dwStart`, `dwSampleSize`) and
/// a WAVEFORMATEX with `extra` after it.
fn audio_strl(tag: u16, channels: u16, rate_hz: u32, block_align: u16, bits: u16, extra: &[u8], strh: (u32, u32, u32, u32)) -> Vec<u8> {
    let (scale, rate, start, sample_size) = strh;
    let mut h = b"auds".to_vec();
    h.extend_from_slice(&[0u8; 4]); // fccHandler
    h.extend_from_slice(&[0u8; 12]); // flags, priority, language, initial frames
    h.extend_from_slice(&scale.to_le_bytes());
    h.extend_from_slice(&rate.to_le_bytes());
    h.extend_from_slice(&start.to_le_bytes());
    h.extend_from_slice(&[0u8; 12]); // length, buffer size, quality
    h.extend_from_slice(&sample_size.to_le_bytes());
    h.extend_from_slice(&[0u8; 8]); // rcFrame
    let mut f = tag.to_le_bytes().to_vec();
    f.extend_from_slice(&channels.to_le_bytes());
    f.extend_from_slice(&rate_hz.to_le_bytes());
    f.extend_from_slice(&0u32.to_le_bytes()); // nAvgBytesPerSec
    f.extend_from_slice(&block_align.to_le_bytes());
    f.extend_from_slice(&bits.to_le_bytes());
    f.extend_from_slice(&(extra.len() as u16).to_le_bytes());
    f.extend_from_slice(extra);
    let mut strl = chunk(b"strh", &h);
    strl.extend_from_slice(&chunk(b"strf", &f));
    list(b"strl", &strl)
}

/// An XVID video stream (0) at 30/1 with one frame, then the audio stream
/// (1) and its `01wb` chunks after the frame.
fn audio_avi(audio: Vec<u8>, chunks: &[&[u8]]) -> Vec<u8> {
    let mut hdrl_body = chunk(b"avih", &[0u8; 56]);
    hdrl_body.extend_from_slice(&video_strl(b"XVID", b"XVID", 320, 240, 30, 1));
    hdrl_body.extend_from_slice(&audio);
    let mut movi_body = chunk(b"00dc", b"frame");
    for c in chunks {
        movi_body.extend_from_slice(&chunk(b"01wb", c));
    }
    let mut riff_body = b"AVI ".to_vec();
    riff_body.extend_from_slice(&list(b"hdrl", &hdrl_body));
    riff_body.extend_from_slice(&list(b"movi", &movi_body));
    let mut file = b"RIFF".to_vec();
    file.extend_from_slice(&(riff_body.len() as u32).to_le_bytes());
    file.extend_from_slice(&riff_body);
    file
}

/// Both demuxers read the same audio: the track and its edit.
fn both_audio(file: &[u8]) -> (crate::demux::AudioTrack, Option<crate::edit::AudioEdit>) {
    let legacy = demux_avi(file).expect("legacy demux");
    let d = demux_avi_streaming_init(bytes::Bytes::from(file.to_vec())).expect("init");
    let track = d.audio().cloned().expect("an audio track");
    let l = legacy.audio.expect("legacy audio");
    assert_eq!(
        (&l.codec, &l.samples, &l.durations, l.timescale),
        (&track.codec, &track.samples, &track.durations, track.timescale)
    );
    assert_eq!(legacy.audio_edit, d.audio_edit());
    (track, d.audio_edit())
}

/// PCM is a byte stream (`dwSampleSize` = `nBlockAlign`): a chunk lasts its
/// bytes over the block size in samples, and a chunk that does not end on a
/// block is still handed over whole (the decoder joins the pieces).
#[test]
fn pcm_audio_is_read_with_its_timeline() {
    let strl = audio_strl(0x0001, 2, 48_000, 4, 16, &[], (1, 48_000, 0, 4));
    let a = vec![1u8; 4000];
    let b = vec![2u8; 4002];
    let file = audio_avi(strl, &[&a, &b, &[3u8; 2]]);
    let (track, edit) = both_audio(&file);
    assert_eq!(track.codec, "pcm_s16le");
    assert_eq!((track.sample_rate, track.channels, track.timescale), (48_000, 2, 48_000));
    assert_eq!(track.samples, vec![a, b, vec![3u8; 2]]);
    // 1000 blocks, 1000 (4002 / 4, floored as ffmpeg does), then a 2-byte
    // chunk that spans no whole block (at least one tick).
    assert_eq!(track.durations, vec![1000, 1000, 1]);
    assert_eq!(edit, None);
    for (bits, codec) in [(8u16, "pcm_u8"), (24, "pcm_s24le"), (32, "pcm_s32le")] {
        let strl = audio_strl(0x0001, 1, 44_100, bits / 8, bits, &[], (1, 44_100, 0, u32::from(bits / 8)));
        assert_eq!(both_audio(&audio_avi(strl, &[&[0u8; 12]])).0.codec, codec);
    }
    let float = audio_strl(0x0003, 1, 48_000, 4, 32, &[], (1, 48_000, 0, 4));
    assert_eq!(both_audio(&audio_avi(float, &[&[0u8; 8]])).0.codec, "pcm_f32le");
    // WAVE_FORMAT_EXTENSIBLE (5.1): the sub-format GUID names the format.
    let mut ext = vec![16, 0, 0x3f, 0, 0, 0]; // valid bits, channel mask
    ext.extend_from_slice(&[1, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71]); // KSDATAFORMAT_SUBTYPE_PCM
    let strl = audio_strl(0xFFFE, 6, 48_000, 12, 16, &ext, (1, 48_000, 0, 12));
    let (track, _) = both_audio(&audio_avi(strl, &[&[0u8; 120]]));
    assert_eq!((track.codec.as_str(), track.channels, track.durations.as_slice()), ("pcm_s16le", 6, &[10u32][..]));
}

/// One frame to a chunk (`dwSampleSize` 0), as ffmpeg writes AAC: every
/// chunk one `dwScale / dwRate` unit (1024 samples here), the empty chunks
/// ffmpeg's muxer leaves behind taking no time — ffmpeg's `get_duration`
/// counts a chunk's bytes over `nBlockAlign`, rounded up. The ASC comes from
/// the WAVEFORMATEX extra bytes.
#[test]
fn aac_frames_are_timed_by_the_stream_header_and_empty_chunks_take_no_time() {
    let strl = audio_strl(0x00FF, 2, 48_000, 768, 16, &[0x11, 0x90], (1024, 48_000, 0, 0));
    let file = audio_avi(strl, &[b"f0", b"", b"", b"f1", b"f2"]);
    let (track, edit) = both_audio(&file);
    assert_eq!(track.codec, "aac");
    assert_eq!(track.asc, vec![0x11, 0x90]);
    assert_eq!((track.sample_rate, track.channels, track.timescale), (48_000, 2, 48_000));
    assert_eq!(track.samples, vec![b"f0".to_vec(), b"f1".to_vec(), b"f2".to_vec()]);
    assert_eq!(track.durations, vec![1024, 1024, 1024]);
    assert_eq!(edit, None);
    // With no nBlockAlign, ffmpeg counts every chunk as one unit, empty or
    // not: the empty chunks are a gap, which the previous frame's duration
    // carries (and before the first frame, a delay).
    let strl = audio_strl(0x00FF, 2, 48_000, 0, 16, &[0x11, 0x90], (1024, 48_000, 0, 0));
    let (track, edit) = both_audio(&audio_avi(strl, &[b"", b"f0", b"", b"", b"f1"]));
    assert_eq!(track.durations, vec![3 * 1024, 1024]);
    assert_eq!(edit, Some(crate::edit::AudioEdit { delay: 1024, media_start: 0, media_end: None }));
}

/// A 48 kHz 192 kb/s stereo AC-3 syncframe header (bsid 8), zero-padded.
fn ac3_frame() -> Vec<u8> {
    let mut f = vec![0x0B, 0x77, 0, 0, 0x14, 0x40, 0x40];
    f.resize(64, 0);
    f
}

/// `dwStart` is where the stream begins, in `dwScale / dwRate` units — what
/// ffmpeg stamps the first packet with: a late start is the edit's delay.
#[test]
fn a_late_start_is_a_delay() {
    let strl = audio_strl(0x0001, 1, 48_000, 2, 16, &[], (1, 48_000, 24_000, 2));
    let (track, edit) = both_audio(&audio_avi(strl, &[&[0u8; 960]]));
    assert_eq!(track.durations, vec![480]);
    assert_eq!(edit, Some(crate::edit::AudioEdit { delay: 24_000, media_start: 0, media_end: None }));
    // On a coarser time base: 15 AC-3 frame units of 4/125 s = 0.48 s.
    let ac3 = ac3_frame();
    let strl = audio_strl(0x2000, 2, 48_000, 3840, 0, &[], (4, 125, 15, 0));
    let (track, edit) = both_audio(&audio_avi(strl, &[&ac3, &ac3]));
    assert_eq!((track.codec.as_str(), track.sample_rate, track.channels), ("ac3", 48_000, 2));
    assert_eq!(track.codec_private.len(), 3, "a dac3 body");
    assert_eq!(track.durations, vec![1536, 1536]);
    assert_eq!(edit.map(|e| e.delay), Some(15 * 1536));
}

/// A format rivet has no path for is surfaced by name with no packets, so
/// the audio stage says which codec it dropped; so is an AAC stream with no
/// AudioSpecificConfig and AC-3 that is not stored a syncframe a chunk.
#[test]
fn an_unusable_format_is_named_not_guessed() {
    for (tag, name) in [(0x0161u16, "wmav2"), (0x0002, "adpcm_ms"), (0x0007, "pcm_mulaw"), (0x1234, "avi_audio_0x1234")] {
        let strl = audio_strl(tag, 1, 48_000, 682, 16, &[], (341, 8000, 0, 682));
        let (track, edit) = both_audio(&audio_avi(strl, &[&[7u8; 682]]));
        assert_eq!(track.codec, name);
        assert!(track.samples.is_empty() && track.durations.is_empty(), "{name}");
        assert_eq!(edit, None);
    }
    let strl = audio_strl(0x00FF, 2, 48_000, 768, 16, &[], (1024, 48_000, 0, 0));
    assert_eq!(both_audio(&audio_avi(strl, &[&[0xFF, 0xF1, 0x50, 0x80]])).0.codec, "aac_adts");
    let strl = audio_strl(0x2000, 2, 48_000, 1, 0, &[], (1, 24_000, 0, 1));
    let (track, _) = both_audio(&audio_avi(strl, &[&[0x12, 0x34, 0x56]]));
    assert_eq!((track.codec.as_str(), track.samples.len()), ("ac3", 0));
}

/// No audio stream, or only empty audio chunks: no track.
#[test]
fn no_audio_stream_no_track() {
    let file = filler_avi(30, 1, 2, 0); // its `01wb` chunks belong to no stream
    assert!(demux_avi(&file).expect("demux").audio.is_none());
    let d = demux_avi_streaming_init(bytes::Bytes::from(file)).expect("init");
    assert!(d.audio().is_none() && d.audio_edit().is_none());
    let strl = audio_strl(0x0001, 1, 48_000, 2, 16, &[], (1, 48_000, 0, 2));
    assert!(demux_avi(&audio_avi(strl, &[b"", b""])).expect("demux").audio.is_none());
}

