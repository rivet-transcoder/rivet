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
    for i in 0..3 {
        let cur_off = movi1_body.len();
        // Chunk header is 8 bytes; data starts at cur_off + 8.
        let c = chunk(b"00dc", &payloads[i]);
        movi1_body.extend_from_slice(&c);
        chunk_data_offsets_in_movi1.push((cur_off + 8, payloads[i].len()));
    }

    // movi#2 body: three 00dc chunks with payloads 3, 4, 5.
    let mut movi2_body = Vec::new();
    let mut chunk_data_offsets_in_movi2 = Vec::new();
    for i in 3..6 {
        let cur_off = movi2_body.len();
        let c = chunk(b"00dc", &payloads[i]);
        movi2_body.extend_from_slice(&c);
        chunk_data_offsets_in_movi2.push((cur_off + 8, payloads[i].len()));
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
    let mut d = demux_avi_streaming_init(&file).expect("OpenDML init");
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
    let d = demux_avi_streaming_init(&file).expect("init");
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
    let d = demux_avi_streaming_init(&file).expect("init");
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

    let mut d = demux_avi_streaming_init(&file).expect("init");
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
