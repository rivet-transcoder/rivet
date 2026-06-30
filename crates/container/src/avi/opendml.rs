//! OpenDML 1.0 super-index support: `indx`/`ix##` chain, `dmlh` total-frames
//! override, and the `avih` fallback for single-RIFF files.

// ---------------------------------------------------------------------------
// Total-frames readers
// ---------------------------------------------------------------------------

/// Read `dwTotalFrames` from the MainAVIHeader (`avih`) chunk that lives
/// as the first chunk inside `LIST hdrl`. AVIMAINHEADER body layout
/// (Microsoft AVI RIFF reference):
///   u32 dwMicroSecPerFrame      // 0..4
///   u32 dwMaxBytesPerSec        // 4..8
///   u32 dwPaddingGranularity    // 8..12
///   u32 dwFlags                 // 12..16
///   u32 dwTotalFrames           // 16..20  ← what we want
///   ...
/// Returns `None` if the avih chunk is missing, shorter than 20 bytes,
/// or the field is zero (some encoders leave it unset; the pipeline
/// then falls back to "unknown" same as TS).
pub(super) fn read_avih_total_frames(hdrl: &[u8]) -> Option<u64> {
    let mut pos = 0;
    while pos + 8 <= hdrl.len() {
        let fcc = &hdrl[pos..pos + 4];
        let size = u32::from_le_bytes([hdrl[pos + 4], hdrl[pos + 5], hdrl[pos + 6], hdrl[pos + 7]])
            as usize;
        let body_start = pos + 8;
        let body_end = body_start + size;
        if body_end > hdrl.len() {
            return None;
        }
        if fcc == b"avih" {
            if size < 20 {
                return None;
            }
            let body = &hdrl[body_start..body_end];
            let total = u32::from_le_bytes([body[16], body[17], body[18], body[19]]);
            return if total > 0 { Some(total as u64) } else { None };
        }
        pos = body_end + (body_end & 1);
    }
    None
}

/// Read `dwTotalFrames` from the OpenDML extension header chunk
/// (`dmlh`) which lives inside `LIST hdrl > LIST odml > dmlh`. For
/// >1 GiB / very long files the spec recommends using this in
/// preference to `avih.dwTotalFrames` because that field is u32 and
/// can wrap. `dmlh.dwTotalFrames` is the first (and for our purposes
/// only) field of the dmlh body. Returns None if the chunk is absent
/// or the field is zero.
pub(super) fn read_dmlh_total_frames(hdrl: &[u8]) -> Option<u64> {
    let mut pos = 0;
    while pos + 8 <= hdrl.len() {
        let fcc = &hdrl[pos..pos + 4];
        let size = u32::from_le_bytes([hdrl[pos + 4], hdrl[pos + 5], hdrl[pos + 6], hdrl[pos + 7]])
            as usize;
        let body_start = pos + 8;
        let body_end = body_start + size;
        if body_end > hdrl.len() {
            return None;
        }
        if fcc == b"LIST" && size >= 4 && &hdrl[body_start..body_start + 4] == b"odml" {
            // Walk the odml LIST body looking for dmlh.
            let mut p = body_start + 4;
            while p + 8 <= body_end {
                let f = &hdrl[p..p + 4];
                let s = u32::from_le_bytes([hdrl[p + 4], hdrl[p + 5], hdrl[p + 6], hdrl[p + 7]])
                    as usize;
                let bs = p + 8;
                let be = bs + s;
                if be > body_end {
                    return None;
                }
                if f == b"dmlh" && s >= 4 {
                    let total =
                        u32::from_le_bytes([hdrl[bs], hdrl[bs + 1], hdrl[bs + 2], hdrl[bs + 3]]);
                    return if total > 0 { Some(total as u64) } else { None };
                }
                p = be + (be & 1);
            }
            return None;
        }
        pos = body_end + (body_end & 1);
    }
    None
}

// ---------------------------------------------------------------------------
// indx / ix## super-index parsers
// ---------------------------------------------------------------------------

/// Locate the `indx` superindex chunk inside the `LIST strl` for the
/// given video stream, and return a list of `(absolute file offset of
/// each ix## chunk header, ix## chunk body size)` references parsed
/// from its AVI_INDEX_OF_INDEXES entries. Returns None when:
/// - the strl LIST doesn't carry an indx (legacy single-`movi` file),
/// - the indx is the rare AVI_INDEX_OF_CHUNKS form (treated like a
///   fancy idx1; the cursor walk handles those files correctly), or
/// - the indx is malformed.
///
/// `indx` chunk body layout (24-byte header + N×16-byte entries for
/// AVI_INDEX_OF_INDEXES, per OpenDML 1.02 §3.7):
///   wLongsPerEntry     u16   // 4 for index-of-indexes
///   bIndexSubType      u8    // 0 for index-of-indexes
///   bIndexType         u8    // 0x00 = AVI_INDEX_OF_INDEXES
///   nEntriesInUse      u32
///   dwChunkId          u32   // fcc the entries refer to (e.g. "00dc")
///   dwReserved[3]      u32×3 // zero
///   then per entry:
///     qwOffset         u64   // absolute file offset of an ix## chunk
///     dwSize           u32   // ix## chunk size (excluding 8-byte hdr)
///     dwDuration       u32   // sample duration covered by this ix##
pub(super) fn locate_stream_indx(
    hdrl: &[u8],
    target_stream_idx: u32,
) -> Option<Vec<(usize, usize)>> {
    let mut stream_idx: u32 = 0;
    let mut pos = 0;
    while pos + 8 <= hdrl.len() {
        let fcc = &hdrl[pos..pos + 4];
        let size = u32::from_le_bytes([hdrl[pos + 4], hdrl[pos + 5], hdrl[pos + 6], hdrl[pos + 7]])
            as usize;
        let body_start = pos + 8;
        let body_end = body_start + size;
        if body_end > hdrl.len() {
            return None;
        }
        if fcc == b"LIST" && size >= 4 && &hdrl[body_start..body_start + 4] == b"strl" {
            if stream_idx == target_stream_idx {
                return parse_indx_in_strl(&hdrl[body_start + 4..body_end]);
            }
            stream_idx += 1;
        }
        pos = body_end + (body_end & 1);
    }
    None
}

pub(super) fn parse_indx_in_strl(strl: &[u8]) -> Option<Vec<(usize, usize)>> {
    let mut pos = 0;
    while pos + 8 <= strl.len() {
        let fcc = &strl[pos..pos + 4];
        let size = u32::from_le_bytes([strl[pos + 4], strl[pos + 5], strl[pos + 6], strl[pos + 7]])
            as usize;
        let body_start = pos + 8;
        let body_end = body_start + size;
        if body_end > strl.len() {
            return None;
        }
        if fcc == b"indx" {
            return parse_indx_body(&strl[body_start..body_end]);
        }
        pos = body_end + (body_end & 1);
    }
    None
}

pub(super) fn parse_indx_body(body: &[u8]) -> Option<Vec<(usize, usize)>> {
    if body.len() < 24 {
        return None;
    }
    let longs_per_entry = u16::from_le_bytes([body[0], body[1]]);
    let _index_sub_type = body[2];
    let index_type = body[3];
    let n_entries = u32::from_le_bytes([body[4], body[5], body[6], body[7]]) as usize;
    // Index-of-chunks form (rare) is handled by the cursor backend.
    if index_type != 0x00 {
        return None;
    }
    if longs_per_entry != 4 {
        return None;
    } // 4 longs = 16 bytes per entry
    let entries_start = 24;
    let mut refs = Vec::with_capacity(n_entries);
    for i in 0..n_entries {
        let off = entries_start + i * 16;
        if off + 16 > body.len() {
            break;
        }
        let qw_offset = u64::from_le_bytes([
            body[off],
            body[off + 1],
            body[off + 2],
            body[off + 3],
            body[off + 4],
            body[off + 5],
            body[off + 6],
            body[off + 7],
        ]);
        let dw_size =
            u32::from_le_bytes([body[off + 8], body[off + 9], body[off + 10], body[off + 11]]);
        // _dw_duration at off+12..off+16 is informational; not needed
        // to walk the chunks.
        let off_us = qw_offset as usize;
        refs.push((off_us, dw_size as usize));
    }
    Some(refs)
}

/// Parse an `ix##` chunk at the given absolute file offset and append
/// each sample chunk's `(absolute payload offset, payload size)` to
/// `out`. Only entries whose chunk fourcc starts with `<prefix>` are
/// kept (filters out the rare case of a multi-stream ix## merged into
/// one file area).
///
/// `ix##` chunk body layout (24-byte header + N×8-byte entries for
/// AVI_INDEX_OF_CHUNKS, per OpenDML 1.02 §3.7):
///   wLongsPerEntry     u16   // 2 for index-of-chunks
///   bIndexSubType      u8    // 0 (frame index)
///   bIndexType         u8    // 0x01 = AVI_INDEX_OF_CHUNKS
///   nEntriesInUse      u32
///   dwChunkId          u32   // fcc the entries reference
///   qwBaseOffset       u64   // entries' dwOffset is relative to this
///   dwReserved         u32   // zero
///   then per entry:
///     dwOffset         u32   // chunk DATA offset from qwBaseOffset
///     dwSize           u32   // chunk DATA size; high bit = NOT keyframe
///
/// Note `dwOffset` points at the chunk DATA, NOT the chunk header
/// (FOURCC + size) per the OpenDML 1.02 conformance spec — the
/// reasoning being that the indx/ix## pre-locates the data so a player
/// can jump directly without re-reading the chunk header. We honor
/// that: the absolute offset we record is `qwBaseOffset + dwOffset`,
/// and `size` is the data-only payload size.
pub(super) fn parse_ix_chunk(
    data: &[u8],
    ix_header_off: usize,
    _ix_size: usize,
    prefix: &[u8; 2],
    out: &mut Vec<(usize, usize)>,
) {
    // The ix_header_off given by the indx superindex points at the
    // ix## chunk's 8-byte RIFF header (FOURCC + LE size). Skip the
    // header then read the body.
    if ix_header_off + 8 > data.len() {
        return;
    }
    let body_start = ix_header_off + 8;
    let body_size = u32::from_le_bytes([
        data[ix_header_off + 4],
        data[ix_header_off + 5],
        data[ix_header_off + 6],
        data[ix_header_off + 7],
    ]) as usize;
    let body_end = body_start.saturating_add(body_size).min(data.len());
    if body_end < body_start + 24 {
        return;
    }
    let body = &data[body_start..body_end];
    let longs_per_entry = u16::from_le_bytes([body[0], body[1]]);
    let _index_sub_type = body[2];
    let index_type = body[3];
    let n_entries = u32::from_le_bytes([body[4], body[5], body[6], body[7]]) as usize;
    let chunk_id: [u8; 4] = body[8..12].try_into().unwrap();
    let qw_base_offset = u64::from_le_bytes([
        body[12], body[13], body[14], body[15], body[16], body[17], body[18], body[19],
    ]) as usize;
    // body[20..24] is dwReserved (zero).
    if index_type != 0x01 {
        return;
    } // not an index-of-chunks
    if longs_per_entry != 2 {
        return;
    }
    // Only this stream's chunks (`<prefix>dc` / `<prefix>db`).
    if chunk_id[0] != prefix[0] || chunk_id[1] != prefix[1] {
        return;
    }
    let kind = chunk_id[3];
    if kind != b'c' && kind != b'b' {
        return;
    }
    let entries_start = 24;
    for i in 0..n_entries {
        let off = entries_start + i * 8;
        if off + 8 > body.len() {
            break;
        }
        let dw_offset =
            u32::from_le_bytes([body[off], body[off + 1], body[off + 2], body[off + 3]]) as usize;
        let dw_size_raw =
            u32::from_le_bytes([body[off + 4], body[off + 5], body[off + 6], body[off + 7]]);
        // High bit = "this is NOT a keyframe" flag; mask off to get the
        // real payload size. (Not used here — we don't track keyframes
        // at the demux layer for AVI; the codec parses I/P/B itself.)
        let dw_size = (dw_size_raw & 0x7FFFFFFF) as usize;
        let abs_off = qw_base_offset.saturating_add(dw_offset);
        out.push((abs_off, dw_size));
    }
}
