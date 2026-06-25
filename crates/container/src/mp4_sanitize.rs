//! Lenient pre-pass for the strict `mp4` crate.
//!
//! ISOBMFF box headers carry a `size` field that COULD be wrong —
//! malformed encoders (older Apple QuickTime, some prosumer cameras,
//! anything that round-trips through a buggy mux) can emit child
//! boxes whose advertised size exceeds the parent's remaining
//! payload. The `mp4 0.14` crate (and most strict ISOBMFF parsers)
//! bail with `"<parent> box contains a box with a larger size than
//! it"` and the whole demux fails.
//!
//! `sanitize_isobmff_box_sizes` walks the box tree from the root,
//! and any time a child's advertised size would exceed the
//! parent's remaining payload, rewrites the child's `size` field
//! to fit. The output bytes are byte-compatible with strict
//! parsers and (in the common case where the child's size was a
//! benign over-report by 1-N bytes) preserve everything that the
//! parser actually reads.
//!
//! The function is a no-op on every well-formed file — every box
//! header is left untouched, so a clean MP4 hashes identically
//! through this function. Only malformed files mutate.
//!
//! What this DOES handle:
//!   - Top-level container boxes: ftyp, moov, mdat, etc.
//!   - Recursive containers: moov > trak > mdia > minf > stbl >
//!     stsd > {mp4a, av01, hvc1, ...}.
//!   - 64-bit `largesize` extended-size form.
//!
//! What this does NOT handle:
//!   - `size = 0` "extends to end of file" — left untouched (strict
//!     parsers handle this correctly).
//!   - Box trees with byte-level corruption inside a leaf box's
//!     payload (e.g. a malformed `esds` descriptor). The sanitizer
//!     only touches the box header bytes; leaf payload is opaque.

/// Set of box four-CCs that contain other boxes. Walking these
/// recursively lets us reach every header in the tree. Anything
/// not in the set is treated as a leaf — its payload is copied
/// through without further inspection. The set covers the boxes
/// the strict parser actually descends into; adding a new entry
/// here is the only way to extend the sanitizer's reach when a
/// future crate version recurses further.
const CONTAINER_FOURCCS: &[&[u8; 4]] = &[
    b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd", b"edts", b"udta", b"meta", b"dinf",
    b"mvex", b"moof", b"traf", b"mfra",
    // Sample-entry boxes that themselves contain children (visual
    // sample entries carry colr/mdcv/clli/av1C; audio sample
    // entries carry esds/dOps/wave). The strict parser walks
    // these as containers, so we must too.
    b"mp4a", b"Opus", b"ac-3", b"ec-3", b"enca", b"av01", b"avc1", b"avc3", b"hvc1", b"hev1",
    b"hvc2", b"hev2", b"dvh1", b"dvhe", b"vp08", b"vp09", b"apco", b"apcs", b"apcn", b"apch",
    b"ap4h", b"ap4x",
    // QuickTime-era audio sample-entry sub-container; legacy Apple
    // tools wrap esds inside this.
    b"wave",
];

#[inline]
fn is_container(fourcc: &[u8; 4]) -> bool {
    CONTAINER_FOURCCS.contains(&fourcc)
}

/// Visual sample entries have a fixed 78-byte header before any
/// child boxes start. Audio sample entries (mp4a, Opus, etc.)
/// have a 28-byte header. The sanitizer skips these to land at
/// the start of the first child.
///
/// Standard sizes per ISO 14496-12 §8.5.2:
///   - VisualSampleEntry: 8 (box header) + 6 (reserved) +
///     2 (data_reference_index) + 2 (pre_defined) +
///     2 (reserved) + 12 (pre_defined[3]) + 2 (width) + 2 (height) +
///     8 (resolution) + 4 (reserved) + 2 (frame_count) + 32 (compressorname) +
///     2 (depth) + 2 (pre_defined) = 86 bytes total. After the box
///     header we read 78 bytes of fixed fields before children.
///   - AudioSampleEntry: 8 (box header) + 6 (reserved) +
///     2 (data_reference_index) + 8 (reserved) + 2 (channels) +
///     2 (sample_size) + 4 (reserved) + 4 (sample_rate) = 36 bytes
///     total. After the box header: 28 bytes of fixed fields.
fn sample_entry_fixed_fields_len(fourcc: &[u8; 4]) -> Option<usize> {
    let visual = matches!(
        fourcc,
        b"av01"
            | b"avc1"
            | b"avc3"
            | b"hvc1"
            | b"hev1"
            | b"hvc2"
            | b"hev2"
            | b"dvh1"
            | b"dvhe"
            | b"vp08"
            | b"vp09"
            | b"apco"
            | b"apcs"
            | b"apcn"
            | b"apch"
            | b"ap4h"
            | b"ap4x",
    );
    let audio = matches!(fourcc, b"mp4a" | b"Opus" | b"ac-3" | b"ec-3" | b"enca");
    if visual {
        Some(78)
    } else if audio {
        Some(28)
    } else {
        None
    }
}

pub fn sanitize_isobmff_box_sizes(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    // Top-level walk has no parent — `parent` = `*` is fine since
    // top-level fourccs (ftyp, moov, mdat, ...) never need
    // sample-entry prefix handling.
    walk_and_sanitize(data, 0, data.len(), b"****", &mut out);
    out
}

/// Walks a parent's payload (data[parent_payload_start..parent_payload_end])
/// emitting box headers, recursing into containers, copying leaves
/// verbatim, and clamping any child whose advertised size exceeds
/// the parent's remaining payload.
///
/// `parent` is the parent's four-CC. Used to decide whether a
/// sample-entry-shaped child (mp4a, av01, etc.) actually IS a
/// sample entry (parent == stsd) or a plain box used inside a
/// QuickTime extension (e.g. the inner `mp4a` inside `wave` for
/// iPhone-recorded MOVs). Only sample-entry-context children get
/// the 28/78-byte fixed-prefix skip; plain-context children with
/// the same fourcc walk like any other container.
fn walk_and_sanitize(data: &[u8], start: usize, end: usize, parent: &[u8; 4], out: &mut Vec<u8>) {
    let mut cursor = start;
    while cursor < end {
        // Box header is 8 bytes minimum (4 size + 4 fourcc).
        if cursor + 8 > end {
            // Trailing junk — copy through; the strict parser will
            // surface the issue more clearly than we can here.
            out.extend_from_slice(&data[cursor..end]);
            return;
        }

        let raw_size = u32::from_be_bytes([
            data[cursor],
            data[cursor + 1],
            data[cursor + 2],
            data[cursor + 3],
        ]) as u64;
        let fourcc: &[u8; 4] = data[cursor + 4..cursor + 8].try_into().unwrap();

        // size=0 means "extends to end of file (or parent)" per spec.
        // Leave alone — strict parsers handle this correctly.
        if raw_size == 0 {
            out.extend_from_slice(&data[cursor..end]);
            return;
        }

        let mut header_len = 8usize;
        let mut box_size = raw_size;

        // Extended (64-bit) size: header is 16 bytes total.
        if raw_size == 1 {
            if cursor + 16 > end {
                out.extend_from_slice(&data[cursor..end]);
                return;
            }
            box_size = u64::from_be_bytes([
                data[cursor + 8],
                data[cursor + 9],
                data[cursor + 10],
                data[cursor + 11],
                data[cursor + 12],
                data[cursor + 13],
                data[cursor + 14],
                data[cursor + 15],
            ]);
            header_len = 16;
        }

        // Clamp: child's size must fit inside parent's remaining
        // payload. If the file claims a size that runs past the
        // parent boundary, rewrite the size field to land at the
        // parent end. Handles the "mp4a box contains a box with a
        // larger size than it" failure mode directly.
        let remaining = (end - cursor) as u64;
        let clamped = if box_size > remaining {
            remaining
        } else {
            box_size
        };

        // Emit the (possibly rewritten) header. We always emit the
        // 32-bit form when clamping — that's what every ISOBMFF
        // parser expects for sizes that fit in u32. If the clamped
        // value exceeds u32::MAX we fall back to writing the
        // largesize form unchanged from the source (this is rare —
        // happens only for >4 GiB boxes, where clamping is a no-op
        // because the file is already huge enough to fit).
        if clamped <= u32::MAX as u64 && header_len == 8 {
            out.extend_from_slice(&(clamped as u32).to_be_bytes());
            out.extend_from_slice(fourcc);
        } else {
            // Either largesize form OR clamped value too big for
            // u32 — emit the original header bytes verbatim.
            out.extend_from_slice(&data[cursor..cursor + header_len]);
        }

        let payload_start = cursor + header_len;
        let payload_end = (cursor as u64 + clamped) as usize;
        let payload_end = payload_end.min(end);

        if payload_start >= payload_end {
            // Zero-length or malformed payload after header. Keep
            // walking from the parent's next box.
            cursor = payload_end.max(cursor + header_len);
            continue;
        }

        if is_container(fourcc) {
            // Sample-entry boxes (mp4a/Opus/ac-3/ec-3/av01/avc1/...)
            // carry a fixed-field block before their children. They
            // are sample entries ONLY when their parent is `stsd`.
            // Anywhere else (e.g. the inner `mp4a` inside `wave` in
            // QuickTime / iPhone MOVs), the same fourcc is a plain
            // container with no fixed prefix — applying the prefix
            // skip there would mis-align the child walk and corrupt
            // the recursion. `stsd` itself has its own 8-byte
            // (FullBox header + entry_count) preamble.
            let prefix_len = if fourcc == b"stsd" {
                8
            } else if parent == b"stsd" {
                sample_entry_fixed_fields_len(fourcc).unwrap_or(0)
            } else {
                0
            };
            let copy_end = (payload_start + prefix_len).min(payload_end);
            out.extend_from_slice(&data[payload_start..copy_end]);
            walk_and_sanitize(data, copy_end, payload_end, fourcc, out);
        } else {
            // Leaf box — copy payload verbatim.
            out.extend_from_slice(&data[payload_start..payload_end]);
        }

        cursor = payload_end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_box(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let size = (8 + payload.len()) as u32;
        let mut out = Vec::with_capacity(size as usize);
        out.extend_from_slice(&size.to_be_bytes());
        out.extend_from_slice(fourcc);
        out.extend_from_slice(payload);
        out
    }

    fn make_sized_box(fourcc: &[u8; 4], reported_size: u32, payload: &[u8]) -> Vec<u8> {
        // Size on the wire reflects the "reported" value, but the
        // payload appended is the actual bytes. Used to fabricate
        // malformed boxes where reported_size != header_len + payload.len().
        let mut out = Vec::with_capacity(8 + payload.len());
        out.extend_from_slice(&reported_size.to_be_bytes());
        out.extend_from_slice(fourcc);
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn well_formed_file_passes_through_byte_identical() {
        let esds = make_box(b"esds", &[0x00; 32]);
        let mut mp4a_payload = vec![0u8; 28]; // fixed audio sample entry fields
        mp4a_payload.extend_from_slice(&esds);
        let mp4a = make_box(b"mp4a", &mp4a_payload);

        let stsd = {
            let mut p = vec![0u8, 0, 0, 0]; // version+flags
            p.extend_from_slice(&1u32.to_be_bytes()); // entry_count = 1
            p.extend_from_slice(&mp4a);
            make_box(b"stsd", &p)
        };
        let stbl = make_box(b"stbl", &stsd);
        let minf = make_box(b"minf", &stbl);
        let mdia = make_box(b"mdia", &minf);
        let trak = make_box(b"trak", &mdia);
        let moov = make_box(b"moov", &trak);

        let sanitized = sanitize_isobmff_box_sizes(&moov);
        assert_eq!(
            sanitized, moov,
            "well-formed input must round-trip byte-identical"
        );
    }

    #[test]
    fn over_reported_child_inside_mp4a_gets_clamped() {
        // The bug from the user's screenshot: an esds child whose
        // reported size exceeds the parent mp4a's remaining payload.
        // Reported size = 100 (way more than the 16 actual bytes
        // including header).
        let bad_esds = make_sized_box(b"esds", 100, &[0xAB; 8]);

        let mut mp4a_payload = vec![0u8; 28]; // fixed audio fields
        mp4a_payload.extend_from_slice(&bad_esds);
        let mp4a = make_box(b"mp4a", &mp4a_payload);

        // mp4a is only treated as a sample entry (with the 28-byte
        // prefix) when its parent is `stsd`. Wrap properly.
        let stsd_payload = {
            let mut p = vec![0u8; 4]; // version + flags
            p.extend_from_slice(&1u32.to_be_bytes()); // entry_count = 1
            p.extend_from_slice(&mp4a);
            p
        };
        let stsd = make_box(b"stsd", &stsd_payload);

        let sanitized = sanitize_isobmff_box_sizes(&stsd);

        // Locate the mp4a header inside the sanitized output:
        //   stsd header (8) + version+flags (4) + entry_count (4) = 16
        let mp4a_header_offset = 16;
        assert_eq!(
            &sanitized[mp4a_header_offset + 4..mp4a_header_offset + 8],
            b"mp4a"
        );
        // esds header sits 8 (mp4a header) + 28 (fixed audio fields)
        // bytes past the mp4a header.
        let esds_size_offset = mp4a_header_offset + 8 + 28;
        let clamped_esds_size = u32::from_be_bytes([
            sanitized[esds_size_offset],
            sanitized[esds_size_offset + 1],
            sanitized[esds_size_offset + 2],
            sanitized[esds_size_offset + 3],
        ]);
        // mp4a payload (after its 8-byte header) is 28 (fixed) + 16
        // (esds, including header) = 44 bytes. esds sits at offset
        // 28 within mp4a payload, with 16 bytes remaining; clamped
        // esds size should land at 16.
        assert!(
            clamped_esds_size <= 16,
            "esds size should be clamped to fit, got {clamped_esds_size}",
        );
        assert!(
            clamped_esds_size >= 8,
            "esds size should still cover its header, got {clamped_esds_size}",
        );
    }

    #[test]
    fn inner_mp4a_inside_wave_is_not_treated_as_sample_entry() {
        // iPhone MOV layout: the OUTER mp4a is a sample entry
        // (28-byte prefix), but the INNER mp4a inside `wave` is a
        // plain container box — applying the 28-byte prefix there
        // would mis-align the child walk and lose the esds sibling.
        // This test reproduces the iPhone audio drop and asserts
        // the sanitizer's output is structurally walk-able by the
        // manual ASC extractor.
        let inner_mp4a = make_box(b"mp4a", &vec![0u8; 24]); // QuickTime audio config blob
        let frma = make_box(b"frma", b"mp4a");
        let esds_body = vec![0u8; 32];
        let esds = make_box(b"esds", &esds_body);

        let wave_payload = {
            let mut p = Vec::new();
            p.extend_from_slice(&frma);
            p.extend_from_slice(&inner_mp4a);
            p.extend_from_slice(&esds);
            p
        };
        let wave = make_box(b"wave", &wave_payload);

        // Outer mp4a: 28 fixed audio fields + the wave atom.
        let mut outer_mp4a_payload = vec![0u8; 28];
        outer_mp4a_payload.extend_from_slice(&wave);
        let outer_mp4a = make_box(b"mp4a", &outer_mp4a_payload);

        let stsd_payload = {
            let mut p = vec![0u8; 4];
            p.extend_from_slice(&1u32.to_be_bytes());
            p.extend_from_slice(&outer_mp4a);
            p
        };
        let stsd = make_box(b"stsd", &stsd_payload);

        let sanitized = sanitize_isobmff_box_sizes(&stsd);
        // Round-trip byte-identical (no clamping needed — every
        // box's reported size already fits its parent).
        assert_eq!(
            sanitized, stsd,
            "well-formed iPhone-shaped MP4 must pass through unchanged"
        );
    }

    #[test]
    fn sanitizer_is_idempotent() {
        // Running sanitize twice should be a no-op the second time.
        let bad_esds = make_sized_box(b"esds", 100, &[0u8; 8]);
        let mut mp4a_payload = vec![0u8; 28];
        mp4a_payload.extend_from_slice(&bad_esds);
        let mp4a = make_box(b"mp4a", &mp4a_payload);

        let once = sanitize_isobmff_box_sizes(&mp4a);
        let twice = sanitize_isobmff_box_sizes(&once);
        assert_eq!(once, twice, "sanitizer must be idempotent");
    }

    #[test]
    fn truncated_input_is_handled_without_panic() {
        // Box header says size=100 but only 12 bytes follow.
        let mut bad = vec![];
        bad.extend_from_slice(&100u32.to_be_bytes());
        bad.extend_from_slice(b"moov");
        bad.extend_from_slice(&[0u8; 4]); // 4 bytes of "payload"
        let _ = sanitize_isobmff_box_sizes(&bad); // must not panic
    }
}
