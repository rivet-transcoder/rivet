//! Opus / dOps helpers for MP4 audio extraction.
//!
//! Box-walking primitives live in `demux/mod.rs` and are reached via
//! `super::super::` (super = audio, super::super = demux).

/// Walk every `trak` looking for one whose `stsd` contains an `Opus`
/// sample entry (RFC 7845 §4.4). Returns the body bytes of the contained
/// `dOps` box (without the 8-byte box header) or None.
///
/// `find_box_body` only follows the FIRST trak it encounters (the video
/// trak), so we have to iterate traks ourselves — same pattern as
/// `extract_aac_asc`.
///
/// 4-cc match is `Opus` exactly (capital O) per spec. We do not match the
/// lowercase `opus` variant — strict players reject that and we shouldn't
/// silently accept input that some downstream stage will choke on.
pub(super) fn extract_mp4_opus_dops_body(data: &[u8]) -> Option<Vec<u8>> {
    let moov = super::super::find_direct_child(data, b"moov")?;
    let mut pos = 0;
    while pos + 8 <= moov.len() {
        let size =
            u32::from_be_bytes([moov[pos], moov[pos + 1], moov[pos + 2], moov[pos + 3]]) as usize;
        let btype = &moov[pos + 4..pos + 8];
        if size < 8 || pos.checked_add(size).is_none_or(|end| end > moov.len()) {
            break;
        }
        if btype == b"trak" {
            let trak_body = &moov[pos + 8..pos + size];
            if let Some(dops) = extract_dops_from_trak(trak_body) {
                return Some(dops);
            }
        }
        pos += size;
    }
    None
}

fn extract_dops_from_trak(trak: &[u8]) -> Option<Vec<u8>> {
    let stsd = super::super::find_box_body(trak, &[b"mdia", b"minf", b"stbl", b"stsd"])?;
    if stsd.len() < 16 {
        return None;
    }
    let mut pos = 8; // skip version/flags/entry_count
    while pos + 8 <= stsd.len() {
        let entry_size =
            u32::from_be_bytes([stsd[pos], stsd[pos + 1], stsd[pos + 2], stsd[pos + 3]]) as usize;
        let entry_type: [u8; 4] = stsd[pos + 4..pos + 8].try_into().ok()?;
        if entry_size < 8 || pos.saturating_add(entry_size) > stsd.len() {
            break;
        }
        if &entry_type == b"Opus" {
            let end = pos + entry_size;
            // AudioSampleEntry layout per ISO/IEC 14496-12 §8.5.2.2: after
            // the 8-byte box header there's a 28-byte fixed preamble
            // (reserved/channelcount/samplesize/etc.) — same as `mp4a` —
            // followed by nested codec-specific boxes. dOps lives there.
            let child_start = pos + 8 + 28;
            if child_start >= end {
                return None;
            }
            return super::super::find_direct_child(&stsd[child_start..end], b"dOps")
                .map(|b| b.to_vec());
        }
        pos += entry_size;
    }
    None
}

/// Convert a `dOps` body (BE numeric fields per RFC 7845 §4.5) back into
/// the OpusHead-form body (LE numeric fields per RFC 7845 §5.1) that the
/// mux side carries in `AudioInfo.codec_private`. This keeps the in-pipeline
/// representation a single canonical form regardless of source container.
///
/// The dOps `Version` field (always 0 on the wire per §4.5) is rewritten
/// to OpusHead `Version` = 1 (RFC 7845 §5.1: "version number, MUST be 1").
pub(super) fn dops_to_opus_head(dops: &[u8]) -> Option<Vec<u8>> {
    if dops.len() < 11 {
        return None;
    }
    // dops[0] = Version (0); dops[1] = OutputChannelCount;
    // dops[2..4] = PreSkip BE; dops[4..8] = InputSampleRate BE;
    // dops[8..10] = OutputGain BE; dops[10] = ChannelMappingFamily.
    let output_channels = dops[1];
    let pre_skip = u16::from_be_bytes([dops[2], dops[3]]);
    let input_sample_rate = u32::from_be_bytes([dops[4], dops[5], dops[6], dops[7]]);
    let output_gain = i16::from_be_bytes([dops[8], dops[9]]);
    let channel_mapping_family = dops[10];

    // Family != 0 → carry the channel mapping table verbatim too.
    let extra_tail = if channel_mapping_family != 0 {
        if dops.len() < 13 {
            return None;
        }
        let tail_len = 2 + dops[12] as usize;
        if dops.len() < 11 + tail_len {
            return None;
        }
        dops[11..11 + tail_len].to_vec()
    } else {
        Vec::new()
    };

    let mut head = Vec::with_capacity(11 + extra_tail.len());
    head.push(1u8); // OpusHead Version = 1
    head.push(output_channels);
    head.extend_from_slice(&pre_skip.to_le_bytes());
    head.extend_from_slice(&input_sample_rate.to_le_bytes());
    head.extend_from_slice(&(output_gain as u16).to_le_bytes());
    head.push(channel_mapping_family);
    head.extend_from_slice(&extra_tail);
    Some(head)
}
