/// AAC / ADTS / AudioSpecificConfig helpers for MP4 audio extraction.
///
/// All functions here are called from `audio/mod.rs`'s `extract_mp4_audio`.
/// Box-walking primitives live in `demux/mod.rs` and are reached via
/// `super::super::` (super = audio, super::super = demux).

// ─── Shared box-tree helpers ─────────────────────────────────────────────────

/// Audio sample-entry fourccs we recognise as carrying an AAC ASC.
///
/// `mp4a` is the standard ISOBMFF AudioSampleEntry. `enca` is the
/// EncryptedSampleEntry wrapper (ISO 23001-7 §6.2) — it carries the
/// same 28-byte AudioSampleEntry prefix with an inner `frma 'mp4a'`
/// declaring the original format, and the esds (with the clear ASC
/// bytes) sits next to the `sinf` ProtectionSchemeInfoBox. For
/// streams using `cenc` "clear" mode, the ASC itself is unencrypted,
/// so passthrough works the same as for `mp4a`.
const AAC_AUDIO_SAMPLE_ENTRIES: &[&[u8; 4]] = &[b"mp4a", b"enca"];

/// Format the first `n` bytes of `bytes` as a hex string for diagnostic
/// log lines. Used by `extract_mp4_audio` so the log records the actual
/// ASC prefix when something downstream fails to parse it — that lets us
/// reproduce iPhone-shaped issues from CloudWatch alone, without needing
/// the user's source file in hand.
pub(super) fn hex_prefix(bytes: &[u8], n: usize) -> String {
    let mut out = String::with_capacity(n * 2);
    for b in bytes.iter().take(n) {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Walk moov/trak*/mdia/minf/stbl/stsd to recover the AAC AudioSpecificConfig.
///
/// Returns the DecoderSpecificInfo payload verbatim. The walk is robust to
/// the kinds of variation iPhone-recorded MOVs throw at us:
///
///   - **Multi-trak files**: iterates every `trak`. Most files have video +
///     audio + (optional) timed metadata. We use the presence of `smhd`
///     (Sound Media Header, ISO 14496-12 §8.4.5.3) to *positively* identify
///     audio traks rather than relying on stsd[0]'s fourcc — that's how we
///     reach the audio data even if the trak is in an unusual order.
///   - **Multi-entry stsd**: iterates every `SampleEntry` inside `stsd`,
///     not just entry[0]. Apple tooling occasionally emits multiple sample
///     entries (e.g. `mp4a` + an alternate config) and we must find the
///     first one that yields a usable ASC.
///   - **enca (Encrypted-But-Clear)**: same 28-byte AudioSampleEntry
///     prefix as `mp4a`, with an inner `frma 'mp4a'` declaring the
///     original format. We treat `enca` as `mp4a` for ASC extraction.
///   - **wave wrapping**: Apple QuickTime nests
///     `mp4a → wave → frma + mp4a + esds`. `find_esds_recursive` descends
///     into `wave` so the esds is found regardless of nesting depth.
///   - **Brute-force fallback**: after the structured walk, if the trak
///     was identified as audio (smhd present) but no ASC came back, we
///     scan the trak buffer linearly for any `esds` box and try to parse
///     an ASC out of it. This is the safety net for unforeseen wrappers
///     (and the "log signpost" — anything that lands here gets a warn so
///     we can codify the new shape into structured handling later).
///
/// Returns `None` only when none of the audio traks yielded a non-empty
/// ASC. Every fall-through here has a `tracing::warn!` so CloudWatch
/// surfaces the exact reason rather than producing audio-less output
/// silently.
pub(super) fn extract_aac_asc(data: &[u8]) -> Option<Vec<u8>> {
    let moov = super::super::find_direct_child(data, b"moov")?;
    let mut pos = 0;
    let mut saw_audio_trak = false;
    while pos + 8 <= moov.len() {
        let size =
            u32::from_be_bytes([moov[pos], moov[pos + 1], moov[pos + 2], moov[pos + 3]]) as usize;
        let btype = &moov[pos + 4..pos + 8];
        if size < 8 || pos.checked_add(size).is_none_or(|end| end > moov.len()) {
            break;
        }
        if btype == b"trak" {
            let trak_body = &moov[pos + 8..pos + size];
            if trak_is_audio(trak_body) {
                saw_audio_trak = true;
                if let Some(asc) = extract_asc_from_trak(trak_body) {
                    return Some(asc);
                }
                // Audio trak identified by smhd but the structured
                // walk came up empty — try a brute-force esds scan
                // before declaring failure.
                if let Some(asc) = brute_force_find_asc_in_trak(trak_body) {
                    tracing::warn!(
                        asc_len = asc.len(),
                        "audio passthrough recovered ASC via brute-force esds scan; \
                         the trak's stsd shape is not in our structured handler. \
                         Capture this file and add coverage so the structured walk \
                         finds it next time."
                    );
                    return Some(asc);
                }
            }
        }
        pos += size;
    }
    if saw_audio_trak {
        tracing::warn!(
            "audio passthrough skipped: identified an audio trak via smhd, but no \
             stsd entry yielded an AudioSpecificConfig. Possible causes: enca with \
             unsupported scheme, sample entry fourcc we don't recognise, esds box \
             missing or corrupt, mp4 sanitizer mis-aligned a wave-wrapped esds."
        );
    } else {
        tracing::warn!(
            "audio passthrough skipped: no trak had a Sound Media Header (smhd). \
             Source may be video-only, or its track headers do not conform to ISOBMFF \
             §8.4.5.3 (smhd is required for audio traks)."
        );
    }
    None
}

/// Quick "is this trak an audio trak?" check. ISO 14496-12 §8.4.5.3
/// requires `smhd` (Sound Media Header) inside `mdia/minf` for every
/// audio trak. Looking for it is a strictly stronger signal than
/// inspecting the first `stsd` entry's fourcc — it's positive evidence
/// of trak intent rather than fourcc-position guessing.
fn trak_is_audio(trak: &[u8]) -> bool {
    super::super::find_box_body(trak, &[b"mdia", b"minf", b"smhd"]).is_some()
}

fn extract_asc_from_trak(trak: &[u8]) -> Option<Vec<u8>> {
    let stsd = super::super::find_box_body(trak, &[b"mdia", b"minf", b"stbl", b"stsd"])?;
    if stsd.len() < 8 {
        tracing::warn!(
            stsd_len = stsd.len(),
            "audio passthrough: stsd shorter than its 8-byte FullBox preamble"
        );
        return None;
    }
    // Skip version/flags (4) + entry_count (4). Sample entries follow.
    let entries = &stsd[8..];
    let mut cursor = 0;
    while cursor + 8 <= entries.len() {
        let entry_size = u32::from_be_bytes([
            entries[cursor],
            entries[cursor + 1],
            entries[cursor + 2],
            entries[cursor + 3],
        ]) as usize;
        let entry_type: &[u8; 4] = entries[cursor + 4..cursor + 8].try_into().unwrap();
        if entry_size < 8 || cursor + entry_size > entries.len() {
            break;
        }

        if AAC_AUDIO_SAMPLE_ENTRIES.contains(&entry_type) {
            // AudioSampleEntry layout per ISOBMFF §8.5.2: 8-byte box
            // header + 28-byte fixed preamble (reserved /
            // channelcount / samplesize / sample_rate Q16) + nested
            // boxes (esds, optional wave wrapper, optional chan).
            if entry_size >= 36 {
                let body = &entries[cursor + 8 + 28..cursor + entry_size];
                if let Some(asc) = find_esds_recursive(body) {
                    return Some(asc);
                }
            }
        }
        cursor += entry_size;
    }
    None
}

/// Last-resort: linearly scan the trak buffer for any `esds` box and
/// try to parse an ASC out of it. Used only when the structured walk
/// (smhd → stsd → mp4a/enca → esds, optionally through `wave`) failed
/// despite the trak being an audio trak. Logs a warn at the call site
/// when this path returns a result so we can codify the source's
/// actual shape into the structured handler later.
fn brute_force_find_asc_in_trak(trak: &[u8]) -> Option<Vec<u8>> {
    let mut pos = 0;
    while pos + 8 <= trak.len() {
        if &trak[pos + 4..pos + 8] == b"esds" {
            let size = u32::from_be_bytes([trak[pos], trak[pos + 1], trak[pos + 2], trak[pos + 3]])
                as usize;
            if size >= 12 && pos + size <= trak.len() {
                // esds body begins after 8-byte box header + 4-byte FullBox preamble.
                let esds_body = &trak[pos + 12..pos + size];
                if let Some(asc) = extract_asc_from_esds(esds_body) {
                    if !asc.is_empty() {
                        return Some(asc);
                    }
                }
            }
        }
        pos += 1;
    }
    None
}

/// Descend into the nested-box children of an mp4a sample entry to
/// find `esds`. Apple QuickTime / iPhone MOV files frequently wrap
/// the esds inside a `wave` container box (legacy from .mov format),
/// so a flat scan of immediate children misses it. Recursing into
/// `wave` (and only `wave` — other sub-boxes are not specified to
/// contain esds) lets us pick it up in either layout.
///
/// Returns the parsed AudioSpecificConfig bytes from the first esds
/// found.
fn find_esds_recursive(body: &[u8]) -> Option<Vec<u8>> {
    extract_asc_from_esds(find_esds_body_recursive(body)?)
}

/// The ES descriptor tree of the first `esds` among `body`'s boxes,
/// descending into `wave` (see [`find_esds_recursive`]).
fn find_esds_body_recursive(body: &[u8]) -> Option<&[u8]> {
    let mut pos = 0;
    while pos + 8 <= body.len() {
        let sub_size =
            u32::from_be_bytes([body[pos], body[pos + 1], body[pos + 2], body[pos + 3]]) as usize;
        let sub_type = &body[pos + 4..pos + 8];
        if sub_size < 8 || pos + sub_size > body.len() {
            break;
        }
        if sub_type == b"esds" {
            // esds body: 1 byte version + 3 flags + ES descriptor tree.
            return Some(&body[pos + 8 + 4..pos + sub_size]);
        }
        if sub_type == b"wave" {
            // QuickTime audio extension. Recurse — esds usually lives
            // inside.
            if let Some(esds) = find_esds_body_recursive(&body[pos + 8..pos + sub_size]) {
                return Some(esds);
            }
        }
        pos += sub_size;
    }
    None
}

/// The `objectTypeIndication` of the first `mp4a` / `enca` sample entry
/// of an audio trak, if any. `mp4a` is not only AAC: ffmpeg's MP4 muxer
/// writes DTS under it with OTI 0xA9 (core), 0xAA / 0xAB (DTS-HD HRA / MA)
/// or 0xAC (DTS Express), and the caller uses this to route those to the
/// DTS path instead of treating the ES descriptor as an AAC config.
pub(super) fn mp4_esds_object_type(data: &[u8]) -> Option<u8> {
    let moov = super::super::find_direct_child(data, b"moov")?;
    let mut pos = 0;
    while pos + 8 <= moov.len() {
        let size =
            u32::from_be_bytes([moov[pos], moov[pos + 1], moov[pos + 2], moov[pos + 3]]) as usize;
        let btype = &moov[pos + 4..pos + 8];
        if size < 8 || pos + size > moov.len() {
            break;
        }
        if btype == b"trak" {
            let trak = &moov[pos + 8..pos + size];
            if trak_is_audio(trak)
                && let Some(stsd) =
                    super::super::find_box_body(trak, &[b"mdia", b"minf", b"stbl", b"stsd"])
                && stsd.len() >= 8
            {
                let entries = &stsd[8..];
                let mut cursor = 0;
                while cursor + 8 <= entries.len() {
                    let entry_size = u32::from_be_bytes([
                        entries[cursor],
                        entries[cursor + 1],
                        entries[cursor + 2],
                        entries[cursor + 3],
                    ]) as usize;
                    if entry_size < 8 || cursor + entry_size > entries.len() {
                        break;
                    }
                    let entry_type: &[u8; 4] = entries[cursor + 4..cursor + 8].try_into().unwrap();
                    if AAC_AUDIO_SAMPLE_ENTRIES.contains(&entry_type) && entry_size >= 36 {
                        let body = &entries[cursor + 8 + 28..cursor + entry_size];
                        if let Some(oti) = find_esds_body_recursive(body).and_then(esds_object_type) {
                            return Some(oti);
                        }
                    }
                    cursor += entry_size;
                }
            }
        }
        pos += size;
    }
    None
}

/// Walk `moov > trak[]` and return true if any audio trak (identified
/// by `smhd`, ISO 14496-12 §8.4.5.3) carries one of our recognised AAC
/// sample-entry fourccs (`mp4a` or `enca`). Walks every stsd entry, not
/// just entry[0], so multi-entry stsd shapes Apple tooling occasionally
/// produces still classify correctly.
///
/// Used as the manual AAC detector that bypasses `mp4 0.14`'s
/// `track.media_type()` — iPhone MOVs trip the crate's classifier when
/// audio carries QuickTime extensions (esds wrapped in `wave`), and the
/// silent-Err path used to drop audio on every upload.
pub(super) fn mp4_has_aac_sample_entry(data: &[u8]) -> bool {
    let Some(moov) = super::super::find_direct_child(data, b"moov") else {
        return false;
    };
    let mut pos = 0;
    while pos + 8 <= moov.len() {
        let size =
            u32::from_be_bytes([moov[pos], moov[pos + 1], moov[pos + 2], moov[pos + 3]]) as usize;
        let btype = &moov[pos + 4..pos + 8];
        if size < 8 || pos + size > moov.len() {
            break;
        }
        if btype == b"trak" {
            let trak_body = &moov[pos + 8..pos + size];
            if !trak_is_audio(trak_body) {
                pos += size;
                continue;
            }
            if let Some(stsd) =
                super::super::find_box_body(trak_body, &[b"mdia", b"minf", b"stbl", b"stsd"])
                && stsd.len() >= 8
            {
                let entries = &stsd[8..];
                let mut cursor = 0;
                while cursor + 8 <= entries.len() {
                    let entry_size = u32::from_be_bytes([
                        entries[cursor],
                        entries[cursor + 1],
                        entries[cursor + 2],
                        entries[cursor + 3],
                    ]) as usize;
                    if entry_size < 8 || cursor + entry_size > entries.len() {
                        break;
                    }
                    let entry_type: &[u8; 4] =
                        entries[cursor + 4..cursor + 8].try_into().unwrap();
                    if AAC_AUDIO_SAMPLE_ENTRIES.contains(&entry_type) {
                        return true;
                    }
                    cursor += entry_size;
                }
            }
        }
        pos += size;
    }
    false
}

/// Parse MPEG-4 descriptor tree rooted at ES_Descriptor and pluck the
/// DecoderSpecificInfo payload. Tags: ES_Descr=0x03, DecoderConfigDescr=0x04,
/// DecoderSpecificInfo=0x05. Each descriptor has a tag byte then a variable
/// length (7 bits per byte, top bit = continuation).
fn extract_asc_from_esds(body: &[u8]) -> Option<Vec<u8>> {
    // DecoderConfigDescriptor: 1 objectTypeIndication + 1 streamType
    // byte + 3 bufferSizeDB + 4 maxBitrate + 4 avgBitrate, then nested.
    let child = decoder_config_descriptor(body)?;
    if child.len() < 13 {
        return None;
    }
    let mut inner_cursor = &child[13..];
    while !inner_cursor.is_empty() {
        let (t, dsi_payload, r) = read_descriptor(inner_cursor)?;
        inner_cursor = r;
        if t == 0x05 {
            return Some(dsi_payload.to_vec());
        }
    }
    None
}

/// The `objectTypeIndication` (ISO/IEC 14496-1 Table 5) of the ES
/// descriptor tree `body`: 0x40 is MPEG-4 audio (AAC), 0x69 / 0x6B are
/// MPEG-2 / MPEG-1 audio, 0xA9..=0xAC are the DTS registrations.
pub(super) fn esds_object_type(body: &[u8]) -> Option<u8> {
    decoder_config_descriptor(body)?.first().copied()
}

/// The DecoderConfigDescriptor (tag 0x04) payload inside the
/// ES_Descriptor (tag 0x03) at the start of `body`.
fn decoder_config_descriptor(body: &[u8]) -> Option<&[u8]> {
    let (tag, payload, _rest) = read_descriptor(body)?;
    if tag != 0x03 {
        return None;
    }
    // ES_Descriptor layout: 2 bytes ES_ID + 1 flags byte + optional fields,
    // then nested descriptors. Flags bit layout (per spec):
    //   streamDependenceFlag (1) | URL_Flag (1) | OCRstreamFlag (1) | streamPriority (5)
    if payload.len() < 3 {
        return None;
    }
    let flags = payload[2];
    let mut off = 3;
    if flags & 0x80 != 0 {
        off += 2;
    } // dependsOn_ES_ID
    if flags & 0x40 != 0 {
        // URL_Flag: 1-byte length + URL string
        if off >= payload.len() {
            return None;
        }
        let url_len = payload[off] as usize;
        off += 1 + url_len;
    }
    if flags & 0x20 != 0 {
        off += 2;
    } // OCR_ES_ID
    if off > payload.len() {
        return None;
    }

    // Iterate children looking for DecoderConfigDescriptor (tag 0x04).
    let mut cursor = &payload[off..];
    while !cursor.is_empty() {
        let (tag, child, rest) = read_descriptor(cursor)?;
        cursor = rest;
        if tag == 0x04 {
            return Some(child);
        }
    }
    None
}

/// Parse a single descriptor: `[tag u8][len ULEB128-ish][payload]`. Returns
/// (tag, payload-slice, remaining-bytes-after-this-descriptor).
fn read_descriptor(data: &[u8]) -> Option<(u8, &[u8], &[u8])> {
    if data.is_empty() {
        return None;
    }
    let tag = data[0];
    let mut pos = 1;
    let mut length: usize = 0;
    for _ in 0..4 {
        if pos >= data.len() {
            return None;
        }
        let b = data[pos];
        pos += 1;
        length = (length << 7) | (b & 0x7F) as usize;
        if b & 0x80 == 0 {
            break;
        }
    }
    if pos + length > data.len() {
        return None;
    }
    let payload = &data[pos..pos + length];
    let rest = &data[pos + length..];
    Some((tag, payload, rest))
}

/// Decode the sampling_frequency out of an ASC per ISO/IEC 14496-3 §1.6.2.1.
/// ASC bitstream: audioObjectType(5) samplingFrequencyIndex(4) ...
/// If index==0xF then 24-bit sample rate follows inline.
pub(super) fn decode_asc_sample_rate(asc: &[u8]) -> Option<u32> {
    if asc.len() < 2 {
        return None;
    }
    let mut br = AscBitReader::new(asc);
    let aot = br.bits(5)?;
    let _extended_aot = if aot == 31 { br.bits(6)? + 32 } else { aot };
    let freq_idx = br.bits(4)? as usize;
    if freq_idx == 0xF {
        let sr = br.bits(24)?;
        Some(sr as u32)
    } else {
        const FREQS: [u32; 13] = [
            96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000,
            7350,
        ];
        FREQS.get(freq_idx).copied()
    }
}

pub(super) fn decode_asc_channels(asc: &[u8]) -> Option<u16> {
    if asc.len() < 2 {
        return None;
    }
    let mut br = AscBitReader::new(asc);
    let aot = br.bits(5)?;
    let _ext = if aot == 31 { br.bits(6)? + 32 } else { aot };
    let freq_idx = br.bits(4)? as usize;
    if freq_idx == 0xF {
        let _ = br.bits(24)?;
    }
    let chan_cfg = br.bits(4)? as u16;
    // chan_cfg 0 means "inspect PCE"; we don't bother — default to 2.
    if chan_cfg == 0 { Some(2) } else { Some(chan_cfg) }
}

struct AscBitReader<'a> {
    data: &'a [u8],
    pos: usize,
}
impl<'a> AscBitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn bits(&mut self, n: u32) -> Option<u64> {
        let mut v: u64 = 0;
        for _ in 0..n {
            let byte = *self.data.get(self.pos / 8)?;
            let bit = (byte >> (7 - (self.pos % 8))) & 1;
            v = (v << 1) | bit as u64;
            self.pos += 1;
        }
        Some(v)
    }
}
