/// Audio extraction from MP4/MOV and MKV/WebM containers.
///
/// Provides `extract_mp4_audio` and `extract_mkv_audio` for passthrough
/// muxing of AAC, Opus, AC-3 and E-AC-3 audio tracks.
use mp4::Mp4Reader;
use matroska_demuxer::{Frame as MkvFrame, MatroskaFile, TrackType as MkvTrackType};
use std::io::Cursor;

use super::AudioTrack;

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

// ─── MP4 / MOV audio extraction ──────────────────────────────────────────────

/// Pull the audio track out of an MP4 / MOV for passthrough.
///
/// ─── Codec families recognised ──────────────────────────────────────
/// (Squad-18 + Squad-23 + Squad-26)
/// - AAC-LC + HE-AAC v1/v2 + xHE-AAC USAC (`mp4a` / `enca` sample entry
///   + `esds`): emits `codec="aac"`, `asc` populated, `codec_private`
///   empty.
/// - Opus (`Opus` sample entry + `dOps`, RFC 7845 §4.4): emits
///   `codec="opus"`, `codec_private` populated with the OpusHead-form
///   body (LE numeric convention), `asc` empty.
/// - AC-3 (`ac-3` sample entry + `dac3`, ETSI TS 102 366 §F.2): emits
///   `codec="ac3"`, `codec_private` populated with the 3-byte dac3 body.
/// - E-AC-3 (`ec-3` sample entry + `dec3`, ETSI TS 102 366 §F.5): emits
///   `codec="eac3"`, `codec_private` populated with the dec3 body.
///
/// Other audio codecs (MP3, Vorbis, ...) log a warning and the track is
/// dropped — pipeline falls back to video-only.
///
/// ─── iPhone / Apple QuickTime resilience ────────────────────────────
///
/// Apple's recorder tooling produces several MOV / MP4 shapes that
/// trip strict ISOBMFF parsers and the `mp4` crate's classifier in
/// particular. The full path here was rebuilt incrementally against
/// real-world iPhone uploads (2026-05-03 → 2026-05-04 → 2026-05-07);
/// the contract has THREE pieces that all must be in place for an
/// iPhone source to round-trip with audio:
///
///   1. **`crates/container/src/mp4_sanitize.rs::sanitize_isobmff_box_sizes`**
///      runs at every MP4 demux entry point. Clamps over-reported
///      child box sizes (legacy QuickTime tooling sometimes emits
///      `wave` children whose advertised size exceeds the parent),
///      and CRITICALLY skips the 28-byte AudioSampleEntry fixed prefix
///      ONLY when the parent fourcc is `stsd` — without that
///      context-aware prefix handling, the inner `mp4a` inside `wave`
///      gets mis-aligned and the recursion loses the `esds` sibling.
///
///   2. **`extract_aac_asc` (this file)** identifies audio traks by
///      `smhd` presence (positive evidence of audio intent — strictly
///      stronger than guessing by stsd[0]'s fourcc), walks ALL stsd
///      entries (not just entry[0] — some Apple sources emit
///      multi-entry stsd), accepts `mp4a` AND `enca`, descends into
///      `wave` via `find_esds_recursive`, and falls back to a
///      brute-force `esds` scan with a warn so unforeseen wrapper
///      shapes still produce audio.
///
///   3. **`mp4_has_aac_sample_entry` (this file)** mirrors the same
///      smhd-based detection so the pre-flight check that bypasses
///      `mp4 0.14`'s broken `track.media_type()` matches the
///      extraction path's notion of "this trak has AAC".
///
/// Diagnostic logging: every silent-drop path here emits a
/// `tracing::warn!` with enough context (codec, hex prefix of ASC,
/// trak structure hint) that the next iPhone-shaped failure mode is
/// reproducible from CloudWatch alone. If you change this method, do
/// NOT remove the warns — add new ones for any new fail paths you
/// introduce.
///
/// Test coverage worth maintaining:
/// - `mp4_sanitize::tests::inner_mp4a_inside_wave_is_not_treated_as_sample_entry`
/// - any future test that constructs an iPhone-shaped synthetic MOV
///   and asserts `extract_mp4_audio` returns `Some(AudioTrack)` with
///   non-empty samples.
pub(super) fn extract_mp4_audio(data: &[u8]) -> Option<AudioTrack> {
    let size = data.len() as u64;
    let cursor = Cursor::new(data);
    let reader = Mp4Reader::read_header(cursor, size).ok()?;
    let track = reader
        .tracks()
        .values()
        .find(|t| t.track_type().ok() == Some(mp4::TrackType::Audio))?;
    let track_id = track.track_id();

    // Detect Opus / AC-3 / E-AC-3 first by sample-entry 4-cc — mp4 0.14's
    // `media_type()` doesn't surface those (it returns `unknown`), so we
    // walk the stsd box manually. AAC stays on the existing mp4-crate
    // path BUT with a manual `mp4a` 4cc fallback for iPhone-recorded
    // MOVs whose audio sample entry wraps esds in a `wave` sub-box —
    // `mp4 0.14`'s media_type() returns Err on those, which previously
    // caused silent audio drop on every iPhone upload. Burned 2026-05-03.
    let opus_dops = extract_mp4_opus_dops_body(data);
    let ac3_cfg = extract_mp4_ac3_dac3_body(data);
    let eac3_cfg = extract_mp4_eac3_dec3_body(data);
    let media_type = track.media_type();
    let crate_says_aac = media_type
        .as_ref()
        .map(|mt| matches!(mt, mp4::MediaType::AAC))
        .unwrap_or(false);
    let manual_says_aac = mp4_has_aac_sample_entry(data);
    let is_aac = crate_says_aac || manual_says_aac;

    if !is_aac && opus_dops.is_none() && ac3_cfg.is_none() && eac3_cfg.is_none() {
        match media_type {
            Ok(mt) => tracing::warn!(
                codec = ?mt,
                "audio passthrough skipped: only AAC / Opus / AC-3 / E-AC-3 are supported"
            ),
            Err(e) => tracing::warn!(
                error = ?e,
                "audio passthrough skipped: mp4 crate could not classify audio sample entry, \
                 and manual stsd walk found no recognized 4cc"
            ),
        }
        return None;
    }

    let timescale = track.timescale();
    let sample_count = track.sample_count();

    if is_aac {
        // Verbatim ASC straight from esds — mp4-rust decodes it into
        // {profile, freq_index, chan_conf} which discards HE-AAC / xHE-AAC
        // extension bits. We walk the box tree ourselves.
        //
        // `extract_aac_asc` is the iPhone-survivable path: walks all
        // traks, identifies audio via smhd, walks all stsd entries,
        // accepts mp4a + enca, descends into wave, and falls back to a
        // brute-force esds scan with a warn. If it returns None, every
        // fail path inside has already logged; we don't need to log here.
        let asc = match extract_aac_asc(data) {
            Some(a) => a,
            None => return None,
        };
        if asc.is_empty() {
            tracing::warn!(
                "AAC track found but AudioSpecificConfig is empty; dropping. \
                 Source has an esds box but its DecoderSpecificInfo descriptor is \
                 zero-length."
            );
            return None;
        }
        // Squad-25: surface the effective output channel count (post-PS
        // upmix for HE-AAC v2 mono PS) and the SBR-doubled output rate
        // for HE-AAC v1/v2. Falls back to the legacy core-only decoder
        // when the structured parser declines (e.g. unrecognised ASC).
        let parsed = crate::aac_asc::parse_aac_asc(&asc);
        let sample_rate = match parsed
            .as_ref()
            .and_then(|p| p.sbr_sample_rate.or(Some(p.sample_rate)))
            .or_else(|| decode_asc_sample_rate(&asc))
        {
            Some(sr) => sr,
            None => {
                tracing::warn!(
                    asc_hex = %hex_prefix(&asc, 16),
                    "AAC ASC sample rate could not be decoded; dropping audio. \
                     Likely an extended sampling-frequency-index escape (0x0F) \
                     pointing at unsupported bytes, or a malformed ASC."
                );
                return None;
            }
        };
        let channels = parsed
            .as_ref()
            .map(crate::aac_asc::effective_output_channels)
            .or_else(|| decode_asc_channels(&asc))
            .unwrap_or(2);

        let mut samples = Vec::with_capacity(sample_count as usize);
        let mut durations = Vec::with_capacity(sample_count as usize);
        // AAC-LC encodes 1024 PCM samples per access unit; AAC-HE
        // (SBR) doubles the OUTPUT to 2048 but the core frame stays
        // 1024 and the track's `mdhd.timescale` typically equals the
        // SOURCE sample rate (not the SBR-doubled rate), so 1024 is
        // the right tick count regardless of HE/non-HE.
        //
        // Fragmented MP4 sources (notably iPhone capture, some
        // screen-recorder outputs) sometimes ship a `traf.trun`
        // without per-sample durations AND a `tfhd`/`mvex.trex` whose
        // `default_sample_duration` is 0. The mp4 crate then surfaces
        // `sample.duration = 0` for every audio access unit, which
        // sums to 0 total and trips the audio/video duration drift
        // validator at job-end (failure mode observed on
        // 2026-05-09 / job 37 — full-length audio dropped despite
        // 12231 of 12318 access units extracting cleanly).
        //
        // Falling back to 1024 ticks per zero-duration sample
        // re-derives the natural per-frame duration. Spec-conformant
        // sources (where `sample.duration` carries the real value)
        // are unaffected — fallback only fires on the 0 case.
        const AAC_LC_CORE_FRAME_SIZE_TICKS: u32 = 1024;

        // Fragmented MP4 path. The mp4 crate's `read_sample` returns
        // garbage (typically the bytes of an adjacent moof box header)
        // for fragmented audio tracks just like it does for video —
        // see `build_fragmented_sample_table`'s docstring for the bug
        // history. Walk moof->traf->trun ourselves and pull sample
        // bytes straight out of `data` at the resolved offsets.
        if let Some(frag) = super::mp4::build_fragmented_sample_table(data, track_id, 0, 0) {
            tracing::info!(
                track_id,
                sample_count = frag.len(),
                "fragmented MP4 audio: built sample table from moof/traf/trun"
            );
            for s in &frag {
                let off = s.offset as usize;
                let sz = s.size as usize;
                let end = match off.checked_add(sz) {
                    Some(e) if e <= data.len() => e,
                    _ => {
                        tracing::warn!(
                            track_id,
                            offset = s.offset,
                            size = s.size,
                            data_len = data.len(),
                            "fragmented audio sample range out of bounds; truncating track"
                        );
                        break;
                    }
                };
                // For AAC, ignore the source trun's per-sample
                // duration entirely — AAC-LC AUs are exactly 1024
                // PCM samples by spec. Source files (Apple / iOS /
                // some web recorders) attach encoder-priming
                // bookkeeping to the first sample's duration
                // (e.g. 3298 ticks for a 1024-PCM-sample frame
                // observed 2026-05-09); propagating that into our
                // output mux makes Chrome MSE reject the audio
                // SourceBuffer with `MediaSource readyState ended`.
                // Fixed 1024 yields a clean contiguous timeline.
                let dur = if is_aac {
                    AAC_LC_CORE_FRAME_SIZE_TICKS
                } else {
                    s.duration_ticks
                };
                durations.push(dur);
                samples.push(data[off..end].to_vec());
            }
        } else {
            // Static moov sample table path — `read_sample` is correct
            // here, the bug is fragmented-only.
            let mut cursor = Cursor::new(data);
            let mut reader = match Mp4Reader::read_header(&mut cursor, size) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(error = %e, "audio passthrough: re-opening MP4 for sample read failed; dropping audio");
                    return None;
                }
            };
            for idx in 1..=sample_count {
                match reader.read_sample(track_id, idx) {
                    Ok(Some(sample)) => {
                        let dur = if is_aac && sample.duration == 0 {
                            AAC_LC_CORE_FRAME_SIZE_TICKS
                        } else {
                            sample.duration
                        };
                        durations.push(dur);
                        samples.push(sample.bytes.to_vec());
                    }
                    Ok(None) => break,
                    Err(e) => {
                        tracing::warn!(
                            track_id,
                            idx,
                            error = %e,
                            "audio passthrough: read_sample error mid-track; \
                             keeping samples read so far ({} of {}) and continuing",
                            samples.len(),
                            sample_count
                        );
                        break;
                    }
                }
            }
        }
        if samples.is_empty() {
            tracing::warn!(
                track_id,
                sample_count,
                "AAC track parsed (ASC + sample table) but read_sample returned 0 \
                 samples — possible mp4 crate stsd / stco parse failure on the source"
            );
            return None;
        }
        return Some(AudioTrack {
            codec: "aac".into(),
            samples,
            sample_rate,
            channels,
            asc,
            codec_private: Vec::new(),
            timescale,
            durations,
        });
    }

    // AC-3 path. The `dac3` body lives in the sample entry; we use it as
    // codec_private. Samples come back via the standard reader path (one
    // AC-3 syncframe per MP4 sample). MP4 stsd preamble already advertises
    // sample_rate (Q16) and channelcount but we re-derive both from the
    // dac3 body for accuracy: the AudioSampleEntry preamble can mis-report
    // (e.g. "48000" for an embedded 32 kHz stream — strict players use the
    // dac3 body anyway).
    if let Some(dac3_body) = ac3_cfg {
        if dac3_body.len() < 3 {
            tracing::warn!("MP4 AC-3 dac3 body shorter than 3 bytes — dropping audio");
            return None;
        }
        let (sr, ch) = ac3_sample_rate_channels_from_dac3(&dac3_body)?;
        let mut cursor = Cursor::new(data);
        let mut reader = Mp4Reader::read_header(&mut cursor, size).ok()?;
        let mut samples = Vec::with_capacity(sample_count as usize);
        let mut durations = Vec::with_capacity(sample_count as usize);
        for idx in 1..=sample_count {
            match reader.read_sample(track_id, idx).ok()? {
                Some(sample) => {
                    durations.push(sample.duration);
                    samples.push(sample.bytes.to_vec());
                }
                None => break,
            }
        }
        if samples.is_empty() {
            return None;
        }
        return Some(AudioTrack {
            codec: "ac3".into(),
            samples,
            sample_rate: sr,
            channels: ch,
            asc: Vec::new(),
            codec_private: dac3_body[..3].to_vec(),
            timescale,
            durations,
        });
    }

    // E-AC-3 path. Same shape as AC-3 — body extracted from `dec3`.
    if let Some(dec3_body) = eac3_cfg {
        if dec3_body.len() < 5 {
            tracing::warn!("MP4 E-AC-3 dec3 body shorter than 5 bytes — dropping audio");
            return None;
        }
        let (sr, ch) = eac3_sample_rate_channels_from_dec3(&dec3_body)?;
        let mut cursor = Cursor::new(data);
        let mut reader = Mp4Reader::read_header(&mut cursor, size).ok()?;
        let mut samples = Vec::with_capacity(sample_count as usize);
        let mut durations = Vec::with_capacity(sample_count as usize);
        for idx in 1..=sample_count {
            match reader.read_sample(track_id, idx).ok()? {
                Some(sample) => {
                    durations.push(sample.duration);
                    samples.push(sample.bytes.to_vec());
                }
                None => break,
            }
        }
        if samples.is_empty() {
            return None;
        }
        return Some(AudioTrack {
            codec: "eac3".into(),
            samples,
            sample_rate: sr,
            channels: ch,
            asc: Vec::new(),
            codec_private: dec3_body,
            timescale,
            durations,
        });
    }

    // Opus path. The dOps body lives in the sample entry; samples (one
    // Opus packet per MP4 sample) come back via the standard reader path
    // since stco / stsc / stsz iteration is codec-agnostic.
    let dops_body = opus_dops?; // body bytes only, no 'dOps' magic
    let opus_head = dops_to_opus_head(&dops_body)?;
    // For MP4-Opus the timescale is mandated 48000 by RFC 7845 §3 and
    // virtually every encoder honours that, but tolerate divergence — the
    // pipeline-level mux re-pins to 48000 when emitting.
    let input_sample_rate =
        u32::from_le_bytes([opus_head[4], opus_head[5], opus_head[6], opus_head[7]]);
    let channels = opus_head[1] as u16;

    let mut cursor = Cursor::new(data);
    let mut reader = Mp4Reader::read_header(&mut cursor, size).ok()?;
    let mut samples = Vec::with_capacity(sample_count as usize);
    let mut durations = Vec::with_capacity(sample_count as usize);
    for idx in 1..=sample_count {
        match reader.read_sample(track_id, idx).ok()? {
            Some(sample) => {
                durations.push(sample.duration);
                samples.push(sample.bytes.to_vec());
            }
            None => break,
        }
    }
    if samples.is_empty() {
        return None;
    }
    Some(AudioTrack {
        codec: "opus".into(),
        samples,
        sample_rate: input_sample_rate,
        channels,
        asc: Vec::new(),
        codec_private: opus_head,
        timescale,
        durations,
    })
}

/// Walk every `trak` looking for one whose `stsd` contains an `ac-3`
/// sample entry (ETSI TS 102 366 §F.2). Returns the body bytes of the
/// contained `dac3` box (without the 8-byte box header) or None.
fn extract_mp4_ac3_dac3_body(data: &[u8]) -> Option<Vec<u8>> {
    extract_mp4_audio_config_body(data, b"ac-3", b"dac3")
}

/// Walk every `trak` looking for one whose `stsd` contains an `ec-3`
/// sample entry (ETSI TS 102 366 §F.5). Returns the body bytes of the
/// contained `dec3` box (without the 8-byte box header) or None.
fn extract_mp4_eac3_dec3_body(data: &[u8]) -> Option<Vec<u8>> {
    extract_mp4_audio_config_body(data, b"ec-3", b"dec3")
}

/// Generic walker — find an audio sample-entry of `entry_fourcc`, return
/// the body of the named codec-config child (`dac3` / `dec3`) inside.
/// Mirrors `extract_mp4_opus_dops_body`'s shape but parameterised on the
/// entry / config 4-cc pair.
fn extract_mp4_audio_config_body(
    data: &[u8],
    entry_fourcc: &[u8; 4],
    cfg_fourcc: &[u8; 4],
) -> Option<Vec<u8>> {
    let moov = super::find_direct_child(data, b"moov")?;
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
            if let Some(cfg) = extract_audio_cfg_from_trak(trak_body, entry_fourcc, cfg_fourcc) {
                return Some(cfg);
            }
        }
        pos += size;
    }
    None
}

fn extract_audio_cfg_from_trak(
    trak: &[u8],
    entry_fourcc: &[u8; 4],
    cfg_fourcc: &[u8; 4],
) -> Option<Vec<u8>> {
    let stsd = super::find_box_body(trak, &[b"mdia", b"minf", b"stbl", b"stsd"])?;
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
        if &entry_type == entry_fourcc {
            let end = pos + entry_size;
            // AudioSampleEntry layout per ISO/IEC 14496-12 §8.5.2.2: after
            // the 8-byte box header there's a 28-byte fixed preamble
            // followed by nested codec-specific boxes.
            let child_start = pos + 8 + 28;
            if child_start >= end {
                return None;
            }
            return super::find_direct_child(&stsd[child_start..end], cfg_fourcc)
                .map(|b| b.to_vec());
        }
        pos += entry_size;
    }
    None
}

/// Decode (sample_rate, channel_count) from a 3-byte `dac3` body per
/// ETSI TS 102 366 §F.4. Bit layout (MSB-first across 24 bits):
///   bits 23..22 fscod          (shift=22)
///   bits 21..17 bsid           (shift=17)
///   bits 16..14 bsmod          (shift=14)
///   bits 13..11 acmod          (shift=11)
///   bit  10     lfeon          (shift=10)
///   bits  9.. 5 bit_rate_code  (shift= 5)
///   bits  4.. 0 reserved (=0)
pub(super) fn ac3_sample_rate_channels_from_dac3(dac3: &[u8]) -> Option<(u32, u16)> {
    if dac3.len() < 3 {
        return None;
    }
    let raw = ((dac3[0] as u32) << 16) | ((dac3[1] as u32) << 8) | dac3[2] as u32;
    let fscod = ((raw >> 22) & 0x03) as u8;
    let acmod = ((raw >> 11) & 0x07) as u8;
    let lfeon = ((raw >> 10) & 0x01) == 1;
    let sr = match fscod {
        0 => 48_000,
        1 => 44_100,
        2 => 32_000,
        _ => return None,
    };
    Some((sr, crate::ac3_sync::channel_count(acmod, lfeon)))
}

/// Decode (sample_rate, channel_count) from a `dec3` body per ETSI TS 102
/// 366 §F.6. Squad-26 only emits / extracts the single-substream form
/// (5-byte body), which is what every vanilla 5.1 / 7.1 E-AC-3 file uses.
pub(super) fn eac3_sample_rate_channels_from_dec3(dec3: &[u8]) -> Option<(u32, u16)> {
    if dec3.len() < 5 {
        return None;
    }
    // Header: data_rate(13b) + num_ind_sub-1(3b) packed in bytes 0..2.
    // Per-substream block starts at bit position 16.
    // bits 16..18 = fscod
    //  18..23 = bsid (=16)
    //  23..24 = reserved
    //  24..25 = asvc
    //  25..28 = bsmod
    //  28..31 = acmod
    //  31..32 = lfeon
    let raw_be = u64::from(dec3[0]) << 32
        | u64::from(dec3[1]) << 24
        | u64::from(dec3[2]) << 16
        | u64::from(dec3[3]) << 8
        | u64::from(dec3[4]);
    // dec3 is 5 bytes total (40 bits) for the single-substream case.
    // Adjust shifts: high bit is bit 39 in our 40-bit value.
    //   bit 39..27 = data_rate (13 bits)  shift=27
    //   bit 26..24 = num_ind_sub-1        shift=24
    //   bit 23..22 = fscod                shift=22
    //   bit 21..17 = bsid                 shift=17
    //   bit 16     = reserved
    //   bit 15     = asvc
    //   bit 14..12 = bsmod
    //   bit 11..9  = acmod                shift=9
    //   bit 8      = lfeon                shift=8
    //   bit 7..5   = reserved
    //   bit 4..1   = num_dep_sub
    //   bit 0      = reserved
    let fscod = ((raw_be >> 22) & 0x03) as u8;
    let acmod = ((raw_be >> 9) & 0x07) as u8;
    let lfeon = ((raw_be >> 8) & 0x01) == 1;
    let sr = crate::ac3_sync::eac3_sample_rate_hz(fscod, 0);
    if sr == 0 {
        return None;
    }
    Some((sr, crate::ac3_sync::channel_count(acmod, lfeon)))
}

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
fn extract_mp4_opus_dops_body(data: &[u8]) -> Option<Vec<u8>> {
    let moov = super::find_direct_child(data, b"moov")?;
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
    let stsd = super::find_box_body(trak, &[b"mdia", b"minf", b"stbl", b"stsd"])?;
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
            return super::find_direct_child(&stsd[child_start..end], b"dOps")
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
fn dops_to_opus_head(dops: &[u8]) -> Option<Vec<u8>> {
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
fn extract_aac_asc(data: &[u8]) -> Option<Vec<u8>> {
    let moov = super::find_direct_child(data, b"moov")?;
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

/// Format the first `n` bytes of `bytes` as a hex string for diagnostic
/// log lines. Used by `extract_mp4_audio` so the log records the actual
/// ASC prefix when something downstream fails to parse it — that lets us
/// reproduce iPhone-shaped issues from CloudWatch alone, without needing
/// the user's source file in hand.
fn hex_prefix(bytes: &[u8], n: usize) -> String {
    let mut out = String::with_capacity(n * 2);
    for b in bytes.iter().take(n) {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Quick "is this trak an audio trak?" check. ISO 14496-12 §8.4.5.3
/// requires `smhd` (Sound Media Header) inside `mdia/minf` for every
/// audio trak. Looking for it is a strictly stronger signal than
/// inspecting the first `stsd` entry's fourcc — it's positive evidence
/// of trak intent rather than fourcc-position guessing.
fn trak_is_audio(trak: &[u8]) -> bool {
    super::find_box_body(trak, &[b"mdia", b"minf", b"smhd"]).is_some()
}

fn extract_asc_from_trak(trak: &[u8]) -> Option<Vec<u8>> {
    let stsd = super::find_box_body(trak, &[b"mdia", b"minf", b"stbl", b"stsd"])?;
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
            let esds_body = &body[pos + 8 + 4..pos + sub_size];
            return extract_asc_from_esds(esds_body);
        }
        if sub_type == b"wave" {
            // QuickTime audio extension. Recurse — esds usually lives
            // inside.
            if let Some(asc) = find_esds_recursive(&body[pos + 8..pos + sub_size]) {
                return Some(asc);
            }
        }
        pos += sub_size;
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
fn mp4_has_aac_sample_entry(data: &[u8]) -> bool {
    let Some(moov) = super::find_direct_child(data, b"moov") else {
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
                super::find_box_body(trak_body, &[b"mdia", b"minf", b"stbl", b"stsd"])
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
        if tag != 0x04 {
            continue;
        }
        // DecoderConfigDescriptor: 1 objectTypeIndication + 1 streamType
        // byte + 3 bufferSizeDB + 4 maxBitrate + 4 avgBitrate, then nested.
        if child.len() < 13 {
            return None;
        }
        let inner = &child[13..];
        let mut inner_cursor = inner;
        while !inner_cursor.is_empty() {
            let (t, dsi_payload, r) = read_descriptor(inner_cursor)?;
            inner_cursor = r;
            if t == 0x05 {
                return Some(dsi_payload.to_vec());
            }
        }
        return None;
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
fn decode_asc_sample_rate(asc: &[u8]) -> Option<u32> {
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

fn decode_asc_channels(asc: &[u8]) -> Option<u16> {
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

// ─── MKV / WebM audio extraction ─────────────────────────────────────────────

/// Pull the audio track out of an MKV / WebM for passthrough. Four codec
/// families are recognised today (Squad-18 + Squad-23 + Squad-26):
/// - `A_AAC`: AAC-LC. CodecPrivate carries the AudioSpecificConfig verbatim.
/// - `A_OPUS`: Opus. CodecPrivate carries the OpusHead body verbatim per
///   RFC 7845 §5.2 (the WebM spec mirrors this) — same bytes the dOps
///   writer needs (in OpusHead LE numeric form).
/// - `A_AC3`: AC-3. CodecPrivate is empty (frames are self-describing); we
///   derive the `dac3` body from the first frame's sync header per
///   ETSI TS 102 366 §F.4.
/// - `A_EAC3`: E-AC-3. Same — empty CodecPrivate; derive `dec3` body from
///   the first frame's sync header per ETSI TS 102 366 §F.6.
///
/// Other audio codec IDs (`A_VORBIS`, `A_MPEG/L3`) log a warning and the
/// track is dropped — pipeline falls back to video-only.
///
/// WebM is a Matroska subset so the same code path covers both.
pub(super) fn extract_mkv_audio(data: &[u8]) -> Option<AudioTrack> {
    let cursor = Cursor::new(data);
    let mut mkv = MatroskaFile::open(cursor).ok()?;

    enum MkvAudioKind {
        Aac,
        Opus,
        Ac3,
        Eac3,
    }

    let (track_number, kind, codec_private_or_empty, sample_rate, channels, default_duration) = {
        let track = mkv
            .tracks()
            .iter()
            .find(|t| t.track_type() == MkvTrackType::Audio)?;
        let codec_id = track.codec_id();
        let kind = match codec_id {
            "A_AAC" => MkvAudioKind::Aac,
            "A_OPUS" => MkvAudioKind::Opus,
            "A_AC3" => MkvAudioKind::Ac3,
            "A_EAC3" => MkvAudioKind::Eac3,
            other => {
                tracing::warn!(
                    codec = other,
                    "audio passthrough skipped: only AAC / Opus / AC-3 / E-AC-3 are supported"
                );
                return None;
            }
        };
        // CodecPrivate is mandatory for AAC / Opus (carries ASC / OpusHead).
        // It's typically EMPTY for AC-3 / E-AC-3 in MKV — frames are
        // self-describing and the dac3 / dec3 body is derived from the
        // first frame's sync header. Tolerate either.
        let codec_private = match kind {
            MkvAudioKind::Aac => {
                let cp = track.codec_private()?.to_vec();
                if cp.is_empty() {
                    return None;
                }
                cp
            }
            MkvAudioKind::Opus => {
                // RFC 7845 §5.2: MKV CodecPrivate carries the full OpusHead
                // packet — magic signature "OpusHead" + body. Our internal
                // AudioTrack.codec_private contract (and the dOps writer in
                // mux.rs) expects the post-magic body only, so strip the
                // 8-byte magic if present. Without this, mux reads
                // codec_private[10] expecting ChannelMappingFamily but
                // actually gets pre-skip's LSB byte of OpusHead.
                let mut cp = track.codec_private()?.to_vec();
                if cp.is_empty() {
                    return None;
                }
                if cp.len() >= 8 && &cp[..8] == b"OpusHead" {
                    cp.drain(..8);
                }
                if cp.is_empty() {
                    return None;
                }
                cp
            }
            MkvAudioKind::Ac3 | MkvAudioKind::Eac3 => track
                .codec_private()
                .map(|p| p.to_vec())
                .unwrap_or_default(),
        };
        let audio = track.audio()?;
        let sr = audio.sampling_frequency() as u32;
        let ch = audio.channels().get() as u16;
        let default_duration = track.default_duration().map(|d| d.get());
        (
            track.track_number().get(),
            kind,
            codec_private,
            sr,
            ch,
            default_duration,
        )
    };

    // Per-codec timescale + per-frame default duration tick conversion.
    //   - AAC: mdhd timescale = sample_rate; natural frame = 1024 samples.
    //   - Opus: mdhd timescale pinned to 48000 per RFC 7845 §3 regardless
    //     of the source's nominal sample_rate; natural frame = 960 samples
    //     (20 ms standard libopus encoder frame).
    //   - AC-3 / E-AC-3: mdhd timescale = sample_rate; natural frame =
    //     1536 samples (6 blocks × 256 / ETSI TS 102 366).
    let timescale = match kind {
        MkvAudioKind::Aac => sample_rate,
        MkvAudioKind::Opus => 48_000,
        MkvAudioKind::Ac3 | MkvAudioKind::Eac3 => sample_rate,
    };
    let default_frame_samples_at_ts = match kind {
        MkvAudioKind::Aac => 1024u64,
        MkvAudioKind::Opus => 960u64,
        MkvAudioKind::Ac3 | MkvAudioKind::Eac3 => 1536u64,
    };
    // For the fallback duration math we need the rate matching the chosen
    // timescale (NOT the source's nominal sample_rate when kind=Opus).
    let timescale_for_fallback = if timescale == 0 { 48_000 } else { timescale };

    let mut samples: Vec<Vec<u8>> = Vec::new();
    let mut durations: Vec<u32> = Vec::new();
    let mut frame = MkvFrame::default();
    loop {
        match mkv.next_frame(&mut frame) {
            Ok(true) => {
                if frame.track == track_number {
                    // Prefer the block's own duration, then default_duration,
                    // then the codec's natural frame size at the chosen
                    // mdhd timescale.
                    let dur_ns = frame.duration.or(default_duration).unwrap_or_else(|| {
                        1_000_000_000u64 * default_frame_samples_at_ts
                            / timescale_for_fallback as u64
                    });
                    // Convert ns → mdhd timescale ticks.
                    let dur_ticks =
                        ((dur_ns as u128) * (timescale as u128) / 1_000_000_000) as u32;
                    durations.push(dur_ticks.max(1));
                    samples.push(std::mem::take(&mut frame.data));
                }
            }
            Ok(false) => break,
            Err(_) => break,
        }
    }

    if samples.is_empty() {
        return None;
    }

    Some(match kind {
        MkvAudioKind::Aac => {
            // Squad-25: MKV `Audio.Channels` is an integer hint and the ASC
            // (CodecPrivate) is canonical for HE-AAC v2 PS upmix + multichannel
            // configs. Prefer the parsed-ASC counts when available; fall back
            // to whatever the MKV header advertised.
            let parsed = crate::aac_asc::parse_aac_asc(&codec_private_or_empty);
            let aac_channels = parsed
                .as_ref()
                .map(crate::aac_asc::effective_output_channels)
                .unwrap_or(channels);
            let aac_sample_rate = parsed
                .as_ref()
                .and_then(|p| p.sbr_sample_rate.or(Some(p.sample_rate)))
                .unwrap_or(sample_rate);
            AudioTrack {
                codec: "aac".into(),
                samples,
                sample_rate: aac_sample_rate,
                channels: aac_channels,
                asc: codec_private_or_empty,
                codec_private: Vec::new(),
                timescale: aac_sample_rate, // mdhd timescale tracks the effective rate
                durations,
            }
        }
        MkvAudioKind::Opus => AudioTrack {
            codec: "opus".into(),
            samples,
            sample_rate,
            channels,
            asc: Vec::new(),
            codec_private: codec_private_or_empty,
            timescale,
            durations,
        },
        MkvAudioKind::Ac3 => {
            // CodecPrivate is empty for AC-3 in MKV. Synthesize the dac3
            // body by walking the first frame's sync header and re-packing
            // per ETSI TS 102 366 §F.4. Per-frame samples already collected.
            let dac3 = match samples
                .first()
                .and_then(|f| crate::ac3_sync::parse_sync_info(f).ok())
            {
                Some(crate::ac3_sync::SyncInfo::Ac3(s)) => {
                    crate::mux::dac3_body_from_sync(&s).to_vec()
                }
                _ => {
                    tracing::warn!(
                        "MKV A_AC3: failed to parse first frame sync header — dropping audio"
                    );
                    return None;
                }
            };
            // Re-derive sample_rate / channel layout from the parsed sync —
            // it's the authoritative source.
            let (sr, ch) =
                ac3_sample_rate_channels_from_dac3(&dac3).unwrap_or((sample_rate, channels));
            AudioTrack {
                codec: "ac3".into(),
                samples,
                sample_rate: sr,
                channels: ch,
                asc: Vec::new(),
                codec_private: dac3,
                timescale: sr,
                durations,
            }
        }
        MkvAudioKind::Eac3 => {
            // Same story for E-AC-3: derive dec3 from the first frame.
            let (dec3, sr, ch) = match samples
                .first()
                .and_then(|f| crate::ac3_sync::parse_sync_info(f).ok())
            {
                Some(crate::ac3_sync::SyncInfo::Eac3(s)) => {
                    // data_rate (kbps / 2) computed from the source frame:
                    //   frame_size_bytes = (frmsiz + 1) * 2
                    //   bitrate_kbps = (frame_size_bytes * 8 * sample_rate) / samples_per_frame / 1000
                    let sr = crate::ac3_sync::eac3_sample_rate_hz(s.fscod, s.fscod2);
                    let spf = crate::ac3_sync::eac3_samples_per_frame(s.numblkscod) as u64;
                    let frame_bytes = ((s.frmsiz as u64) + 1) * 2;
                    let bitrate_kbps = if spf > 0 && sr > 0 {
                        (frame_bytes * 8 * sr as u64) / spf / 1000
                    } else {
                        0
                    };
                    let data_rate = bitrate_kbps.div_ceil(2) as u16;
                    let dec3 = crate::mux::dec3_body_from_sync(&s, data_rate).to_vec();
                    let ch = crate::ac3_sync::channel_count(s.acmod, s.lfeon);
                    (dec3, sr, ch)
                }
                _ => {
                    tracing::warn!(
                        "MKV A_EAC3: failed to parse first frame sync header — dropping audio"
                    );
                    return None;
                }
            };
            AudioTrack {
                codec: "eac3".into(),
                samples,
                sample_rate: sr,
                channels: ch,
                asc: Vec::new(),
                codec_private: dec3,
                timescale: sr,
                durations,
            }
        }
    })
}
