/// Audio extraction from MP4/MOV and MKV/WebM containers.
///
/// Provides `extract_mp4_audio` and `extract_mkv_audio` for passthrough
/// muxing of AAC, Opus, AC-3 and E-AC-3 audio tracks.
use mp4::Mp4Reader;
use matroska_demuxer::{Frame as MkvFrame, MatroskaFile, TrackType as MkvTrackType};
use std::io::Cursor;

use super::AudioTrack;

mod aac;
mod opus;
mod ac3;
#[cfg(test)]
mod tests;

// Functions from sub-modules used by extract_mp4_audio / extract_mkv_audio.
use aac::{extract_aac_asc, mp4_has_aac_sample_entry, decode_asc_sample_rate, decode_asc_channels, hex_prefix};
use opus::{extract_mp4_opus_dops_body, dops_to_opus_head};
use ac3::{extract_mp4_ac3_dac3_body, extract_mp4_eac3_dec3_body};

// Preserve pub(super) visibility for the two decoder helpers so the original
// `super::audio::ac3_sample_rate_channels_from_dac3` / `..._eac3_...` call
// paths from demux siblings remain valid.
pub(crate) use ac3::{ac3_sample_rate_channels_from_dac3, eac3_sample_rate_channels_from_dec3};

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
///   2. **`extract_aac_asc` (aac.rs)** identifies audio traks by
///      `smhd` presence (positive evidence of audio intent — strictly
///      stronger than guessing by stsd[0]'s fourcc), walks ALL stsd
///      entries (not just entry[0] — some Apple sources emit
///      multi-entry stsd), accepts `mp4a` AND `enca`, descends into
///      `wave` via `find_esds_recursive`, and falls back to a
///      brute-force `esds` scan with a warn so unforeseen wrapper
///      shapes still produce audio.
///
///   3. **`mp4_has_aac_sample_entry` (aac.rs)** mirrors the same
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
        /// Decode-only: no MP4 passthrough form, but `codec::audio` can decode
        /// it, so the job layer re-encodes it to Opus.
        Vorbis,
        Mp3,
        /// Passthrough-only: no decoder (the DCA tables are normative data),
        /// but the frames copy into MP4 as `dtsc` verbatim.
        Dts,
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
            // Vorbis and MP3 have no MP4 passthrough form, but `codec::audio`
            // decodes both — so surface them and let the job layer re-encode to
            // Opus. Dropping them here would strand that decode path (and any
            // `--audio-filter`) as unreachable code for Matroska sources.
            "A_DTS" => MkvAudioKind::Dts,
            "A_VORBIS" => MkvAudioKind::Vorbis,
            "A_MPEG/L3" | "A_MPEG/L2" | "A_MPEG/L1" => MkvAudioKind::Mp3,
            other => {
                tracing::warn!(
                    codec = other,
                    "audio track dropped: no passthrough form (AAC / Opus / AC-3 / E-AC-3 / \
                     DTS) and no decoder (Vorbis / MP3) for this codec"
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
            MkvAudioKind::Ac3 | MkvAudioKind::Eac3 | MkvAudioKind::Mp3 | MkvAudioKind::Dts => track
                .codec_private()
                .map(|p| p.to_vec())
                .unwrap_or_default(),
            // Vorbis CodecPrivate is the three Xiph-laced setup headers; the
            // decoder cannot start without them.
            MkvAudioKind::Vorbis => {
                let cp = track.codec_private()?.to_vec();
                if cp.is_empty() {
                    tracing::warn!("A_VORBIS: empty CodecPrivate (no setup headers); dropping");
                    return None;
                }
                cp
            }
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
        MkvAudioKind::Ac3 | MkvAudioKind::Eac3 | MkvAudioKind::Dts => sample_rate,
        // Decode-only kinds never reach a sample table as-is — they're
        // re-encoded to Opus first — so the timescale only has to make the
        // per-packet duration fallback below come out sensibly.
        MkvAudioKind::Vorbis | MkvAudioKind::Mp3 => sample_rate,
    };
    let default_frame_samples_at_ts = match kind {
        MkvAudioKind::Aac => 1024u64,
        MkvAudioKind::Opus => 960u64,
        MkvAudioKind::Ac3 | MkvAudioKind::Eac3 => 1536u64,
        // DTS core: (NBLKS+1) x 32; 512 is the usual Blu-ray core frame.
        MkvAudioKind::Dts => 512u64,
        // MPEG-1 Layer III is 1152 samples per frame; Vorbis blocks vary, so
        // its long block is the useful approximation.
        MkvAudioKind::Mp3 => 1152u64,
        MkvAudioKind::Vorbis => 1024u64,
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
        // Decode-only: carried verbatim for `prepare_audio` to decode and
        // re-encode. `codec_private` holds the Vorbis setup headers, which
        // `codec::audio::create_decoder` needs as `extra_data`.
        MkvAudioKind::Vorbis => AudioTrack {
            codec: "vorbis".into(),
            samples,
            sample_rate,
            channels,
            asc: Vec::new(),
            codec_private: codec_private_or_empty,
            timescale,
            durations,
        },
        MkvAudioKind::Mp3 => AudioTrack {
            codec: "mp3".into(),
            samples,
            sample_rate,
            channels,
            asc: Vec::new(),
            codec_private: Vec::new(),
            timescale,
            durations,
        },
        // DTS passthrough: derive the `ddts` body from the first frame's core
        // sync header, the same way AC-3 derives `dac3`. MKV's CodecPrivate is
        // empty for DTS, so the bitstream is the only source.
        MkvAudioKind::Dts => {
            let first = samples.first()?;
            let core = match crate::dts_sync::parse_core_sync(first) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("MKV A_DTS: {e}; dropping audio");
                    return None;
                }
            };
            let hd = crate::dts_sync::has_hd_extension(first, &core);
            if hd {
                // Passthrough keeps the extension bytes, but the `ddts` box
                // describes the core — a player that can't do DTS-HD still
                // decodes the core, which is the point.
                tracing::info!("MKV A_DTS: DTS-HD extension present; carried through");
            }
            let ddts = crate::mux::ddts_body_from_sync(&core, hd);
            AudioTrack {
                codec: "dts".into(),
                samples,
                sample_rate: core.sample_rate,
                channels: core.channels,
                asc: Vec::new(),
                codec_private: ddts,
                timescale: core.sample_rate,
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
