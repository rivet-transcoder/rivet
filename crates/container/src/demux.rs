use anyhow::{Context, Result, bail};
use codec::frame::{
    ColorMetadata, ColorSpace, ContentLightLevel, MasteringDisplay, PixelFormat, StreamInfo,
    TransferFn,
};
use matroska_demuxer::{
    Colour as MkvColour, Frame as MkvFrame, MasteringMetadata as MkvMastering, MatrixCoefficients,
    MatroskaFile, Primaries, Range as MkvRange, TrackType as MkvTrackType, TransferCharacteristics,
};
use mp4::Mp4Reader;
use std::io::Cursor;

use crate::mp4_sanitize::sanitize_isobmff_box_sizes;

use crate::annexb::{
    AvcConfig, HevcConfig, NaluCodec, ParamSetTracker, length_prefixed_to_annexb_tracked,
    parse_avcc, parse_hvcc,
};
use crate::avi::demux_avi;
use crate::streaming::{DemuxHeader, Sample, StreamingDemuxer};
use crate::ts::demux_ts;
use crate::{MkvColorInfo, MkvMasteringMetadata};

pub struct DemuxResult {
    pub codec: String,
    pub info: StreamInfo,
    pub samples: Vec<Vec<u8>>,
    /// Optional audio track carried through for passthrough muxing. Populated
    /// when the input has an AAC track (MP4: `mp4a` sample entry; MKV codec
    /// id `A_AAC`). Other audio codecs log a warning and are dropped.
    pub audio: Option<AudioTrack>,
}

/// Audio track extracted for passthrough or transcode. Supports two codec
/// families today (Squad-18 + Squad-23):
/// - **AAC-LC**: `codec = "aac"`, `asc` holds the verbatim
///   AudioSpecificConfig bytes sourced from the MP4 esds descriptor (not
///   the mp4 crate's rebuilt form) or MKV `CodecPrivate`, so HE-AAC /
///   xHE-AAC signaling survives the copy. `codec_private` is empty.
/// - **Opus**: `codec = "opus"`, `codec_private` holds the RFC 7845 §5.1
///   `OpusHead` body verbatim — for MKV/WebM that's exactly the
///   `CodecPrivate` element bytes (post-magic — RFC 7845 §5.2 specifies
///   no magic prefix for the MKV CodecPrivate); for MP4-Opus that's the
///   `dOps` body re-serialised in OpusHead's LE numeric convention. `asc`
///   is empty.
///
/// `samples` are codec-native packets (AAC: ADTS-stripped raw access
/// units; Opus: TOC-prefixed Opus packets, one per frame). `durations`
/// are per-sample in `timescale` units.
#[derive(Debug, Clone)]
pub struct AudioTrack {
    pub codec: String,
    pub samples: Vec<Vec<u8>>,
    pub sample_rate: u32,
    pub channels: u16,
    /// AAC-only: AudioSpecificConfig bytes. Empty for non-AAC codecs.
    pub asc: Vec<u8>,
    /// Opus-only: OpusHead body bytes (RFC 7845 §5.1). Empty for non-Opus
    /// codecs. The 8-byte 'OpusHead' magic prefix is NOT included — only
    /// the post-magic body.
    pub codec_private: Vec<u8>,
    pub timescale: u32,
    pub durations: Vec<u32>,
}

/// Dispatch to the right demuxer based on container magic bytes.
pub fn demux(data: &[u8]) -> Result<DemuxResult> {
    match detect_container(data) {
        // MOV shares its demuxer with MP4 — same ISOBMFF box tree, same
        // sample-entry structure. `detect_container` returns "mp4" for
        // both `ftyp mp4*` and `ftyp qt  ` / bare-moov MOVs.
        "mp4" => demux_mp4(data),
        "mkv" => demux_mkv(data),
        "avi" => demux_avi(data),
        "ts" => demux_ts(data),
        other => bail!("unsupported container: {other}"),
    }
}

fn detect_container(data: &[u8]) -> &'static str {
    if data.len() < 12 {
        return "unknown";
    }
    // ISOBMFF: MP4 (`ftyp mp41`/`mp42`/`isom`/...) and MOV (`ftyp qt  `)
    // both land here. Older MOV files sometimes ship without a top-level
    // `ftyp` and lead with `moov` or `mdat` directly — accept those too.
    if &data[4..8] == b"ftyp" || &data[4..8] == b"moov" || &data[4..8] == b"mdat" {
        return "mp4";
    }
    // Matroska/WebM: EBML signature.
    if data[0] == 0x1A && data[1] == 0x45 && data[2] == 0xDF && data[3] == 0xA3 {
        return "mkv";
    }
    // RIFF-based AVI: "RIFF" <size> "AVI ".
    if &data[..4] == b"RIFF" && &data[8..12] == b"AVI " {
        return "avi";
    }
    // MPEG-TS: 0x47 sync byte at offset 0 AND at offset 188 (and 376 if
    // we have the bytes). A single 0x47 appears routinely in random
    // payloads, so require two confirming hits before committing.
    if data[0] == 0x47
        && data.len() > 188
        && data[188] == 0x47
        && (data.len() <= 376 || data[376] == 0x47)
    {
        return "ts";
    }
    "unknown"
}

pub fn demux_mp4(data: &[u8]) -> Result<DemuxResult> {
    // Pre-pass to clamp any over-reported child box sizes (common
    // on iPhone-recorded MP4s where the legacy QuickTime `wave`
    // atom inside `mp4a` exposes child boxes whose advertised size
    // exceeds the parent's remaining payload). The sanitizer is
    // byte-identical on clean files, so this is safe to run
    // unconditionally — only malformed files mutate. See
    // `mp4_sanitize::sanitize_isobmff_box_sizes`.
    let sanitized = sanitize_isobmff_box_sizes(data);
    let data: &[u8] = &sanitized;
    let size = data.len() as u64;
    let cursor = Cursor::new(data);
    let reader = Mp4Reader::read_header(cursor, size).context("reading MP4 header")?;

    let video_track = reader
        .tracks()
        .values()
        .find(|t| t.track_type().ok() == Some(mp4::TrackType::Video))
        .context("no video track in MP4")?;

    let track_id = video_track.track_id();
    let codec_from_mp4 = format_codec(video_track);
    // mp4 0.14 has no av01 sample-entry support: tracks using AV1 come back
    // as "unknown" and the decoder factory would fail. Byte-scan stsd for
    // the av01 fourcc to recover the codec label. Sample iteration still
    // works because stco/stsc/stsz are read independently of the sample
    // entry; AV1-in-MP4 samples are raw OBU streams with no AVCC wrapping.
    let codec = if codec_from_mp4 == "unknown" && has_av01_sample_entry(data) {
        "av1".to_string()
    } else if codec_from_mp4 == "unknown" && hevc_sample_entry_fourcc(data).is_some() {
        // hvc1 sample entry — mp4 0.14 only parses hev1. Same length-
        // prefixed bitstream, different fourcc. We retrieve VPS/SPS/PPS
        // from the hvcC box via byte-scan (below) and convert samples
        // to Annex-B the same way as avc1.
        "h265".to_string()
    } else if codec_from_mp4 == "unknown" && prores_sample_entry_fourcc(data).is_some() {
        // Apple ProRes lives in MOV (which is ISOBMFF, same box tree as
        // MP4) under one of six fourccs — mp4 0.14 returns `unknown` for
        // all of them. Samples are stored as self-contained ProRes frames
        // with no AVCC-style length prefix, so stco/stsc/stsz iteration
        // already reads them correctly — we just need the codec label so
        // downstream decode (legacy-cpu-eng's lane) can dispatch.
        "prores".to_string()
    } else {
        codec_from_mp4
    };
    let width = video_track.width() as u32;
    let height = video_track.height() as u32;
    let sample_count = video_track.sample_count();
    let duration = video_track.duration().as_secs_f64();
    let frame_rate = mp4_frame_rate(video_track, duration);
    let bitrate = video_track.bitrate() as u64;

    // Squad-21: pull `mdcv` and `clli` boxes nested inside the visual
    // sample entry (`stsd > {av01, hvc1, hev1, ...}`) and surface them
    // to ColorMetadata so the muxer can round-trip them. These boxes
    // are an HDR10 / HDR10+ requirement — without them, Apple's player
    // (and many TVs) silently fall back to BT.709 limited even when
    // colr nclx says BT.2020.
    let mp4_color = extract_mp4_visual_color_metadata(data);
    let initial_color_metadata = ColorMetadata {
        mastering_display: mp4_color.mastering_display,
        content_light_level: mp4_color.content_light_level,
        ..Default::default()
    };

    let info = StreamInfo {
        codec: codec.clone(),
        width,
        height,
        frame_rate,
        duration,
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        total_frames: sample_count as u64,
        bitrate,
        // SDR defaults for primaries/transfer/matrix at demux layer —
        // those flow from the decoder's sequence_callback (NVDEC) or
        // SPS VUI parser (HEVC CPU). Mastering display + content
        // light level live in MP4 sample-entry boxes (extracted above)
        // so they CAN come from the demuxer directly.
        color_metadata: initial_color_metadata,
    };

    let cursor = Cursor::new(data);
    let mut reader = Mp4Reader::read_header(cursor, size).context("re-reading MP4 for samples")?;

    let mut samples = Vec::with_capacity(sample_count as usize);

    let needs_annexb = matches!(codec.as_str(), "h264" | "h265");
    // length_size defaults to 4 (the ISOBMFF near-universal pick); when
    // we can reach the avcC/hvcC box we override with the recorded value.
    // A length_size of 2 or even 1 is legal and has been observed in
    // streaming-profile MP4s.
    let (sps_pps, length_size) = if needs_annexb {
        if codec == "h264" {
            match extract_avc_config(data) {
                Some(cfg) => (cfg.parameter_sets, cfg.length_size),
                // mp4 0.14 successfully parsed the avcC high-level but we
                // couldn't recover length_size from the box bytes — fall
                // back to the crate's parsed SPS/PPS and assume 4-byte.
                None => (extract_sps_pps(&reader, track_id), 4u8),
            }
        } else {
            // h265: parse hvcC straight from the box bytes (mp4 0.14
            // doesn't surface either length_size or the hvcC arrays).
            match extract_hevc_config(data) {
                Some(cfg) => (cfg.parameter_sets, cfg.length_size),
                None => (Vec::new(), 4u8),
            }
        }
    } else {
        (Vec::new(), 4u8)
    };

    // Per-stream parameter-set emission tracker (#67/#68). Replaces the
    // older `prepend on sample_idx==1` heuristic, which mishandled
    // ExoPlayer open-GOP MP4s where sample 0 is `SPS + non-IDR slice`
    // and the first IRAP arrives later carrying only a slice NAL.
    // The tracker scans inline NAL types per sample and prepends only
    // the parameter-set kinds that are still missing on the first IRAP.
    let mut avc_tracker = if needs_annexb {
        Some(ParamSetTracker::new(if codec == "h264" {
            NaluCodec::Avc
        } else {
            NaluCodec::Hevc
        }))
    } else {
        None
    };

    for sample_idx in 1..=sample_count {
        let sample = reader
            .read_sample(track_id, sample_idx)
            .context("reading sample")?;

        if let Some(sample) = sample {
            let sample_data = sample.bytes.to_vec();

            if let Some(tracker) = avc_tracker.as_mut() {
                let annexb =
                    length_prefixed_to_annexb_tracked(&sample_data, length_size, tracker, &sps_pps);
                samples.push(annexb);
            } else {
                samples.push(sample_data);
            }
        }
    }

    // Replace the hard-coded yuv420p with a real sniff from the first
    // sample's sequence header. detect() is safe on short/malformed
    // data — falls back to Yuv420p.
    let detected_pf = codec::pixel_format::detect(&codec, &samples);
    let info = StreamInfo {
        pixel_format: detected_pf,
        ..info
    };

    let audio = extract_mp4_audio(data);

    Ok(DemuxResult {
        codec,
        info,
        samples,
        audio,
    })
}

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
fn extract_mp4_audio(data: &[u8]) -> Option<AudioTrack> {
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
        if let Some(frag) = build_fragmented_sample_table(data, track_id, 0, 0) {
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
    let moov = find_direct_child(data, b"moov")?;
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
    let stsd = find_box_body(trak, &[b"mdia", b"minf", b"stbl", b"stsd"])?;
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
            return find_direct_child(&stsd[child_start..end], cfg_fourcc).map(|b| b.to_vec());
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
fn ac3_sample_rate_channels_from_dac3(dac3: &[u8]) -> Option<(u32, u16)> {
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
fn eac3_sample_rate_channels_from_dec3(dec3: &[u8]) -> Option<(u32, u16)> {
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
    let moov = find_direct_child(data, b"moov")?;
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
    let stsd = find_box_body(trak, &[b"mdia", b"minf", b"stbl", b"stsd"])?;
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
            return find_direct_child(&stsd[child_start..end], b"dOps").map(|b| b.to_vec());
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
    let moov = find_direct_child(data, b"moov")?;
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

/// Quick "is this trak an audio trak?" check. ISO 14496-12 §8.4.5.3
/// requires `smhd` (Sound Media Header) inside `mdia/minf` for every
/// audio trak. Looking for it is a strictly stronger signal than
/// inspecting the first `stsd` entry's fourcc — it's positive evidence
/// of trak intent rather than fourcc-position guessing.
fn trak_is_audio(trak: &[u8]) -> bool {
    find_box_body(trak, &[b"mdia", b"minf", b"smhd"]).is_some()
}

fn extract_asc_from_trak(trak: &[u8]) -> Option<Vec<u8>> {
    let stsd = find_box_body(trak, &[b"mdia", b"minf", b"stbl", b"stsd"])?;
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
    let Some(moov) = find_direct_child(data, b"moov") else {
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
            if let Some(stsd) = find_box_body(trak_body, &[b"mdia", b"minf", b"stbl", b"stsd"])
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
            96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
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
    if chan_cfg == 0 {
        Some(2)
    } else {
        Some(chan_cfg)
    }
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

pub fn demux_mkv(data: &[u8]) -> Result<DemuxResult> {
    let cursor = Cursor::new(data);
    let mut mkv =
        MatroskaFile::open(cursor).map_err(|e| anyhow::anyhow!("reading MKV header: {e}"))?;

    // AVC/HEVC in MKV: CodecPrivate holds the avcC / hvcC configuration record
    // verbatim. Length-prefixed Block samples need the same Annex-B conversion
    // we do for MP4, plus VPS/SPS/PPS prepended to the first sample of the
    // track. VP8/VP9/AV1 are self-contained and skip this dance.
    //
    // Snapshot every field we need off TrackEntry before `next_frame` starts
    // mutating `mkv` below — TrackEntry borrows from `mkv` and hold times
    // conflict with the &mut self on `next_frame`.
    let (
        track_number,
        track_uid,
        codec_id,
        width,
        height,
        annexb_prepend,
        length_size,
        color_space,
        mut color_metadata,
        mut color_info,
        track_default_duration_ns,
    ) = {
        let track_info = mkv
            .tracks()
            .iter()
            .find(|t| t.track_type() == MkvTrackType::Video)
            .context("no video track in MKV")?;

        let track_number = track_info.track_number().get();
        let track_uid = track_info.track_uid().get();
        let codec_id = track_info.codec_id().to_string();
        // Per-track DefaultDuration (`0x23E383`, ns per frame) — Matroska's
        // canonical frame-rate hint. Used as the frame_rate fallback when the
        // segment's `Duration` element is absent (live-recorded MKVs and some
        // streaming WebMs ship without one). Squad-32: this fallback was
        // previously missing — frame_rate would silently default to 30.0
        // even when DefaultDuration cleanly described e.g. 23.976 / 60 fps.
        let default_duration_ns = track_info.default_duration().map(|d| d.get());

        // Parse avcC/hvcC CodecPrivate once to recover both the parameter
        // sets and the recorded length_size_minus_one — 4-byte prefixes
        // are the common case, but the spec allows 1 or 2 bytes.
        let (annexb_prepend, length_size): (Vec<Vec<u8>>, u8) = if codec_id == "V_MPEG4/ISO/AVC" {
            let priv_bytes = track_info
                .codec_private()
                .context("V_MPEG4/ISO/AVC CodecPrivate missing")?;
            let cfg = parse_avcc(priv_bytes).context("V_MPEG4/ISO/AVC CodecPrivate malformed")?;
            (cfg.parameter_sets, cfg.length_size)
        } else if codec_id == "V_MPEGH/ISO/HEVC" {
            let priv_bytes = track_info
                .codec_private()
                .context("V_MPEGH/ISO/HEVC CodecPrivate missing")?;
            let cfg = parse_hvcc(priv_bytes).context("V_MPEGH/ISO/HEVC CodecPrivate malformed")?;
            (cfg.parameter_sets, cfg.length_size)
        } else {
            (Vec::new(), 4)
        };

        if mkv_codec_needs_annexb(&codec_id) && annexb_prepend.is_empty() {
            bail!("AVC/HEVC MKV CodecPrivate missing or empty — no parameter sets to prepend");
        }

        let video = track_info
            .video()
            .context("video track missing Video element")?;
        let w = video.pixel_width().get() as u32;
        let h = video.pixel_height().get() as u32;

        // Parse the Colour element into a ColorMetadata + ColorSpace +
        // extended MkvColorInfo. Legacy MKVs without Colour produce the
        // SDR BT.709 default.
        let (color_space, color_metadata, color_info) = match video.colour() {
            Some(colour) => colour_to_pipeline(colour),
            None => (
                ColorSpace::Bt709,
                ColorMetadata::default(),
                MkvColorInfo::default(),
            ),
        };

        (
            track_number,
            track_uid,
            codec_id,
            w,
            h,
            annexb_prepend,
            length_size,
            color_space,
            color_metadata,
            color_info,
            default_duration_ns,
        )
    };

    // Squad-21: matroska-demuxer 0.7's `Colour::new` reads MaxCLL/MaxFALL from
    // the wrong ElementId offset (it actually reads MatrixCoefficients), and
    // `MasteringMetadata::new` reads each `_chromaticity_y` from the matching
    // `_chromaticity_x` ElementId — so all three primaries' y values come back
    // holding the corresponding x value. Re-scan the raw EBML bytes to recover
    // the canonical values; the same workaround already lives in
    // `probe_mkv_color_info`. We MUST also clear the unified
    // `ColorMetadata.content_light_level` and the mastering display y-fields
    // we synthesized from the poisoned typed accessors so a scan miss doesn't
    // leave the wrong value in place.
    color_info.max_cll = None;
    color_info.max_fall = None;
    color_metadata.content_light_level = None;
    if let Some(md) = color_metadata.mastering_display.as_mut() {
        // The y values are poisoned with the matching x values — clear them
        // in case the raw scan can't recover (defensive: leave 0 vs garbage).
        md.primaries_r_y = 0;
        md.primaries_g_y = 0;
        md.primaries_b_y = 0;
    }
    if let Some(local) = color_info.mastering.as_mut() {
        local.primary_r_chromaticity_y = None;
        local.primary_g_chromaticity_y = None;
        local.primary_b_chromaticity_y = None;
    }
    if let Some(fix) = scan_mkv_colour_raw(data) {
        color_info.max_cll = fix.max_cll;
        color_info.max_fall = fix.max_fall;
        if fix.max_cll.is_some() || fix.max_fall.is_some() {
            color_metadata.content_light_level = Some(ContentLightLevel {
                max_cll: fix.max_cll.unwrap_or(0).min(u16::MAX as u32) as u16,
                max_fall: fix.max_fall.unwrap_or(0).min(u16::MAX as u32) as u16,
            });
        }
        // Re-fold the recovered y-chromaticities (HEVC SEI D.2.28 wire
        // domain: 0.00002 increments → multiply by 50_000, saturate to u16).
        let chrom = |v: f64| (v * 50_000.0).round().clamp(0.0, u16::MAX as f64) as u16;
        if let Some(md) = color_metadata.mastering_display.as_mut() {
            if let Some(y) = fix.primary_r_chromaticity_y {
                md.primaries_r_y = chrom(y);
            }
            if let Some(y) = fix.primary_g_chromaticity_y {
                md.primaries_g_y = chrom(y);
            }
            if let Some(y) = fix.primary_b_chromaticity_y {
                md.primaries_b_y = chrom(y);
            }
        }
        if let Some(local) = color_info.mastering.as_mut() {
            if fix.primary_r_chromaticity_y.is_some() {
                local.primary_r_chromaticity_y = fix.primary_r_chromaticity_y;
            }
            if fix.primary_g_chromaticity_y.is_some() {
                local.primary_g_chromaticity_y = fix.primary_g_chromaticity_y;
            }
            if fix.primary_b_chromaticity_y.is_some() {
                local.primary_b_chromaticity_y = fix.primary_b_chromaticity_y;
            }
        }
    }

    let needs_annexb = mkv_codec_needs_annexb(&codec_id);
    let codec = match codec_id.as_str() {
        "V_VP9" => "vp9".to_string(),
        "V_VP8" => "vp8".to_string(),
        "V_AV1" => "av1".to_string(),
        "V_MPEG4/ISO/AVC" => "h264".to_string(),
        "V_MPEGH/ISO/HEVC" => "h265".to_string(),
        other => other.to_lowercase(),
    };

    let timestamp_scale = mkv.info().timestamp_scale().get();
    let duration_ticks = mkv.info().duration().unwrap_or(0.0);
    // timestamp_scale is in ns; duration is in ticks (float)
    let duration = duration_ticks * (timestamp_scale as f64) / 1_000_000_000.0;

    // Tag-based bitrate: preferred over the computed fallback when a
    // muxer wrote a `BIT_RATE` Matroska Tag scoped to our track UID.
    // See `bitrate_from_tags` for scope-resolution details.
    let tag_bitrate = mkv
        .tags()
        .and_then(|tags| bitrate_from_tags(tags, track_uid));
    // Emit the extended metadata we can't (yet) carry on `StreamInfo`
    // on a structured log line — downstream work-items #HDR10 and mux
    // SEI passthrough will read them via `probe_mkv_color_info`.
    if color_info != MkvColorInfo::default() {
        tracing::info!(
            bits_per_channel = ?color_info.bits_per_channel,
            max_cll = ?color_info.max_cll,
            max_fall = ?color_info.max_fall,
            mastering = ?color_info.mastering,
            "MKV Colour: parsed HDR-adjacent metadata"
        );
    }

    let mut samples: Vec<Vec<u8>> = Vec::new();
    let mut frame = MkvFrame::default();
    let mut total_video_bytes: u64 = 0;
    // Same per-stream tracker as the MP4 path. MKV's CodecPrivate carries
    // the avcC / hvcC bytes verbatim, so the same first-IRAP-prepend
    // heuristic applies (and is more robust than the old
    // `is_first_video_sample` flag, which assumed sample 0 was always IRAP).
    let mut mkv_tracker = if needs_annexb {
        Some(ParamSetTracker::new(if codec_id == "V_MPEG4/ISO/AVC" {
            NaluCodec::Avc
        } else {
            NaluCodec::Hevc
        }))
    } else {
        None
    };
    loop {
        match mkv.next_frame(&mut frame) {
            Ok(true) => {
                if frame.track == track_number {
                    let raw = std::mem::take(&mut frame.data);
                    total_video_bytes += raw.len() as u64;
                    if let Some(tracker) = mkv_tracker.as_mut() {
                        let annexb = length_prefixed_to_annexb_tracked(
                            &raw,
                            length_size,
                            tracker,
                            &annexb_prepend,
                        );
                        samples.push(annexb);
                    } else {
                        samples.push(raw);
                    }
                }
            }
            Ok(false) => break,
            Err(e) => bail!("MKV frame read error: {e}"),
        }
    }

    let total_frames = samples.len() as u64;
    // Frame rate fallback chain (Squad-32):
    //   1. samples / segment_duration  (most accurate when both are known)
    //   2. 1 / DefaultDuration          (Matroska's canonical per-frame ns)
    //   3. 30.0                         (last-resort sentinel)
    let frame_rate = if duration > 0.0 {
        total_frames as f64 / duration
    } else if let Some(dd_ns) = track_default_duration_ns.filter(|n| *n > 0) {
        1_000_000_000.0 / dd_ns as f64
    } else {
        30.0
    };

    let detected_pf = codec::pixel_format::detect(&codec, &samples);

    // Bitrate priority: Tag `BIT_RATE` if present → summed sample bytes
    // over the segment duration. Never 0 unless the file has no samples
    // AND no tag (in which case bitrate is genuinely unknowable and we
    // keep the historical 0 sentinel).
    let bitrate = match tag_bitrate {
        Some(b) if b > 0 => b,
        _ => {
            if duration > 0.0 && total_video_bytes > 0 {
                ((total_video_bytes as f64 * 8.0) / duration) as u64
            } else {
                0
            }
        }
    };

    let info = StreamInfo {
        codec: codec.clone(),
        width,
        height,
        frame_rate,
        duration,
        pixel_format: detected_pf,
        color_space,
        total_frames,
        bitrate,
        color_metadata,
    };

    // Audio passthrough uses its own MatroskaFile handle (re-opened) since
    // next_frame above already consumed the stream.
    let audio = extract_mkv_audio(data);

    Ok(DemuxResult {
        codec,
        info,
        samples,
        audio,
    })
}

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
fn extract_mkv_audio(data: &[u8]) -> Option<AudioTrack> {
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
                    let dur_ticks = ((dur_ns as u128) * (timescale as u128) / 1_000_000_000) as u32;
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

/// True for MKV CodecIDs whose samples are length-prefixed (AVCC/HVCC) and
/// require SPS/PPS pulled from the track's CodecPrivate to feed a decoder
/// that expects Annex-B. demux_mkv bails on these until the Annex-B path is
/// wired — currently only VP8/VP9/AV1 are safe through MKV.
fn mkv_codec_needs_annexb(codec_id: &str) -> bool {
    matches!(codec_id, "V_MPEG4/ISO/AVC" | "V_MPEGH/ISO/HEVC")
}

/// Walk the ISOBMFF box tree looking for an `av01` sample entry inside
/// `moov/trak/mdia/minf/stbl/stsd`. Returns true if found at the expected
/// nesting level. Doing a full tree walk (vs naive byte-search for "av01")
/// avoids false positives from sample data in mdat that happens to contain
/// those bytes.
/// Find the HEVC sample-entry fourcc (`hvc1`, `hev1`, `hvc2`, `hev2`,
/// `dvh1`, `dvhe`) in the video track's stsd box. Returns the 4-byte
/// fourcc or None. Used as the mp4 0.14 crate detection fallback —
/// its `media_type()` only returns H265 for `hev1`, so `hvc1` (the
/// Jellyfin corpus's HEVC flavor) needs this path.
fn hevc_sample_entry_fourcc(data: &[u8]) -> Option<[u8; 4]> {
    let path: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"];
    let stsd_body = find_box_body(data, path)?;
    if stsd_body.len() < 16 {
        return None;
    }
    let mut pos = 8; // skip version/flags/entry_count
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        let entry_type: [u8; 4] = stsd_body[pos + 4..pos + 8].try_into().ok()?;
        match &entry_type {
            b"hvc1" | b"hev1" | b"hvc2" | b"hev2" | b"dvh1" | b"dvhe" => {
                return Some(entry_type);
            }
            _ => {}
        }
        if entry_size == 0 {
            break;
        }
        pos = pos.saturating_add(entry_size);
    }
    None
}

/// Look for an Apple ProRes sample entry in the video track's stsd box.
/// Six fourccs cover the product family:
///   apcn = ProRes 422 Standard    apch = ProRes 422 HQ
///   apcs = ProRes 422 LT          apco = ProRes 422 Proxy
///   ap4h = ProRes 4444            ap4x = ProRes 4444 XQ
/// All share the same container layout (self-contained frame samples, no
/// length-prefix wrapping), so from demux's perspective they are
/// interchangeable — we return the first one we see so callers can log
/// which specific profile the input used. Decode dispatch uses the
/// unified `"prores"` codec label produced by `demux_mp4`.
fn prores_sample_entry_fourcc(data: &[u8]) -> Option<[u8; 4]> {
    let path: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"];
    let stsd_body = find_box_body(data, path)?;
    if stsd_body.len() < 16 {
        return None;
    }
    let mut pos = 8;
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        let entry_type: [u8; 4] = stsd_body[pos + 4..pos + 8].try_into().ok()?;
        match &entry_type {
            b"apcn" | b"apch" | b"apcs" | b"apco" | b"ap4h" | b"ap4x" => {
                return Some(entry_type);
            }
            _ => {}
        }
        if entry_size == 0 {
            break;
        }
        pos = pos.saturating_add(entry_size);
    }
    None
}

/// Find the AVC sample entry in MP4 and return its parsed avcC config
/// (length_size + SPS/PPS NAL units). Returns None when no `avc1`/`avc3`
/// sample entry is present or the avcC box is malformed.
fn extract_avc_config(data: &[u8]) -> Option<AvcConfig> {
    let path: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"];
    let stsd_body = find_box_body(data, path)?;
    if stsd_body.len() < 16 {
        return None;
    }

    let mut pos = 8;
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        let entry_type = &stsd_body[pos + 4..pos + 8];
        let is_avc = matches!(entry_type, b"avc1" | b"avc3");
        if !is_avc {
            if entry_size == 0 {
                break;
            }
            pos = pos.saturating_add(entry_size);
            continue;
        }
        let end = pos.saturating_add(entry_size);
        if end > stsd_body.len() {
            return None;
        }
        let child_start = pos + 8 + 78; // VisualSampleEntry fixed header
        if child_start >= end {
            return None;
        }
        let avcc = find_direct_child(&stsd_body[child_start..end], b"avcC")?;
        return parse_avcc(avcc);
    }
    None
}

/// HDR static metadata pulled from the visual sample entry's `mdcv` and
/// `clli` boxes — Squad-21 wires this to ColorMetadata so Squad-20's
/// muxer can round-trip HDR10 mastering display + content light level
/// from any source MP4 / MOV that signals them.
#[derive(Debug, Default, Clone, Copy)]
struct Mp4VisualColorMetadata {
    mastering_display: Option<MasteringDisplay>,
    content_light_level: Option<ContentLightLevel>,
}

/// Walk `moov/trak/mdia/minf/stbl/stsd > {av01, hvc1, hev1, ...}` and
/// pick out the optional `mdcv` and `clli` child boxes.
///
/// Per ISO/IEC 23001-17 (Carriage of static and dynamic metadata in
/// ISOBMFF), `mdcv` and `clli` are direct children of the visual
/// sample entry — same nesting level as `colr`. Layouts:
///
///   `mdcv` body (24 bytes):
///     u16[2] display_primaries[3]   // wire order GBR
///     u16    white_point_x
///     u16    white_point_y
///     u32    max_display_mastering_luminance  (in 0.0001 cd/m²)
///     u32    min_display_mastering_luminance  (in 0.0001 cd/m²)
///
///   `clli` body (4 bytes):
///     u16    max_content_light_level
///     u16    max_pic_average_light_level
fn extract_mp4_visual_color_metadata(data: &[u8]) -> Mp4VisualColorMetadata {
    let path: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"];
    let Some(stsd_body) = find_box_body(data, path) else {
        return Mp4VisualColorMetadata::default();
    };
    if stsd_body.len() < 16 {
        return Mp4VisualColorMetadata::default();
    }

    let mut pos = 8; // skip version/flags/entry_count
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        if entry_size < 8 || pos.saturating_add(entry_size) > stsd_body.len() {
            break;
        }
        let entry_type: [u8; 4] = match stsd_body[pos + 4..pos + 8].try_into() {
            Ok(v) => v,
            Err(_) => break,
        };
        // Visual sample entries — mdcv/clli only live under these.
        let is_visual = matches!(
            &entry_type,
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
                | b"apcn"
                | b"apch"
                | b"apcs"
                | b"apco"
                | b"ap4h"
                | b"ap4x"
        );
        if !is_visual {
            pos = pos.saturating_add(entry_size);
            continue;
        }
        let end = pos.saturating_add(entry_size);
        // VisualSampleEntry header: 8-byte box header + 78 bytes of fixed
        // VisualSampleEntry fields before the first child box. Same
        // offset for every visual sample entry kind.
        let child_start = pos + 8 + 78;
        if child_start >= end {
            return Mp4VisualColorMetadata::default();
        }
        let children = &stsd_body[child_start..end];
        let mut out = Mp4VisualColorMetadata::default();
        if let Some(mdcv) = find_direct_child(children, b"mdcv") {
            out.mastering_display = parse_mp4_mdcv(mdcv);
        }
        if let Some(clli) = find_direct_child(children, b"clli") {
            out.content_light_level = parse_mp4_clli(clli);
        }
        return out;
    }
    Mp4VisualColorMetadata::default()
}

fn parse_mp4_mdcv(body: &[u8]) -> Option<MasteringDisplay> {
    if body.len() < 24 {
        return None;
    }
    let u16be = |o: usize| u16::from_be_bytes([body[o], body[o + 1]]);
    let u32be = |o: usize| u32::from_be_bytes([body[o], body[o + 1], body[o + 2], body[o + 3]]);
    Some(MasteringDisplay {
        // Wire order is GBR per ISO/IEC 23001-17 §7.3.
        primaries_g_x: u16be(0),
        primaries_g_y: u16be(2),
        primaries_b_x: u16be(4),
        primaries_b_y: u16be(6),
        primaries_r_x: u16be(8),
        primaries_r_y: u16be(10),
        white_point_x: u16be(12),
        white_point_y: u16be(14),
        max_luminance: u32be(16),
        min_luminance: u32be(20),
    })
}

fn parse_mp4_clli(body: &[u8]) -> Option<ContentLightLevel> {
    if body.len() < 4 {
        return None;
    }
    Some(ContentLightLevel {
        max_cll: u16::from_be_bytes([body[0], body[1]]),
        max_fall: u16::from_be_bytes([body[2], body[3]]),
    })
}

/// Find the HEVC sample entry in MP4 and return its parsed hvcC config
/// (length_size + VPS/SPS/PPS NAL units in recorded order).
fn extract_hevc_config(data: &[u8]) -> Option<HevcConfig> {
    let path: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"];
    let stsd_body = find_box_body(data, path)?;
    if stsd_body.len() < 16 {
        return None;
    }

    let mut pos = 8;
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        let entry_type = &stsd_body[pos + 4..pos + 8];
        let is_hevc = matches!(
            entry_type,
            b"hvc1" | b"hev1" | b"hvc2" | b"hev2" | b"dvh1" | b"dvhe"
        );
        if !is_hevc {
            if entry_size == 0 {
                break;
            }
            pos = pos.saturating_add(entry_size);
            continue;
        }
        let end = pos.saturating_add(entry_size);
        if end > stsd_body.len() {
            return None;
        }
        let child_start = pos + 8 + 78;
        if child_start >= end {
            return None;
        }
        let hvcc = find_direct_child(&stsd_body[child_start..end], b"hvcC")?;
        return parse_hvcc(hvcc);
    }
    None
}

/// Extract VPS/SPS/PPS NAL units from the `hvcC` config box nested
/// under the HEVC sample entry. The hvcC layout (ISO/IEC 14496-15
/// §8.3.3) puts parameter-set arrays at offset 22, each array as:
/// `array_type u8 | num_nalus u16 BE | [{nalu_len u16 BE, nalu ...}]`.
#[allow(dead_code)]
fn extract_hevc_parameter_sets(data: &[u8]) -> Vec<Vec<u8>> {
    let path: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"];
    let Some(stsd_body) = find_box_body(data, path) else {
        return Vec::new();
    };
    if stsd_body.len() < 16 {
        return Vec::new();
    }

    // Walk the stsd entries, find the HEVC sample entry.
    let mut pos = 8;
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        let entry_type = &stsd_body[pos + 4..pos + 8];
        let is_hevc = matches!(
            entry_type,
            b"hvc1" | b"hev1" | b"hvc2" | b"hev2" | b"dvh1" | b"dvhe"
        );
        if !is_hevc {
            if entry_size == 0 {
                break;
            }
            pos = pos.saturating_add(entry_size);
            continue;
        }

        let end = pos.saturating_add(entry_size);
        if end > stsd_body.len() {
            return Vec::new();
        }
        let entry_body_start = pos + 8;
        // VisualSampleEntry header: 78 bytes between fourcc and first
        // child box (6 reserved + 2 data_ref_idx + 16 pre_defined +
        // 2 width + 2 height + 4x2 resolutions + 4 reserved + 2 frame_count
        // + 32 compressor_name + 2 depth + 2 pre_defined).
        let child_start = entry_body_start + 78;
        if child_start >= end {
            return Vec::new();
        }
        let child_area = &stsd_body[child_start..end];
        let hvcc = match find_direct_child(child_area, b"hvcC") {
            Some(b) => b,
            None => return Vec::new(),
        };
        return parse_hvcc_param_sets(hvcc);
    }
    Vec::new()
}

/// Parse the H.264 AVCDecoderConfigurationRecord (avcC body) to extract
/// SPS and PPS NALU payloads. Layout (ISO/IEC 14496-15 §5.3.3.1):
///   u8  configurationVersion = 1
///   u8  AVCProfileIndication
///   u8  profile_compatibility
///   u8  AVCLevelIndication
///   u8  reserved(6)|lengthSizeMinusOne(2)
///   u8  reserved(3)|numOfSequenceParameterSets(5)
///   // per SPS: u16 nalUnitLength, u8[nalUnitLength] nalUnit
///   u8  numOfPictureParameterSets
///   // per PPS: u16 nalUnitLength, u8[nalUnitLength] nalUnit
/// Used for MKV `V_MPEG4/ISO/AVC` where CodecPrivate is the verbatim avcC body.
///
/// Kept as a back-compat alias over `crate::annexb::parse_avcc` — callers
/// should prefer the new parser that also returns `length_size`.
#[allow(dead_code)]
fn parse_avcc_param_sets(avcc: &[u8]) -> Vec<Vec<u8>> {
    if avcc.len() < 7 {
        return Vec::new();
    }
    let num_sps = (avcc[5] & 0x1F) as usize;
    let mut out = Vec::new();
    let mut cur = 6;
    for _ in 0..num_sps {
        if cur + 2 > avcc.len() {
            return out;
        }
        let nalu_len = u16::from_be_bytes([avcc[cur], avcc[cur + 1]]) as usize;
        cur += 2;
        if cur + nalu_len > avcc.len() {
            return out;
        }
        out.push(avcc[cur..cur + nalu_len].to_vec());
        cur += nalu_len;
    }
    if cur >= avcc.len() {
        return out;
    }
    let num_pps = avcc[cur] as usize;
    cur += 1;
    for _ in 0..num_pps {
        if cur + 2 > avcc.len() {
            return out;
        }
        let nalu_len = u16::from_be_bytes([avcc[cur], avcc[cur + 1]]) as usize;
        cur += 2;
        if cur + nalu_len > avcc.len() {
            return out;
        }
        out.push(avcc[cur..cur + nalu_len].to_vec());
        cur += nalu_len;
    }
    out
}

#[allow(dead_code)]
fn parse_hvcc_param_sets(hvcc: &[u8]) -> Vec<Vec<u8>> {
    // HEVCDecoderConfigurationRecord:
    //   u8  configurationVersion = 1
    //   u8  general_profile_space(2)|tier(1)|profile_idc(5)
    //   u32 general_profile_compatibility_flags
    //   u48 general_constraint_indicator_flags
    //   u8  general_level_idc
    //   u16 reserved(4)|min_spatial_segmentation_idc(12)
    //   u8  reserved(6)|parallelismType(2)
    //   u8  reserved(6)|chroma_format_idc(2)
    //   u8  reserved(5)|bit_depth_luma_minus8(3)
    //   u8  reserved(5)|bit_depth_chroma_minus8(3)
    //   u16 avgFrameRate
    //   u8  constantFrameRate(2)|numTemporalLayers(3)|temporalIdNested(1)|lengthSizeMinusOne(2)
    //   u8  numOfArrays
    //   // per array:
    //   //   u8  array_completeness(1)|reserved(1)|NAL_unit_type(6)
    //   //   u16 numNalus
    //   //   // per nalu:  u16 nalUnitLength, u8[nalUnitLength] nalUnit
    if hvcc.len() < 23 {
        return Vec::new();
    }
    let num_arrays = hvcc[22] as usize;
    let mut out = Vec::new();
    let mut cur = 23;
    for _ in 0..num_arrays {
        if cur + 3 > hvcc.len() {
            break;
        }
        let _array_hdr = hvcc[cur];
        let num_nalus = u16::from_be_bytes([hvcc[cur + 1], hvcc[cur + 2]]) as usize;
        cur += 3;
        for _ in 0..num_nalus {
            if cur + 2 > hvcc.len() {
                return out;
            }
            let nalu_len = u16::from_be_bytes([hvcc[cur], hvcc[cur + 1]]) as usize;
            cur += 2;
            if cur + nalu_len > hvcc.len() {
                return out;
            }
            out.push(hvcc[cur..cur + nalu_len].to_vec());
            cur += nalu_len;
        }
    }
    out
}

fn has_av01_sample_entry(data: &[u8]) -> bool {
    let path: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"];
    let stsd_body = match find_box_body(data, path) {
        Some(b) => b,
        None => return false,
    };
    // stsd: 1 byte version + 3 flags + 4 entry_count + [box header { size u32, type [u8;4] }...]
    if stsd_body.len() < 16 {
        return false;
    }
    let mut pos = 8; // skip version/flags/entry_count
    while pos + 8 <= stsd_body.len() {
        let entry_type = &stsd_body[pos + 4..pos + 8];
        if entry_type == b"av01" {
            return true;
        }
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        if entry_size == 0 {
            break;
        }
        pos = pos.saturating_add(entry_size);
    }
    false
}

/// Follow a box type path from `data` (top level) down and return the body
/// bytes (payload, excluding the 8-byte box header) of the last box in the
/// path, or None if any hop is missing. Handles 32-bit box sizes only —
/// adequate for moov/trak/stsd which are ~KB in practice.
fn find_box_body<'a>(data: &'a [u8], path: &[&[u8; 4]]) -> Option<&'a [u8]> {
    let mut slice = data;
    for (i, target) in path.iter().enumerate() {
        let found = find_direct_child(slice, target)?;
        if i + 1 == path.len() {
            return Some(found);
        }
        slice = found;
    }
    None
}

fn find_direct_child<'a>(data: &'a [u8], target: &[u8; 4]) -> Option<&'a [u8]> {
    let mut pos = 0;
    while pos + 8 <= data.len() {
        let size =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let btype = &data[pos + 4..pos + 8];
        if size < 8 || pos.checked_add(size).is_none_or(|end| end > data.len()) {
            return None;
        }
        if btype == target {
            return Some(&data[pos + 8..pos + size]);
        }
        pos += size;
    }
    None
}

/// Exact video frame rate for an MP4 track. For constant-frame-rate sources —
/// a single `stts` entry — this is `timescale / sample_delta`, which is exact.
/// The naive `sample_count / mdhd.duration` is not: ffmpeg pads `mdhd.duration`
/// by one frame, so a clean 30 fps source reads as 29.9 (300 / 10.033). VFR
/// (multi-entry stts) or a missing/zero stts falls back to that average.
fn mp4_frame_rate(track: &mp4::Mp4Track, duration: f64) -> f64 {
    let stts = &track.trak.mdia.minf.stbl.stts;
    if stts.entries.len() == 1 && stts.entries[0].sample_delta > 0 {
        return track.timescale() as f64 / stts.entries[0].sample_delta as f64;
    }
    if duration > 0.0 {
        track.sample_count() as f64 / duration
    } else {
        30.0
    }
}

fn format_codec(track: &mp4::Mp4Track) -> String {
    match track.media_type() {
        Ok(mp4::MediaType::H264) => "h264".into(),
        Ok(mp4::MediaType::H265) => "h265".into(),
        Ok(mp4::MediaType::VP9) => "vp9".into(),
        _ => "unknown".into(),
    }
}

fn extract_sps_pps(reader: &Mp4Reader<Cursor<&[u8]>>, track_id: u32) -> Vec<Vec<u8>> {
    let mut nalus = Vec::new();
    if let Some(track) = reader.tracks().get(&track_id)
        && let Some(ref avc1) = track.trak.mdia.minf.stbl.stsd.avc1
    {
        for sps in &avc1.avcc.sequence_parameter_sets {
            nalus.push(sps.bytes.to_vec());
        }
        for pps in &avc1.avcc.picture_parameter_sets {
            nalus.push(pps.bytes.to_vec());
        }
    }
    nalus
}

// `avcc_to_annexb` was removed when MP4 and MKV paths converged on the
// shared `crate::annexb::length_prefixed_to_annexb` helper, which also
// honors non-4-byte length prefixes recorded in `lengthSizeMinusOne`.

/// Map a Matroska `Colour` element into our pipeline's color-space,
/// per-H.273 `ColorMetadata`, and extended `MkvColorInfo`. Unspecified
/// sub-elements default to the SDR BT.709 baseline so decoders that
/// never read a Colour element keep behaving exactly as before.
fn colour_to_pipeline(colour: &MkvColour) -> (ColorSpace, ColorMetadata, MkvColorInfo) {
    let matrix_u8 = colour
        .matrix_coefficients()
        .map(matrix_coefficients_to_h273);
    let primaries_u8 = colour.primaries().map(primaries_to_h273);
    let transfer_u8 = colour.transfer_characteristics().map(transfer_to_h273);
    let range = colour.range();

    let color_space = match colour.matrix_coefficients() {
        Some(MatrixCoefficients::Bt709) => ColorSpace::Bt709,
        Some(MatrixCoefficients::Bt470bg) | Some(MatrixCoefficients::Smpte170) => ColorSpace::Bt601,
        Some(MatrixCoefficients::Bt2020Ncl)
        | Some(MatrixCoefficients::Bt2020Cl)
        | Some(MatrixCoefficients::Bt2100) => ColorSpace::Bt2020,
        _ => ColorSpace::Bt709,
    };

    let mastering = colour.mastering_metadata().map(mkv_mastering_to_local);
    let mkv_max_cll = colour.max_cll().and_then(|v| u32::try_from(v).ok());
    let mkv_max_fall = colour.max_fall().and_then(|v| u32::try_from(v).ok());

    // Squad-21: also synthesize the unified ColorMetadata HDR fields from
    // the MKV `MasteringMetadata` + `MaxCLL` / `MaxFALL` so the muxer
    // (Squad-20) can write `mdcv`/`clli` without re-reading the
    // MKV-specific MkvColorInfo struct. matroska-demuxer 0.7's MaxCLL/
    // MaxFALL bug (see `probe_mkv_color_info`) means the values here
    // come from the typed accessor — for the canonical scan we re-read
    // raw bytes in `probe_mkv_color_info`. The two paths agree on
    // well-formed MKVs and disagree only on malformed ones (where the
    // raw scan wins). Pipeline plumbs the raw-scan path for MKV.
    let unified_mastering = mastering.as_ref().and_then(mkv_mastering_to_unified);
    let unified_cll = match (mkv_max_cll, mkv_max_fall) {
        (None, None) => None,
        (cll, fall) => Some(ContentLightLevel {
            max_cll: cll.unwrap_or(0).min(u16::MAX as u32) as u16,
            max_fall: fall.unwrap_or(0).min(u16::MAX as u32) as u16,
        }),
    };

    let color_metadata = ColorMetadata {
        transfer: transfer_u8.map(TransferFn::from_h273).unwrap_or_default(),
        matrix_coefficients: matrix_u8.unwrap_or(1),
        colour_primaries: primaries_u8.unwrap_or(1),
        // H.273 full_range_flag: Matroska Range=2 (Full) sets it; any
        // other value (Broadcast, Defined, Unknown) keeps the studio
        // 16..235 default.
        full_range: matches!(range, Some(MkvRange::Full)),
        // Squad-21 wires MKV float chromaticities + max_cll/fall into
        // the H.265-spec u16 encoding via `mkv_mastering_to_unified` and
        // the f64 → cd/m² conversion above (also recovers around two
        // matroska-demuxer 0.7 bugs that misread MaxCLL/MaxFALL and y
        // chromaticities at the wrong ElementIds).
        mastering_display: unified_mastering,
        content_light_level: unified_cll,
    };

    let extra = MkvColorInfo {
        bits_per_channel: colour.bits_per_channel().and_then(|v| u8::try_from(v).ok()),
        chroma_subsampling_horz: colour
            .chroma_subsampling_horz()
            .and_then(|v| u8::try_from(v).ok()),
        chroma_subsampling_vert: colour
            .chroma_subsampling_vert()
            .and_then(|v| u8::try_from(v).ok()),
        chroma_siting_horz: colour.chroma_sitting_horz().map(chroma_siting_horz_to_u8),
        chroma_siting_vert: colour.chroma_sitting_vert().map(chroma_siting_vert_to_u8),
        max_cll: mkv_max_cll,
        max_fall: mkv_max_fall,
        mastering,
    };

    (color_space, color_metadata, extra)
}

/// Convert the Matroska f64 chromaticities (range 0..=1) and luminance
/// (cd/m²) into the integer encoding the unified `MasteringDisplay`
/// uses (HEVC SEI D.2.28 wire format). Returns `None` when no
/// sub-element of the MasteringMetadata was populated.
fn mkv_mastering_to_unified(m: &MkvMasteringMetadata) -> Option<MasteringDisplay> {
    if m.primary_r_chromaticity_x.is_none()
        && m.primary_g_chromaticity_x.is_none()
        && m.primary_b_chromaticity_x.is_none()
        && m.white_point_chromaticity_x.is_none()
        && m.luminance_max.is_none()
        && m.luminance_min.is_none()
    {
        return None;
    }
    let chrom = |v: Option<f64>| -> u16 {
        // 0.00002 increments per HEVC SEI D.2.28 — map [0.0, ~1.31)
        // into a u16 with saturation.
        let scaled = (v.unwrap_or(0.0) * 50_000.0).round();
        scaled.clamp(0.0, u16::MAX as f64) as u16
    };
    let max_lum = (m.luminance_max.unwrap_or(0.0) * 10_000.0).round();
    let min_lum = (m.luminance_min.unwrap_or(0.0) * 10_000.0).round();
    Some(MasteringDisplay {
        primaries_r_x: chrom(m.primary_r_chromaticity_x),
        primaries_r_y: chrom(m.primary_r_chromaticity_y),
        primaries_g_x: chrom(m.primary_g_chromaticity_x),
        primaries_g_y: chrom(m.primary_g_chromaticity_y),
        primaries_b_x: chrom(m.primary_b_chromaticity_x),
        primaries_b_y: chrom(m.primary_b_chromaticity_y),
        white_point_x: chrom(m.white_point_chromaticity_x),
        white_point_y: chrom(m.white_point_chromaticity_y),
        max_luminance: max_lum.clamp(0.0, u32::MAX as f64) as u32,
        min_luminance: min_lum.clamp(0.0, u32::MAX as f64) as u32,
    })
}

fn mkv_mastering_to_local(m: &MkvMastering) -> MkvMasteringMetadata {
    MkvMasteringMetadata {
        primary_r_chromaticity_x: m.primary_r_chromaticity_x(),
        primary_r_chromaticity_y: m.primary_r_chromaticity_y(),
        primary_g_chromaticity_x: m.primary_g_chromaticity_x(),
        primary_g_chromaticity_y: m.primary_g_chromaticity_y(),
        primary_b_chromaticity_x: m.primary_b_chromaticity_x(),
        primary_b_chromaticity_y: m.primary_b_chromaticity_y(),
        white_point_chromaticity_x: m.white_point_chromaticity_x(),
        white_point_chromaticity_y: m.white_point_chromaticity_y(),
        luminance_max: m.luminance_max(),
        luminance_min: m.luminance_min(),
    }
}

/// MatroskaElement MatrixCoefficients (0x55B1) uses the H.273 numbering
/// 1:1, but the `matroska-demuxer` enum hides the raw u8. Reverse the
/// mapping so downstream (mux `colr nclx`, nvenc encode params) can
/// write the original numeric value back out without re-deriving it.
fn matrix_coefficients_to_h273(m: MatrixCoefficients) -> u8 {
    match m {
        MatrixCoefficients::Identity => 0,
        MatrixCoefficients::Bt709 => 1,
        MatrixCoefficients::Fcc73682 => 4,
        MatrixCoefficients::Bt470bg => 5,
        MatrixCoefficients::Smpte170 => 6,
        MatrixCoefficients::Smpte240 => 7,
        MatrixCoefficients::YCoCg => 8,
        MatrixCoefficients::Bt2020Ncl => 9,
        MatrixCoefficients::Bt2020Cl => 10,
        MatrixCoefficients::SmpteSt2085 => 11,
        MatrixCoefficients::ChromaDerivedNcl => 12,
        MatrixCoefficients::ChromaDerivedCl => 13,
        MatrixCoefficients::Bt2100 => 14,
        MatrixCoefficients::Unknown => 2, // H.273 "unspecified"
    }
}

fn transfer_to_h273(t: TransferCharacteristics) -> u8 {
    match t {
        TransferCharacteristics::Bt709 => 1,
        TransferCharacteristics::Bt407m => 4,
        TransferCharacteristics::Bt407bg => 5,
        TransferCharacteristics::Smpte170 => 6,
        TransferCharacteristics::Smpte240 => 7,
        TransferCharacteristics::Linear => 8,
        TransferCharacteristics::Log => 9,
        TransferCharacteristics::LogSqrt => 10,
        TransferCharacteristics::Iec61966_2_4 => 11,
        TransferCharacteristics::Bt1361 => 12,
        TransferCharacteristics::Iec61966_2_1 => 13,
        TransferCharacteristics::Bt220_10 => 14,
        TransferCharacteristics::Bt220_12 => 15,
        TransferCharacteristics::Bt2100 => 16,
        TransferCharacteristics::SmpteSt428_1 => 17,
        TransferCharacteristics::Hlg => 18,
        TransferCharacteristics::Unknown => 2,
    }
}

fn primaries_to_h273(p: Primaries) -> u8 {
    match p {
        Primaries::Bt709 => 1,
        Primaries::Bt470m => 4,
        Primaries::Bt601 => 5,
        Primaries::Smpte170 => 6,
        Primaries::Smpte240 => 7,
        Primaries::Film => 8,
        Primaries::Bt2020 => 9,
        Primaries::SmpteSt428_1 => 10,
        Primaries::SmpteRp432_2 => 11,
        Primaries::SmpteEg432_2 => 12,
        Primaries::JedecP22 => 22,
        Primaries::Unknown => 2,
    }
}

fn chroma_siting_horz_to_u8(s: matroska_demuxer::ChromaSitingHorz) -> u8 {
    match s {
        matroska_demuxer::ChromaSitingHorz::LeftCollated => 1,
        matroska_demuxer::ChromaSitingHorz::Half => 2,
        matroska_demuxer::ChromaSitingHorz::Unknown => 0,
    }
}

fn chroma_siting_vert_to_u8(s: matroska_demuxer::ChromaSitingVert) -> u8 {
    match s {
        matroska_demuxer::ChromaSitingVert::LeftCollated => 1,
        matroska_demuxer::ChromaSitingVert::Half => 2,
        matroska_demuxer::ChromaSitingVert::Unknown => 0,
    }
}

/// Resolve a track-scoped `BIT_RATE` Matroska Tag to a bits-per-second
/// value. Matroska's tag-scoping rules (spec §"Tagging") say: a Tag
/// applies to the track whose `TagTrackUID` matches, or to every track
/// in the segment if `TagTrackUID` is absent or 0. We prefer an exact
/// UID match, fall back to a segment-wide tag when no per-track value
/// exists.
///
/// `BIT_RATE` is the canonical Matroska target tag name (FFmpeg writes
/// it; the MKVToolNix matrix documents it). Some encoders emit
/// `BPS` / `BPS-eng` instead — we accept both for robustness. Values
/// are strings of base-10 digits in bits per second.
fn bitrate_from_tags(tags: &[matroska_demuxer::Tag], track_uid: u64) -> Option<u64> {
    let matches_track = |tag: &matroska_demuxer::Tag| -> bool {
        match tag.targets() {
            None => true, // Segment-wide — applies to all tracks.
            Some(t) => match t.tag_track_uid() {
                None | Some(0) => true,
                Some(uid) => uid == track_uid,
            },
        }
    };
    let mut segment_wide: Option<u64> = None;
    let mut track_scoped: Option<u64> = None;
    for tag in tags {
        if !matches_track(tag) {
            continue;
        }
        for st in tag.simple_tags() {
            let name = st.name();
            let is_bitrate = name.eq_ignore_ascii_case("BIT_RATE")
                || name.eq_ignore_ascii_case("BPS")
                || name.to_ascii_uppercase().starts_with("BPS-");
            if !is_bitrate {
                continue;
            }
            let Some(val) = st.string() else {
                continue;
            };
            let Ok(parsed) = val.trim().parse::<u64>() else {
                continue;
            };
            let is_track_scoped = tag
                .targets()
                .and_then(|t| t.tag_track_uid())
                .map(|uid| uid == track_uid)
                .unwrap_or(false);
            if is_track_scoped {
                track_scoped = Some(parsed);
            } else if segment_wide.is_none() {
                segment_wide = Some(parsed);
            }
        }
    }
    track_scoped.or(segment_wide)
}

/// Re-open an MKV container solely to extract the extended Colour
/// sub-elements that don't fit on `StreamInfo.color_metadata`
/// (MaxCLL / MaxFALL / SMPTE-2086 mastering primaries / bits_per_channel /
/// chroma siting). Intended for downstream paths that need HDR10 side
/// data for muxing; returns `None` when the file has no video track,
/// no `Colour` element, or isn't a well-formed MKV.
pub fn probe_mkv_color_info(data: &[u8]) -> Option<MkvColorInfo> {
    let cursor = Cursor::new(data);
    let mkv = MatroskaFile::open(cursor).ok()?;
    let track = mkv
        .tracks()
        .iter()
        .find(|t| t.track_type() == MkvTrackType::Video)?;
    let colour = track.video()?.colour()?;
    let (_, _, mut info) = colour_to_pipeline(colour);

    // matroska-demuxer 0.7 has two known bugs we work around with a raw
    // EBML scan (see `scan_mkv_colour_raw` doc):
    //   * `Colour::new` misreads MaxCLL/MaxFALL at the MatrixCoefficients
    //     ElementId offset (so both come back holding the matrix value).
    //   * `MasteringMetadata::new` misreads each `_chromaticity_y` at the
    //     matching `_chromaticity_x` ElementId (so all three primaries' y
    //     values come back holding the corresponding x value).
    // Clear the poisoned fields before the raw scan overrides them so a
    // scan miss doesn't leave the wrong value in place.
    info.max_cll = None;
    info.max_fall = None;
    if let Some(local) = info.mastering.as_mut() {
        local.primary_r_chromaticity_y = None;
        local.primary_g_chromaticity_y = None;
        local.primary_b_chromaticity_y = None;
    }
    if let Some(fix) = scan_mkv_colour_raw(data) {
        info.max_cll = fix.max_cll;
        info.max_fall = fix.max_fall;
        if let Some(local) = info.mastering.as_mut() {
            if fix.primary_r_chromaticity_y.is_some() {
                local.primary_r_chromaticity_y = fix.primary_r_chromaticity_y;
            }
            if fix.primary_g_chromaticity_y.is_some() {
                local.primary_g_chromaticity_y = fix.primary_g_chromaticity_y;
            }
            if fix.primary_b_chromaticity_y.is_some() {
                local.primary_b_chromaticity_y = fix.primary_b_chromaticity_y;
            }
        }
    }
    Some(info)
}

/// Raw-bytes EBML walk for the Colour element's MaxCLL (0x55BC),
/// MaxFALL (0x55BD), and the mastering display chromaticity_y fields
/// (0x55D2 / 0x55D4 / 0x55D6). Used exclusively as a workaround for
/// matroska-demuxer 0.7 bugs:
///   * `Colour::new` reads MaxCLL / MaxFALL from MatrixCoefficients
///     (lib.rs:725..728 in matroska-demuxer-0.7.0/src/lib.rs).
///   * `MasteringMetadata::new` reads `primary_{r,g,b}_chromaticity_y`
///     from the matching X ElementId (lib.rs:846/848/850), so all three
///     y values come back holding the corresponding x value.
/// Returns `None` when the file is not well-formed enough to reach the
/// Colour element, or when neither bug-recovery field is present.
#[derive(Default)]
struct RawColourFix {
    max_cll: Option<u32>,
    max_fall: Option<u32>,
    /// Mastering display y-chromaticity recoveries — Squad-21.
    primary_r_chromaticity_y: Option<f64>,
    primary_g_chromaticity_y: Option<f64>,
    primary_b_chromaticity_y: Option<f64>,
}

fn scan_mkv_colour_raw(data: &[u8]) -> Option<RawColourFix> {
    // Top-level: EBML header (0x1A45DFA3) then Segment (0x18538067).
    // We walk linearly until we find the Segment element and grab its
    // payload bytes — all subsequent work is inside that slice.
    let mut cursor = 0;
    let seg_body: &[u8] = loop {
        let (el, after) = next_ebml_element(data, cursor)?;
        if el.id == 0x18538067 {
            break &data[el.body_start..el.body_start + el.body_len];
        }
        cursor = after;
    };

    // Segment → Tracks (0x1654AE6B). Segment may carry many top-level
    // elements in any order — walk them until we find Tracks.
    let tracks = find_ebml_child(seg_body, 0x1654AE6B)?;
    // Tracks → TrackEntry* (0xAE). Look for the first TrackEntry whose
    // Video sub-element has a Colour; that's the path we care about.
    let mut cur = 0;
    while cur < tracks.len() {
        let (el, after) = next_ebml_element(tracks, cur)?;
        cur = after;
        if el.id != 0xAE {
            continue;
        }
        let entry = &tracks[el.body_start..el.body_start + el.body_len];
        let Some(video) = find_ebml_child(entry, 0xE0) else {
            continue;
        };
        let Some(colour) = find_ebml_child(video, 0x55B0) else {
            continue;
        };

        let mut fix = RawColourFix::default();
        let mut c = 0;
        while c < colour.len() {
            let (ce, after_ce) = match next_ebml_element(colour, c) {
                Some(v) => v,
                None => break,
            };
            c = after_ce;
            let value_bytes = &colour[ce.body_start..ce.body_start + ce.body_len];
            match ce.id {
                0x55BC => {
                    fix.max_cll = read_unsigned(value_bytes).and_then(|v| u32::try_from(v).ok());
                }
                0x55BD => {
                    fix.max_fall = read_unsigned(value_bytes).and_then(|v| u32::try_from(v).ok());
                }
                // MasteringMetadata sub-element (0x55D0). Walk its children
                // and pull the three buggy y-chromaticities so callers can
                // override the typed-accessor reads.
                0x55D0 => {
                    let md = value_bytes;
                    let mut mc = 0;
                    while mc < md.len() {
                        let (mce, after_mce) = match next_ebml_element(md, mc) {
                            Some(v) => v,
                            None => break,
                        };
                        mc = after_mce;
                        let mv = &md[mce.body_start..mce.body_start + mce.body_len];
                        match mce.id {
                            0x55D2 => fix.primary_r_chromaticity_y = read_float(mv),
                            0x55D4 => fix.primary_g_chromaticity_y = read_float(mv),
                            0x55D6 => fix.primary_b_chromaticity_y = read_float(mv),
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        if fix.max_cll.is_some()
            || fix.max_fall.is_some()
            || fix.primary_r_chromaticity_y.is_some()
            || fix.primary_g_chromaticity_y.is_some()
            || fix.primary_b_chromaticity_y.is_some()
        {
            return Some(fix);
        }
    }
    None
}

/// Walk the direct children of `buf` (assumed to be an EBML master
/// element body, NOT starting with the master's own header) and
/// return the payload slice of the first element with id `want`.
fn find_ebml_child(buf: &[u8], want: u32) -> Option<&[u8]> {
    let mut cur = 0;
    while cur < buf.len() {
        let (el, after) = next_ebml_element(buf, cur)?;
        cur = after;
        if el.id == want {
            return Some(&buf[el.body_start..el.body_start + el.body_len]);
        }
    }
    None
}

#[derive(Debug)]
struct RawEbmlElement {
    id: u32,
    body_start: usize,
    body_len: usize,
}

/// Read a single EBML element at `off` within `buf`. Returns the
/// element descriptor plus the byte offset immediately after the
/// element (header + body). Only handles up to 4-byte IDs (all
/// Matroska elements fit) and size VInts up to 8 bytes.
fn next_ebml_element(buf: &[u8], off: usize) -> Option<(RawEbmlElement, usize)> {
    if off >= buf.len() {
        return None;
    }
    let (id, id_len) = read_id_vint(&buf[off..])?;
    let body_off = off + id_len;
    if body_off >= buf.len() {
        return None;
    }
    let (size, size_len) = read_size_vint(&buf[body_off..])?;
    let body_start = body_off + size_len;
    if body_start + size as usize > buf.len() {
        return None;
    }
    let elem = RawEbmlElement {
        id,
        body_start,
        body_len: size as usize,
    };
    Some((elem, body_start + size as usize))
}

/// Read an EBML Class A/B/C/D ID (top-bit marker determines width,
/// 1..=4 bytes). Returns (raw id with marker bits preserved, byte-count).
fn read_id_vint(buf: &[u8]) -> Option<(u32, usize)> {
    if buf.is_empty() {
        return None;
    }
    let first = buf[0];
    let len = if first & 0x80 != 0 {
        1
    } else if first & 0x40 != 0 {
        2
    } else if first & 0x20 != 0 {
        3
    } else if first & 0x10 != 0 {
        4
    } else {
        return None;
    };
    if buf.len() < len {
        return None;
    }
    let mut id: u32 = 0;
    for b in &buf[..len] {
        id = (id << 8) | (*b as u32);
    }
    Some((id, len))
}

/// Read an EBML size VInt (1..=8 bytes). Strips the marker bit and
/// returns the numeric value plus byte-count.
fn read_size_vint(buf: &[u8]) -> Option<(u64, usize)> {
    if buf.is_empty() {
        return None;
    }
    let first = buf[0];
    if first == 0 {
        return None;
    }
    let len = first.leading_zeros() as usize + 1;
    if len > 8 || buf.len() < len {
        return None;
    }
    // Mask off the leading marker bit. `len == 8` (first byte 0x01) has
    // *no* value bits in the first byte — all 56 value bits live in
    // bytes 1..8. `u8 >> 8` is UB, so branch explicitly.
    let mask: u8 = if len == 8 { 0 } else { 0xFFu8 >> len };
    let mut v: u64 = (first & mask) as u64;
    for b in &buf[1..len] {
        v = (v << 8) | (*b as u64);
    }
    Some((v, len))
}

/// Read a big-endian unsigned integer (1..=8 bytes) from a Matroska
/// value payload. Zero-length payloads encode 0.
fn read_unsigned(buf: &[u8]) -> Option<u64> {
    if buf.len() > 8 {
        return None;
    }
    let mut v: u64 = 0;
    for b in buf {
        v = (v << 8) | (*b as u64);
    }
    Some(v)
}

/// Read a big-endian Matroska float payload — 4 bytes encode an f32,
/// 8 bytes encode an f64. Anything else is malformed.
fn read_float(buf: &[u8]) -> Option<f64> {
    match buf.len() {
        4 => {
            let arr: [u8; 4] = buf.try_into().ok()?;
            Some(f32::from_be_bytes(arr) as f64)
        }
        8 => {
            let arr: [u8; 8] = buf.try_into().ok()?;
            Some(f64::from_be_bytes(arr))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Streaming demuxer impls (Squad streaming-migration-55 P1)
// ---------------------------------------------------------------------------
//
// Per-format `StreamingDemuxer` implementations. Each holds only the cursor
// state needed to produce ONE sample at a time — no per-sample
// accumulation. Audio remains buffered (Squad-18 contract preserved).
//
// The legacy `demux()` is implemented at the bottom as a thin adapter:
// `demux_streaming(input)` → drain `next_video_sample()` into a `Vec`.

/// MP4 / MOV streaming demuxer. Owns the input bytes (so its
/// `Mp4Reader<Cursor<Vec<u8>>>` cursor is self-contained) and walks
/// `read_sample(track_id, idx)` one sample at a time. Per-sample
/// AVCC→Annex-B + parameter-set tracking (Squad-14) is preserved.
/// Per-sample location record built when the input is a fragmented
/// MP4. The `mp4` crate (v0.14) returns garbage (typically the bytes
/// of an adjacent `moof` box) from `read_sample` on fragmented inputs
/// — affects BOTH video and audio tracks. Side-stepping `read_sample`
/// for fragmented input by pre-computing sample
/// (file_offset, size, pts, duration) from the moof->traf->trun chain
/// produces correct bytes regardless of track kind. The track filter
/// is `track_id` (parameter on the walker chain) — generic across
/// video/audio/anything else with a track_id.
///
/// Bug history: the audio-extraction path WAS originally claimed to
/// "walk boxes itself" (per a prior comment here) but in fact it
/// called `reader.read_sample(audio_track_id, idx)` — the same buggy
/// path video uses. Burned 2026-05-09: malformed audio segments
/// (8-byte first AU containing the source's `moof` header bytes
/// `00 00 NN NN 6d 6f 6f 66`, every following AU mid-box-tree)
/// passed dedup hash unchanged because they're size-deterministic
/// per source, MSE rejected them with `Number of bands exceeds limit`
/// → SourceBuffer error → MediaSource readyState ended → all video
/// appendBuffer calls failed.
#[derive(Debug, Clone, Copy)]
struct FragSample {
    offset: u64,
    size: u32,
    pts_ticks: i64,
    duration_ticks: u32,
}

pub struct Mp4StreamingDemuxer {
    // Owned for the box-tree slice walkers (extract_*); the reader's
    // cursor consumes a clone.
    data: Vec<u8>,
    reader: Mp4Reader<Cursor<Vec<u8>>>,
    header: DemuxHeader,
    audio: Option<AudioTrack>,
    track_id: u32,
    sample_count: u32,
    next_idx: u32,
    // For AVC/HEVC: codec-specific config. Empty for the rest.
    sps_pps: Vec<Vec<u8>>,
    length_size: u8,
    tracker: Option<ParamSetTracker>,
    /// `Some` when the input is fragmented MP4. Each entry is a
    /// (file_offset, size, pts, duration) tuple resolved from
    /// moof/traf/trun. `next_video_sample` reads bytes directly from
    /// `self.data` at these offsets instead of going through the mp4
    /// crate's `read_sample`.
    fragmented_samples: Option<Vec<FragSample>>,
}

/// Walk top-level `moof` boxes in `data`, gather per-sample
/// (file_offset, size, pts, duration) tuples for the track id matching
/// `track_id` (works for video, audio, or any other track kind).
/// Returns `Some(table)` when the input is fragmented (at least one
/// top-level moof exists), `None` otherwise. An empty `Some(vec![])`
/// means "fragmented, but this track id had no samples in any moof"
/// — that's distinct from non-fragmented (None) and the caller
/// shouldn't fall back to `read_sample` in that case (it'd return
/// the same garbage bytes that prompted the fragmented path in the
/// first place).
///
/// Best-effort: silently skips moofs / trafs / truns that don't parse,
/// or that reference unknown tracks. Each successfully-walked trun
/// contributes its samples in order so the resulting Vec is decode-
/// order across the file.
fn build_fragmented_sample_table(
    data: &[u8],
    track_id: u32,
    default_sample_duration_from_trex: u32,
    default_sample_size_from_trex: u32,
) -> Option<Vec<FragSample>> {
    let mut samples: Vec<FragSample> = Vec::new();
    let mut pos: usize = 0;
    let mut accumulated_pts: i64 = 0;
    let mut found_any_moof = false;

    while pos + 8 <= data.len() {
        let box_size_field = u32::from_be_bytes(data[pos..pos + 4].try_into().ok()?);
        let box_type = &data[pos + 4..pos + 8];
        let (box_size, header_size): (usize, usize) = if box_size_field == 1 {
            // 64-bit largesize form.
            if pos + 16 > data.len() {
                break;
            }
            let big = u64::from_be_bytes(data[pos + 8..pos + 16].try_into().ok()?);
            (big as usize, 16)
        } else if box_size_field == 0 {
            // box extends to EOF — stop walking after this one.
            (data.len() - pos, 8)
        } else {
            (box_size_field as usize, 8)
        };
        if box_size < header_size || pos + box_size > data.len() {
            break;
        }

        if box_type == b"moof" {
            found_any_moof = true;
            let moof_start = pos;
            let moof_end = pos + box_size;
            walk_moof(
                data,
                moof_start + header_size,
                moof_end,
                moof_start as u64,
                track_id,
                default_sample_duration_from_trex,
                default_sample_size_from_trex,
                &mut accumulated_pts,
                &mut samples,
            );
        }
        pos = pos
            .checked_add(box_size)
            .filter(|&n| n <= data.len())
            .unwrap_or(data.len());
    }

    if found_any_moof { Some(samples) } else { None }
}

#[allow(clippy::too_many_arguments)]
fn walk_moof(
    data: &[u8],
    children_start: usize,
    moof_end: usize,
    moof_offset: u64,
    track_id: u32,
    default_sample_duration_from_trex: u32,
    default_sample_size_from_trex: u32,
    accumulated_pts: &mut i64,
    samples: &mut Vec<FragSample>,
) {
    let mut pos = children_start;
    while pos + 8 <= moof_end {
        let size = u32::from_be_bytes(match data[pos..pos + 4].try_into() {
            Ok(b) => b,
            Err(_) => break,
        });
        let typ = &data[pos + 4..pos + 8];
        if size == 0 || size as usize + pos > moof_end {
            break;
        }
        if typ == b"traf" {
            walk_traf(
                data,
                pos + 8,
                pos + size as usize,
                moof_offset,
                track_id,
                default_sample_duration_from_trex,
                default_sample_size_from_trex,
                accumulated_pts,
                samples,
            );
        }
        pos += size as usize;
    }
}

#[allow(clippy::too_many_arguments)]
fn walk_traf(
    data: &[u8],
    children_start: usize,
    traf_end: usize,
    moof_offset: u64,
    track_id: u32,
    default_sample_duration_from_trex: u32,
    default_sample_size_from_trex: u32,
    accumulated_pts: &mut i64,
    samples: &mut Vec<FragSample>,
) {
    // First pass: find tfhd (always first child of traf per spec) +
    // collect tfhd-derived defaults + base_data_offset semantics.
    let mut this_track: Option<u32> = None;
    let mut tfhd_default_sample_duration: u32 = default_sample_duration_from_trex;
    let mut tfhd_default_sample_size: u32 = default_sample_size_from_trex;
    let mut base_data_offset: u64 = moof_offset; // default-base-is-moof
    let mut base_data_offset_explicit = false;
    let mut tfdt_base_pts: Option<i64> = None;

    let mut pos = children_start;
    while pos + 8 <= traf_end {
        let size = u32::from_be_bytes(match data[pos..pos + 4].try_into() {
            Ok(b) => b,
            Err(_) => break,
        });
        let typ = &data[pos + 4..pos + 8];
        if size == 0 || size as usize + pos > traf_end {
            break;
        }
        if typ == b"tfhd" {
            // tfhd: u8 version + u24 flags + u32 track_id + optional fields per flag bits
            if pos + 16 > traf_end {
                pos += size as usize;
                continue;
            }
            let flags = u32::from_be_bytes(match data[pos + 8..pos + 12].try_into() {
                Ok(b) => b,
                Err(_) => break,
            }) & 0x00ff_ffff;
            let tk = u32::from_be_bytes(match data[pos + 12..pos + 16].try_into() {
                Ok(b) => b,
                Err(_) => break,
            });
            this_track = Some(tk);
            let mut p = pos + 16;
            // base_data_offset_present
            if flags & 0x01 != 0 {
                if p + 8 > traf_end {
                    break;
                }
                base_data_offset = u64::from_be_bytes(match data[p..p + 8].try_into() {
                    Ok(b) => b,
                    Err(_) => break,
                });
                base_data_offset_explicit = true;
                p += 8;
            }
            // sample_description_index_present
            if flags & 0x02 != 0 {
                p += 4;
            }
            // default_sample_duration_present
            if flags & 0x08 != 0 {
                if p + 4 > traf_end {
                    break;
                }
                tfhd_default_sample_duration =
                    u32::from_be_bytes(match data[p..p + 4].try_into() {
                        Ok(b) => b,
                        Err(_) => break,
                    });
                p += 4;
            }
            // default_sample_size_present
            if flags & 0x10 != 0 {
                if p + 4 > traf_end {
                    break;
                }
                tfhd_default_sample_size = u32::from_be_bytes(match data[p..p + 4].try_into() {
                    Ok(b) => b,
                    Err(_) => break,
                });
                p += 4;
            }
            // default_sample_flags_present (skip 4 bytes)
            if flags & 0x20 != 0 {
                p += 4;
            }
            // default-base-is-moof flag: when set AND base_data_offset
            // not present, base is the moof start (which is our default).
            let _ = p;
        } else if typ == b"tfdt" {
            // tfdt: version u8 + flags u24 + base_media_decode_time (u32 v0 / u64 v1)
            if pos + 12 > traf_end {
                pos += size as usize;
                continue;
            }
            let version = data[pos + 8];
            if version == 1 {
                if pos + 20 > traf_end {
                    pos += size as usize;
                    continue;
                }
                let bmdt =
                    u64::from_be_bytes(data[pos + 12..pos + 20].try_into().unwrap_or([0; 8]));
                tfdt_base_pts = Some(bmdt as i64);
            } else {
                let bmdt =
                    u32::from_be_bytes(data[pos + 12..pos + 16].try_into().unwrap_or([0; 4]));
                tfdt_base_pts = Some(bmdt as i64);
            }
        }
        pos += size as usize;
    }

    let Some(tk) = this_track else {
        return;
    };
    if tk != track_id {
        return;
    }

    if let Some(bp) = tfdt_base_pts {
        *accumulated_pts = bp;
    }

    // Second pass: walk trun boxes in declaration order.
    let mut pos = children_start;
    while pos + 8 <= traf_end {
        let size = u32::from_be_bytes(match data[pos..pos + 4].try_into() {
            Ok(b) => b,
            Err(_) => break,
        });
        let typ = &data[pos + 4..pos + 8];
        if size == 0 || size as usize + pos > traf_end {
            break;
        }
        if typ == b"trun" {
            walk_trun(
                data,
                pos + 8,
                pos + size as usize,
                if base_data_offset_explicit {
                    base_data_offset
                } else {
                    moof_offset
                },
                tfhd_default_sample_duration,
                tfhd_default_sample_size,
                accumulated_pts,
                samples,
            );
        }
        pos += size as usize;
    }
    let _ = base_data_offset_explicit;
}

#[allow(clippy::too_many_arguments)]
fn walk_trun(
    data: &[u8],
    children_start: usize,
    trun_end: usize,
    base_offset: u64,
    default_sample_duration: u32,
    default_sample_size: u32,
    accumulated_pts: &mut i64,
    samples: &mut Vec<FragSample>,
) {
    if children_start + 8 > trun_end {
        return;
    }
    let version = data[children_start];
    let flags = u32::from_be_bytes(match data[children_start..children_start + 4].try_into() {
        Ok(b) => b,
        Err(_) => return,
    }) & 0x00ff_ffff;
    let sample_count = u32::from_be_bytes(
        match data[children_start + 4..children_start + 8].try_into() {
            Ok(b) => b,
            Err(_) => return,
        },
    );
    let mut p = children_start + 8;
    let mut data_offset_in_trun: i32 = 0;
    if flags & 0x000_001 != 0 {
        if p + 4 > trun_end {
            return;
        }
        data_offset_in_trun = i32::from_be_bytes(match data[p..p + 4].try_into() {
            Ok(b) => b,
            Err(_) => return,
        });
        p += 4;
    }
    if flags & 0x000_004 != 0 {
        // first-sample-flags-present: skip 4 bytes
        p += 4;
    }

    let sample_duration_present = flags & 0x000_100 != 0;
    let sample_size_present = flags & 0x000_200 != 0;
    let sample_flags_present = flags & 0x000_400 != 0;
    let sample_cto_present = flags & 0x000_800 != 0;

    let mut current_offset = base_offset.wrapping_add(data_offset_in_trun as u64);
    for _ in 0..sample_count {
        let dur = if sample_duration_present {
            if p + 4 > trun_end {
                return;
            }
            let d = u32::from_be_bytes(match data[p..p + 4].try_into() {
                Ok(b) => b,
                Err(_) => return,
            });
            p += 4;
            d
        } else {
            default_sample_duration
        };
        let sz = if sample_size_present {
            if p + 4 > trun_end {
                return;
            }
            let s = u32::from_be_bytes(match data[p..p + 4].try_into() {
                Ok(b) => b,
                Err(_) => return,
            });
            p += 4;
            s
        } else {
            default_sample_size
        };
        if sample_flags_present {
            p += 4;
        }
        let cto: i32 = if sample_cto_present {
            if p + 4 > trun_end {
                return;
            }
            let c = if version == 0 {
                u32::from_be_bytes(match data[p..p + 4].try_into() {
                    Ok(b) => b,
                    Err(_) => return,
                }) as i32
            } else {
                i32::from_be_bytes(match data[p..p + 4].try_into() {
                    Ok(b) => b,
                    Err(_) => return,
                })
            };
            p += 4;
            c
        } else {
            0
        };

        if sz > 0 {
            samples.push(FragSample {
                offset: current_offset,
                size: sz,
                pts_ticks: accumulated_pts.saturating_add(cto as i64),
                duration_ticks: dur,
            });
        }
        current_offset = current_offset.saturating_add(sz as u64);
        *accumulated_pts = accumulated_pts.saturating_add(dur as i64);
    }
}

pub(crate) fn demux_mp4_streaming_init(data: &[u8]) -> Result<Mp4StreamingDemuxer> {
    // Same lenient pre-pass as `demux_mp4` — see comment there for
    // the iPhone / QuickTime `wave` atom rationale.
    let owned = sanitize_isobmff_box_sizes(data);
    let size = owned.len() as u64;
    // Build a probe reader against an immutable borrow first — same as
    // legacy `demux_mp4`. This pulls track / codec metadata before we
    // commit the owned buffer to the cursor that backs the streaming
    // reader.
    let probe = Mp4Reader::read_header(Cursor::new(owned.as_slice()), size)
        .context("reading MP4 header")?;

    let video_track = probe
        .tracks()
        .values()
        .find(|t| t.track_type().ok() == Some(mp4::TrackType::Video))
        .context("no video track in MP4")?;

    let track_id = video_track.track_id();
    let codec_from_mp4 = format_codec(video_track);
    let codec = if codec_from_mp4 == "unknown" && has_av01_sample_entry(&owned) {
        "av1".to_string()
    } else if codec_from_mp4 == "unknown" && hevc_sample_entry_fourcc(&owned).is_some() {
        "h265".to_string()
    } else if codec_from_mp4 == "unknown" && prores_sample_entry_fourcc(&owned).is_some() {
        "prores".to_string()
    } else {
        codec_from_mp4
    };
    let width = video_track.width() as u32;
    let height = video_track.height() as u32;
    let sample_count = video_track.sample_count();
    let duration = video_track.duration().as_secs_f64();
    let video_track_timescale = video_track.timescale();
    let frame_rate = mp4_frame_rate(video_track, duration);
    let bitrate = video_track.bitrate() as u64;

    let mp4_color = extract_mp4_visual_color_metadata(&owned);
    let initial_color_metadata = ColorMetadata {
        mastering_display: mp4_color.mastering_display,
        content_light_level: mp4_color.content_light_level,
        ..Default::default()
    };

    let mut info = StreamInfo {
        codec: codec.clone(),
        width,
        height,
        frame_rate,
        duration,
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        total_frames: sample_count as u64,
        bitrate,
        color_metadata: initial_color_metadata,
    };

    let needs_annexb = matches!(codec.as_str(), "h264" | "h265");
    let (sps_pps, length_size) = if needs_annexb {
        if codec == "h264" {
            match extract_avc_config(&owned) {
                Some(cfg) => (cfg.parameter_sets, cfg.length_size),
                None => (extract_sps_pps(&probe, track_id), 4u8),
            }
        } else {
            match extract_hevc_config(&owned) {
                Some(cfg) => (cfg.parameter_sets, cfg.length_size),
                None => (Vec::new(), 4u8),
            }
        }
    } else {
        (Vec::new(), 4u8)
    };

    // Pixel-format detection needs the SPS / sequence header. For hvc1 / avc1
    // the parameter sets live in the sample entry (`sps_pps`), NOT the first
    // VCL sample — detecting on the raw sample alone silently reports 8-bit for
    // a 10-bit Main 10 / Hi10P source, which then mis-sizes the encoder. Detect
    // on the parameter sets (Annex-B) when present; fall back to the first
    // sample for hev1 / avc3 (in-band) and AV1 / VP9 (sequence header in band).
    if sample_count > 0 {
        let detect_input: Vec<u8> = if !sps_pps.is_empty() {
            let mut buf = Vec::new();
            for ps in &sps_pps {
                buf.extend_from_slice(&[0, 0, 0, 1]);
                buf.extend_from_slice(ps);
            }
            buf
        } else {
            let mut probe_for_pf = Mp4Reader::read_header(Cursor::new(owned.as_slice()), size)
                .context("re-reading MP4 for pixel-format probe")?;
            match probe_for_pf.read_sample(track_id, 1) {
                Ok(Some(s)) => s.bytes.to_vec(),
                _ => Vec::new(),
            }
        };
        if !detect_input.is_empty() {
            info.pixel_format = codec::pixel_format::detect(&codec, &[detect_input]);
        }
    }

    drop(probe);

    let audio = extract_mp4_audio(&owned);

    // Build the streaming reader against an owned cursor.
    let reader_cursor = Cursor::new(owned.clone());
    let reader =
        Mp4Reader::read_header(reader_cursor, size).context("opening MP4 streaming reader")?;

    let tracker = if needs_annexb {
        Some(ParamSetTracker::new(if codec == "h264" {
            NaluCodec::Avc
        } else {
            NaluCodec::Hevc
        }))
    } else {
        None
    };

    let _ = needs_annexb; // tracker presence reflects this

    // Detect fragmented MP4 + build a sample table from moof/traf/trun
    // when applicable. The mp4 crate's `read_sample` returns garbage
    // (typically the bytes of an adjacent moof box header) for any
    // fragmented track regardless of kind, so for fragmented input
    // we bypass `read_sample` entirely and read sample bytes directly
    // from `owned` at the offsets in this table. `extract_mp4_audio`
    // does the same against its own `data` slice.
    let fragmented_samples = build_fragmented_sample_table(&owned, track_id, 0, 0).map(|table| {
        tracing::info!(
            track_id,
            sample_count = table.len(),
            "fragmented MP4 detected; built sample table from moof/traf/trun"
        );
        table
    });
    let final_sample_count = match &fragmented_samples {
        Some(table) => table.len() as u32,
        None => sample_count,
    };

    // Recompute frame_rate + duration from fragmented sample timestamps
    // when (a) we built a fragmented sample table AND (b) the static
    // moov sample table was empty or had a zero duration. Pure
    // fragmented MP4 — common from web recorders, screen capture
    // tools, and modern phone exports — leaves moov with no static
    // samples + tkhd.duration=0; the previous fallback was the 30.0
    // sentinel, which silently encoded a 24-fps VFR source as 30-fps
    // CFR and produced ~20% short output. The fragmented sample
    // table's actual duration_ticks (from moof.traf.trun per-sample
    // duration entries) carries the truth. Trust the static table
    // when it's populated — that path was correct already.
    if let Some(table) = fragmented_samples.as_ref() {
        if !table.is_empty() && (sample_count == 0 || duration <= 0.0) && video_track_timescale > 0
        {
            let total_ticks: u64 = table.iter().map(|s| s.duration_ticks as u64).sum();
            if total_ticks > 0 {
                let total_seconds = total_ticks as f64 / video_track_timescale as f64;
                if total_seconds > 0.0 {
                    let avg_fps = table.len() as f64 / total_seconds;
                    info.frame_rate = avg_fps.clamp(1.0, 240.0);
                    info.duration = total_seconds;
                    info.total_frames = table.len() as u64;
                    tracing::info!(
                        track_id,
                        avg_fps,
                        total_seconds,
                        sample_count = table.len(),
                        timescale = video_track_timescale,
                        "fragmented MP4: recomputed frame_rate + duration from \
                         moof/traf/trun timestamps (static moov sample table \
                         was empty)"
                    );
                }
            }
        }
    }
    Ok(Mp4StreamingDemuxer {
        data: owned,
        reader,
        header: DemuxHeader { codec, info },
        audio,
        track_id,
        sample_count: final_sample_count,
        next_idx: 1,
        sps_pps,
        length_size,
        tracker,
        fragmented_samples,
    })
}

impl StreamingDemuxer for Mp4StreamingDemuxer {
    fn header(&self) -> &DemuxHeader {
        &self.header
    }

    fn next_video_sample(&mut self) -> Result<Option<Sample>> {
        // Fragmented MP4 path: pull bytes directly from the input buffer
        // at the offsets we resolved at init time.
        if let Some(table) = self.fragmented_samples.as_ref() {
            let idx_zero_based = (self.next_idx - 1) as usize;
            if idx_zero_based >= table.len() {
                return Ok(None);
            }
            self.next_idx += 1;
            let entry = table[idx_zero_based];
            let off = entry.offset as usize;
            let end = off.saturating_add(entry.size as usize);
            if end > self.data.len() {
                tracing::warn!(
                    idx = idx_zero_based + 1,
                    offset = entry.offset,
                    size = entry.size,
                    data_len = self.data.len(),
                    "fragmented sample reaches past EOF; stopping at the previous frame"
                );
                return Ok(None);
            }
            let raw = self.data[off..end].to_vec();
            let data = if let Some(tracker) = self.tracker.as_mut() {
                length_prefixed_to_annexb_tracked(&raw, self.length_size, tracker, &self.sps_pps)
            } else {
                raw
            };
            return Ok(Some(Sample {
                data,
                pts_ticks: entry.pts_ticks,
                duration_ticks: entry.duration_ticks,
            }));
        }
        loop {
            if self.next_idx > self.sample_count {
                return Ok(None);
            }
            let idx = self.next_idx;
            self.next_idx += 1;
            // Mirror the audio-track tolerance in `extract_mp4_audio`:
            // when a mid-track read_sample fails on a fragmented MP4
            // with a truncated `traf.trun` index — the typical iPhone /
            // Android broken-recording shape — surface a warn and
            // signal soft EOF to the encode loop. The frames that DID
            // demux upstream still flow through, the encoder produces
            // an AV1 sequence header from the first one, and the CMAF
            // muxer's `finalize` writes a valid (truncated) init
            // segment. Without this, a single missing trun entry
            // halfway through a clip would propagate as `TranscodeFailure`
            // for the whole job — the symptom we hit 2026-05-08.
            let s = match self.reader.read_sample(self.track_id, idx) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(
                        track_id = self.track_id,
                        idx,
                        emitted = idx.saturating_sub(1),
                        sample_count = self.sample_count,
                        error = %e,
                        "video stream: read_sample error mid-track; \
                         stopping at sample {} of {} (truncated source — \
                         iPhone fragmented MP4 with a missing trun entry \
                         is the typical cause)",
                        idx.saturating_sub(1),
                        self.sample_count,
                    );
                    return Ok(None);
                }
            };
            let Some(sample) = s else { continue };
            let pts_ticks = sample.start_time as i64;
            let duration_ticks = sample.duration;
            let raw = sample.bytes.to_vec();
            let data = if let Some(tracker) = self.tracker.as_mut() {
                length_prefixed_to_annexb_tracked(&raw, self.length_size, tracker, &self.sps_pps)
            } else {
                raw
            };
            return Ok(Some(Sample {
                data,
                pts_ticks,
                duration_ticks,
            }));
        }
    }

    fn audio(&self) -> Option<&AudioTrack> {
        self.audio.as_ref()
    }
}

impl Mp4StreamingDemuxer {
    /// For tests + the legacy `demux()` adapter: reach back at the
    /// owned input bytes (e.g. for an opt-in re-probe).
    #[allow(dead_code)]
    pub(crate) fn raw_bytes(&self) -> &[u8] {
        &self.data
    }
}

/// MKV / WebM streaming demuxer. Wraps `MatroskaFile` whose `next_frame`
/// API is already pull-shaped, so the streaming impl is a thin wrapper:
/// pull next frame, filter to the video track, AVCC→Annex-B convert if
/// AVC/HEVC, surface as a `Sample`.
pub struct MkvStreamingDemuxer {
    mkv: MatroskaFile<Cursor<Vec<u8>>>,
    header: DemuxHeader,
    audio: Option<AudioTrack>,
    track_number: u64,
    timestamp_scale: u64,
    annexb_prepend: Vec<Vec<u8>>,
    length_size: u8,
    tracker: Option<ParamSetTracker>,
    /// Default-duration in ns from the track header — used as the
    /// fallback per-sample duration when the Block doesn't carry one.
    default_duration_ns: Option<u64>,
    /// Lazily set on the first `next_video_sample()` call by running
    /// `pixel_format::detect` against the first emitted sample.
    /// `header.info.pixel_format` is then patched in place. Subsequent
    /// calls skip the probe (codec sequence headers don't change
    /// mid-stream for the codecs we support).
    pixel_format_detected: bool,
}

pub(crate) fn demux_mkv_streaming_init(data: &[u8]) -> Result<MkvStreamingDemuxer> {
    let owned = data.to_vec();
    // First pass: open with a borrow to harvest header metadata without
    // consuming the buffer that backs the streaming reader.
    let cursor = Cursor::new(owned.as_slice());
    let probe =
        MatroskaFile::open(cursor).map_err(|e| anyhow::anyhow!("reading MKV header: {e}"))?;

    let (
        track_number,
        track_uid,
        codec_id,
        width,
        height,
        annexb_prepend,
        length_size,
        color_space,
        mut color_metadata,
        mut color_info,
        track_default_duration_ns,
    ) = {
        let track_info = probe
            .tracks()
            .iter()
            .find(|t| t.track_type() == MkvTrackType::Video)
            .context("no video track in MKV")?;

        let track_number = track_info.track_number().get();
        let track_uid = track_info.track_uid().get();
        let codec_id = track_info.codec_id().to_string();
        let default_duration_ns = track_info.default_duration().map(|d| d.get());

        let (annexb_prepend, length_size): (Vec<Vec<u8>>, u8) = if codec_id == "V_MPEG4/ISO/AVC" {
            let priv_bytes = track_info
                .codec_private()
                .context("V_MPEG4/ISO/AVC CodecPrivate missing")?;
            let cfg = parse_avcc(priv_bytes).context("V_MPEG4/ISO/AVC CodecPrivate malformed")?;
            (cfg.parameter_sets, cfg.length_size)
        } else if codec_id == "V_MPEGH/ISO/HEVC" {
            let priv_bytes = track_info
                .codec_private()
                .context("V_MPEGH/ISO/HEVC CodecPrivate missing")?;
            let cfg = parse_hvcc(priv_bytes).context("V_MPEGH/ISO/HEVC CodecPrivate malformed")?;
            (cfg.parameter_sets, cfg.length_size)
        } else {
            (Vec::new(), 4)
        };

        if mkv_codec_needs_annexb(&codec_id) && annexb_prepend.is_empty() {
            bail!("AVC/HEVC MKV CodecPrivate missing or empty — no parameter sets to prepend");
        }

        let video = track_info
            .video()
            .context("video track missing Video element")?;
        let w = video.pixel_width().get() as u32;
        let h = video.pixel_height().get() as u32;

        let (color_space, color_metadata, color_info) = match video.colour() {
            Some(colour) => colour_to_pipeline(colour),
            None => (
                ColorSpace::Bt709,
                ColorMetadata::default(),
                MkvColorInfo::default(),
            ),
        };

        (
            track_number,
            track_uid,
            codec_id,
            w,
            h,
            annexb_prepend,
            length_size,
            color_space,
            color_metadata,
            color_info,
            default_duration_ns,
        )
    };

    // Apply the matroska-demuxer 0.7 raw-scan workarounds — same as the
    // legacy demux_mkv path.
    color_info.max_cll = None;
    color_info.max_fall = None;
    color_metadata.content_light_level = None;
    if let Some(md) = color_metadata.mastering_display.as_mut() {
        md.primaries_r_y = 0;
        md.primaries_g_y = 0;
        md.primaries_b_y = 0;
    }
    if let Some(local) = color_info.mastering.as_mut() {
        local.primary_r_chromaticity_y = None;
        local.primary_g_chromaticity_y = None;
        local.primary_b_chromaticity_y = None;
    }
    if let Some(fix) = scan_mkv_colour_raw(&owned) {
        color_info.max_cll = fix.max_cll;
        color_info.max_fall = fix.max_fall;
        if fix.max_cll.is_some() || fix.max_fall.is_some() {
            color_metadata.content_light_level = Some(ContentLightLevel {
                max_cll: fix.max_cll.unwrap_or(0).min(u16::MAX as u32) as u16,
                max_fall: fix.max_fall.unwrap_or(0).min(u16::MAX as u32) as u16,
            });
        }
        let chrom = |v: f64| (v * 50_000.0).round().clamp(0.0, u16::MAX as f64) as u16;
        if let Some(md) = color_metadata.mastering_display.as_mut() {
            if let Some(y) = fix.primary_r_chromaticity_y {
                md.primaries_r_y = chrom(y);
            }
            if let Some(y) = fix.primary_g_chromaticity_y {
                md.primaries_g_y = chrom(y);
            }
            if let Some(y) = fix.primary_b_chromaticity_y {
                md.primaries_b_y = chrom(y);
            }
        }
        if let Some(local) = color_info.mastering.as_mut() {
            if fix.primary_r_chromaticity_y.is_some() {
                local.primary_r_chromaticity_y = fix.primary_r_chromaticity_y;
            }
            if fix.primary_g_chromaticity_y.is_some() {
                local.primary_g_chromaticity_y = fix.primary_g_chromaticity_y;
            }
            if fix.primary_b_chromaticity_y.is_some() {
                local.primary_b_chromaticity_y = fix.primary_b_chromaticity_y;
            }
        }
    }

    let needs_annexb = mkv_codec_needs_annexb(&codec_id);
    let codec = match codec_id.as_str() {
        "V_VP9" => "vp9".to_string(),
        "V_VP8" => "vp8".to_string(),
        "V_AV1" => "av1".to_string(),
        "V_MPEG4/ISO/AVC" => "h264".to_string(),
        "V_MPEGH/ISO/HEVC" => "h265".to_string(),
        other => other.to_lowercase(),
    };

    let timestamp_scale = probe.info().timestamp_scale().get();
    let duration_ticks = probe.info().duration().unwrap_or(0.0);
    let duration = duration_ticks * (timestamp_scale as f64) / 1_000_000_000.0;
    let tag_bitrate = probe
        .tags()
        .and_then(|tags| bitrate_from_tags(tags, track_uid));
    if color_info != MkvColorInfo::default() {
        tracing::info!(
            bits_per_channel = ?color_info.bits_per_channel,
            max_cll = ?color_info.max_cll,
            max_fall = ?color_info.max_fall,
            mastering = ?color_info.mastering,
            "MKV Colour: parsed HDR-adjacent metadata"
        );
    }

    drop(probe);

    // Audio: extract from the owned bytes via a separate MatroskaFile
    // open (same as legacy demux_mkv). The video reader below needs its
    // own clean cursor.
    let audio = extract_mkv_audio(&owned);

    // Build the streaming MKV reader against the owned buffer.
    let mkv = MatroskaFile::open(Cursor::new(owned.clone()))
        .map_err(|e| anyhow::anyhow!("opening MKV streaming reader: {e}"))?;

    // Bitrate / frame_rate / pixel_format are best-effort at construction
    // time. Bitrate falls back to 0 (unknown) if no tag exists; the
    // legacy path computes it by summing sample bytes which is fine for
    // Vec-materialized output but blows the streaming budget. We surface
    // the tag bitrate when present and 0 otherwise — pipeline already
    // tolerates 0 (matches the AVI / TS behaviour).
    let bitrate = tag_bitrate.unwrap_or(0);

    // For frame_rate we apply the Squad-32 fallback chain as far as it
    // goes without the materialized sample count. samples/duration is
    // unknowable in streaming, so use DefaultDuration first then 30.0.
    let frame_rate = if let Some(dd_ns) = track_default_duration_ns.filter(|n| *n > 0) {
        1_000_000_000.0 / dd_ns as f64
    } else if duration > 0.0 {
        // duration-only fallback: assume 30 fps × duration as the floor.
        // This matches what the legacy path produced when sample count
        // was tiny; for normal media DefaultDuration is virtually always
        // present.
        30.0
    } else {
        30.0
    };

    // Pixel format detection requires a sample. For the streaming
    // demuxer's StreamInfo we keep the codec-defaulted Yuv420p — the
    // actual decoded format is whatever the decoder produces.
    // (The legacy `demux_mkv()` adapter re-runs `pixel_format::detect`
    // on the materialized samples after the drain.)
    let pixel_format = PixelFormat::Yuv420p;

    let info = StreamInfo {
        codec: codec.clone(),
        width,
        height,
        frame_rate,
        duration,
        pixel_format,
        color_space,
        total_frames: 0, // unknown until drained
        bitrate,
        color_metadata,
    };

    let tracker = if needs_annexb {
        Some(ParamSetTracker::new(if codec_id == "V_MPEG4/ISO/AVC" {
            NaluCodec::Avc
        } else {
            NaluCodec::Hevc
        }))
    } else {
        None
    };

    let _ = needs_annexb; // tracker presence reflects this
    Ok(MkvStreamingDemuxer {
        mkv,
        header: DemuxHeader { codec, info },
        audio,
        track_number,
        timestamp_scale,
        annexb_prepend,
        length_size,
        tracker,
        default_duration_ns: track_default_duration_ns,
        pixel_format_detected: false,
    })
}

impl StreamingDemuxer for MkvStreamingDemuxer {
    fn header(&self) -> &DemuxHeader {
        &self.header
    }

    fn next_video_sample(&mut self) -> Result<Option<Sample>> {
        let mut frame = MkvFrame::default();
        loop {
            match self.mkv.next_frame(&mut frame) {
                Ok(true) => {
                    if frame.track != self.track_number {
                        continue;
                    }
                    let raw = std::mem::take(&mut frame.data);
                    let data = if let Some(tracker) = self.tracker.as_mut() {
                        length_prefixed_to_annexb_tracked(
                            &raw,
                            self.length_size,
                            tracker,
                            &self.annexb_prepend,
                        )
                    } else {
                        raw
                    };
                    // Lazy pixel-format detection on the first sample.
                    // `pixel_format::detect` only ever reads `samples[0]`,
                    // so a one-shot probe against the first emitted sample
                    // matches the legacy `demux_mkv()` behaviour without
                    // requiring the full Vec to be materialised first.
                    if !self.pixel_format_detected {
                        let detected = codec::pixel_format::detect(
                            &self.header.codec,
                            std::slice::from_ref(&data),
                        );
                        self.header.info.pixel_format = detected;
                        self.pixel_format_detected = true;
                    }
                    let pts_ticks = frame.timestamp.saturating_mul(self.timestamp_scale) as i64;
                    let duration_ticks = frame
                        .duration
                        .or(self.default_duration_ns)
                        .map(|ns| ns.min(u32::MAX as u64) as u32)
                        .unwrap_or(0);
                    return Ok(Some(Sample {
                        data,
                        pts_ticks,
                        duration_ticks,
                    }));
                }
                Ok(false) => return Ok(None),
                Err(e) => bail!("MKV frame read error: {e}"),
            }
        }
    }

    fn audio(&self) -> Option<&AudioTrack> {
        self.audio.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mkv_annexb_guard_flags_avc_and_hevc() {
        assert!(mkv_codec_needs_annexb("V_MPEG4/ISO/AVC"));
        assert!(mkv_codec_needs_annexb("V_MPEGH/ISO/HEVC"));
    }

    #[test]
    fn mkv_annexb_guard_passes_self_contained_codecs() {
        assert!(!mkv_codec_needs_annexb("V_VP9"));
        assert!(!mkv_codec_needs_annexb("V_VP8"));
        assert!(!mkv_codec_needs_annexb("V_AV1"));
        assert!(!mkv_codec_needs_annexb("V_UNKNOWN"));
    }

    #[test]
    fn parse_avcc_extracts_sps_and_pps() {
        // One SPS (6 bytes) + one PPS (4 bytes), no extension fields.
        let sps: [u8; 6] = [0x67, 0x42, 0x00, 0x1e, 0xab, 0x40];
        let pps: [u8; 4] = [0x68, 0xce, 0x3c, 0x80];
        let mut avcc = Vec::new();
        avcc.push(0x01); // configurationVersion
        avcc.push(0x42); // AVCProfileIndication = 66 (Baseline)
        avcc.push(0x00); // profile_compatibility
        avcc.push(0x1e); // AVCLevelIndication = 3.0
        avcc.push(0xff); // reserved(6)=1|lengthSizeMinusOne(2)=3
        avcc.push(0xe1); // reserved(3)=7|numOfSequenceParameterSets(5)=1
        avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&sps);
        avcc.push(0x01); // numOfPictureParameterSets = 1
        avcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&pps);

        let sets = parse_avcc_param_sets(&avcc);
        assert_eq!(sets.len(), 2, "expected SPS + PPS");
        assert_eq!(&sets[0], &sps);
        assert_eq!(&sets[1], &pps);
    }

    #[test]
    fn parse_avcc_truncated_returns_partial() {
        // Truncation mid-SPS should not panic; returns whatever was fully read.
        let avcc: [u8; 6] = [0x01, 0x42, 0x00, 0x1e, 0xff, 0xe1];
        let sets = parse_avcc_param_sets(&avcc);
        assert!(sets.is_empty());
    }

    #[test]
    fn parse_avcc_empty_record_returns_empty() {
        assert!(parse_avcc_param_sets(&[]).is_empty());
        assert!(parse_avcc_param_sets(&[0x01]).is_empty());
    }

    /// Build a minimal box: `[size u32 BE][fourcc 4][payload]`.
    fn mkbox(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let size = (8 + payload.len()) as u32;
        let mut out = Vec::with_capacity(size as usize);
        out.extend_from_slice(&size.to_be_bytes());
        out.extend_from_slice(fourcc);
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn av01_detector_finds_sample_entry_in_nested_stsd() {
        // Minimal valid stsd body: version/flags (4 B) + entry_count=1 (4 B)
        // + one 16-byte av01 box header.
        let mut stsd_body = vec![0u8; 8];
        stsd_body.extend_from_slice(&mkbox(b"av01", &[0u8; 8]));
        let stsd = mkbox(b"stsd", &stsd_body);
        let stbl = mkbox(b"stbl", &stsd);
        let minf = mkbox(b"minf", &stbl);
        let mdia = mkbox(b"mdia", &minf);
        let trak = mkbox(b"trak", &mdia);
        let moov = mkbox(b"moov", &trak);
        assert!(has_av01_sample_entry(&moov));
    }

    #[test]
    fn av01_detector_ignores_av01_in_wrong_place() {
        // av01 bytes floating in mdat must not trigger the detector.
        let mdat = mkbox(b"mdat", b"...av01... garbage");
        assert!(!has_av01_sample_entry(&mdat));
    }

    #[test]
    fn read_size_vint_8_byte_encoding() {
        // size_vint_8 form used by the MKV test builder: `(1 << 56) | size`
        // encoded as 8 bytes big-endian. First byte is 0x01.
        let size: u64 = 1000;
        let v = (1u64 << 56) | size;
        let bytes = v.to_be_bytes();
        let (read, len) = read_size_vint(&bytes).expect("parse 8-byte size");
        assert_eq!(len, 8);
        assert_eq!(read, 1000);
    }

    #[test]
    fn read_size_vint_1_byte_encoding() {
        // 1-byte VInt for value 1: 0x81.
        let (v, l) = read_size_vint(&[0x81]).expect("1-byte size");
        assert_eq!(l, 1);
        assert_eq!(v, 1);
    }

    #[test]
    fn read_id_vint_parses_matroska_ids() {
        assert_eq!(read_id_vint(&[0xAE]), Some((0xAE, 1)));
        assert_eq!(
            read_id_vint(&[0x1A, 0x45, 0xDF, 0xA3, 0xFF]),
            Some((0x1A45DFA3, 4))
        );
        assert_eq!(read_id_vint(&[0x55, 0xB0, 0xFF]), Some((0x55B0, 2)));
    }

    #[test]
    fn av01_detector_returns_false_for_avc1_sample_entry() {
        let mut stsd_body = vec![0u8; 8];
        stsd_body.extend_from_slice(&mkbox(b"avc1", &[0u8; 8]));
        let stsd = mkbox(b"stsd", &stsd_body);
        let stbl = mkbox(b"stbl", &stsd);
        let minf = mkbox(b"minf", &stbl);
        let mdia = mkbox(b"mdia", &minf);
        let trak = mkbox(b"trak", &mdia);
        let moov = mkbox(b"moov", &trak);
        assert!(!has_av01_sample_entry(&moov));
    }

    /// Helper: build a minimal MOV box tree carrying a single sample
    /// entry with the supplied fourcc, nested moov/trak/mdia/minf/stbl/stsd.
    /// The sample entry payload itself is zeros — the prores detector
    /// only looks at the fourcc, not at any internal fields.
    fn mov_with_sample_entry(fourcc: &[u8; 4]) -> Vec<u8> {
        let mut stsd_body = vec![0u8; 8]; // version/flags + entry_count
        stsd_body.extend_from_slice(&mkbox(fourcc, &[0u8; 8]));
        let stsd = mkbox(b"stsd", &stsd_body);
        let stbl = mkbox(b"stbl", &stsd);
        let minf = mkbox(b"minf", &stbl);
        let mdia = mkbox(b"mdia", &minf);
        let trak = mkbox(b"trak", &mdia);
        mkbox(b"moov", &trak)
    }

    #[test]
    fn prores_detector_finds_all_six_fourccs() {
        for fcc in [b"apco", b"apcs", b"apcn", b"apch", b"ap4h", b"ap4x"] {
            let moov = mov_with_sample_entry(fcc);
            let detected = prores_sample_entry_fourcc(&moov)
                .unwrap_or_else(|| panic!("did not detect ProRes fourcc {fcc:?}"));
            assert_eq!(&detected, fcc, "fourcc round-trip for {fcc:?}");
        }
    }

    #[test]
    fn prores_detector_ignores_non_prores_fourccs() {
        // A sample entry whose fourcc is something else (h264, hevc, etc.)
        // must NOT trigger the ProRes detector even when nested correctly.
        for fcc in [b"avc1", b"hvc1", b"av01", b"vp09", b"mp4v"] {
            let moov = mov_with_sample_entry(fcc);
            assert!(
                prores_sample_entry_fourcc(&moov).is_none(),
                "false positive on fourcc {fcc:?}"
            );
        }
    }

    #[test]
    fn prores_detector_returns_none_when_no_stsd() {
        // Bare moov with no stsd path — must safely return None,
        // never panic.
        let moov = mkbox(b"moov", &[0u8; 4]);
        assert!(prores_sample_entry_fourcc(&moov).is_none());
    }

    #[test]
    fn detect_container_recognises_mpeg_ts_sync_pattern() {
        // detect_container is package-private here; we exercise it via
        // a buffer whose first three sync points all land on 0x47.
        let mut buf = vec![0xFFu8; 12];
        buf[0] = 0x47;
        // Pad to length so detect_container can probe offsets 188 and 376.
        while buf.len() < 400 {
            buf.push(0x00);
        }
        buf[188] = 0x47;
        buf[376] = 0x47;
        assert_eq!(detect_container(&buf), "ts");
    }

    #[test]
    fn detect_container_rejects_lone_0x47_byte() {
        // A single 0x47 sync byte must not be enough — random payloads
        // routinely contain it. Demand at least two confirming hits.
        let mut buf = vec![0u8; 400];
        buf[0] = 0x47;
        buf[188] = 0x00; // miss the second probe
        assert_ne!(detect_container(&buf), "ts");
    }

    #[test]
    fn detect_container_recognises_avi_riff_signature() {
        let mut buf: Vec<u8> = b"RIFF".to_vec();
        buf.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        buf.extend_from_slice(b"AVI ");
        buf.extend_from_slice(&[0u8; 32]);
        assert_eq!(detect_container(&buf), "avi");
    }
}
