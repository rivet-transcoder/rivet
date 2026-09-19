use std::path::Path;

use anyhow::{Context, Result, bail};

use codec::audio::filter::AudioFilter;
use codec::audio::{
    AudioCodec, AudioEncoderConfig, create_decoder as audio_decoder,
    create_encoder as audio_encoder,
};
use container::cmaf::CmafAudioMuxer;
use container::demux::AudioTrack;
use container::hls::AudioVariantSpec;
use container::AudioInfo;

use crate::cmaf_util::add_audio_sample_with_segment_flush;
use crate::spec::AudioCodecPolicy;

// ---------------------------------------------------------------------------
// PreparedAudio
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(super) struct PreparedAudio {
    pub(super) info: AudioInfo,
    pub(super) samples: Vec<(Vec<u8>, u32)>,
    pub(super) handling: String,
    /// How the samples are presented, in ticks of `info.timescale`: the
    /// source's audio edit list carried to the output (priming, decoder
    /// preroll and a trim's partial packet hidden; a late start). The identity
    /// for a source without one, which writes no edit list.
    pub(super) edit: container::edit::TrackEdit,
}

impl PreparedAudio {
    pub(super) fn has_samples(&self) -> bool {
        !self.samples.is_empty()
    }

    /// Append another track's samples after this one (for splice concat). The
    /// muxer re-times from the running duration, so the joined audio is gap-free.
    ///
    /// An output track has one edit list: it places the start and the end of
    /// the joined audio exactly, and nothing in between. Inside a join — this
    /// track presenting less than its samples hold, or the next one hiding
    /// samples at its start (priming, a source trim) — a passthrough track can
    /// only be cut at a packet boundary. Each cut goes on the boundary nearest
    /// where the edit wants it, counting the error the previous cut left: the
    /// audio is within half a packet of its pictures at every join, and the
    /// error does not grow with the number of joins. The joined track's edit
    /// then carries the intended total length, so the end is exact again.
    ///
    /// A late start (an empty edit) on a clip after the first cannot be written
    /// inside a join; it is dropped, with a warning, and the join is gap-free.
    /// What of it the clip's video shares — both starting late, as a transport
    /// stream's streams do against its program clock — is taken off before the
    /// join ([`super::splice::trim_audio_to_video`]); what reaches here is the
    /// audio starting after its own pictures.
    pub(super) fn extend(&mut self, other: &PreparedAudio) {
        if self.edit.duration.is_none() && other.edit.is_identity() {
            self.samples.extend(other.samples.iter().cloned());
            return;
        }
        if other.edit.delay != 0 {
            tracing::warn!(
                delay = other.edit.delay,
                "splice: a late audio start inside a join cannot be written; the join is gap-free"
            );
        }
        let total = |s: &[(Vec<u8>, u32)]| s.iter().map(|(_, d)| u64::from(*d)).sum::<u64>();
        // Where this track's presentation ends, in its own media ticks.
        let presented = self.edit.duration.unwrap_or(total(&self.samples).saturating_sub(self.edit.media_time));
        let end = self.edit.media_time + presented;
        // A fixed-frame codec's last packet decodes to a whole frame even when
        // its duration says less: the encoder's end padding, which the track's
        // own edit hid at its end. Inside a join that padding is decoded and
        // played, so the packet is written and counted at its decoded length —
        // counted at its duration, every join ran late by the padding (256
        // samples for an ffmpeg AAC clip, 5.3 ms) and the error grew by that
        // much with each join.
        if let Some(frame) = fixed_frame_ticks(&self.info.codec, &self.samples)
            && let Some(last) = self.samples.last_mut()
        {
            last.1 = last.1.max(frame);
        }
        let (keep, kept_to) = nearest_packet_boundary(&self.samples, end);
        self.samples.truncate(keep);
        // Audio kept past (+) or short of (-) where it should stop: the next
        // track's start moves by as much, so the error does not carry on.
        let overrun = kept_to as i64 - end as i64;
        let skip = (other.edit.media_time as i64 + overrun).max(0) as u64;
        let (drop, dropped_to) = nearest_packet_boundary(&other.samples, skip);
        self.samples.extend(other.samples[drop..].iter().cloned());
        let other_presented =
            other.edit.duration.unwrap_or(total(&other.samples).saturating_sub(other.edit.media_time));
        self.edit.duration = Some(presented + other_presented);
        tracing::info!(
            join_error_ticks = overrun - (dropped_to as i64 - other.edit.media_time as i64),
            timescale = self.info.timescale,
            "splice: audio edit inside the join applied at the nearest packet boundary"
        );
    }
}

/// The frame length, in ticks, of a codec whose every packet decodes to the
/// same number of samples (AAC, AC-3, E-AC-3, DTS): the longest packet
/// duration in the track, of those within half again of the median — a hole
/// in a transport stream's audio lengthens the packet before it by more than
/// half a frame ([`container::edit::AudioGap`]), and is not a frame. `None`
/// for Opus, whose packets legitimately vary.
fn fixed_frame_ticks(codec: &str, samples: &[(Vec<u8>, u32)]) -> Option<u32> {
    let fixed = ["aac", "ac3", "eac3", "dts"].iter().any(|c| codec.eq_ignore_ascii_case(c));
    if !fixed {
        return None;
    }
    let mut durations: Vec<u32> = samples.iter().map(|(_, d)| *d).collect();
    durations.sort_unstable();
    let median = u64::from(*durations.get(durations.len() / 2)?);
    durations.into_iter().rev().find(|&d| 2 * u64::from(d) <= 3 * median)
}

/// The number of leading packets whose end is nearest to `ticks`, and that
/// end. A tie keeps fewer packets.
fn nearest_packet_boundary(samples: &[(Vec<u8>, u32)], ticks: u64) -> (usize, u64) {
    let (mut best, mut best_at, mut at) = (0usize, 0u64, 0u64);
    for (i, (_, d)) in samples.iter().enumerate() {
        if at >= ticks {
            break;
        }
        at += u64::from(*d);
        if at.abs_diff(ticks) < best_at.abs_diff(ticks) {
            best = i + 1;
            best_at = at;
        }
    }
    (best, best_at)
}

// ---------------------------------------------------------------------------
// Audio preparation
// ---------------------------------------------------------------------------

pub(super) fn prepare_audio(
    track: Option<&AudioTrack>,
    // The source's audio edit list (`StreamingDemuxer::audio_edit`), in the
    // track's ticks: priming to hide, a trim, a late start.
    edit: Option<container::edit::AudioEdit>,
    // The holes in the track (`StreamingDemuxer::audio_gaps`): a passthrough
    // carries them in its durations, a decode fills them with silence.
    gaps: &[container::edit::AudioGap],
    policy: AudioCodecPolicy,
    bitrate: Option<u32>,
    filters: &[AudioFilter],
) -> Result<Option<PreparedAudio>> {
    let Some(track) = track else {
        return Ok(None);
    };
    if policy == AudioCodecPolicy::Drop {
        return Ok(None);
    }
    let codec = track.codec.to_ascii_lowercase();
    let passthrough_ok = matches!(codec.as_str(), "aac" | "opus" | "ac3" | "eac3" | "dts");
    let force_opus = policy == AudioCodecPolicy::ForceOpus;
    // A filter has to see PCM, so it forces the decode/encode path. Rather than
    // let a passthrough silently discard the user's `channelmap`, treat the
    // filter as an implicit request to transcode.
    let filtered = !filters.is_empty();

    if passthrough_ok && !filtered && !(force_opus && codec != "opus") {
        let info = passthrough_info(&codec, track);
        // The source's edit, applied exactly: whole packets outside it (beyond
        // the decoder's preroll) are dropped, and the output edit list hides
        // the rest of what the source hid.
        let (packets, out_edit) = match edit {
            Some(e) => {
                let preroll = container::edit::AudioPreroll::for_codec(&codec, track.timescale);
                let cut = container::edit::cut_audio_packets(&track.durations, &e, preroll);
                tracing::info!(
                    codec,
                    kept_from = cut.packets.start,
                    kept_to = cut.packets.end,
                    of = track.samples.len(),
                    edit = ?cut.edit,
                    "audio passthrough: source edit list applied"
                );
                (cut.packets, cut.edit)
            }
            None => (0..track.samples.len(), container::edit::TrackEdit::default()),
        };
        let samples = track.samples[packets.clone()]
            .iter()
            .cloned()
            .zip(track.durations[packets].iter().copied())
            .collect();
        return Ok(Some(PreparedAudio {
            info,
            samples,
            handling: format!("{codec} passthrough"),
            edit: out_edit,
        }));
    }

    // Codecs `codec::audio::create_decoder` can turn into PCM (linear PCM
    // included: it only needs converting).
    let decodable = matches!(codec.as_str(), "mp3" | "vorbis" | "dts" | "ac3" | "eac3")
        || codec::audio::decode::PcmFormat::from_codec(&codec).is_some();
    if decodable || force_opus || filtered {
        if !decodable {
            // No decoder for this source codec, so there's no PCM to re-encode
            // or filter. Say which knob went unhonoured — silently emitting an
            // unfiltered passthrough would be worse than dropping.
            if filtered {
                bail!(
                    "audio filters ({}) need a decodable track, but {codec} has no decoder in \
                     this build — it can only be passed through. Drop the audio filter, or \
                     supply a source whose audio is mp3/vorbis/dts/ac3/eac3/pcm.",
                    codec::audio::filter::chain_to_string(filters)
                );
            }
            tracing::warn!(codec, "cannot transcode to opus; dropping audio");
            return Ok(Some(dropped(codec)));
        }

        let extra: Option<&[u8]> =
            if track.codec_private.is_empty() { None } else { Some(track.codec_private.as_slice()) };
        let mut dec = audio_decoder(&codec, extra, track.sample_rate, track.channels as u8)
            .context("audio decoder")?;

        // The encoder is built before the first frame, so the channel count the
        // filter chain *will* produce has to be known up front.
        let out_channels = codec::audio::filter::output_channels(filters, track.channels as u8)
            .context("audio filter chain")?;
        let mut enc = audio_encoder(AudioEncoderConfig {
            codec: AudioCodec::Opus,
            sample_rate: track.sample_rate,
            channels: out_channels,
            // 0 = let the encoder derive it from the layout (64k per uncoupled
            // stream + 96k per coupled pair — 64k mono, 96k stereo, 320k 5.1).
            bitrate: bitrate.unwrap_or(0),
        })
        .context("opus encoder")?;

        let mut samples: Vec<(Vec<u8>, u32)> = Vec::new();
        let mut pts: i64 = 0;
        // The source's edit, applied to the decoded samples exactly: what it
        // hides is not encoded, nor anything past its end. Its delay goes to
        // the output's edit list.
        let mut window = edit.map(|e| PcmWindow::new(&e, track.timescale, track.sample_rate));
        // Samples (per channel, at the input rate) handed to the encoder: the
        // output's presented length.
        let mut encoded_samples: u64 = 0;
        let mut encode_frame = |enc: &mut Box<dyn codec::audio::AudioEncoder>,
                                frame: &codec::audio::AudioFrame,
                                out: &mut Vec<(Vec<u8>, u32)>|
         -> Result<()> {
            let presented;
            let frame = match window.as_mut() {
                None => frame,
                Some(w) => match w.take(frame) {
                    Some(f) => {
                        presented = f;
                        &presented
                    }
                    None => return Ok(()),
                },
            };
            let filtered = codec::audio::filter::apply_chain(frame, filters)
                .context("audio filter chain")?;
            encoded_samples += (filtered.samples.len() / usize::from(filtered.channels.max(1))) as u64;
            for pkt in enc.encode(&filtered).context("opus encode")? {
                out.push((pkt.data, pkt.duration as u32));
            }
            Ok(())
        };
        let mut holes = gaps.iter().peekable();
        for (index, packet) in track.samples.iter().enumerate() {
            let frames = match dec.decode(packet, pts) {
                Ok(frames) => frames,
                // The decoder exists but this stream uses a tool it refuses by
                // name (DTS: ADPCM prediction, whose code book ETSI does not
                // print). Same outcome as having no decoder at all: a filter
                // that needs PCM is an error, otherwise the track is dropped
                // with the reason rather than emitted wrong.
                Err(codec::audio::AudioError::Unsupported(reason)) => {
                    if filtered {
                        bail!(
                            "audio filters ({}) need the {codec} track decoded, which this build \
                             cannot do for this stream: {reason}",
                            codec::audio::filter::chain_to_string(filters)
                        );
                    }
                    tracing::warn!(codec, %reason, "cannot transcode to opus; dropping audio");
                    return Ok(Some(dropped(codec)));
                }
                Err(e) => return Err(e).context("audio decode"),
            };
            for frame in frames {
                pts = pts.saturating_add((frame.samples.len() as i64) / frame.channels.max(1) as i64);
                encode_frame(&mut enc, &frame, &mut samples)?;
            }
            // A hole the source's timestamps leave after this packet: as much
            // silence, so what follows plays where they put it.
            while let Some(hole) = holes.next_if(|h| h.after_packet == index) {
                let length =
                    container::edit::rescale_round(hole.ticks, track.sample_rate, track.timescale);
                let channels = track.channels as u8;
                let silence = codec::audio::AudioFrame {
                    samples: vec![0.0; length as usize * usize::from(channels)],
                    sample_rate: track.sample_rate,
                    channels,
                    pts,
                };
                pts = pts.saturating_add(length as i64);
                encode_frame(&mut enc, &silence, &mut samples)?;
            }
        }
        for frame in dec.flush().context("audio flush")? {
            encode_frame(&mut enc, &frame, &mut samples)?;
        }
        for pkt in enc.flush().context("opus encoder flush")? {
            samples.push((pkt.data, pkt.duration as u32));
        }
        let info = AudioInfo::opus(48_000, out_channels as u16, enc.extra_data());
        let handling = if out_channels as u16 == track.channels {
            format!("{codec} → opus ({out_channels}ch)")
        } else {
            format!("{codec} → opus ({}ch → {out_channels}ch)", track.channels)
        };
        // The samples are already cut to the source's edit, so what is left for
        // the output's is the source's delay, on the Opus clock, and the
        // encoder's own lookahead: the `dOps` PreSkip, hidden by `media_time`
        // as ffmpeg writes an Opus MP4 (without it the audio plays 6.5 ms
        // late in every player that honours the edit), ending after exactly
        // the samples that went in.
        let edit = container::edit::TrackEdit {
            delay: edit.map_or(0, |e| container::edit::rescale_round(e.delay, 48_000, track.timescale)),
            media_time: u64::from(enc.pre_skip()),
            duration: Some(container::edit::rescale_round(encoded_samples, 48_000, track.sample_rate)),
        };
        return Ok(Some(PreparedAudio { info, samples, handling, edit }));
    }

    Ok(Some(dropped(codec)))
}

/// A single-file job muxes one prepared track into every rung's MP4, whose
/// muxer checks a track before taking it
/// ([`Av1Mp4Muxer::check_audio`](container::mux::Av1Mp4Muxer::check_audio)).
/// A track it refuses leaves every file video-only, so the job reports the
/// audio dropped, with the reason, rather than passed through. HLS writes its
/// audio through the CMAF init segment, which takes any of these tracks.
pub(super) fn fit_single_file(audio: Option<PreparedAudio>) -> Option<PreparedAudio> {
    let a = audio?;
    if !a.has_samples() {
        return Some(a);
    }
    match container::mux::Av1Mp4Muxer::check_audio(&a.info) {
        Ok(()) => Some(a),
        Err(e) => {
            tracing::warn!(handling = %a.handling, "the MP4 muxer refuses this audio ({e:#}); video-only");
            Some(dropped(a.info.codec.to_ascii_lowercase()))
        }
    }
}

fn dropped(codec: String) -> PreparedAudio {
    PreparedAudio {
        info: AudioInfo::aac_lc(48_000, 2, Vec::new()),
        samples: Vec::new(),
        handling: format!("{codec} dropped"),
        edit: container::edit::TrackEdit::default(),
    }
}

/// The decoded samples an audio edit presents, by running position: the
/// transcode path's exact cut, where the passthrough path hides samples with an
/// edit list instead.
pub(super) struct PcmWindow {
    /// Samples (per channel) the edit hides at the start.
    skip: u64,
    /// Samples it presents after them; `None` = to the end.
    keep: Option<u64>,
    /// Samples seen so far.
    at: u64,
}

impl PcmWindow {
    /// The window for `edit` (ticks of `timescale`) over PCM at `sample_rate`.
    pub(super) fn new(edit: &container::edit::AudioEdit, timescale: u32, sample_rate: u32) -> Self {
        let samples = |ticks: u64| container::edit::rescale_round(ticks, sample_rate, timescale);
        let skip = samples(edit.media_start);
        Self { skip, keep: edit.media_end.map(|end| samples(end).saturating_sub(skip)), at: 0 }
    }

    /// The part of the next decoded `frame` inside the window, `None` when none is.
    pub(super) fn take(&mut self, frame: &codec::audio::AudioFrame) -> Option<codec::audio::AudioFrame> {
        let channels = usize::from(frame.channels.max(1));
        let start = self.at;
        let end = start + (frame.samples.len() / channels) as u64;
        self.at = end;
        let lo = self.skip.max(start);
        let hi = self.keep.map_or(end, |keep| (self.skip + keep).min(end));
        if lo >= hi {
            return None;
        }
        let (a, b) = ((lo - start) as usize * channels, (hi - start) as usize * channels);
        Some(codec::audio::AudioFrame {
            samples: frame.samples[a..b].to_vec(),
            sample_rate: frame.sample_rate,
            channels: frame.channels,
            pts: frame.pts,
        })
    }
}

fn passthrough_info(codec: &str, track: &AudioTrack) -> AudioInfo {
    match codec {
        "aac" => AudioInfo::aac_lc(track.sample_rate, track.channels, track.asc.clone()),
        "opus" => AudioInfo::opus(track.sample_rate, track.channels, track.codec_private.clone()),
        "ac3" => AudioInfo::ac3(track.sample_rate, track.channels, track.codec_private.clone()),
        "eac3" => AudioInfo::eac3(track.sample_rate, track.channels, track.codec_private.clone()),
        "dts" => AudioInfo::dts(track.sample_rate, track.channels, track.codec_private.clone()),
        _ => AudioInfo::aac_lc(track.sample_rate, track.channels, track.asc.clone()),
    }
}

// ---------------------------------------------------------------------------
// HLS audio rendition builder
// ---------------------------------------------------------------------------

pub(super) fn build_audio_rendition(
    asset_root: &Path,
    audio: &PreparedAudio,
    segment_seconds: f32,
) -> Result<Option<AudioVariantSpec>> {
    if !audio.has_samples() {
        return Ok(None);
    }
    let audio_dir = asset_root.join("audio");
    let seg_target_ticks = (segment_seconds as f64 * audio.info.timescale as f64).round() as u64;
    let mut muxer = CmafAudioMuxer::new(&audio_dir, audio.info.clone()).context("CmafAudioMuxer::new")?;
    muxer.set_edit(audio.edit).context("placing the HLS audio rendition on the source's audio edit")?;
    for (payload, dur) in &audio.samples {
        add_audio_sample_with_segment_flush(&mut muxer, payload.clone(), *dur, seg_target_ticks)?;
    }
    muxer.flush_segment().context("final audio flush_segment")?;
    let manifest = muxer.finalize().context("CmafAudioMuxer finalize")?;
    let codec_string = match audio.info.codec.as_str() {
        "opus" => "opus".to_string(),
        _ => codec::codec_strings::AAC_LC_CODEC_STRING.to_string(),
    };
    Ok(Some(AudioVariantSpec {
        codec_string,
        channels: audio.info.channels,
        sample_rate: audio.info.sample_rate,
        relative_dir: "audio".to_string(),
        language: "und".to_string(),
        name: "Audio".to_string(),
        manifest,
    }))
}
