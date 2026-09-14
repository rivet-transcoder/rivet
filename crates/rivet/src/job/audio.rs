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
    /// An output track has one edit list, and it can only describe the start
    /// and the end. An edit inside the join — this track ending before its
    /// samples do, or the next one hiding samples or starting late — is applied
    /// here to whole packets instead, as a trim's cut points are.
    pub(super) fn extend(&mut self, other: &PreparedAudio) {
        if self.edit.duration.is_none() && other.edit.is_identity() {
            self.samples.extend(other.samples.iter().cloned());
            return;
        }
        tracing::warn!(
            first = ?self.edit,
            next = ?other.edit,
            "splice: an audio edit inside the join is applied at packet granularity"
        );
        if let Some(presented) = self.edit.duration.take() {
            let end = self.edit.media_time + presented;
            let mut at = 0u64;
            self.samples.retain(|(_, d)| {
                let keep = at < end;
                at += u64::from(*d);
                keep
            });
        }
        let skip = other.edit.media_time;
        let other_end = other.edit.duration.map(|d| skip + d);
        let mut at = 0u64;
        for (payload, d) in &other.samples {
            let here = at;
            at += u64::from(*d);
            if at <= skip {
                continue;
            }
            if other_end.is_some_and(|e| here >= e) {
                break;
            }
            self.samples.push((payload.clone(), *d));
        }
    }
}

// ---------------------------------------------------------------------------
// Audio preparation
// ---------------------------------------------------------------------------

pub(super) fn prepare_audio(
    track: Option<&AudioTrack>,
    // The source's audio edit list (`StreamingDemuxer::audio_edit`), in the
    // track's ticks: priming to hide, a trim, a late start.
    edit: Option<container::edit::AudioEdit>,
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

    // Codecs `codec::audio::create_decoder` can turn into PCM.
    let decodable = matches!(codec.as_str(), "mp3" | "vorbis" | "dts" | "ac3" | "eac3");
    if decodable || force_opus || filtered {
        if !decodable {
            // No decoder for this source codec, so there's no PCM to re-encode
            // or filter. Say which knob went unhonoured — silently emitting an
            // unfiltered passthrough would be worse than dropping.
            if filtered {
                bail!(
                    "audio filters ({}) need a decodable track, but {codec} has no decoder in \
                     this build — it can only be passed through. Drop the audio filter, or \
                     supply a source whose audio is mp3/vorbis/dts/ac3/eac3.",
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
            for pkt in enc.encode(&filtered).context("opus encode")? {
                out.push((pkt.data, pkt.duration as u32));
            }
            Ok(())
        };
        for packet in &track.samples {
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
        // The samples are already cut to the edit; only its delay is left, on
        // the Opus clock.
        let edit = container::edit::TrackEdit {
            delay: edit.map_or(0, |e| container::edit::rescale_round(e.delay, 48_000, track.timescale)),
            ..Default::default()
        };
        return Ok(Some(PreparedAudio { info, samples, handling, edit }));
    }

    Ok(Some(dropped(codec)))
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
