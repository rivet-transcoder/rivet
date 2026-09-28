use std::path::Path;

use anyhow::{Context, Result, bail};

use codec::audio::filter::{AudioFilter, ChannelLayout};
use codec::audio::remix::Remixer;
use codec::audio::{
    AudioCodec, AudioEncoderConfig, create_decoder as audio_decoder,
    create_encoder as audio_encoder,
};
use container::cmaf::CmafAudioMuxer;
use container::demux::AudioTrack;
use container::hls::AudioVariantSpec;
use container::AudioInfo;

use crate::cmaf_util::add_audio_sample_with_segment_flush;
use crate::spec::{AudioBitDepth, AudioChannels, AudioCodecPolicy, Container, FlacLevel, OutputMode, OutputSpec};

// ---------------------------------------------------------------------------
// PreparedAudio
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(super) struct PreparedAudio {
    pub(super) info: AudioInfo,
    pub(super) samples: Vec<(Vec<u8>, u32)>,
    pub(super) handling: String,
    /// The MP3 encoder's name and version (`LAME3.100`) as a bare `.mp3`
    /// file's LAME tag states it: this job's encoder, or for an MP3
    /// passthrough the one the source's tag named. `None` otherwise.
    pub(super) encoder: Option<String>,
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
/// same number of samples (AAC, AC-3, E-AC-3, DTS, MP3): the longest packet
/// duration in the track, of those within half again of the median — a hole
/// in a transport stream's audio lengthens the packet before it by more than
/// half a frame ([`container::edit::AudioGap`]), and is not a frame. `None`
/// for Opus, whose packets legitimately vary.
fn fixed_frame_ticks(codec: &str, samples: &[(Vec<u8>, u32)]) -> Option<u32> {
    let fixed = ["aac", "ac3", "eac3", "dts", "mp3"].iter().any(|c| codec.eq_ignore_ascii_case(c));
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

/// Where the prepared track is going: what the container can carry decides
/// what may pass through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AudioOutput {
    /// A single-file MP4.
    Mp4,
    /// CMAF segments for HLS.
    Cmaf,
    /// A bare `.mp3` file.
    Mp3File,
    /// A native `.flac` stream.
    FlacFile,
}

/// What the spec asks of the audio track, and where it is going.
#[derive(Debug, Clone, Copy)]
pub(super) struct AudioRequest<'a> {
    pub(super) policy: AudioCodecPolicy,
    pub(super) bitrate: Option<u32>,
    pub(super) filters: &'a [AudioFilter],
    pub(super) channels: AudioChannels,
    pub(super) output: AudioOutput,
    /// FLAC / ALAC output depth.
    pub(super) bit_depth: AudioBitDepth,
    pub(super) flac_level: FlacLevel,
}

impl<'a> AudioRequest<'a> {
    pub(super) fn of(spec: &'a OutputSpec) -> Self {
        Self {
            policy: spec.audio,
            bitrate: spec.audio_bitrate,
            filters: &spec.audio_filters,
            channels: spec.audio_channels,
            output: match (&spec.mode, spec.container) {
                (OutputMode::SingleFile, _) => AudioOutput::Mp4,
                (OutputMode::Hls { .. }, _) => AudioOutput::Cmaf,
                // An `.m4a` takes what a single-file MP4 does.
                (OutputMode::AudioOnly, Container::M4a) => AudioOutput::Mp4,
                (OutputMode::AudioOnly, Container::Flac) => AudioOutput::FlacFile,
                (OutputMode::AudioOnly, _) => AudioOutput::Mp3File,
            },
            bit_depth: spec.audio_bit_depth,
            flac_level: spec.flac_level,
        }
    }

    /// `policy` into a single-file MP4, nothing else asked.
    #[cfg(test)]
    pub(super) fn plain(policy: AudioCodecPolicy) -> Self {
        Self {
            policy,
            bitrate: None,
            filters: &[],
            channels: AudioChannels::Source,
            output: AudioOutput::Mp4,
            bit_depth: AudioBitDepth::Source,
            flac_level: FlacLevel::Default,
        }
    }

    /// The codec `track` comes out as when it is transcoded. A lossless
    /// output's depth is the one asked for, else the source's (16 for 16
    /// bits or fewer and for a lossy source, 24 for anything deeper).
    fn encode_codec(&self, track: &AudioTrack) -> AudioCodec {
        let bits_per_sample = || {
            self.bit_depth.bits().unwrap_or(match source_bits(&track.codec.to_ascii_lowercase(), track) {
                Some(b) if b > 16 => 24,
                _ => 16,
            })
        };
        match self.policy {
            AudioCodecPolicy::Flac => AudioCodec::Flac { bits_per_sample: bits_per_sample(), level: self.flac_level },
            AudioCodecPolicy::Alac => AudioCodec::Alac { bits_per_sample: bits_per_sample() },
            AudioCodecPolicy::ForceMp3 => AudioCodec::Mp3,
            _ if self.output == AudioOutput::Mp3File => AudioCodec::Mp3,
            AudioCodecPolicy::ForceAac => AudioCodec::Aac,
            _ => AudioCodec::Opus,
        }
    }

    /// Whether `track` can go into this output as it is. Only the codec and
    /// the container are judged here; what the knobs ask for is not.
    fn carries(&self, track: &AudioTrack) -> bool {
        let codec = track.codec.to_ascii_lowercase();
        // MP4 takes MP3 at the MPEG-1 and MPEG-2 rates (object types 0x6B /
        // 0x69); MPEG-2.5's quarter rates have no object type.
        let mp3_in_mp4 = codec == "mp3" && track.sample_rate >= 16_000;
        match (self.policy, self.output) {
            // A lossless source in the codec asked for is copied, into any
            // output that holds it, unless a depth other than its own is
            // asked for. Its packets are timed in samples, so its track's
            // clock has to be its rate.
            (AudioCodecPolicy::Flac | AudioCodecPolicy::Alac, _) => {
                let wanted = if self.policy == AudioCodecPolicy::Flac { "flac" } else { "alac" };
                codec == wanted
                    && self.bit_depth.bits().is_none_or(|b| Some(b) == source_bits(&codec, track))
                    && track.timescale == track.sample_rate
            }
            (_, AudioOutput::FlacFile) => codec == "flac",
            (_, AudioOutput::Mp3File) => codec == "mp3",
            (AudioCodecPolicy::ForceMp3, AudioOutput::Mp4) => mp3_in_mp4,
            (AudioCodecPolicy::ForceOpus, _) => codec == "opus",
            (AudioCodecPolicy::ForceAac, _) => codec == "aac",
            // Auto: what plays or is kept verbatim on the web path. MP3 goes
            // into a single-file MP4 as it is (a re-encode would only lose
            // quality, and every browser plays it there); CMAF has no MP3.
            (_, AudioOutput::Mp4) => PASSTHROUGH.contains(&codec.as_str()) || mp3_in_mp4,
            (_, AudioOutput::Cmaf) => PASSTHROUGH.contains(&codec.as_str()),
        }
    }
}

/// Codecs a single-file MP4 or an HLS package carries verbatim.
const PASSTHROUGH: [&str; 5] = ["aac", "opus", "ac3", "eac3", "dts"];

pub(super) fn prepare_audio(
    track: Option<&AudioTrack>,
    // The source's audio edit list (`StreamingDemuxer::audio_edit`), in the
    // track's ticks: priming to hide, a trim, a late start.
    edit: Option<container::edit::AudioEdit>,
    // The holes in the track (`StreamingDemuxer::audio_gaps`): a passthrough
    // carries them in its durations, a decode fills them with silence.
    gaps: &[container::edit::AudioGap],
    req: AudioRequest<'_>,
) -> Result<Option<PreparedAudio>> {
    let Some(track) = track else {
        return Ok(None);
    };
    if req.policy == AudioCodecPolicy::Drop {
        return Ok(None);
    }
    let codec = track.codec.to_ascii_lowercase();
    let filters = req.filters;
    // A filter has to see PCM, so it forces the decode/encode path. Rather than
    // let a passthrough silently discard the user's `channelmap`, treat the
    // filter as an implicit request to transcode.
    let filtered = !filters.is_empty();
    // Codecs `codec::audio::create_decoder` can turn into PCM (linear PCM
    // included: it only needs converting).
    let decodable = matches!(codec.as_str(), "mp3" | "mp2" | "vorbis" | "dts" | "ac3" | "eac3" | "opus" | "flac" | "alac")
        || codec::audio::decode::PcmFormat::from_codec(&codec).is_some();
    let wanted = req.channels.layout();
    // rivet does not upmix. The container's count is enough to refuse here;
    // the decoded layout is checked again once it is known.
    if let Some(w) = &wanted
        && w.len() > usize::from(track.channels)
    {
        bail!(
            "audio-channels={} asks for {} channels of a {}-channel {codec} track: rivet does not upmix. \
             Use audio-channels=source, or a layout of at most {} channels",
            req.channels.as_str(),
            w.len(),
            track.channels,
            track.channels
        );
    }
    // The layout asked for is the source's own: nothing to convert.
    let channels_kept = wanted.as_ref().is_none_or(|w| w.len() == usize::from(track.channels));
    let target = req.encode_codec(track);
    let target_name = match target {
        AudioCodec::Opus => "opus",
        AudioCodec::Mp3 => "mp3",
        AudioCodec::Aac => "aac",
        AudioCodec::Flac { .. } => "flac",
        AudioCodec::Alac { .. } => "alac",
    };
    // The codec asked for, but this source cannot be decoded (AAC has no
    // decoder in this build): keeping the source's audio beats emitting none,
    // where the output can hold it. This used to fall through to "dropping
    // audio", so every Opus request for an AAC source came out silent.
    let forced = matches!(
        req.policy,
        AudioCodecPolicy::ForceOpus
            | AudioCodecPolicy::ForceMp3
            | AudioCodecPolicy::ForceAac
            | AudioCodecPolicy::Flac
            | AudioCodecPolicy::Alac
    );
    let unreachable = forced
        && codec != target_name
        && !decodable
        && !matches!(req.output, AudioOutput::Mp3File | AudioOutput::FlacFile)
        && AudioRequest { policy: AudioCodecPolicy::Auto, ..req }.carries(track);

    if !filtered && channels_kept && (req.carries(track) || unreachable) {
        if unreachable {
            tracing::warn!(
                codec,
                "{target_name} requested but {codec} cannot be decoded in this build; passing the source audio through"
            );
        }
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
            handling: if unreachable {
                format!("{codec} passthrough ({target_name} requested; no {codec} decoder)")
            } else {
                format!("{codec} passthrough")
            },
            // A bare MP3's LAME tag named its encoder (`container::mp3`): kept,
            // so a passthrough into another `.mp3` states the same gapless
            // delay under the same name.
            encoder: (codec == "mp3" && !track.codec_private.is_empty())
                .then(|| String::from_utf8_lossy(&track.codec_private).into_owned()),
            edit: out_edit,
        }));
    }

    if !decodable {
        // No decoder for this source codec, so there's no PCM to re-encode,
        // filter or remix. Say which knob went unhonoured — silently emitting
        // an unfiltered passthrough would be worse than dropping.
        if filtered {
            bail!(
                "audio filters ({}) need a decodable track, but {codec} has no decoder in \
                 this build — it can only be passed through. Drop the audio filter, or \
                 supply a source whose audio is mp3/vorbis/opus/dts/ac3/eac3/pcm.",
                codec::audio::filter::chain_to_string(filters)
            );
        }
        if !channels_kept {
            bail!(
                "audio-channels={} needs the {}-channel {codec} track decoded to convert it, but {codec} \
                 has no decoder in this build — it can only be passed through as it is. Use \
                 audio-channels=source.",
                req.channels.as_str(),
                track.channels
            );
        }
        if req.output == AudioOutput::Mp3File {
            bail!(
                "an .mp3 file holds MP3, and this {codec} track can be neither passed into it nor \
                 decoded to encode it ({codec} has no decoder in this build)"
            );
        }
        if req.output == AudioOutput::FlacFile {
            bail!(
                "a native FLAC file holds FLAC, and this {codec} track cannot be decoded to encode it \
                 ({codec} has no decoder in this build)"
            );
        }
        tracing::warn!(codec, "cannot transcode to {target_name}; dropping audio");
        return Ok(Some(dropped(codec)));
    }

    let extra: Option<&[u8]> =
        if track.codec_private.is_empty() { None } else { Some(track.codec_private.as_slice()) };
    let mut dec =
        audio_decoder(&codec, extra, track.sample_rate, track.channels as u8).context("audio decoder")?;
    // Opus decodes at 48 kHz whatever the container says the input was.
    let pcm_rate = if codec == "opus" { 48_000 } else { track.sample_rate };
    // A decoded Opus stream starts with its pre-skip, which an MP4 edit list
    // hides (and is in `edit` then); a container that states none (Matroska
    // keeps it in `CodecDelay`, read nowhere) hides it by the `OpusHead`.
    let edit = edit.or_else(|| opus_pre_skip_edit(&codec, track));
    let mut state = EncodeState::new(req, target);
    // The source's edit, applied to the decoded samples exactly: what it
    // hides is not encoded, nor anything past its end. Its delay goes to the
    // output's edit list.
    let mut window = edit.map(|e| PcmWindow::new(&e, track.timescale, pcm_rate));
    let mut samples: Vec<(Vec<u8>, u32)> = Vec::new();
    let mut pts: i64 = 0;
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
                tracing::warn!(codec, %reason, "cannot transcode to {target_name}; dropping audio");
                return Ok(Some(dropped(codec)));
            }
            Err(e) => return Err(e).context("audio decode"),
        };
        let layout = dec.layout();
        for frame in frames {
            pts = pts.saturating_add((frame.samples.len() as i64) / frame.channels.max(1) as i64);
            let frame = match window.as_mut() {
                None => frame,
                Some(w) => match w.take(&frame) {
                    Some(f) => f,
                    None => continue,
                },
            };
            state.encode(&frame, layout.clone(), &mut samples)?;
        }
        // A hole the source's timestamps leave after this packet: as much
        // silence, so what follows plays where they put it.
        while let Some(hole) = holes.next_if(|h| h.after_packet == index) {
            let length = container::edit::rescale_round(hole.ticks, pcm_rate, track.timescale);
            let (channels, layout) = state.last_input(track.channels as u8);
            let silence = codec::audio::AudioFrame {
                samples: vec![0.0; length as usize * usize::from(channels)],
                sample_rate: pcm_rate,
                channels,
                pts,
            };
            pts = pts.saturating_add(length as i64);
            state.encode(&silence, layout, &mut samples)?;
        }
    }
    let layout = dec.layout();
    for frame in dec.flush().context("audio flush")? {
        let frame = match window.as_mut() {
            None => frame,
            Some(w) => match w.take(&frame) {
                Some(f) => f,
                None => continue,
            },
        };
        state.encode(&frame, layout.clone(), &mut samples)?;
    }
    let Some(done) = state.finish(&mut samples)? else {
        tracing::warn!(codec, "the {codec} track decoded to no audio; dropping it");
        return Ok(Some(dropped(codec)));
    };
    let depth = match target {
        AudioCodec::Flac { bits_per_sample, .. } | AudioCodec::Alac { bits_per_sample } => {
            format!(", {bits_per_sample}-bit")
        }
        _ => String::new(),
    };
    let handling = if done.out_layout.len() == done.in_layout.len() {
        format!("{codec} → {target_name} ({}ch{depth})", done.out_layout.len())
    } else {
        format!("{codec} → {target_name} ({}ch → {}ch{depth})", done.in_layout.len(), done.out_layout.len())
    };
    if done.in_layout != done.out_layout {
        tracing::info!(from = %done.in_layout, to = %done.out_layout, codec, "audio channel layout converted");
    }
    // The samples are already cut to the source's edit, so what is left for
    // the output's is the source's delay, on the output clock, and the
    // encoder's own lead-in: Opus's `dOps` PreSkip (without it the audio plays
    // 6.5 ms late in every player that honours the edit, and ffmpeg writes an
    // Opus MP4 the same way), MP3's encoder and decoder delay. It ends after
    // exactly the samples that went in.
    let out_rate = done.out_rate;
    let edit = container::edit::TrackEdit {
        delay: edit.map_or(0, |e| container::edit::rescale_round(e.delay, out_rate, track.timescale)),
        media_time: u64::from(done.pre_skip),
        duration: Some(container::edit::rescale_round(done.encoded_samples, out_rate, done.in_rate)),
    };
    Ok(Some(PreparedAudio { info: done.info, samples, handling, encoder: done.encoder, edit }))
}

/// The edit that hides a decoded Opus track's pre-skip, from its `OpusHead`,
/// for a track whose container states no edit.
fn opus_pre_skip_edit(codec: &str, track: &AudioTrack) -> Option<container::edit::AudioEdit> {
    if codec != "opus" {
        return None;
    }
    let head = codec::audio::decode::opus::OpusHead::parse(&track.codec_private).ok()?;
    (head.pre_skip > 0).then(|| container::edit::AudioEdit {
        delay: 0,
        media_start: container::edit::rescale_round(u64::from(head.pre_skip), track.timescale, 48_000),
        media_end: None,
    })
}

/// The encoder side of a transcode, built on the first frame: the layout the
/// source turns out to have decides the output's, and the rate it decodes
/// at is the encoder's input.
struct EncodeState<'a> {
    req: AudioRequest<'a>,
    codec: AudioCodec,
    enc: Option<Box<dyn codec::audio::AudioEncoder>>,
    /// The layout the encoder takes.
    out_layout: Option<ChannelLayout>,
    /// The first frame's layout, after the filters (for the report).
    in_layout: Option<ChannelLayout>,
    /// Converts the current input layout to `out_layout`; rebuilt when a
    /// stream changes layout mid-way (an AC-3 programme change).
    remix: Option<Remixer>,
    in_rate: u32,
    /// Samples (per channel, at `in_rate`) handed to the encoder.
    encoded_samples: u64,
    /// The last frame's width and layout, for silence that fills a hole.
    last: Option<(u8, Option<ChannelLayout>)>,
}

/// A finished transcode.
struct Encoded {
    info: AudioInfo,
    in_layout: ChannelLayout,
    out_layout: ChannelLayout,
    in_rate: u32,
    out_rate: u32,
    pre_skip: u16,
    encoded_samples: u64,
    encoder: Option<String>,
}

impl<'a> EncodeState<'a> {
    fn new(req: AudioRequest<'a>, codec: AudioCodec) -> Self {
        Self {
            req,
            codec,
            enc: None,
            out_layout: None,
            in_layout: None,
            remix: None,
            in_rate: 0,
            encoded_samples: 0,
            last: None,
        }
    }

    /// The width and layout of the frames seen last, or of the track.
    fn last_input(&self, track_channels: u8) -> (u8, Option<ChannelLayout>) {
        self.last.clone().unwrap_or((track_channels, None))
    }

    /// Filter, remix and encode one decoded frame whose speakers are
    /// `layout` (`None`: the default for its width).
    fn encode(
        &mut self,
        frame: &codec::audio::AudioFrame,
        layout: Option<ChannelLayout>,
        out: &mut Vec<(Vec<u8>, u32)>,
    ) -> Result<()> {
        self.last = Some((frame.channels, layout.clone()));
        let filters = self.req.filters;
        let filtered = codec::audio::filter::apply_chain(frame, filters).context("audio filter chain")?;
        // The speakers the filtered frame carries: the chain's own output
        // layout when it names one, else the decoder's, else the default.
        let source = match codec::audio::filter::output_layout(filters, frame.channels)
            .context("audio filter chain")?
        {
            Some(l) => l,
            None => match layout {
                Some(l) if l.len() == usize::from(filtered.channels) => l,
                _ => ChannelLayout::default_for(filtered.channels).context("audio channel layout")?,
            },
        };
        if self.enc.is_none() {
            let out_layout = self.output_layout(&source)?;
            let enc = audio_encoder(AudioEncoderConfig {
                codec: self.codec,
                sample_rate: filtered.sample_rate,
                channels: out_layout.len() as u8,
                // 0 = let the encoder derive it from the layout: for Opus 64k
                // per uncoupled stream + 96k per coupled pair (64k mono, 96k
                // stereo, 320k 5.1, 416k 7.1); for MP3 128k stereo, 64k mono;
                // for AAC 64k mono, 128k stereo, 384k 5.1, 512k 7.1.
                bitrate: self.req.bitrate.unwrap_or(0),
            })
            .with_context(|| format!("{:?} encoder", self.codec))?;
            self.enc = Some(enc);
            self.in_rate = filtered.sample_rate;
            self.in_layout = Some(source.clone());
            self.out_layout = Some(out_layout);
        }
        let out_layout = self.out_layout.clone().expect("set with the encoder");
        if self.remix.as_ref().is_none_or(|r| *r.from() != source) {
            if self.remix.is_some() {
                tracing::info!(layout = %source, "audio: the source changed layout mid-stream");
            }
            self.remix = Some(Remixer::new(source, out_layout));
        }
        let remix = self.remix.as_ref().expect("set above");
        let remixed = remix.apply(&filtered)?;
        self.encoded_samples += (remixed.samples.len() / usize::from(remixed.channels.max(1))) as u64;
        let enc = self.enc.as_mut().expect("built above");
        for pkt in enc.encode(&remixed).with_context(|| format!("{:?} encode", self.codec))? {
            out.push((pkt.data, pkt.duration as u32));
        }
        Ok(())
    }

    /// The layout the encoder is built for, from the first frame's.
    fn output_layout(&self, source: &ChannelLayout) -> Result<ChannelLayout> {
        match self.req.channels.layout() {
            Some(wanted) => {
                if wanted.len() > source.len() {
                    bail!(
                        "audio-channels={} asks for {} channels of a {source} ({}-channel) source: rivet \
                         does not upmix. Use audio-channels=source, or a layout of at most {} channels",
                        self.req.channels.as_str(),
                        wanted.len(),
                        source.len(),
                        source.len()
                    );
                }
                Ok(wanted)
            }
            None => match self.codec {
                AudioCodec::Opus => codec::audio::remix::opus_layout(source).with_context(|| {
                    format!("no Opus channel mapping carries a {source} source; set audio-channels")
                }),
                AudioCodec::Mp3 => Ok(codec::audio::remix::mp3_layout(source)),
                AudioCodec::Aac => codec::audio::remix::aac_layout(source).with_context(|| {
                    format!("no AAC channel configuration carries a {source} source; set audio-channels")
                }),
                // FLAC and ALAC carry any layout of up to eight channels as
                // it is: a lossless output changes nothing it need not.
                AudioCodec::Flac { .. } | AudioCodec::Alac { .. } => {
                    if source.len() > 8 {
                        bail!("FLAC and ALAC carry at most 8 channels; this {source} source has {}", source.len());
                    }
                    Ok(source.clone())
                }
            },
        }
    }

    fn finish(mut self, out: &mut Vec<(Vec<u8>, u32)>) -> Result<Option<Encoded>> {
        let Some(mut enc) = self.enc.take() else {
            return Ok(None);
        };
        for pkt in enc.flush().with_context(|| format!("{:?} encoder flush", self.codec))? {
            out.push((pkt.data, pkt.duration as u32));
        }
        let out_layout = self.out_layout.expect("set with the encoder");
        let channels = out_layout.len() as u16;
        let (info, encoder) = match self.codec {
            AudioCodec::Opus => (AudioInfo::opus(self.in_rate, channels, enc.extra_data()), None),
            AudioCodec::Mp3 => (AudioInfo::mp3(enc.sample_rate(), channels), codec::audio::mp3_encoder_name()),
            AudioCodec::Aac => (AudioInfo::aac_lc(enc.sample_rate(), channels, enc.extra_data()), None),
            AudioCodec::Flac { .. } => (AudioInfo::flac(enc.sample_rate(), channels, enc.extra_data()), None),
            AudioCodec::Alac { .. } => (AudioInfo::alac(enc.sample_rate(), channels, enc.extra_data()), None),
        };
        Ok(Some(Encoded {
            info,
            in_layout: self.in_layout.expect("set with the encoder"),
            out_layout,
            in_rate: self.in_rate,
            out_rate: enc.sample_rate(),
            pre_skip: enc.pre_skip(),
            encoded_samples: self.encoded_samples,
            encoder,
        }))
    }
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
        encoder: None,
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
        "mp3" => AudioInfo::mp3(track.sample_rate, track.channels),
        "dts" => AudioInfo::dts(track.sample_rate, track.channels, track.codec_private.clone()),
        "flac" => AudioInfo::flac(track.sample_rate, track.channels, track.codec_private.clone()),
        "alac" => AudioInfo::alac(track.sample_rate, track.channels, track.codec_private.clone()),
        _ => AudioInfo::aac_lc(track.sample_rate, track.channels, track.asc.clone()),
    }
}

/// The integer depth a source's audio has, when it has one: a lossless or
/// PCM source's own (float PCM counts as 24), `None` for a lossy codec.
fn source_bits(codec: &str, track: &AudioTrack) -> Option<u8> {
    use codec::audio::lossless::{alac::AlacConfig, flac::stream_info_from_extra};
    match codec {
        "flac" => stream_info_from_extra(&track.codec_private).ok().map(|i| i.bits_per_sample),
        "alac" => AlacConfig::parse(&track.codec_private).ok().map(|c| c.bit_depth),
        "pcm_u8" => Some(8),
        "pcm_s16le" => Some(16),
        "pcm_s24le" | "pcm_f32le" | "pcm_f64le" => Some(24),
        "pcm_s32le" => Some(32),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// HLS audio rendition builder
// ---------------------------------------------------------------------------

pub(super) fn build_audio_rendition(
    asset_root: &Path,
    audio: &PreparedAudio,
    segment_seconds: f32,
    // Where under the asset root, and the rendition's NAME.
    relative_dir: &str,
    name: &str,
) -> Result<Option<AudioVariantSpec>> {
    if !audio.has_samples() {
        return Ok(None);
    }
    let audio_dir = asset_root.join(relative_dir);
    let seg_target_ticks = (segment_seconds as f64 * audio.info.timescale as f64).round() as u64;
    let mut muxer = CmafAudioMuxer::new(&audio_dir, audio.info.clone()).context("CmafAudioMuxer::new")?;
    muxer.set_edit(audio.edit).context("placing the HLS audio rendition on the source's audio edit")?;
    for (payload, dur) in &audio.samples {
        add_audio_sample_with_segment_flush(&mut muxer, payload.clone(), *dur, seg_target_ticks)?;
    }
    muxer.flush_segment().context("final audio flush_segment")?;
    let manifest = muxer.finalize().context("CmafAudioMuxer finalize")?;
    Ok(Some(AudioVariantSpec {
        codec_string: audio_codec_string(&audio.info),
        channels: audio.info.channels,
        sample_rate: audio.info.sample_rate,
        relative_dir: relative_dir.to_string(),
        language: "und".to_string(),
        name: name.to_string(),
        manifest,
    }))
}

/// The RFC 6381 `codecs` value for a prepared track, the one an HLS
/// `CODECS` attribute and a `<source type>` carry: from the AAC
/// AudioSpecificConfig's object type (`mp4a.40.2` LC, `.5` HE-AAC, `.29`
/// HE-AAC v2, `.42` xHE-AAC), `opus`, `ac-3` / `ec-3` (the sample entries'
/// four-character codes, as Apple's HLS authoring spec writes them), `dtsc`,
/// and `mp3` for MP3 ([`container::mux::MP3_CODEC_STRING`] says why not
/// `mp4a.6B`). Every AAC track used to be called `mp4a.40.2`, and an AC-3
/// or E-AC-3 passthrough was too.
pub(super) fn audio_codec_string(info: &AudioInfo) -> String {
    match info.codec.to_ascii_lowercase().as_str() {
        "opus" => "opus".into(),
        "ac3" => "ac-3".into(),
        "eac3" => "ec-3".into(),
        "dts" => "dtsc".into(),
        "mp3" => container::mux::MP3_CODEC_STRING.into(),
        // The HLS authoring specification's values for lossless audio in fMP4.
        "flac" => "fLaC".into(),
        "alac" => "alac".into(),
        _ => {
            use container::aac_asc::AscSignaling;
            let aot = container::aac_asc::parse_aac_asc(&info.asc_bytes).map(|a| match a.signaling {
                AscSignaling::ExplicitSbr => 5,
                AscSignaling::ExplicitPs => 29,
                _ => a.aot,
            });
            match aot {
                Some(aot) => format!("mp4a.40.{aot}"),
                None => codec::codec_strings::AAC_LC_CODEC_STRING.to_string(),
            }
        }
    }
}
