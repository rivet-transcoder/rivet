//! The single-file muxer a rung writes into, by container: the MP4 muxer for
//! an MP4 or (under the `qt  ` brand) a QuickTime movie, the WebM muxer for a
//! WebM file. One face for the serial and the stitched paths, so neither
//! knows which file it is writing.

use anyhow::{Context, Result};
use codec::encode::EncodedPacket;
use codec::frame::{ColorMetadata, VideoCodec};
use container::demux::subtitle::SubtitleTrack;
use container::mux::Av1Mp4Muxer;
use container::webm::WebmMuxer;

use crate::spec::Container;

pub(super) enum FileMuxer {
    Mp4(Box<Av1Mp4Muxer>),
    WebM(Box<WebmMuxer>),
}

impl FileMuxer {
    /// A muxer for `container`. `inline` keeps H.264 / H.265 parameter sets
    /// in band (`avc3` / `hev1`), for stitched chunks whose sets differ.
    pub(super) fn new(
        container: Container,
        width: u32,
        height: u32,
        frame_rate: f64,
        codec: VideoCodec,
        inline: bool,
    ) -> Result<Self> {
        Ok(match container {
            Container::WebM => FileMuxer::WebM(Box::new(
                WebmMuxer::new(width, height, frame_rate, codec).context("WebmMuxer::new")?,
            )),
            _ => {
                let mut m = if inline {
                    Av1Mp4Muxer::new_with_codec_inline(width, height, frame_rate, codec)
                        .context("Av1Mp4Muxer::new_with_codec_inline")?
                } else {
                    Av1Mp4Muxer::new_with_codec(width, height, frame_rate, codec).context("Av1Mp4Muxer::new_with_codec")?
                };
                m.set_quicktime(container == Container::Mov);
                FileMuxer::Mp4(Box::new(m))
            }
        })
    }

    pub(super) fn set_color_metadata(&mut self, color: ColorMetadata) {
        match self {
            FileMuxer::Mp4(m) => {
                m.set_color_metadata(color);
            }
            FileMuxer::WebM(m) => {
                m.set_color_metadata(color);
            }
        }
    }

    pub(super) fn set_video_delay(&mut self, delay: u64, timescale: u32) {
        match self {
            FileMuxer::Mp4(m) => {
                m.set_video_delay(delay, timescale);
            }
            FileMuxer::WebM(m) => {
                m.set_video_delay(delay, timescale);
            }
        }
    }

    /// Add the prepared audio track, or log and go video-only when the file
    /// refuses it (the job's audio routing has already chosen what each file
    /// takes, so this is a backstop).
    pub(super) fn add_audio(&mut self, audio: &super::audio::PreparedAudio, label: &str) -> Result<()> {
        match self {
            FileMuxer::Mp4(m) => {
                if let Err(e) = m.with_audio(audio.info.clone()) {
                    tracing::warn!(rung = %label, "audio rejected ({e}); video-only");
                    return Ok(());
                }
                m.set_audio_edit(audio.edit);
                for (sample, dur) in &audio.samples {
                    m.add_audio_sample(sample, 0, *dur).context("add_audio_sample")?;
                }
            }
            FileMuxer::WebM(m) => {
                if let Err(e) = m.with_audio(audio.info.clone()) {
                    tracing::warn!(rung = %label, "audio rejected ({e}); video-only");
                    return Ok(());
                }
                m.set_audio_edit(audio.edit);
                for (sample, dur) in &audio.samples {
                    m.add_audio_sample(sample, *dur).context("add_audio_sample")?;
                }
            }
        }
        Ok(())
    }

    /// Attach the selected text subtitle tracks, one `tx3g` track each in an
    /// MP4 / QuickTime movie. A rejection is logged and the rung continues
    /// without that track. A WebM file carries none: rivet writes no WebVTT
    /// track into Matroska, and says so.
    pub(super) fn attach_subtitles(&mut self, subtitles: &[SubtitleTrack], label: &str) {
        match self {
            FileMuxer::Mp4(m) => {
                for s in subtitles {
                    if let Err(e) = m.add_subtitle_track(&s.cues, s.timescale, &s.language) {
                        tracing::warn!(
                            rung = %label,
                            language = %s.language,
                            "subtitle track rejected ({e}); continuing without it"
                        );
                    }
                }
            }
            FileMuxer::WebM(_) => {
                if !subtitles.is_empty() {
                    tracing::warn!(
                        rung = %label,
                        tracks = subtitles.len(),
                        "a WebM output carries no subtitle track; the source's text subtitles are dropped"
                    );
                }
            }
        }
    }

    pub(super) fn add_packet(&mut self, packet: EncodedPacket) -> Result<()> {
        match self {
            FileMuxer::Mp4(m) => m.add_packet(packet),
            FileMuxer::WebM(m) => m.add_packet(packet),
        }
    }

    pub(super) fn finalize(self) -> Result<Vec<u8>> {
        match self {
            FileMuxer::Mp4(m) => Ok(m.finalize().context("finalize")?.to_vec()),
            FileMuxer::WebM(m) => m.finalize().context("finalize"),
        }
    }
}
