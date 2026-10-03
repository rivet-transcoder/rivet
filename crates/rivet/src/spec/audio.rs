//! The lossless audio knobs and the audio-only output's file: builders and
//! the coherence checks [`OutputSpec::validate`] makes of them.

use anyhow::{Result, bail};

use super::{AudioBitDepth, AudioCodecPolicy, Container, FlacLevel, Muxer, OutputMode, OutputSpec};

impl OutputSpec {
    /// The audio alone, in the file `container` names: [`Container::Mp3`]
    /// (what [`Self::audio_only`] builds), [`Container::Flac`],
    /// [`Container::M4a`] or [`Container::Ogg`].
    pub fn audio_only_in(container: Container) -> Self {
        let muxer = match container {
            Container::Flac => Muxer::FlacFile,
            Container::M4a => Muxer::M4aFile,
            Container::Ogg => Muxer::OggFile,
            _ => Muxer::Mp3File,
        };
        Self { container, muxer, ..Self::audio_only() }
    }

    /// The file an audio-only output of `policy` is, unless one is named:
    /// a native `.flac` for FLAC, an `.ogg` for Opus and Vorbis, an `.m4a`
    /// for ALAC, the AAC profiles, AC-3, E-AC-3 and DTS, else an `.mp3`.
    pub fn audio_only_container(policy: AudioCodecPolicy) -> Container {
        use AudioCodecPolicy::*;
        match policy {
            Flac => Container::Flac,
            ForceOpus | ForceVorbis => Container::Ogg,
            Alac | ForceAac | ForceHeAac | ForceHeAacV2 | ForceAc3 | ForceEac3 | ForceDts => Container::M4a,
            Auto | ForceMp3 | Drop => Container::Mp3,
        }
    }

    /// Set the bit depth of FLAC / ALAC output.
    pub fn with_audio_bit_depth(mut self, depth: AudioBitDepth) -> Self {
        self.audio_bit_depth = depth;
        self
    }

    /// Set the FLAC compression effort.
    pub fn with_flac_level(mut self, level: FlacLevel) -> Self {
        self.flac_level = level;
        self
    }

    /// The extension a single-file output is written with: `mp4`, or for
    /// audio-only output `mp3`, `flac`, `m4a`, `ogg` (or `opus` for Opus).
    pub fn file_extension(&self) -> &'static str {
        match (&self.mode, self.container) {
            (OutputMode::AudioOnly, Container::Flac) => "flac",
            (OutputMode::AudioOnly, Container::M4a) => "m4a",
            (OutputMode::AudioOnly, Container::Ogg) if self.audio == AudioCodecPolicy::ForceOpus => "opus",
            (OutputMode::AudioOnly, Container::Ogg) => "ogg",
            (OutputMode::AudioOnly, _) => "mp3",
            (_, Container::Mov) => "mov",
            (_, Container::WebM) => "webm",
            _ => "mp4",
        }
    }

    /// The lossless knobs against the audio policy and the output: each
    /// applies to one codec, a lossless codec has no bitrate, and each
    /// audio-only file holds what it can.
    pub(crate) fn check_lossless_audio(&self) -> Result<()> {
        let lossless = self.audio.is_lossless();
        if lossless && self.audio_bitrate.is_some() {
            bail!(
                "an audio bitrate was given, but audio={} is lossless: its size follows the audio. \
                 Drop audio-bitrate",
                if self.audio == AudioCodecPolicy::Flac { "flac" } else { "alac" }
            );
        }
        if !lossless && self.audio_bit_depth != AudioBitDepth::Source {
            bail!("audio-bit-depth applies to FLAC and ALAC output (audio=flac|alac)");
        }
        if self.audio != AudioCodecPolicy::Flac && self.flac_level != FlacLevel::Default {
            bail!("flac-compression applies to FLAC output (audio=flac)");
        }
        if self.mode == OutputMode::AudioOnly {
            match self.container {
                Container::Flac if self.audio != AudioCodecPolicy::Flac => bail!(
                    "a native FLAC file holds FLAC only; set audio=flac, or audio-container=mp4 for an .m4a"
                ),
                Container::Mp3 if lossless => bail!(
                    "an .mp3 file cannot hold lossless audio; set audio-container=flac (FLAC) or mp4"
                ),
                _ => {}
            }
        }
        Ok(())
    }
}
