//! AC-3 / E-AC-3 decode through the workspace's AC-3 decoder (`ac3`, the
//! `crates/ac3` submodule, the rivet-ac3 repository), adapted to
//! [`AudioDecoder`].
//!
//! - Packets hold one or more whole or partial syncframes (MP4 / Matroska
//!   samples, TS PES payloads, a raw .ac3 / .eac3 file in any chunking):
//!   `ac3::Decoder` resynchronises on 0x0B77, buffers a partial syncframe and
//!   skips a damaged one; the adapter stamps each frame from the first
//!   packet's pts plus the samples decoded since.
//! - AC-3 in full; E-AC-3 independent substream 0 with its dependent
//!   substreams, whose channels replace or supplement substream 0's (ETSI
//!   TS 102 366 §E.2.8.2), so 7.1 (a 5.1 downmix in substream 0, a
//!   dependent substream with the discrete surrounds) decodes as eight
//!   channels. Enhanced coupling and
//!   bsid 9 / 10 are [`AudioError::Unsupported`].
//! - Output is in the native order for the layout (5.1: FL FR FC LFE SL SR;
//!   7.1: FL FR FC LFE BL BR SL SR), which [`AudioDecoder::layout`] names.
//!   No downmix here — `channelmap` does that on PCM.

use crate::audio::filter::{ChannelLabel, ChannelLayout};
use crate::audio::{AudioDecoder, AudioError, AudioFrame};

pub use ac3::{Features, FrameDecoder, Header, Options as Ac3Options, frame_crc_ok, parse_header};

fn decode_error(e: ac3::Error) -> AudioError {
    match e {
        ac3::Error::Decode(m) => AudioError::Decode(m),
        ac3::Error::Unsupported(m) => AudioError::Unsupported(m),
        ac3::Error::InvalidInput(m) => AudioError::Decode(m),
    }
}

/// The rivet label of a speaker the decoder names.
fn label(s: ac3::Speaker) -> ChannelLabel {
    match s {
        ac3::Speaker::FL => ChannelLabel::FL,
        ac3::Speaker::FR => ChannelLabel::FR,
        ac3::Speaker::FC => ChannelLabel::FC,
        ac3::Speaker::LFE => ChannelLabel::LFE,
        ac3::Speaker::BC => ChannelLabel::BC,
        ac3::Speaker::SL => ChannelLabel::SL,
        ac3::Speaker::SR => ChannelLabel::SR,
        ac3::Speaker::BL => ChannelLabel::BL,
        ac3::Speaker::BR => ChannelLabel::BR,
    }
}

/// The layout a syncframe header decodes to (its own substream).
pub fn layout(h: &Header) -> ChannelLayout {
    speakers_layout(&h.speakers())
}

fn speakers_layout(speakers: &[ac3::Speaker]) -> ChannelLayout {
    ChannelLayout::new(speakers.iter().copied().map(label).collect()).expect("distinct speakers")
}

/// [`AudioDecoder`] adapter: takes packets that hold one or more whole or
/// partial syncframes, emits one [`AudioFrame`] per syncframe (an E-AC-3
/// one with its dependent substreams).
pub struct Ac3Decoder {
    inner: ac3::Decoder,
    declared_sample_rate: u32,
    declared_channels: u8,
    next_pts_us: Option<i64>,
    warned_layout: bool,
    layout: Option<ChannelLayout>,
}

impl Ac3Decoder {
    pub fn new(sample_rate: u32, channels: u8) -> Result<Self, AudioError> {
        Self::with_options(sample_rate, channels, Ac3Options::default())
    }

    pub fn with_options(sample_rate: u32, channels: u8, opts: Ac3Options) -> Result<Self, AudioError> {
        if channels > 8 {
            return Err(AudioError::Unsupported(format!(
                "ac3: {channels} channels — AC-3 carries at most 6, E-AC-3 here at most 8 (7.1)"
            )));
        }
        Ok(Self {
            inner: ac3::Decoder::with_options(opts),
            declared_sample_rate: sample_rate,
            declared_channels: channels,
            next_pts_us: None,
            warned_layout: false,
            layout: None,
        })
    }

    /// The header of the most recent syncframe decoded, if any.
    pub fn last_header(&self) -> Option<Header> {
        self.inner.last_header()
    }

    fn frames(&mut self, frames: Vec<ac3::Frame>) -> Vec<AudioFrame> {
        frames
            .into_iter()
            .map(|f| {
                if !self.warned_layout && self.declared_channels != 0 && usize::from(self.declared_channels) != f.channels {
                    tracing::warn!(
                        declared = self.declared_channels,
                        stream = f.channels,
                        "ac3: container channel count differs from the bitstream; using the bitstream's"
                    );
                    self.warned_layout = true;
                }
                let pts = self.next_pts_us.unwrap_or(0);
                let samples = (f.samples.len() / f.channels.max(1)) as i64;
                self.next_pts_us = Some(pts + samples * 1_000_000 / i64::from(f.sample_rate));
                self.layout = Some(speakers_layout(&f.layout));
                AudioFrame { samples: f.samples, sample_rate: f.sample_rate, channels: f.channels as u8, pts }
            })
            .collect()
    }
}

impl AudioDecoder for Ac3Decoder {
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<AudioFrame>, AudioError> {
        if self.next_pts_us.is_none() && !packet.is_empty() {
            self.next_pts_us = Some(pts);
        }
        let frames = self.inner.decode(packet).map_err(decode_error)?;
        Ok(self.frames(frames))
    }

    fn flush(&mut self) -> Result<Vec<AudioFrame>, AudioError> {
        let frames = self.inner.flush().map_err(decode_error)?;
        Ok(self.frames(frames))
    }

    fn layout(&self) -> Option<ChannelLayout> {
        self.layout.clone().or_else(|| self.inner.last_header().map(|h| layout(&h)))
    }
}

impl std::fmt::Debug for Ac3Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ac3Decoder")
            .field("declared_sample_rate", &self.declared_sample_rate)
            .field("declared_channels", &self.declared_channels)
            .field("buffered", &self.inner.buffered())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn header(acmod: u8, lfeon: bool) -> Header {
        Header {
            eac3: false,
            strmtyp: 0,
            substreamid: 0,
            frame_len: 0,
            fscod: 0,
            sample_rate: 48_000,
            numblks: 6,
            acmod,
            lfeon,
            nfchans: match acmod {
                1 => 1,
                0 | 2 => 2,
                3 | 4 => 3,
                5 | 6 => 4,
                _ => 5,
            },
            bsid: 8,
            bsmod: 0,
            dialnorm: 31,
            bitrate_kbps: 448,
        }
    }

    /// The decoder's speakers land on rivet's named layouts (moved here from
    /// the decoder's own test when it became a crate).
    #[test]
    fn layouts_follow_acmod_in_output_order() {
        for (acmod, lfeon, name) in [
            (7, true, "5.1(side)"),
            (7, false, "5.0(side)"),
            (1, false, "mono"),
            (2, false, "stereo"),
            (2, true, "2.1"),
            (3, false, "3.0"),
            (3, true, "3.1"),
            (4, false, "3.0(back)"),
            (5, false, "4.0"),
            (5, true, "4.1"),
            (6, false, "quad(side)"),
        ] {
            let h = header(acmod, lfeon);
            assert_eq!(layout(&h).to_string(), name, "acmod {acmod} lfe {lfeon}");
            assert_eq!(layout(&h).len(), h.channels());
        }
        assert_eq!(layout(&header(4, true)).to_string(), "FL+FR+LFE+BC");
    }

    /// The adapter produces the crate decoder's PCM when the stream arrives
    /// as arbitrary packet boundaries, stamped from the first packet's pts.
    #[test]
    fn adapter_reassembles_split_frames_and_stamps_them() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../rivet/tests/data/audio/tones_51.ac3");
        let es = std::fs::read(&path).expect("tones_51.ac3");
        let mut direct = ac3::Decoder::new();
        let mut expected: Vec<f32> = direct.decode(&es).unwrap().into_iter().flat_map(|f| f.samples).collect();
        expected.extend(direct.flush().unwrap().into_iter().flat_map(|f| f.samples));

        let mut dec = Ac3Decoder::with_options(48_000, 6, Ac3Options { drc_scale: 1.0 }).unwrap();
        let mut out = Vec::new();
        let mut pts_seen = Vec::new();
        for (i, chunk) in es.chunks(1000).enumerate() {
            for f in dec.decode(chunk, if i == 0 { 5_000 } else { 0 }).unwrap() {
                assert_eq!(f.channels, 6);
                assert_eq!(f.sample_rate, 48_000);
                pts_seen.push(f.pts);
                out.extend_from_slice(&f.samples);
            }
        }
        out.extend(dec.flush().unwrap().into_iter().flat_map(|f| f.samples));
        assert_eq!(out, expected);
        assert_eq!(pts_seen[0], 5_000);
        assert_eq!(pts_seen[1], 5_000 + 32_000, "one 1536-sample syncframe is 32 ms");
        assert_eq!(dec.layout(), Some(ChannelLayout::named("5.1(side)")));
    }

    #[test]
    fn more_than_eight_declared_channels_is_unsupported() {
        assert!(Ac3Decoder::new(48_000, 8).is_ok());
        assert!(matches!(Ac3Decoder::new(48_000, 10), Err(AudioError::Unsupported(_))));
    }
}
