//! AC-3 / E-AC-3 decode through the workspace's AC-3 decoder (`ac3`, the
//! `crates/ac3` submodule, the rivet-ac3 repository), adapted to
//! [`AudioDecoder`].
//!
//! - Packets hold one or more whole or partial syncframes (MP4 / Matroska
//!   samples, TS PES payloads, a raw .ac3 / .eac3 file in any chunking):
//!   the adapter resynchronises on 0x0B77, buffers a partial syncframe,
//!   skips a damaged one, and stamps each frame from the first packet's pts
//!   plus the samples decoded since.
//! - AC-3 in full; E-AC-3 independent substream 0, so 7.1 decodes as its
//!   5.1 core. Enhanced coupling and bsid 9 / 10 are
//!   [`AudioError::Unsupported`].
//! - Output is in ffmpeg's native order for the layout (5.1: FL FR FC LFE
//!   SL SR), which [`AudioDecoder::layout`] names. No downmix here —
//!   `channelmap` does that on PCM.
//!
//! The stream loop is the one this module had before the decoder moved
//! out, over [`ac3::FrameDecoder`], so the timestamps and log lines are
//! unchanged; the crate's own `ac3::Decoder` does the same without them.

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

/// The layout a syncframe header decodes to.
pub fn layout(h: &Header) -> ChannelLayout {
    ChannelLayout::new(h.speakers().into_iter().map(label).collect()).expect("distinct speakers")
}

/// [`AudioDecoder`] adapter: takes packets that hold one or more whole or
/// partial syncframes, emits one [`AudioFrame`] per syncframe.
pub struct Ac3Decoder {
    inner: FrameDecoder,
    buf: Vec<u8>,
    declared_sample_rate: u32,
    declared_channels: u8,
    next_pts_us: Option<i64>,
    warned_layout: bool,
}

impl Ac3Decoder {
    pub fn new(sample_rate: u32, channels: u8) -> Result<Self, AudioError> {
        Self::with_options(sample_rate, channels, Ac3Options::default())
    }

    pub fn with_options(sample_rate: u32, channels: u8, opts: Ac3Options) -> Result<Self, AudioError> {
        if channels > 6 {
            return Err(AudioError::Unsupported(format!(
                "ac3: {channels} channels — a single AC-3/E-AC-3 independent substream carries at most 6"
            )));
        }
        Ok(Self {
            inner: FrameDecoder::new(opts.drc_scale),
            buf: Vec::new(),
            declared_sample_rate: sample_rate,
            declared_channels: channels,
            next_pts_us: None,
            warned_layout: false,
        })
    }

    /// The header of the most recent syncframe decoded, if any.
    pub fn last_header(&self) -> Option<Header> {
        self.inner.last_header()
    }

    fn drain(&mut self) -> Result<Vec<AudioFrame>, AudioError> {
        let mut frames = Vec::new();
        let mut pos = 0usize;
        while self.buf.len() - pos >= 8 {
            // resync: find the next 0x0B77
            match self.buf[pos..].windows(2).position(|w| w == [0x0b, 0x77]) {
                Some(0) => {}
                Some(off) => {
                    tracing::debug!(skipped = off, "ac3: resynchronised on 0x0B77");
                    pos += off;
                    if self.buf.len() - pos < 8 {
                        break;
                    }
                }
                None => {
                    pos = self.buf.len().saturating_sub(1);
                    break;
                }
            }
            let hdr = match parse_header(&self.buf[pos..]) {
                Ok(h) => h,
                Err(ac3::Error::Unsupported(e)) => return Err(AudioError::Unsupported(e)),
                Err(e) => {
                    tracing::debug!(error = %decode_error(e), "ac3: bad sync header, skipping a byte");
                    pos += 1;
                    continue;
                }
            };
            if self.buf.len() - pos < hdr.frame_len {
                break;
            }
            let frame = &self.buf[pos..pos + hdr.frame_len];
            let mut pcm = Vec::new();
            match self.inner.decode(frame, &mut pcm) {
                Ok(Some(h)) => {
                    if !self.warned_layout
                        && self.declared_channels != 0
                        && usize::from(self.declared_channels) != h.channels()
                    {
                        tracing::warn!(
                            declared = self.declared_channels,
                            stream = h.channels(),
                            "ac3: container channel count differs from the bitstream; using the bitstream's"
                        );
                        self.warned_layout = true;
                    }
                    let pts = self.next_pts_us.unwrap_or(0);
                    let samples = h.samples() as i64;
                    self.next_pts_us = Some(pts + samples * 1_000_000 / i64::from(h.sample_rate));
                    frames.push(AudioFrame {
                        samples: pcm,
                        sample_rate: h.sample_rate,
                        channels: h.channels() as u8,
                        pts,
                    });
                }
                Ok(None) => {}
                Err(ac3::Error::Unsupported(e)) => return Err(AudioError::Unsupported(e)),
                Err(e) => {
                    // A damaged frame: report it, keep the overlap history
                    // honest and carry on with the next syncframe.
                    tracing::warn!(error = %decode_error(e), "ac3: frame decode failed, skipping");
                    self.inner.reset();
                }
            }
            pos += hdr.frame_len;
        }
        self.buf.drain(..pos);
        Ok(frames)
    }
}

impl AudioDecoder for Ac3Decoder {
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<AudioFrame>, AudioError> {
        if self.next_pts_us.is_none() && !packet.is_empty() {
            self.next_pts_us = Some(pts);
        }
        self.buf.extend_from_slice(packet);
        self.drain()
    }

    fn flush(&mut self) -> Result<Vec<AudioFrame>, AudioError> {
        let frames = self.drain()?;
        self.buf.clear();
        Ok(frames)
    }

    fn layout(&self) -> Option<ChannelLayout> {
        self.inner.last_header().map(|h| layout(&h))
    }
}

impl std::fmt::Debug for Ac3Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ac3Decoder")
            .field("declared_sample_rate", &self.declared_sample_rate)
            .field("declared_channels", &self.declared_channels)
            .field("buffered", &self.buf.len())
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
    fn more_than_six_declared_channels_is_unsupported() {
        assert!(matches!(Ac3Decoder::new(48_000, 8), Err(AudioError::Unsupported(_))));
    }
}
