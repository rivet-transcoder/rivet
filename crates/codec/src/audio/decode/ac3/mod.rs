//! AC-3 / E-AC-3 (Dolby Digital / Digital Plus) decoder, in-tree and
//! pure Rust, written from ATSC A/52:2018.
//!
//! What it decodes
//! ---------------
//! - AC-3 (bsid ≤ 8): every mode, 1–6 channels including LFE — block
//!   switching, dither, coupling with phase flags, rematrixing, delta bit
//!   allocation, dynamic range compression (`dynrng`, applied by default,
//!   scalable through [`Ac3Options::drc_scale`]).
//! - E-AC-3 (bsid 16): independent substream 0 — all `numblkscod` frame
//!   sizes, reduced sample rates, frame-based exponent strategies, the three
//!   SNR-offset strategies, standard coupling, spectral extension (with
//!   attenuation), and the adaptive hybrid transform (vector quantisation
//!   and gain-adaptive quantisation).
//!
//! What it refuses or skips, by name
//! ---------------------------------
//! - Enhanced coupling (`ecplinu = 1`) → `AudioError::Unsupported`.
//! - Dependent substreams and independent substreams other than 0 are
//!   skipped (Annex E §3.8.1 says a reference decoder may), so a 7.1
//!   E-AC-3 stream decodes as its 5.1 core.
//! - bsid 9/10 (Annex D reduced-rate AC-3) → `Unsupported`.
//! - `dialnorm` and heavy compression (`compr`) are not applied, matching
//!   libavcodec's default; transient pre-noise processing is parsed and
//!   ignored (an optional post-process).
//!
//! Output is f32 interleaved in ffmpeg's native order for the layout (for
//! 5.1: FL FR FC LFE SL SR), which is what [`crate::audio::filter`] and the
//! Opus encoder assume for a channel count. No downmix is performed here —
//! `channelmap` does that on PCM.
//!
//! Tables live in [`tables`] with per-table checksum tests; the
//! cross-check against libavcodec lives in `tests/ac3_decode_vectors.rs`.

pub mod bitalloc;
mod bits;
pub mod decoder;
pub mod imdct;
pub mod tables;

use crate::audio::{AudioDecoder, AudioError, AudioFrame};
pub use decoder::{Features, FrameDecoder, Header, frame_crc_ok, parse_header};

/// Decoder options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ac3Options {
    /// Fraction of the stream's dynamic range compression to apply: 1.0
    /// applies `dynrng` exactly as encoded (the A/52 default), 0.0 ignores it.
    pub drc_scale: f32,
}

impl Default for Ac3Options {
    fn default() -> Self {
        Self { drc_scale: 1.0 }
    }
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
                Err(AudioError::Unsupported(e)) => return Err(AudioError::Unsupported(e)),
                Err(e) => {
                    tracing::debug!(error = %e, "ac3: bad sync header, skipping a byte");
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
                Err(AudioError::Unsupported(e)) => return Err(AudioError::Unsupported(e)),
                Err(e) => {
                    // A damaged frame: report it, keep the overlap history
                    // honest and carry on with the next syncframe.
                    tracing::warn!(error = %e, "ac3: frame decode failed, skipping");
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
