//! Opus encoder wrapping `audiopus` (libopus FFI; libopus is BSD,
//! audiopus is ISC). Squad-23's MP4 mux side consumes the packets +
//! `extra_data()` (dOps body per RFC 7845 §4.5) + `pre_skip()` (samples
//! at 48 kHz queried via `OPUS_GET_LOOKAHEAD`).
//!
//! Constraints we enforce on the Opus side:
//! - Native sample rates are 8/12/16/24/48 kHz only. We always run the
//!   internal libopus encoder at 48 kHz and resample the input ourselves
//!   via [`AudioResampler`] when the source isn't 48 k. This keeps
//!   pre_skip semantics simple (always reported in 48 kHz ticks per the
//!   RFC) and means the dOps `InputSampleRate` field cleanly reflects
//!   the original source rate.
//! - Frame sizes must be 2.5/5/10/20/40/60 ms. We use 20 ms = 960
//!   samples at 48 kHz. This is libopus's default and matches what
//!   browsers / WebRTC expect.
//! - Channels: 1 (mono) and 2 (stereo) use the regular `audiopus::coder::Encoder`
//!   API. 3..=8 channels (3.0 / quad / 5.0 / 5.1 / 6.1 / 7.1) use the
//!   libopus Multistream API via `audiopus_sys` FFI (Squad-28). Channel
//!   counts above 8 return [`AudioError::Unsupported`] — RFC 7845
//!   §5.1.1.2 only specifies channel-mapping family 1 for 1..=8 channels.
//!
//! Defaults
//! --------
//! - 96 kbps for stereo, 64 kbps for mono if the caller passes 0.
//!   Multichannel: 64 kbps per uncoupled stream + 96 kbps per coupled
//!   stream (so 5.1 = 96 + 96 + 64 + 64 = 320 kbps total) — well above
//!   transparency for music/speech (Opus reaches transparency around
//!   64 kbps stereo for music).
//! - Application = `Audio` (vs Voip / LowDelay): tuned for fidelity
//!   over latency. Latency from a 20 ms frame size + ~6.5 ms libopus
//!   lookahead is ~26 ms one-way which is fine for offline transcode.
//!
//! Multistream API (Squad-28)
//! --------------------------
//! `audiopus 0.3.0-rc.0` ships a `multistream = []` Cargo feature that's
//! a stub — it gates no Rust code (the high-level wrapper just doesn't
//! exist for the multistream side in this crate version). We call the
//! underlying FFI symbols directly via `audiopus::ffi::*` (which re-exports
//! `audiopus_sys 0.2.2`'s `opus_multistream_encoder_*` functions). The
//! channel-mapping family 1 layouts we wire follow RFC 7845 §5.1.1.2
//! verbatim (3.0 / quad / 5.0 / 5.1 / 6.1 / 7.1).

use audiopus::Application;
use audiopus::Bitrate;
use audiopus::Channels as OpusChannels;
use audiopus::SampleRate;
use audiopus::coder::Encoder as OpusEncoderInner;

use crate::audio::resample::AudioResampler;
use crate::audio::{
    AudioCodec, AudioEncoder, AudioEncoderConfig, AudioError, AudioFrame, EncodedAudioPacket,
};

mod dops;
mod multistream;

#[cfg(test)]
mod tests;

use dops::build_dops;
use multistream::{MultistreamEncoder, surround_mapping_family_1};

/// 20 ms frame at 48 kHz = 960 samples per channel. This is the
/// default/recommended Opus frame size.
const OPUS_FRAME_SAMPLES_48K: usize = 960;
/// Internal encoder rate we always run libopus at. Resample to here
/// from any source rate. Per RFC 7845, pre_skip is always counted in
/// 48 kHz ticks regardless.
const OPUS_INTERNAL_RATE: u32 = 48_000;
/// Maximum bytes per Opus packet per RFC 6716 §3.4 — actual bound is
/// 1275 for 60ms VBR + multistream overhead; we round up to 4000 as
/// audiopus does, which gives a comfortable margin. For multistream
/// the per-frame budget scales with stream count; we use the standard
/// libopus bound of 1275 bytes per stream and cap at 8 streams = 10200
/// bytes (rounded up to 16384 for headroom).
const OPUS_MAX_PACKET_BYTES: usize = 4000;
const OPUS_MAX_MS_PACKET_BYTES: usize = 16_384;
/// Default bitrates per channel-count, in bits/second.
const DEFAULT_BITRATE_MONO: u32 = 64_000;
const DEFAULT_BITRATE_STEREO: u32 = 96_000;

/// Internal dispatch — regular libopus encoder for 1/2 channels, or
/// multistream encoder for 3..=8 channels. The two paths converge at
/// the [`AudioEncoder`] trait surface.
enum OpusInner {
    Regular(OpusEncoderInner),
    /// Owned `OpusMSEncoder*` from `opus_multistream_encoder_create`.
    /// Freed via `opus_multistream_encoder_destroy` in `Drop`.
    Multistream(MultistreamEncoder),
}

pub struct OpusEncoder {
    inner: OpusInner,
    /// Source sample rate the caller will feed.
    in_rate: u32,
    /// Channel count (1..=8).
    channels: u8,
    /// Resampler when in_rate != 48 kHz, else None.
    resampler: Option<AudioResampler>,
    /// Carry of resampled (or directly-fed) samples that didn't fill a
    /// full Opus frame yet. Interleaved planar f32.
    sample_carry: Vec<f32>,
    /// pre_skip in 48 kHz samples — captured at construction.
    pre_skip_48k: u16,
    /// dOps body bytes per RFC 7845 §4.5 — built once at construction
    /// from in_rate + channels + pre_skip + (when multichannel)
    /// streams + coupled_streams + channel mapping.
    extra_data: Vec<u8>,
    /// Running PTS in microseconds. Set on first encode call.
    next_pts_us: Option<i64>,
    /// Microseconds per Opus frame at the configured frame size.
    frame_duration_us: i64,
    /// Reusable encode output buffer to avoid per-frame allocation.
    encode_out: Vec<u8>,
}

impl OpusEncoder {
    pub fn new(config: AudioEncoderConfig) -> Result<Self, AudioError> {
        if config.codec != AudioCodec::Opus {
            return Err(AudioError::Encode(format!(
                "OpusEncoder constructed with codec {:?}",
                config.codec
            )));
        }
        if config.channels == 0 {
            return Err(AudioError::Unsupported(
                "Opus channel count must be >= 1".to_string(),
            ));
        }
        if config.channels > 8 {
            return Err(AudioError::Unsupported(format!(
                "Opus supports up to 8 channels (channel-mapping family 1, RFC 7845 §5.1.1.2); \
                 got {} channels",
                config.channels
            )));
        }
        if config.sample_rate == 0 {
            return Err(AudioError::Encode("input sample_rate is 0".to_string()));
        }

        let channels = config.channels;

        // Construct the inner encoder + capture multistream metadata
        // (streams / coupled_streams / mapping) when on the multistream
        // path. Both paths converge into a single OpusInner.
        let (inner, ms_meta, max_packet_bytes) = if channels <= 2 {
            // Regular API path — Squad-24's original code.
            let opus_channels = match channels {
                1 => OpusChannels::Mono,
                2 => OpusChannels::Stereo,
                _ => unreachable!("channel-count guarded above"),
            };
            let mut enc =
                OpusEncoderInner::new(SampleRate::Hz48000, opus_channels, Application::Audio)
                    .map_err(|e| AudioError::Encode(format!("opus encoder create: {e}")))?;
            let bitrate_bps = if config.bitrate == 0 {
                if channels == 1 {
                    DEFAULT_BITRATE_MONO
                } else {
                    DEFAULT_BITRATE_STEREO
                }
            } else {
                config.bitrate
            };
            enc.set_bitrate(Bitrate::BitsPerSecond(bitrate_bps as i32))
                .map_err(|e| AudioError::Encode(format!("opus set_bitrate: {e}")))?;
            // VBR is the audiopus default but we set it explicitly for
            // documentation; CBR is reserved for streaming use cases not
            // relevant to file output.
            enc.set_vbr(true)
                .map_err(|e| AudioError::Encode(format!("opus set_vbr: {e}")))?;
            (OpusInner::Regular(enc), None, OPUS_MAX_PACKET_BYTES)
        } else {
            // Multistream path: build the family-1 layout, allocate the
            // libopus multistream encoder via FFI.
            let (streams, coupled, mapping) = surround_mapping_family_1(channels)?;
            let mut ms = MultistreamEncoder::new(
                OPUS_INTERNAL_RATE,
                channels,
                streams,
                coupled,
                mapping,
                Application::Audio,
            )?;
            // Default aggregate bitrate scales with streams: 96 kbps per
            // coupled (stereo) + 64 kbps per uncoupled (mono). For 5.1
            // (4 streams, 2 coupled) this is 2*96 + 2*64 = 320 kbps,
            // which is the Opus reference default for surround.
            let bitrate_bps = if config.bitrate == 0 {
                let coupled_u = coupled as u32;
                let mono_u = streams as u32 - coupled_u;
                coupled_u * DEFAULT_BITRATE_STEREO + mono_u * DEFAULT_BITRATE_MONO
            } else {
                config.bitrate
            };
            ms.set_bitrate(bitrate_bps as i32)?;
            ms.set_vbr(true)?;
            (
                OpusInner::Multistream(ms),
                Some((streams, coupled, mapping)),
                OPUS_MAX_MS_PACKET_BYTES,
            )
        };

        // Read the lookahead in 48 kHz ticks regardless of which inner
        // path we took. Both regular + multistream report lookahead in
        // samples-of-the-configured-rate per libopus convention; we
        // configure both at 48 kHz so no scaling is needed.
        let pre_skip_48k_u32 = match &inner {
            OpusInner::Regular(enc) => enc
                .lookahead()
                .map_err(|e| AudioError::Encode(format!("opus lookahead: {e}")))?,
            OpusInner::Multistream(ms) => ms.lookahead()?,
        };
        let pre_skip_48k: u16 = pre_skip_48k_u32.try_into().unwrap_or(u16::MAX);

        // Resampler if needed.
        let resampler = if config.sample_rate == OPUS_INTERNAL_RATE {
            None
        } else {
            // chunk_size: process 20 ms worth of input at a time so the
            // resampler output naturally aligns with Opus's 20 ms frame
            // size. 20 ms at 44.1 kHz = 882 samples, at 22.05 kHz = 441,
            // etc. We round to the nearest integer.
            let chunk = ((config.sample_rate as usize) * 20) / 1000;
            let chunk = chunk.max(1);
            Some(AudioResampler::new(
                config.sample_rate,
                OPUS_INTERNAL_RATE,
                channels,
                chunk,
            )?)
        };

        let extra_data = build_dops(channels, pre_skip_48k, config.sample_rate, ms_meta);

        let frame_duration_us =
            (OPUS_FRAME_SAMPLES_48K as i64 * 1_000_000) / OPUS_INTERNAL_RATE as i64;

        Ok(Self {
            inner,
            in_rate: config.sample_rate,
            channels,
            resampler,
            sample_carry: Vec::with_capacity(OPUS_FRAME_SAMPLES_48K * channels as usize * 4),
            pre_skip_48k,
            extra_data,
            next_pts_us: None,
            frame_duration_us,
            encode_out: vec![0u8; max_packet_bytes],
        })
    }

    /// Drain as many full 20-ms Opus frames as possible from
    /// `sample_carry`. Each successful encode advances `next_pts_us`
    /// by `frame_duration_us`.
    fn drain_packets(&mut self) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        let mut out = Vec::new();
        let chans = self.channels as usize;
        let frame_interleaved_len = OPUS_FRAME_SAMPLES_48K * chans;
        while self.sample_carry.len() >= frame_interleaved_len {
            // Encode the front-most frame.
            let frame_slice = &self.sample_carry[..frame_interleaved_len];
            let n = match &mut self.inner {
                OpusInner::Regular(enc) => enc
                    .encode_float(frame_slice, &mut self.encode_out)
                    .map_err(|e| AudioError::Encode(format!("opus encode_float: {e}")))?,
                OpusInner::Multistream(ms) => {
                    ms.encode_float(frame_slice, OPUS_FRAME_SAMPLES_48K, &mut self.encode_out)?
                }
            };
            // n=0 would be a discontinuous-transmission "no packet"
            // signal — we don't enable DTX so it shouldn't fire, but
            // defensively skip if it does.
            if n > 0 {
                let pts = self.next_pts_us.unwrap_or(0);
                self.next_pts_us = Some(pts + self.frame_duration_us);
                out.push(EncodedAudioPacket {
                    data: self.encode_out[..n].to_vec(),
                    pts,
                    duration: OPUS_FRAME_SAMPLES_48K as i64, // 48 kHz ticks
                });
            }
            self.sample_carry.drain(..frame_interleaved_len);
        }
        Ok(out)
    }
}

impl AudioEncoder for OpusEncoder {
    fn encode(&mut self, frame: &AudioFrame) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        // Channel-count gate. Multichannel (3..=8) is now supported via
        // the Multistream API path (Squad-28); >8 stays Unsupported.
        if frame.channels == 0 || frame.channels > 8 {
            return Err(AudioError::Unsupported(format!(
                "Opus AudioFrame channel count must be 1..=8; got {}",
                frame.channels
            )));
        }
        if frame.channels != self.channels {
            return Err(AudioError::Encode(format!(
                "channel count mismatch: encoder configured for {}, frame has {}",
                self.channels, frame.channels
            )));
        }
        if frame.sample_rate != self.in_rate {
            return Err(AudioError::Encode(format!(
                "sample rate mismatch: encoder configured for {}, frame has {}",
                self.in_rate, frame.sample_rate
            )));
        }

        if self.next_pts_us.is_none() {
            self.next_pts_us = Some(frame.pts);
        }

        // Push samples into carry, possibly via resampler.
        if let Some(r) = self.resampler.as_mut() {
            r.process(frame, &mut self.sample_carry)?;
        } else {
            self.sample_carry.extend_from_slice(&frame.samples);
        }

        self.drain_packets()
    }

    fn flush(&mut self) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        if let Some(r) = self.resampler.as_mut() {
            r.flush(&mut self.sample_carry)?;
        }
        // Pad the final partial frame with silence so libopus can emit
        // a final packet (mux side will use pre_skip + the file's
        // total sample count to know where playable audio ends).
        let chans = self.channels as usize;
        let frame_interleaved_len = OPUS_FRAME_SAMPLES_48K * chans;
        if !self.sample_carry.is_empty() && self.sample_carry.len() < frame_interleaved_len {
            self.sample_carry.resize(frame_interleaved_len, 0.0);
        }
        self.drain_packets()
    }

    fn pre_skip(&self) -> u16 {
        self.pre_skip_48k
    }

    fn extra_data(&self) -> Vec<u8> {
        self.extra_data.clone()
    }
}
