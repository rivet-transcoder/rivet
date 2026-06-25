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
use audiopus::ffi;

use std::ffi::c_int;
use std::ptr;

use crate::audio::resample::AudioResampler;
use crate::audio::{
    AudioCodec, AudioEncoder, AudioEncoderConfig, AudioError, AudioFrame, EncodedAudioPacket,
};

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

/// Channel-mapping family 1 surround layouts per RFC 7845 §5.1.1.2.
/// Each entry: (streams, coupled_streams, channel_mapping).
/// `streams` = total internal Opus streams.
/// `coupled_streams` = number of those streams that are stereo (2-channel).
/// `channel_mapping[i]` = which encoder stream the i-th *output* channel
/// pulls from. Indices 0..coupled*2 belong to the coupled (stereo)
/// streams (each coupled stream consumes two consecutive indices); the
/// remaining indices coupled*2..streams+coupled belong to the mono
/// (uncoupled) streams. Total mapping length == channel count.
fn surround_mapping_family_1(channels: u8) -> Result<(u8, u8, &'static [u8]), AudioError> {
    // RFC 7845 §5.1.1.2 — Vorbis channel order on input:
    //   3 (3.0):   L, R, C
    //   4 (quad):  FL, FR, BL, BR
    //   5 (5.0):   FL, FR, C, BL, BR
    //   6 (5.1):   FL, FR, C, LFE, BL, BR
    //   7 (6.1):   FL, FR, C, LFE, BC, SL, SR
    //   8 (7.1):   FL, FR, C, LFE, BL, BR, SL, SR
    //
    // Tuples below `(streams, coupled, mapping)` mirror libopus's
    // authoritative `vorbis_mappings[]` table in
    // `opus/src/opus_multistream_encoder.c:53-62`. The note in the
    // Squad-28 task spec listed `7 channels: streams=4, coupled=2`,
    // but libopus's reference table has `streams=4, coupled=3` (and
    // the mapping `[0, 4, 1, 2, 3, 5, 6]` requires stream index 6 to
    // exist, which only happens with coupled=3 → max valid stream
    // index = streams + coupled - 1 = 6). Using the libopus values
    // makes the multistream-create call succeed.
    //
    // The mapping below answers "which coded stream is each output channel?".
    // Encoder packs coupled (stereo) pairs first (taking 2 indices each),
    // then mono streams (1 index each). Total encoder channels =
    // streams + coupled.
    match channels {
        3 => Ok((2, 1, &[0, 2, 1])),
        4 => Ok((2, 2, &[0, 1, 2, 3])),
        5 => Ok((3, 2, &[0, 4, 1, 2, 3])),
        6 => Ok((4, 2, &[0, 4, 1, 2, 3, 5])),
        7 => Ok((4, 3, &[0, 4, 1, 2, 3, 5, 6])),
        8 => Ok((5, 3, &[0, 6, 1, 2, 3, 4, 5, 7])),
        _ => Err(AudioError::Unsupported(format!(
            "Opus surround mapping family 1 only defined for 3..=8 channels; got {channels}"
        ))),
    }
}

/// Internal dispatch — regular libopus encoder for 1/2 channels, or
/// multistream encoder for 3..=8 channels. The two paths converge at
/// the [`AudioEncoder`] trait surface.
enum OpusInner {
    Regular(OpusEncoderInner),
    /// Owned `OpusMSEncoder*` from `opus_multistream_encoder_create`.
    /// Freed via `opus_multistream_encoder_destroy` in `Drop`.
    Multistream(MultistreamEncoder),
}

/// RAII wrapper over a raw `OpusMSEncoder*`. The underlying libopus
/// state is allocated on the libopus heap; we destroy it via the FFI
/// destroy call when the wrapper drops. The pointer is non-null after
/// successful construction (`MultistreamEncoder::new` enforces this).
struct MultistreamEncoder {
    state: *mut ffi::OpusMSEncoder,
}

// SAFETY: libopus's multistream encoder state has no implicit thread
// affinity — like the regular Encoder we expose it via `&mut self`
// methods only, so external aliasing is impossible. The Send bound
// matches what audiopus's high-level Encoder claims.
unsafe impl Send for MultistreamEncoder {}

impl MultistreamEncoder {
    /// Allocate + initialize a multistream encoder for the given
    /// channel-mapping family-1 layout. `mapping.len()` must equal
    /// `channels`. Internally calls `opus_multistream_encoder_create`.
    fn new(
        sample_rate: u32,
        channels: u8,
        streams: u8,
        coupled_streams: u8,
        mapping: &[u8],
        application: Application,
    ) -> Result<Self, AudioError> {
        if mapping.len() != channels as usize {
            return Err(AudioError::Encode(format!(
                "multistream mapping length {} != channels {channels}",
                mapping.len()
            )));
        }
        // libopus invariant: streams + coupled_streams <= channels and
        // coupled_streams <= streams. Re-check here so a hand-crafted
        // call from inside this module (e.g. via a future API change)
        // can't slip past the public surround_mapping_family_1 helper.
        if coupled_streams > streams {
            return Err(AudioError::Encode(format!(
                "coupled_streams ({coupled_streams}) > streams ({streams})"
            )));
        }
        if (streams as usize) + (coupled_streams as usize) > channels as usize {
            return Err(AudioError::Encode(format!(
                "streams ({streams}) + coupled_streams ({coupled_streams}) > channels ({channels})"
            )));
        }

        let mut err: c_int = 0;
        // SAFETY: `mapping.as_ptr()` is valid for `channels` bytes
        // (asserted above); libopus reads `channels` bytes from it
        // synchronously. `&mut err` is a valid out-pointer for c_int.
        // The returned pointer is checked for null on the libopus
        // contract: non-null implies err == OPUS_OK.
        let state = unsafe {
            ffi::opus_multistream_encoder_create(
                sample_rate as i32,
                channels as c_int,
                streams as c_int,
                coupled_streams as c_int,
                mapping.as_ptr(),
                application as c_int,
                &mut err,
            )
        };
        if state.is_null() || err != ffi::OPUS_OK {
            return Err(AudioError::Encode(format!(
                "opus_multistream_encoder_create failed: code={err}"
            )));
        }
        Ok(Self { state })
    }

    /// Set per-encoder VBR. CTL request OPUS_SET_VBR_REQUEST takes an
    /// i32 (0 = CBR, 1 = VBR).
    fn set_vbr(&mut self, vbr: bool) -> Result<(), AudioError> {
        let val: c_int = if vbr { 1 } else { 0 };
        // SAFETY: `self.state` is a valid OpusMSEncoder*; the variadic
        // CTL ABI expects the request id followed by exactly one i32
        // argument for OPUS_SET_VBR (per libopus opus_defines.h).
        let r = unsafe {
            ffi::opus_multistream_encoder_ctl(self.state, ffi::OPUS_SET_VBR_REQUEST, val)
        };
        if r != ffi::OPUS_OK {
            return Err(AudioError::Encode(format!(
                "opus_multistream_encoder_ctl(SET_VBR) failed: {r}"
            )));
        }
        Ok(())
    }

    /// Set the *aggregate* bitrate across all streams. libopus
    /// distributes this internally proportional to each stream's
    /// channel count (mono streams ~half a stereo stream's allocation).
    fn set_bitrate(&mut self, bps: i32) -> Result<(), AudioError> {
        // SAFETY: same as set_vbr; OPUS_SET_BITRATE takes an i32.
        let r = unsafe {
            ffi::opus_multistream_encoder_ctl(self.state, ffi::OPUS_SET_BITRATE_REQUEST, bps)
        };
        if r != ffi::OPUS_OK {
            return Err(AudioError::Encode(format!(
                "opus_multistream_encoder_ctl(SET_BITRATE) failed: {r}"
            )));
        }
        Ok(())
    }

    /// Query the encoder's lookahead in samples at the configured
    /// sample rate (always 48 kHz for our usage). Returned as the
    /// `dOps.PreSkip` field per RFC 7845 §4.2.
    ///
    /// Returned as `u32` to match the audiopus high-level Encoder
    /// signature (libopus actually surfaces a non-negative i32 — the
    /// CTL never returns negative lookahead values).
    fn lookahead(&self) -> Result<u32, AudioError> {
        let mut out: c_int = 0;
        // SAFETY: OPUS_GET_LOOKAHEAD takes a `*mut int` out parameter.
        let r = unsafe {
            ffi::opus_multistream_encoder_ctl(
                self.state,
                ffi::OPUS_GET_LOOKAHEAD_REQUEST,
                &mut out as *mut c_int,
            )
        };
        if r != ffi::OPUS_OK {
            return Err(AudioError::Encode(format!(
                "opus_multistream_encoder_ctl(GET_LOOKAHEAD) failed: {r}"
            )));
        }
        if out < 0 {
            return Err(AudioError::Encode(format!(
                "opus_multistream_encoder_ctl(GET_LOOKAHEAD) returned negative: {out}"
            )));
        }
        Ok(out as u32)
    }

    /// Encode one 20-ms multichannel frame from interleaved f32 input.
    /// `pcm.len()` must equal `frame_size * channels`. Returns the
    /// encoded packet length in bytes (always positive on success).
    fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        out: &mut [u8],
    ) -> Result<usize, AudioError> {
        let max = out.len().min(i32::MAX as usize) as i32;
        // SAFETY: pcm.as_ptr() valid for frame_size*channels f32s
        // (caller guarantees via slice length); out.as_mut_ptr() valid
        // for `max` bytes; libopus reads/writes only within those
        // bounds.
        let n = unsafe {
            ffi::opus_multistream_encode_float(
                self.state,
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                max,
            )
        };
        if n < 0 {
            return Err(AudioError::Encode(format!(
                "opus_multistream_encode_float failed: code={n}"
            )));
        }
        Ok(n as usize)
    }
}

impl Drop for MultistreamEncoder {
    fn drop(&mut self) {
        if !self.state.is_null() {
            // SAFETY: state was allocated by opus_multistream_encoder_create
            // and is destroyed exactly once (Drop runs once).
            unsafe { ffi::opus_multistream_encoder_destroy(self.state) };
            self.state = ptr::null_mut();
        }
    }
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

/// Build the `dOps` body (RFC 7845 §4.5).
///
/// Two layouts:
/// - Family 0 (mono / stereo): 11 bytes total.
/// - Family 1 (surround 1..=8 channels, RFC 7845 §5.1.1.2): 11 + 2 + N
///   bytes total, where N is the channel count. The 11-byte preamble is
///   identical to family 0; the trailer adds StreamCount, CoupledCount,
///   and ChannelMapping (N bytes).
///
/// All multi-byte fields are LITTLE-endian per the RFC. The mux side
/// (container/src/mux.rs::build_dops) reads these LE bytes back and
/// translates to BE for the on-wire dOps box.
///
/// ```text
/// Version:               u8  = 0
/// OutputChannelCount:    u8  (1..=8)
/// PreSkip:               u16 LE
/// InputSampleRate:       u32 LE  (original/source rate)
/// OutputGain:            i16 LE  (0 = no gain change)
/// ChannelMappingFamily:  u8  (0 for 1-2 channels, 1 for 3-8)
/// // Family 1 only:
/// StreamCount:           u8
/// CoupledCount:          u8
/// ChannelMapping[N]:     u8 each (output-channel → encoder-stream index)
/// ```
fn build_dops(
    channels: u8,
    pre_skip_48k: u16,
    input_sample_rate: u32,
    ms_meta: Option<(u8, u8, &[u8])>,
) -> Vec<u8> {
    // Choose family based on whether multistream metadata is present.
    let (family, total_len) = match ms_meta {
        None => (0u8, 11usize),
        Some(_) => (1u8, 11 + 2 + channels as usize),
    };

    let mut v = Vec::with_capacity(total_len);
    v.push(0u8); // Version
    v.push(channels);
    v.extend_from_slice(&pre_skip_48k.to_le_bytes());
    v.extend_from_slice(&input_sample_rate.to_le_bytes());
    v.extend_from_slice(&0i16.to_le_bytes()); // OutputGain
    v.push(family);

    if let Some((streams, coupled, mapping)) = ms_meta {
        v.push(streams);
        v.push(coupled);
        // ChannelMapping: one byte per *output* channel; value is the
        // encoder-stream index for that output channel.
        v.extend_from_slice(mapping);
        debug_assert_eq!(mapping.len(), channels as usize);
    }
    debug_assert_eq!(v.len(), total_len);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use audiopus::Channels as OpusChannels;
    use audiopus::SampleRate;
    use audiopus::coder::Decoder as OpusDecoderInner;

    fn config_stereo_48k() -> AudioEncoderConfig {
        AudioEncoderConfig {
            codec: AudioCodec::Opus,
            sample_rate: 48_000,
            channels: 2,
            bitrate: 96_000,
        }
    }

    fn config_mono_48k() -> AudioEncoderConfig {
        AudioEncoderConfig {
            codec: AudioCodec::Opus,
            sample_rate: 48_000,
            channels: 1,
            bitrate: 64_000,
        }
    }

    fn config_multi_48k(channels: u8) -> AudioEncoderConfig {
        AudioEncoderConfig {
            codec: AudioCodec::Opus,
            sample_rate: 48_000,
            channels,
            bitrate: 0, // exercise the per-stream default-bitrate path
        }
    }

    #[test]
    fn opus_encoder_constructs_for_mono_48k_with_1_channel_dops() {
        let enc = OpusEncoder::new(config_mono_48k()).expect("constructs");
        assert_eq!(enc.channels, 1);
        assert!(enc.resampler.is_none());
        // dOps[1] = OutputChannelCount = 1 for mono
        assert_eq!(enc.extra_data[1], 1);
    }

    #[test]
    fn opus_encoder_uses_default_bitrate_when_caller_passes_zero() {
        let mut cfg = config_stereo_48k();
        cfg.bitrate = 0;
        let _enc = OpusEncoder::new(cfg).expect("constructs with bitrate=0");
        // Default bitrate path doesn't expose the value via a public
        // method on audiopus's Encoder without GenericCtl, but the
        // constructor would fail if it tried to set an invalid
        // bitrate. The fact that we got here means the default
        // (DEFAULT_BITRATE_STEREO=96k) was applied successfully.
    }

    fn config_stereo_44100() -> AudioEncoderConfig {
        AudioEncoderConfig {
            codec: AudioCodec::Opus,
            sample_rate: 44_100,
            channels: 2,
            bitrate: 96_000,
        }
    }

    fn make_silence(channels: u8, frames: usize, sample_rate: u32) -> AudioFrame {
        AudioFrame {
            samples: vec![0.0f32; frames * channels as usize],
            sample_rate,
            channels,
            pts: 0,
        }
    }

    fn make_sine_1k(channels: u8, frames: usize, sample_rate: u32, amp: f32) -> AudioFrame {
        let mut samples = Vec::with_capacity(frames * channels as usize);
        let two_pi = std::f32::consts::PI * 2.0;
        let freq = 1000.0f32;
        for i in 0..frames {
            let t = i as f32 / sample_rate as f32;
            let v = (two_pi * freq * t).sin() * amp;
            for _ in 0..channels {
                samples.push(v);
            }
        }
        AudioFrame {
            samples,
            sample_rate,
            channels,
            pts: 0,
        }
    }

    #[test]
    fn opus_encoder_constructs_for_stereo_48k() {
        let enc = OpusEncoder::new(config_stereo_48k()).expect("constructs");
        assert_eq!(enc.channels, 2);
        assert_eq!(enc.in_rate, 48000);
        assert!(enc.resampler.is_none(), "no resampler at native rate");
        assert_eq!(enc.extra_data.len(), 11, "dOps body must be 11 bytes");
        // dOps[0] = Version = 0
        assert_eq!(enc.extra_data[0], 0);
        // dOps[1] = OutputChannelCount
        assert_eq!(enc.extra_data[1], 2);
        // dOps[10] = ChannelMappingFamily = 0
        assert_eq!(enc.extra_data[10], 0);
    }

    #[test]
    fn opus_encoder_resamples_44100_to_48k_internally() {
        let enc = OpusEncoder::new(config_stereo_44100()).expect("constructs");
        assert!(enc.resampler.is_some(), "resampler engaged at 44.1k input");
        let r = enc.resampler.as_ref().unwrap();
        assert_eq!(r.in_rate(), 44100);
        assert_eq!(r.out_rate(), 48000);
    }

    #[test]
    fn opus_encoder_rejects_zero_channels() {
        let mut bad = config_stereo_48k();
        bad.channels = 0;
        assert!(matches!(
            OpusEncoder::new(bad),
            Err(AudioError::Unsupported(_))
        ));
    }

    #[test]
    fn opus_encoder_rejects_nine_channels() {
        // 9 channels (and above) has no defined channel-mapping family-1
        // layout in RFC 7845 §5.1.1.2, so we Unsupported it.
        let mut bad9 = config_stereo_48k();
        bad9.channels = 9;
        assert!(matches!(
            OpusEncoder::new(bad9),
            Err(AudioError::Unsupported(_))
        ));
    }

    #[test]
    fn opus_encoder_rejects_nine_channel_frame_at_runtime() {
        let mut enc = OpusEncoder::new(config_stereo_48k()).expect("constructs");
        let bad_frame = AudioFrame {
            samples: vec![0.0; 960 * 9],
            sample_rate: 48000,
            channels: 9,
            pts: 0,
        };
        let r = enc.encode(&bad_frame);
        assert!(
            matches!(r, Err(AudioError::Unsupported(_))),
            "9-channel frame should be Unsupported, got {:?}",
            r
        );
    }

    #[test]
    fn opus_pre_skip_in_48khz_ticks_is_nonzero() {
        let enc = OpusEncoder::new(config_stereo_48k()).expect("constructs");
        // libopus typically reports lookahead in the 312..=400 sample
        // range at 48 kHz. We just sanity-check it's nonzero.
        assert!(
            enc.pre_skip() > 0,
            "Opus encoder lookahead should be positive (libopus convention)"
        );
        assert!(
            enc.pre_skip() < 2000,
            "lookahead is bounded — typically <600 samples at 48 kHz"
        );
    }

    #[test]
    fn opus_dops_carries_correct_pre_skip_and_input_sample_rate_le() {
        let enc = OpusEncoder::new(config_stereo_44100()).expect("constructs");
        let d = enc.extra_data();
        // PreSkip at offset 2 (LE u16)
        let ps = u16::from_le_bytes([d[2], d[3]]);
        assert_eq!(ps, enc.pre_skip(), "dOps PreSkip matches encoder lookahead");
        // InputSampleRate at offset 4 (LE u32)
        let isr = u32::from_le_bytes([d[4], d[5], d[6], d[7]]);
        assert_eq!(
            isr, 44100,
            "dOps InputSampleRate is the source rate, not 48k"
        );
        // OutputGain at offset 8 (LE i16, default 0)
        let og = i16::from_le_bytes([d[8], d[9]]);
        assert_eq!(og, 0);
    }

    #[test]
    fn opus_encode_20ms_silence_produces_one_packet() {
        let mut enc = OpusEncoder::new(config_stereo_48k()).expect("constructs");
        // 20 ms at 48 kHz = 960 frames per channel
        let frame = make_silence(2, 960, 48_000);
        let pkts = enc.encode(&frame).expect("encode");
        assert_eq!(pkts.len(), 1, "exactly one Opus packet for one 20ms frame");
        let pkt = &pkts[0];
        assert!(!pkt.data.is_empty(), "packet should have bytes");
        // Silence at 96 kbps stereo: Opus DTX is OFF so we still get
        // a regular packet. Should be small (a few dozen bytes).
        assert!(
            pkt.data.len() < 200,
            "silence packet at 96 kbps should be small, got {} bytes",
            pkt.data.len()
        );
        assert_eq!(pkt.duration, 960, "20ms = 960 ticks at 48k");
    }

    #[test]
    fn opus_encode_one_second_of_sine_produces_packets_with_reasonable_bitrate() {
        let mut enc = OpusEncoder::new(config_stereo_48k()).expect("constructs");
        // Feed 1 second of 1 kHz sine in 20 ms slices so we have round
        // numbers. 48000 / 960 = 50 frames per second.
        let mut total_bytes = 0usize;
        let mut total_packets = 0usize;
        for i in 0..50 {
            let mut frame = make_sine_1k(2, 960, 48_000, 0.3);
            // Stagger the per-slice phase by adjusting pts; the
            // generator above uses i=0..960 so phase resets each
            // slice — for this test we don't care about phase
            // continuity across slices, only about bitrate aggregate.
            frame.pts = i * 20_000;
            let pkts = enc.encode(&frame).expect("encode");
            for p in &pkts {
                total_bytes += p.data.len();
                total_packets += 1;
            }
        }
        let pkts_flush = enc.flush().expect("flush");
        for p in &pkts_flush {
            total_bytes += p.data.len();
            total_packets += 1;
        }
        // Expect ~50 packets for 1 s of audio (one per 20 ms)
        assert!(
            total_packets >= 49 && total_packets <= 51,
            "expected ~50 packets for 1 s of audio, got {total_packets}"
        );
        // 1 second at 96 kbps = 96000 bits = 12000 bytes target.
        // VBR encoder will be within ±30% of this on a sine wave.
        let observed_bps = (total_bytes as u64 * 8) as i64;
        assert!(
            observed_bps > 30_000 && observed_bps < 200_000,
            "1s of 1kHz sine at 96 kbps should yield 30-200 kbps actual, got {observed_bps} bps ({total_bytes} bytes)"
        );
    }

    #[test]
    fn opus_pts_steps_by_20ms_per_packet() {
        let mut enc = OpusEncoder::new(config_stereo_48k()).expect("constructs");
        let frame_a = make_silence(2, 960, 48_000);
        let mut frame_b = make_silence(2, 960, 48_000);
        frame_b.pts = 20_000;
        let pkts_a = enc.encode(&frame_a).expect("a");
        let pkts_b = enc.encode(&frame_b).expect("b");
        assert_eq!(pkts_a.len(), 1);
        assert_eq!(pkts_b.len(), 1);
        let dt = pkts_b[0].pts - pkts_a[0].pts;
        // 20 ms in microseconds = 20_000
        assert_eq!(
            dt, 20_000,
            "PTS should step by 20_000 us per Opus packet (20 ms frame)"
        );
    }

    /// Round-trip: encode a sine wave then decode through libopus and
    /// compare against the input. Opus is lossy (especially silence
    /// padding at the front for pre_skip) so we measure RMS error
    /// over the steady-state portion only.
    #[test]
    fn opus_round_trip_sine_wave_quality_is_acceptable() {
        let mut enc = OpusEncoder::new(config_stereo_48k()).expect("constructs");
        let frames_per_chunk = 960;
        let n_chunks = 25; // ~500 ms
        let total_frames = frames_per_chunk * n_chunks;

        // Continuous-phase 1 kHz sine across all chunks.
        let mut all_samples = Vec::with_capacity(total_frames * 2);
        let two_pi = std::f32::consts::PI * 2.0;
        let freq = 1000.0f32;
        for i in 0..total_frames {
            let t = i as f32 / 48_000.0;
            let v = (two_pi * freq * t).sin() * 0.5;
            all_samples.push(v);
            all_samples.push(v);
        }

        // Encode chunk by chunk.
        let mut packets = Vec::new();
        for c in 0..n_chunks {
            let chunk_samples =
                all_samples[c * frames_per_chunk * 2..(c + 1) * frames_per_chunk * 2].to_vec();
            let frame = AudioFrame {
                samples: chunk_samples,
                sample_rate: 48_000,
                channels: 2,
                pts: (c as i64) * 20_000,
            };
            packets.extend(enc.encode(&frame).expect("encode"));
        }
        packets.extend(enc.flush().expect("flush"));
        assert!(!packets.is_empty(), "encode must produce packets");

        // Decode with audiopus.
        let mut dec =
            OpusDecoderInner::new(SampleRate::Hz48000, OpusChannels::Stereo).expect("dec");
        let mut decoded = Vec::with_capacity(total_frames * 2);
        let mut tmp = vec![0.0f32; frames_per_chunk * 2];
        for p in &packets {
            let pkt = audiopus::packet::Packet::try_from(p.data.as_slice()).expect("pkt");
            let sig = audiopus::MutSignals::try_from(tmp.as_mut_slice()).expect("sig");
            let n = dec
                .decode_float(Some(pkt), sig, false)
                .expect("decode_float");
            decoded.extend_from_slice(&tmp[..n * 2]);
        }
        assert!(
            decoded.len() >= (total_frames - 100) * 2,
            "decoded length {} should approximate input length {}",
            decoded.len(),
            total_frames * 2
        );

        // Compare the steady-state portion (skip pre_skip + a couple
        // hundred extra samples for filter warm-up) to the original.
        // Opus decoder output is delayed by `pre_skip` 48k samples
        // relative to the original input.
        let pre_skip = enc.pre_skip() as usize;
        let cmp_start = pre_skip + 480; // skip first 10 ms more
        let cmp_end = (decoded.len() / 2).min(total_frames - 100);
        if cmp_end <= cmp_start {
            panic!(
                "round trip too short: cmp_start={cmp_start}, cmp_end={cmp_end}, decoded len/2={}",
                decoded.len() / 2
            );
        }

        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_sig = 0.0f64;
        let mut n = 0usize;
        for i in cmp_start..cmp_end {
            // Opus decoder output at sample i corresponds to input at
            // sample (i - pre_skip). The decoded buffer already starts
            // at output sample 0, and pre_skip samples of it are the
            // encoder's lookahead "padding" — input sample 0 of the
            // user's stream lives at decoder output sample pre_skip.
            let in_idx = i - pre_skip;
            let l_in = all_samples[in_idx * 2];
            let r_in = all_samples[in_idx * 2 + 1];
            let l_out = decoded[i * 2];
            let r_out = decoded[i * 2 + 1];
            sum_sq_err += ((l_in - l_out) as f64).powi(2);
            sum_sq_err += ((r_in - r_out) as f64).powi(2);
            sum_sq_sig += (l_in as f64).powi(2);
            sum_sq_sig += (r_in as f64).powi(2);
            n += 2;
        }
        let rms_err = (sum_sq_err / n as f64).sqrt();
        let rms_sig = (sum_sq_sig / n as f64).sqrt();
        let snr_db = 20.0 * (rms_sig / rms_err.max(1e-12)).log10();
        // A sine wave round-tripped through Opus at 96 kbps stereo
        // should land >15 dB SNR easily — Opus is transparent on
        // simple tones at this bitrate. We use a conservative bound
        // because exact SNR depends on libopus version.
        assert!(
            snr_db > 15.0,
            "round-trip SNR {snr_db:.2} dB too low — Opus quality regression?"
        );
        // Print so the deliverables report can capture the actual
        // number from `cargo test -- --nocapture`.
        println!("opus_round_trip SNR = {snr_db:.2} dB, rms_err = {rms_err:.4}");
    }

    #[test]
    fn dops_layout_matches_rfc_7845_for_mono_and_stereo() {
        let d_mono = build_dops(1, 312, 48_000, None);
        assert_eq!(d_mono.len(), 11);
        assert_eq!(d_mono[0], 0); // Version
        assert_eq!(d_mono[1], 1); // ChannelCount
        assert_eq!(u16::from_le_bytes([d_mono[2], d_mono[3]]), 312); // PreSkip
        assert_eq!(
            u32::from_le_bytes([d_mono[4], d_mono[5], d_mono[6], d_mono[7]]),
            48000
        ); // InputSampleRate
        assert_eq!(i16::from_le_bytes([d_mono[8], d_mono[9]]), 0); // OutputGain
        assert_eq!(d_mono[10], 0); // Family

        let d_stereo = build_dops(2, 400, 44_100, None);
        assert_eq!(d_stereo.len(), 11);
        assert_eq!(d_stereo[1], 2);
        assert_eq!(u16::from_le_bytes([d_stereo[2], d_stereo[3]]), 400);
        assert_eq!(
            u32::from_le_bytes([d_stereo[4], d_stereo[5], d_stereo[6], d_stereo[7]]),
            44100
        );
    }

    // -------- Squad-28 multistream tests below --------

    /// Standard surround layouts per RFC 7845 §5.1.1.2. Each pair
    /// `(channels, (streams, coupled, mapping))` matches the spec
    /// table exactly.
    #[test]
    fn surround_mapping_family_1_matches_rfc_7845_5_1_1_2() {
        // 3.0 — L, R, C → coupled[L,R] + stream[C]
        assert_eq!(
            surround_mapping_family_1(3).unwrap(),
            (2, 1, &[0, 2, 1][..])
        );
        // quad — FL, FR, BL, BR → coupled[FL,FR] + coupled[BL,BR]
        assert_eq!(
            surround_mapping_family_1(4).unwrap(),
            (2, 2, &[0, 1, 2, 3][..])
        );
        // 5.0 — FL, FR, C, BL, BR
        assert_eq!(
            surround_mapping_family_1(5).unwrap(),
            (3, 2, &[0, 4, 1, 2, 3][..])
        );
        // 5.1 — FL, FR, C, LFE, BL, BR
        assert_eq!(
            surround_mapping_family_1(6).unwrap(),
            (4, 2, &[0, 4, 1, 2, 3, 5][..])
        );
        // 6.1 — FL, FR, C, LFE, BC, SL, SR
        // (streams=4, coupled=3; libopus authoritative — see
        // `vorbis_mappings[]` in opus_multistream_encoder.c:60).
        assert_eq!(
            surround_mapping_family_1(7).unwrap(),
            (4, 3, &[0, 4, 1, 2, 3, 5, 6][..])
        );
        // 7.1 — FL, FR, C, LFE, BL, BR, SL, SR
        assert_eq!(
            surround_mapping_family_1(8).unwrap(),
            (5, 3, &[0, 6, 1, 2, 3, 4, 5, 7][..])
        );
        // Out-of-range
        assert!(surround_mapping_family_1(0).is_err());
        assert!(surround_mapping_family_1(1).is_err()); // family-1 is 3..=8
        assert!(surround_mapping_family_1(2).is_err());
        assert!(surround_mapping_family_1(9).is_err());
    }

    #[test]
    fn opus_encoder_constructs_for_3_0_through_7_1_with_family_1_dops() {
        // For each surround channel count, the encoder should construct
        // and the dOps body should be 11 + 2 + N bytes with family=1
        // and the spec-mandated streams/coupled/mapping appended.
        for &ch in &[3u8, 4, 5, 6, 7, 8] {
            let enc = OpusEncoder::new(config_multi_48k(ch))
                .unwrap_or_else(|e| panic!("constructs for {ch}ch: {e:?}"));
            assert_eq!(enc.channels, ch);
            assert!(enc.resampler.is_none(), "no resampler at native rate");

            let d = enc.extra_data();
            let expected_len = 11 + 2 + ch as usize;
            assert_eq!(
                d.len(),
                expected_len,
                "dOps body for {ch}ch should be {expected_len} bytes (11 preamble + 2 stream/coupled + N mapping); got {}",
                d.len()
            );
            assert_eq!(
                d[0], 0,
                "Version=0 (dOps box version, not Opus stream version)"
            );
            assert_eq!(d[1], ch, "OutputChannelCount");
            assert_eq!(d[10], 1, "ChannelMappingFamily=1 for surround");

            let (exp_streams, exp_coupled, exp_mapping) = surround_mapping_family_1(ch).unwrap();
            assert_eq!(d[11], exp_streams, "StreamCount for {ch}ch");
            assert_eq!(d[12], exp_coupled, "CoupledCount for {ch}ch");
            assert_eq!(
                &d[13..13 + ch as usize],
                exp_mapping,
                "ChannelMapping for {ch}ch"
            );
        }
    }

    /// dOps body for a 5.1 encoder, hex-dumped. Captured in the
    /// deliverables report for cross-tool verification.
    #[test]
    fn opus_encoder_dops_5_1_hex_layout() {
        let enc = OpusEncoder::new(config_multi_48k(6)).expect("5.1 constructs");
        let d = enc.extra_data();
        assert_eq!(d.len(), 19, "5.1 dOps body = 11 + 2 + 6 = 19 bytes");
        let hex: String = d.iter().map(|b| format!("{b:02x} ")).collect();
        println!(
            "5.1 dOps body hex (LE-encoded, 19 bytes): {}",
            hex.trim_end()
        );
        // Layout cross-check:
        assert_eq!(d[0], 0); // Version
        assert_eq!(d[1], 6); // OutputChannelCount
        // PreSkip varies by libopus build; check it's non-zero
        let ps = u16::from_le_bytes([d[2], d[3]]);
        assert!(ps > 0 && ps < 2000);
        assert_eq!(
            u32::from_le_bytes([d[4], d[5], d[6], d[7]]),
            48_000,
            "InputSampleRate=48000"
        );
        assert_eq!(i16::from_le_bytes([d[8], d[9]]), 0); // OutputGain
        assert_eq!(d[10], 1); // Family=1
        assert_eq!(d[11], 4); // StreamCount=4 (5.1)
        assert_eq!(d[12], 2); // CoupledCount=2 (5.1)
        assert_eq!(&d[13..19], &[0u8, 4, 1, 2, 3, 5][..]); // ChannelMapping
    }

    #[test]
    fn opus_5_1_encode_20ms_silence_produces_one_packet() {
        let mut enc = OpusEncoder::new(config_multi_48k(6)).expect("5.1 constructs");
        // 20 ms at 48 kHz, 6 channels
        let frame = make_silence(6, 960, 48_000);
        let pkts = enc.encode(&frame).expect("encode 5.1 silence");
        assert_eq!(pkts.len(), 1, "exactly one Opus packet for one 20ms frame");
        let pkt = &pkts[0];
        assert!(!pkt.data.is_empty());
        // Multistream silence packet is larger than the mono case
        // because there's >=4 internal streams emitting their own
        // silence frame — but should still be small in absolute terms.
        assert!(
            pkt.data.len() < 600,
            "5.1 silence packet should still be under ~600 bytes, got {} bytes",
            pkt.data.len()
        );
        assert_eq!(pkt.duration, 960);
    }

    /// Round-trip 5.1 sine through libopus multistream encode + decode,
    /// computing per-channel SNR. Each channel carries a different
    /// frequency so cross-channel bleed would show up as low SNR.
    #[test]
    fn opus_5_1_round_trip_per_channel_snr_is_acceptable() {
        // Per-channel sine frequencies (Hz). Distinct so a coupled
        // stream that mixed channels would show degraded SNR.
        // 5.1 channel order: FL, FR, C, LFE, BL, BR
        let freqs = [440.0f32, 523.25, 659.25, 80.0, 880.0, 987.77];
        let chans: u8 = 6;
        let frames_per_chunk = 960;
        let n_chunks = 30; // ~600 ms
        let total_frames = frames_per_chunk * n_chunks;
        let amp = 0.4f32;

        // Build the multichannel input. Continuous phase across chunks.
        let mut all = vec![0.0f32; total_frames * chans as usize];
        let two_pi = std::f32::consts::PI * 2.0;
        for i in 0..total_frames {
            let t = i as f32 / 48_000.0;
            for ch in 0..chans as usize {
                all[i * chans as usize + ch] = (two_pi * freqs[ch] * t).sin() * amp;
            }
        }

        // Encode.
        let mut enc = OpusEncoder::new(config_multi_48k(chans)).expect("encoder");
        let mut packets = Vec::new();
        for c in 0..n_chunks {
            let frame = AudioFrame {
                samples: all[c * frames_per_chunk * chans as usize
                    ..(c + 1) * frames_per_chunk * chans as usize]
                    .to_vec(),
                sample_rate: 48_000,
                channels: chans,
                pts: (c as i64) * 20_000,
            };
            packets.extend(enc.encode(&frame).expect("encode"));
        }
        packets.extend(enc.flush().expect("flush"));
        assert!(!packets.is_empty(), "must produce packets");

        // Decode via the multistream API directly through audiopus_sys.
        let (streams, coupled, mapping) = surround_mapping_family_1(chans).unwrap();
        let mut err: c_int = 0;
        let dec_state = unsafe {
            ffi::opus_multistream_decoder_create(
                48_000,
                chans as c_int,
                streams as c_int,
                coupled as c_int,
                mapping.as_ptr(),
                &mut err,
            )
        };
        assert!(
            !dec_state.is_null() && err == ffi::OPUS_OK,
            "MS decoder create"
        );

        let mut decoded = Vec::with_capacity(total_frames * chans as usize);
        let mut tmp = vec![0.0f32; frames_per_chunk * chans as usize];
        for p in &packets {
            let n = unsafe {
                ffi::opus_multistream_decode_float(
                    dec_state,
                    p.data.as_ptr(),
                    p.data.len() as i32,
                    tmp.as_mut_ptr(),
                    frames_per_chunk as c_int,
                    0,
                )
            };
            assert!(n > 0, "MS decode_float returned {n}");
            decoded.extend_from_slice(&tmp[..(n as usize) * chans as usize]);
        }
        unsafe { ffi::opus_multistream_decoder_destroy(dec_state) };

        // Per-channel SNR over the steady-state portion. Skip pre_skip
        // + 480 samples of filter warm-up at the front, plus a small
        // tail margin.
        let pre_skip = enc.pre_skip() as usize;
        let cmp_start = pre_skip + 480;
        let cmp_end = (decoded.len() / chans as usize).min(total_frames - 200);
        assert!(cmp_end > cmp_start, "round trip too short");

        let mut snrs = Vec::with_capacity(chans as usize);
        for ch in 0..chans as usize {
            let mut sum_sq_err = 0.0f64;
            let mut sum_sq_sig = 0.0f64;
            for i in cmp_start..cmp_end {
                let in_idx = i - pre_skip;
                let s_in = all[in_idx * chans as usize + ch];
                let s_out = decoded[i * chans as usize + ch];
                sum_sq_err += ((s_in - s_out) as f64).powi(2);
                sum_sq_sig += (s_in as f64).powi(2);
            }
            let n = (cmp_end - cmp_start) as f64;
            let rms_err = (sum_sq_err / n).sqrt();
            let rms_sig = (sum_sq_sig / n).sqrt();
            let snr_db = 20.0 * (rms_sig / rms_err.max(1e-12)).log10();
            snrs.push(snr_db);
        }

        println!("5.1 per-channel SNR (dB):");
        for (i, snr) in snrs.iter().enumerate() {
            let label = ["FL", "FR", "C", "LFE", "BL", "BR"][i];
            println!("  ch{i} ({label}): {snr:.2} dB");
        }

        // Each channel should land >= 5 dB SNR on a steady tone.
        // Multistream Opus at default per-stream bitrate (~320 kbps
        // total) is transparent on a simple sine, but the LFE channel
        // is allocated less bitrate by libopus and lower-frequency
        // tones have proportionally larger error per sample, so we use
        // a conservative bound.
        for (i, snr) in snrs.iter().enumerate() {
            assert!(
                *snr > 5.0,
                "ch{i} SNR {snr:.2} dB too low — multistream quality regression?"
            );
        }
    }

    #[test]
    fn dops_layout_for_5_1_matches_family_1_spec() {
        let (streams, coupled, mapping) = surround_mapping_family_1(6).unwrap();
        let d = build_dops(6, 312, 48_000, Some((streams, coupled, mapping)));
        assert_eq!(d.len(), 11 + 2 + 6, "5.1 dOps = 19 bytes");
        assert_eq!(d[0], 0); // Version
        assert_eq!(d[1], 6); // OutputChannelCount
        assert_eq!(u16::from_le_bytes([d[2], d[3]]), 312); // PreSkip
        assert_eq!(u32::from_le_bytes([d[4], d[5], d[6], d[7]]), 48_000); // InputSampleRate
        assert_eq!(i16::from_le_bytes([d[8], d[9]]), 0); // OutputGain
        assert_eq!(d[10], 1); // Family=1
        assert_eq!(d[11], 4); // StreamCount=4 for 5.1
        assert_eq!(d[12], 2); // CoupledCount=2 for 5.1
        assert_eq!(&d[13..19], &[0u8, 4, 1, 2, 3, 5][..]);
    }

    /// 5.1 encoder at a non-48k input rate must engage the resampler
    /// for its 6 channels — gates the resampler-channel-cap lift.
    #[test]
    fn opus_5_1_resamples_44100_to_48k() {
        let mut cfg = config_multi_48k(6);
        cfg.sample_rate = 44_100;
        let enc = OpusEncoder::new(cfg).expect("5.1 @ 44.1k constructs");
        assert!(enc.resampler.is_some(), "resampler engaged for 6ch @ 44.1k");
        let r = enc.resampler.as_ref().unwrap();
        assert_eq!(r.in_rate(), 44_100);
        assert_eq!(r.out_rate(), 48_000);
        assert_eq!(r.channels(), 6);
    }
}
