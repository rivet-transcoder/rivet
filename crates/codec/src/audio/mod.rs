//! Audio codec framework.
//!
//! Squad-24 (2026-04-17 PM5): adds the decoder/encoder traits + the wire
//! types Squad-23 (audio mux pipeline) consumes. Decoders cover MP3 and
//! Vorbis (mux already handles AAC/Opus/AC-3 passthrough — no decode
//! needed for those). The encoder side currently exposes Opus only;
//! the user decision on the audio expansion in TODO.md picked Opus over
//! AAC because the libopus binding is BSD/Apache, modern browsers all
//! play Opus-in-MP4, and the iOS-13-and-older floor is acceptable.
//!
//! Wire model
//! ----------
//! - [`AudioFrame`] is the canonical PCM exchange type: f32 in
//!   [-1.0, 1.0], interleaved planar layout (LRLRLR for stereo), with
//!   the source sample rate and channel count carried alongside the
//!   samples and a microsecond-domain PTS.
//! - [`EncodedAudioPacket`] carries one encoder output packet plus
//!   PTS/duration in encoder timescale (Opus = 48000 ticks per second
//!   per RFC 7845 §4.1).
//! - [`AudioDecoder`] / [`AudioEncoder`] traits are object-safe so
//!   pipeline code can hand out `Box<dyn AudioEncoder>`.
//!
//! Pre-skip + extra_data contract (Opus-specific)
//! ----------------------------------------------
//! [`AudioEncoder::pre_skip`] returns the number of *48 kHz* samples of
//! lookahead the libopus encoder injects (queried via
//! `OPUS_GET_LOOKAHEAD` and reported in 48 kHz ticks no matter the
//! configured rate). Squad-23's mux side writes this into the `dOps`
//! body so a conformant decoder discards the lookahead at the start of
//! the file.
//!
//! [`AudioEncoder::extra_data`] returns the `dOps` body bytes per RFC
//! 7845 §4.5: 11 bytes minimum, channel-mapping family 0 (mono/stereo).
//! Multistream (>2 channels) is out of scope for this sprint and
//! returns [`AudioError::Unsupported`].

pub mod decode;
pub mod encode;
pub mod resample;

#[derive(thiserror::Error, Debug)]
pub enum AudioError {
    #[error("decode failed: {0}")]
    Decode(String),
    #[error("encode failed: {0}")]
    Encode(String),
    #[error("resample failed: {0}")]
    Resample(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// One decoded audio frame.
///
/// `samples` is interleaved planar — for stereo the layout is
/// `[L0, R0, L1, R1, ...]`, length `frames * channels`. Values are
/// f32 in `[-1.0, 1.0]`. The encoder side accepts the same layout.
#[derive(Clone, Debug)]
pub struct AudioFrame {
    /// Interleaved planar samples (LRLRLR for stereo) in `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u8,
    /// Presentation timestamp, microseconds, signed (allows negative
    /// pre-roll positions for codecs that emit lookahead frames before
    /// PTS=0 — Opus uses pre_skip rather than negative PTS, but this
    /// keeps the type general).
    pub pts: i64,
}

/// One encoded audio packet leaving the encoder.
#[derive(Clone, Debug)]
pub struct EncodedAudioPacket {
    pub data: Vec<u8>,
    /// PTS in microseconds (matches `AudioFrame::pts` domain).
    pub pts: i64,
    /// Duration in encoder timescale ticks. For Opus this is 48000
    /// ticks/sec (one 20 ms frame = 960 ticks).
    pub duration: i64,
}

#[derive(Clone, Debug)]
pub struct AudioEncoderConfig {
    pub codec: AudioCodec,
    /// Input sample rate the caller will feed [`AudioEncoder::encode`].
    /// The encoder transparently resamples to its native rate (48 kHz
    /// for Opus) when this differs.
    pub sample_rate: u32,
    pub channels: u8,
    /// Target bitrate in bits per second.
    pub bitrate: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioCodec {
    Opus,
}

pub trait AudioDecoder: Send {
    /// Decode one input packet at the given PTS (microseconds). May
    /// return zero or more output frames (zero is normal — some
    /// decoders need to see two frames before emitting one).
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<AudioFrame>, AudioError>;

    /// Drain any frames buffered inside the decoder. Call once at EOS.
    fn flush(&mut self) -> Result<Vec<AudioFrame>, AudioError>;
}

pub trait AudioEncoder: Send {
    /// Encode one input frame. The encoder buffers up to one output
    /// frame's worth of samples internally — Opus's smallest frame is
    /// 2.5 ms, default 20 ms — so this returns 0..N packets.
    fn encode(&mut self, frame: &AudioFrame) -> Result<Vec<EncodedAudioPacket>, AudioError>;

    /// Drain any buffered samples. May produce a final partial packet.
    fn flush(&mut self) -> Result<Vec<EncodedAudioPacket>, AudioError>;

    /// Lookahead samples at 48 kHz (Opus convention). For Opus,
    /// queried via `OPUS_GET_LOOKAHEAD` and scaled to 48 kHz when the
    /// encoder is internally running at a non-48k rate.
    fn pre_skip(&self) -> u16;

    /// The codec-specific extra_data the muxer puts in the sample
    /// entry's config box. For Opus this is the `dOps` body per RFC
    /// 7845 §4.5 (11 bytes for channel-mapping family 0).
    fn extra_data(&self) -> Vec<u8>;
}

/// Construct an audio decoder for the given codec name.
///
/// `codec` is matched case-insensitively. Supported tokens:
/// - `mp3` / `mpeg`
/// - `vorbis` (raw audio packet form — caller is responsible for
///   feeding the three Xiph setup packets first via the `extra_data`
///   parameter on first construction, then the audio packets via
///   `decode`)
///
/// `extra_data`, `sample_rate`, and `channels` come from the demux
/// side's container metadata. For codecs that carry full setup in the
/// stream (MP3) `extra_data` may be `None`.
pub fn create_decoder(
    codec: &str,
    extra_data: Option<&[u8]>,
    sample_rate: u32,
    channels: u8,
) -> Result<Box<dyn AudioDecoder>, AudioError> {
    match codec.to_ascii_lowercase().as_str() {
        "mp3" | "mpeg" | "mp3a" => Ok(Box::new(decode::mp3::Mp3Decoder::new(
            sample_rate,
            channels,
        )?)),
        "vorbis" => Ok(Box::new(decode::vorbis::VorbisDecoder::new(
            extra_data,
            sample_rate,
            channels,
        )?)),
        other => Err(AudioError::Unsupported(format!(
            "audio decoder for codec {other}"
        ))),
    }
}

/// Construct an audio encoder.
pub fn create_encoder(config: AudioEncoderConfig) -> Result<Box<dyn AudioEncoder>, AudioError> {
    match config.codec {
        AudioCodec::Opus => Ok(Box::new(encode::opus::OpusEncoder::new(config)?)),
    }
}
