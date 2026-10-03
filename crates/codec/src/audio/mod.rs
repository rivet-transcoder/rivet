//! Audio codec framework: the decoder/encoder traits, the wire types the
//! job layer's audio pipeline consumes, and the constructors.
//!
//! Every codec is this workspace's own, written clean-room in its own
//! repository and brought in as a submodule: Opus (`crates/opus`), MPEG
//! audio / MP3 (`crates/mp3`), Vorbis (`crates/vorbis`), AAC-LC / HE-AAC /
//! HE-AAC v2 (`crates/aac`), AC-3 / E-AC-3 (`crates/ac3`), DTS (`crates/dts`)
//! and FLAC / ALAC (`crates/lossless`). No C library, no build script, no
//! third-party codec crate. Decoders cover all of them plus linear PCM;
//! encoders all of them (DTS: the core; E-AC-3: up to 5.1).
//!
//! Wire model
//! ----------
//! - [`AudioFrame`] is the canonical PCM exchange type: f32 in
//!   [-1.0, 1.0], interleaved (LRLRLR for stereo), in the pipeline's native
//!   channel order for the layout, with the sample rate and channel count
//!   carried alongside and a microsecond-domain PTS.
//! - [`EncodedAudioPacket`] carries one encoder output packet plus
//!   PTS/duration in the encoder's timescale ([`AudioEncoder::sample_rate`]:
//!   48 000 ticks per second for Opus, the coded rate for the others).
//! - [`AudioDecoder`] / [`AudioEncoder`] traits are object-safe so
//!   pipeline code can hand out `Box<dyn AudioEncoder>`.
//!
//! Pre-skip and extra data
//! -----------------------
//! [`AudioEncoder::pre_skip`] is the number of samples, at the encoder's
//! rate, the decoded stream starts with that are not the input's (the
//! priming an MP4 edit list, an `OpusHead` or a tag frame hides).
//! [`AudioEncoder::extra_data`] is the codec configuration the container's
//! sample entry carries: the `OpusHead` body, the AudioSpecificConfig, the
//! Vorbis headers in Xiph lacing, the FLAC metadata blocks, the ALAC cookie
//! (empty for MP3, AC-3, E-AC-3 and DTS, whose frames carry their own; the
//! muxer derives `dac3` / `dec3` / `ddts` from the first frame).

pub mod decode;
pub mod encode;
pub mod filter;
pub mod remix;
/// FLAC and ALAC: this workspace's own codecs (`crates/lossless`, the
/// rivet-lossless repository), re-exported where they used to live.
pub use ::lossless;
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
    /// The encoder resamples to a rate it codes when this is not one
    /// (48 kHz for Opus; the nearest coded rate for the others).
    pub sample_rate: u32,
    pub channels: u8,
    /// Target bitrate in bits per second; 0 picks the codec's default for
    /// the channel count. Ignored by Vorbis (see `quality`) and the lossless
    /// codecs.
    pub bitrate: u32,
    /// Vorbis quality, -1 (smallest) to 10 (best); `None` is 5. Ignored by
    /// the other codecs.
    pub quality: Option<f32>,
    /// The speakers of the input's channels, in order, when they are not
    /// [`ChannelLayout::default_for`](filter::ChannelLayout::default_for)
    /// the count: what AC-3 and DTS, whose channel arrangements a count
    /// alone does not name (4 channels are 4.0, quad(side) or 3.1), code.
    pub layout: Option<filter::ChannelLayout>,
}

impl AudioEncoderConfig {
    /// `codec` at `sample_rate` / `channels`, `bitrate` (0: the default) and
    /// the default quality.
    pub fn new(codec: AudioCodec, sample_rate: u32, channels: u8, bitrate: u32) -> Self {
        Self { codec, sample_rate, channels, bitrate, quality: None, layout: None }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioCodec {
    Opus,
    /// MPEG-1 Audio Layer III (`crates/mp3`).
    Mp3,
    /// AAC-LC (`crates/aac`, through `encode::aac`).
    Aac,
    /// HE-AAC: AAC-LC at half the rate plus spectral band replication.
    HeAac,
    /// HE-AAC v2: HE-AAC with parametric stereo (stereo only).
    HeAacV2,
    /// Vorbis I (`crates/vorbis`), variable rate by quality.
    Vorbis,
    /// AC-3 (Dolby Digital), `crates/ac3`.
    Ac3,
    /// E-AC-3 (Dolby Digital Plus), `crates/ac3`.
    Eac3,
    /// DTS Coherent Acoustics core (`crates/dts`).
    Dts,
    /// FLAC at the given bit depth (4–32) and effort. `bitrate` is ignored.
    Flac { bits_per_sample: u8, level: encode::flac::FlacLevel },
    /// ALAC at the given bit depth (16, 20, 24 or 32). `bitrate` is ignored.
    Alac { bits_per_sample: u8 },
}

impl AudioCodec {
    /// The short name a job reports the output under: `opus`, `mp3`, `aac`,
    /// `he-aac`, `he-aacv2`, `vorbis`, `ac3`, `eac3`, `dts`, `flac`, `alac`.
    pub fn name(self) -> &'static str {
        match self {
            AudioCodec::Opus => "opus",
            AudioCodec::Mp3 => "mp3",
            AudioCodec::Aac => "aac",
            AudioCodec::HeAac => "he-aac",
            AudioCodec::HeAacV2 => "he-aacv2",
            AudioCodec::Vorbis => "vorbis",
            AudioCodec::Ac3 => "ac3",
            AudioCodec::Eac3 => "eac3",
            AudioCodec::Dts => "dts",
            AudioCodec::Flac { .. } => "flac",
            AudioCodec::Alac { .. } => "alac",
        }
    }

    /// The codec tag of the stream it writes, as a container's `AudioInfo`
    /// and a demuxer's track name it: `aac` for all three AAC profiles.
    pub fn stream_codec(self) -> &'static str {
        match self {
            AudioCodec::HeAac | AudioCodec::HeAacV2 => "aac",
            other => other.name(),
        }
    }
}

pub trait AudioDecoder: Send {
    /// Decode one input packet at the given PTS (microseconds). May
    /// return zero or more output frames (zero is normal — some
    /// decoders need to see two frames before emitting one).
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<AudioFrame>, AudioError>;

    /// Drain any frames buffered inside the decoder. Call once at EOS.
    fn flush(&mut self) -> Result<Vec<AudioFrame>, AudioError>;

    /// The speakers the frames last returned carry, in slot order, when the
    /// stream names them — AC-3's `acmod`, DTS's `AMODE` — and they are not
    /// simply [`ChannelLayout::default_for`](filter::ChannelLayout::default_for)
    /// the channel count (a 6-channel AC-3 stream is 5.1(side), a 4-channel
    /// one 4.0, quad(side) or 3.1). `None` means the default for the count.
    fn layout(&self) -> Option<filter::ChannelLayout> {
        None
    }
}

pub trait AudioEncoder: Send {
    /// Encode one input frame. The encoder buffers up to one output
    /// frame's worth of samples internally — Opus's smallest frame is
    /// 2.5 ms, default 20 ms — so this returns 0..N packets.
    fn encode(&mut self, frame: &AudioFrame) -> Result<Vec<EncodedAudioPacket>, AudioError>;

    /// Drain any buffered samples. May produce a final partial packet.
    fn flush(&mut self) -> Result<Vec<EncodedAudioPacket>, AudioError>;

    /// Samples, at [`Self::sample_rate`], the decoded stream starts with that
    /// are not the input's. For Opus the lookahead from
    /// `OPUS_GET_LOOKAHEAD` (the `dOps` PreSkip, at 48 kHz); for MP3 the
    /// encoder's and decoder's delay.
    fn pre_skip(&self) -> u16;

    /// The codec-specific extra_data the muxer puts in the sample
    /// entry's config box. For Opus this is the `dOps` body per RFC
    /// 7845 §4.5 (11 bytes for channel-mapping family 0).
    fn extra_data(&self) -> Vec<u8>;

    /// The rate the encoded stream is coded at, which is also the timescale
    /// of [`EncodedAudioPacket::duration`]: 48 kHz for Opus whatever the
    /// input, the input's own (or the nearest MPEG-1 rate) for MP3, the
    /// input's own (or the nearest AAC rate, `encode::aac::coding_rate`) for
    /// AAC, the input's own for the lossless codecs.
    fn sample_rate(&self) -> u32 {
        48_000
    }

    /// What a bare file of this stream starts with before its first packet,
    /// once the encoder has been flushed: MP3's `Info` tag frame, which
    /// carries the encoder delay and padding a gapless player trims. `None`
    /// for the other codecs (and before the flush).
    fn file_header(&self) -> Option<Vec<u8>> {
        None
    }
}

/// The pipeline's interleaved channel order is ffmpeg's native order for
/// the layout (`filter::channelmap`'s default layouts: 5.1 = FL FR FC LFE
/// BL BR). Opus channel-mapping family 1 (RFC 7845 §5.1.1.2) and Vorbis
/// order their channels differently (5.1 = FL FC FR RL RR LFE), so at those
/// two boundaries the samples are permuted. For 3..=8 channels this gives,
/// for each RFC / Vorbis slot, the native slot it carries; mono and stereo
/// are identical in both and return `None`.
pub fn rfc7845_family1_order(channels: u8) -> Option<&'static [usize]> {
    // native:  3.0 FL FR FC | quad FL FR BL BR | 5.0 FL FR FC BL BR
    //          5.1 FL FR FC LFE BL BR | 6.1 FL FR FC LFE BC SL SR
    //          7.1 FL FR FC LFE BL BR SL SR
    // RFC:     3 L C R | 4 FL FR RL RR | 5 FL FC FR RL RR | 6 FL FC FR RL RR LFE
    //          7 FL FC FR SL SR RC LFE | 8 FL FC FR SL SR RL RR LFE
    match channels {
        3 => Some(&[0, 2, 1]),
        4 => Some(&[0, 1, 2, 3]),
        5 => Some(&[0, 2, 1, 3, 4]),
        6 => Some(&[0, 2, 1, 4, 5, 3]),
        7 => Some(&[0, 2, 1, 5, 6, 4, 3]),
        8 => Some(&[0, 2, 1, 6, 7, 4, 5, 3]),
        _ => None,
    }
}

/// Construct an audio decoder for the given codec name.
///
/// `codec` is matched case-insensitively. Supported tokens:
/// - `aac` / `mp4a` (AAC-LC, HE-AAC and HE-AAC v2, the last two at their
///   full SBR rate; `extra_data` is the AudioSpecificConfig and packets are
///   raw access units, or with no `extra_data` the packets are ADTS)
/// - `mp3` / `mpeg` (and `mp2` / `mp1`: Layers I and II as well)
/// - `ac3` / `eac3` (one or more syncframes per packet; the decoder
///   resynchronises on 0x0B77 and buffers partial frames)
/// - `vorbis` (raw audio packet form — caller is responsible for
///   feeding the three Xiph setup packets first via the `extra_data`
///   parameter on first construction, then the audio packets via
///   `decode`)
/// - `dts` / `dca` / `dtsc` (DTS Coherent Acoustics core; packets are
///   whole core frames, optionally followed by a DTS-HD extension
///   substream, which is skipped)
/// - `opus` (`extra_data` is the `OpusHead` body, which carries the stream
///   layout of a surround track; output is 48 kHz and includes the
///   pre-skip)
/// - `flac` (one frame per packet; `extra_data` is the metadata blocks,
///   with or without the `fLaC` marker or the `dfLa` version/flags, and
///   supplies what a frame header defers to STREAMINFO)
/// - `alac` (one frame per packet; `extra_data` is the magic cookie, bare
///   or in its `alac` atom, and is required)
/// - `pcm_u8` / `pcm_s16le` / `pcm_s24le` / `pcm_s32le` / `pcm_f32le` /
///   `pcm_f64le` (linear PCM in WAVE channel order; packets are byte runs
///   that need not end on a sample frame)
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
        // Layers I and II decode as well.
        "mp3" | "mpeg" | "mp3a" | "mp2" | "mp1" => Ok(Box::new(decode::mp3::Mp3Decoder::new(
            sample_rate,
            channels,
        )?)),
        "vorbis" => Ok(Box::new(decode::vorbis::VorbisDecoder::new(
            extra_data,
            sample_rate,
            channels,
        )?)),
        // DTS Coherent Acoustics core (MKV `A_DTS` / MP4 `dtsc`). The
        // container's rate/channels are only a cross-check — the core frame
        // header is authoritative.
        "dts" | "dca" | "dtsc" => Ok(Box::new(decode::dts::DtsDecoder::new(
            sample_rate,
            channels,
        )?)),
        // AC-3 and E-AC-3 share one decoder: the bsid field of each
        // syncframe selects the syntax, and `extra_data` (dac3/dec3) carries
        // nothing the frames don't.
        "ac3" | "ac-3" | "eac3" | "ec-3" | "e-ac-3" => Ok(Box::new(decode::ac3::Ac3Decoder::new(
            sample_rate,
            channels,
        )?)),
        "opus" => Ok(Box::new(decode::opus::OpusDecoder::new(extra_data, channels)?)),
        // AAC: the AudioSpecificConfig, or ADTS framing without one. HE-AAC
        // and HE-AAC v2 decode in full, at the SBR rate.
        "aac" | "mp4a" => Ok(Box::new(decode::aac::AacDecoder::new(extra_data)?)),
        // Lossless: FLAC (MP4 `fLaC`, Matroska `A_FLAC`, native streams) and
        // ALAC (MP4 `alac`, Matroska `A_ALAC`).
        "flac" => Ok(Box::new(decode::flac::FlacDecoder::new(extra_data, sample_rate, channels)?)),
        "alac" => Ok(Box::new(decode::alac::AlacDecoder::new(extra_data)?)),
        // Linear PCM (AVI's WAVE formats): the bytes are the samples.
        "pcm_u8" | "pcm_s16le" | "pcm_s24le" | "pcm_s32le" | "pcm_f32le" | "pcm_f64le" => Ok(Box::new(
            decode::pcm::PcmDecoder::new(&codec.to_ascii_lowercase(), sample_rate, channels)?,
        )),
        other => Err(AudioError::Unsupported(format!(
            "audio decoder for codec {other}"
        ))),
    }
}

/// Construct an audio encoder.
pub fn create_encoder(config: AudioEncoderConfig) -> Result<Box<dyn AudioEncoder>, AudioError> {
    use encode::aac::{AacConfig, AacEncoder, Profile};
    let aac_config = AacConfig { sample_rate: config.sample_rate, channels: config.channels, bitrate: config.bitrate };
    let aac = |profile| AacEncoder::with_profile(aac_config.clone(), profile);
    match config.codec {
        AudioCodec::Opus => Ok(Box::new(encode::opus::OpusEncoder::new(config)?)),
        AudioCodec::Mp3 => Ok(Box::new(encode::mp3::Mp3Encoder::new(config)?)),
        AudioCodec::Aac => Ok(Box::new(aac(Profile::Lc)?)),
        AudioCodec::HeAac => Ok(Box::new(aac(Profile::HeAac)?)),
        AudioCodec::HeAacV2 => Ok(Box::new(aac(Profile::HeAacV2)?)),
        AudioCodec::Vorbis => Ok(Box::new(encode::vorbis::VorbisEncoder::new(&config)?)),
        AudioCodec::Ac3 | AudioCodec::Eac3 => Ok(Box::new(encode::ac3::Ac3Encoder::new(&config)?)),
        AudioCodec::Dts => Ok(Box::new(encode::dts::DtsEncoder::new(&config)?)),
        AudioCodec::Flac { bits_per_sample, level } => {
            Ok(Box::new(encode::flac::FlacAudioEncoder::new(&config, bits_per_sample, level)?))
        }
        AudioCodec::Alac { bits_per_sample } => {
            Ok(Box::new(encode::alac::AlacAudioEncoder::new(&config, bits_per_sample)?))
        }
    }
}

// ---- MP3 output parameters ----

/// The MPEG-1 Layer III bitrates, bits per second (ISO/IEC 11172-3
/// §2.4.2.3, free format excluded). CBR output is one of these.
pub const MP3_BITRATES: [u32; 14] = [
    32_000, 40_000, 48_000, 56_000, 64_000, 80_000, 96_000, 112_000, 128_000, 160_000, 192_000,
    224_000, 256_000, 320_000,
];

/// Samples per channel in one MPEG-1 Layer III frame.
pub const MP3_FRAME_SAMPLES: u32 = 1152;

/// The Layer III decoder's own delay, in samples: the synthesis filterbank's
/// 528 plus the one-sample offset every decoder since the ISO reference
/// shares. Players that read a LAME-style tag add it to the tag's encoder
/// delay.
pub const MP3_DECODER_DELAY: u32 = ::mp3::xing::DECODER_DELAY;

const MP3_DEFAULT_BITRATE_MONO: u32 = 64_000;
const MP3_DEFAULT_BITRATE_STEREO: u32 = 128_000;

/// The rate MP3 codes a source of `input` Hz at: an MPEG-1 rate, the one of
/// the input's family.
pub fn mp3_sample_rate(input: u32) -> u32 {
    match input {
        32_000 | 44_100 | 48_000 => input,
        r if r % 11_025 == 0 => 44_100,
        _ => 48_000,
    }
}

/// The CBR default for `channels`.
pub fn mp3_default_bitrate(channels: u8) -> u32 {
    if channels == 1 { MP3_DEFAULT_BITRATE_MONO } else { MP3_DEFAULT_BITRATE_STEREO }
}

/// The rate AC-3, E-AC-3 and DTS code a source of `input` Hz at: 48, 44.1 or
/// 32 kHz, the input's own when it is one, else 44.1 kHz for its family and
/// 48 kHz for the rest.
pub fn dolby_dts_sample_rate(input: u32) -> u32 {
    match input {
        32_000 | 44_100 | 48_000 => input,
        r if r % 11_025 == 0 => 44_100,
        _ => 48_000,
    }
}

impl From<lossless::Error> for AudioError {
    fn from(e: lossless::Error) -> Self {
        match e {
            lossless::Error::Invalid(m) => AudioError::Decode(m),
            lossless::Error::Unsupported(m) => AudioError::Unsupported(m),
        }
    }
}
