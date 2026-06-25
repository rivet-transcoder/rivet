pub mod aac_asc;
pub mod ac3_sync;
pub(crate) mod annexb;
pub mod avi;
pub mod cmaf;
pub mod demux;
pub mod hls;
pub mod mp4_sanitize;
pub mod mux;
pub mod streaming;
pub mod ts;

/// Parameters required to bolt an audio track onto `Av1Mp4Muxer`.
///
/// Four codec families are supported:
/// - **AAC-LC** (Squad-18, task #63 v1): mono or stereo, sample_rate as the
///   mdhd timescale per ISO/IEC 14496-14 standard practice, and the
///   AudioSpecificConfig surfaced verbatim from the demuxer (see
///   `demux::AudioTrack::asc`) so HE-AAC / xHE-AAC signalling bits survive
///   the passthrough intact. Sample entry: `mp4a` + `esds`.
/// - **Opus** (Squad-23): mono or stereo, sample_rate is the source's
///   `InputSampleRate` (typically 48000), `mdhd` timescale is pinned at
///   48000 per RFC 7845 §3 (Opus internally always operates at 48 kHz).
///   Sample entry: `Opus` (4cc per RFC 7845 §4.4 — capital O) + `dOps`
///   (Opus-Specific Box per §4.5). The `OpusHead` body bytes are carried
///   in `codec_private` and emitted verbatim inside `dOps`.
/// - **AC-3** / Dolby Digital (Squad-26): up to 5.1 channels, sample_rate
///   from the source's syncframe (32 / 44.1 / 48 kHz). Sample entry:
///   `ac-3` + `dac3` (ETSI TS 102 366 §F.4 / Annex F). The 3-byte `dac3`
///   body is carried in `codec_private` and emitted verbatim.
/// - **E-AC-3** / Dolby Digital Plus (Squad-26): up to 5.1 channels in v1
///   scope (single independent substream). Sample entry: `ec-3` + `dec3`
///   (ETSI TS 102 366 §F.6). The `dec3` body is carried in `codec_private`
///   and emitted verbatim.
///
/// Discriminator: `codec` field. `"aac"` → AAC path; `"opus"` → Opus path;
/// `"ac3"` → AC-3 path; `"eac3"` → E-AC-3 path. Anything else is rejected
/// at `with_audio()` time.
#[derive(Debug, Clone)]
pub struct AudioInfo {
    /// Human-readable codec tag. Muxer accepts `"aac"` (case-insensitive)
    /// and `"opus"` (case-insensitive). Anything else is rejected with a
    /// clear error — this is intentional (no stubs).
    pub codec: String,
    /// Audio sample rate in Hz. For AAC: typically 44100 / 48000; doubles as
    /// the `mdhd` timescale. For Opus: the source's `InputSampleRate`
    /// (informational; the mdhd timescale is pinned to 48000 per RFC 7845
    /// regardless of this value).
    pub sample_rate: u32,
    /// Channel count. Both codecs support 1 (mono) and 2 (stereo) only;
    /// the muxer bails on other values.
    pub channels: u16,
    /// Audio timescale in ticks per second. AAC: equals `sample_rate`.
    /// Opus: caller should pass 48000 (RFC 7845); the muxer additionally
    /// validates this for the Opus path.
    pub timescale: u32,
    /// AudioSpecificConfig bytes verbatim from the demuxer (AAC only).
    /// Embedded into the `esds` box's DecoderSpecificInfo (tag 0x05)
    /// payload. Empty for non-AAC codecs.
    pub asc_bytes: Vec<u8>,
    /// Codec-private body bytes (Opus / AC-3 / E-AC-3). For Opus this MUST
    /// be the RFC 7845 §5.1 `OpusHead` payload (the same bytes a WebM/MKV
    /// `CodecPrivate` element would carry; see RFC 7845 §5.2 for the
    /// MKV mapping). Emitted verbatim as the body of the `dOps` box
    /// inside the `Opus` sample entry. For AC-3 this carries the 3-byte
    /// `dac3` body (ETSI TS 102 366 §F.4); for E-AC-3 the variable-size
    /// `dec3` body (§F.6). Empty for AAC.
    ///
    /// Layout (RFC 7845 §5.1, 19 bytes minimum for ChannelMappingFamily=0
    /// with the 8-byte 'OpusHead' magic prefix; the magic is NOT carried
    /// in `dOps` — only the post-magic body, which is 11 bytes minimum):
    ///   - `Version` u8 = 1 (in OpusHead; mapped to 0 in dOps per §4.5)
    ///   - `OutputChannelCount` u8
    ///   - `PreSkip` u16 LE  (in OpusHead; converted to BE for dOps per §4.5)
    ///   - `InputSampleRate` u32 LE  (LE in OpusHead, BE in dOps)
    ///   - `OutputGain` i16 LE  (LE in OpusHead, BE in dOps)
    ///   - `ChannelMappingFamily` u8
    ///   - (if family != 0: 1 + 1 + N additional bytes)
    ///
    /// The byte-order conversion between OpusHead (Ogg LE convention) and
    /// dOps (ISOBMFF BE convention) is handled by `build_dops` in mux.rs.
    /// Callers should pass the OpusHead bytes (LE numeric fields) — that's
    /// the form the MKV / WebM demuxer surfaces directly out of CodecPrivate.
    pub codec_private: Vec<u8>,
}

impl AudioInfo {
    /// Convenience constructor for the AAC-LC path. Mirrors Squad-18's
    /// original API surface so existing AAC call sites stay terse.
    pub fn aac_lc(sample_rate: u32, channels: u16, asc_bytes: Vec<u8>) -> Self {
        Self {
            codec: "aac".into(),
            sample_rate,
            channels,
            timescale: sample_rate,
            asc_bytes,
            codec_private: Vec::new(),
        }
    }

    /// Convenience constructor for the Opus path. Pins timescale to 48000
    /// per RFC 7845 §3 — Opus is internally always 48 kHz so the mdhd
    /// timescale, not the source's nominal `InputSampleRate`, is what
    /// drives sample-duration math on every player.
    pub fn opus(input_sample_rate: u32, channels: u16, codec_private: Vec<u8>) -> Self {
        Self {
            codec: "opus".into(),
            sample_rate: input_sample_rate,
            channels,
            timescale: 48_000,
            asc_bytes: Vec::new(),
            codec_private,
        }
    }

    /// Convenience constructor for the AC-3 (Dolby Digital) passthrough
    /// path (Squad-26). `codec_private` carries the 3-byte `dac3` body
    /// payload (ETSI TS 102 366 §F.4) the muxer writes verbatim into the
    /// `dac3` box. mdhd timescale = sample_rate (48000 / 44100 / 32000) —
    /// AC-3 doesn't have Opus's "internally fixed at 48 kHz" rule.
    pub fn ac3(sample_rate: u32, channels: u16, dac3_body: Vec<u8>) -> Self {
        Self {
            codec: "ac3".into(),
            sample_rate,
            channels,
            timescale: sample_rate,
            asc_bytes: Vec::new(),
            codec_private: dac3_body,
        }
    }

    /// Convenience constructor for the E-AC-3 (Dolby Digital Plus) passthrough
    /// path (Squad-26). `codec_private` carries the `dec3` body payload
    /// (ETSI TS 102 366 §F.6) — variable size based on substream count;
    /// minimum ~5 bytes for the single-independent-substream case.
    pub fn eac3(sample_rate: u32, channels: u16, dec3_body: Vec<u8>) -> Self {
        Self {
            codec: "eac3".into(),
            sample_rate,
            channels,
            timescale: sample_rate,
            asc_bytes: Vec::new(),
            codec_private: dec3_body,
        }
    }
}

/// Extended MKV colour/mastering metadata parsed from `Segment → Tracks →
/// TrackEntry → Video → Colour` and its nested `MasteringMetadata`. The
/// core H.273-equivalent fields (matrix / primaries / transfer /
/// full-range) round-trip through `StreamInfo.color_metadata` on
/// `DemuxResult`; this struct exists to carry the rest (bits_per_channel,
/// chroma siting / subsampling offsets, MaxCLL/MaxFALL, SMPTE-2086
/// mastering chromaticities) without requiring a breaking extension of
/// the shared `StreamInfo` type in the `codec` crate.
///
/// Populated by `demux::probe_mkv_color_info` for callers that need it
/// (mux HDR signalling, future SEI passthrough). Returns `None` for
/// non-MKV containers and for MKVs with no `Colour` element.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MkvColorInfo {
    /// MatroskaElement 0x55B2 — decoded bits per channel (e.g. 10 for
    /// HDR10 sources).
    pub bits_per_channel: Option<u8>,
    /// MatroskaElement 0x55B3 — Cb/Cr horizontal subsampling ratio.
    pub chroma_subsampling_horz: Option<u8>,
    /// MatroskaElement 0x55B4 — Cb/Cr vertical subsampling ratio.
    pub chroma_subsampling_vert: Option<u8>,
    /// MatroskaElement 0x55B7 — horizontal chroma siting (0=unspecified,
    /// 1=left-collocated, 2=half).
    pub chroma_siting_horz: Option<u8>,
    /// MatroskaElement 0x55B8 — vertical chroma siting.
    pub chroma_siting_vert: Option<u8>,
    /// MatroskaElement 0x55BC — MaxCLL in cd/m².
    pub max_cll: Option<u32>,
    /// MatroskaElement 0x55BD — MaxFALL in cd/m².
    pub max_fall: Option<u32>,
    /// MatroskaElement 0x55D0 nested — SMPTE ST 2086 mastering display
    /// primaries + luminance. Emitted when any sub-element is present.
    pub mastering: Option<MkvMasteringMetadata>,
}

/// SMPTE ST 2086 mastering display metadata, carried verbatim from the
/// MKV `MasteringMetadata` sub-element. Used by HDR10 mux and by future
/// SEI-passthrough paths to preserve the creator-intended display gamut
/// and min/max luminance.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MkvMasteringMetadata {
    pub primary_r_chromaticity_x: Option<f64>,
    pub primary_r_chromaticity_y: Option<f64>,
    pub primary_g_chromaticity_x: Option<f64>,
    pub primary_g_chromaticity_y: Option<f64>,
    pub primary_b_chromaticity_x: Option<f64>,
    pub primary_b_chromaticity_y: Option<f64>,
    pub white_point_chromaticity_x: Option<f64>,
    pub white_point_chromaticity_y: Option<f64>,
    pub luminance_max: Option<f64>,
    pub luminance_min: Option<f64>,
}
