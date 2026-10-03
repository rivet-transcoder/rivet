//! Policy enums — how video/audio codec, container, muxer, output-mode, color,
//! bit-depth, encode/decode distribution, chunk-seam handling, and GPU family
//! are selected. All types are `pub` and re-exported from the parent `spec`
//! module so callers reach them as `rivet::spec::VideoCodecPolicy`, etc.

use codec::frame::VideoCodec;

pub use codec::frame::ProresProfile;

/// Output **video** codec policy — the video analogue of [`AudioCodecPolicy`].
/// Selects which codec the encoder produces:
/// - `Av1` *(default)* — royalty-clean (AV1 + Opus in MP4 = zero royalty exposure).
/// - `H264` / `H265` — for legacy-player compatibility; they carry the
///   patent-licensing obligations AV1 was chosen to avoid.
/// - `Vp9` / `Vp8` — WebM's codecs (VP9 also in MP4 and HLS), profile 0 /
///   8-bit 4:2:0.
/// - `Mpeg2` / `Mpeg4` — MPEG-2 Video and MPEG-4 Part 2 for players and
///   pipelines that want them (DVD-era hardware, broadcast ingest), 8-bit
///   4:2:0, in MP4 or a QuickTime movie.
/// - `ProRes(profile)` — Apple ProRes in a QuickTime movie, for editing:
///   intra-only, 4:2:2 or 4:4:4, 8- or 10-bit.
///
/// AV1, H.264 and H.265 work for single-file MP4 **and** CMAF/HLS, and so
/// does VP9; the others are single-file only ([`Container`] says which file
/// each goes in). The last five are encoded in software by rivet's own
/// encoders, in every build. Resolve to the encoder/muxer's [`VideoCodec`]
/// with [`VideoCodecPolicy::codec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VideoCodecPolicy {
    #[default]
    Av1,
    H264,
    H265,
    Vp8,
    Vp9,
    Mpeg2,
    Mpeg4,
    ProRes(ProresProfile),
}

impl VideoCodecPolicy {
    /// Resolve to the low-level [`VideoCodec`] the encoder + muxer consume.
    pub fn codec(self) -> VideoCodec {
        match self {
            VideoCodecPolicy::Av1 => VideoCodec::Av1,
            VideoCodecPolicy::H264 => VideoCodec::H264,
            VideoCodecPolicy::H265 => VideoCodec::H265,
            VideoCodecPolicy::Vp8 => VideoCodec::Vp8,
            VideoCodecPolicy::Vp9 => VideoCodec::Vp9,
            VideoCodecPolicy::Mpeg2 => VideoCodec::Mpeg2,
            VideoCodecPolicy::Mpeg4 => VideoCodec::Mpeg4,
            VideoCodecPolicy::ProRes(p) => VideoCodec::ProRes(p),
        }
    }

    /// The file a single-file output of this codec is when none is named:
    /// a QuickTime movie for ProRes, WebM for VP8 and VP9, MP4 otherwise.
    pub fn default_container(self) -> Container {
        match self {
            VideoCodecPolicy::ProRes(_) => Container::Mov,
            VideoCodecPolicy::Vp8 | VideoCodecPolicy::Vp9 => Container::WebM,
            _ => Container::Mp4,
        }
    }

    /// Whether `container` carries this codec in a single-file output: MP4
    /// takes everything but ProRes (`av01`, `avc1`, `hvc1`, `vp08`, `vp09`,
    /// `mp4v`); a QuickTime movie ProRes, H.264, H.265, MPEG-2 and MPEG-4;
    /// WebM VP8 and VP9.
    pub fn fits(self, container: Container) -> bool {
        match container {
            Container::Mp4 => !matches!(self, VideoCodecPolicy::ProRes(_)),
            Container::Mov => matches!(
                self,
                VideoCodecPolicy::ProRes(_)
                    | VideoCodecPolicy::H264
                    | VideoCodecPolicy::H265
                    | VideoCodecPolicy::Mpeg2
                    | VideoCodecPolicy::Mpeg4
            ),
            Container::WebM => matches!(self, VideoCodecPolicy::Vp8 | VideoCodecPolicy::Vp9),
            Container::Cmaf => self.hls_ready(),
            Container::Mp3 | Container::Flac | Container::M4a => false,
        }
    }

    /// Whether a CMAF / HLS package carries this codec: AV1, H.264, H.265
    /// and VP9. VP8, MPEG-2, MPEG-4 Part 2 and ProRes have no CMAF binding.
    pub fn hls_ready(self) -> bool {
        matches!(
            self,
            VideoCodecPolicy::Av1 | VideoCodecPolicy::H264 | VideoCodecPolicy::H265 | VideoCodecPolicy::Vp9
        )
    }

    /// Whether the multi-GPU single-file engine may encode this codec in
    /// chunks and stitch them. Only the web set: the encoders of the other
    /// five are software (the chunks would buy nothing on the GPUs the engine
    /// spreads over), and MPEG-2's open GOPs would not stand alone.
    pub fn chunkable(self) -> bool {
        matches!(self, VideoCodecPolicy::Av1 | VideoCodecPolicy::H264 | VideoCodecPolicy::H265)
    }

    /// The settings spelling: `av1`, `h264`, `h265`, `vp8`, `vp9`, `mpeg2`,
    /// `mpeg4`, `prores` (ProRes 422) or `prores-<profile>`.
    pub fn as_str(self) -> &'static str {
        super::caps::output_codec_label(self.codec())
    }
}

/// Output **audio** codec policy — how the source audio track is handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AudioCodecPolicy {
    /// Passthrough AAC / Opus / AC-3 / E-AC-3 / DTS verbatim, and MP3 into a
    /// single-file MP4; transcode the rest (Vorbis, MP2, PCM, MP3 for HLS)
    /// to Opus; drop anything else. For an [`OutputMode::AudioOnly`] `.mp3`
    /// file it means MP3: an MP3 source passes through, the rest is encoded;
    /// an audio-only `.m4a` takes what a single-file MP4 does.
    #[default]
    Auto,
    /// Keep/produce Opus: passthrough Opus, transcode everything else to Opus.
    /// Single-file MP4, HLS and an audio-only `.m4a` (`audio-container=mp4`);
    /// an audio-only `.mp3` refuses it.
    ForceOpus,
    /// Keep/produce MP3: passthrough MP3, encode everything else to MP3 (CBR,
    /// stereo at most — a surround source is downmixed). Needs the `lame`
    /// feature to encode. Single-file MP4 and audio-only output; not HLS.
    ForceMp3,
    /// Keep/produce AAC-LC: passthrough AAC, encode everything else to
    /// AAC-LC with rivet's own encoder (mono to 7.1, constant rate). The
    /// output that plays on every browser and device, older iOS and Safari
    /// included. Single-file MP4, HLS and an audio-only `.m4a`
    /// (`audio-container=mp4`); an audio-only `.mp3` refuses it.
    ForceAac,
    /// Drop audio entirely (video-only output).
    Drop,
    /// Lossless FLAC: a FLAC source is copied (at its own depth, or when
    /// [`AudioBitDepth`] names its depth), anything decodable is encoded.
    /// Plays from MP4 in Chrome, Edge, Firefox and Safari. Audio-only output
    /// is a native `.flac` (or an `.m4a`).
    Flac,
    /// Lossless ALAC (Apple Lossless), copied or encoded as for FLAC. Plays
    /// natively on Apple platforms and in Safari; not in Chrome, Edge or
    /// Firefox. Audio-only output is an `.m4a`.
    Alac,
}

impl AudioCodecPolicy {
    /// FLAC or ALAC.
    pub fn is_lossless(self) -> bool {
        matches!(self, Self::Flac | Self::Alac)
    }
}

/// Bit depth of lossless ([`AudioCodecPolicy::Flac`] / [`AudioCodecPolicy::Alac`])
/// audio output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AudioBitDepth {
    /// The source's: 16-bit for a source of 16 bits or fewer and for a lossy
    /// source, 24-bit for anything deeper. A 20-bit source is carried in 24
    /// bits exactly; a 32-bit or float source is rounded to 24.
    #[default]
    Source,
    /// 16-bit, rounding a deeper source to the nearest step (no dither).
    Sixteen,
    /// 24-bit.
    TwentyFour,
}

impl AudioBitDepth {
    /// The explicit depth, `None` for [`Self::Source`].
    pub fn bits(self) -> Option<u8> {
        match self {
            Self::Source => None,
            Self::Sixteen => Some(16),
            Self::TwentyFour => Some(24),
        }
    }
}

/// What becomes of an **HE-AAC** (or HE-AAC v2) source track. rivet decodes
/// AAC only as far as its AAC-LC core: spectral band replication and
/// parametric stereo are not implemented, so a decoded HE-AAC track comes
/// out at half its sample rate, with a quarter of its full rate's bandwidth
/// (and HE-AAC v2's core is mono). Passing the source through keeps all of
/// it; decoding the core is what a downmix, a filter or an output that
/// cannot hold AAC needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HeAacPolicy {
    /// Pass the source through where the output can carry it and only a
    /// codec change was asked (re-encoding the core would lose the top of
    /// the spectrum for nothing); decode the core only when the job needs
    /// PCM: a downmix, audio filters, a bare `.mp3` or native `.flac`.
    #[default]
    Auto,
    /// Never decode it: pass it through where the output can carry AAC, and
    /// refuse the job where it cannot, rather than lose the bandwidth.
    Passthrough,
    /// Decode the core whenever the job asks for another codec or a
    /// change, as for any AAC-LC track.
    Core,
}

impl HeAacPolicy {
    /// The settings word: `auto`, `passthrough`, `core`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Passthrough => "passthrough",
            Self::Core => "core",
        }
    }
}

/// Source audio codecs that may **not be decoded** (`audio-decode-deny`).
/// Empty (the default) restricts nothing.
///
/// A denied track is never handed to a decoder: it is passed through where
/// the output can carry it as it is (and only a codec change was asked, as
/// for a codec with no decoder), and the job is refused, naming this
/// setting, where the output needs its PCM — a downmix, an audio filter, a
/// bare `.mp3` or native `.flac`, or an output that cannot hold the codec.
/// An HE-AAC track under a denied `aac` is passed through whatever
/// [`HeAacPolicy`] says, since there is no core to decode.
///
/// The names are the decoders' ([`Self::CODECS`]); each source codec
/// spelling a decoder takes is denied under its name (`mp4a` as `aac`,
/// `ec-3` as `eac3`, every `pcm_*` as `pcm`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct AudioDecodeDeny(u16);

impl AudioDecodeDeny {
    /// Every name the setting takes.
    pub const CODECS: [&'static str; 11] =
        ["aac", "ac3", "alac", "dts", "eac3", "flac", "mp2", "mp3", "opus", "pcm", "vorbis"];

    /// No codec denied.
    pub const NONE: Self = Self(0);

    /// `self` with the codec `name` (one of [`Self::CODECS`]) denied too;
    /// `None` for a name that is not one.
    pub fn with(self, name: &str) -> Option<Self> {
        let i = Self::CODECS.iter().position(|c| *c == name)?;
        Some(Self(self.0 | 1 << i))
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The denied names, in [`Self::CODECS`] order.
    pub fn names(self) -> impl Iterator<Item = &'static str> {
        Self::CODECS.into_iter().enumerate().filter(move |(i, _)| self.0 & (1 << i) != 0).map(|(_, c)| c)
    }

    /// The settings value: the denied names, comma-separated (`""` for none).
    pub fn as_string(self) -> String {
        self.names().collect::<Vec<_>>().join(",")
    }

    /// The name a source track's codec (as a demuxer states it) decodes
    /// under, or `None` for one no audio decoder takes.
    pub fn name_of(codec: &str) -> Option<&'static str> {
        Some(match codec.to_ascii_lowercase().as_str() {
            "aac" | "mp4a" => "aac",
            "mp3" | "mpeg" | "mp3a" => "mp3",
            "mp2" | "mp1" => "mp2",
            "vorbis" => "vorbis",
            "opus" => "opus",
            "flac" => "flac",
            "alac" => "alac",
            "ac3" | "ac-3" => "ac3",
            "eac3" | "ec-3" | "e-ac-3" => "eac3",
            "dts" | "dca" | "dtsc" => "dts",
            c if c.starts_with("pcm_") => "pcm",
            _ => return None,
        })
    }

    /// Whether a source track in `codec` may not be decoded.
    pub fn denies(self, codec: &str) -> bool {
        Self::name_of(codec).is_some_and(|name| self.names().any(|d| d == name))
    }
}

/// Output **channel layout** — how many channels the audio comes out with.
///
/// `Source` keeps the source's layout wherever the output codec can carry
/// it: Opus carries 1–8 channels (a layout Opus has no mapping for goes out
/// in the narrowest one that has a place for every speaker, the missing ones
/// silent — 2.1 as 5.1, 4.0 as 5.0), MP3 at most two (a wider source is
/// downmixed to stereo). The others ask for that layout: a wider source is
/// **downmixed** (ITU-R BS.775, LFE dropped, normalised so nothing clips —
/// see `codec::audio::remix`), and a narrower one is **refused**: rivet does
/// not upmix, and asking for 5.1 from a stereo source is an error, never a
/// stereo file that claims otherwise or six channels made up from two.
///
/// Anything but `Source` on a source that already has that many channels
/// changes nothing (a passthrough stays a passthrough); otherwise the track
/// is decoded (an HE-AAC track as its AAC-LC core; see [`HeAacPolicy`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum AudioChannels {
    #[default]
    Source,
    Mono,
    Stereo,
    /// 5.1: FL FR FC LFE BL BR.
    Surround51,
    /// 7.1: FL FR FC LFE BL BR SL SR.
    Surround71,
}

impl AudioChannels {
    /// The layout asked for; `None` for `Source`.
    pub fn layout(self) -> Option<codec::audio::filter::ChannelLayout> {
        use codec::audio::filter::ChannelLayout;
        match self {
            AudioChannels::Source => None,
            AudioChannels::Mono => Some(ChannelLayout::named("mono")),
            AudioChannels::Stereo => Some(ChannelLayout::named("stereo")),
            AudioChannels::Surround51 => Some(ChannelLayout::named("5.1")),
            AudioChannels::Surround71 => Some(ChannelLayout::named("7.1")),
        }
    }

    /// The settings word: `source`, `mono`, `stereo`, `5.1`, `7.1`.
    pub fn as_str(self) -> &'static str {
        match self {
            AudioChannels::Source => "source",
            AudioChannels::Mono => "mono",
            AudioChannels::Stereo => "stereo",
            AudioChannels::Surround51 => "5.1",
            AudioChannels::Surround71 => "7.1",
        }
    }
}

/// Output **subtitle** policy — which of the source's text subtitle tracks
/// are carried.
///
/// **Text** subtitles (SRT / ASS / WebVTT out of Matroska, `tx3g` / `wvtt`
/// out of MP4) become one `tx3g` track per language in a single-file MP4 and
/// one segmented-WebVTT rendition per language in an HLS package. **Bitmap**
/// subtitles (PGS, VobSub, DVB) have no text representation and are dropped
/// with a warning under every policy.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SubtitlePolicy {
    /// Carry every text track, in source order. The default — it matches
    /// `ffmpeg -c:s copy` for the formats the outputs can hold.
    #[default]
    All,
    /// Emit no subtitle track.
    Drop,
    /// Carry only the tracks whose language is listed, in list order — the
    /// first listed language is the default HLS rendition. Codes match by
    /// language, not spelling: `eng`, `en` and `ENG` are one language, and
    /// `ger` is `deu`. A listed language no track carries is logged, not an
    /// error, so one manifest can serve a mixed library.
    Only(Vec<String>),
}

impl SubtitlePolicy {
    /// The tracks this policy keeps out of `tracks`, in output order.
    pub fn select<'a>(
        &self,
        tracks: &'a [container::demux::subtitle::SubtitleTrack],
    ) -> Vec<&'a container::demux::subtitle::SubtitleTrack> {
        match self {
            SubtitlePolicy::All => tracks.iter().collect(),
            SubtitlePolicy::Drop => Vec::new(),
            SubtitlePolicy::Only(langs) => {
                let mut out: Vec<&container::demux::subtitle::SubtitleTrack> = Vec::new();
                for lang in langs {
                    let mut hit = false;
                    for t in tracks {
                        if container::language::same_language(&t.language, lang)
                            && !out.iter().any(|o| std::ptr::eq(*o, t))
                        {
                            out.push(t);
                            hit = true;
                        }
                    }
                    if !hit {
                        tracing::warn!(
                            language = %lang,
                            available = ?tracks.iter().map(|t| t.language.as_str()).collect::<Vec<_>>(),
                            "subtitles: no text track in this language; skipping it"
                        );
                    }
                }
                out
            }
        }
    }
}

/// Deprecated alias for [`AudioCodecPolicy`] (renamed for symmetry with
/// [`VideoCodecPolicy`]).
#[deprecated(since = "0.1.5", note = "renamed to AudioCodecPolicy")]
pub type AudioPolicy = AudioCodecPolicy;

/// Output container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Container {
    /// Plain MP4 (ISO-BMFF), one self-contained file.
    #[default]
    Mp4,
    /// Fragmented MP4 (CMAF) — `moof`+`mdat` segments, for HLS/DASH.
    Cmaf,
    /// A bare `.mp3` file: MPEG audio frames behind an `Info` frame.
    Mp3,
    /// A native FLAC stream (`.flac`): FLAC only.
    Flac,
    /// An audio-only MP4 (`.m4a`), for any codec the MP4 muxer takes.
    M4a,
    /// A QuickTime movie (`.mov`): the MP4 muxer's box tree under the `qt  `
    /// brand. ProRes is written only into one.
    Mov,
    /// A WebM file (`.webm`, Matroska): VP8 or VP9 video, Opus audio.
    WebM,
}

impl Container {
    /// The settings word: `mp4`, `cmaf`, `mp3`, `flac`, `m4a`, `mov`, `webm`.
    pub fn as_str(self) -> &'static str {
        match self {
            Container::Mp4 => "mp4",
            Container::Cmaf => "cmaf",
            Container::Mp3 => "mp3",
            Container::Flac => "flac",
            Container::M4a => "m4a",
            Container::Mov => "mov",
            Container::WebM => "webm",
        }
    }

    /// The file as a refusal names it: `an MP4`, `a QuickTime movie`, ….
    pub fn file_label(self) -> &'static str {
        match self {
            Container::Mp4 => "an MP4",
            Container::Cmaf => "a CMAF package",
            Container::Mp3 => "an .mp3",
            Container::Flac => "a .flac",
            Container::M4a => "an .m4a",
            Container::Mov => "a QuickTime movie (.mov)",
            Container::WebM => "a WebM file",
        }
    }

    /// The muxer that writes a single-file output in this container.
    pub fn single_file_muxer(self) -> Option<Muxer> {
        match self {
            Container::Mp4 | Container::Mov => Some(Muxer::Mp4File),
            Container::WebM => Some(Muxer::WebmFile),
            _ => None,
        }
    }
}

/// Muxer — how the container bytes are assembled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Muxer {
    /// `Av1Mp4Muxer` — a single faststart MP4 with interleaved A/V.
    #[default]
    Mp4File,
    /// `CmafVideoMuxer` + `CmafAudioMuxer` + HLS playlists.
    CmafHls,
    /// `container::mp3::write_file`.
    Mp3File,
    /// `container::mux::write_native_flac`.
    FlacFile,
    /// `container::mux::write_audio_mp4`.
    M4aFile,
    /// `container::webm::WebmMuxer` — a single WebM file.
    WebmFile,
}

/// The high-level shape of the output.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputMode {
    /// One self-contained file per rung.
    SingleFile,
    /// Segmented CMAF + HLS: a media playlist per rung, a shared audio
    /// rendition, and a master playlist. `segment_seconds` is the target
    /// segment length (segments still break on keyframes).
    Hls { segment_seconds: f32 },
    /// The audio alone, as one file — an `.mp3`, or for lossless audio a
    /// native `.flac` or an `.m4a` ([`Container`]): no video is decoded or
    /// encoded and there are no rungs. Also what a single-file job becomes
    /// when its input has no video.
    AudioOnly,
}

impl Default for OutputMode {
    fn default() -> Self {
        OutputMode::SingleFile
    }
}

/// The decode plan — which card(s) decode, and whether the decode is one
/// pump or split into ranges across the cards. One enum, so the two halves
/// cannot contradict each other: "pin decode to card 2" and "split the decode
/// across every card" are not both sayable.
///
/// One decoder for the whole ladder is one decoder, and once the ladder is
/// wide enough it is the ceiling: every encoder waits on it and adding GPUs
/// changes nothing. When the bitstream allows — an un-spliced H.264 / H.265
/// input whose keyframes fall on chunk boundaries — the source can be cut
/// into ranges that are each decodable from their first sample, one pump per
/// card, so the cards decode different stretches at the same time. The
/// numbering stays continuous across the join and the output is
/// byte-identical to a whole-source decode. See
/// [`plan_decode_ranges`](crate::decode_pump::plan_decode_ranges).
/// Anything that cannot be split safely decodes whole under every variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodePolicy {
    /// Split the decode into one range per decode-capable card of the encode
    /// policy's set, where the source allows; whole otherwise. The default.
    #[default]
    Auto,
    /// One decoder for the whole source, on the first decode-capable card of
    /// the encode policy's set. What every job did before ranges existed; the
    /// control arm of any comparison, and the choice for a host whose decode
    /// engines are already saturated.
    Whole,
    /// One decoder, pinned to this physical GPU index (e.g. decode on an iGPU
    /// while the dGPUs encode). Never split: a split on one card is no split.
    SpecificGpu(u32),
    /// Benchmark every decode-capable GPU on a short prefix of the input
    /// before the job and pin one decoder to the fastest. The engine resolves
    /// this to `SpecificGpu` once the winner is known; a no-op on single-GPU
    /// hosts.
    FastestGpu,
    /// Split into up to this many ranges, round-robin over the decode-capable
    /// cards. More ranges than cards is legal — several pumps then share a
    /// card — and is how the split is exercised on a one-card host.
    Ranges(usize),
}

impl DecodePolicy {
    /// The concrete pinned GPU index, if any. Everything but `SpecificGpu`
    /// returns `None`, so the engine picks from the decode-capable cards.
    pub fn gpu_index(self) -> Option<u32> {
        match self {
            DecodePolicy::SpecificGpu(i) => Some(i),
            _ => None,
        }
    }

    /// Whether the engine should benchmark decoders and resolve a fastest GPU.
    pub fn is_fastest(self) -> bool {
        matches!(self, DecodePolicy::FastestGpu)
    }

    /// How many decode ranges to ask for against a pool of `capacity` cards.
    /// One for anything that pins or benchmarks a single card.
    pub fn ranges_for(self, capacity: usize) -> usize {
        match self {
            DecodePolicy::Auto => capacity.max(1),
            DecodePolicy::Whole | DecodePolicy::SpecificGpu(_) | DecodePolicy::FastestGpu => 1,
            DecodePolicy::Ranges(n) => n.max(1),
        }
    }
}

impl std::str::FromStr for DecodePolicy {
    type Err = String;

    /// Parse the `--decode` value space: `auto`, `whole`, `fastest`, `gpu:N`,
    /// `ranges:N` — and a bare `N`, which is a GPU index (the `--decode-gpu`
    /// spelling this flag grew out of).
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim().to_ascii_lowercase();
        let bad = || {
            format!(
                "decode must be 'auto', 'whole', 'fastest', 'gpu:N', 'ranges:N' or a GPU index; got '{s}'"
            )
        };
        if let Some(n) = s.strip_prefix("gpu:").or_else(|| s.strip_prefix("gpu=")) {
            return n.trim().parse::<u32>().map(DecodePolicy::SpecificGpu).map_err(|_| bad());
        }
        if let Some(n) = s
            .strip_prefix("ranges:")
            .or_else(|| s.strip_prefix("ranges="))
            .or_else(|| s.strip_prefix("split:"))
            .or_else(|| s.strip_prefix("split="))
        {
            return n.trim().parse::<usize>().map(DecodePolicy::Ranges).map_err(|_| bad());
        }
        match s.as_str() {
            "" | "auto" | "split" => Ok(DecodePolicy::Auto),
            "whole" | "none" | "single" => Ok(DecodePolicy::Whole),
            "fastest" => Ok(DecodePolicy::FastestGpu),
            other => other.parse::<u32>().map(DecodePolicy::SpecificGpu).map_err(|_| bad()),
        }
    }
}

/// The encode plan — which cards encode, and how the work is laid across
/// them. One enum, so the halves cannot contradict each other: "one encoder"
/// and "every card" are not both sayable, and there is no second knob that
/// silently turns a multi-GPU job serial.
///
/// Applies to both the single-file and HLS paths. Every spreading variant
/// runs the ladder engine (decode once — split across the cards where the
/// source allows — chunk each rung, encode across the cards, stitch or write
/// segments); `SingleGpu` takes the serial encode path with no chunk overhead,
/// which for single-file output means one encoder per rung and no seams at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EncodePolicy {
    /// Every capable card, **ladder-scheduled**: one worker per card, each
    /// serving every rung and taking the next chunk of whichever rung is
    /// furthest behind. A card idles only when the whole job is out of work,
    /// and a ladder deeper than the GPU count still costs one decode. Falls
    /// back to single-GPU serial encode when only one GPU is present or the
    /// frame count is unknown. The default; measured faster than the pinned
    /// shape.
    #[default]
    AllGpus,
    /// Every capable card, each worker **pinned to its own rungs** (rung `i`
    /// to worker `i mod workers`) — "one rung, one GPU" when the ladder fits
    /// the pool. Predictable placement, and a rung's chunks all come off one
    /// card, at the cost of cards idling when their rungs are blocked. For
    /// benchmarking the two shapes against each other, and for hosts where
    /// placement matters more than throughput.
    PerRung,
    /// A **single** card, one encoder per rung, serial. `None` picks the first
    /// available GPU; `Some(i)` pins to GPU index `i`. Single-file output is
    /// seam-free by construction (there are no chunks); HLS runs one worker.
    SingleGpu(Option<u32>),
    /// Every GPU of one **vendor family** (and only that family),
    /// ladder-scheduled — e.g. `Family(GpuFamily::Nvidia)` on a host with an
    /// NVIDIA discrete + an integrated AMD/Intel GPU uses just the NVIDIA
    /// cards.
    Family(GpuFamily),
}

impl EncodePolicy {
    /// Whether this policy spreads work across more than one card (and so
    /// runs the ladder engine rather than the serial path).
    pub fn spreads(self) -> bool {
        !matches!(self, EncodePolicy::SingleGpu(_))
    }

    /// Whether workers are pinned to their own rungs rather than serving the
    /// whole ladder.
    pub fn pins_rungs(self) -> bool {
        matches!(self, EncodePolicy::PerRung)
    }
}

impl std::str::FromStr for EncodePolicy {
    type Err = String;

    /// Parse the `--encode` value space: `all` (or `ladder`), `per-rung`,
    /// `single`, `gpu:N`, `family:nvidia|amd|intel`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim().to_ascii_lowercase();
        let bad = || {
            format!(
                "encode must be 'all', 'per-rung', 'single', 'gpu:N' or 'family:nvidia|amd|intel'; got '{s}'"
            )
        };
        if let Some(n) = s.strip_prefix("gpu:").or_else(|| s.strip_prefix("gpu=")) {
            return n.trim().parse::<u32>().map(|i| EncodePolicy::SingleGpu(Some(i))).map_err(|_| bad());
        }
        if let Some(f) = s.strip_prefix("family:").or_else(|| s.strip_prefix("family=")) {
            return match f.trim() {
                "nvidia" => Ok(EncodePolicy::Family(GpuFamily::Nvidia)),
                "amd" => Ok(EncodePolicy::Family(GpuFamily::Amd)),
                "intel" => Ok(EncodePolicy::Family(GpuFamily::Intel)),
                _ => Err(bad()),
            };
        }
        match s.as_str() {
            "" | "all" | "auto" | "ladder" => Ok(EncodePolicy::AllGpus),
            "per-rung" | "per_rung" | "perrung" | "pinned" => Ok(EncodePolicy::PerRung),
            "single" | "serial" => Ok(EncodePolicy::SingleGpu(None)),
            _ => Err(bad()),
        }
    }
}

/// A GPU vendor family, for constraining encode to one vendor's devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFamily {
    Nvidia,
    Amd,
    Intel,
}

/// How the multi-GPU **single-file** path keeps quality consistent across the
/// chunk seams it stitches into one continuous video.
///
/// Only relevant when more than one GPU encodes a single file (a spreading
/// [`EncodePolicy`] on a multi-GPU host); single-GPU hosts, `SingleGpu`, and
/// HLS (whose segments are independent by design) are unaffected. AMD (AMF) and
/// Intel (QSV) chunks are already constant-QP, so their seams are quality-flat
/// — this chiefly governs **NVENC**, which otherwise runs VBR per chunk and can
/// leave a mild quality step at the chunk boundaries.
///
/// This is a seam-*quality* choice and nothing else. Wanting no seams at all
/// is not a seam mode, it is an encode plan: [`EncodePolicy::SingleGpu`], one
/// encoder per rung. (There used to be a `Serial` variant here that quietly
/// turned a multi-GPU job serial; that was two knobs for one question.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChunkSeamMode {
    /// Default. Chunk across GPUs for throughput; each chunk uses its encoder's
    /// normal rate control (VBR on NVENC). Fastest; NVENC may show mild quality
    /// steps at the seams on complex content.
    #[default]
    Parallel,
    /// Chunk across GPUs but force **constant-QP** so the seams are
    /// quality-flat, keeping the multi-GPU speedup. The QP is derived from the
    /// `QualityTarget` (via the per-encoder tuning CQ), so quality still tracks
    /// the target — the hand-rolled NVENC sets a real const-QP rather than a
    /// preset default. AMD/QSV are unchanged (already constant-QP).
    ParallelConstQp,
}

/// Output **color** policy — the gamut (which colors are representable) and the
/// transfer curve (SDR vs HDR), plus whether to tonemap an HDR source down. This
/// is the *color* half of the decision; bit depth is the separate [`BitDepth`]
/// half (though the HDR variants here imply 10-bit on their own).
///
/// The decode pump never tonemaps on its own — this policy decides.
///
/// Glossary (the jargon these variants use):
/// - **BT.709** — the standard HD / SDR color gamut. What the vast majority of
///   video uses; "SDR" output means BT.709.
/// - **BT.2020** — the *wide* gamut used by HDR: more saturated, deeper colors.
/// - **PQ** (SMPTE ST 2084) — the HDR10 transfer curve (absolute brightness, up
///   to 10,000 nits).
/// - **HLG** (ARIB STD-B67) — the broadcast-friendly HDR transfer curve
///   (relative brightness; degrades gracefully on SDR screens).
/// - **tonemap** — squeeze an HDR signal's brightness/gamut down into SDR so it
///   looks right on ordinary (BT.709, 8-bit) screens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorPolicy {
    /// **SDR out.** Tonemap HDR (PQ / HLG) sources down to 8-bit **BT.709** SDR;
    /// SDR sources pass through unchanged. The default — maximally web-compatible.
    /// (Convenience builder: [`super::OutputSpec::web_sdr`].)
    #[default]
    TonemapToSdr,
    /// **Verbatim.** Keep the source's gamut, transfer, and bit depth as-is — no
    /// tonemap, no re-signaling. An HDR source stays HDR (needs a 10-bit
    /// encoder); an SDR source stays SDR. (Builder: [`super::OutputSpec::passthrough`].)
    Passthrough,
    /// **HDR10 out.** Force **BT.2020** gamut + **PQ** transfer, 10-bit. Sets
    /// 10-bit on its own, so you do *not* also need [`BitDepth::TenBit`].
    /// (Builder: [`super::OutputSpec::hdr10`].)
    Hdr10,
    /// **HLG out.** Force **BT.2020** gamut + **HLG** transfer, 10-bit. Implies
    /// 10-bit. (Builder: [`super::OutputSpec::hlg`].)
    Hlg,
}

impl ColorPolicy {
    /// Whether the decode pump tonemaps HDR→SDR under this policy.
    pub fn tonemaps(self) -> bool {
        matches!(self, ColorPolicy::TonemapToSdr)
    }

    /// Whether this policy signals HDR (PQ/HLG) in the output bitstream.
    pub fn is_hdr(self) -> bool {
        matches!(self, ColorPolicy::Hdr10 | ColorPolicy::Hlg)
    }
}

/// Output **bit depth** — bits per sample. The on-disk pixel format is *derived*
/// from this (every output codec — AV1, H.264, H.265 — is encoded 4:2:0, the
/// web-safe chroma subsampling):
/// 8-bit → **`yuv420p`**, 10-bit → **`yuv420p10le`** (`le` = little-endian 16-bit
/// words holding 10 valid bits). Bit depth is one axis; gamut + SDR/HDR transfer
/// is the orthogonal [`ColorPolicy`] axis.
///
/// You rarely set this by hand: `Auto` derives it from the color policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BitDepth {
    /// Derive depth from the [`ColorPolicy`]: 8-bit for an SDR tonemap, 10-bit
    /// for HDR (`Hdr10` / `Hlg`), the source's own depth for `Passthrough`. The
    /// default — the right choice almost always.
    #[default]
    Auto,
    /// Force **8-bit** 4:2:0 (`yuv420p`) — universal web compatibility.
    EightBit,
    /// Force **10-bit** 4:2:0 (`yuv420p10le`) — higher precision (banding-free
    /// gradients), and required by the HDR policies. Needs a 10-bit encoder
    /// **for the output codec**, which [`OutputSpec::validate`](super::OutputSpec::validate)
    /// checks: AV1 on NVENC (`nvidia`), AMF (`amd`) or QSV (`qsv`) — the
    /// software AV1 tier (`rav1e-fallback`) is 8-bit; H.265 on those three or
    /// the software tier (`h26x-fallback`, Main 10); H.264 on the software tier
    /// only (`h26x-fallback`, High 10 — no hardware backend has a 10-bit H.264
    /// encoder). See [`CodecOutputCaps`](super::CodecOutputCaps).
    TenBit,
}
