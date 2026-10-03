//! Output specification — *how* a job should be transcoded.
//!
//! A job is described by an [`OutputSpec`]: the [`OutputMode`] (single file
//! vs segmented HLS), the [`VideoCodec`] + [`AudioCodecPolicy`], the [`Container`]
//! and [`Muxer`], and the user-defined ladder of [`Rung`]s (each with its own
//! [`Quality`]). Nothing about the output is hard-coded — the caller decides
//! the shape, the codec, the quality, and the renditions.
//!
//! ```
//! use rivet::spec::{OutputSpec, Rung, Quality};
//!
//! // A 3-rung HLS ladder with 4-second segments.
//! let spec = OutputSpec::hls(
//!     vec![Rung::new(1920, 1080), Rung::new(1280, 720), Rung::new(640, 360)],
//!     4.0,
//! );
//! assert!(spec.validate().is_ok());
//! ```

use anyhow::{Context, Result, bail};
use codec::frame::{ColorMetadata, PixelFormat, TransferFn};

pub use codec::encode::tuning::{QualityTarget as PerceptualTarget, SpeedTier as Speed};

/// The low-level codec identity used by the encoder + muxer, re-exported from
/// [`codec::frame::VideoCodec`]. Most callers pick the codec via
/// [`VideoCodecPolicy`] (the spec-level dimension) and never touch this directly;
/// `VideoCodecPolicy::codec` resolves to it.
pub use codec::frame::VideoCodec;

/// FLAC compression effort ([`OutputSpec::flac_level`]).
pub use codec::audio::encode::flac::FlacLevel;

mod audio;
mod caps;
mod policy;
mod rung;
#[cfg(test)]
mod tests;

pub use caps::{
    CodecOutputCaps, ENCODE_BACKENDS, OUTPUT_CODECS, encode_backend_feature, encode_backend_name,
    encoder_backend_from_name,
    encode_backend_serves, every_codec_output_caps, output_caps_label, output_codec_label,
};
pub use policy::*;
pub use rung::*;

/// Full output specification for a transcode job.
#[derive(Debug, Clone)]
pub struct OutputSpec {
    /// Output shape.
    pub mode: OutputMode,
    /// Output video codec policy (`Av1` default, or `H264` / `H265`).
    pub video_codec: VideoCodecPolicy,
    /// Audio handling.
    pub audio: AudioCodecPolicy,
    /// Target bitrate in **bits per second** for tracks that get transcoded.
    /// `None` lets the encoder pick: for Opus from the channel layout — 64
    /// kbps per uncoupled stream + 96 kbps per coupled (stereo) pair, i.e.
    /// 64k mono, 96k stereo, 320k for 5.1, 416k for 7.1; for MP3 128k stereo,
    /// 64k mono (CBR, one of the MPEG-1 Layer III rates); for AAC 64k mono,
    /// 128k stereo, 384k 5.1, 512k 7.1; for HE-AAC 32k mono, 48k stereo, for
    /// HE-AAC v2 32k; for AC-3 96k mono, 192k stereo, 448k 5.1; for E-AC-3
    /// 96k, 192k, 384k; for DTS the full rate (1536k at 48 kHz). Vorbis takes
    /// [`Self::audio_quality`] instead. Ignored for passthrough tracks, which
    /// keep whatever bitrate they were authored at.
    pub audio_bitrate: Option<u32>,
    /// Vorbis quality, -1 (smallest) to 10 (best); `None` is 5. Vorbis output
    /// only (`audio=vorbis`).
    pub audio_quality: Option<f32>,
    /// The output channel layout. See [`AudioChannels`]: `Source` keeps the
    /// source's where the codec can, the others downmix and never upmix.
    pub audio_channels: AudioChannels,
    /// HLS only: beside a surround audio rendition, a stereo downmix of it
    /// in the same audio group, so a player on stereo hardware takes that
    /// instead of downmixing itself (`EXT-X-MEDIA` with `CHANNELS="2"` and
    /// `"6"`). Nothing is added when the audio is stereo or mono already.
    pub audio_stereo_fallback: bool,
    /// Bit depth of FLAC / ALAC output. See [`AudioBitDepth`].
    pub audio_bit_depth: AudioBitDepth,
    /// An HE-AAC source: decoded in full like any AAC, never decoded, or
    /// decoded as its AAC-LC core. See [`HeAacPolicy`].
    pub he_aac: HeAacPolicy,
    /// Source audio codecs that may not be decoded. See [`AudioDecodeDeny`].
    pub audio_decode_deny: AudioDecodeDeny,
    /// Which identifying metadata of the source (location, device, capture
    /// time, descriptive tags), and how much of it, is carried into a single
    /// file or an audio-only output. Empty, the default, carries none; with
    /// the device not kept, a copied AAC or MP3 stream's encoder name is
    /// cleared too. HLS takes none: see [`Self::validate`].
    pub metadata_keep: container::metadata::Keep,
    /// FLAC compression effort; FLAC output only.
    pub flac_level: FlacLevel,
    /// Which of the source's text subtitle tracks to carry. See
    /// [`SubtitlePolicy`]. A single-file MP4 gets a `tx3g` track per language;
    /// an HLS package gets a segmented-WebVTT rendition per language.
    pub subtitles: SubtitlePolicy,
    /// Audio filters applied to decoded PCM **before** the Opus encoder — today
    /// `channelmap`. See [`codec::audio::filter`].
    ///
    /// A filter forces the track to be decoded and re-encoded, so a non-empty
    /// chain is incompatible with a passthrough track; the audio job reports
    /// that rather than silently ignoring the filter.
    pub audio_filters: Vec<codec::audio::filter::AudioFilter>,
    /// Container format.
    pub container: Container,
    /// Muxer.
    pub muxer: Muxer,
    /// The ladder. Order is preserved; the first rung is treated as the
    /// "primary" for single-file callers that only want one output.
    ///
    /// Each rung's size is a box the source is fitted into — see [`fit`](Self::fit)
    /// and [`crate::fit`].
    pub rungs: Vec<Rung>,
    /// How the source meets each rung's box: `Contain` (the default) keeps its
    /// shape inside the box, `Cover` fills the box and crops, `Pad` letterboxes
    /// to exactly the box, `Stretch` distorts to exactly the box (what an
    /// explicit rung did before fitting). A rung's own [`Rung::fit`] wins.
    pub fit: Fit,
    /// Whether a rung's box turns to the source's orientation (`Auto`, the
    /// default: a 1920x1080 box on a portrait source is 1080x1920) or is used
    /// as written (`Fixed`). A rung's own [`Rung::orientation`] wins.
    pub orientation: Orientation,
    /// Whether a rung may come out larger than the source. Off by default: a
    /// source smaller than a box is produced at its own size, and rungs that
    /// collapse onto the same size are merged. A rung's own [`Rung::upscale`]
    /// wins.
    pub upscale: bool,
    /// Cap the output frame rate. A source faster than the cap is decimated:
    /// the decode pump drops frames so each output frame period gets the
    /// source frame showing at its start, keeping the duration (see
    /// [`crate::decode_pump::decimation`]). A source at or below the cap is
    /// untouched. `None` = source fps.
    pub max_frame_rate: Option<f64>,
    /// Pin hardware encode/decode to this GPU index on multi-GPU hosts.
    /// Kept in sync with `encode_policy` (`SingleGpu(idx)` ⇒ `gpu_index = idx`).
    pub gpu_index: Option<u32>,
    /// How to spread encode work across GPUs. See [`EncodePolicy`].
    pub encode_policy: EncodePolicy,
    /// The decode plan — which card(s) decode and whether the decode is split
    /// into ranges across them. See [`DecodePolicy`]: `Auto` (split, one range
    /// per capable card), `Whole`, `SpecificGpu(i)`, `FastestGpu`, `Ranges(n)`.
    pub decode_policy: DecodePolicy,
    /// GOP length in frames for every rung, when set. `None` = the engine's
    /// default of two seconds at the output frame rate.
    ///
    /// What it governs depends on the output. **Single file:** the encoder's
    /// keyframe cadence, and — on the multi-GPU path — the chunk grid, since a
    /// chunk is a whole number of GOPs. **HLS:** the segment grid is set by
    /// `segment_seconds` and every segment opens on an IDR regardless; a GOP
    /// *shorter* than the segment adds keyframes inside it (for seeking), a
    /// GOP longer than the segment is silently the segment, because every
    /// segment is encoded from a fresh IDR anyway. A rung's own
    /// [`Quality::keyframe_interval`] wins over this for that rung.
    pub gop: Option<u32>,
    /// GOP length in seconds of output, when set and [`gop`](Self::gop) is
    /// not: converted to frames at the output frame rate once that is known
    /// ([`Self::with_constant_rates_resolved`]), the same way the two-second
    /// default is ([`gop_frames_for_seconds`]), and then it is `gop`. `None`
    /// with `gop` unset is the default, [`DEFAULT_GOP_SECONDS`].
    pub gop_seconds: Option<f64>,
    /// Per-rung encoder knobs by *position in the ladder* — softer quality
    /// going down, one tile below 4K, more reference frames, and so on. See
    /// [`RungPolicy`](codec::encode::tuning::RungPolicy): the engine resolves
    /// it against each rung and layers the rung's own
    /// [`Quality::overrides`] on top (the rung-specific knob wins; quality
    /// deltas accumulate). Empty by default, so nothing changes unless asked;
    /// [`RungPolicy::recommended`](codec::encode::tuning::RungPolicy::recommended)
    /// is the measured ladder recommendation, and
    /// [`RungPolicy::parse`](codec::encode::tuning::RungPolicy::parse) reads
    /// the text grammar (`qstep=2;short<=2159:tiles=1x1;any:refs=3`).
    pub rung_policy: codec::encode::tuning::RungPolicy,
    /// Output color / tonemap policy. See [`ColorPolicy`].
    pub color: ColorPolicy,
    /// 4:4:4 → 4:2:0 chroma filter for 4:4:4 sources (`box`, the default,
    /// keeps outputs byte-identical to earlier releases; `lanczos` is the
    /// siting-correct separable Lanczos-2). No effect on 4:2:0 / 4:2:2
    /// sources. Settings key `chroma-downsample`.
    pub chroma_downsample: codec::colorspace::ChromaDownsample,
    /// Output bit depth. See [`BitDepth`].
    pub bit_depth: BitDepth,
    /// How the multi-GPU **single-file** path keeps quality consistent across
    /// the chunk seams it stitches. See [`ChunkSeamMode`].
    pub chunk_seam_mode: ChunkSeamMode,
    /// Video filters applied per-frame **before** per-rung scaling (crop, pad,
    /// flip, rotate, grayscale). Empty = none. See [`codec::filter`].
    pub filters: Vec<codec::filter::VideoFilter>,
    /// Splice **trim in-point**, in seconds from the start of the (single)
    /// input. `None` starts at the beginning. Frames before this point are
    /// decoded-and-dropped; the output timeline is re-based to zero. For
    /// multi-clip concatenation use [`run_splice_job`](crate::run_splice_job)
    /// with a per-clip range instead. Trimmed jobs take the serial encode path.
    pub trim_start: Option<f64>,
    /// Splice **trim out-point**, in seconds. `None` keeps the clip to its end.
    /// The kept range is `[trim_start, trim_end)`.
    pub trim_end: Option<f64>,
    /// Code run at fixed points of the job — the input, the probe, decoded
    /// frames, each artifact, and the end. See [`crate::hooks`]. Empty by
    /// default, which costs nothing.
    pub hooks: crate::hooks::Hooks,
}

impl Default for OutputSpec {
    fn default() -> Self {
        Self {
            mode: OutputMode::SingleFile,
            video_codec: VideoCodecPolicy::Av1,
            audio: AudioCodecPolicy::Auto,
            audio_bitrate: None,
            audio_quality: None,
            audio_channels: AudioChannels::Source,
            audio_stereo_fallback: false,
            audio_bit_depth: AudioBitDepth::Source,
            he_aac: HeAacPolicy::Auto,
            audio_decode_deny: AudioDecodeDeny::NONE,
            metadata_keep: container::metadata::Keep::NONE,
            flac_level: FlacLevel::Default,
            audio_filters: Vec::new(),
            subtitles: SubtitlePolicy::default(),
            container: Container::Mp4,
            muxer: Muxer::Mp4File,
            rungs: Vec::new(),
            fit: Fit::default(),
            orientation: Orientation::default(),
            upscale: false,
            max_frame_rate: None,
            gpu_index: None,
            encode_policy: EncodePolicy::default(),
            decode_policy: DecodePolicy::Auto,
            gop: None,
            gop_seconds: None,
            rung_policy: codec::encode::tuning::RungPolicy::new(),
            color: ColorPolicy::default(),

            chroma_downsample: codec::colorspace::ChromaDownsample::Box,
            bit_depth: BitDepth::default(),
            chunk_seam_mode: ChunkSeamMode::default(),
            filters: Vec::new(),
            trim_start: None,
            trim_end: None,
            hooks: crate::hooks::Hooks::default(),
        }
    }
}

impl OutputSpec {
    /// This spec with `hooks` run on every job it drives.
    pub fn with_hooks(mut self, hooks: crate::hooks::Hooks) -> Self {
        self.hooks = hooks;
        self
    }

    /// One self-contained MP4 per rung (AV1 + Opus/passthrough audio).
    pub fn single_file(rungs: Vec<Rung>) -> Self {
        Self {
            mode: OutputMode::SingleFile,
            container: Container::Mp4,
            muxer: Muxer::Mp4File,
            rungs,
            ..Default::default()
        }
    }

    /// A segmented CMAF + HLS package with the given rungs and segment length.
    pub fn hls(rungs: Vec<Rung>, segment_seconds: f32) -> Self {
        Self {
            mode: OutputMode::Hls { segment_seconds },
            container: Container::Cmaf,
            muxer: Muxer::CmafHls,
            rungs,
            ..Default::default()
        }
    }

    /// The audio alone, as one `.mp3` file: no rungs, no video decoded.
    /// `audio` stays `Auto`, which here means MP3.
    pub fn audio_only() -> Self {
        Self { mode: OutputMode::AudioOnly, container: Container::Mp3, muxer: Muxer::Mp3File, ..Default::default() }
    }

    /// Set the audio policy.
    pub fn with_audio(mut self, audio: AudioCodecPolicy) -> Self {
        self.audio = audio;
        self
    }

    /// Set the target bitrate in bits per second for transcoded audio.
    /// Omit to let the encoder derive it from the channel layout.
    pub fn with_audio_bitrate(mut self, bits_per_second: u32) -> Self {
        self.audio_bitrate = Some(bits_per_second);
        self
    }

    /// Set the output channel layout. See [`AudioChannels`].
    pub fn with_audio_channels(mut self, channels: AudioChannels) -> Self {
        self.audio_channels = channels;
        self
    }

    /// HLS: add a stereo downmix rendition beside a surround one.
    pub fn with_audio_stereo_fallback(mut self, on: bool) -> Self {
        self.audio_stereo_fallback = on;
        self
    }

    /// What an HE-AAC source becomes. See [`HeAacPolicy`].
    pub fn with_he_aac(mut self, policy: HeAacPolicy) -> Self {
        self.he_aac = policy;
        self
    }

    /// Source audio codecs that may not be decoded. See [`AudioDecodeDeny`].
    pub fn with_audio_decode_deny(mut self, deny: AudioDecodeDeny) -> Self {
        self.audio_decode_deny = deny;
        self
    }

    /// The audio codec this spec encodes to when the track is transcoded:
    /// FLAC / ALAC when asked for, the forced codec of a `Force*` policy, MP3
    /// for an `.mp3` audio-only output, Opus otherwise. A lossless depth left
    /// to the source is not known until the track is read; 24 stands in for
    /// it.
    pub fn audio_encode_codec(&self) -> codec::audio::AudioCodec {
        use codec::audio::AudioCodec;
        let bits_per_sample = self.audio_bit_depth.bits().unwrap_or(24);
        match (self.audio, &self.mode) {
            (AudioCodecPolicy::Flac, _) => AudioCodec::Flac { bits_per_sample, level: self.flac_level },
            (AudioCodecPolicy::Alac, _) => AudioCodec::Alac { bits_per_sample },
            (p, _) if p.forced_lossy().is_some() => p.forced_lossy().expect("checked"),
            (_, OutputMode::AudioOnly) if self.container == Container::Mp3 => AudioCodec::Mp3,
            _ => AudioCodec::Opus,
        }
    }

    /// Set the Vorbis quality (-1 to 10).
    pub fn with_audio_quality(mut self, quality: f32) -> Self {
        self.audio_quality = Some(quality);
        self
    }

    /// Set the subtitle policy — every text track, none, or a language list.
    /// See [`SubtitlePolicy`].
    pub fn with_subtitles(mut self, policy: SubtitlePolicy) -> Self {
        self.subtitles = policy;
        self
    }

    /// Set the audio filter chain (`channelmap`) applied before the encoder.
    /// See [`codec::audio::filter`].
    pub fn with_audio_filters(
        mut self,
        filters: Vec<codec::audio::filter::AudioFilter>,
    ) -> Self {
        self.audio_filters = filters;
        self
    }

    /// Cap output frame rate.
    pub fn with_max_frame_rate(mut self, fps: f64) -> Self {
        self.max_frame_rate = Some(fps);
        self
    }

    /// Pin to a GPU index. Implies `EncodePolicy::SingleGpu(Some(idx))`.
    pub fn with_gpu_index(mut self, idx: u32) -> Self {
        self.gpu_index = Some(idx);
        self.encode_policy = EncodePolicy::SingleGpu(Some(idx));
        self
    }

    /// Select the GPU encode policy: a single (optionally pinned) GPU, or all
    /// GPUs (the multi-GPU engine).
    ///
    /// ```no_run
    /// # use rivet::spec::{OutputSpec, EncodePolicy, Rung};
    /// # let rungs: Vec<Rung> = vec![];
    /// // chunk-encode across every GPU and stitch:
    /// let _ = OutputSpec::single_file(rungs.clone()).encode_policy(EncodePolicy::AllGpus);
    /// // serial encode, pinned to GPU 1:
    /// let _ = OutputSpec::single_file(rungs).encode_policy(EncodePolicy::SingleGpu(Some(1)));
    /// ```
    pub fn encode_policy(mut self, policy: EncodePolicy) -> Self {
        self.encode_policy = policy;
        if let EncodePolicy::SingleGpu(idx) = policy {
            self.gpu_index = idx;
        }
        self
    }

    /// Set the [`DecodePolicy`] — `Auto` (split across the capable cards),
    /// `Whole` (one decoder), `SpecificGpu(i)` (decode on an iGPU while dGPUs
    /// encode, say), `FastestGpu` (benchmark decoders up front and pick the
    /// quickest) or `Ranges(n)`.
    pub fn decode_policy(mut self, policy: DecodePolicy) -> Self {
        self.decode_policy = policy;
        self
    }

    /// Set the per-rung [`RungPolicy`](codec::encode::tuning::RungPolicy). See
    /// [`OutputSpec::rung_policy`].
    pub fn with_rung_policy(mut self, policy: codec::encode::tuning::RungPolicy) -> Self {
        self.rung_policy = policy;
        self
    }

    /// Set the GOP length in frames for every rung. See [`OutputSpec::gop`].
    pub fn with_gop(mut self, frames: Option<u32>) -> Self {
        self.gop = frames;
        self
    }

    /// Set the GOP length in seconds of output for every rung. See
    /// [`OutputSpec::gop_seconds`]; a `gop` in frames wins over it.
    pub fn with_gop_seconds(mut self, seconds: Option<f64>) -> Self {
        self.gop_seconds = seconds;
        self
    }

    /// The GOP the multi-GPU single-file path chunks on: `gop`, else
    /// `gop_seconds` at `frame_rate`, else two seconds at `frame_rate`.
    pub fn gop_frames(&self, frame_rate: f64) -> u32 {
        self.gop
            .unwrap_or_else(|| gop_frames_for_seconds(self.gop_seconds.unwrap_or(DEFAULT_GOP_SECONDS), frame_rate))
            .max(1)
    }

    /// The spec with a GOP given in seconds made frames at `frame_rate`:
    /// `gop` set from `gop_seconds` (when `gop` is not already set) and
    /// `gop_seconds` cleared. A spec with no `gop_seconds` comes back
    /// unchanged — the default two seconds stays the default.
    pub fn with_gop_seconds_resolved(&self, frame_rate: f64) -> OutputSpec {
        let mut resolved = self.clone();
        if let Some(seconds) = resolved.gop_seconds.take() {
            if resolved.gop.is_none() {
                resolved.gop = Some(gop_frames_for_seconds(seconds, frame_rate));
            }
        }
        resolved
    }

    /// The spec with `rung_policy` folded into every rung's
    /// [`Quality::overrides`] and the policy itself emptied — what the engine
    /// runs, so no worker has to know the ladder's shape. `rung_policy` is
    /// resolved against each rung's position (index 0 is the largest, as
    /// [`Rung`]s are ordered) and the rung's own overrides are layered on
    /// top: a rung-specific knob wins over the ladder-wide one, and quality
    /// deltas accumulate. A spec with an empty policy comes back unchanged.
    pub fn with_rung_policy_resolved(&self) -> OutputSpec {
        use codec::encode::tuning::RungContext;
        let mut resolved = self.clone();
        let policy_is_empty = self.rung_policy.rules.is_empty() && self.rung_policy.global.is_empty();
        if policy_is_empty && self.gop.is_none() {
            return resolved;
        }
        let rung_count = self.rungs.len();
        for (index, rung) in resolved.rungs.iter_mut().enumerate() {
            if !policy_is_empty {
                let ctx = RungContext { width: rung.width, height: rung.height, index, rung_count };
                let mut from_policy = self.rung_policy.resolve(&ctx);
                // `WxH@standard`: no spec-wide rate reaches this rung (its
                // own, set on the rung, still wins through the merge).
                if rung.standard_rate {
                    from_policy.bitrate = None;
                }
                rung.quality.overrides = from_policy.merge(rung.quality.overrides);
            }
            // The spec-wide GOP reaches every rung two ways, because the two
            // paths read different fields: the serial path applies
            // `Quality::keyframe_interval`; the multi-GPU workers take the
            // chunk grid from the job and honour `overrides.keyframe_interval`
            // for the encoder's own cadence within it. A rung's own values win.
            if let Some(gop) = self.gop {
                if rung.quality.keyframe_interval.is_none() {
                    rung.quality.keyframe_interval = Some(gop);
                }
                if rung.quality.overrides.keyframe_interval.is_none() {
                    rung.quality.overrides.keyframe_interval = Some(gop);
                }
            }
        }
        resolved.rung_policy = codec::encode::tuning::RungPolicy::new();
        resolved
    }

    /// Set the output color / tonemap policy (SDR tonemap vs HDR passthrough).
    pub fn with_color(mut self, color: ColorPolicy) -> Self {
        self.color = color;
        self
    }

    /// Choose the 4:4:4 → 4:2:0 chroma filter (see
    /// [`codec::colorspace::ChromaDownsample`]).
    pub fn with_chroma_downsample(mut self, filter: codec::colorspace::ChromaDownsample) -> Self {
        self.chroma_downsample = filter;
        self
    }

    /// Set the output **bit depth** (`Auto` / `EightBit` / `TenBit`). Sets bits
    /// per sample only — the gamut/SDR-HDR choice is [`Self::with_color`]. For
    /// HDR you usually don't need this (the HDR [`ColorPolicy`] implies 10-bit).
    pub fn with_bit_depth(mut self, depth: BitDepth) -> Self {
        self.bit_depth = depth;
        self
    }

    // ── Color presets ──────────────────────────────────────────────
    // One-call intent shortcuts that bundle the color policy (and the bit depth
    // it implies). Equivalent to the `with_color` / `with_bit_depth` pairs in the
    // comments, but say what you mean. The low-level builders stay available.

    /// **Web-safe SDR** (the default): BT.709 8-bit, tonemapping any HDR source
    /// down. Plays everywhere. Same as `.with_color(TonemapToSdr)
    /// .with_bit_depth(EightBit)`.
    pub fn web_sdr(self) -> Self {
        self.with_color(ColorPolicy::TonemapToSdr)
            .with_bit_depth(BitDepth::EightBit)
    }

    /// **HDR10**: BT.2020 wide gamut + PQ transfer, 10-bit, no tonemap. Needs a
    /// 10-bit HDR encoder for the output codec, which [`Self::validate`]
    /// checks: AV1 on `nvidia` / `amd` / `qsv` (the software AV1 tier is
    /// 8-bit); H.265 on those or `h26x-fallback` (Main 10); H.264 on
    /// `h26x-fallback` only (High 10). Same as
    /// `.with_color(Hdr10)` — the policy already implies 10-bit.
    pub fn hdr10(self) -> Self {
        self.with_color(ColorPolicy::Hdr10)
    }

    /// **HLG**: BT.2020 wide gamut + HLG transfer, 10-bit, no tonemap. Same as
    /// `.with_color(Hlg)`.
    pub fn hlg(self) -> Self {
        self.with_color(ColorPolicy::Hlg)
    }

    /// **Passthrough**: keep the source's gamut, transfer, and bit depth
    /// verbatim. Same as `.with_color(Passthrough)`.
    pub fn passthrough(self) -> Self {
        self.with_color(ColorPolicy::Passthrough)
    }

    /// Set how the multi-GPU single-file path handles chunk seams
    /// (`Parallel` fastest / `ParallelConstQp` seam-flat; seam-free is an
    /// encode plan — [`EncodePolicy::SingleGpu`] — not a seam mode).
    pub fn chunk_seam_mode(mut self, mode: ChunkSeamMode) -> Self {
        self.chunk_seam_mode = mode;
        self
    }

    /// How the source meets every rung's box. See [`OutputSpec::fit`].
    pub fn with_fit(mut self, fit: Fit) -> Self {
        self.fit = fit;
        self
    }

    /// Whether rung boxes turn to the source's orientation. See
    /// [`OutputSpec::orientation`].
    pub fn with_orientation(mut self, orientation: Orientation) -> Self {
        self.orientation = orientation;
        self
    }

    /// Whether rungs may be larger than the source. See [`OutputSpec::upscale`].
    pub fn with_upscale(mut self, upscale: bool) -> Self {
        self.upscale = upscale;
        self
    }

    /// This spec with every rung fitted to a `source` (the picture as it
    /// reaches the scalers — upright, after the filters): each rung's size is
    /// its output size, its label follows, and rungs that came out the same
    /// as an earlier one are gone. Also returns what became of each requested
    /// rung, in request order. See [`crate::fit::fit_rungs`].
    pub fn with_rungs_fitted(&self, source: crate::fit::SourceShape) -> (OutputSpec, Vec<crate::fit::FittedRung>) {
        let (rungs, report) =
            crate::fit::fit_rungs(&self.rungs, source, self.fit, self.orientation, self.upscale);
        (OutputSpec { rungs, ..self.clone() }, report)
    }

    /// Set the per-frame video filter chain (crop / pad / flip / rotate /
    /// grayscale), applied before per-rung scaling. See [`codec::filter`].
    pub fn with_filters(mut self, filters: Vec<codec::filter::VideoFilter>) -> Self {
        self.filters = filters;
        self
    }

    /// **Trim** the single input to the time range `[start, end)` in seconds
    /// (either bound `None` = open). The output is re-based to zero. Trimmed
    /// jobs use the serial encode path. For joining multiple clips, see
    /// [`run_splice_job`](crate::run_splice_job).
    pub fn with_trim(mut self, start: Option<f64>, end: Option<f64>) -> Self {
        self.trim_start = start;
        self.trim_end = end;
        self
    }

    /// Set the output video codec ([`VideoCodecPolicy::Av1`] default; `H264`,
    /// `H265`, `Vp9`, `Vp8`, `Mpeg2`, `Mpeg4`, `ProRes(profile)`). AV1, H.264,
    /// H.265 and VP9 work for single-file output and CMAF/HLS; the others
    /// for single-file output. A single-file spec still in its default MP4
    /// moves to the codec's own file — a `.mov` for ProRes, a `.webm` for
    /// VP8 / VP9 ([`VideoCodecPolicy::default_container`]); call
    /// [`Self::with_container`] after this to pick another.
    pub fn with_video_codec(mut self, codec: VideoCodecPolicy) -> Self {
        self.video_codec = codec;
        if self.mode == OutputMode::SingleFile && self.container == Container::Mp4 {
            self = self.with_container(codec.default_container());
        }
        self
    }

    /// Set the file a single-file output is: [`Container::Mp4`],
    /// [`Container::Mov`] (a QuickTime movie) or [`Container::WebM`], with the
    /// muxer that writes it. [`Self::validate`] refuses a codec the file
    /// cannot carry ([`VideoCodecPolicy::fits`]).
    pub fn with_container(mut self, container: Container) -> Self {
        self.container = container;
        if let Some(muxer) = container.single_file_muxer() {
            self.muxer = muxer;
        }
        self
    }

    /// Whether the decode pump tonemaps HDR→SDR for this spec (policy-driven —
    /// the pump never decides on its own).
    pub fn tonemaps(&self) -> bool {
        self.color.tonemaps()
    }

    /// Resolve the encoder's input `(color_metadata, pixel_format)` for a given
    /// source — which is also what the output is *tagged* as (SPS VUI, `colr`),
    /// so it describes the picture after the pump, not the source. The default
    /// (`TonemapToSdr` + `Auto`) reproduces the legacy source-driven fold: HDR
    /// sources collapse to 8-bit SDR; SDR sources keep their own bit depth and
    /// colour, except that an 8-bit BT.601 / BT.2020 matrix is re-derived to
    /// BT.709 on the way and is tagged as such. `Hdr10`/`Hlg` force BT.2020
    /// 10-bit; `Passthrough` keeps the source; `pixel_format` overrides the
    /// bit depth.
    pub fn resolve_output(
        &self,
        source_color: ColorMetadata,
        source_pixel_format: PixelFormat,
    ) -> (ColorMetadata, PixelFormat) {
        let source_is_hdr = matches!(
            source_color.transfer,
            TransferFn::St2084 | TransferFn::AribStdB67
        );
        // The pump normalises every source onto 4:2:0 at 8 or 10 bits
        // (`colorspace::normalize_layout_to_420`), so that is what the
        // encoder is configured for — never the source's own 4:4:4 / 4:2:2
        // / 12-bit format, which no encoder in the tree accepts.
        let source_pixel_format = encoder_input_format(source_pixel_format);
        // The pump normalises every source onto 4:2:0 at 8 or 10 bits
        // (`colorspace::normalize_layout_to_420`), so that is what the
        // encoder is configured for — never the source's own 4:4:4 / 4:2:2
        // / 12-bit format, which no encoder in the tree accepts.
        let source_pixel_format = encoder_input_format(source_pixel_format);
        let (color, mut pix) = match self.color {
            ColorPolicy::TonemapToSdr => {
                if source_is_hdr {
                    (ColorMetadata::default(), PixelFormat::Yuv420p)
                } else if source_pixel_format == PixelFormat::Yuv420p
                    && matches!(source_color.matrix_coefficients, 5 | 6 | 9 | 10)
                {
                    // The pump's 8-bit SDR path re-derives a BT.601 / BT.2020
                    // matrix to BT.709 (`colorspace::convert_to_yuv420p_bt709`,
                    // keyed on the frame's `ColorSpace`, which the demuxer sets
                    // from this same matrix code), so the picture that comes
                    // out is BT.709-matrixed and the stream and container must
                    // say so. Passing the source's tags through here sent an
                    // smpte170m-tagged H.264 source out with BT.709 pixels and
                    // a smpte170m tag. Only the matrix is converted: range,
                    // primaries and transfer stay what the source said.
                    (ColorMetadata { matrix_coefficients: 1, ..source_color }, PixelFormat::Yuv420p)
                } else {
                    (source_color, source_pixel_format)
                }
            }
            ColorPolicy::Passthrough => (source_color, source_pixel_format),
            // The HDR policies fix the gamut and the transfer tag. The
            // static metadata (mastering display, content light level)
            // describes the content, not the tag, so a source that carried
            // it keeps it: it is what the encoders' SEIs and the container's
            // `mdcv` / `clli` are written from, and replacing it with the
            // default here left an HDR10 source's `--color hdr10` transcode
            // with no mastering display at all while `passthrough` kept it.
            ColorPolicy::Hdr10 | ColorPolicy::Hlg => {
                let transfer = if self.color == ColorPolicy::Hdr10 {
                    TransferFn::St2084
                } else {
                    TransferFn::AribStdB67
                };
                // An SDR source mapped into PQ has a known colour volume; it
                // is signalled when the source says nothing
                // ([`SDR_IN_PQ_MASTERING_DISPLAY`]). HLG carries no static
                // metadata: its signal is scene-referred.
                let sdr_in_pq = transfer == TransferFn::St2084 && !source_is_hdr;
                let color = ColorMetadata {
                    mastering_display: source_color
                        .mastering_display
                        .or(sdr_in_pq.then_some(SDR_IN_PQ_MASTERING_DISPLAY)),
                    content_light_level: source_color
                        .content_light_level
                        .or(sdr_in_pq.then_some(SDR_IN_PQ_CONTENT_LIGHT_LEVEL)),
                    ..hdr_metadata(transfer)
                };
                (color, PixelFormat::Yuv420p10le)
            }
        };
        match self.bit_depth {
            BitDepth::Auto => {}
            BitDepth::EightBit => pix = PixelFormat::Yuv420p,
            BitDepth::TenBit => pix = PixelFormat::Yuv420p10le,
        }
        (color, pix)
    }

    /// The HDR transfer this spec has the decode pump map an SDR `source`
    /// into — see [`sdr_into_hdr`]. `None` when the policy is not HDR or the
    /// source already is.
    pub fn sdr_to_hdr(&self, source: &ColorMetadata) -> Option<TransferFn> {
        let (output, _) = self.resolve_output(*source, PixelFormat::Yuv420p);
        sdr_into_hdr(self.tonemaps(), source, &output)
    }

    /// Refuse, by name and before anything is decoded, a source this spec's
    /// colour policy could only re-tag:
    ///
    /// - `hdr10` on an HLG source, `hlg` on a PQ one: nothing here converts
    ///   between HDR transfers, and one labelled as the other plays wrongly;
    /// - `hdr10` / `hlg` on an SDR source the BT.2408 mapping
    ///   ([`codec::colorspace::SdrToHdr`]) cannot take — linear light, or a
    ///   matrix or primaries code it does not know.
    ///
    /// An SDR source it can take is mapped into the HDR signal by the pump,
    /// not refused; every other policy takes every source.
    pub fn check_source_colour(&self, source: &ColorMetadata) -> Result<()> {
        let (policy, target) = match self.color {
            ColorPolicy::Hdr10 => ("hdr10", TransferFn::St2084),
            ColorPolicy::Hlg => ("hlg", TransferFn::AribStdB67),
            _ => return Ok(()),
        };
        let name = |t: TransferFn| match t {
            TransferFn::St2084 => "PQ (SMPTE ST 2084)",
            _ => "HLG (ARIB STD-B67)",
        };
        match source.transfer {
            t if t == target => Ok(()),
            t @ (TransferFn::St2084 | TransferFn::AribStdB67) => bail!(
                "--color {policy} on a {} source: rivet does not convert between HDR transfers, and {} pixels tagged {} play wrongly. \
                 Use --color passthrough to keep the source's HDR, or --color sdr to tonemap it",
                name(t),
                name(t),
                name(target)
            ),
            _ => codec::colorspace::SdrToHdr::new(source, target)
                .map(|_| ())
                .with_context(|| format!("--color {policy} on an SDR source")),
        }
    }

    /// Reject incoherent specifications — and an output this build cannot
    /// encode for the spec's codec: 10-bit or HDR output needs a backend whose
    /// encoder for that codec is 10-bit / HDR, compiled into this build or
    /// pinned by name through `TRANSCODE_ENCODER_BACKEND` for a single-file
    /// job (see [`CodecOutputCaps`] and `check_encoder_caps`). The refusal
    /// names what the build has for the codec, the pin if one counts, and
    /// which feature would serve the request.
    pub fn validate(&self) -> Result<()> {
        self.validate_with_pin(caps::pinned_encoder_backend())
    }

    /// [`Self::validate`] with the backend pinned by name passed in, rather
    /// than read from the environment, so the rule is testable without
    /// touching process state.
    pub(crate) fn validate_with_pin(&self, pinned: Option<codec::encode::EncoderBackend>) -> Result<()> {
        self.check_audio()?;
        if let Some(seconds) = self.gop_seconds {
            if !seconds.is_finite() || seconds <= 0.0 {
                bail!("gop in seconds must be a positive number of seconds (got {seconds})");
            }
        }
        if matches!(self.container, Container::WebM | Container::Ogg) && !self.metadata_keep.is_empty() {
            bail!(
                "metadata-keep is not available for {} output: rivet writes source metadata into MP4, QuickTime, FLAC and MP3 files",
                if self.container == Container::Ogg { "Ogg" } else { "WebM" }
            );
        }
        if matches!(self.mode, OutputMode::Hls { .. }) && !self.metadata_keep.is_empty() {
            bail!(
                "metadata-keep is not available for HLS output: a player reads no file-level metadata from its segments, so none is written there"
            );
        }
        if self.mode == OutputMode::AudioOnly {
            let muxer = match self.container {
                Container::Mp3 => Muxer::Mp3File,
                Container::Flac => Muxer::FlacFile,
                Container::M4a => Muxer::M4aFile,
                Container::Ogg => Muxer::OggFile,
                other => bail!("AudioOnly mode writes an .mp3, a .flac, an .m4a or an .ogg, not {other:?}"),
            };
            if self.muxer != muxer {
                bail!("AudioOnly mode in Container::{:?} requires Muxer::{muxer:?}", self.container);
            }
            // No video is decoded or encoded, so nothing else applies.
            return Ok(());
        }
        if self.rungs.is_empty() {
            bail!("OutputSpec has no rungs — at least one rendition is required");
        }
        for r in &self.rungs {
            if r.width == 0 || r.height == 0 {
                bail!("rung '{}' has a zero dimension ({}x{})", r.label, r.width, r.height);
            }
            if r.width % 2 != 0 || r.height % 2 != 0 {
                bail!(
                    "rung '{}' has an odd dimension ({}x{}); 4:2:0 requires even dims",
                    r.label,
                    r.width,
                    r.height
                );
            }
        }
        // Container/muxer/mode coherence, and the codec against the file:
        // AV1, H.264, H.265 and VP9 are valid for HLS/CMAF (the CMAF muxer
        // builds av01 / avc1 / hvc1 / vp09 init segments); every codec has a
        // single file (`VideoCodecPolicy::fits`).
        let codec = self.video_codec;
        match self.mode {
            OutputMode::SingleFile => {
                let Some(muxer) = self.container.single_file_muxer() else {
                    bail!(
                        "SingleFile mode writes an MP4, a QuickTime movie (.mov) or a WebM file, not Container::{:?}",
                        self.container
                    );
                };
                if self.muxer != muxer {
                    bail!("SingleFile mode in Container::{:?} requires Muxer::{muxer:?}", self.container);
                }
                if !codec.fits(self.container) {
                    bail!(
                        "{} does not go in {}: {}",
                        codec.as_str(),
                        self.container.file_label(),
                        match codec {
                            VideoCodecPolicy::ProRes(_) => {
                                "ProRes is a QuickTime codec — write a .mov (container=mov)".to_string()
                            }
                            VideoCodecPolicy::Vp8 | VideoCodecPolicy::Vp9 => {
                                "VP8 / VP9 go in a WebM (container=webm) or an MP4 (container=mp4)".to_string()
                            }
                            _ => format!(
                                "write an MP4 (container=mp4){}",
                                if codec.fits(Container::Mov) { " or a QuickTime movie (container=mov)" } else { "" }
                            ),
                        }
                    );
                }
            }
            OutputMode::Hls { segment_seconds } => {
                if self.muxer != Muxer::CmafHls || self.container != Container::Cmaf {
                    bail!("Hls mode requires Container::Cmaf + Muxer::CmafHls");
                }
                if segment_seconds.is_nan() || segment_seconds <= 0.0 {
                    bail!("Hls segment_seconds must be > 0 (got {segment_seconds})");
                }
                if !codec.hls_ready() {
                    bail!(
                        "{} has no CMAF binding and does not play from an HLS package: HLS carries AV1, H.264, H.265 or VP9. Write {} as a single file (mode=single)",
                        codec.as_str(),
                        codec.as_str()
                    );
                }
            }
            OutputMode::AudioOnly => unreachable!("returned above"),
        }
        self.check_codec_limits()?;
        // Subtitles aren't validated against the source here: the spec can't
        // see which languages the source carries, so a requested language
        // with no track is reported by the job layer once it knows.

        // Output color / bit-depth coherence + what this build can produce
        // for the job's codec. Per codec, not the codec-agnostic union: H.264
        // is 8-bit SDR on every hardware backend and 10-bit HDR only on the
        // software `h26x` tier; AV1 is 10-bit SDR on the software AV1 tier.
        if self.color.is_hdr() && matches!(self.bit_depth, BitDepth::EightBit) {
            bail!(
                "color {:?} is HDR and requires 10-bit output, but bit_depth is forced to 8-bit",
                self.color
            );
        }
        self.check_rates()?;
        self.check_encoder_caps(self.pin_honoured(pinned))
    }

    /// What only some codecs' encoders can do, refused by name before a frame
    /// is decoded: the frame sizes the bitstreams can code, B frames where the
    /// codec or its encoder has none, and the rates each of rivet's own
    /// encoders codes (VP8 / VP9 a fixed quantiser, ProRes its profile's
    /// frame size, MPEG-2 / MPEG-4 an average rate with no buffer model).
    pub(crate) fn check_codec_limits(&self) -> Result<()> {
        /// Whether this build has an encoder that codes VP9 at a constant
        /// rate: QSV (`qsv` feature).
        fn vp9_constant_rate_compiled() -> bool {
            codec::encode::compiled_encode_backends().into_iter().any(|b| {
                codec::encode::hardware_encodes(b, codec::frame::VideoCodec::Vp9)
                    && codec::encode::backend_codes_constant_rate(b)
            })
        }
        let codec = self.video_codec;
        let name = codec.as_str();
        let (max_w, max_h) = match codec {
            VideoCodecPolicy::Mpeg2 => (4095, 2800),
            VideoCodecPolicy::Mpeg4 => (8191, 8191),
            VideoCodecPolicy::Vp8 => (16383, 16383),
            VideoCodecPolicy::ProRes(_) => (65535, 65535),
            _ => (u32::MAX, u32::MAX),
        };
        for r in &self.rungs {
            if r.width > max_w || r.height > max_h {
                bail!("rung '{}' is {}x{}; {name} codes at most {max_w}x{max_h}", r.label, r.width, r.height);
            }
        }
        let quantiser_only = matches!(codec, VideoCodecPolicy::Vp8 | VideoCodecPolicy::ProRes(_));
        let average_only = matches!(codec, VideoCodecPolicy::Mpeg2 | VideoCodecPolicy::Mpeg4 | VideoCodecPolicy::Vp9);
        // Each rung as it will be encoded: the rung policy's rules and global
        // set merged under the rung's own overrides.
        let resolved = self.with_rung_policy_resolved();
        if quantiser_only || average_only {
            for r in &resolved.rungs {
                let o = &r.quality.overrides;
                let rate = o.bitrate;
                let mode = o.rate_mode;
                let buffer = o.buffer_ms.filter(|ms| *ms > 0);
                if quantiser_only && (rate.is_some() || mode.is_some() || r.standard_rate) {
                    bail!(
                        "rung '{}' asks for a bit rate, and {name} is coded {}: drop the bitrate",
                        r.label,
                        if matches!(codec, VideoCodecPolicy::ProRes(_)) {
                            "to its profile's frame size (pick the profile for the rate)"
                        } else {
                            "to a fixed quantiser (a quality target or a crf)"
                        }
                    );
                }
                // VP9 at a constant rate is QSV's (Intel's VP9 encoder codes
                // CBR); whether the job's encoders are QSV is the pool's to say
                // (`multigpu::check_rate_pool`), which refuses it by name on
                // rivet's own encoder. MPEG-2 / MPEG-4 have no such encoder.
                let constant = mode == Some(codec::encode::tuning::RateMode::Constant);
                let vp9 = matches!(codec, VideoCodecPolicy::Vp9) && vp9_constant_rate_compiled();
                if average_only && constant && !vp9 {
                    bail!("rung '{}' asks for a constant rate (rate=cbr); the {name} encoder codes an average rate", r.label);
                }
                if average_only && buffer.is_some() && !(vp9 && constant) {
                    bail!("rung '{}' declares a coded picture buffer; the {name} encoder has no buffer model", r.label);
                }
            }
        }
        if let VideoCodecPolicy::ProRes(_) = codec
            && let Some(r) = self.rungs.iter().find(|r| r.quality.crf.is_some())
        {
            bail!(
                "rung '{}' gives a crf, and ProRes has none: its quality is the profile's (proxy, lt, 422, hq, 4444, 4444xq)",
                r.label
            );
        }
        let bframes =
            resolved.rungs.iter().find_map(|r| r.quality.overrides.bframes.filter(|b| *b > 0).map(|b| (r, b)));
        if let Some((r, b)) = bframes {
            match codec {
                VideoCodecPolicy::Vp8 | VideoCodecPolicy::Vp9 | VideoCodecPolicy::ProRes(_) => bail!(
                    "rung '{}' asks for {b} B frames; {name} {}",
                    r.label,
                    if matches!(codec, VideoCodecPolicy::ProRes(_)) {
                        "is intra-only: every frame is a key frame"
                    } else {
                        "has no B frames (its encoder predicts from the previous frame)"
                    }
                ),
                VideoCodecPolicy::Mpeg2 if b > 7 => {
                    bail!("rung '{}' asks for {b} B frames; MPEG-2 codes at most 7 between references", r.label)
                }
                VideoCodecPolicy::Mpeg4 if b > 8 => {
                    bail!("rung '{}' asks for {b} B frames; MPEG-4 Part 2 codes at most 8 between references", r.label)
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// The audio half of [`Self::validate`]: the audio knobs against each
    /// other, the output shape and this build.
    pub(crate) fn check_audio(&self) -> Result<()> {
        use codec::audio::AudioCodec;
        let hls = matches!(self.mode, OutputMode::Hls { .. });
        let audio_only = self.mode == OutputMode::AudioOnly;
        // Every audio knob only reaches the encoder, so pairing one with
        // "drop the audio" is a contradiction worth naming.
        if self.audio == AudioCodecPolicy::Drop {
            if !self.audio_filters.is_empty() {
                bail!(
                    "audio filters were given ({}) but the audio policy is `drop` — \
                     nothing would be filtered",
                    codec::audio::filter::chain_to_string(&self.audio_filters)
                );
            }
            if self.audio_bitrate.is_some() {
                bail!("an audio bitrate was given but the audio policy is `drop`");
            }
            if self.audio_channels != AudioChannels::Source {
                bail!("audio-channels={} was given but the audio policy is `drop`", self.audio_channels.as_str());
            }
            if audio_only {
                bail!("audio-only output with audio=drop has nothing to write");
            }
        }
        self.check_lossless_audio()?;
        // The codec asked for, against the file it goes in.
        let policy = self.audio;
        if !policy.carried_by(self.container, hls) {
            let name = policy.as_str();
            bail!(
                "{}",
                match (policy, hls, self.container) {
                    (AudioCodecPolicy::ForceMp3, true, _) => {
                        // RFC 8216 §3 carries MP3 only in MPEG-2 TS segments or as
                        // packed audio; rivet's HLS is CMAF (fMP4), for which ISO/IEC
                        // 23000-19 defines no MP3 media profile and Apple's authoring
                        // spec lists no MP3. A rendition built anyway is one players
                        // are free to skip.
                        "audio=mp3 is not available for HLS: rivet writes CMAF (fMP4) segments, and neither \
                         the CMAF media profiles nor Apple's HLS authoring spec carry MP3 in fMP4. Use \
                         audio=auto, opus or aac for HLS, or single-file / audio-only output for MP3"
                            .to_string()
                    }
                    (AudioCodecPolicy::ForceVorbis, _, _) => format!(
                        "audio=vorbis goes in a WebM file (container=webm) or an audio-only .ogg \
                         (audio-container=ogg); {} has no Vorbis mapping",
                        if hls { "an HLS (CMAF) package".to_string() } else { self.container.file_label().to_string() }
                    ),
                    (_, _, Container::WebM) => format!(
                        "a WebM file carries Opus or Vorbis audio, and audio={name} asks for another codec: use \
                         audio=opus, vorbis (or auto), or write an MP4 (container=mp4)"
                    ),
                    (_, _, Container::Ogg) => format!(
                        "an Ogg file holds Opus or Vorbis, not what audio={name} makes: use audio=opus or vorbis, \
                         or audio-container=mp4 for an .m4a"
                    ),
                    (_, _, Container::Mp3) => format!(
                        "audio-only output is an .mp3 file, which holds MP3 only: use audio=mp3 (or auto, which \
                         means MP3 there), or audio-container=mp4 for an .m4a (or ogg for Opus and Vorbis)"
                    ),
                    (_, _, Container::Flac) => format!(
                        "a native FLAC file holds FLAC only, not what audio={name} makes; use audio-container=mp4"
                    ),
                    _ => format!("{} cannot hold what audio={name} makes", self.container.file_label()),
                }
            );
        }
        let target = self.audio_encode_codec();
        let mp3 = target == AudioCodec::Mp3;
        if mp3 && matches!(self.audio_channels, AudioChannels::Surround51 | AudioChannels::Surround71) {
            bail!(
                "audio-channels={} with MP3 output: MP3 carries two channels at most (a surround \
                 source is downmixed to stereo). Use audio=opus for surround",
                self.audio_channels.as_str()
            );
        }
        // The widest layout each other codec carries, against the one asked for.
        let max_channels = match target {
            AudioCodec::HeAacV2 => Some((2, "two channels (a surround source is downmixed to stereo)")),
            AudioCodec::Ac3 | AudioCodec::Eac3 | AudioCodec::Dts => {
                Some((6, "5.1 at most (a 7.1 source is downmixed to 5.1)"))
            }
            _ => None,
        };
        if let (Some((max, what)), Some(wanted)) = (max_channels, self.audio_channels.layout())
            && wanted.len() > max
        {
            bail!(
                "audio-channels={} with {} output: {} carries {what}. Use audio=opus, aac or he-aac for {}",
                self.audio_channels.as_str(),
                target.name(),
                target.name(),
                self.audio_channels.as_str()
            );
        }
        if target == AudioCodec::HeAacV2 && self.audio_channels == AudioChannels::Mono {
            bail!("audio-channels=mono with HE-AAC v2: parametric stereo codes a stereo image; use audio=he-aac for mono");
        }
        if let Some(q) = self.audio_quality {
            if target != AudioCodec::Vorbis {
                bail!("audio-quality applies to Vorbis output (audio=vorbis); {} takes audio-bitrate", target.name());
            }
            if !(-1.0..=10.0).contains(&q) {
                bail!("audio-quality {q} is outside Vorbis's -1..=10");
            }
        }
        if self.audio_stereo_fallback {
            if !hls {
                bail!("audio-stereo-fallback is for HLS output (a second audio rendition in the group)");
            }
            if matches!(self.audio_channels, AudioChannels::Mono | AudioChannels::Stereo) {
                bail!(
                    "audio-stereo-fallback with audio-channels={}: the audio is not surround, so there \
                     is nothing to fall back from",
                    self.audio_channels.as_str()
                );
            }
        }
        if let Some(bps) = self.audio_bitrate {
            check_audio_bitrate(target, self.audio.is_lossless(), bps)?;
        }
        Ok(())
    }

    /// Refuse a rung whose rate request cannot be coded — a bitrate beside a
    /// CRF or under `--seam-mode constqp`, a buffer without a bitrate, a
    /// bitrate on AV1 — in the encoder's own words
    /// ([`codec::encode::h26x_sw::rate_refusal`]), with the rung policy
    /// resolved so a rate from `--video-bitrate` or `bitrate=` is judged on
    /// the rung it lands on. Where the job may encode is judged once its pool
    /// is known (`multigpu::check_rate_pool`).
    pub(crate) fn check_rates(&self) -> Result<()> {
        let codec = self.video_codec.codec();
        // `constqp` constant-QPs the chunks of the multi-GPU single-file path.
        // Refused whichever path the job ends up on: the request itself says
        // two different things.
        let constant_qp =
            matches!(self.mode, OutputMode::SingleFile) && self.chunk_seam_mode == ChunkSeamMode::ParallelConstQp;
        // A constant-rate rung with no rate of its own is judged with the
        // default it will be given; the frame rate is not known yet, and only
        // the rate's presence matters here.
        for r in &self.with_constant_rates_resolved(30.0).rungs {
            if let Some(why) =
                codec::encode::h26x_sw::rate_refusal(codec, &r.quality.overrides, r.quality.crf, constant_qp)
            {
                bail!("rung '{}': {why}", r.label);
            }
        }
        Ok(())
    }

    /// The first rung, with the rung policy resolved, that is coded to a
    /// bitrate: its label and rate. `None` for a job of quality targets.
    pub fn bitrate_rung(&self) -> Option<(String, u32)> {
        self.with_rung_policy_resolved()
            .rungs
            .into_iter()
            .find_map(|r| r.quality.overrides.bitrate.map(|bps| (r.label, bps)))
    }

    /// The first rung, with the rung policy resolved, coded to an **average**
    /// bitrate (a bitrate rung that is not `rate=cbr`): its label and rate.
    pub fn average_rate_rung(&self) -> Option<(String, u32)> {
        self.with_rung_policy_resolved().rungs.into_iter().find_map(|r| {
            let o = r.quality.overrides;
            (o.rate_mode != Some(codec::encode::tuning::RateMode::Constant))
                .then_some(o.bitrate)
                .flatten()
                .map(|bps| (r.label, bps))
        })
    }

    /// The first rung, with the rung policy resolved, coded at a **constant**
    /// rate (`rate=cbr`): its label and rate, `None` for the rate when it has
    /// none of its own yet (see [`Self::with_constant_rates_resolved`]).
    pub fn constant_rate_rung(&self) -> Option<(String, Option<u32>)> {
        self.with_rung_policy_resolved().rungs.into_iter().find_map(|r| {
            let o = r.quality.overrides;
            (o.rate_mode == Some(codec::encode::tuning::RateMode::Constant)).then_some((r.label, o.bitrate))
        })
    }

    /// The spec with the rung policy resolved (see
    /// [`Self::with_rung_policy_resolved`]) and every constant-rate rung
    /// (`rate=cbr`) that has no rate of its own — no `@RATE`, no
    /// `--video-bitrate`, no `bitrate=` rule — given the default for its
    /// codec, short side and `frame_rate`
    /// ([`codec::encode::tuning::default_cbr_bitrate`]). The job engine
    /// calls this once the output frame rate is known, so every encoder, and
    /// the HLS playlist, sees an explicit rate. Anything else is unchanged.
    ///
    /// A GOP given in seconds ([`Self::gop_seconds`]) is made frames at
    /// `frame_rate` here too ([`Self::with_gop_seconds_resolved`]), for the
    /// same reason: it needs the output frame rate.
    pub fn with_constant_rates_resolved(&self, frame_rate: f64) -> OutputSpec {
        use codec::encode::tuning::{RateMode, default_cbr_bitrate};
        let mut resolved = self.with_gop_seconds_resolved(frame_rate).with_rung_policy_resolved();
        let codec = self.video_codec.codec();
        for rung in &mut resolved.rungs {
            let short_side = rung.short_side();
            let o = &mut rung.quality.overrides;
            if o.rate_mode == Some(RateMode::Constant) && o.bitrate.is_none() {
                let bps = default_cbr_bitrate(codec, short_side, frame_rate);
                o.bitrate = Some(bps);
                tracing::debug!(rung = %rung.label, bitrate = bps, frame_rate, ?codec, "rate=cbr: the default rate");
            }
        }
        resolved
    }

    /// `pinned`, if this job's mode can reach the one encode path that builds
    /// a backend pinned by name — the serial single-file encoder — else
    /// `None`. The HLS ladder leases its encoders from the pool and never
    /// reads the pin, so counting it there passed a job that then failed
    /// building its encoder after decode had started. A single-file job the
    /// chunk-and-stitch engine takes ignores the pin too; that routing is
    /// decided once the pool is known, and the job is checked again without
    /// the pin there.
    pub(crate) fn pin_honoured(
        &self,
        pinned: Option<codec::encode::EncoderBackend>,
    ) -> Option<codec::encode::EncoderBackend> {
        match self.mode {
            OutputMode::SingleFile => pinned,
            OutputMode::Hls { .. } | OutputMode::AudioOnly => None,
        }
    }

    /// The capability half of [`Self::validate`]: 10-bit / HDR output needs a
    /// backend for the spec's codec that produces it — one compiled into this
    /// build, or the backend `pinned` by name (`TRANSCODE_ENCODER_BACKEND`),
    /// which is built with or without its `-fallback` feature. `validate`
    /// passes the pin from the environment; taking it as an argument keeps the
    /// rule testable without touching process state.
    pub(crate) fn check_encoder_caps(&self, pinned: Option<codec::encode::EncoderBackend>) -> Result<()> {
        caps::check_output_caps(
            self.color,
            self.bit_depth,
            self.video_codec.codec(),
            &codec::encode::compiled_encode_backends(),
            pinned,
        )
    }

    /// The half of the capability check only the source can settle, run once
    /// it is probed and before a frame is decoded. What [`Self::resolve_output`]
    /// makes of a source can be 10-bit (`bit_depth = Auto` keeps a 10-bit
    /// source's depth) or HDR (`color = Passthrough` keeps an HDR source's
    /// transfer) when the spec asked for neither, and [`Self::validate`] could
    /// not see that: a 10-bit source asked for H.264 on an NVENC-only build
    /// passed it, started decoding, and failed building the encoder. The
    /// backend pinned by name counts as `validate` counts it.
    pub(crate) fn check_source(&self, source_color: ColorMetadata, source_pixel_format: PixelFormat) -> Result<()> {
        self.check_source_against(
            source_color,
            source_pixel_format,
            &codec::encode::compiled_encode_backends(),
            self.pin_honoured(caps::pinned_encoder_backend()),
        )
    }

    /// [`Self::check_source`] against the backends `compiled` plus the one
    /// `pinned`, rather than this build's and the environment's, so the rule
    /// is testable on any build.
    pub(crate) fn check_source_against(
        &self,
        source_color: ColorMetadata,
        source_pixel_format: PixelFormat,
        compiled: &[codec::encode::EncoderBackend],
        pinned: Option<codec::encode::EncoderBackend>,
    ) -> Result<()> {
        let (color, pixel_format) = self.resolve_output(source_color, source_pixel_format);
        caps::check_source_output_caps(
            self.color,
            self.bit_depth,
            caps::SourceOutput {
                source_format: source_pixel_format,
                source_transfer: source_color.transfer,
                ten_bit: pixel_format == PixelFormat::Yuv420p10le,
                hdr: matches!(color.transfer, TransferFn::St2084 | TransferFn::AribStdB67),
            },
            self.video_codec.codec(),
            compiled,
            pinned,
        )
    }
}

/// The HDR transfer the decode pump maps an SDR source into (ITU-R BT.2408,
/// [`codec::colorspace::SdrToHdr`]): the output's transfer when the pump does
/// not tonemap, the output is PQ or HLG and the source is neither. `None`
/// otherwise — HDR in and HDR out passes through, SDR out needs no mapping.
///
/// Before the mapping existed an HDR policy on an SDR source only changed the
/// tags: SDR pixels went out labelled PQ or HLG and played as a different,
/// wrong picture.
pub fn sdr_into_hdr(
    tonemap_to_sdr: bool,
    source: &ColorMetadata,
    output: &ColorMetadata,
) -> Option<TransferFn> {
    let hdr = |t: TransferFn| matches!(t, TransferFn::St2084 | TransferFn::AribStdB67);
    (!tonemap_to_sdr && hdr(output.transfer) && !hdr(source.transfer)).then_some(output.transfer)
}

/// The 4:2:0 format the decode pump hands the encoder for a source
/// `format`: `Yuv420p` for every 8-bit layout, `Yuv420p10le` for every
/// 10- and 12-bit one (12-bit is narrowed with rounding; 4:2:2 / 4:4:4 are
/// chroma-downsampled; RGB is matrixed). Mirrors
/// [`codec::colorspace::normalize_layout_to_420`] and
/// [`codec::colorspace::convert_bit_depth_frame`].
pub fn encoder_input_format(format: PixelFormat) -> PixelFormat {
    match codec::colorspace::planar_bit_depth(format) {
        Some(b) if b > 8 => PixelFormat::Yuv420p10le,
        Some(_) => PixelFormat::Yuv420p,
        // Yuva444p10le (10-bit + alpha) narrows to 10-bit 4:2:0; NV12 / NV21
        // and RGB are 8-bit layouts.
        None if format == PixelFormat::Yuva444p10le => PixelFormat::Yuv420p10le,
        None => PixelFormat::Yuv420p,
    }
}

/// The GOP length, in seconds of output, when none is given (`gop` unset):
/// two seconds at the output frame rate. See [`gop_frames_for_seconds`].
pub const DEFAULT_GOP_SECONDS: f64 = 2.0;

/// A GOP of `seconds` in frames at `frame_rate`: rounded to the nearest
/// frame, never fewer than one. The one conversion the default GOP and a
/// GOP given in seconds (`gop=1.5s`) both go through.
pub fn gop_frames_for_seconds(seconds: f64, frame_rate: f64) -> u32 {
    ((frame_rate * seconds).round() as u32).max(1)
}

/// The mastering display an SDR source mapped into PQ is signalled with
/// (SMPTE ST 2086, HEVC SEI 137, MP4 `mdcv`): BT.709 primaries — the gamut the
/// SDR picture has — a D65 white, a peak at HDR reference white, 203 cd/m²
/// (ITU-R BT.2408: SDR white is placed there, and nothing in the mapped
/// picture is brighter), and a black of 0.
///
/// Written because renderers use it. The same SDR-in-PQ pixels, rendered back
/// to SDR and compared with the SDR they came from, came out closest with
/// these values and furthest with none: libplacebo (mpv's renderer), peak
/// detection off, 23.80 dB against 18.56 with no metadata and 22.45 with a
/// BT.2020 / 1000 cd/m² display; with peak detection on (mpv's default) 23.80
/// against 22.24 for both others; rivet's own tonemap 24.73 against 21.40. With
/// no mastering display BT.2408 has a renderer assume the whole 10 000 cd/m²
/// PQ range, and rivet assumes 1000. Apple's HDR metadata guidance recommends
/// `mdcv` / `clli` for HEVC HDR10 and has them carried as SEI when the boxes
/// are absent.
pub const SDR_IN_PQ_MASTERING_DISPLAY: codec::frame::MasteringDisplay =
    codec::frame::MasteringDisplay {
        primaries_r_x: 32000,
        primaries_r_y: 16500,
        primaries_g_x: 15000,
        primaries_g_y: 30000,
        primaries_b_x: 7500,
        primaries_b_y: 3000,
        white_point_x: 15635,
        white_point_y: 16450,
        max_luminance: 2_030_000,
        min_luminance: 0,
    };

/// The content light level an SDR source mapped into PQ is signalled with
/// (CTA-861.3, HEVC SEI 144, MP4 `clli`): MaxCLL and MaxFALL both 203 cd/m², the
/// bound the BT.2408 mapping puts on every pixel. Bounds, not measurements —
/// the values are declared before the frames are seen — so they never claim
/// less light than the picture has; the frame average is usually well below.
pub const SDR_IN_PQ_CONTENT_LIGHT_LEVEL: codec::frame::ContentLightLevel =
    codec::frame::ContentLightLevel {
        max_cll: 203,
        max_fall: 203,
    };

/// BT.2020 HDR color metadata (BT.2020 primaries and non-constant-luminance
/// matrix, limited range) for the given transfer (PQ or HLG).
fn hdr_metadata(transfer: TransferFn) -> ColorMetadata {
    ColorMetadata {
        transfer,
        matrix_coefficients: 9, // BT.2020 non-constant luminance
        colour_primaries: 9,    // BT.2020
        full_range: false,
        ..ColorMetadata::default()
    }
}

/// An `audio-bitrate` against the codec it is for: each codec's own range,
/// as wide as any layout makes it (the encoder narrows it to the track's own
/// layout and rate, and names the range when it does).
fn check_audio_bitrate(target: codec::audio::AudioCodec, lossless: bool, bps: u32) -> Result<()> {
    use codec::audio::AudioCodec;
    use codec::audio::encode::{aac, ac3, dts, opus};
    let range = |name: &str, lo: u32, hi: u32| -> Result<()> {
        if (lo..=hi).contains(&bps) { Ok(()) } else { bail!("audio bitrate {bps} bps is outside {name}'s range ({lo}..={hi})") }
    };
    match target {
        // Refused by `check_lossless_audio`.
        _ if lossless => Ok(()),
        AudioCodec::Mp3 => {
            if codec::audio::MP3_BITRATES.contains(&bps) {
                Ok(())
            } else {
                bail!(
                    "audio bitrate {bps} bps is not an MP3 bitrate: MP3 output is constant bitrate, one of {}",
                    codec::audio::MP3_BITRATES.map(|b| format!("{}k", b / 1000)).join(", ")
                )
            }
        }
        // The widest AAC band: 8 kb/s for mono, and the 13818-7 decoder
        // buffer's ceiling for 7.1 at 48 kHz.
        AudioCodec::Aac => range("AAC", aac::bitrate_range(48_000, 1).0, aac::bitrate_range(48_000, 8).1),
        AudioCodec::HeAac => range(
            "HE-AAC",
            aac::he_aac_bitrate_range(aac::Profile::HeAac, 1).0,
            aac::he_aac_bitrate_range(aac::Profile::HeAac, 8).1,
        ),
        AudioCodec::HeAacV2 => {
            let (lo, hi) = aac::he_aac_bitrate_range(aac::Profile::HeAacV2, 2);
            range("HE-AAC v2", lo, hi)
        }
        AudioCodec::Ac3 | AudioCodec::Eac3 => {
            if ac3::valid_bitrate(target, bps) {
                Ok(())
            } else if target == AudioCodec::Ac3 {
                bail!(
                    "audio bitrate {bps} bps is not an AC-3 bitrate (A/52 Table 5.18): one of {}",
                    ac3::AC3_BITRATES.map(|b| format!("{}k", b / 1000)).join(", ")
                )
            } else {
                bail!("audio bitrate {bps} bps is not an E-AC-3 bitrate: 32k..6144k, in whole kb/s")
            }
        }
        AudioCodec::Dts => {
            if dts::DTS_BITRATES.contains(&bps) {
                Ok(())
            } else {
                bail!(
                    "audio bitrate {bps} bps is not a DTS bitrate (ETSI TS 102 114 Table 5-7): one of {}",
                    dts::DTS_BITRATES.map(|b| if b % 1000 == 0 { format!("{}k", b / 1000) } else { format!("{}k", f64::from(b) / 1000.0) }).join(", ")
                )
            }
        }
        AudioCodec::Vorbis => bail!("audio-bitrate does not apply to Vorbis, which is variable-rate: set audio-quality (-1..=10)"),
        // 6 to 510 kb/s per stream; the widest layout (7.1, five streams).
        AudioCodec::Opus => range("Opus", opus::bitrate_range(1).0, opus::bitrate_range(8).1),
        AudioCodec::Flac { .. } | AudioCodec::Alac { .. } => Ok(()),
    }
}
