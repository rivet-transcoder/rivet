//! One canonical definition of the transcode "knobs", shared by every
//! front-end — the CLI (`transcode` / `splice` / `pipe` / `batch`), the HTTP
//! API, the batch manifest, and the IPC socket. Each surface reads its own
//! syntax (clap flags / JSON / YAML / query string / `key=value`) into a
//! [`TranscodeSettings`], then calls [`TranscodeSettings::into_spec`]. Add a
//! new option **once** here (a field + a line in `into_spec` + a `parse_*`
//! function + an [`apply_kv`](TranscodeSettings::apply_kv) arm) and every
//! surface picks it up, instead of maintaining several copies of the
//! spec-building logic.
//!
//! **This module is the single point of interpretation.** A surface may
//! *validate spelling* in its own way — clap's value enums list the accepted
//! words for `--help` and completion — but the *meaning* of every value is
//! decided by the `parse_*` functions here and nowhere else. The CLI's value
//! enums are pinned to this vocabulary by a test in the binary; the batch
//! manifest and the HTTP API call these functions directly. If two surfaces
//! ever disagree about what a word means, the bug is that one of them stopped
//! calling this module.

use anyhow::{Context, Result, bail};

use crate::spec::{
    AudioBitDepth, AudioChannels, AudioCodecPolicy, AudioDecodeDeny, HeAacPolicy, BitDepth, ChunkSeamMode, ColorPolicy, Container, DecodePolicy,
    EncodePolicy, FlacLevel, GpuFamily, OutputSpec, Quality, Rung,
};

// ── on the absence of a `speed` knob ────────────────────────────────────────
//
// There deliberately isn't one. `Quality` carries two calibrated dimensions —
// `target` (how good) and `tier` (how much effort) — and the per-encoder tuning
// tables (`nvenc_av1_params`, `qsv_av1_params`, …) exist so that a given target
// lands in the same VMAF band whichever backend runs it.
//
// A front-end `speed` flag couldn't improve on that:
//
// - As an encoder-native *number* it isn't portable. NVENC's `P1..P7` runs
//   fast→slow and oneVPL's `TargetUsage 1..7` runs slow→fast, so the same value
//   is the fastest setting on one card and the slowest on another — and with
//   `--gpu-family` or a multi-GPU host the caller can't know which will run the
//   job. It also bypasses the tables above, which is the one thing keeping
//   backends comparable.
// - As an x265-style *name* it would promise a tradeoff curve this hardware
//   doesn't have: the whole tier range is P5↔P7, a few percent of bitrate,
//   where on x265 the preset is the dominant knob.
//
// Library callers who really do know their backend still have the full range:
// build a `Rung::with_quality(Quality { target, tier, speed_preset, .. })`.

/// Output mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Single,
    Hls,
    /// The audio alone, as an `.mp3` file ([`OutputMode::AudioOnly`](crate::spec::OutputMode::AudioOnly)).
    Audio,
}

/// Every optional transcode knob, surface-agnostic. All-`None`/empty is "use the
/// defaults" (source-resolution single file, AV1 + audio passthrough, SDR).
#[derive(Debug, Clone, Default)]
pub struct TranscodeSettings {
    pub mode: Option<Mode>,
    /// Explicit rungs, each with its own bitrate when it names one
    /// (`WxH@RATE`) and its own fit when it names one (`WxH:cover:fixed`).
    /// Wins over `ladder` / `width`. Each size is a box the source is fitted
    /// into; see [`crate::fit`].
    pub rungs: Vec<RungArg>,
    /// How the source meets each rung's box: `contain` (default), `cover`,
    /// `pad` or `stretch`. See [`crate::fit::Fit`].
    pub fit: Option<crate::fit::Fit>,
    /// Whether a rung's box turns to the source's orientation: `auto`
    /// (default) or `fixed`. See [`crate::fit::Orientation`].
    pub orientation: Option<crate::fit::Orientation>,
    /// Whether a rung may be larger than the source. Off by default.
    pub upscale: bool,
    /// Derive a standard ABR ladder from the source.
    pub ladder: bool,
    pub max_short_side: Option<u32>,
    pub segment_seconds: Option<f32>,
    pub crf: Option<u8>,
    /// Perceptual quality target for every rung — `visually_lossless`, `high`,
    /// `standard` (default), `low`, or `vmaf=N` (a VMAF score, mapped to each
    /// backend's quantiser through the calibrated tables). Ignored for a rung
    /// when `crf` is set, since a CRF names the quantiser directly.
    pub target: Option<codec::encode::tuning::QualityTarget>,
    /// GOP length in frames for every rung. `None` = two seconds. See
    /// [`OutputSpec::gop`](crate::spec::OutputSpec::gop) for what it governs
    /// on each output path.
    pub gop: Option<u32>,
    pub audio: Option<AudioCodecPolicy>,
    /// Which text subtitle tracks to carry. `None` = all of them.
    pub subtitles: Option<crate::spec::SubtitlePolicy>,
    /// Target bitrate in bits per second for transcoded audio. `None` lets
    /// the encoder derive it: Opus from the channel layout (64k mono / 96k
    /// stereo / 320k 5.1 / 416k 7.1), MP3 128k stereo / 64k mono.
    pub audio_bitrate: Option<u32>,
    /// Output channel layout: `source` (default), `mono`, `stereo`, `5.1`,
    /// `7.1`. See [`AudioChannels`].
    pub audio_channels: Option<AudioChannels>,
    /// HLS: a stereo downmix rendition beside a surround one.
    pub audio_stereo_fallback: bool,
    /// Bit depth of FLAC / ALAC output: `source` (default), `16` or `24`.
    pub audio_bit_depth: Option<AudioBitDepth>,
    /// An HE-AAC source: `auto` (default), `passthrough` or `core`.
    pub he_aac: Option<HeAacPolicy>,
    /// Source audio codecs that may not be decoded: `aac`, `mp3`, …
    /// comma-separated. `None` / empty restricts nothing.
    pub audio_decode_deny: Option<AudioDecodeDeny>,
    /// FLAC compression effort: `fast`, `default` or `best`.
    pub flac_level: Option<FlacLevel>,
    /// The file an audio-only output is: `mp3`, `flac` or `mp4` (an `.m4a`).
    /// `None` follows the codec — a native `.flac` for `audio=flac`, an
    /// `.m4a` for `audio=alac`, else an `.mp3`.
    pub audio_container: Option<Container>,
    /// Video bitrate in bits per second for every rung that does not name
    /// its own (`WxH@RATE`) or get one from `encode_policy`: the rung is
    /// coded to a rate rather than to `target`. `None` = a quality target, as
    /// always. The native software H.264 / H.265 encoder is the one that
    /// codes to a rate; see [`EncodeOverrides::bitrate`](codec::encode::tuning::EncodeOverrides::bitrate).
    pub video_bitrate: Option<u32>,
    /// Coded picture buffer, in milliseconds, for every bitrate rung that
    /// does not get one from `encode_policy`; `Some(0)` declares none. See
    /// [`EncodeOverrides::buffer_ms`](codec::encode::tuning::EncodeOverrides::buffer_ms).
    pub video_buffer_ms: Option<u32>,
    /// How every bitrate rung that does not get a mode from `encode_policy`
    /// (`rate=`) spends its rate: `None` / `Average` is the average rate
    /// bitrate rungs have always been; `Constant` (`rate-mode=cbr`) is CBR —
    /// the rate is also the maximum, an HRD buffer is declared, and the
    /// hardware encoders hold the rate. A constant-rate rung with no rate of
    /// its own takes `video_bitrate`, else the default for its codec, size
    /// and frame rate ([`default_cbr_bitrate`](codec::encode::tuning::default_cbr_bitrate)).
    /// See [`EncodeOverrides::rate_mode`](codec::encode::tuning::EncodeOverrides::rate_mode).
    pub rate_mode: Option<codec::encode::tuning::RateMode>,
    /// Audio filter chain (`channelmap`) applied to decoded PCM before the Opus
    /// encoder. String surfaces parse `codec::audio::filter::parse_chain` at the
    /// edge, the same way `filters` does for video.
    pub audio_filters: Vec<codec::audio::filter::AudioFilter>,
    pub color: Option<ColorPolicy>,
    /// 4:4:4 → 4:2:0 chroma filter: `box` (default) or `lanczos`.
    pub chroma_downsample: Option<codec::colorspace::ChromaDownsample>,
    pub bit_depth: Option<BitDepth>,
    pub seam: Option<ChunkSeamMode>,
    pub max_fps: Option<f64>,
    /// Pin encode to one GPU index.
    pub gpu: Option<u32>,
    /// Restrict encode to one vendor family.
    pub gpu_family: Option<GpuFamily>,
    /// Use a single GPU (serial), the first available.
    pub single_gpu: bool,
    /// The decode plan: `Auto` (split across the capable cards), `Whole`,
    /// `SpecificGpu(i)`, `FastestGpu`, `Ranges(n)`. See [`DecodePolicy`].
    pub decode_policy: DecodePolicy,
    /// The encode plan, when given as one value (`--encode`, settings key
    /// `encode`): `AllGpus`, `PerRung`, `Family(_)`, `SingleGpu(_)`. `None`
    /// falls back to the older per-flag spellings (`gpu`, `gpu_family`,
    /// `single_gpu`), which stay as aliases.
    pub encode: Option<EncodePolicy>,
    /// `seam=serial` was seen. It predates the split of "seam quality" from
    /// "encode plan" and means `encode=single`; recorded here rather than
    /// written into `encode`, so an explicit `encode` wins whatever order the
    /// two arrived in. Set through [`Self::apply_seam`].
    pub seam_serial: bool,
    /// Per-rung encoder knobs by ladder position: `None` = no policy (the
    /// default), or a policy — [`RungPolicy::recommended`](codec::encode::tuning::RungPolicy::recommended) via the CLI's
    /// `recommended`, or a parsed grammar string. See
    /// [`OutputSpec::rung_policy`](crate::spec::OutputSpec::rung_policy).
    pub encode_policy: Option<codec::encode::tuning::RungPolicy>,
    /// Single-output width/height (the `pipe`/`ipc` scaling knobs). Used only
    /// when neither `rungs` nor `ladder` is set; defaults to the source size.
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// Video filter chain (crop/pad/flip/rotate/grayscale) applied before
    /// per-rung scaling. The canonical structured form; string surfaces parse
    /// `codec::filter::parse_chain` at the edge.
    pub filters: Vec<codec::filter::VideoFilter>,
    /// Output video codec: `av1` (default), `h264`, or `h265`. `None` = av1.
    pub video_codec: Option<crate::spec::VideoCodecPolicy>,
    /// Splice **trim in-point** in seconds (`None` = start of input).
    pub trim_start: Option<f64>,
    /// Splice **trim out-point** in seconds (`None` = end of input).
    pub trim_end: Option<f64>,
}

impl TranscodeSettings {
    /// Build an [`OutputSpec`] from these settings against a source resolution.
    /// This is the **single** spec-building implementation for all surfaces.
    pub fn into_spec(self, src_w: u32, src_h: u32) -> Result<OutputSpec> {
        if self.mode == Some(Mode::Audio) {
            return self.into_audio_only_spec(true);
        }

        // `speed_preset` / `tier` stay at their defaults on purpose — see the
        // note above on why there's no front-end speed knob.
        let quality = Quality {
            crf: self.crf,
            target: self.target.unwrap_or_default(),
            ..Default::default()
        };

        let rungs: Vec<Rung> = if !self.rungs.is_empty() {
            self.rungs
                .iter()
                .map(|r| {
                    // A rung's own `@RATE` is its own override, so it wins over
                    // the policy and over `video_bitrate` (see below).
                    let mut q = quality.clone();
                    q.overrides.bitrate = r.bitrate;
                    let mut rung = Rung::new(r.width, r.height).with_quality(q);
                    rung.fit = r.fit;
                    rung.orientation = r.orientation;
                    rung.upscale = r.upscale;
                    rung
                })
                .collect()
        } else if self.ladder {
            crate::ladder::standard_ladder(src_w, src_h, self.max_short_side)
                .into_iter()
                .map(|r| r.with_quality(quality.clone()))
                .collect()
        } else {
            // Single rung at the requested size, else the source — even-aligned
            // (AV1 4:2:0 needs even dimensions).
            let w = self.width.unwrap_or(src_w) & !1;
            let h = self.height.unwrap_or(src_h) & !1;
            if w == 0 || h == 0 {
                bail!("source resolution unknown ({src_w}x{src_h}); set explicit rungs or width/height");
            }
            vec![Rung::new(w, h).with_quality(quality.clone())]
        };
        if rungs.is_empty() {
            bail!("no rungs to produce");
        }

        let mut spec = match self.mode.unwrap_or(Mode::Single) {
            Mode::Hls => OutputSpec::hls(rungs, self.segment_seconds.unwrap_or(4.0)),
            Mode::Single => OutputSpec::single_file(rungs),
            Mode::Audio => unreachable!("built above"),
        };

        spec.fit = self.fit.unwrap_or_default();
        spec.orientation = self.orientation.unwrap_or_default();
        spec.upscale = self.upscale;
        if let Some(a) = self.audio {
            spec.audio = a;
        }
        if let Some(s) = self.subtitles {
            spec.subtitles = s;
        }
        spec.audio_bitrate = self.audio_bitrate;
        spec.audio_filters = self.audio_filters;
        spec.audio_channels = self.audio_channels.unwrap_or_default();
        spec.audio_stereo_fallback = self.audio_stereo_fallback;
        spec.audio_bit_depth = self.audio_bit_depth.unwrap_or_default();
        spec.he_aac = self.he_aac.unwrap_or_default();
        spec.audio_decode_deny = self.audio_decode_deny.unwrap_or_default();
        spec.flac_level = self.flac_level.unwrap_or_default();
        if self.audio_container.is_some() {
            bail!("audio-container names the file of an audio-only output (mode=audio)");
        }
        spec.max_frame_rate = self.max_fps;
        if let Some(c) = self.color {
            spec = spec.with_color(c);
        }
        if let Some(f) = self.chroma_downsample {
            spec = spec.with_chroma_downsample(f);
        }
        if let Some(b) = self.bit_depth {
            spec = spec.with_bit_depth(b);
        }
        if let Some(s) = self.seam {
            spec = spec.chunk_seam_mode(s);
        }

        // The encode plan: one value wins; else the legacy `seam=serial`
        // (which always meant "one encoder"); else the older per-flag
        // spellings, pinned index > vendor family > single > all.
        spec = if let Some(policy) = self.encode {
            spec.encode_policy(policy)
        } else if self.seam_serial {
            spec.encode_policy(EncodePolicy::SingleGpu(None))
        } else if let Some(idx) = self.gpu {
            spec.encode_policy(EncodePolicy::SingleGpu(Some(idx)))
        } else if let Some(fam) = self.gpu_family {
            spec.encode_policy(EncodePolicy::Family(fam))
        } else if self.single_gpu {
            spec.encode_policy(EncodePolicy::SingleGpu(None))
        } else {
            spec.encode_policy(EncodePolicy::AllGpus)
        };
        spec = spec.decode_policy(self.decode_policy);
        spec = spec.with_gop(self.gop);
        // `video_bitrate` / `video_buffer_ms` / `rate_mode` are "every rung",
        // so they sit beneath the whole policy — its global set and its rules
        // both win — and a rung's own `@RATE` wins over all of it.
        let video_rate = codec::encode::tuning::EncodeOverrides {
            bitrate: self.video_bitrate,
            buffer_ms: self.video_buffer_ms,
            rate_mode: self.rate_mode,
            ..Default::default()
        };
        let mut policy = self.encode_policy.unwrap_or_default();
        policy.global = video_rate.merge(policy.global);
        spec = spec.with_rung_policy(policy);
        spec = spec.with_filters(self.filters);
        spec = spec.with_trim(self.trim_start, self.trim_end);
        if let Some(c) = self.video_codec {
            spec = spec.with_video_codec(c);
        }

        spec.validate().context("invalid output spec")?;
        Ok(spec)
    }

    /// [`Self::into_spec`] against a probed source. A source with no video (a
    /// bare MP3, an M4A: `video_codec` `none`) under a single-file job has
    /// nothing for a rung, so the job is its audio-only form — the one
    /// `mode=audio` asks for — with the video knobs a shared default set may
    /// carry ignored rather than refused. HLS of such a source is refused:
    /// an HLS package needs a video variant.
    pub fn into_spec_for(self, source: &crate::probe::MediaInfo) -> Result<OutputSpec> {
        if source.video_codec == "none" {
            return match self.mode.unwrap_or(Mode::Single) {
                Mode::Single => self.into_audio_only_spec(false),
                Mode::Audio => self.into_audio_only_spec(true),
                Mode::Hls => bail!("the input has no video, and an HLS package needs a video variant: use mode=audio"),
            };
        }
        let (width, height) = source.display_dims();
        self.into_spec(width, height)
    }

    /// The audio-only spec (`mode=audio`): the audio knobs, and — when
    /// `strict`, the mode asked for by name — a refusal for every video one
    /// given, since no video is written.
    fn into_audio_only_spec(self, strict: bool) -> Result<OutputSpec> {
        let video_knobs = [
            ("rung", !self.rungs.is_empty()),
            ("fit", self.fit.is_some()),
            ("orientation", self.orientation.is_some()),
            ("upscale", self.upscale),
            ("ladder", self.ladder),
            ("crf", self.crf.is_some()),
            ("target", self.target.is_some()),
            ("video-bitrate", self.video_bitrate.is_some()),
            ("codec", self.video_codec.is_some()),
            ("filter", !self.filters.is_empty()),
            ("width/height", self.width.is_some() || self.height.is_some()),
        ];
        if let Some((knob, _)) = video_knobs.iter().find(|(_, set)| *set) {
            if strict {
                bail!("mode=audio writes no video, so `{knob}` has nothing to apply to");
            }
            tracing::info!(knob, "the input has no video; the video settings do not apply");
        }
        let audio = self.audio.unwrap_or_default();
        let container = self.audio_container.unwrap_or(OutputSpec::audio_only_container(audio));
        let mut spec = OutputSpec::audio_only_in(container);
        spec.audio = audio;
        spec.audio_bitrate = self.audio_bitrate;
        spec.audio_filters = self.audio_filters;
        spec.audio_channels = self.audio_channels.unwrap_or_default();
        spec.audio_stereo_fallback = self.audio_stereo_fallback;
        spec.audio_bit_depth = self.audio_bit_depth.unwrap_or_default();
        spec.he_aac = self.he_aac.unwrap_or_default();
        spec.audio_decode_deny = self.audio_decode_deny.unwrap_or_default();
        spec.flac_level = self.flac_level.unwrap_or_default();
        spec = spec.with_trim(self.trim_start, self.trim_end);
        spec.validate().context("invalid output spec")?;
        Ok(spec)
    }

    /// Apply one `key=value` setting (the IPC header / generic string form).
    /// Keys mirror the CLI flags. Unknown keys error.
    pub fn apply_kv(&mut self, key: &str, val: &str) -> Result<()> {
        match key {
            "mode" => self.mode = Some(parse_mode(val)?),
            "rung" | "rungs" => {
                for r in val.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    self.rungs.push(parse_rung(r)?);
                }
            }
            "fit" => self.fit = Some(crate::fit::Fit::parse(val)?),
            "orientation" => self.orientation = Some(crate::fit::Orientation::parse(val)?),
            "upscale" => self.upscale = parse_bool(val),
            "ladder" => self.ladder = parse_bool(val),
            "max-short-side" => self.max_short_side = Some(val.parse().context("max-short-side")?),
            "segment-seconds" => self.segment_seconds = Some(val.parse().context("segment-seconds")?),
            "crf" => self.crf = Some(val.parse().context("crf")?),
            "target" | "quality" => self.target = Some(parse_quality_target(val)?),
            "gop" | "keyframe-interval" => self.gop = Some(val.parse().context("gop")?),
            // Accepted and refused by name so an old `speed=6` header gets the
            // reason rather than "unknown setting".
            "speed" | "preset" => bail!(
                "'{key}' is no longer a knob: an encoder-native preset number means opposite \
                 things on different GPUs (NVENC P1 is the fastest, Intel TargetUsage 1 the \
                 slowest), and it bypasses the per-encoder tuning tables that keep quality \
                 comparable across backends. Use `crf` for quality; library callers can still \
                 set `Quality::tier`."
            ),
            "audio" => self.audio = Some(parse_audio(val)?),
            "subtitles" | "subs" => self.subtitles = Some(parse_subtitles(val)?),
            "audio-bitrate" | "ab" => self.audio_bitrate = Some(parse_bitrate(val)?),
            "audio-channels" | "ac" => self.audio_channels = Some(parse_audio_channels(val)?),
            "audio-stereo-fallback" => self.audio_stereo_fallback = parse_bool(val),
            "audio-bit-depth" => self.audio_bit_depth = Some(parse_audio_bit_depth(val)?),
            "he-aac" => self.he_aac = Some(parse_he_aac(val)?),
            "audio-decode-deny" => self.audio_decode_deny = Some(parse_audio_decode_deny(val)?),
            "flac-compression" => self.flac_level = Some(parse_flac_level(val)?),
            "audio-container" => self.audio_container = parse_audio_container(val)?,
            "video-bitrate" | "vb" => self.video_bitrate = Some(parse_bitrate(val)?),
            "video-buffer" => self.video_buffer_ms = Some(parse_buffer(val)?),
            "rate-mode" => self.rate_mode = Some(parse_rate_mode(val)?),
            "audio-filter" | "af" => self.audio_filters = codec::audio::filter::parse_chain(val)?,
            "color" => self.color = Some(parse_color(val)?),
            "chroma-downsample" | "chroma-filter" => {
                self.chroma_downsample = Some(parse_chroma_downsample(val)?)
            }
            "bit-depth" | "pixel-format" => self.bit_depth = Some(parse_bit_depth(val)?),
            "seam" | "seam-mode" => self.apply_seam(val)?,
            "max-fps" => self.max_fps = Some(val.parse().context("max-fps")?),
            "gpu" => self.gpu = Some(val.parse().context("gpu")?),
            "gpu-family" => self.gpu_family = Some(parse_gpu_family(val)?),
            "single-gpu" => self.single_gpu = parse_bool(val),
            "decode" | "decode-gpu" | "decode-split" | "decode-ranges" => {
                self.decode_policy = parse_decode_plan(val)?
            }
            "encode" | "schedule" => self.encode = Some(parse_encode_plan(val)?),
            "encode-policy" => self.encode_policy = Some(parse_encode_policy(val)?),
            "width" => self.width = Some(val.parse().context("width")?),
            "height" => self.height = Some(val.parse().context("height")?),
            "filter" => self.filters = codec::filter::parse_chain(val)?,
            "codec" => self.video_codec = Some(parse_video_codec(val)?),
            o => bail!(
                "unknown setting '{o}' (mode/rung/fit/orientation/upscale/ladder/max-short-side/segment-seconds/crf/\
                 target/gop/video-bitrate/video-buffer/rate-mode/audio/audio-bitrate/audio-filter/\
                 audio-channels/audio-stereo-fallback/audio-bit-depth/he-aac/audio-decode-deny/flac-compression/audio-container/\
                 subtitles/color/bit-depth/seam/\
                 max-fps/encode/decode/gpu/gpu-family/single-gpu/decode-gpu/encode-policy/\
                 width/height/filter/codec)"
            ),
        }
        Ok(())
    }

    /// Parse a whole `key=value key=value …` line into settings.
    pub fn parse_kv_line(line: &str) -> Result<Self> {
        let mut s = Self::default();
        for tok in line.split_whitespace() {
            let (k, v) = tok
                .split_once('=')
                .with_context(|| format!("bad setting '{tok}' (expected key=value)"))?;
            s.apply_kv(k, v)?;
        }
        Ok(s)
    }

    /// Interpret a `seam` value — the one place the legacy `serial` spelling
    /// is understood. `parallel` / `constqp` set the seam quality; `serial`
    /// records that the encode plan is `single` (see [`Self::seam_serial`]).
    pub fn apply_seam(&mut self, raw: &str) -> Result<()> {
        match parse_seam(raw)? {
            SeamValue::Mode(mode) => self.seam = Some(mode),
            SeamValue::EncodeSingle => self.seam_serial = true,
        }
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.mode.is_none()
            && self.rungs.is_empty()
            && self.fit.is_none()
            && self.orientation.is_none()
            && !self.upscale
            && !self.ladder
            && self.max_short_side.is_none()
            && self.segment_seconds.is_none()
            && self.crf.is_none()
            && self.target.is_none()
            && self.gop.is_none()
            && self.audio.is_none()
            && self.subtitles.is_none()
            && self.audio_bitrate.is_none()
            && self.audio_channels.is_none()
            && !self.audio_stereo_fallback
            && self.audio_bit_depth.is_none()
            && self.he_aac.is_none()
            && self.audio_decode_deny.is_none()
            && self.flac_level.is_none()
            && self.audio_container.is_none()
            && self.video_bitrate.is_none()
            && self.video_buffer_ms.is_none()
            && self.rate_mode.is_none()
            && self.audio_filters.is_empty()
            && self.color.is_none()
            && self.bit_depth.is_none()
            && self.seam.is_none()
            && self.max_fps.is_none()
            && self.gpu.is_none()
            && self.gpu_family.is_none()
            && !self.single_gpu
            && self.decode_policy == DecodePolicy::Auto
            && self.encode.is_none()
            && !self.seam_serial
            && self.width.is_none()
            && self.height.is_none()
            && self.filters.is_empty()
            && self.video_codec.is_none()
    }
}

// ── central string vocabulary (the single source of truth) ──────────────

pub fn parse_mode(s: &str) -> Result<Mode> {
    match s {
        "single" => Ok(Mode::Single),
        "hls" => Ok(Mode::Hls),
        "audio" => Ok(Mode::Audio),
        o => bail!("mode must be single|hls|audio, got '{o}'"),
    }
}

pub fn parse_audio(s: &str) -> Result<AudioCodecPolicy> {
    match s {
        "auto" => Ok(AudioCodecPolicy::Auto),
        "opus" => Ok(AudioCodecPolicy::ForceOpus),
        "mp3" => Ok(AudioCodecPolicy::ForceMp3),
        "aac" => Ok(AudioCodecPolicy::ForceAac),
        "flac" => Ok(AudioCodecPolicy::Flac),
        "alac" => Ok(AudioCodecPolicy::Alac),
        "drop" => Ok(AudioCodecPolicy::Drop),
        o => bail!("audio must be auto|opus|mp3|aac|flac|alac|drop, got '{o}'"),
    }
}

/// Parse `audio-bit-depth`: `source`, `16` or `24`.
pub fn parse_audio_bit_depth(s: &str) -> Result<AudioBitDepth> {
    match s {
        "source" => Ok(AudioBitDepth::Source),
        "16" => Ok(AudioBitDepth::Sixteen),
        "24" => Ok(AudioBitDepth::TwentyFour),
        o => bail!("audio-bit-depth must be source|16|24, got '{o}'"),
    }
}

/// Parse `he-aac`: `auto`, `passthrough` or `core`.
pub fn parse_he_aac(s: &str) -> Result<HeAacPolicy> {
    match s {
        "auto" => Ok(HeAacPolicy::Auto),
        "passthrough" => Ok(HeAacPolicy::Passthrough),
        "core" => Ok(HeAacPolicy::Core),
        o => bail!("he-aac must be auto|passthrough|core, got '{o}'"),
    }
}

/// Parse `audio-decode-deny`: source audio codecs that may not be decoded,
/// comma-separated, from [`AudioDecodeDeny::CODECS`]. Empty or `none`
/// denies nothing.
pub fn parse_audio_decode_deny(s: &str) -> Result<AudioDecodeDeny> {
    let mut deny = AudioDecodeDeny::NONE;
    for name in s.split(',').map(str::trim).filter(|n| !n.is_empty()) {
        if name == "none" {
            continue;
        }
        deny = deny.with(name).with_context(|| {
            format!(
                "audio-decode-deny takes audio codec names ({}) or none, comma-separated; got '{name}'",
                AudioDecodeDeny::CODECS.join("|")
            )
        })?;
    }
    Ok(deny)
}

/// Parse `flac-compression`: `fast`, `default` or `best`.
pub fn parse_flac_level(s: &str) -> Result<FlacLevel> {
    match s {
        "fast" => Ok(FlacLevel::Fast),
        "default" => Ok(FlacLevel::Default),
        "best" => Ok(FlacLevel::Best),
        o => bail!("flac-compression must be fast|default|best, got '{o}'"),
    }
}

/// Parse `audio-container`: `auto` (`None`: follow the codec), `mp3`,
/// `flac`, or `mp4` / `m4a`.
pub fn parse_audio_container(s: &str) -> Result<Option<Container>> {
    match s {
        "auto" => Ok(None),
        "mp3" => Ok(Some(Container::Mp3)),
        "flac" => Ok(Some(Container::Flac)),
        "mp4" | "m4a" => Ok(Some(Container::M4a)),
        o => bail!("audio-container must be auto|mp3|flac|mp4, got '{o}'"),
    }
}

/// Parse an output channel layout: `source`, `mono`, `stereo`, `5.1`, `7.1`
/// (and the counts `1`, `2`, `6`, `8`).
pub fn parse_audio_channels(s: &str) -> Result<AudioChannels> {
    match s.trim().to_ascii_lowercase().as_str() {
        "source" => Ok(AudioChannels::Source),
        "mono" | "1" => Ok(AudioChannels::Mono),
        "stereo" | "2" => Ok(AudioChannels::Stereo),
        "5.1" | "6" => Ok(AudioChannels::Surround51),
        "7.1" | "8" => Ok(AudioChannels::Surround71),
        o => bail!("audio-channels must be source|mono|stereo|5.1|7.1, got '{o}'"),
    }
}

/// Parse an `--encode-policy` value: `recommended` (the measured ladder
/// policy), `off` / `none` (an empty policy — the control arm), or the rule
/// grammar (`qstep=2;short<=2159:tiles=1x1;any:refs=3`).
pub fn parse_encode_policy(s: &str) -> Result<codec::encode::tuning::RungPolicy> {
    use codec::encode::tuning::RungPolicy;
    match s.trim().to_ascii_lowercase().as_str() {
        "recommended" | "default" => Ok(RungPolicy::recommended()),
        "off" | "none" => Ok(RungPolicy::new()),
        _ => RungPolicy::parse(s).map_err(anyhow::Error::msg).context("encode-policy"),
    }
}

/// Parse a bitrate the way an ffmpeg command line writes one: a plain count of
/// bits per second, or a `k` / `M` suffix (`240k`, `1.5M`). Decimal SI, matching
/// ffmpeg — `240k` is 240 000 bps, not 245 760. The encode policy grammar's
/// `bitrate=` reads the same spelling through the same function.
pub fn parse_bitrate(s: &str) -> Result<u32> {
    codec::encode::tuning::parse_bitrate(s).map_err(anyhow::Error::msg)
}

/// Parse a coded picture buffer duration (`--video-buffer`): `500ms`, `1s`,
/// `1.5s`, or `0` for none, as whole milliseconds. The encode policy
/// grammar's `buffer=` reads the same spelling through the same function.
pub fn parse_buffer(s: &str) -> Result<u32> {
    codec::encode::tuning::parse_buffer_ms(s).map_err(anyhow::Error::msg)
}

/// Parse a rate mode (`--rate-mode`, settings key `rate-mode`): `cbr` /
/// `constant` for a constant rate, `average` / `abr` for the average rate
/// bitrate rungs have always had. The encode policy grammar's `rate=` reads
/// the same spelling through the same function.
pub fn parse_rate_mode(s: &str) -> Result<codec::encode::tuning::RateMode> {
    s.parse().map_err(anyhow::Error::msg).context("rate-mode")
}

/// Parse a subtitle selection: `all` (the default; `copy` and `keep` are the
/// older spellings), `none` (`drop`), or a comma-separated language list such
/// as `eng,deu` — ISO 639 codes in either length, so `en,de` names the same
/// tracks. The list's order is the output order.
pub fn parse_subtitles(s: &str) -> Result<crate::spec::SubtitlePolicy> {
    use crate::spec::SubtitlePolicy;
    let lowered = s.trim().to_ascii_lowercase();
    match lowered.as_str() {
        "all" | "copy" | "keep" => Ok(SubtitlePolicy::All),
        "none" | "drop" => Ok(SubtitlePolicy::Drop),
        _ => {
            let langs: Vec<String> = lowered
                .split(',')
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(String::from)
                .collect();
            let is_code = |l: &String| (2..=3).contains(&l.len()) && l.bytes().all(|b| b.is_ascii_lowercase());
            if langs.is_empty() || !langs.iter().all(is_code) {
                bail!(
                    "subtitles must be all|none|<lang,lang,...> (ISO 639 codes such as eng,deu \
                     or en,de), got '{s}'"
                );
            }
            Ok(SubtitlePolicy::Only(langs))
        }
    }
}

pub fn parse_color(s: &str) -> Result<ColorPolicy> {
    match s {
        "sdr" => Ok(ColorPolicy::TonemapToSdr),
        "hdr10" => Ok(ColorPolicy::Hdr10),
        "hlg" => Ok(ColorPolicy::Hlg),
        "passthrough" => Ok(ColorPolicy::Passthrough),
        o => bail!("color must be sdr|hdr10|hlg|passthrough, got '{o}'"),
    }
}

/// The `chroma-downsample` vocabulary: `box` (default) | `lanczos`. The
/// words are owned by `codec::colorspace::ChromaDownsample::parse`.
pub fn parse_chroma_downsample(s: &str) -> Result<codec::colorspace::ChromaDownsample> {
    codec::colorspace::ChromaDownsample::parse(s)
}

pub fn parse_bit_depth(s: &str) -> Result<BitDepth> {
    match s {
        "auto" => Ok(BitDepth::Auto),
        "8bit" => Ok(BitDepth::EightBit),
        "10bit" => Ok(BitDepth::TenBit),
        o => bail!("bit-depth must be auto|8bit|10bit, got '{o}'"),
    }
}

/// What a `seam` value asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeamValue {
    /// A seam-quality mode for the multi-GPU single-file path.
    Mode(ChunkSeamMode),
    /// The legacy `serial`: not a seam mode but the encode plan it always
    /// was — one encoder per rung, `encode=single`.
    EncodeSingle,
}

/// The `seam` vocabulary: `parallel`, `constqp`, and the legacy `serial`.
pub fn parse_seam(s: &str) -> Result<SeamValue> {
    match s.trim().to_ascii_lowercase().as_str() {
        "parallel" => Ok(SeamValue::Mode(ChunkSeamMode::Parallel)),
        "constqp" | "const-qp" | "constant-qp" => Ok(SeamValue::Mode(ChunkSeamMode::ParallelConstQp)),
        "serial" => Ok(SeamValue::EncodeSingle),
        o => bail!("seam must be parallel|constqp (or the legacy serial = encode single), got '{o}'"),
    }
}

/// The `target` vocabulary — the perceptual quality target: `visually_lossless`
/// (or `lossless`), `high`, `standard`, `low`, or `vmaf=N`. The same words the
/// policy grammar's `target=` takes; one parser for both.
pub fn parse_quality_target(s: &str) -> Result<codec::encode::tuning::QualityTarget> {
    s.parse().map_err(anyhow::Error::msg).context("target")
}

/// The `decode` vocabulary — the whole decode plan as one value: `auto`,
/// `whole`, `fastest`, `gpu:N`, `ranges:N`, or a bare GPU index (the older
/// `decode-gpu` spelling). See [`DecodePolicy`].
pub fn parse_decode_plan(s: &str) -> Result<DecodePolicy> {
    s.parse().map_err(anyhow::Error::msg).context("decode")
}

/// The `encode` vocabulary — the whole encode plan as one value: `all`,
/// `per-rung`, `single`, `gpu:N`, `family:nvidia|amd|intel` (and `serial`, the
/// older spelling of `single`). See [`EncodePolicy`].
pub fn parse_encode_plan(s: &str) -> Result<EncodePolicy> {
    s.parse().map_err(anyhow::Error::msg).context("encode")
}

pub fn parse_video_codec(s: &str) -> Result<crate::spec::VideoCodecPolicy> {
    use crate::spec::VideoCodecPolicy;
    match s.to_ascii_lowercase().as_str() {
        "av1" | "av01" => Ok(VideoCodecPolicy::Av1),
        "h264" | "avc" | "avc1" | "x264" => Ok(VideoCodecPolicy::H264),
        "h265" | "hevc" | "hvc1" | "x265" => Ok(VideoCodecPolicy::H265),
        o => bail!("codec must be av1|h264|h265, got '{o}'"),
    }
}

pub fn parse_gpu_family(s: &str) -> Result<GpuFamily> {
    match s {
        "nvidia" => Ok(GpuFamily::Nvidia),
        "amd" => Ok(GpuFamily::Amd),
        "intel" => Ok(GpuFamily::Intel),
        o => bail!("gpu-family must be nvidia|amd|intel, got '{o}'"),
    }
}

/// One explicit rung as the surfaces spell it: `WxH`, or `WxH@RATE` for a
/// rung coded to a bitrate (`1280x720@3M`), and fitted its own way
/// (`1080x1920:cover:fixed`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RungArg {
    /// The box's width.
    pub width: u32,
    /// The box's height.
    pub height: u32,
    /// The rung's own bitrate, bits per second, from `@RATE`.
    pub bitrate: Option<u32>,
    /// The rung's own fit, from a `:contain` / `:cover` / `:pad` / `:stretch`.
    pub fit: Option<crate::fit::Fit>,
    /// The rung's own orientation, from `:auto` / `:fixed`.
    pub orientation: Option<crate::fit::Orientation>,
    /// The rung's own upscale, from `:upscale` / `:no-upscale`.
    pub upscale: Option<bool>,
}

impl From<(u32, u32)> for RungArg {
    fn from((width, height): (u32, u32)) -> Self {
        Self { width, height, bitrate: None, fit: None, orientation: None, upscale: None }
    }
}

/// Split a rung's `@RATE` off: the `WxH` part and the rate, read by
/// [`parse_bitrate`]. Every surface's rung reader goes through this, so
/// `@RATE` means the same thing on the CLI, the API, the manifest and the
/// IPC header.
pub fn split_rung_rate(s: &str) -> Result<(&str, Option<u32>)> {
    match s.split_once('@') {
        None => Ok((s, None)),
        Some((size, rate)) => {
            let bps = parse_bitrate(rate).with_context(|| format!("rung '{s}': the rate after `@`"))?;
            Ok((size, Some(bps)))
        }
    }
}

/// Parse a `WxH` rung, e.g. `1280x720`, or `WxH@RATE` (`1280x720@3M`) for a
/// rung coded to that bitrate, followed by any of the rung's own fitting
/// words, each after a `:` — a fit (`contain`, `cover`, `pad`, `stretch`), an
/// orientation (`auto`, `fixed`), `upscale` or `no-upscale`:
/// `1080x1920:cover:fixed`, `1280x720@3M:pad`.
pub fn parse_rung(s: &str) -> Result<RungArg> {
    let mut parts = s.split(':');
    let head = parts.next().unwrap_or_default();
    let (size, bitrate) = split_rung_rate(head)?;
    let (w, h) = size.split_once(['x', 'X']).with_context(|| {
        format!("rung must be WxH or WxH@RATE, e.g. 1280x720, 1280x720@3M or 1080x1920:cover (got '{s}')")
    })?;
    let mut rung = RungArg {
        width: w.trim().parse().context("rung width")?,
        height: h.trim().parse().context("rung height")?,
        ..RungArg::from((0, 0))
    };
    rung.bitrate = bitrate;
    for word in parts.map(|w| w.trim().to_ascii_lowercase()) {
        match word.as_str() {
            "upscale" => rung.upscale = Some(true),
            "no-upscale" | "noupscale" => rung.upscale = Some(false),
            "auto" | "fixed" => rung.orientation = Some(crate::fit::Orientation::parse(&word)?),
            other => {
                rung.fit = Some(crate::fit::Fit::parse(other).with_context(|| {
                    format!(
                        "rung '{s}': `{other}` is not a rung setting (a fit — contain, cover, pad, stretch —                          an orientation — auto, fixed — or upscale / no-upscale)"
                    )
                })?)
            }
        }
    }
    Ok(rung)
}

fn parse_bool(s: &str) -> bool {
    matches!(s.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on" | "y" | "t")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_single_source_resolution() {
        let spec = TranscodeSettings::default().into_spec(1280, 720).unwrap();
        assert!(matches!(spec.mode, crate::spec::OutputMode::SingleFile));
        assert_eq!(spec.rungs.len(), 1);
        assert_eq!((spec.rungs[0].width, spec.rungs[0].height), (1280, 720));
    }

    #[test]
    fn target_and_gop_reach_every_rung_from_any_surface() {
        // `target=vmaf=93 gop=48` as the IPC socket / API / manifest would say
        // it, resolved by the one vocabulary into every rung's Quality.
        let s = TranscodeSettings::parse_kv_line("mode=hls rung=1280x720,640x360 target=vmaf=93 gop=48").unwrap();
        let spec = s.into_spec(1280, 720).unwrap().with_rung_policy_resolved();
        for r in &spec.rungs {
            assert_eq!(r.quality.target, codec::encode::tuning::QualityTarget::Vmaf(93));
            assert_eq!(r.quality.keyframe_interval, Some(48));
            assert_eq!(r.quality.overrides.keyframe_interval, Some(48));
        }
        assert_eq!(spec.gop_frames(30.0), 48);
        // The plain words parse too, and a bad one names itself.
        assert!(parse_quality_target("high").is_ok());
        assert!(parse_quality_target("lossless").is_ok());
        assert!(parse_quality_target("vmaf:88").is_ok());
        assert!(parse_quality_target("shiny").is_err());
        // No gop ⇒ two seconds at the output rate.
        let plain = TranscodeSettings::default().into_spec(1280, 720).unwrap();
        assert_eq!(plain.gop_frames(30.0), 60);
    }

    #[test]
    fn explicit_rungs_and_hls() {
        let s = TranscodeSettings {
            mode: Some(Mode::Hls),
            rungs: vec![(1920, 1080).into(), (1280, 720).into(), (640, 360).into()],
            segment_seconds: Some(6.0),
            crf: Some(28),
            ..Default::default()
        };
        let spec = s.into_spec(1920, 1080).unwrap();
        assert!(matches!(spec.mode, crate::spec::OutputMode::Hls { .. }));
        assert_eq!(spec.rungs.len(), 3);
        assert_eq!(spec.rungs[1].quality.crf, Some(28));
    }

    #[test]
    fn width_height_scales_single_rung() {
        let s = TranscodeSettings {
            width: Some(640),
            height: Some(360),
            ..Default::default()
        };
        let spec = s.into_spec(1280, 720).unwrap();
        assert_eq!((spec.rungs[0].width, spec.rungs[0].height), (640, 360));
    }

    #[test]
    fn kv_line_parses_all_common_keys() {
        let s = TranscodeSettings::parse_kv_line(
            "mode=hls rung=1280x720,640x360 crf=30 audio=opus gpu=1 max-fps=30",
        )
        .unwrap();
        assert_eq!(s.mode, Some(Mode::Hls));
        assert_eq!(s.rungs, vec![(1280, 720).into(), (640, 360).into()]);
        assert_eq!(s.crf, Some(30));
        assert_eq!(s.audio, Some(AudioCodecPolicy::ForceOpus));
        assert_eq!(s.gpu, Some(1));
        assert_eq!(s.max_fps, Some(30.0));
    }

    #[test]
    fn bitrate_accepts_the_ffmpeg_spellings() {
        assert_eq!(parse_bitrate("240k").unwrap(), 240_000);
        assert_eq!(parse_bitrate("240K").unwrap(), 240_000);
        assert_eq!(parse_bitrate("240000").unwrap(), 240_000);
        assert_eq!(parse_bitrate("1.5M").unwrap(), 1_500_000);
        assert!(parse_bitrate("0").is_err());
        assert!(parse_bitrate("-96k").is_err());
        assert!(parse_bitrate("loud").is_err());
    }

    #[test]
    fn kv_carries_the_audio_knobs() {
        let s = TranscodeSettings::parse_kv_line(
            "audio=opus audio-bitrate=240k audio-filter=channelmap=FL-FL|FR-FR:stereo",
        )
        .unwrap();
        assert_eq!(s.audio_bitrate, Some(240_000));
        assert_eq!(s.audio_filters.len(), 1);
        // …and they reach the spec.
        let spec = s.into_spec(1280, 720).unwrap();
        assert_eq!(spec.audio_bitrate, Some(240_000));
        assert_eq!(
            codec::audio::filter::chain_to_string(&spec.audio_filters),
            "channelmap=FL-FL|FR-FR:stereo"
        );
    }

    #[test]
    fn audio_knobs_conflict_with_dropping_audio() {
        // Silently ignoring a filter the user asked for is worse than refusing.
        let s = TranscodeSettings {
            audio: Some(AudioCodecPolicy::Drop),
            audio_filters: codec::audio::filter::parse_chain("channelmap=FL-FL:mono").unwrap(),
            ..Default::default()
        };
        assert!(s.into_spec(1280, 720).is_err());

        let s = TranscodeSettings {
            audio: Some(AudioCodecPolicy::Drop),
            audio_bitrate: Some(240_000),
            ..Default::default()
        };
        assert!(s.into_spec(1280, 720).is_err());
    }

    #[test]
    fn mp3_channels_and_the_audio_mode_are_in_the_vocabulary() {
        let s = TranscodeSettings::parse_kv_line("audio=mp3 audio-channels=stereo audio-bitrate=192k").unwrap();
        assert_eq!(s.audio, Some(AudioCodecPolicy::ForceMp3));
        assert_eq!(s.audio_channels, Some(AudioChannels::Stereo));
        for (word, want) in [
            ("source", AudioChannels::Source),
            ("mono", AudioChannels::Mono),
            ("2", AudioChannels::Stereo),
            ("5.1", AudioChannels::Surround51),
            ("7.1", AudioChannels::Surround71),
        ] {
            assert_eq!(parse_audio_channels(word).unwrap(), want, "{word}");
        }
        assert!(parse_audio_channels("quad").is_err(), "a layout rivet does not produce");
        let hls = TranscodeSettings::parse_kv_line("mode=hls audio-channels=5.1 audio-stereo-fallback=true")
            .unwrap()
            .into_spec(1280, 720)
            .unwrap();
        assert_eq!((hls.audio_channels, hls.audio_stereo_fallback), (AudioChannels::Surround51, true));

        let audio = TranscodeSettings::parse_kv_line("mode=audio").unwrap().into_spec(0, 0).unwrap();
        assert_eq!((audio.mode, audio.rungs.len()), (crate::spec::OutputMode::AudioOnly, 0));
        let err = TranscodeSettings::parse_kv_line("mode=audio crf=28").unwrap().into_spec(0, 0).unwrap_err();
        assert!(format!("{err:#}").contains("writes no video"), "{err:#}");
    }

    #[test]
    fn he_aac_is_in_the_vocabulary() {
        for (word, want) in
            [("auto", HeAacPolicy::Auto), ("passthrough", HeAacPolicy::Passthrough), ("core", HeAacPolicy::Core)]
        {
            assert_eq!(parse_he_aac(word).unwrap(), want, "{word}");
            assert_eq!(want.as_str(), word);
            let spec = TranscodeSettings::parse_kv_line(&format!("he-aac={word}")).unwrap().into_spec(1280, 720).unwrap();
            assert_eq!(spec.he_aac, want);
        }
        assert!(parse_he_aac("sbr").is_err());
        let spec = TranscodeSettings::parse_kv_line("audio=opus").unwrap().into_spec(1280, 720).unwrap();
        assert_eq!(spec.he_aac, HeAacPolicy::Auto, "the default");
        let audio = TranscodeSettings::parse_kv_line("mode=audio he-aac=core").unwrap().into_spec(0, 0).unwrap();
        assert_eq!(audio.he_aac, HeAacPolicy::Core, "the audio-only path keeps it too");
    }

    #[test]
    fn audio_decode_deny_is_in_the_vocabulary() {
        let deny = parse_audio_decode_deny("aac").unwrap();
        assert_eq!(deny.as_string(), "aac");
        assert!(deny.denies("aac") && deny.denies("mp4a") && !deny.denies("opus"));
        let deny = parse_audio_decode_deny(" mp3 , aac,pcm ").unwrap();
        assert_eq!(deny.as_string(), "aac,mp3,pcm", "the names, in canonical order");
        assert!(deny.denies("pcm_s16le") && deny.denies("mp3") && !deny.denies("mp2"));
        for word in ["", "none", ","] {
            assert!(parse_audio_decode_deny(word).unwrap().is_empty(), "'{word}' denies nothing");
        }
        for name in AudioDecodeDeny::CODECS {
            assert_eq!(parse_audio_decode_deny(name).unwrap().as_string(), name);
        }
        let err = parse_audio_decode_deny("aac,wma").unwrap_err();
        assert!(format!("{err:#}").contains("got 'wma'"), "{err:#}");
        assert!(parse_audio_decode_deny("AAC").is_err(), "names are lower case, as every other word");
        assert!(TranscodeSettings::parse_kv_line("audio-decode-deny=aac,bogus").is_err());

        let s = TranscodeSettings::parse_kv_line("audio-decode-deny=aac").unwrap();
        assert!(!s.is_empty());
        let spec = s.into_spec(1280, 720).unwrap();
        assert_eq!(spec.audio_decode_deny.as_string(), "aac");
        let spec = TranscodeSettings::parse_kv_line("audio=opus").unwrap().into_spec(1280, 720).unwrap();
        assert!(spec.audio_decode_deny.is_empty(), "the default denies nothing");
        let audio = TranscodeSettings::parse_kv_line("mode=audio audio-decode-deny=aac,opus")
            .unwrap()
            .into_spec(0, 0)
            .unwrap();
        assert_eq!(audio.audio_decode_deny.as_string(), "aac,opus", "the audio-only path keeps it too");
    }

    /// Every refusal the audio knobs have, at the spec, before any work.
    #[test]
    fn audio_knobs_that_cannot_be_honoured_are_refused_up_front() {
        let refused = |line: &str, needle: &str| {
            let err = TranscodeSettings::parse_kv_line(line).unwrap().into_spec(1280, 720).unwrap_err();
            assert!(format!("{err:#}").contains(needle), "{line}: {err:#}");
        };
        refused("mode=hls audio=mp3", "not available for HLS");
        refused("audio=opus audio-channels=5.1 mode=audio", "cannot hold Opus");
        refused("audio=drop mode=audio", "nothing to write");
        refused("audio=drop audio-channels=stereo", "audio policy is `drop`");
        refused("audio-stereo-fallback=true", "for HLS output");
        refused("mode=hls audio-channels=stereo audio-stereo-fallback=true", "nothing to fall back from");
        if codec::audio::MP3_ENCODE_BUILT {
            refused("audio=mp3 audio-channels=5.1", "two channels at most");
            refused("audio=mp3 audio-bitrate=100k", "not an MP3 bitrate");
            TranscodeSettings::parse_kv_line("audio=mp3 audio-bitrate=320k").unwrap().into_spec(1280, 720).unwrap();
        } else {
            refused("audio=mp3", "`lame` feature");
        }
        // Opus bitrates stay free-form.
        TranscodeSettings::parse_kv_line("audio=opus audio-bitrate=100k").unwrap().into_spec(1280, 720).unwrap();
    }

    #[test]
    fn absurd_audio_bitrates_are_rejected() {
        let s = TranscodeSettings { audio_bitrate: Some(240_000_000), ..Default::default() };
        assert!(s.into_spec(1280, 720).is_err());
        let s = TranscodeSettings { audio_bitrate: Some(1), ..Default::default() };
        assert!(s.into_spec(1280, 720).is_err());
    }

    #[test]
    fn chroma_downsample_reaches_the_spec_and_defaults_to_box() {
        use codec::colorspace::ChromaDownsample;
        let plain = TranscodeSettings::default().into_spec(1280, 720).unwrap();
        assert_eq!(plain.chroma_downsample, ChromaDownsample::Box);
        let s = TranscodeSettings::parse_kv_line("chroma-downsample=lanczos").unwrap();
        assert_eq!(s.chroma_downsample, Some(ChromaDownsample::Lanczos));
        assert_eq!(s.into_spec(1280, 720).unwrap().chroma_downsample, ChromaDownsample::Lanczos);
        let s = TranscodeSettings::parse_kv_line("chroma-filter=box").unwrap();
        assert_eq!(s.chroma_downsample, Some(ChromaDownsample::Box));
        assert!(TranscodeSettings::parse_kv_line("chroma-downsample=bicubic").is_err());
        assert!(parse_chroma_downsample("lanczos").is_ok());
    }

    #[test]
    fn fit_orientation_and_upscale_reach_the_spec_and_the_rungs() {
        use crate::fit::{Fit, Orientation};
        let s = TranscodeSettings::parse_kv_line(
            "codec=h264 rungs=1920x1080,1080x1920@4M:cover:fixed:upscale,640x360:no-upscale fit=pad orientation=fixed upscale=1",
        )
        .unwrap();
        assert_eq!((s.fit, s.orientation, s.upscale), (Some(Fit::Pad), Some(Orientation::Fixed), true));
        assert_eq!(
            s.rungs[1],
            RungArg {
                bitrate: Some(4_000_000),
                fit: Some(Fit::Cover),
                orientation: Some(Orientation::Fixed),
                upscale: Some(true),
                ..(1080, 1920).into()
            }
        );
        assert_eq!(s.rungs[2].upscale, Some(false));
        let spec = s.into_spec(1920, 1080).unwrap();
        assert_eq!((spec.fit, spec.orientation, spec.upscale), (Fit::Pad, Orientation::Fixed, true));
        assert_eq!((spec.rungs[0].fit, spec.rungs[1].fit), (None, Some(Fit::Cover)));
        assert_eq!(spec.rungs[1].quality.overrides.bitrate, Some(4_000_000));

        // Absent, the defaults: contain, auto, no upscale.
        let spec = TranscodeSettings::parse_kv_line("rungs=1280x720").unwrap().into_spec(1920, 1080).unwrap();
        assert_eq!((spec.fit, spec.orientation, spec.upscale), (Fit::Contain, Orientation::Auto, false));
    }

    #[test]
    fn fit_words_are_checked() {
        assert!(TranscodeSettings::parse_kv_line("fit=squash").is_err());
        assert!(TranscodeSettings::parse_kv_line("orientation=sideways").is_err());
        let e = parse_rung("1280x720:zoom").unwrap_err();
        assert!(format!("{e:#}").contains("not a rung setting"), "{e:#}");
        for fit in crate::fit::Fit::ALL {
            assert_eq!(crate::fit::Fit::parse(fit.as_str()).unwrap(), fit);
        }
        // mode=audio refuses them by name, as it does every video knob.
        let e = TranscodeSettings::parse_kv_line("mode=audio fit=cover").unwrap().into_spec(0, 0).unwrap_err();
        assert!(format!("{e:#}").contains("fit"), "{e:#}");
    }

    #[test]
    fn kv_rejects_unknown_key() {
        assert!(TranscodeSettings::parse_kv_line("bogus=1").is_err());
        assert!(TranscodeSettings::parse_kv_line("crf=notanumber").is_err());
    }

    #[test]
    fn speed_is_refused_with_a_reason() {
        // The knob was removed rather than reinterpreted, so an old `speed=6`
        // should explain itself instead of reading as a typo.
        let err = TranscodeSettings::parse_kv_line("speed=6").unwrap_err().to_string();
        assert!(err.contains("tuning tables"), "unhelpful error: {err}");
        assert!(err.contains("crf"), "should point at the knob that remains: {err}");
    }

    #[test]
    fn parsers_reject_garbage() {
        assert!(parse_color("ultrahd").is_err());
        assert!(parse_rung("notarung").is_err());
        assert!(parse_rung("1280x720").is_ok());
        assert!(parse_rung("1280x720@").is_err());
        assert!(parse_rung("1280x720@fast").is_err());
        assert!(parse_buffer("1000").is_err(), "a buffer needs a unit");
    }

    /// `WxH@RATE`, `video-bitrate` and `video-buffer` as every surface writes
    /// them, and who wins: a rung's own `@RATE` over the encode policy over
    /// `video-bitrate`. A rung with nothing named stays a quality target.
    #[test]
    fn video_rates_reach_the_rungs_with_the_rung_winning() {
        let s = TranscodeSettings::parse_kv_line(
            "codec=h264 rung=1920x1080@5M,1280x720,640x360 video-bitrate=1.5M video-buffer=1s \
             encode-policy=step=1:bitrate=3M",
        )
        .unwrap();
        assert_eq!(s.rungs[0], RungArg { bitrate: Some(5_000_000), ..(1920, 1080).into() });
        assert_eq!((s.video_bitrate, s.video_buffer_ms), (Some(1_500_000), Some(1000)));
        let spec = s.into_spec(1920, 1080).unwrap().with_rung_policy_resolved();
        let rates: Vec<_> = spec.rungs.iter().map(|r| (r.quality.overrides.bitrate, r.quality.overrides.buffer_ms)).collect();
        assert_eq!(
            rates,
            vec![(Some(5_000_000), Some(1000)), (Some(3_000_000), Some(1000)), (Some(1_500_000), Some(1000))]
        );
        // The policy's own every-rung set wins over `video-bitrate` too.
        let s = TranscodeSettings::parse_kv_line("codec=h264 rung=1280x720 video-bitrate=1.5M encode-policy=any:bitrate=2M")
            .unwrap();
        let spec = s.into_spec(1280, 720).unwrap().with_rung_policy_resolved();
        assert_eq!(spec.rungs[0].quality.overrides.bitrate, Some(2_000_000));
        // Nothing named: no rate anywhere, the spec as it always was.
        let plain = TranscodeSettings::parse_kv_line("codec=h264 rung=1280x720").unwrap();
        let spec = plain.into_spec(1280, 720).unwrap();
        assert!(spec.rung_policy.global.is_empty() && spec.rung_policy.rules.is_empty());
        assert_eq!(spec.with_rung_policy_resolved().rungs[0].quality.overrides, Default::default());
        // `vb` is the short key, as `ab` is for audio; a buffer needs a unit.
        assert_eq!(TranscodeSettings::parse_kv_line("vb=800k").unwrap().video_bitrate, Some(800_000));
        assert!(TranscodeSettings::parse_kv_line("video-buffer=1000").is_err());
    }

    /// `rate-mode` reaches every rung beneath the policy, in both spellings;
    /// a policy `rate=` rule wins; with no rate named, a constant-rate rung
    /// takes `video-bitrate`, else the default for its codec, size and frame
    /// rate once the frame rate is known.
    #[test]
    fn a_rate_mode_reaches_the_rungs_and_defaults_their_rate() {
        use codec::encode::tuning::{RateMode, default_cbr_bitrate};
        for (word, mode) in [("cbr", RateMode::Constant), ("constant", RateMode::Constant), ("average", RateMode::Average), ("abr", RateMode::Average)] {
            assert_eq!(TranscodeSettings::parse_kv_line(&format!("rate-mode={word}")).unwrap().rate_mode, Some(mode));
        }
        let err = TranscodeSettings::parse_kv_line("rate-mode=vbr").unwrap_err();
        assert!(format!("{err:#}").contains("cbr|constant|average|abr"), "{err:#}");

        let s = TranscodeSettings::parse_kv_line("codec=h264 rung=1920x1080@6M,1280x720,640x360 rate-mode=cbr encode-policy=short<=360:rate=abr,bitrate=500k").unwrap();
        let spec = s.into_spec(1920, 1080).unwrap().with_constant_rates_resolved(60.0);
        let got: Vec<_> = spec.rungs.iter().map(|r| (r.quality.overrides.rate_mode, r.quality.overrides.bitrate)).collect();
        assert_eq!(
            got,
            vec![
                (Some(RateMode::Constant), Some(6_000_000)),
                (Some(RateMode::Constant), Some(default_cbr_bitrate(codec::frame::VideoCodec::H264, 720, 60.0))),
                (Some(RateMode::Average), Some(500_000)),
            ]
        );
        assert_eq!(got[1].1, Some(4_500_000), "720p60 H.264: 3 Mb/s x 1.5");
        // `video-bitrate` is the rate of every rung without its own.
        let s = TranscodeSettings::parse_kv_line("codec=av1 rung=1280x720 rate-mode=cbr video-bitrate=2M").unwrap();
        let spec = s.into_spec(1280, 720).unwrap().with_constant_rates_resolved(30.0);
        assert_eq!(spec.rungs[0].quality.overrides.bitrate, Some(2_000_000));
        // A ladder rung takes the default too; AV1 is half of H.264.
        let s = TranscodeSettings::parse_kv_line("codec=av1 ladder=true max-short-side=1080 rate-mode=cbr").unwrap();
        let spec = s.into_spec(1920, 1080).unwrap().with_constant_rates_resolved(30.0);
        assert_eq!(spec.rungs[0].quality.overrides.bitrate, Some(2_500_000));
        assert!(spec.rungs.iter().all(|r| r.quality.overrides.bitrate.is_some()));
    }

    /// A constant-rate request that cannot be coded is refused while the
    /// spec is built, in the knob's own words.
    #[test]
    fn an_impossible_constant_rate_is_refused_by_name() {
        for (line, words) in [
            ("codec=h264 rung=1280x720 rate-mode=cbr crf=23", &["crf=23", "rate=cbr"][..]),
            ("codec=h264 rung=1280x720 rate-mode=cbr video-buffer=0", &["buffer=0", "rate=cbr"][..]),
            ("mode=single codec=h264 rung=1280x720 rate-mode=cbr seam=constqp", &["constqp", "rate=cbr"][..]),
        ] {
            let err = TranscodeSettings::parse_kv_line(line).unwrap().into_spec(1280, 720).unwrap_err();
            let msg = format!("{err:#}");
            assert!(words.iter().all(|w| msg.contains(w)), "{line}: {msg}");
        }
        // AV1 at a constant rate is a job for the cards, not refused here.
        assert!(TranscodeSettings::parse_kv_line("codec=av1 rung=1280x720 rate-mode=cbr").unwrap().into_spec(1280, 720).is_ok());
    }
}
