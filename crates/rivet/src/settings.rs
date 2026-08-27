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
    AudioCodecPolicy, BitDepth, ChunkSeamMode, ColorPolicy, DecodePolicy, EncodePolicy, GpuFamily,
    OutputSpec, Quality, Rung,
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
}

/// Every optional transcode knob, surface-agnostic. All-`None`/empty is "use the
/// defaults" (source-resolution single file, AV1 + audio passthrough, SDR).
#[derive(Debug, Clone, Default)]
pub struct TranscodeSettings {
    pub mode: Option<Mode>,
    /// Explicit rungs as `(width, height)`. Wins over `ladder` / `width`.
    pub rungs: Vec<(u32, u32)>,
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
    /// Target Opus bitrate in bits per second for transcoded audio. `None` lets
    /// the encoder derive it from the channel layout (64k mono / 96k stereo /
    /// 320k 5.1).
    pub audio_bitrate: Option<u32>,
    /// Audio filter chain (`channelmap`) applied to decoded PCM before the Opus
    /// encoder. String surfaces parse `codec::audio::filter::parse_chain` at the
    /// edge, the same way `filters` does for video.
    pub audio_filters: Vec<codec::audio::filter::AudioFilter>,
    pub color: Option<ColorPolicy>,
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
    /// default), or a policy — [`RungPolicy::recommended`] via the CLI's
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
                .map(|&(w, h)| Rung::new(w, h).with_quality(quality.clone()))
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
        };

        if let Some(a) = self.audio {
            spec.audio = a;
        }
        if let Some(s) = self.subtitles {
            spec.subtitles = s;
        }
        spec.audio_bitrate = self.audio_bitrate;
        spec.audio_filters = self.audio_filters;
        spec.max_frame_rate = self.max_fps;
        if let Some(c) = self.color {
            spec = spec.with_color(c);
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
        if let Some(policy) = self.encode_policy {
            spec = spec.with_rung_policy(policy);
        }
        spec = spec.with_filters(self.filters);
        spec = spec.with_trim(self.trim_start, self.trim_end);
        if let Some(c) = self.video_codec {
            spec = spec.with_video_codec(c);
        }

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
            "audio-filter" | "af" => self.audio_filters = codec::audio::filter::parse_chain(val)?,
            "color" => self.color = Some(parse_color(val)?),
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
                "unknown setting '{o}' (mode/rung/ladder/max-short-side/segment-seconds/crf/\
                 target/gop/audio/audio-bitrate/audio-filter/subtitles/color/bit-depth/seam/\
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
            && !self.ladder
            && self.max_short_side.is_none()
            && self.segment_seconds.is_none()
            && self.crf.is_none()
            && self.target.is_none()
            && self.gop.is_none()
            && self.audio.is_none()
            && self.subtitles.is_none()
            && self.audio_bitrate.is_none()
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
        o => bail!("mode must be single|hls, got '{o}'"),
    }
}

pub fn parse_audio(s: &str) -> Result<AudioCodecPolicy> {
    match s {
        "auto" => Ok(AudioCodecPolicy::Auto),
        "opus" => Ok(AudioCodecPolicy::ForceOpus),
        "drop" => Ok(AudioCodecPolicy::Drop),
        o => bail!("audio must be auto|opus|drop, got '{o}'"),
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
/// ffmpeg — `240k` is 240 000 bps, not 245 760.
pub fn parse_bitrate(s: &str) -> Result<u32> {
    let t = s.trim();
    let (num, scale) = match t.chars().last() {
        Some('k') | Some('K') => (&t[..t.len() - 1], 1_000f64),
        Some('m') | Some('M') => (&t[..t.len() - 1], 1_000_000f64),
        _ => (t, 1f64),
    };
    let v: f64 = num
        .trim()
        .parse()
        .with_context(|| format!("bitrate must be a number with an optional k/M suffix (got '{s}')"))?;
    if !v.is_finite() || v <= 0.0 {
        bail!("bitrate must be positive (got '{s}')");
    }
    let bps = v * scale;
    if bps > u32::MAX as f64 {
        bail!("bitrate '{s}' is too large");
    }
    Ok(bps.round() as u32)
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

/// Parse a `WxH` rung, e.g. `1280x720`.
pub fn parse_rung(s: &str) -> Result<(u32, u32)> {
    let (w, h) = s
        .split_once(['x', 'X'])
        .with_context(|| format!("rung must be WxH, e.g. 1280x720 (got '{s}')"))?;
    Ok((
        w.trim().parse().context("rung width")?,
        h.trim().parse().context("rung height")?,
    ))
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
            rungs: vec![(1920, 1080), (1280, 720), (640, 360)],
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
        assert_eq!(s.rungs, vec![(1280, 720), (640, 360)]);
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
    fn absurd_audio_bitrates_are_rejected() {
        let s = TranscodeSettings { audio_bitrate: Some(240_000_000), ..Default::default() };
        assert!(s.into_spec(1280, 720).is_err());
        let s = TranscodeSettings { audio_bitrate: Some(1), ..Default::default() };
        assert!(s.into_spec(1280, 720).is_err());
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
    }
}
