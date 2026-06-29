//! One canonical definition of the transcode "knobs", shared by every
//! front-end — the CLI (`transcode` / `pipe`), the HTTP API, and the IPC
//! socket. Each surface parses its own syntax (clap flags / JSON / query
//! string / `key=value`) into a [`TranscodeSettings`], then calls
//! [`TranscodeSettings::into_spec`]. Add a new option **once** here (a field +
//! a line in `into_spec` + a `parse_*` arm) and every surface picks it up,
//! instead of maintaining three copies of the spec-building logic.

use anyhow::{Context, Result, bail};

use crate::spec::{
    AudioCodecPolicy, BitDepth, ChunkSeamMode, ColorPolicy, DecodePolicy, EncodePolicy, GpuFamily,
    OutputSpec, Quality, Rung,
};

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
    pub speed: Option<u8>,
    pub audio: Option<AudioCodecPolicy>,
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
    /// How the decode pump picks its GPU: `Auto` (follow the encode policy),
    /// `SpecificGpu(i)`, or `FastestGpu` (benchmark up front). See [`DecodePolicy`].
    pub decode_policy: DecodePolicy,
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
        let quality = Quality {
            crf: self.crf,
            speed_preset: self.speed,
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

        // GPU policy precedence: pinned index > vendor family > single > all.
        spec = if let Some(idx) = self.gpu {
            spec.encode_policy(EncodePolicy::SingleGpu(Some(idx)))
        } else if let Some(fam) = self.gpu_family {
            spec.encode_policy(EncodePolicy::Family(fam))
        } else if self.single_gpu {
            spec.encode_policy(EncodePolicy::SingleGpu(None))
        } else {
            spec.encode_policy(EncodePolicy::AllGpus)
        };
        spec = spec.decode_policy(self.decode_policy);
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
            "speed" => self.speed = Some(val.parse().context("speed")?),
            "audio" => self.audio = Some(parse_audio(val)?),
            "color" => self.color = Some(parse_color(val)?),
            "bit-depth" | "pixel-format" => self.bit_depth = Some(parse_bit_depth(val)?),
            "seam" => self.seam = Some(parse_seam(val)?),
            "max-fps" => self.max_fps = Some(val.parse().context("max-fps")?),
            "gpu" => self.gpu = Some(val.parse().context("gpu")?),
            "gpu-family" => self.gpu_family = Some(parse_gpu_family(val)?),
            "single-gpu" => self.single_gpu = parse_bool(val),
            "decode-gpu" => {
                self.decode_policy = val.parse().map_err(anyhow::Error::msg).context("decode-gpu")?
            }
            "width" => self.width = Some(val.parse().context("width")?),
            "height" => self.height = Some(val.parse().context("height")?),
            "filter" => self.filters = codec::filter::parse_chain(val)?,
            "codec" => self.video_codec = Some(parse_video_codec(val)?),
            o => bail!(
                "unknown setting '{o}' (mode/rung/ladder/crf/speed/audio/color/bit-depth/seam/max-fps/gpu/gpu-family/single-gpu/decode-gpu/width/height/filter/codec)"
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

    pub fn is_empty(&self) -> bool {
        self.mode.is_none()
            && self.rungs.is_empty()
            && !self.ladder
            && self.max_short_side.is_none()
            && self.segment_seconds.is_none()
            && self.crf.is_none()
            && self.speed.is_none()
            && self.audio.is_none()
            && self.color.is_none()
            && self.bit_depth.is_none()
            && self.seam.is_none()
            && self.max_fps.is_none()
            && self.gpu.is_none()
            && self.gpu_family.is_none()
            && !self.single_gpu
            && self.decode_policy == DecodePolicy::Auto
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

pub fn parse_seam(s: &str) -> Result<ChunkSeamMode> {
    match s {
        "parallel" => Ok(ChunkSeamMode::Parallel),
        "constqp" => Ok(ChunkSeamMode::ParallelConstQp),
        "serial" => Ok(ChunkSeamMode::Serial),
        o => bail!("seam must be parallel|constqp|serial, got '{o}'"),
    }
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
    fn kv_rejects_unknown_key() {
        assert!(TranscodeSettings::parse_kv_line("bogus=1").is_err());
        assert!(TranscodeSettings::parse_kv_line("crf=notanumber").is_err());
    }

    #[test]
    fn parsers_reject_garbage() {
        assert!(parse_color("ultrahd").is_err());
        assert!(parse_rung("notarung").is_err());
        assert!(parse_rung("1280x720").is_ok());
    }
}
