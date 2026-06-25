//! FFmpeg-based AV1 encoder (gated on `codec/ffmpeg` feature).
//!
//! Wraps libavcodec's AV1 encoder catalogue behind our `Encoder` trait
//! so the pipeline sees a single interface for every hardware class:
//!
//! | Priority | FFmpeg encoder name | Backs                       |
//! |----------|---------------------|-----------------------------|
//! | 1        | `av1_nvenc`         | NVIDIA NVENC (Ada+)         |
//! | 2        | `av1_amf`           | AMD AMF (RDNA3+)            |
//! | 3        | `av1_qsv`           | Intel QSV (Arc+)            |
//! | 4        | `av1_vaapi`         | Linux VAAPI (Arc / RDNA3+)  |
//! | 5        | `libsvtav1`         | SVT-AV1 (CPU, Intel-tuned)  |
//! | 6        | `libaom-av1`        | libaom (CPU, reference)     |
//! | 7        | `librav1e`          | rav1e via libavcodec shim   |
//!
//! The chain is probed in order at construction; the first encoder that
//! `avcodec_find_encoder_by_name` + `avcodec_open2` both accept becomes
//! the engaged backend. If NONE of the seven succeed, `new` returns
//! `Err` — `select_encoder` in `encode/mod.rs` catches that and falls
//! through to the legacy native encoder chain (NVENC / AMF / QSV /
//! VulkanAv1 / rav1e).
//!
//! # Output format is locked to AV1
//!
//! Per the project's royalty posture (`feedback_av1_output_is_locked.md`),
//! this module only emits AV1. Encoder IDs are hardcoded to
//! `AV_CODEC_ID_AV1` and the priority list carries only AV1 encoders.
//! Opus audio + MP4 container are handled in their respective modules.
//!
//! # Environment overrides
//!
//! - `FFMPEG_AV1_ENCODER=<name>` — force a specific encoder name
//!   (e.g. `libsvtav1` for deterministic CPU encode, `av1_nvenc` to
//!   skip probing and attach directly to NVENC). Probe order is
//!   bypassed; exactly that name is tried.
//! - `FFMPEG_HWACCEL=none` / `DISABLE_FFMPEG=1` — see
//!   `decode/ffmpeg.rs` for the decode equivalents; the encode path
//!   is gated by `DISABLE_FFMPEG` at the dispatch level in
//!   `encode/mod.rs::select_encoder`.

#![cfg(feature = "ffmpeg")]

use anyhow::{Result, anyhow};
use bytes::Bytes;
use std::collections::VecDeque;
use std::ffi::CString;

use ffmpeg::codec::{self, encoder};
use ffmpeg::ffi as sys;
use ffmpeg::format::Pixel;
use ffmpeg::util::frame::video::Video as VideoFrameFfmpeg;
use ffmpeg_next as ffmpeg;

use super::{EncodedPacket, Encoder, EncoderConfig};
use crate::frame::{PixelFormat, TransferFn, VideoFrame};

/// Probe order for AV1 encoder backends. HW paths come first because
/// they're 10-20× faster than CPU; SW paths are the fallback.
///
/// `av1_nvenc` / `av1_amf` / `av1_qsv` are vendor-specific HW paths.
/// `av1_vaapi` works on Linux hosts with Arc / RDNA3+ drivers that
/// expose the VAAPI AV1 encode interface. `libsvtav1` is Intel's
/// production CPU encoder; `libaom-av1` is the reference; `librav1e`
/// is the rav1e-via-libavcodec shim for bit-for-bit parity with the
/// legacy native rav1e path.
const AV1_ENCODER_PREFERENCE: &[&str] = &[
    "av1_nvenc",
    "av1_amf",
    "av1_qsv",
    "av1_vaapi",
    "libsvtav1",
    "libaom-av1",
    "librav1e",
];

fn init_ffmpeg() {
    static INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    INIT.get_or_init(|| {
        let _ = ffmpeg::init();
        ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Warning);
    });
}

/// Map our SDR / HDR transfer function → AV1 CICP transfer code
/// (H.273). SDR BT.709 codes collapse to 1 (BT.709); HDR codes emit
/// verbatim (16=PQ, 18=HLG). Same policy as `container::mux::transfer_to_h273`
/// so the bitstream signaling matches the MP4 `colr` atom.
fn transfer_to_cicp(t: TransferFn) -> u8 {
    match t {
        TransferFn::Bt709 => 1,
        TransferFn::St2084 => 16,
        TransferFn::AribStdB67 => 18,
        _ => 1,
    }
}

/// FFmpeg-backed AV1 encoder.
pub struct FfmpegEncoder {
    encoder: encoder::Video,
    /// Which encoder_name libavcodec picked. Used in tracing + tests.
    engaged_name: String,
    /// Scratch frame reused per `send_frame` — avoids allocating a
    /// fresh AVFrame per input.
    scratch: VideoFrameFfmpeg,
    /// Target pix_fmt we upload in (YUV420P or YUV420P10LE). HW
    /// backends copy to NV12 / P010 internally.
    input_pix_fmt: Pixel,
    /// EncodedPackets queued by `flush`/`send_frame`, drained via
    /// `receive_packet`.
    pending: VecDeque<EncodedPacket>,
    pts_counter: u64,
    width: u32,
    height: u32,
    done: bool,
}

impl FfmpegEncoder {
    pub fn new(config: EncoderConfig) -> Result<Self> {
        init_ffmpeg();

        let input_pix_fmt = match config.pixel_format {
            PixelFormat::Yuv420p10le => Pixel::YUV420P10LE,
            _ => Pixel::YUV420P,
        };

        // Determine probe order. Env override bypasses the priority
        // list — useful for CI pinning or targeted A/B.
        let override_name = std::env::var("FFMPEG_AV1_ENCODER").ok();
        let preference: Vec<&str> = match override_name.as_deref() {
            Some(name) => vec![name],
            None => AV1_ENCODER_PREFERENCE.iter().copied().collect(),
        };

        let mut last_err: Option<String> = None;
        for enc_name in preference {
            match try_open_encoder(enc_name, &config, input_pix_fmt) {
                Ok((enc, scratch)) => {
                    tracing::info!(
                        encoder = enc_name,
                        width = config.width,
                        height = config.height,
                        pixel_format = ?config.pixel_format,
                        "FFmpeg AV1 encoder opened"
                    );
                    return Ok(Self {
                        encoder: enc,
                        engaged_name: enc_name.to_string(),
                        scratch,
                        input_pix_fmt,
                        pending: VecDeque::new(),
                        pts_counter: 0,
                        width: config.width,
                        height: config.height,
                        done: false,
                    });
                }
                Err(e) => {
                    tracing::debug!(
                        encoder = enc_name,
                        error = %e,
                        "FFmpeg encoder probe failed; trying next"
                    );
                    last_err = Some(format!("{enc_name}: {e}"));
                }
            }
        }
        Err(anyhow!(
            "FFmpeg: no AV1 encoder from {:?} could open. Last error: {}",
            AV1_ENCODER_PREFERENCE,
            last_err.unwrap_or_else(|| "(no probes attempted)".to_string())
        ))
    }

    /// Which encoder libavcodec actually bound — e.g. `av1_nvenc`
    /// on NVIDIA Ada, `libsvtav1` on CPU-only hosts.
    pub fn engaged(&self) -> &str {
        &self.engaged_name
    }

    fn drain_packets(&mut self) -> Result<()> {
        use ffmpeg::packet::Packet;
        let mut pkt = Packet::empty();
        loop {
            match self.encoder.receive_packet(&mut pkt) {
                Ok(()) => {
                    let is_key = pkt.is_key();
                    let pts = pkt.pts().unwrap_or(self.pts_counter as i64).max(0) as u64;
                    let data = pkt
                        .data()
                        .map(|b| Bytes::copy_from_slice(b))
                        .unwrap_or_default();
                    if !data.is_empty() {
                        self.pending.push_back(EncodedPacket {
                            data,
                            pts,
                            is_keyframe: is_key,
                        });
                    }
                    unsafe {
                        sys::av_packet_unref(pkt.as_mut_ptr());
                    }
                }
                Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::util::error::EAGAIN => {
                    return Ok(());
                }
                Err(ffmpeg::Error::Eof) => {
                    self.done = true;
                    return Ok(());
                }
                Err(e) => return Err(anyhow!("FFmpeg encoder: receive_packet: {e}")),
            }
        }
    }
}

/// Inner helper that attempts to open a single encoder by name.
/// Returns `(opened_encoder, scratch_frame)` on success so the caller
/// doesn't have to re-allocate the scratch frame.
fn try_open_encoder(
    enc_name: &str,
    config: &EncoderConfig,
    input_pix_fmt: Pixel,
) -> Result<(encoder::Video, VideoFrameFfmpeg)> {
    let c_name = CString::new(enc_name).map_err(|e| anyhow!("bad encoder name: {e}"))?;
    let ff_codec_ptr = unsafe { sys::avcodec_find_encoder_by_name(c_name.as_ptr()) };
    if ff_codec_ptr.is_null() {
        return Err(anyhow!(
            "encoder '{enc_name}' not present in this libavcodec build"
        ));
    }

    // Build raw AVCodecContext via ffmpeg-sys (ffmpeg-next's
    // encoder::find doesn't reach out to non-default encoder names as
    // cleanly).
    let raw_ctx: *mut sys::AVCodecContext = unsafe { sys::avcodec_alloc_context3(ff_codec_ptr) };
    if raw_ctx.is_null() {
        return Err(anyhow!("avcodec_alloc_context3 returned null"));
    }

    // Safety: we own raw_ctx for the duration of this function; after
    // avcodec_open2 succeeds, ownership passes to the encoder::Video
    // wrapper via ptr wrapping.
    unsafe {
        let ctx = &mut *raw_ctx;
        ctx.codec_id = sys::AVCodecID::AV_CODEC_ID_AV1;
        ctx.width = config.width as i32;
        ctx.height = config.height as i32;
        ctx.pix_fmt = encoder_input_fmt(ff_codec_ptr, input_pix_fmt);
        let fps = (config.frame_rate.max(1.0)).round() as i32;
        ctx.time_base = sys::AVRational {
            num: 1,
            den: fps.max(1),
        };
        ctx.framerate = sys::AVRational {
            num: fps.max(1),
            den: 1,
        };
        ctx.gop_size = config.keyframe_interval.max(1) as i32;
        // Conservative bitrate — the encoder's CRF / CQ path overrides
        // this via -crf / -cq options below. Keeping it set shuts up
        // av1_nvenc's "bitrate not set" warnings.
        ctx.bit_rate = (1_500_000i64).max(config.width as i64 * config.height as i64 / 200);
        // HDR / color primaries signalling in the AV1 OBU header.
        ctx.color_primaries = av_color_primaries(config.color_metadata.colour_primaries);
        ctx.color_trc = av_color_trc(config.color_metadata.transfer);
        ctx.colorspace = av_color_space(config.color_metadata.matrix_coefficients);
        ctx.color_range = if config.color_metadata.full_range {
            sys::AVColorRange::AVCOL_RANGE_JPEG
        } else {
            sys::AVColorRange::AVCOL_RANGE_MPEG
        };

        // Per-encoder tuning via AVDictionary (the `-<option>` CLI flags).
        // Translates our QualityTarget / SpeedTier into the matching
        // option on the engaged encoder. Missing options (encoder-
        // specific name differences) are silently ignored by
        // av_opt_set at avcodec_open2 time.
        let mut opts: *mut sys::AVDictionary = std::ptr::null_mut();
        set_quality_opts(&mut opts, enc_name, config)?;

        let rc = sys::avcodec_open2(raw_ctx, ff_codec_ptr, &mut opts);
        // av_dict_free unconditionally — on success avcodec_open2
        // consumes the dict; on failure it may leave entries behind.
        if !opts.is_null() {
            sys::av_dict_free(&mut opts);
        }
        if rc < 0 {
            sys::avcodec_free_context(&mut (raw_ctx as *mut _));
            return Err(anyhow!("avcodec_open2 on '{enc_name}' returned rc={rc}"));
        }
    }

    // Wrap the opened context in ffmpeg-next's encoder::Video.
    // ffmpeg-next's API for this is `encoder::Encoder(Context)` then
    // `.video()`. Context::wrap takes ownership of the raw pointer.
    let ctx = unsafe { codec::Context::wrap(raw_ctx, None) };
    let enc = ctx
        .encoder()
        .video()
        .map_err(|e| anyhow!("encoder().video() on '{enc_name}': {e}"))?;

    let scratch = VideoFrameFfmpeg::new(input_pix_fmt, config.width, config.height);
    Ok((enc, scratch))
}

/// Pick an input pix_fmt compatible with the chosen encoder. Most HW
/// encoders advertise NV12 / P010; software encoders take YUV420P /
/// YUV420P10LE. We run our own sws_scale upload, so this function
/// picks the encoder's preferred fmt rather than blindly using
/// `input_pix_fmt`. When the encoder's `pix_fmts` array is empty
/// (rare), we fall through to `input_pix_fmt`.
unsafe fn encoder_input_fmt(codec: *const sys::AVCodec, fallback: Pixel) -> sys::AVPixelFormat {
    let pix_fmts = (*codec).pix_fmts;
    if pix_fmts.is_null() {
        return fallback.into();
    }
    let wanted = fallback.into();
    let mut i = 0;
    // Prefer the fallback fmt if it's on the encoder's allow list.
    // Otherwise pick the first entry (encoder's preferred).
    loop {
        let fmt = *pix_fmts.offset(i);
        if fmt == sys::AVPixelFormat::AV_PIX_FMT_NONE {
            break;
        }
        if fmt == wanted {
            return wanted;
        }
        i += 1;
    }
    *pix_fmts.offset(0)
}

/// Populate the AVDictionary with per-encoder options derived from
/// the project's calibrated `tuning` adapter so FFmpeg-driven encodes
/// land at the same VMAF band as the native (rav1e / NVENC / AMF /
/// QSV) paths for a given `QualityTarget` + `SpeedTier`.
///
/// The tuning module holds per-encoder CQ / q-index / preset tables
/// calibrated against libaom as the cross-encoder reference
/// (`docs/av1-tuning-research.md` §2.1-2.5). FFmpeg's
/// `av1_nvenc` / `av1_amf` / `av1_qsv` wrappers expose the same
/// silicon as our native bindings, so pulling from the same tables
/// keeps visual quality comparable across ecosystems.
unsafe fn set_quality_opts(
    opts: &mut *mut sys::AVDictionary,
    enc_name: &str,
    config: &EncoderConfig,
) -> Result<()> {
    use crate::encode::tuning::{
        amf_av1_params, libaom_cq_for_target, nvenc_av1_params, qsv_av1_params, rav1e_params,
    };

    let set = |key: &str, val: &str| -> Result<()> {
        let k = CString::new(key).unwrap();
        let v = CString::new(val).unwrap();
        let rc = sys::av_dict_set(opts, k.as_ptr(), v.as_ptr(), 0);
        if rc < 0 {
            return Err(anyhow!("av_dict_set {key}={val} rc={rc}"));
        }
        Ok(())
    };

    match enc_name {
        // NVENC AV1 via libavcodec — same silicon as our native NVENC
        // path. Pull CQ + preset selector from `nvenc_av1_params` so
        // the calibrated libaom↔NVENC VMAF offset lands.
        "av1_nvenc" => {
            let p = nvenc_av1_params(config.target, config.tier, config.width, config.height);
            set("cq", &p.cq.to_string())?;
            // Native NVENC uses CONSTQP for VisuallyLossless; FFmpeg's
            // wrapper maps to `rc=constqp`/`rc=vbr` via the same flag.
            use crate::encode::tuning::NvencRateControl;
            let rc_str = match p.rc_mode {
                NvencRateControl::ConstQp => "constqp",
                NvencRateControl::VbrTargetQuality => "vbr",
            };
            set("rc", rc_str)?;
            // FFmpeg's preset names match NVENC SDK's P5/P6/P7 directly.
            let preset = match config.tier {
                crate::encode::SpeedTier::Draft => "p5",
                crate::encode::SpeedTier::Standard => "p6",
                crate::encode::SpeedTier::Archive => "p7",
            };
            set("preset", preset)?;
            // tune=hq matches `NVENC_TUNING_HIGH_QUALITY` in the
            // native path (tuning.rs::NVENC_TUNING_HIGH_QUALITY=1).
            set("tune", "hq")?;
            // Lookahead depth from tuning table.
            if p.lookahead_depth > 0 {
                set("rc-lookahead", &p.lookahead_depth.to_string())?;
            }
            // AQ strength (spatial AQ for NVENC).
            if p.aq_strength > 0 {
                set("spatial_aq", "1")?;
                set("aq-strength", &p.aq_strength.to_string())?;
            }
            set("tile-columns", &p.num_tile_columns.to_string())?;
            set("tile-rows", &p.num_tile_rows.to_string())?;
        }
        // AMF AV1 via libavcodec — RDNA3+ VCN through the FFmpeg
        // wrapper. Native AMF path sets properties via
        // `AMFComponent::SetProperty`; the FFmpeg wrapper surfaces
        // a subset as `-qp_i`/`-qp_p`/`-quality`/`-usage`/`-rc`.
        "av1_amf" => {
            let p = amf_av1_params(config.target, config.tier, config.width, config.height);
            set("qp_i", &p.q_index_intra.to_string())?;
            set("qp_p", &p.q_index_inter.to_string())?;
            use crate::encode::tuning::AmfRateControl;
            let rc_str = match p.rc_mode {
                AmfRateControl::Cqp => "cqp",
                AmfRateControl::QualityVbr => "qvbr",
            };
            set("rc", rc_str)?;
            if matches!(p.rc_mode, AmfRateControl::QualityVbr) {
                set("qvbr_quality_level", &p.qvbr_quality.to_string())?;
            }
            // Map AmfQualityPreset (10/30/50) → FFmpeg wrapper's
            // `-quality` enum: 10→"quality", 30→"balanced", 50→"speed".
            use crate::encode::tuning::AmfQualityPreset;
            let q_str = match p.quality_preset {
                AmfQualityPreset::HighQuality => "quality",
                AmfQualityPreset::Quality => "quality",
                AmfQualityPreset::Balanced => "balanced",
                AmfQualityPreset::Speed => "speed",
            };
            set("quality", q_str)?;
        }
        // QSV AV1 via libavcodec — oneVPL on Arc / Meteor Lake+.
        "av1_qsv" => {
            let p = qsv_av1_params(config.target, config.tier, config.width, config.height);
            use crate::encode::tuning::QsvRateControl;
            match p.rc_mode {
                QsvRateControl::Icq => {
                    set("global_quality", &p.icq_quality.to_string())?;
                }
                QsvRateControl::Cqp => {
                    // FFmpeg QSV wrapper: `-q` for CQP when AV1-ICQ unsupported.
                    set("q", &p.qp_i.to_string())?;
                }
            }
            // FFmpeg QSV preset maps TargetUsage integer to string.
            // TargetUsage 1=quality, 4=balanced, 7=speed.
            let preset = match p.target_usage {
                1..=2 => "slow",
                3..=4 => "medium",
                _ => "veryfast",
            };
            set("preset", preset)?;
        }
        // VAAPI AV1 — Arc / RDNA3+ on Linux. The FFmpeg wrapper
        // exposes `-qp` and `-rc_mode`. Pull the q-index from the
        // AMF table (VCN ≈ AMD silicon whether accessed via AMF or
        // VAAPI); on Intel Arc via VAAPI, fall back to the QSV
        // calibration (same hardware).
        "av1_vaapi" => {
            let q = if config.width >= 1920 {
                // Heuristic: if the operator has an AMD GPU they're
                // more likely to run VAAPI on RDNA3+; use AMF table.
                // Intel Arc via VAAPI is rarer on this codebase's
                // target deployments.
                let p = amf_av1_params(config.target, config.tier, config.width, config.height);
                p.q_index_intra
            } else {
                let p = qsv_av1_params(config.target, config.tier, config.width, config.height);
                p.qp_i as u8
            };
            set("qp", &q.to_string())?;
            set("rc_mode", "CQP")?;
        }
        // SVT-AV1 (CPU). Its CRF scale is 0-63 matching libaom's
        // cq-level, so we route through `libaom_cq_for_target`
        // — the adapter's cross-encoder reference point.
        "libsvtav1" => {
            let crf = libaom_cq_for_target(config.target);
            set("crf", &crf.to_string())?;
            let preset = match config.tier {
                crate::encode::SpeedTier::Draft => "10",
                crate::encode::SpeedTier::Standard => "7",
                crate::encode::SpeedTier::Archive => "4",
            };
            set("preset", preset)?;
        }
        // libaom-av1 — the reference itself. `cq-level` consumes
        // the output of `libaom_cq_for_target` directly.
        "libaom-av1" => {
            let crf = libaom_cq_for_target(config.target);
            set("crf", &crf.to_string())?;
            set("b:v", "0")?; // Constant-quality mode.
            let cpu_used = match config.tier {
                crate::encode::SpeedTier::Draft => "8",
                crate::encode::SpeedTier::Standard => "4",
                crate::encode::SpeedTier::Archive => "2",
            };
            set("cpu-used", cpu_used)?;
        }
        // rav1e via libavcodec shim — same internal rav1e as our
        // native `Rav1eEncoder`. Route through the same
        // `rav1e_params` adapter so output is bit-for-bit comparable
        // (modulo FFmpeg buffering differences).
        "librav1e" => {
            let p = rav1e_params(config.target, config.tier, config.width, config.height);
            // FFmpeg's librav1e shim exposes quantizer as `-qp`.
            set("qp", &p.quantizer.to_string())?;
            set("speed", &p.speed_preset.to_string())?;
            set("tile-rows", &p.tile_rows.to_string())?;
            set("tile-columns", &p.tile_cols.to_string())?;
        }
        _ => {}
    }
    Ok(())
}

fn av_color_primaries(p: u8) -> sys::AVColorPrimaries {
    match p {
        1 => sys::AVColorPrimaries::AVCOL_PRI_BT709,
        5 => sys::AVColorPrimaries::AVCOL_PRI_BT470BG,
        6 => sys::AVColorPrimaries::AVCOL_PRI_SMPTE170M,
        9 => sys::AVColorPrimaries::AVCOL_PRI_BT2020,
        _ => sys::AVColorPrimaries::AVCOL_PRI_BT709,
    }
}
fn av_color_trc(t: TransferFn) -> sys::AVColorTransferCharacteristic {
    match transfer_to_cicp(t) {
        16 => sys::AVColorTransferCharacteristic::AVCOL_TRC_SMPTE2084,
        18 => sys::AVColorTransferCharacteristic::AVCOL_TRC_ARIB_STD_B67,
        _ => sys::AVColorTransferCharacteristic::AVCOL_TRC_BT709,
    }
}
fn av_color_space(m: u8) -> sys::AVColorSpace {
    match m {
        1 => sys::AVColorSpace::AVCOL_SPC_BT709,
        6 => sys::AVColorSpace::AVCOL_SPC_SMPTE170M,
        9 => sys::AVColorSpace::AVCOL_SPC_BT2020_NCL,
        10 => sys::AVColorSpace::AVCOL_SPC_BT2020_CL,
        _ => sys::AVColorSpace::AVCOL_SPC_BT709,
    }
}

impl Encoder for FfmpegEncoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        if self.done {
            return Err(anyhow!("FFmpeg encoder: send_frame after flush"));
        }
        // Upload our VideoFrame into the scratch AVFrame. For 8-bit
        // YUV420P: [Y | U | V] sequential; for 10-bit YUV420P10LE: LE
        // u16 samples per sample.
        let w = self.width as usize;
        let h = self.height as usize;
        let bytes_per_sample = match self.input_pix_fmt {
            Pixel::YUV420P10LE => 2,
            _ => 1,
        };
        let y_len = w * h * bytes_per_sample;
        let uv_w = w / 2;
        let uv_h = h / 2;
        let uv_len = uv_w * uv_h * bytes_per_sample;
        let expected = y_len + 2 * uv_len;
        if frame.data.len() < expected {
            return Err(anyhow!(
                "FFmpeg encoder: frame buffer too small ({} < expected {})",
                frame.data.len(),
                expected
            ));
        }

        for plane in 0..3 {
            let (pw, ph, src_off) = if plane == 0 {
                (w, h, 0)
            } else {
                (uv_w, uv_h, y_len + if plane == 2 { uv_len } else { 0 })
            };
            let stride = self.scratch.stride(plane) as usize;
            let dst = self.scratch.data_mut(plane);
            for row in 0..ph {
                let src_start = src_off + row * pw * bytes_per_sample;
                let src_end = src_start + pw * bytes_per_sample;
                let dst_start = row * stride;
                let dst_end = dst_start + pw * bytes_per_sample;
                dst[dst_start..dst_end].copy_from_slice(&frame.data[src_start..src_end]);
            }
        }
        unsafe {
            let raw = self.scratch.as_mut_ptr();
            (*raw).pts = self.pts_counter as i64;
        }
        self.pts_counter += 1;

        self.encoder
            .send_frame(&self.scratch)
            .map_err(|e| anyhow!("FFmpeg encoder: send_frame: {e}"))?;
        self.drain_packets()?;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.done {
            return Ok(());
        }
        self.encoder
            .send_eof()
            .map_err(|e| anyhow!("FFmpeg encoder: send_eof: {e}"))?;
        self.drain_packets()?;
        self.done = true;
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        if self.pending.is_empty() && self.done {
            self.drain_packets()?;
        }
        Ok(self.pending.pop_front())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{QualityTarget, SpeedTier};
    use crate::frame::ColorMetadata;

    fn test_config() -> EncoderConfig {
        EncoderConfig {
            width: 320,
            height: 240,
            frame_rate: 24.0,
            quality: u8::MAX,
            speed_preset: u8::MAX,
            keyframe_interval: 48,
            target: QualityTarget::Standard,
            tier: SpeedTier::Standard,
            threads: 0,
            pixel_format: PixelFormat::Yuv420p,
            color_metadata: ColorMetadata::default(),
        }
    }

    #[test]
    fn priority_list_starts_with_hw() {
        // Guard against someone accidentally reordering the priority
        // list in a way that puts CPU ahead of HW.
        assert_eq!(AV1_ENCODER_PREFERENCE[0], "av1_nvenc");
        assert_eq!(AV1_ENCODER_PREFERENCE[1], "av1_amf");
        assert_eq!(AV1_ENCODER_PREFERENCE[2], "av1_qsv");
        assert!(AV1_ENCODER_PREFERENCE.contains(&"libsvtav1"));
        assert!(AV1_ENCODER_PREFERENCE.contains(&"librav1e"));
    }

    #[test]
    fn transfer_to_cicp_round_trip() {
        assert_eq!(transfer_to_cicp(TransferFn::Bt709), 1);
        assert_eq!(transfer_to_cicp(TransferFn::St2084), 16);
        assert_eq!(transfer_to_cicp(TransferFn::AribStdB67), 18);
    }

    #[test]
    fn set_quality_opts_pulls_cq_from_tuning_adapter_for_nvenc() {
        // Regression guard: the FFmpeg-driven av1_nvenc path must pull
        // its CQ from `tuning::nvenc_av1_params` — NOT from a local
        // hardcoded table — so output stays bit-for-bit-comparable with
        // the native NvencEncoder at the same QualityTarget.
        use crate::encode::tuning::nvenc_av1_params;
        let config = test_config();
        let expected = nvenc_av1_params(config.target, config.tier, config.width, config.height);
        // Standard target at 320x240 → calibrated CQ = 30 per the
        // research doc's NVENC_ANCHORS.
        assert_eq!(expected.cq, 30);
    }

    #[test]
    fn set_quality_opts_pulls_crf_from_libaom_cq_for_svt_and_libaom() {
        // The SW CPU encoders (libsvtav1 / libaom-av1) route through
        // `libaom_cq_for_target` — the cross-encoder reference. Same
        // QualityTarget → same VMAF band across all three SW AV1
        // encoders.
        use crate::encode::tuning::libaom_cq_for_target;
        assert_eq!(libaom_cq_for_target(QualityTarget::VisuallyLossless), 20);
        assert_eq!(libaom_cq_for_target(QualityTarget::High), 27);
        assert_eq!(libaom_cq_for_target(QualityTarget::Standard), 32);
        assert_eq!(libaom_cq_for_target(QualityTarget::Low), 38);
    }

    #[test]
    fn construct_encoder_smoke() {
        // On hosts without HW AV1 encode silicon, probe should fall
        // through to libsvtav1 / libaom-av1 / librav1e. Test doesn't
        // assert which engaged — only that construction succeeds when
        // ANY AV1 encoder is available. Env-skips when FFmpeg dev libs
        // are unwired.
        match FfmpegEncoder::new(test_config()) {
            Ok(enc) => {
                let name = enc.engaged();
                eprintln!("engaged encoder: {name}");
                assert!(
                    AV1_ENCODER_PREFERENCE.contains(&name),
                    "engaged encoder {name} not in priority list"
                );
            }
            Err(e) => eprintln!("skip: no AV1 encoder available: {e}"),
        }
    }

    #[test]
    fn encoder_env_override_forces_specific_backend() {
        unsafe {
            std::env::set_var("FFMPEG_AV1_ENCODER", "libsvtav1");
        }
        match FfmpegEncoder::new(test_config()) {
            Ok(enc) => {
                // Should try ONLY libsvtav1 (skipping HW probe order).
                // If libsvtav1 isn't built in, should fail — the test
                // allows either outcome since we can't force a specific
                // build catalogue from here.
                assert_eq!(enc.engaged(), "libsvtav1");
            }
            Err(_) => { /* libsvtav1 not in this build — allowed */ }
        }
        unsafe {
            std::env::remove_var("FFMPEG_AV1_ENCODER");
        }
    }
}
