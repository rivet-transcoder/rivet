//! NVENC AV1 encoder via `shiguredo_nvcodec` (Apache-2.0).
//!
//! Replaces a ~2200-line hand-rolled `nvEncodeAPI` FFI mirror with the
//! upstream-maintained `shiguredo_nvcodec` crate (sister to `shiguredo_vpl`
//! / `shiguredo_amf`). The crate bindgens the NVIDIA Video Codec SDK headers
//! at build time (needs libclang) and dlopens the CUDA + NVENC runtime — no
//! build-time CUDA link. Gated behind the `nvidia` cargo feature; when the
//! feature is off the encoder compiles to a construction-erroring stub
//! (`nvenc_stub.rs`).
//!
//! Input: `Yuv420p` 8-bit (interleaved to NV12) or `Yuv420p10le` 10-bit
//! (interleaved to P010 — 10-bit samples in the high bits of each `u16`) on the
//! way in. 10-bit unlocks HDR output **without the `ffmpeg` feature**: the AV1
//! Main profile is 8/10-bit, NVENC derives the bit depth from the input
//! `BufferFormat`, and the muxer writes the `colr`/`mdcv`/`clli` HDR atoms from
//! the job's `ColorMetadata`.
//!
//! Quality model: by default we map our perceptual `QualityTarget` to a target
//! bitrate via a bits-per-pixel heuristic and run **VBR** — `shiguredo_nvcodec`'s
//! `EncoderConfig` exposes `average_bitrate` + a rate-control *mode* but no
//! constant-QP *value*. When `EncoderConfig.constant_qp` is set (the multi-GPU
//! single-file `ChunkSeamMode::ParallelConstQp` path, to keep chunk seams
//! quality-flat) we select `RateControlMode::ConstQp`; the wrapper then uses the
//! encoder **preset's default QP** (we can't pass a specific QP), so the
//! `QualityTarget`→bitrate mapping is skipped in that mode.
//!
//! Platform note: `shiguredo_nvcodec` compiles on **Linux** (the production /
//! Docker target) but NOT on a **Windows MSVC** host — the MSVC ABI types the
//! SDK's C enums as signed `int` (`i32`) while the crate expects `u32`
//! (`unsigned int`, which is what clang produces under the Linux ABI). Build
//! the `nvidia` feature on Linux. See `Cargo.toml` for the full note.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow, bail};
use bytes::Bytes;

use shiguredo_nvcodec::{
    Av1EncoderConfig, Av1Profile, BufferFormat, CodecConfig, EncodeOptions, EncodedFrame,
    Encoder as NvEncoder, EncoderConfig as NvConfig, Error as NvError, FnEncodeHandler, PictureType,
    Preset, RateControlMode, TuningInfo,
};

use super::tuning::{QualityTarget, SpeedTier};
use super::{EncodedPacket, Encoder, EncoderConfig};
use crate::frame::{PixelFormat, VideoFrame};

type Collector = Arc<Mutex<VecDeque<EncodedPacket>>>;

pub struct NvencEncoder {
    config: EncoderConfig,
    inner: NvEncoder<FnEncodeHandler<u64>>,
    collected: Collector,
    flushed: bool,
    scratch: Vec<u8>,
    width: u32,
    height: u32,
    frame_counter: u64,
}

// The inner encoder drives a worker thread and our collector is an
// Arc<Mutex<_>>; the whole thing is Send. Asserted explicitly to match the
// `Encoder: Send` trait bound (the dispatcher returns `Box<dyn Encoder>`).
unsafe impl Send for NvencEncoder {}

impl NvencEncoder {
    pub fn new(config: EncoderConfig, gpu_index: u32) -> Result<Self> {
        // 8-bit → NV12, 10-bit → semi-planar P010 (NVENC's YUV420_10BIT). The
        // encoded AV1 bit depth follows the input buffer format.
        let buffer_format = match config.pixel_format {
            PixelFormat::Yuv420p => BufferFormat::Nv12,
            PixelFormat::Yuv420p10le => BufferFormat::Yuv420_10bit,
            other => bail!("NVENC encoder expects Yuv420p or Yuv420p10le, got {other:?}"),
        };

        let (framerate_num, framerate_den) = frame_rate_rational(config.frame_rate);
        // Rate control: constant-QP (seam-flat chunked single-file, set by the
        // multi-GPU path under `ChunkSeamMode::ParallelConstQp`) vs VBR with a
        // bitrate derived from the perceptual quality target. The wrapper exposes
        // no QP *value* — in ConstQp it uses the encoder preset's default QP — so
        // we pass no bitrate (it's ignored / rejected in ConstQp mode anyway).
        let (rate_control_mode, average_bitrate) = if config.constant_qp {
            (RateControlMode::ConstQp, None)
        } else {
            let br = target_bitrate(config.target, config.width, config.height, config.frame_rate);
            (RateControlMode::Vbr, Some(br))
        };

        let nv_cfg = NvConfig {
            codec: CodecConfig::Av1(Av1EncoderConfig {
                profile: Some(Av1Profile::Main),
                idr_period: Some(config.keyframe_interval.max(1)),
            }),
            width: config.width,
            height: config.height,
            max_encode_width: None,
            max_encode_height: None,
            framerate_num,
            framerate_den,
            average_bitrate,
            preset: preset_for_tier(config.tier),
            tuning_info: TuningInfo::HIGH_QUALITY,
            rate_control_mode,
            gop_length: Some(config.keyframe_interval.max(1)),
            frame_interval_p: 1, // no B-frames (matches the prior repo policy)
            buffer_format,
            device_id: gpu_index as i32,
        };

        let collected: Collector = Arc::new(Mutex::new(VecDeque::new()));
        let sink = Arc::clone(&collected);
        let handler =
            FnEncodeHandler::new(move |frame: std::result::Result<EncodedFrame<u64>, NvError>| {
                match frame {
                    Ok(f) => {
                        let is_keyframe =
                            matches!(f.picture_type(), PictureType::I | PictureType::Idr);
                        let pts = *f.user_data();
                        sink.lock().unwrap().push_back(EncodedPacket {
                            data: Bytes::copy_from_slice(f.data()),
                            pts,
                            is_keyframe,
                        });
                    }
                    Err(e) => tracing::warn!("NVENC encode callback error: {e:?}"),
                }
            });

        let inner = NvEncoder::new(nv_cfg, handler).map_err(|e| {
            anyhow!("shiguredo_nvcodec::Encoder::new (gpu_index={gpu_index}): {e:?}")
        })?;

        // NV12 is 1 byte/sample (w·h·3/2); P010 is 2 bytes/sample (w·h·3).
        let bytes_per_sample = if matches!(config.pixel_format, PixelFormat::Yuv420p10le) {
            2
        } else {
            1
        };
        let scratch_len =
            (config.width as usize) * (config.height as usize) * 3 / 2 * bytes_per_sample;
        Ok(Self {
            width: config.width,
            height: config.height,
            config,
            inner,
            collected,
            flushed: false,
            scratch: vec![0u8; scratch_len],
            frame_counter: 0,
        })
    }

    /// Interleave a tightly-packed `Yuv420p` frame into NV12 in the scratch
    /// buffer (Y plane verbatim, then interleaved U/V). The wrapper expects a
    /// tightly-packed `width × height × 3 / 2` NV12 buffer.
    fn fill_nv12(&mut self, frame: &VideoFrame) -> Result<()> {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;
        let uv = (w / 2) * (h / 2);
        if frame.data.len() < y_size + 2 * uv {
            bail!(
                "NVENC: Yuv420p frame too small for {w}×{h}: have {} need {}",
                frame.data.len(),
                y_size + 2 * uv
            );
        }
        self.scratch[..y_size].copy_from_slice(&frame.data[..y_size]);
        let u = &frame.data[y_size..y_size + uv];
        let v = &frame.data[y_size + uv..y_size + 2 * uv];
        let dst = &mut self.scratch[y_size..y_size + 2 * uv];
        for i in 0..uv {
            dst[2 * i] = u[i];
            dst[2 * i + 1] = v[i];
        }
        Ok(())
    }

    /// Interleave a `Yuv420p10le` frame into semi-planar **P010** in the scratch
    /// buffer: Y plane then interleaved U/V, each 10-bit sample placed in the
    /// high 10 bits of a little-endian `u16` (`<< 6` — the P010 layout NVENC's
    /// `YUV420_10BIT` expects). The encoded AV1 stays 4:2:0 Main-profile 10-bit,
    /// which is web-safe; the muxer tags it HDR via `colr`/`mdcv`/`clli`.
    fn fill_p010(&mut self, frame: &VideoFrame) -> Result<()> {
        let w = self.width as usize;
        let h = self.height as usize;
        let (cw, ch) = (w / 2, h / 2);
        let y_samples = w * h;
        let c_samples = cw * ch;
        let need = (y_samples + 2 * c_samples) * 2; // Yuv420p10le bytes
        if frame.data.len() < need {
            bail!(
                "NVENC: Yuv420p10le frame too small for {w}×{h}: have {} need {}",
                frame.data.len(),
                need
            );
        }
        let src = &frame.data;
        let rd = |off: usize| u16::from_le_bytes([src[off], src[off + 1]]) << 6;
        let dst = &mut self.scratch;
        for i in 0..y_samples {
            dst[i * 2..i * 2 + 2].copy_from_slice(&rd(i * 2).to_le_bytes());
        }
        let y_bytes = y_samples * 2;
        let (u_off, v_off) = (y_bytes, y_bytes + c_samples * 2);
        for i in 0..c_samples {
            let u = rd(u_off + i * 2);
            let v = rd(v_off + i * 2);
            let d = y_bytes + i * 4;
            dst[d..d + 2].copy_from_slice(&u.to_le_bytes());
            dst[d + 2..d + 4].copy_from_slice(&v.to_le_bytes());
        }
        Ok(())
    }
}

impl Encoder for NvencEncoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        if frame.format != self.config.pixel_format {
            bail!(
                "NVENC encoder configured for {:?} but received {:?}",
                self.config.pixel_format,
                frame.format
            );
        }
        if frame.width != self.width || frame.height != self.height {
            bail!(
                "NVENC encoder fixed at {}×{}, received {}×{} (scale before encode)",
                self.width,
                self.height,
                frame.width,
                frame.height
            );
        }
        match frame.format {
            PixelFormat::Yuv420p => self.fill_nv12(frame)?,
            PixelFormat::Yuv420p10le => self.fill_p010(frame)?,
            other => bail!("NVENC: unexpected frame format {other:?}"),
        }
        let opts = EncodeOptions {
            force_intra: false,
            force_idr: false,
            output_spspps: false,
        };
        self.inner
            .encode(&self.scratch, &opts, self.frame_counter)
            .map_err(|e| anyhow!("shiguredo_nvcodec::Encoder::encode: {e:?}"))?;
        self.frame_counter += 1;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.flushed {
            return Ok(());
        }
        self.flushed = true;
        self.inner
            .flush()
            .map_err(|e| anyhow!("shiguredo_nvcodec::Encoder::flush: {e:?}"))?;
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        Ok(self.collected.lock().unwrap().pop_front())
    }
}

/// Map our perceptual quality target to a target bitrate. `shiguredo_nvcodec`
/// is bitrate-driven, so we estimate bits/pixel/frame by quality target and
/// scale by resolution × frame rate. Clamped to a sane envelope.
fn target_bitrate(target: QualityTarget, width: u32, height: u32, fps: f64) -> u32 {
    let bpp = match target {
        QualityTarget::VisuallyLossless => 0.16,
        QualityTarget::High => 0.10,
        QualityTarget::Standard => 0.07,
        QualityTarget::Low => 0.045,
        QualityTarget::Vmaf(v) => 0.03 + (v as f64 / 100.0) * 0.10,
    };
    let bps = bpp * (width as f64) * (height as f64) * fps.max(1.0);
    (bps as u64).clamp(100_000, 60_000_000) as u32
}

fn preset_for_tier(tier: SpeedTier) -> Preset {
    match tier {
        SpeedTier::Draft => Preset::P5,
        SpeedTier::Standard => Preset::P6,
        SpeedTier::Archive => Preset::P7,
    }
}

fn frame_rate_rational(fps: f64) -> (u32, u32) {
    if (fps.fract()).abs() < 1e-6 {
        (fps.round().max(1.0) as u32, 1)
    } else {
        ((fps * 1000.0).round().max(1.0) as u32, 1000)
    }
}
