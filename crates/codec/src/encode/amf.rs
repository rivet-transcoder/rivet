//! AMD AMF AV1 encoder via `shiguredo_amf` (Apache-2.0).
//!
//! Replaces a hand-rolled AMF FFI mirror with the upstream-maintained
//! `shiguredo_amf` crate (sister to `shiguredo_vpl` / `shiguredo_nvcodec`).
//! The crate bindgens the AMD AMF SDK headers at build time (needs libclang)
//! and dlopens the AMF runtime. Gated behind the `amd` cargo feature; when the
//! feature is off the encoder compiles to a construction-erroring stub
//! (`amf_stub.rs`).
//!
//! Input: `Yuv420p` frames, copied into an AMF host `Surface` as NV12 (Y plane
//! verbatim, interleaved U/V) honoring the surface's plane pitches.
//!
//! Quality model: AMF exposes per-frame QP (`qpi`/`qpp`/`qpb`), so unlike the
//! bitrate-only NVENC wrapper we run **constant-QP** using the tuning
//! adapter's AV1 q-index values — quality-stable across content.
//!
//! GPU pinning: `shiguredo_amf` selects the default AMF adapter; per-GPU
//! pinning on a multi-AMD host is not exposed by the wrapper (a `gpu_index`
//! other than 0 is accepted but logged).
//!
//! Platform note: `shiguredo_amf` compiles on **Linux** (the production /
//! Docker target) but NOT on a **Windows MSVC** host — the MSVC ABI types the
//! SDK's C enums as signed `int` (`i32`) while the crate expects `u32`. Build
//! the `amd` feature on Linux. See `Cargo.toml` for the full note.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow, bail};
use bytes::Bytes;

use shiguredo_amf::amf::Surface;
use shiguredo_amf::{
    Av1EncoderConfig, Av1Profile, CodecConfig, EncodeOptions, EncodedFrame, Encoder as AmfEnc,
    EncoderConfig as AmfConfig, Error as AmfError, FnEncodeHandler, FrameFormat, PictureType,
    RateControlMode, frame_type,
};

use super::tuning;
use super::{EncodedPacket, Encoder, EncoderConfig};
use crate::frame::{PixelFormat, VideoFrame};

type Collector = Arc<Mutex<VecDeque<EncodedPacket>>>;

pub struct AmfEncoder {
    config: EncoderConfig,
    inner: AmfEnc<FnEncodeHandler<u64>>,
    collected: Collector,
    flushed: bool,
    width: u32,
    height: u32,
    frame_counter: u64,
}

unsafe impl Send for AmfEncoder {}

impl AmfEncoder {
    pub fn new(config: EncoderConfig, gpu_index: u32) -> Result<Self> {
        // 8-bit → NV12, 10-bit → P010 (AMF_SURFACE_P010). 10-bit AV1 Main is
        // web-safe; the muxer tags HDR via colr/mdcv/clli.
        let frame_format = match config.pixel_format {
            PixelFormat::Yuv420p => FrameFormat::Nv12,
            PixelFormat::Yuv420p10le => FrameFormat::P010,
            other => bail!("AMF encoder expects Yuv420p or Yuv420p10le, got {other:?}"),
        };
        if gpu_index != 0 {
            tracing::warn!(
                gpu_index,
                "shiguredo_amf selects the default AMF adapter; multi-GPU pinning not supported"
            );
        }

        let (fr_num, fr_den) = frame_rate_rational(config.frame_rate);
        let tp = tuning::amf_av1_params(config.target, config.tier, config.width, config.height);

        let mut amf_cfg = AmfConfig::new(
            CodecConfig::Av1(Av1EncoderConfig {
                profile: Some(Av1Profile::Main),
            }),
            config.width,
            config.height,
            frame_format,
            fr_num,
            fr_den,
            RateControlMode::Cqp,
        );
        // Constant-QP with the tuning adapter's AV1 q-index (0..255).
        amf_cfg.qpi = Some(tp.q_index_intra as u16);
        amf_cfg.qpp = Some(tp.q_index_inter as u16);
        amf_cfg.qpb = Some(tp.q_index_inter as u16);
        amf_cfg.gop_pic_size =
            Some(config.keyframe_interval.clamp(1, u16::MAX as u32) as u16);

        let collected: Collector = Arc::new(Mutex::new(VecDeque::new()));
        let sink = Arc::clone(&collected);
        let handler =
            FnEncodeHandler::new(move |frame: std::result::Result<EncodedFrame<u64>, AmfError>| {
                match frame {
                    Ok(f) => {
                        let is_keyframe =
                            matches!(f.picture_type(), PictureType::I | PictureType::Idr);
                        let pts = *f.user_data();
                        let buf = f.buffer();
                        let size = buf.get_size() as usize;
                        let native = buf.get_native() as *const u8;
                        let data = if native.is_null() || size == 0 {
                            Bytes::new()
                        } else {
                            // SAFETY: AMF guarantees `native` points to `size`
                            // valid bytes for the lifetime of the buffer; we
                            // copy them out immediately.
                            Bytes::copy_from_slice(unsafe {
                                std::slice::from_raw_parts(native, size)
                            })
                        };
                        sink.lock().unwrap().push_back(EncodedPacket {
                            data,
                            pts,
                            is_keyframe,
                        });
                    }
                    Err(e) => tracing::warn!("AMF encode callback error: {e:?}"),
                }
            });

        let inner = AmfEnc::new(amf_cfg, handler)
            .map_err(|e| anyhow!("shiguredo_amf::Encoder::new (gpu_index={gpu_index}): {e:?}"))?;

        Ok(Self {
            width: config.width,
            height: config.height,
            config,
            inner,
            collected,
            flushed: false,
            frame_counter: 0,
        })
    }

    /// Copy a `Yuv420p` frame into an AMF host `Surface` as NV12, honoring the
    /// surface's plane pitches.
    fn fill_surface(surface: &Surface, frame: &VideoFrame) -> Result<()> {
        let w = frame.width as usize;
        let h = frame.height as usize;
        let y_size = w * h;
        let uv = (w / 2) * (h / 2);
        if frame.data.len() < y_size + 2 * uv {
            bail!(
                "AMF: Yuv420p frame too small for {w}×{h}: have {} need {}",
                frame.data.len(),
                y_size + 2 * uv
            );
        }

        // Plane 0 = Y.
        let yp = surface
            .get_plane_at(0)
            .map_err(|e| anyhow!("AMF surface Y plane: {e:?}"))?;
        let y_ptr = yp.get_native() as *mut u8;
        let y_pitch = yp.get_hpitch() as usize;
        if y_ptr.is_null() || y_pitch < w {
            bail!("AMF: bad Y plane (null={}, pitch={y_pitch}, w={w})", y_ptr.is_null());
        }
        let y_src = &frame.data[..y_size];
        for row in 0..h {
            // SAFETY: dst row [row*pitch, row*pitch+w) is in-bounds (pitch ≥ w,
            // plane height ≥ h); src row [row*w, row*w+w) is in-bounds.
            unsafe {
                std::ptr::copy_nonoverlapping(y_src.as_ptr().add(row * w), y_ptr.add(row * y_pitch), w);
            }
        }

        // Plane 1 = interleaved UV.
        let uvp = surface
            .get_plane_at(1)
            .map_err(|e| anyhow!("AMF surface UV plane: {e:?}"))?;
        let uv_ptr = uvp.get_native() as *mut u8;
        let uv_pitch = uvp.get_hpitch() as usize;
        let cw = w / 2;
        let ch = h / 2;
        if uv_ptr.is_null() || uv_pitch < cw * 2 {
            bail!("AMF: bad UV plane (null={}, pitch={uv_pitch}, need={})", uv_ptr.is_null(), cw * 2);
        }
        let u = &frame.data[y_size..y_size + uv];
        let v = &frame.data[y_size + uv..y_size + 2 * uv];
        for row in 0..ch {
            // SAFETY: dst row is `uv_pitch ≥ cw*2` wide; we write 2*cw bytes.
            let dst_row = unsafe { uv_ptr.add(row * uv_pitch) };
            for col in 0..cw {
                unsafe {
                    *dst_row.add(2 * col) = u[row * cw + col];
                    *dst_row.add(2 * col + 1) = v[row * cw + col];
                }
            }
        }
        Ok(())
    }

    /// Copy a `Yuv420p10le` frame into an AMF **P010** host `Surface`: 10-bit
    /// samples in the high 10 bits of each `u16` (`<< 6`), honoring plane
    /// pitches. Output stays 4:2:0 Main-profile 10-bit AV1 (web-safe); HDR is
    /// tagged by the muxer's colr/mdcv/clli atoms.
    fn fill_surface_p010(surface: &Surface, frame: &VideoFrame) -> Result<()> {
        let w = frame.width as usize;
        let h = frame.height as usize;
        let (cw, ch) = (w / 2, h / 2);
        let (y_samples, c_samples) = (w * h, cw * ch);
        let need = (y_samples + 2 * c_samples) * 2;
        if frame.data.len() < need {
            bail!(
                "AMF: Yuv420p10le frame too small for {w}×{h}: have {} need {}",
                frame.data.len(),
                need
            );
        }
        let src = &frame.data;
        let rd = |off: usize| u16::from_le_bytes([src[off], src[off + 1]]) << 6;

        // Plane 0 = Y (one u16 per sample).
        let yp = surface
            .get_plane_at(0)
            .map_err(|e| anyhow!("AMF P010 Y plane: {e:?}"))?;
        let y_ptr = yp.get_native() as *mut u8;
        let y_pitch = yp.get_hpitch() as usize;
        if y_ptr.is_null() || y_pitch < w * 2 {
            bail!("AMF: bad P010 Y plane (null={}, pitch={y_pitch}, need={})", y_ptr.is_null(), w * 2);
        }
        for row in 0..h {
            let dst_row = unsafe { y_ptr.add(row * y_pitch) };
            for col in 0..w {
                let b = rd((row * w + col) * 2).to_le_bytes();
                // SAFETY: pitch ≥ w*2, plane height ≥ h.
                unsafe {
                    *dst_row.add(col * 2) = b[0];
                    *dst_row.add(col * 2 + 1) = b[1];
                }
            }
        }

        // Plane 1 = interleaved UV (two u16 per chroma pixel).
        let uvp = surface
            .get_plane_at(1)
            .map_err(|e| anyhow!("AMF P010 UV plane: {e:?}"))?;
        let uv_ptr = uvp.get_native() as *mut u8;
        let uv_pitch = uvp.get_hpitch() as usize;
        if uv_ptr.is_null() || uv_pitch < cw * 4 {
            bail!("AMF: bad P010 UV plane (null={}, pitch={uv_pitch}, need={})", uv_ptr.is_null(), cw * 4);
        }
        let y_bytes = y_samples * 2;
        let (u_off, v_off) = (y_bytes, y_bytes + c_samples * 2);
        for row in 0..ch {
            let dst_row = unsafe { uv_ptr.add(row * uv_pitch) };
            for col in 0..cw {
                let idx = row * cw + col;
                let ub = rd(u_off + idx * 2).to_le_bytes();
                let vb = rd(v_off + idx * 2).to_le_bytes();
                // SAFETY: pitch ≥ cw*4.
                unsafe {
                    *dst_row.add(col * 4) = ub[0];
                    *dst_row.add(col * 4 + 1) = ub[1];
                    *dst_row.add(col * 4 + 2) = vb[0];
                    *dst_row.add(col * 4 + 3) = vb[1];
                }
            }
        }
        Ok(())
    }
}

impl Encoder for AmfEncoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        if frame.format != self.config.pixel_format {
            bail!(
                "AMF encoder configured for {:?} but received {:?}",
                self.config.pixel_format,
                frame.format
            );
        }
        if frame.width != self.width || frame.height != self.height {
            bail!(
                "AMF encoder fixed at {}×{}, received {}×{} (scale before encode)",
                self.width,
                self.height,
                frame.width,
                frame.height
            );
        }
        let surface = self
            .inner
            .alloc_surface()
            .map_err(|e| anyhow!("shiguredo_amf::Encoder::alloc_surface: {e:?}"))?;
        match frame.format {
            PixelFormat::Yuv420p => Self::fill_surface(&surface, frame)?,
            PixelFormat::Yuv420p10le => Self::fill_surface_p010(&surface, frame)?,
            other => bail!("AMF: unexpected frame format {other:?}"),
        }
        let opts = EncodeOptions {
            frame_type: frame_type::UNKNOWN,
        };
        self.inner
            .encode(surface, &opts, self.frame_counter)
            .map_err(|e| anyhow!("shiguredo_amf::Encoder::encode: {e:?}"))?;
        self.frame_counter += 1;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.flushed {
            return Ok(());
        }
        self.flushed = true;
        self.inner
            .finish()
            .map_err(|e| anyhow!("shiguredo_amf::Encoder::finish: {e:?}"))?;
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        Ok(self.collected.lock().unwrap().pop_front())
    }
}

fn frame_rate_rational(fps: f64) -> (u32, u32) {
    if (fps.fract()).abs() < 1e-6 {
        (fps.round().max(1.0) as u32, 1)
    } else {
        ((fps * 1000.0).round().max(1.0) as u32, 1000)
    }
}
