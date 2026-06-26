//! Intel QSV / oneVPL AV1 encoder.
//!
//! Wraps `shiguredo_vpl::Encoder` (vendored at `crates/shiguredo_vpl`,
//! Apache-2.0). Replaces a 2233-line hand-rolled FFI mirror that was
//! "spec-conformant-by-review only, never run on real Intel HW". The
//! wrapper crate static-links libvpl 2.16 and bindgens the headers, so
//! we pick up upstream's exact struct sizes / field orderings.
//!
//! Hardware target: Intel Arc / Meteor Lake / Lunar Lake / Battle
//! Mage. Codec dispatch is gated to AV1 in this file because the
//! pipeline's encoder trait targets AV1 only — h264/h265/vp9 encode
//! through libvpl is supported by the wrapper crate but unused here.
//!
//! Input format: this encoder consumes `Yuv420p` frames from the
//! pipeline. The wrapper expects NV12 (semi-planar) so we
//! interleave the U / V planes on the way in. 10-bit (`Yuv420p10le`
//! → P010) is not yet wired in the wrapper crate — 10-bit jobs fall
//! through to the next encoder tier (rav1e on systems without NVENC
//! Ada+).

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use shiguredo_vpl::{
    Av1EncoderConfig, Av1Profile, CodecConfig, EncodeOptions, EncoderConfig as VplEncoderConfig,
    FrameFormat, PictureType, RateControlMode,
};

use super::tuning::{self};
use super::{AUTO_FROM_TARGET, EncodedPacket, Encoder, EncoderConfig};
use crate::frame::{PixelFormat, VideoFrame};

pub struct QsvEncoder {
    config: EncoderConfig,
    inner: shiguredo_vpl::Encoder,
    encoded_packets: Vec<EncodedPacket>,
    packet_cursor: usize,
    flushed: bool,
    /// Reusable NV12 staging buffer — sized at construction to
    /// `coded_w × coded_h × 3 / 2` (matches the encoder's expectation).
    nv12_scratch: Vec<u8>,
    coded_w: u32,
    coded_h: u32,
    frame_counter: u64,
}

unsafe impl Send for QsvEncoder {}

impl QsvEncoder {
    pub fn new(config: EncoderConfig, gpu_index: u32) -> Result<Self> {
        // Tuning adapter — same path as NVENC / AMF: maps
        // (target, tier, w, h) to QSV-native knobs.
        let tp = tuning::qsv_av1_params(config.target, config.tier, config.width, config.height);

        // 8-bit only on this backend. 10-bit (P010) is a wrapper-crate gap —
        // `shiguredo_vpl`'s `FrameFormat` exposes Nv12 / Yuy2 / Bgra, no P010 —
        // so unlike NVENC (Yuv420_10bit) and AMF (P010), QSV can't do HDR
        // without the `ffmpeg` feature yet. Fail fast so the dispatcher falls
        // through to the next encoder tier instead of silently downgrading.
        match config.pixel_format {
            PixelFormat::Yuv420p => {}
            PixelFormat::Yuv420p10le => bail!(
                "QSV encoder: 10-bit (Yuv420p10le → P010) not exposed by shiguredo_vpl; \
                 use the nvidia/amd/ffmpeg path for HDR. Falling through to next tier."
            ),
            other => bail!("QSV encoder expects Yuv420p / Yuv420p10le, got {other:?}"),
        }

        // 16-pixel coding alignment matches what the wrapper crate's
        // `coded_size()` does internally for AV1.
        let coded_w = ((config.width + 15) & !15).max(16);
        let coded_h = ((config.height + 15) & !15).max(16);

        // Use ICQ (Intelligent Constant Quality) — matches the
        // tuning adapter's `icq_quality` directly. CQP is also
        // available (set qpi/qpp/qpb on the EncoderConfig) but ICQ
        // gives consistent perceptual quality across scene
        // complexity, which matches our pipeline's "target VMAF"
        // policy. Allow caller to override via the legacy `quality`
        // sentinel.
        let icq_quality: u16 = if config.quality == AUTO_FROM_TARGET {
            tp.icq_quality
        } else {
            (config.quality as u16).clamp(1, 51)
        };
        let rc_mode = RateControlMode::Icq;

        // Frame rate as rational. Rounding to integer FPS is fine for
        // every preset our pipeline uses (24 / 25 / 30 / 60); precise
        // 1001-family rationals (23.976, 29.97) plumb through if we
        // pass the source's exact (num, den) — but our pipeline only
        // carries `frame_rate: f64`, so derive a sensible (num, den).
        let (fr_num, fr_den) = if (config.frame_rate.fract() - 0.0).abs() < 1e-6 {
            (config.frame_rate as u32, 1u32)
        } else {
            ((config.frame_rate * 1000.0).round() as u32, 1000u32)
        };

        let codec_cfg = CodecConfig::Av1(Av1EncoderConfig {
            profile: Some(Av1Profile::Main),
        });
        // Pin this encoder session to the requested physical adapter.
        // `adapter_selector_for_gpu_index` maps our 0-based PCI-bus-
        // ordered gpu_index to a libvpl DRM render node so sessions
        // spread across physical Arc cards instead of stacking on
        // adapter 0.
        let adapter = crate::gpu::adapter_selector_for_gpu_index(gpu_index)?;
        let mut vpl_cfg = VplEncoderConfig::new(
            adapter,
            codec_cfg,
            coded_w,
            coded_h,
            FrameFormat::Nv12,
            fr_num,
            fr_den,
            rc_mode,
        );
        vpl_cfg.gop_pic_size = Some(config.keyframe_interval.min(u16::MAX as u32) as u16);
        vpl_cfg.gop_ref_dist = Some(1); // no B-frames (matches NVENC default in this repo)
        vpl_cfg.target_usage = Some(tp.target_usage);
        vpl_cfg.icq_quality = Some(icq_quality);

        let inner = shiguredo_vpl::Encoder::new(vpl_cfg).with_context(|| {
            format!(
                "shiguredo_vpl::Encoder::new (gpu_index={gpu_index}, adapter={adapter:?}) — \
             Intel adapter visible? /dev/dri exposed?"
            )
        })?;

        let nv12_scratch_len = (coded_w as usize) * (coded_h as usize) * 3 / 2;
        Ok(Self {
            config,
            inner,
            encoded_packets: Vec::new(),
            packet_cursor: 0,
            flushed: false,
            nv12_scratch: vec![0u8; nv12_scratch_len],
            coded_w,
            coded_h,
            frame_counter: 0,
        })
    }

    /// Convert a `Yuv420p` frame to NV12 in `self.nv12_scratch`.
    /// Returns Err if the frame's dims don't fit in the encoder's
    /// coded surface (caller cannot resize mid-stream).
    fn yuv420p_to_nv12(&mut self, frame: &VideoFrame) -> Result<()> {
        let w = frame.width as usize;
        let h = frame.height as usize;
        let coded_w = self.coded_w as usize;
        let coded_h = self.coded_h as usize;
        if w > coded_w || h > coded_h {
            bail!(
                "QSV encoder: frame {}×{} larger than coded surface {}×{}",
                w,
                h,
                coded_w,
                coded_h
            );
        }
        let y_size = w * h;
        let uv_size = (w / 2) * (h / 2);
        if frame.data.len() < y_size + 2 * uv_size {
            bail!(
                "QSV encoder: Yuv420p frame too small for {}×{}: have {} bytes need {}",
                w,
                h,
                frame.data.len(),
                y_size + 2 * uv_size
            );
        }

        // Pre-fill the scratch with neutral YUV black BEFORE the
        // content copy. Content rows/cols overwrite [0..h]×[0..w]; the
        // padding region (rows [h..coded_h], cols [w..coded_w]) keeps
        // these init values. Wrong fill = wrong padding decode:
        //
        //   - Old code zeroed the whole buffer (Y=0, U=0, V=0). The
        //     decoder reads NV12 padding as Y=0 + chroma=(0, 0) which
        //     in BT.709 limited renders RGB(0, 154, 0) — bright green
        //     bars on the right/bottom edges of any variant whose
        //     dimensions aren't multiples of 16 (most non-16:9
        //     sources after the resolution-snap). User-reported
        //     2026-05-09.
        //
        //   - Correct: Y plane = 16 (BT.709 limited black floor),
        //     UV plane = (128, 128) neutral chroma. Decoded padding
        //     renders as black bars (visually compatible with
        //     letterboxed content) regardless of variant alignment.
        //
        // Future improvement: feed coded_w/coded_h to the encoder
        // separately from the displayable width/height so the AV1
        // sequence header advertises the smaller actual frame size
        // and players crop the padding entirely. The shiguredo_vpl
        // wrapper takes one (w, h) pair today; revisit when it gains
        // a separate display-size knob.
        let y_plane_len = coded_w * coded_h;
        for b in &mut self.nv12_scratch[..y_plane_len] {
            *b = 16;
        }
        for b in &mut self.nv12_scratch[y_plane_len..] {
            *b = 128;
        }

        // Y plane: copy row-by-row honouring coded pitch.
        let y_src = &frame.data[..y_size];
        for row in 0..h {
            let dst_off = row * coded_w;
            let src_off = row * w;
            self.nv12_scratch[dst_off..dst_off + w].copy_from_slice(&y_src[src_off..src_off + w]);
        }

        // UV plane: interleave U and V, place at coded_w × coded_h offset.
        let u_src = &frame.data[y_size..y_size + uv_size];
        let v_src = &frame.data[y_size + uv_size..y_size + 2 * uv_size];
        let cw = w / 2;
        let ch = h / 2;
        let coded_cw = coded_w / 2;
        let uv_base = y_plane_len;
        for row in 0..ch {
            let dst_row_base = uv_base + row * coded_w;
            for col in 0..cw {
                self.nv12_scratch[dst_row_base + 2 * col] = u_src[row * cw + col];
                self.nv12_scratch[dst_row_base + 2 * col + 1] = v_src[row * cw + col];
            }
            // The remaining `coded_cw - cw` columns keep the neutral
            // chroma (128) preset above.
            let _ = coded_cw;
        }

        Ok(())
    }

    /// Drain any encoded frames the wrapper has finished into our
    /// pending queue.
    fn drain_inner(&mut self) {
        while let Some(packet) = self.inner.next_frame() {
            let is_keyframe = matches!(packet.picture_type(), PictureType::Idr | PictureType::I);
            let pts = packet.timestamp();
            self.encoded_packets.push(EncodedPacket {
                data: Bytes::copy_from_slice(packet.data()),
                pts,
                is_keyframe,
            });
        }
    }
}

impl Encoder for QsvEncoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        if frame.format != PixelFormat::Yuv420p {
            bail!(
                "QSV encoder configured for {:?} but received {:?}",
                self.config.pixel_format,
                frame.format
            );
        }
        self.yuv420p_to_nv12(frame)?;
        let opts = EncodeOptions { frame_type: 0 };
        self.inner
            .encode(&self.nv12_scratch, &opts)
            .context("shiguredo_vpl::Encoder::encode")?;
        self.frame_counter += 1;
        self.drain_inner();
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if self.flushed {
            return Ok(());
        }
        self.flushed = true;
        self.inner
            .finish()
            .context("shiguredo_vpl::Encoder::finish")?;
        self.drain_inner();
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        if self.packet_cursor < self.encoded_packets.len() {
            let pkt = self.encoded_packets[self.packet_cursor].clone();
            self.packet_cursor += 1;
            Ok(Some(pkt))
        } else {
            Ok(None)
        }
    }
}
