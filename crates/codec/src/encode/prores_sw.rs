//! Apple ProRes encode — this workspace's own encoder (`crates/prores`, the
//! rivet-prores repository), written clean-room from SMPTE RDD 36:2022.
//!
//! The only ProRes encoder in the tree, built directly by
//! [`select_encoder`](super::select_encoder) for a ProRes job; see
//! [`native`](super::native).
//!
//! # What it takes and writes
//!
//! Every frame a key frame — ProRes is intra-only — one MOV sample per frame,
//! in display order. The profile (Proxy, LT, 422, HQ, 4444, 4444 XQ) comes
//! with the codec (`VideoCodec::ProRes(profile)`) and sets the chroma format
//! and the target frame size the encoder's per-slice quantiser search lands
//! under; there is no other quality knob, so a CRF or a bitrate is refused by
//! name.
//!
//! The pipeline hands every encoder 4:2:0 (8- or 10-bit), so this adapter
//! upsamples chroma to the profile's 4:2:2 or 4:4:4: vertically by linear
//! interpolation between the 4:2:0 rows (MPEG siting: a chroma row sits
//! midway between two luma rows), horizontally — 4:4:4 only — by averaging
//! neighbours (co-sited columns). The frame is coded at the input's depth;
//! the encoder scales samples to its 10- / 12-bit internal precision.
//!
//! # Colour
//!
//! The frame header's colour primaries, transfer characteristic and matrix
//! (H.273 codes) are written from `color_metadata`, so a BT.2020 PQ / HLG
//! picture says so in the bitstream as well as in the `colr` box: ProRes is
//! 10-bit with HDR here.

use std::collections::VecDeque;

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use super::native::{check_frame, frame_rate_ratio, refuse_any_rate};
use super::{AUTO_FROM_TARGET, EncodedPacket, Encoder, EncoderConfig};
use crate::frame::{ColorMetadata, PixelFormat, ProresProfile, TransferFn, VideoCodec, VideoFrame};

/// A ProRes encoder behind rivet's [`Encoder`] trait.
pub struct ProresEncoder {
    inner: prores::Encoder,
    width: u32,
    height: u32,
    profile: prores::Profile,
    format: PixelFormat,
    metadata: prores::Metadata,
    ready: VecDeque<EncodedPacket>,
}

/// The crate's profile for the pipeline's.
pub fn profile(p: ProresProfile) -> prores::Profile {
    match p {
        ProresProfile::Proxy => prores::Profile::Proxy,
        ProresProfile::Lt => prores::Profile::Lt,
        ProresProfile::Standard => prores::Profile::Standard,
        ProresProfile::Hq => prores::Profile::Hq,
        ProresProfile::P4444 => prores::Profile::P4444,
        ProresProfile::P4444Xq => prores::Profile::P4444Xq,
    }
}

/// The H.273 transfer code for a pipeline transfer (the BT.709 family as 1).
fn transfer_code(tf: TransferFn) -> u8 {
    match tf {
        TransferFn::Bt709 | TransferFn::Unspecified => 1,
        TransferFn::Bt470Bg => 4,
        TransferFn::Linear => 8,
        TransferFn::St2084 => 16,
        TransferFn::AribStdB67 => 18,
    }
}

/// The frame header's metadata: the colour codes and the frame rate code.
fn metadata(c: &ColorMetadata, frame_rate: f64) -> prores::Metadata {
    let (n, d) = frame_rate_ratio(frame_rate);
    prores::Metadata {
        aspect_ratio: 0,
        frame_rate_code: prores::Metadata::frame_rate_code_for(n, d),
        color_primaries: c.colour_primaries,
        transfer_characteristic: transfer_code(c.transfer),
        matrix_coefficients: c.matrix_coefficients,
    }
}

impl ProresEncoder {
    /// Build an encoder for `config` (codec ProRes, `yuv420p` or
    /// `yuv420p10le`).
    pub fn new(config: EncoderConfig) -> Result<Self> {
        let VideoCodec::ProRes(p) = config.codec else {
            bail!("the ProRes encoder encodes ProRes, not {}", config.codec.label());
        };
        if !matches!(config.pixel_format, PixelFormat::Yuv420p | PixelFormat::Yuv420p10le) {
            bail!("the ProRes encoder takes 8- or 10-bit 4:2:0 from the pipeline, not {:?}", config.pixel_format);
        }
        if config.quality != AUTO_FROM_TARGET {
            bail!(
                "ProRes is coded to its profile's frame size, and this rung gives a crf ({}): pick the profile \
                 (proxy, lt, 422, hq, 4444, 4444xq) instead",
                config.quality
            );
        }
        refuse_any_rate("ProRes", &config)?;
        let profile = profile(p);
        let mut cfg = prores::Config::new(profile);
        cfg.alpha = prores::AlphaType::None;
        Ok(Self {
            inner: prores::Encoder::new(cfg),
            width: config.width,
            height: config.height,
            profile,
            format: config.pixel_format,
            metadata: metadata(&config.color_metadata, config.frame_rate),
            ready: VecDeque::new(),
        })
    }

    /// The pipeline's 4:2:0 frame as the profile's 4:2:2 / 4:4:4 one.
    fn picture(&self, data: &[u8]) -> Result<prores::Frame> {
        let ten = self.format == PixelFormat::Yuv420p10le;
        let sample = |i: usize| -> u16 {
            if ten { u16::from_le_bytes([data[2 * i], data[2 * i + 1]]) } else { u16::from(data[i]) }
        };
        let chroma = self.profile.chroma();
        let mut f = prores::Frame::new(self.width, self.height, chroma, if ten { 10 } else { 8 })
            .context("the ProRes encoder refused the frame size")?;
        let (w, h) = (self.width as usize, self.height as usize);
        for (i, s) in f.plane_mut(0).iter_mut().enumerate() {
            *s = sample(i);
        }
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let four_four = chroma == prores::ChromaFormat::Yuv444;
        let out_w = if four_four { w } else { cw };
        for plane in 1..3 {
            let base = w * h + (plane - 1) * cw * ch;
            let c = |x: usize, y: usize| -> u32 { u32::from(sample(base + y.min(ch - 1) * cw + x.min(cw - 1))) };
            // 4:2:0 → 4:2:2: output row y lies a quarter of a chroma row from
            // its nearest input row, toward the next one.
            let vertical = |x: usize, y: usize| -> u32 {
                let k = y / 2;
                let other = if y.is_multiple_of(2) { k.saturating_sub(1) } else { k + 1 };
                (3 * c(x, k) + c(x, other) + 2) / 4
            };
            let out = f.plane_mut(plane);
            for y in 0..h {
                for x in 0..out_w {
                    out[y * out_w + x] = if four_four {
                        if x % 2 == 0 {
                            vertical(x / 2, y)
                        } else {
                            (vertical(x / 2, y) + vertical(x / 2 + 1, y)).div_ceil(2)
                        }
                    } else {
                        vertical(x, y)
                    } as u16;
                }
            }
        }
        f.metadata = self.metadata;
        Ok(f)
    }
}

impl Encoder for ProresEncoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        let want = check_frame("ProRes", frame, self.width, self.height, &[self.format])?;
        let picture = self.picture(&frame.data[..want])?;
        let data = self.inner.encode(&picture).context("the ProRes encoder refused a frame")?;
        self.ready.push_back(EncodedPacket { data: Bytes::from(data), pts: frame.pts, is_keyframe: true });
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        Ok(self.ready.pop_front())
    }

    /// Every frame is a key frame already.
    fn force_keyframe_next(&mut self) -> Result<()> {
        Ok(())
    }

    /// Nothing carries over between frames; only the queue is cleared.
    fn reset(&mut self) -> Result<()> {
        self.ready.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn psnr(a: &[u16], b: &[u16], max: f64) -> f64 {
        let mse: f64 =
            a.iter().zip(b).map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2)).sum::<f64>() / a.len() as f64;
        10.0 * (max * max / mse.max(1e-9)).log10()
    }

    /// Every profile: a key frame per picture, decoding to the source's luma
    /// within the profile's quality, the colour codes in the header.
    #[test]
    fn every_profile_round_trips() {
        let (w, h) = (64u32, 48u32);
        let src = super::super::native::test_picture(w, h, 3);
        for p in ProresProfile::ALL {
            let color = ColorMetadata { colour_primaries: 9, matrix_coefficients: 9, transfer: TransferFn::St2084, ..Default::default() };
            let config = EncoderConfig {
                width: w,
                height: h,
                frame_rate: 25.0,
                codec: VideoCodec::ProRes(p),
                color_metadata: color,
                ..Default::default()
            };
            let mut enc = ProresEncoder::new(config).unwrap();
            enc.send_frame(&src).unwrap();
            let packet = enc.receive_packet().unwrap().unwrap();
            assert!(packet.is_keyframe);
            let decoded = prores::Decoder::with_bit_depth(8).unwrap().decode(&packet.data).unwrap();
            assert_eq!(decoded.chroma, profile(p).chroma(), "{p:?}");
            assert_eq!((decoded.metadata.color_primaries, decoded.metadata.transfer_characteristic), (9, 16));
            assert_eq!(decoded.metadata.frame_rate(), Some((25, 1)));
            let luma: Vec<u16> = src.data[..(w * h) as usize].iter().map(|&v| u16::from(v)).collect();
            let q = psnr(decoded.plane(0), &luma, 255.0);
            // A 64x48 picture of fine diagonal detail at the profile's
            // area-scaled frame size: Proxy and LT have very few bytes for it.
            let floor = match p {
                ProresProfile::Proxy => 25.0,
                ProresProfile::Lt => 30.0,
                _ => 35.0,
            };
            assert!(q > floor, "{p:?}: {q:.1} dB");
        }
    }

    #[test]
    fn a_crf_is_refused_by_name() {
        let config = EncoderConfig {
            width: 64,
            height: 48,
            codec: VideoCodec::ProRes(ProresProfile::Hq),
            quality: 20,
            ..Default::default()
        };
        assert!(ProresEncoder::new(config).err().expect("refused").to_string().contains("profile"));
    }
}
