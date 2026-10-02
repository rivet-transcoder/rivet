//! VP9 through this workspace's own decoder (`crates/vp9`, the rivet-vp9
//! repository), written clean-room from the VP9 bitstream specification and
//! bit-exact on the public test vectors (profiles 0-3).
//!
//! The software tier for VP9: NVDEC, AMF and QSV take it first where they
//! can, and this decodes it everywhere else. Each sample is one compressed
//! frame or a superframe; a hidden frame produces nothing, and a superframe
//! shows at most one. 8, 10 and 12-bit 4:2:0, 4:2:2 and 4:4:4 map to the
//! pipeline's formats; 4:4:0 and RGB-coded streams have no place in it and are
//! refused.

use std::collections::VecDeque;

use anyhow::{Result, bail};
use bytes::Bytes;

use super::Decoder;
use crate::frame::{ColorSpace, PixelFormat, StreamInfo, VideoFrame};

/// The codec labels the VP9 tier serves.
pub fn supports(codec_lower: &str) -> bool {
    matches!(codec_lower, "vp9" | "vp09")
}

/// A VP9 decoder behind rivet's [`Decoder`] trait.
pub struct Vp9Decoder {
    inner: vp9::Decoder,
    info: StreamInfo,
    ready: VecDeque<VideoFrame>,
    next_pts: u64,
}

impl Vp9Decoder {
    pub fn new(info: StreamInfo) -> Result<Self> {
        let codec = info.codec.to_ascii_lowercase();
        if !supports(&codec) {
            bail!("the VP9 decoder decodes VP9, not '{codec}'");
        }
        Ok(Self { inner: vp9::Decoder::new(), info, ready: VecDeque::new(), next_pts: 0 })
    }

    fn convert(&mut self, frame: vp9::Frame) -> Result<VideoFrame> {
        use vp9::ChromaFormat as C;
        let format = match (frame.chroma, frame.bit_depth) {
            (C::Yuv420, 8) => PixelFormat::Yuv420p,
            (C::Yuv420, 10) => PixelFormat::Yuv420p10le,
            (C::Yuv420, 12) => PixelFormat::Yuv420p12le,
            (C::Yuv422, 8) => PixelFormat::Yuv422p,
            (C::Yuv422, 10) => PixelFormat::Yuv422p10le,
            (C::Yuv422, 12) => PixelFormat::Yuv422p12le,
            (C::Yuv444, 8) => PixelFormat::Yuv444p,
            (C::Yuv444, 10) => PixelFormat::Yuv444p10le,
            (C::Yuv444, 12) => PixelFormat::Yuv444p12le,
            (chroma, depth) => {
                bail!("VP9 decoded a {chroma:?} {depth}-bit picture, which has no pixel format in the pipeline")
            }
        };
        let color_space = match frame.color_space {
            vp9::ColorSpace::Bt709 => ColorSpace::Bt709,
            vp9::ColorSpace::Bt2020 => ColorSpace::Bt2020,
            vp9::ColorSpace::Bt601 | vp9::ColorSpace::Smpte170 | vp9::ColorSpace::Smpte240 => ColorSpace::Bt601,
            vp9::ColorSpace::Rgb => bail!("this VP9 stream is coded as RGB, which the pipeline does not take"),
            _ => self.info.color_space,
        };
        let pts = self.next_pts;
        self.next_pts += 1;
        Ok(VideoFrame::new(Bytes::from(frame.data), frame.width, frame.height, format, color_space, pts))
    }
}

impl Decoder for Vp9Decoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    fn push_sample(&mut self, data: &[u8]) -> Result<()> {
        let decoded = self
            .inner
            .decode(data)
            .map_err(anyhow::Error::new)
            .map_err(|e| e.context("the VP9 decoder could not decode a frame"))?;
        if let Some(frame) = decoded {
            let frame = self.convert(frame)?;
            self.ready.push_back(frame);
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        Ok(())
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        Ok(self.ready.pop_front())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(codec: &str) -> StreamInfo {
        StreamInfo {
            codec: codec.to_string(),
            width: 0,
            height: 0,
            frame_rate: 0.0,
            duration: 0.0,
            pixel_format: PixelFormat::Yuv420p,
            color_space: ColorSpace::Bt601,
            total_frames: 0,
            bitrate: 0,
            color_metadata: Default::default(),
        }
    }

    #[test]
    fn only_vp9_constructs() {
        assert!(Vp9Decoder::new(info("vp9")).is_ok());
        assert!(Vp9Decoder::new(info("vp8")).is_err());
    }

    /// Frames from the crate's encoder come back as pipeline frames, in order.
    #[test]
    fn encoded_frames_come_back_in_order() {
        let (w, h) = (64u32, 48u32);
        let mut enc = vp9::Encoder::new(vp9::Config::new(w, h));
        let mut dec = Vp9Decoder::new(info("vp9")).expect("decoder");
        for n in 0..3u32 {
            let mut frame = vp9::Frame::new(w, h, 8, vp9::ChromaFormat::Yuv420);
            for (i, s) in frame.data.iter_mut().enumerate() {
                *s = ((i as u32 + n * 7) % 251) as u8;
            }
            let packet = enc.encode(&frame).expect("encode");
            dec.push_sample(&packet).expect("decode");
        }
        dec.finish().unwrap();
        for pts in 0..3 {
            let f = dec.decode_next().unwrap().expect("a frame");
            assert_eq!((f.width, f.height, f.format, f.pts), (w, h, PixelFormat::Yuv420p, pts));
            assert_eq!(f.data.len(), (w * h * 3 / 2) as usize);
        }
        assert!(dec.decode_next().unwrap().is_none());
    }

    #[test]
    fn garbage_is_an_error() {
        let mut dec = Vp9Decoder::new(info("vp9")).expect("decoder");
        assert!(dec.push_sample(&[0xff, 0xff, 0xff]).is_err());
    }
}
