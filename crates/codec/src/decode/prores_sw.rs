//! ProRes through this workspace's own decoder (`crates/prores`, the
//! rivet-prores repository), written clean-room from SMPTE RDD 36.
//!
//! Every profile shares one syntax, so one decoder serves all six. Each MOV
//! sample is one ProRes frame (`icpf`); ProRes is intra-only, so a sample in
//! is a frame out, in order. 4:2:2 comes out 10-bit and 4:4:4 12-bit (the
//! depths the profiles are defined at). An alpha plane, when a 4444 frame has
//! one, is dropped: the pipeline has no alpha.

use std::collections::VecDeque;

use anyhow::{Result, bail};
use bytes::Bytes;

use super::Decoder;
use crate::frame::{ColorSpace, PixelFormat, StreamInfo, VideoFrame};

/// The codec labels the ProRes tier serves: the label demux gives every
/// ProRes track, and the six profile fourccs.
pub fn supports(codec_lower: &str) -> bool {
    matches!(codec_lower, "prores" | "apco" | "apcs" | "apcn" | "apch" | "ap4h" | "ap4x")
}

/// A ProRes decoder behind rivet's [`Decoder`] trait.
pub struct ProresDecoder {
    inner: prores::Decoder,
    info: StreamInfo,
    ready: VecDeque<VideoFrame>,
    next_pts: u64,
}

impl ProresDecoder {
    pub fn new(info: StreamInfo) -> Result<Self> {
        let codec = info.codec.to_ascii_lowercase();
        if !supports(&codec) {
            bail!("the ProRes decoder decodes ProRes, not '{codec}'");
        }
        Ok(Self { inner: prores::Decoder::new(), info, ready: VecDeque::new(), next_pts: 0 })
    }

    /// The frame's colour matrix, from its header (ITU-T H.273 codes), or the
    /// stream's when the header leaves it unspecified.
    fn color_space(&self, metadata: &prores::Metadata) -> ColorSpace {
        match metadata.matrix_coefficients {
            1 => ColorSpace::Bt709,
            5 | 6 => ColorSpace::Bt601,
            9 | 10 => ColorSpace::Bt2020,
            _ => self.info.color_space,
        }
    }

    fn convert(&mut self, frame: prores::Frame) -> Result<VideoFrame> {
        let format = match (frame.chroma, frame.bit_depth) {
            (prores::ChromaFormat::Yuv422, 10) => PixelFormat::Yuv422p10le,
            (prores::ChromaFormat::Yuv444, 12) => PixelFormat::Yuv444p12le,
            (chroma, depth) => {
                bail!("ProRes decoded a {chroma:?} {depth}-bit picture, which has no pixel format in the pipeline")
            }
        };
        let color_space = self.color_space(&frame.metadata);
        let pts = self.next_pts;
        self.next_pts += 1;
        Ok(VideoFrame::new(
            Bytes::from(frame.to_le_bytes()),
            frame.width,
            frame.height,
            format,
            color_space,
            pts,
        ))
    }
}

impl Decoder for ProresDecoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    fn push_sample(&mut self, data: &[u8]) -> Result<()> {
        let frame = self
            .inner
            .decode(data)
            .map_err(anyhow::Error::new)
            .map_err(|e| e.context("the ProRes decoder could not decode a frame"))?;
        let frame = self.convert(frame)?;
        self.ready.push_back(frame);
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
            pixel_format: PixelFormat::Yuv422p10le,
            color_space: ColorSpace::Bt709,
            total_frames: 0,
            bitrate: 0,
            color_metadata: Default::default(),
        }
    }

    fn picture(width: u32, height: u32, chroma: prores::ChromaFormat, depth: u32) -> prores::Frame {
        let mut frame = prores::Frame::new(width, height, chroma, depth).expect("frame");
        let mid = 1u16 << (depth - 1);
        for (i, s) in frame.data.iter_mut().enumerate() {
            *s = (mid + (i % 97) as u16) & ((1 << depth) - 1);
        }
        frame
    }

    #[test]
    fn only_prores_constructs() {
        assert!(ProresDecoder::new(info("prores")).is_ok());
        assert!(ProresDecoder::new(info("apch")).is_ok());
        assert!(ProresDecoder::new(info("h264")).is_err());
    }

    /// A frame from the crate's encoder comes back as one pipeline frame of the
    /// right format, size and order.
    #[test]
    fn encoded_frames_come_back_in_order() {
        for (profile, chroma, depth, format) in [
            (prores::Profile::Hq, prores::ChromaFormat::Yuv422, 10, PixelFormat::Yuv422p10le),
            (prores::Profile::P4444, prores::ChromaFormat::Yuv444, 12, PixelFormat::Yuv444p12le),
        ] {
            let encoder = prores::Encoder::new(prores::Config::new(profile));
            let mut dec = ProresDecoder::new(info("prores")).expect("decoder");
            for _ in 0..2 {
                let sample = encoder.encode(&picture(64, 32, chroma, depth)).expect("encode");
                dec.push_sample(&sample).expect("decode");
            }
            dec.finish().unwrap();
            for pts in 0..2 {
                let f = dec.decode_next().unwrap().expect("a frame");
                assert_eq!((f.width, f.height, f.format, f.pts), (64, 32, format, pts));
                let bytes_per_sample = 2;
                let planes = if chroma == prores::ChromaFormat::Yuv422 { 2 } else { 3 };
                assert_eq!(f.data.len(), 64 * 32 * planes * bytes_per_sample);
            }
            assert!(dec.decode_next().unwrap().is_none());
        }
    }

    #[test]
    fn garbage_is_an_error() {
        let mut dec = ProresDecoder::new(info("prores")).expect("decoder");
        assert!(dec.push_sample(&[0, 1, 2, 3]).is_err());
    }
}
