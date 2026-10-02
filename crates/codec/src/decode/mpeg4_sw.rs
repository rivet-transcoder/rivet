//! MPEG-4 Part 2 Visual through this workspace's own decoder (`crates/mpeg4`,
//! the rivet-mpeg4 repository), written clean-room from ISO/IEC 14496-2:
//! Simple and Advanced Simple Profile (B-VOPs, quarter-pel, GMC, interlace),
//! DivX packed bitstreams, and the H.263 short video header.
//!
//! The software tier for MPEG-4 Part 2: NVDEC takes it first where there is
//! one, and this decodes it everywhere else. The decoder configures itself
//! from the VOL header in the stream (as AVI carries it). Samples are
//! bitstream bytes in any chunking; frames come out in display order, and
//! `finish` drains what B-VOP reordering holds back. Output is 8-bit 4:2:0.

use std::collections::VecDeque;

use anyhow::{Result, bail};
use bytes::Bytes;

use super::Decoder;
use crate::frame::{PixelFormat, StreamInfo, VideoFrame};

/// The codec labels the MPEG-4 Part 2 tier serves.
pub fn supports(codec_lower: &str) -> bool {
    matches!(codec_lower, "mpeg4" | "mp4v" | "mpeg4part2" | "xvid" | "divx")
}

/// An MPEG-4 Part 2 decoder behind rivet's [`Decoder`] trait.
pub struct Mpeg4Decoder {
    inner: mpeg4::Decoder,
    info: StreamInfo,
    ready: VecDeque<VideoFrame>,
    next_pts: u64,
}

impl Mpeg4Decoder {
    pub fn new(info: StreamInfo) -> Result<Self> {
        let codec = info.codec.to_ascii_lowercase();
        if !supports(&codec) {
            bail!("the MPEG-4 Part 2 decoder decodes MPEG-4 Visual, not '{codec}'");
        }
        Ok(Self { inner: mpeg4::Decoder::new(), info, ready: VecDeque::new(), next_pts: 0 })
    }

    fn take(&mut self, frames: Vec<mpeg4::Frame>) {
        for frame in frames {
            let pts = self.next_pts;
            self.next_pts += 1;
            self.ready.push_back(VideoFrame::new(
                Bytes::from(frame.data),
                frame.width,
                frame.height,
                PixelFormat::Yuv420p,
                self.info.color_space,
                pts,
            ));
        }
    }
}

impl Decoder for Mpeg4Decoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    fn push_sample(&mut self, data: &[u8]) -> Result<()> {
        let frames = self
            .inner
            .decode(data)
            .map_err(anyhow::Error::new)
            .map_err(|e| e.context("the MPEG-4 Part 2 decoder could not decode the stream"))?;
        self.take(frames);
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        let frames = self.inner.flush();
        self.take(frames);
        Ok(())
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        Ok(self.ready.pop_front())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::ColorSpace;

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
    fn only_mpeg4_constructs() {
        assert!(Mpeg4Decoder::new(info("mpeg4")).is_ok());
        assert!(Mpeg4Decoder::new(info("mpeg2")).is_err());
    }

    /// A stream from the crate's encoder, its VOL in-band, comes back as every
    /// frame in display order once finished.
    #[test]
    fn encoded_frames_come_back_in_display_order() {
        let (w, h) = (64u32, 48u32);
        let mut enc = mpeg4::Encoder::new(mpeg4::EncoderConfig::new(w, h, 25)).expect("encoder");
        let mut stream = Vec::new();
        for n in 0..4u32 {
            let mut frame = mpeg4::Frame::new(w, h);
            for (i, s) in frame.data.iter_mut().enumerate() {
                *s = ((i as u32 + n * 7) % 251) as u8;
            }
            stream.extend(enc.encode(&frame).expect("encode"));
        }
        stream.extend(enc.finish().expect("finish"));

        let mut dec = Mpeg4Decoder::new(info("mpeg4")).expect("decoder");
        dec.push_sample(&stream).expect("decode");
        dec.finish().unwrap();
        for pts in 0..4 {
            let f = dec.decode_next().unwrap().expect("a frame");
            assert_eq!((f.width, f.height, f.format, f.pts), (w, h, PixelFormat::Yuv420p, pts));
            assert_eq!(f.data.len(), (w * h * 3 / 2) as usize);
        }
        assert!(dec.decode_next().unwrap().is_none());
    }
}
