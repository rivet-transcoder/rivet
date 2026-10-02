//! MPEG-2 Video (and MPEG-1 video) through this workspace's own decoder
//! (`crates/mpeg2`, the rivet-mpeg2 repository), written clean-room from
//! ITU-T H.262; the whole ISO/IEC 13818-4 conformance suite decodes.
//!
//! The software tier for MPEG-2: NVDEC takes it first where there is one, and
//! this decodes it everywhere else. Samples are elementary-stream bytes in any
//! chunking; frames come out in display order (the decoder reorders), and
//! `finish` drains the one it holds back. Output is 8-bit 4:2:0 or 4:2:2.

use std::collections::VecDeque;

use anyhow::{Result, bail};
use bytes::Bytes;

use super::Decoder;
use crate::frame::{ColorSpace, PixelFormat, StreamInfo, VideoFrame};

/// The codec labels the MPEG-2 tier serves (MPEG-1 video included).
pub fn supports(codec_lower: &str) -> bool {
    matches!(codec_lower, "mpeg2" | "mpeg2video" | "m2v" | "mpeg1" | "mpeg1video" | "m1v")
}

/// An MPEG-2 decoder behind rivet's [`Decoder`] trait.
pub struct Mpeg2Decoder {
    inner: mpeg2::Decoder,
    info: StreamInfo,
    ready: VecDeque<VideoFrame>,
    next_pts: u64,
}

impl Mpeg2Decoder {
    pub fn new(info: StreamInfo) -> Result<Self> {
        let codec = info.codec.to_ascii_lowercase();
        if !supports(&codec) {
            bail!("the MPEG-2 decoder decodes MPEG-1/2 video, not '{codec}'");
        }
        Ok(Self { inner: mpeg2::Decoder::new(), info, ready: VecDeque::new(), next_pts: 0 })
    }

    /// The colour matrix the sequence signals (ITU-T H.273 codes), or the
    /// stream's when it signals none.
    fn color_space(&self) -> ColorSpace {
        match self.inner.sequence().and_then(|s| s.colour_description).map(|(_, _, matrix)| matrix) {
            Some(1) => ColorSpace::Bt709,
            Some(5 | 6) => ColorSpace::Bt601,
            Some(9 | 10) => ColorSpace::Bt2020,
            _ => self.info.color_space,
        }
    }

    fn take(&mut self, frames: Vec<mpeg2::Frame>) -> Result<()> {
        let color_space = self.color_space();
        for frame in frames {
            let format = match frame.chroma {
                mpeg2::ChromaFormat::Yuv420 => PixelFormat::Yuv420p,
                mpeg2::ChromaFormat::Yuv422 => PixelFormat::Yuv422p,
                #[allow(unreachable_patterns)]
                other => bail!("MPEG-2 decoded a {other:?} picture, which has no pixel format in the pipeline"),
            };
            let pts = self.next_pts;
            self.next_pts += 1;
            self.ready.push_back(VideoFrame::new(
                Bytes::from(frame.data),
                frame.width,
                frame.height,
                format,
                color_space,
                pts,
            ));
        }
        Ok(())
    }
}

impl Decoder for Mpeg2Decoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    fn push_sample(&mut self, data: &[u8]) -> Result<()> {
        let frames = self
            .inner
            .decode(data)
            .map_err(anyhow::Error::new)
            .map_err(|e| e.context("the MPEG-2 decoder could not decode the stream"))?;
        self.take(frames)
    }

    fn finish(&mut self) -> Result<()> {
        let frames = self
            .inner
            .flush()
            .map_err(anyhow::Error::new)
            .map_err(|e| e.context("the MPEG-2 decoder could not finish the stream"))?;
        self.take(frames)
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
    fn only_mpeg_video_constructs() {
        assert!(Mpeg2Decoder::new(info("mpeg2video")).is_ok());
        assert!(Mpeg2Decoder::new(info("mpeg1")).is_ok());
        assert!(Mpeg2Decoder::new(info("h264")).is_err());
    }

    /// A stream from the crate's encoder, B-frames included, comes back as
    /// every frame in display order once finished.
    #[test]
    fn encoded_frames_come_back_in_display_order() {
        let (w, h) = (64u32, 48u32);
        let mut enc = mpeg2::Encoder::new(mpeg2::EncoderConfig::new(w, h)).expect("encoder");
        let mut stream = Vec::new();
        for n in 0..5u32 {
            let mut frame = mpeg2::Frame::new(w, h, mpeg2::ChromaFormat::Yuv420);
            for (i, s) in frame.data.iter_mut().enumerate() {
                *s = ((i as u32 + n * 7) % 251) as u8;
            }
            stream.extend(enc.encode(&frame).expect("encode"));
        }
        stream.extend(enc.finish().expect("finish"));

        let mut dec = Mpeg2Decoder::new(info("mpeg2")).expect("decoder");
        // In two arbitrary chunks, as a demuxer may hand it over.
        let (a, b) = stream.split_at(stream.len() / 3);
        dec.push_sample(a).expect("decode");
        dec.push_sample(b).expect("decode");
        dec.finish().unwrap();
        for pts in 0..5 {
            let f = dec.decode_next().unwrap().expect("a frame");
            assert_eq!((f.width, f.height, f.format, f.pts), (w, h, PixelFormat::Yuv420p, pts));
            assert_eq!(f.data.len(), (w * h * 3 / 2) as usize);
        }
        assert!(dec.decode_next().unwrap().is_none());
    }
}
