//! Native H.264 / HEVC decode in software — rivet's own decoders.
//!
//! The `h26x` crate is this workspace's pure-Rust H.264 and H.265 decoder
//! pair: written from the ITU-T specifications, bit-exact against the JVT and
//! JCT-VC conformance suites, frame- and wavefront-threaded, with AVX2 and
//! NEON kernels chosen at run time. No C, no system library, nothing to
//! install on a build host — so unlike libavcodec it is always compiled, and
//! unlike openh264 it handles the profiles that actually arrive (High, 8x8
//! transform, CABAC B-frames, weighted prediction; Main/Main 10/Main 12,
//! WPP, tiles, SAO...).
//!
//! # Where it sits
//!
//! Below every hardware tier — a fixed-function block is still faster and
//! costs no CPU — and first among the software ones for the two codecs it
//! serves. libavcodec, when the build has it, is the tier *behind* this one:
//! it catches what these decoders refuse (interlaced H.264, 4:2:2, the odd
//! profile), because a stream `h26x` cannot decode says so up front, as an
//! [`h26x::Error::Unsupported`], and the dispatcher hands the stream on.
//!
//! # Threads
//!
//! Each decoder runs its own worker pool sized to the machine
//! (`H26X_THREADS` overrides). The pipeline decodes a source once for the whole
//! ladder, so one decoder owning the cores is the intended shape.
//!
//! # Annex-B, whole access units
//!
//! Samples arrive as Annex-B access units (the demuxers convert AVCC/HVCC).
//! Every NAL is fed separately so an error names the position, and pictures
//! are collected as soon as the decoder has them finished — the frame
//! threading means several may be in flight behind the one just pushed.

use std::collections::VecDeque;

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use super::Decoder;
use crate::frame::{PixelFormat, StreamInfo, VideoFrame};

/// The two decoders behind one face.
enum Inner {
    H264(h26x::h264::H264Decoder),
    Hevc(h26x::hevc::HevcDecoder),
}

impl Inner {
    fn push_nal(&mut self, nal: &[u8]) -> h26x::Result<()> {
        match self {
            Inner::H264(d) => d.push_nal(nal),
            Inner::Hevc(d) => d.push_nal(nal),
        }
    }
    fn try_next_picture(&mut self) -> Option<h26x::Picture> {
        match self {
            Inner::H264(d) => d.try_next_picture(),
            Inner::Hevc(d) => d.try_next_picture(),
        }
    }
    fn next_picture(&mut self) -> Option<h26x::Picture> {
        match self {
            Inner::H264(d) => d.next_picture(),
            Inner::Hevc(d) => d.next_picture(),
        }
    }
    fn flush(&mut self) -> h26x::Result<()> {
        match self {
            Inner::H264(d) => d.flush(),
            Inner::Hevc(d) => d.flush(),
        }
    }
}

/// Software H.264 / HEVC decoder on the native `h26x` crate.
pub struct H26xDecoder {
    inner: Inner,
    info: StreamInfo,
    ready: VecDeque<VideoFrame>,
    /// Frames are numbered in output order from zero; the container's
    /// timestamps are not carried through this interface and every consumer
    /// re-timestamps anyway (the same convention as the AV1 tier).
    next_pts: u64,
    /// Whether the decoder has produced anything yet — after that a refusal is
    /// a stream error, not a capability question.
    produced: bool,
}

/// Whether the native tier serves `codec_lower`.
pub fn supports(codec_lower: &str) -> bool {
    matches!(
        codec_lower,
        "h264" | "avc1" | "avc" | "h265" | "hevc" | "hvc1" | "hev1" | "hvc2" | "hev2"
    )
}

impl H26xDecoder {
    /// Build a decoder for `info.codec` (or the label passed by the
    /// dispatcher, already lower-cased and stored in `info`).
    pub fn new(info: StreamInfo) -> Result<Self> {
        let codec = info.codec.to_ascii_lowercase();
        let inner = match codec.as_str() {
            "h264" | "avc1" | "avc" => Inner::H264(h26x::h264::H264Decoder::new()),
            "h265" | "hevc" | "hvc1" | "hev1" | "hvc2" | "hev2" => {
                Inner::Hevc(h26x::hevc::HevcDecoder::new())
            }
            other => bail!("h26x decodes H.264 and HEVC, not '{other}'"),
        };
        Ok(Self { inner, info, ready: VecDeque::new(), next_pts: 0, produced: false })
    }

    /// Turn a decoded picture into a [`VideoFrame`], or say why the format has
    /// no place in the pipeline's pixel formats.
    fn convert(&mut self, pic: h26x::Picture) -> Result<VideoFrame> {
        use h26x::ChromaFormat as C;
        let (format, shift): (PixelFormat, u32) = match (pic.chroma, pic.bit_depth) {
            (C::Yuv420 | C::Monochrome, 8) => (PixelFormat::Yuv420p, 0),
            (C::Yuv420 | C::Monochrome, 9) => (PixelFormat::Yuv420p10le, 1),
            (C::Yuv420 | C::Monochrome, 10) => (PixelFormat::Yuv420p10le, 0),
            (C::Yuv420 | C::Monochrome, 12) => (PixelFormat::Yuv420p12le, 0),
            (C::Yuv422, 8) => (PixelFormat::Yuv422p, 0),
            (C::Yuv422, 9) => (PixelFormat::Yuv422p10le, 1),
            (C::Yuv422, 10) => (PixelFormat::Yuv422p10le, 0),
            (C::Yuv444, 8) => (PixelFormat::Yuv444p, 0),
            (C::Yuv444, 9) => (PixelFormat::Yuv444p10le, 1),
            (C::Yuv444, 10) => (PixelFormat::Yuv444p10le, 0),
            (chroma, depth) => bail!(
                "h26x decoded a {chroma:?} {depth}-bit picture, which has no pixel format in the pipeline"
            ),
        };
        let (w, h) = (pic.width as usize, pic.height as usize);
        let bytes_per_sample = if pic.bit_depth > 8 { 2 } else { 1 };
        // Monochrome travels as 4:2:0 with grey chroma (below), so its chroma
        // planes are sized for that.
        let (sw, sh) = if pic.chroma == C::Monochrome { (2, 2) } else { pic.chroma.subsampling() };
        let (cw, ch) = (w.div_ceil(sw as usize), h.div_ceil(sh as usize));
        let mut out = Vec::with_capacity((w * h + 2 * cw * ch) * bytes_per_sample);
        // Planes are tightly packed already; a 9-bit picture is widened into
        // the 10-bit format's range on the way through.
        for plane in &pic.planes {
            if shift == 0 {
                out.extend_from_slice(&plane.data);
            } else {
                for pair in plane.data.chunks_exact(2) {
                    let v = u16::from_le_bytes([pair[0], pair[1]]) << shift;
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        if pic.chroma == C::Monochrome {
            // Grey chroma, so a luma-only stream still travels as 4:2:0 —
            // the pipeline has no grey format and every consumer expects three
            // planes.
            let mid: u16 = 1 << (pic.bit_depth + shift - 1);
            for _ in 0..2 * cw * ch {
                if bytes_per_sample == 1 {
                    out.push(mid as u8);
                } else {
                    out.extend_from_slice(&mid.to_le_bytes());
                }
            }
        }
        // The bitstream is authoritative over whatever the container said.
        self.info.width = pic.width;
        self.info.height = pic.height;
        self.info.pixel_format = format;

        let pts = self.next_pts;
        self.next_pts += 1;
        self.produced = true;
        Ok(VideoFrame::new(
            Bytes::from(out),
            pic.width,
            pic.height,
            format,
            self.info.color_space,
            pts,
        ))
    }

    /// Queue every picture the decoder has finished.
    fn collect_ready(&mut self) -> Result<()> {
        while let Some(pic) = self.inner.try_next_picture() {
            let frame = self.convert(pic)?;
            self.ready.push_back(frame);
        }
        Ok(())
    }
}

impl Decoder for H26xDecoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    fn push_sample(&mut self, data: &[u8]) -> Result<()> {
        for nal in h26x::nal::annexb_nals(data) {
            match self.inner.push_nal(nal) {
                Ok(()) => {}
                // A feature these decoders do not implement. Said up front, on
                // the parameter set, so the dispatcher can hand the stream to
                // the next tier with nothing lost.
                Err(e @ h26x::Error::Unsupported(_)) => {
                    return Err(anyhow::Error::new(e))
                        .context("the native H.264/HEVC decoder does not support this stream");
                }
                // Malformed data. Not fatal once the decoder is running: a
                // stream that starts mid-GOP, or carries a damaged NAL, errors
                // until its next keyframe, and failing the job there would
                // refuse material every other decoder accepts. Before the
                // first picture it is a refusal, and the next tier gets a go.
                Err(e) => {
                    if !self.produced {
                        return Err(anyhow::Error::new(e))
                            .context("the native H.264/HEVC decoder could not start on this stream");
                    }
                    tracing::debug!(error = %e, "h26x rejected a NAL; continuing");
                }
            }
        }
        self.collect_ready()
    }

    fn finish(&mut self) -> Result<()> {
        if let Err(e) = self.inner.flush() {
            tracing::warn!(error = %e, "h26x flush reported an error; the tail may be short");
        }
        while let Some(pic) = self.inner.next_picture() {
            let frame = self.convert(pic)?;
            self.ready.push_back(frame);
        }
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
            color_space: ColorSpace::Bt709,
            total_frames: 0,
            bitrate: 0,
            color_metadata: Default::default(),
        }
    }

    #[test]
    fn both_codecs_construct_and_others_do_not() {
        assert!(H26xDecoder::new(info("h264")).is_ok());
        assert!(H26xDecoder::new(info("hevc")).is_ok());
        assert!(H26xDecoder::new(info("hvc1")).is_ok());
        assert!(H26xDecoder::new(info("av1")).is_err());
        assert!(supports("avc1") && supports("hev1") && !supports("vp9"));
    }

    #[test]
    fn an_unsupported_stream_is_refused_up_front() {
        // An H.264 SPS with frame_mbs_only_flag = 0 (interlaced): the decoder
        // says so on the parameter set, before any slice, so the tier above
        // can hand the stream on with nothing lost.
        let mut d = H26xDecoder::new(info("h264")).expect("decoder");
        // The SPS, PPS and the start of the first IDR slice of an x264
        // `--interlaced` encode (High profile, level 2.1, frame_mbs_only_flag
        // = 0), taken from a real stream rather than hand-assembled. The
        // refusal comes on the first slice, before any picture exists.
        let sps: &[u8] = &[
            0, 0, 0, 1, 0x67, 0x64, 0x00, 0x15, 0xac, 0xd9, 0x41, 0x43, 0x3f, 0x2c, 0xd4, 0x18, 0x04,
            0x19, 0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x00, 0x03, 0x00, 0x32, 0x1f, 0x14, 0x29, 0x96,
        ];
        let pps: &[u8] = &[0, 0, 0, 1, 0x68, 0xfa, 0xa3, 0xcb, 0x22, 0xc0];
        let idr: &[u8] = &[
            0, 0, 0, 1, 0x65, 0x88, 0x82, 0x0b, 0x1f, 0xf1, 0xb0, 0xac, 0x5b, 0xf1, 0x5a, 0x16, 0x1d,
            0xc6, 0x1d, 0x1a, 0xfd, 0xa0, 0x06, 0xc1, 0x52, 0x38, 0xd0, 0xdb, 0xad, 0x95, 0xa2, 0x07,
            0xde, 0x61, 0x88, 0xfd, 0xfa, 0xcf, 0xd7, 0xdc, 0xde, 0x9a, 0x72, 0x88,
        ];
        d.push_sample(sps).expect("a parameter set is accepted");
        d.push_sample(pps).expect("a parameter set is accepted");
        let err = d.push_sample(idr).expect_err("interlaced is unsupported");
        assert!(format!("{err:#}").contains("unsupported"), "{err:#}");
        assert!(d.decode_next().expect("drain").is_none());
    }

    #[test]
    fn rubbish_before_the_first_picture_is_a_refusal_not_a_panic() {
        let mut d = H26xDecoder::new(info("hevc")).expect("decoder");
        let r = d.push_sample(&[0, 0, 0, 1, 0x40, 0x01, 0xff, 0xff, 0xff]);
        // Either outcome is acceptable here; what matters is that it neither
        // panics nor produces a frame.
        let _ = r;
        assert!(d.decode_next().expect("drain").is_none());
    }
}
