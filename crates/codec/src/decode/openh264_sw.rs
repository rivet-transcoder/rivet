//! OpenH264 — H.264 decode in software, as the last resort.
//!
//! The counterpart to [`rav1d_sw`](super::rav1d_sw), for the codec that
//! actually arrives. AV1 already had a software tier; H.264 did not, and H.264
//! is what cameras, phones and every existing library produce — so on a host
//! with no GPU the pipeline could accept a job, download it, probe it, and then
//! fail at the decoder with nothing to fall back to.
//!
//! That combination is worse than it sounds. The *encode* side already falls
//! back to rav1e, so such a host reports itself able to transcode and fails
//! only once real work arrives.
//!
//! # Why OpenH264 and not ffmpeg
//!
//! This crate does not depend on ffmpeg in any capacity and this module does
//! not change that. `openh264` builds Cisco's decoder from vendored C via
//! `cc` — no system library, no pkg-config, nothing to install on a build
//! host. It is the same shape as the audio decoders already here (`minimp3`)
//! and as the AV1 fallback: a codec library that ships through cargo.
//!
//! # Gated at the build, not just the dispatch
//!
//! `openh264-fallback` turns on both this module and the `openh264`
//! dependency. Gating the build too, rather than compiling always and only
//! gating the dispatch, is deliberate here: the dependency compiles a C
//! codec, and a host with hardware decode should not pay that build cost for
//! a tier it will never reach.
//!
//! # Annex-B, one NAL at a time
//!
//! OpenH264 wants a single NAL unit per `decode` call and returns a picture
//! only once one is complete — so most calls legitimately produce nothing.
//! Samples arrive here as whole access units, which is several NALs, so each
//! is split before being fed in. Feeding a whole access unit as one packet
//! decodes the first NAL and silently drops the rest, which shows up much later
//! as a video missing most of its frames.
//!
//! # 4:2:0 8-bit only
//!
//! Which is what OpenH264 supports and what the overwhelming majority of H.264
//! in the wild is. Anything else is a hardware decoder's job and says so.

use anyhow::Result;
use bytes::Bytes;
use openh264::decoder::{Decoder as OpenH264Decoder, DecoderConfig};
use openh264::formats::YUVSource;
use openh264::nal_units;

use super::Decoder;
use crate::frame::{ColorSpace, PixelFormat, StreamInfo, VideoFrame};

/// Software H.264 decoder.
pub struct OpenH264SwDecoder {
    inner: OpenH264Decoder,
    info: StreamInfo,
    ready: std::collections::VecDeque<VideoFrame>,
    /// Frames are numbered in decode order from zero.
    ///
    /// The container's timestamps are not carried through here: this decoder
    /// is handed samples without them, and every consumer downstream is
    /// re-timestamping anyway. Matching the AV1 fallback rather than inventing
    /// a second convention.
    next_pts: u64,
}

impl OpenH264SwDecoder {
    /// Build a decoder. `info` is what the container already knows; the
    /// dimensions are corrected from the first decoded picture, because a
    /// container header and a sequence parameter set do disagree in the wild.
    pub fn new(info: StreamInfo) -> Result<Self> {
        let inner = OpenH264Decoder::with_api_config(
            openh264::OpenH264API::from_source(),
            DecoderConfig::new(),
        )
        .map_err(|e| anyhow::anyhow!("openh264 decoder could not be created: {e}"))?;

        Ok(Self {
            inner,
            info,
            ready: std::collections::VecDeque::new(),
            next_pts: 0,
        })
    }

    /// Copy one decoded picture into a [`VideoFrame`].
    ///
    /// Row-wise, because OpenH264 hands back planes with their own stride and
    /// a flat copy shears the picture progressively down the frame — the same
    /// trap as the AV1 fallback, and one that looks like a decoder bug rather
    /// than a copy bug.
    fn convert(info: &mut StreamInfo, next_pts: &mut u64, yuv: &impl YUVSource) -> VideoFrame {
        let (w, h) = yuv.dimensions();
        let (stride_y, stride_u, stride_v) = yuv.strides();
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));

        let mut out = Vec::with_capacity(w * h + 2 * cw * ch);

        for (plane, pw, ph, stride) in [
            (yuv.y(), w, h, stride_y),
            (yuv.u(), cw, ch, stride_u),
            (yuv.v(), cw, ch, stride_v),
        ] {
            for row in 0..ph {
                let start = row * stride;
                // Defensive rather than trusting the arithmetic: a short final
                // row would panic on the slice, and a decoder that produced
                // one should not take the process down with it.
                let end = (start + pw).min(plane.len());
                if start >= plane.len() {
                    break;
                }
                out.extend_from_slice(&plane[start..end]);
            }
        }

        // The bitstream is authoritative over whatever the container said.
        info.width = w as u32;
        info.height = h as u32;

        let pts = *next_pts;
        *next_pts += 1;

        VideoFrame::new(
            Bytes::from(out),
            w as u32,
            h as u32,
            PixelFormat::Yuv420p,
            ColorSpace::Bt709,
            pts,
        )
    }
}

impl Decoder for OpenH264SwDecoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    fn push_sample(&mut self, data: &[u8]) -> Result<()> {
        // One NAL per call — see the module docs. A sample is an access unit,
        // which is several.
        for nal in nal_units(data) {
            match self.inner.decode(nal) {
                Ok(Some(yuv)) => {
                    let frame = Self::convert(&mut self.info, &mut self.next_pts, &yuv);
                    self.ready.push_back(frame);
                }
                // No picture yet. The ordinary case: parameter sets and the
                // leading slices of a frame all decode to nothing.
                Ok(None) => {}
                // Not fatal. A stream that starts mid-GOP produces errors until
                // the first keyframe, and failing the job there would refuse
                // material every other decoder accepts.
                Err(e) => {
                    tracing::debug!(error = %e, "openh264 rejected a NAL; continuing");
                }
            }
        }

        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        // Drain whatever the decoder was still holding. Without this the tail
        // of every video is lost — the frames are decoded, buffered, and never
        // asked for.
        match self.inner.flush_remaining() {
            Ok(remaining) => {
                for yuv in remaining {
                    let frame = Self::convert(&mut self.info, &mut self.next_pts, &yuv);
                    self.ready.push_back(frame);
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "openh264 flush failed; the tail may be short");
            }
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

    fn info() -> StreamInfo {
        StreamInfo {
            width: 0,
            height: 0,
            ..Default::default()
        }
    }

    #[test]
    fn a_decoder_can_be_created() {
        // The build half of this: `openh264` compiles its vendored C, and a
        // decoder that cannot be constructed means the fallback tier silently
        // never engages.
        assert!(OpenH264SwDecoder::new(info()).is_ok());
    }

    #[test]
    fn rubbish_does_not_fail_the_job() {
        let mut decoder = OpenH264SwDecoder::new(info()).expect("decoder");

        // A stream that starts mid-GOP errors until its first keyframe, which
        // is ordinary. Returning `Err` here would refuse material every other
        // decoder accepts.
        assert!(decoder.push_sample(&[0, 0, 0, 1, 0x65, 0xff, 0xff]).is_ok());
        assert!(decoder.decode_next().expect("drain").is_none());
    }
}
