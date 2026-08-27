//! Native H.264 / H.265 encode in software — rivet's own encoders.
//!
//! The `h26x` crate is this workspace's pure-Rust H.264 and H.265 codec pair.
//! Its decoders are bit-exact against the JVT and JCT-VC conformance suites;
//! its encoders are the mirror of them, built on the same reconstruction
//! kernels, and held to a four-property gate (`h26x/tools/verify_encode.sh`):
//! our decoder reproduces the encoder's own reconstruction byte for byte
//! (**SELF**), libavcodec agrees with our decoder (**CROSS**), PSNR is reported,
//! and a rate objective, where one is set, is hit. No C, no system library,
//! nothing to install on a build host — so, like the decoders, this module is
//! always compiled.
//!
//! # Where it sits
//!
//! Below every hardware tier: a fixed-function block is faster and costs no
//! CPU. It is the last tier for the two codecs it serves, the way
//! [`rav1e_sw`](super::rav1e_sw) is for AV1, and it exists for the same
//! hosts — a laptop, a CI runner, a container with no GPU attached — where a
//! slow file beats a diagnostic.
//!
//! # Always built; the feature decides whether it is *reached*
//!
//! `h26x-fallback` gates whether [`select_encoder`](super::select_encoder)
//! **falls back** here on its own when every hardware backend has declined.
//! Off by default, for the reason `rav1e-fallback` is: a throughput fleet
//! quietly degrading into a CPU encoder reads as a capacity problem rather
//! than the missing driver it is. A caller that wants software encoding can
//! always ask for it by name, feature or no feature.
//!
//! # What it takes
//!
//! 8-bit 4:2:0. The encoders refuse deeper samples by name today (their
//! reconstruction planes are `u8`; the decoders go to 14-bit and the encoders
//! will follow), and the pipeline's other chroma layouts are converted before
//! the encoder anyway. H.264 is 8-bit only on every backend in this crate;
//! for H.265 this is the one 8-bit-only path, and a 10-bit request is
//! refused rather than narrowed.
//!
//! # Threads
//!
//! Each encoder runs its own worker pool sized to `threads`, or to the
//! machine when that is zero (`H26X_THREADS` overrides). The pipeline runs a
//! ladder's rungs as separate encoders, so a caller running several at once
//! should hand each a share.
//!
//! # Order
//!
//! No B pictures, so coding order is display order and a packet's timestamp
//! is the one its frame arrived with. The hardware tiers here are configured
//! the same way; a B-pyramid would need the muxer to carry composition
//! offsets, which it does not.

use std::collections::VecDeque;

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use super::{AUTO_FROM_TARGET, EncodedPacket, Encoder, EncoderConfig};
use crate::encode::tuning::h26x_sw_params_with;
use crate::frame::{PixelFormat, VideoCodec, VideoFrame};

/// The two encoders behind one face.
enum Inner {
    H264(h26x::encode::h264::H264Encoder),
    Hevc(h26x::encode::h265::H265Encoder),
}

impl Inner {
    fn push(&mut self, picture: &[u8]) -> h26x::Result<Vec<h26x::encode::Access>> {
        match self {
            Inner::H264(e) => e.push(picture),
            Inner::Hevc(e) => e.push(picture),
        }
    }
    fn flush(&mut self) -> h26x::Result<Vec<h26x::encode::Access>> {
        match self {
            Inner::H264(e) => e.flush(),
            Inner::Hevc(e) => e.flush(),
        }
    }
    fn frame_bytes(&self) -> usize {
        match self {
            Inner::H264(e) => e.frame_bytes(),
            Inner::Hevc(e) => e.frame_bytes(),
        }
    }
    fn force_idr(&mut self) {
        match self {
            Inner::H264(e) => e.force_idr(),
            Inner::Hevc(e) => e.force_idr(),
        }
    }
}

/// Software H.264 / H.265 encoder on the native `h26x` crate.
pub struct H26xEncoder {
    inner: Inner,
    /// The configuration `inner` was built from, kept so `reset` can build
    /// it again.
    cfg: h26x::encode::Config,
    codec: VideoCodec,
    width: u32,
    height: u32,
    /// Timestamps in the order frames were pushed. The encoder numbers each
    /// coded picture by its position in coding order, and with no B pictures
    /// that is the order of arrival, so a packet's `encode_index` is its index
    /// here. Kept as a growing table rather than a queue so a forced IDR,
    /// which reorders nothing, cannot desynchronise it either.
    pts: Vec<u64>,
    /// Packets coded but not yet collected.
    ready: VecDeque<EncodedPacket>,
}

impl H26xEncoder {
    /// Whether this tier serves `codec`.
    pub fn supports(codec: VideoCodec) -> bool {
        matches!(codec, VideoCodec::H264 | VideoCodec::H265)
    }

    /// Build an encoder for `config`.
    ///
    /// Fails rather than silently degrading when the codec or frame format is
    /// not one the native encoders take. The caller's chain has already
    /// exhausted the hardware tiers by this point, so a clear error is more
    /// useful than a picture with the planes misread — or a 10-bit request
    /// shipped at 8.
    pub fn new(config: EncoderConfig) -> Result<Self> {
        if !Self::supports(config.codec) {
            bail!(
                "the native h26x encoders produce H.264 and H.265, not {:?}",
                config.codec
            );
        }
        if config.pixel_format != PixelFormat::Yuv420p {
            bail!(
                "the native h26x software encoders take 8-bit 4:2:0 (yuv420p) today, got {:?}. \
                 For 10-bit H.265 use a hardware backend (NVENC / QSV); H.264 is 8-bit on \
                 every backend.",
                config.pixel_format
            );
        }

        let p = h26x_sw_params_with(config.codec, config.target, config.tier, &config.overrides);
        // The CRF escape hatch is already in this codec's currency (0..51),
        // and `resolve_overrides` has applied any per-rung delta to it, so it
        // replaces the derived quantiser outright.
        let qp = if config.quality == AUTO_FROM_TARGET {
            p.qp
        } else {
            config.quality.min(51)
        };

        // Zero means "decide for me", and the encoder's own zero means one
        // worker per core, which is the same answer — but the runtime's count
        // respects a container CPU quota where a core count does not, and a
        // job worker that ignores its quota starves everything sharing the
        // box. So resolve it here rather than passing the zero through.
        let threads = if config.threads > 0 {
            config.threads
        } else {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        };

        let cfg = h26x::encode::Config {
            width: config.width,
            height: config.height,
            bit_depth: 8,
            chroma: h26x::ChromaFormat::Yuv420,
            // The encoder's zero means "every picture an IDR", which is not
            // what a caller leaving the interval unset wants.
            gop: if config.keyframe_interval == 0 { 250 } else { config.keyframe_interval },
            bframes: 0,
            max_refs: 1,
            rate: h26x::encode::RateControl::ConstantQp(qp),
            entropy: h26x::encode::Entropy::Cabac,
            transform_8x8: p.transform_8x8,
            subparts: p.subparts,
            sao: p.sao,
            threads,
            fps: (config.frame_rate.round() as u32).max(1),
            cpb_ms: 0,
        };

        let inner = Self::build_inner(config.codec, &cfg)?;

        tracing::warn!(
            codec = ?config.codec,
            width = config.width,
            height = config.height,
            qp,
            transform_8x8 = p.transform_8x8,
            subparts = p.subparts,
            sao = p.sao,
            threads,
            "no {:?} encode silicon available — falling back to the native software encoder, \
             which is far slower than any hardware backend",
            config.codec
        );

        Ok(Self {
            inner,
            cfg,
            codec: config.codec,
            width: config.width,
            height: config.height,
            pts: Vec::new(),
            ready: VecDeque::new(),
        })
    }

    fn build_inner(codec: VideoCodec, cfg: &h26x::encode::Config) -> Result<Inner> {
        Ok(match codec {
            VideoCodec::H264 => Inner::H264(
                h26x::encode::h264::H264Encoder::new(cfg.clone())
                    .context("the native H.264 encoder rejected the configuration")?,
            ),
            VideoCodec::H265 => Inner::Hevc(
                h26x::encode::h265::H265Encoder::new(cfg.clone())
                    .context("the native H.265 encoder rejected the configuration")?,
            ),
            VideoCodec::Av1 => unreachable!("checked by supports()"),
        })
    }

    /// Queue every access unit the encoder handed back.
    fn collect(&mut self, units: Vec<h26x::encode::Access>) -> Result<()> {
        for a in units {
            let idx = usize::try_from(a.encode_index).context("encode index overflow")?;
            let pts = match self.pts.get(idx) {
                Some(&pts) => pts,
                // Cannot happen with no B pictures — every coded picture was
                // pushed first — but a wrong timestamp is the kind of error
                // that plays fine and drifts, so refuse rather than guess.
                None => bail!(
                    "h26x coded picture {} before any frame with that index was pushed",
                    a.encode_index
                ),
            };
            self.ready.push_back(EncodedPacket {
                data: Bytes::from(a.data),
                pts,
                is_keyframe: a.keyframe,
            });
        }
        Ok(())
    }
}

impl Encoder for H26xEncoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        if frame.format != PixelFormat::Yuv420p {
            bail!(
                "the native h26x encoders take 8-bit 4:2:0 only, got {:?}. Convert with the \
                 colorspace filter before the encoder, or use a hardware backend for this format.",
                frame.format
            );
        }
        if frame.width != self.width || frame.height != self.height {
            bail!(
                "frame is {}x{} but the encoder was configured for {}x{}",
                frame.width,
                frame.height,
                self.width,
                self.height
            );
        }
        let want = self.inner.frame_bytes();
        // A frame buffer may carry padding after the planes; the encoder wants
        // exactly its three planes, so hand it that prefix and no more.
        if frame.data.len() < want {
            bail!(
                "frame buffer is {} bytes, too short for {}x{} 4:2:0 ({} expected)",
                frame.data.len(),
                self.width,
                self.height,
                want
            );
        }
        self.pts.push(frame.pts);
        let units = self
            .inner
            .push(&frame.data[..want])
            .with_context(|| format!("the native {:?} encoder refused a frame", self.codec))?;
        self.collect(units)
    }

    fn flush(&mut self) -> Result<()> {
        let units = self
            .inner
            .flush()
            .with_context(|| format!("the native {:?} encoder failed to flush", self.codec))?;
        self.collect(units)
    }

    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        Ok(self.ready.pop_front())
    }

    fn force_keyframe_next(&mut self) -> Result<()> {
        // Supported, which matters: the chunked path discards a lead-in and
        // needs the first kept frame promoted to an IDR or the chunk will not
        // stand alone.
        self.inner.force_idr();
        Ok(())
    }

    /// Rebuild the inner encoder from its own configuration.
    ///
    /// A rebuild *is* the reset here, and it is the cheaper of the two ways
    /// to get one. The native encoders own no threads, no device and no
    /// surface ring — the decoders have the worker pool, the encoders do not
    /// — so construction is a few derived tables (geometry, the intra
    /// kernels) and empty vectors: measured at 7 us (H.264) / 0.6 us (H.265) for a
    /// 640x360 session (`tests/h26x_sw_reset.rs`), against tens of
    /// milliseconds of encode for the shortest chunk the ladder makes. A reset that instead walked the encoder's
    /// state clearing references, the scheduler, `frame_num`, `idr_pic_id`
    /// and the rate ledger would save nothing measurable and add a second
    /// path to the "fresh stream" invariant that `new` already owns.
    ///
    /// What is *not* rebuilt is this wrapper's identity: the caller's
    /// session pool keeps the `Box<dyn Encoder>` and its counters see a
    /// reuse, which is what makes the software tier behave like the hardware
    /// ones under the same pool.
    fn reset(&mut self) -> Result<()> {
        self.inner = Self::build_inner(self.codec, &self.cfg)?;
        self.pts.clear();
        self.ready.clear();
        tracing::debug!(
            event = "h26x_sw.reset",
            codec = ?self.codec,
            "native h26x session reset (inner encoder rebuilt; the face and its pool slot survive)"
        );
        Ok(())
    }
}
