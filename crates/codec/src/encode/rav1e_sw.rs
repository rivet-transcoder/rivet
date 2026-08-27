//! rav1e — AV1 encode in software, as the last resort.
//!
//! Every other encoder in this crate needs silicon: NVENC wants Ada or newer,
//! AMF wants RDNA3, QSV wants Arc or Meteor Lake. On a host with none of them
//! the dispatch chain used to end in an error telling the operator to
//! reprovision, which is the right answer for a throughput-oriented fleet and
//! the wrong one for everything else — a laptop, a CI runner, a container on a
//! cloud instance with no GPU attached, or a machine whose driver failed to
//! load this morning.
//!
//! This tier exists so those cases produce a file instead of a diagnostic.
//!
//! # Always built; the feature decides whether it is *reached*
//!
//! This module compiles unconditionally, so it is always testable and a caller
//! that wants software encoding can always ask for it by name. The `rav1e`
//! feature gates something narrower and more useful: whether
//! [`select_encoder`](super::select_encoder) **falls back** here on its own
//! when every hardware backend has declined.
//!
//! That distinction is the point. Falling back silently is a policy decision,
//! not a capability one: rav1e at a speed preset that preserves quality is one
//! to two orders of magnitude slower than a hardware encoder, so a fleet that
//! quietly degraded into it would look like a capacity problem rather than the
//! missing driver it actually is. A workstation or a CI runner wants exactly
//! the opposite. The feature is how a build says which it is.
//!
//! It is tried last either way, and when it engages it says so at `warn`, once,
//! with the reason.
//!
//! # Threads
//!
//! rav1e is internally parallel and will use every core it is given. Left
//! unbounded that starves whatever else shares the box — in a job worker,
//! typically the other jobs. `threads` is therefore pinned to the encoder
//! config's own budget when one is set, and otherwise to the parallelism the
//! runtime reports, which is the container's CPU quota rather than the host's
//! core count when one is imposed.

use anyhow::{Context, Result};
use bytes::Bytes;

use super::{EncodedPacket, Encoder, EncoderConfig};
use crate::encode::tuning::rav1e_params;
use crate::frame::{PixelFormat, VideoFrame};

/// Software AV1 encoder.
pub struct Rav1eEncoder {
    ctx: rav1e::Context<u8>,
    width: u32,
    height: u32,
    /// rav1e reports frame order, not presentation timestamps, so the
    /// timestamps the caller gave us are queued and re-attached as packets
    /// come out. AV1 has no B-pyramid reordering in the configuration used
    /// here, so this stays first-in-first-out.
    pts_queue: std::collections::VecDeque<u64>,
    /// Set by `force_keyframe_next`, consumed by the next `send_frame`.
    force_key: bool,
}

impl Rav1eEncoder {
    /// Build an encoder for `config`.
    ///
    /// Fails rather than silently degrading when the frame format is not one
    /// rav1e can take. The caller's chain has already exhausted the hardware
    /// tiers by this point, so a clear error here is more useful than a
    /// picture with the chroma planes misread.
    pub fn new(config: EncoderConfig) -> Result<Self> {
        let p = rav1e_params(config.target, config.tier, config.width, config.height);

        // Zero means "decide for me". The runtime's answer respects a
        // container CPU quota where the host core count does not, which is the
        // difference between a well-behaved job worker and one that starves
        // everything sharing the box.
        let threads = if config.threads > 0 {
            config.threads
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        };

        let enc = rav1e::EncoderConfig {
            width: config.width as usize,
            height: config.height as usize,
            speed_settings: rav1e::prelude::SpeedSettings::from_preset(p.speed_preset),
            quantizer: p.quantizer,
            tile_rows: p.tile_rows,
            tile_cols: p.tile_cols,
            // 8-bit 4:2:0 only, matching what `send_frame` accepts. Widening
            // this means widening the plane copy with it; the two must not
            // drift apart.
            bit_depth: 8,
            chroma_sampling: rav1e::prelude::ChromaSampling::Cs420,
            // The caller's GOP. Left unset, rav1e keeps its own default (240
            // frames), so the 2 s cadence the ladder asked for came out as
            // one keyframe per *chunk*: invisible on HLS, where a segment is
            // a fresh encoder anyway, and wrong on the chunked single file,
            // whose chunks are ten GOPs long (found by the IDR-cadence gate,
            // 2026-08-27). Zero keeps rav1e's default, as the other tiers
            // treat an unset interval; `force_keyframe_next` still lands a
            // key frame wherever it is asked.
            min_key_frame_interval: 0,
            max_key_frame_interval: if config.keyframe_interval == 0 { 240 } else { u64::from(config.keyframe_interval) },
            ..Default::default()
        };

        let cfg = rav1e::Config::new()
            .with_encoder_config(enc)
            .with_threads(threads);

        let ctx: rav1e::Context<u8> = cfg
            .new_context()
            .context("rav1e rejected the encoder configuration")?;

        tracing::warn!(
            width = config.width,
            height = config.height,
            speed_preset = p.speed_preset,
            quantizer = p.quantizer,
            threads,
            "no AV1 encode silicon available — falling back to rav1e software encoding, \
             which is far slower than any hardware backend"
        );

        Ok(Self {
            ctx,
            width: config.width,
            height: config.height,
            pts_queue: std::collections::VecDeque::new(),
            force_key: false,
        })
    }

    /// Copy one plane into a rav1e plane.
    ///
    /// `copy_from_raw_u8` rather than writing `plane.data` directly. A rav1e
    /// `Plane` is not a bare pixel rectangle: it carries a padding border, and
    /// the visible top-left corner sits at `(xorigin, yorigin)` inside the
    /// allocation. Indexing from zero writes the whole frame into the border,
    /// which encodes cleanly and decodes to a uniform grey picture — no error
    /// anywhere, just no image.
    ///
    /// The API handles origin, stride and sample width together, which is
    /// three things not to get individually wrong.
    fn fill_plane(dst: &mut rav1e::prelude::Plane<u8>, src: &[u8], width: usize) {
        dst.copy_from_raw_u8(src, width, 1);
    }
}

impl Encoder for Rav1eEncoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        if frame.format != PixelFormat::Yuv420p {
            anyhow::bail!(
                "rav1e fallback encodes 8-bit 4:2:0 only, got {:?}. Convert with the colorspace \
                 filter before the encoder, or use a hardware backend for this format.",
                frame.format
            );
        }
        if frame.width != self.width || frame.height != self.height {
            anyhow::bail!(
                "frame is {}x{} but the encoder was configured for {}x{}",
                frame.width,
                frame.height,
                self.width,
                self.height
            );
        }

        let w = self.width as usize;
        let h = self.height as usize;
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));

        let luma = w * h;
        let chroma = cw * ch;
        if frame.data.len() < luma + 2 * chroma {
            anyhow::bail!(
                "frame buffer is {} bytes, too short for {}x{} 4:2:0 ({} expected)",
                frame.data.len(),
                w,
                h,
                luma + 2 * chroma
            );
        }

        let mut picture = self.ctx.new_frame();
        Self::fill_plane(&mut picture.planes[0], &frame.data[..luma], w);
        Self::fill_plane(&mut picture.planes[1], &frame.data[luma..luma + chroma], cw);
        Self::fill_plane(
            &mut picture.planes[2],
            &frame.data[luma + chroma..luma + 2 * chroma],
            cw,
        );

        let params = if std::mem::take(&mut self.force_key) {
            Some(rav1e::prelude::FrameParameters {
                frame_type_override: rav1e::prelude::FrameTypeOverride::Key,
                ..Default::default()
            })
        } else {
            None
        };

        self.ctx
            .send_frame((picture, params))
            .context("rav1e refused a frame")?;
        self.pts_queue.push_back(frame.pts);
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.ctx.flush();
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        // Loops, because `Encoded` is not "nothing to give" — it means rav1e
        // consumed work and has no *packet* yet, and the very next call may
        // produce one. Returning `None` for it ends the caller's
        // `while let Some(..)` drain, which silently truncated a five-frame
        // encode to one packet with no error anywhere.
        //
        // `None` is reserved for the two states that really mean "stop":
        // needing more input, and the end of a flushed stream.
        loop {
            match self.ctx.receive_packet() {
                Ok(pkt) => {
                    let is_keyframe = matches!(pkt.frame_type, rav1e::prelude::FrameType::KEY);
                    // Prefer the timestamp we were handed. rav1e's own `input_frameno`
                    // counts frames, and a caller working in anything other than
                    // frame numbers (a container writing 90 kHz ticks, say) would get
                    // a stream whose timing is silently wrong.
                    let pts = self.pts_queue.pop_front().unwrap_or(pkt.input_frameno);
                    return Ok(Some(EncodedPacket {
                        data: Bytes::from(pkt.data),
                        pts,
                        is_keyframe,
                    }));
                }
                // Work happened, no packet yet — ask again.
                Err(rav1e::prelude::EncoderStatus::Encoded) => continue,
                // The two that really mean stop: more input needed, or the end of
                // a flushed stream.
                Err(rav1e::prelude::EncoderStatus::NeedMoreData)
                | Err(rav1e::prelude::EncoderStatus::LimitReached) => return Ok(None),
                Err(e) => return Err(anyhow::anyhow!("rav1e encode failed: {e:?}")),
            }
        }
    }

    fn force_keyframe_next(&mut self) -> Result<()> {
        // Supported, which matters: the chunked path discards a lead-in and
        // needs the first kept frame promoted to a keyframe or the chunk will
        // not stand alone. Without this the fallback could not participate in
        // chunked encoding at all.
        self.force_key = true;
        Ok(())
    }

    // `reset` is deliberately left at the trait default (`ResetUnsupported`).
    // rav1e's `Context` has no restart: once flushed it is finished, and a
    // keyframe override does not clear its rate-control history or its
    // reference set. The caller's session pool sees the refusal by type and
    // rebuilds — the behaviour every chunk had before pooling existed.
}
