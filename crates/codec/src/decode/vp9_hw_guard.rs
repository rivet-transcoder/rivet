//! A guard in front of a hardware VP9 decoder: it watches the stream for
//! what that decoder cannot do, and hands the stream to rivet's own decoder
//! — from the last key frame, without a seam — the moment it appears.
//!
//! # Why
//!
//! A hardware VP9 decoder that takes a stream can still decode parts of it
//! wrongly, and silently. Measured on the AMF decoder (Ryzen 9 9950X iGPU,
//! `tests/hw_vpx_decode.rs`, against the WebM project's test vectors):
//!
//! - a `show_existing_frame` packet produces no picture: a stream that
//!   re-shows a reference comes out with frames missing
//!   (`vp90-2-10-show-existing-frame2`: 8 of 16);
//! - a frame whose size differs from the first key frame's — a resize at a
//!   key frame, or an inter frame predicting from scaled references —
//!   comes out at the first size (`vp90-2-05-resize`,
//!   `vp90-2-13-mv-with-scaling`).
//!
//! Everything else it decodes bit-exact. Both features are rare in files
//! (libvpx uses neither by default) and both are visible in the
//! uncompressed header before a decoder sees the frame
//! (`crate::vp9_header`), so the guard can act on them exactly when they
//! occur rather than refusing VP9 on the hardware altogether.
//!
//! # How
//!
//! The guard keeps every packet since the last key frame. When a packet
//! needs what the hardware lacks, it drains the hardware (every picture it
//! owes for what it was given), builds rivet's own decoder, replays the
//! kept packets and the new one into it, drops the pictures the hardware
//! already returned for them, and from then on decodes in software. VP9
//! fixes the reconstruction bit-exactly, so the pictures on either side of
//! the switch are the ones a single decoder would have made. Output
//! timestamps are renumbered by the guard, 0, 1, 2, … across the switch.
//!
//! What each vendor's decoder is trusted with is its [`Vp9HwPolicy`]: only
//! what a test has shown it decodes bit-exact. Untested hardware is trusted
//! with neither feature — a switch costs speed, never a wrong picture.

use std::collections::VecDeque;

use anyhow::{Result, bail};

use super::{Decoder, StreamInfo};
use crate::frame::VideoFrame;
use crate::vp9_header::{self, CodedFrame, FrameHeader, RefSizes};

/// What a hardware VP9 decoder has been shown to decode bit-exactly, beyond
/// streams of one frame size coded without any of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vp9HwPolicy {
    /// `show_existing_frame` packets produce their picture.
    pub show_existing: bool,
    /// Frames whose size differs from the first key frame's (resizes,
    /// scaled references) come out at their own size, correctly.
    pub size_changes: bool,
    /// Frames coded with `error_resilient_mode`.
    pub error_resilient: bool,
    /// Frames coded with `segmentation_enabled`.
    pub segmentation: bool,
    /// `intra_only` frames.
    pub intra_only: bool,
    /// Profile 2, 10-bit 4:2:0.
    pub ten_bit: bool,
    /// The smallest frame the decoder is documented to take.
    pub min_size: (u32, u32),
    /// The largest.
    pub max_size: (u32, u32),
    /// The most frames one packet may carry: a superframe of more goes to
    /// rivet's own decoder. Two — a hidden frame and the frame that shows —
    /// is the usual alt-ref shape and decodes bit-exact on AMF; the
    /// "big superframe" stress vectors (eight frames in one packet) hung the
    /// AMD iGPU's video engine (LiveKernelEvent 141 / a2000002) after
    /// `AMF_RESOLUTION_CHANGED` and three wrong pictures.
    pub max_frames_per_packet: usize,
}

impl Vp9HwPolicy {
    /// None of the features; 8- and 10-bit at any size. The vendor
    /// policies below narrow the sizes to what each vendor documents.
    pub const BASELINE: Self = Self {
        show_existing: false,
        size_changes: false,
        error_resilient: false,
        segmentation: false,
        intra_only: false,
        ten_bit: true,
        min_size: (1, 1),
        max_size: (u32::MAX, u32::MAX),
        max_frames_per_packet: 2,
    };

    /// Which of `header`'s features this policy does not trust, by name.
    pub fn untrusted(&self, header: &CodedFrame) -> Option<&'static str> {
        if header.error_resilient && !self.error_resilient {
            Some("error_resilient_mode")
        } else if header.segmentation && !self.segmentation {
            Some("segmentation")
        } else if header.intra_only && !self.intra_only {
            Some("an intra_only frame")
        } else {
            None
        }
    }
}

/// AMF — used only with `RIVET_AMF_VP9=1` (the dispatcher's `amf_takes`:
/// the iGPU's video engine timed out during these runs, once on input
/// nothing here could screen). Measured on a Ryzen 9 9950X iGPU against the WebM project's VP9
/// vectors (`tests/hw_vpx_decode.rs`): no feature trusted, and at most a
/// hidden frame plus the frame that shows in one packet. A
/// `show_existing_frame` produces no picture; frames of another size come out
/// at the first size; every stream with segmentation enabled decodes wrongly
/// from its first segmented frame (`vp90-2-09-aq2`, `vp90-2-15-segkey*`,
/// `vp90-2-19-skip-01`, the 24 `vp90-2-02-size-*-05-resize-*`); `intra_only`
/// frames are answered `AMF_RESOLUTION_CHANGED`; an eight-frame superframe
/// (`vp90-2-07-frame_parallel_big_superframe`) was answered
/// `AMF_RESOLUTION_CHANGED` after three wrong pictures, and the video engine
/// timed out (LiveKernelEvent 141). Error-resilient streams without those
/// decoded bit-exact in the vector runs, but are not trusted until shown safe
/// on their own. 8- and 10-bit 4:2:0 up to 8192x8192 — "VP9 8,10b: 8K" for
/// every VCN in AMD's AMF wiki table (GPU and APU HW Features and Support) —
/// and from 16x16, the smallest size the decoder is initialised at here.
pub const AMF_POLICY: Vp9HwPolicy =
    Vp9HwPolicy { min_size: (16, 16), max_size: (8192, 8192), ..Vp9HwPolicy::BASELINE };
/// QSV: no feature trusted until a run on Intel hardware shows otherwise.
/// Up to 16384x16384 for 8- and 10-bit VP9 decode on DG2 / MTL and later
/// (Intel media-driver `docs/media_features.md`); 16x16 at least.
pub const QSV_POLICY: Vp9HwPolicy =
    Vp9HwPolicy { min_size: (16, 16), max_size: (16384, 16384), ..Vp9HwPolicy::BASELINE };
/// NVDEC: never run on NVIDIA hardware for this project, so no feature
/// trusted. VP9 from 128x128 to 8192x8192 (NVDEC Programming Guide, NVDEC
/// capabilities); 10-bit where `cuvidGetDecoderCaps` says so, which the
/// decoder asks before it creates a session.
pub const NVDEC_POLICY: Vp9HwPolicy =
    Vp9HwPolicy { min_size: (128, 128), max_size: (8192, 8192), ..Vp9HwPolicy::BASELINE };

/// The most bytes of packets kept since the last key frame. A key frame
/// interval this large is unusual; past it the guard stops being able to
/// switch, and a stream that then needs a switch fails by name instead of
/// decoding wrongly.
const KEEP_LIMIT: usize = 256 << 20;

/// Builds a fresh hardware decoder for a stream of the given shape: what the
/// guard restarts the hardware with at a key frame that changes the size.
pub type Rebuild = Box<dyn FnMut(&StreamInfo) -> Result<Box<dyn Decoder>> + Send>;

/// See the module docs.
pub struct Vp9HardwareGuard {
    label: &'static str,
    info: StreamInfo,
    policy: Vp9HwPolicy,
    hw: Option<Box<dyn Decoder>>,
    sw: Option<Box<dyn Decoder>>,
    refs: RefSizes,
    /// The first key frame's size: what the hardware decodes at.
    stream_size: Option<(u32, u32)>,
    /// The size and depth the hardware decoder was set up for, from the
    /// container (`StreamInfo`): a stream that is not what it was told is
    /// not handed to it.
    init_size: Option<(u32, u32)>,
    init_ten_bit: bool,
    /// Packets from the last key frame on, pushed to the hardware.
    kept: Vec<Vec<u8>>,
    kept_bytes: usize,
    /// `kept` outgrew [`KEEP_LIMIT`] since the last key frame.
    kept_overflowed: bool,
    /// Pictures shown by the packets before `kept[0]`.
    shown_before_kept: u64,
    /// Pictures shown by every packet pushed to the hardware.
    shown_total: u64,
    /// Pictures the hardware has returned.
    hw_frames: u64,
    /// The size of each picture the hardware owes, in order, from the
    /// headers: what its output is cropped to (a surface rounds an odd
    /// size up — AMF returns 352x288 for a 351x287 stream).
    owed_sizes: VecDeque<Option<(u32, u32)>>,
    ready: VecDeque<VideoFrame>,
    next_pts: u64,
    finished: bool,
    /// See [`Self::with_rebuild`].
    rebuild: Option<Rebuild>,
    /// Hardware restarts at a resizing key frame so far.
    restarts: u32,
}

impl Vp9HardwareGuard {
    /// Guard `hw`, a hardware decoder for the VP9 stream `info`.
    pub fn new(label: &'static str, hw: Box<dyn Decoder>, info: StreamInfo, policy: Vp9HwPolicy) -> Self {
        let init_size = (info.width > 0 && info.height > 0).then_some((info.width, info.height));
        let init_ten_bit = info.pixel_format == crate::frame::PixelFormat::Yuv420p10le;
        Self {
            label,
            info,
            policy,
            hw: Some(hw),
            sw: None,
            refs: RefSizes::default(),
            stream_size: None,
            init_size,
            init_ten_bit,
            kept: Vec::new(),
            kept_bytes: 0,
            kept_overflowed: false,
            shown_before_kept: 0,
            shown_total: 0,
            hw_frames: 0,
            owed_sizes: VecDeque::new(),
            ready: VecDeque::new(),
            next_pts: 0,
            finished: false,
            rebuild: None,
            restarts: 0,
        }
    }

    /// Let the guard restart the hardware at a key frame of a new size
    /// rather than leave it: the old decoder is drained of every picture it
    /// owes and a new one is built for the new size — the "Drain / Terminate
    /// / Init" AMF's `core/Result.h` prescribes for `AMF_RESOLUTION_CHANGED`,
    /// done before the decoder can be handed the frame. A size change at an
    /// inter frame (scaled references) is not a restart: it still goes to
    /// rivet's own decoder unless the policy trusts size changes.
    pub fn with_rebuild(mut self, rebuild: Rebuild) -> Self {
        self.rebuild = Some(rebuild);
        self
    }

    /// Hardware restarts at a resizing key frame so far.
    pub fn restarts(&self) -> u32 {
        self.restarts
    }

    /// Whether the guard has handed the stream to rivet's own decoder.
    pub fn switched(&self) -> bool {
        self.sw.is_some()
    }

    fn emit(&mut self, mut frame: VideoFrame) {
        frame.pts = self.next_pts;
        self.next_pts += 1;
        self.ready.push_back(frame);
    }

    fn pull_hw(&mut self) -> Result<()> {
        while let Some(hw) = self.hw.as_mut() {
            let Some(frame) = hw.decode_next()? else { break };
            self.hw_frames += 1;
            let frame = match self.owed_sizes.pop_front().flatten() {
                Some((w, h)) => crop(frame, w, h),
                None => frame,
            };
            self.emit(frame);
        }
        Ok(())
    }

    fn pull_sw(&mut self) -> Result<()> {
        while let Some(sw) = self.sw.as_mut() {
            let Some(frame) = sw.decode_next()? else { break };
            self.emit(frame);
        }
        Ok(())
    }

    /// What `packet` asks of the decoder that this hardware is not trusted
    /// with, if anything — read before anything decodes it. Updates the
    /// reference sizes and the key-frame bookkeeping either way.
    #[allow(clippy::type_complexity)]
    fn needs_software(&mut self, packet: &[u8]) -> (Option<String>, Vec<Option<(u32, u32)>>, Option<(u32, u32)>) {
        let mut why = None;
        let mut shown = Vec::new();
        let mut restart = None;
        let frames = vp9::superframe::split(packet);
        if frames.len() > self.policy.max_frames_per_packet {
            why = Some(format!(
                "a superframe of {} frames (this decoder takes at most {})",
                frames.len(),
                self.policy.max_frames_per_packet
            ));
        }
        for (i, frame) in frames.iter().enumerate() {
            let Some(header) = vp9_header::peek(frame) else { continue };
            if let FrameHeader::Coded(c) = &header
                && c.key
                && i == 0
            {
                // A new random-access point: what came before is no longer
                // needed for a replay.
                self.kept.clear();
                self.kept_bytes = 0;
                self.kept_overflowed = false;
                self.shown_before_kept = self.shown_total;
            }
            match header {
                FrameHeader::ShowExisting { slot } => {
                    shown.push(self.refs.slot(slot));
                    if !self.policy.show_existing && why.is_none() {
                        why = Some(format!("show_existing_frame (slot {slot})"));
                    }
                }
                FrameHeader::Coded(c) => {
                    if why.is_none()
                        && let Some(feature) = self.policy.untrusted(&c)
                    {
                        why = Some(feature.to_string());
                    }
                    if why.is_none()
                        && let Some(color) = c.color
                    {
                        why = self.format_refusal(c.profile, color);
                    }
                    let size = self.refs.apply(&header);
                    // A key frame opening the packet at a new size, on a
                    // guard that can rebuild the hardware: a restart.
                    if c.key
                        && i == 0
                        && why.is_none()
                        && self.rebuild.is_some()
                        && let (Some(new), Some(old)) = (size, self.stream_size)
                        && new != old
                        && self.size_refusal(new, false).is_none()
                    {
                        self.stream_size = Some(new);
                        self.init_size = Some(new);
                        restart = Some(new);
                    }
                    if why.is_none()
                        && let Some(size) = size
                    {
                        why = self.size_refusal(size, c.key && self.stream_size.is_none());
                    }
                    if c.show {
                        shown.push(size);
                    }
                    if c.key && self.stream_size.is_none() {
                        self.stream_size = size;
                    }
                    if let (Some(size), Some(first)) = (size, self.stream_size)
                        && size != first
                        && !self.policy.size_changes
                        && why.is_none()
                    {
                        why = Some(format!(
                            "a {}x{} frame in a stream that began at {}x{}",
                            size.0, size.1, first.0, first.1
                        ));
                    }
                }
            }
        }
        (why, shown, restart)
    }

    /// Drain the hardware of what it owes and build a new decoder for a
    /// stream of `size`: see [`Self::with_rebuild`].
    fn restart(&mut self, size: (u32, u32)) -> Result<()> {
        if let Some(mut old) = self.hw.take() {
            old.finish()?;
            while let Some(frame) = old.decode_next()? {
                self.hw_frames += 1;
                let frame = match self.owed_sizes.pop_front().flatten() {
                    Some((w, h)) => crop(frame, w, h),
                    None => frame,
                };
                self.emit(frame);
            }
        }
        let mut info = self.info.clone();
        (info.width, info.height) = size;
        let rebuild = self.rebuild.as_mut().expect("restart only with a rebuild");
        self.hw = Some(rebuild(&info)?);
        self.restarts += 1;
        tracing::info!(
            decoder = self.label,
            width = size.0,
            height = size.1,
            "VP9 key frame at a new size: hardware decoder restarted at it"
        );
        Ok(())
    }

    /// Why a frame of this format is not handed to the hardware, if it is
    /// not: anything but 4:2:0, 12 bits, or a depth other than the one the
    /// decoder was set up for.
    fn format_refusal(&self, profile: u8, color: vp9_header::ColorConfig) -> Option<String> {
        if color.rgb || color.subsampling != (1, 1) {
            return Some(format!("profile {profile}: not 4:2:0"));
        }
        match color.bit_depth {
            8 if !self.init_ten_bit => None,
            10 if self.init_ten_bit && self.policy.ten_bit => None,
            depth => Some(format!(
                "a {depth}-bit frame on a decoder set up for {} bits",
                if self.init_ten_bit { 10 } else { 8 }
            )),
        }
    }

    /// Why a frame of `size` is not handed to the hardware, if it is not:
    /// outside the vendor's documented range, or — for the stream's first
    /// key frame — not the size the decoder was set up for.
    fn size_refusal(&self, size: (u32, u32), first_key: bool) -> Option<String> {
        let (min, max) = (self.policy.min_size, self.policy.max_size);
        if size.0 < min.0 || size.1 < min.1 || size.0 > max.0 || size.1 > max.1 {
            return Some(format!(
                "a {}x{} frame, outside the {}x{} to {}x{} this decoder takes",
                size.0, size.1, min.0, min.1, max.0, max.1
            ));
        }
        match self.init_size {
            Some(init) if first_key && size != init => Some(format!(
                "a stream that opens at {}x{} while its container says {}x{}",
                size.0, size.1, init.0, init.1
            )),
            _ => None,
        }
    }

    /// Hand the stream to rivet's own decoder: see the module docs. `kept`
    /// already holds the packet that asked for it.
    fn switch(&mut self, why: &str) -> Result<()> {
        if self.kept_overflowed {
            bail!(
                "the {} VP9 decoder cannot decode this stream ({why}), and more than {} MiB has passed since \
                 its last key frame, too much to hand to the software decoder",
                self.label,
                KEEP_LIMIT >> 20
            );
        }
        // Everything the hardware owes for what it was given — as far as it
        // can still give it: after a hardware failure, what it cannot is
        // replayed from the kept packets like the rest.
        if let Some(mut hw) = self.hw.take() {
            match hw.finish() {
                Ok(()) => loop {
                    match hw.decode_next() {
                        Ok(Some(frame)) => {
                            self.hw_frames += 1;
                            let frame = match self.owed_sizes.pop_front().flatten() {
                                Some((w, h)) => crop(frame, w, h),
                                None => frame,
                            };
                            self.emit(frame);
                        }
                        Ok(None) => break,
                        Err(e) => {
                            tracing::debug!(error = %format!("{e:#}"), "hardware VP9 decoder failed while draining for a switch");
                            break;
                        }
                    }
                },
                Err(e) => tracing::debug!(error = %format!("{e:#}"), "hardware VP9 decoder failed to drain for a switch"),
            }
        }
        let mut sw: Box<dyn Decoder> = Box::new(super::vp9_sw::Vp9Decoder::new(self.info.clone())?);
        let mut replayed = Vec::new();
        for packet in &self.kept {
            sw.push_sample(packet)?;
            while let Some(frame) = sw.decode_next()? {
                replayed.push(frame);
            }
        }
        // The hardware has already returned the pictures of the kept
        // packets it could decode; the rest are new.
        let already = (self.hw_frames.saturating_sub(self.shown_before_kept) as usize).min(replayed.len());
        tracing::info!(
            decoder = self.label,
            reason = why,
            replayed_packets = self.kept.len(),
            pictures_already_out = already,
            "the hardware VP9 decoder cannot decode what comes next; continuing in rivet's own decoder from the last key frame"
        );
        for frame in replayed.into_iter().skip(already) {
            self.emit(frame);
        }
        self.kept = Vec::new();
        self.kept_bytes = 0;
        if self.finished {
            sw.finish()?;
        }
        self.sw = Some(sw);
        self.pull_sw()
    }
}

impl Vp9HardwareGuard {
    /// The hardware failed outright. Its frames so far are good (the
    /// features it mishandles silently are caught before it sees them), so
    /// the stream continues in software from the last key frame, as for a
    /// feature it is not trusted with.
    fn recover(&mut self, e: anyhow::Error) -> Result<()> {
        if self.sw.is_some() {
            return Err(e);
        }
        self.switch(&format!("the hardware decoder failed: {e:#}"))
    }
}

impl Decoder for Vp9HardwareGuard {
    fn stream_info(&self) -> &StreamInfo {
        match (&self.hw, &self.sw) {
            (Some(hw), _) => hw.stream_info(),
            (None, Some(sw)) => sw.stream_info(),
            (None, None) => &self.info,
        }
    }

    fn push_sample(&mut self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        if let Some(sw) = self.sw.as_mut() {
            sw.push_sample(data)?;
            return self.pull_sw();
        }
        let (why, shown, restart) = self.needs_software(data);
        if self.kept_bytes + data.len() > KEEP_LIMIT {
            self.kept_overflowed = true;
            self.kept = Vec::new();
            self.kept_bytes = 0;
        } else if !self.kept_overflowed {
            self.kept.push(data.to_vec());
            self.kept_bytes += data.len();
        }
        if let Some(why) = why {
            return self.switch(&why);
        }
        if let Some(size) = restart
            && let Err(e) = self.restart(size)
        {
            return self.recover(e);
        }
        // A packet shows at most one picture (VP9 Annex B); a packet the
        // headers could not be read from is counted as the decoder will.
        if !shown.is_empty() || vp9_header::packet_shows(data) {
            self.shown_total += 1;
            self.owed_sizes.push_back(shown.last().copied().flatten());
        }
        let pushed = self.hw.as_mut().expect("hardware until the switch").push_sample(data);
        match pushed.and_then(|()| self.pull_hw()) {
            Ok(()) => Ok(()),
            Err(e) => self.recover(e),
        }
    }

    fn finish(&mut self) -> Result<()> {
        self.finished = true;
        if let Some(sw) = self.sw.as_mut() {
            sw.finish()?;
            return self.pull_sw();
        }
        let finished = match self.hw.as_mut() {
            Some(hw) => hw.finish(),
            None => Ok(()),
        };
        match finished.and_then(|()| self.pull_hw()) {
            Ok(()) => Ok(()),
            Err(e) => self.recover(e),
        }
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        if self.ready.is_empty() {
            if self.sw.is_some() {
                self.pull_sw()?;
            } else if let Err(e) = self.pull_hw() {
                self.recover(e)?;
            }
        }
        Ok(self.ready.pop_front())
    }
}

/// The top-left `w` x `h` of a planar 4:2:0 frame (8- or 10-bit); any
/// other frame, or one not larger than that, unchanged.
fn crop(frame: VideoFrame, w: u32, h: u32) -> VideoFrame {
    use crate::frame::PixelFormat;
    let bytes = match frame.format {
        PixelFormat::Yuv420p => 1usize,
        PixelFormat::Yuv420p10le => 2,
        _ => return frame,
    };
    if (w, h) == (frame.width, frame.height) || w > frame.width || h > frame.height {
        return frame;
    }
    let (sw, sh) = (frame.width as usize, frame.height as usize);
    let (dw, dh) = (w as usize, h as usize);
    let (scw, sch) = (sw.div_ceil(2), sh.div_ceil(2));
    let (dcw, dch) = (dw.div_ceil(2), dh.div_ceil(2));
    if frame.data.len() < (sw * sh + 2 * scw * sch) * bytes {
        return frame;
    }
    let mut out = Vec::with_capacity((dw * dh + 2 * dcw * dch) * bytes);
    for row in 0..dh {
        out.extend_from_slice(&frame.data[row * sw * bytes..(row * sw + dw) * bytes]);
    }
    for plane in 0..2 {
        let base = (sw * sh + plane * scw * sch) * bytes;
        for row in 0..dch {
            out.extend_from_slice(&frame.data[base + row * scw * bytes..base + (row * scw + dcw) * bytes]);
        }
    }
    VideoFrame::new(out.into(), w, h, frame.format, frame.color_space, frame.pts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{ColorSpace, PixelFormat};

    fn info() -> StreamInfo {
        StreamInfo {
            codec: "vp9".into(),
            width: 0,
            height: 0,
            frame_rate: 30.0,
            duration: 0.0,
            pixel_format: PixelFormat::Yuv420p,
            color_space: ColorSpace::Bt709,
            total_frames: 0,
            bitrate: 0,
            color_metadata: Default::default(),
        }
    }

    /// A stand-in "hardware" decoder that is rivet's own decoder but drops
    /// every `show_existing_frame` picture and reports how many packets it
    /// saw — the AMF behaviour, without the GPU.
    struct DropsShowExisting {
        inner: crate::decode::vp9_sw::Vp9Decoder,
    }

    impl Decoder for DropsShowExisting {
        fn stream_info(&self) -> &StreamInfo {
            self.inner.stream_info()
        }
        fn push_sample(&mut self, data: &[u8]) -> Result<()> {
            let se = vp9::superframe::split(data)
                .iter()
                .all(|f| matches!(vp9_header::peek(f), Some(FrameHeader::ShowExisting { .. })));
            if se {
                return Ok(());
            }
            self.inner.push_sample(data)
        }
        fn finish(&mut self) -> Result<()> {
            self.inner.finish()
        }
        fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
            self.inner.decode_next()
        }
    }

    fn all(mut d: Box<dyn Decoder>, packets: &[Vec<u8>]) -> Vec<VideoFrame> {
        let mut out = Vec::new();
        for p in packets {
            d.push_sample(p).unwrap();
            while let Some(f) = d.decode_next().unwrap() {
                out.push(f);
            }
        }
        d.finish().unwrap();
        while let Some(f) = d.decode_next().unwrap() {
            out.push(f);
        }
        out
    }

    /// The committed `show_existing_frame` vector through a decoder that
    /// drops those pictures: the guard switches at the first one and the
    /// output is every picture rivet's own decoder makes, in order,
    /// numbered 0.. .
    #[test]
    fn a_show_existing_frame_switches_without_a_seam() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../vp9/tests/data/vp90-2-10-show-existing-frame2.webm");
        let Ok(data) = std::fs::read(&path) else {
            eprintln!("SKIPPED: {} not present", path.display());
            return;
        };
        let demuxed = container::demux::demux(&data).expect("demux");
        let packets: Vec<Vec<u8>> = demuxed.samples.iter().map(|s| s.to_vec()).collect();
        let want = all(Box::new(crate::decode::vp9_sw::Vp9Decoder::new(info()).unwrap()), &packets);
        let lossy = DropsShowExisting { inner: crate::decode::vp9_sw::Vp9Decoder::new(info()).unwrap() };
        let unguarded = all(
            Box::new(DropsShowExisting { inner: crate::decode::vp9_sw::Vp9Decoder::new(info()).unwrap() }),
            &packets,
        );
        assert!(unguarded.len() < want.len(), "the stand-in drops pictures");
        let guard = Vp9HardwareGuard::new("test", Box::new(lossy), info(), Vp9HwPolicy::BASELINE);
        let got = all(Box::new(guard), &packets);
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.pts, i as u64);
            assert!(g.data[..] == w.data[..], "picture {i}");
        }
    }

    /// Trusted with the feature, the guard never switches; not trusted, it
    /// does, at the first packet.
    #[test]
    fn a_trusted_feature_stays_on_the_hardware() {
        let (w, h) = (64u32, 48u32);
        let mut enc = vp9::Encoder::new(vp9::Config::new(w, h));
        let frame = vp9::Frame::new(w, h, 8, vp9::ChromaFormat::Yuv420);
        let packets: Vec<Vec<u8>> = (0..4).map(|_| enc.encode(&frame).unwrap()).collect();
        let hw = Box::new(crate::decode::vp9_sw::Vp9Decoder::new(info()).unwrap());
        // rivet's own encoder codes its inter frames error-resilient.
        let policy = Vp9HwPolicy { error_resilient: true, ..Vp9HwPolicy::BASELINE };
        let mut guard = Vp9HardwareGuard::new("test", hw, info(), policy);
        for p in &packets {
            guard.push_sample(p).unwrap();
        }
        guard.finish().unwrap();
        assert!(!guard.switched());
        let mut n = 0;
        while let Some(f) = guard.decode_next().unwrap() {
            assert_eq!(f.pts, n);
            n += 1;
        }
        assert_eq!(n, 4);

        let hw = Box::new(crate::decode::vp9_sw::Vp9Decoder::new(info()).unwrap());
        let mut guard = Vp9HardwareGuard::new("test", hw, info(), Vp9HwPolicy::BASELINE);
        for p in &packets {
            guard.push_sample(p).unwrap();
        }
        assert!(guard.switched());
    }

    /// A stream whose first key frame is not the size the container gave,
    /// or is outside the vendor's range, never reaches the hardware.
    #[test]
    fn a_stream_that_is_not_what_the_decoder_was_set_up_for_stays_in_software() {
        let (w, h) = (64u32, 48u32);
        let mut enc = vp9::Encoder::new(vp9::Config::new(w, h));
        let key = enc.encode(&vp9::Frame::new(w, h, 8, vp9::ChromaFormat::Yuv420)).unwrap();
        let trusting = Vp9HwPolicy { error_resilient: true, ..Vp9HwPolicy::BASELINE };
        let cases: [(u32, u32, PixelFormat, Vp9HwPolicy); 3] = [
            (320, 240, PixelFormat::Yuv420p, trusting),
            (64, 48, PixelFormat::Yuv420p10le, trusting),
            (64, 48, PixelFormat::Yuv420p, Vp9HwPolicy { min_size: (128, 128), ..trusting }),
        ];
        for (cw, ch, format, policy) in cases {
            let mut i = info();
            (i.width, i.height, i.pixel_format) = (cw, ch, format);
            let hw = Box::new(crate::decode::vp9_sw::Vp9Decoder::new(info()).unwrap());
            let mut guard = Vp9HardwareGuard::new("test", hw, i, policy);
            guard.push_sample(&key).unwrap();
            assert!(guard.switched(), "{cw}x{ch} {format:?} {policy:?}");
            let f = guard.decode_next().unwrap().expect("the picture, from software");
            assert_eq!((f.width, f.height, f.pts), (w, h, 0));
        }
        // What it was set up for: stays.
        let mut i = info();
        (i.width, i.height) = (w, h);
        let hw = Box::new(crate::decode::vp9_sw::Vp9Decoder::new(info()).unwrap());
        let mut guard = Vp9HardwareGuard::new("test", hw, i, trusting);
        guard.push_sample(&key).unwrap();
        assert!(!guard.switched());
    }

    #[test]
    fn crop_takes_the_top_left_of_each_plane() {
        // 4x4 → 3x3: luma rows of 3, chroma 2x2 out of 2x2.
        let data: Vec<u8> = (0..24).collect();
        let f = VideoFrame::new(data.into(), 4, 4, PixelFormat::Yuv420p, ColorSpace::Bt709, 7);
        let c = crop(f, 3, 3);
        assert_eq!((c.width, c.height, c.pts), (3, 3, 7));
        assert_eq!(&c.data[..], &[0, 1, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23][..]);
        // 10-bit, 4x2 → 2x2.
        let data: Vec<u8> = (0..24).collect();
        let f = VideoFrame::new(data.into(), 4, 2, PixelFormat::Yuv420p10le, ColorSpace::Bt709, 0);
        let c = crop(f, 2, 2);
        assert_eq!(&c.data[..], &[0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 20, 21][..]);
    }

    /// Two key frames of different sizes: with a rebuild, the hardware is
    /// restarted at the second and the stream never leaves it; without one,
    /// it goes to software at the second. Either way every picture is
    /// rivet's own decoder's.
    #[test]
    fn a_key_frame_at_a_new_size_restarts_the_hardware() {
        let trusting = Vp9HwPolicy { error_resilient: true, ..Vp9HwPolicy::BASELINE };
        let mut packets = Vec::new();
        for (w, h) in [(64u32, 48u32), (96, 64)] {
            let mut enc = vp9::Encoder::new(vp9::Config::new(w, h));
            let frame = vp9::Frame::new(w, h, 8, vp9::ChromaFormat::Yuv420);
            for _ in 0..3 {
                packets.push(enc.encode(&frame).unwrap());
            }
        }
        let want = all(Box::new(crate::decode::vp9_sw::Vp9Decoder::new(info()).unwrap()), &packets);
        assert_eq!(want.len(), 6);

        let built = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter = built.clone();
        let mut i = info();
        (i.width, i.height) = (64, 48);
        let hw = Box::new(crate::decode::vp9_sw::Vp9Decoder::new(info()).unwrap());
        let mut guard = Vp9HardwareGuard::new("test", hw, i.clone(), trusting).with_rebuild(Box::new(move |info| {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            assert_eq!((info.width, info.height), (96, 64));
            Ok(Box::new(crate::decode::vp9_sw::Vp9Decoder::new(info.clone())?) as Box<dyn Decoder>)
        }));
        let mut got = Vec::new();
        for p in &packets {
            guard.push_sample(p).unwrap();
            while let Some(f) = guard.decode_next().unwrap() {
                got.push(f);
            }
        }
        guard.finish().unwrap();
        while let Some(f) = guard.decode_next().unwrap() {
            got.push(f);
        }
        assert!(!guard.switched());
        assert_eq!(guard.restarts(), 1);
        assert_eq!(built.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(got.len(), 6);
        for (n, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!((g.width, g.height, g.pts), (w.width, w.height, n as u64));
            assert!(g.data[..] == w.data[..], "picture {n}");
        }

        let hw = Box::new(crate::decode::vp9_sw::Vp9Decoder::new(info()).unwrap());
        let mut guard = Vp9HardwareGuard::new("test", hw, i, trusting);
        for p in &packets {
            guard.push_sample(p).unwrap();
        }
        assert!(guard.switched());
    }

    /// A superframe of more frames than the policy takes never reaches the
    /// hardware: the guard switches at it, and the pictures are rivet's.
    #[test]
    fn a_big_superframe_goes_to_software() {
        let (w, h) = (64u32, 48u32);
        let mut enc = vp9::Encoder::new(vp9::Config::new(w, h));
        let frame = vp9::Frame::new(w, h, 8, vp9::ChromaFormat::Yuv420);
        let key = enc.encode(&frame).unwrap();
        let inter: Vec<Vec<u8>> = (0..3).map(|_| enc.encode(&frame).unwrap()).collect();
        // Three shown frames in one packet: not a stream a decoder shows
        // correctly, but enough to count frames — the guard must not hand
        // it to the hardware, whatever is in it.
        let big = vp9::superframe::join(&[&inter[0], &inter[1], &inter[2]]);
        let trusting = Vp9HwPolicy { error_resilient: true, ..Vp9HwPolicy::BASELINE };
        struct Refuses;
        impl Decoder for Refuses {
            fn stream_info(&self) -> &StreamInfo {
                unreachable!()
            }
            fn push_sample(&mut self, data: &[u8]) -> Result<()> {
                assert!(vp9::superframe::split(data).len() <= 2, "a big superframe reached the hardware");
                Ok(())
            }
            fn finish(&mut self) -> Result<()> {
                Ok(())
            }
            fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
                Ok(None)
            }
        }
        let mut guard = Vp9HardwareGuard::new("test", Box::new(Refuses), info(), trusting);
        guard.push_sample(&key).unwrap();
        assert!(!guard.switched());
        guard.push_sample(&big).unwrap();
        assert!(guard.switched());
    }
}
