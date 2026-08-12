//! rav1d — AV1 decode in software, as the last resort.
//!
//! The counterpart to `encode/rav1e_sw.rs`, and it exists for the same reason:
//! the hardware decoders need silicon and a driver, and when neither is there
//! the useful outcome is a slow decode rather than a diagnostic.
//!
//! It matters slightly more on the decode side than the encode side. A host
//! can be perfectly capable of *encoding* AV1 and unable to *decode* it —
//! NVDEC gained AV1 in Ampere, NVENC only in Ada — so "is the GPU new enough"
//! is two different questions and this answers the harder one for free.
//!
//! # Always built; the feature decides whether it is *reached*
//!
//! This module compiles unconditionally. The `rav1d-fallback` feature gates
//! whether [`create_decoder_on`](super::create_decoder_on) **falls back** here
//! once the hardware and FFmpeg tiers have declined — a policy question, not a
//! capability one, and the same reasoning as the encode side.
//!
//! It is tried last either way, and engagement is logged at `warn`.
//!
//! # Why this is FFI, when rav1d is a Rust crate
//!
//! rav1d is a line-by-line port of dav1d that deliberately exposes dav1d's **C
//! ABI** — `dav1d_open`, `dav1d_send_data`, `dav1d_get_picture` — rather than an
//! idiomatic Rust surface, so it can be a drop-in replacement for the C
//! library. There is no safe API to call; the unsafety is inherent to the
//! crate's design rather than something chosen here.
//!
//! That is also why it fits: every other backend in this crate (NVENC, NVDEC,
//! AMF, QSV) is hand-rolled FFI over a vendor ABI, and this is the same shape.
//! The unsafety is confined to this file, and each block says what it is
//! relying on.
//!
//! # Buffering
//!
//! `push_sample` hands one temporal unit to the decoder and immediately drains
//! whatever became available; `decode_next` serves from that queue. `EAGAIN` is
//! not an error in either direction — it is the decoder saying "more input" or
//! "collect what you have", and treating it as a failure is the classic way to
//! turn a working decoder into an intermittent one.

use std::ffi::{c_int, c_void};

use anyhow::Result;
use bytes::Bytes;

use rav1d::include::dav1d::data::Dav1dData;
use rav1d::include::dav1d::dav1d::Dav1dSettings;
use rav1d::include::dav1d::headers::DAV1D_PIXEL_LAYOUT_I420;
use rav1d::include::dav1d::picture::Dav1dPicture;

use super::Decoder;
use crate::frame::{ColorSpace, PixelFormat, StreamInfo, VideoFrame};

/// dav1d returns `-EAGAIN` for both "needs more data" and "nothing ready yet".
const EAGAIN: c_int = -11;

// The entry points, declared rather than imported.
//
// rav1d exports these as `#[no_mangle] extern "C"` symbols — they are dav1d's C
// ABI, and they are not re-exported as Rust paths, so a caller binds them the
// way a C consumer would. The context is opaque here for the same reason it is
// opaque in dav1d's own header: its concrete type is private to the crate, and
// a consumer only ever holds the pointer.
//
// `Dav1dResult` is `#[repr(transparent)]` over `c_int`, so returning `c_int` is
// ABI-identical and saves naming a type that carries no information here.
unsafe extern "C" {
    fn dav1d_default_settings(s: *mut Dav1dSettings);
    fn dav1d_open(c_out: *mut *mut c_void, s: *const Dav1dSettings) -> c_int;
    fn dav1d_send_data(c: *mut c_void, data: *mut Dav1dData) -> c_int;
    fn dav1d_get_picture(c: *mut c_void, out: *mut Dav1dPicture) -> c_int;
    fn dav1d_picture_unref(p: *mut Dav1dPicture);
    fn dav1d_data_create(buf: *mut Dav1dData, sz: usize) -> *mut u8;
    // Declared for completeness of the ABI surface, and deliberately unused:
    // dav1d's `flush` is a seek/reset that DISCARDS buffered pictures, not a
    // drain. Calling it at end-of-stream is what turned a five-frame decode
    // into zero frames. Left here, named and explained, so the next person
    // reaching for it finds the warning rather than the function.
    #[allow(dead_code)]
    fn dav1d_flush(c: *mut c_void);
    fn dav1d_close(c_out: *mut *mut c_void);
}

/// Software AV1 decoder.
pub struct Rav1dDecoder {
    /// Opaque decoder context. Null only between `dav1d_close` and drop.
    ctx: *mut c_void,
    info: StreamInfo,
    ready: std::collections::VecDeque<VideoFrame>,
    /// AV1 carries no container timestamps of its own, so frames are numbered
    /// in decode order. Monotonic from zero, which is what a muxer that is
    /// re-timestamping anyway expects.
    next_pts: u64,
}

// The decoder context is owned exclusively by this struct and never shared;
// `Decoder` requires `Send` and every backend in this crate is used the same
// way — constructed on one thread, driven from one thread.
unsafe impl Send for Rav1dDecoder {}

impl Rav1dDecoder {
    /// Build a decoder. `info` is what the container already knows; the
    /// dimensions are corrected from the first decoded picture, because a
    /// container header and a sequence header do disagree in the wild.
    pub fn new(info: StreamInfo) -> Result<Self> {
        let mut settings = std::mem::MaybeUninit::<Dav1dSettings>::uninit();
        let mut ctx: *mut c_void = std::ptr::null_mut();

        // SAFETY: `dav1d_default_settings` fully initialises the struct it is
        // given, and `dav1d_open` writes a context into `ctx` on success. Both
        // pointers are to live locals that outlive the calls.
        let rc = unsafe {
            dav1d_default_settings(settings.as_mut_ptr());
            let mut settings = settings.assume_init();
            // Let rav1d size its own thread pool. Pinning it to one thread
            // makes the fallback slower than it needs to be, and this tier is
            // already the slow path.
            settings.n_threads = 0;
            dav1d_open(&mut ctx as *mut _, &settings as *const _)
        };
        if rc < 0 {
            anyhow::bail!("rav1d refused to initialise (dav1d_open returned {rc})");
        }
        if ctx.is_null() {
            anyhow::bail!("rav1d reported success but produced no decoder context");
        }

        tracing::warn!(
            width = info.width,
            height = info.height,
            "no AV1 hardware decoder available — falling back to rav1d software decoding, \
             which is substantially slower than a fixed-function decoder"
        );

        Ok(Self {
            ctx,
            info,
            ready: std::collections::VecDeque::new(),
            next_pts: 0,
        })
    }

    /// Move every picture rav1d currently holds into `ready`.
    ///
    /// `at_eos` changes what `EAGAIN` means. Mid-stream it means "I need more
    /// data" and this returns. At end of stream it is also how dav1d *enters*
    /// drain mode — the first `EAGAIN` with no input pending flips the flag,
    /// and the frames still inside the reorder delay come out on the calls
    /// after it. Returning on the first one leaves them there: five frames in,
    /// one frame out, and nothing that looks like an error.
    ///
    /// So at EOS an `EAGAIN` is only believed once it has been seen twice with
    /// no picture in between.
    fn drain_inner(&mut self, at_eos: bool) -> Result<()> {
        let mut idle_eagains = 0;
        loop {
            let mut pic = std::mem::MaybeUninit::<Dav1dPicture>::zeroed();

            // SAFETY: the context came from `dav1d_open` and is still open;
            // `pic` is a live local. A zeroed `Dav1dPicture` is the documented
            // way to hand dav1d an output slot.
            let rc = unsafe { dav1d_get_picture(self.ctx, pic.as_mut_ptr()) };

            if rc == EAGAIN {
                if !at_eos {
                    return Ok(());
                }
                idle_eagains += 1;
                if idle_eagains >= 2 {
                    return Ok(());
                }
                continue;
            }
            idle_eagains = 0;
            if rc < 0 {
                anyhow::bail!("rav1d decode failed (dav1d_get_picture returned {rc})");
            }

            // SAFETY: a non-negative return means dav1d initialised the picture.
            let mut pic = unsafe { pic.assume_init() };
            let converted = self.convert(&pic);

            // Unref before propagating any error, or a failed conversion leaks
            // the picture's buffers for the life of the process.
            // SAFETY: `pic` was produced by `dav1d_get_picture` and is unrefed
            // exactly once.
            unsafe { dav1d_picture_unref(&mut pic as *mut _) };

            self.ready.push_back(converted?);
        }
    }

    /// Mid-stream drain: collect whatever is ready and return.
    fn drain(&mut self) -> Result<()> {
        self.drain_inner(false)
    }

    /// Copy a decoded picture into the crate's own tightly-packed planar frame.
    ///
    /// Row-wise, because dav1d hands back planes with their own stride and a
    /// flat copy shears the picture progressively down the frame — the same
    /// trap as the encode side, and one that looks like a decoder bug.
    fn convert(&mut self, pic: &Dav1dPicture) -> Result<VideoFrame> {
        if pic.p.layout != DAV1D_PIXEL_LAYOUT_I420 || pic.p.bpc != 8 {
            anyhow::bail!(
                "rav1d fallback handles 8-bit 4:2:0 only; this stream is {}-bit layout {}. \
                 Use a hardware decoder (NVDEC / AMF / QSV) for it.",
                pic.p.bpc,
                pic.p.layout
            );
        }

        let w = pic.p.w as usize;
        let h = pic.p.h as usize;
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));

        let mut out = Vec::with_capacity(w * h + 2 * cw * ch);
        // Planes 0..3 are Y, U, V. dav1d carries two strides: [0] for luma,
        // [1] shared by both chroma planes.
        for (plane, pw, ph, stride) in [
            (0usize, w, h, pic.stride[0] as usize),
            (1, cw, ch, pic.stride[1] as usize),
            (2, cw, ch, pic.stride[1] as usize),
        ] {
            let base = pic.data[plane]
                .ok_or_else(|| anyhow::anyhow!("rav1d returned a picture with no plane {plane}"))?
                .as_ptr() as *const u8;

            for row in 0..ph {
                // SAFETY: dav1d guarantees `stride * height` readable bytes per
                // plane, and `pw <= stride` for every layout it produces.
                let line = unsafe { std::slice::from_raw_parts(base.add(row * stride), pw) };
                out.extend_from_slice(line);
            }
        }

        // The sequence header is authoritative over whatever the container said.
        self.info.width = w as u32;
        self.info.height = h as u32;

        let pts = self.next_pts;
        self.next_pts += 1;

        Ok(VideoFrame::new(
            Bytes::from(out),
            w as u32,
            h as u32,
            PixelFormat::Yuv420p,
            ColorSpace::Bt709,
            pts,
        ))
    }
}

impl Decoder for Rav1dDecoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    fn push_sample(&mut self, data: &[u8]) -> Result<()> {
        let mut buf = std::mem::MaybeUninit::<Dav1dData>::zeroed();

        // SAFETY: `dav1d_data_create` allocates `len` bytes inside the data
        // struct and returns a pointer to them, or null on failure.
        let dst = unsafe { dav1d_data_create(buf.as_mut_ptr(), data.len()) };
        if dst.is_null() {
            anyhow::bail!("rav1d could not allocate {} bytes for a sample", data.len());
        }
        // SAFETY: `dst` points to exactly `data.len()` writable bytes, and the
        // two regions do not overlap.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()) };

        // SAFETY: `buf` was initialised by `dav1d_data_create` above.
        let mut buf = unsafe { buf.assume_init() };

        loop {
            // SAFETY: the context is open; `buf` is a live local. On a partial
            // consume dav1d updates `buf` in place, which is why it is passed
            // again on the retry rather than rebuilt.
            let rc = unsafe { dav1d_send_data(self.ctx, &mut buf as *mut _) };

            if rc == EAGAIN {
                // The decoder is full because pictures are waiting to be
                // collected. Draining and retrying is the documented handling;
                // failing here would strand a perfectly decodable stream.
                self.drain()?;
                continue;
            }
            if rc < 0 {
                anyhow::bail!("rav1d rejected a sample (dav1d_send_data returned {rc})");
            }
            break;
        }

        self.drain()
    }

    fn finish(&mut self) -> Result<()> {
        // Drain only. `dav1d_flush` is NOT a drain — it is dav1d's seek/reset,
        // and it DISCARDS every buffered picture. Calling it here decoded five
        // frames into nothing: the encoder produced packets, the decoder
        // accepted them, and zero came out, which reads like a decode failure
        // rather than the wrong function.
        //
        // dav1d's actual end-of-stream is to stop sending and keep pulling:
        // with no data pending it drains its reorder delay and then answers
        // EAGAIN for real.
        self.drain_inner(true)
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        Ok(self.ready.pop_front())
    }
}

impl Drop for Rav1dDecoder {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            // SAFETY: closed exactly once — `dav1d_close` takes the context by
            // out-pointer and clears it, and nothing else touches `ctx` after.
            unsafe { dav1d_close(&mut self.ctx as *mut _) };
        }
    }
}
