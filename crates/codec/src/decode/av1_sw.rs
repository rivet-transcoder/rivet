//! AV1 decode in software — this workspace's own decoder (`crates/av1`, the
//! rivet-av1 repository), written clean-room from the AV1 Bitstream &
//! Decoding Process Specification and bit-exact on all 244 AOM test vectors
//! and all 3 015 Argon conformance streams.
//!
//! # Where it sits
//!
//! Behind every hardware tier — NVDEC (Ampere and newer), AMF and QSV decode
//! AV1 in fixed-function silicon at hundreds of frames a second — and the
//! only software tier for AV1. Like the workspace's other own decoders (h26x,
//! VP8, VP9, MPEG-2, MPEG-4, ProRes) it is always compiled and always in the
//! chain: there is no feature to turn it on. (It replaced rav1d, which sat
//! behind a `rav1d-fallback` feature; that name is kept as a no-op alias.)
//!
//! It matters more on the decode side than the encode side: NVDEC gained AV1
//! in Ampere, NVENC only in Ada, so a host can encode AV1 in hardware and
//! still have no way to decode it.
//!
//! # Throughput, and its threads
//!
//! The decoder decodes the tiles of a frame in parallel and runs its
//! post-filters (loop filter, CDEF, loop restoration) in bands on several
//! threads, with AVX2 / NEON in its hottest kernels (CDEF, the inter
//! prediction filters, the inverse transforms, the loop filter): 60-70 MP/s on
//! one thread and 75-115 with four on the software AV1 encoder's own 720p / 1080p
//! output (the crate's figures; more tiles, more parallelism). This adapter
//! gives it [`decode_threads`] threads (`RIVET_AV1_DECODE_THREADS`, default
//! up to four) and takes the decode off the caller's thread as well: each
//! decoder runs on a worker thread of its own, a few temporal units ahead of
//! the caller, so the pipeline's colour conversion, scaling and encoding
//! overlap the decode rather than waiting for it.
//! `RIVET_AV1_DECODE_THREAD=0` decodes on the caller's thread instead.
//!
//! # Colour
//!
//! The sequence header's colour description (primaries, transfer, matrix,
//! range) and the HDR10 metadata OBUs (mastering display, content light
//! level) a stream carries replace what the container said in
//! [`stream_info`](Decoder::stream_info)'s `color_metadata`, as each frame
//! comes out.
//!
//! # Output
//!
//! Every layout and depth AV1 has: 4:2:0, 4:2:2 and 4:4:4 at 8, 10 and 12
//! bits, as the planar formats the native HEVC decoder already emits (the
//! decode pump normalises them for the encoders). Monochrome (4:0:0 — an AVIF
//! alpha plane, a greyscale still) comes out as 4:2:0 with neutral chroma:
//! the same picture, in a format every consumer reads. Chroma planes are
//! `ceil(w / 2)` wide for an odd width. Film grain is applied, as the
//! specification's output process does.

use std::collections::VecDeque;
use std::sync::mpsc;
use std::thread::JoinHandle;

use anyhow::{Context, Result, anyhow, bail};
use bytes::Bytes;

use super::Decoder;
use crate::frame::{ColorSpace, PixelFormat, StreamInfo, VideoFrame};

/// The codec labels the AV1 tier serves.
pub fn supports(codec_lower: &str) -> bool {
    matches!(codec_lower, "av1" | "av01")
}

/// How many temporal units the worker may hold before `push_sample` waits:
/// enough to keep it busy while the caller converts and encodes, few enough
/// that decoded 4K frames do not pile up.
const IN_FLIGHT: usize = 3;

/// What the worker sends back for each temporal unit.
type Decoded = std::result::Result<Vec<av1::Frame>, av1::Error>;

/// An AV1 decoder behind rivet's [`Decoder`] trait.
pub struct Av1Decoder {
    info: StreamInfo,
    engine: Engine,
    ready: VecDeque<VideoFrame>,
    /// AV1 carries no container timestamps of its own, so frames are numbered
    /// in output order. Monotonic from zero, which is what a muxer that is
    /// re-timestamping anyway expects.
    next_pts: u64,
}

enum Engine {
    /// Decoding on the caller's thread.
    Inline(Box<av1::Decoder>),
    /// Decoding on a worker; `samples` is `None` once `finish` closed it.
    Worker {
        samples: Option<mpsc::SyncSender<Vec<u8>>>,
        results: mpsc::Receiver<Decoded>,
        handle: Option<JoinHandle<()>>,
        /// Temporal units sent and not yet answered.
        pending: usize,
    },
}

/// Threads each decoder may use for tiles and post-filters:
/// `RIVET_AV1_DECODE_THREADS`, else up to four (the pipeline runs other work
/// beside the decode, and other decodes beside this one).
pub fn decode_threads() -> usize {
    std::env::var("RIVET_AV1_DECODE_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()).min(4))
}

fn worker_disabled() -> bool {
    matches!(
        std::env::var("RIVET_AV1_DECODE_THREAD").as_deref().map(str::to_ascii_lowercase).as_deref(),
        Ok("0" | "false" | "no" | "off")
    )
}

impl Av1Decoder {
    /// Build a decoder. `info` is what the container already knows; the
    /// dimensions and format are corrected from the first decoded picture,
    /// because a container header and a sequence header do disagree in the
    /// wild.
    pub fn new(info: StreamInfo) -> Result<Self> {
        let codec = info.codec.to_ascii_lowercase();
        if !supports(&codec) {
            bail!("the AV1 decoder decodes AV1, not '{codec}'");
        }
        let threads = decode_threads();
        let engine = if worker_disabled() {
            let mut decoder = av1::Decoder::new();
            decoder.set_threads(threads);
            Engine::Inline(Box::new(decoder))
        } else {
            let (sample_tx, sample_rx) = mpsc::sync_channel::<Vec<u8>>(IN_FLIGHT);
            let (result_tx, results) = mpsc::channel::<Decoded>();
            let handle = std::thread::Builder::new()
                .name("rivet-av1-decode".into())
                .spawn(move || {
                    let mut decoder = av1::Decoder::new();
                    decoder.set_threads(threads);
                    for sample in sample_rx {
                        let decoded = decoder.decode_all(&sample);
                        let failed = decoded.is_err();
                        if result_tx.send(decoded).is_err() || failed {
                            // The adapter is gone, or the stream is broken:
                            // a corrupt temporal unit leaves the reference
                            // state undefined, so nothing after it decodes.
                            return;
                        }
                    }
                })
                .context("starting the AV1 decode thread")?;
            Engine::Worker { samples: Some(sample_tx), results, handle: Some(handle), pending: 0 }
        };
        tracing::info!(
            backend = "av1",
            width = info.width,
            height = info.height,
            threads,
            "AV1 software decode engaged (rivet's own decoder)"
        );
        Ok(Self { info, engine, ready: VecDeque::new(), next_pts: 0 })
    }

    fn accept(&mut self, decoded: Decoded) -> Result<()> {
        let frames = decoded.map_err(|e| anyhow!("AV1 decode failed: {e}"))?;
        for frame in frames {
            let converted = self.convert(frame)?;
            self.ready.push_back(converted);
        }
        Ok(())
    }

    /// Collect what the worker has finished; with `wait`, everything it was
    /// given.
    fn collect(&mut self, wait: bool) -> Result<()> {
        loop {
            let Engine::Worker { results, pending, .. } = &mut self.engine else {
                return Ok(());
            };
            if *pending == 0 {
                return Ok(());
            }
            let got = if wait {
                results.recv().map_err(|_| anyhow!("the AV1 decode thread ended early"))?
            } else {
                match results.try_recv() {
                    Ok(got) => got,
                    Err(mpsc::TryRecvError::Empty) => return Ok(()),
                    Err(mpsc::TryRecvError::Disconnected) => bail!("the AV1 decode thread ended early"),
                }
            };
            *pending -= 1;
            self.accept(got)?;
        }
    }

    fn convert(&mut self, frame: av1::Frame) -> Result<VideoFrame> {
        use av1::ChromaFormat as C;
        let format = match (frame.chroma, frame.bit_depth) {
            (C::Mono | C::Yuv420, 8) => PixelFormat::Yuv420p,
            (C::Mono | C::Yuv420, 10) => PixelFormat::Yuv420p10le,
            (C::Mono | C::Yuv420, 12) => PixelFormat::Yuv420p12le,
            (C::Yuv422, 8) => PixelFormat::Yuv422p,
            (C::Yuv422, 10) => PixelFormat::Yuv422p10le,
            (C::Yuv422, 12) => PixelFormat::Yuv422p12le,
            (C::Yuv444, 8) => PixelFormat::Yuv444p,
            (C::Yuv444, 10) => PixelFormat::Yuv444p10le,
            (C::Yuv444, 12) => PixelFormat::Yuv444p12le,
            (chroma, depth) => bail!("AV1 decoded a {chroma:?} {depth}-bit picture, which AV1 does not define"),
        };
        let color_space = match frame.color.matrix_coefficients {
            1 => ColorSpace::Bt709,
            9 | 10 => ColorSpace::Bt2020,
            5 | 6 => ColorSpace::Bt601,
            _ => self.info.color_space,
        };
        let (w, h) = (frame.width, frame.height);
        let data = if frame.chroma == C::Mono {
            // Neutral chroma: the middle of the sample range.
            let bps = frame.bytes_per_sample();
            let chroma = (w.div_ceil(2) * h.div_ceil(2)) as usize;
            let mut data = frame.data;
            let mid: u16 = 1 << (frame.bit_depth - 1);
            data.reserve(2 * chroma * bps);
            for _ in 0..2 * chroma {
                if bps == 2 {
                    data.extend_from_slice(&mid.to_le_bytes());
                } else {
                    data.push(mid as u8);
                }
            }
            data
        } else {
            frame.data
        };
        // The sequence header is authoritative over whatever the container said.
        self.info.width = w;
        self.info.height = h;
        self.info.pixel_format = format;
        self.info.color_space = color_space;
        apply_color(&mut self.info.color_metadata, &frame.color, &frame.hdr);
        let pts = self.next_pts;
        self.next_pts += 1;
        Ok(VideoFrame::new(Bytes::from(data), w, h, format, color_space, pts))
    }
}

/// What a decoded frame says of its colour, over what the container said: the
/// colour description when the sequence header has one, the HDR10 metadata
/// when the stream carries it (in the pipeline's units: chromaticities in
/// 0.00002 steps, luminance in 0.0001 cd/m²).
fn apply_color(meta: &mut crate::frame::ColorMetadata, c: &av1::ColorInfo, hdr: &av1::HdrMetadata) {
    use crate::frame::{ContentLightLevel, MasteringDisplay, TransferFn};
    // All three unspecified: the stream says nothing; keep the container's.
    if c.color_primaries != 2 || c.transfer_characteristics != 2 || c.matrix_coefficients != 2 {
        meta.transfer = TransferFn::from_h273(c.transfer_characteristics as u8);
        meta.colour_primaries = c.color_primaries as u8;
        meta.matrix_coefficients = c.matrix_coefficients as u8;
        meta.full_range = c.full_range;
    }
    if let Some(cl) = hdr.content_light {
        meta.content_light_level = Some(ContentLightLevel { max_cll: cl.max_cll, max_fall: cl.max_fall });
    }
    if let Some(md) = hdr.mastering_display {
        let xy = |v: u16| ((u64::from(v) * 50_000 + 32_768) / 65_536).min(65_535) as u16;
        meta.mastering_display = Some(MasteringDisplay {
            primaries_r_x: xy(md.primaries[0][0]),
            primaries_r_y: xy(md.primaries[0][1]),
            primaries_g_x: xy(md.primaries[1][0]),
            primaries_g_y: xy(md.primaries[1][1]),
            primaries_b_x: xy(md.primaries[2][0]),
            primaries_b_y: xy(md.primaries[2][1]),
            white_point_x: xy(md.white_point[0]),
            white_point_y: xy(md.white_point[1]),
            max_luminance: ((u64::from(md.luminance_max) * 10_000 + 128) / 256).min(u64::from(u32::MAX)) as u32,
            min_luminance: ((u64::from(md.luminance_min) * 10_000 + 8_192) / 16_384).min(u64::from(u32::MAX)) as u32,
        });
    }
}

impl Decoder for Av1Decoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    fn push_sample(&mut self, data: &[u8]) -> Result<()> {
        match &mut self.engine {
            Engine::Inline(decoder) => {
                let decoded = decoder.decode_all(data);
                self.accept(decoded)
            }
            Engine::Worker { samples, pending, .. } => {
                let tx = samples.as_ref().ok_or_else(|| anyhow!("an AV1 sample after the end of the stream"))?;
                if tx.send(data.to_vec()).is_err() {
                    // The worker stopped on an error it has already sent:
                    // collect it, so the caller sees the decoder's reason.
                    self.collect(true)?;
                    bail!("the AV1 decode thread ended early");
                }
                *pending += 1;
                self.collect(false)
            }
        }
    }

    fn finish(&mut self) -> Result<()> {
        if let Engine::Worker { samples, .. } = &mut self.engine {
            // Closing the channel ends the worker once it has decoded the
            // rest; `collect` waits for every answer.
            samples.take();
        }
        self.collect(true)?;
        if let Engine::Worker { handle, .. } = &mut self.engine
            && let Some(handle) = handle.take()
        {
            handle.join().map_err(|_| anyhow!("the AV1 decode thread panicked"))?;
        }
        Ok(())
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        if self.ready.is_empty() {
            self.collect(false)?;
        }
        Ok(self.ready.pop_front())
    }
}

impl Drop for Av1Decoder {
    fn drop(&mut self) {
        if let Engine::Worker { samples, results, handle, .. } = &mut self.engine {
            samples.take();
            // Unblock a worker waiting to send, then let it run out.
            while results.try_recv().is_ok() {}
            if let Some(handle) = handle.take() {
                // The worker ends as soon as its current temporal unit does.
                let _ = handle.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> StreamInfo {
        StreamInfo {
            codec: "av1".into(),
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

    /// Temporal units from rivet-av1's own encoder: `n` frames of a moving
    /// gradient at `w` x `h`.
    fn stream(w: u32, h: u32, n: u32, bit_depth: u32) -> Vec<Vec<u8>> {
        let mut cfg = av1::Config::new(w, h);
        cfg.bit_depth = bit_depth;
        cfg.keyframe_interval = 4;
        let mut enc = av1::Encoder::new(cfg);
        (0..n)
            .map(|i| {
                let mut f = av1::Frame::new(w, h, bit_depth, av1::ChromaFormat::Yuv420);
                for p in 0..3 {
                    let pl = f.planes[p];
                    for y in 0..pl.height {
                        for x in 0..pl.width {
                            let v = ((x * 3 + y * 2 + i * 5 + p as u32 * 40) % 200 + 20) as u16;
                            f.set_sample(p, x, y, v << (bit_depth - 8));
                        }
                    }
                }
                enc.encode(&f).unwrap()
            })
            .collect()
    }

    fn decode_all(units: &[Vec<u8>]) -> Vec<VideoFrame> {
        let mut dec = Av1Decoder::new(info()).unwrap();
        let mut out = Vec::new();
        for u in units {
            dec.push_sample(u).unwrap();
            while let Some(f) = dec.decode_next().unwrap() {
                out.push(f);
            }
        }
        dec.finish().unwrap();
        while let Some(f) = dec.decode_next().unwrap() {
            out.push(f);
        }
        out
    }

    #[test]
    fn every_frame_comes_out_in_order() {
        let units = stream(64, 48, 9, 8);
        let frames = decode_all(&units);
        assert_eq!(frames.len(), 9);
        for (i, f) in frames.iter().enumerate() {
            assert_eq!((f.pts, f.width, f.height, f.format), (i as u64, 64, 48, PixelFormat::Yuv420p));
            assert_eq!(f.data.len(), 64 * 48 * 3 / 2);
        }
    }

    #[test]
    fn ten_bit_is_carried() {
        let frames = decode_all(&stream(34, 18, 2, 10));
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].format, PixelFormat::Yuv420p10le);
        assert_eq!(frames[0].data.len(), (34 * 18 + 2 * 17 * 9) * 2);
    }

    /// An HDR10 stream: its colour description and mastering display /
    /// content light level reach `stream_info`, in the pipeline's units.
    #[test]
    fn hdr10_signalling_reaches_the_stream_info() {
        let mut cfg = av1::Config::new(32, 32);
        cfg.bit_depth = 10;
        cfg.color = av1::ColorInfo {
            color_primaries: 9,
            transfer_characteristics: 16,
            matrix_coefficients: 9,
            full_range: false,
            chroma_sample_position: 0,
        };
        cfg.hdr = av1::HdrMetadata {
            content_light: Some(av1::ContentLightLevel { max_cll: 1000, max_fall: 400 }),
            mastering_display: Some(av1::MasteringDisplay {
                primaries: [[46_399, 19_136], [11_141, 52_167], [8_585, 3_015]],
                white_point: [20_493, 21_561],
                luminance_max: 1000 << 8,
                luminance_min: 82,
            }),
        };
        let mut enc = av1::Encoder::new(cfg);
        let f = av1::Frame::new(32, 32, 10, av1::ChromaFormat::Yuv420);
        let unit = enc.encode(&f).unwrap();
        let mut dec = Av1Decoder::new(info()).unwrap();
        dec.push_sample(&unit).unwrap();
        dec.finish().unwrap();
        assert!(dec.decode_next().unwrap().is_some());
        let m = dec.stream_info().color_metadata;
        assert_eq!(m.transfer, crate::frame::TransferFn::St2084);
        assert_eq!((m.colour_primaries, m.matrix_coefficients, m.full_range), (9, 9, false));
        assert_eq!(dec.stream_info().color_space, ColorSpace::Bt2020);
        let md = m.mastering_display.expect("mastering display");
        // 46399 / 65536 = 0.70799 -> 35400 steps of 0.00002.
        assert_eq!(md.primaries_r_x, 35_400);
        assert_eq!(md.max_luminance, 10_000_000);
        assert_eq!(md.min_luminance, 50);
        assert_eq!(m.content_light_level.map(|c| (c.max_cll, c.max_fall)), Some((1000, 400)));
    }

    #[test]
    fn a_corrupt_stream_is_an_error_not_a_hang() {
        let mut units = stream(32, 32, 3, 8);
        let half = units[1].len() / 2;
        units[1].truncate(half);
        let mut dec = Av1Decoder::new(info()).unwrap();
        let mut failed = false;
        for u in &units {
            if dec.push_sample(u).is_err() {
                failed = true;
                break;
            }
            while let Ok(Some(_)) = dec.decode_next() {}
        }
        failed |= dec.finish().is_err();
        assert!(failed, "a broken temporal unit must surface as an error");
    }
}
