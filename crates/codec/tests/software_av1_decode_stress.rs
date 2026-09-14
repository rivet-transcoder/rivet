//! rav1d's worker pool, oversubscribed: many threaded decodes at once on two cores.
//!
//! The regression guard for an intermittent hang in `software_av1_roundtrip`
//! (merge gate, 2026-09-14). In a debug build rav1d tracks every borrow of its
//! shared line buffers (`DisjointMut`) and panics on an overlap. Its fallback
//! CDEF borrows the top row from two pixels left of the block, and reads from
//! the third when the block has no left neighbour (rav1d 1.1.0 `src/cdef.rs`,
//! `padding`); at the start of each superblock row those two pixels belong to
//! the neighbouring slot of the line buffer, which another worker may be
//! writing (`backup2lines`). When the two borrows meet:
//!
//! - if the reader borrows second, the panic is inside an `extern "C"` DSP
//!   function, which cannot unwind, and the process aborts;
//! - if the writer borrows second, the panic unwinds and ends that worker
//!   thread. Its task never completes, its frame is never marked finished, and
//!   `dav1d_send_data` or `dav1d_get_picture` waits for it forever. The panic
//!   message goes to the test's captured output, which is never printed
//!   because the test never finishes.
//!
//! The fix is `[profile.dev.package.rav1d] debug-assertions = false` in the
//! workspace `Cargo.toml`. Without it this test fails by name: the process
//! abort, or the watchdog below, rather than a gate that sits there.
//!
//! Two things make the borrows meet often enough to test. Squeezing the process
//! onto two cores preempts workers mid-task: 400 runs of the round trip under
//! full CPU load on all 32 threads never tripped it, while pinned to two cores
//! 3 runs in 400 did. And tall, narrow frames put a row-start block in every one
//! of many superblock rows, so each decode offers many meetings instead of one:
//! before the fix this shape aborted in 14 of 15 runs, typically 12-15 s in,
//! where 128x128 frames aborted in 3 of 5.

use std::sync::Arc;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use codec::decode::Decoder;
use codec::decode::rav1d_sw::Rav1dDecoder;
use codec::encode::rav1e_sw::Rav1eEncoder;
use codec::encode::{Encoder, EncoderConfig, QualityTarget, SpeedTier};
use codec::frame::{ColorMetadata, ColorSpace, PixelFormat, StreamInfo, VideoCodec, VideoFrame};

const W: u32 = 64;
const H: u32 = 512;
const FRAMES: usize = 5;
/// Decoders running at once, each with rav1d's automatically sized pool.
const DECODERS: usize = 8;
/// Decodes per decoder thread.
const ROUNDS: usize = 20;
/// No decode finishing for this long means a worker died or deadlocked. One
/// decode here takes about a second squeezed onto two cores.
const WATCHDOG: Duration = Duration::from_secs(120);

/// A hard vertical edge with a little texture, moving from frame to frame, so
/// CDEF has something to filter in every superblock row.
fn textured_frame(pts: u64) -> VideoFrame {
    let (w, h) = (W as usize, H as usize);
    let (cw, ch) = (w / 2, h / 2);
    let edge = w / 2 + (pts as usize * 3) % 16;
    let mut data = Vec::with_capacity(w * h + 2 * cw * ch);
    for y in 0..h {
        for x in 0..w {
            let base = if x < edge { 40 } else { 210 };
            data.push(base + ((x ^ y) & 7) as u8);
        }
    }
    data.extend(std::iter::repeat_n(128u8, 2 * cw * ch));
    VideoFrame::new(
        data.into(),
        W,
        H,
        PixelFormat::Yuv420p,
        ColorSpace::Bt709,
        pts,
    )
}

fn encode() -> Vec<Vec<u8>> {
    let mut enc = Rav1eEncoder::new(EncoderConfig {
        width: W,
        height: H,
        frame_rate: 30.0,
        quality: u8::MAX,
        speed_preset: u8::MAX,
        keyframe_interval: 30,
        target: QualityTarget::Standard,
        tier: SpeedTier::Draft,
        threads: 1,
        pixel_format: PixelFormat::Yuv420p,
        color_metadata: ColorMetadata::default(),
        gpu_index: None,
        gpu_vendor: None,
        codec: VideoCodec::Av1,
        constant_qp: false,
        overrides: Default::default(),
    })
    .expect("rav1e should construct");
    for pts in 0..FRAMES as u64 {
        enc.send_frame(&textured_frame(pts))
            .expect("rav1e accepts a frame");
    }
    enc.flush().expect("flush");
    let mut packets = Vec::new();
    while let Some(pkt) = enc.receive_packet().expect("receive") {
        packets.push(pkt.data.to_vec());
    }
    assert_eq!(packets.len(), FRAMES);
    packets
}

/// One full decode through the wrapper, returning how many frames came out.
fn decode(packets: &[Vec<u8>]) -> usize {
    let mut dec = Rav1dDecoder::new(StreamInfo {
        codec: "av1".to_string(),
        width: W,
        height: H,
        frame_rate: 30.0,
        duration: FRAMES as f64 / 30.0,
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        total_frames: FRAMES as u64,
        bitrate: 0,
        color_metadata: ColorMetadata::default(),
    })
    .expect("rav1d should construct");
    let mut frames = 0;
    for pkt in packets {
        dec.push_sample(pkt).expect("rav1d accepts a packet");
        while dec.decode_next().expect("decode").is_some() {
            frames += 1;
        }
    }
    dec.finish().expect("finish");
    while dec.decode_next().expect("drain").is_some() {
        frames += 1;
    }
    frames
}

/// Restrict this test process to two logical cores, so rav1d's workers are
/// preempted mid-task constantly. It has its own test binary, so no other test
/// is slowed by it.
#[cfg(windows)]
fn squeeze_onto_two_cores() {
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn SetProcessAffinityMask(process: *mut std::ffi::c_void, mask: usize) -> i32;
    }
    // SAFETY: `GetCurrentProcess` returns a pseudo-handle that is always valid
    // for the calling process, and the mask names cores 0 and 1, which exist on
    // any host that runs this test.
    let ok = unsafe { SetProcessAffinityMask(GetCurrentProcess(), 0b11) };
    assert!(ok != 0, "SetProcessAffinityMask failed");
}

#[cfg(not(windows))]
fn squeeze_onto_two_cores() {}

#[test]
fn many_threaded_rav1d_decodes_on_two_cores_all_finish() {
    let packets = Arc::new(encode());
    squeeze_onto_two_cores();

    let (tx, rx) = mpsc::channel();
    for _ in 0..DECODERS {
        let (tx, packets) = (tx.clone(), Arc::clone(&packets));
        std::thread::spawn(move || {
            for _ in 0..ROUNDS {
                if tx.send(decode(&packets)).is_err() {
                    return;
                }
            }
        });
    }
    drop(tx);

    let total = DECODERS * ROUNDS;
    for done in 0..total {
        match rx.recv_timeout(WATCHDOG) {
            Ok(frames) => assert_eq!(frames, FRAMES, "decode {done} returned {frames} frames"),
            Err(RecvTimeoutError::Timeout) => panic!(
                "no rav1d decode finished for {WATCHDOG:?} after {done} of {total}: a rav1d \
                 worker thread died or deadlocked (look for a DisjointMut panic in the output)"
            ),
            Err(RecvTimeoutError::Disconnected) => {
                panic!("a decoder thread panicked after {done} of {total} decodes finished")
            }
        }
    }
}
