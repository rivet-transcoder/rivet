//! Make an AC-3 / E-AC-3 elementary stream deterministic for cross-checking:
//! clear every `dithflag` (§5.4.3.2; Annex E §2.3.4.2) so that no decoder
//! substitutes noise for zero-bit mantissas, and re-solve `crc1` / `crc2`
//! (§7.10.1) so the frames stay valid. Nothing in the syntax depends on the
//! flags — they only select the §7.3.4 reconstruction — so the modified stream
//! parses exactly as before, and two conformant decoders must now agree on it
//! to float rounding instead of differing by two independent noise sequences.
//!
//! E-AC-3 frames with `dithflage = 0` carry no flags (they default to 1) and
//! are left as they are; the tool reports how many blocks kept their dither.
//! Frames the decoder refuses (other substreams, unsupported tools) are copied
//! unchanged. Leading junk before the first syncframe is dropped.
//!
//!     cargo run -p rivet-codec --example ac3_strip_dither -- in.ac3 out.ac3

use codec::audio::decode::ac3::{FrameDecoder, frame_crc_ok, parse_header};

/// CRC-16, generator x¹⁶ + x¹⁵ + x² + 1, zero initial state (§7.10.1).
fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0;
    for &b in data {
        crc ^= u16::from(b) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 { (crc << 1) ^ 0x8005 } else { crc << 1 };
        }
    }
    crc
}

/// The 16-bit word `w` with `crc16(w ‖ body) == 0`. The CRC is linear over
/// GF(2), so `crc16(w ‖ body) = crc16(w ‖ 0…0) ⊕ crc16(body)`; the 16 basis
/// words give a 16×16 system solved by elimination.
fn solve_prefix_crc(body: &[u8]) -> u16 {
    let mut target = crc16(body);
    let mut rows: Vec<(u16, u16)> = (0..16)
        .map(|i| {
            let w: u16 = 1 << i;
            let mut buf = w.to_be_bytes().to_vec();
            buf.resize(2 + body.len(), 0);
            (crc16(&buf), w)
        })
        .collect();
    let mut w = 0u16;
    for bit in (0..16).rev() {
        let Some(p) = rows.iter().position(|r| r.0 & (1 << bit) != 0) else { continue };
        let piv = rows.remove(p);
        for r in &mut rows {
            if r.0 & (1 << bit) != 0 {
                r.0 ^= piv.0;
                r.1 ^= piv.1;
            }
        }
        if target & (1 << bit) != 0 {
            target ^= piv.0;
            w ^= piv.1;
        }
    }
    assert_eq!(target, 0, "crc prefix not solvable");
    w
}

fn clear_bit(frame: &mut [u8], bit: usize) {
    frame[bit / 8] &= !(0x80 >> (bit % 8));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: ac3_strip_dither <in.ac3|in.eac3> <out>");
        std::process::exit(2);
    }
    let es = std::fs::read(&args[1]).expect("read input");
    let mut out = Vec::with_capacity(es.len());
    let mut dec = FrameDecoder::new(0.0);
    let mut scratch = Vec::new();
    let (mut frames, mut stripped, mut kept, mut copied, mut skipped_bytes) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut pos = 0usize;
    while pos + 8 <= es.len() {
        let hdr = match parse_header(&es[pos..]) {
            Ok(h) if pos + h.frame_len <= es.len() && frame_crc_ok(&es[pos..pos + h.frame_len]) => h,
            Ok(h) if pos + h.frame_len > es.len() => break,
            _ => {
                skipped_bytes += 1;
                pos += 1;
                continue;
            }
        };
        let mut frame = es[pos..pos + hdr.frame_len].to_vec();
        pos += hdr.frame_len;
        frames += 1;
        scratch.clear();
        match dec.decode(&frame, &mut scratch) {
            Ok(Some(_)) => {
                let offsets: Vec<Option<usize>> = dec.dithflag_bit_offsets().to_vec();
                for off in offsets {
                    match off {
                        Some(bit) => {
                            for ch in 0..hdr.nfchans {
                                clear_bit(&mut frame, bit + ch);
                            }
                            stripped += 1;
                        }
                        None => kept += 1,
                    }
                }
                if !hdr.eac3 {
                    // crc1 covers bytes 2..5/8 of the frame (§7.10.1); solve it
                    // first because crc2 covers it.
                    let words = hdr.frame_len / 2;
                    let five8 = ((words >> 1) + (words >> 3)) * 2;
                    let crc1 = solve_prefix_crc(&frame[4..five8]);
                    frame[2..4].copy_from_slice(&crc1.to_be_bytes());
                }
                let n = frame.len();
                let crc2 = crc16(&frame[2..n - 2]);
                frame[n - 2..].copy_from_slice(&crc2.to_be_bytes());
                assert!(frame_crc_ok(&frame), "frame {frames}: crc re-solve failed");
            }
            Ok(None) => copied += 1,
            Err(e) => {
                eprintln!("frame {frames}: {e} — copied unchanged");
                copied += 1;
            }
        }
        out.extend_from_slice(&frame);
    }
    std::fs::write(&args[2], &out).expect("write output");
    println!(
        "{frames} frames: dither cleared in {stripped} blocks, {kept} blocks had no flags (E-AC-3 dithflage=0), {copied} frames copied unchanged, {skipped_bytes} leading/junk bytes dropped"
    );
}
