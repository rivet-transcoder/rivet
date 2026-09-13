//! Decode an AC-3 / E-AC-3 elementary stream to raw f32le PCM with the
//! in-tree decoder — the counterpart of `ffmpeg -i x.ac3 -f f32le`.
//!
//!     cargo run -p rivet-codec --example ac3_decode -- in.ac3 out.f32 [drc_scale] [noise_seed|off]
//!
//! Prints the stream layout, the decoder's dither / dynrng statistics and the
//! coding tools the stream exercised. Leading junk before the first
//! syncframe is skipped and a truncated final frame dropped, as libavcodec
//! does.

use std::io::Write;

use codec::audio::decode::ac3::{FrameDecoder, frame_crc_ok, parse_header};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: ac3_decode <in.ac3|in.eac3> <out.f32le> [drc_scale=1.0] [noise_seed|off]");
        std::process::exit(2);
    }
    let drc: f32 = args.get(3).map(|s| s.parse().expect("drc_scale")).unwrap_or(1.0);
    let es = std::fs::read(&args[1]).expect("read input");
    let mut dec = FrameDecoder::new(drc);
    match args.get(4).map(String::as_str) {
        Some("off") => dec.set_noise_fill(false),
        Some(seed) => dec.set_noise_seed(seed.parse().expect("noise_seed")),
        None => {}
    }
    let mut pcm = Vec::new();
    let mut pos = 0usize;
    let mut frames = 0usize;
    let mut bad_crc = 0usize;
    let mut skipped = 0usize;
    let mut last = None;
    while pos + 8 <= es.len() {
        let hdr = match parse_header(&es[pos..]) {
            Ok(h) => h,
            Err(e) => {
                if frames > 0 {
                    eprintln!("frame {frames} at byte {pos}: {e}");
                    break;
                }
                skipped += 1;
                pos += 1;
                continue;
            }
        };
        if pos + hdr.frame_len > es.len() {
            eprintln!("frame {frames}: truncated ({} of {} bytes), dropped", es.len() - pos, hdr.frame_len);
            break;
        }
        let frame = &es[pos..pos + hdr.frame_len];
        if !frame_crc_ok(frame) {
            bad_crc += 1;
        }
        match dec.decode(frame, &mut pcm) {
            Ok(Some(h)) => last = Some(h),
            Ok(None) => {}
            Err(e) => eprintln!("frame {frames}: {e}"),
        }
        frames += 1;
        pos += hdr.frame_len;
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[2]).expect("create output"));
    for v in &pcm {
        out.write_all(&v.to_le_bytes()).unwrap();
    }
    out.flush().unwrap();
    let (dithered, total) = dec.dither_stats();
    let (drc_blocks, blocks) = dec.drc_stats();
    match last {
        Some(h) => println!(
            "{} frames ({} junk bytes skipped), {}{} acmod {} lfe {} → {} ch @ {} Hz, {} kbit/s; {} samples/ch; bad crc {}; dithered {}/{} bins; dynrng blocks {}/{}\nfeatures: {}",
            frames,
            skipped,
            if h.eac3 { "E-AC-3" } else { "AC-3" },
            if h.eac3 { format!(" ({} blk)", h.numblks) } else { String::new() },
            h.acmod,
            h.lfeon,
            h.channels(),
            h.sample_rate,
            h.bitrate_kbps,
            pcm.len() / h.channels().max(1),
            bad_crc,
            dithered,
            total,
            drc_blocks,
            blocks,
            dec.features()
        ),
        None => println!("no decodable frames"),
    }
}
