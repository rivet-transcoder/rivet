//! Decode an AC-3 / E-AC-3 elementary stream to raw f32le PCM with the
//! in-tree decoder — the counterpart of `ffmpeg -i x.ac3 -f f32le`.
//!
//!     cargo run -p rivet-codec --example ac3_decode -- in.ac3 out.f32 [drc_scale] [noise_seed|off]
//!
//! Prints the stream layout, the decoder's dither / dynrng statistics and the
//! coding tools the stream exercised. Leading junk before the first
//! syncframe is skipped and a truncated final frame dropped, as libavcodec
//! does. `RUST_LOG=trace` prints the decoder's per-frame / per-block syntax
//! trace on stderr; `AC3_DECODE_FRAMES=1` prints one line per syncframe with
//! the coding tools that frame used (the deltas of the `Features` counters),
//! for lining up against a per-frame error report.

use std::io::Write;

use codec::audio::decode::ac3::{FrameDecoder, frame_crc_ok, parse_header};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: ac3_decode <in.ac3|in.eac3> <out.f32le> [drc_scale=1.0] [noise_seed|off]");
        std::process::exit(2);
    }
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .without_time()
        .init();
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
    let per_frame = std::env::var_os("AC3_DECODE_FRAMES").is_some();
    let mut prev_feat = dec.features();
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
        if per_frame {
            let f = dec.features();
            let d = |a: u64, b: u64| a - b;
            eprintln!(
                "frame {frames}: blksw {} cpl {} phsflg {} remat {} deltba {} dynrng {} spx {} spxatten {} aht {} gaq {} vq {} gaqbins {} large {} mixed-blksw {:?} dith-offsets {:?}",
                d(f.blksw_chblocks, prev_feat.blksw_chblocks),
                d(f.cpl_blocks, prev_feat.cpl_blocks),
                d(f.phsflg_blocks, prev_feat.phsflg_blocks),
                d(f.remat_blocks, prev_feat.remat_blocks),
                d(f.deltba_blocks, prev_feat.deltba_blocks),
                d(f.dynrng_blocks, prev_feat.dynrng_blocks),
                d(f.spx_blocks, prev_feat.spx_blocks),
                d(f.spx_atten_channels, prev_feat.spx_atten_channels),
                d(f.aht_channels, prev_feat.aht_channels),
                d(f.gaq_channels, prev_feat.gaq_channels),
                d(f.vq_bins, prev_feat.vq_bins),
                d(f.gaq_bins, prev_feat.gaq_bins),
                d(f.gaq_large_mantissas, prev_feat.gaq_large_mantissas),
                dec.mixed_transform_blocks().iter().filter(|&&b| b / 6 == frames as u64).map(|b| b % 6).collect::<Vec<_>>(),
                dec.dithflag_bit_offsets(),
            );
            prev_feat = f;
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
