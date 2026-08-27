//! Per-frame timing and output dump for the denoise family, on a raw
//! `yuv420p` clip.
//!
//! ```text
//! denoise_bench <clip.yuv> <w> <h> [-n FRAMES] [-m a,b,..] [--out DIR] [--chain "spec"]
//! ```
//!
//! Every method is applied frame by frame through the public filter API
//! (exactly what the decode pump does), each frame timed on its own so a
//! paired comparison between two runs of the *same binary* — one with
//! `RIVET_DENOISE_MAX_SIMD=none`, one without — can take the median of
//! per-frame ratios rather than comparing two means. Prints one
//! `FRAME method idx ms` line per frame and one `RESULT method frames
//! median_ms mean_ms` line per method.
//!
//! With `--out DIR` the filtered frames are written to `DIR/<method>.yuv`, so
//! an external `md5sum` proves two builds produce the same bytes.

use std::io::Write;
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use codec::filter::{DenoiseMethod, FilterChain, VideoFilter, parse_chain};
use codec::{ColorSpace, PixelFormat, VideoFrame};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: denoise_bench <clip.yuv> <w> <h> [-n FRAMES] [-m a,b,..] [--out DIR] [--chain SPEC]");
        std::process::exit(2);
    }
    let path = &args[1];
    let w: u32 = args[2].parse().expect("width");
    let h: u32 = args[3].parse().expect("height");
    let mut max_frames = usize::MAX;
    let mut methods: Vec<String> = vec![
        "bilateral".into(),
        "gaussian".into(),
        "median".into(),
        "mean".into(),
        "nlmeans".into(),
        "anisotropic".into(),
        "nlmeans_params".into(),
    ];
    let mut out_dir: Option<String> = None;
    let mut chain: Option<String> = None;
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "-n" => {
                max_frames = args[i + 1].parse().expect("frames");
                i += 2;
            }
            "-m" => {
                methods = args[i + 1].split(',').map(|s| s.to_string()).collect();
                i += 2;
            }
            "--out" => {
                out_dir = Some(args[i + 1].clone());
                i += 2;
            }
            "--chain" => {
                chain = Some(args[i + 1].clone());
                methods = vec!["chain".into()];
                i += 2;
            }
            o => panic!("unknown arg {o}"),
        }
    }

    let data = std::fs::read(path).expect("reading clip");
    let frame_len = (w as usize * h as usize * 3) / 2;
    let n = (data.len() / frame_len).min(max_frames);
    assert!(n > 0, "clip shorter than one frame");
    let frames: Vec<VideoFrame> = (0..n)
        .map(|i| {
            VideoFrame::new(
                Bytes::copy_from_slice(&data[i * frame_len..(i + 1) * frame_len]),
                w,
                h,
                PixelFormat::Yuv420p,
                ColorSpace::Bt709,
                i as u64,
            )
        })
        .collect();

    for m in &methods {
        let filter = match m.as_str() {
            "bilateral" => VideoFilter::Denoise { method: DenoiseMethod::Bilateral, strength: 0.8 },
            "gaussian" => VideoFilter::Denoise { method: DenoiseMethod::Gaussian, strength: 0.8 },
            "median" => VideoFilter::Denoise { method: DenoiseMethod::Median, strength: 0.8 },
            "mean" => VideoFilter::Denoise { method: DenoiseMethod::Mean, strength: 0.8 },
            "nlmeans" => VideoFilter::Denoise { method: DenoiseMethod::Nlmeans, strength: 0.8 },
            "anisotropic" => {
                VideoFilter::Denoise { method: DenoiseMethod::Anisotropic, strength: 0.8 }
            }
            "nlmeans_params" => VideoFilter::Nlmeans { s: 1.0, p: 7, pc: 5, r: 3, rc: 3 },
            "chain" => {
                let spec = chain.as_deref().expect("--chain");
                let filters = parse_chain(spec).expect("parsing --chain");
                run_chain(spec, &filters, &frames, out_dir.as_deref());
                continue;
            }
            o => panic!("unknown method {o}"),
        };
        let filters = [filter];
        run_chain(m, &filters, &frames, out_dir.as_deref());
    }
}

fn run_chain(name: &str, filters: &[VideoFilter], frames: &[VideoFrame], out_dir: Option<&str>) {
    let chain = Arc::new(FilterChain::prepare(filters).expect("preparing chain"));
    let mut out = out_dir.map(|d| {
        std::fs::create_dir_all(d).expect("out dir");
        let file = name.replace(['=', ':', ','], "_");
        std::io::BufWriter::new(
            std::fs::File::create(format!("{d}/{file}.yuv")).expect("creating output"),
        )
    });
    let mut times = Vec::with_capacity(frames.len());
    for (i, f) in frames.iter().enumerate() {
        let t = Instant::now();
        let r = chain.apply(f.clone()).expect("filter");
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        times.push(ms);
        println!("FRAME {name} {i} {ms:.3}");
        if let Some(o) = out.as_mut() {
            o.write_all(&r.data).expect("writing output");
        }
    }
    let mut sorted = times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!("RESULT {name} {} {median:.3} {mean:.3}", times.len());
}
