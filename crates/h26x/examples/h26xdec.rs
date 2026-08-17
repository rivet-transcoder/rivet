//! Decode an Annex-B H.264 stream and print one line per output frame in
//! libavcodec `framemd5` style (frame index and the MD5 of the packed
//! planar picture), or write raw YUV.
//!
//!   h26xdec <input.264> [out.yuv]

use std::io::Write;

fn md5_hex(data: &[u8]) -> String {
    let d = md5::compute(data);
    format!("{:x}", d)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: h26xdec <input.264|.265> [out.yuv]");
        std::process::exit(2);
    }
    let data = std::fs::read(&args[1]).expect("read input");
    let mut out = args.get(2).map(|p| std::fs::File::create(p).expect("create output"));
    let mut dec = h26x::h264::H264Decoder::new();
    let mut n = 0usize;
    let mut emit = |pic: h26x::Picture, out: &mut Option<std::fs::File>| {
        let packed = pic.packed();
        println!("{},{},{},{}x{},{}", n, pic.poc, pic.decode_index, pic.width, pic.height, md5_hex(&packed));
        if let Some(f) = out {
            f.write_all(&packed).unwrap();
        }
        n += 1;
    };
    // Feed NAL by NAL so an error names the position.
    let mut nals = 0usize;
    for nal in h26x::nal::annexb_nals(&data) {
        nals += 1;
        if let Err(e) = dec.push_nal(nal) {
            eprintln!("error at NAL {nals} (type {}): {e}", nal[0] & 0x1f);
            std::process::exit(1);
        }
        while let Some(pic) = dec.next_picture() {
            emit(pic, &mut out);
        }
    }
    if let Err(e) = dec.flush() {
        eprintln!("error at flush: {e}");
        std::process::exit(1);
    }
    while let Some(pic) = dec.next_picture() {
        emit(pic, &mut out);
    }
    eprintln!("{n} frames, {} warnings", dec.warnings());
}
