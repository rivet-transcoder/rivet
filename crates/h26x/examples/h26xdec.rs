//! Decode an Annex-B H.264 or HEVC stream and print one line per output
//! frame in libavcodec `framemd5` style (frame index and the MD5 of the
//! packed planar picture), or write raw YUV.
//!
//!   h26xdec <input.264|.265> [out.yuv]
//!
//! The codec is taken from the extension (`.265`/`.hevc`/`.h265` = HEVC,
//! anything else = H.264).

use std::io::Write;

fn md5_hex(data: &[u8]) -> String {
    let d = md5::compute(data);
    format!("{:x}", d)
}

/// The two decoders behind one face.
enum Dec {
    H264(h26x::h264::H264Decoder),
    Hevc(h26x::hevc::HevcDecoder),
}

impl Dec {
    fn push_nal(&mut self, nal: &[u8]) -> h26x::Result<()> {
        match self {
            Dec::H264(d) => d.push_nal(nal),
            Dec::Hevc(d) => d.push_nal(nal),
        }
    }
    fn next_picture(&mut self) -> Option<h26x::Picture> {
        match self {
            Dec::H264(d) => d.next_picture(),
            Dec::Hevc(d) => d.next_picture(),
        }
    }
    /// Non-blocking: only pictures that are already finished.
    fn try_next_picture(&mut self) -> Option<h26x::Picture> {
        match self {
            Dec::H264(d) => d.try_next_picture(),
            Dec::Hevc(d) => d.try_next_picture(),
        }
    }
    fn flush(&mut self) -> h26x::Result<()> {
        match self {
            Dec::H264(d) => d.flush(),
            Dec::Hevc(d) => d.flush(),
        }
    }
    fn warnings(&self) -> u64 {
        match self {
            Dec::H264(d) => d.warnings(),
            Dec::Hevc(d) => d.warnings(),
        }
    }
    fn nal_type(&self, nal: &[u8]) -> u8 {
        match self {
            Dec::H264(_) => nal[0] & 0x1f,
            Dec::Hevc(_) => (nal[0] >> 1) & 0x3f,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: h26xdec <input.264|.265> [out.yuv]");
        std::process::exit(2);
    }
    let path = &args[1];
    let data = std::fs::read(path).expect("read input");
    let mut out = args.get(2).map(|p| std::fs::File::create(p).expect("create output"));
    let lower = path.to_ascii_lowercase();
    let hevc = lower.ends_with(".265") || lower.ends_with(".hevc") || lower.ends_with(".h265");
    let mut dec = if hevc { Dec::Hevc(h26x::hevc::HevcDecoder::new()) } else { Dec::H264(h26x::h264::H264Decoder::new()) };
    let mut n = 0usize;
    // H26XDEC_NOMD5=1 skips hashing (and packing) to time the decoder alone.
    let no_md5 = std::env::var_os("H26XDEC_NOMD5").is_some();
    let mut emit = |pic: h26x::Picture, out: &mut Option<std::fs::File>| {
        if no_md5 && out.is_none() {
            println!("{},{},{},{}x{}", n, pic.poc, pic.decode_index, pic.width, pic.height);
        } else {
            let mut packed = pic.packed();
            if pic.chroma == h26x::ChromaFormat::Monochrome {
                // Like libavcodec's yuv420p output for 4:0:0: grey chroma
                // planes follow the luma, so the hashes compare.
                let (cw, ch) = (pic.width.div_ceil(2) as usize, pic.height.div_ceil(2) as usize);
                let bps = if pic.bit_depth > 8 { 2 } else { 1 };
                let mid = 1u16 << (pic.bit_depth - 1);
                for _ in 0..2 * cw * ch {
                    if bps == 1 {
                        packed.push(mid as u8);
                    } else {
                        packed.extend_from_slice(&mid.to_le_bytes());
                    }
                }
            }
            println!("{},{},{},{}x{},{}", n, pic.poc, pic.decode_index, pic.width, pic.height, md5_hex(&packed));
            if let Some(f) = out {
                f.write_all(&packed).unwrap();
            }
        }
        n += 1;
    };
    // Feed NAL by NAL so an error names the position.
    let mut nals = 0usize;
    for nal in h26x::nal::annexb_nals(&data) {
        nals += 1;
        if let Err(e) = dec.push_nal(nal) {
            eprintln!("error at NAL {nals} (type {}): {e}", dec.nal_type(nal));
            std::process::exit(1);
        }
        // Collect what is ready without stalling the pipeline behind the
        // oldest picture still decoding.
        while let Some(pic) = dec.try_next_picture() {
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
