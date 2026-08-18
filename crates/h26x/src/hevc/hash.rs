//! Decoded picture hash SEI (H.265 D.2.20 / D.3.20): MD5, CRC or checksum
//! per colour component of the decoded (uncropped) picture. With
//! `H26X_VERIFY_HASH=1` the decoder checks every output picture that
//! carried one and counts a warning per mismatch — the conformance
//! bitstreams all carry them, which makes a self-contained correctness
//! check without a reference decoder.

use super::frame::{Frame, Sample};
use crate::bitreader::BitReader;
use crate::picture::ChromaFormat;

/// One picture's expected hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PictureHash {
    /// `hash_type` 0: MD5 of each component's sample bytes.
    Md5([[u8; 16]; 3]),
    /// `hash_type` 1: CRC-16 (CCITT polynomial) of each component.
    Crc([u16; 3]),
    /// `hash_type` 2: the position-masked byte checksum of each component.
    Checksum([u32; 3]),
}

/// Whether hash verification is on (`H26X_VERIFY_HASH=1`).
pub fn verify_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("H26X_VERIFY_HASH").is_some_and(|v| v == "1" || v == "true"))
}

/// Find a `decoded_picture_hash` message in an SEI RBSP (the payload after
/// the two NAL header bytes, emulation prevention removed).
pub fn parse_sei(rbsp: &[u8], chroma_format_idc: u32) -> Option<PictureHash> {
    let mut i = 0usize;
    while i + 2 <= rbsp.len() {
        // payload_type / payload_size: ff-extended bytes.
        let mut ptype = 0usize;
        while i < rbsp.len() && rbsp[i] == 0xFF {
            ptype += 255;
            i += 1;
        }
        ptype += *rbsp.get(i)? as usize;
        i += 1;
        let mut psize = 0usize;
        while i < rbsp.len() && rbsp[i] == 0xFF {
            psize += 255;
            i += 1;
        }
        psize += *rbsp.get(i)? as usize;
        i += 1;
        let payload = rbsp.get(i..i + psize)?;
        i += psize;
        if ptype == 132 {
            let comps = if chroma_format_idc == 0 { 1 } else { 3 };
            let mut r = BitReader::new(payload);
            let hash_type = r.bits(8);
            return match hash_type {
                0 => {
                    let mut m = [[0u8; 16]; 3];
                    for c in m.iter_mut().take(comps) {
                        for b in c.iter_mut() {
                            *b = r.bits(8) as u8;
                        }
                    }
                    Some(PictureHash::Md5(m))
                }
                1 => {
                    let mut v = [0u16; 3];
                    for c in v.iter_mut().take(comps) {
                        *c = r.bits(16) as u16;
                    }
                    Some(PictureHash::Crc(v))
                }
                2 => {
                    let mut v = [0u32; 3];
                    for c in v.iter_mut().take(comps) {
                        *c = r.bits(32);
                    }
                    Some(PictureHash::Checksum(v))
                }
                _ => None,
            };
        }
        // rbsp trailing bits after the last message.
        if i < rbsp.len() && rbsp[i] == 0x80 && i + 1 == rbsp.len() {
            break;
        }
    }
    None
}

/// Check `frame` against `hash`; the mismatching components on failure.
pub fn verify<S: Sample>(frame: &Frame<S>, hash: &PictureHash) -> Result<(), String> {
    let comps = if frame.chroma == ChromaFormat::Monochrome { 1 } else { 3 };
    let mut bad = Vec::new();
    for c in 0..comps {
        let plane = match c {
            0 => &frame.y,
            1 => &frame.cb,
            _ => &frame.cr,
        };
        let bd = frame.bit_depth;
        let wide = bd > 8;
        // The component's samples in raster order, as bytes (low byte first
        // when the depth needs two).
        let mut bytes = Vec::with_capacity(plane.width * plane.height * if wide { 2 } else { 1 });
        for y in 0..plane.height {
            let off = plane.offset(0, y as isize);
            for &s in &plane.data[off..off + plane.width] {
                let v = s.to_i32() as u32;
                bytes.push((v & 0xFF) as u8);
                if wide {
                    bytes.push((v >> 8) as u8);
                }
            }
        }
        let ok = match hash {
            PictureHash::Md5(m) => md5(&bytes) == m[c],
            PictureHash::Crc(v) => crc16(&bytes) == v[c],
            PictureHash::Checksum(v) => checksum(&bytes, plane.width, wide) == v[c],
        };
        if !ok {
            bad.push(c);
        }
    }
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!("component(s) {bad:?} differ from the picture hash SEI"))
    }
}

/// CRC as specified: 16-bit, polynomial 0x1021, seeded 0xFFFF, bit by
/// bit MSB first, then flushed with 16 zero bits.
fn crc16(data: &[u8]) -> u16 {
    let mut crc: u32 = 0xFFFF;
    for &b in data {
        for bit in 0..8 {
            let msb = (crc >> 15) & 1;
            let v = ((b >> (7 - bit)) & 1) as u32;
            crc = (((crc << 1) + v) & 0xFFFF) ^ (msb * 0x1021);
        }
    }
    for _ in 0..16 {
        let msb = (crc >> 15) & 1;
        crc = ((crc << 1) & 0xFFFF) ^ (msb * 0x1021);
    }
    crc as u16
}

/// Checksum as specified: bytes xor-masked by their sample position.
fn checksum(bytes: &[u8], width: usize, wide: bool) -> u32 {
    let per = if wide { 2 } else { 1 };
    let mut sum: u32 = 0;
    for (i, chunk) in bytes.chunks_exact(per).enumerate() {
        let x = i % width;
        let y = i / width;
        let mask = ((x & 0xFF) ^ (y & 0xFF) ^ (x >> 8) ^ (y >> 8)) as u32;
        sum = sum.wrapping_add((chunk[0] as u32) ^ mask);
        if wide {
            sum = sum.wrapping_add((chunk[1] as u32) ^ mask);
        }
    }
    sum
}

/// MD5 (RFC 1321).
pub fn md5(data: &[u8]) -> [u8; 16] {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11,
        16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: [u32; 64] = std::array::from_fn(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32);
    let mut a0: u32 = 0x67452301;
    let mut b0: u32 = 0xefcdab89;
    let mut c0: u32 = 0x98badcfe;
    let mut d0: u32 = 0x10325476;
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_le_bytes());
    for chunk in msg.chunks_exact(64) {
        let m: [u32; 16] = std::array::from_fn(|i| u32::from_le_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]));
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f2 = f.wrapping_add(a).wrapping_add(k[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f2.rotate_left(S[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    let mut out = [0u8; 16];
    out[..4].copy_from_slice(&a0.to_le_bytes());
    out[4..8].copy_from_slice(&b0.to_le_bytes());
    out[8..12].copy_from_slice(&c0.to_le_bytes());
    out[12..].copy_from_slice(&d0.to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn md5_known_answers() {
        let hex = |d: [u8; 16]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(hex(md5(b"")), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(hex(md5(b"abc")), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(hex(md5(b"The quick brown fox jumps over the lazy dog")), "9e107d9d372bb6826bd81d3542a419d6");
        let long: Vec<u8> = (0..1000u32).map(|i| (i * 7 % 251) as u8).collect();
        assert_eq!(hex(md5(&long)), hex(md5(&long)));
    }
}
