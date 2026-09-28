//! What the FLAC and ALAC codecs share: MSB-first bit I/O, linear
//! prediction analysis, and the conversion between integer PCM and the
//! pipeline's f32 samples.
//!
//! # Integer PCM through an f32 pipeline
//!
//! The pipeline carries audio as f32 in `[-1.0, 1.0]`. An integer sample of
//! `bits` bits maps to `s / 2^(bits-1)`, and back by the inverse: a power of
//! two scale, so the round trip is exact whenever the integer fits the f32
//! significand — every depth up to 24 bits. A 32-bit source loses its low
//! bits on the way through; the decoders' integer output
//! ([`crate::audio::decode::flac::FlacDecoder::decode_int`], …) is exact at
//! every depth.

pub mod alac;
pub(crate) mod bits;
pub mod flac;
pub(crate) mod lpc;

/// An integer sample of `bits` bits as the pipeline's f32.
pub(crate) fn int_to_f32(s: i32, bits: u32) -> f32 {
    (f64::from(s) / (1u64 << (bits - 1)) as f64) as f32
}

/// A pipeline f32 sample as a `bits`-bit integer: scaled, rounded to the
/// nearest, and clipped to the range. Exact inverse of [`int_to_f32`].
pub(crate) fn f32_to_int(x: f32, bits: u32) -> i32 {
    let scale = (1u64 << (bits - 1)) as f64;
    let max = scale - 1.0;
    (f64::from(x) * scale).round().clamp(-scale, max) as i32
}

/// Interleaved integer samples as the pipeline's f32.
pub(crate) fn ints_to_f32(samples: &[i32], bits: u32) -> Vec<f32> {
    samples.iter().map(|&s| int_to_f32(s, bits)).collect()
}

/// `v` sign-extended from its low `bits` bits.
pub(crate) fn sign_extend(v: i64, bits: u32) -> i64 {
    if bits >= 64 {
        return v;
    }
    let shift = 64 - bits;
    (v << shift) >> shift
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_pcm_survives_the_f32_pipeline_up_to_24_bits() {
        for bits in [8u32, 16, 20, 24] {
            let lo = -(1i32 << (bits - 1));
            let hi = (1i32 << (bits - 1)) - 1;
            for s in [lo, lo + 1, -1, 0, 1, hi - 1, hi, lo / 3, hi / 7] {
                assert_eq!(f32_to_int(int_to_f32(s, bits), bits), s, "{bits}-bit {s}");
            }
        }
        assert_eq!(f32_to_int(1.5, 16), 32767, "clipped");
        assert_eq!(f32_to_int(-1.5, 16), -32768, "clipped");
    }

    #[test]
    fn sign_extension() {
        assert_eq!(sign_extend(0x1FFFF, 17), -1);
        assert_eq!(sign_extend(0xFFFF, 17), 0xFFFF);
        assert_eq!(sign_extend(-5, 64), -5);
    }
}
