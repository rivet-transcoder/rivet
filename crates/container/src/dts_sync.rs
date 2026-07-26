//! DTS Coherent Acoustics core sync-frame parsing, for `dtsc` passthrough.
//!
//! rivet has no DTS **decoder** — the DCA quantisation and Huffman tables are
//! normative data that would have to be transcribed from the spec, the same
//! blocker AC-3 decode has (see TODO.md). What it does have, and what this
//! module enables, is **passthrough**: a DTS track can be carried into MP4
//! byte-for-byte, exactly as AC-3 and E-AC-3 already are.
//!
//! That only needs the core substream's frame header, which is a fixed field
//! layout — no tables beyond two small enumerations (sample rate and channel
//! arrangement).
//!
//! Reference: ETSI TS 102 114 §5.3 (core frame header) and Annex E / the DTS
//! 9302J81100 MP4 mapping for the `DTSSpecificBox`.

use thiserror::Error;

/// Core substream sync word, 16-bit big-endian ("normal" DTS). The 14-bit and
/// little-endian variants exist but only ever appear in raw `.dts`/`.cpt`
/// captures, never inside Matroska or MP4, so they're rejected rather than
/// silently half-parsed.
pub const DTS_CORE_SYNC: u32 = 0x7FFE_8001;

/// Substream sync word for DTS-HD extensions (DTS-HD MA / High Resolution).
/// A DTS-HD track is a core frame followed by one of these; passthrough keeps
/// both, so this is only used to recognise that the stream is DTS-HD.
pub const DTS_HD_SYNC: u32 = 0x6472_7473;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DtsError {
    #[error("DTS: buffer too short for a core frame header ({0} bytes)")]
    TooShort(usize),
    #[error("DTS: no 16-bit big-endian core sync word (0x7FFE8001) at the start of the frame")]
    NoSync,
    #[error("DTS: reserved/invalid sample-rate code {0}")]
    BadSampleRate(u8),
}

/// The fields of a DTS core sync frame that the container layer needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DtsSyncInfo {
    /// Sample rate in Hz, from `SFREQ`.
    pub sample_rate: u32,
    /// Total channel count including LFE.
    pub channels: u16,
    /// Whether the low-frequency-effects channel is present (`LFF` != 0).
    pub lfe: bool,
    /// `AMODE` — the channel arrangement code, kept verbatim for `ddts`.
    pub amode: u8,
    /// Core frame size in bytes (`FSIZE + 1`).
    pub frame_size: usize,
    /// PCM samples per channel in this frame: `(NBLKS + 1) * 32`.
    pub samples_per_frame: u32,
    /// `RATE` — the transmission bitrate code.
    pub rate_code: u8,
    /// Nominal bitrate in bits/s, or `None` for the open/variable codes.
    pub bit_rate: Option<u32>,
}

/// `SFREQ` → Hz (ETSI TS 102 114 Table 5-5). Zero marks a reserved code.
const SAMPLE_RATES: [u32; 16] = [
    0, 8_000, 16_000, 32_000, 0, 0, 11_025, 22_050, 44_100, 0, 0, 12_000, 24_000, 48_000, 0, 0,
];

/// `RATE` → bits/s (Table 5-7). The last three codes are open / variable /
/// lossless, which have no single nominal rate.
const BIT_RATES: [u32; 32] = [
    32_000, 56_000, 64_000, 96_000, 112_000, 128_000, 192_000, 224_000, 256_000, 320_000, 384_000,
    448_000, 512_000, 576_000, 640_000, 768_000, 960_000, 1_024_000, 1_152_000, 1_280_000,
    1_344_000, 1_408_000, 1_411_200, 1_472_000, 1_536_000, 1_920_000, 2_048_000, 3_072_000,
    3_840_000, 0, 0, 0,
];

/// `AMODE` → number of *full-range* channels (Table 5-4). LFE is counted
/// separately via `LFF`. Codes past the table are custom arrangements, which
/// passthrough can't describe — callers fall back to the container's own
/// channel count.
const AMODE_CHANNELS: [u16; 16] = [1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 6, 6, 6, 7, 8, 8];

/// Number of full-range channels for an `AMODE` code, if it's one of the
/// standard arrangements.
pub fn amode_channels(amode: u8) -> Option<u16> {
    AMODE_CHANNELS.get(amode as usize).copied()
}

/// Parse a DTS **core** sync frame header from the start of `bytes`.
///
/// Bit layout after the 32-bit sync word (ETSI TS 102 114 §5.3.1):
///
/// ```text
///   FTYPE   1    frame type
///   SHORT   5    deficit sample count
///   CPF     1    CRC present
///   NBLKS   7    blocks in frame, minus one   -> 32 PCM samples each
///   FSIZE  14    frame size in bytes, minus one
///   AMODE   6    channel arrangement
///   SFREQ   4    sample rate code
///   RATE    5    bitrate code
///   (1 reserved) (FixedBit)
///   DYNF    1
///   TIMEF   1
///   AUXF    1
///   HDCD    1
///   EXT_AUDIO_ID 3
///   EXT_AUDIO    1
///   ASPF         1
///   LFF          2    low-frequency effects flag
/// ```
pub fn parse_core_sync(bytes: &[u8]) -> Result<DtsSyncInfo, DtsError> {
    // Sync (4) + the fields above run to bit 78, so 10 bytes is the minimum.
    if bytes.len() < 14 {
        return Err(DtsError::TooShort(bytes.len()));
    }
    let sync = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if sync != DTS_CORE_SYNC {
        return Err(DtsError::NoSync);
    }

    let mut r = BitCursor { bytes, bit: 32 };
    let _ftype = r.bits(1);
    let _short = r.bits(5);
    let _cpf = r.bits(1);
    let nblks = r.bits(7);
    let fsize = r.bits(14);
    let amode = r.bits(6) as u8;
    let sfreq = r.bits(4) as u8;
    let rate_code = r.bits(5) as u8;
    let _fixed = r.bits(1);
    let _dynf = r.bits(1);
    let _timef = r.bits(1);
    let _auxf = r.bits(1);
    let _hdcd = r.bits(1);
    let _ext_audio_id = r.bits(3);
    let _ext_audio = r.bits(1);
    let _aspf = r.bits(1);
    let lff = r.bits(2);

    let sample_rate = SAMPLE_RATES[sfreq as usize & 0xF];
    if sample_rate == 0 {
        return Err(DtsError::BadSampleRate(sfreq));
    }
    // LFF: 0 = none, 1 = present (128-sample interpolation), 2 = present (64),
    // 3 = invalid. Treat the invalid code as "no LFE" rather than failing the
    // whole frame — the channel count comes out one low at worst.
    let lfe = lff == 1 || lff == 2;
    let full_range = amode_channels(amode).unwrap_or(2);

    Ok(DtsSyncInfo {
        sample_rate,
        channels: full_range + u16::from(lfe),
        lfe,
        amode,
        frame_size: fsize as usize + 1,
        samples_per_frame: (nblks + 1) * 32,
        rate_code,
        bit_rate: match BIT_RATES.get(rate_code as usize & 0x1F).copied() {
            Some(0) | None => None,
            Some(b) => Some(b),
        },
    })
}

/// Whether a DTS-HD extension substream follows the core frame — i.e. the
/// track is DTS-HD MA / High Resolution rather than plain DTS.
pub fn has_hd_extension(bytes: &[u8], core: &DtsSyncInfo) -> bool {
    let at = core.frame_size;
    bytes.len() >= at + 4
        && u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
            == DTS_HD_SYNC
}

/// Big-endian MSB-first bit reader over a byte slice. Reads past the end
/// return zero, which the length check above already rules out for the header.
struct BitCursor<'a> {
    bytes: &'a [u8],
    bit: usize,
}

impl BitCursor<'_> {
    fn bits(&mut self, n: usize) -> u32 {
        let mut v = 0u32;
        for _ in 0..n {
            let byte = self.bytes.get(self.bit >> 3).copied().unwrap_or(0);
            let b = (byte >> (7 - (self.bit & 7))) & 1;
            v = (v << 1) | b as u32;
            self.bit += 1;
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic core frame header with the given field values, so the
    /// bit offsets can be checked without a real DTS file.
    fn frame(nblks: u32, fsize: u32, amode: u32, sfreq: u32, rate: u32, lff: u32) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();
        let mut push = |v: u32, n: usize| {
            for i in (0..n).rev() {
                bits.push(((v >> i) & 1) as u8);
            }
        };
        push(DTS_CORE_SYNC, 32);
        push(0, 1); // FTYPE
        push(0, 5); // SHORT
        push(0, 1); // CPF
        push(nblks, 7);
        push(fsize, 14);
        push(amode, 6);
        push(sfreq, 4);
        push(rate, 5);
        push(0, 1); // FixedBit
        push(0, 1); // DYNF
        push(0, 1); // TIMEF
        push(0, 1); // AUXF
        push(0, 1); // HDCD
        push(0, 3); // EXT_AUDIO_ID
        push(0, 1); // EXT_AUDIO
        push(0, 1); // ASPF
        push(lff, 2);
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let mut out = vec![0u8; bits.len() / 8];
        for (i, b) in bits.iter().enumerate() {
            out[i / 8] |= b << (7 - (i % 8));
        }
        out.resize(out.len().max(16), 0);
        out
    }

    #[test]
    fn parses_a_48k_5_1_frame() {
        // sfreq 13 = 48 kHz, amode 9 = 5 full-range channels, LFF=1 adds LFE.
        let f = frame(15, 2011, 9, 13, 24, 1);
        let info = parse_core_sync(&f).unwrap();
        assert_eq!(info.sample_rate, 48_000);
        assert_eq!(info.amode, 9);
        assert!(info.lfe);
        assert_eq!(info.channels, 6, "5 full-range + LFE");
        assert_eq!(info.frame_size, 2012, "FSIZE is size-minus-one");
        assert_eq!(info.samples_per_frame, 512, "(15+1) blocks x 32 samples");
        assert_eq!(info.bit_rate, Some(1_536_000));
    }

    #[test]
    fn stereo_without_lfe() {
        let f = frame(31, 1023, 2, 13, 10, 0);
        let info = parse_core_sync(&f).unwrap();
        assert_eq!(info.channels, 2);
        assert!(!info.lfe);
        assert_eq!(info.samples_per_frame, 1024);
        assert_eq!(info.bit_rate, Some(384_000));
    }

    #[test]
    fn rejects_a_non_dts_buffer() {
        assert_eq!(parse_core_sync(&[0u8; 32]), Err(DtsError::NoSync));
        // An AC-3 frame must not be mistaken for DTS.
        let mut ac3 = vec![0u8; 32];
        ac3[0] = 0x0B;
        ac3[1] = 0x77;
        assert_eq!(parse_core_sync(&ac3), Err(DtsError::NoSync));
        assert!(matches!(parse_core_sync(&[0x7F, 0xFE]), Err(DtsError::TooShort(2))));
    }

    #[test]
    fn rejects_reserved_sample_rates() {
        for bad in [0u32, 4, 5, 9, 10, 14, 15] {
            let f = frame(15, 2011, 9, bad, 24, 1);
            assert!(
                matches!(parse_core_sync(&f), Err(DtsError::BadSampleRate(_))),
                "sfreq {bad} should be rejected"
            );
        }
    }

    #[test]
    fn variable_bitrate_codes_have_no_nominal_rate() {
        // 29..31 are open / variable / lossless.
        for open in [29u32, 30, 31] {
            let f = frame(15, 2011, 9, 13, open, 1);
            assert_eq!(parse_core_sync(&f).unwrap().bit_rate, None);
        }
    }

    #[test]
    fn detects_an_hd_extension_after_the_core() {
        let mut f = frame(15, 511, 9, 13, 24, 1);
        let core = parse_core_sync(&f).unwrap();
        assert_eq!(core.frame_size, 512);
        assert!(!has_hd_extension(&f, &core), "nothing after the core yet");
        f.resize(512, 0);
        f.extend_from_slice(&DTS_HD_SYNC.to_be_bytes());
        assert!(has_hd_extension(&f, &core), "DTS-HD substream should be seen");
    }

    #[test]
    fn amode_table_covers_the_standard_arrangements() {
        assert_eq!(amode_channels(0), Some(1), "mono");
        assert_eq!(amode_channels(2), Some(2), "L/R stereo");
        assert_eq!(amode_channels(9), Some(5), "3F2R — the 5.1 core");
        assert_eq!(amode_channels(16), None, "custom arrangements aren't described");
    }
}
