//! Sampling-rate dependent tables of ISO/IEC 13818-7:2004: the
//! sampling_frequency_index (Table 35), the scalefactor band offsets
//! (Tables 45 to 53).

/// Scalefactor band offsets for long windows at 44.1 and 48 kHz (Table 45):
/// 49 bands, the last entry closing the final band at 1024.
const SWB_LONG_48: [u16; 50] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 1024,
];

/// Long windows at 32 kHz (Table 47): 51 bands.
const SWB_LONG_32: [u16; 52] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120, 132, 144, 160,
    176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
    736, 768, 800, 832, 864, 896, 928, 960, 992, 1024,
];

/// Long windows at 22.05 and 24 kHz (Table 52): 47 bands.
const SWB_LONG_24: [u16; 48] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 136,
    148, 160, 172, 188, 204, 220, 240, 260, 284, 308, 336, 364, 396, 432, 468, 508, 552, 600, 652,
    704, 768, 832, 896, 960, 1024,
];

/// Short windows at 32, 44.1 and 48 kHz (Table 46): 14 bands.
const SWB_SHORT_48: [u16; 15] = [0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96, 112, 128];

/// Short windows at 22.05 and 24 kHz (Table 53): 15 bands.
const SWB_SHORT_24: [u16; 16] = [
    0, 4, 8, 12, 16, 20, 24, 28, 36, 44, 52, 64, 76, 92, 108, 128,
];

/// Everything the encoder needs to know about one supported sampling rate.
#[derive(Debug, Clone, Copy)]
pub(super) struct RateTables {
    pub rate: u32,
    /// sampling_frequency_index (Table 35).
    pub index: u8,
    pub swb_long: &'static [u16],
    pub swb_short: &'static [u16],
}

/// The sampling rates this encoder codes natively. Other rates are the
/// caller's to resample; the standard's rates below 22.05 kHz and above
/// 48 kHz are left out on purpose (speech-band and high-resolution rates are
/// not what AAC-LC delivery to the web needs).
pub(super) fn rate_tables(rate: u32) -> Option<RateTables> {
    let t = |index, swb_long: &'static [u16], swb_short: &'static [u16]| RateTables {
        rate,
        index,
        swb_long,
        swb_short,
    };
    match rate {
        48_000 => Some(t(3, &SWB_LONG_48, &SWB_SHORT_48)),
        44_100 => Some(t(4, &SWB_LONG_48, &SWB_SHORT_48)),
        32_000 => Some(t(5, &SWB_LONG_32, &SWB_SHORT_48)),
        24_000 => Some(t(6, &SWB_LONG_24, &SWB_SHORT_24)),
        22_050 => Some(t(7, &SWB_LONG_24, &SWB_SHORT_24)),
        _ => None,
    }
}

/// The rates [`rate_tables`] accepts, for error messages and callers that
/// pick a resampling target.
pub const SUPPORTED_RATES: [u32; 5] = [48_000, 44_100, 32_000, 24_000, 22_050];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_tables_are_increasing_multiples_of_four_ending_at_the_transform_size() {
        for rate in SUPPORTED_RATES {
            let t = rate_tables(rate).unwrap();
            for (swb, end) in [(t.swb_long, 1024), (t.swb_short, 128)] {
                assert_eq!(swb[0], 0);
                assert_eq!(*swb.last().unwrap(), end);
                for w in swb.windows(2) {
                    assert!(w[1] > w[0] && (w[1] - w[0]) % 4 == 0, "{rate}: {w:?}");
                }
            }
        }
        assert_eq!(rate_tables(48_000).unwrap().swb_long.len(), 50);
        assert_eq!(rate_tables(32_000).unwrap().swb_long.len(), 52);
        assert_eq!(rate_tables(22_050).unwrap().swb_long.len(), 48);
        assert_eq!(rate_tables(22_050).unwrap().swb_short.len(), 16);
    }
}
