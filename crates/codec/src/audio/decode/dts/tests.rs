//! Decoder-level tests. The conformance comparison against libavcodec's
//! decode of ffmpeg-made vectors lives in `crates/codec/tests/dts_core.rs`
//! (it needs the fixture files); these cover the frame-level contracts.

use super::*;

/// A synthetic bit-stream header (Table 5-1 fields only), for the refusals
/// that trigger before any audio is parsed.
fn header(ftype: u32, nblks: u32, fsize: u32, amode: u32, sfreq: u32, lff: u32, vernum: u32) -> Vec<u8> {
    pack(&header_bits(ftype, nblks, fsize, amode, sfreq, lff, vernum), fsize)
}

fn pack(bits: &[u8], fsize: u32) -> Vec<u8> {
    let mut out = vec![0u8; bits.len().div_ceil(8)];
    for (i, b) in bits.iter().enumerate() {
        out[i / 8] |= b << (7 - (i % 8));
    }
    out.resize(fsize as usize + 1, 0);
    out
}

/// The Table 5-1 header followed by `SUBFS` = 0 and `PCHS`, as bits.
fn header_bits(ftype: u32, nblks: u32, fsize: u32, amode: u32, sfreq: u32, lff: u32, vernum: u32) -> Vec<u8> {
    let mut bits: Vec<u8> = Vec::new();
    let mut push = |v: u32, n: usize| {
        for i in (0..n).rev() {
            bits.push(((v >> i) & 1) as u8);
        }
    };
    push(CORE_SYNC, 32);
    push(ftype, 1);
    push(31, 5); // SHORT
    push(0, 1); // CPF
    push(nblks, 7);
    push(fsize, 14);
    push(amode, 6);
    push(sfreq, 4);
    push(24, 5); // RATE
    push(0, 1); // FixedBit
    push(0, 4); // DYNF TIMEF AUXF HDCD
    push(0, 3); // EXT_AUDIO_ID
    push(0, 1); // EXT_AUDIO
    push(0, 1); // ASPF
    push(lff, 2);
    push(1, 1); // HFLAG
    push(0, 1); // FILTS
    push(vernum, 4);
    push(0, 2); // CHIST
    push(0, 3); // PCMR
    push(0, 2); // SUMF SUMS
    push(0, 4); // DIALNORM
    // Primary audio coding header: SUBFS=0, PCHS=channels-1, then zeros.
    push(0, 4);
    push(AMODE_CHANNELS[amode as usize] as u32 - 1, 3);
    bits
}

/// A 5.1 frame whose first subframe predicts its first subband (`PMODE` = 1).
/// Every other header field is 0, which is valid: two active subbands per
/// channel, none VQ, no joint coding, code books A, `SEL` 0 everywhere (so
/// every `ADJ` is transmitted), one subsubframe.
fn frame_with_adpcm() -> Vec<u8> {
    let mut bits = header_bits(1, 15, 2047, 9, 13, 2, 7);
    // SUBS 5×5, VQSUB 5×5, JOINX 5×3, THUFF 5×2, SHUFF 5×3, BHUFF 5×3, SEL
    // 5×(1 + 4×2 + 5×3), ADJ 5×(2 + 4×2 + 5×2), then SSC 2 + PSC 3.
    bits.extend(std::iter::repeat_n(0, 25 + 25 + 15 + 10 + 15 + 15 + 120 + 100 + 5));
    bits.push(1); // PMODE[0][0]
    bits.extend(std::iter::repeat_n(0, 9 + 12)); // the other PMODEs, PVQ[0][0]
    pack(&bits, 2047)
}

#[test]
fn adpcm_prediction_is_refused_by_name() {
    let mut d = DtsDecoder::new(48_000, 6).unwrap();
    let err = d.decode(&frame_with_adpcm(), 0).unwrap_err();
    assert!(matches!(err, AudioError::Unsupported(_)), "{err}");
    assert!(err.to_string().contains("ADPCM prediction (PMODE = 1 in 1 subbands)"), "{err}");
    assert!(err.to_string().contains("D.10.1"), "{err}");
}

#[test]
fn rejects_non_dts_packets_by_name() {
    let mut d = DtsDecoder::new(48_000, 6).unwrap();
    let err = d.decode(&[0x0B, 0x77, 0, 0, 0, 0, 0, 0, 0, 0], 0).unwrap_err();
    assert!(err.to_string().contains("0x7FFE8001"), "{err}");
}

#[test]
fn a_short_packet_is_truncated_not_zero_filled() {
    let mut d = DtsDecoder::new(48_000, 6).unwrap();
    let f = header(1, 15, 2047, 9, 13, 2, 7);
    let err = d.decode(&f[..100], 0).unwrap_err();
    assert!(err.to_string().contains("truncated"), "{err}");
}

#[test]
fn termination_frames_are_refused_by_name() {
    let mut d = DtsDecoder::new(48_000, 6).unwrap();
    let f = header(0, 15, 2047, 9, 13, 2, 7);
    let err = d.decode(&f, 0).unwrap_err();
    assert!(matches!(err, AudioError::Unsupported(_)), "{err}");
    assert!(err.to_string().contains("termination frame"), "{err}");
}

#[test]
fn incompatible_encoder_revisions_are_refused_by_name() {
    let mut d = DtsDecoder::new(48_000, 6).unwrap();
    let f = header(1, 15, 2047, 9, 13, 2, 9);
    let err = d.decode(&f, 0).unwrap_err();
    assert!(err.to_string().contains("VERNUM 9"), "{err}");
}

#[test]
fn arrangements_without_a_pipeline_layout_are_refused_by_name() {
    // AMODE 10 = CL + CR + L + R + SL + SR needs XCh.
    let err = output_layout(10, false).unwrap_err();
    assert!(err.to_string().contains("XCh"), "{err}");
    // 3.0 + LFE has no named layout.
    let err = output_layout(5, true).unwrap_err();
    assert!(err.to_string().contains("3.1"), "{err}");
    // L + R + S.
    assert!(output_layout(6, false).is_err());
    // And a frame carrying one is refused before any audio is parsed.
    let mut d = DtsDecoder::new(48_000, 6).unwrap();
    let f = header(1, 15, 2047, 10, 13, 0, 7);
    let err = d.decode(&f, 0).unwrap_err();
    assert!(matches!(err, AudioError::Unsupported(_)), "{err}");
}

#[test]
fn the_five_one_core_maps_onto_the_family_1_order() {
    use Slot::*;
    // Core order is C L R SL SR (+LFE); pipeline order is FL FR FC LFE SL SR.
    let (name, slots) = output_layout(9, true).unwrap();
    assert_eq!(name, "5.1(side)");
    assert_eq!(slots, vec![Core(1), Core(2), Core(0), Lfe, Core(3), Core(4)]);
    let (name, slots) = output_layout(2, true).unwrap();
    assert_eq!(name, "2.1");
    assert_eq!(slots, vec![Core(0), Core(1), Lfe]);
    let (name, slots) = output_layout(0, false).unwrap();
    assert_eq!(name, "mono");
    assert_eq!(slots, vec![Core(0)]);
}

#[test]
fn sum_difference_pairs_follow_amode_channel_order() {
    assert_eq!(sum_diff_pairs(9), (Some((1, 2)), Some((3, 4))), "C L R SL SR");
    assert_eq!(sum_diff_pairs(2), (Some((0, 1)), None), "L R");
    assert_eq!(sum_diff_pairs(0), (None, None));
}

#[test]
fn decoder_construction_bounds_the_channel_count() {
    assert!(DtsDecoder::new(48_000, 0).is_err());
    assert!(DtsDecoder::new(48_000, 7).is_err());
    assert!(DtsDecoder::new(48_000, 6).is_ok());
}
