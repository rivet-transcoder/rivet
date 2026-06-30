//! `dOps` body builder per RFC 7845 §4.5.

/// Build the `dOps` body (RFC 7845 §4.5).
///
/// Two layouts:
/// - Family 0 (mono / stereo): 11 bytes total.
/// - Family 1 (surround 1..=8 channels, RFC 7845 §5.1.1.2): 11 + 2 + N
///   bytes total, where N is the channel count. The 11-byte preamble is
///   identical to family 0; the trailer adds StreamCount, CoupledCount,
///   and ChannelMapping (N bytes).
///
/// All multi-byte fields are LITTLE-endian per the RFC. The mux side
/// (container/src/mux.rs::build_dops) reads these LE bytes back and
/// translates to BE for the on-wire dOps box.
///
/// ```text
/// Version:               u8  = 0
/// OutputChannelCount:    u8  (1..=8)
/// PreSkip:               u16 LE
/// InputSampleRate:       u32 LE  (original/source rate)
/// OutputGain:            i16 LE  (0 = no gain change)
/// ChannelMappingFamily:  u8  (0 for 1-2 channels, 1 for 3-8)
/// // Family 1 only:
/// StreamCount:           u8
/// CoupledCount:          u8
/// ChannelMapping[N]:     u8 each (output-channel → encoder-stream index)
/// ```
pub(super) fn build_dops(
    channels: u8,
    pre_skip_48k: u16,
    input_sample_rate: u32,
    ms_meta: Option<(u8, u8, &[u8])>,
) -> Vec<u8> {
    // Choose family based on whether multistream metadata is present.
    let (family, total_len) = match ms_meta {
        None => (0u8, 11usize),
        Some(_) => (1u8, 11 + 2 + channels as usize),
    };

    let mut v = Vec::with_capacity(total_len);
    v.push(0u8); // Version
    v.push(channels);
    v.extend_from_slice(&pre_skip_48k.to_le_bytes());
    v.extend_from_slice(&input_sample_rate.to_le_bytes());
    v.extend_from_slice(&0i16.to_le_bytes()); // OutputGain
    v.push(family);

    if let Some((streams, coupled, mapping)) = ms_meta {
        v.push(streams);
        v.push(coupled);
        // ChannelMapping: one byte per *output* channel; value is the
        // encoder-stream index for that output channel.
        v.extend_from_slice(mapping);
        debug_assert_eq!(mapping.len(), channels as usize);
    }
    debug_assert_eq!(v.len(), total_len);
    v
}
