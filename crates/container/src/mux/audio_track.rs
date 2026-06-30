use crate::AudioInfo;
use crate::ac3_sync::{Ac3SyncInfo, Eac3SyncInfo};
use super::boxes::BoxBuilder;
use super::boxes::write_unity_matrix;
use super::video_track::{build_mdhd, build_dinf};
use super::sample_table::{AudioBuildPlan, build_stsc, build_stsz, build_stco, build_co64};
use super::AudioCodecKind;

// ---- Audio trak / mdia / minf / stbl / mp4a / esds ---------------------------
// These layers match ISO/IEC 14496-12/14 for an AAC sound track sharing
// mdat with the video track. Offsets are supplied by the finalize planner;
// the builders just embed them.

pub(super) fn build_audio_trak(
    plan: &AudioBuildPlan,
    duration_in_movie_ts: u64,
    chunk_offsets: &[u64],
    use_co64: bool,
) -> Vec<u8> {
    let tkhd = build_audio_tkhd(duration_in_movie_ts);
    let mdia = build_audio_mdia(plan, chunk_offsets, use_co64);

    let mut b = BoxBuilder::new(b"trak");
    b.extend(&tkhd);
    b.extend(&mdia);
    b.finish()
}

fn build_audio_tkhd(duration_in_movie_ts: u64) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"tkhd");
    b.u8(0); // version
    b.extend(&[0, 0, 0x03]); // flags: track_enabled | track_in_movie
    b.u32(0); // creation_time
    b.u32(0); // modification_time
    b.u32(2); // track_ID (audio is track 2)
    b.u32(0); // reserved
    b.u32(duration_in_movie_ts as u32);
    b.u32(0); // reserved
    b.u32(0);
    b.u16(0); // layer
    b.u16(0x0001); // alternate_group (1 for audio; lets players swap tracks within the group)
    b.u16(0x0100); // volume 1.0 (audio)
    b.u16(0); // reserved
    write_unity_matrix(&mut b);
    b.u32(0); // width = 0 for audio
    b.u32(0); // height = 0 for audio
    b.finish()
}

fn build_audio_mdia(plan: &AudioBuildPlan, chunk_offsets: &[u64], use_co64: bool) -> Vec<u8> {
    let mdhd = build_mdhd(plan.info.timescale, plan.total_duration_in_own_ts);
    let hdlr = build_audio_hdlr();
    let minf = build_audio_minf(plan, chunk_offsets, use_co64);

    let mut b = BoxBuilder::new(b"mdia");
    b.extend(&mdhd);
    b.extend(&hdlr);
    b.extend(&minf);
    b.finish()
}

fn build_audio_hdlr() -> Vec<u8> {
    let mut b = BoxBuilder::new(b"hdlr");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(0); // pre_defined
    b.extend(b"soun"); // handler_type
    b.u32(0);
    b.u32(0);
    b.u32(0);
    b.extend(b"SoundHandler\0");
    b.finish()
}

fn build_audio_minf(plan: &AudioBuildPlan, chunk_offsets: &[u64], use_co64: bool) -> Vec<u8> {
    let smhd = build_smhd();
    let dinf = build_dinf();
    let stbl = build_audio_stbl(plan, chunk_offsets, use_co64);

    let mut b = BoxBuilder::new(b"minf");
    b.extend(&smhd);
    b.extend(&dinf);
    b.extend(&stbl);
    b.finish()
}

fn build_smhd() -> Vec<u8> {
    let mut b = BoxBuilder::new(b"smhd");
    b.u8(0);
    b.extend(&[0, 0, 0]); // flags
    b.u16(0); // balance (0 = center)
    b.u16(0); // reserved
    b.finish()
}

fn build_audio_stbl(plan: &AudioBuildPlan, chunk_offsets: &[u64], use_co64: bool) -> Vec<u8> {
    let stsd = build_audio_stsd(&plan.info);
    let stts = build_audio_stts(&plan.durations);
    let stsc = build_stsc(plan.sample_sizes.len() as u32, plan.samples_per_chunk);
    let stsz = build_stsz(&plan.sample_sizes);
    let chunk_offset_box = if use_co64 {
        build_co64(chunk_offsets)
    } else {
        build_stco(chunk_offsets)
    };

    let mut b = BoxBuilder::new(b"stbl");
    b.extend(&stsd);
    b.extend(&stts);
    b.extend(&stsc);
    b.extend(&stsz);
    b.extend(&chunk_offset_box);
    b.finish()
}

pub(crate) fn build_audio_stsd(info: &AudioInfo) -> Vec<u8> {
    // Dispatch on codec — AAC → mp4a + esds; Opus → Opus + dOps;
    // AC-3 → ac-3 + dac3; E-AC-3 → ec-3 + dec3. The AudioSampleEntry
    // preamble is shared (same v0 layout per ISO/IEC 14496-12 §8.5.2.2 =
    // 36 bytes total before child boxes); only the 4-cc and the
    // codec-specific child differ.
    let kind = AudioCodecKind::from_codec_tag(&info.codec)
        .expect("with_audio gate already validated codec tag");
    let entry = match kind {
        AudioCodecKind::Aac => build_mp4a(info),
        AudioCodecKind::Opus => build_opus_sample_entry(info),
        AudioCodecKind::Ac3 => build_ac3_sample_entry(info),
        AudioCodecKind::Eac3 => build_ec3_sample_entry(info),
    };
    let mut b = BoxBuilder::new(b"stsd");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(1); // entry_count
    b.extend(&entry);
    b.finish()
}

/// AudioSampleEntryV0 per ISO/IEC 14496-12 §8.5.2.2, followed by the esds
/// descriptor tree per ISO/IEC 14496-14 / 14496-1 §7.2.6.5.
///
/// `channelcount` reflects the actual decoded-output channel count as
/// surfaced by the demuxer. For HE-AAC v2 PS (1-channel core) the demuxer
/// upmixes to 2; for 5.1 / 7.1 the AAC channelConfiguration is passed
/// straight through (Squad-25). When channels ≥ 3, an Apple `chan`
/// (Channel Layout) box is appended after `esds` so iOS Safari /
/// QuickTime / AVFoundation render the correct multichannel layout
/// rather than defaulting to L+R downmix.
pub(super) fn build_mp4a(info: &AudioInfo) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"mp4a");
    // SampleEntry header
    for _ in 0..6 {
        b.u8(0);
    } // reserved[6] = 0
    b.u16(1); // data_reference_index
    // AudioSampleEntry v0 body
    b.u32(0); // reserved (was version + revision_level in v0 QuickTime)
    b.u32(0); // reserved (vendor in v0 QuickTime)
    b.u16(info.channels); // channel_count (driven by demux)
    b.u16(16); // sample_size (bits)
    b.u16(0); // pre_defined
    b.u16(0); // reserved
    b.u32(info.sample_rate << 16); // samplerate 16.16 fixed-point
    // esds child (carries the AudioSpecificConfig verbatim)
    b.extend(&build_esds(info));
    // Apple Channel Layout (`chan`) box for multichannel AAC. Per
    // QuickTime File Format Spec §"Channel Layout Box" the box nests
    // *inside* the `mp4a` AudioSampleEntry alongside `esds`.
    if let Some(chan) = build_chan_box(info.channels) {
        b.extend(&chan);
    }
    b.finish()
}

/// Apple Channel Layout (`chan`) box for ≥3-channel audio. Per the QuickTime
/// File Format Specification, §"Channel Layout Box", and CoreAudioBaseTypes.h
/// (`AudioChannelLayout`):
///
///   - `mChannelLayoutTag` (u32 BE): one of the standard layout tags. The
///     low 16 bits carry the channel count and the high 16 bits identify
///     the layout. Returned by Apple's `kAudioChannelLayoutTag_*` macros.
///   - `mChannelBitmap` (u32 BE) = 0 — only used when the tag is
///     `kAudioChannelLayoutTag_UseChannelBitmap`.
///   - `mNumberChannelDescriptions` (u32 BE) = 0 — only used when the tag
///     is `kAudioChannelLayoutTag_UseChannelDescriptions`.
///
/// Total payload: 12 bytes. Box size: 20 bytes (8-byte header + 12-byte body).
///
/// Returns `None` for mono / stereo (Apple defaults to standard mono /
/// L+R already, no `chan` box needed). Returns `None` for unsupported
/// channel counts — caller's `with_audio` gate already restricts to the
/// supported set; this function uses `None` as a defence-in-depth.
///
/// Standard layouts emitted (channels in this order in the bitstream):
///   - 5.1 → `kAudioChannelLayoutTag_MPEG_5_1_C` = `(114 << 16) | 6`
///     = `0x00720006`. Channels: L, R, C, LFE, Ls, Rs.
///   - 7.1 → `kAudioChannelLayoutTag_MPEG_7_1_C` = `(127 << 16) | 8`
///     = `0x007F0008`. Channels: L, R, C, LFE, Ls, Rs, Lc, Rc.
///
/// 7.1 + Atmos and other extended / object-based layouts are NOT emitted
/// here (caller's `with_audio` gate already rejects them). Adding a wrong
/// `chan` tag is worse than omitting the box — Apple players would map
/// channels to the wrong speakers.
pub(crate) fn build_chan_box(channels: u16) -> Option<Vec<u8>> {
    let tag: u32 = match channels {
        1 | 2 => return None,    // Apple default is correct
        6 => (114u32 << 16) | 6, // kAudioChannelLayoutTag_MPEG_5_1_C
        7 => (127u32 << 16) | 8, // kAudioChannelLayoutTag_MPEG_7_1_C
        _ => return None,        // unsupported (gate already rejected)
    };
    let mut b = BoxBuilder::new(b"chan");
    b.u32(tag); // mChannelLayoutTag
    b.u32(0); // mChannelBitmap
    b.u32(0); // mNumberChannelDescriptions
    Some(b.finish())
}

/// `Opus` sample entry per RFC 7845 §4.4. Same generic AudioSampleEntry v0
/// layout as `mp4a` (per ISO/IEC 14496-12 §8.5.2.2) followed by the
/// Opus-Specific Box `dOps`.
///
/// 4-cc is `Opus` exactly — capital O lowercase pus, that spelling is
/// load-bearing per RFC 7845 §4.4 ("the four-character code shall be set
/// to 'Opus'"). Lowercase variants like `opus` will be rejected by
/// strict players (e.g. macOS / iOS AVFoundation).
///
/// `samplerate` field at the AudioSampleEntry level is set to
/// 48000 << 16 (16.16 fixed-point form of 48000) to match the
/// `InputSampleRate` we emit inside dOps. Apple's AVFoundation reads this
/// field; storing the source's nominal rate (e.g. 44100) would mismatch
/// the dOps body and confuse strict validators.
///
/// `channelcount` carries the actual decoded output channel count
/// (matches `OutputChannelCount` in dOps for ChannelMappingFamily=0).
pub(super) fn build_opus_sample_entry(info: &AudioInfo) -> Vec<u8> {
    // RFC 7845 §4.4: 4-cc is exactly 'Opus' (capital O).
    let mut b = BoxBuilder::new(b"Opus");
    // SampleEntry header
    for _ in 0..6 {
        b.u8(0);
    } // reserved[6] = 0
    b.u16(1); // data_reference_index
    // AudioSampleEntry v0 body
    b.u32(0); // reserved
    b.u32(0); // reserved
    b.u16(info.channels); // channel_count
    b.u16(16); // sample_size (bits) — informational for Opus
    b.u16(0); // pre_defined
    b.u16(0); // reserved
    // Opus is internally always 48 kHz (RFC 6716). The sample-entry
    // samplerate is the playback / mdhd-aligned rate. Pin to 48000 << 16.
    b.u32(48_000u32 << 16); // samplerate 16.16 fixed-point = 48000
    // dOps child
    b.extend(&build_dops(info));
    b.finish()
}

/// `dOps` Opus-Specific Box per RFC 7845 §4.5.
///
/// Body layout (11 bytes minimum for ChannelMappingFamily=0):
///   - `Version` u8 = 0
///   - `OutputChannelCount` u8
///   - `PreSkip` u16 BE
///   - `InputSampleRate` u32 BE
///   - `OutputGain` i16 BE (Q8 dB; 0 = no gain)
///   - `ChannelMappingFamily` u8
///   - (when family != 0: StreamCount u8 + CoupledCount u8 + ChannelMapping[N])
///
/// Byte-order conversion: the source `codec_private` carries the OpusHead
/// body in **Ogg / WebM little-endian** convention (PreSkip / InputSampleRate
/// / OutputGain are LE) — that's what falls out of WebM/MKV `CodecPrivate`
/// directly, and what an Opus encoder library (libopusenc) emits when
/// asked for OpusHead. RFC 7845 §4.5 mandates **big-endian** for the same
/// fields inside `dOps`. We translate field-by-field rather than copying
/// bytes verbatim.
///
/// `Version`: OpusHead carries Version=1 (its own encoding); RFC 7845 §4.5
/// requires Version=0 in dOps (this is THE box version, not the Opus
/// stream version). We force-write 0 here regardless of what the input
/// `codec_private[0]` says.
pub(super) fn build_dops(info: &AudioInfo) -> Vec<u8> {
    let p = &info.codec_private;
    debug_assert!(
        p.len() >= 11,
        "with_audio gate must enforce dOps minimum size"
    );

    // OpusHead → dOps numeric field translation.
    // Layout of input bytes (OpusHead, after the 8-byte 'OpusHead' magic
    // which the demuxer already strips):
    //   [0]    Version (u8) — OpusHead version, NOT the dOps version
    //   [1]    OutputChannelCount (u8)
    //   [2..4] PreSkip (u16 LE)
    //   [4..8] InputSampleRate (u32 LE)
    //   [8..10] OutputGain (i16 LE, Q8 dB)
    //   [10]   ChannelMappingFamily (u8)
    //   // Family != 0 trailer (Squad-28, RFC 7845 §5.1.1):
    //   [11]   StreamCount (u8)
    //   [12]   CoupledCount (u8)
    //   [13..13+N]  ChannelMapping (u8 per output channel)
    let output_channels = p[1];
    let pre_skip = u16::from_le_bytes([p[2], p[3]]);
    let input_sample_rate = u32::from_le_bytes([p[4], p[5], p[6], p[7]]);
    let output_gain = i16::from_le_bytes([p[8], p[9]]);
    let channel_mapping_family = p[10];

    let mut b = BoxBuilder::new(b"dOps");
    b.u8(0); // Version (RFC 7845 §4.5: MUST be 0)
    b.u8(output_channels); // OutputChannelCount
    b.u16(pre_skip); // PreSkip (BE — was LE in OpusHead)
    b.u32(input_sample_rate); // InputSampleRate (BE)
    // i16 output gain → wire u16 BE (two's complement preserved across the cast).
    b.u16(output_gain as u16); // OutputGain (BE Q8)
    b.u8(channel_mapping_family); // ChannelMappingFamily

    // ChannelMappingFamily != 0 → ChannelMappingTable follows
    // (RFC 7845 §5.1.1). For family 1 (Squad-28 multichannel) the
    // table is StreamCount + CoupledCount + ChannelMapping[N]. The
    // encoder packed these immediately after the 11-byte preamble in
    // its `extra_data()` output, so the demuxed `codec_private` buffer
    // already carries them in the correct order — we copy verbatim
    // (no endianness conversion: u8 fields).
    if channel_mapping_family != 0 {
        // with_audio's family-1 validation gate ensured codec_private
        // has the trailing bytes; this assert is just forward protection
        // against a future caller bypassing the gate.
        let trailer_len = 2 + output_channels as usize;
        debug_assert!(
            p.len() >= 11 + trailer_len,
            "family={channel_mapping_family} requires {trailer_len} more bytes after the 11-byte preamble; codec_private has {}",
            p.len()
        );
        b.u8(p[11]); // StreamCount
        b.u8(p[12]); // CoupledCount
        for i in 0..output_channels as usize {
            b.u8(p[13 + i]); // ChannelMapping[i]
        }
    }

    b.finish()
}

// ---- Squad-26: AC-3 / E-AC-3 sample entries + dac3 / dec3 boxes --------------
//
// Per ETSI TS 102 366 v1.4.1 Annex F:
//   §F.2 — AC-3 in MP4 / 3GP: 4cc 'ac-3' AudioSampleEntry + 'dac3' config box.
//   §F.4 — `dac3` body layout (3 bytes total payload, 11-byte total box):
//     fscod         2 bits   (0=48k 1=44.1k 2=32k)
//     bsid          5 bits   (=8 for AC-3 — verified from sync header)
//     bsmod         3 bits
//     acmod         3 bits
//     lfeon         1 bit
//     bit_rate_code 5 bits
//     reserved      5 bits   = 0
//   §F.5 — E-AC-3: 4cc 'ec-3' + 'dec3' config box.
//   §F.6 — `dec3` body: data_rate (13b) + num_ind_sub-1 (3b) followed by
//     N independent-substream descriptors (3 bytes each, plus 9-bit
//     chan_loc when num_dep_sub>0). Squad-26 emits the single-substream
//     case (5 bytes total payload, 13-byte box).
//
// Squad-26 hard-restricts to:
//   - AC-3 5.1 / stereo / mono (acmod 1, 2, 7 with optional LFE)
//   - E-AC-3 single independent substream (num_ind_sub=0 wire encoding,
//     num_dep_sub=0). Vanilla 5.1 is the dominant case in the wild.

/// `ac-3` AudioSampleEntry per ETSI TS 102 366 §F.2. Same generic
/// AudioSampleEntry v0 layout (per ISO/IEC 14496-12 §8.5.2.2) as `mp4a` /
/// `Opus` — 28-byte fixed body after the box header — followed by the
/// `dac3` Config Box.
///
/// 4cc is `ac-3` exactly (with the hyphen, ASCII bytes 0x61 0x63 0x2D
/// 0x33). NOT `ac3` — strict players reject the dehyphenated form.
///
/// `samplerate` field at the AudioSampleEntry level is set to
/// `info.sample_rate << 16`. AC-3 samples are 32 / 44.1 / 48 kHz.
///
/// `channelcount` carries the actual decoded output channel count
/// (acmod-derived) — informational; players use the dac3 body for the
/// authoritative channel layout.
pub(super) fn build_ac3_sample_entry(info: &AudioInfo) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"ac-3");
    // SampleEntry header
    for _ in 0..6 {
        b.u8(0);
    } // reserved[6] = 0
    b.u16(1); // data_reference_index
    // AudioSampleEntry v0 body
    b.u32(0); // reserved
    b.u32(0); // reserved
    b.u16(info.channels); // channel_count (informational)
    b.u16(16); // sample_size (bits) — informational
    b.u16(0); // pre_defined
    b.u16(0); // reserved
    b.u32(info.sample_rate << 16); // samplerate 16.16 fixed-point
    b.extend(&build_dac3(info)); // dac3 child
    b.finish()
}

/// `ec-3` AudioSampleEntry per ETSI TS 102 366 §F.5. Mirrors `ac-3` with a
/// different 4cc and a `dec3` (rather than `dac3`) child config box.
pub(super) fn build_ec3_sample_entry(info: &AudioInfo) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"ec-3");
    for _ in 0..6 {
        b.u8(0);
    }
    b.u16(1);
    b.u32(0);
    b.u32(0);
    b.u16(info.channels);
    b.u16(16);
    b.u16(0);
    b.u16(0);
    b.u32(info.sample_rate << 16);
    b.extend(&build_dec3(info));
    b.finish()
}

/// `dac3` AC-3 Config Box per ETSI TS 102 366 §F.4. Box header is 8 bytes;
/// payload is exactly 3 bytes (24 bits packed MSB-first). Total = 11 bytes.
///
/// Bit layout (all MSB-first within the 3-byte payload):
/// ```text
///   bit  0..2   fscod          (2 bits)
///   bit  2..7   bsid           (5 bits)
///   bit  7..10  bsmod          (3 bits)
///   bit 10..13  acmod          (3 bits)
///   bit 13..14  lfeon          (1 bit)
///   bit 14..19  bit_rate_code  (5 bits)
///   bit 19..24  reserved       (5 bits, must be 0)
/// ```
///
/// The 3 payload bytes carried in `info.codec_private` are emitted verbatim
/// — the demuxer side already serialised them per the spec, so this builder
/// is a thin wrapper. The 3-byte length contract is checked by `with_audio`.
pub(super) fn build_dac3(info: &AudioInfo) -> Vec<u8> {
    debug_assert_eq!(
        info.codec_private.len(),
        3,
        "with_audio gate must enforce dac3 body == 3 bytes"
    );
    let mut b = BoxBuilder::new(b"dac3");
    b.extend(&info.codec_private);
    b.finish()
}

/// `dec3` E-AC-3 Config Box per ETSI TS 102 366 §F.6. Box header is 8 bytes;
/// payload is variable size depending on independent / dependent substream
/// count. For the single-independent-substream / no-dependent-substream
/// case (Squad-26's scope) the payload is 5 bytes:
///
/// ```text
///   bit  0..13   data_rate          (13 bits, kbps / 2)
///   bit 13..16   num_ind_sub - 1    (3 bits — 0 = 1 substream)
///   per independent substream:
///     bit 0..2    fscod            (2 bits)
///     bit 2..7    bsid             (5 bits, =16 for E-AC-3)
///     bit 7..8    reserved         (1 bit, =0)
///     bit 8..9    asvc             (1 bit)
///     bit 9..12   bsmod            (3 bits)
///     bit 12..15  acmod            (3 bits)
///     bit 15..16  lfeon            (1 bit)
///     bit 16..19  reserved         (3 bits, =0)
///     bit 19..23  num_dep_sub      (4 bits, =0 in Squad-26 scope)
///     // (if num_dep_sub > 0: chan_loc 9 bits — not emitted here)
/// ```
///
/// The body is carried in `info.codec_private` and emitted verbatim;
/// `with_audio` validates length ≥ 5. Demuxer-side construction of these
/// bytes happens in `demux::derive_dec3_from_eac3_sync`.
pub(super) fn build_dec3(info: &AudioInfo) -> Vec<u8> {
    debug_assert!(
        info.codec_private.len() >= 5,
        "with_audio gate must enforce dec3 body >= 5 bytes"
    );
    let mut b = BoxBuilder::new(b"dec3");
    b.extend(&info.codec_private);
    b.finish()
}

/// Construct the 3-byte `dac3` body from a parsed AC-3 sync header. Used
/// by the demuxer (derive from first frame) and by tests.
///
/// Bit layout per ETSI TS 102 366 §F.4 (fscod 2 | bsid 5 | bsmod 3 |
/// acmod 3 | lfeon 1 | bit_rate_code 5 | reserved 5).
pub fn dac3_body_from_sync(s: &Ac3SyncInfo) -> [u8; 3] {
    let mut bw = MsbBitWriter::new();
    bw.put(2, s.fscod as u32);
    bw.put(5, s.bsid as u32);
    bw.put(3, s.bsmod as u32);
    bw.put(3, s.acmod as u32);
    bw.put(1, if s.lfeon { 1 } else { 0 });
    bw.put(5, s.bit_rate_code as u32);
    bw.put(5, 0); // reserved
    let bytes = bw.finish();
    // Exactly 24 bits = 3 bytes (compile-time invariant of the layout).
    [bytes[0], bytes[1], bytes[2]]
}

/// Construct the 5-byte single-substream `dec3` body from a parsed E-AC-3
/// sync header. Used by the demuxer (derive from first frame) and by tests.
///
/// `data_rate` is the source-frame nominal kbps / 2 per §F.6. Compute it
/// from the source: `data_rate = ceil((frame_size_bytes * 8 * sample_rate /
/// samples_per_frame) / 2 / 1000)`. We accept it as a parameter so the
/// caller can supply either the frame-derived value or a stored/best-known
/// value; for vanilla 5.1 48 kHz E-AC-3 at 384 kbps this is 192.
pub fn dec3_body_from_sync(s: &Eac3SyncInfo, data_rate_div2_kbps: u16) -> [u8; 5] {
    let mut bw = MsbBitWriter::new();
    // Header: data_rate (13b) + num_ind_sub - 1 (3b). num_ind_sub = 1 in
    // Squad-26's scope, so the wire field is 0.
    bw.put(13, (data_rate_div2_kbps & 0x1FFF) as u32);
    bw.put(3, 0); // num_ind_sub - 1 = 0
    // Per-independent-substream block (3 bytes for the no-dep-sub case).
    bw.put(2, s.fscod as u32);
    bw.put(5, 16); // bsid pinned to 16 per §F.6
    bw.put(1, 0); // reserved
    bw.put(1, 0); // asvc — Squad-26 doesn't carry alternate-stream signalling
    bw.put(3, s.bsmod as u32);
    bw.put(3, s.acmod as u32);
    bw.put(1, if s.lfeon { 1 } else { 0 });
    bw.put(3, 0); // reserved
    bw.put(4, 0); // num_dep_sub = 0 (Squad-26 scope)
    let bytes = bw.finish();
    debug_assert_eq!(bytes.len(), 5, "dec3 single-substream body must be 5 bytes");
    [bytes[0], bytes[1], bytes[2], bytes[3], bytes[4]]
}

/// MSB-first bit writer used to pack the dac3 / dec3 bodies. Keeps layout
/// math local to the box builders so the bit boundaries stay obvious in
/// review.
struct MsbBitWriter {
    bytes: Vec<u8>,
    bit_pos: usize,
}

impl MsbBitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_pos: 0,
        }
    }
    fn put(&mut self, n: usize, v: u32) {
        debug_assert!(n <= 24);
        for i in (0..n).rev() {
            let bit = ((v >> i) & 0x01) as u8;
            if self.bit_pos.is_multiple_of(8) {
                self.bytes.push(0);
            }
            let byte_idx = self.bit_pos / 8;
            let bit_idx = 7 - (self.bit_pos % 8);
            self.bytes[byte_idx] |= bit << bit_idx;
            self.bit_pos += 1;
        }
    }
    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

/// Emit `esds` box = FullBox(v=0 f=0) + ES_Descriptor tree per 14496-1.
/// See task spec for layout — we materialise each child into a temp Vec
/// first to compute exact lengths, then wrap the parent descriptors in
/// variable-length headers via `write_descriptor_length`.
fn build_esds(info: &AudioInfo) -> Vec<u8> {
    // Innermost: DecoderSpecificInfo (tag 0x05) payload = ASC bytes verbatim.
    let asc_len = info.asc_bytes.len() as u32;
    let mut dsi = Vec::new();
    dsi.push(0x05u8);
    write_descriptor_length(&mut dsi, asc_len);
    dsi.extend_from_slice(&info.asc_bytes);

    // DecoderConfigDescriptor (tag 0x04): 13-byte fixed preamble + DSI.
    // Fields:
    //   objectTypeIndication u8 = 0x40 (MPEG-4 Audio)
    //   streamType u6 | upStream u1 | reserved u1 => (0x05 << 2) | 0x01 = 0x15
    //   bufferSizeDB u24 = 0
    //   maxBitrate u32 = 0
    //   avgBitrate u32 = 0
    let mut dcd_payload = Vec::new();
    dcd_payload.push(0x40); // AAC / MPEG-4 Audio
    dcd_payload.push((0x05 << 2) | 0x01); // AudioStream | upstream=1
    dcd_payload.extend_from_slice(&[0, 0, 0]); // bufferSizeDB
    dcd_payload.extend_from_slice(&0u32.to_be_bytes()); // maxBitrate
    dcd_payload.extend_from_slice(&0u32.to_be_bytes()); // avgBitrate
    dcd_payload.extend_from_slice(&dsi);
    let mut dcd = Vec::new();
    dcd.push(0x04);
    write_descriptor_length(&mut dcd, dcd_payload.len() as u32);
    dcd.extend_from_slice(&dcd_payload);

    // SLConfigDescriptor (tag 0x06): one byte payload = predefined=2 (MP4 reserved).
    let mut slc = Vec::new();
    slc.push(0x06);
    write_descriptor_length(&mut slc, 1);
    slc.push(0x02);

    // ES_Descriptor (tag 0x03): ES_ID u16=0 + flags u8=0 + DCD + SLC.
    let mut es_payload = Vec::new();
    es_payload.extend_from_slice(&0u16.to_be_bytes()); // ES_ID
    es_payload.push(0); // flags
    es_payload.extend_from_slice(&dcd);
    es_payload.extend_from_slice(&slc);
    let mut es = Vec::new();
    es.push(0x03);
    write_descriptor_length(&mut es, es_payload.len() as u32);
    es.extend_from_slice(&es_payload);

    // FullBox(0)
    let mut b = BoxBuilder::new(b"esds");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.extend(&es);
    b.finish()
}

/// Write a variable-length MPEG-4 descriptor length field. For len < 128
/// emits a single byte. For larger values emits a 4-byte continuation
/// sequence per ISO/IEC 14496-1 (high bit set on every byte but the last,
/// low 7 bits carry 7 bits of the length MSB-first).
///
/// Historical note: the `read_descriptor` peer in demux.rs caps at 4 bytes
/// of continuation, so we use 4 bytes consistently on the write side above
/// the 128 threshold — this keeps round-trip compatibility with our own
/// demuxer and is what ffmpeg / mp4box emit.
fn write_descriptor_length(buf: &mut Vec<u8>, len: u32) {
    if len < 128 {
        buf.push(len as u8);
        return;
    }
    buf.push(((len >> 21) & 0x7F) as u8 | 0x80);
    buf.push(((len >> 14) & 0x7F) as u8 | 0x80);
    buf.push(((len >> 7) & 0x7F) as u8 | 0x80);
    buf.push((len & 0x7F) as u8);
}

/// Audio stts: one entry per run of samples with identical durations.
/// AAC typically has uniform 1024-sample frames so this collapses to a
/// single (count, delta) entry, but we handle runs defensively — some
/// demuxed streams have a shorter tail sample.
pub(super) fn build_audio_stts(durations: &[u32]) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"stts");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    // First pass: count runs.
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &d in durations {
        if let Some(last) = runs.last_mut()
            && last.1 == d
        {
            last.0 += 1;
            continue;
        }
        runs.push((1, d));
    }
    b.u32(runs.len() as u32);
    for (count, delta) in runs {
        b.u32(count);
        b.u32(delta);
    }
    b.finish()
}
