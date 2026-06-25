//! AudioSpecificConfig (ASC) parser for AAC family streams
//! (Squad-25, HE-AAC + multichannel AAC passthrough).
//!
//! Reference: ISO/IEC 14496-3 §1.6.2 — `AudioSpecificConfig()` syntax,
//! and §1.6.5 — backward-compatible explicit signaling for SBR / PS.
//!
//! ## Why this exists separately from `decode_asc_*`
//!
//! `demux.rs` already has tiny `decode_asc_sample_rate` /
//! `decode_asc_channels` helpers that handle the **core** ASC layer only.
//! That was sufficient for AAC-LC stereo passthrough (Squad-18) but loses
//! the SBR / PS extension layer that distinguishes:
//!   - AAC-LC (AOT=2)
//!   - HE-AAC v1 (AOT=2 core wrapped by AOT=5 SBR extension)
//!   - HE-AAC v2 (AOT=2 core wrapped by AOT=29 PS extension carrying AOT=5 SBR)
//!
//! and the **implicit** vs **explicit** signaling forms (ISO 14496-3 §1.6.5):
//! - Implicit: the ASC contains only `audioObjectType=2` (LC). The actual
//!   stream may still carry SBR/PS payloads in the AAC bitstream, but the
//!   ASC does not advertise them. **Apple Core Audio (and AVFoundation)
//!   silently downgrade implicit-signaled HE-AAC to mono 22.05 kHz core**
//!   — playback is technically correct against the LC layer but loses the
//!   stereo upmix and the high-frequency band, so listeners hear quiet,
//!   muffled audio.
//! - Explicit: the ASC starts with `audioObjectType=5` (SBR), then carries
//!   the `extensionSamplingFrequencyIndex` followed by an inner
//!   `audioObjectType=2` for the LC core. This is the form Apple players
//!   require to honour the full HE-AAC output.
//!
//! ## What this module exports
//!
//! - `parse_aac_asc` — parse a 2..16-byte ASC, return a structured
//!   `ParsedAsc { aot, sample_rate, channels, sbr_present, ps_present,
//!    sbr_sample_rate, signaling }`. Pure; no allocations beyond the
//!   returned struct.
//! - `effective_output_channels` — apply the PS upmix rule (HE-AAC v2
//!   PS: 1-channel core → 2-channel output) so the demuxer can surface
//!   the post-decoder channel count rather than the pre-decoder one.
//! - `upgrade_to_explicit_signaling` — rewrite an implicit-signaled
//!   HE-AAC ASC (`AOT=2 + sfi + chan + ...`) into the explicit form
//!   (`AOT=5 + sbr_sfi + AOT=2 + ...`) so Apple players honour the
//!   SBR / PS extension. Returns `None` for ASCs that are already
//!   explicit, ASCs that aren't HE-AAC eligible (e.g. AAC-LC > 24 kHz
//!   has no SBR upgrade because SBR doubles the rate to >48 kHz,
//!   which the AAC profile already covers natively at 32/44.1/48 kHz),
//!   or ASCs we can't safely rewrite.
//!
//! ## What this module does NOT do
//!
//! - It does not parse PCE (Programme Config Element). The synthetic
//!   `channelConfiguration=0` pathway is handled by callers that fall
//!   back to a sane default (`AudioInfo.channels=2`) — explicit PCE
//!   support is out of scope for HE-AAC + 5.1/7.1 passthrough.
//! - It does not parse the GASpecificConfig body beyond the
//!   `frameLengthFlag`/`dependsOnCoreCoder`/`extensionFlag` field
//!   prefix needed to find the SBR-extension trailer. Most fields
//!   downstream of GASpecificConfig don't affect the SBR/PS detection.

/// Sampling frequency table per ISO/IEC 14496-3 Table 1.16.
/// Index 0xF means "24-bit explicit rate follows inline".
pub const SFI_FREQS: [u32; 13] = [
    96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350,
];

/// Per ISO 14496-3 §1.6.5: how the SBR / PS extension layer was signaled
/// inside the ASC, if at all. Drives Apple-compat handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AscSignaling {
    /// Pure AAC-LC (or other non-HE AOT). No extension layer signaled.
    NoExtension,
    /// AAC-LC core with the SBR / PS layer signaled implicitly — that is,
    /// the ASC starts with `AOT=2` and **does not** advertise SBR or PS,
    /// but the audio bitstream itself contains SBR / PS payloads. Apple
    /// players silently downgrade to mono 22.05 kHz. Must be upgraded to
    /// `Explicit*` before muxing for Apple-compat or rejected.
    ImplicitMaybe,
    /// SBR / PS layer signaled explicitly via the `AOT=5` (SBR) or
    /// `AOT=29` (PS) leading byte. Apple-compatible.
    ExplicitSbr,
    /// PS extension explicitly signaled (`AOT=29` leading byte). PS implies
    /// SBR (PS rides on top of SBR per ISO 14496-3 §6).
    ExplicitPs,
}

/// Parsed AudioSpecificConfig fields.
#[derive(Debug, Clone)]
pub struct ParsedAsc {
    /// Core `audioObjectType` (the LC profile in HE-AAC bitstreams).
    /// Value range after escape decoding: 2 (AAC-LC), 5 (SBR — only if
    /// the explicit-signaling form puts SBR up front), 29 (PS — same
    /// caveat), 42 (xHE-AAC USAC), etc.
    pub aot: u8,
    /// Sampling rate of the AAC core in Hz. For HE-AAC this is the
    /// half-rate core; the SBR-extended output rate (typically 2×) is
    /// in `sbr_sample_rate` when present.
    pub sample_rate: u32,
    /// Channel configuration as advertised in the ASC's `channelConfiguration`
    /// field per ISO 14496-3 Table 1.19. Values 0..7 map directly to channel
    /// counts (1=mono, 2=stereo, 3=3.0, 4=4.0, 5=5.0, 6=5.1, 7=7.1; value 0
    /// means "consult the PCE", which we don't parse here — caller's choice
    /// to default).
    pub channels: u16,
    /// True when the ASC explicitly signals an SBR layer (AOT=5 leading
    /// byte) or has the implicit form's heuristic backbone (always false
    /// for `Implicit*` because we can only confirm SBR by parsing the
    /// AAC bitstream, which lives in samples, not the ASC).
    pub sbr_present: bool,
    /// True when the ASC explicitly signals a PS layer (AOT=29 leading
    /// byte). PS implies SBR.
    pub ps_present: bool,
    /// Output sample rate of the SBR-doubled stream when `sbr_present`.
    /// For explicit signaling this comes from `extensionSamplingFrequencyIndex`;
    /// it equals `2 × sample_rate` for the canonical HE-AAC profile.
    pub sbr_sample_rate: Option<u32>,
    /// How the SBR / PS layer was signaled. See [`AscSignaling`] for the
    /// Apple-compat consequences.
    pub signaling: AscSignaling,
}

/// Parse an AudioSpecificConfig per ISO/IEC 14496-3 §1.6.2.1.
///
/// On success returns a [`ParsedAsc`] describing the AAC family layer
/// stack. Returns `None` for malformed / truncated input or for ASCs
/// whose AOT we don't model (the caller should reject in that case).
///
/// Recognized AOTs:
/// - 2 (AAC-LC) — most common; HE-AAC uses this as the core layer.
/// - 5 (SBR) — used as the leading AOT in HE-AAC v1 explicit signaling.
/// - 29 (PS) — used as the leading AOT in HE-AAC v2 explicit signaling.
///
/// Other AOTs are surfaced as-is in `aot` with `signaling=NoExtension`
/// so the caller can decide whether to accept (e.g. xHE-AAC=42) or
/// reject.
///
/// ASC bit layout for the relevant AOTs:
///
/// ```text
/// AOT (5 bits, escape via 31+6) | SFI (4 bits, 0xF→24-bit inline) |
/// channelConfiguration (4 bits) |
///   if (AOT == 5 || AOT == 29) {
///     extensionSamplingFrequencyIndex (4 bits, 0xF→24-bit) |
///     inner_AOT (5 bits, escape) |
///     [GASpecificConfig follows for core inner AOT]
///   } else {
///     [GASpecificConfig follows]
///   }
/// ```
///
/// Per ISO 14496-3 §1.6.5 / §4.5.1.1, when explicit signaling is used:
/// - The **outer** `samplingFrequencyIndex` = the **SBR-output** rate
///   (typically 2× the AAC core rate). E.g. 48000 for an HE-AAC v1
///   stream whose core operates at 24000.
/// - `extensionSamplingFrequencyIndex` = the **SBR rate** = the same
///   value as the outer SFI per spec recommendation.
/// - The inner core AAC operates at **half** the SBR rate.
pub fn parse_aac_asc(asc: &[u8]) -> Option<ParsedAsc> {
    // ASC needs at least 2 bytes to carry AOT (5b) + SFI (4b) + channelConfig (4b).
    if asc.len() < 2 {
        return None;
    }
    let mut br = BitReader::new(asc);
    let leading_aot = read_aot(&mut br)?;
    let leading_sfi = br.bits(4)? as usize;
    let leading_sample_rate = decode_sfi(leading_sfi, &mut br)?;
    let leading_chan_cfg = br.bits(4)? as u16;

    // Explicit-signaling form: the leading AOT is SBR (5) or PS (29). The
    // *outer* SFI is the SBR-output (extended) rate per ISO 14496-3 §1.6.5.
    // Next we read `extensionSamplingFrequencyIndex` (typically the same
    // value, expressed redundantly per spec) and then the inner core AOT
    // (typically AOT=2 for LC).
    if leading_aot == 5 || leading_aot == 29 {
        let ext_sfi = br.bits(4)? as usize;
        let _sbr_rate_redundant = decode_sfi(ext_sfi, &mut br)?;
        let core_aot = read_aot(&mut br)?;
        // Outer SFI is the SBR/output rate; core operates at half that.
        let sbr_output_rate = leading_sample_rate;
        let core_rate = sbr_output_rate / 2;
        return Some(ParsedAsc {
            aot: core_aot,
            sample_rate: core_rate,
            channels: leading_chan_cfg,
            sbr_present: true,
            ps_present: leading_aot == 29,
            sbr_sample_rate: Some(sbr_output_rate),
            signaling: if leading_aot == 29 {
                AscSignaling::ExplicitPs
            } else {
                AscSignaling::ExplicitSbr
            },
        });
    }

    // Plain / core AOT path. AAC-LC (2) is the common case; xHE-AAC (42),
    // ER-AAC-LC (17) etc. surface here too. We don't try to chase the
    // GASpecificConfig tail to find a back-door SBR signal — that's the
    // implicit form, and we mark it `ImplicitMaybe` ONLY for the AOT=2 +
    // ≤24 kHz core combination that's the canonical HE-AAC implicit shape.
    let signaling = if leading_aot == 2 && leading_sample_rate <= 24_000 {
        AscSignaling::ImplicitMaybe
    } else {
        AscSignaling::NoExtension
    };

    Some(ParsedAsc {
        aot: leading_aot,
        sample_rate: leading_sample_rate,
        channels: leading_chan_cfg,
        sbr_present: false,
        ps_present: false,
        sbr_sample_rate: None,
        signaling,
    })
}

/// Effective decoded-output channel count for an HE-AAC family stream.
/// Per ISO/IEC 14496-3 §6 ("Parametric Stereo"), a 1-channel core wrapped
/// in PS upmixes to 2-channel output. SBR alone does not change the
/// channel count.
///
/// This is the value the demuxer surfaces in `AudioInfo.channels`, NOT
/// the raw `channelConfiguration` from the ASC. Players honour the
/// effective count for buffer allocation and downstream renderers.
pub fn effective_output_channels(parsed: &ParsedAsc) -> u16 {
    let raw = parsed.channels;
    // PS upmix: mono LC + PS → stereo output.
    if parsed.ps_present && raw == 1 {
        return 2;
    }
    // channelConfiguration=0 is "consult PCE" — we default to 2 (matches
    // the existing demux fallback) so callers never see 0.
    if raw == 0 { 2 } else { raw }
}

/// Rewrite an implicit-signaled HE-AAC ASC into explicit form. Returns
/// `None` when no rewrite is needed or possible:
/// - The ASC is already explicit (`AOT=5` or `AOT=29` leading byte).
/// - The ASC isn't HE-AAC-eligible (core sample rate > 24 kHz, since SBR
///   would push it past the canonical 48 kHz output rate).
/// - The ASC is malformed.
///
/// On success returns the explicit-form ASC bytes. Layout produced per
/// ISO 14496-3 §1.6.2.1 / §1.6.5:
///
///   `outerAOT=5 (5b) | outerSFI=SBR_rate (4b) | channelConfiguration (4b) |
///    extensionSamplingFrequencyIndex=SBR_rate (4b) | innerAOT=2 (5b) |
///    GASpecificConfig tail bits...`
///
/// Where `SBR_rate = 2 × original_core_rate` (per the canonical HE-AAC
/// rate doubling). If `2×core` matches a known SFI we use the index; else
/// we emit the 24-bit explicit rate (`sfi=0xF` + `samplingFrequency u24`).
///
/// The original ASC's `channelConfiguration` and post-channelConfiguration
/// `GASpecificConfig` tail bits are copied verbatim (after re-shifting to
/// the new positions in the bit stream).
pub fn upgrade_to_explicit_signaling(asc: &[u8]) -> Option<Vec<u8>> {
    let parsed = parse_aac_asc(asc)?;
    if parsed.signaling != AscSignaling::ImplicitMaybe {
        return None;
    }
    if parsed.aot != 2 {
        return None;
    }
    // SBR doubles the sample rate; for AAC-LC ≤24 kHz this lands at ≤48 kHz,
    // which is the supported HE-AAC profile range.
    if parsed.sample_rate > 24_000 {
        return None;
    }
    let sbr_rate = parsed.sample_rate * 2;
    let sbr_sfi = sfi_for_rate(sbr_rate);

    // We need the original ASC bit layout as raw bits so we can copy the
    // post-channelConfiguration GASpecificConfig tail verbatim. Build a bit
    // reader, skip the 5+4(+24)+4 prefix, and read the rest as a tail.
    let mut br = BitReader::new(asc);
    br.bits(5)?; // AOT=2
    let leading_sfi = br.bits(4)? as usize;
    if leading_sfi == 0xF {
        br.bits(24)?; // skip inline 24-bit core rate (we re-derive)
    }
    br.bits(4)?; // channelConfiguration (we already have it in `parsed.channels`)

    // Drain remaining bits into a tail-bit-buffer (the GASpecificConfig).
    let tail_bits: Vec<u8> = drain_remaining_bits(&mut br);

    // Build the explicit-form bitstream.
    let mut bw = BitWriter::new();
    bw.bits(5, 5); // outer AOT=5 (SBR)
    match sbr_sfi {
        Some(idx) => bw.bits(idx, 4), // outer SFI = SBR rate
        None => {
            bw.bits(0xF, 4); // 0xF → 24-bit inline
            bw.bits(sbr_rate, 24);
        }
    }
    bw.bits(parsed.channels as u32, 4); // channelConfiguration
    // extensionSamplingFrequencyIndex = same SBR rate (per spec recommendation).
    match sbr_sfi {
        Some(idx) => bw.bits(idx, 4),
        None => {
            bw.bits(0xF, 4);
            bw.bits(sbr_rate, 24);
        }
    }
    bw.bits(2, 5); // inner AOT=2 (LC core)
    for bit in &tail_bits {
        bw.bits(*bit as u32, 1);
    }
    Some(bw.into_bytes())
}

/// Reverse-lookup an SFI table index for the given Hz rate. Returns `None`
/// when no canonical index matches (caller falls back to the 0xF inline form).
fn sfi_for_rate(rate: u32) -> Option<u32> {
    SFI_FREQS.iter().position(|r| *r == rate).map(|p| p as u32)
}

/// Read the (possibly extended) audioObjectType field per ISO 14496-3 §1.6.2.1:
///   if audioObjectType == 31:
///       audioObjectType = 32 + audioObjectTypeExt (6 bits)
fn read_aot(br: &mut BitReader<'_>) -> Option<u8> {
    let raw = br.bits(5)? as u8;
    if raw == 31 {
        let ext = br.bits(6)? as u8;
        Some(32 + ext)
    } else {
        Some(raw)
    }
}

/// Decode a samplingFrequencyIndex into the corresponding rate. Index 0xF
/// triggers a 24-bit inline rate field per ISO 14496-3 Table 1.16.
fn decode_sfi(sfi: usize, br: &mut BitReader<'_>) -> Option<u32> {
    if sfi == 0xF {
        let r = br.bits(24)? as u32;
        if r == 0 { None } else { Some(r) }
    } else {
        SFI_FREQS.get(sfi).copied()
    }
}

/// Read remaining bits from the bit-reader as a flat Vec<u8> of {0,1}.
fn drain_remaining_bits(br: &mut BitReader<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    while let Some(b) = br.bits(1) {
        out.push(b as u8);
    }
    out
}

/// MSB-first bit reader over a byte slice. Mirrors the AscBitReader in
/// `demux.rs` but stays in this module so we don't tangle the parser
/// with the rest of the demux state.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn bits(&mut self, n: u32) -> Option<u64> {
        let mut v: u64 = 0;
        for _ in 0..n {
            let byte = *self.data.get(self.pos / 8)?;
            let bit = (byte >> (7 - (self.pos % 8))) & 1;
            v = (v << 1) | bit as u64;
            self.pos += 1;
        }
        Some(v)
    }
}

/// MSB-first bit writer, byte-padded with zeros at the end.
struct BitWriter {
    buf: Vec<u8>,
    bit_pos: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            bit_pos: 0,
        }
    }

    fn bits(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            let bit = ((value >> i) & 1) as u8;
            if self.bit_pos.is_multiple_of(8) {
                self.buf.push(0);
            }
            let byte_idx = (self.bit_pos / 8) as usize;
            let bit_offset = 7 - (self.bit_pos % 8) as u8;
            self.buf[byte_idx] |= bit << bit_offset;
            self.bit_pos += 1;
        }
    }

    fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AAC-LC stereo @ 48 kHz: AOT=2, SFI=3 (48000), chan=2.
    /// 00010 0011 0010 000 = 0001 0001 1001 0000 = 0x11 0x90.
    #[test]
    fn parse_aac_lc_stereo_48k() {
        let asc = vec![0x11, 0x90];
        let p = parse_aac_asc(&asc).expect("parse should succeed");
        assert_eq!(p.aot, 2, "AOT");
        assert_eq!(p.sample_rate, 48_000, "sample_rate");
        assert_eq!(p.channels, 2, "channels");
        assert!(!p.sbr_present);
        assert!(!p.ps_present);
        assert_eq!(p.signaling, AscSignaling::NoExtension);
        assert_eq!(effective_output_channels(&p), 2);
    }

    /// AAC-LC stereo @ 44.1 kHz: AOT=2, SFI=4, chan=2.
    /// 00010 0100 0010 000 = 0001 0010 0001 0000 = 0x12 0x10.
    #[test]
    fn parse_aac_lc_stereo_44_1k() {
        let asc = vec![0x12, 0x10];
        let p = parse_aac_asc(&asc).expect("parse should succeed");
        assert_eq!(p.aot, 2);
        assert_eq!(p.sample_rate, 44_100);
        assert_eq!(p.channels, 2);
        assert_eq!(p.signaling, AscSignaling::NoExtension);
    }

    /// HE-AAC v1 5.1 explicit signaling at 48 kHz output (24 kHz LC core).
    /// Per ISO 14496-3 §1.6.5 the *outer* SFI = SBR-output rate (48000),
    /// `extensionSamplingFrequencyIndex` = same value, and the inner core
    /// AAC operates at half (24000).
    ///   AOT=5 (00101)
    ///   outer SFI = 3 → 48000 (0011)
    ///   channelConfiguration = 6 → 5.1 (0110)
    ///   extensionSamplingFrequencyIndex = 3 → 48000 (0011)
    ///   inner AOT = 2 → LC (00010)
    /// Bits: 00101 0011 0110 0011 00010 = 22 bits.
    /// 00101001 10110001 10001000 = 0x29 0xB1 0x88.
    #[test]
    fn parse_he_aac_v1_5_1_explicit() {
        let asc = vec![0x29, 0xB1, 0x88];
        let p = parse_aac_asc(&asc).expect("parse should succeed");
        // The reported `aot` is the inner core (LC=2). The leading AOT=5
        // disappears into `signaling=ExplicitSbr` + `sbr_present`.
        assert_eq!(p.aot, 2, "core aot should be LC");
        assert_eq!(p.sample_rate, 24_000, "LC core rate (half of SBR)");
        assert_eq!(p.channels, 6, "5.1 channel config");
        assert!(p.sbr_present, "SBR layer must be present");
        assert!(!p.ps_present, "PS not present in v1");
        assert_eq!(p.sbr_sample_rate, Some(48_000), "SBR output rate");
        assert_eq!(p.signaling, AscSignaling::ExplicitSbr);
        assert_eq!(effective_output_channels(&p), 6);
    }

    /// HE-AAC v2 mono PS explicit signaling at 44.1 kHz output (22.05 kHz LC core):
    ///   AOT=29 (11101)
    ///   outer SFI = 4 → 44100 (0100)  ← SBR-output rate
    ///   channelConfiguration = 1 → mono (0001)
    ///   extensionSamplingFrequencyIndex = 4 → 44100 (0100)
    ///   inner AOT = 2 → LC (00010)
    /// Bits: 11101 0100 0001 0100 00010 = 22 bits.
    /// 11101010 00001010 00001000 = 0xEA 0x0A 0x08.
    #[test]
    fn parse_he_aac_v2_mono_ps_explicit() {
        let asc = vec![0xEA, 0x0A, 0x08];
        let p = parse_aac_asc(&asc).expect("parse should succeed");
        assert_eq!(p.aot, 2, "core aot is LC");
        assert_eq!(p.sample_rate, 22_050, "core rate (half of SBR)");
        assert_eq!(p.channels, 1, "mono channel config (PS upmixes at decode)");
        assert!(p.sbr_present, "PS implies SBR");
        assert!(p.ps_present, "PS layer signaled");
        assert_eq!(p.sbr_sample_rate, Some(44_100));
        assert_eq!(p.signaling, AscSignaling::ExplicitPs);
        // PS upmix: 1-channel core → 2-channel effective output.
        assert_eq!(effective_output_channels(&p), 2);
    }

    /// HE-AAC implicit signaling: a plain AAC-LC ASC at low core rate
    /// (≤24 kHz) is the canonical implicit-HE shape. parse_aac_asc must
    /// flag `ImplicitMaybe` so the caller can decide to upgrade or reject.
    ///   AOT=2, SFI=6 (24000), chan=1 → 00010 0110 0001 000
    ///   = 0001 0011 0000 1000 = 0x13 0x08.
    #[test]
    fn parse_implicit_he_aac_flagged() {
        let asc = vec![0x13, 0x08];
        let p = parse_aac_asc(&asc).expect("parse");
        assert_eq!(p.aot, 2);
        assert_eq!(p.sample_rate, 24_000);
        assert_eq!(
            p.signaling,
            AscSignaling::ImplicitMaybe,
            "low-rate AAC-LC must be flagged as implicit-HE candidate"
        );
    }

    /// upgrade_to_explicit_signaling rewrites a 24 kHz mono AAC-LC ASC into
    /// the explicit HE-AAC v1 form: leading AOT=5, outer SFI=SBR rate (48 kHz),
    /// then channelConfig + extSFI + inner AOT=2.
    #[test]
    fn upgrade_24k_mono_lc_to_explicit_he_aac_v1() {
        let asc = vec![0x13, 0x08]; // AOT=2 SFI=6 (24000) chan=1
        let upgraded =
            upgrade_to_explicit_signaling(&asc).expect("upgrade should succeed for ≤24 kHz LC");
        let reparsed = parse_aac_asc(&upgraded).expect("upgraded ASC parses");
        assert_eq!(
            reparsed.signaling,
            AscSignaling::ExplicitSbr,
            "upgraded ASC must be explicit-SBR"
        );
        // After upgrade: outer SFI = 48000 (SBR), inner core = 24000 (half).
        assert_eq!(reparsed.sample_rate, 24_000, "core rate is half of SBR");
        assert_eq!(
            reparsed.sbr_sample_rate,
            Some(48_000),
            "SBR rate is 2× core"
        );
        assert_eq!(reparsed.channels, 1);
    }

    /// upgrade_to_explicit_signaling refuses to upgrade ASCs whose core
    /// rate is too high to plausibly be HE-AAC (>24 kHz core would push
    /// SBR output > 48 kHz, off-spec for the canonical HE profile).
    #[test]
    fn upgrade_rejects_high_rate_lc() {
        let asc = vec![0x11, 0x90]; // AOT=2 SFI=3 (48000) chan=2
        // 48 kHz core → 96 kHz SBR. Off-spec; refuse the upgrade.
        // (Actually our gate is "core rate <= 24 kHz" so 48 kHz is rejected
        // because signaling is NoExtension, not because of the rate gate;
        // either way the function must return None.)
        assert!(upgrade_to_explicit_signaling(&asc).is_none());
    }

    /// upgrade_to_explicit_signaling refuses to re-upgrade an already-explicit ASC.
    #[test]
    fn upgrade_rejects_already_explicit() {
        let asc = vec![0x29, 0x89, 0x98]; // HE-AAC v1 5.1 explicit
        assert!(upgrade_to_explicit_signaling(&asc).is_none());
    }

    /// AOT escape: AOT=31 → 6-bit extension. AOT=42 (xHE-AAC USAC) is the
    /// canonical case. Bit pattern: 11111 (raw=31) | 001010 (ext=10) → 32+10=42.
    /// Then SFI=3 (0011) chan=2 (0010), pad. 17 bits.
    /// 11111 001010 0011 0010 0 = 1111 1001 0100 0110 0100 = 0xF9 0x46 0x40.
    #[test]
    fn parse_xhe_aac_aot_escape() {
        let asc = vec![0xF9, 0x46, 0x40];
        let p = parse_aac_asc(&asc).expect("parse");
        assert_eq!(p.aot, 42, "xHE-AAC USAC AOT=42 via escape");
        assert_eq!(p.sample_rate, 48_000);
        assert_eq!(p.channels, 2);
    }

    /// 24-bit inline sample rate: SFI=0xF then 24-bit rate. Use 88200 just
    /// to differ from the table values for variety.
    /// AOT=2, SFI=0xF, rate=88200 (0x015888), chan=2.
    /// Bits: 00010 1111 [88200 in 24 bits] 0010
    /// 88200 = 0000 0000 0000 0001 0101 1000 1000 1000 → low 24 bits
    /// = 0000 0001 0101 1000 1000 1000.
    /// Concatenation: 00010 1111 000000010101100010001000 0010
    /// = 0001 0111 1000 0000 1010 1100 0100 0100 0010 = 5 bytes
    /// 17807ac442 first 4 bytes... let's just compute.
    #[test]
    fn parse_inline_24bit_sample_rate() {
        // Construct via BitWriter to avoid hand-bit-fiddling errors.
        let mut bw = BitWriter::new();
        bw.bits(2, 5); // AOT=2
        bw.bits(0xF, 4); // SFI=0xF
        bw.bits(88_200, 24); // inline rate
        bw.bits(2, 4); // chan=2
        let asc = bw.into_bytes();
        let p = parse_aac_asc(&asc).expect("parse");
        assert_eq!(p.aot, 2);
        assert_eq!(p.sample_rate, 88_200);
        assert_eq!(p.channels, 2);
    }

    /// Bit writer round-trip smoke test.
    #[test]
    fn bit_writer_roundtrip() {
        let mut bw = BitWriter::new();
        bw.bits(0b10101, 5);
        bw.bits(0b1100, 4);
        let bytes = bw.into_bytes();
        // 10101 1100 0 = 0xAE 0x00; total written = 9 bits → padded to 2 bytes.
        // Wait: 10101 1100 = 9 bits. Layout: 1010 1110 0000 0000 = 0xAE 0x00.
        assert_eq!(bytes, vec![0xAE, 0x00]);
        let mut br = BitReader::new(&bytes);
        assert_eq!(br.bits(5), Some(0b10101));
        assert_eq!(br.bits(4), Some(0b1100));
    }

    /// 5.1 ASC with channelConfiguration=6 (the standard MPEG L/R/C/LFE/Ls/Rs ordering).
    /// Plain AAC-LC core at 48 kHz: AOT=2 SFI=3 chan=6.
    /// 00010 0011 0110 000 = 0001 0001 1011 0000 = 0x11 0xB0.
    #[test]
    fn parse_aac_lc_5_1_at_48k() {
        let asc = vec![0x11, 0xB0];
        let p = parse_aac_asc(&asc).expect("parse");
        assert_eq!(p.aot, 2);
        assert_eq!(p.sample_rate, 48_000);
        assert_eq!(p.channels, 6, "5.1 channel config");
        assert_eq!(
            p.signaling,
            AscSignaling::NoExtension,
            "48 kHz LC has no implicit HE-AAC interpretation"
        );
        assert_eq!(effective_output_channels(&p), 6);
    }

    /// 7.1 ASC with channelConfiguration=7. Plain AAC-LC at 48 kHz.
    /// 00010 0011 0111 000 = 0001 0001 1011 1000 = 0x11 0xB8.
    #[test]
    fn parse_aac_lc_7_1_at_48k() {
        let asc = vec![0x11, 0xB8];
        let p = parse_aac_asc(&asc).expect("parse");
        assert_eq!(p.channels, 7, "7.1 channel config");
        assert_eq!(effective_output_channels(&p), 7);
    }

    /// Empty / truncated input must return None.
    #[test]
    fn parse_rejects_truncated() {
        assert!(parse_aac_asc(&[]).is_none());
        assert!(parse_aac_asc(&[0x12]).is_none());
    }
}
