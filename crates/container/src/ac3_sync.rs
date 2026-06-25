//! AC-3 / E-AC-3 bitstream sync-header parser.
//!
//! Pure-Rust, decoder-free. We only walk enough of the syncframe to populate
//! the `dac3` / `dec3` MP4 sample-entry config fields. No coefficient parsing.
//!
//! Refs:
//! - **AC-3**: ETSI TS 102 366 v1.4.1 (Annex E) — same wire format as ATSC
//!   A/52. Syncword = 0x0B77 (BE). bsid<=8 (no sub-stream extensions).
//! - **E-AC-3**: ETSI TS 102 366 §E.1.2 / §E.1.3 — bsid==16; the same
//!   0x0B77 syncword starts each independent / dependent substream frame.
//!
//! Squad-26 (AC-3 + E-AC-3 passthrough into MP4) — pure-Rust per task notes
//! ("Do NOT introduce a Dolby decoder").

/// Parsed AC-3 sync-header fields needed to populate the MP4 `dac3`
/// AudioSpecificConfig box per ETSI TS 102 366 §F.4 (AC3SpecificBox).
///
/// All fields come straight off the BSI (Bit Stream Information) header:
///   syncinfo (5 bytes) + bsi { bsid bsmod acmod cmixlev/surmixlev dsurmod
///   lfeon ... }.
///
/// `bit_rate_code` and `fscod` come from the syncinfo block (frmsizecod
/// upper 5 bits = bit_rate_code; fscod = 2 bits at top of syncinfo
/// after the 4-byte sync prefix).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ac3SyncInfo {
    /// fscod (2 bits): sample rate code. 0=48k, 1=44.1k, 2=32k, 3=reserved.
    pub fscod: u8,
    /// bit_rate_code (5 bits, ETSI TS 102 366 Table F.6 / Table 4.6):
    /// indexes the nominal bit-rate table 0..=18 → 32..=640 kbps.
    pub bit_rate_code: u8,
    /// bsid (5 bits): bit-stream identification. AC-3 = 8; bsid==16 marks
    /// E-AC-3 (different parser path).
    pub bsid: u8,
    /// bsmod (3 bits): bit-stream mode (CM, music, dialogue, etc.).
    pub bsmod: u8,
    /// acmod (3 bits): audio coding mode / channel layout. See ETSI Table
    /// F.4: 0 = 1+1 dual mono, 1 = 1/0 mono, 2 = 2/0 stereo, 3 = 3/0,
    /// 4 = 2/1, 5 = 3/1, 6 = 2/2, 7 = 3/2 (5.1 if lfeon=1).
    pub acmod: u8,
    /// lfeon (1 bit): low-frequency-effects channel present.
    pub lfeon: bool,
}

/// Parsed E-AC-3 sync-header fields needed to populate the MP4 `dec3`
/// AudioSpecificConfig box per ETSI TS 102 366 §F.6 (EC3SpecificBox).
///
/// Independent-substream subset only — no dependent substream fields are
/// extracted (`num_dep_sub` defaults to 0). Vanilla 5.1 E-AC-3 is the
/// dominant case in the wild and fits this profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Eac3SyncInfo {
    /// strmtyp (2 bits). 0 = independent. We only support strmtyp=0 frames
    /// (Squad-26 scope; dependent / independent-substream-w/-deps deferred).
    pub strmtyp: u8,
    /// substreamid (3 bits) — 0 for vanilla E-AC-3.
    pub substreamid: u8,
    /// frmsiz (11 bits): frame size in 16-bit words minus one (frame size
    /// in bytes = (frmsiz + 1) * 2). Squad-26 uses this only to derive
    /// data_rate for the dec3 box.
    pub frmsiz: u16,
    /// fscod (2 bits): 0=48k 1=44.1k 2=32k 3=use_fscod2 (reduced-rate).
    pub fscod: u8,
    /// fscod2 (2 bits): only valid when fscod==3. 0=24k 1=22.05k 2=16k.
    pub fscod2: u8,
    /// numblkscod (2 bits): 0..=3 → 1/2/3/6 audio blocks per frame.
    pub numblkscod: u8,
    /// acmod (3 bits): channel layout — same encoding as AC-3 (Table F.4).
    pub acmod: u8,
    /// lfeon (1 bit): LFE channel present.
    pub lfeon: bool,
    /// bsid (5 bits): always 16 for E-AC-3 (11..=16 reserved for AC-3
    /// extensions; we only emit bsid==16 in dec3).
    pub bsid: u8,
    /// dialnorm (5 bits) — informational; dec3 doesn't carry it.
    pub dialnorm: u8,
    /// bsmod (3 bits) — only present when compre==1 / dialnorm valid; we
    /// emit zero when absent (matches ffmpeg behaviour).
    pub bsmod: u8,
}

/// Parsed bitstream sync info — discriminated union returned by the
/// codec-agnostic `parse_sync_info` entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncInfo {
    Ac3(Ac3SyncInfo),
    Eac3(Eac3SyncInfo),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SyncError {
    #[error("AC-3/E-AC-3 sync: input shorter than minimum syncframe")]
    Truncated,
    #[error("AC-3/E-AC-3 sync: missing 0x0B77 syncword at offset 0")]
    MissingSyncword,
    #[error("AC-3/E-AC-3 sync: reserved fscod=3 outside E-AC-3 reduced-rate path")]
    ReservedFscod,
    #[error("AC-3/E-AC-3 sync: bsid {0} outside the AC-3 (≤10) / E-AC-3 (16) ranges supported")]
    UnsupportedBsid(u8),
}

/// Discriminate AC-3 vs E-AC-3 from the bsid field and parse the relevant
/// sync header. Both wire formats share the leading 0x0B77 sync word.
///
/// The bsid byte sits at byte offset 5 in either format:
///   syncword 0x0B77 (2) | crc1 (2) | fscod+frmsizecod (1) | bsid... ←
/// For AC-3 (bsid≤10) the upper 5 bits of byte 5 are bsid; the lower 3
/// are bsmod. For E-AC-3 (bsid=16) the byte layout is different but bsid
/// still occupies the top 5 bits of byte 5 (after a strmtyp+substreamid+
/// frmsiz reordering at bytes 2-4). The shared "bsid is the top 5 bits
/// of byte 5" property gives us the discriminator without parsing the
/// rest first.
pub fn parse_sync_info(bytes: &[u8]) -> Result<SyncInfo, SyncError> {
    if bytes.len() < 6 {
        return Err(SyncError::Truncated);
    }
    if bytes[0] != 0x0B || bytes[1] != 0x77 {
        return Err(SyncError::MissingSyncword);
    }
    // bsid lives in the top 5 bits of byte 5 in BOTH AC-3 and E-AC-3
    // wire layouts (E-AC-3 §E.1.3.1.1; AC-3 §F.5.4.2.4).
    let bsid = bytes[5] >> 3;
    if bsid <= 10 {
        Ok(SyncInfo::Ac3(parse_ac3(bytes)?))
    } else if bsid == 16 {
        Ok(SyncInfo::Eac3(parse_eac3(bytes)?))
    } else {
        Err(SyncError::UnsupportedBsid(bsid))
    }
}

/// Parse the AC-3 syncframe BSI prefix per ETSI TS 102 366 §F.5.4.2.
/// Layout (bit positions, MSB-first within each byte):
///   syncinfo (40 bits = 5 bytes):
///     syncword       16 bits = 0x0B77
///     crc1           16 bits  (skipped)
///     fscod           2 bits  → byte 4, top 2
///     frmsizecod      6 bits  → byte 4, low 6
///   bsi (variable, but the prefix is fixed):
///     bsid            5 bits  → byte 5, top 5
///     bsmod           3 bits  → byte 5, low 3
///     acmod           3 bits  → byte 6, top 3
///     [cmixlev/surmixlev 2/2 bits when acmod warrants — skipped]
///     [dsurmod        2 bits when acmod==2 — skipped]
///     lfeon           1 bit   → varies by acmod (see below)
///
/// The lfeon position depends on which optional cmix/surmix/dsurmod fields
/// are present; we walk the bit cursor through them rather than guessing.
fn parse_ac3(bytes: &[u8]) -> Result<Ac3SyncInfo, SyncError> {
    if bytes.len() < 7 {
        return Err(SyncError::Truncated);
    }
    let mut br = BitReader::new(bytes);
    br.skip(16); // syncword
    br.skip(16); // crc1
    let fscod = br.read(2) as u8;
    if fscod == 3 {
        return Err(SyncError::ReservedFscod);
    }
    let frmsizecod = br.read(6) as u8;
    let bit_rate_code = frmsizecod >> 1; // upper 5 bits index Table F.6
    let bsid = br.read(5) as u8;
    let bsmod = br.read(3) as u8;
    let acmod = br.read(3) as u8;

    // Skip the optional cmix/surmix/dsurmod fields that precede lfeon
    // per §F.5.4.2.4 (the standard documents this as a chain of `if`s).
    if (acmod & 0x01) != 0 && acmod != 0x01 {
        br.skip(2); // cmixlev (2 bits) — present when 3 front channels and not mono
    }
    if (acmod & 0x04) != 0 {
        br.skip(2); // surmixlev (2 bits) — present when surround channels
    }
    if acmod == 0x02 {
        br.skip(2); // dsurmod (2 bits) — only for stereo
    }
    let lfeon = br.read(1) == 1;

    Ok(Ac3SyncInfo {
        fscod,
        bit_rate_code,
        bsid,
        bsmod,
        acmod,
        lfeon,
    })
}

/// Parse the E-AC-3 independent-substream syncframe per ETSI TS 102 366
/// §E.1.3.1.1 (`syncinfo()` + `bsi()`).
///
/// Layout (bit positions, MSB-first):
///   syncword       16 bits = 0x0B77
///   strmtyp         2 bits
///   substreamid     3 bits
///   frmsiz         11 bits
///   fscod           2 bits
///   fscod2 / numblkscod 2 bits  (which one depends on fscod)
///   acmod           3 bits
///   lfeon           1 bit
///   bsid            5 bits  (=16 for E-AC-3)
///   dialnorm        5 bits
///   compre          1 bit
///   if compre: compr 8 bits
///   ... (rest of bsi we don't need)
fn parse_eac3(bytes: &[u8]) -> Result<Eac3SyncInfo, SyncError> {
    if bytes.len() < 8 {
        return Err(SyncError::Truncated);
    }
    let mut br = BitReader::new(bytes);
    br.skip(16); // syncword
    let strmtyp = br.read(2) as u8;
    let substreamid = br.read(3) as u8;
    let frmsiz = br.read(11) as u16;
    let fscod = br.read(2) as u8;
    let (fscod2, numblkscod) = if fscod == 3 {
        // Reduced sample-rate mode; fscod2 occupies these 2 bits and the
        // frame is implicitly 6 audio blocks (numblkscod==3 equivalent).
        (br.read(2) as u8, 3u8)
    } else {
        (0u8, br.read(2) as u8)
    };
    let acmod = br.read(3) as u8;
    let lfeon = br.read(1) == 1;
    let bsid = br.read(5) as u8;
    if bsid != 16 {
        return Err(SyncError::UnsupportedBsid(bsid));
    }
    let dialnorm = br.read(5) as u8;
    let compre = br.read(1) == 1;
    let bsmod = 0u8;
    if compre {
        // compr (8 bits) — discarded; bsmod sits a few fields later in the
        // bsi but isn't critical for dec3 (ffmpeg writes 0 unless an addbsi
        // payload describes a film/music differentiator). Leave at 0.
        br.skip(8);
        // We'd continue parsing chanmap / mixmdat / infomdat / addbsi to
        // recover bsmod from the addbsi block, but Squad-26's scope is the
        // single-substream 5.1 case. ffmpeg / x265's MP4 muxer also writes
        // bsmod=0 unless an explicit cli flag overrides — matches us.
        let _ = bsmod;
    }
    Ok(Eac3SyncInfo {
        strmtyp,
        substreamid,
        frmsiz,
        fscod,
        fscod2,
        numblkscod,
        acmod,
        lfeon,
        bsid,
        dialnorm,
        bsmod,
    })
}

/// Channel count derived from acmod + lfeon per ETSI TS 102 366 Table F.4.
/// 1+1 dual-mono (acmod==0) gets two distinct mono streams (count=2). All
/// other modes follow the conventional layout: 1.0 / 2.0 / 3.0 / 2.1 /
/// 3.1 / 2.2 / 3.2 plus an optional LFE.
pub fn channel_count(acmod: u8, lfeon: bool) -> u16 {
    let base = match acmod {
        0 => 2, // 1+1 (dual mono)
        1 => 1, // 1/0 (mono)
        2 => 2, // 2/0 (stereo)
        3 => 3, // 3/0 (L C R)
        4 => 3, // 2/1 (L R S)
        5 => 4, // 3/1 (L C R S)
        6 => 4, // 2/2 (L R Ls Rs)
        7 => 5, // 3/2 (L C R Ls Rs)
        _ => 0,
    };
    base + if lfeon { 1 } else { 0 }
}

/// Nominal bit-rate (in kbps) for an AC-3 frame given `bit_rate_code`
/// (frmsizecod >> 1) per ETSI TS 102 366 Table F.6 / ATSC A/52 Table 5.18.
/// 0..=18 are valid; everything above is reserved (returns 0).
pub fn ac3_bit_rate_kbps(bit_rate_code: u8) -> u32 {
    match bit_rate_code {
        0 => 32,
        1 => 40,
        2 => 48,
        3 => 56,
        4 => 64,
        5 => 80,
        6 => 96,
        7 => 112,
        8 => 128,
        9 => 160,
        10 => 192,
        11 => 224,
        12 => 256,
        13 => 320,
        14 => 384,
        15 => 448,
        16 => 512,
        17 => 576,
        18 => 640,
        _ => 0,
    }
}

/// Sample rate in Hz from the AC-3 fscod (Table F.5). Reserved (3) is
/// invalid for AC-3 — caller already rejected it; for E-AC-3 with fscod==3
/// the sample rate comes from `eac3_sample_rate_hz` instead.
pub fn ac3_sample_rate_hz(fscod: u8) -> u32 {
    match fscod {
        0 => 48_000,
        1 => 44_100,
        2 => 32_000,
        _ => 0,
    }
}

/// Sample rate in Hz for an E-AC-3 frame. fscod==3 selects the reduced-
/// rate table (24/22.05/16 kHz); otherwise the standard 48/44.1/32 kHz
/// table applies.
pub fn eac3_sample_rate_hz(fscod: u8, fscod2: u8) -> u32 {
    if fscod < 3 {
        return ac3_sample_rate_hz(fscod);
    }
    match fscod2 {
        0 => 24_000,
        1 => 22_050,
        2 => 16_000,
        _ => 0,
    }
}

/// Number of audio samples per E-AC-3 syncframe = numblkscod-derived
/// blocks × 256 samples/block. AC-3 syncframes are always 6 blocks ×
/// 256 = 1536 samples.
pub fn eac3_samples_per_frame(numblkscod: u8) -> u32 {
    let blocks = match numblkscod & 0x03 {
        0 => 1u32,
        1 => 2u32,
        2 => 3u32,
        _ => 6u32,
    };
    blocks * 256
}

/// MSB-first bit reader scoped to a borrowed byte slice. Used only for
/// the BSI prefix walk — no allocation, bounded read sizes (≤16 bits
/// per `read` call). Caller is responsible for not over-reading past
/// the input length; we return zero-padded bits for any bit past the
/// end (matches the H.264 `more_rbsp_data` defensive style).
struct BitReader<'a> {
    data: &'a [u8],
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, bit_pos: 0 }
    }
    fn skip(&mut self, n: usize) {
        self.bit_pos += n;
    }
    fn read(&mut self, n: usize) -> u32 {
        debug_assert!(n <= 24, "BitReader::read: cap is 24 bits per call");
        let mut value: u32 = 0;
        for _ in 0..n {
            let byte_idx = self.bit_pos / 8;
            let bit_idx = 7 - (self.bit_pos % 8);
            let bit = if byte_idx < self.data.len() {
                (self.data[byte_idx] >> bit_idx) & 0x01
            } else {
                0
            };
            value = (value << 1) | bit as u32;
            self.bit_pos += 1;
        }
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic AC-3 syncframe header: only the first ~7 bytes
    /// matter for our parser. Field order (bit-by-bit) per §F.5.4.2:
    ///   syncword 0x0B77 | crc1=0 | fscod | frmsizecod | bsid bsmod acmod
    ///   [optional 2/4 bits] | lfeon | rest...
    /// `frmsizecod` upper 5 bits encode bit_rate_code per Table F.6; lower
    /// 1 bit is the 1/2-frame indicator we don't care about. So
    /// `frmsizecod = bit_rate_code << 1`.
    fn synth_ac3_header(
        fscod: u8,
        bit_rate_code: u8,
        bsid: u8,
        bsmod: u8,
        acmod: u8,
        lfeon: bool,
    ) -> Vec<u8> {
        let mut bw = BitWriter::new();
        bw.put(16, 0x0B77); // syncword
        bw.put(16, 0); // crc1 (don't care)
        bw.put(2, fscod as u32);
        bw.put(6, (bit_rate_code as u32) << 1);
        bw.put(5, bsid as u32);
        bw.put(3, bsmod as u32);
        bw.put(3, acmod as u32);
        if (acmod & 0x01) != 0 && acmod != 0x01 {
            bw.put(2, 0); // cmixlev
        }
        if (acmod & 0x04) != 0 {
            bw.put(2, 0); // surmixlev
        }
        if acmod == 0x02 {
            bw.put(2, 0); // dsurmod
        }
        bw.put(1, if lfeon { 1 } else { 0 });
        // Pad with zeros so the buffer has the minimum length the parser
        // checks (7 bytes is enough but we go to 12 for safety).
        while bw.bytes.len() < 12 {
            bw.put(8, 0);
        }
        bw.flush()
    }

    /// Build a synthetic E-AC-3 independent syncframe header. Per §E.1.3.1.1.
    fn synth_eac3_header(
        strmtyp: u8,
        substreamid: u8,
        frmsiz: u16,
        fscod: u8,
        numblkscod: u8,
        acmod: u8,
        lfeon: bool,
    ) -> Vec<u8> {
        let mut bw = BitWriter::new();
        bw.put(16, 0x0B77);
        bw.put(2, strmtyp as u32);
        bw.put(3, substreamid as u32);
        bw.put(11, frmsiz as u32);
        bw.put(2, fscod as u32);
        bw.put(2, numblkscod as u32);
        bw.put(3, acmod as u32);
        bw.put(1, if lfeon { 1 } else { 0 });
        bw.put(5, 16); // bsid = 16 for E-AC-3
        bw.put(5, 0); // dialnorm
        bw.put(1, 0); // compre = 0
        while bw.bytes.len() < 16 {
            bw.put(8, 0);
        }
        bw.flush()
    }

    struct BitWriter {
        bytes: Vec<u8>,
        bit_pos: usize,
    }
    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                bit_pos: 0,
            }
        }
        fn put(&mut self, n: usize, v: u32) {
            // MSB-first
            for i in (0..n).rev() {
                let bit = ((v >> i) & 0x01) as u8;
                if self.bit_pos % 8 == 0 {
                    self.bytes.push(0);
                }
                let byte_idx = self.bit_pos / 8;
                let bit_idx = 7 - (self.bit_pos % 8);
                self.bytes[byte_idx] |= bit << bit_idx;
                self.bit_pos += 1;
            }
        }
        fn flush(self) -> Vec<u8> {
            self.bytes
        }
    }

    #[test]
    fn parse_ac3_5_1_384k_48k() {
        // Canonical 5.1 384 kbps 48 kHz AC-3: fscod=0, bit_rate_code=14
        // (Table F.6 row 14 = 384), bsid=8, bsmod=0, acmod=7 (3/2), lfeon=1.
        let bytes = synth_ac3_header(0, 14, 8, 0, 7, true);
        let info = parse_sync_info(&bytes).expect("must parse");
        match info {
            SyncInfo::Ac3(ac3) => {
                assert_eq!(ac3.fscod, 0, "fscod=0 → 48 kHz");
                assert_eq!(ac3.bit_rate_code, 14, "Table F.6 idx 14 = 384 kbps");
                assert_eq!(ac3.bsid, 8, "AC-3 bsid = 8");
                assert_eq!(ac3.bsmod, 0);
                assert_eq!(ac3.acmod, 7, "acmod=7 → 3/2 (5.1 with LFE)");
                assert!(ac3.lfeon);
                assert_eq!(channel_count(ac3.acmod, ac3.lfeon), 6);
                assert_eq!(ac3_bit_rate_kbps(ac3.bit_rate_code), 384);
                assert_eq!(ac3_sample_rate_hz(ac3.fscod), 48_000);
            }
            _ => panic!("expected AC-3"),
        }
    }

    #[test]
    fn parse_ac3_stereo_192k() {
        // 2.0 stereo 192 kbps 48 kHz. acmod=2, lfeon=0, bit_rate_code=10.
        let bytes = synth_ac3_header(0, 10, 8, 0, 2, false);
        let info = parse_sync_info(&bytes).expect("parse");
        match info {
            SyncInfo::Ac3(ac3) => {
                assert_eq!(ac3.acmod, 2);
                assert!(!ac3.lfeon);
                assert_eq!(channel_count(ac3.acmod, ac3.lfeon), 2);
                assert_eq!(ac3_bit_rate_kbps(ac3.bit_rate_code), 192);
            }
            _ => panic!("expected AC-3"),
        }
    }

    #[test]
    fn parse_ac3_mono_64k() {
        // 1.0 mono 64 kbps. acmod=1, lfeon=0, bit_rate_code=4.
        let bytes = synth_ac3_header(0, 4, 8, 0, 1, false);
        match parse_sync_info(&bytes).expect("parse") {
            SyncInfo::Ac3(ac3) => {
                assert_eq!(ac3.acmod, 1);
                assert_eq!(channel_count(ac3.acmod, ac3.lfeon), 1);
                assert_eq!(ac3_bit_rate_kbps(ac3.bit_rate_code), 64);
            }
            _ => panic!("AC-3 expected"),
        }
    }

    #[test]
    fn parse_eac3_5_1_independent() {
        // Vanilla 5.1 E-AC-3, indep substream 0, fscod=0 (48 kHz),
        // numblkscod=3 (6 blocks → 1536 samples/frame), acmod=7, lfeon=1.
        // frmsiz = 0x05F → frame_size_bytes = (0x05F + 1) * 2 = 192.
        let bytes = synth_eac3_header(0, 0, 0x05F, 0, 3, 7, true);
        match parse_sync_info(&bytes).expect("parse") {
            SyncInfo::Eac3(e) => {
                assert_eq!(e.strmtyp, 0);
                assert_eq!(e.substreamid, 0);
                assert_eq!(e.frmsiz, 0x05F);
                assert_eq!(e.fscod, 0);
                assert_eq!(e.numblkscod, 3);
                assert_eq!(e.acmod, 7);
                assert!(e.lfeon);
                assert_eq!(e.bsid, 16);
                assert_eq!(channel_count(e.acmod, e.lfeon), 6);
                assert_eq!(eac3_samples_per_frame(e.numblkscod), 1536);
                assert_eq!(eac3_sample_rate_hz(e.fscod, e.fscod2), 48_000);
            }
            _ => panic!("expected E-AC-3"),
        }
    }

    #[test]
    fn parse_rejects_bad_syncword() {
        let mut bytes = synth_ac3_header(0, 14, 8, 0, 7, true);
        bytes[0] = 0xAA;
        assert_eq!(parse_sync_info(&bytes), Err(SyncError::MissingSyncword));
    }

    #[test]
    fn parse_rejects_truncated() {
        let bytes = vec![0x0B, 0x77];
        assert_eq!(parse_sync_info(&bytes), Err(SyncError::Truncated));
    }

    #[test]
    fn parse_rejects_unknown_bsid() {
        // Build an AC-3-style header with bsid=12 (between 10 and 16 = reserved).
        let bytes = synth_ac3_header(0, 14, 12, 0, 7, true);
        assert_eq!(parse_sync_info(&bytes), Err(SyncError::UnsupportedBsid(12)));
    }

    #[test]
    fn channel_count_table() {
        // No-LFE
        assert_eq!(channel_count(0, false), 2); // 1+1 dual mono
        assert_eq!(channel_count(1, false), 1); // mono
        assert_eq!(channel_count(2, false), 2); // stereo
        assert_eq!(channel_count(3, false), 3); // 3/0
        assert_eq!(channel_count(4, false), 3); // 2/1
        assert_eq!(channel_count(5, false), 4); // 3/1
        assert_eq!(channel_count(6, false), 4); // 2/2
        assert_eq!(channel_count(7, false), 5); // 3/2
        // LFE
        assert_eq!(channel_count(7, true), 6); // 5.1
        assert_eq!(channel_count(2, true), 3); // 2.1
    }

    #[test]
    fn bit_rate_table_spans_zero_to_640() {
        assert_eq!(ac3_bit_rate_kbps(0), 32);
        assert_eq!(ac3_bit_rate_kbps(8), 128);
        assert_eq!(ac3_bit_rate_kbps(14), 384);
        assert_eq!(ac3_bit_rate_kbps(18), 640);
        assert_eq!(ac3_bit_rate_kbps(19), 0); // reserved
    }
}
