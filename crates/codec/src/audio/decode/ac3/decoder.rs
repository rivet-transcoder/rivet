//! Syncframe decoding: bit stream syntax (A/52:2018 §5.3 and Annex E §2.2),
//! exponent decoding (§7.1.3), mantissa dequantisation (§7.3 and Annex E
//! §3.4.4), channel coupling (§7.4), spectral extension (Annex E §3.6),
//! rematrixing (§7.5), dynamic range compression (§7.7.1) and the transform
//! (`imdct`). One `FrameDecoder` holds the only state that outlives a
//! syncframe: the overlap-add delay lines and the dither generator.
//!
//! Scope: AC-3 (bsid ≤ 8) in full; E-AC-3 (bsid 16) independent substream 0
//! with standard coupling, AHT (VQ + GAQ) and spectral extension. Enhanced
//! coupling (`ecplinu`), dependent substreams and additional independent
//! substreams are refused or skipped by name below.

use super::bitalloc::{BaParams, DeltaBa, Kind, compute_bap, fast_gain, snr_offset};
use super::bits::BitReader;
use super::imdct::imdct_block;
use super::tables::{
    ASYM_MANT_BITS, BITRATE_KBPS, DEFCPLBNDSTRC, DEFSPXBNDSTRC, FRMEXPSTR, FRMSIZETAB,
    GAQ_REMAP_A, GAQ_REMAP_B, HEBAP_MANT_BITS, SPXATTENTAB, SPXBANDTABLE, SYM_QUANT_3,
    SYM_QUANT_5, SYM_QUANT_7, SYM_QUANT_11, SYM_QUANT_15, vq_table,
};
use crate::audio::AudioError;

/// Transform coefficients per block.
const NB: usize = 256;
/// Most full-bandwidth channels (acmod 7 = 3/2).
pub(super) const MAX_FBW: usize = 5;
const CPL: usize = MAX_FBW;
const LFE: usize = MAX_FBW + 1;
const NCH: usize = MAX_FBW + 2;
const REUSE: u8 = 0;

/// Everything from `syncinfo` + `bsi` (+ the E-AC-3 `audfrm` header fields
/// the caller cares about). Parsed by [`parse_header`] before any block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// `bsid == 16` (Annex E syntax).
    pub eac3: bool,
    /// E-AC-3 `strmtyp`: 0 independent, 1 dependent, 2 AC-3-converted. Always 0 for AC-3.
    pub strmtyp: u8,
    /// E-AC-3 `substreamid`; 0 for AC-3.
    pub substreamid: u8,
    /// Syncframe length in bytes (Table 5.18, or `(frmsiz + 1) * 2`).
    pub frame_len: usize,
    pub fscod: u8,
    pub sample_rate: u32,
    /// Audio blocks in this syncframe (6 for AC-3; 1/2/3/6 for E-AC-3).
    pub numblks: usize,
    pub acmod: u8,
    pub lfeon: bool,
    /// Full-bandwidth channels, from `acmod` (Table 5.8).
    pub nfchans: usize,
    pub bsid: u8,
    pub bsmod: u8,
    pub dialnorm: u8,
    /// Nominal bit rate in kbit/s (AC-3: Table 5.18; E-AC-3: derived from the frame length).
    pub bitrate_kbps: u32,
}

impl Header {
    /// Output channels (fbw + LFE). Dual mono counts as 2.
    pub fn channels(&self) -> usize {
        self.nfchans + usize::from(self.lfeon)
    }
    /// PCM samples per channel this syncframe produces.
    pub fn samples(&self) -> usize {
        self.numblks * NB
    }
}

/// Table 5.8: full-bandwidth channels per `acmod`.
pub(super) fn nfchans_for(acmod: u8) -> usize {
    match acmod {
        0 => 2,
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 3,
        5 => 4,
        6 => 4,
        7 => 5,
        _ => 0,
    }
}

fn err(msg: impl Into<String>) -> AudioError {
    AudioError::Decode(msg.into())
}

/// Parse `syncinfo()` and the fixed-position fields of `bsi()` — enough to
/// size the frame and know its layout. The full `bsi` walk happens in
/// [`FrameDecoder::decode`].
pub fn parse_header(data: &[u8]) -> Result<Header, AudioError> {
    if data.len() < 8 {
        return Err(err("ac3: frame shorter than the 8-byte sync header"));
    }
    if data[0] != 0x0b || data[1] != 0x77 {
        return Err(err(format!("ac3: missing 0x0B77 syncword (got {:02x}{:02x})", data[0], data[1])));
    }
    let bsid = data[5] >> 3;
    if bsid <= 8 {
        let fscod = data[4] >> 6;
        let frmsizecod = data[4] & 0x3f;
        if fscod == 3 {
            return Err(err("ac3: reserved fscod 3"));
        }
        if frmsizecod as usize >= FRMSIZETAB.len() {
            return Err(err(format!("ac3: frmsizecod {frmsizecod} out of range")));
        }
        let mut br = BitReader::new(data);
        br.skip(40)?;
        let bsid = br.read(5)? as u8;
        let bsmod = br.read(3)? as u8;
        let acmod = br.read(3)? as u8;
        if (acmod & 1) != 0 && acmod != 1 {
            br.read(2)?; // cmixlev
        }
        if (acmod & 4) != 0 {
            br.read(2)?; // surmixlev
        }
        if acmod == 2 {
            br.read(2)?; // dsurmod
        }
        let lfeon = br.read_bit()?;
        let dialnorm = br.read(5)? as u8;
        Ok(Header {
            eac3: false,
            strmtyp: 0,
            substreamid: 0,
            frame_len: usize::from(FRMSIZETAB[frmsizecod as usize][fscod as usize]) * 2,
            fscod,
            sample_rate: sample_rate_for(fscod, 0),
            numblks: 6,
            acmod,
            lfeon,
            nfchans: nfchans_for(acmod),
            bsid,
            bsmod,
            dialnorm,
            bitrate_kbps: u32::from(BITRATE_KBPS[(frmsizecod >> 1) as usize]),
        })
    } else if bsid == 16 {
        let mut br = BitReader::new(data);
        br.skip(16)?;
        let strmtyp = br.read(2)? as u8;
        let substreamid = br.read(3)? as u8;
        let frmsiz = br.read(11)? as usize;
        let fscod = br.read(2)? as u8;
        let (fscod2, numblkscod) = if fscod == 3 { (br.read(2)? as u8, 3u8) } else { (0, br.read(2)? as u8) };
        if fscod == 3 && fscod2 == 3 {
            return Err(err("eac3: reserved fscod2 3"));
        }
        let acmod = br.read(3)? as u8;
        let lfeon = br.read_bit()?;
        let _bsid = br.read(5)?;
        let dialnorm = br.read(5)? as u8;
        let numblks = [1usize, 2, 3, 6][numblkscod as usize];
        let frame_len = (frmsiz + 1) * 2;
        let sample_rate = sample_rate_for(fscod, fscod2);
        let samples = numblks * NB;
        let bitrate_kbps = ((frame_len as u64 * 8 * u64::from(sample_rate)) / (samples as u64 * 1000)) as u32;
        Ok(Header {
            eac3: true,
            strmtyp,
            substreamid,
            frame_len,
            fscod,
            sample_rate,
            numblks,
            acmod,
            lfeon,
            nfchans: nfchans_for(acmod),
            bsid: 16,
            bsmod: 0,
            dialnorm,
            bitrate_kbps,
        })
    } else {
        Err(AudioError::Unsupported(format!(
            "ac3: bsid {bsid} — only AC-3 (bsid ≤ 8) and E-AC-3 (bsid 16) are decoded; bsid 9/10 (Annex D reduced sample rate) is not"
        )))
    }
}

fn sample_rate_for(fscod: u8, fscod2: u8) -> u32 {
    match fscod {
        0 => 48_000,
        1 => 44_100,
        2 => 32_000,
        _ => match fscod2 {
            0 => 24_000,
            1 => 22_050,
            _ => 16_000,
        },
    }
}

/// Per-channel decode state; one instance per fbw channel, one for the
/// coupling channel (index [`CPL`]) and one for the LFE ([`LFE`]).
#[derive(Clone)]
struct Chan {
    blksw: bool,
    dith: bool,
    incpl: bool,
    inspx: bool,
    expstr: u8,
    bwcod: u8,
    endmant: usize,
    exps: [u8; NB],
    fsnroffst: u8,
    fgaincod: u8,
    deltbae: u8,
    delta: DeltaBa,
    bap: [u8; NB],
    // coupling coordinates (fbw channels)
    cplcoe: bool,
    first_cplcos: bool,
    cplco_band: [f32; 18],
    // spectral extension (fbw channels)
    first_spxcos: bool,
    spxcoe: bool,
    spxblnd: u8,
    spxco: [f32; 17],
    nblend: [f32; 17],
    sblend: [f32; 17],
    inspxatten: bool,
    spxattencod: u8,
    // AHT
    ahtinu: i8,
    aht: Option<Box<[[f32; NB]; 6]>>,
    coeffs: [f32; NB],
}

impl Default for Chan {
    fn default() -> Self {
        Self {
            blksw: false,
            dith: true,
            incpl: false,
            inspx: false,
            expstr: REUSE,
            bwcod: 0,
            endmant: 0,
            exps: [0; NB],
            fsnroffst: 0,
            fgaincod: 4,
            deltbae: 2,
            delta: DeltaBa::default(),
            bap: [0; NB],
            cplcoe: false,
            first_cplcos: true,
            cplco_band: [0.0; 18],
            first_spxcos: true,
            spxcoe: false,
            spxblnd: 0,
            spxco: [0.0; 17],
            nblend: [0.0; 17],
            sblend: [0.0; 17],
            inspxatten: false,
            spxattencod: 0,
            ahtinu: 0,
            aht: None,
            coeffs: [0.0; NB],
        }
    }
}

/// Shared state for the grouped 3-, 5- and 11-level mantissas, which are
/// packed in triples/pairs across channel boundaries within a block (§7.3.5).
#[derive(Default)]
struct Groups {
    b1: (u8, [u8; 3]),
    b2: (u8, [u8; 3]),
    b4: (u8, [u8; 2]),
}

/// A syncframe decoder. Keep one per stream: it carries the overlap-add
/// history across frames.
pub struct FrameDecoder {
    delay: Vec<[f32; NB]>,
    rng: u32,
    drc_scale: f32,
    chans: Vec<Chan>,
    last: Option<Header>,
    dithered_bins: u64,
    total_bins: u64,
    drc_blocks: u64,
    blocks: u64,
    noise_fill: bool,
    dith_offsets: [Option<usize>; 6],
    feat: Features,
}

/// Coding tools a stream has exercised so far, counted by
/// [`FrameDecoder::features`]. The cross-check report prints these so a green
/// vector cannot pass vacuously — a stream that never couples says nothing
/// about coupling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Features {
    /// AC-3 (bsid ≤ 8) syncframes decoded.
    pub ac3_frames: u64,
    /// E-AC-3 (bsid 16) syncframes decoded.
    pub eac3_frames: u64,
    /// E-AC-3 syncframes with fewer than six blocks (`numblkscod` < 3).
    pub short_frames: u64,
    /// E-AC-3 syncframes at a reduced sample rate (`fscod` = 3).
    pub reduced_rate_frames: u64,
    /// E-AC-3 syncframes taking exponent strategies from Table E2.10 (`expstre` = 0).
    pub frmexpstr_frames: u64,
    /// Channel-blocks decoded with the 256-sample transform (`blksw` = 1).
    pub blksw_chblocks: u64,
    /// Blocks with channel coupling in use.
    pub cpl_blocks: u64,
    /// Coupled 2/0 blocks with phase flags in use.
    pub phsflg_blocks: u64,
    /// 2/0 blocks with at least one rematrixing band flagged.
    pub remat_blocks: u64,
    /// Blocks carrying delta bit allocation information.
    pub deltba_blocks: u64,
    /// Blocks with a non-unity `dynrng` word.
    pub dynrng_blocks: u64,
    /// Blocks with spectral extension in use.
    pub spx_blocks: u64,
    /// Channel-frames with spectral extension attenuation.
    pub spx_atten_channels: u64,
    /// Channel-frames coded with the adaptive hybrid transform.
    pub aht_channels: u64,
    /// AHT channel-frames with gain-adaptive quantisation gains (`gaqmod` ≠ 0).
    pub gaq_channels: u64,
    /// AHT bins coded by vector quantisation (hebap 1..=7).
    pub vq_bins: u64,
    /// AHT bins coded by (gain-adaptive) scalar quantisation (hebap ≥ 8).
    pub gaq_bins: u64,
    /// GAQ mantissas that arrived with the large-mantissa tag.
    pub gaq_large_mantissas: u64,
}

impl std::fmt::Display for Features {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ac3 {} eac3 {} (short {} reduced-rate {} frmexpstr {}); blksw {} cpl {} phsflg {} remat {} deltba {} dynrng {} spx {} spxatten {} aht {} gaq {} (vq bins {} gaq bins {} large {})",
            self.ac3_frames,
            self.eac3_frames,
            self.short_frames,
            self.reduced_rate_frames,
            self.frmexpstr_frames,
            self.blksw_chblocks,
            self.cpl_blocks,
            self.phsflg_blocks,
            self.remat_blocks,
            self.deltba_blocks,
            self.dynrng_blocks,
            self.spx_blocks,
            self.spx_atten_channels,
            self.aht_channels,
            self.gaq_channels,
            self.vq_bins,
            self.gaq_bins,
            self.gaq_large_mantissas
        )
    }
}

/// Per-syncframe parse state (fields that live from `audfrm` to the last block).
struct Frame {
    hdr: Header,
    // E-AC-3 audfrm
    ahte: bool,
    snroffststr: u8,
    blkswe: bool,
    dithflage: bool,
    bamode: bool,
    frmfgaincode: bool,
    dbaflde: bool,
    skipflde: bool,
    cplstre: [bool; 6],
    cplinu_blk: [bool; 6],
    cplexpstr_blk: [u8; 6],
    chexpstr_blk: [[u8; MAX_FBW]; 6],
    lfeexpstr_blk: [u8; 6],
    cplahtinu: i8,
    lfeahtinu: i8,
    frmcsnroffst: u8,
    frmfsnroffst: u8,
    // block state
    cplinu: bool,
    cplbegf: u8,
    cplendf: i32,
    ncplsubnd: usize,
    ncplbnd: usize,
    cplbndstrc: [u8; 18],
    phsflginu: bool,
    phsflg: [bool; 18],
    cplstrtmant: usize,
    cplendmant: usize,
    cplfleak: u8,
    cplsleak: u8,
    first_cplleak: bool,
    spxinu: bool,
    spxstrtf: u8,
    spxbegf: u8,
    spx_begin_subbnd: usize,
    spx_end_subbnd: usize,
    spxbndstrc: [u8; 17],
    nspxbnds: usize,
    spxbndsz: [usize; 17],
    rematflg: [bool; 4],
    nrematbd: usize,
    ba: BaParams,
    csnroffst: u8,
    dynrng: u8,
    dynrng2: u8,
}

impl FrameDecoder {
    /// `drc_scale` in 0.0..=1.0: 1.0 applies `dynrng` as the stream asks
    /// (the spec's required default), 0.0 ignores it, in between applies the
    /// gain raised to that power (partial compression, §7.7.1.2).
    pub fn new(drc_scale: f32) -> Self {
        Self {
            delay: vec![[0.0; NB]; NCH],
            rng: 0x2545_f491,
            drc_scale: drc_scale.clamp(0.0, 1.0),
            chans: vec![Chan::default(); NCH],
            last: None,
            dithered_bins: 0,
            total_bins: 0,
            drc_blocks: 0,
            blocks: 0,
            noise_fill: true,
            dith_offsets: [None; 6],
            feat: Features::default(),
        }
    }

    /// Diagnostic switch for the §7.3.4 noise fill of zero-bit mantissas and
    /// the Annex E §3.6.4.2.4 spectral-extension noise blend. A conformant
    /// decode keeps it on (the default); the cross-check test turns it off to
    /// measure exactly how much of its output is noise.
    pub fn set_noise_fill(&mut self, enabled: bool) {
        self.noise_fill = enabled;
    }

    /// Bit offset, within the syncframe last decoded, of each audio block's
    /// `dithflag` field (`None` where E-AC-3 `dithflage = 0` omits the
    /// flags). Nothing in the syntax depends on those bits — they only select
    /// the §7.3.4 reconstruction — so the cross-check tooling
    /// (`examples/ac3_strip_dither.rs`) clears them to make a stream
    /// deterministic.
    pub fn dithflag_bit_offsets(&self) -> &[Option<usize>] {
        let n = self.last.map_or(0, |h| h.numblks);
        &self.dith_offsets[..n]
    }

    /// Which coding tools the stream has exercised so far.
    pub fn features(&self) -> Features {
        self.feat
    }

    /// `(blocks with a non-unity dynrng word, blocks decoded)` so far —
    /// tells whether a stream exercised dynamic range compression at all.
    pub fn drc_stats(&self) -> (u64, u64) {
        (self.drc_blocks, self.blocks)
    }

    /// Reseed the dither / spectral-extension noise generator. Two decodes
    /// of the same stream with different seeds differ by exactly the
    /// §7.3.4 noise fill, which is how the cross-check test measures the
    /// irreducible disagreement between conformant decoders.
    pub fn set_noise_seed(&mut self, seed: u32) {
        self.rng = seed | 1;
    }

    /// `(dithered, total)` mantissa bins seen so far: how many zero-bit
    /// mantissas were noise-filled versus all mantissas decoded.
    pub fn dither_stats(&self) -> (u64, u64) {
        (self.dithered_bins, self.total_bins)
    }

    /// The header of the last syncframe decoded.
    pub fn last_header(&self) -> Option<Header> {
        self.last
    }

    /// Reset the overlap-add history (after a discontinuity).
    pub fn reset(&mut self) {
        for d in &mut self.delay {
            *d = [0.0; NB];
        }
    }

    /// xorshift32 dither source; uniform in (-1, 1).
    fn rand(&mut self) -> f32 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        (x as i32 as f32) / 2_147_483_648.0
    }

    /// Decode one complete syncframe starting at `data[0]` (the syncword).
    /// Appends `samples * channels` interleaved f32 PCM to `out` in ffmpeg's
    /// native channel order for the layout (fronts, LFE, then surrounds).
    /// Returns `Ok(None)` for E-AC-3 substreams other than independent
    /// substream 0, which are skipped per Annex E §3.8.1.
    pub fn decode(&mut self, data: &[u8], out: &mut Vec<f32>) -> Result<Option<Header>, AudioError> {
        let hdr = parse_header(data)?;
        if data.len() < hdr.frame_len {
            return Err(err(format!("ac3: frame needs {} bytes, got {}", hdr.frame_len, data.len())));
        }
        if hdr.eac3 && (hdr.strmtyp == 1 || hdr.substreamid != 0) {
            tracing::trace!(
                strmtyp = hdr.strmtyp,
                substreamid = hdr.substreamid,
                "eac3: skipping substream (only independent substream 0 is decoded)"
            );
            return Ok(None);
        }
        if let Some(prev) = self.last
            && (prev.acmod != hdr.acmod || prev.lfeon != hdr.lfeon || prev.sample_rate != hdr.sample_rate)
        {
            self.reset();
        }
        let data = &data[..hdr.frame_len];
        if hdr.eac3 {
            self.feat.eac3_frames += 1;
            self.feat.short_frames += u64::from(hdr.numblks != 6);
            self.feat.reduced_rate_frames += u64::from(hdr.fscod == 3);
        } else {
            self.feat.ac3_frames += 1;
        }
        self.dith_offsets = [None; 6];
        let mut br = BitReader::new(data);
        let mut f = Frame::new(hdr);
        for c in &mut self.chans {
            let keep_first = Chan::default();
            *c = Chan { aht: c.aht.take(), ..keep_first };
        }
        if hdr.eac3 {
            parse_eac3_bsi(&mut br, &hdr)?;
            self.parse_audfrm(&mut br, &mut f)?;
        } else {
            parse_ac3_bsi(&mut br, &hdr)?;
        }
        let nch = hdr.channels();
        let start = out.len();
        out.resize(start + hdr.samples() * nch, 0.0);
        let order = output_order(hdr.acmod, hdr.lfeon);
        let mut pcm = [0.0f32; NB];
        for blk in 0..hdr.numblks {
            self.decode_block(&mut br, &mut f, blk)?;
            for (slot, &ch) in order.iter().enumerate() {
                let blksw = if ch == LFE { false } else { self.chans[ch].blksw };
                let coeffs = self.chans[ch].coeffs;
                imdct_block(&coeffs, blksw, &mut self.delay[ch], &mut pcm);
                let base = start + blk * NB * nch;
                for (n, v) in pcm.iter().enumerate() {
                    out[base + n * nch + slot] = *v;
                }
            }
        }
        self.last = Some(hdr);
        Ok(Some(hdr))
    }

    /// Annex E §2.2.3 `audfrm()`.
    fn parse_audfrm(&mut self, br: &mut BitReader, f: &mut Frame) -> Result<(), AudioError> {
        let hdr = f.hdr;
        let nb = hdr.numblks;
        let (expstre, ahte) = if nb == 6 { (br.read_bit()?, br.read_bit()?) } else { (true, false) };
        f.ahte = ahte;
        self.feat.frmexpstr_frames += u64::from(!expstre);
        f.snroffststr = br.read(2)? as u8;
        if f.snroffststr == 3 {
            return Err(err("eac3: reserved snroffststr 3"));
        }
        let transproce = br.read_bit()?;
        f.blkswe = br.read_bit()?;
        f.dithflage = br.read_bit()?;
        f.bamode = br.read_bit()?;
        f.frmfgaincode = br.read_bit()?;
        f.dbaflde = br.read_bit()?;
        f.skipflde = br.read_bit()?;
        let spxattene = br.read_bit()?;
        // coupling strategy per block
        if hdr.acmod > 1 {
            f.cplstre[0] = true;
            f.cplinu_blk[0] = br.read_bit()?;
            for blk in 1..nb {
                f.cplstre[blk] = br.read_bit()?;
                f.cplinu_blk[blk] = if f.cplstre[blk] { br.read_bit()? } else { f.cplinu_blk[blk - 1] };
            }
        }
        // exponent strategies
        let ncplblks = f.cplinu_blk[..nb].iter().filter(|&&c| c).count();
        if expstre {
            for blk in 0..nb {
                if f.cplinu_blk[blk] {
                    f.cplexpstr_blk[blk] = br.read(2)? as u8;
                }
                for ch in 0..hdr.nfchans {
                    f.chexpstr_blk[blk][ch] = br.read(2)? as u8;
                }
            }
        } else {
            if hdr.acmod > 1 && ncplblks > 0 {
                let code = br.read(5)? as usize;
                for blk in 0..nb {
                    f.cplexpstr_blk[blk] = FRMEXPSTR[code][blk];
                }
            }
            for ch in 0..hdr.nfchans {
                let code = br.read(5)? as usize;
                for blk in 0..nb {
                    f.chexpstr_blk[blk][ch] = FRMEXPSTR[code][blk];
                }
            }
        }
        if hdr.lfeon {
            for blk in 0..nb {
                f.lfeexpstr_blk[blk] = br.read(1)? as u8;
            }
        }
        // converter exponent strategy (E-AC-3 → AC-3 converter side info; unused)
        if hdr.strmtyp == 0 {
            let convexpstre = if nb != 6 { br.read_bit()? } else { true };
            if convexpstre {
                for _ in 0..hdr.nfchans {
                    br.read(5)?;
                }
            }
        }
        // AHT
        if ahte {
            let ncplregs = (0..6)
                .filter(|&blk| f.cplstre[blk] || f.cplexpstr_blk[blk] != REUSE)
                .count();
            f.cplahtinu = if ncplblks == 6 && ncplregs == 1 { i8::from(br.read_bit()?) } else { 0 };
            for ch in 0..hdr.nfchans {
                let nchregs = (0..6).filter(|&blk| f.chexpstr_blk[blk][ch] != REUSE).count();
                self.chans[ch].ahtinu = if nchregs == 1 { i8::from(br.read_bit()?) } else { 0 };
            }
            if hdr.lfeon {
                let nlferegs = (0..6).filter(|&blk| f.lfeexpstr_blk[blk] != REUSE).count();
                f.lfeahtinu = if nlferegs == 1 { i8::from(br.read_bit()?) } else { 0 };
            }
        }
        self.chans[CPL].ahtinu = f.cplahtinu;
        self.chans[LFE].ahtinu = f.lfeahtinu;
        // frame SNR offsets
        if f.snroffststr == 0 {
            f.frmcsnroffst = br.read(6)? as u8;
            f.frmfsnroffst = br.read(4)? as u8;
        }
        // transient pre-noise processing: parsed, not applied (optional
        // post-process; libavcodec skips it too).
        if transproce {
            for _ in 0..hdr.nfchans {
                if br.read_bit()? {
                    br.read(10)?; // transprocloc
                    br.read(8)?; // transproclen
                }
            }
        }
        // spectral extension attenuation
        if spxattene {
            for ch in 0..hdr.nfchans {
                self.chans[ch].inspxatten = br.read_bit()?;
                if self.chans[ch].inspxatten {
                    self.chans[ch].spxattencod = br.read(5)? as u8;
                    self.feat.spx_atten_channels += 1;
                }
            }
        }
        // block start info
        let blkstrtinfoe = if nb != 1 { br.read_bit()? } else { false };
        if blkstrtinfoe {
            let words = (hdr.frame_len / 2) as u32;
            let nblkstrtbits = (nb - 1) * (4 + (32 - (words - 1).leading_zeros()) as usize);
            br.skip(nblkstrtbits)?;
        }
        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!("audfrm: {:?} expstre {} ahte {} snroffststr {} blkswe {} dithflage {} bamode {} frmfgaincode {} dbaflde {} skipflde {} cplstre {:?} cplinu {:?} cplexpstr {:?} chexpstr {:?} frmcsnr {} frmfsnr {} pos {}",
                hdr, expstre, ahte, f.snroffststr, f.blkswe, f.dithflage, f.bamode, f.frmfgaincode, f.dbaflde, f.skipflde,
                &f.cplstre[..nb], &f.cplinu_blk[..nb], &f.cplexpstr_blk[..nb], &f.chexpstr_blk[..nb], f.frmcsnroffst, f.frmfsnroffst, br.pos());
        }
        // syntax state initialisation
        for ch in 0..hdr.nfchans {
            self.chans[ch].first_spxcos = true;
            self.chans[ch].first_cplcos = true;
        }
        f.first_cplleak = true;
        Ok(())
    }

    /// One `audblk()` — syntax, then the block's signal processing up to the
    /// transform coefficients (`Chan::coeffs`).
    fn decode_block(&mut self, br: &mut BitReader, f: &mut Frame, blk: usize) -> Result<(), AudioError> {
        let hdr = f.hdr;
        let eac3 = hdr.eac3;
        let nf = hdr.nfchans;

        // --- block switch and dither flags ------------------------------------
        if !eac3 || f.blkswe {
            for ch in 0..nf {
                self.chans[ch].blksw = br.read_bit()?;
            }
            self.feat.blksw_chblocks += (0..nf).filter(|&ch| self.chans[ch].blksw).count() as u64;
        }
        if !eac3 || f.dithflage {
            self.dith_offsets[blk] = Some(br.pos());
            for ch in 0..nf {
                self.chans[ch].dith = br.read_bit()?;
            }
        }
        // --- dynamic range ----------------------------------------------------
        if br.read_bit()? {
            f.dynrng = br.read(8)? as u8;
        } else if blk == 0 {
            f.dynrng = 0;
        }
        if hdr.acmod == 0 {
            if br.read_bit()? {
                f.dynrng2 = br.read(8)? as u8;
            } else if blk == 0 {
                f.dynrng2 = 0;
            }
        }
        // --- spectral extension strategy (E-AC-3) -----------------------------
        if eac3 {
            let spxstre = if blk == 0 { true } else { br.read_bit()? };
            if spxstre {
                f.spxinu = br.read_bit()?;
                if f.spxinu {
                    if hdr.acmod == 1 {
                        self.chans[0].inspx = true;
                    } else {
                        for ch in 0..nf {
                            self.chans[ch].inspx = br.read_bit()?;
                        }
                    }
                    f.spxstrtf = br.read(2)? as u8;
                    f.spxbegf = br.read(3)? as u8;
                    let spxendf = br.read(3)? as usize;
                    let b = f.spxbegf as usize;
                    f.spx_begin_subbnd = if b < 6 { b + 2 } else { b * 2 - 3 };
                    f.spx_end_subbnd = if spxendf < 3 { spxendf + 5 } else { spxendf * 2 + 3 };
                    if f.spx_begin_subbnd >= f.spx_end_subbnd {
                        return Err(err("eac3: spectral extension begin sub-band ≥ end sub-band"));
                    }
                    if br.read_bit()? {
                        for bnd in f.spx_begin_subbnd + 1..f.spx_end_subbnd {
                            f.spxbndstrc[bnd] = br.read(1)? as u8;
                        }
                    }
                    f.nspxbnds = 1;
                    f.spxbndsz[0] = 12;
                    for bnd in f.spx_begin_subbnd + 1..f.spx_end_subbnd {
                        if f.spxbndstrc[bnd] == 0 {
                            f.spxbndsz[f.nspxbnds] = 12;
                            f.nspxbnds += 1;
                        } else {
                            f.spxbndsz[f.nspxbnds - 1] += 12;
                        }
                    }
                } else {
                    for ch in 0..nf {
                        self.chans[ch].inspx = false;
                        self.chans[ch].first_spxcos = true;
                    }
                }
            }
            // spectral extension coordinates
            if f.spxinu {
                for ch in 0..nf {
                    let c = &mut self.chans[ch];
                    if c.inspx {
                        c.spxcoe = if c.first_spxcos {
                            c.first_spxcos = false;
                            true
                        } else {
                            br.read_bit()?
                        };
                        if c.spxcoe {
                            c.spxblnd = br.read(5)? as u8;
                            let mstrspxco = br.read(2)? as u32;
                            for bnd in 0..f.nspxbnds {
                                let exp = br.read(4)?;
                                let mant = br.read(2)? as f32;
                                let temp = if exp == 15 { mant / 4.0 } else { (mant + 4.0) / 8.0 };
                                c.spxco[bnd] = temp / (1u64 << (exp + 3 * mstrspxco)) as f32;
                            }
                            // §3.6.4.2.1 blending factors, computed when new
                            // coordinates arrive.
                            let noffset = f32::from(c.spxblnd) / 32.0;
                            let mut spxmant = usize::from(SPXBANDTABLE[f.spx_begin_subbnd]);
                            let end = f32::from(SPXBANDTABLE[f.spx_end_subbnd]);
                            for bnd in 0..f.nspxbnds {
                                let bandsize = f.spxbndsz[bnd];
                                let nratio = ((spxmant as f32 + 0.5 * bandsize as f32) / end - noffset).clamp(0.0, 1.0);
                                c.nblend[bnd] = nratio.sqrt();
                                c.sblend[bnd] = (1.0 - nratio).sqrt();
                                spxmant += bandsize;
                            }
                        }
                    } else {
                        c.first_spxcos = true;
                    }
                }
            }
        }
        // --- coupling strategy ------------------------------------------------
        // E-AC-3 decided cplinu per block in audfrm (and for acmod ≤ 1 it is
        // simply 0 with no cplstre at all); AC-3 signals both here.
        let cplstre = if eac3 { f.cplstre[blk] } else { br.read_bit()? };
        if eac3 {
            f.cplinu = f.cplinu_blk[blk];
        }
        if cplstre {
            f.cplinu = if eac3 { f.cplinu_blk[blk] } else { br.read_bit()? };
            if f.cplinu {
                if eac3 && br.read_bit()? {
                    return Err(AudioError::Unsupported(
                        "eac3: enhanced coupling (ecplinu=1) — not implemented (no cross-check vector: libavcodec refuses it too)".into(),
                    ));
                }
                if eac3 && hdr.acmod == 2 {
                    self.chans[0].incpl = true;
                    self.chans[1].incpl = true;
                } else {
                    for ch in 0..nf {
                        self.chans[ch].incpl = br.read_bit()?;
                    }
                }
                if hdr.acmod == 2 {
                    f.phsflginu = br.read_bit()?;
                }
                f.cplbegf = br.read(4)? as u8;
                f.cplendf = if !eac3 || !f.spxinu {
                    br.read(4)? as i32
                } else if f.spxbegf < 6 {
                    i32::from(f.spxbegf) - 2
                } else {
                    i32::from(f.spxbegf) * 2 - 7
                };
                let ncplsubnd = 3 + f.cplendf - i32::from(f.cplbegf);
                if ncplsubnd < 1 || i32::from(f.cplbegf) + ncplsubnd > 18 {
                    return Err(err(format!(
                        "ac3: invalid coupling range cplbegf={} cplendf={}",
                        f.cplbegf, f.cplendf
                    )));
                }
                f.ncplsubnd = ncplsubnd as usize;
                f.cplstrtmant = usize::from(f.cplbegf) * 12 + 37;
                f.cplendmant = f.cplstrtmant + f.ncplsubnd * 12;
                // The transmitted bits are indexed relative to cplbegf
                // (AC-3 Table 5.3) while E-AC-3's default structure (Table
                // E2.12) is per absolute sub-band; the array is kept absolute.
                let cplbndstrce = if eac3 { br.read_bit()? } else { true };
                let b0 = usize::from(f.cplbegf);
                if cplbndstrce {
                    for bnd in 1..f.ncplsubnd {
                        f.cplbndstrc[b0 + bnd] = br.read(1)? as u8;
                    }
                }
                f.ncplbnd = f.ncplsubnd
                    - f.cplbndstrc[b0 + 1..b0 + f.ncplsubnd].iter().map(|&b| usize::from(b)).sum::<usize>();
            } else {
                for ch in 0..nf {
                    self.chans[ch].incpl = false;
                    self.chans[ch].first_cplcos = true;
                }
                f.first_cplleak = true;
                f.phsflginu = false;
            }
        } else if blk == 0 && !eac3 {
            return Err(err("ac3: cplstre must be 1 in block 0"));
        }
        if f.cplinu {
            self.feat.cpl_blocks += 1;
            self.feat.phsflg_blocks += u64::from(hdr.acmod == 2 && f.phsflginu);
        }
        self.feat.spx_blocks += u64::from(f.spxinu);
        // --- coupling coordinates -------------------------------------------
        if f.cplinu {
            for ch in 0..nf {
                let c = &mut self.chans[ch];
                if c.incpl {
                    c.cplcoe = if eac3 && c.first_cplcos {
                        c.first_cplcos = false;
                        true
                    } else {
                        br.read_bit()?
                    };
                    if c.cplcoe {
                        let mstrcplco = br.read(2)?;
                        for bnd in 0..f.ncplbnd {
                            let exp = br.read(4)?;
                            let mant = br.read(4)? as f32;
                            let temp = if exp == 15 { mant / 16.0 } else { (mant + 16.0) / 32.0 };
                            c.cplco_band[bnd] = temp / (1u64 << (exp + 3 * mstrcplco)) as f32;
                        }
                    }
                } else if eac3 {
                    c.first_cplcos = true;
                }
            }
            if hdr.acmod == 2 && f.phsflginu && (self.chans[0].cplcoe || self.chans[1].cplcoe) {
                for bnd in 0..f.ncplbnd {
                    f.phsflg[bnd] = br.read_bit()?;
                }
            }
        }
        // --- rematrixing ------------------------------------------------------
        if hdr.acmod == 2 {
            let rematstr = if eac3 && blk == 0 { true } else { br.read_bit()? };
            if rematstr {
                f.nrematbd = if f.cplinu {
                    match f.cplbegf {
                        0 => 2,
                        1 | 2 => 3,
                        _ => 4,
                    }
                } else if f.spxinu {
                    if f.spxbegf < 2 { 3 } else { 4 }
                } else {
                    4
                };
                for bnd in 0..f.nrematbd {
                    f.rematflg[bnd] = br.read_bit()?;
                }
            }
            self.feat.remat_blocks += u64::from(f.rematflg[..f.nrematbd].iter().any(|&r| r));
        }
        // --- exponent strategy and channel bandwidth --------------------------
        if !eac3 {
            if f.cplinu {
                self.chans[CPL].expstr = br.read(2)? as u8;
            }
            for ch in 0..nf {
                self.chans[ch].expstr = br.read(2)? as u8;
            }
            if hdr.lfeon {
                self.chans[LFE].expstr = br.read(1)? as u8;
            }
        } else {
            self.chans[CPL].expstr = f.cplexpstr_blk[blk];
            for ch in 0..nf {
                self.chans[ch].expstr = f.chexpstr_blk[blk][ch];
            }
            self.chans[LFE].expstr = f.lfeexpstr_blk[blk];
        }
        for ch in 0..nf {
            let c = &mut self.chans[ch];
            if c.expstr != REUSE && !c.incpl && !c.inspx {
                c.bwcod = br.read(6)? as u8;
                if c.bwcod > 60 {
                    return Err(err(format!("ac3: chbwcod {} > 60", c.bwcod)));
                }
            } else if blk == 0 && c.expstr == REUSE {
                return Err(err("ac3: chexpstr reuse in block 0"));
            }
            c.endmant = if c.incpl {
                f.cplstrtmant
            } else if c.inspx {
                usize::from(SPXBANDTABLE[f.spx_begin_subbnd])
            } else {
                (usize::from(c.bwcod) + 12) * 3 + 37
            };
        }
        self.chans[LFE].endmant = 7;
        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!("blk {blk} blksw {:?} dith {:?} pos {} cplinu {} cplbegf {} cplendf {} ncplsubnd {} ncplbnd {} phsflginu {} expstr {:?} bwcod {:?} endmant {:?} incpl {:?} rematflg {:?} nrematbd {}",
                (0..nf).map(|c| self.chans[c].blksw).collect::<Vec<_>>(), (0..nf).map(|c| self.chans[c].dith).collect::<Vec<_>>(), br.pos(), f.cplinu, f.cplbegf, f.cplendf, f.ncplsubnd, f.ncplbnd, f.phsflginu,
                (0..nf).map(|c| self.chans[c].expstr).collect::<Vec<_>>(),
                (0..nf).map(|c| self.chans[c].bwcod).collect::<Vec<_>>(),
                (0..nf).map(|c| self.chans[c].endmant).collect::<Vec<_>>(),
                (0..nf).map(|c| self.chans[c].incpl).collect::<Vec<_>>(), f.rematflg, f.nrematbd);
        }
        // --- exponents --------------------------------------------------------
        if f.cplinu && self.chans[CPL].expstr != REUSE {
            let absexp = (br.read(4)? << 1) as u8;
            let grpsize = grp_size(self.chans[CPL].expstr);
            let ncplgrps = (f.cplendmant - f.cplstrtmant) / (3 * grpsize);
            let mut tmp = [0u8; 280];
            decode_exponents(br, ncplgrps, grpsize, absexp, &mut tmp)?;
            let c = &mut self.chans[CPL];
            c.exps[f.cplstrtmant..f.cplendmant].copy_from_slice(&tmp[1..=f.cplendmant - f.cplstrtmant]);
        }
        for ch in 0..nf {
            let c = &mut self.chans[ch];
            if c.expstr != REUSE {
                let absexp = br.read(4)? as u8;
                let grpsize = grp_size(c.expstr);
                let nchgrps = (c.endmant - 1).div_ceil(3 * grpsize);
                let mut tmp = [0u8; 280];
                decode_exponents(br, nchgrps, grpsize, absexp, &mut tmp)?;
                c.exps[..c.endmant].copy_from_slice(&tmp[..c.endmant]);
                br.read(2)?; // gainrng — informational
            }
        }
        if hdr.lfeon && self.chans[LFE].expstr != REUSE {
            let absexp = br.read(4)? as u8;
            let mut tmp = [0u8; 280];
            decode_exponents(br, 2, 1, absexp, &mut tmp)?;
            self.chans[LFE].exps[..7].copy_from_slice(&tmp[..7]);
        }
        // --- bit allocation parameters ---------------------------------------
        if !eac3 || f.bamode {
            if br.read_bit()? {
                let sd = br.read(2)? as u8;
                let fd = br.read(2)? as u8;
                let sg = br.read(2)? as u8;
                let db = br.read(2)? as u8;
                let fl = br.read(3)? as u8;
                f.ba = BaParams::from_codes(sd, fd, sg, db, fl);
            } else if blk == 0 && !eac3 {
                return Err(err("ac3: baie must be 1 in block 0"));
            }
        }
        // --- SNR offsets ------------------------------------------------------
        if eac3 && f.snroffststr == 0 {
            f.csnroffst = f.frmcsnroffst;
            for c in &mut self.chans {
                c.fsnroffst = f.frmfsnroffst;
            }
        } else {
            let snroffste = if eac3 && blk == 0 { true } else { br.read_bit()? };
            if snroffste {
                f.csnroffst = br.read(6)? as u8;
                if eac3 && f.snroffststr == 1 {
                    let blkfsnroffst = br.read(4)? as u8;
                    for c in &mut self.chans {
                        c.fsnroffst = blkfsnroffst;
                    }
                } else {
                    if f.cplinu {
                        self.chans[CPL].fsnroffst = br.read(4)? as u8;
                        if !eac3 {
                            self.chans[CPL].fgaincod = br.read(3)? as u8;
                        }
                    }
                    for ch in 0..nf {
                        self.chans[ch].fsnroffst = br.read(4)? as u8;
                        if !eac3 {
                            self.chans[ch].fgaincod = br.read(3)? as u8;
                        }
                    }
                    if hdr.lfeon {
                        self.chans[LFE].fsnroffst = br.read(4)? as u8;
                        if !eac3 {
                            self.chans[LFE].fgaincod = br.read(3)? as u8;
                        }
                    }
                }
            } else if blk == 0 {
                return Err(err("ac3: snroffste must be 1 in block 0"));
            }
        }
        // --- fast gain codes (E-AC-3) ----------------------------------------
        if eac3 {
            let fgaincode = if f.frmfgaincode { br.read_bit()? } else { false };
            if fgaincode {
                if f.cplinu {
                    self.chans[CPL].fgaincod = br.read(3)? as u8;
                }
                for ch in 0..nf {
                    self.chans[ch].fgaincod = br.read(3)? as u8;
                }
                if hdr.lfeon {
                    self.chans[LFE].fgaincod = br.read(3)? as u8;
                }
            } else {
                for c in &mut self.chans {
                    c.fgaincod = 4;
                }
            }
            if hdr.strmtyp == 0 && br.read_bit()? {
                br.read(10)?; // convsnroffst
            }
        }
        // --- coupling leak ----------------------------------------------------
        if f.cplinu {
            let cplleake = if eac3 && f.first_cplleak {
                f.first_cplleak = false;
                true
            } else {
                br.read_bit()?
            };
            if cplleake {
                f.cplfleak = br.read(3)? as u8;
                f.cplsleak = br.read(3)? as u8;
            }
        }
        // --- delta bit allocation -------------------------------------------
        if (!eac3 || f.dbaflde) && br.read_bit()? {
            self.feat.deltba_blocks += 1;
            if f.cplinu {
                self.chans[CPL].deltbae = br.read(2)? as u8;
            }
            for ch in 0..nf {
                self.chans[ch].deltbae = br.read(2)? as u8;
            }
            let mut order: Vec<usize> = Vec::with_capacity(6);
            if f.cplinu {
                order.push(CPL);
            }
            order.extend(0..nf);
            for &ch in &order {
                let c = &mut self.chans[ch];
                match c.deltbae {
                    1 => {
                        let nseg = br.read(3)? as u8 + 1;
                        c.delta.nseg = nseg;
                        for seg in 0..nseg as usize {
                            c.delta.offst[seg] = br.read(5)? as u8;
                            c.delta.len[seg] = br.read(4)? as u8;
                            c.delta.ba[seg] = br.read(3)? as u8;
                        }
                    }
                    3 => return Err(err("ac3: reserved deltbae 3")),
                    _ => {}
                }
            }
        }
        // --- skip field -------------------------------------------------------
        if (!eac3 || f.skipflde) && br.read_bit()? {
            let skipl = br.read(9)? as usize;
            br.skip(skipl * 8)?;
        }
        // --- bit allocation ---------------------------------------------------
        self.run_bit_allocation(f);
        if tracing::enabled!(tracing::Level::TRACE) {
            tracing::trace!("ba: csnr {} fsnr {:?} fgain {:?} sd/fd/sg/db/fl {:?} nzbap {:?} bap1 {:?} exps0..8 {:?}",
                f.csnroffst,
                (0..NCH).map(|c| self.chans[c].fsnroffst).collect::<Vec<_>>(),
                (0..NCH).map(|c| self.chans[c].fgaincod).collect::<Vec<_>>(),
                f.ba,
                (0..nf).map(|c| self.chans[c].bap[..self.chans[c].endmant].iter().filter(|&&b| b != 0).count()).collect::<Vec<_>>(),
                (0..nf).map(|c| self.chans[c].bap[..self.chans[c].endmant].iter().filter(|&&b| b == 1).count()).collect::<Vec<_>>(),
                (0..nf).map(|c| self.chans[c].exps[30..42].to_vec()).collect::<Vec<_>>());
        }
        // --- mantissas --------------------------------------------------------
        for c in &mut self.chans {
            c.coeffs = [0.0; NB];
        }
        let mut groups = Groups::default();
        let mut got_cplchan = false;
        for ch in 0..nf {
            let aht = self.chans[ch].ahtinu;
            if aht == 0 {
                self.read_mantissas(br, &mut groups, ch, 0, self.chans[ch].endmant, true)?;
            } else if aht == 1 {
                self.read_aht_mantissas(br, ch, 0, self.chans[ch].endmant)?;
                self.chans[ch].ahtinu = -1;
            }
            if aht != 0 {
                self.apply_aht_block(ch, blk, 0, self.chans[ch].endmant, true);
            }
            if f.cplinu && self.chans[ch].incpl && !got_cplchan {
                match self.chans[CPL].ahtinu {
                    0 => self.read_mantissas(br, &mut groups, CPL, f.cplstrtmant, f.cplendmant, false)?,
                    1 => {
                        self.read_aht_mantissas(br, CPL, f.cplstrtmant, f.cplendmant)?;
                        self.chans[CPL].ahtinu = -1;
                    }
                    _ => {}
                }
                if self.chans[CPL].ahtinu != 0 {
                    self.apply_aht_block(CPL, blk, f.cplstrtmant, f.cplendmant, false);
                }
                got_cplchan = true;
            }
        }
        if hdr.lfeon {
            match self.chans[LFE].ahtinu {
                0 => self.read_mantissas(br, &mut groups, LFE, 0, 7, false)?,
                1 => {
                    self.read_aht_mantissas(br, LFE, 0, 7)?;
                    self.chans[LFE].ahtinu = -1;
                }
                _ => {}
            }
            if self.chans[LFE].ahtinu != 0 {
                self.apply_aht_block(LFE, blk, 0, 7, false);
            }
        }
        // --- decoupling (§7.4.3) ----------------------------------------------
        if f.cplinu {
            // expand band coordinates / phase flags to sub-bands
            let mut band_of_sub = [0usize; 18];
            let mut band = 0usize;
            for sb in 0..f.ncplsubnd {
                if sb > 0 && f.cplbndstrc[usize::from(f.cplbegf) + sb] == 0 {
                    band += 1;
                }
                band_of_sub[sb] = band;
            }
            for ch in 0..nf {
                if !self.chans[ch].incpl {
                    continue;
                }
                let dith = self.chans[ch].dith && self.noise_fill;
                for sb in 0..f.ncplsubnd {
                    let bnd = band_of_sub[sb];
                    let mut co = self.chans[ch].cplco_band[bnd] * 8.0;
                    if hdr.acmod == 2 && ch == 1 && f.phsflginu && f.phsflg[bnd] {
                        co = -co;
                    }
                    let base = (usize::from(f.cplbegf) + sb) * 12 + 37;
                    for bin in base..base + 12 {
                        let v = if self.chans[CPL].bap[bin] == 0 && self.chans[CPL].ahtinu == 0 {
                            if dith { self.rand() * 0.707 * exp_scale(self.chans[CPL].exps[bin]) } else { 0.0 }
                        } else {
                            self.chans[CPL].coeffs[bin]
                        };
                        self.chans[ch].coeffs[bin] = v * co;
                    }
                }
            }
        }
        // --- spectral extension (Annex E §3.6.4) ------------------------------
        if f.spxinu {
            for ch in 0..nf {
                if self.chans[ch].inspx {
                    self.spectral_extension(f, ch);
                }
            }
        }
        // --- rematrixing (§7.5.4) ---------------------------------------------
        if hdr.acmod == 2 {
            let bounds = [13usize, 25, 37, 61, 253];
            let end = self.chans[0].endmant.min(self.chans[1].endmant).max(13);
            for bnd in 0..f.nrematbd {
                if !f.rematflg[bnd] {
                    continue;
                }
                let lo = bounds[bnd];
                let hi = bounds[bnd + 1].min(end);
                for bin in lo..hi {
                    let l = self.chans[0].coeffs[bin];
                    let r = self.chans[1].coeffs[bin];
                    self.chans[0].coeffs[bin] = l + r;
                    self.chans[1].coeffs[bin] = l - r;
                }
            }
        }
        // --- dynamic range compression (§7.7.1.2) -----------------------------
        self.blocks += 1;
        if f.dynrng != 0 || (hdr.acmod == 0 && f.dynrng2 != 0) {
            self.drc_blocks += 1;
            self.feat.dynrng_blocks += 1;
        }
        if self.drc_scale > 0.0 {
            let g1 = dynrng_gain(f.dynrng).powf(self.drc_scale);
            let g2 = if hdr.acmod == 0 { dynrng_gain(f.dynrng2).powf(self.drc_scale) } else { g1 };
            for ch in 0..nf {
                let g = if hdr.acmod == 0 && ch == 1 { g2 } else { g1 };
                if g != 1.0 {
                    for v in &mut self.chans[ch].coeffs {
                        *v *= g;
                    }
                }
            }
            if hdr.lfeon && g1 != 1.0 {
                for v in &mut self.chans[LFE].coeffs {
                    *v *= g1;
                }
            }
        }
        Ok(())
    }

    /// §7.2.2 for every active channel of the block.
    fn run_bit_allocation(&mut self, f: &Frame) {
        let hdr = f.hdr;
        let nf = hdr.nfchans;
        let mut active: Vec<usize> = Vec::with_capacity(NCH);
        if f.cplinu {
            active.push(CPL);
        }
        active.extend(0..nf);
        if hdr.lfeon {
            active.push(LFE);
        }
        // §7.2.2.1.1 special case: every SNR offset zero → no bits at all.
        let all_zero = f.csnroffst == 0 && active.iter().all(|&ch| self.chans[ch].fsnroffst == 0);
        for &ch in &active {
            let (start, end) = match ch {
                CPL => (f.cplstrtmant, f.cplendmant),
                LFE => (0, 7),
                _ => (0, self.chans[ch].endmant),
            };
            let c = &mut self.chans[ch];
            if all_zero {
                c.bap[start..end].fill(0);
                continue;
            }
            let kind = match ch {
                CPL => Kind::Cpl {
                    fastleak: (i32::from(f.cplfleak) << 8) + 768,
                    slowleak: (i32::from(f.cplsleak) << 8) + 768,
                },
                LFE => Kind::Lfe,
                _ => Kind::Fbw,
            };
            let delta = if ch != LFE && c.deltbae != 2 && c.delta.nseg > 0 { Some(&c.delta) } else { None };
            let mut bap = [0u8; NB];
            compute_bap(
                &c.exps,
                start,
                end,
                hdr.fscod,
                &f.ba,
                fast_gain(c.fgaincod),
                snr_offset(f.csnroffst, c.fsnroffst),
                kind,
                delta,
                c.ahtinu != 0,
                &mut bap,
            );
            c.bap[start..end].copy_from_slice(&bap[start..end]);
        }
    }

    /// §7.3: unpack `bap`-sized mantissas for bins `start..end` of channel
    /// `ch` into `coeffs`, scaled by the exponent. `dither` enables §7.3.4
    /// noise fill for zero-bit mantissas when the channel's `dithflag` is set.
    fn read_mantissas(
        &mut self,
        br: &mut BitReader,
        g: &mut Groups,
        ch: usize,
        start: usize,
        end: usize,
        dither: bool,
    ) -> Result<(), AudioError> {
        let dith = dither && self.chans[ch].dith && self.noise_fill;
        self.total_bins += (end - start) as u64;
        for bin in start..end {
            let bap = self.chans[ch].bap[bin];
            let m = match bap {
                0 => {
                    if dith {
                        self.dithered_bins += 1;
                        self.rand() * 0.707
                    } else {
                        0.0
                    }
                }
                1 => {
                    if g.b1.0 == 0 {
                        let code = br.read(5)?;
                        g.b1.1 = [(code / 9).min(2) as u8, ((code % 9) / 3) as u8, (code % 3) as u8];
                        g.b1.0 = 3;
                    }
                    let v = g.b1.1[3 - g.b1.0 as usize];
                    g.b1.0 -= 1;
                    SYM_QUANT_3[v as usize]
                }
                2 => {
                    if g.b2.0 == 0 {
                        let code = br.read(7)?;
                        g.b2.1 = [(code / 25).min(4) as u8, ((code % 25) / 5) as u8, (code % 5) as u8];
                        g.b2.0 = 3;
                    }
                    let v = g.b2.1[3 - g.b2.0 as usize];
                    g.b2.0 -= 1;
                    SYM_QUANT_5[v as usize]
                }
                3 => SYM_QUANT_7[(br.read(3)? as usize).min(6)],
                4 => {
                    if g.b4.0 == 0 {
                        let code = br.read(7)?;
                        g.b4.1 = [(code / 11).min(10) as u8, (code % 11) as u8];
                        g.b4.0 = 2;
                    }
                    let v = g.b4.1[2 - g.b4.0 as usize];
                    g.b4.0 -= 1;
                    SYM_QUANT_11[v as usize]
                }
                5 => SYM_QUANT_15[(br.read(4)? as usize).min(14)],
                _ => {
                    let bits = u32::from(ASYM_MANT_BITS[(bap - 6) as usize]);
                    br.read_signed(bits)? as f32 / (1u32 << (bits - 1)) as f32
                }
            };
            self.chans[ch].coeffs[bin] = m * exp_scale(self.chans[ch].exps[bin]);
        }
        Ok(())
    }

    /// Annex E §2.2.4 / §3.4.4: read the six blocks' worth of AHT mantissas
    /// for bins `start..end` of channel `ch` (VQ or GAQ per `hebap`), then
    /// invert the DCT (§3.4.5) into `Chan::aht[blk][bin]`.
    fn read_aht_mantissas(&mut self, br: &mut BitReader, ch: usize, start: usize, end: usize) -> Result<(), AudioError> {
        let gaqmod = br.read(2)? as u8;
        self.feat.aht_channels += 1;
        self.feat.gaq_channels += u64::from(gaqmod != 0);
        let endbap: u8 = if gaqmod < 2 { 12 } else { 17 };
        // §3.4.2 helper variables
        let mut gaqbin = [0i8; NB];
        let mut active = 0usize;
        for bin in start..end {
            let h = self.chans[ch].bap[bin];
            gaqbin[bin] = if h > 7 && h < endbap {
                active += 1;
                1
            } else if h >= endbap {
                -1
            } else {
                0
            };
        }
        let sections = match gaqmod {
            0 => 0,
            1 | 2 => active,
            _ => active.div_ceil(3),
        };
        // gains, expanded to a log2 gain per GAQ bin in ascending order
        let mut log_gain = [0u8; NB];
        let mut gains: Vec<u8> = Vec::with_capacity(active);
        for _ in 0..sections {
            match gaqmod {
                1 => gains.push(br.read(1)? as u8),
                2 => gains.push(br.read(1)? as u8 * 2),
                3 => {
                    let grp = br.read(5)?;
                    gains.push((grp / 9).min(2) as u8);
                    gains.push(((grp % 9) / 3) as u8);
                    gains.push((grp % 3) as u8);
                }
                _ => {}
            }
        }
        let mut gi = 0usize;
        for bin in start..end {
            if gaqbin[bin] == 1 {
                log_gain[bin] = gains.get(gi).copied().unwrap_or(0);
                gi += 1;
            }
        }
        let mut pre = Box::new([[0.0f32; NB]; 6]);
        for bin in start..end {
            let hebap = self.chans[ch].bap[bin];
            match hebap {
                0 => {}
                1..=7 => {
                    self.feat.vq_bins += 1;
                    let idx = br.read(u32::from(HEBAP_MANT_BITS[hebap as usize]))? as usize;
                    let vq = vq_table(hebap);
                    let v = vq.get(idx).copied().unwrap_or([0; 6]);
                    for (n, x) in v.iter().enumerate() {
                        pre[n][bin] = f32::from(*x) / 32768.0;
                    }
                }
                _ => {
                    self.feat.gaq_bins += 1;
                    let m = u32::from(HEBAP_MANT_BITS[hebap as usize]);
                    let row = (hebap - 8) as usize;
                    let lg = if gaqbin[bin] == 1 { u32::from(log_gain[bin]) } else { 0 };
                    let gbits = m - lg;
                    for n in 0..6 {
                        let v = br.read_signed(gbits)?;
                        pre[n][bin] = if lg > 0 && v == -(1 << (gbits - 1)) {
                            // large mantissa: tag found, then the long codeword
                            self.feat.gaq_large_mantissas += 1;
                            let lbits = if lg == 1 { m - 1 } else { m };
                            let x = br.read_signed(lbits)? as f32 / (1u32 << (lbits - 1)) as f32;
                            let a = f32::from(GAQ_REMAP_A[row][lg as usize]) / 32768.0;
                            let b = f32::from(GAQ_REMAP_B[row][lg as usize][usize::from(x < 0.0)]) / 32768.0;
                            x + a * x + b
                        } else if lg > 0 {
                            // small mantissa: attenuated by 1/Gk, no remap
                            v as f32 / (1u32 << (gbits - 1)) as f32 / (1u32 << lg) as f32
                        } else {
                            let x = v as f32 / (1u32 << (m - 1)) as f32;
                            let a = f32::from(GAQ_REMAP_A[row][0]) / 32768.0;
                            x + a * x
                        };
                    }
                }
            }
        }
        // §3.4.5 inverse DCT across the six blocks
        let mut out = Box::new([[0.0f32; NB]; 6]);
        let sqrt2 = std::f32::consts::SQRT_2;
        for bin in start..end {
            for m in 0..6 {
                let mut acc = 0.0f32;
                for j in 0..6 {
                    let r = if j == 0 { 1.0 / sqrt2 } else { 1.0 };
                    let ang = (j * (2 * m + 1)) as f32 * std::f32::consts::PI / 12.0;
                    acc += r * pre[j][bin] * ang.cos();
                }
                out[m][bin] = sqrt2 * acc;
            }
        }
        self.chans[ch].aht = Some(out);
        Ok(())
    }

    /// Scale block `blk`'s AHT mantissas by the (frame-constant) exponents
    /// into `coeffs`, with §7.3.4 dither for zero-bit bins.
    fn apply_aht_block(&mut self, ch: usize, blk: usize, start: usize, end: usize, dither: bool) {
        let dith = dither && self.chans[ch].dith && self.noise_fill;
        let Some(aht) = self.chans[ch].aht.take() else { return };
        for bin in start..end {
            let m = if self.chans[ch].bap[bin] == 0 {
                if dith { self.rand() * 0.707 } else { 0.0 }
            } else {
                aht[blk][bin]
            };
            self.chans[ch].coeffs[bin] = m * exp_scale(self.chans[ch].exps[bin]);
        }
        self.chans[ch].aht = Some(aht);
    }

    /// Annex E §3.6.4 high-frequency synthesis for one channel.
    fn spectral_extension(&mut self, f: &Frame, ch: usize) {
        let copystart = usize::from(SPXBANDTABLE[f.spxstrtf as usize]);
        let copyend = usize::from(SPXBANDTABLE[f.spx_begin_subbnd]);
        let nspx = f.nspxbnds;
        let mut wrapflag = [false; 17];
        // §3.6.4.1 translation
        {
            let tc = &mut self.chans[ch].coeffs;
            let mut copyindex = copystart;
            let mut insertindex = copyend;
            for bnd in 0..nspx {
                let bandsize = f.spxbndsz[bnd];
                if copyindex + bandsize > copyend {
                    copyindex = copystart;
                    wrapflag[bnd] = true;
                }
                for _ in 0..bandsize {
                    if copyindex == copyend {
                        copyindex = copystart;
                    }
                    tc[insertindex] = tc[copyindex];
                    insertindex += 1;
                    copyindex += 1;
                }
            }
        }
        // §3.6.4.2.2 banded RMS energy of the translated coefficients
        let mut rms = [0.0f32; 17];
        {
            let tc = &self.chans[ch].coeffs;
            let mut spxmant = copyend;
            for bnd in 0..nspx {
                let bandsize = f.spxbndsz[bnd];
                let acc: f32 = tc[spxmant..spxmant + bandsize].iter().map(|v| v * v).sum();
                rms[bnd] = (acc / bandsize as f32).sqrt();
                spxmant += bandsize;
            }
        }
        // §3.6.4.2.3 notch filter at the baseband border and wrap points
        if self.chans[ch].inspxatten {
            let att = &SPXATTENTAB[self.chans[ch].spxattencod as usize];
            let tc = &mut self.chans[ch].coeffs;
            let notch = |tc: &mut [f32; NB], filtbin: usize| {
                let taps = [att[0], att[1], att[2], att[1], att[0]];
                for (i, t) in taps.iter().enumerate() {
                    if let Some(v) = tc.get_mut(filtbin + i) {
                        *v *= t;
                    }
                }
            };
            notch(tc, copyend.saturating_sub(2));
            let mut filtbin = copyend + f.spxbndsz[0];
            for bnd in 1..nspx {
                if wrapflag[bnd] {
                    notch(tc, filtbin.saturating_sub(2));
                }
                filtbin += f.spxbndsz[bnd];
            }
        }
        // §3.6.4.2.4 noise blending and §3.6.4.3 coordinate scaling
        let sqrt3 = 3.0f32.sqrt();
        let mut spxmant = copyend;
        for bnd in 0..nspx {
            let bandsize = f.spxbndsz[bnd];
            let nscale = rms[bnd] * self.chans[ch].nblend[bnd];
            let sscale = self.chans[ch].sblend[bnd];
            let co = self.chans[ch].spxco[bnd] * 32.0;
            for bin in spxmant..spxmant + bandsize {
                // zero-mean, unit variance; the diagnostic switch that turns
                // off the §7.3.4 dither turns this off too.
                let noise = if self.noise_fill { self.rand() * sqrt3 } else { 0.0 };
                let tc = &mut self.chans[ch].coeffs;
                tc[bin] = (tc[bin] * sscale + noise * nscale) * co;
            }
            spxmant += bandsize;
        }
    }
}

impl Frame {
    fn new(hdr: Header) -> Self {
        Self {
            hdr,
            ahte: false,
            snroffststr: 2,
            blkswe: true,
            dithflage: true,
            bamode: true,
            frmfgaincode: true,
            dbaflde: true,
            skipflde: true,
            cplstre: [false; 6],
            cplinu_blk: [false; 6],
            cplexpstr_blk: [REUSE; 6],
            chexpstr_blk: [[REUSE; MAX_FBW]; 6],
            lfeexpstr_blk: [REUSE; 6],
            cplahtinu: 0,
            lfeahtinu: 0,
            frmcsnroffst: 0,
            frmfsnroffst: 0,
            cplinu: false,
            cplbegf: 0,
            cplendf: 0,
            ncplsubnd: 0,
            ncplbnd: 0,
            cplbndstrc: DEFCPLBNDSTRC,
            phsflginu: false,
            phsflg: [false; 18],
            cplstrtmant: 37,
            cplendmant: 37,
            cplfleak: 0,
            cplsleak: 0,
            first_cplleak: false,
            spxinu: false,
            spxstrtf: 0,
            spxbegf: 0,
            spx_begin_subbnd: 0,
            spx_end_subbnd: 0,
            spxbndstrc: DEFSPXBNDSTRC,
            nspxbnds: 0,
            spxbndsz: [0; 17],
            rematflg: [false; 4],
            nrematbd: 0,
            ba: BaParams::from_codes(2, 1, 1, 2, 7),
            csnroffst: 0,
            dynrng: 0,
            dynrng2: 0,
        }
    }
}

/// Exponents per group for a strategy code (Table 7.4): D15 → 1, D25 → 2, D45 → 4.
fn grp_size(expstr: u8) -> usize {
    match expstr {
        1 => 1,
        2 => 2,
        _ => 4,
    }
}

/// `2^-exp` for the 5-bit exponents 0..=24.
fn exp_scale(exp: u8) -> f32 {
    1.0 / (1u32 << exp.min(24)) as f32
}

/// §7.1.3: decode `ngrps` 7-bit grouped values into absolute exponents.
/// `out[0] = absexp`, `out[1..=ngrps*3*grpsize]` follow.
fn decode_exponents(br: &mut BitReader, ngrps: usize, grpsize: usize, absexp: u8, out: &mut [u8]) -> Result<(), AudioError> {
    out[0] = absexp;
    let mut prev = i32::from(absexp);
    let mut i = 1usize;
    for _ in 0..ngrps {
        let mut expacc = br.read(7)? as i32;
        let d0 = expacc / 25;
        expacc -= 25 * d0;
        let d1 = expacc / 5;
        let d2 = expacc - 5 * d1;
        for d in [d0, d1, d2] {
            prev += d - 2;
            if !(0..=24).contains(&prev) {
                return Err(err(format!("ac3: absolute exponent {prev} out of 0..=24")));
            }
            for _ in 0..grpsize {
                if i < out.len() {
                    out[i] = prev as u8;
                }
                i += 1;
            }
        }
    }
    Ok(())
}

/// §7.7.1.2: linear gain for an 8-bit `dynrng` word (`X0X1X2.Y3..Y7`):
/// `2^(X+1) · (0.1Y3Y4Y5Y6Y7)₂` with X a 3-bit two's-complement integer.
pub(super) fn dynrng_gain(dynrng: u8) -> f32 {
    let x = i32::from(dynrng >> 5);
    let x = if x >= 4 { x - 8 } else { x };
    let y = f32::from(32 + (dynrng & 31)) / 64.0;
    y * 2f32.powi(x + 1)
}

/// Internal channel indices in output order: fronts as FL FR FC, then LFE,
/// then the surround(s) — ffmpeg's native order for each layout, which is
/// what `channelmap` and the Opus encoder assume for a channel count.
fn output_order(acmod: u8, lfeon: bool) -> Vec<usize> {
    let nf = nfchans_for(acmod);
    let mut v: Vec<usize> = match acmod {
        1 => vec![0],
        0 | 2 | 4 | 6 => vec![0, 1],
        _ => vec![0, 2, 1], // L C R → FL FR FC
    };
    if lfeon {
        v.push(LFE);
    }
    let fronts = if acmod == 1 { 1 } else if acmod & 1 == 1 { 3 } else { 2 };
    v.extend(fronts..nf);
    v
}

/// AC-3 `bsi()` (Table 5.2) after the fixed 8-byte prefix; skips everything
/// the decoder does not act on.
fn parse_ac3_bsi(br: &mut BitReader, hdr: &Header) -> Result<(), AudioError> {
    br.skip(40)?; // syncinfo
    br.read(5)?; // bsid
    br.read(3)?; // bsmod
    let acmod = br.read(3)? as u8;
    if (acmod & 1) != 0 && acmod != 1 {
        br.read(2)?;
    }
    if (acmod & 4) != 0 {
        br.read(2)?;
    }
    if acmod == 2 {
        br.read(2)?;
    }
    br.read(1)?; // lfeon
    br.read(5)?; // dialnorm
    if br.read_bit()? {
        br.read(8)?; // compr
    }
    if br.read_bit()? {
        br.read(8)?; // langcod
    }
    if br.read_bit()? {
        br.read(7)?; // mixlevel + roomtyp
    }
    if hdr.acmod == 0 {
        br.read(5)?; // dialnorm2
        if br.read_bit()? {
            br.read(8)?;
        }
        if br.read_bit()? {
            br.read(8)?;
        }
        if br.read_bit()? {
            br.read(7)?;
        }
    }
    br.read(2)?; // copyrightb, origbs
    if br.read_bit()? {
        br.read(14)?; // timecod1
    }
    if br.read_bit()? {
        br.read(14)?; // timecod2
    }
    if br.read_bit()? {
        let addbsil = br.read(6)? as usize;
        br.skip((addbsil + 1) * 8)?;
    }
    Ok(())
}

/// E-AC-3 `bsi()` (Table E1.2). Everything after the 8 fixed bytes is
/// metadata this decoder does not act on; it is walked, not interpreted.
fn parse_eac3_bsi(br: &mut BitReader, hdr: &Header) -> Result<(), AudioError> {
    br.skip(16)?; // syncword
    let strmtyp = br.read(2)? as u8;
    br.read(3)?; // substreamid
    br.read(11)?; // frmsiz
    let fscod = br.read(2)?;
    let numblkscod = if fscod == 3 {
        br.read(2)?; // fscod2
        3
    } else {
        br.read(2)?
    };
    let acmod = br.read(3)? as u8;
    let lfeon = br.read_bit()?;
    br.read(5)?; // bsid
    br.read(5)?; // dialnorm
    if br.read_bit()? {
        br.read(8)?; // compr
    }
    if acmod == 0 {
        br.read(5)?; // dialnorm2
        if br.read_bit()? {
            br.read(8)?; // compr2
        }
    }
    if strmtyp == 1 && br.read_bit()? {
        br.read(16)?; // chanmap
    }
    if br.read_bit()? {
        // mixing metadata
        if acmod > 2 {
            br.read(2)?; // dmixmod
        }
        if (acmod & 1) != 0 && acmod > 2 {
            br.read(6)?; // ltrtcmixlev, lorocmixlev
        }
        if (acmod & 4) != 0 {
            br.read(6)?; // ltrtsurmixlev, lorosurmixlev
        }
        if lfeon && br.read_bit()? {
            br.read(5)?; // lfemixlevcod
        }
        if strmtyp == 0 {
            if br.read_bit()? {
                br.read(6)?; // pgmscl
            }
            if acmod == 0 && br.read_bit()? {
                br.read(6)?; // pgmscl2
            }
            if br.read_bit()? {
                br.read(6)?; // extpgmscl
            }
            let mixdef = br.read(2)?;
            match mixdef {
                1 => {
                    br.read(5)?;
                }
                2 => {
                    br.read(12)?;
                }
                3 => {
                    let mixdeflen = br.read(5)? as usize;
                    br.skip(8 * (mixdeflen + 2))?;
                }
                _ => {}
            }
            if acmod < 2 {
                if br.read_bit()? {
                    br.read(14)?; // panmean, paninfo
                }
                if acmod == 0 && br.read_bit()? {
                    br.read(14)?;
                }
            }
            if br.read_bit()? {
                // frame mixing configuration
                if numblkscod == 0 {
                    br.read(5)?;
                } else {
                    for _ in 0..hdr.numblks {
                        if br.read_bit()? {
                            br.read(5)?;
                        }
                    }
                }
            }
        }
    }
    if br.read_bit()? {
        // informational metadata
        br.read(5)?; // bsmod, copyrightb, origbs
        if acmod == 2 {
            br.read(4)?; // dsurmod, dheadphonmod
        }
        if acmod >= 6 {
            br.read(2)?; // dsurexmod
        }
        if br.read_bit()? {
            br.read(8)?; // mixlevel, roomtyp, adconvtyp
        }
        if acmod == 0 && br.read_bit()? {
            br.read(8)?;
        }
        if fscod < 3 {
            br.read(1)?; // sourcefscod
        }
    }
    if strmtyp == 0 && numblkscod != 3 {
        br.read(1)?; // convsync
    }
    if strmtyp == 2 {
        let blkid = if numblkscod == 3 { true } else { br.read_bit()? };
        if blkid {
            br.read(6)?; // frmsizecod
        }
    }
    if br.read_bit()? {
        let addbsil = br.read(6)? as usize;
        br.skip((addbsil + 1) * 8)?;
    }
    Ok(())
}

/// CRC-16 with generator x¹⁶ + x¹⁵ + x² + 1 over the syncframe minus its
/// syncword (§7.10.1: "the sync word is not covered by either CRC"): for a
/// valid AC-3 or E-AC-3 frame the register reads zero after `crc2`.
pub fn frame_crc_ok(frame: &[u8]) -> bool {
    let mut crc: u16 = 0;
    for &b in frame.iter().skip(2) {
        crc ^= u16::from(b) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 { (crc << 1) ^ 0x8005 } else { crc << 1 };
        }
    }
    crc == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynrng_gain_follows_table_7_29() {
        assert!((dynrng_gain(0) - 1.0).abs() < 1e-6);
        // X = 011 (+3), Y = 00000 → 2^4 · 0.5 = 8.0 (+18.06 dB)… the table's
        // "+24.08 dB, 4 left" is for Y = 11111 ≈ 63/64.
        assert!((dynrng_gain(0b011_00000) - 8.0).abs() < 1e-6);
        assert!((dynrng_gain(0b011_11111) - 15.75).abs() < 1e-6);
        // X = 100 (−4): 2^-3 · 0.5 = 1/16 (−24.08 dB)
        assert!((dynrng_gain(0b100_00000) - 1.0 / 16.0).abs() < 1e-7);
        // X = 111 (−1), Y = 0 → 0 dB · 0.5 = 0.5 (−6.02 dB)
        assert!((dynrng_gain(0b111_00000) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn exponent_ungrouping_matches_the_spec_pseudo_code() {
        // Two D25 groups: values (2,3,1) → mapped 25*2+5*3+1 = 66, and (2,2,2) → 62.
        let bytes = [(66u8 << 1) | (62 >> 6), (62 << 2)];
        let mut br = BitReader::new(&bytes);
        let mut out = [0u8; 16];
        decode_exponents(&mut br, 2, 2, 5, &mut out).unwrap();
        // dexp: 0,+1,-1, 0,0,0 → abs: 5,6,5, 5,5,5, each doubled (D25)
        assert_eq!(&out[..13], &[5, 5, 5, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5]);
        // Running below 0 is a bitstream error, not a wrap.
        let bytes = [0u8, 0]; // group 0 → dexp −2,−2,−2
        let mut br = BitReader::new(&bytes);
        assert!(decode_exponents(&mut br, 1, 1, 3, &mut out).is_err());
    }

    #[test]
    fn output_order_is_ffmpeg_native() {
        assert_eq!(output_order(7, true), vec![0, 2, 1, LFE, 3, 4]); // L C R Ls Rs → FL FR FC LFE SL SR
        assert_eq!(output_order(2, false), vec![0, 1]);
        assert_eq!(output_order(4, true), vec![0, 1, LFE, 2]); // L R S → FL FR LFE BC
        assert_eq!(output_order(3, false), vec![0, 2, 1]);
        assert_eq!(output_order(1, true), vec![0, LFE]);
        assert_eq!(output_order(0, false), vec![0, 1]);
    }

    #[test]
    fn crc16_of_a_frame_with_appended_remainder_is_zero() {
        // Build a payload, compute its CRC by long division, append, verify.
        let payload = [0x0b, 0x77, 0x12, 0x34, 0x56, 0x78, 0x9a];
        let mut crc: u16 = 0;
        for &b in &payload[2..] {
            crc ^= u16::from(b) << 8;
            for _ in 0..8 {
                crc = if crc & 0x8000 != 0 { (crc << 1) ^ 0x8005 } else { crc << 1 };
            }
        }
        let mut frame = payload.to_vec();
        frame.extend_from_slice(&crc.to_be_bytes());
        assert!(frame_crc_ok(&frame));
        frame[3] ^= 1;
        assert!(!frame_crc_ok(&frame));
    }
}
