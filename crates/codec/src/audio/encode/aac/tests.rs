//! End-to-end tests of the AAC encoder.
//!
//! `RefDecoder` is a small AAC-LC decoder written for these tests from
//! ISO/IEC 13818-7 alone (the subset this encoder emits: SCE / CPE / LFE /
//! FIL / END, long and short windows, M/S; no TNS, pulse, intensity or
//! prediction, which it rejects). It re-parses every access unit bit by bit,
//! so a syntax slip in the encoder shows up as a parse failure or a garbled
//! signal here, and it lets CI measure SNR without any external decoder.
//! The ffmpeg tests at the bottom decode the same streams with an
//! independent implementation when `ffmpeg` is on PATH.

use std::collections::HashMap;
use std::f64::consts::PI;

use super::codebooks::{SCALEFACTOR, SPECTRUM};
use super::mdct::{self, Mdct};
use super::tables;
use super::*;
use crate::audio::AudioFrame;

// ---------------------------------------------------------------- reader

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn bit(&mut self) -> u32 {
        let byte = *self
            .data
            .get(self.pos / 8)
            .expect("read past the end of the access unit");
        let b = (byte >> (7 - self.pos % 8)) & 1;
        self.pos += 1;
        u32::from(b)
    }

    fn get(&mut self, n: u32) -> u32 {
        (0..n).fold(0, |v, _| (v << 1) | self.bit())
    }
}

/// Decode one codeword of `book` by growing a prefix until it matches.
fn huff(r: &mut BitReader, book: &HashMap<(u8, u32), usize>) -> usize {
    let mut code = 0u32;
    for len in 1..=19u8 {
        code = (code << 1) | r.bit();
        if let Some(&i) = book.get(&(len, code)) {
            return i;
        }
    }
    panic!("no codeword matches at bit {}", r.pos);
}

fn index_book(entries: &[(u8, u32)]) -> HashMap<(u8, u32), usize> {
    entries
        .iter()
        .enumerate()
        .map(|(i, &(l, c))| ((l, c), i))
        .collect()
}

// ---------------------------------------------------------------- decoder

#[derive(Clone)]
struct Ics {
    seq: u32,
    max_sfb: usize,
    group_len: Vec<usize>,
    /// Dequantized, de-interleaved spectrum: spec[window][line].
    spec: Vec<Vec<f32>>,
}

struct RefDecoder {
    tables: RateTables,
    sf_book: HashMap<(u8, u32), usize>,
    spec_books: Vec<HashMap<(u8, u32), usize>>,
    long: Vec<f32>,
    short: Vec<f32>,
    imdct_long: Mdct,
    imdct_short: Mdct,
    /// Overlap per output channel, in element order.
    overlap: Vec<Vec<f32>>,
    /// Window sequences seen, per element channel.
    pub seqs: Vec<Vec<u32>>,
}

const UNSIGNED: [bool; 12] = [
    false, false, false, true, true, false, false, true, true, true, true, true,
];
const DIM: [usize; 12] = [0, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2];
const LAV: [i32; 12] = [0, 1, 1, 2, 2, 4, 4, 7, 7, 12, 12, 16];

impl RefDecoder {
    fn new(rate: u32) -> Self {
        Self {
            tables: tables::rate_tables(rate).unwrap(),
            sf_book: index_book(&SCALEFACTOR),
            spec_books: (0..12).map(|cb| index_book(SPECTRUM[cb])).collect(),
            long: mdct::sine_window(2048),
            short: mdct::sine_window(256),
            imdct_long: Mdct::new(1024),
            imdct_short: Mdct::new(128),
            overlap: Vec::new(),
            seqs: Vec::new(),
        }
    }

    fn ics_info(&self, r: &mut BitReader) -> Ics {
        assert_eq!(r.get(1), 0, "ics_reserved_bit");
        let seq = r.get(2);
        assert_eq!(
            r.get(1),
            0,
            "window_shape: the encoder only uses sine windows"
        );
        let (max_sfb, group_len) = if seq == 2 {
            let max_sfb = r.get(4) as usize;
            let grouping = r.get(7);
            let mut g = vec![1usize];
            for i in 0..7 {
                if grouping & (1 << (6 - i)) != 0 {
                    *g.last_mut().unwrap() += 1;
                } else {
                    g.push(1);
                }
            }
            (max_sfb, g)
        } else {
            let max_sfb = r.get(6) as usize;
            assert_eq!(r.get(1), 0, "predictor_data_present is not LC");
            (max_sfb, vec![1])
        };
        let (swb, nwin) = if seq == 2 {
            (self.tables.swb_short, 8)
        } else {
            (self.tables.swb_long, 1)
        };
        assert!(
            max_sfb < swb.len(),
            "max_sfb {max_sfb} beyond the band table"
        );
        Ics {
            seq,
            max_sfb,
            group_len,
            spec: vec![vec![0.0; usize::from(*swb.last().unwrap())]; nwin],
        }
    }

    fn ics(&self, r: &mut BitReader, info: Option<&Ics>) -> Ics {
        let global_gain = r.get(8) as i32;
        let mut ics = match info {
            Some(i) => i.clone(),
            None => self.ics_info(r),
        };
        let short = ics.seq == 2;
        let swb = if short {
            self.tables.swb_short
        } else {
            self.tables.swb_long
        };
        let ngroups = ics.group_len.len();
        // section_data()
        let (len_bits, esc) = if short { (3, 7) } else { (5, 31) };
        let mut band_cb = vec![vec![0u8; ics.max_sfb]; ngroups];
        for cbs in band_cb.iter_mut() {
            let mut k = 0;
            while k < ics.max_sfb {
                let cb = r.get(4) as u8;
                assert!(cb <= 11, "codebook {cb} is not produced by this encoder");
                let mut len = 0;
                loop {
                    let incr = r.get(len_bits) as usize;
                    len += incr;
                    if incr != esc {
                        break;
                    }
                }
                assert!(k + len <= ics.max_sfb, "section runs past max_sfb");
                cbs[k..k + len].fill(cb);
                k += len;
            }
        }
        // scale_factor_data()
        let mut sf = vec![vec![0i32; ics.max_sfb]; ngroups];
        let mut last = global_gain;
        for g in 0..ngroups {
            for s in 0..ics.max_sfb {
                if band_cb[g][s] != 0 {
                    last += huff(r, &self.sf_book) as i32 - 60;
                    assert!((0..=255).contains(&last), "scalefactor {last} out of range");
                    sf[g][s] = last;
                }
            }
        }
        assert_eq!(r.get(1), 0, "pulse_data_present");
        assert_eq!(r.get(1), 0, "tns_data_present");
        assert_eq!(r.get(1), 0, "gain_control_data_present");
        // spectral_data(), de-interleaved on the fly.
        let mut w0 = 0;
        for g in 0..ngroups {
            let glen = ics.group_len[g];
            for s in 0..ics.max_sfb {
                let cb = usize::from(band_cb[g][s]);
                let width = usize::from(swb[s + 1] - swb[s]);
                let mut vals = vec![0i32; width * glen];
                if cb != 0 {
                    let mut k = 0;
                    while k < vals.len() {
                        let mut idx = huff(r, &self.spec_books[cb]);
                        let dim = DIM[cb];
                        let (modulus, off) = if UNSIGNED[cb] {
                            (LAV[cb] + 1, 0)
                        } else {
                            (2 * LAV[cb] + 1, LAV[cb])
                        };
                        let mut t = [0i32; 4];
                        for d in (0..dim).rev() {
                            t[d] = (idx % modulus as usize) as i32 - off;
                            idx /= modulus as usize;
                        }
                        if UNSIGNED[cb] {
                            for v in t.iter_mut().take(dim) {
                                if *v != 0 && r.bit() == 1 {
                                    *v = -*v;
                                }
                            }
                            if cb == 11 {
                                for v in t.iter_mut().take(dim) {
                                    if v.abs() == 16 {
                                        let mut n = 0;
                                        while r.bit() == 1 {
                                            n += 1;
                                        }
                                        let word = r.get(n + 4) as i32;
                                        let mag = (1 << (n + 4)) + word;
                                        assert!(mag <= 8191, "escape value {mag} above 8191");
                                        *v = v.signum() * mag;
                                    }
                                }
                            }
                        }
                        vals[k..k + dim].copy_from_slice(&t[..dim]);
                        k += dim;
                    }
                }
                let gain = 2f64.powf(0.25 * f64::from(sf[g][s] - 100));
                for win in 0..glen {
                    for bin in 0..width {
                        let q = vals[win * width + bin];
                        let x = f64::from(q.signum()) * f64::from(q.abs()).powf(4.0 / 3.0) * gain;
                        ics.spec[w0 + win][usize::from(swb[s]) + bin] = x as f32;
                    }
                }
            }
            w0 += glen;
        }
        ics
    }

    /// Decode one access unit to planar channels in element order.
    fn decode(&mut self, au: &[u8]) -> Vec<Vec<f32>> {
        let mut r = BitReader::new(au);
        let mut channels: Vec<Ics> = Vec::new();
        loop {
            let id = r.get(3);
            match id {
                0 | 3 => {
                    r.get(4);
                    channels.push(self.ics(&mut r, None));
                }
                1 => {
                    r.get(4);
                    assert_eq!(r.get(1), 1, "the encoder always sends a common window");
                    let info = self.ics_info(&mut r);
                    let nswb = if info.seq == 2 {
                        self.tables.swb_short.len()
                    } else {
                        self.tables.swb_long.len()
                    } - 1;
                    let ngroups = info.group_len.len();
                    let mode = r.get(2);
                    assert_ne!(mode, 3, "reserved ms_mask_present");
                    let mut ms = vec![vec![mode == 2; nswb]; ngroups];
                    if mode == 1 {
                        for row in ms.iter_mut() {
                            for v in row.iter_mut().take(info.max_sfb) {
                                *v = r.get(1) == 1;
                            }
                        }
                    }
                    let mut a = self.ics(&mut r, Some(&info));
                    let mut b = self.ics(&mut r, Some(&info));
                    let swb = if info.seq == 2 {
                        self.tables.swb_short
                    } else {
                        self.tables.swb_long
                    };
                    let mut w0 = 0;
                    for (g, &glen) in info.group_len.iter().enumerate() {
                        for win in w0..w0 + glen {
                            for s in 0..info.max_sfb {
                                if ms[g][s] {
                                    for k in usize::from(swb[s])..usize::from(swb[s + 1]) {
                                        let (m, sd) = (a.spec[win][k], b.spec[win][k]);
                                        a.spec[win][k] = m + sd;
                                        b.spec[win][k] = m - sd;
                                    }
                                }
                            }
                        }
                        w0 += glen;
                    }
                    channels.push(a);
                    channels.push(b);
                }
                6 => {
                    let mut cnt = r.get(4) as usize;
                    if cnt == 15 {
                        cnt += r.get(8) as usize - 1;
                    }
                    for _ in 0..cnt {
                        r.get(8);
                    }
                }
                7 => break,
                other => panic!("unexpected element id {other}"),
            }
        }
        // byte_alignment() and nothing after it.
        assert!(
            au.len() * 8 - r.pos < 8,
            "{} trailing bits after END",
            au.len() * 8 - r.pos
        );
        if self.overlap.is_empty() {
            self.overlap = vec![vec![0.0; 1024]; channels.len()];
            self.seqs = vec![Vec::new(); channels.len()];
        }
        let mut out = Vec::with_capacity(channels.len());
        for (c, ics) in channels.iter().enumerate() {
            self.seqs[c].push(ics.seq);
            let mut z = vec![0.0f32; 2048];
            if ics.seq == 2 {
                let mut y = vec![0.0f32; 256];
                for (j, s) in ics.spec.iter().enumerate() {
                    self.imdct_short.inverse(s, &mut y);
                    for n in 0..256 {
                        z[448 + 128 * j + n] += y[n] * self.short[n];
                    }
                }
            } else {
                let seq = match ics.seq {
                    0 => WindowSequence::OnlyLong,
                    1 => WindowSequence::LongStart,
                    _ => WindowSequence::LongStop,
                };
                let w = mdct::long_window(seq, &self.long, &self.short);
                let mut y = vec![0.0f32; 2048];
                self.imdct_long.inverse(&ics.spec[0], &mut y);
                for n in 0..2048 {
                    z[n] = y[n] * w[n];
                }
            }
            let pcm: Vec<f32> = (0..1024)
                .map(|n| (z[n] + self.overlap[c][n]) / 32768.0)
                .collect();
            self.overlap[c].copy_from_slice(&z[1024..]);
            out.push(pcm);
        }
        out
    }
}

// ---------------------------------------------------------------- helpers

/// Element order of each configuration, as native channel slots (the
/// reference decoder outputs channels in element order).
fn element_order(channels: u8) -> Vec<usize> {
    channel_elements(channels)
        .unwrap()
        .1
        .iter()
        .flat_map(|(kind, ch)| {
            if *kind == ElementKind::Cpe {
                ch.to_vec()
            } else {
                vec![ch[0]]
            }
        })
        .collect()
}

struct Encoded {
    aus: Vec<Vec<u8>>,
    /// Decoded, delay-compensated, planar in native channel order.
    decoded: Vec<Vec<f32>>,
    seqs: Vec<Vec<u32>>,
}

fn encode_and_decode(input: &[Vec<f32>], rate: u32, bitrate: u32, switching: bool) -> Encoded {
    let channels = input.len() as u8;
    let mut enc = AacEncoder::new(AacConfig {
        sample_rate: rate,
        channels,
        bitrate,
    })
    .unwrap();
    if !switching {
        enc.disable_block_switching();
    }
    let len = input[0].len();
    let mut aus = Vec::new();
    // Feed in uneven chunks to exercise the internal buffering.
    let mut at = 0;
    let mut chunk = 700;
    while at < len {
        let n = chunk.min(len - at);
        let mut samples = Vec::with_capacity(n * input.len());
        for i in at..at + n {
            for ch in input {
                samples.push(ch[i]);
            }
        }
        let frame = AudioFrame {
            samples,
            sample_rate: rate,
            channels,
            pts: 0,
        };
        aus.extend(enc.encode(&frame).unwrap().into_iter().map(|p| p.data));
        at += n;
        chunk = if chunk == 700 { 1500 } else { 700 };
    }
    aus.extend(enc.flush().unwrap().into_iter().map(|p| p.data));
    assert_eq!(aus.len(), (len + 1024).div_ceil(1024));

    let mut dec = RefDecoder::new(rate);
    let order = element_order(channels);
    let mut decoded = vec![Vec::new(); input.len()];
    for au in &aus {
        for (k, pcm) in dec.decode(au).into_iter().enumerate() {
            decoded[order[k]].extend(pcm);
        }
    }
    for ch in &mut decoded {
        ch.drain(..ENCODER_DELAY as usize);
        ch.truncate(len);
    }
    let mut seqs = vec![Vec::new(); input.len()];
    for (k, s) in dec.seqs.into_iter().enumerate() {
        seqs[order[k]] = s;
    }
    Encoded { aus, decoded, seqs }
}

/// SNR without the first two frames and the last one: the test signals
/// switch on out of digital silence at sample 0 and are cut off at the end,
/// transients of the test rather than of the signal, and the steady state is
/// what these figures are about.
fn steady_snr_db(reference: &[f32], decoded: &[f32]) -> f64 {
    let end = reference.len() - 1024;
    snr_db(&reference[2048..end], &decoded[2048..end])
}

fn snr_db(reference: &[f32], decoded: &[f32]) -> f64 {
    let (mut s, mut n) = (0.0f64, 0.0f64);
    for (&a, &b) in reference.iter().zip(decoded) {
        s += f64::from(a) * f64::from(a);
        n += f64::from(a - b) * f64::from(a - b);
    }
    10.0 * (s / n.max(1e-30)).log10()
}

/// Mean over 1024-sample segments of the per-segment SNR, each clamped to
/// [-10, 90] dB, skipping near-silent segments.
fn segmental_snr_db(reference: &[f32], decoded: &[f32]) -> f64 {
    let mut total = 0.0;
    let mut count = 0;
    for (a, b) in reference
        .as_chunks::<1024>()
        .0
        .iter()
        .zip(decoded.as_chunks::<1024>().0)
    {
        let e: f64 = a.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        if e < 1024.0 * 1e-8 {
            continue;
        }
        total += snr_db(a, b).clamp(-10.0, 90.0);
        count += 1;
    }
    total / f64::from(count.max(1))
}

fn sine(freq: f64, amp: f64, rate: u32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (amp * (2.0 * PI * freq * i as f64 / f64::from(rate)).sin()) as f32)
        .collect()
}

struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) as f32 / (1 << 24) as f32 * 2.0 - 1.0
    }
}

/// A music-like test signal: harmonic notes (eight partials, 1/h
/// amplitudes) with vibrato and decaying envelopes, a new note every quarter
/// second, over a low-passed noise bed of amplitude `noise`; `seed` varies
/// the note sequence and the noise per channel.
fn music_with(rate: u32, len: usize, seed: u32, noise: f32) -> Vec<f32> {
    let mut rng = Lcg(seed);
    let notes = [220.0, 277.18, 329.63, 440.0, 369.99, 293.66, 246.94, 392.0];
    let fs = f64::from(rate);
    let mut lp = 0.0f32;
    (0..len)
        .map(|i| {
            let t = i as f64 / fs;
            let step = (t * 4.0) as usize;
            let local = t * 4.0 - step as f64;
            let env = (-3.0 * local).exp();
            let f0 = notes[(step + seed as usize) % notes.len()]
                * (1.0 + 0.004 * (2.0 * PI * 5.0 * t).sin());
            let mut v = 0.0;
            for h in 1..=8 {
                v += (2.0 * PI * f0 * h as f64 * t).sin() / h as f64;
            }
            lp = 0.9 * lp + 0.1 * rng.next();
            (0.25 * env * v) as f32 + noise * lp
        })
        .collect()
}

fn music(rate: u32, len: usize, seed: u32) -> Vec<f32> {
    music_with(rate, len, seed, 0.02)
}

/// Castanet-like clicks: bursts of decaying noise after silence.
fn clicks(rate: u32, len: usize) -> (Vec<f32>, Vec<usize>) {
    let mut rng = Lcg(99);
    let mut x = vec![0.0f32; len];
    let period = rate as usize / 3;
    let mut onsets = Vec::new();
    let mut at = period / 2 + 317;
    while at + 4096 < len {
        onsets.push(at);
        for n in 0..3000 {
            x[at + n] = 0.8 * (-(n as f32) / 300.0).exp() * rng.next();
        }
        at += period;
    }
    (x, onsets)
}

fn bitrate_of(aus: &[Vec<u8>], rate: u32) -> f64 {
    let bytes: usize = aus.iter().map(|a| a.len()).sum();
    bytes as f64 * 8.0 * f64::from(rate) / (aus.len() as f64 * 1024.0)
}

// ---------------------------------------------------------------- tests

#[test]
fn inverse_transform_matches_the_standards_imdct() {
    let mut m = Mdct::new(128);
    let mut rng = Lcg(3);
    let spec: Vec<f32> = (0..128).map(|_| rng.next()).collect();
    let mut fast = vec![0.0f32; 256];
    m.inverse(&spec, &mut fast);
    for (n, &v) in fast.iter().enumerate() {
        let direct: f64 = 2.0 / 256.0
            * spec
                .iter()
                .enumerate()
                .map(|(k, &x)| {
                    f64::from(x) * (2.0 * PI / 256.0 * (n as f64 + 64.5) * (k as f64 + 0.5)).cos()
                })
                .sum::<f64>();
        assert!((f64::from(v) - direct).abs() < 1e-4, "{n}: {v} vs {direct}");
    }
}

#[test]
fn config_rejects_what_it_cannot_code() {
    let cfg = |sample_rate, channels, bitrate| AacConfig {
        sample_rate,
        channels,
        bitrate,
    };
    assert!(AacEncoder::new(cfg(96_000, 2, 0)).is_err());
    assert!(AacEncoder::new(cfg(48_000, 7, 0)).is_err());
    assert!(AacEncoder::new(cfg(48_000, 9, 0)).is_err());
    assert!(AacEncoder::new(cfg(48_000, 2, 1_000_000)).is_err());
    assert!(AacEncoder::new(cfg(48_000, 2, 4_000)).is_err());
    let e = AacEncoder::new(cfg(44_100, 6, 0)).unwrap();
    assert_eq!(e.channel_configuration(), 6);
    assert_eq!(e.extra_data(), vec![0x12, 0x30]);
    assert_eq!(
        AacEncoder::new(cfg(48_000, 8, 0))
            .unwrap()
            .channel_configuration(),
        7
    );
}

#[test]
fn stereo_sine_decodes_cleanly_at_every_rate() {
    for rate in tables::SUPPORTED_RATES {
        let len = rate as usize;
        let l = sine(997.0, 0.5, rate, len);
        let r = sine(1499.0, 0.3, rate, len);
        let out = encode_and_decode(&[l.clone(), r.clone()], rate, 128_000, true);
        let (sl, sr) = (
            steady_snr_db(&l, &out.decoded[0]),
            steady_snr_db(&r, &out.decoded[1]),
        );
        eprintln!("{rate} Hz stereo sines @128k: SNR L {sl:.1} dB, R {sr:.1} dB");
        assert!(sl > 50.0 && sr > 50.0, "{rate}: {sl} / {sr}");
    }
}

#[test]
fn music_like_signal_holds_its_bit_rate_and_quality() {
    let rate = 48_000;
    let len = rate as usize * 3;
    for (name, noise) in [("tonal", 0.0f32), ("with noise bed", 0.02)] {
        let input = [
            music_with(rate, len, 1, noise),
            music_with(rate, len, 2, noise),
        ];
        for bitrate in [
            32_000u32, 64_000, 96_000, 128_000, 192_000, 256_000, 320_000,
        ] {
            let out = encode_and_decode(&input, rate, bitrate, true);
            let actual = bitrate_of(&out.aus, rate);
            let snr = steady_snr_db(&input[0], &out.decoded[0]);
            let seg = segmental_snr_db(&input[0], &out.decoded[0]);
            eprintln!(
                "music ({name}) stereo @{bitrate}: {actual:.0} b/s, SNR {snr:.1} dB, segSNR {seg:.1} dB"
            );
            // Constant rate at the buffer level: the total never strays from
            // the nominal by more than the decoder buffer (6144 bits a channel).
            let seconds = out.aus.len() as f64 * 1024.0 / f64::from(rate);
            let slack = 2.0 * 6144.0 / seconds;
            assert!(
                (actual - f64::from(bitrate)).abs() <= slack,
                "{bitrate}: {actual}"
            );
            if noise == 0.0 && bitrate >= 128_000 {
                assert!(snr > 25.0 && seg > 35.0, "{bitrate}: {snr} / {seg}");
            }
        }
    }
}

#[test]
fn quality_rises_with_bit_rate() {
    let rate = 44_100;
    let len = rate as usize * 2;
    let input = [music(rate, len, 5)];
    let mut last = f64::NEG_INFINITY;
    for bitrate in [32_000u32, 64_000, 128_000] {
        let out = encode_and_decode(&input, rate, bitrate, true);
        let snr = steady_snr_db(&input[0], &out.decoded[0]);
        eprintln!("music mono @{bitrate}: SNR {snr:.1} dB");
        assert!(snr > last, "{bitrate}: {snr} <= {last}");
        last = snr;
    }
}

#[test]
fn silence_and_short_inputs_encode() {
    let out = encode_and_decode(&[vec![0.0; 100]], 48_000, 64_000, true);
    assert_eq!(out.aus.len(), 2);
    assert!(out.decoded[0].iter().all(|&v| v.abs() < 1e-6));
    let out = encode_and_decode(
        &[vec![0.0; 48_000], vec![0.0; 48_000]],
        48_000,
        128_000,
        true,
    );
    assert!(out.decoded[1].iter().all(|&v| v.abs() < 1e-6));
    // Garbage in (NaN, infinities) is zeroed rather than poisoning a frame.
    let mut bad = vec![0.0f32; 4096];
    bad[100] = f32::NAN;
    bad[2000] = f32::INFINITY;
    let out = encode_and_decode(&[bad], 48_000, 64_000, true);
    assert!(out.decoded[0].iter().all(|&v| v.abs() < 1e-6));
}

#[test]
fn transients_switch_to_short_windows_and_suppress_pre_echo() {
    let rate = 48_000;
    let len = rate as usize * 3;
    let (x, onsets) = clicks(rate, len);
    // Error energy (the input is silent there) in a span before each onset.
    let energy_before = |dec: &[f32], from: usize, to: usize| -> f64 {
        onsets
            .iter()
            .map(|&o| {
                dec[o - from..o - to]
                    .iter()
                    .map(|&v| f64::from(v) * f64::from(v))
                    .sum::<f64>()
            })
            .sum()
    };
    let with = encode_and_decode(std::slice::from_ref(&x), rate, 64_000, true);
    let without = encode_and_decode(std::slice::from_ref(&x), rate, 64_000, false);
    assert!(
        with.seqs[0].contains(&2),
        "no EIGHT_SHORT_SEQUENCE for a click train"
    );
    assert!(!without.seqs[0].contains(&2));
    // Legal transitions only: nothing long follows a START, and a STOP only
    // follows short blocks.
    for pair in with.seqs[0].windows(2) {
        match (pair[0], pair[1]) {
            (1, s) => assert_eq!(s, 2, "{pair:?}"),
            (2, s) => assert!(s == 2 || s == 3, "{pair:?}"),
            (0 | 3, s) => assert!(s == 0 || s == 1, "{pair:?}"),
            _ => unreachable!(),
        }
    }
    // A short window straddling the onset still smears its noise over its
    // own 256 samples, so the last short hop (128 samples, 2.7 ms: within
    // backward masking) is reported apart from the 21 ms before it, where a
    // long window spreads pre-echo and short windows should leave none.
    let db = |a: f64, b: f64| 10.0 * (a / b.max(1e-30)).log10();
    eprintln!(
        "clicks @64k mono, error energy before {} onsets:",
        onsets.len()
    );
    let mut gains = Vec::new();
    for (from, to, name) in [
        (1024, 128, "21.3..2.7 ms before"),
        (128, 0, "last 2.7 ms"),
        (480, 0, "last 10 ms"),
    ] {
        let (long, short) = (
            energy_before(&without.decoded[0], from, to),
            energy_before(&with.decoded[0], from, to),
        );
        eprintln!(
            "  {name}: long windows only {long:.2e}, block switching {short:.2e} ({:.1} dB less)",
            db(long, short)
        );
        gains.push(db(long, short));
    }
    assert!(
        gains[0] > 20.0,
        "block switching left pre-echo ahead of the onsets"
    );
    assert!(gains[1] >= 0.0, "short windows made the onset window worse");
}

#[test]
fn multichannel_layouts_keep_each_channel_in_its_slot() {
    let rate = 48_000;
    let len = rate as usize;
    for channels in [3u8, 4, 5, 6, 8] {
        // A distinct frequency per native slot; the LFE (slot 3 of 5.1/7.1)
        // gets one below 120 Hz.
        let has_lfe = channels >= 6;
        let input: Vec<Vec<f32>> = (0..channels as usize)
            .map(|c| {
                let f = if has_lfe && c == 3 {
                    60.0
                } else {
                    400.0 + 300.0 * c as f64
                };
                sine(f, 0.3, rate, len)
            })
            .collect();
        let out = encode_and_decode(&input, rate, 0, true);
        for (c, (inp, dec)) in input.iter().zip(&out.decoded).enumerate() {
            let snr = steady_snr_db(inp, dec);
            eprintln!("{channels} ch, slot {c}: SNR {snr:.1} dB");
            assert!(snr > 20.0, "{channels} channels, slot {c}: {snr}");
        }
    }
}

// ---------------------------------------------------------------- ffmpeg

/// Whether an `ffmpeg` binary is on PATH; the tests below are skipped
/// (with a note) when it is not, as on CI runners without it.
fn ffmpeg_available() -> bool {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Decode an ADTS stream with ffmpeg: returns interleaved f32 samples and
/// ffmpeg's stderr (with `-v error`, empty unless something was wrong).
fn ffmpeg_decode(adts: &[u8], tag: &str) -> (Vec<f32>, String) {
    let dir = std::env::temp_dir();
    let src = dir.join(format!("rivet-aac-{}-{tag}.aac", std::process::id()));
    std::fs::write(&src, adts).unwrap();
    let out = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-xerror", "-i"])
        .arg(&src)
        .args(["-f", "f32le", "-"])
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&src);
    let pcm = out
        .stdout
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    (pcm, String::from_utf8_lossy(&out.stderr).into_owned())
}

fn to_adts(enc: &AacEncoder, aus: &[Vec<u8>]) -> Vec<u8> {
    aus.iter()
        .flat_map(|au| adts_frame(enc.sampling_index(), enc.channel_configuration(), au))
        .collect()
}

#[test]
fn ffmpeg_decodes_every_rate_bit_rate_and_layout_without_errors() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH: skipping the external-decoder check");
        return;
    }
    let mut cases = Vec::new();
    for rate in tables::SUPPORTED_RATES {
        for (channels, bitrates) in [
            (1u8, vec![32_000u32, 64_000, 128_000]),
            (2, vec![32_000, 96_000, 128_000, 192_000, 320_000]),
        ] {
            for b in bitrates {
                if b <= bitrate_range(rate, channels).1 {
                    cases.push((rate, channels, b));
                }
            }
        }
    }
    for channels in [3u8, 4, 5, 6, 8] {
        cases.push((48_000, channels, 0));
        cases.push((44_100, channels, 0));
    }
    for (rate, channels, bitrate) in cases {
        let len = rate as usize * 2;
        let (clk, _) = clicks(rate, len);
        let lfe = if channels >= 6 { Some(3) } else { None };
        let input: Vec<Vec<f32>> = (0..channels as usize)
            .map(|c| {
                if Some(c) == lfe {
                    return sine(50.0, 0.4, rate, len);
                }
                let m = music(rate, len, c as u32 + 1);
                m.iter().zip(&clk).map(|(a, b)| a + 0.5 * b).collect()
            })
            .collect();
        let mut enc = AacEncoder::new(AacConfig {
            sample_rate: rate,
            channels,
            bitrate,
        })
        .unwrap();
        let mut samples = Vec::with_capacity(len * channels as usize);
        for i in 0..len {
            for ch in &input {
                samples.push(ch[i]);
            }
        }
        let frame = AudioFrame {
            samples,
            sample_rate: rate,
            channels,
            pts: 0,
        };
        let mut aus: Vec<Vec<u8>> = enc
            .encode(&frame)
            .unwrap()
            .into_iter()
            .map(|p| p.data)
            .collect();
        aus.extend(enc.flush().unwrap().into_iter().map(|p| p.data));
        let (pcm, err) = ffmpeg_decode(
            &to_adts(&enc, &aus),
            &format!("{rate}-{channels}-{bitrate}"),
        );
        assert!(
            err.trim().is_empty(),
            "{rate} Hz, {channels} ch, {bitrate} b/s: ffmpeg said: {err}"
        );
        assert_eq!(
            pcm.len(),
            aus.len() * 1024 * channels as usize,
            "{rate}/{channels}/{bitrate}"
        );
        // Match every decoded channel to the input slot it reproduces best;
        // each must reproduce one, and no two the same one.
        let n = channels as usize;
        let mut mapping = Vec::new();
        let mut worst = f64::INFINITY;
        for k in 0..n {
            let dec: Vec<f32> = pcm
                .iter()
                .skip(n * 1024 + k)
                .step_by(n)
                .copied()
                .take(len)
                .collect();
            let (slot, snr) = (0..n)
                .map(|s| (s, snr_db(&input[s], &dec)))
                .fold((0, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
            mapping.push(slot);
            worst = worst.min(snr);
        }
        eprintln!(
            "ffmpeg {rate} Hz {channels} ch @{bitrate}: {:.0} b/s, output->input slots {mapping:?}, worst SNR {worst:.1} dB",
            bitrate_of(&aus, rate)
        );
        assert!(worst > 1.0, "{rate}/{channels}/{bitrate}: {worst}");
        let mut sorted = mapping.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), n, "{rate}/{channels}/{bitrate}: {mapping:?}");
    }
}

/// The reference decoder above and ffmpeg's decoder are independent
/// readings of the standard; on the same streams they must produce the same
/// samples (to float rounding). This is what lets the CI-side SNR figures
/// stand for what a real decoder hears.
#[test]
fn reference_decoder_agrees_with_ffmpeg() {
    if !ffmpeg_available() {
        eprintln!("ffmpeg not on PATH: skipping the external-decoder check");
        return;
    }
    for (rate, channels, bitrate) in [
        (48_000u32, 2u8, 128_000u32),
        (44_100, 6, 0),
        (32_000, 1, 48_000),
    ] {
        let len = rate as usize * 2;
        let (clk, _) = clicks(rate, len);
        let input: Vec<Vec<f32>> = (0..channels as usize)
            .map(|c| {
                let m = music(rate, len, c as u32 + 3);
                m.iter().zip(&clk).map(|(a, b)| a + 0.5 * b).collect()
            })
            .collect();
        let mut enc = AacEncoder::new(AacConfig {
            sample_rate: rate,
            channels,
            bitrate,
        })
        .unwrap();
        let samples: Vec<f32> = (0..len)
            .flat_map(|i| input.iter().map(move |ch| ch[i]))
            .collect();
        let frame = AudioFrame {
            samples,
            sample_rate: rate,
            channels,
            pts: 0,
        };
        let mut aus: Vec<Vec<u8>> = enc
            .encode(&frame)
            .unwrap()
            .into_iter()
            .map(|p| p.data)
            .collect();
        aus.extend(enc.flush().unwrap().into_iter().map(|p| p.data));

        let (pcm, err) = ffmpeg_decode(&to_adts(&enc, &aus), &format!("agree-{rate}-{channels}"));
        assert!(err.trim().is_empty(), "ffmpeg said: {err}");
        let mut dec = RefDecoder::new(rate);
        let order = element_order(channels);
        let mut ours = vec![Vec::new(); channels as usize];
        for au in &aus {
            for (k, ch) in dec.decode(au).into_iter().enumerate() {
                ours[order[k]].extend(ch);
            }
        }
        let n = channels as usize;
        for (c, ch) in ours.iter().enumerate() {
            let theirs: Vec<f32> = pcm.iter().skip(c).step_by(n).copied().collect();
            assert_eq!(theirs.len(), ch.len());
            let agree = snr_db(&theirs, ch);
            eprintln!(
                "{rate} Hz {channels} ch, channel {c}: reference vs ffmpeg decode agree to {agree:.1} dB"
            );
            assert!(agree > 70.0, "{rate}/{channels}, channel {c}: {agree}");
        }
    }
}
