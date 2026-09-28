//! AAC-LC encoder, written in this crate from the standards (ISO/IEC 13818-7
//! and ISO/IEC 14496-3) and the published literature cited in each module;
//! see `docs/decisions.md` for the provenance. Output is raw access units
//! (one raw_data_block each) plus the AudioSpecificConfig for the MP4 `esds`;
//! [`adts_frame`] wraps an access unit for MPEG-TS.
//!
//! Scope
//! -----
//! - AAC-LC (audioObjectType 2), 1024-sample frames, sine windows with
//!   long/short block switching, M/S stereo per scalefactor band, and a
//!   constant-NMR rate loop with a bit reservoir (constant bit rate at the
//!   decoder-buffer level; the per-frame size varies).
//! - Coded at 22.05, 24, 32, 44.1 or 48 kHz; any other input rate is
//!   resampled on the way in ([`coding_rate`]), the resampler's delay
//!   trimmed so the output stays aligned with the input.
//! - Channel configurations 1–7 (ISO/IEC 13818-7 Table 42), from rivet's
//!   native (ffmpeg) channel order: mono, stereo, 3.0, 4.0, 5.0, 5.1, 7.1.
//!   Six and seven-channel layouts other than 5.1, and quad, have no
//!   configuration and are rejected; the caller remaps them.
//! - Not used: TNS, intensity stereo, PNS, the pulse tool. Each is optional
//!   for an encoder; leaving them out costs efficiency, never conformance.
//!
//! Timing
//! ------
//! Frame `k` transforms input samples `1024(k-1) .. 1024(k+1)`, so a decoder
//! outputs everything 1024 samples late: [`ENCODER_DELAY`] samples of
//! priming lead the stream and the muxer trims them (MP4 edit list). The
//! block-switching decision for frame `k` looks at the 1024 samples centred
//! on `1024k` — the span its eight short windows cover — and also needs
//! frame `k+1`'s, since a long frame before a short one must be a
//! LONG_START; so the encoder holds one frame of lookahead internally.
//! Packet timestamps count from the first input frame's PTS in steps of
//! 1024 samples, like the Opus encoder's: the priming is signalled apart
//! from the timestamps.

mod bits;
mod codebooks;
mod huffman;
mod mdct;
mod psy;
mod quant;
mod syntax;
mod tables;

#[cfg(test)]
mod tests;

pub use syntax::{adts_frame, adts_header, audio_specific_config};
pub use tables::SUPPORTED_RATES;

use crate::audio::resample::AudioResampler;
use crate::audio::{AudioEncoder, AudioError, AudioFrame, EncodedAudioPacket};

use bits::BitWriter;
use mdct::{Mdct, WindowSequence};
use psy::{AttackDetector, BandPsy, Zone};
use quant::{ChannelFrame, Layout, Quantized};
use tables::RateTables;

/// Samples per channel in one access unit.
pub const FRAME_SAMPLES: usize = 1024;
/// Priming samples at the start of the stream (one frame of MDCT overlap).
pub const ENCODER_DELAY: u32 = 1024;

/// Encoder settings.
#[derive(Clone, Debug)]
pub struct AacConfig {
    /// The input's sample rate; the stream is coded at [`coding_rate`] of it.
    pub sample_rate: u32,
    /// 1, 2, 3, 4, 5, 6 or 8, in rivet's native channel order.
    pub channels: u8,
    /// Target bit rate in bits per second for all channels together; 0
    /// picks [`default_bitrate`].
    pub bitrate: u32,
}

/// The rate a stream from `input_rate` is coded at: the input's own when
/// the encoder codes it natively ([`SUPPORTED_RATES`]), otherwise the
/// nearest of them in the input's family (multiples of 11.025 kHz stay in
/// the 44.1 kHz family, the rest go to the 48 kHz one), never below 22.05
/// kHz and never above 48 kHz.
pub fn coding_rate(input_rate: u32) -> u32 {
    if SUPPORTED_RATES.contains(&input_rate) {
        return input_rate;
    }
    let cd_family = input_rate.is_multiple_of(11_025);
    match (input_rate, cd_family) {
        (r, true) if r < 22_050 => 22_050,
        (_, true) => 44_100,
        (r, false) if r < 24_000 => 24_000,
        (r, false) if r <= 32_000 => 32_000,
        _ => 48_000,
    }
}

/// Default bit rate for a channel count: 64 kb/s mono, 128 kb/s stereo,
/// 384 kb/s 5.1, 512 kb/s 7.1, and 64 kb/s per extra main channel between.
pub fn default_bitrate(channels: u8) -> u32 {
    match channels {
        1 => 64_000,
        2 => 128_000,
        6 => 384_000,
        8 => 512_000,
        n => 64_000 * u32::from(n),
    }
}

/// The bit rates a stream of `channels` at `rate` Hz can have: at least
/// 8 kb/s per main channel, and at most what the decoder input buffer of
/// ISO/IEC 13818-7 8.2.2 allows a constant-rate stream (6144 bits per main
/// channel per frame: 288 kb/s a channel at 48 kHz, 144 kb/s at 24 kHz).
/// The LFE channel is not a main channel.
pub fn bitrate_range(rate: u32, channels: u8) -> (u32, u32) {
    let main = u64::from(channels) - u64::from(channels >= 6);
    (
        (8_000 * main) as u32,
        (6144 * main * u64::from(rate) / 1024) as u32,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ElementKind {
    Sce,
    Cpe,
    Lfe,
}

/// One syntactic element: which input channels it carries and its state.
struct Element {
    kind: ElementKind,
    tag: u8,
    /// Native channel slots (the second is unused for SCE/LFE).
    ch: [usize; 2],
    prev_seq: WindowSequence,
}

/// An element and the native channel slots it carries.
type ElementSlots = (ElementKind, [usize; 2]);

/// The element list of a channel configuration (Table 42), from the native
/// channel order: 3.0 = FL FR FC, 4.0 = FL FR FC BC, 5.0 = FL FR FC BL BR,
/// 5.1 = FL FR FC LFE BL BR, 7.1 = FL FR FC LFE BL BR SL SR.
///
/// Configuration 7 is "centre, front pair, a second front pair (the outer
/// one), surround pair, LFE"; rivet's 7.1 sends its side pair as that second
/// pair and its back pair as the surround pair, so both pairs keep their
/// front-to-back order.
fn channel_elements(channels: u8) -> Option<(u8, Vec<ElementSlots>)> {
    use ElementKind::*;
    let e = match channels {
        1 => (1, vec![(Sce, [0, 0])]),
        2 => (2, vec![(Cpe, [0, 1])]),
        3 => (3, vec![(Sce, [2, 0]), (Cpe, [0, 1])]),
        4 => (4, vec![(Sce, [2, 0]), (Cpe, [0, 1]), (Sce, [3, 0])]),
        5 => (5, vec![(Sce, [2, 0]), (Cpe, [0, 1]), (Cpe, [3, 4])]),
        6 => (
            6,
            vec![(Sce, [2, 0]), (Cpe, [0, 1]), (Cpe, [4, 5]), (Lfe, [3, 0])],
        ),
        8 => (
            7,
            vec![
                (Sce, [2, 0]),
                (Cpe, [0, 1]),
                (Cpe, [6, 7]),
                (Cpe, [4, 5]),
                (Lfe, [3, 0]),
            ],
        ),
        _ => return None,
    };
    Some(e)
}

/// Audio bandwidth for a bit rate per main channel: the classic trade of
/// bandwidth against coding noise at low rates.
fn bandwidth_hz(bits_per_channel: u32, rate: u32) -> f64 {
    const POINTS: [(f64, f64); 8] = [
        (16_000.0, 6_000.0),
        (24_000.0, 9_000.0),
        (32_000.0, 11_500.0),
        (48_000.0, 14_500.0),
        (64_000.0, 16_500.0),
        (80_000.0, 18_000.0),
        (96_000.0, 19_500.0),
        (128_000.0, 20_000.0),
    ];
    let b = f64::from(bits_per_channel);
    let hz = if b <= POINTS[0].0 {
        POINTS[0].1
    } else if b >= POINTS[POINTS.len() - 1].0 {
        POINTS[POINTS.len() - 1].1
    } else {
        let k = POINTS.windows(2).position(|p| b < p[1].0).unwrap();
        let (x0, y0) = POINTS[k];
        let (x1, y1) = POINTS[k + 1];
        y0 + (y1 - y0) * (b - x0) / (x1 - x0)
    };
    hz.min(f64::from(rate) / 2.0)
}

struct ChannelState {
    /// Queued input, 16-bit scale; `pcm[0]` is sample `1024(k-1)` of the
    /// next frame `k` to encode.
    pcm: Vec<f32>,
    detector: AttackDetector,
    /// Transient analysis of the current frame's zone, computed one frame
    /// ahead.
    zone: Option<Zone>,
    /// The last long block's raw thresholds, for pre-echo control.
    prev_nb: Option<Vec<f32>>,
    /// Coded lines in a long window (bandwidth, or the LFE's 12).
    max_line: usize,
}

/// Bit reservoir and frame budget (ISO/IEC 13818-7 8.2.2, Annex C.7.3).
struct RateControl {
    /// bitrate * 1024 and the sample rate: frame `k` has
    /// floor((k+1) * num / den) - floor(k * num / den) mean bits.
    num: u64,
    den: u64,
    frame: u64,
    /// Bits saved in the reservoir.
    reservoir: i64,
    max_reservoir: i64,
    /// Running mean of the log of the frames' perceptual entropy.
    log_pe_avg: f64,
    /// The previous frame's noise-to-mask offset in dB, the next search's
    /// starting point.
    lambda: f32,
}

impl RateControl {
    fn mean_bits(&self) -> i64 {
        ((self.frame + 1) * self.num / self.den - self.frame * self.num / self.den) as i64
    }
}

pub struct AacEncoder {
    tables: RateTables,
    channels: u8,
    channel_configuration: u8,
    elements: Vec<Element>,
    chans: Vec<ChannelState>,
    long_windows: [Vec<f32>; 3],
    short_window: Vec<f32>,
    mdct_long: Mdct,
    mdct_short: Mdct,
    psy_long: BandPsy,
    psy_short: BandPsy,
    rc: RateControl,
    asc: [u8; 2],
    /// The input's sample rate, and the resampler to the coding rate when
    /// they differ, with the output frames of its delay still to drop.
    in_rate: u32,
    resampler: Option<AudioResampler>,
    resample_skip: usize,
    resampled: Vec<f32>,
    /// Input sample frames received, per channel, at the input rate.
    samples_in: u64,
    frames_out: u64,
    first_pts: Option<i64>,
    /// Test knob: never switch to short blocks.
    block_switching: bool,
}

impl AacEncoder {
    pub fn new(config: AacConfig) -> Result<Self, AudioError> {
        if config.sample_rate == 0 {
            return Err(AudioError::Encode("input sample_rate is 0".to_string()));
        }
        let rate = coding_rate(config.sample_rate);
        let tables = tables::rate_tables(rate).expect("coding_rate picks a coded rate");
        let (channel_configuration, layout) =
            channel_elements(config.channels).ok_or_else(|| {
                AudioError::Unsupported(format!(
                    "AAC encoder has no channel configuration for {} channels \
                 (1, 2, 3, 4, 5, 6 and 8 are supported: mono, stereo, 3.0, 4.0, 5.0, 5.1, 7.1)",
                    config.channels
                ))
            })?;
        let main_channels = u32::from(config.channels) - u32::from(config.channels >= 6);
        let bitrate = if config.bitrate == 0 {
            default_bitrate(config.channels)
        } else {
            config.bitrate
        };
        let (min_bitrate, max_bitrate) = bitrate_range(rate, config.channels);
        if bitrate < min_bitrate || bitrate > max_bitrate {
            return Err(AudioError::Unsupported(format!(
                "AAC bit rate {bitrate} b/s for {} channels at {rate} Hz (allowed {min_bitrate}..={max_bitrate})",
                config.channels
            )));
        }

        let mut tags = [0u8; 4];
        let elements = layout
            .into_iter()
            .map(|(kind, ch)| {
                let tag = &mut tags[kind as usize];
                let e = Element {
                    kind,
                    tag: *tag,
                    ch,
                    prev_seq: WindowSequence::OnlyLong,
                };
                *tag += 1;
                e
            })
            .collect::<Vec<_>>();

        let cutoff = bandwidth_hz(bitrate / main_channels, rate);
        let max_line = ((cutoff / (f64::from(rate) / 2.0) * 1024.0).ceil() as usize).min(1024);
        let mut chans: Vec<ChannelState> = (0..config.channels)
            .map(|_| ChannelState {
                pcm: vec![0.0; FRAME_SAMPLES],
                detector: AttackDetector::default(),
                zone: None,
                prev_nb: None,
                max_line,
            })
            .collect();
        for e in &elements {
            if e.kind == ElementKind::Lfe {
                chans[e.ch[0]].max_line = 12;
            }
        }

        let long = mdct::sine_window(2048);
        let short = mdct::sine_window(256);
        let long_windows = [
            mdct::long_window(WindowSequence::OnlyLong, &long, &short),
            mdct::long_window(WindowSequence::LongStart, &long, &short),
            mdct::long_window(WindowSequence::LongStop, &long, &short),
        ];
        let (resampler, resample_skip) = if rate == config.sample_rate {
            (None, 0)
        } else {
            (
                Some(AudioResampler::new(
                    config.sample_rate,
                    rate,
                    config.channels,
                    1024,
                )?),
                resampler_delay(config.sample_rate, rate)?,
            )
        };
        let mean = u64::from(bitrate) * 1024 / u64::from(rate);
        let buffer = 6144 * i64::from(main_channels);
        Ok(Self {
            psy_long: BandPsy::new(tables.swb_long, 1024, tables.rate),
            psy_short: BandPsy::new(tables.swb_short, 128, tables.rate),
            asc: audio_specific_config(tables.index, channel_configuration),
            tables,
            channels: config.channels,
            channel_configuration,
            elements,
            chans,
            long_windows,
            short_window: short,
            mdct_long: Mdct::new(1024),
            mdct_short: Mdct::new(128),
            rc: RateControl {
                num: u64::from(bitrate) * 1024,
                den: u64::from(rate),
                frame: 0,
                // A constant-rate decoder fills its input buffer before it
                // starts (13818-7 8.2.3), so the reservoir starts full.
                reservoir: buffer - mean as i64,
                max_reservoir: buffer - mean as i64,
                log_pe_avg: 0.0,
                lambda: 0.0,
            },
            in_rate: config.sample_rate,
            resample_skip,
            resampler,
            resampled: Vec::new(),
            samples_in: 0,
            frames_out: 0,
            first_pts: None,
            block_switching: true,
        })
    }

    /// The AudioSpecificConfig (ISO/IEC 14496-3 1.6.2.1) for the MP4 `esds`.
    pub fn audio_specific_config(&self) -> [u8; 2] {
        self.asc
    }

    /// The rate the stream is coded at (the timescale of its packets).
    pub fn coding_rate(&self) -> u32 {
        self.tables.rate
    }

    /// sampling_frequency_index, for an ADTS header.
    pub fn sampling_index(&self) -> u8 {
        self.tables.index
    }

    /// The channel configuration signalled in the ASC / ADTS header.
    pub fn channel_configuration(&self) -> u8 {
        self.channel_configuration
    }

    #[cfg(test)]
    fn disable_block_switching(&mut self) {
        self.block_switching = false;
    }

    /// Queue interleaved input (at the coding rate) per channel, in the
    /// 16-bit scale the decoder's output is defined in. A NaN or a wild
    /// value would poison every band it touches, so they are zeroed /
    /// clamped (8x full scale still codes: the scalefactor range covers it).
    fn push(&mut self, samples: &[f32]) {
        let n = usize::from(self.channels);
        for (c, ch) in self.chans.iter_mut().enumerate() {
            ch.pcm.extend(samples.iter().skip(c).step_by(n).map(|&s| {
                if s.is_finite() {
                    s.clamp(-8.0, 8.0) * 32768.0
                } else {
                    0.0
                }
            }));
        }
    }

    /// [`Self::push`] for the resampler's output, less its delay.
    fn push_resampled(&mut self, samples: &[f32]) {
        let n = usize::from(self.channels);
        let skip = self.resample_skip.min(samples.len() / n);
        self.resample_skip -= skip;
        self.push(&samples[skip * n..]);
    }

    fn encode_ready(&mut self) -> Vec<EncodedAudioPacket> {
        let mut out = Vec::new();
        while self.chans[0].pcm.len() >= 3 * FRAME_SAMPLES {
            out.push(self.encode_frame());
        }
        out
    }

    fn encode_frame(&mut self) -> EncodedAudioPacket {
        // Transients: this frame's zone (computed a frame ago, except at the
        // very start) and the next frame's.
        let mut next_zones = Vec::with_capacity(self.chans.len());
        for ch in &mut self.chans {
            if ch.zone.is_none() {
                ch.zone = Some(ch.detector.analyse(&ch.pcm[512..1536]));
            }
            next_zones.push(ch.detector.analyse(&ch.pcm[1536..2560]));
        }

        // Window sequence per element.
        let mut layouts = Vec::with_capacity(self.elements.len());
        for el in &mut self.elements {
            let chs: &[usize] = if el.kind == ElementKind::Cpe {
                &el.ch
            } else {
                &el.ch[..1]
            };
            let switching = self.block_switching && el.kind != ElementKind::Lfe;
            let now = switching
                && chs
                    .iter()
                    .any(|&c| self.chans[c].zone.unwrap().attack.is_some());
            let next = switching && chs.iter().any(|&c| next_zones[c].attack.is_some());
            let prev_short = el.prev_seq.is_short();
            let seq = if now || (next && prev_short) {
                // No window turns from short to short within one long
                // frame, so a long frame between two short ones is short too.
                WindowSequence::EightShort
            } else if next {
                WindowSequence::LongStart
            } else if prev_short {
                WindowSequence::LongStop
            } else {
                WindowSequence::OnlyLong
            };
            el.prev_seq = seq;
            let layout = if seq.is_short() {
                let mut energy = [0.0f32; 8];
                let mut attack: Option<usize> = None;
                for &c in chs {
                    let z = self.chans[c].zone.unwrap();
                    for (e, ze) in energy.iter_mut().zip(z.energy) {
                        *e += ze;
                    }
                    attack = match (attack, z.attack) {
                        (Some(a), Some(b)) => Some(a.min(b)),
                        (a, b) => a.or(b),
                    };
                }
                Layout {
                    seq,
                    group_len: psy::group_short_windows(&energy, attack),
                    swb: self.tables.swb_short,
                }
            } else {
                Layout {
                    seq,
                    group_len: vec![1],
                    swb: self.tables.swb_long,
                }
            };
            layouts.push(layout);
        }

        // Analysis: transform + masking thresholds, per channel.
        let mut frames: Vec<Option<ChannelFrame>> = (0..self.chans.len()).map(|_| None).collect();
        let assignments: Vec<(usize, usize)> = self
            .elements
            .iter()
            .enumerate()
            .flat_map(|(e, el)| {
                let n = if el.kind == ElementKind::Cpe { 2 } else { 1 };
                el.ch[..n].iter().map(move |&c| (e, c)).collect::<Vec<_>>()
            })
            .collect();
        for (e, c) in assignments {
            frames[c] = Some(self.analyse(c, layouts[e].clone()));
        }
        let mut frames: Vec<ChannelFrame> = frames.into_iter().map(|f| f.unwrap()).collect();

        // M/S per band for each channel pair.
        let mut ms: Vec<Vec<bool>> = vec![Vec::new(); self.elements.len()];
        for (e, el) in self.elements.iter().enumerate() {
            if el.kind == ElementKind::Cpe {
                let [a, b] = el.ch;
                let (fa, fb) = two_mut(&mut frames, a, b);
                ms[e] = decide_ms(fa, fb);
            }
        }

        // Rate control.
        let mean = self.rc.mean_bits();
        // A frame's share follows its perceptual entropy relative to the
        // recent (geometric) mean, square-root compressed so one hard frame
        // cannot starve the ones after it.
        let log_pe = (frames.iter().map(|f| f64::from(f.pe)).sum::<f64>() + 1.0).ln();
        if self.rc.frame == 0 {
            self.rc.log_pe_avg = log_pe;
        }
        let demand = (0.5 * (log_pe - self.rc.log_pe_avg)).exp().clamp(0.6, 2.0);
        let target =
            mean as f64 * demand + 0.3 * (self.rc.reservoir - self.rc.max_reservoir / 2) as f64;
        // Whatever the reservoir cannot hold would only turn into fill bits,
        // so spend it; and never more than it holds.
        let floor = (mean + self.rc.reservoir - self.rc.max_reservoir).max(mean / 2);
        let budget = (target as i64).clamp(floor, mean + self.rc.reservoir.max(0));
        self.rc.log_pe_avg = 0.9 * self.rc.log_pe_avg + 0.1 * log_pe;
        // Room for the END element and byte alignment.
        let element_budget = (budget - 3 - 7).max(0) as usize;
        let coded = self.search(&frames, &ms, element_budget);

        // Bitstream.
        let mut w = BitWriter::with_capacity(budget as usize / 8 + 64);
        for (e, el) in self.elements.iter().enumerate() {
            match el.kind {
                ElementKind::Sce | ElementKind::Lfe => {
                    let c = el.ch[0];
                    w.put(
                        if el.kind == ElementKind::Sce {
                            syntax::ID_SCE
                        } else {
                            syntax::ID_LFE
                        },
                        3,
                    );
                    w.put(u32::from(el.tag), 4);
                    syntax::write_ics(&mut w, &frames[c], &coded[c], false);
                }
                ElementKind::Cpe => {
                    let [a, b] = el.ch;
                    let layout = &frames[a].layout;
                    let max_sfb = coded[a].max_sfb;
                    w.put(syntax::ID_CPE, 3);
                    w.put(u32::from(el.tag), 4);
                    w.put(1, 1); // common_window
                    syntax::write_ics_info(&mut w, layout, max_sfb);
                    let mode = syntax::ms_mask_mode(&ms[e], layout, max_sfb);
                    syntax::write_ms_mask(&mut w, mode, &ms[e], layout, max_sfb);
                    syntax::write_ics(&mut w, &frames[a], &coded[a], true);
                    syntax::write_ics(&mut w, &frames[b], &coded[b], true);
                }
            }
        }
        // Pad with fill elements when the reservoir would overflow: a
        // constant-rate decoder buffer cannot hold the surplus.
        let aligned = |bits: usize| (bits + 3).div_ceil(8) * 8;
        let used = aligned(w.len_bits()) as i64;
        let surplus = self.rc.reservoir + mean - used - self.rc.max_reservoir;
        if surplus > 0 {
            let target = used + surplus;
            let mut fill = Vec::new();
            let mut bits = w.len_bits();
            while (aligned(bits) as i64) < target {
                let need = (target - aligned(bits) as i64) as usize;
                let payload = need
                    .saturating_sub(7)
                    .div_ceil(8)
                    .clamp(1, syntax::MAX_FILL_PAYLOAD);
                fill.push(payload);
                bits += syntax::fill_element_bits(payload);
            }
            for payload in fill {
                syntax::write_fill_element(&mut w, payload);
            }
        }
        w.put(syntax::ID_END, 3);
        w.align();
        let data = w.into_bytes();
        self.rc.reservoir += mean - (data.len() * 8) as i64;
        self.rc.frame += 1;

        for ch in &mut self.chans {
            ch.pcm.drain(..FRAME_SAMPLES);
        }
        for (ch, z) in self.chans.iter_mut().zip(next_zones) {
            ch.zone = Some(z);
        }
        let first = self.first_pts.unwrap_or(0);
        let pts = first
            + (self.frames_out * FRAME_SAMPLES as u64 * 1_000_000 / u64::from(self.tables.rate))
                as i64;
        self.frames_out += 1;
        EncodedAudioPacket {
            data,
            pts,
            duration: FRAME_SAMPLES as i64,
        }
    }

    /// Transform one channel and compute its thresholds for `layout`.
    fn analyse(&mut self, c: usize, layout: Layout) -> ChannelFrame {
        let ch = &mut self.chans[c];
        let mut coef = vec![0.0f32; 1024];
        if layout.short() {
            let swb = self.tables.swb_short;
            let nswb = swb.len() - 1;
            let mut spec = vec![[0.0f32; 128]; 8];
            let mut thr_w = Vec::with_capacity(8);
            let mut z = vec![0.0f32; 256];
            for (j, s) in spec.iter_mut().enumerate() {
                let at = 448 + 128 * j;
                for (n, zv) in z.iter_mut().enumerate() {
                    *zv = ch.pcm[at + n] * self.short_window[n];
                }
                self.mdct_short.forward(&z, s);
                thr_w.push(self.psy_short.analyse(swb, s, None).thr);
            }
            ch.prev_nb = None;
            let ngroups = layout.num_groups();
            let mut thr = vec![0.0f32; ngroups * nswb];
            let mut floor = vec![0.0f32; ngroups * nswb];
            let mut w0 = 0;
            for g in 0..ngroups {
                let len = layout.group_len[g];
                for sfb in 0..nswb {
                    let r = layout.band(g, sfb);
                    let (lo, hi) = (usize::from(swb[sfb]), usize::from(swb[sfb + 1]));
                    let width = hi - lo;
                    let mut min_thr = f32::INFINITY;
                    for i in 0..len {
                        coef[r.start + i * width..r.start + (i + 1) * width]
                            .copy_from_slice(&spec[w0 + i][lo..hi]);
                        min_thr = min_thr.min(thr_w[w0 + i][sfb]);
                    }
                    thr[g * nswb + sfb] = min_thr * len as f32;
                    floor[g * nswb + sfb] = self.psy_short.ath()[sfb] * len as f32;
                }
                w0 += len;
            }
            let limit_line = ch.max_line.div_ceil(8);
            let band_limit = swb
                .iter()
                .take(nswb)
                .filter(|&&o| usize::from(o) < limit_line)
                .count();
            ChannelFrame::new(
                layout,
                coef,
                vec![0.0; ngroups * nswb],
                thr,
                floor,
                band_limit,
            )
        } else {
            let swb = self.tables.swb_long;
            let nswb = swb.len() - 1;
            let win = match layout.seq {
                WindowSequence::OnlyLong => &self.long_windows[0],
                WindowSequence::LongStart => &self.long_windows[1],
                _ => &self.long_windows[2],
            };
            let z: Vec<f32> = ch.pcm[..2048].iter().zip(win).map(|(x, w)| x * w).collect();
            self.mdct_long.forward(&z, &mut coef);
            let m = self.psy_long.analyse(swb, &coef, ch.prev_nb.as_deref());
            ch.prev_nb = Some(m.nb);
            let band_limit = if ch.max_line <= 12 {
                // LFE: only the lowest twelve lines may be non-zero (8.4.1).
                swb.windows(2)
                    .take_while(|b| usize::from(b[1]) <= 12)
                    .count()
            } else {
                swb.iter()
                    .take(nswb)
                    .filter(|&&o| usize::from(o) < ch.max_line)
                    .count()
            };
            ChannelFrame::new(
                layout,
                coef,
                m.energy,
                m.thr,
                self.psy_long.ath().to_vec(),
                band_limit,
            )
        }
    }

    /// Find the smallest common noise-to-mask offset (in dB) whose frame
    /// fits `budget` bits of elements, by bracketing from the previous
    /// frame's offset and bisecting.
    fn search(
        &mut self,
        frames: &[ChannelFrame],
        ms: &[Vec<bool>],
        budget: usize,
    ) -> Vec<Quantized> {
        const LAMBDA_MIN: f32 = -60.0;
        const LAMBDA_MAX: f32 = 90.0;
        let mut fit: Option<(f32, Vec<Quantized>)> = None;
        let mut over: Option<f32> = None;
        let mut lambda = self.rc.lambda.clamp(LAMBDA_MIN, LAMBDA_MAX);
        let mut step = 3.0f32;
        for _ in 0..16 {
            let (coded, bits) = self.code_elements(frames, ms, lambda);
            if bits <= budget {
                fit = Some((lambda, coded));
            } else {
                over = Some(lambda);
            }
            lambda = match (&fit, over) {
                (Some((f, _)), Some(o)) => {
                    if f - o < 0.25 {
                        break;
                    }
                    (f + o) / 2.0
                }
                (Some((f, _)), None) => {
                    if *f <= LAMBDA_MIN {
                        break;
                    }
                    let l = (f - step).max(LAMBDA_MIN);
                    step *= 2.0;
                    l
                }
                (None, _) => {
                    if lambda >= LAMBDA_MAX {
                        break;
                    }
                    let l = (lambda + step).min(LAMBDA_MAX);
                    step *= 2.0;
                    l
                }
            };
        }
        match fit {
            Some((l, coded)) => {
                self.rc.lambda = l;
                coded
            }
            // Even at the largest offset the frame is over budget (a bit rate
            // this low is refused at construction, so this is a corner case):
            // send the emptiest frame there is.
            None => self.code_elements(frames, ms, f32::INFINITY).0,
        }
    }

    /// Quantize every channel at noise offset `lambda` dB; returns the
    /// per-channel results and the element bits (END excluded).
    fn code_elements(
        &self,
        frames: &[ChannelFrame],
        ms: &[Vec<bool>],
        lambda: f32,
    ) -> (Vec<Quantized>, usize) {
        let scale = if lambda.is_finite() {
            10f32.powf(lambda / 10.0)
        } else {
            f32::INFINITY
        };
        let mut coded: Vec<Quantized> = vec![Quantized::default(); frames.len()];
        let mut bits = 0usize;
        for (e, el) in self.elements.iter().enumerate() {
            match el.kind {
                ElementKind::Sce | ElementKind::Lfe => {
                    let c = el.ch[0];
                    let mut q = Quantized::quantize(&frames[c], scale);
                    let max_sfb = q.max_sfb;
                    q.finish(&frames[c], max_sfb);
                    bits += 7 + syntax::ics_info_bits(&frames[c].layout) + q.body_bits;
                    coded[c] = q;
                }
                ElementKind::Cpe => {
                    let [a, b] = el.ch;
                    let mut qa = Quantized::quantize(&frames[a], scale);
                    let mut qb = Quantized::quantize(&frames[b], scale);
                    let max_sfb = qa.max_sfb.max(qb.max_sfb);
                    qa.finish(&frames[a], max_sfb);
                    qb.finish(&frames[b], max_sfb);
                    let layout = &frames[a].layout;
                    let mode = syntax::ms_mask_mode(&ms[e], layout, max_sfb);
                    bits += 7
                        + 1
                        + syntax::ics_info_bits(layout)
                        + syntax::ms_mask_bits(mode, layout, max_sfb);
                    bits += qa.body_bits + qb.body_bits;
                    coded[a] = qa;
                    coded[b] = qb;
                }
            }
        }
        (coded, bits)
    }
}

/// The delay, in output samples, of the resampler from `in_rate` to
/// `out_rate`: where an impulse at the first input sample comes out. Measured
/// rather than computed, so it holds whatever the filter's design.
fn resampler_delay(in_rate: u32, out_rate: u32) -> Result<usize, AudioError> {
    let mut r = AudioResampler::new(in_rate, out_rate, 1, 1024)?;
    let mut impulse = vec![0.0f32; 1024];
    impulse[0] = 1.0;
    let mut out = Vec::new();
    for samples in [impulse, vec![0.0; 1024]] {
        let frame = AudioFrame {
            samples,
            sample_rate: in_rate,
            channels: 1,
            pts: 0,
        };
        r.process(&frame, &mut out)?;
    }
    Ok(out
        .iter()
        .enumerate()
        .fold((0, 0.0f32), |best, (i, &v)| {
            if v.abs() > best.1 { (i, v.abs()) } else { best }
        })
        .0)
}

fn two_mut<T>(v: &mut [T], a: usize, b: usize) -> (&mut T, &mut T) {
    assert!(a < b);
    let (lo, hi) = v.split_at_mut(b);
    (&mut lo[a], &mut hi[0])
}

/// Per-band M/S decision for a channel pair sharing one window layout
/// (after Johnston & Ferreira, "Sum-difference stereo transform coding",
/// ICASSP 1992, and Annex C.6.1): code a band as M = (L+R)/2, S = (L-R)/2
/// when that needs fewer bits by the perceptual-entropy estimate.
///
/// The decoder rebuilds L = M + S and R = M - S, so noise in M and S adds up
/// in both outputs. Giving M and S each half of min(thr_L, thr_R) keeps the
/// reconstructed L and R within their own thresholds — M/S is never allowed
/// to buy bits with audible noise.
fn decide_ms(a: &mut ChannelFrame, b: &mut ChannelFrame) -> Vec<bool> {
    let layout = a.layout.clone();
    let nswb = layout.num_swb();
    let n = layout.num_groups() * nswb;
    let mut used = vec![false; n];
    let bits = |e: f32, t: f32| if e > t { (e / t).log2() } else { 0.0 };
    for g in 0..layout.num_groups() {
        for sfb in 0..nswb.min(a.band_limit) {
            let i = g * nswb + sfb;
            let r = layout.band(g, sfb);
            let (mut em, mut es) = (0.0f32, 0.0f32);
            for k in r.clone() {
                let m = 0.5 * (a.coef[k] + b.coef[k]);
                let s = 0.5 * (a.coef[k] - b.coef[k]);
                em += m * m;
                es += s * s;
            }
            let (ta, tb) = (a.thr[i], b.thr[i]);
            let tms = 0.5 * ta.min(tb);
            let lr = bits(a.energy[i], ta) + bits(b.energy[i], tb);
            let msb = bits(em, tms) + bits(es, tms);
            if msb < lr {
                used[i] = true;
                for k in r {
                    let (l, rr) = (a.coef[k], b.coef[k]);
                    a.coef[k] = 0.5 * (l + rr);
                    b.coef[k] = 0.5 * (l - rr);
                }
                a.thr[i] = tms;
                b.thr[i] = tms;
                let floor = 0.5 * a.floor[i].min(b.floor[i]);
                a.floor[i] = floor;
                b.floor[i] = floor;
            }
        }
    }
    a.refresh();
    b.refresh();
    used
}

impl AudioEncoder for AacEncoder {
    fn encode(&mut self, frame: &AudioFrame) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        if frame.channels != self.channels {
            return Err(AudioError::Encode(format!(
                "channel count mismatch: encoder configured for {}, frame has {}",
                self.channels, frame.channels
            )));
        }
        if frame.sample_rate != self.in_rate {
            return Err(AudioError::Encode(format!(
                "sample rate mismatch: encoder configured for {}, frame has {}",
                self.in_rate, frame.sample_rate
            )));
        }
        if self.first_pts.is_none() {
            self.first_pts = Some(frame.pts);
        }
        self.samples_in += (frame.samples.len() / usize::from(self.channels)) as u64;
        match self.resampler.as_mut() {
            None => self.push(&frame.samples),
            Some(r) => {
                let mut out = std::mem::take(&mut self.resampled);
                out.clear();
                r.process(frame, &mut out)?;
                self.push_resampled(&out);
                self.resampled = out;
            }
        }
        Ok(self.encode_ready())
    }

    fn flush(&mut self) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        if let Some(mut r) = self.resampler.take() {
            // Silence behind the input pushes the filter's delayed tail out.
            let tail = AudioFrame {
                samples: vec![0.0; r.chunk_size() * usize::from(self.channels)],
                sample_rate: self.in_rate,
                channels: self.channels,
                pts: 0,
            };
            let mut out = Vec::new();
            r.process(&tail, &mut out)?;
            r.flush(&mut out)?;
            self.push_resampled(&out);
        }
        // Enough frames that the decoder's output covers the priming plus
        // every input sample (counted at the coding rate).
        let coded = (u128::from(self.samples_in) * u128::from(self.tables.rate))
            .div_ceil(u128::from(self.in_rate)) as u64;
        let needed = if coded == 0 {
            0
        } else {
            (coded + u64::from(ENCODER_DELAY)).div_ceil(FRAME_SAMPLES as u64)
        };
        let remaining = needed.saturating_sub(self.frames_out) as usize;
        if remaining == 0 {
            return Ok(Vec::new());
        }
        let want = 3 * FRAME_SAMPLES + (remaining - 1) * FRAME_SAMPLES;
        for ch in &mut self.chans {
            if ch.pcm.len() < want {
                ch.pcm.resize(want, 0.0);
            }
        }
        let mut out = Vec::with_capacity(remaining);
        for _ in 0..remaining {
            out.push(self.encode_frame());
        }
        Ok(out)
    }

    /// Priming samples at the stream's own rate (not 48 kHz ticks, as for
    /// Opus): the muxer's edit list skips them.
    fn pre_skip(&self) -> u16 {
        ENCODER_DELAY as u16
    }

    /// The AudioSpecificConfig.
    fn extra_data(&self) -> Vec<u8> {
        self.asc.to_vec()
    }

    fn sample_rate(&self) -> u32 {
        self.tables.rate
    }
}
