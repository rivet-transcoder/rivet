//! Sample-rate conversion using rubato's `SincFixedIn` (high-quality
//! windowed-sinc with band-limited interpolation).
//!
//! Common case: 44.1 kHz MP3 source → 48 kHz Opus encoder. Less common:
//! 22.05 / 32 / 96 kHz inputs. rubato handles arbitrary float ratios so
//! we just compute `out_rate / in_rate` and feed it to SincFixedIn.
//!
//! Squad-23 doesn't call this directly — it goes through the Opus
//! encoder, which holds an `Option<Resampler>` that's `Some` whenever
//! the input rate differs from 48 kHz.
//!
//! Layout conversion
//! -----------------
//! Rubato wants non-interleaved (`Vec<Vec<f32>>`, one inner Vec per
//! channel) — the codec module's [`AudioFrame`] uses interleaved planar
//! to match the rest of the codebase. We deinterleave at input,
//! re-interleave at output. The cost is one extra pair of allocations
//! per frame; for 20 ms of stereo at 48 kHz this is 1920 samples — a
//! few µs at most, well below the per-frame budget of ~20 ms.
//!
//! PTS through resampling
//! ----------------------
//! Output PTS = input PTS. The lookahead delay rubato adds to satisfy
//! its sinc filter is collapsed into the encoder's pre_skip when
//! Opus is the consumer; downstream callers should not see PTS drift.

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use crate::audio::{AudioError, AudioFrame};

/// Default sinc filter length for our usage. 256 is rubato's
/// recommended starting point — gives transition band roll-off well
/// below typical Opus inaudibility threshold for 44.1 → 48 conversion.
const SINC_LEN: usize = 256;
/// Cutoff relative to Nyquist of the lower sample rate. 0.95 is
/// rubato's recommended starting value — leaves a small guard band so
/// the Blackman-Harris window fully suppresses the alias mirror.
const F_CUTOFF: f32 = 0.95;
/// Oversampling factor for the sinc table. 256 is a balance between
/// memory and quality recommended by rubato's docs.
const OVERSAMPLING: usize = 256;

pub struct AudioResampler {
    resampler: SincFixedIn<f32>,
    in_rate: u32,
    out_rate: u32,
    channels: u8,
    chunk_size: usize,
    /// Reusable input buffer (deinterleaved) so we don't allocate per
    /// frame in the hot path.
    deinterleaved: Vec<Vec<f32>>,
    /// Carryover of input samples that didn't fill a chunk on the
    /// previous call (deinterleaved). On the next push we prepend them
    /// to the new input.
    carry: Vec<Vec<f32>>,
}

impl AudioResampler {
    /// Construct a resampler for `in_rate` → `out_rate` with
    /// `channels` channels processing `chunk_size` input frames per
    /// call. Returns an error if any rate is zero.
    pub fn new(
        in_rate: u32,
        out_rate: u32,
        channels: u8,
        chunk_size: usize,
    ) -> Result<Self, AudioError> {
        if in_rate == 0 || out_rate == 0 {
            return Err(AudioError::Resample(format!(
                "invalid sample rate {in_rate} -> {out_rate}"
            )));
        }
        // Squad-28: lifted the 1..=2 channel cap so multichannel Opus
        // (3..=8 channels through a multistream encoder, RFC 7845 §5.1.1
        // family 1) can resample its input. Rubato handles arbitrary
        // channel counts — the deinterleave/re-interleave loop below is
        // already N-channel general. We cap at 8 because the dOps
        // ChannelMappingTable is only specified for 1..=8 channels in
        // the standard surround layouts (RFC 7845 §5.1.1.2) and matches
        // the upper bound the multistream encoder enforces.
        if channels == 0 || channels > 8 {
            return Err(AudioError::Unsupported(format!(
                "resampler channel count {channels} (must be 1..=8)"
            )));
        }
        if chunk_size == 0 {
            return Err(AudioError::Resample("chunk_size must be > 0".to_string()));
        }

        let params = SincInterpolationParameters {
            sinc_len: SINC_LEN,
            f_cutoff: F_CUTOFF,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: OVERSAMPLING,
            window: WindowFunction::BlackmanHarris2,
        };

        let ratio = f64::from(out_rate) / f64::from(in_rate);
        let resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, channels as usize)
            .map_err(|e| AudioError::Resample(format!("rubato init: {e:?}")))?;

        let deinterleaved = vec![vec![0.0f32; chunk_size]; channels as usize];
        let carry = vec![Vec::new(); channels as usize];

        Ok(Self {
            resampler,
            in_rate,
            out_rate,
            channels,
            chunk_size,
            deinterleaved,
            carry,
        })
    }

    pub fn in_rate(&self) -> u32 {
        self.in_rate
    }
    pub fn out_rate(&self) -> u32 {
        self.out_rate
    }
    pub fn channels(&self) -> u8 {
        self.channels
    }
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }
    /// How many output samples late the output runs: the sinc filter's
    /// group delay, at the output rate.
    pub fn delay(&self) -> usize {
        self.resampler.output_delay()
    }

    /// Process `frame.samples` (interleaved) and append output samples
    /// (interleaved) into `out`. Carries any partial input chunk
    /// internally for the next call.
    ///
    /// The output PTS is the same as the input PTS — the resampler
    /// itself doesn't expose its internal lookahead in our wire model
    /// (the encoder converts that into pre_skip ticks at the file
    /// header level).
    // Re-interleaving indexes frame-major, channel-minor; the index reads plainest.
    #[allow(clippy::needless_range_loop)]
    pub fn process(&mut self, frame: &AudioFrame, out: &mut Vec<f32>) -> Result<(), AudioError> {
        if frame.channels != self.channels {
            return Err(AudioError::Resample(format!(
                "channel mismatch: resampler={}, frame={}",
                self.channels, frame.channels
            )));
        }
        if frame.sample_rate != self.in_rate {
            return Err(AudioError::Resample(format!(
                "sample rate mismatch: resampler in_rate={}, frame={}",
                self.in_rate, frame.sample_rate
            )));
        }

        // Deinterleave + carry: append into self.carry per-channel.
        let chans = self.channels as usize;
        let frames = frame.samples.len() / chans;
        for ch in 0..chans {
            let base = self.carry[ch].len();
            self.carry[ch].reserve(frames);
            for i in 0..frames {
                self.carry[ch].push(frame.samples[i * chans + ch]);
            }
            // (base used only for the `reserve` hint; index is reused
            // as a bookkeeping witness that the per-channel push
            // ordering is correct.)
            debug_assert_eq!(self.carry[ch].len(), base + frames);
        }

        // Drain as many full chunks as we have carry for.
        while self.carry[0].len() >= self.chunk_size {
            for ch in 0..chans {
                self.deinterleaved[ch].copy_from_slice(&self.carry[ch][..self.chunk_size]);
            }
            for ch in 0..chans {
                self.carry[ch].drain(..self.chunk_size);
            }
            let result = self
                .resampler
                .process(&self.deinterleaved, None)
                .map_err(|e| AudioError::Resample(format!("rubato process: {e:?}")))?;
            // Re-interleave into `out`.
            let n_out = result[0].len();
            out.reserve(n_out * chans);
            for i in 0..n_out {
                for ch in 0..chans {
                    out.push(result[ch][i]);
                }
            }
        }

        Ok(())
    }

    /// Flush any carry by zero-padding to a full chunk and processing
    /// it. Useful at end-of-stream to drain the rubato sinc filter.
    #[allow(clippy::needless_range_loop)] // same interleave as `process`
    pub fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), AudioError> {
        let chans = self.channels as usize;
        let n = self.carry[0].len();
        if n == 0 {
            return Ok(());
        }
        for ch in 0..chans {
            self.carry[ch].resize(self.chunk_size, 0.0);
            self.deinterleaved[ch].copy_from_slice(&self.carry[ch][..self.chunk_size]);
            self.carry[ch].clear();
        }
        let result = self
            .resampler
            .process(&self.deinterleaved, None)
            .map_err(|e| AudioError::Resample(format!("rubato flush: {e:?}")))?;
        let n_out = result[0].len();
        out.reserve(n_out * chans);
        for i in 0..n_out {
            for ch in 0..chans {
                out.push(result[ch][i]);
            }
        }
        Ok(())
    }
}

/// A resampler whose output lines up with its input: the filter's delay is
/// dropped from the front, and at the end exactly as many samples come out
/// as the input's length at the output rate (rounded up). What an encoder
/// whose coding rate is not the input's puts in front of itself, so the
/// stream's priming is the codec's own alone. With equal rates it passes the
/// samples through untouched.
pub struct AlignedResampler {
    inner: Option<AudioResampler>,
    in_rate: u32,
    out_rate: u32,
    channels: u8,
    /// Output samples (per channel) of the filter's delay still to drop.
    skip: usize,
    /// Input and output samples per channel so far.
    samples_in: u64,
    samples_out: u64,
}

impl AlignedResampler {
    /// From `in_rate` to `out_rate`, `channels` interleaved.
    pub fn new(in_rate: u32, out_rate: u32, channels: u8) -> Result<Self, AudioError> {
        let (inner, skip) = if in_rate == out_rate {
            (None, 0)
        } else {
            (Some(AudioResampler::new(in_rate, out_rate, channels, 1024)?), measured_delay(in_rate, out_rate)?)
        };
        Ok(Self { inner, in_rate, out_rate, channels, skip, samples_in: 0, samples_out: 0 })
    }

    /// Whether the rates differ (anything is resampled at all).
    pub fn is_active(&self) -> bool {
        self.inner.is_some()
    }

    pub fn in_rate(&self) -> u32 {
        self.in_rate
    }

    pub fn out_rate(&self) -> u32 {
        self.out_rate
    }

    /// The input's samples per channel so far, at the output rate, rounded up:
    /// what the output holds once [`Self::flush`] has run.
    pub fn target_len(&self) -> u64 {
        (u128::from(self.samples_in) * u128::from(self.out_rate)).div_ceil(u128::from(self.in_rate)) as u64
    }

    /// Resample `frame` (interleaved, at the input rate), appending to `out`.
    pub fn process(&mut self, frame: &AudioFrame, out: &mut Vec<f32>) -> Result<(), AudioError> {
        let ch = usize::from(self.channels);
        self.samples_in += (frame.samples.len() / ch.max(1)) as u64;
        match self.inner.as_mut() {
            None => {
                out.extend_from_slice(&frame.samples);
                self.samples_out = self.samples_in;
                Ok(())
            }
            Some(r) => {
                let mut tmp = Vec::new();
                r.process(frame, &mut tmp)?;
                self.take(&tmp, out);
                Ok(())
            }
        }
    }

    /// The end of the input: the filter's delayed tail, cut to [`Self::target_len`].
    pub fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), AudioError> {
        let ch = usize::from(self.channels);
        let Some(mut r) = self.inner.take() else {
            return Ok(());
        };
        let mut tmp = Vec::new();
        // Silence behind the input pushes the delayed tail out; enough of it
        // to cover the delay whatever the chunk size.
        let blocks = (self.skip / r.chunk_size().max(1)) + 2;
        for _ in 0..blocks {
            let tail = AudioFrame {
                samples: vec![0.0; r.chunk_size() * ch],
                sample_rate: self.in_rate,
                channels: self.channels,
                pts: 0,
            };
            r.process(&tail, &mut tmp)?;
        }
        r.flush(&mut tmp)?;
        self.take(&tmp, out);
        let target = self.target_len();
        if self.samples_out > target {
            let extra = (self.samples_out - target) as usize;
            out.truncate(out.len() - extra.min(out.len() / ch.max(1)) * ch);
            self.samples_out = target;
        }
        Ok(())
    }

    /// Append `samples` (resampled) to `out`, less the delay still to drop.
    fn take(&mut self, samples: &[f32], out: &mut Vec<f32>) {
        let ch = usize::from(self.channels).max(1);
        let n = samples.len() / ch;
        let skip = self.skip.min(n);
        self.skip -= skip;
        out.extend_from_slice(&samples[skip * ch..n * ch]);
        self.samples_out += (n - skip) as u64;
    }
}

/// The delay, in output samples, of the resampler from `in_rate` to
/// `out_rate`: where an impulse at the first input sample comes out. Measured
/// rather than computed, so it holds whatever the filter's design.
pub fn measured_delay(in_rate: u32, out_rate: u32) -> Result<usize, AudioError> {
    let mut r = AudioResampler::new(in_rate, out_rate, 1, 1024)?;
    let mut impulse = vec![0.0f32; 1024];
    impulse[0] = 1.0;
    let mut out = Vec::new();
    for samples in [impulse, vec![0.0; 1024]] {
        let frame = AudioFrame { samples, sample_rate: in_rate, channels: 1, pts: 0 };
        r.process(&frame, &mut out)?;
    }
    Ok(out
        .iter()
        .enumerate()
        .fold((0, 0.0f32), |best, (i, &v)| if v.abs() > best.1 { (i, v.abs()) } else { best })
        .0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_44100_to_48000_preserves_sample_count_within_tolerance() {
        // Process exactly one chunk of 44100 samples (1 second of mono
        // audio at 44.1 kHz). Expect output sample count close to
        // 48000 (within 1% — rubato's SincFixedIn emits per its
        // internal sinc-len delay, which depends on SINC_LEN and the
        // ratio, so the exact output count is not exactly `ratio *
        // chunk_size`; it's `chunk_size * ratio - filter_delay` on
        // the first call, then ramps to `ratio * chunk_size` once
        // the filter is primed).
        let chunk = 44100;
        let mut r = AudioResampler::new(44100, 48000, 1, chunk).expect("resampler");
        let frame = AudioFrame {
            samples: vec![0.0f32; chunk],
            sample_rate: 44100,
            channels: 1,
            pts: 0,
        };
        let mut out = Vec::new();
        r.process(&frame, &mut out).expect("process");
        let diff = (out.len() as i64 - 48000).abs();
        assert!(
            diff <= 480, // < 10 ms at 48k — within 1% of 48000
            "expected ~48000 output samples, got {} (diff {} — sinc filter delay from SINC_LEN)",
            out.len(),
            diff
        );
    }

    #[test]
    fn resample_rejects_zero_rates() {
        assert!(AudioResampler::new(0, 48000, 1, 1024).is_err());
        assert!(AudioResampler::new(44100, 0, 1, 1024).is_err());
    }

    #[test]
    fn resample_rejects_unsupported_channels() {
        // 0 and >8 are out of range; 6 (5.1) is now legal for Squad-28.
        assert!(AudioResampler::new(44100, 48000, 0, 1024).is_err());
        assert!(AudioResampler::new(44100, 48000, 9, 1024).is_err());
        // 6-channel resampler must construct successfully — gates the
        // 5.1-channel Opus multistream path.
        assert!(AudioResampler::new(44100, 48000, 6, 1024).is_ok());
    }

    #[test]
    fn resample_input_validation_catches_channel_mismatch() {
        let mut r = AudioResampler::new(44100, 48000, 2, 1024).expect("resampler");
        let frame = AudioFrame {
            samples: vec![0.0f32; 1024],
            sample_rate: 44100,
            channels: 1,
            pts: 0,
        };
        let mut out = Vec::new();
        assert!(r.process(&frame, &mut out).is_err());
    }

    #[test]
    fn resample_input_validation_catches_rate_mismatch() {
        let mut r = AudioResampler::new(44100, 48000, 2, 1024).expect("resampler");
        let frame = AudioFrame {
            samples: vec![0.0f32; 2048],
            sample_rate: 22050,
            channels: 2,
            pts: 0,
        };
        let mut out = Vec::new();
        assert!(r.process(&frame, &mut out).is_err());
    }

    #[test]
    fn resample_stereo_44100_to_48000_interleaved_layout_preserved() {
        let chunk = 44100;
        let mut r = AudioResampler::new(44100, 48000, 2, chunk).expect("resampler");
        // Build stereo input: left channel = +0.1, right = -0.1.
        let mut samples = Vec::with_capacity(chunk * 2);
        for _ in 0..chunk {
            samples.push(0.1f32);
            samples.push(-0.1f32);
        }
        let frame = AudioFrame {
            samples,
            sample_rate: 44100,
            channels: 2,
            pts: 0,
        };
        let mut out = Vec::new();
        r.process(&frame, &mut out).expect("process");
        assert!(out.len() % 2 == 0, "stereo output must be even");
        // Check left ≈ +0.1, right ≈ -0.1 in steady state (skip the
        // first ~sinc_len samples worth of filter warm-up).
        let warmup = 512;
        let mut ok_l = 0;
        let mut ok_r = 0;
        for i in (warmup..out.len()).step_by(2) {
            if (out[i] - 0.1).abs() < 0.05 {
                ok_l += 1;
            }
            if (out[i + 1] - (-0.1)).abs() < 0.05 {
                ok_r += 1;
            }
        }
        assert!(
            ok_l > 100,
            "L channel should converge near 0.1; got {ok_l} matches"
        );
        assert!(
            ok_r > 100,
            "R channel should converge near -0.1; got {ok_r} matches"
        );
    }

    /// A tone resampled comes out as long as the input at the new rate and
    /// in time with it: matched against the same tone generated at the new
    /// rate, to within half a sample (the filter's delay is a whole number of
    /// output samples only to the nearest).
    #[test]
    fn aligned_output_is_in_time_and_exactly_as_long() {
        for (from, to) in [(44_100u32, 48_000u32), (96_000, 48_000), (22_050, 44_100), (16_000, 24_000)] {
            let n = from as usize; // one second
            let tone = |rate: u32, t: f64| (0.5 * (2.0 * std::f64::consts::PI * 440.0 * t / f64::from(rate)).sin()) as f32;
            let input: Vec<f32> = (0..n).map(|i| tone(from, i as f64)).collect();
            let mut r = AlignedResampler::new(from, to, 1).unwrap();
            let mut out = Vec::new();
            for c in input.chunks(777) {
                let frame = AudioFrame { samples: c.to_vec(), sample_rate: from, channels: 1, pts: 0 };
                r.process(&frame, &mut out).unwrap();
            }
            r.flush(&mut out).unwrap();
            assert_eq!(out.len(), to as usize, "{from} -> {to}");
            let (lag, snr) = (-2..=2)
                .map(|q| {
                    let lag = f64::from(q) * 0.25;
                    let (mut s, mut e) = (0.0f64, 0.0f64);
                    for (i, &v) in out.iter().enumerate().skip(2000).take(to as usize - 4000) {
                        let want = tone(to, i as f64 + lag);
                        s += f64::from(want).powi(2);
                        e += f64::from(want - v).powi(2);
                    }
                    (lag, 10.0 * (s / e.max(1e-30)).log10())
                })
                .fold((0.0, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
            eprintln!("{from} -> {to}: {snr:.1} dB at {lag:+} samples");
            assert!(snr > 40.0, "{from} -> {to}: {snr:.1} dB at {lag:+}");
        }
    }
}
