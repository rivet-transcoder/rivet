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
        // (3..=8 channels via libopus's Multistream API, RFC 7845 §5.1.1
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

    /// Process `frame.samples` (interleaved) and append output samples
    /// (interleaved) into `out`. Carries any partial input chunk
    /// internally for the next call.
    ///
    /// The output PTS is the same as the input PTS — the resampler
    /// itself doesn't expose its internal lookahead in our wire model
    /// (the encoder converts that into pre_skip ticks at the file
    /// header level).
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
}
