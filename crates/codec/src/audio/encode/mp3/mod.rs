//! MP3 (MPEG-1 Audio Layer III) encoder: constant bitrate, mono or stereo,
//! at 32 / 44.1 / 48 kHz.
//!
//! The encoding is LAME's, loaded at run time behind the `lame` feature
//! ([`lame`] says how, and why it is not linked). Everything around it is
//! this crate's: the choice of output rate and the resampling to it, the
//! defaults, and cutting LAME's byte stream into one packet per frame, which
//! is what an MP4 sample and a transport stream PES carry.
//!
//! - **Rate.** MPEG-1 Layer III codes 32, 44.1 and 48 kHz. Those pass
//!   through; anything else is resampled here ([`mp3_sample_rate`]): the
//!   11.025 kHz family (22.05, 88.2, 176.4 kHz) to 44.1 kHz, everything else
//!   to 48 kHz. LAME is told its output rate outright, because left to itself
//!   it drops to the MPEG-2 half rates at low bitrates, which not every
//!   player takes.
//! - **Bitrate.** CBR on the MPEG-1 ladder ([`MP3_BITRATES`]); `0` is 128k
//!   stereo, 64k mono.
//! - **Channels.** One or two; joint stereo for two. A wider layout is the
//!   caller's to downmix first ([`crate::audio::remix::mp3_layout`]).
//! - **Delay.** The decoded stream starts [`AudioEncoder::pre_skip`] samples
//!   late: LAME's encoder delay (576), the decoder's (529, the synthesis
//!   filterbank every Layer III decoder has), and the resampler's. A
//!   container hides them with an edit list, or a LAME tag's delay field in a
//!   bare `.mp3`.
//! - **No Xing/LAME tag frame.** LAME's own tag frame is switched off: in an
//!   MP4 it would be a sample that decodes to a frame of silence. The `.mp3`
//!   writer adds an `Info` frame of its own.

use std::ffi::c_int;

use crate::audio::resample::AudioResampler;
use crate::audio::{
    AudioCodec, AudioEncoder, AudioEncoderConfig, AudioError, AudioFrame, EncodedAudioPacket, MP3_BITRATES,
    MP3_DECODER_DELAY, MP3_FRAME_SAMPLES, mp3_default_bitrate, mp3_sample_rate,
};

mod lame;

use lame::Lame;

/// LAME's `-q 2`: its recommended "near-best, not too slow" noise shaping.
const LAME_QUALITY: c_int = 2;

/// Whether the LAME library can be loaded on this host (and its version).
pub fn lame_version() -> Result<String, AudioError> {
    Lame::get().map(Lame::version_string)
}

pub struct Mp3Encoder {
    lame: &'static Lame,
    flags: lame::Flags,
    in_rate: u32,
    out_rate: u32,
    channels: u8,
    resampler: Option<AudioResampler>,
    /// Interleaved samples at `out_rate` waiting to go to LAME.
    carry: Vec<f32>,
    /// LAME's output not yet cut into whole frames.
    bytes: Vec<u8>,
    /// LAME's output buffer, sized per `lame.h`'s worst case.
    out: Vec<u8>,
    pre_skip: u32,
    next_pts_us: i64,
}

// SAFETY: the LAME state is owned by this value and only reached through
// `&mut self`.
unsafe impl Send for Mp3Encoder {}

impl Mp3Encoder {
    pub fn new(config: AudioEncoderConfig) -> Result<Self, AudioError> {
        if config.codec != AudioCodec::Mp3 {
            return Err(AudioError::Encode(format!("Mp3Encoder constructed with codec {:?}", config.codec)));
        }
        if !(1..=2).contains(&config.channels) {
            return Err(AudioError::Unsupported(format!(
                "MP3 carries one or two channels; got {} (downmix first)",
                config.channels
            )));
        }
        if config.sample_rate == 0 {
            return Err(AudioError::Encode("input sample_rate is 0".into()));
        }
        let bitrate = if config.bitrate == 0 { mp3_default_bitrate(config.channels) } else { config.bitrate };
        if !MP3_BITRATES.contains(&bitrate) {
            return Err(AudioError::Unsupported(format!(
                "{bitrate} bps is not an MPEG-1 Layer III bitrate (32k..320k: {})",
                MP3_BITRATES.map(|b| format!("{}k", b / 1000)).join(", ")
            )));
        }
        let lame = Lame::get()?;
        let out_rate = mp3_sample_rate(config.sample_rate);
        // SAFETY: `lame_init` allocates a fresh state or returns null.
        let flags = unsafe { (lame.init)() };
        if flags.is_null() {
            return Err(AudioError::Encode("lame_init failed".into()));
        }
        let mut enc = Self {
            lame,
            flags,
            in_rate: config.sample_rate,
            out_rate,
            channels: config.channels,
            resampler: None,
            carry: Vec::new(),
            bytes: Vec::new(),
            out: Vec::new(),
            pre_skip: 0,
            next_pts_us: 0,
        };
        let mode = if config.channels == 1 { lame::MONO } else { lame::JOINT_STEREO };
        // SAFETY: `flags` is a live state; each setter takes it and an int.
        unsafe {
            let set = |f: unsafe extern "C" fn(lame::Flags, c_int) -> c_int, v: c_int| f(flags, v);
            // Only `init_params` validates, so the setters' results say nothing.
            set(lame.set_in_samplerate, out_rate as c_int);
            set(lame.set_out_samplerate, out_rate as c_int);
            set(lame.set_num_channels, c_int::from(config.channels));
            set(lame.set_mode, mode);
            set(lame.set_vbr, lame::VBR_OFF);
            set(lame.set_brate, (bitrate / 1000) as c_int);
            set(lame.set_quality, LAME_QUALITY);
            set(lame.set_write_vbr_tag, 0);
            let r = (lame.init_params)(flags);
            if r < 0 {
                return Err(AudioError::Encode(format!(
                    "lame_init_params failed ({r}) for {out_rate} Hz, {} ch, {} kbps",
                    config.channels,
                    bitrate / 1000
                )));
            }
            let framesize = (lame.get_framesize)(flags);
            if framesize != MP3_FRAME_SAMPLES as c_int {
                return Err(AudioError::Encode(format!(
                    "LAME chose {framesize}-sample frames at {out_rate} Hz; expected MPEG-1's {MP3_FRAME_SAMPLES}"
                )));
            }
            enc.pre_skip = (lame.get_encoder_delay)(flags).max(0) as u32 + MP3_DECODER_DELAY;
        }
        if config.sample_rate != out_rate {
            let chunk = (config.sample_rate as usize * 20 / 1000).max(1);
            let r = AudioResampler::new(config.sample_rate, out_rate, config.channels, chunk)?;
            enc.pre_skip += r.delay() as u32;
            enc.resampler = Some(r);
        }
        Ok(enc)
    }

    /// Hand everything in `carry` to LAME, then cut whole frames off `bytes`.
    fn drain(&mut self, flush: bool) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        let ch = usize::from(self.channels);
        let n = self.carry.len() / ch;
        if n > 0 {
            // lame.h: "mp3buf_size ... worst case 1.25*num_samples + 7200".
            self.out.resize(n * 5 / 4 + 7200, 0);
            let cap = self.out.len() as c_int;
            // SAFETY: `carry` holds `n` frames of `ch` floats; `out` holds `cap` bytes.
            let written = unsafe {
                if ch == 2 {
                    (self.lame.encode_interleaved)(self.flags, self.carry.as_ptr(), n as c_int, self.out.as_mut_ptr(), cap)
                } else {
                    let p = self.carry.as_ptr();
                    (self.lame.encode_planar)(self.flags, p, p, n as c_int, self.out.as_mut_ptr(), cap)
                }
            };
            if written < 0 {
                return Err(AudioError::Encode(format!("lame_encode_buffer failed: {written}")));
            }
            self.bytes.extend_from_slice(&self.out[..written as usize]);
            self.carry.clear();
        }
        if flush {
            self.out.resize(7200, 0);
            // SAFETY: `out` holds 7200 bytes, the bound passed.
            let written = unsafe { (self.lame.flush)(self.flags, self.out.as_mut_ptr(), 7200) };
            if written < 0 {
                return Err(AudioError::Encode(format!("lame_encode_flush failed: {written}")));
            }
            self.bytes.extend_from_slice(&self.out[..written as usize]);
        }
        let mut packets = Vec::new();
        let mut at = 0;
        while let Some(len) = frame_len(&self.bytes[at..]) {
            if self.bytes.len() - at < len {
                break;
            }
            packets.push(EncodedAudioPacket {
                data: self.bytes[at..at + len].to_vec(),
                pts: self.next_pts_us,
                duration: i64::from(MP3_FRAME_SAMPLES),
            });
            self.next_pts_us += i64::from(MP3_FRAME_SAMPLES) * 1_000_000 / i64::from(self.out_rate);
            at += len;
        }
        if at == 0 && self.bytes.len() >= 4 && frame_len(&self.bytes).is_none() {
            return Err(AudioError::Encode("LAME output does not start on a frame header".into()));
        }
        self.bytes.drain(..at);
        if flush && !self.bytes.is_empty() {
            return Err(AudioError::Encode(format!("{} bytes of LAME output after the last whole frame", self.bytes.len())));
        }
        Ok(packets)
    }
}

/// The length of the MPEG-1 Layer III frame whose header opens `b`, or
/// `None` when `b` does not open with one (or is too short to say).
fn frame_len(b: &[u8]) -> Option<usize> {
    if b.len() < 4 || b[0] != 0xFF || b[1] & 0xFE != 0xFA {
        return None;
    }
    let kbps = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0][usize::from(b[2] >> 4)];
    let rate = [44_100, 48_000, 32_000, 0][usize::from((b[2] >> 2) & 3)];
    if kbps == 0 || rate == 0 {
        return None;
    }
    Some(144 * kbps * 1000 / rate + usize::from((b[2] >> 1) & 1))
}

impl AudioEncoder for Mp3Encoder {
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
        match self.resampler.as_mut() {
            Some(r) => r.process(frame, &mut self.carry)?,
            None => self.carry.extend_from_slice(&frame.samples),
        }
        self.drain(false)
    }

    fn flush(&mut self) -> Result<Vec<EncodedAudioPacket>, AudioError> {
        if let Some(r) = self.resampler.as_mut() {
            r.flush(&mut self.carry)?;
        }
        self.drain(true)
    }

    /// LAME's encoder delay plus the decoder's and the resampler's, in
    /// samples at [`Self::sample_rate`].
    fn pre_skip(&self) -> u16 {
        self.pre_skip.min(u32::from(u16::MAX)) as u16
    }

    /// MP3 has no decoder configuration: its `esds` carries none.
    fn extra_data(&self) -> Vec<u8> {
        Vec::new()
    }

    fn sample_rate(&self) -> u32 {
        self.out_rate
    }
}

impl Drop for Mp3Encoder {
    fn drop(&mut self) {
        // SAFETY: `flags` came from `lame_init` and is closed exactly once.
        unsafe { (self.lame.close)(self.flags) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_outside_mpeg1_resample_to_the_nearest_family() {
        for (input, out) in [
            (48_000, 48_000),
            (44_100, 44_100),
            (32_000, 32_000),
            (22_050, 44_100),
            (11_025, 44_100),
            (88_200, 44_100),
            (96_000, 48_000),
            (16_000, 48_000),
            (8_000, 48_000),
        ] {
            assert_eq!(mp3_sample_rate(input), out, "{input}");
        }
    }

    #[test]
    fn frame_lengths_follow_the_header() {
        // 128 kbps, 44.1 kHz, no padding: 417 bytes; padded: 418.
        assert_eq!(frame_len(&[0xFF, 0xFB, 0x90, 0x64]), Some(417));
        assert_eq!(frame_len(&[0xFF, 0xFB, 0x92, 0x64]), Some(418));
        // 320 kbps at 48 kHz: 960.
        assert_eq!(frame_len(&[0xFF, 0xFB, 0xE4, 0x00]), Some(960));
        // Not Layer III / not MPEG-1 / free format / truncated.
        assert_eq!(frame_len(&[0xFF, 0xFD, 0x90, 0x64]), None);
        assert_eq!(frame_len(&[0xFF, 0xF3, 0x90, 0x64]), None);
        assert_eq!(frame_len(&[0xFF, 0xFB, 0x00, 0x64]), None);
        assert_eq!(frame_len(&[0xFF, 0xFB]), None);
    }

    /// LAME present: a second of stereo 44.1 kHz comes back as whole frames
    /// of the asked-for bitrate, and `RIVET_REQUIRE_LAME` turns a missing
    /// library into a failure rather than a skip (CI installs it).
    #[test]
    fn encodes_whole_cbr_frames_when_lame_is_present() {
        if let Err(e) = lame_version() {
            assert!(std::env::var_os("RIVET_REQUIRE_LAME").is_none(), "{e}");
            eprintln!("skipping: {e}");
            return;
        }
        let mut enc = Mp3Encoder::new(AudioEncoderConfig {
            codec: AudioCodec::Mp3,
            sample_rate: 44_100,
            channels: 2,
            bitrate: 0,
        })
        .unwrap();
        assert_eq!(enc.sample_rate(), 44_100);
        assert_eq!(enc.pre_skip(), 576 + 529, "LAME's delay + the decoder's");
        let samples = (0..44_100 * 2).map(|i| 0.25 * ((i / 2) as f32 * 0.0627).sin()).collect();
        let mut packets = enc.encode(&AudioFrame { samples, sample_rate: 44_100, channels: 2, pts: 0 }).unwrap();
        packets.extend(enc.flush().unwrap());
        // Every input sample and LAME's 576 of delay, in whole frames, and
        // no more than the final partial frame and LAME's flush frame past it.
        let coded = packets.len() * 1152;
        assert!((44_100 + 576..44_100 + 576 + 2 * 1152).contains(&coded), "{} frames", packets.len());
        for p in &packets {
            assert_eq!(p.duration, 1152);
            assert_eq!(p.data[0..2], [0xFF, 0xFB], "MPEG-1 Layer III, no CRC");
            assert_eq!(p.data[2] >> 4, 9, "128 kbps");
            assert_eq!(p.data.len(), frame_len(&p.data).unwrap());
        }
        assert_eq!(p_mode(&packets[packets.len() / 2].data), 1, "joint stereo");
    }

    fn p_mode(frame: &[u8]) -> u8 {
        frame[3] >> 6
    }

    #[test]
    fn a_resampled_source_adds_the_resampler_delay() {
        if lame_version().is_err() {
            return;
        }
        let enc = Mp3Encoder::new(AudioEncoderConfig {
            codec: AudioCodec::Mp3,
            sample_rate: 22_050,
            channels: 1,
            bitrate: 0,
        })
        .unwrap();
        assert_eq!(enc.sample_rate(), 44_100);
        assert!(enc.pre_skip() > 576 + 529);
    }

    #[test]
    fn off_ladder_bitrates_and_surround_are_refused() {
        let cfg = |channels, bitrate| AudioEncoderConfig { codec: AudioCodec::Mp3, sample_rate: 48_000, channels, bitrate };
        let e = Mp3Encoder::new(cfg(2, 100_000)).err().unwrap();
        assert!(e.to_string().contains("not an MPEG-1 Layer III bitrate"), "{e}");
        let e = Mp3Encoder::new(cfg(6, 0)).err().unwrap();
        assert!(e.to_string().contains("one or two channels"), "{e}");
    }
}
