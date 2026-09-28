//! ALAC (Apple Lossless) decoder, clean-room from the published format
//! description.
//!
//! One packet is one ALAC frame: a sequence of single-channel (SCE, LFE)
//! and channel-pair (CPE) elements closed by an END element, each either
//! predicted and Rice coded or escaped to raw samples. The magic cookie
//! (`ALACSpecificConfig`) from the container gives the bit depth (16, 20,
//! 24 or 32), the channel count (1–8), the frame length and the coder's
//! parameters. See [`crate::audio::lossless::alac`] for the coding scheme.
//!
//! The decoded channels are reordered from ALAC's layouts (which lead with
//! the centre channel) to the pipeline's native order; see
//! [`native_from_alac`] for the layout each count lands on.

use crate::audio::lossless::alac::{
    AlacConfig, ID_CCE, ID_CPE, ID_DSE, ID_END, ID_FIL, ID_LFE, ID_PCE, ID_SCE, RiceParams, decode_residuals,
    native_from_alac, unpredict,
};
use crate::audio::lossless::bits::BitReader;
use crate::audio::lossless::{ints_to_f32, sign_extend};
use crate::audio::{AudioDecoder, AudioError, AudioFrame};

fn err(msg: impl Into<String>) -> AudioError {
    AudioError::Decode(format!("alac: {}", msg.into()))
}

/// Decode one frame to per-channel samples in ALAC channel order.
pub fn decode_frame(config: &AlacConfig, packet: &[u8]) -> Result<Vec<Vec<i64>>, AudioError> {
    let channels = usize::from(config.num_channels);
    let depth = u32::from(config.bit_depth);
    let mut br = BitReader::new(packet, "alac");
    let mut out: Vec<Vec<i64>> = Vec::with_capacity(channels);
    let mut frame_samples: Option<usize> = None;
    loop {
        let tag = br.read_u32(3)?;
        match tag {
            ID_SCE | ID_LFE | ID_CPE => {
                let nch = if tag == ID_CPE { 2 } else { 1 };
                if out.len() + nch > channels {
                    return Err(err(format!("frame holds more than the cookie's {channels} channels")));
                }
                let _instance = br.read_u32(4)?;
                if br.read_u32(12)? != 0 {
                    return Err(err("element header's reserved bits are set"));
                }
                let partial = br.read_bit()?;
                let shift_bytes = br.read_u32(2)?;
                let escape = br.read_bit()?;
                let n = if partial { br.read_u32(32)? as usize } else { config.frame_length as usize };
                if n > config.frame_length as usize {
                    return Err(err(format!("{n} samples in a frame of at most {}", config.frame_length)));
                }
                if *frame_samples.get_or_insert(n) != n {
                    return Err(err("elements of one frame disagree on its length"));
                }
                let chans = if escape {
                    let mut chans = vec![Vec::with_capacity(n); nch];
                    for _ in 0..n {
                        for ch in chans.iter_mut() {
                            ch.push(br.read_signed(depth)?);
                        }
                    }
                    chans
                } else {
                    decode_compressed(&mut br, config, nch, n, shift_bytes)?
                };
                out.extend(chans);
            }
            ID_DSE => {
                let _instance = br.read_u32(4)?;
                let align = br.read_bit()?;
                let mut count = br.read_u32(8)? as usize;
                if count == 255 {
                    count += br.read_u32(8)? as usize;
                }
                if align {
                    br.align();
                }
                br.skip(count * 8)?;
            }
            ID_FIL => {
                let mut count = br.read_u32(4)? as usize;
                if count == 15 {
                    count += br.read_u32(8)? as usize;
                    count -= 1;
                }
                br.skip(count * 8)?;
            }
            ID_CCE | ID_PCE => return Err(AudioError::Unsupported(format!("alac: element type {tag}"))),
            ID_END => break,
            _ => unreachable!("a 3-bit tag"),
        }
    }
    if out.len() != channels {
        return Err(err(format!("frame holds {} of the cookie's {channels} channels", out.len())));
    }
    Ok(out)
}

fn decode_compressed(
    br: &mut BitReader<'_>,
    config: &AlacConfig,
    nch: usize,
    n: usize,
    shift_bytes: u32,
) -> Result<Vec<Vec<i64>>, AudioError> {
    let depth = u32::from(config.bit_depth);
    let shift = shift_bytes * 8;
    if shift >= depth {
        return Err(err(format!("{shift_bytes} shifted bytes of {depth}-bit audio")));
    }
    let chan_bits = depth - shift + (nch as u32 - 1);
    let mix_bits = br.read_u32(8)?;
    let mix_res = br.read_signed(8)?;
    struct Pred {
        mode: u32,
        den_shift: u32,
        pb_factor: u32,
        coefs: Vec<i16>,
    }
    let mut preds = Vec::with_capacity(nch);
    for _ in 0..nch {
        let mode = br.read_u32(4)?;
        let den_shift = br.read_u32(4)?;
        let pb_factor = br.read_u32(3)?;
        let order = br.read_u32(5)? as usize;
        let coefs = (0..order).map(|_| Ok(br.read_signed(16)? as i16)).collect::<Result<_, AudioError>>()?;
        preds.push(Pred { mode, den_shift, pb_factor, coefs });
    }
    // The low bytes, raw and interleaved, come before the residuals.
    let mut low: Vec<Vec<i64>> = vec![Vec::new(); nch];
    if shift > 0 {
        for ch in low.iter_mut() {
            ch.reserve(n);
        }
        for _ in 0..n {
            for ch in low.iter_mut() {
                ch.push(br.read(shift)? as i64);
            }
        }
    }
    let mut chans = Vec::with_capacity(nch);
    for p in &preds {
        let params = RiceParams::new(config, p.pb_factor);
        let res = decode_residuals(br, &params, n, chan_bits)?;
        let mut data: Vec<i64> = res.into_iter().map(i64::from).collect();
        if p.mode != 0 {
            unpredict(&mut data, &[], 31, 0, chan_bits);
        }
        let order = p.coefs.len();
        if order == 31 {
            unpredict(&mut data, &[], 31, 0, chan_bits);
        } else {
            unpredict(&mut data, &p.coefs, order, p.den_shift, chan_bits);
        }
        chans.push(data);
    }
    if nch == 2 && mix_res != 0 {
        let (u, v) = chans.split_at_mut(1);
        for (u, v) in u[0].iter_mut().zip(v[0].iter_mut()) {
            let l = *u + *v - ((mix_res * *v) >> mix_bits);
            let r = l - *v;
            *u = l;
            *v = r;
        }
    }
    if shift > 0 {
        for (ch, low) in chans.iter_mut().zip(&low) {
            for (s, &l) in ch.iter_mut().zip(low) {
                *s = (*s << shift) | l;
            }
        }
    }
    for ch in chans.iter_mut() {
        for s in ch.iter_mut() {
            *s = sign_extend(*s, depth);
        }
    }
    Ok(chans)
}

/// ALAC through the [`AudioDecoder`] surface.
pub struct AlacDecoder {
    config: AlacConfig,
    first_pts_us: Option<i64>,
    samples_decoded: u64,
}

impl AlacDecoder {
    /// `extra_data` is the magic cookie, in any of the wrappings
    /// [`AlacConfig::parse`] takes; it is required.
    pub fn new(extra_data: Option<&[u8]>) -> Result<Self, AudioError> {
        let extra = extra_data.filter(|e| !e.is_empty()).ok_or_else(|| err("no magic cookie"))?;
        Ok(Self { config: AlacConfig::parse(extra)?, first_pts_us: None, samples_decoded: 0 })
    }

    pub fn config(&self) -> &AlacConfig {
        &self.config
    }

    /// Decode one packet to interleaved integer samples in the pipeline's
    /// channel order, at the cookie's bit depth.
    pub fn decode_int(&mut self, packet: &[u8]) -> Result<Vec<i32>, AudioError> {
        let chans = decode_frame(&self.config, packet)?;
        let n = chans.first().map_or(0, Vec::len);
        let order = native_from_alac(self.config.num_channels);
        let chans = &chans;
        let out: Vec<i32> = (0..n).flat_map(|i| order.iter().map(move |&c| chans[c][i] as i32)).collect();
        self.samples_decoded += n as u64;
        Ok(out)
    }
}

impl AudioDecoder for AlacDecoder {
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<AudioFrame>, AudioError> {
        let first_pts_us = *self.first_pts_us.get_or_insert(pts);
        let before = self.samples_decoded;
        let samples = self.decode_int(packet)?;
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let rate = self.config.sample_rate.max(1);
        Ok(vec![AudioFrame {
            samples: ints_to_f32(&samples, u32::from(self.config.bit_depth)),
            sample_rate: rate,
            channels: self.config.num_channels,
            pts: first_pts_us + (before as i64 * 1_000_000) / i64::from(rate),
        }])
    }

    fn flush(&mut self) -> Result<Vec<AudioFrame>, AudioError> {
        Ok(Vec::new())
    }

    /// ALAC's four-channel layout is 4.0 (C L R Cs), not the quad the
    /// pipeline assumes of four channels. Its eight-channel layout's front
    /// left- and right-of-centre pair has no label here, and rides in the
    /// 7.1 slots SL / SR.
    fn layout(&self) -> Option<crate::audio::filter::ChannelLayout> {
        (self.config.num_channels == 4).then(|| crate::audio::filter::ChannelLayout::named("4.0"))
    }
}
