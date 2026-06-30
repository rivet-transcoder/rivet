//! Multistream (surround) encoder primitives: the RAII [`MultistreamEncoder`]
//! wrapper over `OpusMSEncoder*` (libopus FFI) and the channel-mapping
//! family-1 table per RFC 7845 §5.1.1.2.

use audiopus::Application;
use audiopus::ffi;
use std::ffi::c_int;
use std::ptr;

use crate::audio::AudioError;

/// Channel-mapping family 1 surround layouts per RFC 7845 §5.1.1.2.
/// Each entry: (streams, coupled_streams, channel_mapping).
/// `streams` = total internal Opus streams.
/// `coupled_streams` = number of those streams that are stereo (2-channel).
/// `channel_mapping[i]` = which encoder stream the i-th *output* channel
/// pulls from. Indices 0..coupled*2 belong to the coupled (stereo)
/// streams (each coupled stream consumes two consecutive indices); the
/// remaining indices coupled*2..streams+coupled belong to the mono
/// (uncoupled) streams. Total mapping length == channel count.
pub(super) fn surround_mapping_family_1(
    channels: u8,
) -> Result<(u8, u8, &'static [u8]), AudioError> {
    // RFC 7845 §5.1.1.2 — Vorbis channel order on input:
    //   3 (3.0):   L, R, C
    //   4 (quad):  FL, FR, BL, BR
    //   5 (5.0):   FL, FR, C, BL, BR
    //   6 (5.1):   FL, FR, C, LFE, BL, BR
    //   7 (6.1):   FL, FR, C, LFE, BC, SL, SR
    //   8 (7.1):   FL, FR, C, LFE, BL, BR, SL, SR
    //
    // Tuples below `(streams, coupled, mapping)` mirror libopus's
    // authoritative `vorbis_mappings[]` table in
    // `opus/src/opus_multistream_encoder.c:53-62`. The note in the
    // Squad-28 task spec listed `7 channels: streams=4, coupled=2`,
    // but libopus's reference table has `streams=4, coupled=3` (and
    // the mapping `[0, 4, 1, 2, 3, 5, 6]` requires stream index 6 to
    // exist, which only happens with coupled=3 → max valid stream
    // index = streams + coupled - 1 = 6). Using the libopus values
    // makes the multistream-create call succeed.
    //
    // The mapping below answers "which coded stream is each output channel?".
    // Encoder packs coupled (stereo) pairs first (taking 2 indices each),
    // then mono streams (1 index each). Total encoder channels =
    // streams + coupled.
    match channels {
        3 => Ok((2, 1, &[0, 2, 1])),
        4 => Ok((2, 2, &[0, 1, 2, 3])),
        5 => Ok((3, 2, &[0, 4, 1, 2, 3])),
        6 => Ok((4, 2, &[0, 4, 1, 2, 3, 5])),
        7 => Ok((4, 3, &[0, 4, 1, 2, 3, 5, 6])),
        8 => Ok((5, 3, &[0, 6, 1, 2, 3, 4, 5, 7])),
        _ => Err(AudioError::Unsupported(format!(
            "Opus surround mapping family 1 only defined for 3..=8 channels; got {channels}"
        ))),
    }
}

/// RAII wrapper over a raw `OpusMSEncoder*`. The underlying libopus
/// state is allocated on the libopus heap; we destroy it via the FFI
/// destroy call when the wrapper drops. The pointer is non-null after
/// successful construction (`MultistreamEncoder::new` enforces this).
pub(super) struct MultistreamEncoder {
    state: *mut ffi::OpusMSEncoder,
}

// SAFETY: libopus's multistream encoder state has no implicit thread
// affinity — like the regular Encoder we expose it via `&mut self`
// methods only, so external aliasing is impossible. The Send bound
// matches what audiopus's high-level Encoder claims.
unsafe impl Send for MultistreamEncoder {}

impl MultistreamEncoder {
    /// Allocate + initialize a multistream encoder for the given
    /// channel-mapping family-1 layout. `mapping.len()` must equal
    /// `channels`. Internally calls `opus_multistream_encoder_create`.
    pub(super) fn new(
        sample_rate: u32,
        channels: u8,
        streams: u8,
        coupled_streams: u8,
        mapping: &[u8],
        application: Application,
    ) -> Result<Self, AudioError> {
        if mapping.len() != channels as usize {
            return Err(AudioError::Encode(format!(
                "multistream mapping length {} != channels {channels}",
                mapping.len()
            )));
        }
        // libopus invariant: streams + coupled_streams <= channels and
        // coupled_streams <= streams. Re-check here so a hand-crafted
        // call from inside this module (e.g. via a future API change)
        // can't slip past the public surround_mapping_family_1 helper.
        if coupled_streams > streams {
            return Err(AudioError::Encode(format!(
                "coupled_streams ({coupled_streams}) > streams ({streams})"
            )));
        }
        if (streams as usize) + (coupled_streams as usize) > channels as usize {
            return Err(AudioError::Encode(format!(
                "streams ({streams}) + coupled_streams ({coupled_streams}) > channels ({channels})"
            )));
        }

        let mut err: c_int = 0;
        // SAFETY: `mapping.as_ptr()` is valid for `channels` bytes
        // (asserted above); libopus reads `channels` bytes from it
        // synchronously. `&mut err` is a valid out-pointer for c_int.
        // The returned pointer is checked for null on the libopus
        // contract: non-null implies err == OPUS_OK.
        let state = unsafe {
            ffi::opus_multistream_encoder_create(
                sample_rate as i32,
                channels as c_int,
                streams as c_int,
                coupled_streams as c_int,
                mapping.as_ptr(),
                application as c_int,
                &mut err,
            )
        };
        if state.is_null() || err != ffi::OPUS_OK {
            return Err(AudioError::Encode(format!(
                "opus_multistream_encoder_create failed: code={err}"
            )));
        }
        Ok(Self { state })
    }

    /// Set per-encoder VBR. CTL request OPUS_SET_VBR_REQUEST takes an
    /// i32 (0 = CBR, 1 = VBR).
    pub(super) fn set_vbr(&mut self, vbr: bool) -> Result<(), AudioError> {
        let val: c_int = if vbr { 1 } else { 0 };
        // SAFETY: `self.state` is a valid OpusMSEncoder*; the variadic
        // CTL ABI expects the request id followed by exactly one i32
        // argument for OPUS_SET_VBR (per libopus opus_defines.h).
        let r = unsafe {
            ffi::opus_multistream_encoder_ctl(self.state, ffi::OPUS_SET_VBR_REQUEST, val)
        };
        if r != ffi::OPUS_OK {
            return Err(AudioError::Encode(format!(
                "opus_multistream_encoder_ctl(SET_VBR) failed: {r}"
            )));
        }
        Ok(())
    }

    /// Set the *aggregate* bitrate across all streams. libopus
    /// distributes this internally proportional to each stream's
    /// channel count (mono streams ~half a stereo stream's allocation).
    pub(super) fn set_bitrate(&mut self, bps: i32) -> Result<(), AudioError> {
        // SAFETY: same as set_vbr; OPUS_SET_BITRATE takes an i32.
        let r = unsafe {
            ffi::opus_multistream_encoder_ctl(self.state, ffi::OPUS_SET_BITRATE_REQUEST, bps)
        };
        if r != ffi::OPUS_OK {
            return Err(AudioError::Encode(format!(
                "opus_multistream_encoder_ctl(SET_BITRATE) failed: {r}"
            )));
        }
        Ok(())
    }

    /// Query the encoder's lookahead in samples at the configured
    /// sample rate (always 48 kHz for our usage). Returned as the
    /// `dOps.PreSkip` field per RFC 7845 §4.2.
    ///
    /// Returned as `u32` to match the audiopus high-level Encoder
    /// signature (libopus actually surfaces a non-negative i32 — the
    /// CTL never returns negative lookahead values).
    pub(super) fn lookahead(&self) -> Result<u32, AudioError> {
        let mut out: c_int = 0;
        // SAFETY: OPUS_GET_LOOKAHEAD takes a `*mut int` out parameter.
        let r = unsafe {
            ffi::opus_multistream_encoder_ctl(
                self.state,
                ffi::OPUS_GET_LOOKAHEAD_REQUEST,
                &mut out as *mut c_int,
            )
        };
        if r != ffi::OPUS_OK {
            return Err(AudioError::Encode(format!(
                "opus_multistream_encoder_ctl(GET_LOOKAHEAD) failed: {r}"
            )));
        }
        if out < 0 {
            return Err(AudioError::Encode(format!(
                "opus_multistream_encoder_ctl(GET_LOOKAHEAD) returned negative: {out}"
            )));
        }
        Ok(out as u32)
    }

    /// Encode one 20-ms multichannel frame from interleaved f32 input.
    /// `pcm.len()` must equal `frame_size * channels`. Returns the
    /// encoded packet length in bytes (always positive on success).
    pub(super) fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        out: &mut [u8],
    ) -> Result<usize, AudioError> {
        let max = out.len().min(i32::MAX as usize) as i32;
        // SAFETY: pcm.as_ptr() valid for frame_size*channels f32s
        // (caller guarantees via slice length); out.as_mut_ptr() valid
        // for `max` bytes; libopus reads/writes only within those
        // bounds.
        let n = unsafe {
            ffi::opus_multistream_encode_float(
                self.state,
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                max,
            )
        };
        if n < 0 {
            return Err(AudioError::Encode(format!(
                "opus_multistream_encode_float failed: code={n}"
            )));
        }
        Ok(n as usize)
    }
}

impl Drop for MultistreamEncoder {
    fn drop(&mut self) {
        if !self.state.is_null() {
            // SAFETY: state was allocated by opus_multistream_encoder_create
            // and is destroyed exactly once (Drop runs once).
            unsafe { ffi::opus_multistream_encoder_destroy(self.state) };
            self.state = ptr::null_mut();
        }
    }
}
