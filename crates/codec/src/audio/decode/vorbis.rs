//! Vorbis decoder wrapping `lewton` (pure-Rust, MIT/Apache-2.0).
//!
//! Vorbis storage in MKV / WebM doesn't include the OGG container —
//! MKV's `CodecPrivate` carries the three Xiph setup headers (ident,
//! comment, setup) packed in the Xiph "lacing" layout, then each
//! `Block` holds one raw audio packet. Squad-23's audio mux side will
//! receive Vorbis sources from MKV demux this way; we accept the
//! packed CodecPrivate as `extra_data` on construction and use lewton's
//! lower-level packet API.
//!
//! For OGG-Vorbis files (`.ogg` / `.oga`) the MP4 mux side hands us
//! whole audio packets the OGG demuxer split out — same code path,
//! since lewton's per-packet API is stateless across the container.
//!
//! Xiph lacing layout (used by MKV CodecPrivate, FLAC METADATA_BLOCK,
//! and several other places that need to pack 3 variable-length blobs
//! into a flat byte buffer):
//! - byte 0: number of headers minus one (so byte 0 = 2 for Vorbis)
//! - bytes 1..N: lacing values for headers 0..N-2 (each: a sequence of
//!   0xFF bytes terminated by a non-0xFF byte; the lengths sum)
//! - then the headers in order (header 0, 1, ..., N-1). The last
//!   header's length is computed from the total CodecPrivate length
//!   minus the lacing prefix and the sum of explicit lacing lengths.

use lewton::audio::{PreviousWindowRight, read_audio_packet_generic};
use lewton::header::{
    HeaderReadError, IdentHeader, SetupHeader, read_header_ident, read_header_setup,
};

use crate::audio::{AudioDecoder, AudioError, AudioFrame};

pub struct VorbisDecoder {
    ident: IdentHeader,
    setup: SetupHeader,
    pwr: PreviousWindowRight,
    declared_sample_rate: u32,
    /// Caller-declared channel count (defaults to ident header value
    /// when the container reports 0). Held for cross-check; current
    /// path uses the per-packet decoded channel count from lewton.
    #[allow(dead_code)]
    declared_channels: u8,
    /// Running PTS in microseconds. Set on first `decode` call.
    next_pts_us: Option<i64>,
}

impl VorbisDecoder {
    /// Construct a decoder. `extra_data` MUST be the Xiph-laced
    /// concatenation of the three Vorbis setup packets (ident +
    /// comment + setup), as MKV CodecPrivate carries it. For OGG
    /// sources, the demuxer is responsible for assembling the same
    /// layout from the first three packets before calling here.
    pub fn new(
        extra_data: Option<&[u8]>,
        sample_rate: u32,
        channels: u8,
    ) -> Result<Self, AudioError> {
        let extra = extra_data.ok_or_else(|| {
            AudioError::Decode(
                "vorbis decoder needs CodecPrivate-style setup headers as extra_data".to_string(),
            )
        })?;
        let (ident_bytes, _comment_bytes, setup_bytes) = parse_xiph_lacing(extra)?;

        let ident = read_header_ident(ident_bytes)
            .map_err(|e| AudioError::Decode(format!("vorbis ident header: {}", header_err(&e))))?;

        // Cross-check container claims against the bitstream's own.
        // The container side may report 0 if the demuxer didn't have
        // Audio metadata; we tolerate that and trust the ident header.
        let cs = if sample_rate == 0 {
            ident.audio_sample_rate
        } else {
            sample_rate
        };
        let cc = if channels == 0 {
            ident.audio_channels
        } else {
            channels
        };

        if ident.audio_channels == 0 || ident.audio_channels > 2 {
            return Err(AudioError::Unsupported(format!(
                "vorbis channel count {} (this decoder routes >2 channels through resampler/encoder which only supports mono/stereo)",
                ident.audio_channels
            )));
        }

        let setup = read_header_setup(
            setup_bytes,
            ident.audio_channels,
            (ident.blocksize_0, ident.blocksize_1),
        )
        .map_err(|e| AudioError::Decode(format!("vorbis setup header: {}", header_err(&e))))?;

        Ok(Self {
            ident,
            setup,
            pwr: PreviousWindowRight::new(),
            declared_sample_rate: cs,
            declared_channels: cc,
            next_pts_us: None,
        })
    }
}

impl AudioDecoder for VorbisDecoder {
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<AudioFrame>, AudioError> {
        if self.next_pts_us.is_none() {
            self.next_pts_us = Some(pts);
        }
        if packet.is_empty() {
            return Ok(Vec::new());
        }

        // Vorbis returns Vec<Vec<f32>> per channel (planar). We flatten
        // to interleaved planar to match AudioFrame's contract.
        let decoded: Vec<Vec<f32>> = read_audio_packet_generic::<Vec<Vec<f32>>>(
            &self.ident,
            &self.setup,
            packet,
            &mut self.pwr,
        )
        .map_err(|e| AudioError::Decode(format!("vorbis audio packet: {e:?}")))?;

        if decoded.is_empty() {
            return Ok(Vec::new());
        }
        let channels = decoded.len() as u8;
        if channels == 0 {
            return Ok(Vec::new());
        }
        let frames_per_channel = decoded[0].len();
        if frames_per_channel == 0 {
            return Ok(Vec::new());
        }

        let mut interleaved = Vec::with_capacity(frames_per_channel * channels as usize);
        for i in 0..frames_per_channel {
            for ch in 0..channels as usize {
                let s = decoded[ch][i];
                // lewton already produces f32 in [-1, 1]; clamp
                // defensively to match AudioFrame's contract.
                interleaved.push(s.clamp(-1.0, 1.0));
            }
        }

        let pts_us = self.next_pts_us.unwrap_or(pts);
        let frame_us = (frames_per_channel as i64 * 1_000_000) / self.declared_sample_rate as i64;
        self.next_pts_us = Some(pts_us + frame_us);

        Ok(vec![AudioFrame {
            samples: interleaved,
            sample_rate: self.declared_sample_rate,
            channels,
            pts: pts_us,
        }])
    }

    fn flush(&mut self) -> Result<Vec<AudioFrame>, AudioError> {
        // Vorbis is stateless after each packet flush — there's no
        // tail buffer to drain. PreviousWindowRight only matters for
        // the next packet's IMDCT overlap-add.
        Ok(Vec::new())
    }
}

/// Parse a 3-element Xiph lacing buffer. Used by MKV CodecPrivate for
/// Vorbis (and FLAC). Returns the three header byte slices.
fn parse_xiph_lacing(bytes: &[u8]) -> Result<(&[u8], &[u8], &[u8]), AudioError> {
    if bytes.is_empty() {
        return Err(AudioError::Decode("vorbis extra_data is empty".to_string()));
    }
    let n_minus_1 = bytes[0] as usize;
    if n_minus_1 != 2 {
        return Err(AudioError::Decode(format!(
            "vorbis extra_data lacing prefix says n-1={n_minus_1}, expected 2 (3 headers)"
        )));
    }
    // Read lacing values for headers 0..N-2 (so for N=3, 2 values).
    let mut cursor = 1usize;
    let mut lengths = [0usize; 2];
    for slot in lengths.iter_mut() {
        let mut total = 0usize;
        loop {
            if cursor >= bytes.len() {
                return Err(AudioError::Decode(
                    "vorbis extra_data ended inside Xiph lacing length".to_string(),
                ));
            }
            let v = bytes[cursor] as usize;
            cursor += 1;
            total += v;
            if v != 0xFF {
                break;
            }
        }
        *slot = total;
    }
    let len0 = lengths[0];
    let len1 = lengths[1];
    let header_bytes_start = cursor;
    if header_bytes_start + len0 + len1 > bytes.len() {
        return Err(AudioError::Decode(format!(
            "vorbis extra_data: lacing lengths {} + {} + tail exceed buffer ({} bytes after prefix, total {})",
            len0,
            len1,
            bytes.len() - header_bytes_start,
            bytes.len()
        )));
    }
    let len2 = bytes.len() - header_bytes_start - len0 - len1;
    let h0 = &bytes[header_bytes_start..header_bytes_start + len0];
    let h1 = &bytes[header_bytes_start + len0..header_bytes_start + len0 + len1];
    let h2 = &bytes[header_bytes_start + len0 + len1..header_bytes_start + len0 + len1 + len2];
    Ok((h0, h1, h2))
}

fn header_err(e: &HeaderReadError) -> String {
    format!("{e:?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xiph_lacing_parses_simple_three_segment_buffer() {
        // Header lengths 30, 19, 5 packed:
        // prefix: [2, 30, 19, then 30 + 19 + 5 = 54 bytes of payload]
        let mut buf = vec![2u8, 30, 19];
        buf.extend(std::iter::repeat(0xAAu8).take(30));
        buf.extend(std::iter::repeat(0xBBu8).take(19));
        buf.extend(std::iter::repeat(0xCCu8).take(5));
        let (a, b, c) = parse_xiph_lacing(&buf).expect("parses");
        assert_eq!(a.len(), 30);
        assert_eq!(b.len(), 19);
        assert_eq!(c.len(), 5);
        assert!(a.iter().all(|x| *x == 0xAA));
        assert!(b.iter().all(|x| *x == 0xBB));
        assert!(c.iter().all(|x| *x == 0xCC));
    }

    #[test]
    fn xiph_lacing_handles_long_runs() {
        // Length-260 segment encodes as 0xFF 0x05 (255 + 5).
        let mut buf = vec![2u8, 0xFF, 0x05, 0x10];
        buf.extend(std::iter::repeat(0u8).take(260));
        buf.extend(std::iter::repeat(1u8).take(16));
        buf.extend(std::iter::repeat(2u8).take(8));
        let (a, b, c) = parse_xiph_lacing(&buf).expect("parses");
        assert_eq!(a.len(), 260);
        assert_eq!(b.len(), 16);
        assert_eq!(c.len(), 8);
    }

    #[test]
    fn xiph_lacing_rejects_wrong_header_count() {
        let buf = vec![1u8, 5, 5];
        assert!(parse_xiph_lacing(&buf).is_err());
    }

    #[test]
    fn xiph_lacing_rejects_truncated_buffer() {
        let buf = vec![2u8, 30, 19, 0, 0]; // only 2 payload bytes claimed >0
        assert!(parse_xiph_lacing(&buf).is_err());
    }

    #[test]
    fn vorbis_decoder_rejects_missing_extra_data() {
        let r = VorbisDecoder::new(None, 44100, 2);
        assert!(matches!(r, Err(AudioError::Decode(_))));
    }

    #[test]
    fn vorbis_decoder_rejects_garbage_extra_data() {
        // Looks like proper lacing prefix but ident header parser bails.
        let extra = vec![2u8, 30, 19, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let r = VorbisDecoder::new(Some(&extra), 44100, 2);
        assert!(matches!(r, Err(AudioError::Decode(_))));
    }
}
