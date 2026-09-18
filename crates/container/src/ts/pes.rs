//! PES (Packetised Elementary Stream) header parsing and access-unit scanning.
//!
//! Provides:
//! - `parse_pes_header` — strip the PES header from a video PES payload and
//!   return the ES start offset plus any PTS.
//! - `VideoStreamScan` / `scan_first_video_au` — walk enough packets on the
//!   active video PID to capture the first access unit (for SPS dim parsing)
//!   and a window of PTSes (for frame-rate estimation).


use super::{TS_PACKET, TS_SYNC};
use crate::demux::hdr::{ColourWindow, HeadNals};

/// Parse a PES header at the start of `payload`. Returns the byte
/// offset of the elementary-stream payload within `payload`, plus any
/// PTS we extracted. PES layout (ISO/IEC 13818-1 §2.4.3.6):
///   start_code(0x000001) + stream_id(8) + PES_packet_length(16)
///   flags(16) + PES_header_data_length(8) + header_extension(...) + ES data
pub(super) fn parse_pes_header(payload: &[u8]) -> Option<(usize, Option<u64>)> {
    if payload.len() < 9 {
        return None;
    }
    if payload[0] != 0 || payload[1] != 0 || payload[2] != 1 {
        return None;
    }
    let stream_id = payload[3];
    // Video streams are 0xE0..=0xEF. Other stream_ids (audio, padding,
    // program streams) aren't what we want; bail so the caller can drop
    // the sample.
    if !(0xE0..=0xEF).contains(&stream_id) {
        return None;
    }
    // The two PES flag bytes live at offsets 6-7.
    let flags = payload[7];
    let pts_dts_flags = (flags >> 6) & 0x03;
    let header_data_len = payload[8] as usize;
    let es_start = 9 + header_data_len;
    if es_start > payload.len() {
        return None;
    }
    let pts = if pts_dts_flags == 0b10 || pts_dts_flags == 0b11 {
        // PTS occupies bytes 9..14. Layout: 4 marker bits + PTS[32..30]
        //   + 1 marker, 15 bits PTS[29..15] + 1 marker, 15 bits PTS[14..0] + 1 marker.
        if payload.len() < 14 {
            return None;
        }
        let p0 = ((payload[9] >> 1) & 0x07) as u64;
        let p1 = (((payload[10] as u64) << 7) | ((payload[11] as u64) >> 1)) & 0x7FFF;
        let p2 = (((payload[12] as u64) << 7) | ((payload[13] as u64) >> 1)) & 0x7FFF;
        Some((p0 << 30) | (p1 << 15) | p2)
    } else {
        None
    };
    Some((es_start, pts))
}

/// Result of a single-pass scan over the active video PID: the first
/// access unit's bytes (for SPS / seq-header dim extraction) plus a
/// window of PTSes (for frame-rate estimation).
pub(super) struct VideoStreamScan {
    pub(super) first_au: Option<Vec<u8>>,
    /// H.264 / HEVC: the colour window over the head of the stream
    /// ([`ColourWindow`]) — its SPS and SEIs, up to the first access unit
    /// that carries an SPS. `None` for other codecs.
    pub(super) head: Option<HeadNals>,
    pub(super) ptses: Vec<u64>,
    /// H.264 / HEVC that opens mid-GOP: the access units before its first
    /// random-access point. `None` when the first access unit is one, or none
    /// comes within [`crate::demux::hdr::COLOUR_WINDOW_ACCESS_UNITS`].
    pub(super) leading: Option<LeadingSkip>,
}

/// The access units a stream opens with before its first random-access
/// point (an IDR; for HEVC any IRAP). They reference pictures the stream
/// does not carry, so no decoder can decode them: ffmpeg's decoders drop them,
/// NVDEC drops them, and rivet's native decoder refuses a stream that starts
/// with one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LeadingSkip {
    /// How many access units come before the first random-access point.
    pub(super) units: usize,
    /// The random-access point's PTS minus the earliest PTS among those
    /// units, 90 kHz: the time they would have filled, which the video keeps
    /// as a late start so what follows stays where it was.
    pub(super) delay_ticks: u64,
}

impl VideoStreamScan {
    /// What the stream's dimensions and pixel format are read from: the
    /// colour window when it found an SPS — a stream cut mid-GOP has none in
    /// its first access unit — else the first access unit.
    pub(super) fn parameter_au(&self) -> Option<&Vec<u8>> {
        match &self.head {
            Some(head) if head.has_sps => Some(&head.annexb),
            _ => self.first_au.as_ref(),
        }
    }

    /// What the stream's colour is read from: the colour window, else the
    /// first access unit.
    pub(super) fn colour_au(&self) -> Option<&[u8]> {
        self.head
            .as_ref()
            .map(|head| head.annexb.as_slice())
            .or(self.first_au.as_deref())
    }
}

/// Walk TS packets on the active video PID and reassemble the first
/// complete access unit into a single contiguous byte buffer. "Complete"
/// = from the first PUSI on the target PID up to (but not including)
/// the second PUSI; if there's no second PUSI before EOF we return
/// whatever we've accumulated so far. Also collects up to
/// `max_pts_samples` successive PTSes off the video PID so the caller
/// can derive a frame rate from their inter-arrival span. For H.264 / HEVC
/// (`codec` `"h264"` / `"h265"`) the walk goes on, access unit by access unit,
/// until the colour window has the stream's first SPS and the first
/// random-access point has been seen ([`LeadingSkip`]), or the walk reaches
/// the window's bound ([`crate::demux::hdr::COLOUR_WINDOW_ACCESS_UNITS`]).
///
/// Used by the streaming demuxer's init path to populate
/// `StreamInfo.width` / `.height` from the codec's SPS (H.264 / HEVC)
/// or sequence header (MPEG-2) — AND a correct `frame_rate` from the
/// PTS window — before any downstream consumer reads `header()`.
/// Walks the same packets `next_video_sample` would walk later; state
/// is local to this fn so the main walk state isn't disturbed.
pub(super) fn scan_first_video_au(
    data: &[u8],
    packets: usize,
    packet_stride: usize,
    prefix_len: usize,
    video_pid: u16,
    max_pts_samples: usize,
    codec: &str,
) -> VideoStreamScan {
    let mut accumulator: Vec<u8> = Vec::new();
    let mut first_au: Option<Vec<u8>> = None;
    let mut window = ColourWindow::new(codec);
    let mut ptses: Vec<u64> = Vec::new();
    // Inside an access unit being collected, and the PTS of its PES.
    let mut in_au = false;
    let mut au_pts: Option<u64> = None;
    // The search for the first random-access point: access units closed so
    // far, the earliest PTS before it, and where it was found.
    let nal_codec = match codec {
        "h264" => Some(crate::nal_mux::NalMuxCodec::H264),
        "h265" => Some(crate::nal_mux::NalMuxCodec::H265),
        _ => None,
    };
    let mut units = 0usize;
    let mut leading_min_pts: Option<u64> = None;
    let mut irap: Option<(usize, Option<u64>)> = None;
    let searching = |irap: &Option<(usize, Option<u64>)>, units: usize| {
        nal_codec.is_some()
            && irap.is_none()
            && units < crate::demux::hdr::COLOUR_WINDOW_ACCESS_UNITS
    };
    // Access units are wanted until the first is in hand, the colour window
    // (if any) is closed and the random-access point found.
    let wanting = |first_au: &Option<Vec<u8>>,
                   window: &Option<ColourWindow>,
                   irap: &Option<(usize, Option<u64>)>,
                   units: usize| {
        first_au.is_none()
            || window.as_ref().is_some_and(|w| !w.is_closed())
            || searching(irap, units)
    };
    let mut close = |au: Vec<u8>,
                     pts: Option<u64>,
                     first_au: &mut Option<Vec<u8>>,
                     window: &mut Option<ColourWindow>,
                     irap: &mut Option<(usize, Option<u64>)>,
                     units: &mut usize| {
        if searching(irap, *units) {
            if nal_codec.is_some_and(|c| crate::nal_mux::sample_is_keyframe(&au, c)) {
                *irap = Some((*units, pts));
            } else if let Some(p) = pts {
                leading_min_pts = Some(leading_min_pts.map_or(p, |m| m.min(p)));
            }
        }
        *units += 1;
        if let Some(w) = window.as_mut() {
            w.push(&au);
        }
        if first_au.is_none() {
            *first_au = Some(au);
        }
    };
    for i in 0..packets {
        let start = i * packet_stride + prefix_len;
        let pkt = &data[start..start + TS_PACKET];
        if pkt[0] != TS_SYNC {
            continue;
        }
        let pid = (((pkt[1] & 0x1F) as u16) << 8) | pkt[2] as u16;
        if pid != video_pid {
            continue;
        }
        let pusi = pkt[1] & 0x40 != 0;
        let scramble = (pkt[3] >> 6) & 0x03;
        if scramble != 0 {
            continue;
        } // encrypted; skip probe
        let adaptation = (pkt[3] >> 4) & 0x03;
        let has_payload = adaptation & 0x01 != 0;
        let has_adaptation = adaptation & 0x02 != 0;
        if !has_payload {
            continue;
        }
        let mut offset = 4usize;
        if has_adaptation {
            if offset >= TS_PACKET {
                continue;
            }
            let adap_len = pkt[offset] as usize;
            offset += 1 + adap_len;
            if offset > TS_PACKET {
                continue;
            }
        }
        if offset >= TS_PACKET {
            continue;
        }
        let payload = &pkt[offset..];

        if pusi {
            // A PUSI closes the access unit being collected.
            if in_au {
                in_au = false;
                let au = std::mem::take(&mut accumulator);
                close(
                    au,
                    au_pts,
                    &mut first_au,
                    &mut window,
                    &mut irap,
                    &mut units,
                );
            }
            if let Some((es_start, pts)) = parse_pes_header(payload) {
                if let Some(p) = pts
                    && ptses.len() < max_pts_samples
                {
                    ptses.push(p);
                }
                if wanting(&first_au, &window, &irap, units) {
                    if es_start < payload.len() {
                        accumulator.extend_from_slice(&payload[es_start..]);
                    }
                    in_au = true;
                    au_pts = pts;
                }
            }
        } else if in_au {
            accumulator.extend_from_slice(payload);
        }

        // Early exit once every target is hit.
        if !wanting(&first_au, &window, &irap, units) && ptses.len() >= max_pts_samples {
            break;
        }
    }
    // EOF with an access unit still open — take whatever's accumulated.
    if in_au && !accumulator.is_empty() {
        close(
            accumulator,
            au_pts,
            &mut first_au,
            &mut window,
            &mut irap,
            &mut units,
        );
    }
    let leading = match irap {
        Some((at, irap_pts)) if at > 0 => Some(LeadingSkip {
            units: at,
            delay_ticks: irap_pts
                .zip(leading_min_pts)
                .map_or(0, |(irap_pts, earliest)| irap_pts.saturating_sub(earliest)),
        }),
        _ => None,
    };
    VideoStreamScan {
        first_au,
        head: window.map(|w| w.finish("ts")),
        ptses,
        leading,
    }
}
