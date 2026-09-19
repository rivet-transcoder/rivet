//! Where a transport stream's clock starts again.
//!
//! A program's timestamps run on one time base until a system time-base
//! discontinuity (ISO/IEC 13818-1 §2.4.3.5): a broadcast splice, or two
//! recordings concatenated byte for byte. The multiplexer marks it with the
//! `discontinuity_indicator` of the PCR PID's adaptation field, and from there
//! the PCR — and the PTSes of the PES packets that follow — belong to a new
//! time base. A plain `cat a.ts b.ts` marks nothing, but its PCR jumps.
//!
//! [`time_base_breaks`] finds both on the PCR PID. A stream's timeline is then
//! cut ([`Segmenter`]) into [`Stretch`]es at the first PES packet after a
//! break, and at a PTS that jumps as no continuous stream does — more than ten
//! seconds on, or more than one second back (ffmpeg's `-dts_delta_threshold`
//! is the same ten seconds) — for a stream whose program carries no PCR, or
//! whose timestamps jump on their own.

use super::clock::{PTS_HZ, unwrap_pts};
use super::{TS_PACKET, TS_SYNC};

/// A forward PTS / PCR jump no continuous stream makes, 90 kHz.
const JUMP_FORWARD: i64 = 10 * PTS_HZ as i64;
/// A backward one: decoding order puts a PTS behind the one before it by a
/// few frames at most.
const JUMP_BACK: i64 = PTS_HZ as i64;

/// The `PCR_PID` of the program whose PMT is on `pmt_pid`.
pub(super) fn pcr_pid(
    data: &[u8],
    (packets, packet_stride, prefix_len): (usize, usize, usize),
    pmt_pid: u16,
) -> Option<u16> {
    (0..packets).find_map(|i| {
        let start = i * packet_stride + prefix_len;
        let pkt = &data[start..start + TS_PACKET];
        let pid = (((pkt[1] & 0x1F) as u16) << 8) | pkt[2] as u16;
        if pkt[0] != TS_SYNC || pid != pmt_pid || pkt[1] & 0x40 == 0 {
            return None;
        }
        super::ts_psi_payload(pkt).and_then(super::pat_pmt::parse_pmt_pcr_pid)
    })
}

/// The adaptation field of a packet, after its length byte; `None` when it
/// has none (or only the length byte).
fn adaptation_field(pkt: &[u8]) -> Option<&[u8]> {
    if pkt[3] & 0x20 == 0 {
        return None;
    }
    let len = pkt[4] as usize;
    (len > 0).then(|| &pkt[5..(5 + len).min(TS_PACKET)])
}

/// The packet's PCR base (90 kHz; the 27 MHz extension dropped), when its
/// adaptation field carries one.
fn pcr(pkt: &[u8]) -> Option<u64> {
    let af = adaptation_field(pkt)?;
    if af[0] & 0x10 == 0 || af.len() < 7 {
        return None;
    }
    Some(
        (u64::from(af[1]) << 25)
            | (u64::from(af[2]) << 17)
            | (u64::from(af[3]) << 9)
            | (u64::from(af[4]) << 1)
            | (u64::from(af[5]) >> 7),
    )
}

/// The packet indices at which the program's time base breaks: a packet on
/// the PCR PID with its `discontinuity_indicator` set, or whose PCR jumps back
/// or more than ten seconds on from the PCR before it (unwrapped across the
/// 33-bit wrap). Ascending. The indicator breaks only a time base some PCR has
/// set: a stream whose very first packets carry it (a cut from a longer one,
/// an HLS segment) starts on one time base, not two.
pub(super) fn time_base_breaks(
    data: &[u8],
    (packets, packet_stride, prefix_len): (usize, usize, usize),
    pcr_pid: Option<u16>,
) -> Vec<usize> {
    let Some(pcr_pid) = pcr_pid else {
        return Vec::new();
    };
    let mut breaks = Vec::new();
    let mut last: Option<i64> = None;
    for i in 0..packets {
        let start = i * packet_stride + prefix_len;
        let pkt = &data[start..start + TS_PACKET];
        let pid = (((pkt[1] & 0x1F) as u16) << 8) | pkt[2] as u16;
        if pkt[0] != TS_SYNC || pid != pcr_pid {
            continue;
        }
        let flagged = last.is_some() && adaptation_field(pkt).is_some_and(|af| af[0] & 0x80 != 0);
        let value = pcr(pkt);
        let jumped = match (last, value) {
            (Some(prev), Some(v)) => {
                let v = unwrap_pts(v, prev);
                v < prev || v - prev > JUMP_FORWARD
            }
            _ => false,
        };
        if flagged || jumped {
            breaks.push(i);
            // A new time base: the next PCR is not compared with the old one.
            last = None;
        }
        if let Some(v) = value {
            last = Some(match last {
                Some(prev) => unwrap_pts(v, prev),
                None => v as i64,
            });
        }
    }
    breaks
}

/// Which stretch of one time base a PES packet is in: the time-base breaks
/// before it, and the PTS jumps in its own stream since the last of them.
/// Every stream of a program counts the same breaks, so the video's stretch
/// and the audio's after a break are the same `Stretch`; a jump is a stream's
/// own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub(super) struct Stretch {
    pub(super) breaks: usize,
    pub(super) jumps: usize,
}

/// Cuts one stream's PES packets into [`Stretch`]es: a PES opens a new one
/// when a time-base break lies between it and the PES before it, or when its
/// PTS jumps from that PES's by more than ten seconds on or one second back.
pub(super) struct Segmenter<'a> {
    breaks: &'a [usize],
    /// The stretch of the PES before.
    stretch: Stretch,
    /// The previous PES's PTS, unwrapped on its stretch's timeline.
    prev: Option<i64>,
}

impl<'a> Segmenter<'a> {
    pub(super) fn new(breaks: &'a [usize]) -> Self {
        Self {
            breaks,
            stretch: Stretch::default(),
            prev: None,
        }
    }

    /// The PES starting in packet `packet` with PTS `pts` (as the stream has
    /// it): the stretch it is in, and its PTS unwrapped on that stretch's
    /// timeline.
    pub(super) fn place(&mut self, packet: usize, pts: Option<u64>) -> (Stretch, Option<i64>) {
        let breaks = self.breaks.partition_point(|&b| b <= packet);
        if breaks != self.stretch.breaks {
            self.stretch = Stretch { breaks, jumps: 0 };
            self.prev = None;
        }
        let unwrapped = pts.map(|p| self.prev.map_or(p as i64, |prev| unwrap_pts(p, prev)));
        if let (Some(prev), Some(p)) = (self.prev, unwrapped)
            && (p - prev > JUMP_FORWARD || prev - p > JUMP_BACK)
        {
            self.stretch.jumps += 1;
            self.prev = None;
        }
        let placed = pts.map(|p| self.prev.map_or(p as i64, |prev| unwrap_pts(p, prev)));
        if placed.is_some() {
            self.prev = placed;
        }
        (self.stretch, placed)
    }
}
