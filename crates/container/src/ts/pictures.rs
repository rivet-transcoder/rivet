//! How many frames a decoder makes of a transport stream's video, counted from
//! the packets before anything is decoded.
//!
//! A transport stream has no frame count and no duration of its own; the
//! pipeline plans from them (HLS segments, the multi-GPU chunk grid, progress,
//! a thumbnail's position). One PES packet per access unit is the usual
//! carriage, so the count is the PES packets on the video PID — less the ones
//! the reader drops before the first random-access point, and the RASL
//! pictures a decoder starting at an HEVC CRA does not output (the caller
//! takes those off). An interlaced H.264 stream is the exception: a muxer may
//! carry each field of a field-coded frame in its own PES (ffmpeg does), and a
//! decoder pairs the two into one frame. There the count, and the frame rate
//! — which the PES timestamps would give per field — come from the slice
//! headers: every picture's first slice says whether it is a frame or a field,
//! which parity, and its `frame_num`.

use super::discontinuity::{Segmenter, Stretch};
use super::pes::parse_pes_header;
use super::{TS_PACKET, TS_SYNC};

/// What [`count_frames`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FrameCount {
    /// Frames a decoder outputs: one per PES packet, or for a field-coded
    /// stream one per frame picture and per pair of field pictures.
    pub(super) frames: u64,
    /// Whether any field picture was seen.
    pub(super) field_coded: bool,
    /// Of a field-coded stream, the PTSes (unwrapped, in stream order) of the
    /// first PES packets a frame starts in, up to the count asked for: what
    /// its frame rate is read from, where the PES rate counts fields.
    pub(super) frame_ptses: Vec<i64>,
    /// The stretches of one time base the video runs in, in order
    /// ([`super::discontinuity`]); one for a stream with no discontinuity.
    pub(super) segments: Vec<VideoSegment>,
}

/// One stretch of the video on a single time base.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct VideoSegment {
    /// Which stretch it is.
    pub(super) stretch: Stretch,
    /// Frames counted before it: where its first frame falls in the output.
    pub(super) frames_before: u64,
    /// The PTSes of its frames (those whose PES carries one), on its own
    /// unwrapped timeline, ascending: presentation order, which is the order
    /// the output presents them in.
    pub(super) ptses: Vec<i64>,
}

/// How an interlaced H.264 stream's slice headers are read, from its SPS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FieldSyntax {
    /// `log2_max_frame_num_minus4 + 4`: the width of `frame_num`.
    log2_max_frame_num: u32,
    /// `separate_colour_plane_flag`: a `colour_plane_id` precedes `frame_num`.
    separate_colour_plane: bool,
}

impl FieldSyntax {
    /// The first H.264 SPS in `annexb`, when it lets pictures be fields
    /// (`frame_mbs_only_flag` 0); `None` for a progressive stream, another
    /// codec, or no SPS.
    pub(super) fn from_annexb(codec: &str, annexb: &[u8]) -> Option<Self> {
        if codec != "h264" {
            return None;
        }
        let nal = h26x::nal::annexb_nals(annexb)
            .find(|nal| nal.first().is_some_and(|b| b & 0x1f == 7))?;
        let sps = h26x::h264::Sps::parse(&h26x::nal::unescape_rbsp(&nal[1..])).ok()?;
        (!sps.frame_mbs_only).then_some(Self {
            log2_max_frame_num: sps.log2_max_frame_num,
            separate_colour_plane: sps.separate_colour_plane,
        })
    }
}

/// Count the frames of the video on `video_pid`, from its PES packets after
/// the first `skip` (the access units the reader drops before a mid-GOP
/// stream's first random-access point). `fields` is the stream's
/// [`FieldSyntax`] when it may code fields: then the elementary stream is
/// walked for its slice headers, and `max_ptses` frame-starting PTSes are
/// kept for the frame rate. Otherwise only the PES headers are read. The
/// walk also cuts the stream at the program's time-base `breaks` (and at a
/// PTS that jumps) into [`VideoSegment`]s.
#[allow(clippy::too_many_arguments)]
pub(super) fn count_frames(
    data: &[u8],
    packets: usize,
    packet_stride: usize,
    prefix_len: usize,
    video_pid: u16,
    skip: usize,
    fields: Option<FieldSyntax>,
    max_ptses: usize,
    breaks: &[usize],
) -> FrameCount {
    let mut pes_seen = 0usize;
    let mut frames = 0u64;
    let mut walker = fields.map(|syntax| FieldWalker::new(syntax, max_ptses));
    let mut segmenter = Segmenter::new(breaks);
    let mut segments: Vec<VideoSegment> = Vec::new();
    let mut in_counted_pes = false;
    for i in 0..packets {
        let start = i * packet_stride + prefix_len;
        let pkt = &data[start..start + TS_PACKET];
        if pkt[0] != TS_SYNC {
            continue;
        }
        let pid = (((pkt[1] & 0x1F) as u16) << 8) | pkt[2] as u16;
        if pid != video_pid || (pkt[3] >> 6) & 0x03 != 0 {
            continue;
        }
        let adaptation = (pkt[3] >> 4) & 0x03;
        if adaptation & 0x01 == 0 {
            continue;
        }
        let mut offset = 4usize;
        if adaptation & 0x02 != 0 {
            offset += 1 + pkt[offset] as usize;
        }
        if offset >= TS_PACKET {
            continue;
        }
        let payload = &pkt[offset..];
        if pkt[1] & 0x40 != 0 {
            let Some((es_start, pts)) = parse_pes_header(payload) else {
                in_counted_pes = false;
                continue;
            };
            let (stretch, pts) = segmenter.place(i, pts);
            pes_seen += 1;
            in_counted_pes = pes_seen > skip;
            if !in_counted_pes {
                continue;
            }
            if segments.last().is_none_or(|s| s.stretch != stretch) {
                segments.push(VideoSegment {
                    stretch,
                    frames_before: walker.as_ref().map_or(frames, |w| w.frames),
                    ptses: Vec::new(),
                });
            }
            let current = segments.last_mut().expect("pushed above");
            match walker.as_mut() {
                None => {
                    frames += 1;
                    current.ptses.extend(pts);
                }
                Some(w) => {
                    w.start_pes(pts);
                    w.feed(payload.get(es_start..).unwrap_or(&[]));
                    current.ptses.append(&mut w.started);
                }
            }
        } else if in_counted_pes && let Some(w) = walker.as_mut() {
            w.feed(payload);
            if let Some(current) = segments.last_mut() {
                current.ptses.append(&mut w.started);
            }
        }
    }
    if let Some(w) = walker.as_mut() {
        w.flush();
        if let Some(current) = segments.last_mut() {
            current.ptses.append(&mut w.started);
        }
    }
    for segment in &mut segments {
        segment.ptses.sort_unstable();
    }
    match walker {
        None => FrameCount {
            frames,
            field_coded: false,
            frame_ptses: Vec::new(),
            segments,
        },
        Some(w) => FrameCount {
            segments,
            ..w.finish()
        },
    }
}

/// The NAL units after a start code, read far enough for a slice header's
/// `field_pic_flag`: the bytes of each are gathered across PES and TS packet
/// boundaries, up to [`FieldWalker::HEAD`].
struct FieldWalker {
    syntax: FieldSyntax,
    /// Consecutive zero bytes just seen (a start code is two or more, then 1).
    zeros: u32,
    /// The head of the NAL unit being gathered, when one is.
    head: Option<Vec<u8>>,
    /// The PTS of the PES packet being walked, until its first picture takes it.
    pes_pts: Option<i64>,
    /// A first field waiting for its second: its `frame_num` and parity.
    open_field: Option<(u32, bool)>,
    frames: u64,
    field_coded: bool,
    frame_ptses: Vec<i64>,
    max_ptses: usize,
    /// The PTSes of the frames started since the caller last took them.
    started: Vec<i64>,
}

impl FieldWalker {
    /// Bytes of a NAL unit read: its header byte, then enough of a slice
    /// header for `field_pic_flag` and `bottom_field_flag` — five exp-Golomb
    /// and fixed fields, far inside this even with emulation prevention.
    const HEAD: usize = 24;

    fn new(syntax: FieldSyntax, max_ptses: usize) -> Self {
        Self {
            syntax,
            zeros: 0,
            head: None,
            pes_pts: None,
            open_field: None,
            frames: 0,
            field_coded: false,
            frame_ptses: Vec::new(),
            max_ptses,
            started: Vec::new(),
        }
    }

    fn start_pes(&mut self, pts: Option<i64>) {
        self.pes_pts = pts;
    }

    fn feed(&mut self, bytes: &[u8]) {
        for &b in bytes {
            if let Some(head) = self.head.as_mut() {
                head.push(b);
                if head.len() == Self::HEAD {
                    let head = self.head.take().unwrap_or_default();
                    self.nal(&head);
                }
            }
            if self.zeros >= 2 && b == 1 {
                // A start code: whatever was being gathered ends here (it was
                // shorter than HEAD, like an access unit delimiter).
                if let Some(head) = self.head.take() {
                    self.nal(&head[..head.len().saturating_sub(3)]);
                }
                self.head = Some(Vec::with_capacity(Self::HEAD));
            }
            self.zeros = if b == 0 { self.zeros + 1 } else { 0 };
        }
    }

    /// One NAL unit's head: a slice NAL (1, 5) whose `first_mb_in_slice` is 0
    /// opens a picture.
    fn nal(&mut self, head: &[u8]) {
        let Some(&header) = head.first() else { return };
        if !matches!(header & 0x1f, 1 | 5) {
            return;
        }
        let rbsp = h26x::nal::unescape_rbsp(&head[1..]);
        let mut r = Bits::new(&rbsp);
        let picture = (|| {
            if r.ue()? != 0 {
                return None;
            }
            r.ue()?; // slice_type
            r.ue()?; // pic_parameter_set_id
            if self.syntax.separate_colour_plane {
                r.bits(2)?; // colour_plane_id
            }
            let frame_num = r.bits(self.syntax.log2_max_frame_num)?;
            let field = r.bits(1)? == 1;
            let bottom = field && r.bits(1)? == 1;
            Some((frame_num, field, bottom))
        })();
        let Some((frame_num, field, bottom)) = picture else {
            return;
        };
        // A field completes the frame its opposite-parity partner of the same
        // frame_num opened; anything else starts a frame.
        let second_field = field
            && self
                .open_field
                .is_some_and(|(num, parity)| num == frame_num && parity != bottom);
        let pts = self.pes_pts.take();
        if second_field {
            self.open_field = None;
            return;
        }
        self.frames += 1;
        self.field_coded |= field;
        self.open_field = field.then_some((frame_num, bottom));
        if let Some(p) = pts {
            self.started.push(p);
            if self.frame_ptses.len() < self.max_ptses {
                self.frame_ptses.push(p);
            }
        }
    }

    /// Reads the NAL unit the stream ends in.
    fn flush(&mut self) {
        if let Some(head) = self.head.take() {
            self.nal(&head);
        }
    }

    fn finish(self) -> FrameCount {
        FrameCount {
            frames: self.frames,
            field_coded: self.field_coded,
            frame_ptses: self.frame_ptses,
            segments: Vec::new(),
        }
    }
}

/// An MSB-first bit reader with exp-Golomb, over an RBSP.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn bits(&mut self, n: u32) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            let byte = *self.data.get(self.pos / 8)?;
            v = (v << 1) | u32::from((byte >> (7 - self.pos % 8)) & 1);
            self.pos += 1;
        }
        Some(v)
    }

    fn ue(&mut self) -> Option<u32> {
        let mut leading = 0u32;
        while self.bits(1)? == 0 {
            leading += 1;
            if leading > 31 {
                return None;
            }
        }
        Some((1u32 << leading) - 1 + self.bits(leading)?)
    }
}
