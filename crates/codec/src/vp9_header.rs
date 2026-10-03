//! The start of a VP9 frame's uncompressed header, read far enough to know
//! what the frame does to the stream: whether it is a key frame, whether it
//! is shown, whether it only re-shows a reference (`show_existing_frame`),
//! which of the eight reference slots it refreshes, and its frame size —
//! the last of which an inter frame may take from a reference
//! (`frame_size_with_refs`), so a reader that wants it must follow the
//! reference slots. [`RefSizes`] does that.
//!
//! Written from the VP9 Bitstream & Decoding Process Specification v0.6,
//! §6.2 (`uncompressed_header`, `frame_sync_code`, `color_config`,
//! `frame_size`, `render_size`, `frame_size_with_refs`,
//! `read_interpolation_filter`, `loop_filter_params`, `quantization_params`
//! and the first bit of `segmentation_params`) and §7.2 (their semantics).
//! Reading stops at `segmentation_enabled`; nothing after it is needed
//! here.
//!
//! Users: the QSV VP9 encoder (key frames, hidden frames) and the guard in
//! front of the hardware VP9 decoders (`decode::vp9_hw_guard`), which needs
//! to see `show_existing_frame` and frame-size changes before a decoder
//! does.

/// What [`peek`] read from one frame (not a superframe: split those first,
/// `vp9::superframe::split`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameHeader {
    /// `show_existing_frame = 1`: the frame shows reference slot `slot`
    /// again and decodes nothing.
    ShowExisting { slot: u8 },
    /// A coded frame.
    Coded(CodedFrame),
}

/// The fields of a coded frame's header [`peek`] reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodedFrame {
    /// `Profile` (0..=3).
    pub profile: u8,
    /// `frame_type == KEY_FRAME`.
    pub key: bool,
    /// `intra_only` (always false on a key frame or a shown frame's
    /// `show_frame = 1` path, where the spec infers it 0).
    pub intra_only: bool,
    /// `show_frame`.
    pub show: bool,
    /// The reference slots the frame overwrites: every slot for a key frame
    /// (§7.2: `refresh_frame_flags = 0xFF`), else `refresh_frame_flags`.
    pub refresh: u8,
    /// The frame size, or where to find it.
    pub size: FrameSize,
    /// `error_resilient_mode`.
    pub error_resilient: bool,
    /// `segmentation_enabled` (§6.2.11).
    pub segmentation: bool,
    /// The colour config a key frame or an intra-only frame carries (an
    /// intra-only frame of profile 0 implies 8-bit 4:2:0); `None` on an
    /// inter frame, which inherits it.
    pub color: Option<ColorConfig>,
}

/// What `color_config()` (§6.2.2) says about the picture format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorConfig {
    /// 8, 10 or 12.
    pub bit_depth: u8,
    /// `subsampling_x`, `subsampling_y`: (1, 1) is 4:2:0.
    pub subsampling: (u8, u8),
    /// `color_space == CS_RGB`.
    pub rgb: bool,
}

/// Where a coded frame's size comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameSize {
    /// Coded in the header (`frame_size()`).
    Explicit { width: u32, height: u32 },
    /// `found_ref = 1`: the size of the reference in slot `slot`.
    FromRef { slot: u8 },
}

/// A big-endian bit reader over the header bytes (§4.9 `f(n)`).
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Bits<'_> {
    fn f(&mut self, n: u32) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            let byte = *self.data.get(self.pos / 8)?;
            v = (v << 1) | u32::from((byte >> (7 - self.pos % 8)) & 1);
            self.pos += 1;
        }
        Some(v)
    }
}

/// `CS_RGB` (§7.2, `color_space`).
const CS_RGB: u32 = 7;
/// `frame_sync_code`: 0x49, 0x83, 0x42 (§6.2.1, §7.2.1).
const SYNC_CODE: u32 = 0x49_83_42;

/// `color_config()` (§6.2.2).
fn color_config(b: &mut Bits, profile: u8) -> Option<ColorConfig> {
    let bit_depth = if profile >= 2 {
        if b.f(1)? == 1 { 12 } else { 10 } // ten_or_twelve_bit
    } else {
        8
    };
    let color_space = b.f(3)?;
    let rgb = color_space == CS_RGB;
    let subsampling = if !rgb {
        b.f(1)?; // color_range
        if profile == 1 || profile == 3 {
            let ss_x = b.f(1)? as u8;
            let ss_y = b.f(1)? as u8;
            b.f(1)?; // reserved_zero
            (ss_x, ss_y)
        } else {
            (1, 1)
        }
    } else {
        if profile == 1 || profile == 3 {
            b.f(1)?; // reserved_zero
        }
        (0, 0)
    };
    Some(ColorConfig { bit_depth, subsampling, rgb })
}

/// `frame_size()` (§6.2.3): `frame_width_minus_1`, `frame_height_minus_1`.
fn frame_size(b: &mut Bits) -> Option<FrameSize> {
    let width = b.f(16)? + 1;
    let height = b.f(16)? + 1;
    Some(FrameSize::Explicit { width, height })
}

/// Read the start of one frame's uncompressed header. `None` when the bytes
/// are not a VP9 frame header (wrong `frame_marker`, a key or intra-only
/// frame without the sync code, too short).
pub fn peek(frame: &[u8]) -> Option<FrameHeader> {
    let mut b = Bits { data: frame, pos: 0 };
    if b.f(2)? != 2 {
        return None; // frame_marker
    }
    let profile_low = b.f(1)?;
    let profile_high = b.f(1)?;
    let profile = ((profile_high << 1) + profile_low) as u8;
    if profile == 3 {
        b.f(1)?; // reserved_zero
    }
    if b.f(1)? == 1 {
        let slot = b.f(3)? as u8; // frame_to_show_map_idx
        return Some(FrameHeader::ShowExisting { slot });
    }
    let key = b.f(1)? == 0; // frame_type: KEY_FRAME = 0
    let show = b.f(1)? == 1;
    let error_resilient = b.f(1)? == 1;
    let (intra_only, refresh, size, color) = if key {
        if b.f(24)? != SYNC_CODE {
            return None;
        }
        let color = color_config(&mut b, profile)?;
        let size = frame_size(&mut b)?;
        render_size(&mut b)?;
        (false, 0xFF, size, Some(color))
    } else {
        let intra_only = if show { false } else { b.f(1)? == 1 };
        if !error_resilient {
            b.f(2)?; // reset_frame_context
        }
        if intra_only {
            if b.f(24)? != SYNC_CODE {
                return None;
            }
            let color = if profile > 0 {
                color_config(&mut b, profile)?
            } else {
                ColorConfig { bit_depth: 8, subsampling: (1, 1), rgb: false }
            };
            let refresh = b.f(8)? as u8;
            let size = frame_size(&mut b)?;
            render_size(&mut b)?;
            (true, refresh, size, Some(color))
        } else {
            let refresh = b.f(8)? as u8;
            let mut ref_slots = [0u8; 3];
            for slot in &mut ref_slots {
                *slot = b.f(3)? as u8; // ref_frame_idx[i]
                b.f(1)?; // ref_frame_sign_bias
            }
            // frame_size_with_refs(): the first reference whose found_ref is
            // set gives the size; none set, and it is coded. render_size()
            // follows either way.
            let mut size = None;
            for slot in ref_slots {
                if b.f(1)? == 1 {
                    size = Some(FrameSize::FromRef { slot });
                    break;
                }
            }
            let size = match size {
                Some(s) => s,
                None => frame_size(&mut b)?,
            };
            render_size(&mut b)?;
            b.f(1)?; // allow_high_precision_mv
            if b.f(1)? == 0 {
                b.f(2)?; // is_filter_switchable = 0: raw_interpolation_filter
            }
            (false, refresh, size, None)
        }
    };
    if !error_resilient {
        b.f(2)?; // refresh_frame_context, frame_parallel_decoding_mode
    }
    b.f(2)?; // frame_context_idx
    // loop_filter_params() (§6.2.8).
    b.f(9)?; // loop_filter_level, loop_filter_sharpness
    if b.f(1)? == 1 && b.f(1)? == 1 {
        // loop_filter_delta_enabled, loop_filter_delta_update: four ref
        // deltas and two mode deltas, each su(6) behind an update bit.
        for _ in 0..6 {
            if b.f(1)? == 1 {
                b.f(7)?;
            }
        }
    }
    // quantization_params() (§6.2.9): base_q_idx, then three read_delta_q().
    b.f(8)?;
    for _ in 0..3 {
        if b.f(1)? == 1 {
            b.f(5)?; // su(4)
        }
    }
    let segmentation = b.f(1)? == 1;
    Some(FrameHeader::Coded(CodedFrame { profile, key, intra_only, show, refresh, size, error_resilient, segmentation, color }))
}

/// `render_size()` (§6.2.4), read and discarded.
fn render_size(b: &mut Bits) -> Option<()> {
    if b.f(1)? == 1 {
        b.f(32)?; // render_width_minus_1, render_height_minus_1
    }
    Some(())
}

/// Whether a packet — one frame or a superframe — opens with a key frame:
/// the sync sample test. A superframe whose first frame is a key frame is a
/// random-access point (§B.3: the frames of a superframe decode in order).
pub fn packet_is_keyframe(packet: &[u8]) -> bool {
    let frames = vp9::superframe::split(packet);
    matches!(frames.first().and_then(|f| peek(f)), Some(FrameHeader::Coded(CodedFrame { key: true, .. })))
}

/// Whether a packet shows a picture: some frame in it has `show_frame = 1`
/// or is a `show_existing_frame`.
pub fn packet_shows(packet: &[u8]) -> bool {
    vp9::superframe::split(packet).iter().any(|f| match peek(f) {
        Some(FrameHeader::ShowExisting { .. }) => true,
        Some(FrameHeader::Coded(c)) => c.show,
        None => false,
    })
}

/// The sizes of the eight reference slots, followed frame by frame, so an
/// inter frame's `found_ref` size can be resolved.
#[derive(Debug, Clone, Default)]
pub struct RefSizes {
    slots: [Option<(u32, u32)>; 8],
}

impl RefSizes {
    /// Apply one frame's header: its size (resolved through the slots), and
    /// the slots it refreshes. Returns the frame's size, or `None` for a
    /// `show_existing_frame` (which decodes nothing) or a size taken from a
    /// slot nothing has filled yet.
    pub fn apply(&mut self, header: &FrameHeader) -> Option<(u32, u32)> {
        let FrameHeader::Coded(c) = header else { return None };
        let size = match c.size {
            FrameSize::Explicit { width, height } => Some((width, height)),
            FrameSize::FromRef { slot } => self.slots[usize::from(slot)],
        };
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if c.refresh & (1 << i) != 0 {
                *slot = size;
            }
        }
        size
    }

    /// The size in reference slot `slot`, if one has been filled.
    pub fn slot(&self, slot: u8) -> Option<(u32, u32)> {
        self.slots.get(usize::from(slot)).copied().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame from rivet's own encoder: the first is a key frame of the
    /// configured size, shown, refreshing every slot; the second an inter
    /// frame whose size comes from a reference.
    #[test]
    fn reads_rivets_own_frames() {
        let (w, h) = (66u32, 40u32);
        let mut enc = vp9::Encoder::new(vp9::Config::new(w, h));
        let frame = vp9::Frame::new(w, h, 8, vp9::ChromaFormat::Yuv420);
        let key = enc.encode(&frame).expect("encode");
        let inter = enc.encode(&frame).expect("encode");
        let FrameHeader::Coded(k) = peek(&key).expect("key header") else { panic!("not coded") };
        assert!(k.key && k.show && !k.intra_only);
        assert_eq!(k.refresh, 0xFF);
        assert_eq!(k.size, FrameSize::Explicit { width: w, height: h });
        assert!(packet_is_keyframe(&key) && packet_shows(&key));

        let FrameHeader::Coded(p) = peek(&inter).expect("inter header") else { panic!("not coded") };
        assert!(!p.key && p.show);
        assert!(!packet_is_keyframe(&inter));
        let mut refs = RefSizes::default();
        assert_eq!(refs.apply(&FrameHeader::Coded(k)), Some((w, h)));
        assert_eq!(refs.apply(&FrameHeader::Coded(p)), Some((w, h)));
    }

    /// Hand-built headers, bit by bit from §6.2: a profile 0 key frame of
    /// 352x288; a hidden inter frame refreshing slot 2; a
    /// `show_existing_frame` of slot 5; and a profile 3 key frame (the
    /// reserved bit after the profile, the 4:4:4 colour config).
    #[test]
    fn hand_built_headers() {
        fn pack(bits: &[(u32, u32)]) -> Vec<u8> {
            let mut out = Vec::new();
            let mut n = 0usize;
            for &(v, len) in bits {
                for i in (0..len).rev() {
                    if n % 8 == 0 {
                        out.push(0);
                    }
                    if (v >> i) & 1 == 1 {
                        *out.last_mut().unwrap() |= 0x80 >> (n % 8);
                    }
                    n += 1;
                }
            }
            out.extend([0, 0, 0, 0]);
            out
        }
        // marker, profile_low, profile_high, show_existing, frame_type=KEY,
        // show_frame, error_res, sync, color_space=BT709(2), color_range,
        // width-1, height-1.
        // Then render_size (same), refresh_frame_context +
        // frame_parallel_decoding_mode, frame_context_idx, loop filter level
        // + sharpness, delta_enabled = 1, delta_update = 1 with one ref delta
        // (su(6)) updated, base_q_idx, one delta_q coded (su(4)), and
        // segmentation_enabled = 1.
        let key = pack(&[
            (2, 2), (0, 1), (0, 1), (0, 1), (0, 1), (1, 1), (0, 1), (SYNC_CODE, 24), (2, 3), (0, 1), (351, 16), (287, 16),
            (0, 1), (0, 2), (0, 2), (0, 9), (1, 1), (1, 1), (1, 1), (0b1000001, 7), (0, 1), (0, 1), (0, 1), (0, 1),
            (60, 8), (1, 1), (0b00011, 5), (0, 1), (0, 1), (1, 1),
        ]);
        assert_eq!(
            peek(&key),
            Some(FrameHeader::Coded(CodedFrame {
                profile: 0,
                key: true,
                intra_only: false,
                show: true,
                refresh: 0xFF,
                size: FrameSize::Explicit { width: 352, height: 288 },
                error_resilient: false,
                segmentation: true,
                color: Some(ColorConfig { bit_depth: 8, subsampling: (1, 1), rgb: false }),
            }))
        );
        // Hidden inter frame: show_frame 0, intra_only 0, reset_frame_context,
        // refresh 0b100, three refs (idx, sign), found_ref on the second.
        let hidden = pack(&[
            (2, 2), (0, 1), (0, 1), (0, 1), (1, 1), (0, 1), (0, 1),
            (0, 1), (0, 2), (0b100, 8),
            (0, 3), (0, 1), (1, 3), (0, 1), (2, 3), (0, 1),
            (0, 1), (1, 1),
            // render_size, allow_high_precision_mv, is_filter_switchable = 0
            // + raw_interpolation_filter, the context bits, loop filter
            // (no deltas), base_q_idx, no delta_q, segmentation_enabled = 0.
            (0, 1), (1, 1), (0, 1), (2, 2), (0, 2), (0, 2), (0, 9), (0, 1), (40, 8), (0, 3), (0, 1),
        ]);
        let h = peek(&hidden).expect("hidden");
        assert_eq!(
            h,
            FrameHeader::Coded(CodedFrame {
                profile: 0,
                key: false,
                intra_only: false,
                show: false,
                refresh: 0b100,
                size: FrameSize::FromRef { slot: 1 },
                error_resilient: false,
                segmentation: false,
                color: None,
            })
        );
        assert!(!packet_shows(&hidden));
        let se = pack(&[(2, 2), (0, 1), (0, 1), (1, 1), (5, 3)]);
        assert_eq!(peek(&se), Some(FrameHeader::ShowExisting { slot: 5 }));
        assert!(packet_shows(&se) && !packet_is_keyframe(&se));
        // Profile 3: low=1, high=1, reserved_zero, then as before with a
        // 12-bit 4:4:4 colour config (ten_or_twelve_bit, cs, range, ss_x,
        // ss_y, reserved).
        let p3 = pack(&[
            (2, 2), (1, 1), (1, 1), (0, 1), (0, 1), (0, 1), (1, 1), (0, 1),
            (SYNC_CODE, 24), (1, 1), (2, 3), (0, 1), (0, 1), (0, 1), (0, 1),
            (63, 16), (31, 16),
        ]);
        let Some(FrameHeader::Coded(c)) = peek(&p3) else { panic!("p3") };
        assert_eq!((c.profile, c.key, c.size), (3, true, FrameSize::Explicit { width: 64, height: 32 }));
        assert_eq!(c.color, Some(ColorConfig { bit_depth: 12, subsampling: (0, 0), rgb: false }));

        let mut refs = RefSizes::default();
        assert_eq!(refs.apply(&peek(&key).unwrap()), Some((352, 288)));
        assert_eq!(refs.apply(&h), Some((352, 288)));
        assert_eq!(refs.slot(2), Some((352, 288)));
        assert_eq!(refs.apply(&FrameHeader::ShowExisting { slot: 2 }), None);
    }

    #[test]
    fn not_vp9_is_none() {
        assert_eq!(peek(&[]), None);
        assert_eq!(peek(&[0x00, 0x00]), None);
        // A key frame header whose sync code is wrong.
        assert_eq!(peek(&[0x82, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]), None);
    }
}
