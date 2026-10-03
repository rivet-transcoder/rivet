//! VP8 / VP9 glue for the muxers: what a frame's uncompressed header says
//! (key frame, profile, depth, chroma), the `vpcC` configuration record of the
//! ISOBMFF binding, the VP9 level a stream fits, and the RFC 6381 `codecs`
//! string.
//!
//! Written from the WebM project's *VP Codec ISO Media File Format Binding*
//! (v1.0): `vp08` / `vp09` sample entries carry a `VPCodecConfigurationBox`
//! (`vpcC`, a version-1 FullBox) and the `codecs` parameter is
//! `vp09.PP.LL.DD.CC.cp.tc.mc.FF`; the VP9 levels are the table of the VP9
//! bitstream specification's Annex A.

use frame::ColorMetadata;

/// What a muxer needs from one VP8 / VP9 frame's uncompressed header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VpxFrameInfo {
    /// A key frame: a decoder may start here.
    pub key_frame: bool,
    /// VP9 profile (0 to 3); 0 for VP8.
    pub profile: u8,
    /// Luma bit depth (8, 10, 12). Known on VP9 key frames; 8 otherwise.
    pub bit_depth: u8,
    /// `(subsampling_x, subsampling_y)`; (1, 1) for 4:2:0.
    pub subsampling: (u8, u8),
    /// The colour range flag (VP9 key frames; VP8's `clamping_type` is not it).
    pub full_range: bool,
}

/// The uncompressed header of a VP8 frame (RFC 6386 §9.1): bit 0 of the
/// frame tag is 0 for a key frame. `None` for fewer than three bytes.
pub fn vp8_frame_info(frame: &[u8]) -> Option<VpxFrameInfo> {
    let tag = *frame.first()?;
    if frame.len() < 3 {
        return None;
    }
    Some(VpxFrameInfo {
        key_frame: tag & 1 == 0,
        profile: 0,
        bit_depth: 8,
        subsampling: (1, 1),
        full_range: false,
    })
}

/// The uncompressed header of a VP9 frame (VP9 bitstream §6.2) — or of the
/// last frame of a superframe, which is the one shown. `None` when the bytes
/// are not a VP9 frame. A `show_existing_frame` header is reported as an
/// inter frame.
pub fn vp9_frame_info(frame: &[u8]) -> Option<VpxFrameInfo> {
    let frame = last_superframe_frame(frame).unwrap_or(frame);
    let mut r = Bits { data: frame, pos: 0 };
    if r.read(2)? != 2 {
        return None;
    }
    let low = r.read(1)?;
    let high = r.read(1)?;
    let profile = ((high << 1) | low) as u8;
    if profile == 3 {
        r.read(1)?;
    }
    let mut info = VpxFrameInfo { key_frame: false, profile, bit_depth: 8, subsampling: (1, 1), full_range: false };
    if r.read(1)? == 1 {
        // show_existing_frame
        return Some(info);
    }
    let frame_type = r.read(1)?;
    r.read(1)?; // show_frame
    r.read(1)?; // error_resilient_mode
    if frame_type != 0 {
        return Some(info);
    }
    info.key_frame = true;
    if r.read(24)? != 0x49_83_42 {
        return None;
    }
    if profile >= 2 {
        info.bit_depth = if r.read(1)? == 1 { 12 } else { 10 };
    }
    let color_space = r.read(3)?;
    if color_space != 7 {
        // Not CS_RGB.
        info.full_range = r.read(1)? == 1;
        if profile == 1 || profile == 3 {
            info.subsampling = (r.read(1)? as u8, r.read(1)? as u8);
        }
    } else {
        info.full_range = true;
        info.subsampling = (0, 0);
    }
    Some(info)
}

/// The last frame of a VP9 superframe (Annex B), or `None` when `data` is not
/// one.
fn last_superframe_frame(data: &[u8]) -> Option<&[u8]> {
    let marker = *data.last()?;
    if marker & 0xe0 != 0xc0 {
        return None;
    }
    let frames = usize::from(marker & 0x7) + 1;
    let size_bytes = usize::from((marker >> 3) & 0x3) + 1;
    let index = 2 + size_bytes * frames;
    if data.len() < index || data[data.len() - index] != marker {
        return None;
    }
    let sizes = &data[data.len() - index + 1..data.len() - 1];
    let mut offset = 0usize;
    let mut last = None;
    for i in 0..frames {
        let mut size = 0usize;
        for b in 0..size_bytes {
            size |= usize::from(sizes[i * size_bytes + b]) << (8 * b);
        }
        last = Some((offset, size));
        offset += size;
    }
    let (start, size) = last?;
    data.get(start..start + size)
}

/// The lowest VP9 level (10 = 1.0 … 62 = 6.2) whose picture size and luma
/// sample rate cover `width` x `height` at `frame_rate` (VP9 Annex A,
/// Table A-1's `MaxPictureSize` and `MaxLumaSampleRate`).
pub fn vp9_level(width: u32, height: u32, frame_rate: f64) -> u8 {
    const LEVELS: [(u8, u64, u64); 14] = [
        (10, 36_864, 829_440),
        (11, 73_728, 2_764_800),
        (20, 122_880, 4_608_000),
        (21, 245_760, 9_216_000),
        (30, 552_960, 20_736_000),
        (31, 983_040, 36_864_000),
        (40, 2_228_224, 83_558_400),
        (41, 2_228_224, 160_432_128),
        (50, 8_912_896, 311_951_360),
        (51, 8_912_896, 588_251_136),
        (52, 8_912_896, 1_176_502_272),
        (60, 35_651_584, 1_176_502_272),
        (61, 35_651_584, 2_353_004_544),
        (62, 35_651_584, 4_706_009_088),
    ];
    let size = u64::from(width) * u64::from(height);
    let rate = (size as f64 * frame_rate.max(0.0)).ceil() as u64;
    LEVELS.iter().find(|(_, s, r)| size <= *s && rate <= *r).map_or(62, |(l, _, _)| *l)
}

/// The VP codec configuration a sample entry, a CMAF init segment and the
/// `codecs` string share.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VpxConfig {
    pub profile: u8,
    pub level: u8,
    pub bit_depth: u8,
    /// `chromaSubsampling`: 0 4:2:0 vertical (MPEG-2 siting), 1 4:2:0
    /// co-located, 2 4:2:2, 3 4:4:4.
    pub chroma_subsampling: u8,
    pub full_range: bool,
    pub colour_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
}

impl VpxConfig {
    /// The configuration of a stream whose first frame is `first` (a key
    /// frame), coded at `width` x `height` and `frame_rate`, tagged with
    /// `color`. `vp9` selects the VP9 header reader; VP8 is profile 0, 8-bit
    /// 4:2:0.
    pub fn from_stream(vp9: bool, first: &[u8], width: u32, height: u32, frame_rate: f64, color: &ColorMetadata) -> Self {
        let info = if vp9 { vp9_frame_info(first) } else { vp8_frame_info(first) };
        let info = info.unwrap_or(VpxFrameInfo {
            key_frame: true,
            profile: 0,
            bit_depth: 8,
            subsampling: (1, 1),
            full_range: false,
        });
        let chroma_subsampling = match info.subsampling {
            (1, 1) => 0,
            (1, 0) => 2,
            _ => 3,
        };
        VpxConfig {
            profile: info.profile,
            // VP8 has no levels: the field is VP9's.
            level: if vp9 { vp9_level(width, height, frame_rate) } else { 0 },
            bit_depth: info.bit_depth,
            chroma_subsampling,
            full_range: info.full_range || color.full_range,
            colour_primaries: color.colour_primaries,
            transfer_characteristics: crate::mux::transfer_to_h273(color.transfer),
            matrix_coefficients: color.matrix_coefficients,
        }
    }

    /// The `vpcC` box (a version-1 FullBox), header included.
    pub fn vpcc_box(&self) -> Vec<u8> {
        let mut body = vec![1u8, 0, 0, 0]; // version 1, flags 0
        body.push(self.profile);
        body.push(self.level);
        body.push((self.bit_depth << 4) | ((self.chroma_subsampling & 0x7) << 1) | u8::from(self.full_range));
        body.push(self.colour_primaries);
        body.push(self.transfer_characteristics);
        body.push(self.matrix_coefficients);
        body.extend_from_slice(&0u16.to_be_bytes()); // codecInitializationDataSize
        let mut out = Vec::with_capacity(body.len() + 8);
        out.extend_from_slice(&((body.len() + 8) as u32).to_be_bytes());
        out.extend_from_slice(b"vpcC");
        out.extend_from_slice(&body);
        out
    }

    /// The RFC 6381 `codecs` value, full form: `vp09.00.31.08.00.01.01.01.00`.
    pub fn codecs_string(&self, fourcc: &str) -> String {
        format!(
            "{fourcc}.{:02}.{:02}.{:02}.{:02}.{:02}.{:02}.{:02}.{:02}",
            self.profile,
            self.level,
            self.bit_depth,
            self.chroma_subsampling,
            self.colour_primaries,
            self.transfer_characteristics,
            self.matrix_coefficients,
            u8::from(self.full_range)
        )
    }

    /// Read a `vpcC` box body (after the 8-byte header).
    pub fn parse_vpcc_body(body: &[u8]) -> Option<Self> {
        if body.len() < 12 || body[0] != 1 {
            return None;
        }
        Some(VpxConfig {
            profile: body[4],
            level: body[5],
            bit_depth: body[6] >> 4,
            chroma_subsampling: (body[6] >> 1) & 0x7,
            full_range: body[6] & 1 == 1,
            colour_primaries: body[7],
            transfer_characteristics: body[8],
            matrix_coefficients: body[9],
        })
    }
}

/// An MSB-first bit reader over a frame header.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Bits<'_> {
    fn read(&mut self, n: u32) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            let byte = *self.data.get(self.pos / 8)?;
            v = (v << 1) | u32::from((byte >> (7 - self.pos % 8)) & 1);
            self.pos += 1;
        }
        Some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_follow_the_table() {
        assert_eq!(vp9_level(320, 240, 30.0), 20);
        assert_eq!(vp9_level(256, 144, 30.0), 11);
        assert_eq!(vp9_level(640, 360, 30.0), 21);
        assert_eq!(vp9_level(1280, 720, 30.0), 31);
        assert_eq!(vp9_level(1920, 1080, 30.0), 40);
        assert_eq!(vp9_level(1920, 1080, 60.0), 41);
        assert_eq!(vp9_level(3840, 2160, 30.0), 50);
    }

    #[test]
    fn a_vpcc_round_trips_and_names_its_codecs_string() {
        let c = VpxConfig {
            profile: 0,
            level: 31,
            bit_depth: 8,
            chroma_subsampling: 0,
            full_range: false,
            colour_primaries: 1,
            transfer_characteristics: 1,
            matrix_coefficients: 1,
        };
        let b = c.vpcc_box();
        assert_eq!(&b[4..8], b"vpcC");
        assert_eq!(b.len(), 20);
        assert_eq!(VpxConfig::parse_vpcc_body(&b[8..]), Some(c));
        assert_eq!(c.codecs_string("vp09"), "vp09.00.31.08.00.01.01.01.00");
    }

    #[test]
    fn vp8_key_frames_are_tag_bit_zero() {
        assert!(vp8_frame_info(&[0x10, 0x02, 0x00, 0x9d, 0x01, 0x2a]).unwrap().key_frame);
        assert!(!vp8_frame_info(&[0x11, 0x02, 0x00]).unwrap().key_frame);
        assert!(vp8_frame_info(&[0x10]).is_none());
    }

    #[test]
    fn a_vp9_profile_0_key_frame_header_reads() {
        // frame_marker 10, profile 0 (0, 0), show_existing 0, frame_type 0,
        // show_frame 1, error_res 0 → 1000_0010 = 0x82; sync 49 83 42;
        // color_space 2 (BT.709, 3 bits) + color_range 0 → 010 0 ....
        let hdr = [0x82, 0x49, 0x83, 0x42, 0b0100_0000, 0, 0, 0];
        let info = vp9_frame_info(&hdr).unwrap();
        assert!(info.key_frame);
        assert_eq!((info.profile, info.bit_depth, info.subsampling, info.full_range), (0, 8, (1, 1), false));
        // An inter frame: frame_type 1.
        assert!(!vp9_frame_info(&[0x86, 0, 0]).unwrap().key_frame);
    }
}
