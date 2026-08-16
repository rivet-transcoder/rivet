//! AV1 OBU (Open Bitstream Unit) finding and byte-offset utilities.
//!
//! Provides the low-level OBU iteration that the higher-level parsers
//! (`sequence`, `frame`) and Vulkan decode infrastructure depend on.

use super::frame::parse_av1_frame_header;
use super::sequence::Av1SequenceHeader;

/// Find the first AV1 OBU of the given obu_type. AV1 OBU header:
///   obu_forbidden_bit(1) | obu_type(4) | obu_extension_flag(1)
///   | obu_has_size_field(1) | obu_reserved_1bit(1)
/// followed by an optional 1-byte extension, optional LEB128 size,
/// then payload. For simplicity we require obu_has_size_field=1 which
/// all muxed AV1 satisfies.
pub(super) fn find_av1_obu(data: &[u8], target_type: u8) -> Option<&[u8]> {
    find_av1_obu_with_offset(data, target_type).map(|(bytes, _)| bytes)
}

/// Public re-export so the Vulkan Video decoder can extract the byte
/// range of an OBU from a demuxed sample.
pub fn find_av1_obu_with_offset_pub(data: &[u8], target_type: u8) -> Option<(&[u8], usize)> {
    find_av1_obu_with_offset(data, target_type)
}

/// Returns the OBU payload slice AND the byte offset at which it
/// starts inside `data`. The offset is what callers need to translate
/// an in-OBU bit/byte position (e.g. tile_group start after
/// byte_alignment()) to an absolute position in the sample buffer.
pub(super) fn find_av1_obu_with_offset(data: &[u8], target_type: u8) -> Option<(&[u8], usize)> {
    let mut i = 0;
    while i < data.len() {
        let header = data[i];
        let obu_type = (header >> 3) & 0x0F;
        let extension_flag = (header >> 2) & 0x01;
        let has_size_field = (header >> 1) & 0x01;
        i += 1;
        if extension_flag == 1 {
            i += 1;
        }
        if has_size_field == 0 {
            return None;
        }
        let (size, leb_bytes) = read_leb128(&data[i..])?;
        i += leb_bytes;
        if obu_type == target_type {
            let end = (i + size as usize).min(data.len());
            return Some((&data[i..end], i));
        }
        i += size as usize;
    }
    None
}

fn read_leb128(data: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0u64;
    for i in 0..8 {
        if i >= data.len() {
            return None;
        }
        let byte = data[i];
        value |= ((byte & 0x7F) as u64) << (i * 7);
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None
}

/// Locate the byte offset, within `sample`, of the uncompressed_header
/// payload of the first Frame OBU (obu_type 3 or 6). Returns None if
/// no such OBU is found.
///
/// AV1 OBU layout: 1-byte header + optional 1-byte extension + LEB128
/// size + payload. For a Frame OBU (type 6), the payload begins with
/// uncompressed_header_obu() — so the byte offset we return is the
/// first byte of uncompressed_header() in the original sample buffer.
/// Vulkan `VkVideoDecodeAV1PictureInfoKHR::frameHeaderOffset` wants
/// exactly this value.
pub fn av1_frame_header_offset(sample: &[u8]) -> Option<u32> {
    let mut i = 0usize;
    while i < sample.len() {
        let header = sample[i];
        let obu_type = (header >> 3) & 0x0F;
        let extension_flag = (header >> 2) & 0x01;
        let has_size_field = (header >> 1) & 0x01;
        let mut p = i + 1;
        if extension_flag == 1 {
            p += 1;
        }
        let (size, leb) = if has_size_field == 1 {
            let (s, n) = read_leb128(&sample[p..])?;
            p += n;
            (s as usize, n)
        } else {
            // OBU has_size_field=0 is legal but we don't handle it
            // (AV1 in MP4 always sets it).
            return None;
        };
        let _ = leb;
        if obu_type == 3 || obu_type == 6 {
            return Some(p as u32);
        }
        p += size;
        i = p;
    }
    None
}

/// Locate the byte offset of the first tile_group_obu payload within
/// the sample buffer, used for
/// `VkVideoDecodeAV1PictureInfoKHR::pTileOffsets`. Two shapes:
/// - Separate Frame Header OBU (type 3) + Tile Group OBU (type 4):
///   return the type-4 OBU payload start.
/// - Frame OBU (type 6) (frame header + tile group in one OBU):
///   return `frame_OBU_payload_start + tile_group_offset_in_obu`
///   where the in-OBU offset comes from `parse_av1_frame_header`
///   (the byte-aligned position after uncompressed_header).
///
/// Returns None when neither shape is found or the parser bails.
pub fn av1_tile_group_offset(sample: &[u8], seq: &Av1SequenceHeader) -> Option<u32> {
    // If a standalone Tile Group OBU (type 4) exists, use its payload
    // start directly — no uncompressed_header to skip past.
    let mut i = 0usize;
    while i < sample.len() {
        let header = sample[i];
        let obu_type = (header >> 3) & 0x0F;
        let extension_flag = (header >> 2) & 0x01;
        let has_size_field = (header >> 1) & 0x01;
        let mut p = i + 1;
        if extension_flag == 1 {
            p += 1;
        }
        let size = if has_size_field == 1 {
            let (s, n) = read_leb128(&sample[p..])?;
            p += n;
            s as usize
        } else {
            return None;
        };
        if obu_type == 4 {
            return Some(p as u32);
        }
        p += size;
        i = p;
    }
    // Frame OBU (type 6): combine the OBU payload start with the
    // in-OBU offset from the parsed frame header.
    let (_obu_bytes, payload_offset) = find_av1_obu_with_offset(sample, 6)?;
    let hdr = parse_av1_frame_header(sample, seq)?;
    Some(payload_offset as u32 + hdr.tile_group_offset_in_obu)
}

/// Backwards-compatible shim — uses an empty-ish sequence header
/// default that only works for the fallback path (standalone type-4
/// OBU). Callers with access to the parsed sequence header should
/// use `av1_tile_group_offset` (the seq-aware form) instead.
pub fn av1_tile_group_offset_fallback(sample: &[u8]) -> Option<u32> {
    let mut i = 0usize;
    while i < sample.len() {
        let header = sample[i];
        let obu_type = (header >> 3) & 0x0F;
        let extension_flag = (header >> 2) & 0x01;
        let has_size_field = (header >> 1) & 0x01;
        let mut p = i + 1;
        if extension_flag == 1 {
            p += 1;
        }
        let size = if has_size_field == 1 {
            let (s, n) = read_leb128(&sample[p..])?;
            p += n;
            s as usize
        } else {
            return None;
        };
        if obu_type == 4 {
            return Some(p as u32);
        }
        p += size;
        i = p;
    }
    av1_frame_header_offset(sample)
}
/// Whether an AV1 packet opens a random-access point.
///
/// # Why this is not read from the encoder
///
/// oneVPL leaves `mfxBitstream::FrameType` at zero for AV1 on the iHD
/// runtime — the field is an H.264/HEVC concept and the AV1 encoder simply
/// does not populate it. Trusting it means every packet reports itself as a
/// delta frame, which is silently harmless in a single MP4 (the `stss` table
/// ends up empty and players cope) and fatal in CMAF: a segment must open on
/// a sync sample, so the segmenter can never cut, the whole video accumulates
/// in one pending buffer, and the trailing flush fails with
/// "first pending sample is not a keyframe".
///
/// The bitstream always knows. A sequence header OBU only accompanies a
/// random-access point, and a frame OBU states its own `frame_type`, so this
/// reads the answer rather than asking the encoder for it.
pub fn av1_packet_is_keyframe(sample: &[u8]) -> bool {
    let mut i = 0usize;

    while i < sample.len() {
        let header = sample[i];
        let obu_type = (header >> 3) & 0x0F;
        let extension_flag = (header >> 2) & 0x01;
        let has_size_field = (header >> 1) & 0x01;

        let mut p = i + 1;
        if extension_flag == 1 {
            p += 1;
        }

        // `has_size_field = 0` is legal in the wild but never produced by an
        // encoder writing into a container, and without a size there is no
        // way to reach the next OBU. Unknown rather than false would be a
        // lie; refusing to guess is the honest answer.
        if has_size_field != 1 {
            return false;
        }

        let Some((size, leb)) = read_leb128(&sample[p..]) else {
            return false;
        };
        p += leb;

        // OBU_SEQUENCE_HEADER. It is only sent where a decoder may start,
        // which is precisely a random-access point.
        if obu_type == 1 {
            return true;
        }

        // OBU_FRAME_HEADER / OBU_FRAME — the frame states its own type.
        if obu_type == 3 || obu_type == 6 {
            let Some(&first) = sample.get(p) else {
                return false;
            };

            // uncompressed_header(): show_existing_frame(1), then
            // frame_type(2) when it is a new frame. A shown existing frame
            // codes no picture of its own and can never open a segment.
            if (first >> 7) & 1 == 1 {
                return false;
            }

            // KEY_FRAME = 0, INTER = 1, INTRA_ONLY = 2, SWITCH = 3.
            return (first >> 5) & 0b11 == 0;
        }

        p += size as usize;
        i = p;
    }

    false
}
