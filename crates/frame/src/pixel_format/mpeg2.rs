//! MPEG-2 sequence header parser — ISO/IEC 13818-2 §6.2.2.1 + §6.2.2.3.

use super::bitreader::BitReader;

/// Parsed MPEG-2 sequence header + (optional) sequence extension.
///
/// MPEG-2 video §6.2.2.1/§6.2.2.3 (ISO/IEC 13818-2): the 12-bit
/// `horizontal_size_value` / `vertical_size_value` from the sequence
/// header, optionally extended to 14 bits by the 2-bit
/// `horizontal_size_extension` / `vertical_size_extension` fields in a
/// `sequence_extension()` start-code-prefixed NAL. Pure MPEG-1
/// (start code 0xB3 but no 0xB5 extension) stays 12-bit — produces
/// the same 12-bit result via the extension-less path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mpeg2SeqInfo {
    pub width: u32,
    pub height: u32,
}

/// MPEG-2 sequence header scan — ISO/IEC 13818-2 §6.2.2.1 (sequence
/// header, start code `00 00 01 B3`) + §6.2.2.3 (sequence extension,
/// start code `00 00 01 B5` with `extension_start_code_identifier==1`).
///
/// The sequence header carries 12-bit `horizontal_size_value` and
/// `vertical_size_value`, tight for sizes ≤ 4095. The optional sequence
/// extension prepends 2-bit `_extension` fields that, when combined,
/// bring the total to 14 bits (sizes ≤ 16383). Pure MPEG-1 (start code
/// 0xB3 only, no 0xB5) never has the extension and stays 12-bit.
pub fn parse_mpeg2_sequence_header(sample: &[u8]) -> Option<Mpeg2SeqInfo> {
    // Walk bytes looking for 00 00 01 B3 (sequence_header_code). The
    // following 3 bytes carry horizontal(12) + vertical(12).
    let seq_hdr_start = find_mpeg2_start_code(sample, 0xB3)?;
    let hdr_body_off = seq_hdr_start + 4;
    if hdr_body_off + 3 > sample.len() {
        return None;
    }
    let b = &sample[hdr_body_off..hdr_body_off + 3];
    let mut width = (((b[0] as u32) << 4) | ((b[1] as u32) >> 4)) & 0x0FFF;
    let mut height = (((b[1] as u32 & 0x0F) << 8) | (b[2] as u32)) & 0x0FFF;

    // Look for a subsequent sequence_extension that upgrades the 12-bit
    // values to 14-bit. Only scan forward from seq_hdr_start; a
    // sequence_extension before the first sequence_header is
    // nonsensical and we shouldn't confuse the parse.
    let search_from = hdr_body_off + 3;
    if search_from < sample.len()
        && let Some(ext_start) = find_mpeg2_start_code(&sample[search_from..], 0xB5)
    {
        let ext_body_off = search_from + ext_start + 4;
        if ext_body_off + 3 <= sample.len() {
            let mut br = BitReader::new(&sample[ext_body_off..]);
            if let Some(id) = br.read_bits(4)
                && id == 1
            {
                // sequence_extension §6.2.2.3:
                //   extension_start_code_identifier  u(4) = 0001   (already read)
                //   profile_and_level_indication     u(8)
                //   progressive_sequence             u(1)
                //   chroma_format                    u(2)
                //   horizontal_size_extension        u(2)
                //   vertical_size_extension          u(2)
                let _profile_level = br.read_bits(8)?;
                let _progressive = br.read_bits(1)?;
                let _chroma = br.read_bits(2)?;
                let h_ext = br.read_bits(2)?;
                let v_ext = br.read_bits(2)?;
                width |= h_ext << 12;
                height |= v_ext << 12;
            }
        }
    }

    if width == 0 || height == 0 {
        return None;
    }
    Some(Mpeg2SeqInfo { width, height })
}

/// Scan for an MPEG-2 start code (0x00 0x00 0x01 <target>) byte-aligned.
/// Returns the file offset of the leading 0x00 on success.
fn find_mpeg2_start_code(data: &[u8], target: u8) -> Option<usize> {
    let mut i = 0;
    while i + 4 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 && data[i + 3] == target {
            return Some(i);
        }
        i += 1;
    }
    None
}
