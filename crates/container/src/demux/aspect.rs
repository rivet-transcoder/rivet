//! The source's sample aspect ratio: the shape of one stored pixel.
//!
//! Most video has square samples, and its stored size is its shape. An
//! anamorphic source does not: a PAL DVD's 720x576 is shown 16:9 (samples
//! 64:45 wide) or 4:3 (16:15), and HDV's 1440x1080 is shown 1920x1080 (4:3).
//! A resize that takes the stored size for the shape squashes or stretches
//! the picture, so every demuxer finds the ratio and puts it on its header.
//!
//! Where it is said, in order: the container (an MP4 `pasp` box, a Matroska
//! `DisplayWidth`/`DisplayHeight` in pixels), then the stream (an H.264 or
//! HEVC SPS VUI's `aspect_ratio_info`, an MPEG-2 sequence header's
//! `aspect_ratio_information`). The container wins where both speak, as it
//! does for colour, because it is what a player honours. Nothing said is
//! square.

/// Square samples.
pub const SQUARE: (u32, u32) = (1, 1);

/// `(n, d)` in lowest terms; `None` for a zero term, which no format
/// defines as a shape.
pub(crate) fn reduce((n, d): (u64, u64)) -> Option<(u32, u32)> {
    if n == 0 || d == 0 {
        return None;
    }
    let (mut a, mut b) = (n, d);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    let (n, d) = (n / a, d / a);
    Some((u32::try_from(n).ok()?, u32::try_from(d).ok()?))
}

/// An MP4 `pasp` box body (ISO/IEC 14496-12 §12.1.4): `hSpacing`,
/// `vSpacing`, each a u32.
pub(crate) fn parse_pasp(body: &[u8]) -> Option<(u32, u32)> {
    let h = u32::from_be_bytes(body.get(0..4)?.try_into().ok()?);
    let v = u32::from_be_bytes(body.get(4..8)?.try_into().ok()?);
    reduce((h.into(), v.into()))
}

/// The sample aspect ratio a Matroska track's display size implies for its
/// pixel size: `DisplayWidth x DisplayHeight` is the shape the picture is
/// shown at, so a sample is `(dw / pw) : (dh / ph)` wide. Only for a display
/// unit of pixels (the default); centimetres, inches and a bare aspect ratio
/// state a physical size, not a shape to resample to.
pub(crate) fn from_display_size(pixel: (u32, u32), display: (u64, u64)) -> Option<(u32, u32)> {
    let (pw, ph) = (u64::from(pixel.0), u64::from(pixel.1));
    reduce((display.0.checked_mul(ph)?, display.1.checked_mul(pw)?))
}

/// What the stream itself states: the first SPS's VUI for H.264 and HEVC
/// (`parameter_sets` out of band, else the access unit `first_au`), the
/// sequence header for MPEG-2. `width x height` is the stored picture, which
/// MPEG-2's display ratio is turned into a sample ratio against.
pub(crate) fn from_bitstream(
    codec: &str,
    parameter_sets: &[Vec<u8>],
    first_au: Option<&[u8]>,
    width: u32,
    height: u32,
) -> Option<(u32, u32)> {
    match codec {
        "h264" | "avc" | "avc1" | "h265" | "hevc" => {
            let in_band: Vec<Vec<u8>> = first_au
                .map(|au| h26x::nal::annexb_nals(au).map(<[u8]>::to_vec).collect())
                .unwrap_or_default();
            from_parameter_sets(codec, parameter_sets).or_else(|| from_parameter_sets(codec, &in_band))
        }
        "mpeg2" => {
            let dar = parameter_sets
                .iter()
                .map(Vec::as_slice)
                .chain(first_au)
                .find_map(|s| frame::pixel_format::parse_mpeg2_display_aspect(s, width, height))?;
            // A sample is DAR / (stored shape) wide.
            reduce((u64::from(dar.0) * u64::from(height), u64::from(dar.1) * u64::from(width)))
        }
        _ => None,
    }
}

/// The `aspect_ratio_info` of the first SPS among `parameter_sets` (one NAL
/// unit each, with or without a start code).
fn from_parameter_sets(codec: &str, parameter_sets: &[Vec<u8>]) -> Option<(u32, u32)> {
    for entry in parameter_sets {
        let nal: &[u8] = if entry.starts_with(&[0, 0, 0, 1]) {
            &entry[4..]
        } else if entry.starts_with(&[0, 0, 1]) {
            &entry[3..]
        } else {
            entry
        };
        let Some(&first) = nal.first() else { continue };
        let sar = match codec {
            "h265" | "hevc" if (first >> 1) & 0x3f == 33 => {
                let rbsp = h26x::nal::unescape_rbsp(nal);
                h26x::hevc::Sps::parse(rbsp.get(2..)?).ok()?.vui?.sample_aspect
            }
            "h264" | "avc" | "avc1" if first & 0x1f == 7 => {
                let rbsp = h26x::nal::unescape_rbsp(&nal[1..]);
                h26x::h264::Sps::parse(&rbsp).ok()?.vui?.sample_aspect
            }
            _ => continue,
        };
        return sar.and_then(|(w, h)| reduce((w.into(), h.into())));
    }
    None
}

/// The container's say, else the stream's, else square — logged when it is
/// not square, since it changes every output's size.
pub(crate) fn resolve(container: Option<(u32, u32)>, stream: impl FnOnce() -> Option<(u32, u32)>, label: &str) -> (u32, u32) {
    let (sar, from) = match container {
        Some(sar) => (sar, "container"),
        None => match stream() {
            Some(sar) => (sar, "stream"),
            None => return SQUARE,
        },
    };
    if sar != SQUARE {
        tracing::info!(
            container = label,
            from,
            sample_aspect = %format!("{}:{}", sar.0, sar.1),
            "source has non-square samples; outputs are sized by its display shape"
        );
    }
    sar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratios_are_reduced_and_zero_is_nothing() {
        assert_eq!(reduce((128, 90)), Some((64, 45)));
        assert_eq!(reduce((1, 1)), Some(SQUARE));
        assert_eq!(reduce((0, 1)), None);
        assert_eq!(reduce((1, 0)), None);
    }

    #[test]
    fn pasp_reads_its_two_spacings() {
        assert_eq!(parse_pasp(&[0, 0, 0, 64, 0, 0, 0, 45]), Some((64, 45)));
        assert_eq!(parse_pasp(&[0, 0, 0, 1, 0, 0, 0, 1]), Some(SQUARE));
        assert_eq!(parse_pasp(&[0, 0, 0, 0, 0, 0, 0, 1]), None, "a zero spacing is no shape");
        assert_eq!(parse_pasp(&[0, 0, 0, 1]), None, "truncated");
    }

    #[test]
    fn a_matroska_display_size_gives_the_sample_shape() {
        // 720x576 shown at 1024x576: 16:9, samples 64:45.
        assert_eq!(from_display_size((720, 576), (1024, 576)), Some((64, 45)));
        // Displayed at its own size: square.
        assert_eq!(from_display_size((1920, 1080), (1920, 1080)), Some(SQUARE));
        // HDV: 1440x1080 shown 1920x1080.
        assert_eq!(from_display_size((1440, 1080), (1920, 1080)), Some((4, 3)));
    }

    #[test]
    fn an_mpeg2_display_ratio_becomes_a_sample_ratio() {
        // sequence_header_code, 720x576, aspect_ratio_information 3 (16:9),
        // frame_rate_code 3 (25).
        let seq = [0x00, 0x00, 0x01, 0xB3, 0x2D, 0x02, 0x40, 0x33, 0xFF, 0xFF, 0xE0, 0x18];
        assert_eq!(from_bitstream("mpeg2", &[], Some(&seq), 720, 576), Some((64, 45)));
        // Code 2: 4:3 on the same picture is 16:15.
        let mut four_three = seq;
        four_three[7] = 0x23;
        assert_eq!(from_bitstream("mpeg2", &[], Some(&four_three), 720, 576), Some((16, 15)));
        // Code 1: square.
        let mut square = seq;
        square[7] = 0x13;
        assert_eq!(from_bitstream("mpeg2", &[], Some(&square), 720, 576), Some(SQUARE));
    }

    #[test]
    fn an_h264_vui_ratio_is_read_in_or_out_of_band() {
        // libx264, `setsar=64/45`: aspect_ratio_idc 255, 64:45.
        let sps = vec![
            0x67, 0x64, 0x00, 0x0a, 0xac, 0xd9, 0x44, 0x26, 0xff, 0xc0, 0x10, 0x00, 0x0b, 0x44,
            0x00, 0x00, 0x03, 0x00, 0x04, 0x00, 0x00, 0x03, 0x00, 0xc8, 0x3c, 0x48, 0x96, 0x58,
        ];
        assert_eq!(from_bitstream("h264", std::slice::from_ref(&sps), None, 64, 64), Some((64, 45)));
        let mut au = vec![0, 0, 0, 1];
        au.extend_from_slice(&sps);
        assert_eq!(from_bitstream("h264", &[], Some(&au), 64, 64), Some((64, 45)));
    }

    #[test]
    fn the_container_is_taken_over_the_stream() {
        assert_eq!(resolve(Some((4, 3)), || Some((64, 45)), "test"), (4, 3));
        assert_eq!(resolve(None, || Some((64, 45)), "test"), (64, 45));
        assert_eq!(resolve(None, || None, "test"), SQUARE);
    }
}
