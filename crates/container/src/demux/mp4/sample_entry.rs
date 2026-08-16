/// Sample-entry detection and parameter-set extraction helpers.
///
/// These functions walk the ISOBMFF box tree to identify the video codec and
/// pull decoder configuration (SPS/PPS for AVC, VPS/SPS/PPS for HEVC) out of
/// the `stsd` sample-entry. They are shared by the legacy `demux_mp4` entry
/// point (in `mod.rs`) and the streaming init path (in `streaming.rs`).
///
/// Box-walking primitives (`find_box_body`, `find_direct_child`) live in
/// `demux/mod.rs`; they are accessed via `super::super::` from this file.
use crate::annexb::{AvcConfig, HevcConfig, parse_avcc, parse_hvcc};

// ---------------------------------------------------------------------------
// AV1 / HEVC / ProRes fourcc detection
// ---------------------------------------------------------------------------

/// Walk the ISOBMFF box tree looking for an `av01` sample entry inside
/// `moov/trak/mdia/minf/stbl/stsd`. Returns true if found at the expected
/// nesting level. Doing a full tree walk (vs naive byte-search for "av01")
/// avoids false positives from sample data in mdat that happens to contain
/// those bytes.
pub(crate) fn has_av01_sample_entry(data: &[u8]) -> bool {
    let Some(stsd_body) = super::super::find_video_stsd(data) else {
        return false;
    };
    if stsd_body.len() < 16 {
        return false;
    }
    let mut pos = 8; // skip version/flags/entry_count
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        if entry_size == 0 {
            break;
        }
        if pos + 4 < stsd_body.len() && &stsd_body[pos + 4..pos + 8] == b"av01" {
            return true;
        }
        pos = pos.saturating_add(entry_size);
    }
    false
}

/// Find the HEVC sample-entry fourcc (`hvc1`, `hev1`, `hvc2`, `hev2`,
/// `dvh1`, `dvhe`) in the video track's stsd box. Returns the 4-byte
/// fourcc or None. Used as the mp4 0.14 crate detection fallback —
/// its `media_type()` only returns H265 for `hev1`, so `hvc1` (the
/// Jellyfin corpus's HEVC flavor) needs this path.
pub(super) fn hevc_sample_entry_fourcc(data: &[u8]) -> Option<[u8; 4]> {
    let stsd_body = super::super::find_video_stsd(data)?;
    if stsd_body.len() < 16 {
        return None;
    }
    let mut pos = 8; // skip version/flags/entry_count
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        let entry_type: [u8; 4] = stsd_body[pos + 4..pos + 8].try_into().ok()?;
        match &entry_type {
            b"hvc1" | b"hev1" | b"hvc2" | b"hev2" | b"dvh1" | b"dvhe" => {
                return Some(entry_type);
            }
            _ => {}
        }
        if entry_size == 0 {
            break;
        }
        pos = pos.saturating_add(entry_size);
    }
    None
}

/// Look for an Apple ProRes sample entry in the video track's stsd box.
/// Six fourccs cover the product family:
///   apcn = ProRes 422 Standard    apch = ProRes 422 HQ
///   apcs = ProRes 422 LT          apco = ProRes 422 Proxy
///   ap4h = ProRes 4444            ap4x = ProRes 4444 XQ
/// All share the same container layout (self-contained frame samples, no
/// length-prefix wrapping), so from demux's perspective they are
/// interchangeable — we return the first one we see so callers can log
/// which specific profile the input used. Decode dispatch uses the
/// unified `"prores"` codec label produced by `demux_mp4`.
pub(crate) fn prores_sample_entry_fourcc(data: &[u8]) -> Option<[u8; 4]> {
    let stsd_body = super::super::find_video_stsd(data)?;
    if stsd_body.len() < 16 {
        return None;
    }
    let mut pos = 8;
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        let entry_type: [u8; 4] = stsd_body[pos + 4..pos + 8].try_into().ok()?;
        match &entry_type {
            b"apcn" | b"apch" | b"apcs" | b"apco" | b"ap4h" | b"ap4x" => {
                return Some(entry_type);
            }
            _ => {}
        }
        if entry_size == 0 {
            break;
        }
        pos = pos.saturating_add(entry_size);
    }
    None
}

// ---------------------------------------------------------------------------
// AVC / HEVC config extraction (avcC / hvcC box parsing)
// ---------------------------------------------------------------------------

/// Find the AVC sample entry in MP4 and return its parsed avcC config
/// (length_size + SPS/PPS NAL units). Returns None when no `avc1`/`avc3`
/// sample entry is present or the avcC box is malformed.
pub(super) fn extract_avc_config(data: &[u8]) -> Option<AvcConfig> {
    let stsd_body = super::super::find_video_stsd(data)?;
    if stsd_body.len() < 16 {
        return None;
    }

    let mut pos = 8;
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        let entry_type = &stsd_body[pos + 4..pos + 8];
        let is_avc = matches!(entry_type, b"avc1" | b"avc3");
        if !is_avc {
            if entry_size == 0 {
                break;
            }
            pos = pos.saturating_add(entry_size);
            continue;
        }
        let end = pos.saturating_add(entry_size);
        if end > stsd_body.len() {
            return None;
        }
        let child_start = pos + 8 + 78; // VisualSampleEntry fixed header
        if child_start >= end {
            return None;
        }
        let avcc = super::super::find_direct_child(&stsd_body[child_start..end], b"avcC")?;
        return parse_avcc(avcc);
    }
    None
}

pub(super) fn extract_hevc_config(data: &[u8]) -> Option<HevcConfig> {
    let stsd_body = super::super::find_video_stsd(data)?;
    if stsd_body.len() < 16 {
        return None;
    }
    let mut pos = 8;
    while pos + 8 <= stsd_body.len() {
        let entry_size = u32::from_be_bytes([
            stsd_body[pos],
            stsd_body[pos + 1],
            stsd_body[pos + 2],
            stsd_body[pos + 3],
        ]) as usize;
        let entry_type = &stsd_body[pos + 4..pos + 8];
        let is_hevc = matches!(
            entry_type,
            b"hvc1" | b"hev1" | b"hvc2" | b"hev2" | b"dvh1" | b"dvhe"
        );
        if !is_hevc {
            if entry_size == 0 {
                break;
            }
            pos = pos.saturating_add(entry_size);
            continue;
        }
        let end = pos.saturating_add(entry_size);
        if end > stsd_body.len() {
            return None;
        }
        let child_start = pos + 8 + 78; // VisualSampleEntry fixed header
        if child_start >= end {
            return None;
        }
        let hvcc = super::super::find_direct_child(&stsd_body[child_start..end], b"hvcC")?;
        return parse_hvcc(hvcc);
    }
    None
}

#[allow(dead_code)]
pub(super) fn extract_hevc_parameter_sets(data: &[u8]) -> Vec<Vec<u8>> {
    extract_hevc_config(data)
        .map(|cfg| cfg.parameter_sets)
        .unwrap_or_default()
}

/// Parse the SPS/PPS parameter sets out of an avcC box (as a `Vec<Vec<u8>>`
/// of raw NAL units without start codes). Used by tests and as the fallback
/// when `extract_avc_config` is unavailable. Returns an empty Vec on any
/// parse failure — callers must tolerate that.
#[allow(dead_code)]
pub(crate) fn parse_avcc_param_sets(avcc: &[u8]) -> Vec<Vec<u8>> {
    parse_avcc(avcc)
        .map(|cfg| cfg.parameter_sets)
        .unwrap_or_default()
}

#[allow(dead_code)]
pub(super) fn parse_hvcc_param_sets(hvcc: &[u8]) -> Vec<Vec<u8>> {
    parse_hvcc(hvcc)
        .map(|cfg| cfg.parameter_sets)
        .unwrap_or_default()
}
