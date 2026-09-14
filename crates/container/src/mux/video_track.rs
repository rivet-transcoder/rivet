use frame::ColorMetadata;
use super::boxes::{BoxBuilder, write_unity_matrix, parse_seq_header_params};
use super::sample_table::{build_stsc, build_stsz, build_stco, build_co64};

// ---- Video trak / mdia / minf / stbl / stsd -----------------------------------

/// Video trak builder. `duration_in_movie_ts` goes into tkhd (the movie
/// header's clock); `duration_in_mdhd_ts` goes into mdhd (the track's own
/// clock). For video the two timescales are currently pinned equal at
/// 90 kHz, but the split is kept so the audio path, which has a distinct
/// mdhd timescale (= sample_rate), uses the same builder pattern.
pub(super) fn build_video_trak(
    width: u32,
    height: u32,
    mdhd_timescale: u32,
    duration_in_movie_ts: u64,
    duration_in_mdhd_ts: u64,
    frame_duration: u32,
    sample_sizes: &[u32],
    keyframe_indices: &[u32],
    composition_offsets: Option<&[i32]>,
    config_obus: &[u8],
    chunk_offsets: &[u64],
    samples_per_chunk: u32,
    use_co64: bool,
    color_metadata: &ColorMetadata,
    // `edts` for a track that does not present from time zero (a late start);
    // `None` writes the trak exactly as before edits existed.
    edts: Option<&[u8]>,
) -> Vec<u8> {
    let tkhd = build_video_tkhd(width, height, duration_in_movie_ts);
    let mdia = build_video_mdia(
        width,
        height,
        mdhd_timescale,
        duration_in_mdhd_ts,
        frame_duration,
        sample_sizes,
        keyframe_indices,
        composition_offsets,
        config_obus,
        chunk_offsets,
        samples_per_chunk,
        use_co64,
        color_metadata,
    );

    let mut b = BoxBuilder::new(b"trak");
    b.extend(&tkhd);
    if let Some(edts) = edts {
        b.extend(edts);
    }
    b.extend(&mdia);
    b.finish()
}

fn build_video_tkhd(width: u32, height: u32, duration: u64) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"tkhd");
    b.u8(0); // version
    b.extend(&[0, 0, 0x03]); // flags: track_enabled | track_in_movie
    b.u32(0); // creation_time
    b.u32(0); // modification_time
    b.u32(1); // track_ID
    b.u32(0); // reserved
    b.u32(duration as u32);
    b.u32(0); // reserved
    b.u32(0);
    b.u16(0); // layer
    b.u16(0); // alternate_group
    b.u16(0); // volume (0 for video)
    b.u16(0); // reserved
    write_unity_matrix(&mut b);
    b.u32(width << 16); // width as 16.16
    b.u32(height << 16);
    b.finish()
}

fn build_video_mdia(
    width: u32,
    height: u32,
    timescale: u32,
    duration: u64,
    frame_duration: u32,
    sample_sizes: &[u32],
    keyframe_indices: &[u32],
    composition_offsets: Option<&[i32]>,
    config_obus: &[u8],
    chunk_offsets: &[u64],
    samples_per_chunk: u32,
    use_co64: bool,
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let mdhd = build_mdhd(timescale, duration);
    let hdlr = build_video_hdlr();
    let minf = build_minf(
        width,
        height,
        frame_duration,
        sample_sizes,
        keyframe_indices,
        composition_offsets,
        config_obus,
        chunk_offsets,
        samples_per_chunk,
        use_co64,
        color_metadata,
    );

    let mut b = BoxBuilder::new(b"mdia");
    b.extend(&mdhd);
    b.extend(&hdlr);
    b.extend(&minf);
    b.finish()
}

pub(super) fn build_mdhd(timescale: u32, duration: u64) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"mdhd");
    b.u8(0); // version
    b.extend(&[0, 0, 0]); // flags
    b.u32(0); // creation_time
    b.u32(0); // modification_time
    b.u32(timescale);
    b.u32(duration as u32);
    b.u16(0x55c4); // language 'und'
    b.u16(0); // pre_defined
    b.finish()
}

fn build_video_hdlr() -> Vec<u8> {
    let mut b = BoxBuilder::new(b"hdlr");
    b.u8(0); // version
    b.extend(&[0, 0, 0]); // flags
    b.u32(0); // pre_defined
    b.extend(b"vide"); // handler_type
    b.u32(0); // reserved[0]
    b.u32(0); // reserved[1]
    b.u32(0); // reserved[2]
    b.extend(b"VideoHandler\0");
    b.finish()
}

pub(super) fn build_dinf() -> Vec<u8> {
    let mut dref = BoxBuilder::new(b"dref");
    dref.u8(0);
    dref.extend(&[0, 0, 0]);
    dref.u32(1); // entry_count
    let mut url = BoxBuilder::new(b"url ");
    url.u8(0);
    url.extend(&[0, 0, 0x01]); // self-contained
    dref.extend(&url.finish());

    let mut b = BoxBuilder::new(b"dinf");
    b.extend(&dref.finish());
    b.finish()
}

fn build_minf(
    width: u32,
    height: u32,
    frame_duration: u32,
    sample_sizes: &[u32],
    keyframe_indices: &[u32],
    composition_offsets: Option<&[i32]>,
    config_obus: &[u8],
    chunk_offsets: &[u64],
    samples_per_chunk: u32,
    use_co64: bool,
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let vmhd = build_vmhd();
    let dinf = build_dinf();
    let stbl = build_stbl(
        width,
        height,
        frame_duration,
        sample_sizes,
        keyframe_indices,
        composition_offsets,
        config_obus,
        chunk_offsets,
        samples_per_chunk,
        use_co64,
        color_metadata,
    );

    let mut b = BoxBuilder::new(b"minf");
    b.extend(&vmhd);
    b.extend(&dinf);
    b.extend(&stbl);
    b.finish()
}

fn build_vmhd() -> Vec<u8> {
    let mut b = BoxBuilder::new(b"vmhd");
    b.u8(0);
    b.extend(&[0, 0, 0x01]); // flags (always 1)
    b.u16(0); // graphicsmode
    b.u16(0);
    b.u16(0);
    b.u16(0); // opcolor
    b.finish()
}

fn build_stbl(
    width: u32,
    height: u32,
    frame_duration: u32,
    sample_sizes: &[u32],
    keyframe_indices: &[u32],
    composition_offsets: Option<&[i32]>,
    config_obus: &[u8],
    chunk_offsets: &[u64],
    samples_per_chunk: u32,
    use_co64: bool,
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let stsd = build_stsd(width, height, config_obus, color_metadata);
    let stts = build_stts(sample_sizes.len() as u32, frame_duration);
    let ctts = composition_offsets.map(build_ctts);
    let stsc = build_stsc(sample_sizes.len() as u32, samples_per_chunk);
    let stsz = build_stsz(sample_sizes);
    let chunk_offset_box = if use_co64 {
        build_co64(chunk_offsets)
    } else {
        build_stco(chunk_offsets)
    };
    let stss_box = if !keyframe_indices.is_empty() && keyframe_indices.len() < sample_sizes.len() {
        Some(build_stss(keyframe_indices))
    } else {
        None
    };

    let mut b = BoxBuilder::new(b"stbl");
    b.extend(&stsd);
    b.extend(&stts);
    if let Some(ct) = &ctts {
        b.extend(ct);
    }
    if let Some(ss) = &stss_box {
        b.extend(ss);
    }
    b.extend(&stsc);
    b.extend(&stsz);
    b.extend(&chunk_offset_box);
    b.finish()
}

/// `stsd` wrapping a single, pre-built visual sample entry (`av01` / `avc1` /
/// `hvc1` — the caller builds the codec-appropriate one). The trailing params
/// are vestigial (the entry already carries width/height/colour) and kept only
/// so the threading call sites don't change.
fn build_stsd(
    _width: u32,
    _height: u32,
    video_sample_entry: &[u8],
    _color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"stsd");
    b.u8(0);
    b.extend(&[0, 0, 0]); // flags
    b.u32(1); // entry_count
    b.extend(video_sample_entry);
    b.finish()
}

fn build_stts(sample_count: u32, frame_duration: u32) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"stts");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(1); // entry_count
    b.u32(sample_count);
    b.u32(frame_duration);
    b.finish()
}

/// `ctts` — Composition Time to Sample (ISO/IEC 14496-12 §8.6.1.3), version
/// 1: each sample's `CT − DT` as a **signed** 32-bit offset, run-length
/// coded over consecutive equal offsets.
///
/// Version 1 rather than version 0 plus an edit list: with signed offsets
/// the decode timeline starts at zero and the first picture is presented at
/// zero, with no `elst` for a reader to honour or ignore (ffmpeg's
/// `negative_cts_offsets`; DASH-IF's recommended form). A B picture is
/// presented *before* it is decoded on this timeline; every reader that
/// knows version 1 shifts the decode times back by the largest negative
/// offset, which is what `ffprobe` reports as a negative first `dts`.
///
/// Only written when some offset is non-zero — see `crate::reorder`.
pub(super) fn build_ctts(offsets: &[i32]) -> Vec<u8> {
    let mut runs: Vec<(u32, i32)> = Vec::new();
    for &o in offsets {
        match runs.last_mut() {
            Some((count, last)) if *last == o => *count += 1,
            _ => runs.push((1, o)),
        }
    }
    let mut b = BoxBuilder::new(b"ctts");
    b.u8(1); // version 1: signed sample_offset
    b.extend(&[0, 0, 0]);
    b.u32(runs.len() as u32);
    for (count, offset) in runs {
        b.u32(count);
        b.u32(offset as u32); // two's complement of the i32
    }
    b.finish()
}

fn build_stss(keyframes: &[u32]) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"stss");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(keyframes.len() as u32);
    for &k in keyframes {
        b.u32(k);
    }
    b.finish()
}

// ---- Visual sample entries: av01 / avc1 / hvc1 --------------------------------

/// AV1 visual sample entry per AV1-ISOBMFF v1.3.0 §2.2. Fourcc is `av01`
/// — there is no `hvc1`/`hev1`-style variant for AV1; the configOBU
/// transport mode is selected via flags inside `av1C` itself, not via a
/// separate sample entry name.
///
/// Children, in order:
/// 1. `av1C` — AV1CodecConfigurationRecord (REQUIRED).
/// 2. `colr` — nclx triple + full_range (REQUIRED for Apple, Squad-18).
/// 3. `mdcv` — Mastering Display Color Volume (HDR only, Squad-20).
/// 4. `clli` — Content Light Level Info (HDR only, Squad-20).
///
/// The HDR atoms `mdcv` and `clli` are emitted only when
/// `ColorMetadata.mastering_display` / `.content_light_level` are
/// `Some(_)`. AV1-ISOBMFF v1.3.0 §2.3.4 + §2.3.5 specify the order
/// `colr → mdcv → clli` inside the visual sample entry; players that
/// scan for `mdcv` / `clli` (browsers via Media Capabilities API,
/// AVFoundation) read the box-tree by 4cc, so order is recommended
/// but not load-bearing — we match the spec anyway.
pub(crate) fn build_av01(
    width: u32,
    height: u32,
    config_obus: &[u8],
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let av1c = build_av1c(config_obus);
    let colr = build_colr_nclx(color_metadata);
    let mdcv = color_metadata.mastering_display.as_ref().map(build_mdcv);
    let clli = color_metadata.content_light_level.as_ref().map(build_clli);
    let mut b = BoxBuilder::new(b"av01");
    // VisualSampleEntry
    for _ in 0..6 {
        b.u8(0);
    } // reserved[6]
    b.u16(1); // data_reference_index
    b.u16(0); // pre_defined
    b.u16(0); // reserved
    for _ in 0..3 {
        b.u32(0);
    } // pre_defined[3]
    b.u16(width as u16);
    b.u16(height as u16);
    b.u32(0x00480000); // horiz 72 dpi
    b.u32(0x00480000); // vert 72 dpi
    b.u32(0); // reserved
    b.u16(1); // frame_count (frames per sample)
    // compressorname: 1 length byte + 31 bytes
    b.u8(0);
    for _ in 0..31 {
        b.u8(0);
    }
    b.u16(0x0018); // depth
    b.u16(0xFFFF); // pre_defined
    b.extend(&av1c);
    b.extend(&colr);
    if let Some(mdcv) = &mdcv {
        b.extend(mdcv);
    }
    if let Some(clli) = &clli {
        b.extend(clli);
    }
    b.finish()
}

/// Write the 78-byte ISO 14496-12 `VisualSampleEntry` header (shared by
/// `av01` / `avc1` / `hvc1`) into a freshly-opened sample-entry box.
fn push_visual_sample_entry_header(b: &mut BoxBuilder, width: u32, height: u32) {
    for _ in 0..6 {
        b.u8(0);
    } // reserved[6]
    b.u16(1); // data_reference_index
    b.u16(0); // pre_defined
    b.u16(0); // reserved
    for _ in 0..3 {
        b.u32(0);
    } // pre_defined[3]
    b.u16(width as u16);
    b.u16(height as u16);
    b.u32(0x00480000); // horiz 72 dpi
    b.u32(0x00480000); // vert 72 dpi
    b.u32(0); // reserved
    b.u16(1); // frame_count
    b.u8(0);
    for _ in 0..31 {
        b.u8(0);
    } // compressorname
    b.u16(0x0018); // depth
    b.u16(0xFFFF); // pre_defined
}

/// Append `colr` + (HDR) `mdcv`/`clli` to a visual sample entry.
fn push_color_boxes(b: &mut BoxBuilder, color_metadata: &ColorMetadata) {
    b.extend(&build_colr_nclx(color_metadata));
    if let Some(md) = color_metadata.mastering_display.as_ref() {
        b.extend(&build_mdcv(md));
    }
    if let Some(cll) = color_metadata.content_light_level.as_ref() {
        b.extend(&build_clli(cll));
    }
}

/// Remove H.264/H.265 emulation-prevention bytes (`00 00 03` → `00 00`) so the
/// raw profile/tier/level fields can be read by byte offset. Returns the RBSP.
fn strip_emulation(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let n = data.len();
    let mut i = 0;
    while i < n {
        if i + 2 < n && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 3 {
            out.push(0);
            out.push(0);
            i += 3; // drop the 0x03; the following byte is handled next iter
        } else {
            out.push(data[i]);
            i += 1;
        }
    }
    out
}

/// H.264 `avc1` visual sample entry (avcC + colr [+ HDR atoms]).
/// H.264 visual sample entry (avcC + colr [+ HDR atoms]). `fourcc` is `avc1`
/// (out-of-band parameter sets) or `avc3` (in-band, for the inline stitch).
pub(crate) fn build_avc1(
    width: u32,
    height: u32,
    avcc: &[u8],
    color_metadata: &ColorMetadata,
    fourcc: &[u8; 4],
) -> Vec<u8> {
    let mut b = BoxBuilder::new(fourcc);
    push_visual_sample_entry_header(&mut b, width, height);
    b.extend(avcc);
    push_color_boxes(&mut b, color_metadata);
    b.finish()
}

/// H.265 visual sample entry (hvcC + colr [+ HDR atoms]). `fourcc` is `hvc1`
/// (out-of-band parameter sets) or `hev1` (in-band, for the inline stitch).
pub(crate) fn build_hvc1(
    width: u32,
    height: u32,
    hvcc: &[u8],
    color_metadata: &ColorMetadata,
    fourcc: &[u8; 4],
) -> Vec<u8> {
    let mut b = BoxBuilder::new(fourcc);
    push_visual_sample_entry_header(&mut b, width, height);
    b.extend(hvcc);
    push_color_boxes(&mut b, color_metadata);
    b.finish()
}

/// AVCDecoderConfigurationRecord (`avcC`) per ISO 14496-15 §5.3.3.1. Profile /
/// compatibility / level come verbatim from the first SPS (NAL payload bytes
/// 1..4). 4-byte NAL length prefixes (`lengthSizeMinusOne = 3`).
///
/// Every profile but Baseline, Main and Extended (66 / 77 / 88) — High, High
/// 10, High 4:2:2, High 4:4:4 Predictive and the rest — carries the record's
/// high-profile extension after the PPS array (§5.3.3.1.2): `chroma_format`,
/// `bit_depth_luma_minus8` and `bit_depth_chroma_minus8`, parsed from the SPS
/// as [`build_hvcc`] parses its own, then `numOfSequenceParameterSetExt`,
/// zero (the muxer collects no SPS extension NAL units). A reader that sizes
/// its surfaces from the record rather than the SPS takes a High 10 stream for
/// 8-bit 4:2:0 without it. ffmpeg's writer adds the same four bytes under the
/// same condition — `fd f8 f8 00` for 8-bit High, `fd fa fa 00` for High 10.
pub(crate) fn build_avcc(sps: &[Vec<u8>], pps: &[Vec<u8>]) -> Vec<u8> {
    let first = sps.first().map(|s| s.as_slice()).unwrap_or(&[]);
    let (profile, compat, level) = if first.len() >= 4 {
        (first[1], first[2], first[3])
    } else {
        (0x64, 0x00, 0x1f) // High @ L3.1 fallback
    };
    let mut body = vec![
        1, // configurationVersion
        profile,
        compat,
        level,
        0xFF,                             // reserved(6)=1 | lengthSizeMinusOne = 3
        0xE0 | (sps.len() as u8 & 0x1F), // reserved(3)=1 | numOfSPS
    ];
    for s in sps {
        body.extend_from_slice(&(s.len() as u16).to_be_bytes());
        body.extend_from_slice(s);
    }
    body.push(pps.len() as u8); // numOfPPS
    for p in pps {
        body.extend_from_slice(&(p.len() as u16).to_be_bytes());
        body.extend_from_slice(p);
    }
    if !matches!(profile, 66 | 77 | 88) {
        // An SPS that does not parse is described as 4:2:0 8-bit, the
        // same default `build_hvcc` falls back to.
        let (chroma_format, luma_m8, chroma_m8) = avc_sps_format(first).unwrap_or((1, 0, 0));
        body.push(0xFC | (chroma_format & 0x03)); // reserved(6) | chroma_format
        body.push(0xF8 | (luma_m8 & 0x07)); // reserved(5) | bit_depth_luma_minus8
        body.push(0xF8 | (chroma_m8 & 0x07)); // reserved(5) | bit_depth_chroma_minus8
        body.push(0); // numOfSequenceParameterSetExt
    }
    let mut b = BoxBuilder::new(b"avcC");
    b.extend(&body);
    b.finish()
}

/// `(chroma_format_idc, bit_depth_luma_minus8, bit_depth_chroma_minus8)` of
/// one H.264 SPS NAL unit (header byte included, no start code), read with
/// the pipeline's own SPS parser — the reader the demuxer's format detection
/// uses. `None` when it does not parse.
fn avc_sps_format(sps_nal: &[u8]) -> Option<(u8, u8, u8)> {
    // parse_h264_sps wants Annex-B; prepend a start code to the raw NAL.
    let mut annexb = vec![0u8, 0, 0, 1];
    annexb.extend_from_slice(sps_nal);
    let info = frame::pixel_format::parse_h264_sps(&annexb)?;
    Some((
        info.chroma_format_idc,
        info.bit_depth_luma.checked_sub(8)?,
        info.bit_depth_chroma.checked_sub(8)?,
    ))
}

/// HEVCDecoderConfigurationRecord (`hvcC`) per ISO 14496-15 §8.3.3.1.2. The
/// 12-byte general profile_tier_level is copied from the first SPS (RBSP bytes
/// 3..15 — after the 2-byte NAL header + the 1-byte vps_id/max_sub/nesting).
/// Chroma + bit depth are pinned to 4:2:0 8-bit (our SDR output). VPS/SPS/PPS
/// arrays follow. 4-byte NAL length prefixes.
pub(crate) fn build_hvcc(vps: &[Vec<u8>], sps: &[Vec<u8>], pps: &[Vec<u8>]) -> Vec<u8> {
    let mut ptl = [0u8; 12];
    // Bit depth (minus 8) + chroma format parsed from the SPS — 0/1 for Main
    // 4:2:0 8-bit, 2/1 for Main 10 (10-bit 4:2:0). The hvcC carries these
    // explicitly (bytes 16-18), so a Main 10 stream must report 2 here or
    // strict decoders mis-configure the surface.
    let (mut bit_depth_luma_m8, mut bit_depth_chroma_m8, mut chroma_format) = (0u8, 0u8, 1u8);
    if let Some(s) = sps.first() {
        let rbsp = strip_emulation(s);
        if rbsp.len() >= 15 {
            ptl.copy_from_slice(&rbsp[3..15]);
        } else {
            ptl[0] = 0x01; // Main profile
            ptl[11] = 123; // level 4.1
        }
        // parse_hevc_sps wants Annex-B; prepend a start code to the raw NAL.
        let mut annexb = vec![0u8, 0, 0, 1];
        annexb.extend_from_slice(s);
        if let Some(info) = frame::pixel_format::parse_hevc_sps(&annexb) {
            bit_depth_luma_m8 = info.bit_depth_luma.saturating_sub(8);
            bit_depth_chroma_m8 = info.bit_depth_chroma.saturating_sub(8);
            chroma_format = info.chroma_format_idc;
        }
    }
    let mut body = Vec::new();
    body.push(1); // configurationVersion
    body.extend_from_slice(&ptl); // [1..13] general PTL
    body.extend_from_slice(&[0xF0, 0x00]); // [13-14] reserved | min_spatial_segmentation_idc=0
    body.push(0xFC); // [15] reserved | parallelismType=0
    body.push(0xFC | (chroma_format & 0x03)); // [16] reserved | chromaFormat
    body.push(0xF8 | (bit_depth_luma_m8 & 0x07)); // [17] reserved | bitDepthLumaMinus8
    body.push(0xF8 | (bit_depth_chroma_m8 & 0x07)); // [18] reserved | bitDepthChromaMinus8
    body.extend_from_slice(&[0, 0]); // [19-20] avgFrameRate=0
    body.push(0x0F); // [21] cfr=0 | numTemporalLayers=1 | tidNested=1 | lengthSizeMinusOne=3
    let arrays: [(u8, &[Vec<u8>]); 3] = [(32, vps), (33, sps), (34, pps)];
    let present: Vec<&(u8, &[Vec<u8>])> = arrays.iter().filter(|(_, v)| !v.is_empty()).collect();
    body.push(present.len() as u8); // numOfArrays
    for (nal_type, set) in present {
        body.push(0x80 | nal_type); // array_completeness=1 | reserved=0 | NAL_unit_type
        body.extend_from_slice(&(set.len() as u16).to_be_bytes());
        for nal in *set {
            body.extend_from_slice(&(nal.len() as u16).to_be_bytes());
            body.extend_from_slice(nal);
        }
    }
    let mut b = BoxBuilder::new(b"hvcC");
    b.extend(&body);
    b.finish()
}

// ---- Color metadata boxes (colr / mdcv / clli) --------------------------------

/// Map the pipeline's `TransferFn` enum back into an H.273
/// `transfer_characteristics` u8 for the `colr nclx` writer. The
/// pipeline's enum is lossy — `Bt709` covers H.273 codes 1, 6, 14, 15 —
/// so we collapse to the canonical code (1 = BT.709) for the SDR family
/// and the spec-defined codes for the HDR transfers.
pub(super) fn transfer_to_h273(transfer: frame::TransferFn) -> u8 {
    use frame::TransferFn;
    match transfer {
        TransferFn::Bt709 => 1,
        TransferFn::Bt470Bg => 4,
        TransferFn::Linear => 8,
        TransferFn::St2084 => 16,
        TransferFn::AribStdB67 => 18,
        // H.273 reserves 2 for "unspecified". Apple's player treats
        // unspecified as BT.709 limited, which is what the rest of this
        // code already assumes — so there's no behaviour change between
        // emitting 2 and emitting 1 here. Emit 2 to stay honest about
        // what the source told us.
        TransferFn::Unspecified => 2,
    }
}

/// Emit a `colr` box with `colour_type='nclx'` per ISO/IEC 14496-12 §12.1.5
/// and ICC's nclx subtype definition. Layout:
///
///   size u32 | 'colr' | colour_type[4] | colour_primaries u16
///   | transfer_characteristics u16 | matrix_coefficients u16
///   | full_range_flag(1) + reserved(7)
///
/// `nclx` is the right colour_type for video distribution (vs `nclc`
/// which is QuickTime-flavored or `rICC`/`prof` for embedded ICC
/// profiles). Apple's player and ffmpeg both honour it.
pub(super) fn build_colr_nclx(color_metadata: &ColorMetadata) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"colr");
    b.extend(b"nclx");
    b.u16(color_metadata.colour_primaries as u16);
    b.u16(transfer_to_h273(color_metadata.transfer) as u16);
    b.u16(color_metadata.matrix_coefficients as u16);
    // full_range_flag is the high bit of a single packed byte; the low 7
    // bits are reserved-zero per ISO 23001-8.
    let full_range_byte: u8 = if color_metadata.full_range {
        0x80
    } else {
        0x00
    };
    b.u8(full_range_byte);
    b.finish()
}

/// Emit a `mdcv` (Mastering Display Color Volume) box per ISO/IEC
/// 23001-17 §7.3 / AV1-ISOBMFF v1.3.0 §2.3.4. Carries SMPTE ST 2086
/// metadata. Layout:
///
///   size u32 (=32) | 'mdcv' | display_primaries_G_x u16 | _G_y u16
///   | _B_x u16 | _B_y u16 | _R_x u16 | _R_y u16
///   | white_point_x u16 | white_point_y u16
///   | max_display_mastering_luminance u32
///   | min_display_mastering_luminance u32
///
/// Total payload = 8×2 + 2×4 = 24 bytes; with 8-byte header → 32 bytes.
///
/// Box type is `'mdcv'` per AV1-ISOBMFF / 14496-12 v6, NOT the older
/// `'SmDm'` from QuickTime-flavored MOV. Browsers + AVFoundation read
/// `'mdcv'`. The byte order is the standard u16/u32 BE everything else
/// in the file uses.
///
/// Field encoding follows HEVC SEI 137 (`mastering_display_colour_volume`)
/// byte for byte — the box is the SEI payload — including the primaries'
/// order: `display_primaries[c]` is indexed green, blue, red (the order
/// every HEVC / AV1 writer uses and the order libavformat's `mdcv`
/// reader assumes). Writing them red-first put BT.2020's green
/// chromaticity where ffprobe reports `red_x`, and swapped them again
/// through this crate's own reader (`demux::hdr::parse_mp4_mdcv`, which
/// reads G, B, R).
///   - Chromaticities are u16 in increments of 0.00002 (so a value of
///     35400 ↔ x=0.708, the BT.2020 red primary).
///   - Luminances are u32 in increments of 0.0001 cd/m² (so 10_000_000
///     ↔ 1000 nits, the canonical HDR10 max).
///
/// We do not normalize/clamp here — the input struct carries spec-domain
/// integers already (Squad-21's probe is responsible for that conversion
/// from float chromaticities / nits).
pub(super) fn build_mdcv(md: &frame::MasteringDisplay) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"mdcv");
    b.u16(md.primaries_g_x);
    b.u16(md.primaries_g_y);
    b.u16(md.primaries_b_x);
    b.u16(md.primaries_b_y);
    b.u16(md.primaries_r_x);
    b.u16(md.primaries_r_y);
    b.u16(md.white_point_x);
    b.u16(md.white_point_y);
    b.u32(md.max_luminance);
    b.u32(md.min_luminance);
    b.finish()
}

/// Emit a `clli` (Content Light Level Information) box per ISO/IEC
/// 14496-12 §12.1.6 / AV1-ISOBMFF v1.3.0 §2.3.5. Carries CTA-861.3
/// metadata. Layout:
///
///   size u32 (=12) | 'clli' | max_content_light_level u16
///   | max_pic_average_light_level u16
///
/// Total payload = 4 bytes; with 8-byte header → 12 bytes.
///
/// Box type is `'clli'`, NOT `'CoLL'` (the older MOV variant). Both
/// fields are integer cd/m² (nits); MaxCLL is the peak pixel anywhere
/// in the stream, MaxFALL is the peak frame-average. The HDR10
/// reference values are typically MaxCLL ≈ 1000 nits / MaxFALL ≈
/// 400 nits, but we write whatever the source declared verbatim.
pub(super) fn build_clli(cll: &frame::ContentLightLevel) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"clli");
    b.u16(cll.max_cll);
    b.u16(cll.max_fall);
    b.finish()
}

fn build_av1c(config_obus: &[u8]) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"av1C");
    // marker=1, version=1 -> 0x81
    b.u8(0x81);
    // seq_profile=0, seq_level_idx_0=0 (default; parse from OBU if present)
    let (
        seq_profile,
        seq_level_idx_0,
        seq_tier_0,
        high_bitdepth,
        twelve_bit,
        monochrome,
        chroma_sub_x,
        chroma_sub_y,
        chroma_sample_position,
    ) = parse_seq_header_params(config_obus);
    b.u8(((seq_profile & 0x7) << 5) | (seq_level_idx_0 & 0x1F));
    let byte3 = ((seq_tier_0 & 0x1) << 7)
        | ((high_bitdepth as u8 & 0x1) << 6)
        | ((twelve_bit as u8 & 0x1) << 5)
        | ((monochrome as u8 & 0x1) << 4)
        | ((chroma_sub_x & 0x1) << 3)
        | ((chroma_sub_y & 0x1) << 2)
        | (chroma_sample_position & 0x3);
    b.u8(byte3);
    // initial_presentation_delay_present=0, reserved bits=0
    b.u8(0);
    // configOBUs
    b.extend(config_obus);
    b.finish()
}
