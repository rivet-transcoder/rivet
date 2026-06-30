use anyhow::Result;
use codec::frame::{ColorMetadata, VideoCodec};
use super::sample_table::AudioBuildPlan;
use super::video_track::build_video_trak;
use super::audio_track::build_audio_trak;

// ---- Generic box infrastructure -----------------------------------------------

pub(crate) fn write_unity_matrix(b: &mut BoxBuilder) {
    b.u32(0x00010000);
    b.u32(0);
    b.u32(0);
    b.u32(0);
    b.u32(0x00010000);
    b.u32(0);
    b.u32(0);
    b.u32(0);
    b.u32(0x40000000);
}

pub(crate) struct BoxBuilder {
    buf: Vec<u8>,
}

impl BoxBuilder {
    pub(crate) fn new(box_type: &[u8; 4]) -> Self {
        let mut buf = Vec::with_capacity(64);
        buf.extend_from_slice(&[0, 0, 0, 0]); // size placeholder
        buf.extend_from_slice(box_type);
        Self { buf }
    }

    pub(crate) fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    pub(crate) fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }
    pub(crate) fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }
    pub(crate) fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }
    pub(crate) fn extend(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }

    /// Current byte length of the buffer (header + payload written so far).
    /// Used by the CMAF muxer to record the position of `trun.data_offset`
    /// so it can be patched once the moof's final size is known.
    pub(crate) fn current_len(&self) -> usize {
        self.buf.len()
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        let size = self.buf.len() as u32;
        self.buf[0..4].copy_from_slice(&size.to_be_bytes());
        self.buf
    }
}

// ---- ftyp / moov / mvhd -------------------------------------------------------

/// Build `ftyp` for AV1-in-MP4 with Apple-device compatibility.
///
/// Per AV1-ISOBMFF v1.3.0 §2.1, an AV1-bearing ISOBMFF file SHALL list
/// `av01` in its `compatible_brands`. Apple's QuickTime / iOS Safari
/// stack additionally requires a structural ISOBMFF brand: `iso6`
/// (ISO/IEC 14496-12 sixth edition — covers `co64`, `mehd` v1, etc.)
/// is the right choice here because the muxer's co64 / large-mdat
/// extensions need the v6 spec scope to be conformant. `mp42`
/// (ISO/IEC 14496-14 second edition) is the conventional brand
/// downstream players key off when deciding AAC / mp4a parsing rules,
/// so we list it as well.
///
/// `major_brand` is set to `iso6` so a strict parser that rejects an
/// `isom`/`mp41`-major file with a co64 box (mp41 predates the v6
/// definition) accepts the output.
pub(super) fn build_ftyp(codec: VideoCodec) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"ftyp");
    b.extend(b"iso6"); // major_brand (v6 of 14496-12; covers co64/largesize)
    b.u32(512); // minor_version (matches FFmpeg / mp4box convention)
    b.extend(b"iso6"); // compatible: structural baseline
    b.extend(b"iso2"); // compatible: 14496-12 second edition (legacy parsers)
    // codec brand: av01 (AV1-ISOBMFF §2.1, REQUIRED) / avc1 (H.264) / hvc1 (H.265)
    b.extend(codec.sample_entry_fourcc().as_bytes());
    b.extend(b"mp41"); // compatible: classic 14496-14 (older players)
    b.extend(b"mp42"); // compatible: 14496-14 second edition (AAC parsing rules)
    b.finish()
}

/// Video-only back-compat wrapper, used by existing tests. New code flows
/// through `build_moov_any` which handles the 1-trak / 2-trak case
/// uniformly.
#[cfg(test)]
pub(super) fn build_moov(
    width: u32,
    height: u32,
    timescale: u32,
    duration: u64,
    frame_duration: u32,
    sample_sizes: &[u32],
    keyframe_indices: &[u32],
    config_obus: &[u8],
    chunk_offsets: &[u64],
    samples_per_chunk: u32,
    use_co64: bool,
) -> Vec<u8> {
    build_moov_any(
        width,
        height,
        timescale,
        timescale,
        duration,
        duration,
        frame_duration,
        sample_sizes,
        keyframe_indices,
        config_obus,
        chunk_offsets,
        samples_per_chunk,
        None,
        &[],
        use_co64,
        &ColorMetadata::default(),
    )
}

/// Build moov with video trak plus optional audio trak. `movie_timescale`
/// governs mvhd; `video_timescale` is video mdhd's own clock. When audio is
/// present we pin both movie and video to the same 90 kHz reference so
/// durations don't need a per-trak rate rescale.
pub(super) fn build_moov_any(
    width: u32,
    height: u32,
    video_timescale: u32,
    movie_timescale: u32,
    movie_duration: u64,
    video_duration_in_video_ts: u64,
    frame_duration: u32,
    sample_sizes: &[u32],
    keyframe_indices: &[u32],
    config_obus: &[u8],
    video_chunk_offsets: &[u64],
    video_spc: u32,
    audio_plan: Option<&AudioBuildPlan>,
    audio_chunk_offsets: &[u64],
    use_co64: bool,
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    // next_track_ID starts at 3 when audio is present (video=1, audio=2).
    let next_track_id: u32 = if audio_plan.is_some() { 3 } else { 2 };
    let mvhd = build_mvhd_v2(movie_timescale, movie_duration, next_track_id);
    // Video track duration expressed in movie timescale.
    let video_duration_movie: u64 = if video_timescale == movie_timescale {
        video_duration_in_video_ts
    } else {
        ((video_duration_in_video_ts as u128) * movie_timescale as u128
            / video_timescale.max(1) as u128) as u64
    };
    let video_trak = build_video_trak(
        width,
        height,
        video_timescale,
        video_duration_movie,
        video_duration_in_video_ts,
        frame_duration,
        sample_sizes,
        keyframe_indices,
        config_obus,
        video_chunk_offsets,
        video_spc,
        use_co64,
        color_metadata,
    );

    let mut b = BoxBuilder::new(b"moov");
    b.extend(&mvhd);
    b.extend(&video_trak);
    if let Some(plan) = audio_plan {
        let audio_trak = build_audio_trak(
            plan,
            plan.total_duration_in_movie_ts,
            audio_chunk_offsets,
            use_co64,
        );
        b.extend(&audio_trak);
    }
    b.finish()
}

/// mvhd v2: takes `next_track_ID`. When audio is present we increment past
/// the audio track ID, otherwise past the video track ID (existing
/// behaviour: next_track_ID=2). Original `build_mvhd` fed 2 hard-coded.
fn build_mvhd_v2(timescale: u32, duration: u64, next_track_id: u32) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"mvhd");
    b.u8(0); // version
    b.extend(&[0, 0, 0]); // flags
    b.u32(0); // creation_time
    b.u32(0); // modification_time
    b.u32(timescale);
    b.u32(duration as u32);
    b.u32(0x00010000); // rate 1.0
    b.u16(0x0100); // volume 1.0
    b.u16(0); // reserved
    b.u32(0); // reserved
    b.u32(0);
    write_unity_matrix(&mut b);
    for _ in 0..6 {
        b.u32(0);
    } // pre_defined
    b.u32(next_track_id);
    b.finish()
}

// ---- AV1 OBU parsing ----------------------------------------------------------

/// Scan OBU stream for OBU_SEQUENCE_HEADER and return a re-emitted copy with
/// obu_has_size_field=1 (required for av1C configOBUs per AV1-ISOBMFF §2.3.3).
///
/// Requires the encoder to emit Low-Overhead-Bitstream (LOB) format with
/// obu_has_size_field set on every OBU — this is the case for rav1e and NVENC.
/// If has_size==0, bail rather than stuff frame data into configOBUs: without
/// a size field the parser can't know where one OBU ends and the next begins.
pub(crate) fn extract_sequence_header(data: &[u8]) -> Result<Vec<u8>> {
    let mut pos = 0;
    while pos < data.len() {
        let header_byte = data[pos];
        pos += 1;
        let obu_type = (header_byte >> 3) & 0x0F;
        let extension_flag = (header_byte >> 2) & 0x1;
        let has_size = (header_byte >> 1) & 0x1;
        if has_size == 0 {
            anyhow::bail!(
                "AV1 packet uses Annex-B style OBUs (obu_has_size_field=0); \
                 expected LOB format from the encoder"
            );
        }
        if extension_flag != 0 {
            if pos >= data.len() {
                anyhow::bail!("truncated OBU extension header");
            }
            pos += 1;
        }
        let (size64, size_len) = read_leb128(&data[pos..])?;
        let size = size64 as usize;
        pos += size_len;
        if pos + size > data.len() {
            anyhow::bail!("OBU payload extends past packet");
        }
        if obu_type == 1 {
            // Re-emit header with ext=0, has_size=1, no temporal/spatial ID.
            let header: u8 = (1 << 3) | (1 << 1);
            let mut out = Vec::with_capacity(1 + 8 + size);
            out.push(header);
            write_leb128(&mut out, size as u64);
            out.extend_from_slice(&data[pos..pos + size]);
            return Ok(out);
        }
        pos += size;
    }
    anyhow::bail!("no OBU_SEQUENCE_HEADER found in first packet")
}

pub(super) fn read_leb128(data: &[u8]) -> Result<(u64, usize)> {
    let mut value: u64 = 0;
    let mut len = 0usize;
    for i in 0..8 {
        if i >= data.len() {
            anyhow::bail!("truncated leb128");
        }
        let byte = data[i];
        value |= ((byte & 0x7F) as u64) << (i * 7);
        len += 1;
        if (byte & 0x80) == 0 {
            return Ok((value, len));
        }
    }
    anyhow::bail!("leb128 too long")
}

pub(super) fn write_leb128(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
            out.push(byte);
        } else {
            out.push(byte);
            return;
        }
    }
}

/// Parse AV1 sequence header OBU to extract parameters needed for av1C.
///
/// Returns `(seq_profile, seq_level_idx_0, seq_tier_0,
///          high_bitdepth, twelve_bit, monochrome,
///          chroma_subsampling_x, chroma_subsampling_y, chroma_sample_position)`.
///
/// Defaults match 8-bit 4:2:0 Main profile if parsing fails — the resulting
/// av1C will still be valid for typical rav1e output (profile 0 level 0).
pub(super) fn parse_seq_header_params(obu: &[u8]) -> (u8, u8, u8, bool, bool, bool, u8, u8, u8) {
    if obu.len() < 2 {
        return (0, 0, 0, false, false, false, 1, 1, 0);
    }
    // Skip OBU header + leb128 size
    let mut pos = 1;
    if obu[0] & 0x02 != 0 {
        // has_size: parse leb128
        match read_leb128(&obu[pos..]) {
            Ok((_, len)) => pos += len,
            Err(_) => return (0, 0, 0, false, false, false, 1, 1, 0),
        }
    }
    if pos >= obu.len() {
        return (0, 0, 0, false, false, false, 1, 1, 0);
    }

    let mut br = BitReader::new(&obu[pos..]);
    let seq_profile = br.bits(3).unwrap_or(0) as u8;
    let _still_picture = br.bits(1).unwrap_or(0);
    let reduced_still_picture_header = br.bits(1).unwrap_or(0);

    let (seq_level_idx_0, seq_tier_0) = if reduced_still_picture_header != 0 {
        (br.bits(5).unwrap_or(0) as u8, 0)
    } else {
        let timing_info_present = br.bits(1).unwrap_or(0);
        if timing_info_present != 0 {
            let _num_units = br.bits(32);
            let _time_scale = br.bits(32);
            let equal_pts = br.bits(1).unwrap_or(0);
            if equal_pts != 0 {
                let _nticks = read_uvlc(&mut br);
            }
            let decoder_model_info_present = br.bits(1).unwrap_or(0);
            if decoder_model_info_present != 0 {
                let _bdlm1 = br.bits(5);
                let _nts = br.bits(32);
                let _brslm1 = br.bits(5);
                let _frpdlm1 = br.bits(5);
            }
        }
        let initial_display_delay_present = br.bits(1).unwrap_or(0);
        let operating_points_cnt_minus_1 = br.bits(5).unwrap_or(0);
        let mut level0 = 0u8;
        let mut tier0 = 0u8;
        for i in 0..=operating_points_cnt_minus_1 {
            let _operating_point_idc = br.bits(12).unwrap_or(0);
            let seq_level_idx_i = br.bits(5).unwrap_or(0) as u8;
            let seq_tier_i = if seq_level_idx_i > 7 {
                br.bits(1).unwrap_or(0) as u8
            } else {
                0
            };
            if i == 0 {
                level0 = seq_level_idx_i;
                tier0 = seq_tier_i;
            }
            // Decoder model / initial_display_delay skipping
            // decoder_model_info_present always 0 in our path above; skip its conditional fields.
            if initial_display_delay_present != 0 {
                let present = br.bits(1).unwrap_or(0);
                if present != 0 {
                    let _iddm1 = br.bits(4);
                }
            }
        }
        (level0, tier0)
    };

    let frame_width_bits_minus_1 = br.bits(4).unwrap_or(0);
    let frame_height_bits_minus_1 = br.bits(4).unwrap_or(0);
    let _max_frame_width_minus_1 = br.bits(frame_width_bits_minus_1 + 1);
    let _max_frame_height_minus_1 = br.bits(frame_height_bits_minus_1 + 1);

    if reduced_still_picture_header == 0 {
        let frame_id_numbers_present = br.bits(1).unwrap_or(0);
        if frame_id_numbers_present != 0 {
            let _delta_fid_len = br.bits(4);
            let _add_fid_len = br.bits(3);
        }
    }
    let _use_128x128 = br.bits(1);
    let _enable_filter_intra = br.bits(1);
    let _enable_intra_edge_filter = br.bits(1);
    if reduced_still_picture_header == 0 {
        let _enable_interintra = br.bits(1);
        let _enable_masked = br.bits(1);
        let _enable_warped = br.bits(1);
        let _enable_dual_filter = br.bits(1);
        let _enable_order_hint = br.bits(1);
        let enable_order_hint = _enable_order_hint.unwrap_or(0);
        if enable_order_hint != 0 {
            let _enable_jnt_comp = br.bits(1);
            let _enable_ref_frame_mvs = br.bits(1);
        }
        let seq_choose_screen_detection_tools = br.bits(1).unwrap_or(0);
        let seq_force_screen_content_tools = if seq_choose_screen_detection_tools != 0 {
            2
        } else {
            br.bits(1).unwrap_or(0)
        };
        if seq_force_screen_content_tools > 0 {
            let seq_choose_integer_mv = br.bits(1).unwrap_or(0);
            if seq_choose_integer_mv == 0 {
                let _seq_force_integer_mv = br.bits(1);
            }
        }
        if enable_order_hint != 0 {
            let _order_hint_bits_minus_1 = br.bits(3);
        }
    }
    let _enable_superres = br.bits(1);
    let _enable_cdef = br.bits(1);
    let _enable_restoration = br.bits(1);

    // color_config() per AV1 §5.5.2
    let high_bitdepth = br.bits(1).unwrap_or(0) != 0;
    let twelve_bit = if seq_profile == 2 && high_bitdepth {
        br.bits(1).unwrap_or(0) != 0
    } else {
        false
    };
    let monochrome = if seq_profile == 1 {
        false
    } else {
        br.bits(1).unwrap_or(0) != 0
    };
    let color_description_present = br.bits(1).unwrap_or(0) != 0;
    let (color_primaries, transfer_characteristics, matrix_coefficients) =
        if color_description_present {
            let cp = br.bits(8).unwrap_or(2) as u8;
            let tc = br.bits(8).unwrap_or(2) as u8;
            let mc = br.bits(8).unwrap_or(2) as u8;
            (cp, tc, mc)
        } else {
            (2u8, 2u8, 2u8) // CP_UNSPECIFIED / TC_UNSPECIFIED / MC_UNSPECIFIED
        };
    let (subsampling_x, subsampling_y, chroma_sample_position) = if monochrome {
        // color_range
        let _color_range = br.bits(1);
        (1u8, 1u8, 0u8)
    } else if color_primaries == 1 /* CP_BT_709 */
        && transfer_characteristics == 13 /* TC_SRGB */
        && matrix_coefficients == 0
    /* MC_IDENTITY */
    {
        // color_range is implicitly full (1), RGB 4:4:4
        (0u8, 0u8, 0u8)
    } else {
        let _color_range = br.bits(1);
        let (sx, sy) = if seq_profile == 0 {
            (1u8, 1u8)
        } else if seq_profile == 1 {
            (0u8, 0u8)
        } else {
            let bit_depth = if high_bitdepth {
                if twelve_bit { 12 } else { 10 }
            } else {
                8
            };
            if bit_depth == 12 {
                let sxb = br.bits(1).unwrap_or(1) as u8;
                let syb = if sxb != 0 {
                    br.bits(1).unwrap_or(1) as u8
                } else {
                    0
                };
                (sxb, syb)
            } else {
                (1u8, 0u8)
            }
        };
        let csp = if sx != 0 && sy != 0 {
            br.bits(2).unwrap_or(0) as u8
        } else {
            0u8
        };
        (sx, sy, csp)
    };
    // separate_uv_deltas follows but we don't emit it; parser state ends here.

    (
        seq_profile,
        seq_level_idx_0,
        seq_tier_0,
        high_bitdepth,
        twelve_bit,
        monochrome,
        subsampling_x,
        subsampling_y,
        chroma_sample_position,
    )
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn bits(&mut self, n: u32) -> Option<u32> {
        let mut v: u32 = 0;
        for _ in 0..n {
            if self.pos / 8 >= self.data.len() {
                return None;
            }
            let byte = self.data[self.pos / 8];
            let bit = (byte >> (7 - (self.pos % 8))) & 1;
            v = (v << 1) | bit as u32;
            self.pos += 1;
        }
        Some(v)
    }
}

fn read_uvlc(br: &mut BitReader) -> u32 {
    let mut leading_zeros = 0u32;
    while leading_zeros < 32 {
        match br.bits(1) {
            Some(0) => leading_zeros += 1,
            Some(_) => break,
            None => return 0,
        }
    }
    if leading_zeros >= 32 {
        return u32::MAX;
    }
    let value = br.bits(leading_zeros).unwrap_or(0);
    value + ((1u32 << leading_zeros) - 1)
}
