//! Per-rung codec invariant: types + the validate/set helper.

use anyhow::{Result, anyhow};
use std::sync::RwLock;
use codec::frame::VideoCodec;
use codec::pixel_format::{
    Av1SequenceHeader, H264SpsInfo, HevcSpsInfo, parse_av1_sequence_header, parse_h264_sps,
    parse_hevc_sps,
};

/// Mandatory AV1 sequence-header fields that every encoder
/// contributing segments to a single rendition MUST agree on.
///
/// Why these specific fields: each is part of the codec-init contract
/// that the player sets up once from `av1C` and expects to hold for
/// every segment. The decoder re-parses the inline OBU sequence
/// header in each segment's IDR; if its parsed values disagree with
/// the av1C from `init.mp4` on any of these fields, strict decoders
/// (dav1d in conformance mode, Safari AVFoundation, hls.js+libdav1d)
/// will reject the segment. Optional fields not listed here (timing
/// info presence, decoder model presence, film grain `present` flag,
/// operating-point details) are tolerated by every major player; we
/// deliberately don't check them so that NVENC + QSV + AMF + the software AV1 encoder
/// can co-exist on one rendition without cosmetic byte differences
/// triggering false rejections.
///
/// First worker on a rung SETS the invariant. Subsequent workers
/// (helpers from any vendor) COMPARE; mismatch fails the run loudly
/// instead of silently corrupting output.
/// Per-rung codec invariant. Each chunk encoded on a different GPU must agree on
/// these decode-init fields, or strict players reject the stitched stream. AV1
/// compares sequence-header fields; H.264/H.265 compare the SPS profile / level
/// / chroma / bit-depth / dims (the `avcC`/`hvcC` decode-init contract).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RungCodecInvariant {
    Av1(Av1Invariant),
    /// Shared by H.264 + H.265 — a rung is single-codec, so the variant only
    /// ever compares chunks of the same codec.
    H26x(H26xInvariant),
    /// VP9 (the one other codec an HLS ladder carries): the profile, depth
    /// and chroma of the first key frame's uncompressed header — what a
    /// `vp09` init segment's `vpcC` records for every segment.
    Vp9 { profile: u8, bit_depth: u8, subsampling: (u8, u8) },
    /// A codec only rivet's own encoder writes, whose first packets have
    /// nothing a decoder is initialised from beyond the codec itself.
    Codec(VideoCodec),
}

impl RungCodecInvariant {
    /// Human-readable diff for error messages. Empty when the two agree.
    pub(super) fn describe_diff(&self, other: &Self) -> String {
        if self == other {
            return String::new();
        }
        match (self, other) {
            (RungCodecInvariant::Av1(a), RungCodecInvariant::Av1(b)) => a.describe_diff(b),
            _ => format!("rung={self:?}, this worker={other:?}"),
        }
    }
}

/// H.264 / H.265 decode-init invariant, derived from the SPS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H26xInvariant {
    pub profile_idc: u8,
    pub level_idc: u8,
    pub chroma_format_idc: u8,
    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    pub width: u32,
    pub height: u32,
}

impl H26xInvariant {
    fn from_h264(sps: &H264SpsInfo) -> Self {
        Self {
            profile_idc: sps.profile_idc,
            level_idc: sps.level_idc,
            chroma_format_idc: sps.chroma_format_idc,
            bit_depth_luma: sps.bit_depth_luma,
            bit_depth_chroma: sps.bit_depth_chroma,
            width: sps.width.unwrap_or(0),
            height: sps.height.unwrap_or(0),
        }
    }

    fn from_h265(sps: &HevcSpsInfo) -> Self {
        Self {
            profile_idc: sps.profile_idc,
            level_idc: sps.level_idc,
            chroma_format_idc: sps.chroma_format_idc,
            bit_depth_luma: sps.bit_depth_luma,
            bit_depth_chroma: sps.bit_depth_chroma,
            width: sps.width.unwrap_or(0),
            height: sps.height.unwrap_or(0),
        }
    }
}

/// AV1 sequence-header invariant — every encoder contributing segments to a
/// single rendition MUST agree on these fields.
///
/// Why these specific fields: each is part of the codec-init contract that the
/// player sets up once from `av1C` and expects to hold for every segment. The
/// decoder re-parses the inline OBU sequence header in each segment's IDR; if
/// its parsed values disagree with the av1C from `init.mp4`, strict decoders
/// (dav1d in conformance mode, Safari AVFoundation, hls.js+libdav1d) reject the
/// segment. Optional fields (timing info, decoder model, film grain present,
/// operating points) are tolerated by every major player; we deliberately don't
/// check them so NVENC + QSV + AMF + the software AV1 encoder co-exist on one rendition without
/// cosmetic byte differences triggering false rejections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1Invariant {
    pub seq_profile: u8,
    pub seq_level_idx_0: u8,
    pub seq_tier_0: u8,
    pub bit_depth: u8,
    pub monochrome: bool,
    pub chroma_subsampling_x: bool,
    pub chroma_subsampling_y: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub color_range: bool,
    pub max_frame_width_minus1: u32,
    pub max_frame_height_minus1: u32,
    pub still_picture: bool,
}

impl Av1Invariant {
    pub fn from_sequence_header(sh: &Av1SequenceHeader) -> Self {
        Self {
            seq_profile: sh.seq_profile,
            seq_level_idx_0: sh.seq_level_idx_0,
            seq_tier_0: sh.seq_tier_0,
            bit_depth: sh.bit_depth,
            monochrome: sh.monochrome,
            chroma_subsampling_x: sh.chroma_subsampling_x,
            chroma_subsampling_y: sh.chroma_subsampling_y,
            color_primaries: sh.color_primaries,
            transfer_characteristics: sh.transfer_characteristics,
            matrix_coefficients: sh.matrix_coefficients,
            color_range: sh.color_range,
            max_frame_width_minus1: sh.max_frame_width_minus1,
            max_frame_height_minus1: sh.max_frame_height_minus1,
            still_picture: sh.still_picture,
        }
    }

    /// Human-readable diff for error messages.
    fn describe_diff(&self, other: &Self) -> String {
        let mut diffs = Vec::new();
        macro_rules! diff_field {
            ($field:ident) => {
                if self.$field != other.$field {
                    diffs.push(format!(
                        "{}: rung={:?}, this worker={:?}",
                        stringify!($field),
                        self.$field,
                        other.$field
                    ));
                }
            };
        }
        diff_field!(seq_profile);
        diff_field!(seq_level_idx_0);
        diff_field!(seq_tier_0);
        diff_field!(bit_depth);
        diff_field!(monochrome);
        diff_field!(chroma_subsampling_x);
        diff_field!(chroma_subsampling_y);
        diff_field!(color_primaries);
        diff_field!(transfer_characteristics);
        diff_field!(matrix_coefficients);
        diff_field!(color_range);
        diff_field!(max_frame_width_minus1);
        diff_field!(max_frame_height_minus1);
        diff_field!(still_picture);
        diffs.join("; ")
    }
}

/// Outcome of comparing a worker's first packet against the rung's
/// codec invariant. The caller — `run_encoder_worker_blocking` —
/// branches on this to decide whether to keep encoding, soft-fail
/// (requeue the chunk for another worker), or hard-fail (parse error
/// from a malformed bitstream).
#[derive(Debug)]
pub enum InvariantCheck {
    /// First worker on the rung. Invariant has been recorded.
    SetByThisWorker,
    /// Matches the rung's invariant. Proceed to publish.
    Matched,
    /// Mandatory fields mismatch. Worker should requeue its chunk and
    /// exit cleanly; the rung continues with workers whose vendors
    /// agree with the invariant the first worker set. **Mission-
    /// critical jobs DO NOT abort on this** — only this one helper's
    /// contribution is lost, and another worker picks up the chunk.
    Mismatched { diff: String },
}

/// Parse a worker's first packet, derive the codec invariant, and
/// compare-or-set it against the per-rung slot. Returns
/// [`InvariantCheck`] on a successful parse; an `Err` only on
/// malformed bitstream (the encoder failed to emit an
/// `OBU_SEQUENCE_HEADER` at all, which is a configuration bug that
/// nothing downstream can recover from).
pub fn validate_or_set_rung_invariant(
    rung_idx: usize,
    gpu_vendor: Option<codec::gpu::GpuVendor>,
    slot: &RwLock<Option<RungCodecInvariant>>,
    first_packet: &[u8],
    codec: VideoCodec,
) -> Result<InvariantCheck> {
    // Derive the codec invariant from the worker's first encoded packet: AV1
    // from the OBU sequence header, H.264/H.265 from the SPS in the Annex-B AU.
    let observed = match codec {
        VideoCodec::Av1 => {
            let parsed = parse_av1_sequence_header(first_packet).ok_or_else(|| {
                anyhow!(
                    "rung {} (vendor {:?}): could not parse AV1 sequence header from first \
                     encoded packet; encoder did not emit OBU_SEQUENCE_HEADER as required for \
                     segment alignment",
                    rung_idx,
                    gpu_vendor,
                )
            })?;
            RungCodecInvariant::Av1(Av1Invariant::from_sequence_header(&parsed))
        }
        VideoCodec::H264 => {
            let sps = parse_h264_sps(first_packet).ok_or_else(|| {
                anyhow!(
                    "rung {} (vendor {:?}): could not parse H.264 SPS from first encoded packet; \
                     encoder did not emit an SPS NAL on the first IDR",
                    rung_idx,
                    gpu_vendor,
                )
            })?;
            RungCodecInvariant::H26x(H26xInvariant::from_h264(&sps))
        }
        VideoCodec::H265 => {
            let sps = parse_hevc_sps(first_packet).ok_or_else(|| {
                anyhow!(
                    "rung {} (vendor {:?}): could not parse H.265 SPS from first encoded packet; \
                     encoder did not emit an SPS NAL on the first IRAP",
                    rung_idx,
                    gpu_vendor,
                )
            })?;
            RungCodecInvariant::H26x(H26xInvariant::from_h265(&sps))
        }
        VideoCodec::Vp9 => {
            let info = container::vpx::vp9_frame_info(first_packet).filter(|i| i.key_frame).ok_or_else(|| {
                anyhow!(
                    "rung {} (vendor {:?}): the first encoded VP9 packet is not a key frame",
                    rung_idx,
                    gpu_vendor,
                )
            })?;
            RungCodecInvariant::Vp9 { profile: info.profile, bit_depth: info.bit_depth, subsampling: info.subsampling }
        }
        other => RungCodecInvariant::Codec(other),
    };

    // Fast path: read lock, check if set + matches.
    if let Some(existing) = &*slot.read().unwrap() {
        if existing == &observed {
            return Ok(InvariantCheck::Matched);
        }
        return Ok(InvariantCheck::Mismatched {
            diff: existing.describe_diff(&observed),
        });
    }
    // First worker — write under write-lock with double-check (race
    // against another worker setting the slot between read and write).
    let mut w = slot.write().unwrap();
    match &*w {
        Some(existing) if existing != &observed => Ok(InvariantCheck::Mismatched {
            diff: existing.describe_diff(&observed),
        }),
        Some(_) => Ok(InvariantCheck::Matched),
        None => {
            tracing::info!(
                rung_idx,
                gpu_vendor = ?gpu_vendor,
                ?codec,
                invariant = ?observed,
                "rung codec invariant captured from first worker"
            );
            *w = Some(observed);
            Ok(InvariantCheck::SetByThisWorker)
        }
    }
}
