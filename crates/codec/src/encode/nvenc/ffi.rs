//! NVENC initialization and codec-config `#[repr(C)]` structs.

use std::ffi::c_void;

use super::constants::Guid;

// ─── NVENC API structs ────────────────────────────────────────────
//
// Layouts mirror vendor/nvidia/nvEncodeAPI.h (SDK 12.2). Where a Rust
// field is more granular than the header stub, the extended layout is
// taken from the full SDK 12.2 headers (`NV_ENC_RC_PARAMS`,
// `NV_ENC_CONFIG_AV1`, `NV_ENC_LOCK_BITSTREAM`). Compile-time
// `const_assert!`s at the bottom of this file verify exact sizes.

#[repr(C)]
pub(super) struct NvEncOpenEncodeSessionExParams {
    pub(super) version: u32,
    pub(super) device_type: u32,
    pub(super) device: *mut c_void,
    pub(super) reserved: *mut c_void,
    pub(super) api_version: u32,
    pub(super) reserved1: [u32; 253],
    pub(super) reserved2: [*mut c_void; 64],
}

/// `NV_ENC_INITIALIZE_PARAMS` — SDK 13.0 layout
/// (vendor/nvidia/nvEncodeAPI.h:2233-2292).
///
/// MAJOR LAYOUT CHANGE FROM SDK 12.2: the six u32 boolean fields after
/// `enable_ptd` (reportSliceOffsets, enableSubFrameWrite,
/// enableExternalMEHints, enableMEOnlyMode, enableWeightedPrediction,
/// enableOutputInVidmem) plus the `reserved[3]` block were COLLAPSED into
/// a single 32-bit bitfield word + new `privDataSize` u32 + `reserved` u32
/// + `privData` (void*). SDK 13 also packed in 4 NEW bitfield slots
/// (splitEncodeMode:4, enableReconFrameOutput:1, enableOutputStats:1,
/// enableUniDirectionalB:1) which the bitfield word now owns.
///
/// Other deltas:
///   - `maxMEHintCountsPerBlock[2]` is now `NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE[2]`
///     (was the wrong `[u32; 2]` mirror in 12.2 — incidentally compensated
///      for by the trailing reserved[287] over-size). Each element is 16 bytes
///     (1 bitfield u32 + 3 u32 reserved) → 32 bytes total.
///   - `numStateBuffers` (NEW in SDK 13) — encoding-without-state-advance.
///   - `outputStatsLevel` (NEW in SDK 13) — pairs with the new bitfield slot
///     `enableOutputStats`.
///   - `reserved1` shrunk from `[u32; 287]` → `[u32; 284]` to make room
///     for the two NEW u32 fields above. Total struct size unchanged at
///     1800 bytes — the new fields steal trailing reserved space.
///
/// The override block in NvencEncoder::new only sets `enable_encode_async = 0`
/// and `enable_ptd = 1`, so the bitfield rewrite is a no-op for our caller
/// (every other flag was already implicitly zero via mem::zeroed). New helpers
/// AV1_INIT_BIT_* below are provided for completeness in case a future caller
/// wants to set splitEncodeMode etc.
#[repr(C)]
pub(super) struct NvEncInitializeParams {
    pub(super) version: u32,
    pub(super) encode_guid: Guid,
    pub(super) preset_guid: Guid,
    pub(super) encode_width: u32,
    pub(super) encode_height: u32,
    pub(super) dar_width: u32,
    pub(super) dar_height: u32,
    pub(super) frame_rate_num: u32,
    pub(super) frame_rate_den: u32,
    pub(super) enable_encode_async: u32,
    pub(super) enable_ptd: u32,
    /// Bitfield word collapsing 11 boolean / packed flags. See INIT_BIT_*
    /// constants below for the layout.
    pub(super) flags: u32,
    pub(super) priv_data_size: u32,
    pub(super) reserved: u32,
    pub(super) priv_data: *mut c_void,
    pub(super) encode_config: *mut c_void,
    pub(super) max_encode_width: u32,
    pub(super) max_encode_height: u32,
    /// Each element is `NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE`
    /// = 1 bitfield u32 + 3 u32 reserved = 16 bytes. We mirror as a flat
    /// 8-u32 array (`[u32; 8]` = 32 bytes) since we never set any of the
    /// hint counts in this service (external ME hints are not used).
    pub(super) max_me_hint_counts_per_block: [u32; 8],
    pub(super) tuning_info: u32,
    pub(super) buffer_format: u32,
    /// NEW in SDK 13: number of state buffers for stateless encode flows.
    /// Always 0 (= single state buffer = our use case).
    pub(super) num_state_buffers: u32,
    /// NEW in SDK 13: granularity for encoded-frame output stats. Always 0
    /// (= NV_ENC_OUTPUT_STATS_NONE). Pairs with the new `enableOutputStats`
    /// bit (bit 9 in the flags word above) which we leave at 0.
    pub(super) output_stats_level: u32,
    pub(super) reserved1: [u32; 284],
    pub(super) reserved2: [*mut c_void; 64],
}

/// `NV_ENC_RECONFIGURE_PARAMS` (nvEncodeAPI.h, unchanged SDK 12.2 → 13.0):
///
/// ```c
/// typedef struct _NV_ENC_RECONFIGURE_PARAMS {
///     uint32_t                 version;
///     NV_ENC_INITIALIZE_PARAMS reInitEncodeParams;
///     uint32_t                 resetEncoder : 1;
///     uint32_t                 forceIDR     : 1;
///     uint32_t                 reserved     : 30;
/// } NV_ENC_RECONFIGURE_PARAMS;
/// ```
///
/// `reInitEncodeParams` holds pointers, so it is 8-aligned and starts at
/// offset 8; the bitfield word follows at 1808 and the struct pads to 1816.
/// The `const_assert!` below pins that.
#[repr(C)]
pub(super) struct NvEncReconfigureParams {
    pub(super) version: u32,
    pub(super) re_init_encode_params: NvEncInitializeParams,
    /// bit 0 `resetEncoder`, bit 1 `forceIDR`.
    pub(super) flags: u32,
}

/// `NV_ENC_RECONFIGURE_PARAMS.resetEncoder`: discard the rate-control and
/// lookahead state and start the stream over.
pub(super) const RECONFIGURE_BIT_RESET_ENCODER: u32 = 1 << 0;
/// `NV_ENC_RECONFIGURE_PARAMS.forceIDR`: the next picture opens a new GOP.
pub(super) const RECONFIGURE_BIT_FORCE_IDR: u32 = 1 << 1;

// Bitfield helpers for `NvEncInitializeParams.flags`. SDK 13 packs 11
// flags into one u32. Bit layout (LSB first):
//   bit  0:    reportSliceOffsets
//   bit  1:    enableSubFrameWrite
//   bit  2:    enableExternalMEHints
//   bit  3:    enableMEOnlyMode
//   bit  4:    enableWeightedPrediction
//   bits 5-8:  splitEncodeMode (4 bits, NEW in SDK 13)
//   bit  9:    enableOutputInVidmem
//   bit  10:   enableReconFrameOutput (NEW)
//   bit  11:   enableOutputStats (NEW)
//   bit  12:   enableUniDirectionalB (NEW)
//   bits 13-31: reservedBitFields
#[allow(dead_code)]
pub(super) const INIT_BIT_REPORT_SLICE_OFFSETS: u32 = 1 << 0;
#[allow(dead_code)]
pub(super) const INIT_BIT_ENABLE_SUB_FRAME_WRITE: u32 = 1 << 1;
#[allow(dead_code)]
pub(super) const INIT_BIT_ENABLE_EXTERNAL_ME_HINTS: u32 = 1 << 2;
#[allow(dead_code)]
pub(super) const INIT_BIT_ENABLE_ME_ONLY_MODE: u32 = 1 << 3;
#[allow(dead_code)]
pub(super) const INIT_BIT_ENABLE_WEIGHTED_PREDICTION: u32 = 1 << 4;
#[allow(dead_code)]
pub(super) const INIT_BIT_ENABLE_OUTPUT_IN_VIDMEM: u32 = 1 << 9;
#[allow(dead_code)]
pub(super) const INIT_BIT_ENABLE_RECON_FRAME_OUTPUT: u32 = 1 << 10;
#[allow(dead_code)]
pub(super) const INIT_BIT_ENABLE_OUTPUT_STATS: u32 = 1 << 11;
#[allow(dead_code)]
pub(super) const INIT_BIT_ENABLE_UNI_DIRECTIONAL_B: u32 = 1 << 12;

/// Rate-control params. Layout is the full SDK 12.2 definition from
/// `_NV_ENC_RC_PARAMS` (not the minimal vendor stub at
/// `vendor/nvidia/nvEncodeAPI.h:154-170`).
///
/// For AV1 under NVENC the `targetQuality` field range is 0..63 for
/// AV1 (0..51 only applies to H.264/HEVC, per NVENC SDK 13 §3.8.3).
#[repr(C)]
pub(super) struct NvEncRcParams {
    pub(super) version: u32,
    pub(super) rate_control_mode: u32,
    pub(super) const_qp_inter_p: u32,
    pub(super) const_qp_inter_b: u32,
    pub(super) const_qp_intra: u32,
    pub(super) average_bitrate: u32,
    pub(super) max_bitrate: u32,
    pub(super) vbv_buffer_size: u32,
    pub(super) vbv_initial_delay: u32,
    /// Bitfield packed as SDK does (enableMinQP, enableMaxQP, enableAQ,
    /// enableLookahead, aqStrength nibble, etc).
    pub(super) flags: u32,
    pub(super) min_qp_inter_p: u32,
    pub(super) min_qp_inter_b: u32,
    pub(super) min_qp_intra: u32,
    pub(super) max_qp_inter_p: u32,
    pub(super) max_qp_inter_b: u32,
    pub(super) max_qp_intra: u32,
    pub(super) initial_rc_qp_inter_p: u32,
    pub(super) initial_rc_qp_inter_b: u32,
    pub(super) initial_rc_qp_intra: u32,
    /// 12 bytes covering `temporallayerIdxMask: u32` + `temporalLayerQP[8]: u8`
    /// in SDK 13 (vendor/nvidia/nvEncodeAPI.h:1586-1589). Was wrongly mirrored
    /// as `[u32; 2]` (8 bytes) in the SDK 12.2 mirror — that 4-byte deficit
    /// shifted every subsequent field by 4 bytes which the trailing reserved
    /// padding silently absorbed at the END but mis-mapped the intermediate
    /// field offsets. We always set this to all-zero (no temporal-layer
    /// bitrate plumbing in this service).
    pub(super) temporally_layer_bitrate_ratio: [u32; 3],
    /// CQ quality target. Range: 0..63 for AV1, 0..51 for H.264/HEVC.
    pub(super) target_quality: u8,
    /// 8-bit fractional part of `target_quality` (8.8 fixed-point).
    /// Left at 0 — whole-step CQ is enough for our VMAF bands.
    pub(super) target_quality_lsb: u8,
    pub(super) lookahead_depth: u16,
    pub(super) low_delay_key_frame_scale: u32,
    pub(super) qp_map_mode: u32,
    pub(super) multi_pass: u32,
    pub(super) alpha_layer_bitrate_ratio: u32,
    pub(super) cbqpi_ofs: i8,
    pub(super) cbqpp_ofs: i8,
    pub(super) crqpi_ofs: i8,
    pub(super) crqpp_ofs: i8,
    pub(super) reserved: [u32; 4],
}

/// AV1-specific codec config. Lives inside NV_ENC_CONFIG's
/// encodeCodecConfig union. SDK 13.0 layout — see
/// vendor/nvidia/nvEncodeAPI.h `_NV_ENC_CONFIG_AV1`.
///
/// MAJOR LAYOUT CHANGE FROM SDK 12.2: all the boolean enable_*
/// fields collapsed into ONE 32-bit bitfield word at offset 16. The
/// override block in NvencEncoder::new now sets that bitfield via
/// shifts/masks. Three new groups of fields appended after the
/// original layout: outputBitDepth/inputBitDepth (replaced
/// pixel_bit_depth_minus_8), numFwdRefs/numBwdRefs typed enums,
/// ltrNumFrames/numTemporalLayers/tfLevel temporal-layer config.
#[repr(C)]
pub(super) struct NvEncConfigAv1 {
    pub(super) level: u32,
    pub(super) tier: u32,
    pub(super) min_part_size: u32,
    pub(super) max_part_size: u32,
    /// Bitfield word — see ::AV1_BIT_* constants. The SDK declares
    /// these as C bitfields; we mirror as a single u32 + helpers
    /// because Rust doesn't have C-style bitfield syntax. Bit
    /// layout (LSB first):
    ///   bit  0:    outputAnnexBFormat (0 = LOB / MP4-friendly, 1 = AnnexB)
    ///   bit  1:    enableTimingInfo
    ///   bit  2:    enableDecoderModelInfo
    ///   bit  3:    enableFrameIdNumbers
    ///   bit  4:    disableSeqHdr
    ///   bit  5:    repeatSeqHdr (set 1 for keyframe-seekable MP4)
    ///   bit  6:    enableIntraRefresh
    ///   bits 7-8:  chromaFormatIDC (1 for 4:2:0)
    ///   bit  9:    enableBitstreamPadding
    ///   bit  10:   enableCustomTileConfig
    ///   bit  11:   enableFilmGrainParams
    ///   bit  12:   enableLTR
    ///   bit  13:   enableTemporalSVC
    ///   bit  14:   outputMaxCll
    ///   bit  15:   outputMasteringDisplay
    ///   bits 16+:  reserved
    pub(super) flags: u32,
    pub(super) idr_period: u32,
    pub(super) intra_refresh_period: u32,
    pub(super) intra_refresh_cnt: u32,
    pub(super) max_num_ref_frames_in_dpb: u32,
    pub(super) num_tile_columns: u32,
    pub(super) num_tile_rows: u32,
    pub(super) reserved2: u32,
    pub(super) tile_widths: *mut u32,
    pub(super) tile_heights: *mut u32,
    pub(super) max_temporal_layers_minus1: u32,
    pub(super) color_primaries: u32,
    pub(super) transfer_characteristics: u32,
    pub(super) matrix_coefficients: u32,
    pub(super) color_range: u32,
    pub(super) chroma_sample_position: u32,
    pub(super) use_b_frames_as_ref: u32,
    pub(super) film_grain_params: *mut c_void,
    pub(super) num_fwd_refs: u32,
    pub(super) num_bwd_refs: u32,
    /// `NV_ENC_BIT_DEPTH` enum — the literal bit depth (8 or 10).
    pub(super) output_bit_depth: u32,
    pub(super) input_bit_depth: u32,
    pub(super) ltr_num_frames: u32,
    pub(super) num_temporal_layers: u32,
    pub(super) tf_level: u32,
    pub(super) reserved1: [u32; 230],
    pub(super) reserved3: [*mut c_void; 62],
}

// `NV_ENC_BIT_DEPTH` enum values (nvEncodeAPI.h). The driver wants the literal
// bit depth, not a 0/1 ordinal — both AV1 and HEVC config unions use this enum.
pub(super) const NV_ENC_BIT_DEPTH_8: u32 = 8;
pub(super) const NV_ENC_BIT_DEPTH_10: u32 = 10;

/// Partial SDK-13 `NV_ENC_CONFIG_HEVC` view — only the leading fields up to
/// `outputBitDepth` / `inputBitDepth` (offsets 200 / 204), overlaid on the
/// codec-config union to set Main 10 bit depth without mirroring the whole
/// struct. The nested `NV_ENC_CONFIG_HEVC_VUI_PARAMETERS` (= H.264 VUI, 28×u32
/// = 112 bytes) sits between `maxTemporalLayersMinus1` and `ltrTrustMode`.
#[repr(C)]
pub(super) struct NvEncConfigHevcBitDepth {
    pub(super) level: u32,
    pub(super) tier: u32,
    pub(super) min_cu_size: u32,
    pub(super) max_cu_size: u32,
    pub(super) flags: u32,
    pub(super) idr_period: u32,
    pub(super) intra_refresh_period: u32,
    pub(super) intra_refresh_cnt: u32,
    pub(super) max_num_ref_frames_in_dpb: u32,
    pub(super) ltr_num_frames: u32,
    pub(super) vps_id: u32,
    pub(super) sps_id: u32,
    pub(super) pps_id: u32,
    pub(super) slice_mode: u32,
    pub(super) slice_mode_data: u32,
    pub(super) max_temporal_layers_minus1: u32,
    pub(super) hevc_vui: [u32; 28],
    pub(super) ltr_trust_mode: u32,
    pub(super) use_b_frames_as_ref: u32,
    pub(super) num_ref_l0: u32,
    pub(super) num_ref_l1: u32,
    pub(super) tf_level: u32,
    pub(super) disable_deblocking_filter_idc: u32,
    pub(super) output_bit_depth: u32,
    pub(super) input_bit_depth: u32,
}
const _: () = assert!(std::mem::offset_of!(NvEncConfigHevcBitDepth, output_bit_depth) == 200);
const _: () = assert!(std::mem::offset_of!(NvEncConfigHevcBitDepth, input_bit_depth) == 204);

// Bitfield positions in NvEncConfigAv1.flags. Used by the override
// block to set specific enable flags without bit-twiddling at the
// call site.
pub(super) const AV1_BIT_OUTPUT_ANNEXB_FORMAT: u32 = 1 << 0;
#[allow(dead_code)]
pub(super) const AV1_BIT_ENABLE_TIMING_INFO: u32 = 1 << 1;
pub(super) const AV1_BIT_REPEAT_SEQ_HDR: u32 = 1 << 5;
// chromaFormatIDC occupies bits 7-8; value 1 (= 4:2:0) goes in bit 7.
pub(super) const AV1_CHROMA_FORMAT_IDC_420: u32 = 1 << 7;

/// Full encode config containing RC params + codec-specific union.
/// Size is fixed per SDK version; `const_assert!` verifies. SDK 13.0
/// layout — see vendor/nvidia/nvEncodeAPI.h `_NV_ENC_CONFIG`.
///
/// 2026-05-01 audit (#2): the `encodeCodecConfig` slot is the C
/// `NV_ENC_CODEC_CONFIG` UNION whose `sizeof` is driven by the LARGEST
/// variant — NV_ENC_CONFIG_H264 at 1792 bytes (HEVC=1560, AV1=1552).
/// Our previous mirror sized the slot to NV_ENC_CONFIG_AV1 (1552)
/// only, so the struct was 240 bytes shorter than the C ABI expected.
/// The driver then read/wrote 240 bytes past the end of our
/// stack-allocated `enc_config` during `NvEncInitializeEncoder` /
/// `NvEncGetEncodePresetConfigEx` — undefined behaviour that worked by
/// luck on prior runs. Verified against `sizeof(NV_ENC_CONFIG)` from
/// the vendored SDK 13 header: 3584. Trailing `_codec_config_pad`
/// brings the union slot up to 1792 without touching `codec_config_av1`'s
/// field offsets (which the encoder code reads at relative offsets).
///
/// Pre-SDK-13 audit gap (#1): this mirror was MISSING the trailing
/// `reserved2: [void*; 64]` field — the C struct has had it since at
/// least 12.2. Restored 2026-05-01 alongside the SDK 13 refresh.
#[repr(C)]
pub(super) struct NvEncConfig {
    pub(super) version: u32,
    pub(super) profile_guid: Guid,
    pub(super) gop_length: u32,
    pub(super) frame_interval_p: u32,
    pub(super) mono_chrome_encoding: u32,
    pub(super) frame_field_mode: u32,
    pub(super) mv_precision: u32,
    pub(super) rc_params: NvEncRcParams,
    pub(super) codec_config_av1: NvEncConfigAv1,
    /// Trailing pad to widen the `NV_ENC_CODEC_CONFIG` union slot from
    /// the AV1 variant size (1552) up to the H.264 variant size (1792)
    /// — which is what the C union sizes to. Driver may write into
    /// these bytes for variant-agnostic reserved fields; we keep them
    /// zero-initialised via `mem::zeroed()`. The encoder reads
    /// `codec_config_av1.flags` / `.idr_period` / etc. which all live
    /// inside the AV1-sized region BEFORE this pad, so the override
    /// block in `NvencEncoder::new` is unaffected.
    pub(super) _codec_config_pad: [u32; 60],
    pub(super) reserved: [u32; 278],
    pub(super) reserved2: [*mut c_void; 64],
}

/// `_NV_ENC_PRESET_CONFIG` — wrapper around `NV_ENC_CONFIG` that
/// `NvEncGetEncodePresetConfigEx` populates with preset+tuning
/// defaults. SDK 13.0 layout (added a leading `reserved` u32, grew
/// reserved1 from 255 → 256):
///   u32 version
///   u32 reserved   ← NEW in SDK 13
///   NV_ENC_CONFIG presetCfg
///   u32 reserved1[256]   ← was [255] in 12.2
///   void* reserved2[64]
#[repr(C)]
pub(super) struct NvEncPresetConfig {
    pub(super) version: u32,
    pub(super) reserved: u32,
    pub(super) preset_cfg: NvEncConfig,
    pub(super) reserved1: [u32; 256],
    pub(super) reserved2: [*mut c_void; 64],
}

// ─── Compile-time size assertions for init/config structs ─────────

// NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS (vendor/nvidia/nvEncodeAPI.h:144-152).
// u32 + u32 + ptr + ptr + u32 + u32[253] + ptr[64] = 1552.
const _: () = assert!(std::mem::size_of::<NvEncOpenEncodeSessionExParams>() == 1552);

// NV_ENC_INITIALIZE_PARAMS (vendor/nvidia/nvEncodeAPI.h:173-200). After
// GUID+14×u32+3×u32+ptr+2×u32+[u32;2]+2×u32+287×u32+64×ptr with ptr
// alignment: 1800 bytes measured.
const _: () = assert!(std::mem::size_of::<NvEncInitializeParams>() == 1800);
// NV_ENC_RECONFIGURE_PARAMS: 4 (version) + 4 (pad to the 8-aligned params)
// + 1800 + 4 (bitfield word) + 4 (tail pad) = 1816.
const _: () = assert!(std::mem::size_of::<NvEncReconfigureParams>() == 1816);
const _: () = assert!(std::mem::offset_of!(NvEncReconfigureParams, re_init_encode_params) == 8);
const _: () = assert!(std::mem::offset_of!(NvEncReconfigureParams, flags) == 1808);

// NV_ENC_RC_PARAMS — SDK 13.0 layout
// (vendor/nvidia/nvEncodeAPI.h:1555-1627). SDK 13 inserted
// `lookaheadLevel: NV_ENC_LOOKAHEAD_LEVEL (u32)` between
// `reserved2` and `viewBitrateRatios[]` and grew temporallayerIdxMask
// + temporalLayerQP[8] from the previous mis-mirrored 8-byte slot to
// the spec-correct 12 bytes. Total grew 124 → 128 bytes.
const _: () = assert!(std::mem::size_of::<NvEncRcParams>() == 128);

// NV_ENC_CONFIG_AV1 — SDK 13.0 `_NV_ENC_CONFIG_AV1`. Bitfield-packed
// flags + 14×u32 + 2×ptr (tile_widths/heights) + 6×u32 enums + 1×ptr
// (film_grain_params) + 7×u32 + reserved1[230] + reserved3[62 ptrs]
// with alignment padding. MSVC x64 size: 1552.
const _: () = assert!(std::mem::size_of::<NvEncConfigAv1>() == 1552);

// NV_ENC_CONFIG — SDK 13.0 `_NV_ENC_CONFIG` (version macro ver=9 at
// vendor/nvidia/nvEncodeAPI.h:2200). u32 + GUID(16) + 5×u32 +
// RC_PARAMS(128) + CODEC_CONFIG_UNION(1792, sized to H264 variant) +
// reserved[278] + reserved2[64 ptrs] = 3584 bytes. Verified against
// `sizeof(NV_ENC_CONFIG)` printed by a small C harness compiled
// against the vendored SDK 13 header on the dev box.
const _: () = assert!(std::mem::size_of::<NvEncConfig>() == 3584);

// NV_ENC_PRESET_CONFIG — SDK 13.0 `_NV_ENC_PRESET_CONFIG` (added a
// leading reserved u32 + grew reserved1 from [255] to [256]).
// 2×u32 + NV_ENC_CONFIG(3584) + reserved1[256] + reserved2[64 ptrs]
// = 5128 bytes.
const _: () = assert!(std::mem::size_of::<NvEncPresetConfig>() == 5128);
