use bytes::Bytes;

/// Output video codec for the encoder + muxer. AV1 is the project default
/// (royalty-clean); H.264 / H.265 are selectable for compatibility with
/// legacy players (they carry patent-licensing obligations — see the docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VideoCodec {
    #[default]
    Av1,
    H264,
    H265,
}

impl VideoCodec {
    /// Short lowercase label (`"av1"` / `"h264"` / `"h265"`).
    pub fn label(self) -> &'static str {
        match self {
            VideoCodec::Av1 => "av1",
            VideoCodec::H264 => "h264",
            VideoCodec::H265 => "h265",
        }
    }

    /// The ISOBMFF visual sample-entry fourcc (`av01` / `avc1` / `hvc1`).
    pub fn sample_entry_fourcc(self) -> &'static str {
        match self {
            VideoCodec::Av1 => "av01",
            VideoCodec::H264 => "avc1",
            VideoCodec::H265 => "hvc1",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Yuv420p,
    Yuv420p10le,
    Yuv420p12le,
    Yuv422p,
    Yuv422p10le,
    Yuv444p,
    Yuv444p10le,
    /// 4-plane 10-bit 4:4:4 with alpha. Y/Cb/Cr stored as u16 LE in the
    /// 0..=1023 range (10-bit sample domain). Alpha stored as u16 LE in
    /// the 0..=65535 range (16-bit precision — RDD 36 §7.7 alpha stream
    /// carries 16-bit samples for `ap4h`/`ap4x`, we preserve that rather
    /// than re-quantize down to 10-bit). Matches the ffmpeg
    /// `yuva444p10le` naming convention but the alpha plane is
    /// effectively 16-bit — documented limitation for downstream pipeline
    /// consumers (which today only accept 8-bit YUV420p; roadmap item #5
    /// tracks 10-bit end-to-end, after which a further extension can
    /// carry alpha too).
    Yuva444p10le,
    Nv12,
    Nv21,
    Rgb24,
    Rgba32,
}

impl PixelFormat {
    pub fn bytes_per_frame(&self, width: u32, height: u32) -> usize {
        let pixels = (width as usize) * (height as usize);
        match self {
            Self::Yuv420p | Self::Nv12 | Self::Nv21 => pixels * 3 / 2,
            Self::Yuv420p10le | Self::Yuv420p12le => pixels * 3,
            Self::Yuv422p => pixels * 2,
            Self::Yuv422p10le => pixels * 4,
            Self::Yuv444p => pixels * 3,
            Self::Yuv444p10le => pixels * 6,
            // 4 planes × 2 bytes/sample. Alpha is 16-bit, Y/Cb/Cr are
            // 10-bit stored in 16-bit containers — total 8 bytes/pixel.
            Self::Yuva444p10le => pixels * 8,
            Self::Rgb24 => pixels * 3,
            Self::Rgba32 => pixels * 4,
        }
    }

    /// ffmpeg-compatible string. Used in probe payloads so downstream
    /// consumers (Laravel, validators) see the same names the Python
    /// implementation emitted.
    pub fn as_ffmpeg_str(&self) -> &'static str {
        match self {
            Self::Yuv420p => "yuv420p",
            Self::Yuv420p10le => "yuv420p10le",
            Self::Yuv420p12le => "yuv420p12le",
            Self::Yuv422p => "yuv422p",
            Self::Yuv422p10le => "yuv422p10le",
            Self::Yuv444p => "yuv444p",
            Self::Yuv444p10le => "yuv444p10le",
            Self::Yuva444p10le => "yuva444p10le",
            Self::Nv12 => "nv12",
            Self::Nv21 => "nv21",
            Self::Rgb24 => "rgb24",
            Self::Rgba32 => "rgba",
        }
    }

    pub fn from_chroma_and_depth(chroma_idc: u8, bit_depth: u8) -> Self {
        match (chroma_idc, bit_depth) {
            (1, 8) => Self::Yuv420p,
            (1, 10) => Self::Yuv420p10le,
            (1, 12) => Self::Yuv420p12le,
            (2, 8) => Self::Yuv422p,
            (2, 10) => Self::Yuv422p10le,
            (3, 8) => Self::Yuv444p,
            (3, 10) => Self::Yuv444p10le,
            _ => Self::Yuv420p, // defensive default
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Bt601,
    Bt709,
    /// Covers both Bt2020 non-constant luminance and constant luminance
    /// matrix variants (ITU-T H.273 matrix_coefficients 9 and 10). The
    /// distinction rarely matters at the decode interface — downstream
    /// mux writes it into the `colr nclx` box's matrix_coefficients
    /// field which is carried separately on `StreamInfo` via the raw
    /// `matrix_coefficients` u8, not re-derived from this enum.
    Bt2020,
}

/// Transfer characteristics per ITU-T H.273 §8.2 / H.265 Table E.4.
/// Carried on `StreamInfo` so the MP4 mux's `colr nclx` writer can
/// round-trip HDR10 (ST2084) and HLG content without losing metadata.
///
/// Separate from `ColorSpace` so existing call sites — every decoder
/// emits a `VideoFrame { color_space, .. }` and every colorspace
/// converter / encoder dispatches on it — continue to compile
/// unchanged. The transfer function is orthogonal to the matrix
/// coefficients for pipeline purposes; only the mux needs both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransferFn {
    /// Gamma 2.2 / Rec. 709 (H.273 value 1). Default for SDR content.
    #[default]
    Bt709,
    /// Gamma 2.8 / BT.470BG.
    Bt470Bg,
    /// Linear (H.273 value 8).
    Linear,
    /// SMPTE ST 2084 / PQ (H.273 value 16). HDR10.
    St2084,
    /// ARIB STD-B67 / HLG (H.273 value 18). Broadcast HDR.
    AribStdB67,
    /// Unspecified or unmapped — consumers fall back to Bt709 gamma.
    Unspecified,
}

impl TransferFn {
    /// Map an H.273 `transfer_characteristics` value to the subset of
    /// transfers this pipeline knows about. Unknown values collapse
    /// to `Unspecified`.
    pub fn from_h273(value: u8) -> Self {
        match value {
            1 | 6 | 14 | 15 => Self::Bt709, // Rec.709 family
            4 => Self::Bt470Bg,
            8 => Self::Linear,
            16 => Self::St2084,
            18 => Self::AribStdB67,
            _ => Self::Unspecified,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VideoFrame {
    pub data: Bytes,
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub color_space: ColorSpace,
    pub pts: u64,
}

impl VideoFrame {
    pub fn new(
        data: Bytes,
        width: u32,
        height: u32,
        format: PixelFormat,
        color_space: ColorSpace,
        pts: u64,
    ) -> Self {
        Self {
            data,
            width,
            height,
            format,
            color_space,
            pts,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamInfo {
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub frame_rate: f64,
    pub duration: f64,
    pub pixel_format: PixelFormat,
    pub color_space: ColorSpace,
    pub total_frames: u64,
    pub bitrate: u64,
    /// HDR-relevant metadata. Bundled into one sub-struct that defaults
    /// to SDR BT.709 so every existing `StreamInfo { ... }` literal in
    /// the codebase compiles unchanged via `..Default::default()` or
    /// direct field init; only HDR-aware sites (nvdec sequence_callback,
    /// HEVC/AV1 SPS/VUI parsers, MP4 mux `colr nclx` writer) populate
    /// non-default values.
    pub color_metadata: ColorMetadata,
}

/// HDR / wide-gamut metadata carried from SPS VUI through the pipeline
/// to the MP4 mux `colr nclx` box. All values default to an SDR
/// BT.709 baseline so un-annotated StreamInfo constructions stay
/// backward-compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorMetadata {
    /// Transfer function per ITU-T H.273. Defaults to Bt709 for SDR.
    /// HDR10 streams populate this with St2084 (PQ); HLG broadcasts
    /// with AribStdB67.
    pub transfer: TransferFn,
    /// Raw H.273 `matrix_coefficients` (0–255). Preserves the ncl/cl
    /// distinction the ColorSpace enum collapses: 9 = BT.2020 NCL,
    /// 10 = BT.2020 CL. Mux writes this verbatim into `colr nclx`.
    pub matrix_coefficients: u8,
    /// Raw H.273 `colour_primaries` (0–255). Written verbatim into
    /// `colr nclx`.
    pub colour_primaries: u8,
    /// `full_range_flag` (H.273): false = studio/limited-range (16..235),
    /// true = full-range (0..255). HEVC SPS VUI exposes this directly.
    pub full_range: bool,
    /// HDR10 mastering display color volume (SMPTE ST 2086, HEVC SEI 137,
    /// AV1 metadata OBU type 2 HDR_MDCV, MP4 `mdcv` box, MKV
    /// `MasteringMetadata`). `None` for SDR sources or when the
    /// upstream did not signal it. Carried to the MP4 mux's `mdcv` box
    /// per ISO/IEC 14496-12 §12.1.6 / AV1-ISOBMFF v1.3.0 (Squad-20).
    /// Populated by Squad-21 from HEVC SEI 137 / AV1 metadata OBU
    /// `METADATA_TYPE_HDR_MDCV` / MP4 `mdcv` / MKV `MasteringMetadata`.
    /// Without it, Apple devices fall back to BT.709 limited even when
    /// `colr nclx` signals BT.2020.
    pub mastering_display: Option<MasteringDisplay>,
    /// HDR10 content light level info (CTA-861.3, HEVC SEI 144, AV1
    /// metadata OBU type 1 HDR_CLL, MP4 `clli`, MKV `MaxCLL` +
    /// `MaxFALL`). `None` for SDR or unsignalled HDR. Carried to the
    /// MP4 mux's `clli` box per ISO/IEC 14496-12 §12.1.6 / AV1-ISOBMFF
    /// v1.3.0 (Squad-20). Populated by Squad-21 from HEVC SEI 144 / AV1
    /// metadata OBU `METADATA_TYPE_HDR_CLL` / MP4 `clli` / MKV.
    pub content_light_level: Option<ContentLightLevel>,
}

impl Default for ColorMetadata {
    fn default() -> Self {
        // SDR BT.709 baseline: matrix=1, primaries=1, transfer=Bt709,
        // studio range. Matches the implicit behavior of every existing
        // decoder that didn't previously populate color metadata.
        Self {
            transfer: TransferFn::Bt709,
            matrix_coefficients: 1,
            colour_primaries: 1,
            full_range: false,
            mastering_display: None,
            content_light_level: None,
        }
    }
}

/// HDR10 Mastering Display Color Volume per SMPTE ST 2086 / HEVC SEI
/// message 137 (D.2.28 in the H.265 spec) / AV1 Metadata OBU
/// `METADATA_TYPE_HDR_MDCV`. Wire-encoded into the MP4 `mdcv` box as
/// 8 × u16 BE primaries/white-point + 2 × u32 BE luminance, total 24
/// bytes payload.
///
/// **Units (per the spec):**
/// - `primaries_*_x` / `primaries_*_y` / `white_point_*` are in
///   increments of 0.00002 of the CIE 1931 chromaticity diagram. The
///   wire format is the unscaled u16 (e.g. BT.2020 red x=0.708 →
///   `(0.708 / 0.00002) = 35400`).
/// - `max_luminance` and `min_luminance` are in increments of 0.0001
///   cd/m² (nits). The wire format is the unscaled u32 (e.g.
///   1000 nits → `10_000_000`).
///
/// **Field-name contract** with Squad-21 (probe HDR): these names are
/// load-bearing — the probe imports this struct and populates it
/// directly from the SEI/OBU payload. Do not rename without coordinating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MasteringDisplay {
    pub primaries_r_x: u16,
    pub primaries_r_y: u16,
    pub primaries_g_x: u16,
    pub primaries_g_y: u16,
    pub primaries_b_x: u16,
    pub primaries_b_y: u16,
    pub white_point_x: u16,
    pub white_point_y: u16,
    pub max_luminance: u32,
    pub min_luminance: u32,
}

/// HDR10 Content Light Level Information per CTA-861.3 / HEVC SEI 144
/// (content_light_level_info) / AV1 Metadata OBU
/// `METADATA_TYPE_HDR_CLL`. Wire-encoded into the MP4 `clli` box as
/// 2 × u16 BE, total 4 bytes payload.
///
/// **Units (per the spec):**
/// - `max_cll` — Maximum Content Light Level, peak luminance of the
///   brightest pixel anywhere in the stream, in cd/m² (integer nits).
/// - `max_fall` — Maximum Frame-Average Light Level, peak per-frame
///   average luminance, in cd/m² (integer nits).
///
/// **Field-name contract** with Squad-21: load-bearing names; do not
/// rename without coordinating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContentLightLevel {
    pub max_cll: u16,
    pub max_fall: u16,
}
