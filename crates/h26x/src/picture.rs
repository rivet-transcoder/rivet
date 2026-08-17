//! The decoded picture both decoders hand back.

/// Chroma sampling of a decoded picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaFormat {
    /// Luma only (`chroma_format_idc` 0).
    Monochrome,
    /// 4:2:0.
    Yuv420,
    /// 4:2:2.
    Yuv422,
    /// 4:4:4.
    Yuv444,
}

impl ChromaFormat {
    /// `(SubWidthC, SubHeightC)` — how many luma samples one chroma sample
    /// covers in each direction (1 for monochrome, which has no chroma).
    pub fn subsampling(self) -> (u32, u32) {
        match self {
            ChromaFormat::Monochrome => (1, 1),
            ChromaFormat::Yuv420 => (2, 2),
            ChromaFormat::Yuv422 => (2, 1),
            ChromaFormat::Yuv444 => (1, 1),
        }
    }
    /// From `chroma_format_idc`.
    pub fn from_idc(idc: u32) -> Option<Self> {
        Some(match idc {
            0 => ChromaFormat::Monochrome,
            1 => ChromaFormat::Yuv420,
            2 => ChromaFormat::Yuv422,
            3 => ChromaFormat::Yuv444,
            _ => return None,
        })
    }
}

/// One plane of samples, tightly packed (stride == width), 8-bit samples as
/// one byte each, higher bit depths as little-endian `u16` (values in the low
/// bits).
#[derive(Debug, Clone)]
pub struct Plane {
    /// Sample data.
    pub data: Vec<u8>,
    /// Width in samples.
    pub width: u32,
    /// Height in samples.
    pub height: u32,
}

/// A decoded, cropped picture in output order.
#[derive(Debug, Clone)]
pub struct Picture {
    /// Luma width after cropping.
    pub width: u32,
    /// Luma height after cropping.
    pub height: u32,
    /// Bits per sample (8, 9, 10, 12).
    pub bit_depth: u32,
    /// Chroma sampling.
    pub chroma: ChromaFormat,
    /// Y, then Cb, then Cr (the last two absent for monochrome).
    pub planes: Vec<Plane>,
    /// Picture order count — the standard's own display-order key. Frames
    /// come out sorted by it; a caller can use it to detect gaps.
    pub poc: i32,
    /// Decode-order index of the frame (0 for the first decoded picture).
    pub decode_index: u64,
}

impl Picture {
    /// The planes concatenated: Y then Cb then Cr — the layout of a packed
    /// planar frame (and of libavcodec's `framemd5`).
    pub fn packed(&self) -> Vec<u8> {
        let n: usize = self.planes.iter().map(|p| p.data.len()).sum();
        let mut out = Vec::with_capacity(n);
        for p in &self.planes {
            out.extend_from_slice(&p.data);
        }
        out
    }
}
