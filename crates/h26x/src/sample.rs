//! The sample type a decoder works in: `u8` for 8-bit streams, `u16` for
//! 9–16-bit ones. Everything below the NAL layer of both decoders is generic
//! over it, so 8-bit pictures move half the bytes and the SIMD kernels work
//! on twice the lanes.

/// A picture sample: `u8` (8-bit streams) or `u16` (deeper).
pub trait Sample: Copy + Default + Send + Sync + PartialEq + Eq + std::fmt::Debug + 'static {
    /// Bytes per sample.
    const BYTES: usize;
    /// Widen.
    fn to_i32(self) -> i32;
    /// Narrow (the value must be in range).
    fn from_i32(v: i32) -> Self;
    /// Fill the SIMD entries of the HEVC kernel table for this sample type.
    fn install_simd(dsp: &mut crate::dsp::hevc::HevcDsp<Self>, cpu: crate::dsp::Cpu);
    /// Fill the SIMD entries of the H.264 kernel table for this sample type.
    fn install_h264_simd(dsp: &mut crate::dsp::h264::H264Dsp<Self>, cpu: crate::dsp::Cpu);
}

impl Sample for u8 {
    const BYTES: usize = 1;
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u8
    }
    fn install_simd(dsp: &mut crate::dsp::hevc::HevcDsp<Self>, cpu: crate::dsp::Cpu) {
        crate::dsp::hevc::install_simd_u8(dsp, cpu);
    }
    fn install_h264_simd(dsp: &mut crate::dsp::h264::H264Dsp<Self>, cpu: crate::dsp::Cpu) {
        crate::dsp::h264::install_simd_u8(dsp, cpu);
    }
}

impl Sample for u16 {
    const BYTES: usize = 2;
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u16
    }
    fn install_simd(dsp: &mut crate::dsp::hevc::HevcDsp<Self>, cpu: crate::dsp::Cpu) {
        crate::dsp::hevc::install_simd_u16(dsp, cpu);
    }
    fn install_h264_simd(_dsp: &mut crate::dsp::h264::H264Dsp<Self>, _cpu: crate::dsp::Cpu) {
        // No 16-bit H.264 SIMD kernels yet: the scalar table stands.
    }
}
