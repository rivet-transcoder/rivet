//! Pixel kernels: interpolation filters, inverse transforms, deblocking and
//! SAO edges — the loops a decoder spends its time in.
//!
//! Every kernel exists as a scalar reference implementation, and the hot ones
//! also as SIMD (x86-64 AVX2, AArch64 NEON). Which runs is decided once per process
//! by [`Cpu::detect`] and threaded through as function pointers, the way
//! libavcodec's `*dsp_init` tables work, so the decoders never branch on the
//! CPU themselves and the scalar path stays the executable specification the
//! SIMD path is tested against.

pub mod h264;
#[cfg(target_arch = "x86_64")]
pub mod h264_avx2;
#[cfg(target_arch = "aarch64")]
pub mod h264_neon;
pub mod hevc;
#[cfg(target_arch = "x86_64")]
pub mod hevc_avx2;
#[cfg(target_arch = "aarch64")]
pub mod hevc_neon;

/// What the running CPU can do, detected once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cpu {
    /// x86-64 with AVX2 (implies SSE4.1 / SSSE3).
    pub avx2: bool,
    /// x86-64 with SSE4.1 (a superset of the SSSE3 kernels need).
    pub sse41: bool,
    /// AArch64 NEON (baseline on every AArch64 CPU).
    pub neon: bool,
}

impl Cpu {
    /// Detect the running CPU's SIMD extensions.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let avx2 = std::is_x86_feature_detected!("avx2");
            let sse41 = std::is_x86_feature_detected!("sse4.1");
            return Self { avx2, sse41, neon: false };
        }
        #[cfg(target_arch = "aarch64")]
        {
            return Self { avx2: false, sse41: false, neon: true };
        }
        #[allow(unreachable_code)]
        Self { avx2: false, sse41: false, neon: false }
    }

    /// Scalar only — the reference paths. What the SIMD kernels are checked
    /// against, and what a `H26X_NO_SIMD=1` environment asks for.
    pub const SCALAR: Self = Self { avx2: false, sse41: false, neon: false };

    /// [`Self::detect`], unless the environment turns SIMD off.
    pub fn detect_honouring_env() -> Self {
        if std::env::var_os("H26X_NO_SIMD").is_some_and(|v| v == "1" || v == "true") {
            Self::SCALAR
        } else {
            Self::detect()
        }
    }
}
