//! Runtime SIMD dispatch for the denoise kernels.
//!
//! Three tiers — scalar, 128-bit SSE4.1 and 256-bit AVX2 — chosen once per
//! process from what the CPU advertises ([`Tier::detect`]) and capped by the
//! `RIVET_DENOISE_MAX_SIMD` environment variable (`avx2`, `sse41`, `none`), so
//! every rung of the ladder can be exercised for bit-exactness on a machine
//! that would otherwise always take the top one, and so a same-binary timing
//! control is one environment variable away.
//!
//! Each kernel is written **once** as a generic body over the [`Simd`] trait —
//! a lane-width-agnostic view of the handful of vector operations the denoise
//! family needs — and instantiated per tier by [`tiered!`], which wraps the
//! body in a `#[target_feature]` function so the intrinsics inline into it.
//! The scalar tier is always the hand-written reference loop, never the
//! generic body with one lane: the reference is the spec, and the generic body
//! is what the tests hold against it.
//!
//! Every op here maps to exactly one IEEE-754 operation (or an integer one),
//! and the kernel bodies perform them in the same order as the reference
//! loops, which is what makes the vector paths bit-identical rather than
//! merely close. There is deliberately no fused multiply-add.

#![cfg_attr(not(any(target_arch = "x86", target_arch = "x86_64")), allow(dead_code))]

use std::sync::OnceLock;

/// The vector tier a kernel runs at. Ordered: a cap is a `min`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Tier {
    /// The hand-written reference loops.
    Scalar,
    /// 128-bit SSE4.1 (4 × f32 / 8 × u16 / 16 × u8 lanes).
    Sse41,
    /// 256-bit AVX2 (8 × f32 / 16 × u16 / 32 × u8 lanes).
    Avx2,
}

impl Tier {
    /// The tier the kernels run at in this process: what the host advertises,
    /// capped by `RIVET_DENOISE_MAX_SIMD`. Resolved once and cached, so the
    /// environment is read at most once per process.
    pub(crate) fn detect() -> Tier {
        static TIER: OnceLock<Tier> = OnceLock::new();
        *TIER.get_or_init(|| {
            let env = std::env::var("RIVET_DENOISE_MAX_SIMD").ok();
            let tier = Tier::cap(Tier::host(), env.as_deref());
            tracing::debug!(host = Tier::host().name(), tier = tier.name(), "denoise SIMD tier");
            tier
        })
    }

    /// What the CPU supports, ignoring the environment.
    pub(crate) fn host() -> Tier {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                return Tier::Avx2;
            }
            if std::is_x86_feature_detected!("sse4.1") {
                return Tier::Sse41;
            }
        }
        Tier::Scalar
    }

    /// Apply a `RIVET_DENOISE_MAX_SIMD` value to a detected tier. A cap only
    /// ever lowers; an unrecognised value is ignored (with a warning) rather
    /// than silently downgrading.
    pub(crate) fn cap(host: Tier, env: Option<&str>) -> Tier {
        match env.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") | Some("avx2") => host,
            Some("sse41") | Some("sse4.1") | Some("sse") => host.min(Tier::Sse41),
            Some("none") | Some("scalar") | Some("0") => Tier::Scalar,
            Some(other) => {
                tracing::warn!(
                    value = other,
                    "RIVET_DENOISE_MAX_SIMD not understood (want avx2|sse41|none); ignored"
                );
                host
            }
        }
    }

    /// The name the docs and logs use.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Tier::Scalar => "scalar",
            Tier::Sse41 => "sse4.1",
            Tier::Avx2 => "avx2",
        }
    }

    /// Every tier this host can run, lowest first — what the bit-exactness
    /// tests iterate over.
    #[cfg(test)]
    pub(crate) fn available() -> Vec<Tier> {
        let host = Tier::host();
        [Tier::Scalar, Tier::Sse41, Tier::Avx2].into_iter().filter(|t| *t <= host).collect()
    }
}

/// The vector operations the denoise kernels need, at one lane width.
///
/// Every method is `unsafe` because the intrinsics behind them are only sound
/// to execute on a CPU that has the feature — which the kernel bodies are only
/// reached on, through [`tiered!`] after [`Tier::detect`]. Pointer arguments
/// must be readable / writable for the vector's width.
pub(crate) trait Simd: Copy {
    /// `f32` / `i32` lanes per vector.
    const LANES: usize;
    /// `u16` lanes per vector (`2 × LANES`).
    const LANES16: usize;
    /// `u8` lanes per vector (`4 × LANES`).
    const LANES8: usize;

    /// A vector of `LANES` × `f32`.
    type F: Copy;
    /// A vector of `LANES` × `i32`.
    type I: Copy;
    /// A vector of `LANES16` × `u16`.
    type H: Copy;
    /// A vector of `LANES8` × `u8`.
    type B: Copy;

    // ── f32 ────────────────────────────────────────────────────────────────
    /// `LANES` bytes → f32 lanes (exact).
    unsafe fn load_u8_f32(p: *const u8) -> Self::F;
    unsafe fn load_f32(p: *const f32) -> Self::F;
    unsafe fn store_f32(p: *mut f32, v: Self::F);
    /// Store f32 lanes that are already integral and in `0..=255` as `LANES` bytes.
    unsafe fn store_f32_u8(p: *mut u8, v: Self::F);
    unsafe fn set1_f32(x: f32) -> Self::F;
    unsafe fn add_f32(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn sub_f32(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn mul_f32(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn div_f32(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn floor_f32(a: Self::F) -> Self::F;
    unsafe fn min_f32(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn max_f32(a: Self::F, b: Self::F) -> Self::F;
    /// Lane mask (all ones where `a >= b`).
    unsafe fn cmpge_f32(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn and_f32(a: Self::F, b: Self::F) -> Self::F;
    unsafe fn i32_to_f32(a: Self::I) -> Self::F;
    /// `table[idx]` per lane. Every index must be in bounds.
    unsafe fn gather_f32(table: &[f32], idx: Self::I) -> Self::F;

    // ── i32 ────────────────────────────────────────────────────────────────
    /// `LANES` bytes → i32 lanes.
    unsafe fn load_u8_i32(p: *const u8) -> Self::I;
    unsafe fn load_u32(p: *const u32) -> Self::I;
    unsafe fn store_u32(p: *mut u32, v: Self::I);
    unsafe fn set1_i32(x: i32) -> Self::I;
    unsafe fn add_i32(a: Self::I, b: Self::I) -> Self::I;
    unsafe fn sub_i32(a: Self::I, b: Self::I) -> Self::I;
    unsafe fn mullo_i32(a: Self::I, b: Self::I) -> Self::I;
    unsafe fn abs_i32(a: Self::I) -> Self::I;
    unsafe fn min_i32(a: Self::I, b: Self::I) -> Self::I;
    /// Inclusive prefix sum across the lanes (wrapping).
    unsafe fn prefix_sum_i32(a: Self::I) -> Self::I;
    /// The last lane.
    unsafe fn extract_last_i32(a: Self::I) -> i32;

    // ── u16 ────────────────────────────────────────────────────────────────
    /// `LANES16` bytes → u16 lanes.
    unsafe fn load_u8_u16(p: *const u8) -> Self::H;
    unsafe fn load_u16(p: *const u16) -> Self::H;
    unsafe fn store_u16(p: *mut u16, v: Self::H);
    /// Store u16 lanes that are `<= 255` as `LANES16` bytes.
    unsafe fn store_u16_u8(p: *mut u8, v: Self::H);
    unsafe fn set1_u16(x: u16) -> Self::H;
    unsafe fn add_u16(a: Self::H, b: Self::H) -> Self::H;
    /// `(a * b) >> 16` per unsigned lane.
    unsafe fn mulhi_u16(a: Self::H, b: Self::H) -> Self::H;

    // ── u8 ─────────────────────────────────────────────────────────────────
    unsafe fn load_u8(p: *const u8) -> Self::B;
    unsafe fn store_u8(p: *mut u8, v: Self::B);
    unsafe fn min_u8(a: Self::B, b: Self::B) -> Self::B;
    unsafe fn max_u8(a: Self::B, b: Self::B) -> Self::B;
}

/// `x.round().clamp(0.0, 255.0)` for non-negative `x`, lane-wise, bit-exact
/// with the scalar expression: `x − floor(x)` is exact below 2²³, so testing
/// that fraction against ½ is exactly round-half-away-from-zero for `x ≥ 0`.
#[inline(always)]
pub(crate) unsafe fn round_clamp_u8<S: Simd>(x: S::F) -> S::F {
    unsafe {
        let r = S::floor_f32(x);
        let frac = S::sub_f32(x, r);
        let up = S::and_f32(S::cmpge_f32(frac, S::set1_f32(0.5)), S::set1_f32(1.0));
        let r = S::add_f32(r, up);
        S::min_f32(S::max_f32(r, S::set1_f32(0.0)), S::set1_f32(255.0))
    }
}

/// Define `fn $name(tier: Tier, args..)` that runs `$body::<Avx2>` /
/// `$body::<Sse41>` inside a `#[target_feature]` wrapper for the vector tiers
/// and `$scalar(args..)` for the scalar one. The body must be an
/// `#[inline(always)] unsafe fn $body<S: Simd>(args..)` with the same
/// parameter list.
macro_rules! tiered {
    (fn $name:ident($($arg:ident : $ty:ty),* $(,)?) => $body:ident, scalar $scalar:path) => {
        fn $name(tier: $crate::filter::denoise::simd::Tier, $($arg: $ty),*) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                #[target_feature(enable = "avx2")]
                unsafe fn avx2($($arg: $ty),*) {
                    unsafe { $body::<$crate::filter::denoise::simd::Avx2>($($arg),*) }
                }
                #[target_feature(enable = "sse4.1")]
                unsafe fn sse41($($arg: $ty),*) {
                    unsafe { $body::<$crate::filter::denoise::simd::Sse41>($($arg),*) }
                }
                // SAFETY: `tier` came from `Tier::detect` (or a test's
                // `Tier::available`), so the host has the feature; a cap only
                // ever lowers it.
                match tier {
                    $crate::filter::denoise::simd::Tier::Avx2 => return unsafe { avx2($($arg),*) },
                    $crate::filter::denoise::simd::Tier::Sse41 => return unsafe { sse41($($arg),*) },
                    $crate::filter::denoise::simd::Tier::Scalar => {}
                }
            }
            let _ = tier;
            $scalar($($arg),*)
        }
    };
}
pub(crate) use tiered;

// ── AVX2 ────────────────────────────────────────────────────────────────────

/// 256-bit lanes.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
pub(crate) struct Avx2;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2_impl {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use super::{Avx2, Simd};

    impl Simd for Avx2 {
        const LANES: usize = 8;
        const LANES16: usize = 16;
        const LANES8: usize = 32;
        type F = __m256;
        type I = __m256i;
        type H = __m256i;
        type B = __m256i;

        #[inline(always)]
        unsafe fn load_u8_f32(p: *const u8) -> __m256 {
            unsafe { _mm256_cvtepi32_ps(Self::load_u8_i32(p)) }
        }
        #[inline(always)]
        unsafe fn load_f32(p: *const f32) -> __m256 {
            unsafe { _mm256_loadu_ps(p) }
        }
        #[inline(always)]
        unsafe fn store_f32(p: *mut f32, v: __m256) {
            unsafe { _mm256_storeu_ps(p, v) }
        }
        #[inline(always)]
        unsafe fn store_f32_u8(p: *mut u8, v: __m256) {
            unsafe {
                // Integral lanes, so the conversion is exact whatever the
                // rounding mode. Pack 32 → 16 → 8 within each 128-bit half,
                // then the first four bytes of each half are the lanes.
                let i = _mm256_cvtps_epi32(v);
                let h = _mm256_packus_epi32(i, i);
                let b = _mm256_packus_epi16(h, h);
                let lo = _mm_cvtsi128_si32(_mm256_castsi256_si128(b));
                let hi = _mm_cvtsi128_si32(_mm256_extracti128_si256::<1>(b));
                (p as *mut i32).write_unaligned(lo);
                (p.add(4) as *mut i32).write_unaligned(hi);
            }
        }
        #[inline(always)]
        unsafe fn set1_f32(x: f32) -> __m256 {
            unsafe { _mm256_set1_ps(x) }
        }
        #[inline(always)]
        unsafe fn add_f32(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_add_ps(a, b) }
        }
        #[inline(always)]
        unsafe fn sub_f32(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_sub_ps(a, b) }
        }
        #[inline(always)]
        unsafe fn mul_f32(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_mul_ps(a, b) }
        }
        #[inline(always)]
        unsafe fn div_f32(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_div_ps(a, b) }
        }
        #[inline(always)]
        unsafe fn floor_f32(a: __m256) -> __m256 {
            unsafe { _mm256_floor_ps(a) }
        }
        #[inline(always)]
        unsafe fn min_f32(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_min_ps(a, b) }
        }
        #[inline(always)]
        unsafe fn max_f32(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_max_ps(a, b) }
        }
        #[inline(always)]
        unsafe fn cmpge_f32(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_cmp_ps::<_CMP_GE_OQ>(a, b) }
        }
        #[inline(always)]
        unsafe fn and_f32(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_and_ps(a, b) }
        }
        #[inline(always)]
        unsafe fn i32_to_f32(a: __m256i) -> __m256 {
            unsafe { _mm256_cvtepi32_ps(a) }
        }
        #[inline(always)]
        unsafe fn gather_f32(table: &[f32], idx: __m256i) -> __m256 {
            unsafe { _mm256_i32gather_ps::<4>(table.as_ptr(), idx) }
        }

        #[inline(always)]
        unsafe fn load_u8_i32(p: *const u8) -> __m256i {
            unsafe { _mm256_cvtepu8_epi32(_mm_loadl_epi64(p as *const __m128i)) }
        }
        #[inline(always)]
        unsafe fn load_u32(p: *const u32) -> __m256i {
            unsafe { _mm256_loadu_si256(p as *const __m256i) }
        }
        #[inline(always)]
        unsafe fn store_u32(p: *mut u32, v: __m256i) {
            unsafe { _mm256_storeu_si256(p as *mut __m256i, v) }
        }
        #[inline(always)]
        unsafe fn set1_i32(x: i32) -> __m256i {
            unsafe { _mm256_set1_epi32(x) }
        }
        #[inline(always)]
        unsafe fn add_i32(a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_add_epi32(a, b) }
        }
        #[inline(always)]
        unsafe fn sub_i32(a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_sub_epi32(a, b) }
        }
        #[inline(always)]
        unsafe fn mullo_i32(a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_mullo_epi32(a, b) }
        }
        #[inline(always)]
        unsafe fn abs_i32(a: __m256i) -> __m256i {
            unsafe { _mm256_abs_epi32(a) }
        }
        #[inline(always)]
        unsafe fn min_i32(a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_min_epi32(a, b) }
        }
        #[inline(always)]
        unsafe fn prefix_sum_i32(s: __m256i) -> __m256i {
            unsafe {
                // `slli_si256` shifts within each 128-bit half, so two
                // shift-adds give a prefix per half; then broadcast the low
                // half's total (its lane 3) into the high half and add.
                let s = _mm256_add_epi32(s, _mm256_slli_si256::<4>(s));
                let s = _mm256_add_epi32(s, _mm256_slli_si256::<8>(s));
                let low_total =
                    _mm256_shuffle_epi32::<0xFF>(_mm256_permute2x128_si256::<0x08>(s, s));
                _mm256_add_epi32(s, low_total)
            }
        }
        #[inline(always)]
        unsafe fn extract_last_i32(a: __m256i) -> i32 {
            unsafe { _mm256_extract_epi32::<7>(a) }
        }

        #[inline(always)]
        unsafe fn load_u8_u16(p: *const u8) -> __m256i {
            unsafe { _mm256_cvtepu8_epi16(_mm_loadu_si128(p as *const __m128i)) }
        }
        #[inline(always)]
        unsafe fn load_u16(p: *const u16) -> __m256i {
            unsafe { _mm256_loadu_si256(p as *const __m256i) }
        }
        #[inline(always)]
        unsafe fn store_u16(p: *mut u16, v: __m256i) {
            unsafe { _mm256_storeu_si256(p as *mut __m256i, v) }
        }
        #[inline(always)]
        unsafe fn store_u16_u8(p: *mut u8, v: __m256i) {
            unsafe {
                // Pack within each 128-bit half; the first eight bytes of each
                // half are that half's lanes.
                let b = _mm256_packus_epi16(v, v);
                _mm_storel_epi64(p as *mut __m128i, _mm256_castsi256_si128(b));
                _mm_storel_epi64(p.add(8) as *mut __m128i, _mm256_extracti128_si256::<1>(b));
            }
        }
        #[inline(always)]
        unsafe fn set1_u16(x: u16) -> __m256i {
            unsafe { _mm256_set1_epi16(x as i16) }
        }
        #[inline(always)]
        unsafe fn add_u16(a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_add_epi16(a, b) }
        }
        #[inline(always)]
        unsafe fn mulhi_u16(a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_mulhi_epu16(a, b) }
        }

        #[inline(always)]
        unsafe fn load_u8(p: *const u8) -> __m256i {
            unsafe { _mm256_loadu_si256(p as *const __m256i) }
        }
        #[inline(always)]
        unsafe fn store_u8(p: *mut u8, v: __m256i) {
            unsafe { _mm256_storeu_si256(p as *mut __m256i, v) }
        }
        #[inline(always)]
        unsafe fn min_u8(a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_min_epu8(a, b) }
        }
        #[inline(always)]
        unsafe fn max_u8(a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_max_epu8(a, b) }
        }
    }
}

// ── SSE4.1 ──────────────────────────────────────────────────────────────────

/// 128-bit lanes.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
pub(crate) struct Sse41;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse41_impl {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    use super::{Simd, Sse41};

    impl Simd for Sse41 {
        const LANES: usize = 4;
        const LANES16: usize = 8;
        const LANES8: usize = 16;
        type F = __m128;
        type I = __m128i;
        type H = __m128i;
        type B = __m128i;

        #[inline(always)]
        unsafe fn load_u8_f32(p: *const u8) -> __m128 {
            unsafe { _mm_cvtepi32_ps(Self::load_u8_i32(p)) }
        }
        #[inline(always)]
        unsafe fn load_f32(p: *const f32) -> __m128 {
            unsafe { _mm_loadu_ps(p) }
        }
        #[inline(always)]
        unsafe fn store_f32(p: *mut f32, v: __m128) {
            unsafe { _mm_storeu_ps(p, v) }
        }
        #[inline(always)]
        unsafe fn store_f32_u8(p: *mut u8, v: __m128) {
            unsafe {
                let i = _mm_cvtps_epi32(v);
                let h = _mm_packus_epi32(i, i);
                let b = _mm_packus_epi16(h, h);
                (p as *mut i32).write_unaligned(_mm_cvtsi128_si32(b));
            }
        }
        #[inline(always)]
        unsafe fn set1_f32(x: f32) -> __m128 {
            unsafe { _mm_set1_ps(x) }
        }
        #[inline(always)]
        unsafe fn add_f32(a: __m128, b: __m128) -> __m128 {
            unsafe { _mm_add_ps(a, b) }
        }
        #[inline(always)]
        unsafe fn sub_f32(a: __m128, b: __m128) -> __m128 {
            unsafe { _mm_sub_ps(a, b) }
        }
        #[inline(always)]
        unsafe fn mul_f32(a: __m128, b: __m128) -> __m128 {
            unsafe { _mm_mul_ps(a, b) }
        }
        #[inline(always)]
        unsafe fn div_f32(a: __m128, b: __m128) -> __m128 {
            unsafe { _mm_div_ps(a, b) }
        }
        #[inline(always)]
        unsafe fn floor_f32(a: __m128) -> __m128 {
            unsafe { _mm_floor_ps(a) }
        }
        #[inline(always)]
        unsafe fn min_f32(a: __m128, b: __m128) -> __m128 {
            unsafe { _mm_min_ps(a, b) }
        }
        #[inline(always)]
        unsafe fn max_f32(a: __m128, b: __m128) -> __m128 {
            unsafe { _mm_max_ps(a, b) }
        }
        #[inline(always)]
        unsafe fn cmpge_f32(a: __m128, b: __m128) -> __m128 {
            unsafe { _mm_cmpge_ps(a, b) }
        }
        #[inline(always)]
        unsafe fn and_f32(a: __m128, b: __m128) -> __m128 {
            unsafe { _mm_and_ps(a, b) }
        }
        #[inline(always)]
        unsafe fn i32_to_f32(a: __m128i) -> __m128 {
            unsafe { _mm_cvtepi32_ps(a) }
        }
        #[inline(always)]
        unsafe fn gather_f32(table: &[f32], idx: __m128i) -> __m128 {
            // No gather below AVX2: four scalar loads.
            unsafe {
                let i0 = _mm_extract_epi32::<0>(idx) as usize;
                let i1 = _mm_extract_epi32::<1>(idx) as usize;
                let i2 = _mm_extract_epi32::<2>(idx) as usize;
                let i3 = _mm_extract_epi32::<3>(idx) as usize;
                _mm_set_ps(
                    *table.get_unchecked(i3),
                    *table.get_unchecked(i2),
                    *table.get_unchecked(i1),
                    *table.get_unchecked(i0),
                )
            }
        }

        #[inline(always)]
        unsafe fn load_u8_i32(p: *const u8) -> __m128i {
            unsafe { _mm_cvtepu8_epi32(_mm_cvtsi32_si128((p as *const i32).read_unaligned())) }
        }
        #[inline(always)]
        unsafe fn load_u32(p: *const u32) -> __m128i {
            unsafe { _mm_loadu_si128(p as *const __m128i) }
        }
        #[inline(always)]
        unsafe fn store_u32(p: *mut u32, v: __m128i) {
            unsafe { _mm_storeu_si128(p as *mut __m128i, v) }
        }
        #[inline(always)]
        unsafe fn set1_i32(x: i32) -> __m128i {
            unsafe { _mm_set1_epi32(x) }
        }
        #[inline(always)]
        unsafe fn add_i32(a: __m128i, b: __m128i) -> __m128i {
            unsafe { _mm_add_epi32(a, b) }
        }
        #[inline(always)]
        unsafe fn sub_i32(a: __m128i, b: __m128i) -> __m128i {
            unsafe { _mm_sub_epi32(a, b) }
        }
        #[inline(always)]
        unsafe fn mullo_i32(a: __m128i, b: __m128i) -> __m128i {
            unsafe { _mm_mullo_epi32(a, b) }
        }
        #[inline(always)]
        unsafe fn abs_i32(a: __m128i) -> __m128i {
            unsafe { _mm_abs_epi32(a) }
        }
        #[inline(always)]
        unsafe fn min_i32(a: __m128i, b: __m128i) -> __m128i {
            unsafe { _mm_min_epi32(a, b) }
        }
        #[inline(always)]
        unsafe fn prefix_sum_i32(s: __m128i) -> __m128i {
            unsafe {
                let s = _mm_add_epi32(s, _mm_slli_si128::<4>(s));
                _mm_add_epi32(s, _mm_slli_si128::<8>(s))
            }
        }
        #[inline(always)]
        unsafe fn extract_last_i32(a: __m128i) -> i32 {
            unsafe { _mm_extract_epi32::<3>(a) }
        }

        #[inline(always)]
        unsafe fn load_u8_u16(p: *const u8) -> __m128i {
            unsafe { _mm_cvtepu8_epi16(_mm_loadl_epi64(p as *const __m128i)) }
        }
        #[inline(always)]
        unsafe fn load_u16(p: *const u16) -> __m128i {
            unsafe { _mm_loadu_si128(p as *const __m128i) }
        }
        #[inline(always)]
        unsafe fn store_u16(p: *mut u16, v: __m128i) {
            unsafe { _mm_storeu_si128(p as *mut __m128i, v) }
        }
        #[inline(always)]
        unsafe fn store_u16_u8(p: *mut u8, v: __m128i) {
            unsafe { _mm_storel_epi64(p as *mut __m128i, _mm_packus_epi16(v, v)) }
        }
        #[inline(always)]
        unsafe fn set1_u16(x: u16) -> __m128i {
            unsafe { _mm_set1_epi16(x as i16) }
        }
        #[inline(always)]
        unsafe fn add_u16(a: __m128i, b: __m128i) -> __m128i {
            unsafe { _mm_add_epi16(a, b) }
        }
        #[inline(always)]
        unsafe fn mulhi_u16(a: __m128i, b: __m128i) -> __m128i {
            unsafe { _mm_mulhi_epu16(a, b) }
        }

        #[inline(always)]
        unsafe fn load_u8(p: *const u8) -> __m128i {
            unsafe { _mm_loadu_si128(p as *const __m128i) }
        }
        #[inline(always)]
        unsafe fn store_u8(p: *mut u8, v: __m128i) {
            unsafe { _mm_storeu_si128(p as *mut __m128i, v) }
        }
        #[inline(always)]
        unsafe fn min_u8(a: __m128i, b: __m128i) -> __m128i {
            unsafe { _mm_min_epu8(a, b) }
        }
        #[inline(always)]
        unsafe fn max_u8(a: __m128i, b: __m128i) -> __m128i {
            unsafe { _mm_max_epu8(a, b) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_env_cap_only_ever_lowers() {
        assert_eq!(Tier::cap(Tier::Avx2, None), Tier::Avx2);
        assert_eq!(Tier::cap(Tier::Avx2, Some("avx2")), Tier::Avx2);
        assert_eq!(Tier::cap(Tier::Avx2, Some("sse41")), Tier::Sse41);
        assert_eq!(Tier::cap(Tier::Avx2, Some("SSE4.1")), Tier::Sse41);
        assert_eq!(Tier::cap(Tier::Avx2, Some("none")), Tier::Scalar);
        assert_eq!(Tier::cap(Tier::Avx2, Some("scalar")), Tier::Scalar);
        // A cap above the host is not a promotion.
        assert_eq!(Tier::cap(Tier::Sse41, Some("avx2")), Tier::Sse41);
        assert_eq!(Tier::cap(Tier::Scalar, Some("sse41")), Tier::Scalar);
        // Nonsense is ignored, not a downgrade.
        assert_eq!(Tier::cap(Tier::Avx2, Some("neon")), Tier::Avx2);
    }

    #[test]
    fn available_tiers_start_at_scalar_and_end_at_the_host() {
        let tiers = Tier::available();
        assert_eq!(tiers[0], Tier::Scalar);
        assert_eq!(*tiers.last().unwrap(), Tier::host());
        assert!(tiers.windows(2).all(|w| w[0] < w[1]));
    }

    /// The lane-wise rounding must agree with `f32::round` — it is the last
    /// step of every float kernel, and a `floor(x + 0.5)` would not (it
    /// rounds `0.49999997` up).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn round_clamp_matches_f32_round() {
        #[inline(always)]
        unsafe fn body<S: Simd>(input: &[f32], out: &mut [u8]) {
            unsafe {
                let mut x = 0;
                while x + S::LANES <= input.len() {
                    let v = S::load_f32(input.as_ptr().add(x));
                    S::store_f32_u8(out.as_mut_ptr().add(x), round_clamp_u8::<S>(v));
                    x += S::LANES;
                }
            }
        }
        fn scalar(input: &[f32], out: &mut [u8]) {
            for (o, &v) in out.iter_mut().zip(input) {
                *o = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        tiered!(fn run(input: &[f32], out: &mut [u8]) => body, scalar scalar);

        // Every quarter step, the awkward neighbours of every half, and the
        // clamp ends.
        let mut input: Vec<f32> = (0..=1024).map(|i| i as f32 * 0.25).collect();
        for i in 0..256 {
            let half = i as f32 + 0.5;
            input.push(half);
            input.push(f32::from_bits(half.to_bits() - 1));
            input.push(f32::from_bits(half.to_bits() + 1));
        }
        input.extend_from_slice(&[0.49999997, 254.99998, 255.4, 255.5, 300.0, 0.0]);
        while input.len() % 8 != 0 {
            input.push(1.5);
        }
        let mut want = vec![0u8; input.len()];
        scalar(&input, &mut want);
        for tier in Tier::available() {
            let mut got = vec![0u8; input.len()];
            run(tier, &input, &mut got);
            assert_eq!(got, want, "{tier:?} rounding diverged");
        }
    }
}
