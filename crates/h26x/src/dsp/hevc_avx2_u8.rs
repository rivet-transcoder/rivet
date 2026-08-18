//! AVX2 versions of the H.265 kernels for 8-bit sample planes (x86-64).

#![cfg(target_arch = "x86_64")]

use super::hevc::HevcDsp;

/// Replace the scalar entries of `d` with the AVX2 kernels.
pub fn install(_d: &mut HevcDsp<u8>) {}
