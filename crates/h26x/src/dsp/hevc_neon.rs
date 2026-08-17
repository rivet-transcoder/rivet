//! NEON versions of the H.265 kernels (AArch64).

#![cfg(target_arch = "aarch64")]

use super::hevc::HevcDsp;

/// Replace the scalar entries of `d` with the NEON kernels.
pub fn install(_d: &mut HevcDsp) {}
