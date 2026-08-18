//! NEON versions of the H.264 kernels (AArch64).

#![cfg(target_arch = "aarch64")]

use super::h264::H264Dsp;

/// Replace the scalar entries of `d` with the NEON kernels.
pub fn install(_d: &mut H264Dsp) {}
