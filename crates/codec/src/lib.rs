/// D3D11-device-on-a-specific-AMD-adapter helper for AMF (Windows-only): lets
/// AMF's `InitDX11` bind to the AMD GPU instead of DXGI adapter 0.
#[cfg(all(windows, feature = "amd"))]
pub mod amf_device;
pub mod audio;
pub mod codec_strings;
pub mod colorspace;
// CUDA init serialization — used only by the hand-rolled NVENC/NVDEC FFI.
#[cfg(feature = "nvidia")]
pub(crate) mod cuda_lock;
pub mod decode;
pub mod encode;
pub mod filter;
pub mod frame;
pub mod gpu;
pub mod hevc_sei;
pub mod pixel_format;
pub mod probe;
#[cfg(feature = "qsv")]
pub(crate) mod qsv_ffi;
pub mod tonemap;

pub use frame::{ColorSpace, PixelFormat, VideoFrame};
pub use gpu::{GpuDevice, GpuVendor};
