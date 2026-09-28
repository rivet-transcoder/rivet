//! The LAME entry points the MP3 encoder uses, loaded at run time.
//!
//! LAME is LGPL, and rivet does not link or ship it: the library is found
//! with `dlopen` (`libmp3lame.so.0` on Linux, `libmp3lame.0.dylib` on macOS,
//! `libmp3lame.dll` on Windows, or the path in `RIVET_LAME_LIBRARY`), the
//! same boundary the GPU runtimes sit behind. A host without it fails the
//! first MP3 encode with an error saying which library to install, not at
//! start-up. The signatures mirror `lame.h` (LAME 3.100; every call here has
//! been in the API since 3.99, which added the IEEE-float input).

use std::ffi::{CStr, c_char, c_int, c_uchar, c_void};
use std::sync::OnceLock;

use crate::audio::AudioError;

/// `lame_global_flags *`.
pub(super) type Flags = *mut c_void;

/// `MPEG_mode` from `lame.h`.
pub(super) const JOINT_STEREO: c_int = 1;
pub(super) const MONO: c_int = 3;
/// `vbr_mode::vbr_off`: constant bitrate.
pub(super) const VBR_OFF: c_int = 0;

/// The library names tried, in order, when `RIVET_LAME_LIBRARY` is unset.
const NAMES: &[&str] = &[
    "libmp3lame.so.0",
    "libmp3lame.so",
    "libmp3lame.0.dylib",
    "libmp3lame.dylib",
    "libmp3lame.dll",
    "libmp3lame-0.dll",
];

type SetInt = unsafe extern "C" fn(Flags, c_int) -> c_int;
type GetInt = unsafe extern "C" fn(Flags) -> c_int;

pub(super) struct Lame {
    pub(super) init: unsafe extern "C" fn() -> Flags,
    pub(super) close: unsafe extern "C" fn(Flags) -> c_int,
    pub(super) init_params: GetInt,
    pub(super) set_in_samplerate: SetInt,
    pub(super) set_out_samplerate: SetInt,
    pub(super) set_num_channels: SetInt,
    pub(super) set_brate: SetInt,
    pub(super) set_mode: SetInt,
    pub(super) set_vbr: SetInt,
    pub(super) set_quality: SetInt,
    pub(super) set_write_vbr_tag: SetInt,
    pub(super) get_encoder_delay: GetInt,
    pub(super) get_framesize: GetInt,
    /// `lame_encode_buffer_interleaved_ieee_float`: stereo, `[-1, 1]` floats.
    pub(super) encode_interleaved: unsafe extern "C" fn(Flags, *const f32, c_int, *mut c_uchar, c_int) -> c_int,
    /// `lame_encode_buffer_ieee_float`: planar; mono passes one plane twice.
    pub(super) encode_planar:
        unsafe extern "C" fn(Flags, *const f32, *const f32, c_int, *mut c_uchar, c_int) -> c_int,
    pub(super) flush: unsafe extern "C" fn(Flags, *mut c_uchar, c_int) -> c_int,
    pub(super) version: unsafe extern "C" fn() -> *const c_char,
    /// Declared last: every pointer above points into it.
    _lib: libloading::Library,
}

// SAFETY: the table holds plain function pointers into a library that stays
// loaded for the life of the process; LAME's calls are reentrant across
// distinct `lame_global_flags`.
unsafe impl Send for Lame {}
unsafe impl Sync for Lame {}

impl Lame {
    /// The process-wide library, loaded on first use.
    pub(super) fn get() -> Result<&'static Lame, AudioError> {
        static LAME: OnceLock<Result<Lame, String>> = OnceLock::new();
        LAME.get_or_init(Self::load).as_ref().map_err(|e| AudioError::Unsupported(e.clone()))
    }

    /// `get_lame_version()`, e.g. `3.100`.
    pub(super) fn version_string(&self) -> String {
        // SAFETY: LAME returns a pointer to a static NUL-terminated string.
        unsafe { CStr::from_ptr((self.version)()) }.to_string_lossy().into_owned()
    }

    fn load() -> Result<Lame, String> {
        let explicit = std::env::var_os("RIVET_LAME_LIBRARY");
        let lib = match &explicit {
            // SAFETY: loading LAME runs no initialisers with preconditions.
            Some(path) => unsafe { libloading::Library::new(path) }
                .map_err(|e| format!("RIVET_LAME_LIBRARY={}: {e}", path.to_string_lossy()))?,
            None => NAMES
                .iter()
                .find_map(|n| unsafe { libloading::Library::new(n) }.ok())
                .ok_or_else(|| {
                    "MP3 encoding needs the LAME library at run time and none was found \
                     (tried libmp3lame.so.0 / libmp3lame.dylib / libmp3lame.dll): install it \
                     (Debian/Ubuntu `libmp3lame0`, Fedora `lame-libs`, macOS `brew install lame`) \
                     or point RIVET_LAME_LIBRARY at the library file"
                        .to_string()
                })?,
        };
        macro_rules! sym {
            ($name:literal) => {
                // SAFETY: the symbol's type is the one `lame.h` declares.
                *unsafe { lib.get($name) }.map_err(|e| {
                    format!(
                        "the LAME library lacks {} ({e}); LAME 3.99 or newer is needed",
                        String::from_utf8_lossy(&$name[..$name.len() - 1])
                    )
                })?
            };
        }
        Ok(Lame {
            init: sym!(b"lame_init\0"),
            close: sym!(b"lame_close\0"),
            init_params: sym!(b"lame_init_params\0"),
            set_in_samplerate: sym!(b"lame_set_in_samplerate\0"),
            set_out_samplerate: sym!(b"lame_set_out_samplerate\0"),
            set_num_channels: sym!(b"lame_set_num_channels\0"),
            set_brate: sym!(b"lame_set_brate\0"),
            set_mode: sym!(b"lame_set_mode\0"),
            set_vbr: sym!(b"lame_set_VBR\0"),
            set_quality: sym!(b"lame_set_quality\0"),
            set_write_vbr_tag: sym!(b"lame_set_bWriteVbrTag\0"),
            get_encoder_delay: sym!(b"lame_get_encoder_delay\0"),
            get_framesize: sym!(b"lame_get_framesize\0"),
            encode_interleaved: sym!(b"lame_encode_buffer_interleaved_ieee_float\0"),
            encode_planar: sym!(b"lame_encode_buffer_ieee_float\0"),
            flush: sym!(b"lame_encode_flush\0"),
            version: sym!(b"get_lame_version\0"),
            _lib: lib,
        })
    }
}
