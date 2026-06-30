// mdat writing is performed inline in `Av1Mp4Muxer::finalize_to_file` (mod.rs).
// The mdat box header bytes are constructed there directly, including the
// largesize (64-bit) upgrade when the payload exceeds u32::MAX − 8.
// No separate helper functions are needed; this module is kept as a
// placeholder to satisfy the directory split.
