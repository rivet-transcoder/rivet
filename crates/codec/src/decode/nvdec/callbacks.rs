//! CUVID parser callbacks: sequence / decode / display and the AV1
//! operating-point hook.  All wrapped in `catch_unwind` to prevent
//! Rust panics from unwinding across the FFI boundary into the driver.

use std::ffi::c_void;
use std::os::raw::{c_int, c_uint, c_ulong};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use crate::frame::ColorSpace;
use super::NvdecError;
use super::convert::{output_geometry, validate_format};
use super::ffi::{
    CU_MEMORYTYPE_DEVICE, CU_MEMORYTYPE_HOST,
    CUVID_CHROMA_420, CUVID_CREATE_PREFER_CUVID, CUVID_FMT_NV12, CUVID_FMT_P016, CUVID_H264,
    CUdeviceptr, CUvideodecoder,
    CudaMemcpy2D, CuVideoDecodeCaps, CuVideoDecodeCreateInfo, CuVideoDispInfo, CuVideoFormat,
    CuVideoPicParams, CuVideoProcParams,
};
use super::state::{CallbackState, DecodedFrame};

// ─── Callbacks ─────────────────────────────────────────────────────
//
// Each callback body is wrapped in std::panic::catch_unwind. Unwinding
// across an `extern "C"` boundary is UB per the Rustonomicon. If a
// Rust panic escapes into cuvidParseVideoData we get memory corruption
// at best. catch_unwind gives us a defined path: convert to error and
// return 0 (which tells the parser to abort cleanly).

pub unsafe extern "C" fn sequence_callback(
    user_data: *mut c_void,
    format: *mut CuVideoFormat,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if user_data.is_null() || format.is_null() {
                return 0;
            }
            let state = &mut *(user_data as *mut CallbackState);
            let fmt = &*format;

            // Task #39 hardening: verify the driver-reported codec matches
            // what we told the parser to expect. A mismatch here means the
            // CUVIDEOFORMAT struct layout drifted (bytes mean different
            // things than Rust thinks) OR we set up the parser for the
            // wrong codec OR the driver is quietly reinterpreting the
            // stream. Any of those is a catastrophic misconfiguration we
            // want to abort on immediately rather than proceed into
            // undefined decode behaviour. Using tracing::warn! + typed
            // error rather than assert! so the failure is diagnosable in
            // prod log aggregators without crashing the worker process.
            if fmt.codec != state.codec_type {
                tracing::warn!(
                    expected = state.codec_type,
                    got = fmt.codec,
                    "NVDEC sequence_callback codec mismatch — ABI drift suspected"
                );
                state.set_error(format!(
                    "sequence_callback codec mismatch: expected {} got {}",
                    state.codec_type, fmt.codec
                ));
                return 0;
            }

            // Honor the parser's declared minimum; pad up for pipelining but
            // cap at 32 so small-VRAM GPUs (e.g. Jetson, T4, A10) don't OOM
            // when the decoder reserves 4K×32 surfaces per stream.
            let num_surfaces = (fmt.min_num_decode_surfaces as c_uint).clamp(20, 32) as c_ulong;

            // The picture is the stream's display area (H.264 frame
            // cropping / HEVC conformance window / AV1 and VP9 frame size),
            // which the driver reports beside the padded coded surface. The
            // decoder below is created with that rectangle as its
            // `display_area` and a target of the same size, so the surface
            // it maps out is the cropped picture and the frame carries the
            // display dimensions — never the coded ones (a 640x360 stream is
            // coded 640x368; the pump resampled those 368 rows into 360 on
            // every NVDEC job, 20 dB down against the source).
            let geo = output_geometry(
                fmt.coded_width,
                fmt.coded_height,
                fmt.display_area_left,
                fmt.display_area_top,
                fmt.display_area_right,
                fmt.display_area_bottom,
            );

            // INFO level because this is a backend-engaged signal —
            // operators want to see it in prod logs to confirm NVDEC is
            // actually taking H.264/HEVC/VP9/AV1 traffic rather than
            // silently falling back to CPU. Fires once per sequence
            // (on first IDR and on mid-stream resolution changes).
            tracing::info!(
                codec = fmt.codec,
                width = geo.width,
                height = geo.height,
                coded_width = fmt.coded_width,
                coded_height = fmt.coded_height,
                display_area = ?(
                    fmt.display_area_left,
                    fmt.display_area_top,
                    fmt.display_area_right,
                    fmt.display_area_bottom
                ),
                chroma = fmt.chroma_format,
                bit_depth = fmt.bit_depth_luma_minus8 + 8,
                surfaces = num_surfaces,
                "NVDEC backend engaged"
            );
            if geo.coded_fallback {
                tracing::warn!(
                    codec = fmt.codec,
                    coded_width = fmt.coded_width,
                    coded_height = fmt.coded_height,
                    display_area = ?(
                        fmt.display_area_left,
                        fmt.display_area_top,
                        fmt.display_area_right,
                        fmt.display_area_bottom
                    ),
                    "NVDEC reported no usable display area; the picture is the padded coded surface"
                );
            }

            // Reject non-4:2:0 sources up-front. The NV12 buffer sizing and
            // the chroma deinterleave loop in decode_next both assume 4:2:0
            // subsampling. If the driver reports something else, fail
            // cleanly rather than corrupt the output.
            //
            // Chroma format values (SDK cudaVideoChromaFormat):
            //   0 = Monochrome, 1 = 4:2:0, 2 = 4:2:2, 3 = 4:4:4
            // HEVC Range Extensions profiles produce 4:2:2 or 4:4:4 in
            // the wild (perf-analyst flagged a test HEVC sample as RExt);
            // those land here and bubble up as a WARN in decode/mod.rs.
            // Route both chroma + bit-depth reject through the same
            // pure-Rust validator the unit tests exercise, so the
            // reject matrix can't drift between the callback and the
            // standalone validate_format() public API
            // (codec-review-2 HIGH-1 + HIGH-2).
            if let Some(err) = validate_format(
                fmt.chroma_format,
                fmt.bit_depth_luma_minus8,
                fmt.coded_width,
                fmt.coded_height,
            ) {
                // Actionable warn carries the codec, the resolution,
                // and the structured reject reason so operators
                // running multiple GPU hosts can attribute the
                // reject to a specific worker. The log fields mirror
                // the NvdecError variant contents.
                match &err {
                    NvdecError::UnsupportedChroma { label, .. } => {
                        tracing::warn!(
                            codec = state.codec_type,
                            w = fmt.coded_width,
                            h = fmt.coded_height,
                            chroma = fmt.chroma_format,
                            chroma_label = *label,
                            "NVDEC rejecting: chroma {} unsupported",
                            label
                        );
                    }
                    NvdecError::UnsupportedPixelFormat { bit_depth } => {
                        tracing::warn!(
                            codec = state.codec_type,
                            w = fmt.coded_width,
                            h = fmt.coded_height,
                            bit_depth = bit_depth,
                            "NVDEC rejecting: {}-bit content unsupported",
                            bit_depth
                        );
                    }
                    // validate_format never returns this (it's set by the
                    // cuvidGetDecoderCaps pre-flight below), but match exhaustively.
                    NvdecError::UnsupportedByHardware { reason } => {
                        tracing::warn!(codec = state.codec_type, "NVDEC rejecting: {reason}");
                    }
                }
                state.set_typed_error(err);
                return 0;
            }

            // cuvidGetDecoderCaps pre-flight — ask the GPU whether its NVDEC can
            // actually decode this (codec, chroma, bit-depth) + frame size before
            // cuvidCreateDecoder, so an unsupported combo or oversized frame is a
            // clean typed error instead of a cryptic driver failure. Conservative:
            // only reject on an explicit "not supported" / over-max; a query that
            // itself errors falls through to the create path (don't block on it).
            {
                let mut caps: CuVideoDecodeCaps = std::mem::zeroed();
                caps.codec_type = state.codec_type;
                caps.chroma_format = fmt.chroma_format;
                caps.bit_depth_minus8 = fmt.bit_depth_luma_minus8 as u32;
                if (state.cuvid_get_decoder_caps)(&mut caps) == 0 {
                    if caps.is_supported == 0 {
                        let reason = format!(
                            "GPU NVDEC does not support codec={} chroma={} {}-bit",
                            state.codec_type,
                            fmt.chroma_format,
                            fmt.bit_depth_luma_minus8 + 8
                        );
                        tracing::warn!(
                            codec = state.codec_type,
                            chroma = fmt.chroma_format,
                            bit_depth = fmt.bit_depth_luma_minus8 + 8,
                            "NVDEC rejecting: {reason}"
                        );
                        state.set_typed_error(NvdecError::UnsupportedByHardware { reason });
                        return 0;
                    }
                    if caps.max_width > 0
                        && caps.max_height > 0
                        && (fmt.coded_width > caps.max_width || fmt.coded_height > caps.max_height)
                    {
                        let reason = format!(
                            "frame {}x{} exceeds NVDEC max {}x{}",
                            fmt.coded_width, fmt.coded_height, caps.max_width, caps.max_height
                        );
                        tracing::warn!(
                            w = fmt.coded_width,
                            h = fmt.coded_height,
                            max_w = caps.max_width,
                            max_h = caps.max_height,
                            "NVDEC rejecting: {reason}"
                        );
                        state.set_typed_error(NvdecError::UnsupportedByHardware { reason });
                        return 0;
                    }
                    tracing::debug!(
                        codec = state.codec_type,
                        max_w = caps.max_width,
                        max_h = caps.max_height,
                        "NVDEC capability validated"
                    );
                }
            }

            let is_high_bit_depth = fmt.bit_depth_luma_minus8 > 0;
            // Record the bit depth on the callback state so
            // display_callback knows whether to use the NV12 (1 byte
            // per sample) or P016 (2 bytes per sample) copy path.
            state.bit_depth_luma_minus8 = fmt.bit_depth_luma_minus8;

            // Color metadata from CUVIDEOFORMAT.video_signal_description.
            // SDK layout (SDK 12.2 nvcuvid.h):
            //   byte 0: video_format:3 | video_full_range_flag:1 | reserved:4
            //   byte 1: color_primaries
            //   byte 2: transfer_characteristics
            //   byte 3: matrix_coefficients
            // Values per ITU-T H.273 / H.265 §E.3.1:
            //   matrix_coefficients  = 1  → BT.709
            //                          5  → BT.601 625
            //                          6  → BT.601 525
            //                          9  → BT.2020 non-constant luminance
            //                          10 → BT.2020 constant luminance
            // transfer_characteristics = 16 (PQ / SMPTE ST 2084) indicates
            // HDR10; we still tag as BT.2020 since ColorSpace doesn't yet
            // have an HDR10 variant — the transfer curve is a separate
            // concern from the matrix. If/when a downstream consumer
            // needs to distinguish, StreamInfo can grow a TransferFn
            // field.
            let cp = fmt.video_signal_description[1];
            let tc = fmt.video_signal_description[2];
            let mc = fmt.video_signal_description[3];
            let full_range = (fmt.video_signal_description[0] >> 3) & 1 == 1;
            state.vui_colour_primaries = cp;
            state.vui_transfer_characteristics = tc;
            state.vui_matrix_coefficients = mc;
            state.vui_full_range_flag = full_range;
            state.color_space = match mc {
                1 => ColorSpace::Bt709,
                5 | 6 => ColorSpace::Bt601,
                9 | 10 => ColorSpace::Bt2020,
                _ => {
                    // Unspecified (0/2) or unknown: infer from bit depth.
                    // HDR10 streams always hit the 10-bit path; non-HDR
                    // 10-bit streams are rare enough to tag BT.2020 as
                    // a conservative default.
                    if is_high_bit_depth {
                        ColorSpace::Bt2020
                    } else {
                        ColorSpace::Bt709
                    }
                }
            };
            tracing::info!(
                matrix_coefficients = mc,
                color_primaries = fmt.video_signal_description[1],
                transfer = fmt.video_signal_description[2],
                color_space = ?state.color_space,
                "NVDEC color metadata"
            );

            if state.decoder.is_none() {
                let mut create_info: CuVideoDecodeCreateInfo = std::mem::zeroed();
                create_info.code_width = geo.coded_width as c_ulong;
                create_info.coded_height = geo.coded_height as c_ulong;
                create_info.num_decode_surfaces = num_surfaces;
                create_info.codec_type = state.codec_type;
                create_info.chroma_format = CUVID_CHROMA_420;
                // Explicitly prefer the CUVID (native NVDEC) backend rather
                // than letting the driver pick DXVA on Windows. Matches
                // ffmpeg libavcodec/cuviddec.c. This is the leading
                // suspect for the H.264 segfault seen on Windows — a
                // DXVA-backed decoder hands back surfaces with different
                // pitch/layout semantics than our cuMemcpy2D assumes.
                create_info.creation_flags = CUVID_CREATE_PREFER_CUVID;
                create_info.bit_depth_minus8 = fmt.bit_depth_luma_minus8 as c_ulong;
                // P016 surface for 10/12-bit, NV12 for 8-bit. P016 lays
                // out 16 bits per sample with the high-order bits
                // carrying the actual 10/12-bit value (low bits zero).
                create_info.output_format = if is_high_bit_depth {
                    CUVID_FMT_P016
                } else {
                    CUVID_FMT_NV12
                };
                // Progressive → Weave (0, no-op).
                // Interlaced → codec-dependent:
                //   H.264 → Adaptive (2): best quality for MBAFF/PAFF
                //   that dominates the H.264 interlaced corpus.
                //   HEVC  → Bob (1): the driver rejects Adaptive for
                //   HEVC interlaced streams with INVALID_ARG (see
                //   codec-review-2 MEDIUM-5 and nvdec-segfault-hunt.md);
                //   Bob is the highest-quality mode the driver will
                //   accept for HEVC.
                //   Other codecs → Bob (1) as a safe default.
                create_info.deinterlace_mode = if fmt.progressive_sequence != 0 {
                    0
                } else if state.codec_type == CUVID_H264 {
                    2
                } else {
                    1
                };
                // Crop on the way out: the display rectangle of the coded
                // surface maps 1:1 onto a target of its own size, exactly
                // as ffmpeg's cuviddec sets the decoder up. The mapped
                // output surface is then `geo.width` x `geo.height` luma
                // rows followed by the chroma rows (display_callback reads
                // it with `state.height` as the chroma plane's row offset).
                create_info.display_area_left = geo.display_left as i16;
                create_info.display_area_top = geo.display_top as i16;
                create_info.display_area_right = geo.display_right as i16;
                create_info.display_area_bottom = geo.display_bottom as i16;
                create_info.target_width = geo.width as c_ulong;
                create_info.target_height = geo.height as c_ulong;
                create_info.target_rect_left = 0;
                create_info.target_rect_top = 0;
                create_info.target_rect_right = geo.width as i16;
                create_info.target_rect_bottom = geo.height as i16;
                // ffmpeg uses 1 output surface; we use 4 for better
                // pipelining between display_callback and the decoder.
                // Some drivers reject > 4 on older GPUs.
                create_info.num_output_surfaces = 4;
                // Leave max_width / max_height as zero per ffmpeg
                // (memset'd to zero; never written). Setting them equal
                // to coded dimensions rejects any future resolution
                // upshift within the stream and has been seen to trigger
                // INVALID_ARG on some driver versions.
                create_info.max_width = 0;
                create_info.max_height = 0;

                state.width = geo.width;
                state.height = geo.height;

                let mut decoder: CUvideodecoder = ptr::null_mut();
                let rc = (state.cuvid_create_decoder)(&mut decoder, &mut create_info);
                if rc != 0 {
                    state.set_error(format!("cuvidCreateDecoder failed: {rc}"));
                    return 0;
                }
                state.decoder = Some(decoder);
            }

            num_surfaces as c_int
        }))
        .unwrap_or(0)
    }
}

pub unsafe extern "C" fn decode_callback(
    user_data: *mut c_void,
    pic_params: *mut CuVideoPicParams,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if user_data.is_null() || pic_params.is_null() {
                return 0;
            }
            let state = &mut *(user_data as *mut CallbackState);

            let Some(decoder) = state.decoder else {
                state.set_error("decode_callback before decoder created");
                return 0;
            };

            let rc = (state.cuvid_decode_picture)(decoder, pic_params);
            if rc != 0 {
                state.set_error(format!("cuvidDecodePicture failed: {rc}"));
                return 0;
            }
            1
        }))
        .unwrap_or(0)
    }
}

pub unsafe extern "C" fn display_callback(
    user_data: *mut c_void,
    disp_info: *mut CuVideoDispInfo,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if user_data.is_null() || disp_info.is_null() {
                return 0;
            }
            let state = &mut *(user_data as *mut CallbackState);
            let info = &*disp_info;

            let Some(decoder) = state.decoder else {
                state.set_error("display_callback before decoder created");
                return 0;
            };

            // NVDEC occasionally returns a sentinel picture_index on parse
            // recovery paths (observed < 0 on truncated streams). Passing
            // it back to cuvidMapVideoFrame can segfault inside the driver.
            if info.picture_index < 0 {
                state.set_error(format!(
                    "display_callback picture_index invalid: {}",
                    info.picture_index
                ));
                return 0;
            }

            let mut proc_params: CuVideoProcParams = std::mem::zeroed();
            proc_params.progressive_frame = info.progressive_frame;
            proc_params.second_field = 0;
            proc_params.top_field_first = info.top_field_first;
            proc_params.unpaired_field = 0;

            let mut frame_ptr: CUdeviceptr = 0;
            let mut pitch: c_uint = 0;
            let rc = (state.cuvid_map_video_frame)(
                decoder,
                info.picture_index,
                &mut frame_ptr,
                &mut pitch,
                &mut proc_params,
            );
            if rc != 0 {
                state.set_error(format!("cuvidMapVideoFrame failed: {rc}"));
                return 0;
            }

            // The display picture's size — what the decoder was created to
            // output (its target), not the coded surface. The mapped frame
            // is laid out as `height` luma rows then ceil(height/2) chroma
            // rows, each `pitch` bytes wide.
            let width = state.width as usize;
            let height = state.height as usize;
            // 1 byte/sample for NV12, 2 bytes/sample for P016.
            // Chroma plane is ceil(width/2) × ceil(height/2) samples of
            // interleaved UV. Because chroma is stored as UV pairs side
            // by side at chroma resolution, its row width in bytes is
            // 2 * ceil(w/2) * bytes_per_sample — which for even widths
            // equals the luma row width. For odd widths NVDEC already
            // pads up to ceil(w/2) when outputting NV12/P016, so using
            // the luma row_bytes is still the correct copy stride.
            //
            // codec-review-2 MEDIUM-4: previously we used `height/2`
            // here which silently truncated the last chroma row for
            // odd-height streams (1080 is even, but 1079-height HDR
            // tests and 4:2:0 film transfers hit the odd case). The
            // missing row showed up as a green band at the bottom of
            // the frame after NV12→planar.
            let bytes_per_sample = if state.bit_depth_luma_minus8 > 0 {
                2
            } else {
                1
            };
            let row_bytes = width * bytes_per_sample;
            let chroma_height = height.div_ceil(2);
            let y_bytes = row_bytes * height;
            let uv_bytes = row_bytes * chroma_height;
            let mut host_buf = vec![0u8; y_bytes + uv_bytes];

            let mut luma_copy: CudaMemcpy2D = std::mem::zeroed();
            luma_copy.src_memory_type = CU_MEMORYTYPE_DEVICE;
            luma_copy.src_device = frame_ptr;
            luma_copy.src_pitch = pitch as usize;
            luma_copy.dst_memory_type = CU_MEMORYTYPE_HOST;
            luma_copy.dst_host = host_buf.as_mut_ptr() as *mut c_void;
            luma_copy.dst_pitch = row_bytes;
            luma_copy.width_in_bytes = row_bytes;
            luma_copy.height = height;
            let rc = (state.cu_memcpy2d)(&luma_copy);
            if rc != 0 {
                (state.cuvid_unmap_video_frame)(decoder, frame_ptr);
                state.set_error(format!("cuMemcpy2D (luma) failed: {rc}"));
                return 0;
            }

            let chroma_src = frame_ptr + (pitch as CUdeviceptr) * (height as CUdeviceptr);
            let mut chroma_copy: CudaMemcpy2D = std::mem::zeroed();
            chroma_copy.src_memory_type = CU_MEMORYTYPE_DEVICE;
            chroma_copy.src_device = chroma_src;
            chroma_copy.src_pitch = pitch as usize;
            chroma_copy.dst_memory_type = CU_MEMORYTYPE_HOST;
            chroma_copy.dst_host = host_buf[y_bytes..].as_mut_ptr() as *mut c_void;
            chroma_copy.dst_pitch = row_bytes;
            chroma_copy.width_in_bytes = row_bytes;
            // ceil(h/2) rows — see host_buf sizing above. The driver
            // always emits ceil(h/2) chroma rows regardless of parity;
            // the previous `height / 2` dropped the last row on odd h.
            chroma_copy.height = chroma_height;
            let rc = (state.cu_memcpy2d)(&chroma_copy);

            let _ = (state.cuvid_unmap_video_frame)(decoder, frame_ptr);

            if rc != 0 {
                state.set_error(format!("cuMemcpy2D (chroma) failed: {rc}"));
                return 0;
            }

            if let Ok(mut c) = state.collector.lock() {
                // push_back so the streaming reader (NvdecStreamingDecoder)
                // can pop_front in display order. Eager callers drain
                // sequentially after teardown — same observed order
                // either way.
                c.frames.push_back(DecodedFrame {
                    nv12: host_buf,
                    width: state.width,
                    height: state.height,
                    bit_depth_minus8: state.bit_depth_luma_minus8,
                    color_space: state.color_space,
                    timestamp: info.timestamp,
                });
            }
            1
        }))
        .unwrap_or(0)
    }
}

/// AV1 operating-point callback for `pfn_get_operating_point` on
/// `CuVideoParserParams` (NVIDIA Video Codec SDK 12.x).
///
/// PROBLEMS.md §"NVDEC AV1 — CUVIDEOFORMAT layout mismatch" hypothesis:
/// on AV1 streams, when the callback is *not* set, some SDK versions
/// don't fully populate the `CUVIDEOFORMAT` named fields before
/// `pfn_sequence_callback` fires — we observed `chroma_format=3`
/// (4:4:4) and `coded_width=coded_height=0` on a clean SVT-AV1 4:2:0
/// source, which means the whole struct was being read at the wrong
/// offset. FFmpeg's `libavcodec/cuviddec.c::cuvid_handle_operating_point`
/// always wires this callback for AV1; the parser may take a different
/// code path depending on whether it's set.
///
/// Return value encoding (per SDK nvcuvid.h):
///   `(output_all_layers << 16) | operating_point_index`
///
/// We pick operating point 0 (always present — the base layer for
/// scalable streams, the entire bitstream for single-layer streams)
/// with `output_all_layers = 0`. Matches FFmpeg's default and what
/// mainstream players use for non-scalable AV1.
///
/// The callback is wired on every `CuVideoParserParams` setup
/// regardless of codec; non-AV1 codecs ignore it (the SDK only calls
/// it from the AV1 parser path). Cost is one fn-pointer populated
/// per parser construction.
pub unsafe extern "C" fn get_operating_point_callback(
    _user_data: *mut c_void,
    _op_info: *mut c_void,
) -> c_int {
    // catch_unwind defence: the callback runs on an NVIDIA-driver-
    // owned thread; a Rust panic crossing a C boundary is UB. Matches
    // the other callbacks in this file.
    catch_unwind(AssertUnwindSafe(|| 0_i32)).unwrap_or(0)
}
