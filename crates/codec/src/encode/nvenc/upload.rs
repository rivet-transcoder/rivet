//! NVENC surface upload: copy YUV frames into locked input buffers.

use anyhow::{bail, Result};
use std::ptr;

use crate::frame::VideoFrame;

use super::buffers::NvEncLockInputBuffer;
use super::constants::{NV_ENC_LOCK_INPUT_BUFFER_VER, NV_ENC_SUCCESS};
use super::session::EncodeSession;

/// Copy an 8-bit YUV420p frame into a locked NVENC IYUV surface.
/// Layout: Y plane, then U plane, then V plane, each laid out
/// contiguously at the surface's pitch.
///
/// MEDIUM-6: chroma plane dims use round-up `(w+1)/2, (h+1)/2`
/// so odd widths/heights don't truncate the last column/row
/// (mirrors the NVDEC fix in systems-review-2 M-N1).
pub(super) unsafe fn upload_frame(
    session: &EncodeSession,
    frame: &VideoFrame,
    slot: usize,
) -> Result<u32> {
    unsafe {
        let input_buffer = session.input_buffers[slot];
        let mut lock: NvEncLockInputBuffer = std::mem::zeroed();
        lock.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
        lock.input_buffer = input_buffer;
        let rc = (session.fn_lock_input_buffer)(session.encoder, &mut lock);
        if rc != NV_ENC_SUCCESS {
            bail!("NvEncLockInputBuffer failed: {rc}");
        }

        let pitch = lock.pitch as usize;
        let w = session.width as usize;
        let h = session.height as usize;
        // Round-up chroma dims for 4:2:0.
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let y_size = w * h;
        let uv_size = cw * ch;

        if frame.data.len() < y_size + 2 * uv_size {
            (session.fn_unlock_input_buffer)(session.encoder, input_buffer);
            bail!("frame data too small for {}x{} YUV420p", w, h);
        }

        let dst = lock.buffer_data_ptr as *mut u8;

        // Y plane: one row at a time to honor the surface pitch.
        for row in 0..h {
            let src = frame.data.as_ptr().add(row * w);
            let dst_row = dst.add(row * pitch);
            ptr::copy_nonoverlapping(src, dst_row, w);
        }

        // U plane: starts at dst + pitch*h on the surface. NVENC's
        // IYUV layout uses HALF-PITCH for chroma rows (the
        // surface is allocated as pitch*h Y + (pitch/2)*ch U +
        // (pitch/2)*ch V — a 4:2:0 chroma subsampling with
        // proportionally narrower chroma rows). The previous
        // mirror used full pitch for chroma rows, which hid on
        // sub-1080p sources where the driver-allocated surface
        // happened to have enough headroom but reliably SIGSEGV'd
        // at 4K (3840×2160 with pitch=4096): full-pitch chroma
        // ended up at offset 13.27 MiB while the driver's IYUV
        // surface only allocates ~13.27 MiB total — chroma writes
        // ran off the end. Verified 2026-05-01 against the 4K
        // segfault repro on the dev box.
        let chroma_pitch = pitch / 2;
        let u_dst_base = dst.add(pitch * h);
        let u_src_base = frame.data.as_ptr().add(y_size);
        for row in 0..ch {
            let src = u_src_base.add(row * cw);
            let dst_row = u_dst_base.add(row * chroma_pitch);
            ptr::copy_nonoverlapping(src, dst_row, cw);
        }

        // V plane: follows U at chroma_pitch*ch further in.
        let v_dst_base = u_dst_base.add(chroma_pitch * ch);
        let v_src_base = u_src_base.add(uv_size);
        for row in 0..ch {
            let src = v_src_base.add(row * cw);
            let dst_row = v_dst_base.add(row * chroma_pitch);
            ptr::copy_nonoverlapping(src, dst_row, cw);
        }

        let rc = (session.fn_unlock_input_buffer)(session.encoder, input_buffer);
        if rc != NV_ENC_SUCCESS {
            bail!("NvEncUnlockInputBuffer failed: {rc}");
        }
        Ok(lock.pitch)
    }
}

/// Copy a 10-bit YUV420p frame (`Yuv420p10le` — u16 LE per sample,
/// valid value in the lower 10 bits) into a locked
/// `NV_ENC_BUFFER_FORMAT_YUV420_10BIT` surface (P010-style — u16 LE
/// per sample, valid value in the **upper 10 bits**, i.e.
/// `sample_10bit << 6`).
///
/// `pitch` from `NvEncLockInputBuffer` is in **bytes**, not samples;
/// for a 10-bit surface that's 2× the sample count per row.
/// Plane layout is semi-planar (P010): the Y plane, then one
/// interleaved `U0 V0 U1 V1 …` chroma plane at the luma pitch, 2
/// bytes/sample — not IYUV's separate U and V planes. The planar
/// source U and V planes are interleaved on the copy.
///
/// Round-up `(w+1)/2`, `(h+1)/2` chroma dims for odd dims, same as
/// the 8-bit path (MEDIUM-6 in codec-review-3).
pub(super) unsafe fn upload_frame_10bit(
    session: &EncodeSession,
    frame: &VideoFrame,
    slot: usize,
) -> Result<u32> {
    unsafe {
        let input_buffer = session.input_buffers[slot];
        let mut lock: NvEncLockInputBuffer = std::mem::zeroed();
        lock.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
        lock.input_buffer = input_buffer;
        let rc = (session.fn_lock_input_buffer)(session.encoder, &mut lock);
        if rc != NV_ENC_SUCCESS {
            bail!("NvEncLockInputBuffer failed: {rc}");
        }

        let pitch_bytes = lock.pitch as usize;
        let w = session.width as usize;
        let h = session.height as usize;
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        // Frame data layout (Yuv420p10le): Y plane (w*h u16) + U
        // plane (cw*ch u16) + V plane (cw*ch u16). Bytes are
        // 2× the sample counts.
        let y_bytes = w * h * 2;
        let uv_bytes = cw * ch * 2;
        if frame.data.len() < y_bytes + 2 * uv_bytes {
            (session.fn_unlock_input_buffer)(session.encoder, input_buffer);
            bail!(
                "frame data too small for {}x{} Yuv420p10le: need {} bytes, got {}",
                w,
                h,
                y_bytes + 2 * uv_bytes,
                frame.data.len()
            );
        }

        let dst = lock.buffer_data_ptr as *mut u8;
        let src_ptr = frame.data.as_ptr();

        // Y plane: w samples per row, 2*w bytes per row, shift each
        // u16 left by 6 to satisfy the SDK's upper-10-bit
        // convention.
        for row in 0..h {
            let src_row = src_ptr.add(row * w * 2) as *const u16;
            let dst_row = dst.add(row * pitch_bytes) as *mut u16;
            for col in 0..w {
                // `<<6` keeps the significant bits in the
                // upper 10 of the 16-bit container; the bottom 6
                // bits are zero (matches NVDEC P016 output
                // emitted by Squad-6 before its `>>6` normalize).
                let sample = (*src_row.add(col)) & 0x03FF;
                *dst_row.add(col) = sample << 6;
            }
        }

        // UV plane: `NV_ENC_BUFFER_FORMAT_YUV420_10BIT` is **semi-planar**
        // (P010-style) — a single interleaved chroma plane `U0 V0 U1 V1 …`,
        // NOT separate U/V planes. It starts at dst + pitch_bytes*h and uses
        // the FULL luma pitch stride (each chroma row packs cw U+V pairs =
        // cw*2 samples = the same byte width as a luma row). Writing U and V
        // as separate half-pitch planes (the planar IYUV layout) decodes as
        // garbage chroma (Y PSNR fine, U/V ~6 dB). Source is planar
        // Yuv420p10le, so interleave its separate U/V planes here.
        let uv_dst_base = dst.add(pitch_bytes * h);
        let u_src_base = src_ptr.add(y_bytes) as *const u16;
        let v_src_base = src_ptr.add(y_bytes + uv_bytes) as *const u16;
        for row in 0..ch {
            let u_src_row = u_src_base.add(row * cw);
            let v_src_row = v_src_base.add(row * cw);
            let dst_row = uv_dst_base.add(row * pitch_bytes) as *mut u16;
            for col in 0..cw {
                let u = (*u_src_row.add(col)) & 0x03FF;
                let v = (*v_src_row.add(col)) & 0x03FF;
                *dst_row.add(col * 2) = u << 6;
                *dst_row.add(col * 2 + 1) = v << 6;
            }
        }

        let rc = (session.fn_unlock_input_buffer)(session.encoder, input_buffer);
        if rc != NV_ENC_SUCCESS {
            bail!("NvEncUnlockInputBuffer failed: {rc}");
        }
        Ok(lock.pitch)
    }
}
