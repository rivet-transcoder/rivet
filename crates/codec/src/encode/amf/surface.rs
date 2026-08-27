//! AMF surface RAII guard and NV12/P010 frame upload helpers.
//!
//! `SurfaceGuard` wraps the caller-held ref on an AMF surface so it is
//! released on every exit path (panic, `?`, `bail!`). `upload_frame_static`
//! allocates a fresh AMF surface and copies a `VideoFrame` into it;
//! `copy_yuv420p_to_nv12_surface` and `copy_yuv420p10le_to_p010_surface`
//! do the per-format byte work.

use anyhow::{Result, bail};
use std::ffi::c_void;
use std::ptr;

use crate::frame::VideoFrame;

// Items from ffi.rs accessed via the parent (amf) module's private-use
// re-export (`use self::ffi::*;` in mod.rs).
use super::{
    AMF_MEMORY_HOST, AMF_OK, AMF_PLANE_UV, AMF_PLANE_Y, AMF_SURFACE_NV12, AMF_SURFACE_P010,
    AmfContextObj, AmfPlaneObj, AmfSurfaceObj, AmfSurfaceVtbl, result_name,
};

// ─── RAII surface guard ──────────────────────────────────────────
//
// Wraps the caller-held ref on an AMF surface so it gets released on every
// exit path — including `bail!`, `?` early-return, and panic unwind (which
// catch_unwind converts to an error). Drop is a no-op after
// `transfer_to_encoder` marks the ref as consumed by `SubmitInput` returning
// AMF_OK / AMF_NEED_MORE_INPUT.
pub(super) struct SurfaceGuard {
    pub(super) surface: *mut c_void,
    owned: bool,
}

impl SurfaceGuard {
    pub(super) fn new(surface: *mut c_void) -> Self {
        Self {
            surface,
            owned: true,
        }
    }

    /// Marks the caller-held ref as transferred to the encoder. After
    /// this, `Drop` will NOT release. Call this immediately after the
    /// `SubmitInput` call that returned `AMF_OK` / `AMF_NEED_MORE_INPUT`.
    pub(super) fn transfer_to_encoder(&mut self) {
        self.owned = false;
    }

    pub(super) fn as_ptr(&self) -> *mut c_void {
        self.surface
    }
}

impl Drop for SurfaceGuard {
    fn drop(&mut self) {
        if self.owned && !self.surface.is_null() {
            unsafe {
                let obj = self.surface as *mut AmfSurfaceObj;
                let vt = &*(*obj).vtbl;
                (vt.data.ps.release)(self.surface);
            }
        }
    }
}

// ─── Session snapshot ────────────────────────────────────────────

/// Plain-data snapshot of the fields `upload_frame_static` needs. Used
/// so we can hold session pointers across a self-mutating call without
/// fighting the borrow checker.
#[derive(Clone, Copy)]
pub(super) struct SessionSnapshot {
    pub(super) context: *mut c_void,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) pts_timescale: u64,
    /// `AMF_SURFACE_NV12` or `AMF_SURFACE_P010`.
    pub(super) surface_format: i32,
}

// ─── Surface upload ───────────────────────────────────────────────

/// Copy a YUV420p (8-bit) or Yuv420p10le (10-bit) frame into a freshly-
/// allocated AMF surface. The surface format must already have been
/// captured into `snap.surface_format` at session-create time —
/// dispatching here per-frame would silently mismatch the encoder
/// component's Init format.
///
/// Returns an AMF-owned surface pointer; caller must Release when
/// done (SubmitInput keeps its own internal ref, so one Release
/// balances one AllocSurface regardless of SubmitInput outcome).
pub(super) unsafe fn upload_frame_static(
    snap: &SessionSnapshot,
    frame: &VideoFrame,
) -> Result<*mut c_void> {
    unsafe {
        let context_obj = snap.context as *mut AmfContextObj;
        let context_vt = &*(*context_obj).vtbl;

        let mut surface: *mut c_void = ptr::null_mut();
        let rc = (context_vt.alloc_surface)(
            snap.context,
            AMF_MEMORY_HOST,
            snap.surface_format,
            snap.width as i32,
            snap.height as i32,
            &mut surface,
        );
        if rc != AMF_OK || surface.is_null() {
            bail!(
                "AMFContext::AllocSurface({}x{} fmt={}) failed: {rc} ({})",
                snap.width,
                snap.height,
                snap.surface_format,
                result_name(rc),
            );
        }

        let surface_obj = surface as *mut AmfSurfaceObj;
        let surface_vt = &*(*surface_obj).vtbl;

        let y_plane = (surface_vt.get_plane)(surface, AMF_PLANE_Y);
        let uv_plane = (surface_vt.get_plane)(surface, AMF_PLANE_UV);
        if y_plane.is_null() || uv_plane.is_null() {
            (surface_vt.data.ps.release)(surface);
            bail!(
                "AMF surface (fmt={}) missing Y or UV plane",
                snap.surface_format
            );
        }

        // Per-format upload. Both branches share the plane geometry +
        // PTS write; they only differ in per-sample byte width and the
        // P010 `<<6` shift.
        let upload_result = match snap.surface_format {
            AMF_SURFACE_NV12 => copy_yuv420p_to_nv12_surface(
                surface,
                surface_vt,
                y_plane,
                uv_plane,
                snap.width,
                snap.height,
                frame,
            ),
            AMF_SURFACE_P010 => copy_yuv420p10le_to_p010_surface(
                surface,
                surface_vt,
                y_plane,
                uv_plane,
                snap.width,
                snap.height,
                frame,
            ),
            other => {
                (surface_vt.data.ps.release)(surface);
                bail!("AMF surface format {other} not supported by uploader");
            }
        };
        upload_result?;

        (surface_vt.data.set_pts)(surface, (frame.pts * snap.pts_timescale) as i64);

        Ok(surface)
    }
}

/// 8-bit YUV420p → AMF NV12 surface. Y plane: byte copy at
/// surface pitch. UV plane: interleave U + V from separate planes
/// into the single NV12 chroma plane.
unsafe fn copy_yuv420p_to_nv12_surface(
    surface: *mut c_void,
    surface_vt: &AmfSurfaceVtbl,
    y_plane: *mut c_void,
    uv_plane: *mut c_void,
    width: u32,
    height: u32,
    frame: &VideoFrame,
) -> Result<()> {
    unsafe {
        let w = width as usize;
        let h = height as usize;
        let y_size = w * h;
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        let uv_size = cw * ch;

        if frame.data.len() < y_size + 2 * uv_size {
            (surface_vt.data.ps.release)(surface);
            bail!(
                "frame data too small for {}x{} YUV420p: need {} bytes, got {}",
                w,
                h,
                y_size + 2 * uv_size,
                frame.data.len()
            );
        }

        let y_plane_obj = y_plane as *mut AmfPlaneObj;
        let y_vt = &*(*y_plane_obj).vtbl;
        let y_dst = (y_vt.get_native)(y_plane) as *mut u8;
        let y_pitch = (y_vt.get_h_pitch)(y_plane) as usize;
        if y_dst.is_null() {
            (surface_vt.data.ps.release)(surface);
            bail!("AMF Y plane native pointer is null — surface not host-mapped?");
        }
        for row in 0..h {
            let src = frame.data.as_ptr().add(row * w);
            let dst = y_dst.add(row * y_pitch);
            ptr::copy_nonoverlapping(src, dst, w);
        }

        let uv_plane_obj = uv_plane as *mut AmfPlaneObj;
        let uv_vt = &*(*uv_plane_obj).vtbl;
        let uv_dst = (uv_vt.get_native)(uv_plane) as *mut u8;
        let uv_pitch = (uv_vt.get_h_pitch)(uv_plane) as usize;
        if uv_dst.is_null() {
            (surface_vt.data.ps.release)(surface);
            bail!("AMF UV plane native pointer is null — surface not host-mapped?");
        }
        let u_src_base = frame.data.as_ptr().add(y_size);
        let v_src_base = u_src_base.add(uv_size);
        for row in 0..ch {
            let u_src = u_src_base.add(row * cw);
            let v_src = v_src_base.add(row * cw);
            let dst_row = uv_dst.add(row * uv_pitch);
            for col in 0..cw {
                *dst_row.add(col * 2) = *u_src.add(col);
                *dst_row.add(col * 2 + 1) = *v_src.add(col);
            }
        }
        Ok(())
    }
}

/// 10-bit `Yuv420p10le` → AMF P010 surface. Same plane geometry as
/// NV12 but each sample is 2 bytes; P010 stores the valid 10-bit
/// value in the **upper 10 bits** of the 16-bit word (`core/Surface.h:62`),
/// so we shift each source sample left by 6 on the way in. Source format
/// keeps the value in the **lower 10 bits** (matches NVDEC `>>6`-normalized
/// surface output from Squad-6).
unsafe fn copy_yuv420p10le_to_p010_surface(
    surface: *mut c_void,
    surface_vt: &AmfSurfaceVtbl,
    y_plane: *mut c_void,
    uv_plane: *mut c_void,
    width: u32,
    height: u32,
    frame: &VideoFrame,
) -> Result<()> {
    unsafe {
        let w = width as usize;
        let h = height as usize;
        let cw = w.div_ceil(2);
        let ch = h.div_ceil(2);
        // 2 bytes per sample.
        let y_bytes = w * h * 2;
        let uv_bytes = cw * ch * 2;

        if frame.data.len() < y_bytes + 2 * uv_bytes {
            (surface_vt.data.ps.release)(surface);
            bail!(
                "frame data too small for {}x{} Yuv420p10le: need {} bytes, got {}",
                w,
                h,
                y_bytes + 2 * uv_bytes,
                frame.data.len()
            );
        }

        let y_plane_obj = y_plane as *mut AmfPlaneObj;
        let y_vt = &*(*y_plane_obj).vtbl;
        let y_dst = (y_vt.get_native)(y_plane) as *mut u8;
        let y_pitch_bytes = (y_vt.get_h_pitch)(y_plane) as usize;
        if y_dst.is_null() {
            (surface_vt.data.ps.release)(surface);
            bail!("AMF P010 Y plane native pointer is null");
        }

        let src_ptr = frame.data.as_ptr();

        // Y plane: w samples per row. The source rows may be unaligned
        // (the frame buffer is a byte slice), so read them as bytes.
        for row in 0..h {
            let src_row = src_ptr.add(row * w * 2);
            let dst_row = y_dst.add(row * y_pitch_bytes) as *mut u16;
            for col in 0..w {
                let sample = u16::from_le_bytes([*src_row.add(col * 2), *src_row.add(col * 2 + 1)]) & 0x03FF;
                dst_row.add(col).write_unaligned(sample << 6);
            }
        }

        let uv_plane_obj = uv_plane as *mut AmfPlaneObj;
        let uv_vt = &*(*uv_plane_obj).vtbl;
        let uv_dst = (uv_vt.get_native)(uv_plane) as *mut u8;
        let uv_pitch_bytes = (uv_vt.get_h_pitch)(uv_plane) as usize;
        if uv_dst.is_null() {
            (surface_vt.data.ps.release)(surface);
            bail!("AMF P010 UV plane native pointer is null");
        }
        let u_src_base = src_ptr.add(y_bytes);
        let v_src_base = u_src_base.add(uv_bytes);
        // UV plane: cw samples (cw*2 bytes) per row interleaved as
        // U,V,U,V… (pitch is in bytes).
        for row in 0..ch {
            let u_src = u_src_base.add(row * cw * 2);
            let v_src = v_src_base.add(row * cw * 2);
            let dst_row = uv_dst.add(row * uv_pitch_bytes) as *mut u16;
            for col in 0..cw {
                let u = u16::from_le_bytes([*u_src.add(col * 2), *u_src.add(col * 2 + 1)]) & 0x03FF;
                let v = u16::from_le_bytes([*v_src.add(col * 2), *v_src.add(col * 2 + 1)]) & 0x03FF;
                dst_row.add(col * 2).write_unaligned(u << 6);
                dst_row.add(col * 2 + 1).write_unaligned(v << 6);
            }
        }
        Ok(())
    }
}
