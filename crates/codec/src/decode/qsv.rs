//! Intel QSV / oneVPL decoder.
//!
//! Wraps the `shiguredo_vpl` crate (the only mature Rust binding for
//! libvpl on crates.io as of 2026-05). That crate static-links libvpl
//! 2.16 and bindgen's the headers; we just plug its high-level
//! `Decoder` API into the trait dispatcher in `decode/mod.rs`.
//!
//! Runtime requirements (provided by the prod Dockerfile + dev box):
//! - `libmfx-gen1.2` — Intel oneVPL GPU runtime, dlopen'd by the
//!   static libvpl from the wrapper crate.
//! - `libva2` + `libva-drm2` — VA-API loader.
//! - `intel-media-va-driver-non-free` — Intel iHD driver, the actual
//!   GPU backend.
//! - The container needs the Intel `/dev/dri/renderDXXX` exposed.
//!
//! Output: NV12 8-bit (Y plane + interleaved UV). The pipeline
//! prefers `Yuv420p` so we deinterleave on the way out.
//!
//! 10-bit: `shiguredo_vpl` 2026.1.x exposes only NV12. P010
//! support is a planned upstream addition; until then 10-bit / HDR
//! sources fall through to the next decoder tier.

use anyhow::{Context, Result, bail};
use bytes::{Bytes, BytesMut};
use std::collections::VecDeque;

use shiguredo_vpl::AdapterSelector;
use shiguredo_vpl::{DecodedFrame, DecoderCodec, DecoderConfig};

use super::Decoder;
use crate::frame::{ColorSpace, PixelFormat, StreamInfo, VideoFrame};
use crate::gpu::adapter_selector_for_gpu_index;

/// Map a codec string from `StreamInfo` to the wrapper crate's
/// codec enum. Returns `None` for codecs the wrapper doesn't
/// support (currently VP8 / MPEG-2 — vainfo lists VLD entrypoints
/// for them on Arc but the wrapper's `DecoderCodec` enum doesn't).
pub fn shiguredo_codec_for(codec_lower: &str) -> Option<DecoderCodec> {
    Some(match codec_lower {
        "h264" | "avc1" | "avc" => DecoderCodec::H264,
        "h265" | "hevc" | "hvc1" | "hev1" | "hvc2" | "hev2" => DecoderCodec::Hevc,
        "vp9" | "vp09" => DecoderCodec::Vp9,
        "av1" | "av01" => DecoderCodec::Av1,
        _ => return None,
    })
}

pub struct QsvDecoder {
    info: StreamInfo,
    inner: shiguredo_vpl::Decoder,
    pending: VecDeque<VideoFrame>,
    finished: bool,
}

unsafe impl Send for QsvDecoder {}

impl QsvDecoder {
    /// Construct a QSV decoder pinned to `gpu_index`'s physical
    /// adapter. `gpu_index` is the same numbering this crate uses in
    /// `crate::gpu::detect_intel` — sequential 0..N in PCI bus order.
    /// `adapter_selector_for_gpu_index` correlates that index to a
    /// libvpl DRM render node via `list_adapters()` (PCI-address
    /// match), so each session lands on its intended physical card
    /// instead of stacking on adapter 0.
    pub fn new(info: StreamInfo, gpu_index: u32) -> Result<Self> {
        let codec_lower = info.codec.to_ascii_lowercase();
        let codec = shiguredo_codec_for(&codec_lower).ok_or_else(|| {
            anyhow::anyhow!("QSV decoder: codec '{codec_lower}' not in shiguredo_vpl enum")
        })?;
        let adapter: AdapterSelector = adapter_selector_for_gpu_index(gpu_index)?;
        let inner =
            shiguredo_vpl::Decoder::new(DecoderConfig::new(adapter, codec)).with_context(|| {
                format!(
                    "shiguredo_vpl::Decoder::new (gpu_index={gpu_index}, adapter={adapter:?}) — \
                 Intel adapter visible? /dev/dri exposed?"
                )
            })?;
        Ok(Self {
            info,
            inner,
            pending: VecDeque::new(),
            finished: false,
        })
    }

    /// Drain all currently-ready frames from the wrapper into our
    /// pending queue, converting NV12 → Yuv420p along the way.
    fn drain_inner(&mut self) -> Result<()> {
        while let Some(decoded) = self.inner.next_frame() {
            let frame = nv12_decoded_to_yuv420p(decoded)?;
            self.pending.push_back(frame);
        }
        Ok(())
    }
}

impl Decoder for QsvDecoder {
    fn stream_info(&self) -> &StreamInfo {
        &self.info
    }

    fn push_sample(&mut self, data: &[u8]) -> Result<()> {
        self.inner
            .decode(data)
            .context("shiguredo_vpl::Decoder::decode")?;
        self.drain_inner()
    }

    fn finish(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        self.inner
            .finish()
            .context("shiguredo_vpl::Decoder::finish")?;
        self.drain_inner()
    }

    fn decode_next(&mut self) -> Result<Option<VideoFrame>> {
        Ok(self.pending.pop_front())
    }
}

/// NV12 (Y plane + interleaved UV) → Yuv420p (Y / U / V planar).
/// 8-bit only — wrapper crate doesn't expose P010 yet.
fn nv12_decoded_to_yuv420p(decoded: DecodedFrame) -> Result<VideoFrame> {
    let w = decoded.width();
    let h = decoded.height();
    if w == 0 || h == 0 {
        bail!("QSV decoder returned zero-dimension frame");
    }
    let y_size = w * h;
    let uv_planar_size = (w / 2) * (h / 2);
    let need = y_size + 2 * (w * h / 4);
    let bytes = decoded.into_data();
    if bytes.len() < y_size + uv_planar_size * 2 {
        // The interleaved UV is `(w/2) * (h/2) * 2` bytes — same as
        // `w * h / 2`. Total NV12 = w*h + w*h/2 = w*h*3/2. Cross-check
        // before we deinterleave.
        if bytes.len() < y_size + (w * h / 2) {
            bail!(
                "QSV decoder returned {} bytes for {}×{} NV12; need {}",
                bytes.len(),
                w,
                h,
                w * h * 3 / 2
            );
        }
    }
    let _ = need;

    let mut out = BytesMut::with_capacity(y_size + 2 * uv_planar_size);

    // Y plane: straight copy.
    out.extend_from_slice(&bytes[..y_size]);

    // UV interleaved → split into U plane then V plane.
    let uv = &bytes[y_size..y_size + (w * h / 2)];
    let mut u_plane = Vec::with_capacity(uv_planar_size);
    let mut v_plane = Vec::with_capacity(uv_planar_size);
    let mut i = 0;
    while i + 1 < uv.len() {
        u_plane.push(uv[i]);
        v_plane.push(uv[i + 1]);
        i += 2;
    }
    out.extend_from_slice(&u_plane);
    out.extend_from_slice(&v_plane);

    Ok(VideoFrame::new(
        Bytes::from(out),
        w as u32,
        h as u32,
        PixelFormat::Yuv420p,
        // The wrapper doesn't surface BT.601 vs BT.709 from the bitstream
        // — assume BT.709 (the common modern default). The colorspace
        // converter in the pipeline trusts the source's `ColorMetadata`
        // on `StreamInfo` rather than the per-frame `color_space` for
        // matrix correction, so this default is a safe placeholder.
        ColorSpace::Bt709,
        0,
    ))
}
