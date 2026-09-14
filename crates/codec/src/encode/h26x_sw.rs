//! Native H.264 / H.265 encode in software — rivet's own encoders.
//!
//! The `h26x` crate is this workspace's pure-Rust H.264 and H.265 codec pair.
//! Its decoders are bit-exact against the JVT and JCT-VC conformance suites;
//! its encoders are the mirror of them, built on the same reconstruction
//! kernels, and held to a four-property gate (`h26x/tools/verify_encode.sh`):
//! our decoder reproduces the encoder's own reconstruction byte for byte
//! (**SELF**), libavcodec agrees with our decoder (**CROSS**), PSNR is reported,
//! and a rate objective, where one is set, is hit. No C, no system library,
//! nothing to install on a build host — so, like the decoders, this module is
//! always compiled.
//!
//! # Where it sits
//!
//! Below every hardware tier: a fixed-function block is faster and costs no
//! CPU. It is the last tier for the two codecs it serves, the way
//! [`rav1e_sw`](super::rav1e_sw) is for AV1, and it exists for the same
//! hosts — a laptop, a CI runner, a container with no GPU attached — where a
//! slow file beats a diagnostic.
//!
//! # Always built; the feature decides whether it is *reached*
//!
//! `h26x-fallback` gates whether [`select_encoder`](super::select_encoder)
//! **falls back** here on its own when every hardware backend has declined.
//! Off by default, for the reason `rav1e-fallback` is: a throughput fleet
//! quietly degrading into a CPU encoder reads as a capacity problem rather
//! than the missing driver it is. A caller that wants software encoding can
//! always ask for it by name, feature or no feature.
//!
//! # What it takes
//!
//! 4:2:0 at 8 bits for both codecs, and at 10 bits for H.265 (Main 10, the
//! HDR path — little-endian `u16` planes, the pipeline's `yuv420p10le`).
//! The H.264 encoder is 8-bit today and refuses deeper by name, as does
//! H.264 on every hardware backend here, so a 10-bit H.264 request is
//! refused rather than narrowed. The pipeline's other chroma layouts are
//! converted before the encoder anyway.
//!
//! # Colour
//!
//! The stream says what colour it is: `config.color_metadata` becomes the
//! SPS VUI's colour description (`video_signal_type_present_flag` — the
//! H.273 primaries / transfer / matrix codes and the range flag), the same
//! signalling the hardware backends write, so a BT.2020 PQ or HLG picture
//! is shown as HDR by players that read the bitstream before the
//! container's `colr` box (and by the ones that never read the box).
//! Written for every stream, SDR included — rivet always knows its output
//! colour, and a stream that says BT.709 is better than one that leaves
//! the player to assume it. `backend_output_caps` therefore reports this
//! tier as 10-bit **with** HDR, and an `Hdr10` / `Hlg` policy validates on
//! a build with no GPU. The HDR10 static metadata, when the source had it
//! (`mastering_display`, `content_light_level`), goes into the bitstream
//! too — the mastering display colour volume and content light level SEIs
//! (137 / 144) in every IDR access unit, for both codecs — beside the
//! container's `mdcv` / `clli`.
//!
//! # Threads
//!
//! Each encoder runs its own worker pool sized to `threads`, or to the
//! machine when that is zero (`H26X_THREADS` overrides). The pipeline runs a
//! ladder's rungs as separate encoders, so a caller running several at once
//! should hand each a share.
//!
//! # Order
//!
//! B pictures are enabled from `overrides.bframes` (non-pyramid: a fixed run
//! of B pictures between anchors, matching the NVENC/QSV plumbing). With
//! them coding order is not display order, so a coded picture's timestamp is
//! not the one that just arrived — it is the timestamp of the picture at the
//! access unit's stream-wide *display* index. The encoder reports that index
//! (`Access::display`); the muxer carries the composition offsets it implies.
//! Left at zero the tier is byte-identical to the no-B one it replaced.

use std::collections::VecDeque;

use anyhow::{Context, Result, bail};
use bytes::Bytes;

use super::{AUTO_FROM_TARGET, EncodedPacket, Encoder, EncoderConfig};
use crate::encode::tuning::h26x_sw_params_with;
use crate::frame::{
    ColorMetadata, ContentLightLevel, MasteringDisplay, PixelFormat, TransferFn, VideoCodec, VideoFrame,
};

/// The pipeline's mastering display as the encoder's: the same ten
/// integers in the same units (chromaticities in 0.00002, luminances in
/// 0.0001 cd/m² — both structs are the SEI's wire values), regrouped by
/// primary.
fn mastering_display(m: &MasteringDisplay) -> h26x::encode::MasteringDisplay {
    h26x::encode::MasteringDisplay {
        red: (m.primaries_r_x, m.primaries_r_y),
        green: (m.primaries_g_x, m.primaries_g_y),
        blue: (m.primaries_b_x, m.primaries_b_y),
        white_point: (m.white_point_x, m.white_point_y),
        max_luminance: m.max_luminance,
        min_luminance: m.min_luminance,
    }
}

/// The pipeline's content light level as the encoder's: two cd/m² values.
fn content_light(c: &ContentLightLevel) -> h26x::encode::ContentLightLevel {
    h26x::encode::ContentLightLevel { max_cll: c.max_cll, max_fall: c.max_fall }
}

/// The H.273 `transfer_characteristics` code the SPS VUI carries for a
/// pipeline transfer — the inverse of [`TransferFn::from_h273`], which
/// folds the whole BT.709 family (1, 6, 14, 15) onto `Bt709`; this writes
/// the family's canonical 1. `Unspecified` is written as 1 too, as the
/// three hardware backends write it (`nvenc::helpers::transfer_to_h273`
/// and its AMF / QSV twins): every player reads an unsignalled transfer
/// as BT.709 anyway, and the four tiers should describe the same picture
/// the same way.
fn transfer_to_h273(tf: TransferFn) -> u8 {
    match tf {
        TransferFn::Bt709 => 1,
        TransferFn::Bt470Bg => 4,
        TransferFn::Linear => 8,
        TransferFn::St2084 => 16,
        TransferFn::AribStdB67 => 18,
        TransferFn::Unspecified => 1,
    }
}

/// The VUI colour description for the pipeline's colour metadata: the
/// primaries and matrix are already H.273 codes and are copied, the
/// transfer is mapped by [`transfer_to_h273`], the range flag is the
/// range flag. HDR10 metadata (BT.2020 + PQ) becomes `9 / 16 / 9`, HLG
/// `9 / 18 / 9`, the SDR default `1 / 1 / 1` limited range.
fn colour_description(cm: &ColorMetadata) -> h26x::encode::ColourDescription {
    h26x::encode::ColourDescription {
        primaries: cm.colour_primaries,
        transfer: transfer_to_h273(cm.transfer),
        matrix: cm.matrix_coefficients,
        full_range: cm.full_range,
    }
}

/// The two encoders behind one face.
enum Inner {
    H264(h26x::encode::h264::H264Encoder),
    Hevc(h26x::encode::h265::H265Encoder),
}

impl Inner {
    fn push(&mut self, picture: &[u8]) -> h26x::Result<Vec<h26x::encode::Access>> {
        match self {
            Inner::H264(e) => e.push(picture),
            Inner::Hevc(e) => e.push(picture),
        }
    }
    fn flush(&mut self) -> h26x::Result<Vec<h26x::encode::Access>> {
        match self {
            Inner::H264(e) => e.flush(),
            Inner::Hevc(e) => e.flush(),
        }
    }
    fn frame_bytes(&self) -> usize {
        match self {
            Inner::H264(e) => e.frame_bytes(),
            Inner::Hevc(e) => e.frame_bytes(),
        }
    }
    fn force_idr(&mut self) {
        match self {
            Inner::H264(e) => e.force_idr(),
            Inner::Hevc(e) => e.force_idr(),
        }
    }
}

/// Software H.264 / H.265 encoder on the native `h26x` crate.
pub struct H26xEncoder {
    inner: Inner,
    /// The configuration `inner` was built from, kept so `reset` can build
    /// it again.
    cfg: h26x::encode::Config,
    codec: VideoCodec,
    width: u32,
    height: u32,
    /// The one pixel format this instance accepts, fixed at construction:
    /// the encoder's bit depth is in its SPS, so a frame of another depth
    /// cannot be taken mid-stream.
    format: PixelFormat,
    /// Timestamps in the order frames were pushed, indexed by stream-wide
    /// display index. A coded picture names the picture it codes by its
    /// display index (`Access::display`), which is where its timestamp sits
    /// here — the row is right whether or not the picture was reordered.
    /// A growing table rather than a queue so a forced IDR, which shifts the
    /// indices of nothing, cannot desynchronise it either.
    pts: Vec<u64>,
    /// Packets coded but not yet collected.
    ready: VecDeque<EncodedPacket>,
}

impl H26xEncoder {
    /// Whether this tier serves `codec`.
    pub fn supports(codec: VideoCodec) -> bool {
        matches!(codec, VideoCodec::H264 | VideoCodec::H265)
    }

    /// Build an encoder for `config`.
    ///
    /// Fails rather than silently degrading when the codec or frame format is
    /// not one the native encoders take. The caller's chain has already
    /// exhausted the hardware tiers by this point, so a clear error is more
    /// useful than a picture with the planes misread — or a 10-bit request
    /// shipped at 8.
    pub fn new(config: EncoderConfig) -> Result<Self> {
        if !Self::supports(config.codec) {
            bail!(
                "the native h26x encoders produce H.264 and H.265, not {:?}",
                config.codec
            );
        }
        // 4:2:0 at 8 bits for both codecs; 10 bits for H.265 (Main 10), the
        // HDR path. The H.264 encoder is still 8-bit and refuses deeper by
        // name — as does every hardware backend for H.264 — so a 10-bit
        // H.264 request is refused here rather than narrowed.
        let bit_depth = match (config.pixel_format, config.codec) {
            (PixelFormat::Yuv420p, _) => 8,
            (PixelFormat::Yuv420p10le, VideoCodec::H265) => 10,
            (PixelFormat::Yuv420p10le, VideoCodec::H264) => bail!(
                "the native H.264 encoder is 8-bit only (as is H.264 on every backend here); \
                 got yuv420p10le. Use --codec h265 for 10-bit output."
            ),
            (other, _) => bail!(
                "the native h26x software encoders take 4:2:0 at 8 bits (yuv420p), or 10 bits \
                 for H.265 (yuv420p10le); got {other:?}. Convert with the colorspace filter \
                 before the encoder."
            ),
        };

        let p = h26x_sw_params_with(config.codec, config.target, config.tier, &config.overrides);
        // The CRF escape hatch is already in this codec's currency (0..51),
        // and `resolve_overrides` has applied any per-rung delta to it, so it
        // replaces the derived quantiser outright.
        let qp = if config.quality == AUTO_FROM_TARGET {
            p.qp
        } else {
            config.quality.min(51)
        };

        // Zero means "decide for me", and the encoder's own zero means one
        // worker per core, which is the same answer — but the runtime's count
        // respects a container CPU quota where a core count does not, and a
        // job worker that ignores its quota starves everything sharing the
        // box. So resolve it here rather than passing the zero through.
        let threads = if config.threads > 0 {
            config.threads
        } else {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        };

        let cfg = h26x::encode::Config {
            width: config.width,
            height: config.height,
            bit_depth,
            chroma: h26x::ChromaFormat::Yuv420,
            // The encoder's zero means "every picture an IDR", which is not
            // what a caller leaving the interval unset wants.
            gop: if config.keyframe_interval == 0 { 250 } else { config.keyframe_interval },
            // Consecutive B pictures between anchors, non-pyramid — the same
            // grammar the hardware tiers read from the same override. The
            // encoder bumps `max_refs` to 2 itself when this is non-zero (a B
            // needs both anchors marked), so 1 is the honest floor to declare
            // here; over-declaring would only enlarge the DPB the SPS asks a
            // decoder to allocate.
            bframes: u32::from(config.overrides.bframes.unwrap_or(0)),
            max_refs: 1,
            rate: h26x::encode::RateControl::ConstantQp(qp),
            entropy: h26x::encode::Entropy::Cabac,
            transform_8x8: p.transform_8x8,
            subparts: p.subparts,
            sao: p.sao,
            threads,
            fps: (config.frame_rate.round() as u32).max(1),
            cpb_ms: 0,
            // Always: the pipeline resolved an output colour, and the stream
            // should say it rather than leave the player to assume BT.709
            // (right for SDR, wrong for everything this field exists for).
            colour: Some(colour_description(&config.color_metadata)),
            // The HDR10 static metadata, when the source carried it: an SEI
            // each in every IDR access unit, beside the container's boxes.
            mastering_display: config.color_metadata.mastering_display.as_ref().map(mastering_display),
            content_light: config.color_metadata.content_light_level.as_ref().map(content_light),
        };

        let inner = Self::build_inner(config.codec, &cfg)?;

        tracing::warn!(
            codec = ?config.codec,
            width = config.width,
            height = config.height,
            qp,
            transform_8x8 = p.transform_8x8,
            subparts = p.subparts,
            sao = p.sao,
            threads,
            colour = ?cfg.colour,
            hdr10_static_metadata = cfg.mastering_display.is_some() || cfg.content_light.is_some(),
            "no {:?} encode silicon available — falling back to the native software encoder, \
             which is far slower than any hardware backend",
            config.codec
        );

        Ok(Self {
            inner,
            cfg,
            codec: config.codec,
            width: config.width,
            height: config.height,
            format: config.pixel_format,
            pts: Vec::new(),
            ready: VecDeque::new(),
        })
    }

    fn build_inner(codec: VideoCodec, cfg: &h26x::encode::Config) -> Result<Inner> {
        Ok(match codec {
            VideoCodec::H264 => Inner::H264(
                h26x::encode::h264::H264Encoder::new(cfg.clone())
                    .context("the native H.264 encoder rejected the configuration")?,
            ),
            VideoCodec::H265 => Inner::Hevc(
                h26x::encode::h265::H265Encoder::new(cfg.clone())
                    .context("the native H.265 encoder rejected the configuration")?,
            ),
            VideoCodec::Av1 => unreachable!("checked by supports()"),
        })
    }

    /// Queue every access unit the encoder handed back.
    fn collect(&mut self, units: Vec<h26x::encode::Access>) -> Result<()> {
        for a in units {
            // The packet carries the picture it codes by *display* index; its
            // timestamp is the one that picture arrived with. Using the coding
            // index instead would be right only without B pictures and silently
            // wrong with them — a drift that plays fine, so it is exactly the
            // thing to get from the encoder rather than infer.
            let idx = usize::try_from(a.display).context("display index overflow")?;
            let pts = match self.pts.get(idx) {
                Some(&pts) => pts,
                None => bail!(
                    "h26x coded picture claims display index {} but only {} frames were pushed",
                    a.display,
                    self.pts.len()
                ),
            };
            self.ready.push_back(EncodedPacket {
                data: Bytes::from(a.data),
                pts,
                is_keyframe: a.keyframe,
            });
        }
        Ok(())
    }
}

impl Encoder for H26xEncoder {
    fn send_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        if frame.format != self.format {
            bail!(
                "the native h26x encoder was configured for {:?} and got a {:?} frame. Convert \
                 with the colorspace filter before the encoder.",
                self.format,
                frame.format
            );
        }
        if frame.width != self.width || frame.height != self.height {
            bail!(
                "frame is {}x{} but the encoder was configured for {}x{}",
                frame.width,
                frame.height,
                self.width,
                self.height
            );
        }
        let want = self.inner.frame_bytes();
        // A frame buffer may carry padding after the planes; the encoder wants
        // exactly its three planes, so hand it that prefix and no more.
        if frame.data.len() < want {
            bail!(
                "frame buffer is {} bytes, too short for {}x{} {:?} ({} expected)",
                frame.data.len(),
                self.width,
                self.height,
                self.format,
                want
            );
        }
        self.pts.push(frame.pts);
        let units = self
            .inner
            .push(&frame.data[..want])
            .with_context(|| format!("the native {:?} encoder refused a frame", self.codec))?;
        self.collect(units)
    }

    fn flush(&mut self) -> Result<()> {
        let units = self
            .inner
            .flush()
            .with_context(|| format!("the native {:?} encoder failed to flush", self.codec))?;
        self.collect(units)
    }

    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        Ok(self.ready.pop_front())
    }

    fn force_keyframe_next(&mut self) -> Result<()> {
        // Supported, which matters: the chunked path discards a lead-in and
        // needs the first kept frame promoted to an IDR or the chunk will not
        // stand alone.
        self.inner.force_idr();
        Ok(())
    }

    /// Rebuild the inner encoder from its own configuration.
    ///
    /// A rebuild *is* the reset here, and it is the cheaper of the two ways
    /// to get one. The native encoders own no threads, no device and no
    /// surface ring — the decoders have the worker pool, the encoders do not
    /// — so construction is a few derived tables (geometry, the intra
    /// kernels) and empty vectors: measured at 7 us (H.264) / 0.6 us (H.265) for a
    /// 640x360 session (`tests/h26x_sw_reset.rs`), against tens of
    /// milliseconds of encode for the shortest chunk the ladder makes. A reset that instead walked the encoder's
    /// state clearing references, the scheduler, `frame_num`, `idr_pic_id`
    /// and the rate ledger would save nothing measurable and add a second
    /// path to the "fresh stream" invariant that `new` already owns.
    ///
    /// What is *not* rebuilt is this wrapper's identity: the caller's
    /// session pool keeps the `Box<dyn Encoder>` and its counters see a
    /// reuse, which is what makes the software tier behave like the hardware
    /// ones under the same pool.
    fn reset(&mut self) -> Result<()> {
        self.inner = Self::build_inner(self.codec, &self.cfg)?;
        self.pts.clear();
        self.ready.clear();
        tracing::debug!(
            event = "h26x_sw.reset",
            codec = ?self.codec,
            "native h26x session reset (inner encoder rebuilt; the face and its pool slot survive)"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::tuning::{QualityTarget, SpeedTier};
    use crate::frame::ColorSpace;

    /// Every transfer the pipeline names comes back as itself through the
    /// decoder-side reader, so what the SPS says is what the source said.
    /// `Unspecified` is the one that cannot: it is written as BT.709 (as
    /// the hardware backends write it) and reads back as BT.709.
    #[test]
    fn transfer_codes_round_trip_through_from_h273() {
        for tf in [
            TransferFn::Bt709,
            TransferFn::Bt470Bg,
            TransferFn::Linear,
            TransferFn::St2084,
            TransferFn::AribStdB67,
        ] {
            assert_eq!(TransferFn::from_h273(transfer_to_h273(tf)), tf, "{tf:?}");
        }
        assert_eq!(TransferFn::from_h273(transfer_to_h273(TransferFn::Unspecified)), TransferFn::Bt709);
    }

    /// HDR10 metadata becomes the three codes ffprobe names bt2020 /
    /// smpte2084 / bt2020nc; the SDR default becomes BT.709 limited.
    #[test]
    fn hdr10_metadata_becomes_the_bt2020_pq_codes() {
        let hdr10 = ColorMetadata {
            transfer: TransferFn::St2084,
            matrix_coefficients: 9,
            colour_primaries: 9,
            full_range: false,
            ..ColorMetadata::default()
        };
        let c = colour_description(&hdr10);
        assert_eq!((c.primaries, c.transfer, c.matrix, c.full_range), (9, 16, 9, false));
        let sdr = colour_description(&ColorMetadata::default());
        assert_eq!((sdr.primaries, sdr.transfer, sdr.matrix, sdr.full_range), (1, 1, 1, false));
    }

    /// One coded H.264 access unit for `cm`, 64x64 grey.
    fn first_access_unit(cm: ColorMetadata) -> bytes::Bytes {
        let cfg = EncoderConfig {
            width: 64,
            height: 64,
            frame_rate: 30.0,
            quality: u8::MAX,
            speed_preset: u8::MAX,
            keyframe_interval: 30,
            target: QualityTarget::Standard,
            tier: SpeedTier::Draft,
            threads: 1,
            pixel_format: PixelFormat::Yuv420p,
            color_metadata: cm,
            gpu_index: None,
            gpu_vendor: None,
            codec: VideoCodec::H264,
            constant_qp: false,
            overrides: Default::default(),
        };
        let mut enc = H26xEncoder::new(cfg).expect("encoder");
        let frame = VideoFrame::new(
            vec![128u8; 64 * 64 * 3 / 2].into(),
            64,
            64,
            PixelFormat::Yuv420p,
            ColorSpace::Bt709,
            0,
        );
        enc.send_frame(&frame).expect("frame");
        enc.flush().expect("flush");
        enc.receive_packet().expect("packet").expect("one coded picture").data
    }

    /// The NAL units of one H.264 access unit with `unit_type`, without
    /// their one-byte headers, emulation prevention removed.
    fn nals_of_type(au: &[u8], unit_type: u8) -> Vec<Vec<u8>> {
        h26x::nal::annexb_nals(au)
            .filter(|n| h26x::nal::H264NalHeader::parse(n).map(|h| h.unit_type) == Some(unit_type))
            .map(|n| h26x::nal::unescape_rbsp(&n[1..]))
            .collect()
    }

    /// The colour reaches the stream: encode one frame with a full-range
    /// BT.709 description and read the SPS back with the crate's own
    /// (public) H.264 parser — the reader the decoders use, not a second
    /// bit-level reading here. Both native encoders take the same
    /// `Config`, so a `colour` left `None` fails this for H.265 too.
    #[test]
    fn the_colour_description_is_in_the_sps_the_encoder_writes() {
        let cm = ColorMetadata {
            transfer: TransferFn::Bt709,
            matrix_coefficients: 1,
            colour_primaries: 1,
            full_range: true,
            ..ColorMetadata::default()
        };
        let au = first_access_unit(cm);
        let sps = nals_of_type(&au, 7);
        let [sps] = sps.as_slice() else { panic!("one SPS in the first access unit, got {}", sps.len()) };
        let sps = h26x::h264::Sps::parse(sps).expect("the SPS parses");
        let vui = sps.vui.as_ref().expect("the SPS carries a VUI");
        assert_eq!(vui.colour_description, Some((1, 1, 1)));
        assert!(vui.full_range, "video_full_range_flag");
    }

    /// The HDR10 static metadata reaches the stream as the two SEIs, and
    /// as the bytes x265 writes for the same values (the fixture h26x's
    /// own writer test holds; here it proves the plumbing regroups the ten
    /// integers into the right fields — red into red, max above min).
    /// The crate has no reader for these SEIs; the gate's ffprobe probe
    /// is the reader, and these are the bytes it read as
    /// `red_x=34000/50000 … max_luminance=10000000/10000`.
    #[test]
    fn the_hdr10_static_metadata_is_in_the_seis_the_encoder_writes() {
        let cm = ColorMetadata {
            transfer: TransferFn::St2084,
            matrix_coefficients: 9,
            colour_primaries: 9,
            full_range: false,
            mastering_display: Some(MasteringDisplay {
                primaries_r_x: 34000,
                primaries_r_y: 16000,
                primaries_g_x: 13250,
                primaries_g_y: 34500,
                primaries_b_x: 7500,
                primaries_b_y: 3000,
                white_point_x: 15635,
                white_point_y: 16450,
                max_luminance: 10_000_000,
                min_luminance: 1,
            }),
            content_light_level: Some(ContentLightLevel { max_cll: 1000, max_fall: 400 }),
        };
        let au = first_access_unit(cm);
        let seis: Vec<String> = nals_of_type(&au, 6)
            .iter()
            .map(|n| n.iter().map(|b| format!("{b:02x}")).collect())
            .collect();
        // payloadType 137, 24 bytes (G B R WP, max, min), trailing bits;
        // emulation prevention already removed by `nals_of_type`.
        let mdcv = "891833c286c41d4c0bb884d03e803d134042009896800000000180".to_string();
        // payloadType 144, 4 bytes: 1000, 400.
        let cll = "900403e8019080".to_string();
        assert!(seis.contains(&mdcv), "mastering display SEI {mdcv} not among {seis:?}");
        assert!(seis.contains(&cll), "content light level SEI {cll} not among {seis:?}");
        // And none without the metadata.
        let au = first_access_unit(ColorMetadata::default());
        assert!(nals_of_type(&au, 6).is_empty(), "no SEI for SDR metadata");
    }
}
