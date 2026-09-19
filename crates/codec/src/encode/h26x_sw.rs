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
//! 4:2:0 at 8 or 10 bits for both codecs — at 10, little-endian `u16`
//! planes, the pipeline's `yuv420p10le`. H.265 is written as Main / Main 10,
//! H.264 as High / High 10 (`profile_idc` 110, the depth in the SPS's
//! `bit_depth_luma_minus8`). This is the only tier here with 10-bit H.264:
//! no hardware backend has a High 10 encoder (NVENC has no High 10 profile
//! GUID, oneVPL no `AVC High 10`, AMF no 10-bit `Profile`), which is why
//! [`backend_output_caps_for`](super::backend_output_caps_for) reports H.264
//! at 10 bits for this backend alone. Any other format is refused by name
//! rather than narrowed; the pipeline converts its other chroma layouts
//! before the encoder anyway.
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
//! container's `mdcv` / `clli`. The chroma siting (`chroma_sample_loc_type`,
//! which the encoders can also write) is not signalled: the pipeline does
//! not carry one, and a wrong siting costs more than none.
//!
//! # Threads
//!
//! Each encoder runs its own worker pool sized to `threads`, or to the
//! machine when that is zero (`H26X_THREADS` overrides). The pipeline runs a
//! ladder's rungs as separate encoders, so a caller running several at once
//! should hand each a share.
//!
//! # Rate
//!
//! Constant QP — the tuning table's quantiser for the rung's target, or its
//! CRF — unless the rung names a bitrate (`EncodeOverrides::bitrate`). Then
//! the encoder's own rate controller picks a quantiser per picture to spend
//! it (`RateControl::Bitrate`), with the rung's coded picture buffer
//! (`buffer_ms`, which writes the HRD and holds every picture inside it)
//! and, for H.265, its lookahead. Every encoder is a stream of its own and
//! its controller starts from nothing: the whole file on the serial path, a
//! chunk after its lead-in on the chunked one, and one segment on the HLS
//! ladder. A request this tier cannot code — a rate beside a CRF or under
//! `constant_qp`, a buffer without a rate — is refused by name
//! ([`rate_refusal`]); the hardware backends refuse a rate outright.
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

/// The most an H.265 stream from this tier may declare as its NAL HRD rate,
/// and as its buffer: Level 4.0's Main-tier `MaxBR` and `MaxCPB` (12 000
/// kbit/s and 12 000 kbit, H.265 table A.8) times `CpbNalFactor` 1100 / 1000.
/// The h26x encoder writes `general_level_idc` 120 (Level 4.0) into every
/// H.265 stream, so a buffer declared past these is a stream that breaks its
/// own level's limits.
pub const H265_LEVEL4_NAL_LIMIT: u64 = 13_200_000;

/// Why this tier cannot code a rung's rate request, or `None` when it can.
///
/// `crf` is the rung's CRF when it names one, and `constant_qp` whether its
/// encode must be constant-QP (the chunked path's `--seam-mode constqp`). The
/// encoder checks this before building anything, and rivet's spec
/// validation calls it for every rung, so a request that cannot be coded is
/// refused before a frame is decoded, in the same words either way.
pub fn rate_refusal(
    codec: VideoCodec,
    overrides: &super::tuning::EncodeOverrides,
    crf: Option<u8>,
    constant_qp: bool,
) -> Option<String> {
    let named_buffer = overrides.buffer_ms.unwrap_or(0);
    let Some(bps) = overrides.bitrate else {
        let buffer = named_buffer;
        return (buffer > 0).then(|| {
            format!(
                "buffer={buffer}ms names a coded picture buffer, which constrains a rate, and this rung has \
                 no bitrate: name one (`--video-bitrate`, `--rung WxH@RATE` or `bitrate=`) or drop the buffer"
            )
        });
    };
    if bps == 0 {
        return Some("bitrate=0 is not a rate: name a positive bitrate, or none for a quality target".into());
    }
    if codec == VideoCodec::Av1 {
        return Some(format!(
            "a bitrate rung ({bps} bit/s) is coded by the native software H.264 / H.265 encoder only; no AV1 \
             encoder here codes to a bitrate. Encode the rung to a quality target (`--target`), or choose \
             `--codec h264` / `h265`"
        ));
    }
    if let Some(q) = crf {
        return Some(format!(
            "crf={q} names a quantiser and bitrate={bps} a rate, and a rung is coded to one or the other: drop \
             one of them"
        ));
    }
    if constant_qp {
        return Some(format!(
            "`--seam-mode constqp` codes every chunk at a constant quantiser, and this rung is coded to a rate \
             (bitrate={bps}): drop one of them"
        ));
    }
    if codec == VideoCodec::H265 {
        if overrides.lookahead_frames.is_some_and(|n| n > 250) {
            return Some(format!(
                "lookahead={} is above the 250 pictures the native H.265 encoder holds back at most",
                overrides.lookahead_frames.unwrap_or(0)
            ));
        }
        // The buffer the rung will declare: its own, or the table's default.
        let buffer = overrides.buffer_ms.unwrap_or(super::tuning::H26X_SW_BITRATE_BUFFER_MS);
        let which = if overrides.buffer_ms.is_some() { "" } else { " (the default for a bitrate rung)" };
        let buffer_bits = u64::from(bps) * u64::from(buffer) / 1000;
        if buffer > 0 && (u64::from(bps) > H265_LEVEL4_NAL_LIMIT || buffer_bits > H265_LEVEL4_NAL_LIMIT) {
            return Some(format!(
                "bitrate={bps} with buffer={buffer}ms{which} declares a {buffer_bits}-bit buffer at {bps} bit/s, \
                 and the native H.265 encoder labels every stream Level 4.0, whose limits are \
                 {H265_LEVEL4_NAL_LIMIT} bit/s and {H265_LEVEL4_NAL_LIMIT} bits: lower the rate or the buffer, or \
                 declare none (`buffer=0`)"
            ));
        }
    }
    None
}

/// The rate to hand the encoder so that each picture's budget is
/// `bps / frame_rate`.
///
/// **A workaround.** h26x's `Config::fps` is a whole number and its rate
/// controller budgets `bps / fps` per picture, so a 29.97 fps rung coded at
/// `fps` 30 would spend 0.1 % under its target and a 12.5 fps one 4 % under.
/// Scaling the rate by `fps / frame_rate` puts each picture's budget right;
/// a declared buffer then states the scaled rate, consistently with the
/// whole frame rate its VUI timing carries. Remove it once `Config` takes a
/// rational frame rate (on the h26x backlog), when both are exact.
fn encoder_bps(bps: u32, frame_rate: f64, fps: u32) -> u32 {
    if !(frame_rate.is_finite() && frame_rate > 0.0) {
        return bps;
    }
    (f64::from(bps) * f64::from(fps) / frame_rate).round().clamp(1.0, f64::from(u32::MAX)) as u32
}

/// The two encoders behind one face, both boxed. The H.264 encoder is about
/// 26 KB and the H.265 one about 1.2 KB, and clippy's `large_enum_variant`
/// fires on any difference past 200 bytes: boxing only the larger one leaves
/// the other as the large variant against an 8-byte box. An `Inner` is built
/// once per session, so the indirection costs nothing measurable.
enum Inner {
    H264(Box<h26x::encode::h264::H264Encoder>),
    Hevc(Box<h26x::encode::h265::H265Encoder>),
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
    /// The display index of the frame `force_keyframe_next` promised an IDR,
    /// until the encoder is told. See `force_keyframe_next`.
    force_at: Option<u64>,
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
        // 4:2:0 at 8 or 10 bits for both codecs: H.265 Main / Main 10, H.264
        // High / High 10. The encoders pick the profile from the depth; the
        // `u16` planes are the layout both take at 10 bits.
        let bit_depth = match config.pixel_format {
            PixelFormat::Yuv420p => 8,
            PixelFormat::Yuv420p10le => 10,
            other => bail!(
                "the native h26x software encoders take 4:2:0 at 8 bits (yuv420p) or 10 bits \
                 (yuv420p10le); got {other:?}. Convert with the colorspace filter before the \
                 encoder."
            ),
        };

        let p = h26x_sw_params_with(config.codec, config.target, config.tier, &config.overrides);
        // A quadtree depth on H.264 is refused by name, in the words of the
        // knob the caller wrote. The table's H.264 row is 0, so only a
        // `cu_depth=` override gets here.
        if config.codec == VideoCodec::H264 && p.max_cu_depth > 0 {
            bail!(
                "cu_depth={} names an H.265 coding quadtree depth; the native H.264 encoder codes \
                 16x16 macroblocks and has no quadtree. Leave cu_depth unset (or 0) for an H.264 rung.",
                p.max_cu_depth
            );
        }
        let o = &config.overrides;
        let crf = (config.quality != AUTO_FROM_TARGET).then_some(config.quality);
        if let Some(why) = rate_refusal(config.codec, o, crf, config.constant_qp) {
            bail!("{why}");
        }
        // The CRF escape hatch is already in this codec's currency (0..51),
        // and `resolve_overrides` has applied any per-rung delta to it, so it
        // replaces the derived quantiser outright.
        let qp = crf.map_or(p.qp, |q| q.min(51));
        let fps = (config.frame_rate.round() as u32).max(1);

        // Constant QP unless the rung names a rate. A request this tier
        // cannot honour is said, not dropped: a lookahead informs a rate
        // controller, which a constant-QP rung does not have, and the H.264
        // rate controller has no calibrated lookahead (the encoder refuses
        // one by name).
        let (rate, cpb_ms, lookahead) = match p.bitrate {
            None => {
                if o.lookahead_frames.is_some_and(|n| n > 0) {
                    tracing::warn!(
                        lookahead_frames = ?o.lookahead_frames,
                        "this rung is constant-QP: a lookahead informs a rate controller and there \
                         is none here, so the request is ignored (name a bitrate to have one)"
                    );
                }
                (h26x::encode::RateControl::ConstantQp(qp), 0, 0)
            }
            Some(bps) => {
                let lookahead = if config.codec == VideoCodec::H265 {
                    p.lookahead
                } else {
                    if p.lookahead > 0 {
                        tracing::warn!(
                            lookahead = p.lookahead,
                            "the native H.264 rate controller has no calibrated lookahead, so the \
                             request is ignored and the rung is rate-controlled from the past only"
                        );
                    }
                    0
                };
                if o.quality_delta != 0 || o.quality_target.is_some() {
                    tracing::info!(
                        bitrate = bps,
                        quality_delta = o.quality_delta,
                        quality_target = ?o.quality_target,
                        "this rung is coded to a bitrate: its quality target and delta are not consulted"
                    );
                }
                let rate = h26x::encode::RateControl::Bitrate { bps: encoder_bps(bps, config.frame_rate, fps) };
                (rate, p.buffer_ms, lookahead)
            }
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
            // Constant QP, or the rung's rate with its buffer (`cpb_ms`, 0
            // declares none) and lookahead — see the `match` above.
            rate,
            entropy: h26x::encode::Entropy::Cabac,
            transform_8x8: p.transform_8x8,
            subparts: p.subparts,
            sao: p.sao,
            threads,
            fps,
            cpb_ms,
            // The encoders' opt-in tools, both codecs, from the tuning table
            // unless an override names them (`aq=`, `wp=`). Adaptive
            // quantisation is off at every target; weighted prediction is on
            // at every target, measured in docs/codec-encode.md ("Weighted
            // prediction by default"). A tool left off keeps the stream
            // byte-identical to one from an encoder that never had it.
            aq_strength: f32::from(p.aq_strength_tenths) / 10.0,
            lookahead,
            weighted_pred: p.weighted_pred,
            // Always: the pipeline resolved an output colour, and the stream
            // should say it rather than leave the player to assume BT.709
            // (right for SDR, wrong for everything this field exists for).
            colour: Some(colour_description(&config.color_metadata)),
            // Nothing: the pipeline does not carry the chroma siting
            // (`ColorMetadata` has no such field — the decoders here do not
            // report it and the 4:4:4 downsampler's siting is not
            // threaded through), and a wrong siting is worse than none.
            chroma_loc: None,
            // The HDR10 static metadata, when the source carried it: an SEI
            // each in every IDR access unit, beside the container's boxes.
            mastering_display: config.color_metadata.mastering_display.as_ref().map(mastering_display),
            content_light: config.color_metadata.content_light_level.as_ref().map(content_light),
            // Always a number from the tuning table, never `None`: `None` is
            // "whatever the h26x crate's default is this release", and a
            // submodule bump that moves that default would silently change
            // every software H.265 stream's bytes and encode time. H.264's
            // row is 0 — it has no quadtree and refuses a depth above 0.
            max_cu_depth: Some(p.max_cu_depth),
            // Progressive: the pipeline hands the encoder frames, not fields,
            // and carries no field order to code them by. `field_coding`
            // means nothing without `interlace`; `Paff` is the crate's
            // default, spelled out so the literal names every field.
            interlace: None,
            field_coding: h26x::encode::FieldCoding::Paff,
        };

        let inner = Self::build_inner(config.codec, &cfg)?;

        tracing::warn!(
            codec = ?config.codec,
            width = config.width,
            height = config.height,
            bit_depth,
            rate = ?cfg.rate,
            cpb_ms = cfg.cpb_ms,
            lookahead = cfg.lookahead,
            transform_8x8 = p.transform_8x8,
            subparts = p.subparts,
            sao = p.sao,
            aq_strength = cfg.aq_strength,
            weighted_pred = cfg.weighted_pred,
            max_cu_depth = ?cfg.max_cu_depth,
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
            force_at: None,
        })
    }

    fn build_inner(codec: VideoCodec, cfg: &h26x::encode::Config) -> Result<Inner> {
        Ok(match codec {
            VideoCodec::H264 => Inner::H264(Box::new(
                h26x::encode::h264::H264Encoder::new(cfg.clone())
                    .context("the native H.264 encoder rejected the configuration")?,
            )),
            VideoCodec::H265 => Inner::Hevc(Box::new(
                h26x::encode::h265::H265Encoder::new(cfg.clone())
                    .context("the native H.265 encoder rejected the configuration")?,
            )),
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
        // The picture the encoder offers its scheduler while taking this
        // frame is the one `lookahead` frames back; a forced IDR meant for it
        // is handed over now (see `force_keyframe_next`).
        let display = self.pts.len() as u64;
        if self.force_at.is_some() && display.checked_sub(u64::from(self.cfg.lookahead)) == self.force_at {
            self.inner.force_idr();
            self.force_at = None;
        }
        self.pts.push(frame.pts);
        let units = self
            .inner
            .push(&frame.data[..want])
            .with_context(|| format!("the native {:?} encoder refused a frame", self.codec))?;
        self.collect(units)
    }

    fn flush(&mut self) -> Result<()> {
        // A forced IDR still owed goes to the first picture the flush offers,
        // which has to be the one it was promised to.
        if let Some(at) = self.force_at.take() {
            let first = (self.pts.len() as u64).saturating_sub(u64::from(self.cfg.lookahead));
            if at != first {
                bail!(
                    "a keyframe was forced at frame {at}, but the encoder's {}-frame lookahead offers \
                     frame {first} first on flush: fewer frames followed the forced one than the \
                     lookahead holds, and it cannot be placed",
                    self.cfg.lookahead
                );
            }
            self.inner.force_idr();
        }
        let units = self
            .inner
            .flush()
            .with_context(|| format!("the native {:?} encoder failed to flush", self.codec))?;
        self.collect(units)
    }

    fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
        Ok(self.ready.pop_front())
    }

    /// Supported, which matters: the chunked path discards a lead-in and
    /// needs the first kept frame promoted to an IDR, or the chunk will not
    /// stand alone.
    ///
    /// **A workaround for the h26x H.265 encoder under a lookahead.** Its
    /// `force_idr` promotes the next picture *offered to its scheduler*, and
    /// a lookahead of `n` offers each picture `n` frames after it is pushed,
    /// so called here it would land `n` frames early, on a lead-in picture,
    /// and the chunk would open on a picture that predicts from discarded
    /// ones (measured: lookahead 4, keyframe at frame 6 for frame 10). So the
    /// frame is remembered and the encoder told on the push that offers it
    /// (`send_frame`), or on `flush`. Without a lookahead that is the next
    /// push, exactly as before. Remove this once `force_idr` names the next
    /// picture pushed (on the h26x backlog).
    fn force_keyframe_next(&mut self) -> Result<()> {
        self.force_at = Some(self.pts.len() as u64);
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
        self.force_at = None;
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

    /// An encoder for `codec` with `overrides`, handed four 64x64 frames of a
    /// picture that brightens each frame (an IDR, then P pictures), and what
    /// it coded. The left half is flat and the right half textured, so its
    /// blocks differ in luma variance — which is what adaptive quantisation
    /// reads: its offsets are zero-mean over the picture, and a picture whose
    /// blocks all share one variance gets no offset anywhere. Returns the
    /// configuration the adapter built — the one the h26x encoder was
    /// constructed from — and the packets.
    fn encode_with(
        codec: VideoCodec,
        overrides: crate::encode::tuning::EncodeOverrides,
    ) -> (h26x::encode::Config, Vec<bytes::Bytes>) {
        encode_at(codec, SpeedTier::Draft, overrides)
    }

    /// Picture `i` of the four [`encode_with`] codes: 64x64 4:2:0 planes.
    fn test_picture(i: u64) -> Vec<u8> {
        let mut data: Vec<u8> = (0..64 * 64)
            .map(|p| {
                let (x, y) = (p % 64, p / 64);
                ((if x < 24 { 20 } else { ((x ^ y) & 0x1f) * 3 + 90 }) + i as usize * 12) as u8
            })
            .collect();
        data.extend(std::iter::repeat_n(128u8, 2 * 32 * 32));
        data
    }

    /// The 64x64 configuration [`encode_at`] builds its encoder from.
    fn config_at(codec: VideoCodec, tier: SpeedTier, overrides: crate::encode::tuning::EncodeOverrides) -> EncoderConfig {
        EncoderConfig {
            width: 64,
            height: 64,
            frame_rate: 30.0,
            quality: u8::MAX,
            speed_preset: u8::MAX,
            keyframe_interval: 30,
            target: QualityTarget::Standard,
            tier,
            threads: 1,
            pixel_format: PixelFormat::Yuv420p,
            color_metadata: ColorMetadata::default(),
            gpu_index: None,
            gpu_vendor: None,
            codec,
            constant_qp: false,
            overrides,
        }
    }

    /// [`encode_with`] at speed tier `tier`.
    fn encode_at(
        codec: VideoCodec,
        tier: SpeedTier,
        overrides: crate::encode::tuning::EncodeOverrides,
    ) -> (h26x::encode::Config, Vec<bytes::Bytes>) {
        let mut enc = H26xEncoder::new(config_at(codec, tier, overrides)).expect("encoder");
        let mut packets = Vec::new();
        for i in 0..4u64 {
            let frame = VideoFrame::new(test_picture(i).into(), 64, 64, PixelFormat::Yuv420p, ColorSpace::Bt709, i);
            enc.send_frame(&frame).expect("frame");
            while let Some(p) = enc.receive_packet().expect("packet") {
                packets.push(p.data);
            }
        }
        enc.flush().expect("flush");
        while let Some(p) = enc.receive_packet().expect("packet") {
            packets.push(p.data);
        }
        (enc.cfg.clone(), packets)
    }

    /// Every H.265 PPS NAL unit in `packets`, header included.
    fn hevc_pps(packets: &[bytes::Bytes]) -> Vec<Vec<u8>> {
        packets
            .iter()
            .flat_map(|p| {
                h26x::nal::annexb_nals(p)
                    .filter(|n| n.len() > 1 && (n[0] >> 1) & 0x3f == 34)
                    .map(|n| n.to_vec())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// The `aq` / `wp` overrides reach the configuration the h26x encoder is
    /// built from, each on its own, for both codecs, and change the stream.
    /// In H.265 both tools are PPS syntax (`cu_qp_delta_enabled_flag`,
    /// `weighted_pred_flag`), so each one's PPS differs from the knob-off
    /// PPS and from the other's; the crate's H.265 PPS reader is not public,
    /// so that half compares bytes. In H.264 AQ has no switch — it is the
    /// `mb_qp_delta` every coded macroblock already carries — so the H.264
    /// half reads the PPS with the crate's parser for weighted prediction
    /// and compares the coded pictures for AQ. Without an override both
    /// codecs build 0.0 / weighted prediction on (their table rows), so each
    /// half turns weighted prediction off to see it move.
    #[test]
    fn the_opt_in_tools_reach_the_encoder_config_and_the_stream() {
        use crate::encode::tuning::EncodeOverrides;
        let aq = EncodeOverrides { aq_strength_tenths: Some(10), ..Default::default() };
        let wp_off = EncodeOverrides { weighted_pred: Some(false), ..Default::default() };

        let (on_cfg, on) = encode_with(VideoCodec::H265, EncodeOverrides::default());
        let (aq_cfg, aq_out) = encode_with(VideoCodec::H265, aq);
        let (off_cfg, off) = encode_with(VideoCodec::H265, wp_off);
        assert_eq!((on_cfg.aq_strength, on_cfg.weighted_pred, on_cfg.lookahead), (0.0, true, 0), "H.265 table row");
        assert_eq!((aq_cfg.aq_strength, aq_cfg.weighted_pred), (1.0, true), "H.265 aq");
        assert_eq!((off_cfg.aq_strength, off_cfg.weighted_pred), (0.0, false), "H.265 wp=off");

        let (on_pps, aq_pps, off_pps) = (hevc_pps(&on), hevc_pps(&aq_out), hevc_pps(&off));
        assert_eq!(on_pps.len(), 1, "one PPS per stream");
        assert_ne!(aq_pps, on_pps, "aq=1.0 left the PPS as it was");
        assert_ne!(off_pps, on_pps, "wp=off left the PPS as it was");
        assert_ne!(aq_pps, off_pps, "the two tools wrote the same PPS");

        let (on_cfg, on) = encode_with(VideoCodec::H264, EncodeOverrides::default());
        let (aq_cfg, aq_out) = encode_with(VideoCodec::H264, aq);
        let (off_cfg, off) = encode_with(VideoCodec::H264, wp_off);
        assert_eq!((on_cfg.aq_strength, on_cfg.weighted_pred, on_cfg.lookahead), (0.0, true, 0), "H.264 table row");
        assert_eq!((aq_cfg.aq_strength, aq_cfg.weighted_pred), (1.0, true), "H.264 aq");
        assert_eq!((off_cfg.aq_strength, off_cfg.weighted_pred), (0.0, false), "H.264 wp=off");
        let weighted = |packets: &[bytes::Bytes]| -> Vec<bool> {
            let sps: Vec<h26x::h264::Sps> = packets
                .iter()
                .flat_map(|p| nals_of_type(p, 7))
                .map(|s| h26x::h264::Sps::parse(&s).expect("the SPS parses"))
                .collect();
            packets
                .iter()
                .flat_map(|p| nals_of_type(p, 8))
                .map(|pps| {
                    h26x::h264::Pps::parse(&pps, &|_| sps.first().cloned()).expect("the PPS parses").weighted_pred
                })
                .collect()
        };
        // The H.264 encoder repeats its PPS in every access unit here; every
        // copy has to say the same thing.
        let (on_w, off_w) = (weighted(&on), weighted(&off));
        assert!(!on_w.is_empty() && on_w.iter().all(|&w| w), "the table's wp did not reach the H.264 PPS: {on_w:?}");
        assert!(!off_w.is_empty() && off_w.iter().all(|w| !w), "wp=off did not reach the H.264 PPS: {off_w:?}");
        assert_ne!(aq_out, on, "aq=1.0 left the H.264 pictures as they were");
    }

    /// The table's weighted-prediction row reaches the encoder for both codecs
    /// at every tier, with no override. The configuration carries it, the
    /// stream is byte for byte the one an explicit `wp=` naming the same
    /// value writes, and it differs from the stream of the other value (the
    /// PPS's `weighted_pred_flag` alone moves it). So a row that stopped
    /// arriving shows in the bytes as well as in the configuration.
    #[test]
    fn the_tables_weighted_prediction_reaches_both_encoders() {
        use crate::encode::tuning::{EncodeOverrides, h26x_sw_params};
        let wp = |on: bool| EncodeOverrides { weighted_pred: Some(on), ..Default::default() };
        for tier in [SpeedTier::Draft, SpeedTier::Standard, SpeedTier::Archive] {
            for codec in [VideoCodec::H264, VideoCodec::H265] {
                let want = h26x_sw_params(codec, QualityTarget::Standard, tier).weighted_pred;
                let (cfg, packets) = encode_at(codec, tier, EncodeOverrides::default());
                assert_eq!(cfg.weighted_pred, want, "{codec:?} {tier:?}: the configuration does not carry the table's wp");
                let (_, same) = encode_at(codec, tier, wp(want));
                let (_, other) = encode_at(codec, tier, wp(!want));
                assert_eq!(packets, same, "{codec:?} {tier:?}: the default stream is not wp={want}'s stream");
                assert_ne!(packets, other, "{codec:?} {tier:?}: wp={} coded the same stream", !want);
            }
        }
    }

    /// The coding quadtree depth reaches the H.265 encoder as the tuning
    /// table's number at every tier — never `None`, which would be whatever
    /// default the h26x crate has this release. The configuration says so,
    /// and the stream is byte for byte the one an encoder built directly at
    /// the table's depth writes. The content splits (the table's depth and
    /// another write different streams), so a depth that did not arrive
    /// shows in the bytes as well as in the configuration. H.264 carries 0.
    #[test]
    fn the_tables_cu_depth_reaches_the_h265_encoder() {
        use crate::encode::tuning::{EncodeOverrides, h26x_sw_params};
        let direct = |cfg: &h26x::encode::Config, depth: u32| -> Vec<bytes::Bytes> {
            let mut e = h26x::encode::h265::H265Encoder::new(h26x::encode::Config { max_cu_depth: Some(depth), ..cfg.clone() })
                .expect("encoder");
            let mut out = Vec::new();
            for i in 0..4 {
                out.extend(e.push(&test_picture(i)).expect("picture").into_iter().map(|a| bytes::Bytes::from(a.data)));
            }
            out.extend(e.flush().expect("flush").into_iter().map(|a| bytes::Bytes::from(a.data)));
            out
        };
        for tier in [SpeedTier::Draft, SpeedTier::Standard, SpeedTier::Archive] {
            let want = h26x_sw_params(VideoCodec::H265, QualityTarget::Standard, tier).max_cu_depth;
            let (cfg, packets) = encode_at(VideoCodec::H265, tier, EncodeOverrides::default());
            assert_eq!(cfg.max_cu_depth, Some(want), "{tier:?}: the configuration does not carry the table's depth");
            assert_eq!(packets, direct(&cfg, want), "{tier:?}: the stream is not the table depth's stream");
            let other = if want == 0 { 2 } else { 0 };
            assert_ne!(direct(&cfg, want), direct(&cfg, other), "{tier:?}: depth {want} and {other} coded the same stream");
        }
        let (h264, _) = encode_at(VideoCodec::H264, SpeedTier::Archive, EncodeOverrides::default());
        assert_eq!(h264.max_cu_depth, Some(0), "H.264 has no quadtree");
    }

    /// A `cu_depth=` override replaces the table's depth for H.265 at a tier
    /// whose row it is not (0 at `Standard`, 2 at `Draft`): the configuration
    /// carries it and the stream moves with it. An H.264 rung that names a
    /// depth above 0 is refused by name; `cu_depth=0` on H.264 is the H.264
    /// row itself.
    #[test]
    fn a_cu_depth_override_reaches_the_h265_encoder_and_h264_refuses_it() {
        use crate::encode::tuning::EncodeOverrides;
        let depth = |d: u8| EncodeOverrides { cu_depth: Some(d), ..Default::default() };
        let (table, table_out) = encode_at(VideoCodec::H265, SpeedTier::Standard, EncodeOverrides::default());
        let (std0, std0_out) = encode_at(VideoCodec::H265, SpeedTier::Standard, depth(0));
        let (draft2, _) = encode_at(VideoCodec::H265, SpeedTier::Draft, depth(2));
        assert_eq!(table.max_cu_depth, Some(2), "the Standard row");
        assert_eq!(std0.max_cu_depth, Some(0), "cu_depth=0 at Standard");
        assert_eq!(draft2.max_cu_depth, Some(2), "cu_depth=2 at Draft");
        assert_ne!(std0_out, table_out, "cu_depth=0 coded the table depth's stream");

        let err = H26xEncoder::new(config_at(VideoCodec::H264, SpeedTier::Standard, depth(1)))
            .err()
            .expect("an H.264 rung with cu_depth=1 must refuse");
        let msg = format!("{err:#}");
        assert!(msg.contains("cu_depth=1") && msg.contains("H.264"), "{msg}");
        let (h264, _) = encode_at(VideoCodec::H264, SpeedTier::Standard, depth(0));
        assert_eq!(h264.max_cu_depth, Some(0), "cu_depth=0 on H.264");
    }

    /// Width, height and frame rate of the clip [`encode_clip`] codes.
    const CLIP: (u32, u32, f64) = (96, 64, 30.0);

    /// Picture `i` of a clip with somewhere for bits to go: a pseudo-random
    /// texture panning two samples a picture, over a gradient, with chroma
    /// that moves too. Deterministic, so every run codes the same stream.
    fn clip_picture(i: u32) -> Vec<u8> {
        let (w, h, _) = CLIP;
        let hash = |x: u32, y: u32| -> u32 {
            let mut v = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77);
            v ^= v >> 15;
            v.wrapping_mul(0xC2B2_AE3D) >> 24
        };
        let mut data: Vec<u8> =
            (0..w * h).map(|p| (((p % w) + 2 * i) / 4 * 3 + p / w + hash((p % w) + 2 * i, p / w) / 3) as u8).collect();
        let (cw, ch) = (w / 2, h / 2);
        data.extend((0..cw * ch).map(|p| (96 + ((p % cw) + i) % 64) as u8));
        data.extend((0..cw * ch).map(|p| (160 - (p / cw + i) % 48) as u8));
        data
    }

    /// Code `frames` pictures of [`clip_picture`] at `frame_rate` with
    /// `overrides`; the configuration the h26x encoder was built from and
    /// the Annex B stream.
    fn encode_clip(
        codec: VideoCodec,
        overrides: crate::encode::tuning::EncodeOverrides,
        frames: u32,
        frame_rate: f64,
    ) -> (h26x::encode::Config, Vec<u8>) {
        let (w, h, _) = CLIP;
        let cfg = EncoderConfig { width: w, height: h, frame_rate, ..config_at(codec, SpeedTier::Draft, overrides) };
        let mut enc = H26xEncoder::new(cfg).expect("encoder");
        let mut out = Vec::new();
        for i in 0..frames {
            let frame = VideoFrame::new(clip_picture(i).into(), w, h, PixelFormat::Yuv420p, ColorSpace::Bt709, u64::from(i));
            enc.send_frame(&frame).expect("frame");
            while let Some(p) = enc.receive_packet().expect("packet") {
                out.extend_from_slice(&p.data);
            }
        }
        enc.flush().expect("flush");
        while let Some(p) = enc.receive_packet().expect("packet") {
            out.extend_from_slice(&p.data);
        }
        (enc.cfg.clone(), out)
    }

    fn bitrate(bps: u32) -> crate::encode::tuning::EncodeOverrides {
        crate::encode::tuning::EncodeOverrides { bitrate: Some(bps), ..Default::default() }
    }

    /// A rate with `buffer=0`: no coded picture buffer declared.
    fn unbuffered(bps: u32) -> crate::encode::tuning::EncodeOverrides {
        crate::encode::tuning::EncodeOverrides { buffer_ms: Some(0), ..bitrate(bps) }
    }

    /// A rung with no rate is the constant-QP encode it always was — the
    /// table's quantiser, no buffer, no lookahead — for both codecs; a rung
    /// with one reaches the encoder as its rate controller's target.
    #[test]
    fn a_bitrate_reaches_the_encoder_and_none_stays_constant_qp() {
        use crate::encode::tuning::h26x_sw_params;
        for codec in [VideoCodec::H264, VideoCodec::H265] {
            let (cqp, _) = encode_at(codec, SpeedTier::Draft, Default::default());
            let qp = h26x_sw_params(codec, QualityTarget::Standard, SpeedTier::Draft).qp;
            assert_eq!((cqp.rate, cqp.cpb_ms, cqp.lookahead), (h26x::encode::RateControl::ConstantQp(qp), 0, 0), "{codec:?}");
            // A rate gets the table's one-second buffer unless it names its
            // own, and `buffer=0` declares none.
            let (abr, _) = encode_at(codec, SpeedTier::Draft, bitrate(250_000));
            assert_eq!((abr.rate, abr.cpb_ms), (h26x::encode::RateControl::Bitrate { bps: 250_000 }, 1000), "{codec:?}");
            let o = crate::encode::tuning::EncodeOverrides { buffer_ms: Some(500), ..bitrate(250_000) };
            let (buffered, _) = encode_at(codec, SpeedTier::Draft, o);
            assert_eq!(buffered.cpb_ms, 500, "{codec:?}");
            let (none, _) = encode_at(codec, SpeedTier::Draft, unbuffered(250_000));
            assert_eq!(none.cpb_ms, 0, "{codec:?}");
        }
    }

    /// The rate controller is steering: four times the target codes a
    /// clearly larger stream, and each lands within the gate's band of what
    /// it was asked for. A controller that ignored its target would code the
    /// two targets alike.
    #[test]
    fn two_targets_code_ordered_sizes_near_their_targets() {
        let frames = 60;
        let secs = f64::from(frames) / CLIP.2;
        for codec in [VideoCodec::H264, VideoCodec::H265] {
            let achieved = |bps: u32| -> f64 {
                let (_, stream) = encode_clip(codec, bitrate(bps), frames, CLIP.2);
                stream.len() as f64 * 8.0 / secs / f64::from(bps)
            };
            let (low, high) = (achieved(150_000), achieved(600_000));
            let (low_bits, high_bits) = (low * 150_000.0, high * 600_000.0);
            assert!(high_bits > 2.0 * low_bits, "{codec:?}: 600k coded {high_bits:.0} bit/s, 150k {low_bits:.0}");
            for (ratio, target) in [(low, "150k"), (high, "600k")] {
                assert!((0.5..=2.0).contains(&ratio), "{codec:?} at {target}: {ratio:.3} of target");
            }
        }
    }

    /// A buffer is declared, and the stream keeps to it: the crate's HRD
    /// checker, reading only the stream, finds the declared rate and size
    /// (snapped down to what the syntax carries) and no underflow, for both
    /// codecs.
    #[test]
    fn a_buffered_rung_declares_its_buffer_and_conforms() {
        for codec in [VideoCodec::H264, VideoCodec::H265] {
            let o = crate::encode::tuning::EncodeOverrides { buffer_ms: Some(1000), ..bitrate(300_000) };
            let (_, stream) = encode_clip(codec, o, 60, CLIP.2);
            let report = h26x::encode::hrd::verify(&stream).expect("the stream declares an HRD");
            assert_eq!(report.bit_rate, 300_000 / 64 * 64, "{codec:?}: declared rate");
            assert_eq!(report.cpb_size, 300_000 / 16 * 16, "{codec:?}: a one-second buffer");
            assert!(report.conforms(), "{codec:?}: {report:?}");
            let (_, abr) = encode_clip(codec, unbuffered(300_000), 60, CLIP.2);
            assert!(h26x::encode::hrd::verify(&abr).is_err(), "{codec:?}: no buffer, no HRD");
        }
    }

    /// The rate handed to the encoder puts each picture's budget at
    /// `bps / frame_rate`, whatever whole frame rate the encoder was given.
    #[test]
    fn the_encoder_rate_follows_the_real_frame_rate() {
        assert_eq!(encoder_bps(3_000_000, 30.0, 30), 3_000_000);
        assert_eq!(encoder_bps(3_000_000, 25.0, 25), 3_000_000);
        assert_eq!(encoder_bps(3_000_000, 30_000.0 / 1001.0, 30), 3_003_000);
        assert_eq!(encoder_bps(3_000_000, 24_000.0 / 1001.0, 24), 3_003_000);
        assert_eq!(encoder_bps(1_000_000, 12.5, 13), 1_040_000);
        assert_eq!(encoder_bps(1_000_000, 0.0, 1), 1_000_000, "an unknown rate is left alone");
        let (cfg, _) = encode_clip(VideoCodec::H264, bitrate(1_000_000), 2, 30_000.0 / 1001.0);
        assert_eq!((cfg.fps, cfg.rate), (30, h26x::encode::RateControl::Bitrate { bps: 1_001_000 }));
    }

    /// A lookahead reaches an H.265 bitrate rung's encoder; H.264's rate
    /// controller has none, and a constant-QP rung no controller, so both
    /// code with none (and say so) rather than being refused.
    #[test]
    fn a_lookahead_reaches_h265_bitrate_rungs_only() {
        use crate::encode::tuning::EncodeOverrides;
        let la = |o: EncodeOverrides| EncodeOverrides { lookahead_frames: Some(8), ..o };
        let (h265, _) = encode_at(VideoCodec::H265, SpeedTier::Draft, la(bitrate(250_000)));
        let (h264, _) = encode_at(VideoCodec::H264, SpeedTier::Draft, la(bitrate(250_000)));
        let (cqp, _) = encode_at(VideoCodec::H265, SpeedTier::Draft, la(EncodeOverrides::default()));
        assert_eq!((h265.lookahead, h264.lookahead, cqp.lookahead), (8, 0, 0));
    }

    /// `force_keyframe_next` makes the next frame *sent* an IDR, with or
    /// without a lookahead holding pictures back — the chunked path calls it
    /// on the first kept frame after a lead-in and slices by that frame, so
    /// a keyframe landing on a lead-in picture instead leaves the chunk
    /// opening on a picture that predicts from discarded ones.
    #[test]
    fn a_forced_keyframe_lands_on_the_next_frame_sent_under_a_lookahead() {
        use crate::encode::tuning::EncodeOverrides;
        for lookahead in [0u32, 4] {
            let o = EncodeOverrides { lookahead_frames: Some(lookahead), ..bitrate(300_000) };
            let (w, h, fps) = CLIP;
            let cfg = EncoderConfig { width: w, height: h, frame_rate: fps, ..config_at(VideoCodec::H265, SpeedTier::Draft, o) };
            let mut enc = H26xEncoder::new(cfg).expect("encoder");
            let mut keyframes = Vec::new();
            for i in 0..20u32 {
                if i == 10 {
                    enc.force_keyframe_next().expect("force");
                }
                let frame = VideoFrame::new(clip_picture(i).into(), w, h, PixelFormat::Yuv420p, ColorSpace::Bt709, u64::from(i));
                enc.send_frame(&frame).expect("frame");
                while let Some(p) = enc.receive_packet().expect("packet") {
                    if p.is_keyframe {
                        keyframes.push(p.pts);
                    }
                }
            }
            enc.flush().expect("flush");
            while let Some(p) = enc.receive_packet().expect("packet") {
                if p.is_keyframe {
                    keyframes.push(p.pts);
                }
            }
            assert_eq!(keyframes, vec![0, 10], "lookahead {lookahead}: keyframes at {keyframes:?}");
        }
        // Forced within a lookahead of the end, the keyframe has no picture
        // the flush could give it to: an error, not a chunk that is wrong.
        let o = EncodeOverrides { lookahead_frames: Some(4), ..bitrate(300_000) };
        let (w, h, fps) = CLIP;
        let cfg = EncoderConfig { width: w, height: h, frame_rate: fps, ..config_at(VideoCodec::H265, SpeedTier::Draft, o) };
        let mut enc = H26xEncoder::new(cfg).expect("encoder");
        for i in 0..12u32 {
            if i == 10 {
                enc.force_keyframe_next().expect("force");
            }
            let frame = VideoFrame::new(clip_picture(i).into(), w, h, PixelFormat::Yuv420p, ColorSpace::Bt709, u64::from(i));
            enc.send_frame(&frame).expect("frame");
        }
        let err = enc.flush().expect_err("an unplaceable keyframe must be an error");
        assert!(format!("{err:#}").contains("cannot be placed"), "{err:#}");
    }

    /// Every rate request this tier cannot code is refused by name before
    /// the encoder is built, in the words of the knob the caller wrote.
    #[test]
    fn impossible_rate_requests_are_refused_by_name() {
        use crate::encode::tuning::EncodeOverrides;
        let refuse = |cfg: EncoderConfig, words: &[&str]| {
            let err = H26xEncoder::new(cfg).err().expect("must refuse");
            let msg = format!("{err:#}");
            assert!(words.iter().all(|w| msg.contains(w)), "{words:?} not all in: {msg}");
        };
        let at = |codec, o| config_at(codec, SpeedTier::Draft, o);
        refuse(EncoderConfig { quality: 28, ..at(VideoCodec::H264, bitrate(500_000)) }, &["crf=28", "bitrate=500000"]);
        refuse(EncoderConfig { constant_qp: true, ..at(VideoCodec::H265, bitrate(500_000)) }, &["constqp", "bitrate=500000"]);
        refuse(at(VideoCodec::H264, EncodeOverrides { buffer_ms: Some(500), ..Default::default() }), &["buffer=500ms", "no bitrate"]);
        refuse(at(VideoCodec::H265, EncodeOverrides { lookahead_frames: Some(251), ..bitrate(500_000) }), &["lookahead=251"]);
        refuse(
            at(VideoCodec::H265, EncodeOverrides { buffer_ms: Some(1000), ..bitrate(20_000_000) }),
            &["bitrate=20000000", "Level 4.0", "buffer=0"],
        );
        refuse(at(VideoCodec::H265, bitrate(0)), &["bitrate=0"]);
        // The default buffer counts: a rate past the level with no buffer
        // named is refused, saying the buffer was the default.
        refuse(at(VideoCodec::H265, bitrate(20_000_000)), &["buffer=1000ms (the default", "buffer=0"]);
        let av1 = rate_refusal(VideoCodec::Av1, &bitrate(500_000), None, false).expect("AV1 has no rate tier");
        assert!(av1.contains("AV1") && av1.contains("--codec h264"), "{av1}");
        // What stays legal: a large H.265 rate with no buffer declares
        // nothing about the level's HRD limits; a buffer of 0 is no buffer.
        assert_eq!(rate_refusal(VideoCodec::H265, &unbuffered(20_000_000), None, false), None);
        let zero = EncodeOverrides { buffer_ms: Some(0), ..Default::default() };
        assert_eq!(rate_refusal(VideoCodec::H264, &zero, Some(28), true), None);
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
