//! What this build can encode, **per output codec** — the capability half of
//! [`OutputSpec::validate`](super::OutputSpec::validate), and what
//! `rivet capabilities` reports.
//!
//! The codec-agnostic [`codec::encode::build_output_caps`] cannot say that
//! H.264 is 8-bit SDR on NVENC / AMF / QSV but 10-bit HDR on the software
//! `h26x` tier, or that the software AV1 tier (rav1e) is 8-bit while the
//! software H.265 tier is 10-bit. A job has one output codec, so it is checked
//! against the answer for that codec:
//! [`codec::encode::backend_output_caps_for`] over the backends compiled into
//! this build ([`codec::encode::compiled_encode_backends`]).

use anyhow::{Result, bail};
use codec::encode::{EncoderBackend, OutputCaps, backend_output_caps_for};
use codec::frame::VideoCodec;

use super::{BitDepth, ColorPolicy};

/// Every encode backend rivet has, in dispatch-preference order — hardware
/// first, then software; the order `codec::encode::encode_backends` lists the
/// compiled ones in.
pub const ENCODE_BACKENDS: [EncoderBackend; 5] = [
    EncoderBackend::Nvenc,
    EncoderBackend::Amf,
    EncoderBackend::Qsv,
    EncoderBackend::Rav1e,
    EncoderBackend::H26x,
];

/// Every output codec rivet encodes, in `--codec` order.
pub const OUTPUT_CODECS: [VideoCodec; 3] = [VideoCodec::Av1, VideoCodec::H264, VideoCodec::H265];

/// The 8-bit SDR floor every encode path meets.
const EIGHT_BIT_SDR: OutputCaps = OutputCaps {
    max_bit_depth: 8,
    hdr: false,
};

/// The backend's name as `rivet capabilities` and `TRANSCODE_ENCODER_BACKEND`
/// spell it.
pub fn encode_backend_name(backend: EncoderBackend) -> &'static str {
    match backend {
        EncoderBackend::Nvenc => "nvenc",
        EncoderBackend::Amf => "amf",
        EncoderBackend::Qsv => "qsv",
        EncoderBackend::H26x => "h26x",
        EncoderBackend::Rav1e => "rav1e",
    }
}

/// The cargo feature that puts `backend` in the dispatch chain.
pub fn encode_backend_feature(backend: EncoderBackend) -> &'static str {
    match backend {
        EncoderBackend::Nvenc => "nvidia",
        EncoderBackend::Amf => "amd",
        EncoderBackend::Qsv => "qsv",
        EncoderBackend::H26x => "h26x-fallback",
        EncoderBackend::Rav1e => "rav1e-fallback",
    }
}

fn is_hardware(backend: EncoderBackend) -> bool {
    matches!(
        backend,
        EncoderBackend::Nvenc | EncoderBackend::Amf | EncoderBackend::Qsv
    )
}

/// Whether `backend` encodes `codec` at all: the hardware backends serve every
/// output codec, rav1e AV1 only, h26x H.264 / H.265 only.
pub fn encode_backend_serves(backend: EncoderBackend, codec: VideoCodec) -> bool {
    match backend {
        EncoderBackend::Nvenc | EncoderBackend::Amf | EncoderBackend::Qsv => true,
        EncoderBackend::Rav1e => codec == VideoCodec::Av1,
        EncoderBackend::H26x => matches!(codec, VideoCodec::H264 | VideoCodec::H265),
    }
}

/// The codec's name as `--codec` spells it.
pub fn output_codec_label(codec: VideoCodec) -> &'static str {
    match codec {
        VideoCodec::Av1 => "av1",
        VideoCodec::H264 => "h264",
        VideoCodec::H265 => "h265",
    }
}

/// `"10-bit HDR"` / `"8-bit SDR"`.
pub fn output_caps_label(caps: OutputCaps) -> String {
    format!(
        "{}-bit {}",
        caps.max_bit_depth,
        if caps.hdr { "HDR" } else { "SDR" }
    )
}

/// One output codec's capabilities over a set of encode backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecOutputCaps {
    /// The output codec.
    pub codec: VideoCodec,
    /// The best of each capability over [`Self::backends`], from the 8-bit SDR
    /// floor. For this build's set it is what
    /// [`codec::encode::build_output_caps_for`] says.
    pub caps: OutputCaps,
    /// Each backend of the set that encodes `codec`, with its capabilities for
    /// it, in the set's order. Empty when none does.
    pub backends: Vec<(EncoderBackend, OutputCaps)>,
}

impl CodecOutputCaps {
    /// `codec`'s capabilities over `backends`.
    pub fn over(codec: VideoCodec, backends: &[EncoderBackend]) -> Self {
        let backends: Vec<(EncoderBackend, OutputCaps)> = backends
            .iter()
            .copied()
            .filter(|&b| encode_backend_serves(b, codec))
            .map(|b| (b, backend_output_caps_for(b, codec)))
            .collect();
        let caps = backends
            .iter()
            .fold(EIGHT_BIT_SDR, |acc, (_, c)| OutputCaps {
                max_bit_depth: acc.max_bit_depth.max(c.max_bit_depth),
                hdr: acc.hdr || c.hdr,
            });
        Self {
            codec,
            caps,
            backends,
        }
    }

    /// `codec`'s capabilities on this build, over
    /// [`codec::encode::compiled_encode_backends`].
    pub fn of_this_build(codec: VideoCodec) -> Self {
        Self::over(codec, &codec::encode::compiled_encode_backends())
    }
}

/// Refuse an output policy that `backends` cannot encode for `codec`.
///
/// Ten bits (an HDR colour policy, or a forced 10-bit depth) needs a backend
/// whose `codec` encoder is 10-bit; HDR needs one that signals it. The error
/// says what the set has for `codec`, and which backends would serve the
/// request by the feature that compiles them in (and, for AV1, the silicon).
/// Only those two are checked: an 8-bit SDR policy passes on any set, an empty
/// one included — whether the build has an encoder for the codec at all is
/// found out when the job builds one, as it always was.
pub(crate) fn check_output_caps(
    color: ColorPolicy,
    bit_depth: BitDepth,
    codec: VideoCodec,
    backends: &[EncoderBackend],
) -> Result<()> {
    let have = CodecOutputCaps::over(codec, backends);
    let needs_10bit = color.is_hdr() || matches!(bit_depth, BitDepth::TenBit);
    if needs_10bit && have.caps.max_bit_depth < 10 {
        bail!(
            "{}",
            refusal(&have, "at 10 bits", color, bit_depth, |c| c.max_bit_depth
                >= 10)
        );
    }
    if color.is_hdr() && !have.caps.hdr {
        bail!(
            "{}",
            refusal(&have, "with HDR", color, bit_depth, |c| c.hdr)
        );
    }
    Ok(())
}

/// The silicon a hardware backend needs for `codec`, where rivet knows it is
/// narrower than "the vendor's encoder".
fn hardware_silicon_for(codec: VideoCodec) -> Option<&'static str> {
    match codec {
        VideoCodec::Av1 => {
            Some("on a GPU with AV1 encode: NVIDIA Ada+, AMD RDNA3+, Intel Arc / Meteor Lake+")
        }
        VideoCodec::H264 | VideoCodec::H265 => None,
    }
}

/// `a`, `a or b`, `a, b or c`.
fn or_list(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [one] => one.clone(),
        [init @ .., last] => format!("{} or {last}", init.join(", ")),
    }
}

fn refusal(
    have: &CodecOutputCaps,
    what: &str,
    color: ColorPolicy,
    bit_depth: BitDepth,
    meets: impl Fn(OutputCaps) -> bool,
) -> String {
    let codec = output_codec_label(have.codec);
    let has = if have.backends.is_empty() {
        format!("this build has no {codec} encoder")
    } else {
        let list: Vec<String> = have
            .backends
            .iter()
            .map(|&(b, c)| format!("{} ({})", encode_backend_name(b), output_caps_label(c)))
            .collect();
        format!("this build encodes {codec} with {}", list.join(", "))
    };
    let features = |bs: &[EncoderBackend]| -> String {
        let names: Vec<String> = bs
            .iter()
            .map(|&b| format!("`{}`", encode_backend_feature(b)))
            .collect();
        or_list(&names)
    };
    let serving: Vec<EncoderBackend> = ENCODE_BACKENDS
        .iter()
        .copied()
        .filter(|&b| encode_backend_serves(b, have.codec))
        .collect();
    let (able, short): (Vec<EncoderBackend>, Vec<EncoderBackend>) = serving
        .iter()
        .copied()
        .partition(|&b| meets(backend_output_caps_for(b, have.codec)));
    let (hardware, software): (Vec<EncoderBackend>, Vec<EncoderBackend>) =
        able.iter().copied().partition(|&b| is_hardware(b));

    let mut msg = format!(
        "{codec} {what} (color={color:?}, bit_depth={bit_depth:?}) cannot be encoded: {has}. "
    );
    let mut needs: Vec<String> = Vec::new();
    if !hardware.is_empty() {
        let silicon = hardware_silicon_for(have.codec)
            .map(|s| format!(", {s}"))
            .unwrap_or_default();
        needs.push(format!(
            "a hardware encoder (build with {}{silicon})",
            features(&hardware)
        ));
    }
    if !software.is_empty() {
        needs.push(format!(
            "the software tier (build with {})",
            features(&software)
        ));
    }
    if needs.is_empty() {
        msg.push_str(&format!("No encoder in rivet produces {codec} {what}"));
        return msg;
    }
    msg.push_str(&format!("{codec} {what} needs {}", needs.join(" or ")));
    if hardware.is_empty() {
        msg.push_str(&format!("; no hardware backend encodes {codec} {what}"));
    }
    for b in short.into_iter().filter(|&b| !is_hardware(b)) {
        msg.push_str(&format!(
            "; the software {codec} tier (`{}`) is {}",
            encode_backend_feature(b),
            output_caps_label(backend_output_caps_for(b, have.codec))
        ));
    }
    msg
}
