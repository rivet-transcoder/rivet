//! Audio filters — per-frame transforms applied to decoded [`AudioFrame`]s
//! **between decode and encode**.
//!
//! The video side's [`crate::filter`] shape, mirrored for audio: a list of
//! [`AudioFilter`] values is the canonical representation, with an
//! ffmpeg-`-filter:a`-style textual serialization that round-trips
//! (`parse_chain(&chain_to_string(c)) == c`).
//!
//! Only filters that change *which* samples go where live here — anything that
//! changes the sample rate is the encoder's job (it resamples to 48 kHz for
//! Opus on its own).
//!
//! ## Why this only applies to transcoded audio
//!
//! A filter has to see PCM, so it runs only when the track is decoded and
//! re-encoded. A passthrough track (AAC / Opus / AC-3 / E-AC-3 copied verbatim)
//! never becomes PCM, so asking for an audio filter on one is an error rather
//! than a silent no-op — see `rivet`'s audio job for where that's enforced.

use std::fmt;

use anyhow::{Result, bail};

use super::AudioFrame;

mod channelmap;
#[cfg(test)]
mod tests;

pub use channelmap::{ChannelLabel, ChannelLayout};

/// One audio-filter step.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum AudioFilter {
    /// **Remap / reorder / select** channels, matching
    /// `ffmpeg -filter:a channelmap=MAP[:LAYOUT]`.
    ///
    /// Each pair says "input channel *X* becomes output channel *Y*". The output
    /// layout is `layout` when given, otherwise it's inferred from the output
    /// labels in the order they were written. Channels the map doesn't mention
    /// are dropped; an output channel no pair feeds is silent.
    ChannelMap {
        /// `(input, output)` speaker positions, in the order written.
        pairs: Vec<(ChannelLabel, ChannelLabel)>,
        /// Explicit output layout (ffmpeg's second argument). `None` = infer
        /// from `pairs`.
        #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
        layout: Option<ChannelLayout>,
    },
}

impl fmt::Display for AudioFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioFilter::ChannelMap { pairs, layout } => {
                write!(f, "channelmap=")?;
                for (i, (src, dst)) in pairs.iter().enumerate() {
                    if i > 0 {
                        f.write_str("|")?;
                    }
                    write!(f, "{src}-{dst}")?;
                }
                if let Some(l) = layout {
                    write!(f, ":{l}")?;
                }
                Ok(())
            }
        }
    }
}

/// A whole chain as a comma-separated textual string (the inverse of
/// [`parse_chain`]).
pub fn chain_to_string(chain: &[AudioFilter]) -> String {
    chain.iter().map(|f| f.to_string()).collect::<Vec<_>>().join(",")
}

/// Parse an ffmpeg-`-filter:a`-style chain, e.g.
/// `"channelmap=FL-FL|FR-FR|FC-FC|LFE-LFE|SL-BL|SR-BR:5.1"`.
pub fn parse_chain(s: &str) -> Result<Vec<AudioFilter>> {
    let mut out = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        out.push(parse_one(part)?);
    }
    if out.is_empty() {
        bail!("empty audio filter chain");
    }
    Ok(out)
}

fn parse_one(spec: &str) -> Result<AudioFilter> {
    let (name, args) = match spec.split_once('=') {
        Some((n, a)) => (n.trim(), a.trim()),
        None => (spec.trim(), ""),
    };
    match name {
        "channelmap" | "channelsplit" => channelmap::parse(args),
        o => bail!("unknown audio filter '{o}' (want channelmap)"),
    }
}

/// Apply a whole chain to one decoded frame, in order.
pub fn apply_chain(frame: &AudioFrame, chain: &[AudioFilter]) -> Result<AudioFrame> {
    let mut f = frame.clone();
    for filter in chain {
        f = apply(&f, filter)?;
    }
    Ok(f)
}

/// Apply one filter to one decoded frame.
pub fn apply(frame: &AudioFrame, filter: &AudioFilter) -> Result<AudioFrame> {
    match filter {
        AudioFilter::ChannelMap { pairs, layout } => channelmap::apply(frame, pairs, layout.as_ref()),
    }
}

/// How many channels a chain produces given `in_channels` on the way in.
///
/// The encoder has to be configured before the first frame arrives, so the
/// channel count has to be knowable from the chain alone — this is that answer,
/// and it's an error (not a guess) if the chain can't accept `in_channels`.
pub fn output_channels(chain: &[AudioFilter], in_channels: u8) -> Result<u8> {
    let mut ch = in_channels;
    for filter in chain {
        ch = match filter {
            AudioFilter::ChannelMap { pairs, layout } => {
                channelmap::output_layout(pairs, layout.as_ref(), ch)?.len() as u8
            }
        };
    }
    Ok(ch)
}
