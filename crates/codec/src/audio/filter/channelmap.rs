//! `channelmap` — remap, reorder, or select audio channels.
//!
//! Mirrors `ffmpeg -filter:a channelmap=MAP[:LAYOUT]`. The map is a `|`-separated
//! list of `IN-OUT` speaker-position pairs; the optional layout names the output.
//!
//! ```text
//! channelmap=FL-FL|FR-FR|FC-FC|LFE-LFE|SL-BL|SR-BR:5.1
//! ```
//!
//! That example is the common one: a 5.1 source whose surrounds are labelled
//! *side* (`SL`/`SR`) re-tagged as the *back* (`BL`/`BR`) positions that MP4 and
//! Opus's channel-mapping family 1 expect.

use std::fmt;
use std::str::FromStr;

use anyhow::{Result, anyhow, bail};

use super::AudioFilter;
use crate::audio::AudioFrame;

/// A speaker position. The names are ffmpeg's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "String", into = "String"))]
pub enum ChannelLabel {
    /// Front left.
    FL,
    /// Front right.
    FR,
    /// Front centre.
    FC,
    /// Low-frequency effects.
    LFE,
    /// Back left.
    BL,
    /// Back right.
    BR,
    /// Back centre.
    BC,
    /// Side left.
    SL,
    /// Side right.
    SR,
}

impl fmt::Display for ChannelLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ChannelLabel::FL => "FL",
            ChannelLabel::FR => "FR",
            ChannelLabel::FC => "FC",
            ChannelLabel::LFE => "LFE",
            ChannelLabel::BL => "BL",
            ChannelLabel::BR => "BR",
            ChannelLabel::BC => "BC",
            ChannelLabel::SL => "SL",
            ChannelLabel::SR => "SR",
        })
    }
}

impl FromStr for ChannelLabel {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_uppercase().as_str() {
            "FL" => ChannelLabel::FL,
            "FR" => ChannelLabel::FR,
            // A mono track's single channel is front-centre; ffmpeg also
            // accepts the bare "MONO" spelling for it.
            "FC" | "MONO" => ChannelLabel::FC,
            "LFE" => ChannelLabel::LFE,
            "BL" => ChannelLabel::BL,
            "BR" => ChannelLabel::BR,
            "BC" => ChannelLabel::BC,
            "SL" => ChannelLabel::SL,
            "SR" => ChannelLabel::SR,
            o => bail!(
                "unknown channel '{o}' (want FL|FR|FC|LFE|BL|BR|BC|SL|SR)"
            ),
        })
    }
}

impl TryFrom<String> for ChannelLabel {
    type Error = anyhow::Error;
    fn try_from(s: String) -> Result<Self> {
        s.parse()
    }
}

impl From<ChannelLabel> for String {
    fn from(c: ChannelLabel) -> String {
        c.to_string()
    }
}

/// An ordered list of speaker positions — what each interleaved sample slot
/// means. Order *is* the layout: slot `i` carries `labels[i]`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "String", into = "String"))]
pub struct ChannelLayout(Vec<ChannelLabel>);

/// The named layouts, in the channel order **RFC 7845 §5.1.1.2** specifies for
/// Opus channel-mapping family 1 — which is also ffmpeg's native order for the
/// same names, and what `crate::audio::encode::opus`'s multistream table
/// assumes on input. Keeping one order across demux → filter → encode is what
/// makes a `channelmap` mean the same thing at every stage.
const NAMED_LAYOUTS: &[(&str, &[ChannelLabel])] = {
    use ChannelLabel::*;
    &[
        ("mono", &[FC]),
        ("stereo", &[FL, FR]),
        ("2.1", &[FL, FR, LFE]),
        ("3.0", &[FL, FR, FC]),
        ("4.0", &[FL, FR, FC, BC]),
        ("quad", &[FL, FR, BL, BR]),
        ("5.0", &[FL, FR, FC, BL, BR]),
        ("5.0(side)", &[FL, FR, FC, SL, SR]),
        ("5.1", &[FL, FR, FC, LFE, BL, BR]),
        ("5.1(side)", &[FL, FR, FC, LFE, SL, SR]),
        ("6.1", &[FL, FR, FC, LFE, BC, SL, SR]),
        ("7.1", &[FL, FR, FC, LFE, BL, BR, SL, SR]),
    ]
};

impl ChannelLayout {
    /// Build a layout from an explicit ordered list. Rejects duplicates — two
    /// slots claiming the same speaker has no meaning.
    pub fn new(labels: Vec<ChannelLabel>) -> Result<Self> {
        if labels.is_empty() {
            bail!("a channel layout needs at least one channel");
        }
        for (i, l) in labels.iter().enumerate() {
            if labels[..i].contains(l) {
                bail!("channel {l} appears twice in the layout");
            }
        }
        Ok(Self(labels))
    }

    /// The **default layout for a channel count** — the RFC 7845 order the Opus
    /// encoder and rivet's demuxers agree on. This is the layout an input track
    /// is assumed to carry when the container doesn't say otherwise.
    pub fn default_for(channels: u8) -> Result<Self> {
        let labels: &[ChannelLabel] = match channels {
            1 => NAMED_LAYOUTS[0].1,  // mono
            2 => NAMED_LAYOUTS[1].1,  // stereo
            3 => NAMED_LAYOUTS[3].1,  // 3.0
            4 => NAMED_LAYOUTS[5].1,  // quad
            5 => NAMED_LAYOUTS[6].1,  // 5.0
            6 => NAMED_LAYOUTS[8].1,  // 5.1
            7 => NAMED_LAYOUTS[10].1, // 6.1
            8 => NAMED_LAYOUTS[11].1, // 7.1
            o => bail!("no default channel layout for {o} channels (1..=8 supported)"),
        };
        Ok(Self(labels.to_vec()))
    }

    /// The channels, in slot order.
    pub fn labels(&self) -> &[ChannelLabel] {
        &self.0
    }

    /// Channel count.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the layout has no channels. Never true for a constructed layout —
    /// [`new`](Self::new) rejects the empty case — but the linter wants it next
    /// to `len`.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Slot index of a speaker position, if the layout has one.
    pub fn index_of(&self, label: ChannelLabel) -> Option<usize> {
        self.0.iter().position(|&l| l == label)
    }
}

impl fmt::Display for ChannelLayout {
    /// The canonical name when the layout is a named one (`5.1`), else the
    /// explicit `FL+FR+…` spelling. Both forms parse back.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some((name, _)) = NAMED_LAYOUTS.iter().find(|(_, l)| *l == self.0.as_slice()) {
            return f.write_str(name);
        }
        let joined =
            self.0.iter().map(|l| l.to_string()).collect::<Vec<_>>().join("+");
        f.write_str(&joined)
    }
}

impl FromStr for ChannelLayout {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let t = s.trim();
        if t.is_empty() {
            bail!("empty channel layout");
        }
        let lower = t.to_ascii_lowercase();
        if let Some((_, labels)) = NAMED_LAYOUTS.iter().find(|(n, _)| *n == lower) {
            return Ok(Self(labels.to_vec()));
        }
        // A bare channel count is the default layout for that count ("6" = 5.1).
        if let Ok(n) = t.parse::<u8>() {
            return Self::default_for(n);
        }
        // Explicit `FL+FR+…`.
        let labels = t
            .split('+')
            .map(|p| p.parse::<ChannelLabel>())
            .collect::<Result<Vec<_>>>()
            .map_err(|e| anyhow!("bad channel layout '{t}': {e}"))?;
        Self::new(labels)
    }
}

impl TryFrom<String> for ChannelLayout {
    type Error = anyhow::Error;
    fn try_from(s: String) -> Result<Self> {
        s.parse()
    }
}

impl From<ChannelLayout> for String {
    fn from(l: ChannelLayout) -> String {
        l.to_string()
    }
}

/// Parse the argument of `channelmap=…`.
pub(super) fn parse(args: &str) -> Result<AudioFilter> {
    if args.is_empty() {
        bail!("channelmap needs a map, e.g. channelmap=FL-FL|FR-FR:stereo");
    }
    // `MAP[:LAYOUT]` — the map itself never contains ':', so the first one
    // splits off the layout.
    let (map, layout) = match args.split_once(':') {
        Some((m, l)) => (m.trim(), Some(l.trim().parse::<ChannelLayout>()?)),
        None => (args.trim(), None),
    };

    let entries: Vec<&str> = map.split('|').map(str::trim).filter(|s| !s.is_empty()).collect();
    if entries.is_empty() {
        bail!("channelmap needs at least one channel pair");
    }

    let mut pairs = Vec::with_capacity(entries.len());
    for (slot, entry) in entries.iter().enumerate() {
        let (src, dst) = match entry.split_once('-') {
            Some((s, d)) => (s.parse::<ChannelLabel>()?, d.parse::<ChannelLabel>()?),
            // Positional form (`channelmap=FR|FL`): the i-th entry names the
            // input channel that feeds output slot i, so the output position
            // comes from the declared layout.
            None => {
                let src = entry.parse::<ChannelLabel>()?;
                let dst = layout
                    .as_ref()
                    .and_then(|l| l.labels().get(slot).copied())
                    .ok_or_else(|| {
                        anyhow!(
                            "channelmap entry '{entry}' has no output channel; write it as \
                             'IN-OUT' or give a layout with at least {} channels",
                            slot + 1
                        )
                    })?;
                (src, dst)
            }
        };
        pairs.push((src, dst));
    }

    for (i, (_, dst)) in pairs.iter().enumerate() {
        if pairs[..i].iter().any(|(_, d)| d == dst) {
            bail!("channelmap writes output channel {dst} twice");
        }
    }

    Ok(AudioFilter::ChannelMap { pairs, layout })
}

/// Work out how the **input** is laid out, given only its channel count and the
/// map the user wrote.
///
/// A container rarely tells us more than "6 channels", and 6 channels is `5.1`
/// with *back* surrounds or `5.1(side)` with *side* ones — the distinction that
/// makes `SL-BL|SR-BR` worth writing at all. So the map is taken as evidence:
/// if the default layout for this channel count doesn't have every channel the
/// map reads, look for the named layout of that width that does. One match is
/// the answer; several is ambiguous and none is a mistake, and both say so.
fn input_layout(
    pairs: &[(ChannelLabel, ChannelLabel)],
    in_channels: u8,
) -> Result<ChannelLayout> {
    let has_all =
        |l: &ChannelLayout| pairs.iter().all(|(src, _)| l.index_of(*src).is_some());

    let default = ChannelLayout::default_for(in_channels)?;
    if has_all(&default) {
        return Ok(default);
    }

    let candidates: Vec<ChannelLayout> = NAMED_LAYOUTS
        .iter()
        .filter(|(_, labels)| labels.len() == in_channels as usize)
        .map(|(_, labels)| ChannelLayout(labels.to_vec()))
        .filter(has_all)
        .collect();

    match candidates.as_slice() {
        [only] => {
            tracing::debug!(
                channels = in_channels,
                layout = %only,
                "channelmap: read the input as {only} (the map names channels {default} lacks)"
            );
            Ok(only.clone())
        }
        [] => {
            let missing: Vec<String> = pairs
                .iter()
                .map(|(s, _)| *s)
                .filter(|s| default.index_of(*s).is_none())
                .map(|s| s.to_string())
                .collect();
            bail!(
                "channelmap reads channel(s) {}, which no {in_channels}-channel layout has \
                 (the default for {in_channels} channels is {default})",
                missing.join(", ")
            )
        }
        many => bail!(
            "channelmap is ambiguous for a {in_channels}-channel input — the channels it reads \
             fit {}; name the input channels unambiguously",
            many.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(" and ")
        ),
    }
}

/// Resolve the output layout for a map: the explicit one when given, else the
/// output labels in the order they were written.
///
/// `in_channels` is validated here too — the map can't read a channel no layout
/// of that width has, and that's worth catching before the encoder is built
/// rather than on the first frame.
pub(super) fn output_layout(
    pairs: &[(ChannelLabel, ChannelLabel)],
    layout: Option<&ChannelLayout>,
    in_channels: u8,
) -> Result<ChannelLayout> {
    input_layout(pairs, in_channels)?;
    match layout {
        Some(l) => Ok(l.clone()),
        None => ChannelLayout::new(pairs.iter().map(|(_, d)| *d).collect()),
    }
}

/// Apply the map to one decoded frame.
pub(super) fn apply(
    frame: &AudioFrame,
    pairs: &[(ChannelLabel, ChannelLabel)],
    layout: Option<&ChannelLayout>,
) -> Result<AudioFrame> {
    let in_ch = frame.channels;
    let input = input_layout(pairs, in_ch)?;
    let output = output_layout(pairs, layout, in_ch)?;
    let out_ch = output.len();

    // Precompute, per output slot, which input slot feeds it. An output channel
    // no pair mentions stays silent rather than picking up a neighbour.
    let routing: Vec<Option<usize>> = output
        .labels()
        .iter()
        .map(|&dst| {
            pairs
                .iter()
                .find(|(_, d)| *d == dst)
                .and_then(|(src, _)| input.index_of(*src))
        })
        .collect();

    let frames = frame.samples.len() / in_ch.max(1) as usize;
    let mut samples = vec![0f32; frames * out_ch];
    for f in 0..frames {
        let src_base = f * in_ch as usize;
        let dst_base = f * out_ch;
        for (slot, route) in routing.iter().enumerate() {
            if let Some(src_slot) = route {
                samples[dst_base + slot] = frame.samples[src_base + src_slot];
            }
        }
    }

    Ok(AudioFrame {
        samples,
        sample_rate: frame.sample_rate,
        channels: out_ch as u8,
        pts: frame.pts,
    })
}
