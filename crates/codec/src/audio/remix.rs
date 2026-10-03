//! Channel-layout conversion of decoded PCM: the downmix matrices, and the
//! layout each output codec carries a source in.
//!
//! [`remix_matrix`] builds the coefficients that take one [`ChannelLayout`]
//! to another. It is a downmix and a relabel, never an upmix: an output
//! speaker the input has nothing for is silent. The coefficients are ITU-R
//! BS.775's (Table 2) — a centre goes to the front pair at −3 dB, a surround
//! to its front side at −3 dB — with the conventions ffmpeg's `swresample`
//! uses where BS.775 has nothing to say:
//!
//! - the **LFE is dropped** when the output has none, as BS.775 and A/52's
//!   own downmix (§7.8) do; a low-frequency channel folded into full-range
//!   speakers is louder than its mix intended;
//! - a **side** and a **back** pair are the same surrounds when the other
//!   side has only one of them (5.1(side) → 5.1 is a relabel, gain 1), and
//!   fold into each other at −3 dB when the input has both (7.1 → 5.1);
//! - a **back centre** splits into a surround pair at −3 dB, or into the
//!   fronts at −6 dB when there is no surround;
//! - **mono** is the stereo downmix folded at −3 dB per side.
//!
//! The matrix is then **normalised so no output can clip**: when any output
//! channel's coefficients sum (in magnitude) to more than 1, every
//! coefficient is divided by the largest such sum, which keeps the balance
//! between channels. 5.1 → stereo therefore comes out as
//! `L = 0.414·FL + 0.293·FC + 0.293·SL` (and the mirror for R), −7.7 dB on
//! the fronts: that is the price of a full-scale centre and surround never
//! clipping, and what ffmpeg's `-ac 2` produces too.

use anyhow::{Result, bail};

use super::AudioFrame;
use super::filter::{ChannelLabel, ChannelLayout};

use ChannelLabel::*;

/// −3 dB.
const HALF_POWER: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// The matrix taking `from` to `to`, row-major: `to.len()` rows of
/// `from.len()` coefficients, `out[o] = Σ m[o][i] · in[i]`.
pub fn remix_matrix(from: &ChannelLayout, to: &ChannelLayout) -> Vec<f32> {
    let mono = ChannelLayout::named("mono");
    let mut m = if *to == mono && *from != mono {
        // Mono is the stereo downmix folded: build that, then sum its rows
        // at −3 dB each.
        let stereo = ChannelLayout::named("stereo");
        let s = route_all(from, &stereo);
        (0..from.len()).map(|i| HALF_POWER * (s[i] + s[from.len() + i])).collect()
    } else {
        route_all(from, to)
    };
    let widest = m
        .chunks(from.len())
        .map(|row| row.iter().map(|c| c.abs()).sum::<f32>())
        .fold(0.0f32, f32::max);
    if widest > 1.0 + 1e-6 {
        m.iter_mut().for_each(|c| *c /= widest);
    }
    m
}

/// Every input channel routed into `to`, unnormalised.
fn route_all(from: &ChannelLayout, to: &ChannelLayout) -> Vec<f32> {
    let mut m = vec![0.0f32; to.len() * from.len()];
    for (i, &label) in from.labels().iter().enumerate() {
        for (o, gain) in route(label, from, to) {
            m[o * from.len() + i] += gain;
        }
    }
    m
}

/// Where one input speaker goes in `to`, and at what gain.
fn route(label: ChannelLabel, from: &ChannelLayout, to: &ChannelLayout) -> Vec<(usize, f32)> {
    let at = |l: ChannelLabel| to.index_of(l);
    if let Some(o) = at(label) {
        return vec![(o, 1.0)];
    }
    // The fronts pair, when the output has it.
    let fronts = || at(FL).zip(at(FR));
    match label {
        FC => fronts().map_or_else(Vec::new, |(l, r)| vec![(l, HALF_POWER), (r, HALF_POWER)]),
        // A front the output lacks (it has a centre only): −3 dB into it.
        FL | FR => at(FC).map_or_else(Vec::new, |c| vec![(c, HALF_POWER)]),
        LFE => Vec::new(),
        BL | BR | SL | SR => {
            let left = matches!(label, BL | SL);
            // The same surround under the other name: a relabel when the input
            // has only this pair, a −3 dB fold when it has both.
            let twin = match label {
                BL => SL,
                BR => SR,
                SL => BL,
                _ => BR,
            };
            if let Some(o) = at(twin) {
                return vec![(o, if from.has(twin) { HALF_POWER } else { 1.0 })];
            }
            if let Some(o) = at(BC) {
                return vec![(o, HALF_POWER)];
            }
            at(if left { FL } else { FR }).map_or_else(Vec::new, |o| vec![(o, HALF_POWER)])
        }
        BC => {
            let pair = at(BL).zip(at(BR)).or_else(|| at(SL).zip(at(SR)));
            match (pair, fronts()) {
                (Some((l, r)), _) => vec![(l, HALF_POWER), (r, HALF_POWER)],
                (None, Some((l, r))) => vec![(l, 0.5), (r, 0.5)],
                (None, None) => Vec::new(),
            }
        }
    }
}

/// Converts frames from one layout to another with [`remix_matrix`].
#[derive(Debug, Clone)]
pub struct Remixer {
    from: ChannelLayout,
    to: ChannelLayout,
    matrix: Vec<f32>,
}

impl Remixer {
    pub fn new(from: ChannelLayout, to: ChannelLayout) -> Self {
        let matrix = remix_matrix(&from, &to);
        Self { from, to, matrix }
    }

    pub fn from(&self) -> &ChannelLayout {
        &self.from
    }

    pub fn to(&self) -> &ChannelLayout {
        &self.to
    }

    /// Whether the conversion leaves every sample where it is.
    pub fn is_identity(&self) -> bool {
        self.from == self.to
    }

    /// One frame in `from`'s layout, in `to`'s. A frame whose channel count is
    /// not `from`'s is refused rather than read as some other layout.
    pub fn apply(&self, frame: &AudioFrame) -> Result<AudioFrame> {
        let (inn, out) = (self.from.len(), self.to.len());
        if usize::from(frame.channels) != inn {
            bail!(
                "remix {} → {}: the frame has {} channels, not {inn}",
                self.from,
                self.to,
                frame.channels
            );
        }
        if self.is_identity() {
            return Ok(frame.clone());
        }
        let frames = frame.samples.len() / inn;
        let mut samples = vec![0.0f32; frames * out];
        for (src, dst) in frame.samples.chunks_exact(inn).zip(samples.chunks_exact_mut(out)) {
            for (o, d) in dst.iter_mut().enumerate() {
                let row = &self.matrix[o * inn..(o + 1) * inn];
                *d = row.iter().zip(src).map(|(c, s)| c * s).sum();
            }
        }
        Ok(AudioFrame { samples, sample_rate: frame.sample_rate, channels: out as u8, pts: frame.pts })
    }
}

/// The Opus channel-mapping family 1 layouts (RFC 7845 §5.1.1.2), by channel
/// count, in the pipeline's order. The encoder permutes into the RFC's.
const OPUS_LAYOUTS: [&str; 8] = ["mono", "stereo", "3.0", "quad", "5.0", "5.1", "6.1", "7.1"];

/// The layout Opus carries `source` in: the source's own when family 0 or 1
/// has it, else the narrowest one that has a place for every speaker the
/// source has — its side pair as the back pair (or the reverse), a back
/// centre split into the surround pair — with the speakers the source lacks
/// left silent. 2.1 goes out as 5.1 with a silent centre and surrounds, 4.0
/// as 5.0 with its back centre in both surrounds: the layout changes shape,
/// no content is made up, and the LFE is never folded away.
pub fn opus_layout(source: &ChannelLayout) -> Option<ChannelLayout> {
    OPUS_LAYOUTS.iter().map(|n| ChannelLayout::named(n)).find(|candidate| {
        candidate.len() >= source.len()
            && source.labels().iter().all(|&l| carries(candidate, source, l))
    })
}

/// Whether `to` has a place for `from`'s speaker `label` without a downmix.
fn carries(to: &ChannelLayout, from: &ChannelLayout, label: ChannelLabel) -> bool {
    let twin = match label {
        BL => Some(SL),
        BR => Some(SR),
        SL => Some(BL),
        SR => Some(BR),
        _ => None,
    };
    to.has(label)
        || twin.is_some_and(|t| to.has(t) && !from.has(t))
        || (label == BC && ((to.has(BL) && to.has(BR)) || (to.has(SL) && to.has(SR))))
}

/// The channel configurations of AAC (ISO/IEC 13818-7 Table 42) as the
/// named layouts the AAC encoder takes, in the pipeline's order: 3.0 is
/// configuration 3, 4.0 (a back centre) 4, 5.0 5, 5.1 6, 7.1 7.
const AAC_LAYOUTS: [&str; 7] = ["mono", "stereo", "3.0", "4.0", "5.0", "5.1", "7.1"];

/// The layout AAC carries `source` in, found as for Opus
/// ([`opus_layout`]): the source's own when a channel configuration has it,
/// else the narrowest one with a place for every speaker the source has, the
/// rest silent. Quad goes out as 5.0 (a silent centre), 2.1 as 5.1, 6.1 as
/// 7.1 with its back centre in both back channels.
pub fn aac_layout(source: &ChannelLayout) -> Option<ChannelLayout> {
    AAC_LAYOUTS.iter().map(|n| ChannelLayout::named(n)).find(|candidate| {
        candidate.len() >= source.len()
            && source.labels().iter().all(|&l| carries(candidate, source, l))
    })
}

/// The layout MP3 carries `source` in: mono and stereo as they are,
/// everything else downmixed to stereo (MPEG-1 Layer III has two channels at
/// most).
pub fn mp3_layout(source: &ChannelLayout) -> ChannelLayout {
    let mono = ChannelLayout::named("mono");
    if *source == mono { mono } else { ChannelLayout::named("stereo") }
}

/// The channel arrangements of AC-3 / E-AC-3 (A/52 `acmod` 1/0 to 3/2, each
/// with or without the LFE) and of the DTS core (`AMODE` 0, 2 and 5–9, the
/// same arrangements), as named layouts in the pipeline's order and from
/// the narrowest: `3.0(back)` is 2/1, `4.0` 3/1, `quad(side)` 2/2,
/// `5.1(side)` 3/2 with the LFE.
const SURROUND_CORE_LAYOUTS: [&str; 11] =
    ["mono", "stereo", "2.1", "3.0", "3.0(back)", "3.1", "4.0", "quad(side)", "4.1", "5.0(side)", "5.1(side)"];

/// The layout AC-3, E-AC-3 and DTS carry `source` in, found as for Opus
/// ([`opus_layout`]): the source's own when an arrangement has it, else the
/// narrowest with a place for every speaker the source has (quad as
/// quad(side), 5.1 as 5.1(side), 6.1's back centre split into the side
/// pair). A source wider than 5.1 that none of them carries (7.1) is
/// downmixed to 5.1(side).
pub fn surround_core_layout(source: &ChannelLayout) -> ChannelLayout {
    SURROUND_CORE_LAYOUTS
        .iter()
        .map(|n| ChannelLayout::named(n))
        .find(|candidate| {
            candidate.len() >= source.len() && source.labels().iter().all(|&l| carries(candidate, source, l))
        })
        .unwrap_or_else(|| ChannelLayout::named("5.1(side)"))
}

/// The layout E-AC-3 carries `source` in: [`surround_core_layout`]'s, but
/// a source wider than 5.1 with a place in 7.1 (FL FR FC LFE BL BR SL SR —
/// the independent substream's 3/2 and LFE, and a dependent substream on
/// the back surrounds) stays 7.1 rather than being downmixed.
pub fn eac3_layout(source: &ChannelLayout) -> ChannelLayout {
    let core = surround_core_layout(source);
    let seven_one = ChannelLayout::named("7.1");
    let downmixed = !source.labels().iter().all(|&l| carries(&core, source, l));
    if downmixed && source.labels().iter().all(|&l| carries(&seven_one, source, l)) { seven_one } else { core }
}

/// The layout Vorbis carries `source` in: Vorbis I §4.3.9 defines the same
/// eight arrangements as Opus channel-mapping family 1, so [`opus_layout`]'s.
pub fn vorbis_layout(source: &ChannelLayout) -> Option<ChannelLayout> {
    opus_layout(source)
}

/// The layout HE-AAC v2 carries `source` in: stereo, a mono source spread to
/// both sides and a wider one downmixed (parametric stereo codes a stereo
/// image and nothing else).
pub fn he_aac_v2_layout(_source: &ChannelLayout) -> ChannelLayout {
    ChannelLayout::named("stereo")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(name: &str) -> ChannelLayout {
        name.parse().unwrap()
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    /// Coefficient taking input `i` to output `o`.
    fn coef(m: &[f32], from: &ChannelLayout, to: &ChannelLayout, i: ChannelLabel, o: ChannelLabel) -> f32 {
        m[to.index_of(o).unwrap() * from.len() + from.index_of(i).unwrap()]
    }

    #[test]
    fn aac_carries_every_layout_in_a_channel_configuration() {
        for (src, out) in [
            ("mono", "mono"),
            ("stereo", "stereo"),
            ("3.0", "3.0"),
            ("4.0", "4.0"),
            ("quad", "5.0"),
            ("2.1", "5.1"),
            ("5.0", "5.0"),
            ("5.0(side)", "5.0"),
            ("5.1", "5.1"),
            ("5.1(side)", "5.1"),
            ("6.1", "7.1"),
            ("7.1", "7.1"),
        ] {
            assert_eq!(aac_layout(&layout(src)), Some(layout(out)), "{src}");
        }
    }

    #[test]
    fn five_one_to_stereo_is_bs775_normalised() {
        for src in ["5.1", "5.1(side)"] {
            let (from, to) = (layout(src), layout("stereo"));
            let m = remix_matrix(&from, &to);
            let sur = if src == "5.1" { (BL, BR) } else { (SL, SR) };
            // BS.775: L = FL + 0.707 FC + 0.707 Ls, over its sum 2.414.
            let n = 1.0 + 2.0 * HALF_POWER;
            assert!(close(coef(&m, &from, &to, FL, FL), 1.0 / n));
            assert!(close(coef(&m, &from, &to, FC, FL), HALF_POWER / n));
            assert!(close(coef(&m, &from, &to, FC, FR), HALF_POWER / n));
            assert!(close(coef(&m, &from, &to, sur.0, FL), HALF_POWER / n));
            assert!(close(coef(&m, &from, &to, sur.1, FR), HALF_POWER / n));
            assert_eq!(coef(&m, &from, &to, FL, FR), 0.0, "no crosstalk");
            assert_eq!(coef(&m, &from, &to, sur.0, FR), 0.0);
            assert_eq!(coef(&m, &from, &to, LFE, FL), 0.0, "the LFE is dropped");
            assert_eq!(coef(&m, &from, &to, LFE, FR), 0.0);
            // No row can clip: full scale on every input sums to at most 1.
            for row in m.chunks(from.len()) {
                assert!(row.iter().sum::<f32>() <= 1.0 + 1e-6);
            }
        }
    }

    #[test]
    fn stereo_and_five_one_fold_to_mono() {
        let (from, to) = (layout("stereo"), layout("mono"));
        assert_eq!(remix_matrix(&from, &to), vec![0.5, 0.5]);
        let (from, to) = (layout("5.1"), layout("mono"));
        let m = remix_matrix(&from, &to);
        // 0.707·(L + R) + C + 0.5·(Ls + Rs), over 3.414.
        let n = 2.0 * HALF_POWER + 1.0 + 1.0;
        assert!(close(coef(&m, &from, &to, FL, FC), HALF_POWER / n));
        assert!(close(coef(&m, &from, &to, FC, FC), 1.0 / n));
        assert!(close(coef(&m, &from, &to, BL, FC), 0.5 / n));
        assert_eq!(coef(&m, &from, &to, LFE, FC), 0.0);
        assert!(close(m.iter().sum::<f32>(), 1.0));
    }

    #[test]
    fn side_and_back_surrounds_relabel_or_fold() {
        // 5.1(side) → 5.1: the surrounds move slots at unity; nothing else.
        let (from, to) = (layout("5.1(side)"), layout("5.1"));
        let m = remix_matrix(&from, &to);
        assert_eq!(coef(&m, &from, &to, SL, BL), 1.0);
        assert_eq!(coef(&m, &from, &to, SR, BR), 1.0);
        assert_eq!(coef(&m, &from, &to, FC, FC), 1.0);
        assert_eq!(m.iter().filter(|&&c| c != 0.0).count(), 6, "a permutation");
        // 7.1 → 5.1: both pairs into the back pair at −3 dB, normalised.
        let (from, to) = (layout("7.1"), layout("5.1"));
        let m = remix_matrix(&from, &to);
        let n = 1.0 + HALF_POWER;
        assert!(close(coef(&m, &from, &to, BL, BL), 1.0 / n));
        assert!(close(coef(&m, &from, &to, SL, BL), HALF_POWER / n));
        assert!(close(coef(&m, &from, &to, FL, FL), 1.0 / n));
        assert!(close(coef(&m, &from, &to, LFE, LFE), 1.0 / n), "kept when the output has one");
    }

    #[test]
    fn a_back_centre_splits_into_the_surrounds_or_the_fronts() {
        let (from, to) = (layout("4.0"), layout("5.0"));
        let m = remix_matrix(&from, &to);
        assert!(close(coef(&m, &from, &to, BC, BL), HALF_POWER));
        assert!(close(coef(&m, &from, &to, BC, BR), HALF_POWER));
        let (from, to) = (layout("3.0(back)"), layout("stereo"));
        let m = remix_matrix(&from, &to);
        assert!(close(coef(&m, &from, &to, BC, FL), 0.5 / 1.5));
        assert!(close(coef(&m, &from, &to, FL, FL), 1.0 / 1.5));
    }

    #[test]
    fn remixing_a_frame_moves_the_samples() {
        let r = Remixer::new(layout("5.1"), layout("stereo"));
        // One frame: only the centre at full scale.
        let frame = AudioFrame { samples: vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0], sample_rate: 48_000, channels: 6, pts: 7 };
        let out = r.apply(&frame).unwrap();
        assert_eq!((out.channels, out.pts, out.samples.len()), (2, 7, 2));
        let c = HALF_POWER / (1.0 + 2.0 * HALF_POWER);
        assert!(close(out.samples[0], c) && close(out.samples[1], c));
        let stereo = AudioFrame { samples: vec![0.1, 0.2], sample_rate: 48_000, channels: 2, pts: 0 };
        assert!(r.apply(&stereo).is_err(), "a frame of the wrong width is refused");
        assert!(Remixer::new(layout("stereo"), layout("stereo")).is_identity());
    }

    #[test]
    fn opus_carries_each_source_layout_without_a_downmix() {
        for (src, opus) in [
            ("mono", "mono"),
            ("stereo", "stereo"),
            ("2.1", "5.1"),
            ("3.0", "3.0"),
            ("3.0(back)", "quad"),
            ("3.1", "5.1"),
            ("4.0", "5.0"),
            ("quad", "quad"),
            ("quad(side)", "quad"),
            ("4.1", "5.1"),
            ("5.0", "5.0"),
            ("5.0(side)", "5.0"),
            ("5.1", "5.1"),
            ("5.1(side)", "5.1"),
            ("6.1", "6.1"),
            ("7.1", "7.1"),
        ] {
            let got = opus_layout(&layout(src)).unwrap();
            assert_eq!(got, layout(opus), "{src}");
            // And the matrix to it moves every source speaker at unity or a
            // −3 dB split, never folding one into another's slot.
            let m = remix_matrix(&layout(src), &got);
            for (i, _) in layout(src).labels().iter().enumerate() {
                let column: f32 = (0..got.len()).map(|o| m[o * layout(src).len() + i].powi(2)).sum();
                assert!(close(column, 1.0), "{src}: input {i} keeps its power ({column})");
            }
        }
    }

    #[test]
    fn mp3_is_mono_or_stereo() {
        assert_eq!(mp3_layout(&layout("mono")), layout("mono"));
        assert_eq!(mp3_layout(&layout("stereo")), layout("stereo"));
        assert_eq!(mp3_layout(&layout("5.1(side)")), layout("stereo"));
    }

    #[test]
    fn ac3_and_dts_carry_their_arrangements_and_downmix_the_rest() {
        for (src, out) in [
            ("mono", "mono"),
            ("stereo", "stereo"),
            ("2.1", "2.1"),
            ("3.0", "3.0"),
            ("3.0(back)", "3.0(back)"),
            ("3.1", "3.1"),
            ("4.0", "4.0"),
            ("quad", "quad(side)"),
            ("quad(side)", "quad(side)"),
            ("4.1", "4.1"),
            ("5.0", "5.0(side)"),
            ("5.0(side)", "5.0(side)"),
            ("5.1", "5.1(side)"),
            ("5.1(side)", "5.1(side)"),
            ("6.1", "5.1(side)"),
            ("7.1", "5.1(side)"),
        ] {
            assert_eq!(surround_core_layout(&layout(src)), layout(out), "{src}");
        }
        assert_eq!(he_aac_v2_layout(&layout("5.1")), layout("stereo"));
        assert_eq!(vorbis_layout(&layout("5.1(side)")), Some(layout("5.1")));
    }
}
