//! A text grammar for [`RungPolicy`], and the recommended ladder policy.
//!
//! # Why a grammar
//!
//! A policy is the kind of thing that gets typed into a Helm values file, a
//! CLI flag or a job manifest at 2am. Expressing it as the rule vocabulary
//! rather than a fixed list of knobs means an operator can replace any of it
//! without a build:
//!
//! ```text
//! qstep=2;top:q=-2;short<=2159:tiles=1x1;any:refs=3
//! ```
//!
//! - Rules are separated by `;` and applied in the order written; later wins.
//! - A rule is `selector:key=value,key=value`.
//! - Selectors: `any` (or `*`), `top`, `below_top`, `step=N` (N positions
//!   below the top), `short<=N`, `short>=N`.
//! - Keys: `q` (quality delta, signed, libaom-CQ steps), `tiles` (`CxR`),
//!   `gop` (frames), `lookahead` (frames), `bframes`, `refs`, `multipass`
//!   (`on`/`off`), `speed` (`draft`/`standard`/`archive`), `target`
//!   (`visually_lossless`/`high`/`standard`/`low`/`vmaf=N`), `grain`
//!   (`on`/`off`).
//! - `qstep=N` on its own is the compounding per-rung step
//!   ([`RungPolicy::with_quality_step_per_rung`]).
//!
//! An empty string is an empty policy — byte-for-byte the behaviour with no
//! policy at all, which is the control arm of any benchmark and the reason
//! the empty case is a tested property rather than an assumption.
//!
//! # Why a recommended policy
//!
//! Every rung of every ladder used to be encoded at one quality — the same
//! value for a 1920×960 rung as for a 480×240 one. That is how a 240p rung
//! came to spend four times the bits per pixel of the top rung and the four
//! lower rungs came to be 65% of a job's storage. The lower rungs exist to be
//! cheap; a constant quantizer makes them expensive, because the same
//! quantizer at a quarter of the resolution is a far finer quantizer in terms
//! of what an eye can resolve. [`LadderPolicy`] is the shape that measured
//! well, with every number a field.

use std::str::FromStr;

use super::{
    EncodeOverrides, QualityTarget, RungPolicy, RungSelector, SpeedTier, TileGrid,
};

/// The recommended ladder policy, as numbers. `Default` is the measured
/// recommendation; [`Self::into_policy`] turns it into rules.
///
/// - **Softer going down.** `quality_step` libaom-CQ-equivalent steps per
///   position below the top, compounding. Roughly −15–20% bitrate per step at
///   these sizes on hardware AV1.
/// - **Nothing extra at the top.** `top_bonus` used to spend `2` back on the
///   rung most people watch. Measured with VMAF against the source, that bought
///   **+0.62 VMAF for +23.3% bytes** — the worst trade in the ladder, on the
///   rung that dominates storage — so it is `0` by default. Still a knob: the
///   measurement is one clip of synthetic content, and grain or fast motion may
///   justify spending there once measured.
/// - **One tile below 4K.** Tiles cost about 1% quality each and buy encoder
///   parallelism a ladder already gets from running rungs concurrently, on
///   separate GPUs.
/// - **Three reference frames.** oneVPL was asking for one — a P-frame could
///   predict only from its immediate predecessor. Unlike B-frames this needs no
///   reordering, so it is a pool-size question rather than a latency one.
///
/// Lookahead, B-frames and multi-pass are *off* by default and deliberately.
/// They are worth 10–20% BD-rate and they make the encoder hold input
/// surfaces; the surface-reuse bug that produced duplicated frames was exactly
/// that. They are one field away, so they can be turned on for a measured
/// comparison rather than on an argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LadderPolicy {
    /// Compounding quality step per position below the top rung.
    pub quality_step: i16,
    /// How much sharper the top rung is than the ladder's base quality.
    pub top_bonus: i16,
    /// Rungs with a short side *below* this get a single tile. `0` disables.
    pub single_tile_below_short_side: u32,
    /// Reference frames to request. `0` leaves the encoder's default.
    pub reference_frames: u8,
    /// Rate-control lookahead depth, in frames.
    pub lookahead_frames: Option<u32>,
    /// Consecutive B-frames.
    pub bframes: Option<u8>,
    /// Multi-pass rate control.
    pub multi_pass: Option<bool>,
}

impl Default for LadderPolicy {
    fn default() -> Self {
        Self {
            quality_step: 2,
            top_bonus: 0,
            single_tile_below_short_side: 2160,
            reference_frames: 3,
            lookahead_frames: None,
            bframes: None,
            multi_pass: None,
        }
    }
}

impl LadderPolicy {
    /// The rule set these numbers describe.
    pub fn into_policy(self) -> RungPolicy {
        let mut policy = RungPolicy::new()
            .with_global(EncodeOverrides {
                reference_frames: (self.reference_frames > 0).then_some(self.reference_frames),
                lookahead_frames: self.lookahead_frames,
                bframes: self.bframes,
                multi_pass: self.multi_pass,
                ..Default::default()
            })
            .with_quality_step_per_rung(self.quality_step);

        if self.top_bonus != 0 {
            policy = policy.with_rule(
                RungSelector::Top,
                EncodeOverrides { quality_delta: -self.top_bonus, ..Default::default() },
            );
        }

        if self.single_tile_below_short_side > 0 {
            policy = policy.with_rule(
                // `- 1`: "below 4K" means a 2160 rung keeps its tiles.
                RungSelector::ShortSideAtMost(self.single_tile_below_short_side.saturating_sub(1)),
                EncodeOverrides { tiles: Some(TileGrid::SINGLE), ..Default::default() },
            );
        }

        policy
    }
}

impl RungPolicy {
    /// The recommended ladder policy — [`LadderPolicy::default`] as rules.
    pub fn recommended() -> Self {
        LadderPolicy::default().into_policy()
    }

    /// Parse the rule grammar documented at the [module level](self).
    ///
    /// Errors name the offending fragment: "invalid policy" would not help
    /// anybody find the missing colon.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let mut policy = RungPolicy::new();

        for fragment in spec.split(';').map(str::trim).filter(|f| !f.is_empty()) {
            // `qstep=N` is the one rule that expands into many, so it is
            // written as a bare directive rather than pretending to have a
            // selector.
            if let Some(value) = fragment.strip_prefix("qstep=") {
                let step: i16 = value
                    .trim()
                    .parse()
                    .map_err(|_| format!("`{fragment}`: qstep wants a whole number of steps"))?;
                policy = policy.with_quality_step_per_rung(step);
                continue;
            }

            let (selector_text, assignments) = fragment
                .split_once(':')
                .ok_or_else(|| format!("`{fragment}`: expected `selector:key=value`"))?;
            let selector = parse_selector(selector_text.trim())
                .ok_or_else(|| format!("`{fragment}`: `{selector_text}` is not a selector"))?;
            let overrides = parse_overrides(assignments, fragment)?;

            // `any` is the global set rather than a rule, so that a spec
            // reading `any:refs=3;top:refs=5` resolves the way it looks.
            if selector == RungSelector::Any {
                policy = policy.with_global(overrides);
            } else {
                policy = policy.with_rule(selector, overrides);
            }
        }

        Ok(policy)
    }
}

impl FromStr for RungPolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

fn parse_selector(text: &str) -> Option<RungSelector> {
    if let Some(value) = text.strip_prefix("step=") {
        return value.trim().parse().ok().map(RungSelector::StepsBelowTop);
    }
    if let Some(value) = text.strip_prefix("short<=") {
        return value.trim().parse().ok().map(RungSelector::ShortSideAtMost);
    }
    if let Some(value) = text.strip_prefix("short>=") {
        return value.trim().parse().ok().map(RungSelector::ShortSideAtLeast);
    }
    match text {
        "any" | "*" => Some(RungSelector::Any),
        "top" => Some(RungSelector::Top),
        "below_top" => Some(RungSelector::BelowTop),
        _ => None,
    }
}

fn parse_overrides(assignments: &str, fragment: &str) -> Result<EncodeOverrides, String> {
    let mut overrides = EncodeOverrides::default();

    for pair in assignments.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let (key, value) =
            pair.split_once('=').ok_or_else(|| format!("`{fragment}`: `{pair}` is not `key=value`"))?;
        let (key, value) = (key.trim(), value.trim());
        let bad = || format!("`{fragment}`: `{value}` is not a valid `{key}`");

        match key {
            "q" => overrides.quality_delta = value.parse().map_err(|_| bad())?,
            "tiles" => overrides.tiles = Some(parse_tiles(value).ok_or_else(bad)?),
            "gop" => overrides.keyframe_interval = Some(value.parse().map_err(|_| bad())?),
            "lookahead" => overrides.lookahead_frames = Some(value.parse().map_err(|_| bad())?),
            "bframes" => overrides.bframes = Some(value.parse().map_err(|_| bad())?),
            "refs" => overrides.reference_frames = Some(value.parse().map_err(|_| bad())?),
            "multipass" => overrides.multi_pass = Some(parse_bool(value).ok_or_else(bad)?),
            "grain" => overrides.film_grain = Some(parse_bool(value).ok_or_else(bad)?),
            "speed" => overrides.speed_tier = Some(parse_tier(value).ok_or_else(bad)?),
            "target" => overrides.quality_target = Some(parse_target(value).ok_or_else(bad)?),
            _ => return Err(format!("`{fragment}`: `{key}` is not a knob")),
        }
    }

    Ok(overrides)
}

fn parse_tiles(value: &str) -> Option<TileGrid> {
    let (columns, rows) = value.split_once(['x', 'X'])?;
    Some(TileGrid { columns: columns.trim().parse().ok()?, rows: rows.trim().parse().ok()? })
}

fn parse_tier(value: &str) -> Option<SpeedTier> {
    match value.to_ascii_lowercase().as_str() {
        "draft" => Some(SpeedTier::Draft),
        "standard" => Some(SpeedTier::Standard),
        "archive" => Some(SpeedTier::Archive),
        _ => None,
    }
}

fn parse_target(value: &str) -> Option<QualityTarget> {
    if let Some(score) = value.to_ascii_lowercase().strip_prefix("vmaf=") {
        return score.trim().parse().ok().map(QualityTarget::Vmaf);
    }
    match value.to_ascii_lowercase().as_str() {
        "visually_lossless" | "lossless" => Some(QualityTarget::VisuallyLossless),
        "high" => Some(QualityTarget::High),
        "standard" => Some(QualityTarget::Standard),
        "low" => Some(QualityTarget::Low),
        _ => None,
    }
}

/// `1`/`on`/`true`/`yes` and `0`/`off`/`false`/`no`, case-insensitively.
pub fn parse_bool(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "true" | "yes" => Some(true),
        "0" | "off" | "false" | "no" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::tuning::RungContext;

    fn rung(index: usize, short: u32, count: usize) -> RungContext {
        RungContext { width: short * 2, height: short, index, rung_count: count }
    }

    #[test]
    fn the_recommended_ladder_gets_cheaper_going_down() {
        // The whole point. Without this the 240p rung costs four times the
        // bits per pixel of the top one.
        let policy = RungPolicy::recommended();
        let deltas: Vec<i16> = [1080, 720, 480, 360, 240]
            .iter()
            .enumerate()
            .map(|(index, &short)| policy.resolve(&rung(index, short, 5)).quality_delta)
            .collect();

        assert!(
            deltas.windows(2).all(|pair| pair[1] > pair[0]),
            "quality does not soften monotonically: {deltas:?}",
        );
        // The top rung sits at the base quality — no bonus, no penalty. It
        // used to be spent `-2` sharper; measured, that cost 23% of its bytes
        // for 0.62 VMAF.
        assert_eq!(deltas[0], 0, "the top rung should sit at the base quality");
    }

    #[test]
    fn the_recommended_ladder_asks_for_more_than_one_reference_frame() {
        let overrides = RungPolicy::recommended().resolve(&rung(0, 1080, 5));
        assert_eq!(overrides.reference_frames, Some(3));
    }

    #[test]
    fn the_recommended_ladder_leaves_frame_buffering_alone() {
        // Lookahead and B-frames make the encoder hold input surfaces, which is
        // precisely what produced duplicated frames. Available, not assumed.
        let overrides = RungPolicy::recommended().resolve(&rung(0, 1080, 5));
        assert_eq!(overrides.lookahead_frames, None);
        assert_eq!(overrides.bframes, None);
        assert_eq!(overrides.multi_pass, None);
    }

    #[test]
    fn tiles_collapse_below_4k_and_survive_at_it() {
        let policy = RungPolicy::recommended();
        assert_eq!(policy.resolve(&rung(1, 1080, 5)).tiles, Some(TileGrid::SINGLE));
        assert_eq!(policy.resolve(&rung(0, 2160, 5)).tiles, None, "4K should keep its tiles");
    }

    #[test]
    fn a_top_bonus_is_a_sharper_top_rung() {
        let policy = LadderPolicy { top_bonus: 2, ..Default::default() }.into_policy();
        assert_eq!(policy.resolve(&rung(0, 1080, 5)).quality_delta, -2);
        assert_eq!(policy.resolve(&rung(1, 720, 5)).quality_delta, 2);
    }

    #[test]
    fn a_spec_replaces_the_default_rather_than_adding_to_it() {
        let policy = RungPolicy::parse("any:refs=5").expect("valid spec");
        let overrides = policy.resolve(&rung(3, 360, 5));
        assert_eq!(overrides.reference_frames, Some(5));
        assert_eq!(overrides.quality_delta, 0, "the default step leaked into an explicit spec");
    }

    #[test]
    fn every_documented_knob_parses() {
        // The module doc is the interface. If a knob listed there does not
        // parse, the doc is a lie and this is where it gets caught.
        let spec = "qstep=3;\
                    any:refs=4,lookahead=8,bframes=2,multipass=on,grain=off;\
                    top:q=-2,tiles=2x2,speed=archive,target=vmaf=95;\
                    below_top:gop=120;\
                    step=2:q=1;\
                    short<=480:target=low;\
                    short>=1080:speed=standard";
        let policy: RungPolicy = spec.parse().expect("documented spec should parse");

        let top = policy.resolve(&rung(0, 1080, 5));
        assert_eq!(top.tiles, Some(TileGrid { columns: 2, rows: 2 }));
        assert_eq!(top.quality_target, Some(QualityTarget::Vmaf(95)));
        assert_eq!(top.speed_tier, Some(SpeedTier::Standard), "the later rule should win");
        assert_eq!(top.reference_frames, Some(4));
        assert_eq!(top.quality_delta, -2);

        let third = policy.resolve(&rung(2, 480, 5));
        assert_eq!(third.keyframe_interval, Some(120));
        assert_eq!(third.quality_target, Some(QualityTarget::Low));
        // qstep 3 twice below the top, plus the `step=2` rule's own +1.
        assert_eq!(third.quality_delta, 7);
    }

    #[test]
    fn a_malformed_spec_says_which_fragment() {
        for (spec, needle) in [
            ("top", "expected `selector:key=value`"),
            ("sideways:q=1", "is not a selector"),
            ("top:q=sideways", "is not a valid `q`"),
            ("top:wobble=1", "is not a knob"),
            ("top:q", "is not `key=value`"),
            ("qstep=lots", "qstep wants"),
        ] {
            let error = RungPolicy::parse(spec).expect_err("should have rejected {spec}");
            assert!(error.contains(needle), "{spec:?} said {error:?}, wanted {needle:?}");
        }
    }

    #[test]
    fn the_empty_spec_is_genuinely_empty() {
        // The control arm of any benchmark. It has to be reachable and it has
        // to be genuinely empty.
        let policy = RungPolicy::parse("").expect("empty spec");
        assert!(policy.rules.is_empty());
        assert!(policy.global.is_empty());
        assert!(policy.resolve(&rung(3, 360, 5)).is_empty());
    }
}
