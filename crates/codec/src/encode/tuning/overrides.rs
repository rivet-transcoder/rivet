//! Backend-agnostic encode overrides, and the policy that resolves them.
//!
//! # Why this exists
//!
//! Everything in `adapters.rs` derives its numbers from two enums — a
//! [`QualityTarget`] and a [`SpeedTier`] — and nothing else. That is a fine
//! default and a poor ceiling: it produced one quality value for every rung of
//! an ABR ladder, which is how a 240p rung ended up spending four times the
//! bits per pixel of its 1080p sibling. The lower rungs of a ladder exist to be
//! cheap, and a constant quantizer makes them expensive.
//!
//! The fix is not a special case for "ICQ +2 per rung". It is that the caller —
//! the service that knows what a rung is *for* — gets to say so, and this
//! module is the vocabulary it says it in. Every knob a backend understands is
//! expressed here once, backend-agnostically, and each adapter applies whatever
//! it can honour.
//!
//! # The shape
//!
//! [`EncodeOverrides`] is the knob set. Every field is optional and `None`
//! means "leave whatever the target and tier already chose" — so an empty
//! override is exactly today's behaviour, and a caller only names what it wants
//! to change.
//!
//! [`RungPolicy`] resolves overrides for one rung: a global set, then any
//! number of [`RungRule`]s whose [`RungSelector`] matches, applied in order.
//! Later rules win. That is enough to express "every rung two steps softer than
//! the one above it", "nothing above 1080p", "the top rung gets a little back",
//! and things nobody has thought of yet, without this module knowing what any
//! of them mean.
//!
//! # What it deliberately is not
//!
//! It is not a per-encoder settings blob. A caller that wants to set
//! `NV_ENC_CONFIG_AV1::numFwdRefs` directly should not be able to, because that
//! request cannot be honoured by QSV or rav1e and a ladder that silently
//! behaves differently per GPU vendor is worse than one that is merely
//! suboptimal. Knobs live here only once they mean something everywhere — or
//! once the adapters that cannot honour them say so.

use super::{QualityTarget, SpeedTier};

/// An explicit tile grid. Columns × rows, literal (not log2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileGrid {
    pub columns: u8,
    pub rows: u8,
}

impl TileGrid {
    pub const SINGLE: Self = Self { columns: 1, rows: 1 };

    pub fn tiles(self) -> u32 {
        u32::from(self.columns) * u32::from(self.rows)
    }
}

/// One knob set, backend-agnostic.
///
/// `None`/`0` means "unchanged". An `EncodeOverrides::default()` applied to any
/// adapter must produce byte-identical parameters to not applying it at all —
/// there is a test for exactly that, because the whole mechanism is only safe
/// if the empty case is provably inert.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EncodeOverrides {
    /// Shift quality, in **libaom-CQ-equivalent steps**.
    ///
    /// Positive is always "smaller file, lower quality" on every backend. The
    /// native scales disagree about units *and* direction — QSV ICQ is 1..51
    /// counting up as quality falls, NVENC CQ is 0..63 the same way, NVENC's
    /// VBR `targetQuality` is 0..100 counting *down*, rav1e's quantizer is
    /// 0..255 at roughly four times libaom's scale — so a raw "native units"
    /// delta would mean five different things.
    ///
    /// libaom CQ is the currency this module already converts through
    /// (`libaom_cq_for_target`), so it is the one used here: each adapter
    /// converts a step into its own scale and applies the sign its scale needs.
    /// A caller says "+2 softer" once and it means the same change on an Arc and
    /// on a 5060.
    ///
    /// Roughly 15-20% bitrate per step on AV1 hardware encoders at the sizes
    /// this service produces — an observation from this content, not a promise.
    pub quality_delta: i16,

    /// Replace the quality target outright. For per-title work, where a
    /// measured VMAF for *this* clip beats any fixed tier.
    pub quality_target: Option<QualityTarget>,

    /// Replace the speed tier — e.g. spend more CPU on the rung most people
    /// watch and less on the ones they do not.
    pub speed_tier: Option<SpeedTier>,

    /// Force a tile grid instead of the resolution-derived default. Tiles cost
    /// roughly 1% quality each and buy encoder parallelism this service gets
    /// from running rungs concurrently instead.
    pub tiles: Option<TileGrid>,

    /// Keyframe interval in frames. Segment length drives this for HLS, so it
    /// is normally computed rather than set — but a caller that knows its
    /// segmentation should be able to say so.
    pub keyframe_interval: Option<u32>,

    /// Rate-control lookahead depth in frames; `Some(0)` disables it.
    ///
    /// **Not free to enable.** A backend that buffers frames holds the input
    /// surface it was given, and an encoder whose surface pool assumes
    /// one-in-one-out will hand the next frame the same memory — which is a
    /// silently corrupted picture, not an error. Only set this on a backend
    /// whose pool selects by "the runtime has released this", which is why the
    /// adapters treat it as a request rather than an instruction.
    pub lookahead_frames: Option<u32>,

    /// Consecutive B-frames; `Some(0)` means none. Same buffering caveat as
    /// [`Self::lookahead_frames`] — B-frames imply reordering, which implies
    /// the encoder holds pictures.
    pub bframes: Option<u8>,

    /// How many reference frames the encoder may predict from.
    ///
    /// AV1 allows up to seven; this service was asking for **one**, which lets
    /// a P-frame reference only its immediate predecessor and costs more
    /// compression than the missing B-frames do. Unlike B-frames it needs no
    /// reordering — the encoder holds more surfaces, which is a pool-sizing
    /// question, not an output-ordering one.
    pub reference_frames: Option<u8>,

    /// Multi-pass rate control, where the backend has it.
    pub multi_pass: Option<bool>,

    /// Request AV1 film-grain synthesis.
    ///
    /// Plumbed, and off. NVENC exposes `enableFilmGrainParams` but will not
    /// analyse grain for you — the caller must hand it a populated
    /// `NV_ENC_FILM_GRAIN_PARAMS_AV1`, so honouring this means writing a grain
    /// estimator. oneVPL's vendored headers expose no equivalent at all. The
    /// knob is here so the plumbing exists and the gap is visible; adapters
    /// currently ignore it rather than pretending.
    pub film_grain: Option<bool>,
}

impl EncodeOverrides {
    /// Whether this override would change anything.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Layer `other` on top of `self`. Anything `other` names wins; anything it
    /// leaves `None` keeps this value. `quality_delta` accumulates, because two
    /// rules each asking for "a bit softer" should compound rather than the
    /// second silently discarding the first.
    pub fn merge(self, other: Self) -> Self {
        Self {
            quality_delta: self.quality_delta.saturating_add(other.quality_delta),
            quality_target: other.quality_target.or(self.quality_target),
            speed_tier: other.speed_tier.or(self.speed_tier),
            tiles: other.tiles.or(self.tiles),
            keyframe_interval: other.keyframe_interval.or(self.keyframe_interval),
            lookahead_frames: other.lookahead_frames.or(self.lookahead_frames),
            bframes: other.bframes.or(self.bframes),
            reference_frames: other.reference_frames.or(self.reference_frames),
            multi_pass: other.multi_pass.or(self.multi_pass),
            film_grain: other.film_grain.or(self.film_grain),
        }
    }
}

/// Where a rung sits in the ladder it belongs to.
///
/// Passed to [`RungPolicy::resolve`] so a rule can talk about position
/// ("two steps below the top") as well as size ("anything under 480p"). A
/// caller that has no ladder can use [`RungContext::standalone`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RungContext {
    pub width: u32,
    pub height: u32,
    /// 0 is the largest rung.
    pub index: usize,
    pub rung_count: usize,
}

impl RungContext {
    /// A single encode that is not part of a ladder.
    pub fn standalone(width: u32, height: u32) -> Self {
        Self { width, height, index: 0, rung_count: 1 }
    }

    /// The short side, which is what "1080p" has always meant here regardless
    /// of orientation.
    pub fn short_side(&self) -> u32 {
        self.width.min(self.height)
    }

    /// Steps below the largest rung. 0 for the top rung.
    pub fn steps_below_top(&self) -> usize {
        self.index
    }
}

/// Which rungs a rule applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RungSelector {
    /// Every rung.
    Any,
    /// The largest rung.
    Top,
    /// Everything except the largest.
    BelowTop,
    /// Exactly this many steps below the top.
    StepsBelowTop(usize),
    /// Short side at or below this.
    ShortSideAtMost(u32),
    /// Short side at or above this.
    ShortSideAtLeast(u32),
}

impl RungSelector {
    pub fn matches(&self, rung: &RungContext) -> bool {
        match *self {
            Self::Any => true,
            Self::Top => rung.index == 0,
            Self::BelowTop => rung.index > 0,
            Self::StepsBelowTop(n) => rung.steps_below_top() == n,
            Self::ShortSideAtMost(px) => rung.short_side() <= px,
            Self::ShortSideAtLeast(px) => rung.short_side() >= px,
        }
    }
}

/// One conditional knob set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RungRule {
    pub selector: RungSelector,
    pub overrides: EncodeOverrides,
}

impl RungRule {
    pub fn new(selector: RungSelector, overrides: EncodeOverrides) -> Self {
        Self { selector, overrides }
    }
}

/// What a caller hands the encoder factory: a global knob set plus rules.
///
/// Resolution is `global`, then every matching rule in declaration order, each
/// merged on top of the last. Order is the caller's, and later wins — so a
/// broad rule can set a floor and a narrow one can carve an exception out of
/// it, which is the usual way these get written.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RungPolicy {
    pub global: EncodeOverrides,
    pub rules: Vec<RungRule>,
}

impl RungPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply to every rung.
    pub fn with_global(mut self, overrides: EncodeOverrides) -> Self {
        self.global = self.global.merge(overrides);
        self
    }

    pub fn with_rule(mut self, selector: RungSelector, overrides: EncodeOverrides) -> Self {
        self.rules.push(RungRule::new(selector, overrides));
        self
    }

    /// A quality step that compounds down the ladder.
    ///
    /// `step` per position below the top: the second rung is one step softer,
    /// the third two, and so on. This is the shape an ABR ladder wants and the
    /// thing a single quality target cannot express — expressed here as an
    /// ordinary rule set rather than as a special case inside an adapter, so
    /// that a caller wanting a different curve can write one.
    pub fn with_quality_step_per_rung(mut self, step: i16) -> Self {
        if step == 0 {
            return self;
        }
        // Bounded: a ladder deeper than this is not a ladder, and an unbounded
        // loop here would let a bad `rung_count` generate rules for ever.
        for position in 1..=MAX_LADDER_DEPTH {
            self.rules.push(RungRule::new(
                RungSelector::StepsBelowTop(position),
                EncodeOverrides {
                    quality_delta: step.saturating_mul(position as i16),
                    ..Default::default()
                },
            ));
        }
        self
    }

    /// The knob set for one rung.
    pub fn resolve(&self, rung: &RungContext) -> EncodeOverrides {
        self.rules
            .iter()
            .filter(|rule| rule.selector.matches(rung))
            .fold(self.global, |acc, rule| acc.merge(rule.overrides))
    }
}

/// How many rungs deep [`RungPolicy::with_quality_step_per_rung`] will
/// generate rules for. The ladder this service builds tops out at seven
/// standard short sides plus a source rung.
pub const MAX_LADDER_DEPTH: usize = 16;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_override_changes_nothing() {
        // The whole mechanism rests on this: applying a default override has to
        // be indistinguishable from not applying one, or every existing caller
        // silently changes behaviour the day this lands.
        let empty = EncodeOverrides::default();

        assert!(empty.is_empty());
        assert_eq!(empty.merge(EncodeOverrides::default()), empty);
    }

    #[test]
    fn merging_accumulates_quality_and_lets_the_later_value_win() {
        let first = EncodeOverrides {
            quality_delta: 2,
            bframes: Some(0),
            multi_pass: Some(false),
            ..Default::default()
        };
        let second = EncodeOverrides {
            quality_delta: 3,
            bframes: Some(2),
            ..Default::default()
        };

        let merged = first.merge(second);

        // Two rules each asking for "softer" compound; a rule that names a
        // value replaces one that named a different value; a value only the
        // earlier rule named survives.
        assert_eq!(merged.quality_delta, 5);
        assert_eq!(merged.bframes, Some(2));
        assert_eq!(merged.multi_pass, Some(false));
    }

    fn ladder_rung(index: usize, short_side: u32) -> RungContext {
        RungContext { width: short_side * 2, height: short_side, index, rung_count: 5 }
    }

    #[test]
    fn a_per_rung_step_compounds_down_the_ladder() {
        let policy = RungPolicy::new().with_quality_step_per_rung(2);

        assert_eq!(policy.resolve(&ladder_rung(0, 1080)).quality_delta, 0);
        assert_eq!(policy.resolve(&ladder_rung(1, 720)).quality_delta, 2);
        assert_eq!(policy.resolve(&ladder_rung(2, 480)).quality_delta, 4);
        assert_eq!(policy.resolve(&ladder_rung(3, 360)).quality_delta, 6);
        assert_eq!(policy.resolve(&ladder_rung(4, 240)).quality_delta, 8);
    }

    #[test]
    fn the_top_rung_can_be_given_some_back() {
        // The shape the ladder actually wants: everything below the top gets
        // cheaper, and the rung most people watch gets a little sharper.
        let policy = RungPolicy::new()
            .with_quality_step_per_rung(2)
            .with_rule(
                RungSelector::Top,
                EncodeOverrides { quality_delta: -2, ..Default::default() },
            );

        assert_eq!(policy.resolve(&ladder_rung(0, 1080)).quality_delta, -2);
        assert_eq!(policy.resolve(&ladder_rung(1, 720)).quality_delta, 2);
    }

    #[test]
    fn selectors_pick_out_the_rungs_they_name() {
        let top = ladder_rung(0, 1080);
        let bottom = ladder_rung(4, 240);

        assert!(RungSelector::Any.matches(&top));
        assert!(RungSelector::Top.matches(&top));
        assert!(!RungSelector::Top.matches(&bottom));
        assert!(RungSelector::BelowTop.matches(&bottom));
        assert!(RungSelector::ShortSideAtMost(360).matches(&bottom));
        assert!(!RungSelector::ShortSideAtMost(360).matches(&top));
        assert!(RungSelector::ShortSideAtLeast(1080).matches(&top));
    }

    #[test]
    fn a_narrow_rule_can_carve_an_exception_out_of_a_broad_one() {
        // Declaration order is the caller's, and later wins. This is how an
        // exception gets written, so it is worth pinning.
        let policy = RungPolicy::new()
            .with_rule(
                RungSelector::Any,
                EncodeOverrides { bframes: Some(3), ..Default::default() },
            )
            .with_rule(
                RungSelector::ShortSideAtMost(240),
                EncodeOverrides { bframes: Some(0), ..Default::default() },
            );

        assert_eq!(policy.resolve(&ladder_rung(0, 1080)).bframes, Some(3));
        assert_eq!(policy.resolve(&ladder_rung(4, 240)).bframes, Some(0));
    }

    #[test]
    fn a_standalone_encode_is_the_top_and_only_rung() {
        let policy = RungPolicy::new().with_quality_step_per_rung(2);
        let solo = RungContext::standalone(1920, 1080);

        assert_eq!(solo.steps_below_top(), 0);
        assert_eq!(policy.resolve(&solo).quality_delta, 0);
    }

    #[test]
    fn short_side_is_orientation_independent() {
        let landscape = RungContext::standalone(1920, 1080);
        let portrait = RungContext::standalone(1080, 1920);

        assert_eq!(landscape.short_side(), 1080);
        assert_eq!(portrait.short_side(), 1080);
    }
}
