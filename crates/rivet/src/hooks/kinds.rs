//! The kinds of hook: one trait per place in the pipeline, each handed only
//! what exists there. See the [module docs](super) for the table of where each
//! hooks in.

use std::fmt;
use std::sync::Arc;

use anyhow::{Result, bail};

use super::{
    ArtifactEvent, CompletedEvent, FailedEvent, FrameEvent, FrameSampling, Hook, HookContext, HookEvent, HookOutcome,
    HookPolicy, Hooks, ProbeEvent, SourceEvent, Stage, StageSet, StillEvent,
};

/// Which kind of hook an entry is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookKind {
    Source,
    Probe,
    DecodedFrame,
    EncoderFrame,
    Still,
    Artifact,
    Completed,
    Failed,
    /// A general [`Hook`] at the stages it lists.
    Event,
}

impl HookKind {
    pub fn as_str(self) -> &'static str {
        match self {
            HookKind::Source => "source",
            HookKind::Probe => "probe",
            HookKind::DecodedFrame => "decoded-frame",
            HookKind::EncoderFrame => "encoder-frame",
            HookKind::Still => "still",
            HookKind::Artifact => "artifact",
            HookKind::Completed => "completed",
            HookKind::Failed => "failed",
            HookKind::Event => "event",
        }
    }
}

impl fmt::Display for HookKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What an output is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArtifactKind {
    /// A self-contained video file (a single-file rung).
    Video,
    /// An audio-only file (`.mp3`, `.flac`, `.m4a`).
    Audio,
    /// A still image.
    Image,
    /// An HLS rendition: a directory of segments and its playlist.
    Rendition,
    /// An HLS master playlist.
    Playlist,
}

impl ArtifactKind {
    pub const ALL: [ArtifactKind; 5] =
        [ArtifactKind::Video, ArtifactKind::Audio, ArtifactKind::Image, ArtifactKind::Rendition, ArtifactKind::Playlist];

    pub fn as_str(self) -> &'static str {
        match self {
            ArtifactKind::Video => "video",
            ArtifactKind::Audio => "audio",
            ArtifactKind::Image => "image",
            ArtifactKind::Rendition => "rendition",
            ArtifactKind::Playlist => "playlist",
        }
    }
}

impl fmt::Display for ArtifactKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for ArtifactKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "video" => ArtifactKind::Video,
            "audio" => ArtifactKind::Audio,
            "image" => ArtifactKind::Image,
            "rendition" => ArtifactKind::Rendition,
            "playlist" => ArtifactKind::Playlist,
            other => bail!("unknown artifact kind `{other}` (video, audio, image, rendition, playlist)"),
        })
    }
}

// ---------------------------------------------------------------------------
// The traits
// ---------------------------------------------------------------------------

/// The source bytes as they arrived, before anything parses them. Once per
/// source: each clip of a splice is its own.
pub trait SourceHook: Send + Sync {
    fn on_source(&self, ctx: &HookContext, source: &SourceEvent) -> Result<HookOutcome>;

    fn describe(&self) -> String {
        "source hook".into()
    }
}

/// The source's description, once demuxed (or an image's header read), before
/// anything is decoded.
pub trait ProbeHook: Send + Sync {
    fn on_probe(&self, ctx: &HookContext, probe: &ProbeEvent) -> Result<HookOutcome>;

    fn describe(&self) -> String {
        "probe hook".into()
    }
}

/// Video frames as the decoder produced them, turned upright: before
/// tonemapping, chroma or bit-depth conversion, the spec's filters or any
/// scaling — what the source holds, whatever output is asked for. The pixel
/// format is the decoder's (`yuv420p`, `nv12`, `yuv420p10le`, ...).
pub trait DecodedFrameHook: Send + Sync {
    /// Which frames it is handed.
    fn sampling(&self) -> FrameSampling {
        FrameSampling::default()
    }

    fn on_decoded_frame(&self, ctx: &HookContext, frame: &FrameEvent) -> Result<HookOutcome>;

    fn describe(&self) -> String {
        "decoded-frame hook".into()
    }
}

/// Video frames exactly as the encoders receive them: after the colour
/// pipeline (tonemap, chroma, bit depth) and the spec's filters, before each
/// rung's scaling. Always `yuv420p` or `yuv420p10le`.
pub trait EncoderFrameHook: Send + Sync {
    /// Which frames it is handed.
    fn sampling(&self) -> FrameSampling {
        FrameSampling::default()
    }

    fn on_encoder_frame(&self, ctx: &HookContext, frame: &FrameEvent) -> Result<HookOutcome>;

    fn describe(&self) -> String {
        "encoder-frame hook".into()
    }
}

/// Still pictures in an image job: the decoded image, or each still taken from
/// a video. Upright 8-bit RGBA, every one (no sampling).
pub trait StillHook: Send + Sync {
    fn on_still(&self, ctx: &HookContext, still: &StillEvent) -> Result<HookOutcome>;

    fn describe(&self) -> String {
        "still hook".into()
    }
}

/// The job's outputs, before they are returned: each output of a kind in
/// [`kinds`](Self::kinds).
pub trait ArtifactHook: Send + Sync {
    /// The kinds of output it is handed. Default: every kind.
    fn kinds(&self) -> Vec<ArtifactKind> {
        ArtifactKind::ALL.to_vec()
    }

    fn on_artifact(&self, ctx: &HookContext, artifact: &ArtifactEvent) -> Result<HookOutcome>;

    fn describe(&self) -> String {
        "artifact hook".into()
    }
}

/// The job made everything it was asked for. Its last gate: a rejection here
/// still fails the job.
pub trait CompletedHook: Send + Sync {
    fn on_completed(&self, ctx: &HookContext, done: &CompletedEvent) -> Result<HookOutcome>;

    fn describe(&self) -> String {
        "completed hook".into()
    }
}

/// The job failed, a rejection by a hook included. Its verdict changes
/// nothing; it runs once per job.
pub trait FailedHook: Send + Sync {
    fn on_failed(&self, ctx: &HookContext, failed: &FailedEvent) -> Result<HookOutcome>;

    fn describe(&self) -> String {
        "failed hook".into()
    }
}

// ---------------------------------------------------------------------------
// Adapters onto the general trait
// ---------------------------------------------------------------------------

/// One adapter per kind: the kind's stage, its event unwrapped for its
/// method, and any extra methods it has (sampling, accepted artifact kinds).
macro_rules! adapter {
    ($name:ident, $trait:ident, $kind:ident, $method:ident { $($extra:tt)* }) => {
        struct $name<H>(H);

        impl<H: $trait> Hook for $name<H> {
            fn stages(&self) -> StageSet {
                StageSet::of(&[Stage::$kind])
            }
            fn call(&self, ctx: &HookContext, event: &HookEvent) -> Result<HookOutcome> {
                match event {
                    HookEvent::$kind(e) => self.0.$method(ctx, e),
                    _ => Ok(HookOutcome::proceed()),
                }
            }
            fn describe(&self) -> String {
                self.0.describe()
            }
            fn kind(&self) -> HookKind {
                HookKind::$kind
            }
            $($extra)*
        }
    };
}

adapter!(SourceAdapter, SourceHook, Source, on_source {});
adapter!(ProbeAdapter, ProbeHook, Probe, on_probe {});
adapter!(DecodedFrameAdapter, DecodedFrameHook, DecodedFrame, on_decoded_frame {
    fn sampling(&self) -> FrameSampling {
        self.0.sampling()
    }
});
adapter!(EncoderFrameAdapter, EncoderFrameHook, EncoderFrame, on_encoder_frame {
    fn sampling(&self) -> FrameSampling {
        self.0.sampling()
    }
});
adapter!(StillAdapter, StillHook, Still, on_still {});
adapter!(ArtifactAdapter, ArtifactHook, Artifact, on_artifact {
    fn artifact_kinds(&self) -> Vec<ArtifactKind> {
        self.0.kinds()
    }
});
adapter!(CompletedAdapter, CompletedHook, Completed, on_completed {});
adapter!(FailedAdapter, FailedHook, Failed, on_failed {});

// ---------------------------------------------------------------------------
// Registration by kind
// ---------------------------------------------------------------------------

macro_rules! register {
    ($(#[$doc:meta])* $plain:ident, $with:ident, $trait:ident, $adapter:ident) => {
        $(#[$doc])*
        pub fn $plain(self, name: impl Into<String>, hook: impl $trait + 'static) -> Self {
            self.$with(name, hook, HookPolicy::default())
        }

        #[doc = concat!("[`", stringify!($plain), "`](Self::", stringify!($plain), ") with a [`HookPolicy`].")]
        pub fn $with(self, name: impl Into<String>, hook: impl $trait + 'static, policy: HookPolicy) -> Self {
            self.with_policy(name, Arc::new($adapter(hook)), policy)
        }
    };
}

impl Hooks {
    register!(
        /// Register a [`SourceHook`]: the source bytes, before parsing.
        source, source_with, SourceHook, SourceAdapter
    );
    register!(
        /// Register a [`ProbeHook`]: the source's description.
        probe, probe_with, ProbeHook, ProbeAdapter
    );
    register!(
        /// Register a [`DecodedFrameHook`]: video frames as decoded.
        decoded_frames, decoded_frames_with, DecodedFrameHook, DecodedFrameAdapter
    );
    register!(
        /// Register an [`EncoderFrameHook`]: video frames as the encoders receive them.
        encoder_frames, encoder_frames_with, EncoderFrameHook, EncoderFrameAdapter
    );
    register!(
        /// Register a [`StillHook`]: still pictures in an image job.
        stills, stills_with, StillHook, StillAdapter
    );
    register!(
        /// Register an [`ArtifactHook`]: the job's outputs of the kinds it accepts.
        artifacts, artifacts_with, ArtifactHook, ArtifactAdapter
    );
    register!(
        /// Register a [`CompletedHook`]: the job finished.
        completed, completed_with, CompletedHook, CompletedAdapter
    );
    register!(
        /// Register a [`FailedHook`]: the job failed.
        failed, failed_with, FailedHook, FailedAdapter
    );
}

// ---------------------------------------------------------------------------
// Shared hooks
// ---------------------------------------------------------------------------

/// An `Arc` of a hook is that hook: one value can be registered at several
/// points (a decoded-frame and an encoder-frame hook sharing state), or kept
/// by the caller to read back after the job.
macro_rules! shared {
    ($trait:ident { $($method:ident($event:ty)),* } $(, $extra:ident -> $ret:ty)*) => {
        impl<T: $trait + ?Sized> $trait for Arc<T> {
            $(fn $method(&self, ctx: &HookContext, event: &$event) -> Result<HookOutcome> {
                (**self).$method(ctx, event)
            })*
            $(fn $extra(&self) -> $ret {
                (**self).$extra()
            })*
            fn describe(&self) -> String {
                (**self).describe()
            }
        }
    };
}

shared!(SourceHook { on_source(SourceEvent) });
shared!(ProbeHook { on_probe(ProbeEvent) });
shared!(DecodedFrameHook { on_decoded_frame(FrameEvent) }, sampling -> FrameSampling);
shared!(EncoderFrameHook { on_encoder_frame(FrameEvent) }, sampling -> FrameSampling);
shared!(StillHook { on_still(StillEvent) });
shared!(ArtifactHook { on_artifact(ArtifactEvent) }, kinds -> Vec<ArtifactKind>);
shared!(CompletedHook { on_completed(CompletedEvent) });
shared!(FailedHook { on_failed(FailedEvent) });
