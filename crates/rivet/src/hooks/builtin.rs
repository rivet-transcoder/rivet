//! The built-in hooks: compute and record, nothing more.

use anyhow::{Context, Result};
use serde_json::{Map, Value};

use super::digest::DigestAlgorithm;
use super::phash::{self, PerceptualAlgorithm};
use super::{
    ArtifactData, ArtifactEvent, ArtifactHook, ArtifactKind, DecodedFrameHook, EncoderFrameHook, FrameEvent,
    FrameSampling, HookContext, HookOutcome, SourceEvent, SourceHook, StillEvent, StillHook,
};

/// A [`SourceHook`] recording content digests of the source bytes, under each
/// algorithm's name (`sha256`, `sha1`, `md5`), and the byte count (`bytes`).
pub struct SourceDigest {
    pub algorithms: Vec<DigestAlgorithm>,
}

impl SourceDigest {
    pub fn new(algorithms: &[DigestAlgorithm]) -> Self {
        Self { algorithms: algorithms.to_vec() }
    }
}

impl SourceHook for SourceDigest {
    fn on_source(&self, _ctx: &HookContext, source: &SourceEvent) -> Result<HookOutcome> {
        Ok(digests(&self.algorithms, &source.bytes, HookOutcome::proceed().annotate("bytes", source.bytes.len())))
    }

    fn describe(&self) -> String {
        format!("source digest ({})", names(&self.algorithms))
    }
}

/// An [`ArtifactHook`] recording content digests of each output of the kinds
/// it accepts. A directory output (an HLS rendition) records an object of
/// file → digest under each algorithm's name; anything else its digest.
pub struct ArtifactDigest {
    pub algorithms: Vec<DigestAlgorithm>,
    pub kinds: Vec<ArtifactKind>,
}

impl ArtifactDigest {
    /// Every kind of output.
    pub fn new(algorithms: &[DigestAlgorithm]) -> Self {
        Self { algorithms: algorithms.to_vec(), kinds: ArtifactKind::ALL.to_vec() }
    }

    /// Only outputs of `kinds`.
    pub fn kinds(mut self, kinds: &[ArtifactKind]) -> Self {
        self.kinds = kinds.to_vec();
        self
    }
}

impl ArtifactHook for ArtifactDigest {
    fn kinds(&self) -> Vec<ArtifactKind> {
        self.kinds.clone()
    }

    fn on_artifact(&self, _ctx: &HookContext, artifact: &ArtifactEvent) -> Result<HookOutcome> {
        let outcome = HookOutcome::proceed();
        Ok(match &artifact.data {
            ArtifactData::Bytes(b) => digests(&self.algorithms, b, outcome.annotate("bytes", b.len())),
            ArtifactData::File(p) => {
                let data = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
                digests(&self.algorithms, &data, outcome.annotate("bytes", data.len()))
            }
            ArtifactData::Directory { path, files } => {
                let mut per_algo: Vec<Map<String, Value>> = vec![Map::new(); self.algorithms.len()];
                for f in files {
                    let full = path.join(f);
                    let data = std::fs::read(&full).with_context(|| format!("reading {}", full.display()))?;
                    for (a, map) in self.algorithms.iter().zip(per_algo.iter_mut()) {
                        map.insert(f.clone(), Value::String(a.hex(&data)));
                    }
                }
                self.algorithms
                    .iter()
                    .zip(per_algo)
                    .fold(outcome, |o, (a, map)| o.annotate(a.as_str(), Value::Object(map)))
            }
        })
    }

    fn describe(&self) -> String {
        let kinds: Vec<&str> = self.kinds.iter().map(|k| k.as_str()).collect();
        format!("artifact digest ({}) of {}", names(&self.algorithms), kinds.join(", "))
    }
}

/// Perceptual hashes ([`phash`]) of pictures, each recorded under its
/// algorithm's name (`phash`, `dhash`, `ahash`) as 16 hex digits.
///
/// One type for each place a picture can be hooked: register it as a
/// [`DecodedFrameHook`] ([`Hooks::decoded_frames`](super::Hooks::decoded_frames))
/// for the source's frames as decoded, an [`EncoderFrameHook`]
/// ([`Hooks::encoder_frames`](super::Hooks::encoder_frames)) for the frames the
/// encoders receive, or a [`StillHook`] ([`Hooks::stills`](super::Hooks::stills))
/// for an image job's pictures. `sampling` applies to the two frame kinds.
pub struct PerceptualFingerprint {
    pub algorithms: Vec<PerceptualAlgorithm>,
    pub sampling: FrameSampling,
}

impl PerceptualFingerprint {
    /// One frame a second.
    pub fn new(algorithms: &[PerceptualAlgorithm]) -> Self {
        Self { algorithms: algorithms.to_vec(), sampling: FrameSampling::default() }
    }

    pub fn sampling(mut self, sampling: FrameSampling) -> Self {
        self.sampling = sampling;
        self
    }

    fn hash(&self, frame: &codec::frame::VideoFrame) -> Result<HookOutcome> {
        let luma = super::frame::luma8(frame)?;
        let (w, h) = (frame.width as usize, frame.height as usize);
        self.algorithms.iter().try_fold(HookOutcome::proceed(), |o, a| {
            Ok(o.annotate(a.as_str(), phash::to_hex(a.hash_luma(&luma, w, h)?)))
        })
    }

    fn describe_at(&self, place: &str) -> String {
        let algos: Vec<&str> = self.algorithms.iter().map(|a| a.as_str()).collect();
        format!("perceptual fingerprint ({}) of {place}", algos.join(", "))
    }
}

impl DecodedFrameHook for PerceptualFingerprint {
    fn sampling(&self) -> FrameSampling {
        self.sampling
    }
    fn on_decoded_frame(&self, _ctx: &HookContext, frame: &FrameEvent) -> Result<HookOutcome> {
        self.hash(&frame.frame)
    }
    fn describe(&self) -> String {
        self.describe_at("decoded frames")
    }
}

impl EncoderFrameHook for PerceptualFingerprint {
    fn sampling(&self) -> FrameSampling {
        self.sampling
    }
    fn on_encoder_frame(&self, _ctx: &HookContext, frame: &FrameEvent) -> Result<HookOutcome> {
        self.hash(&frame.frame)
    }
    fn describe(&self) -> String {
        self.describe_at("encoder frames")
    }
}

impl StillHook for PerceptualFingerprint {
    fn on_still(&self, _ctx: &HookContext, still: &StillEvent) -> Result<HookOutcome> {
        self.hash(&still.frame)
    }
    fn describe(&self) -> String {
        self.describe_at("stills")
    }
}

fn digests(algorithms: &[DigestAlgorithm], data: &[u8], outcome: HookOutcome) -> HookOutcome {
    algorithms.iter().fold(outcome, |o, a| o.annotate(a.as_str(), a.hex(data)))
}

fn names(algorithms: &[DigestAlgorithm]) -> String {
    algorithms.iter().map(|a| a.as_str()).collect::<Vec<_>>().join(", ")
}
