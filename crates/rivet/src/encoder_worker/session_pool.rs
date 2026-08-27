//! One worker's encoder session, kept between chunks.
//!
//! The chunked single-file path used to build a fresh encoder for every
//! chunk. That was the simplest way to make each chunk an independently
//! decodable IDR-led GOP — which the stitcher depends on, because chunks are
//! encoded out of order across cards and concatenated — and it cost a
//! session construction per chunk: on NVENC a CUDA context, a driver session,
//! a capability query, a preset query and two sixteen-slot buffer rings;
//! about 1300 times on a feature-length file.
//!
//! [`Encoder::reset`] is the same guarantee without the construction: the
//! backend restarts its stream in place, so the next frame is an IDR that
//! opens a closed GOP and nothing of the previous chunk — references,
//! rate-control history, queued packets — survives. This pool is the small
//! amount of bookkeeping that lets a worker use it.
//!
//! # Shape
//!
//! One slot per pool, one pool per worker. A ladder worker holds one GPU
//! lease and serves every rung, so it hops between encoder configurations
//! (one per rung); the slot holds the session of whichever rung it served
//! last. A matching configuration is reused after a reset; a different one
//! evicts. Holding a session per rung instead would multiply live driver
//! sessions per card by the ladder depth — consumer NVIDIA drivers cap
//! those, and every one owns two rings of surfaces — for a saving that is
//! only a rung hop, which the scheduler makes rarer than a chunk.
//!
//! # When the backend cannot reset
//!
//! [`ResetUnsupported`] is a type, not a message, and the pool matches on it:
//! that backend is rebuilt for every chunk, silently, exactly as before. A
//! reset that *fails* is also a rebuild, but at `warn`, because it means a
//! backend that claims the capability could not deliver it on this stream.
//!
//! # Counters
//!
//! [`PoolStats`] is the evidence that the pool did what it says: `built`
//! counts constructions, `reused` counts resets that succeeded. On a run
//! that used to construct N encoders, `built + reused == N` and the saving
//! is `reused`. Workers log the stats when they exit.
//!
//! # The control
//!
//! `RIVET_ENCODER_POOL=off` makes every pool drop its session on release, so
//! every chunk builds — the behaviour before this module existed, in the same
//! binary. That is what a before/after measurement compares against; a
//! separate build is a separate set of variables.

use anyhow::{Context, Result};
use codec::encode::{self, Encoder, EncoderConfig, ResetUnsupported};

/// Builds an encoder for a configuration. The default is
/// [`encode::select_encoder`]; tests inject a fake.
pub type EncoderBuilder = dyn Fn(&EncoderConfig) -> Result<Box<dyn Encoder>> + Send;

/// What the pool has done so far. See the module note on what the numbers
/// prove.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Sessions constructed from scratch.
    pub built: u64,
    /// Sessions reused after a successful `reset`.
    pub reused: u64,
    /// Sessions dropped because the next chunk wanted a different
    /// configuration.
    pub evicted: u64,
    /// Sessions dropped because the backend has no reset path
    /// ([`ResetUnsupported`]); each of these was rebuilt.
    pub reset_unsupported: u64,
    /// Sessions dropped because a reset that should have worked failed;
    /// each of these was rebuilt.
    pub reset_failed: u64,
}

struct Pooled {
    config: EncoderConfig,
    encoder: Box<dyn Encoder>,
}

/// One worker's pooled encoder session. Not `Sync`, not shared: each
/// worker thread owns its own.
pub struct EncoderSessionPool {
    slot: Option<Pooled>,
    builder: Box<EncoderBuilder>,
    stats: PoolStats,
    /// `false` under `RIVET_ENCODER_POOL=off`: sessions are dropped on
    /// release and every chunk builds.
    reuse: bool,
}

impl Default for EncoderSessionPool {
    fn default() -> Self {
        Self::new()
    }
}

impl EncoderSessionPool {
    /// A pool that builds through [`encode::select_encoder`]. Honours
    /// `RIVET_ENCODER_POOL=off` (see the module note).
    pub fn new() -> Self {
        let mut pool = Self::with_builder(Box::new(|config: &EncoderConfig| {
            encode::select_encoder(config.clone(), None)
        }));
        if std::env::var("RIVET_ENCODER_POOL").is_ok_and(|v| v == "off") {
            pool.reuse = false;
            tracing::info!(
                event = "encoder_pool.disabled",
                "RIVET_ENCODER_POOL=off: encoder sessions will not be reused; every chunk builds"
            );
        }
        pool
    }

    /// A pool that builds through `builder`. For tests, and for callers
    /// that pin a backend by name.
    pub fn with_builder(builder: Box<EncoderBuilder>) -> Self {
        Self { slot: None, builder, stats: PoolStats::default(), reuse: true }
    }

    /// The same pool with reuse switched off: the control path, for a
    /// measurement or a test.
    pub fn without_reuse(mut self) -> Self {
        self.reuse = false;
        self
    }

    /// A session ready to encode a new stream for `config`: the pooled one,
    /// reset, when its configuration matches; otherwise a new one.
    ///
    /// The pool is empty on return — the caller owns the session until it
    /// hands it back with [`release`](Self::release). A session that is not
    /// released (the chunk failed) is simply gone, and the next call builds.
    pub fn acquire(&mut self, config: &EncoderConfig) -> Result<Box<dyn Encoder>> {
        if let Some(pooled) = self.slot.take() {
            if pooled.config == *config {
                let mut encoder = pooled.encoder;
                match encoder.reset() {
                    Ok(()) => {
                        self.stats.reused += 1;
                        tracing::debug!(
                            event = "encoder_pool.reuse",
                            reused = self.stats.reused,
                            built = self.stats.built,
                            "encoder session reused after reset"
                        );
                        return Ok(encoder);
                    }
                    Err(e) if e.downcast_ref::<ResetUnsupported>().is_some() => {
                        self.stats.reset_unsupported += 1;
                        tracing::debug!(
                            event = "encoder_pool.rebuild",
                            "encoder backend cannot reset; rebuilding for this chunk"
                        );
                    }
                    Err(e) => {
                        self.stats.reset_failed += 1;
                        tracing::warn!(
                            event = "encoder_pool.reset_failed",
                            error = %format!("{e:#}"),
                            "encoder session reset failed; rebuilding for this chunk"
                        );
                    }
                }
                // Dropped here: a session whose reset was refused or failed
                // is not one to encode another stream on.
            } else {
                self.stats.evicted += 1;
                tracing::debug!(
                    event = "encoder_pool.evict",
                    from = %describe(&pooled.config),
                    to = %describe(config),
                    "encoder session evicted: next chunk wants a different configuration"
                );
                drop(pooled);
            }
        }
        let encoder = (self.builder)(config).context("creating encoder for chunk")?;
        self.stats.built += 1;
        tracing::debug!(
            event = "encoder_pool.build",
            built = self.stats.built,
            reused = self.stats.reused,
            config = %describe(config),
            "encoder session built"
        );
        Ok(encoder)
    }

    /// Hand a session back after its stream has been flushed and drained.
    /// Replaces whatever was in the slot (there is nothing there after
    /// `acquire`, so nothing is lost).
    pub fn release(&mut self, config: &EncoderConfig, encoder: Box<dyn Encoder>) {
        if self.reuse {
            self.slot = Some(Pooled { config: config.clone(), encoder });
        }
        // Otherwise dropped here: the control path builds every chunk.
    }

    /// Whether a session is being held.
    pub fn is_empty(&self) -> bool {
        self.slot.is_none()
    }

    /// The counters so far.
    pub fn stats(&self) -> PoolStats {
        self.stats
    }
}

/// The parts of a configuration that tell two rungs apart, for the log.
fn describe(c: &EncoderConfig) -> String {
    format!("{:?} {}x{} gpu={:?}", c.codec, c.width, c.height, c.gpu_index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use codec::encode::EncodedPacket;
    use codec::frame::VideoFrame;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A backend that only counts. `reset` behaves as configured.
    struct Fake {
        resets: Arc<AtomicUsize>,
        reset: FakeReset,
    }

    #[derive(Clone, Copy)]
    enum FakeReset {
        Works,
        Unsupported,
        Fails,
    }

    impl Encoder for Fake {
        fn send_frame(&mut self, _: &VideoFrame) -> Result<()> {
            Ok(())
        }
        fn flush(&mut self) -> Result<()> {
            Ok(())
        }
        fn receive_packet(&mut self) -> Result<Option<EncodedPacket>> {
            Ok(None)
        }
        fn reset(&mut self) -> Result<()> {
            match self.reset {
                FakeReset::Works => {
                    self.resets.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
                FakeReset::Unsupported => Err(ResetUnsupported.into()),
                FakeReset::Fails => anyhow::bail!("driver said no"),
            }
        }
    }

    fn pool(reset: FakeReset) -> (EncoderSessionPool, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let built = Arc::new(AtomicUsize::new(0));
        let resets = Arc::new(AtomicUsize::new(0));
        let (b, r) = (Arc::clone(&built), Arc::clone(&resets));
        let pool = EncoderSessionPool::with_builder(Box::new(move |_| {
            b.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(Fake { resets: Arc::clone(&r), reset }) as Box<dyn Encoder>)
        }));
        (pool, built, resets)
    }

    fn config(width: u32) -> EncoderConfig {
        EncoderConfig { width, height: 16, ..EncoderConfig::default() }
    }

    #[test]
    fn a_matching_config_is_reused_after_one_reset() {
        let (mut pool, built, resets) = pool(FakeReset::Works);
        let a = config(16);
        let enc = pool.acquire(&a).unwrap();
        assert!(pool.is_empty(), "acquire hands the session out");
        pool.release(&a, enc);
        let _enc = pool.acquire(&a).unwrap();
        assert_eq!(built.load(Ordering::SeqCst), 1, "one construction for two chunks");
        assert_eq!(resets.load(Ordering::SeqCst), 1, "the reuse went through reset");
        assert_eq!(pool.stats(), PoolStats { built: 1, reused: 1, ..Default::default() });
    }

    #[test]
    fn a_different_config_evicts_and_builds() {
        let (mut pool, built, resets) = pool(FakeReset::Works);
        let (a, b) = (config(16), config(32));
        let enc = pool.acquire(&a).unwrap();
        pool.release(&a, enc);
        let _enc = pool.acquire(&b).unwrap();
        assert_eq!(built.load(Ordering::SeqCst), 2);
        assert_eq!(resets.load(Ordering::SeqCst), 0, "an evicted session is not reset");
        assert_eq!(pool.stats(), PoolStats { built: 2, evicted: 1, ..Default::default() });
    }

    #[test]
    fn a_backend_without_reset_is_rebuilt_every_time() {
        let (mut pool, built, _) = pool(FakeReset::Unsupported);
        let a = config(16);
        for _ in 0..3 {
            let enc = pool.acquire(&a).unwrap();
            pool.release(&a, enc);
        }
        assert_eq!(built.load(Ordering::SeqCst), 3, "no reset path means one build per chunk");
        assert_eq!(pool.stats(), PoolStats { built: 3, reset_unsupported: 2, ..Default::default() });
    }

    #[test]
    fn a_failed_reset_is_rebuilt_and_counted_separately() {
        let (mut pool, built, _) = pool(FakeReset::Fails);
        let a = config(16);
        let enc = pool.acquire(&a).unwrap();
        pool.release(&a, enc);
        let _enc = pool.acquire(&a).unwrap();
        assert_eq!(built.load(Ordering::SeqCst), 2);
        assert_eq!(pool.stats(), PoolStats { built: 2, reset_failed: 1, ..Default::default() });
    }

    #[test]
    fn a_session_not_released_is_gone() {
        // The chunk failed and the encoder was dropped with it: the next
        // chunk builds rather than finding a half-used session.
        let (mut pool, built, resets) = pool(FakeReset::Works);
        let a = config(16);
        let enc = pool.acquire(&a).unwrap();
        drop(enc);
        let _enc = pool.acquire(&a).unwrap();
        assert_eq!(built.load(Ordering::SeqCst), 2);
        assert_eq!(resets.load(Ordering::SeqCst), 0);
        assert_eq!(pool.stats().reused, 0);
    }

    #[test]
    fn the_control_path_builds_every_chunk_and_never_resets() {
        let (pool, built, resets) = pool(FakeReset::Works);
        let mut pool = pool.without_reuse();
        let a = config(16);
        for _ in 0..3 {
            let enc = pool.acquire(&a).unwrap();
            pool.release(&a, enc);
            assert!(pool.is_empty(), "nothing is kept on the control path");
        }
        assert_eq!(built.load(Ordering::SeqCst), 3);
        assert_eq!(resets.load(Ordering::SeqCst), 0);
        assert_eq!(pool.stats(), PoolStats { built: 3, ..Default::default() });
    }

    #[test]
    fn the_builder_error_names_the_chunk() {
        let mut pool = EncoderSessionPool::with_builder(Box::new(|_| anyhow::bail!("no silicon")));
        let err = match pool.acquire(&config(16)) {
            Ok(_) => panic!("a builder that fails must fail the acquire"),
            Err(e) => e,
        };
        assert!(format!("{err:#}").contains("creating encoder for chunk"), "{err:#}");
        assert!(format!("{err:#}").contains("no silicon"), "{err:#}");
        assert_eq!(pool.stats().built, 0, "a failed build is not a build");
    }
}
