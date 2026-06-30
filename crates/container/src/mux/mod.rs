use anyhow::{Context, Result};
use bytes::Bytes;
use codec::encode::EncodedPacket;
use codec::frame::{ColorMetadata, VideoCodec};

use crate::nal_mux::{NalMuxCodec, NalSampleWriter};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use tempfile::NamedTempFile;

use crate::AudioInfo;

mod boxes;
mod video_track;
mod audio_track;
mod sample_table;
mod mdat;
#[cfg(test)]
mod tests;

// Re-exports for external crate callers that import from `container::mux::*`.
pub(crate) use boxes::{BoxBuilder, write_unity_matrix, extract_sequence_header};
pub(crate) use video_track::{build_av01, build_avc1, build_hvc1, build_avcc, build_hvcc};
pub(crate) use audio_track::build_audio_stsd;
pub use audio_track::{dac3_body_from_sync, dec3_body_from_sync};

// Internal imports used by impl Av1Mp4Muxer below.
use boxes::{build_ftyp, build_moov_any};
use sample_table::{AudioBuildPlan, chunk_count_of, plan_interleaved_layout};

/// Streams mdat payload bytes to a tempfile while keeping only small
/// per-packet metadata vectors in RAM. At 15 min 1080p60 and ~500 kB/sample
/// average the metadata Vecs are ~700 KB total; the packet payload (~500 MB
/// per variant at AV1 CQ 32) stays on disk.
///
/// Faststart is preserved: `finalize_to_file` writes ftyp + moov first,
/// then streams the tempfile's mdat bytes into the final output.
///
/// API:
/// - `new(w, h, fps)` — constructs a spooled muxer, creating the tempfile
///   immediately. Fails if tempdir is unwritable.
/// - `add_packet(packet)` — appends packet payload to the tempfile and
///   records size/sync metadata.
/// - `with_audio(info)` — registers an optional audio track. Codec dispatch
///   happens here on `info.codec` (`"aac"` / `"opus"` / `"ac3"` / `"eac3"`).
///   Must be called before `add_audio_sample`. Bails on unsupported codecs
///   or channel counts — no silent degradation.
/// - `add_audio_sample(sample, pts_ticks, duration_ticks)` — appends one
///   audio access unit plus per-sample metadata. Requires `with_audio`
///   first.
/// - `finalize_to_file(&Path)` — writes ftyp + moov + mdat payload to the
///   target path. Consumes self.
/// - `finalize()` — backward-compat shim that reads the finalized file into
///   a `Bytes`. Useful for small tests; callers hitting the RAM ceiling
///   should use `finalize_to_file` + `ObjectStore::upload_file`.
pub struct Av1Mp4Muxer {
    width: u32,
    height: u32,
    frame_rate: f64,
    mdat_tmp: NamedTempFile,
    mdat_writer: BufWriter<File>,
    sample_sizes: Vec<u32>,
    keyframe_indices: Vec<u32>,
    first_packet_header: Option<Vec<u8>>,
    packet_count: u32,
    mdat_payload_bytes: u64,
    audio: Option<AudioTrackState>,
    /// Color metadata copied from the source `StreamInfo` so the visual
    /// sample entry can carry an Apple-compliant `colr nclx` box. Defaults
    /// to BT.709 SDR limited-range — Apple silently assumes that when
    /// `colr` is absent, so the default is correct for SDR sources but
    /// breaks BT.2020 / HDR clips. Real values arrive via `with_color`.
    color_metadata: ColorMetadata,
    /// Test-only override forcing the muxer to emit the 64-bit `largesize`
    /// mdat header even when the payload would fit in the 32-bit `size`
    /// field. Pre-existing payload size computation otherwise leaves the
    /// largesize branch untestable without producing a 4 GiB tempfile.
    /// Production callers leave this `false`; tests flip it on to assert
    /// the bit-layout of the largesize header is correct.
    ///
    /// Must be a regular field (not `#[cfg(test)]`-gated) so integration
    /// tests in `tests/` — which compile against the release library
    /// without `cfg(test)` — can flip it via `force_largesize_mdat_for_test`.
    #[doc(hidden)]
    force_largesize_mdat: bool,
    /// Output video codec. Drives the sample-entry fourcc + config box at
    /// finalize (`av01`/`av1C`, `avc1`/`avcC`, or `hvc1`/`hvcC`).
    codec: VideoCodec,
    /// For H.264 / H.265: repackages the encoder's Annex-B frames into
    /// length-prefixed mdat samples and collects the SPS/PPS(/VPS) for the
    /// config box. `None` for AV1 (which stores OBUs verbatim).
    nal_writer: Option<NalSampleWriter>,
    /// Inline-parameter-set mode (H.264/H.265 multi-GPU stitch): keep SPS/PPS
    /// inline per access unit + emit the `avc3`/`hev1` sample entry instead of
    /// `avc1`/`hvc1`, so chunks from independent encoders self-describe.
    inline_param_sets: bool,
}

/// Per-muxer audio track state: info + spooling tempfile + per-sample
/// metadata. Kept internal; populated via `with_audio` + `add_audio_sample`.
struct AudioTrackState {
    info: AudioInfo,
    audio_tmp: NamedTempFile,
    audio_writer: BufWriter<File>,
    sample_sizes: Vec<u32>,
    durations: Vec<u32>,
    total_duration_ticks: u64,
    mdat_payload_bytes: u64,
}

/// Internal discriminator chosen at `with_audio` time. Saves us re-parsing
/// the codec string at every builder call site (build_audio_stsd, etc.) and
/// keeps the AAC / Opus / AC-3 / E-AC-3 dispatch in one place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AudioCodecKind {
    Aac,
    Opus,
    Ac3,
    Eac3,
}

impl AudioCodecKind {
    pub(super) fn from_codec_tag(codec: &str) -> Option<Self> {
        if codec.eq_ignore_ascii_case("aac") {
            Some(Self::Aac)
        } else if codec.eq_ignore_ascii_case("opus") {
            Some(Self::Opus)
        } else if codec.eq_ignore_ascii_case("ac3") || codec.eq_ignore_ascii_case("ac-3") {
            Some(Self::Ac3)
        } else if codec.eq_ignore_ascii_case("eac3") || codec.eq_ignore_ascii_case("e-ac-3") {
            Some(Self::Eac3)
        } else {
            None
        }
    }
}

impl Av1Mp4Muxer {
    /// AV1 muxer (the default + back-compatible constructor).
    pub fn new(width: u32, height: u32, frame_rate: f64) -> Result<Self> {
        Self::new_with_codec(width, height, frame_rate, VideoCodec::Av1)
    }

    /// Muxer for the given output `codec` — `Av1` (`av01`/`av1C`), `H264`
    /// (`avc1`/`avcC`), or `H265` (`hvc1`/`hvcC`). H.264/H.265 callers feed the
    /// encoder's **Annex-B** packets; the muxer repackages them to
    /// length-prefixed samples + collects the parameter sets.
    pub fn new_with_codec(
        width: u32,
        height: u32,
        frame_rate: f64,
        codec: VideoCodec,
    ) -> Result<Self> {
        Self::new_with_codec_opts(width, height, frame_rate, codec, false)
    }

    /// Like [`new_with_codec`] but with **inline parameter sets** for H.264/H.265
    /// (the multi-GPU stitch). Each access unit keeps its own SPS/PPS(/VPS) and
    /// the sample entry is `avc3`/`hev1`, so chunks from independent encoders
    /// (possibly different vendors) decode with their own parameter sets.
    pub fn new_with_codec_inline(
        width: u32,
        height: u32,
        frame_rate: f64,
        codec: VideoCodec,
    ) -> Result<Self> {
        Self::new_with_codec_opts(width, height, frame_rate, codec, true)
    }

    fn new_with_codec_opts(
        width: u32,
        height: u32,
        frame_rate: f64,
        codec: VideoCodec,
        inline_param_sets: bool,
    ) -> Result<Self> {
        let mdat_tmp = NamedTempFile::new().context("creating mdat tempfile")?;
        let handle = mdat_tmp
            .reopen()
            .context("reopening mdat tempfile for write")?;
        let mdat_writer = BufWriter::new(handle);
        let make = |c: NalMuxCodec| {
            if inline_param_sets {
                NalSampleWriter::new_inline(c)
            } else {
                NalSampleWriter::new(c)
            }
        };
        let nal_writer = match codec {
            VideoCodec::Av1 => None,
            VideoCodec::H264 => Some(make(NalMuxCodec::H264)),
            VideoCodec::H265 => Some(make(NalMuxCodec::H265)),
        };
        Ok(Self {
            width,
            height,
            frame_rate,
            mdat_tmp,
            mdat_writer,
            sample_sizes: Vec::new(),
            keyframe_indices: Vec::new(),
            first_packet_header: None,
            packet_count: 0,
            mdat_payload_bytes: 0,
            audio: None,
            color_metadata: ColorMetadata::default(),
            force_largesize_mdat: false,
            codec,
            nal_writer,
            inline_param_sets,
        })
    }

    /// Test-only knob to exercise the 64-bit mdat largesize header without
    /// crafting a multi-GiB payload. Production callers do not touch this —
    /// the natural threshold (`mdat_payload + 8 > u32::MAX`) selects
    /// largesize when the file genuinely needs it.
    #[doc(hidden)]
    pub fn force_largesize_mdat_for_test(&mut self) -> &mut Self {
        self.force_largesize_mdat = true;
        self
    }

    /// Carry the source's color metadata into the visual sample entry's
    /// `colr nclx` box. Apple QuickTime / iOS Safari silently assume
    /// BT.709 limited-range when `colr` is missing, which corrupts
    /// BT.2020 HDR / wide-gamut clips. Pipeline calls this once after
    /// demux but before any `add_packet` — though calling order is
    /// not load-bearing because the metadata is only consumed by the
    /// finalize-time `build_av01` builder.
    pub fn set_color_metadata(&mut self, color_metadata: ColorMetadata) -> &mut Self {
        self.color_metadata = color_metadata;
        self
    }

    pub fn add_packet(&mut self, packet: EncodedPacket) -> Result<()> {
        // AV1: store the OBU stream verbatim (the first packet carries the
        // sequence header we embed in av1C). H.264/H.265: repackage the
        // Annex-B frame into a length-prefixed mdat sample, capturing the
        // parameter sets for the avcC/hvcC config box.
        match &mut self.nal_writer {
            None => {
                // AV1: one OBU sample per packet.
                if self.first_packet_header.is_none() {
                    self.first_packet_header = Some(packet.data.to_vec());
                }
                self.write_sample(&packet.data.clone(), packet.is_keyframe)?;
            }
            Some(_) => {
                // H.264/H.265: a packet may carry several access units; split it
                // into one length-prefixed sample per frame (per-AU keyframe).
                let writer = self.nal_writer.as_mut().unwrap();
                let samples = writer.push_packet(&packet.data);
                for au in samples {
                    self.write_sample(&au.data, au.is_keyframe)?;
                }
            }
        }
        Ok(())
    }

    /// Append one finished sample to the mdat tempfile + update the per-sample
    /// tables (size, keyframe index, payload total).
    fn write_sample(&mut self, sample: &[u8], is_keyframe: bool) -> Result<()> {
        let size = sample.len() as u32;
        self.mdat_writer
            .write_all(sample)
            .context("writing sample to mdat tempfile")?;
        self.sample_sizes.push(size);
        self.packet_count = self
            .packet_count
            .checked_add(1)
            .context("packet count overflow")?;
        if is_keyframe {
            self.keyframe_indices.push(self.packet_count);
        }
        self.mdat_payload_bytes = self
            .mdat_payload_bytes
            .checked_add(size as u64)
            .context("mdat payload overflow")?;
        Ok(())
    }

    /// before `add_audio_sample`. Validates codec ∈ {AAC family, Opus,
    /// AC-3, E-AC-3} with codec-appropriate channel-count gates —
    /// anything outside the supported envelope must fail loudly (no
    /// silent degradation, no stubs).
    ///
    /// AAC family path (Squad-18 + Squad-25): emits `mp4a` sample entry +
    /// `esds` descriptor tree carrying the AudioSpecificConfig verbatim,
    /// plus an Apple `chan` (Channel Layout) box for ≥3-channel streams
    /// so iOS Safari / QuickTime / AVFoundation render the correct layout
    /// instead of defaulting to L+R. Accepts:
    ///   - AAC-LC (AOT=2), mono / stereo / 5.1 / 7.1
    ///   - HE-AAC v1 (explicit-signaled SBR; ASC starts AOT=5)
    ///   - HE-AAC v2 (explicit-signaled PS; ASC starts AOT=29)
    ///
    /// Implicit-signaled HE-AAC (AOT=2 leading byte at low core rate ≤24 kHz)
    /// is rejected — the caller (`pipeline::transcode::route_audio`) is
    /// responsible for upgrading the ASC via
    /// `aac_asc::upgrade_to_explicit_signaling` before reaching the mux.
    ///
    /// Opus path (Squad-23 + Squad-28, RFC 7845): emits `Opus` sample entry
    /// + `dOps` (Opus-Specific Box) carrying the OpusHead body verbatim.
    /// Mono / stereo via ChannelMappingFamily=0 (Squad-23) or 3..=8
    /// channels via ChannelMappingFamily=1 surround layouts (Squad-28).
    /// Requires `info.codec_private` populated with the appropriate-form
    /// OpusHead body. The mdhd timescale is pinned to 48000 per RFC 7845
    /// §3 — the `info.timescale` is validated equal.
    ///
    /// AC-3 path (Squad-26, ETSI TS 102 366 §F.2): emits `ac-3` sample
    /// entry + `dac3` config box carrying the 3-byte body verbatim. Up
    /// to 5.1 channels. Sample rates 32 / 44.1 / 48 kHz only.
    ///
    /// E-AC-3 path (Squad-26, ETSI TS 102 366 §F.5): emits `ec-3` sample
    /// entry + `dec3` config box. Up to 5.1 channels in v1 scope (single
    /// independent substream). Sample rates 16 / 22.05 / 24 / 32 / 44.1 /
    /// 48 kHz.
    ///
    /// Returns `&mut Self` for builder-style chaining. The audio tempfile
    /// is created eagerly so tempdir failures surface here rather than at
    /// `add_audio_sample` time.
    pub fn with_audio(&mut self, info: AudioInfo) -> Result<&mut Self> {
        // Codec dispatch: AAC, Opus, AC-3, E-AC-3 are the supported
        // families. Other codec tags (mp3, vorbis, ...) are intentionally
        // rejected here so the pipeline fall-back path in `transcode.rs` can
        // surface a clean warn and emit video-only.
        let codec_kind = AudioCodecKind::from_codec_tag(&info.codec).ok_or_else(|| {
            anyhow::anyhow!(
                "audio mux: only AAC-LC, Opus, AC-3, E-AC-3 are supported; got codec '{}'",
                info.codec
            )
        })?;
        // Per-codec channel-count gates.
        // - AAC: standard MPEG channelConfiguration values 1 (mono) /
        //   2 (stereo) / 6 (5.1) / 7 (7.1). Multichannel adds an Apple
        //   `chan` box (Squad-25) for QuickTime / AVFoundation rendering.
        // - Opus: 1..=8. Mono/stereo via ChannelMappingFamily=0 (Squad-23);
        //   3..=8 ride the dOps family-1 surround trailer per RFC 7845
        //   §5.1.1.2 (Squad-28 multistream).
        // - AC-3 / E-AC-3: up to 6 channels (5.1). The real layout lives
        //   in `acmod`+`lfeon` inside the dac3/dec3 body; the
        //   AudioSampleEntry channelcount is informational. v1 scope keeps
        //   things tight at 5.1.
        match codec_kind {
            AudioCodecKind::Aac => {
                if !matches!(info.channels, 1 | 2 | 6 | 7) {
                    anyhow::bail!(
                        "audio mux: AAC supports mono/stereo/5.1(channels=6)/7.1(channels=7) layouts; \
                         got {} channels — extended Atmos / object layouts are not supported",
                        info.channels
                    );
                }
            }
            AudioCodecKind::Opus => {
                if info.channels < 1 || info.channels > 8 {
                    anyhow::bail!(
                        "audio mux: Opus supports 1..=8 channels; got {}",
                        info.channels
                    );
                }
            }
            AudioCodecKind::Ac3 | AudioCodecKind::Eac3 => {
                if !(1..=6).contains(&info.channels) {
                    anyhow::bail!(
                        "audio mux: AC-3 / E-AC-3 channel count must be 1..=6 (mono..5.1); got {}",
                        info.channels
                    );
                }
            }
        }
        if info.sample_rate == 0 {
            anyhow::bail!("audio mux: sample_rate must be > 0");
        }
        if info.timescale == 0 {
            anyhow::bail!("audio mux: timescale must be > 0");
        }
        match codec_kind {
            AudioCodecKind::Aac => {
                if info.asc_bytes.is_empty() {
                    anyhow::bail!("audio mux: AudioSpecificConfig bytes missing");
                }
                // Parse the ASC's leading AOT (with the 5-bit raw + 6-bit
                // extension escape per ISO 14496-3 §1.6.2.1) so HE-AAC
                // explicit signaling isn't rejected by a naive `>>3 & 0x1F`
                // peek. Squad-25 lifts the prior AAC-LC-only gate.
                let parsed = crate::aac_asc::parse_aac_asc(&info.asc_bytes)
                    .with_context(|| "audio mux: failed to parse AudioSpecificConfig")?;
                use crate::aac_asc::AscSignaling;
                match parsed.signaling {
                    AscSignaling::ImplicitMaybe => {
                        anyhow::bail!(
                            "audio mux: ASC uses implicit HE-AAC signaling (AOT=2 core at \
                             {} Hz with no SBR/PS layer in the ASC). Apple players silently \
                             downgrade to mono 22.05 kHz core. Caller must upgrade with \
                             aac_asc::upgrade_to_explicit_signaling before muxing.",
                            parsed.sample_rate
                        );
                    }
                    AscSignaling::NoExtension
                    | AscSignaling::ExplicitSbr
                    | AscSignaling::ExplicitPs => {
                        // AOT=2 (LC), AOT=5 (SBR-wrapped LC), AOT=29 (PS-wrapped LC),
                        // and AOT=42 (xHE-AAC USAC) are all accepted at the mux
                        // level. The `esds` writer emits the ASC verbatim so the
                        // decoder receives whatever signaling the ASC carries.
                        let core_aot = parsed.aot;
                        if !matches!(core_aot, 2 | 42) {
                            anyhow::bail!(
                                "audio mux: only AAC-LC (AOT=2) and xHE-AAC USAC (AOT=42) \
                                 cores are supported; ASC core AOT={}",
                                core_aot
                            );
                        }
                    }
                }
            }
            AudioCodecKind::Opus => {
                // OpusHead body without the 8-byte 'OpusHead' magic is 11
                // bytes minimum for ChannelMappingFamily=0 (RFC 7845 §5.1).
                // Reject anything shorter — the dOps writer can't synthesize
                // a missing field and producing an empty box would silently
                // break every player.
                if info.codec_private.len() < 11 {
                    anyhow::bail!(
                        "audio mux: Opus codec_private must be ≥11 bytes (RFC 7845 §5.1 \
                         minimum body for ChannelMappingFamily=0); got {} bytes",
                        info.codec_private.len()
                    );
                }
                // RFC 7845 §3: the audio mdhd timescale MUST be 48000 for
                // Opus. The CALLER pins this in `AudioInfo::opus(...)`; if
                // they hand-built an `AudioInfo` with a different timescale
                // we reject loudly so a downstream stts mismatch can't
                // silently shift PTS by a small fraction.
                if info.timescale != 48_000 {
                    anyhow::bail!(
                        "audio mux: Opus mdhd timescale must be 48000 (RFC 7845 §3); \
                         got timescale={}",
                        info.timescale
                    );
                }
                // ChannelMappingFamily byte (offset 10 in the OpusHead body
                // we emit into dOps). Family 0 is mono/stereo (1..=2
                // channels). Family 1 (Squad-28) is surround for 1..=8
                // channels; requires a 2 + N byte trailer
                // (StreamCount + CoupledCount + ChannelMapping[N]) per
                // RFC 7845 §5.1.1. Family 255 (arbitrary mappings) and
                // any other unknown family are rejected.
                let cmf = info.codec_private[10];
                match cmf {
                    0 => {
                        // RFC 7845 §5.1.1: family 0 is defined for
                        // 1..=2 channels only.
                        if info.channels > 2 {
                            anyhow::bail!(
                                "audio mux: Opus ChannelMappingFamily=0 only supports 1..=2 channels; got {}",
                                info.channels
                            );
                        }
                    }
                    1 => {
                        // Family 1 needs StreamCount + CoupledCount +
                        // ChannelMapping[channels] after the 11-byte
                        // preamble. Total dOps body = 11 + 2 + N.
                        let n = info.channels as usize;
                        let needed = 11 + 2 + n;
                        if info.codec_private.len() < needed {
                            anyhow::bail!(
                                "audio mux: Opus family=1 codec_private must be ≥{needed} bytes \
                                 (11 preamble + 2 stream/coupled + {n} mapping); got {}",
                                info.codec_private.len()
                            );
                        }
                        let stream_count = info.codec_private[11];
                        let coupled_count = info.codec_private[12];
                        // libopus invariants (RFC 7845 §5.1.1):
                        //   - StreamCount >= 1
                        //   - CoupledCount <= StreamCount
                        //   - StreamCount + CoupledCount <= 255 (always
                        //     true at our scale)
                        //   - StreamCount + CoupledCount <= channels
                        //     (every encoder stream covers >=1 channel)
                        if stream_count < 1 {
                            anyhow::bail!(
                                "audio mux: Opus family=1 StreamCount must be >= 1; got {stream_count}"
                            );
                        }
                        if coupled_count > stream_count {
                            anyhow::bail!(
                                "audio mux: Opus family=1 CoupledCount ({coupled_count}) > StreamCount ({stream_count})"
                            );
                        }
                        if (stream_count as u16) + (coupled_count as u16) > info.channels {
                            anyhow::bail!(
                                "audio mux: Opus family=1 StreamCount ({stream_count}) + CoupledCount ({coupled_count}) > channels ({})",
                                info.channels
                            );
                        }
                        // ChannelMapping[i] must be < streams +
                        // coupled (i.e. a valid encoder-stream index).
                        let mapping_max = stream_count + coupled_count;
                        for i in 0..n {
                            let m = info.codec_private[13 + i];
                            if m >= mapping_max {
                                anyhow::bail!(
                                    "audio mux: Opus family=1 ChannelMapping[{i}]={m} \
                                     exceeds streams+coupled ({mapping_max})"
                                );
                            }
                        }
                    }
                    other => {
                        anyhow::bail!(
                            "audio mux: only Opus ChannelMappingFamily 0 (mono/stereo) and 1 (surround 1..=8) supported; \
                             got family={other}"
                        );
                    }
                }
            }
            AudioCodecKind::Ac3 => {
                // dac3 body is exactly 3 bytes per ETSI TS 102 366 §F.4
                // (fscod 2b | bsid 5b | bsmod 3b | acmod 3b | lfeon 1b |
                //  bit_rate_code 5b | reserved 5b => 24 bits total).
                if info.codec_private.len() != 3 {
                    anyhow::bail!(
                        "audio mux: AC-3 codec_private (dac3 body) must be exactly 3 bytes \
                         per ETSI TS 102 366 §F.4; got {} bytes",
                        info.codec_private.len()
                    );
                }
                // Sample rate sanity per ETSI TS 102 366 Table F.5.
                match info.sample_rate {
                    32_000 | 44_100 | 48_000 => {}
                    other => anyhow::bail!(
                        "audio mux: AC-3 sample_rate must be 32000 / 44100 / 48000; got {}",
                        other
                    ),
                }
            }
            AudioCodecKind::Eac3 => {
                // dec3 body is variable-size; minimum body is 5 bytes for a
                // single independent substream with no dependent substreams
                // (data_rate 13b + num_ind_sub 3b = 2B + per-indep-substream
                //  fscod/bsid/asvc/bsmod/acmod/lfeon/num_dep_sub fields
                //  packed into the next 3 bytes). Reject anything shorter.
                if info.codec_private.len() < 5 {
                    anyhow::bail!(
                        "audio mux: E-AC-3 codec_private (dec3 body) must be ≥5 bytes \
                         per ETSI TS 102 366 §F.6; got {} bytes",
                        info.codec_private.len()
                    );
                }
                // E-AC-3 sample rates: 32 / 44.1 / 48 kHz at "full" rate
                // plus 16 / 22.05 / 24 kHz "reduced rate" (fscod==3 path).
                match info.sample_rate {
                    16_000 | 22_050 | 24_000 | 32_000 | 44_100 | 48_000 => {}
                    other => anyhow::bail!(
                        "audio mux: E-AC-3 sample_rate must be 16000 / 22050 / 24000 / 32000 / \
                         44100 / 48000; got {}",
                        other
                    ),
                }
            }
        }
        if self.audio.is_some() {
            anyhow::bail!("audio mux: with_audio called twice");
        }
        let audio_tmp = NamedTempFile::new().context("creating audio mdat tempfile")?;
        let handle = audio_tmp
            .reopen()
            .context("reopening audio tempfile for write")?;
        let audio_writer = BufWriter::new(handle);
        self.audio = Some(AudioTrackState {
            info,
            audio_tmp,
            audio_writer,
            sample_sizes: Vec::new(),
            durations: Vec::new(),
            total_duration_ticks: 0,
            mdat_payload_bytes: 0,
        });
        Ok(self)
    }

    /// Append one audio access unit (AAC AU / Opus packet / AC-3 syncframe /
    /// E-AC-3 syncframe). `pts_ticks` is currently informational only —
    /// ISOBMFF doesn't store per-sample PTS directly; stts durations imply
    /// a running clock starting at 0. We accept it in the API to keep the
    /// signature extensible (edit-lists / ctts for offset signalling can
    /// land here later).
    pub fn add_audio_sample(
        &mut self,
        sample: &[u8],
        _pts_ticks: u64,
        duration_ticks: u32,
    ) -> Result<()> {
        let audio = self
            .audio
            .as_mut()
            .context("audio mux: add_audio_sample called before with_audio")?;
        if sample.is_empty() {
            anyhow::bail!("audio mux: refusing to add empty audio access unit");
        }
        audio
            .audio_writer
            .write_all(sample)
            .context("writing audio sample to tempfile")?;
        audio.sample_sizes.push(sample.len() as u32);
        let dur = if duration_ticks == 0 {
            // Codec-aware default frame duration. AAC: 1024 samples (the
            // natural transform length); Opus: 960 ticks @ 48 kHz = 20 ms
            // (the standard libopus encoder frame size); AC-3: 1536 samples
            // per syncframe (6 blocks × 256 samples per ETSI TS 102 366);
            // E-AC-3: 1536 samples for the dominant numblkscod=3 / 6-block
            // case (other numblkscod values would be 256/512/768 — caller
            // should override). Most common defaults; callers can override
            // with an explicit non-zero `duration_ticks`.
            match AudioCodecKind::from_codec_tag(&audio.info.codec) {
                Some(AudioCodecKind::Aac) => 1024,
                Some(AudioCodecKind::Opus) => 960,
                Some(AudioCodecKind::Ac3) | Some(AudioCodecKind::Eac3) => 1536,
                None => 1024, // unreachable: with_audio gates the codec tag
            }
        } else {
            duration_ticks
        };
        audio.durations.push(dur);
        audio.total_duration_ticks = audio
            .total_duration_ticks
            .checked_add(dur as u64)
            .context("audio total duration overflow")?;
        audio.mdat_payload_bytes = audio
            .mdat_payload_bytes
            .checked_add(sample.len() as u64)
            .context("audio mdat payload overflow")?;
        Ok(())
    }

    /// Write ftyp + moov + mdat into `output_path`. Faststart preserved.
    ///
    /// When audio is present (via `with_audio`), writes an interleaved mdat
    /// with chunk-alternation: one ~1s video chunk then one ~1s audio chunk,
    /// repeated until both tracks are drained. stco/co64 entries in each
    /// trak's stbl point at the first sample of that trak's chunk inside
    /// the shared mdat.
    pub fn finalize_to_file(mut self, output_path: &Path) -> Result<()> {
        if self.packet_count == 0 {
            anyhow::bail!("cannot finalize MP4 with zero packets");
        }
        self.mdat_writer.flush().context("flushing mdat tempfile")?;
        if let Some(ref mut audio) = self.audio {
            audio
                .audio_writer
                .flush()
                .context("flushing audio mdat tempfile")?;
            if audio.sample_sizes.is_empty() {
                // Caller called with_audio but never pushed a sample. Safer
                // to drop the audio track than emit an empty audio trak
                // that confuses players.
                tracing::warn!(
                    "audio mux: with_audio called but no samples pushed; dropping audio"
                );
                self.audio = None;
            }
        }

        // 90 kHz matches ffmpeg/x264/x265 and divides evenly for 23.976 /
        // 29.97 / 59.94 fps.
        let video_timescale: u32 = 90_000;
        let frame_duration: u32 = ((video_timescale as f64) / self.frame_rate)
            .round()
            .max(1.0) as u32;
        let total_video_duration: u64 = frame_duration as u64 * self.packet_count as u64;

        // Build the visual sample entry up front (codec-dispatched). For AV1
        // it embeds the sequence-header OBU in av1C; for H.264/H.265 it embeds
        // the parameter sets captured during add_packet in avcC/hvcC.
        let video_sample_entry = match self.codec {
            VideoCodec::Av1 => {
                let first_packet = self
                    .first_packet_header
                    .as_ref()
                    .context("first packet header missing; add_packet never called?")?;
                let av1_obus = extract_sequence_header(first_packet)
                    .context("extracting AV1 sequence header OBU from first packet")?;
                build_av01(self.width, self.height, &av1_obus, &self.color_metadata)
            }
            VideoCodec::H264 => {
                let w = self.nal_writer.as_ref().context("H.264 nal writer missing")?;
                if !w.has_param_sets() {
                    anyhow::bail!("H.264 mux: no SPS/PPS captured from the encoder bitstream");
                }
                let avcc = build_avcc(&w.sps, &w.pps);
                // `avc3` signals in-band parameter sets (inline-stitch mode);
                // `avc1` requires them out-of-band only.
                let fourcc = if self.inline_param_sets { b"avc3" } else { b"avc1" };
                build_avc1(self.width, self.height, &avcc, &self.color_metadata, fourcc)
            }
            VideoCodec::H265 => {
                let w = self.nal_writer.as_ref().context("H.265 nal writer missing")?;
                if !w.has_param_sets() {
                    anyhow::bail!("H.265 mux: no VPS/SPS/PPS captured from the encoder bitstream");
                }
                let hvcc = build_hvcc(&w.vps, &w.sps, &w.pps);
                // `hev1` signals in-band parameter sets; `hvc1` is out-of-band.
                let fourcc = if self.inline_param_sets { b"hev1" } else { b"hvc1" };
                build_hvc1(self.width, self.height, &hvcc, &self.color_metadata, fourcc)
            }
        };

        let ftyp = build_ftyp(self.codec);

        // Chunking policy: one second per chunk, capped at 120 for video
        // and 200 for audio. Matching ~1 s per chunk on both sides keeps
        // seek granularity consistent and bounds stsc/stco table sizes.
        let video_spc: u32 = (self.frame_rate.round() as u32).max(1).min(120);

        // Pre-compute audio chunking + per-track totals so the movie header
        // can report `max(video_duration, audio_duration)` in movie timescale.
        // Choose movie timescale = max(video, audio) timescales so both
        // durations convert integer-cleanly (we use video's 90 kHz which is
        // already a multiple of all common audio rates' divisors in the
        // chosen target — but we do the conversion explicitly either way
        // since 48000 ∤ 90000; we round-to-nearest which is what ISOBMFF
        // players expect for track duration display).
        let movie_timescale: u32 = video_timescale;

        let audio_plan: Option<AudioBuildPlan> = self.audio.as_ref().map(|a| {
            // Chunking policy: aim for ~1 second of audio per chunk.
            // Frame size differs by codec — AAC = 1024 samples / frame,
            // Opus = 960 samples / frame at 48 kHz (the standard encoder
            // frame size; callers using 2.5 / 5 / 10 / 40 / 60 ms frames
            // would diverge but the chunk-size cap and the 1-second
            // target both still apply, so the worst case is a slightly
            // suboptimal chunk granularity rather than a structurally
            // broken file). The mdhd timescale is `a.info.timescale`
            // (sample_rate for AAC, 48000 for Opus).
            let frames_per_sec = match AudioCodecKind::from_codec_tag(&a.info.codec) {
                Some(AudioCodecKind::Opus) => (a.info.timescale as f64) / 960.0,
                // AC-3 / E-AC-3: 1536 samples per syncframe (6 blocks × 256).
                Some(AudioCodecKind::Ac3) | Some(AudioCodecKind::Eac3) => {
                    (a.info.timescale as f64) / 1536.0
                }
                Some(AudioCodecKind::Aac) | None => (a.info.timescale as f64) / 1024.0,
            };
            let audio_spc = (frames_per_sec.round() as u32).max(1).min(200);
            let audio_duration_movie: u64 =
                ((a.total_duration_ticks as u128) * movie_timescale as u128
                    / a.info.timescale.max(1) as u128) as u64;
            AudioBuildPlan {
                info: a.info.clone(),
                sample_sizes: a.sample_sizes.clone(),
                durations: a.durations.clone(),
                total_duration_in_own_ts: a.total_duration_ticks,
                total_duration_in_movie_ts: audio_duration_movie,
                samples_per_chunk: audio_spc,
            }
        });

        let video_duration_movie: u64 = total_video_duration; // video uses 90 kHz == movie
        let movie_duration: u64 = match audio_plan.as_ref() {
            Some(p) => video_duration_movie.max(p.total_duration_in_movie_ts),
            None => video_duration_movie,
        };

        // Video-side mdat byte total stays in self; audio side is in plan.
        let video_payload_bytes = self.mdat_payload_bytes;
        let audio_payload_bytes = audio_plan
            .as_ref()
            .map(|p| p.sample_sizes.iter().map(|&s| s as u64).sum::<u64>())
            .unwrap_or(0);
        let mdat_payload_total = video_payload_bytes
            .checked_add(audio_payload_bytes)
            .context("combined mdat payload overflow")?;

        // mdat box-size policy. The 32-bit `size` field maxes at
        // u32::MAX; the box header is 8 bytes (size + type). When the box
        // body alone would push the total past u32::MAX - 8, we switch to
        // the ISOBMFF 14496-12 §4.2 largesize form: `size = 1` (32 bits),
        // `type = 'mdat'`, then a 64-bit `largesize` field carrying the
        // total box length (header + payload). Header grows from 8 → 16
        // bytes which means stco/co64 offsets must reflect the post-header
        // start.
        let mdat_payload_plus_short_header = 8u64
            .checked_add(mdat_payload_total)
            .context("mdat short-header size overflow")?;
        // Production: pick largesize iff the payload + short header
        // exceeds u32. Tests can force largesize on to exercise the
        // bit-layout without crafting a 4 GiB tempfile.
        let use_largesize_mdat =
            mdat_payload_plus_short_header > u32::MAX as u64 || self.force_largesize_mdat;
        let mdat_header_len: u64 = if use_largesize_mdat { 16 } else { 8 };
        let mdat_box_size: u64 = mdat_header_len
            .checked_add(mdat_payload_total)
            .context("mdat box size overflow")?;

        // Two-pass moov construction. On pass 1 we need placeholder offsets
        // of consistent widths to size the moov; on pass 2 we use the real
        // offsets computed against the planned mdat layout.
        let video_chunk_count = chunk_count_of(self.sample_sizes.len(), video_spc);
        let audio_chunk_count = audio_plan
            .as_ref()
            .map(|p| chunk_count_of(p.sample_sizes.len(), p.samples_per_chunk))
            .unwrap_or(0);
        let video_zero_offsets: Vec<u64> = vec![0; video_chunk_count];
        let audio_zero_offsets: Vec<u64> = vec![0; audio_chunk_count];

        let moov_co64_size = build_moov_any(
            self.width,
            self.height,
            video_timescale,
            movie_timescale,
            movie_duration,
            total_video_duration,
            frame_duration,
            &self.sample_sizes,
            &self.keyframe_indices,
            &video_sample_entry,
            &video_zero_offsets,
            video_spc,
            audio_plan.as_ref(),
            &audio_zero_offsets,
            true,
            &self.color_metadata,
        )
        .len() as u64;

        let upper_bound: u64 = (ftyp.len() as u64)
            .checked_add(moov_co64_size)
            .context("moov size overflow")?
            .checked_add(mdat_header_len)
            .context("mdat header overflow")?
            .checked_add(mdat_payload_total)
            .context("mdat payload overflow")?;
        let use_co64 = upper_bound > u32::MAX as u64;

        let moov_without_offsets = build_moov_any(
            self.width,
            self.height,
            video_timescale,
            movie_timescale,
            movie_duration,
            total_video_duration,
            frame_duration,
            &self.sample_sizes,
            &self.keyframe_indices,
            &video_sample_entry,
            &video_zero_offsets,
            video_spc,
            audio_plan.as_ref(),
            &audio_zero_offsets,
            use_co64,
            &self.color_metadata,
        );

        let mdat_offset_in_file = (ftyp.len() + moov_without_offsets.len()) as u64;
        let first_sample_file_offset = mdat_offset_in_file + mdat_header_len;
        if !use_co64 && first_sample_file_offset > u32::MAX as u64 {
            anyhow::bail!(
                "internal: chose stco but first_sample_file_offset {} exceeds u32",
                first_sample_file_offset
            );
        }

        // Compute interleaved chunk offsets. No audio → contiguous video
        // chunks (unchanged behaviour). Audio present → alternating video,
        // audio, video, audio, ..., tail is whichever side has samples left.
        let (video_chunk_offsets, audio_chunk_offsets, interleave_plan) = plan_interleaved_layout(
            first_sample_file_offset,
            &self.sample_sizes,
            video_spc,
            audio_plan.as_ref(),
        );
        debug_assert_eq!(video_chunk_offsets.len(), video_chunk_count);
        debug_assert_eq!(audio_chunk_offsets.len(), audio_chunk_count);

        let moov = build_moov_any(
            self.width,
            self.height,
            video_timescale,
            movie_timescale,
            movie_duration,
            total_video_duration,
            frame_duration,
            &self.sample_sizes,
            &self.keyframe_indices,
            &video_sample_entry,
            &video_chunk_offsets,
            video_spc,
            audio_plan.as_ref(),
            &audio_chunk_offsets,
            use_co64,
            &self.color_metadata,
        );

        assert_eq!(
            moov.len(),
            moov_without_offsets.len(),
            "moov size must be stable across rebuild"
        );

        // Stream final layout: ftyp + moov + mdat-header + mdat-payload.
        let out_file = File::create(output_path)
            .with_context(|| format!("creating output file {}", output_path.display()))?;
        let mut out = BufWriter::new(out_file);
        out.write_all(&ftyp).context("writing ftyp")?;
        out.write_all(&moov).context("writing moov")?;
        if use_largesize_mdat {
            // size=1 sentinel, then 'mdat', then 64-bit largesize.
            out.write_all(&1u32.to_be_bytes())
                .context("writing mdat largesize sentinel")?;
            out.write_all(b"mdat").context("writing mdat type")?;
            out.write_all(&mdat_box_size.to_be_bytes())
                .context("writing mdat largesize")?;
        } else {
            let mdat_size_u32 = mdat_box_size as u32;
            out.write_all(&mdat_size_u32.to_be_bytes())
                .context("writing mdat size")?;
            out.write_all(b"mdat").context("writing mdat type")?;
        }

        // Stream mdat bytes per the interleave plan. Each InterleaveStep
        // records which track and how many bytes to copy from that track's
        // tempfile. We reopen both tempfiles once and copy by range so we
        // never buffer the full payload.
        let video_payload_handle = self
            .mdat_tmp
            .reopen()
            .context("reopening mdat tempfile for read")?;
        let mut video_payload = BufReader::new(video_payload_handle);
        video_payload
            .seek(SeekFrom::Start(0))
            .context("rewinding mdat tempfile")?;

        let mut audio_payload: Option<BufReader<File>> = match self.audio.as_ref() {
            Some(a) => {
                let h = a
                    .audio_tmp
                    .reopen()
                    .context("reopening audio mdat tempfile for read")?;
                let mut r = BufReader::new(h);
                r.seek(SeekFrom::Start(0))
                    .context("rewinding audio mdat tempfile")?;
                Some(r)
            }
            None => None,
        };

        let mut video_copied: u64 = 0;
        let mut audio_copied: u64 = 0;
        for step in &interleave_plan {
            match step.track {
                sample_table::InterleaveTrack::Video => {
                    let copied =
                        std::io::copy(&mut (&mut video_payload).take(step.bytes), &mut out)
                            .context("copying video chunk into mdat")?;
                    if copied != step.bytes {
                        anyhow::bail!(
                            "video chunk short read: wanted {}, got {}",
                            step.bytes,
                            copied
                        );
                    }
                    video_copied += copied;
                }
                sample_table::InterleaveTrack::Audio => {
                    let audio_r = audio_payload.as_mut().context(
                        "internal: interleave plan has audio step but no audio tempfile",
                    )?;
                    let copied = std::io::copy(&mut audio_r.take(step.bytes), &mut out)
                        .context("copying audio chunk into mdat")?;
                    if copied != step.bytes {
                        anyhow::bail!(
                            "audio chunk short read: wanted {}, got {}",
                            step.bytes,
                            copied
                        );
                    }
                    audio_copied += copied;
                }
            }
        }
        if video_copied != video_payload_bytes {
            anyhow::bail!(
                "video mdat payload length mismatch: expected {}, copied {}",
                video_payload_bytes,
                video_copied
            );
        }
        if audio_copied != audio_payload_bytes {
            anyhow::bail!(
                "audio mdat payload length mismatch: expected {}, copied {}",
                audio_payload_bytes,
                audio_copied
            );
        }
        out.flush().context("flushing output")?;

        Ok(())
    }

    /// Back-compat: finalize into memory. Writes to a second tempfile then
    /// reads it back. Callers hitting the 4 GB ceiling should use
    /// `finalize_to_file` instead.
    pub fn finalize(self) -> Result<Bytes> {
        let tmp = NamedTempFile::new().context("creating finalize buffer tempfile")?;
        let path = tmp.path().to_path_buf();
        self.finalize_to_file(&path)?;
        let mut f = File::open(&path).context("reopening finalize buffer tempfile")?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf).context("reading finalize buffer")?;
        Ok(Bytes::from(buf))
    }
}
