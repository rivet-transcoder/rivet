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
enum AudioCodecKind {
    Aac,
    Opus,
    Ac3,
    Eac3,
}

impl AudioCodecKind {
    fn from_codec_tag(codec: &str) -> Option<Self> {
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
        let mdat_tmp = NamedTempFile::new().context("creating mdat tempfile")?;
        let handle = mdat_tmp
            .reopen()
            .context("reopening mdat tempfile for write")?;
        let mdat_writer = BufWriter::new(handle);
        let nal_writer = match codec {
            VideoCodec::Av1 => None,
            VideoCodec::H264 => Some(NalSampleWriter::new(NalMuxCodec::H264)),
            VideoCodec::H265 => Some(NalSampleWriter::new(NalMuxCodec::H265)),
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
        let sample: Vec<u8> = match &mut self.nal_writer {
            None => {
                if self.first_packet_header.is_none() {
                    self.first_packet_header = Some(packet.data.to_vec());
                }
                packet.data.to_vec()
            }
            Some(writer) => writer.push_frame(&packet.data),
        };
        let size = sample.len() as u32;
        self.mdat_writer
            .write_all(&sample)
            .context("writing packet to mdat tempfile")?;
        self.sample_sizes.push(size);
        self.packet_count = self
            .packet_count
            .checked_add(1)
            .context("packet count overflow")?;
        if packet.is_keyframe {
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
                build_avc1(self.width, self.height, &avcc, &self.color_metadata)
            }
            VideoCodec::H265 => {
                let w = self.nal_writer.as_ref().context("H.265 nal writer missing")?;
                if !w.has_param_sets() {
                    anyhow::bail!("H.265 mux: no VPS/SPS/PPS captured from the encoder bitstream");
                }
                let hvcc = build_hvcc(&w.vps, &w.sps, &w.pps);
                build_hvc1(self.width, self.height, &hvcc, &self.color_metadata)
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
                InterleaveTrack::Video => {
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
                InterleaveTrack::Audio => {
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

/// Audio build plan shared between sizing passes and the final moov emit.
/// Holds the post-flush AAC metadata plus the derived chunking policy.
struct AudioBuildPlan {
    info: AudioInfo,
    sample_sizes: Vec<u32>,
    durations: Vec<u32>,
    total_duration_in_own_ts: u64,
    total_duration_in_movie_ts: u64,
    samples_per_chunk: u32,
}

/// One contiguous copy from one source tempfile to the output. The finalize
/// loop walks a Vec<InterleaveStep> and copies `bytes` from the chosen
/// track's tempfile into the output stream, which keeps peak RAM bounded.
#[derive(Debug, Clone, Copy)]
struct InterleaveStep {
    track: InterleaveTrack,
    bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InterleaveTrack {
    Video,
    Audio,
}

fn chunk_count_of(sample_count: usize, spc: u32) -> usize {
    if sample_count == 0 {
        return 0;
    }
    let spc = spc.max(1) as usize;
    sample_count.div_ceil(spc)
}

/// Compute chunk byte size arrays — one entry per chunk, summing sample
/// sizes inside each chunk.
fn chunk_byte_sizes(sample_sizes: &[u32], spc: u32) -> Vec<u64> {
    let spc = spc.max(1) as usize;
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < sample_sizes.len() {
        let end = (i + spc).min(sample_sizes.len());
        let mut total: u64 = 0;
        for &s in &sample_sizes[i..end] {
            total += s as u64;
        }
        out.push(total);
        i = end;
    }
    out
}

/// Plan the interleaved mdat layout + assign per-track chunk offsets.
/// Chunk-alternation: emit one video chunk then one audio chunk, repeating
/// until both are drained; tail chunks for whichever track has more chunks.
/// This gives ~1 s interleave granularity on both sides which matches the
/// spc policy (video: frame_rate fps / 1 chunk; audio: ~46 chunks/s worth).
fn plan_interleaved_layout(
    first_sample_file_offset: u64,
    video_sample_sizes: &[u32],
    video_spc: u32,
    audio_plan: Option<&AudioBuildPlan>,
) -> (Vec<u64>, Vec<u64>, Vec<InterleaveStep>) {
    let video_chunks = chunk_byte_sizes(video_sample_sizes, video_spc);
    let audio_chunks = match audio_plan {
        Some(p) => chunk_byte_sizes(&p.sample_sizes, p.samples_per_chunk),
        None => Vec::new(),
    };

    let mut video_offsets: Vec<u64> = Vec::with_capacity(video_chunks.len());
    let mut audio_offsets: Vec<u64> = Vec::with_capacity(audio_chunks.len());
    let mut plan: Vec<InterleaveStep> = Vec::with_capacity(video_chunks.len() + audio_chunks.len());

    let mut cursor = first_sample_file_offset;
    let mut vi = 0usize;
    let mut ai = 0usize;
    loop {
        if vi < video_chunks.len() {
            video_offsets.push(cursor);
            let size = video_chunks[vi];
            plan.push(InterleaveStep {
                track: InterleaveTrack::Video,
                bytes: size,
            });
            cursor = cursor.saturating_add(size);
            vi += 1;
        }
        if ai < audio_chunks.len() {
            audio_offsets.push(cursor);
            let size = audio_chunks[ai];
            plan.push(InterleaveStep {
                track: InterleaveTrack::Audio,
                bytes: size,
            });
            cursor = cursor.saturating_add(size);
            ai += 1;
        }
        if vi >= video_chunks.len() && ai >= audio_chunks.len() {
            break;
        }
    }

    (video_offsets, audio_offsets, plan)
}

/// Build `ftyp` for AV1-in-MP4 with Apple-device compatibility.
///
/// Per AV1-ISOBMFF v1.3.0 §2.1, an AV1-bearing ISOBMFF file SHALL list
/// `av01` in its `compatible_brands`. Apple's QuickTime / iOS Safari
/// stack additionally requires a structural ISOBMFF brand: `iso6`
/// (ISO/IEC 14496-12 sixth edition — covers `co64`, `mehd` v1, etc.)
/// is the right choice here because the muxer's co64 / large-mdat
/// extensions need the v6 spec scope to be conformant. `mp42`
/// (ISO/IEC 14496-14 second edition) is the conventional brand
/// downstream players key off when deciding AAC / mp4a parsing rules,
/// so we list it as well.
///
/// `major_brand` is set to `iso6` so a strict parser that rejects an
/// `isom`/`mp41`-major file with a co64 box (mp41 predates the v6
/// definition) accepts the output.
fn build_ftyp(codec: VideoCodec) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"ftyp");
    b.extend(b"iso6"); // major_brand (v6 of 14496-12; covers co64/largesize)
    b.u32(512); // minor_version (matches FFmpeg / mp4box convention)
    b.extend(b"iso6"); // compatible: structural baseline
    b.extend(b"iso2"); // compatible: 14496-12 second edition (legacy parsers)
    // codec brand: av01 (AV1-ISOBMFF §2.1, REQUIRED) / avc1 (H.264) / hvc1 (H.265)
    b.extend(codec.sample_entry_fourcc().as_bytes());
    b.extend(b"mp41"); // compatible: classic 14496-14 (older players)
    b.extend(b"mp42"); // compatible: 14496-14 second edition (AAC parsing rules)
    b.finish()
}

/// Video-only back-compat wrapper, used by existing tests. New code flows
/// through `build_moov_any` which handles the 1-trak / 2-trak case
/// uniformly.
#[cfg(test)]
fn build_moov(
    width: u32,
    height: u32,
    timescale: u32,
    duration: u64,
    frame_duration: u32,
    sample_sizes: &[u32],
    keyframe_indices: &[u32],
    config_obus: &[u8],
    chunk_offsets: &[u64],
    samples_per_chunk: u32,
    use_co64: bool,
) -> Vec<u8> {
    build_moov_any(
        width,
        height,
        timescale,
        timescale,
        duration,
        duration,
        frame_duration,
        sample_sizes,
        keyframe_indices,
        config_obus,
        chunk_offsets,
        samples_per_chunk,
        None,
        &[],
        use_co64,
        &ColorMetadata::default(),
    )
}

/// Build moov with video trak plus optional audio trak. `movie_timescale`
/// governs mvhd; `video_timescale` is video mdhd's own clock. When audio is
/// present we pin both movie and video to the same 90 kHz reference so
/// durations don't need a per-trak rate rescale.
fn build_moov_any(
    width: u32,
    height: u32,
    video_timescale: u32,
    movie_timescale: u32,
    movie_duration: u64,
    video_duration_in_video_ts: u64,
    frame_duration: u32,
    sample_sizes: &[u32],
    keyframe_indices: &[u32],
    config_obus: &[u8],
    video_chunk_offsets: &[u64],
    video_spc: u32,
    audio_plan: Option<&AudioBuildPlan>,
    audio_chunk_offsets: &[u64],
    use_co64: bool,
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    // next_track_ID starts at 3 when audio is present (video=1, audio=2).
    let next_track_id: u32 = if audio_plan.is_some() { 3 } else { 2 };
    let mvhd = build_mvhd_v2(movie_timescale, movie_duration, next_track_id);
    // Video track duration expressed in movie timescale.
    let video_duration_movie: u64 = if video_timescale == movie_timescale {
        video_duration_in_video_ts
    } else {
        ((video_duration_in_video_ts as u128) * movie_timescale as u128
            / video_timescale.max(1) as u128) as u64
    };
    let video_trak = build_video_trak(
        width,
        height,
        video_timescale,
        video_duration_movie,
        video_duration_in_video_ts,
        frame_duration,
        sample_sizes,
        keyframe_indices,
        config_obus,
        video_chunk_offsets,
        video_spc,
        use_co64,
        color_metadata,
    );

    let mut b = BoxBuilder::new(b"moov");
    b.extend(&mvhd);
    b.extend(&video_trak);
    if let Some(plan) = audio_plan {
        let audio_trak = build_audio_trak(
            plan,
            plan.total_duration_in_movie_ts,
            audio_chunk_offsets,
            use_co64,
        );
        b.extend(&audio_trak);
    }
    b.finish()
}

/// mvhd v2: takes `next_track_ID`. When audio is present we increment past
/// the audio track ID, otherwise past the video track ID (existing
/// behaviour: next_track_ID=2). Original `build_mvhd` fed 2 hard-coded.
fn build_mvhd_v2(timescale: u32, duration: u64, next_track_id: u32) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"mvhd");
    b.u8(0); // version
    b.extend(&[0, 0, 0]); // flags
    b.u32(0); // creation_time
    b.u32(0); // modification_time
    b.u32(timescale);
    b.u32(duration as u32);
    b.u32(0x00010000); // rate 1.0
    b.u16(0x0100); // volume 1.0
    b.u16(0); // reserved
    b.u32(0); // reserved
    b.u32(0);
    write_unity_matrix(&mut b);
    for _ in 0..6 {
        b.u32(0);
    } // pre_defined
    b.u32(next_track_id);
    b.finish()
}

/// Video trak builder. `duration_in_movie_ts` goes into tkhd (the movie
/// header's clock); `duration_in_mdhd_ts` goes into mdhd (the track's own
/// clock). For video the two timescales are currently pinned equal at
/// 90 kHz, but the split is kept so the audio path, which has a distinct
/// mdhd timescale (= sample_rate), uses the same builder pattern.
fn build_video_trak(
    width: u32,
    height: u32,
    mdhd_timescale: u32,
    duration_in_movie_ts: u64,
    duration_in_mdhd_ts: u64,
    frame_duration: u32,
    sample_sizes: &[u32],
    keyframe_indices: &[u32],
    config_obus: &[u8],
    chunk_offsets: &[u64],
    samples_per_chunk: u32,
    use_co64: bool,
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let tkhd = build_video_tkhd(width, height, duration_in_movie_ts);
    let mdia = build_video_mdia(
        width,
        height,
        mdhd_timescale,
        duration_in_mdhd_ts,
        frame_duration,
        sample_sizes,
        keyframe_indices,
        config_obus,
        chunk_offsets,
        samples_per_chunk,
        use_co64,
        color_metadata,
    );

    let mut b = BoxBuilder::new(b"trak");
    b.extend(&tkhd);
    b.extend(&mdia);
    b.finish()
}

fn build_video_tkhd(width: u32, height: u32, duration: u64) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"tkhd");
    b.u8(0); // version
    b.extend(&[0, 0, 0x03]); // flags: track_enabled | track_in_movie
    b.u32(0); // creation_time
    b.u32(0); // modification_time
    b.u32(1); // track_ID
    b.u32(0); // reserved
    b.u32(duration as u32);
    b.u32(0); // reserved
    b.u32(0);
    b.u16(0); // layer
    b.u16(0); // alternate_group
    b.u16(0); // volume (0 for video)
    b.u16(0); // reserved
    write_unity_matrix(&mut b);
    b.u32(width << 16); // width as 16.16
    b.u32(height << 16);
    b.finish()
}

fn build_video_mdia(
    width: u32,
    height: u32,
    timescale: u32,
    duration: u64,
    frame_duration: u32,
    sample_sizes: &[u32],
    keyframe_indices: &[u32],
    config_obus: &[u8],
    chunk_offsets: &[u64],
    samples_per_chunk: u32,
    use_co64: bool,
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let mdhd = build_mdhd(timescale, duration);
    let hdlr = build_video_hdlr();
    let minf = build_minf(
        width,
        height,
        frame_duration,
        sample_sizes,
        keyframe_indices,
        config_obus,
        chunk_offsets,
        samples_per_chunk,
        use_co64,
        color_metadata,
    );

    let mut b = BoxBuilder::new(b"mdia");
    b.extend(&mdhd);
    b.extend(&hdlr);
    b.extend(&minf);
    b.finish()
}

fn build_mdhd(timescale: u32, duration: u64) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"mdhd");
    b.u8(0); // version
    b.extend(&[0, 0, 0]); // flags
    b.u32(0); // creation_time
    b.u32(0); // modification_time
    b.u32(timescale);
    b.u32(duration as u32);
    b.u16(0x55c4); // language 'und'
    b.u16(0); // pre_defined
    b.finish()
}

fn build_video_hdlr() -> Vec<u8> {
    let mut b = BoxBuilder::new(b"hdlr");
    b.u8(0); // version
    b.extend(&[0, 0, 0]); // flags
    b.u32(0); // pre_defined
    b.extend(b"vide"); // handler_type
    b.u32(0); // reserved[0]
    b.u32(0); // reserved[1]
    b.u32(0); // reserved[2]
    b.extend(b"VideoHandler\0");
    b.finish()
}

// -------- Audio trak / mdia / minf / stbl / mp4a / esds ----------------
// These layers match ISO/IEC 14496-12/14 for an AAC sound track sharing
// mdat with the video track. Offsets are supplied by the finalize planner;
// the builders just embed them.

fn build_audio_trak(
    plan: &AudioBuildPlan,
    duration_in_movie_ts: u64,
    chunk_offsets: &[u64],
    use_co64: bool,
) -> Vec<u8> {
    let tkhd = build_audio_tkhd(duration_in_movie_ts);
    let mdia = build_audio_mdia(plan, chunk_offsets, use_co64);

    let mut b = BoxBuilder::new(b"trak");
    b.extend(&tkhd);
    b.extend(&mdia);
    b.finish()
}

fn build_audio_tkhd(duration_in_movie_ts: u64) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"tkhd");
    b.u8(0); // version
    b.extend(&[0, 0, 0x03]); // flags: track_enabled | track_in_movie
    b.u32(0); // creation_time
    b.u32(0); // modification_time
    b.u32(2); // track_ID (audio is track 2)
    b.u32(0); // reserved
    b.u32(duration_in_movie_ts as u32);
    b.u32(0); // reserved
    b.u32(0);
    b.u16(0); // layer
    b.u16(0x0001); // alternate_group (1 for audio; lets players swap tracks within the group)
    b.u16(0x0100); // volume 1.0 (audio)
    b.u16(0); // reserved
    write_unity_matrix(&mut b);
    b.u32(0); // width = 0 for audio
    b.u32(0); // height = 0 for audio
    b.finish()
}

fn build_audio_mdia(plan: &AudioBuildPlan, chunk_offsets: &[u64], use_co64: bool) -> Vec<u8> {
    let mdhd = build_mdhd(plan.info.timescale, plan.total_duration_in_own_ts);
    let hdlr = build_audio_hdlr();
    let minf = build_audio_minf(plan, chunk_offsets, use_co64);

    let mut b = BoxBuilder::new(b"mdia");
    b.extend(&mdhd);
    b.extend(&hdlr);
    b.extend(&minf);
    b.finish()
}

fn build_audio_hdlr() -> Vec<u8> {
    let mut b = BoxBuilder::new(b"hdlr");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(0); // pre_defined
    b.extend(b"soun"); // handler_type
    b.u32(0);
    b.u32(0);
    b.u32(0);
    b.extend(b"SoundHandler\0");
    b.finish()
}

fn build_audio_minf(plan: &AudioBuildPlan, chunk_offsets: &[u64], use_co64: bool) -> Vec<u8> {
    let smhd = build_smhd();
    let dinf = build_dinf();
    let stbl = build_audio_stbl(plan, chunk_offsets, use_co64);

    let mut b = BoxBuilder::new(b"minf");
    b.extend(&smhd);
    b.extend(&dinf);
    b.extend(&stbl);
    b.finish()
}

fn build_smhd() -> Vec<u8> {
    let mut b = BoxBuilder::new(b"smhd");
    b.u8(0);
    b.extend(&[0, 0, 0]); // flags
    b.u16(0); // balance (0 = center)
    b.u16(0); // reserved
    b.finish()
}

fn build_audio_stbl(plan: &AudioBuildPlan, chunk_offsets: &[u64], use_co64: bool) -> Vec<u8> {
    let stsd = build_audio_stsd(&plan.info);
    let stts = build_audio_stts(&plan.durations);
    let stsc = build_stsc(plan.sample_sizes.len() as u32, plan.samples_per_chunk);
    let stsz = build_stsz(&plan.sample_sizes);
    let chunk_offset_box = if use_co64 {
        build_co64(chunk_offsets)
    } else {
        build_stco(chunk_offsets)
    };

    let mut b = BoxBuilder::new(b"stbl");
    b.extend(&stsd);
    b.extend(&stts);
    b.extend(&stsc);
    b.extend(&stsz);
    b.extend(&chunk_offset_box);
    b.finish()
}

pub(crate) fn build_audio_stsd(info: &AudioInfo) -> Vec<u8> {
    // Dispatch on codec — AAC → mp4a + esds; Opus → Opus + dOps;
    // AC-3 → ac-3 + dac3; E-AC-3 → ec-3 + dec3. The AudioSampleEntry
    // preamble is shared (same v0 layout per ISO/IEC 14496-12 §8.5.2.2 =
    // 36 bytes total before child boxes); only the 4-cc and the
    // codec-specific child differ.
    let kind = AudioCodecKind::from_codec_tag(&info.codec)
        .expect("with_audio gate already validated codec tag");
    let entry = match kind {
        AudioCodecKind::Aac => build_mp4a(info),
        AudioCodecKind::Opus => build_opus_sample_entry(info),
        AudioCodecKind::Ac3 => build_ac3_sample_entry(info),
        AudioCodecKind::Eac3 => build_ec3_sample_entry(info),
    };
    let mut b = BoxBuilder::new(b"stsd");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(1); // entry_count
    b.extend(&entry);
    b.finish()
}

/// AudioSampleEntryV0 per ISO/IEC 14496-12 §8.5.2.2, followed by the esds
/// descriptor tree per ISO/IEC 14496-14 / 14496-1 §7.2.6.5.
///
/// `channelcount` reflects the actual decoded-output channel count as
/// surfaced by the demuxer. For HE-AAC v2 PS (1-channel core) the demuxer
/// upmixes to 2; for 5.1 / 7.1 the AAC channelConfiguration is passed
/// straight through (Squad-25). When channels ≥ 3, an Apple `chan`
/// (Channel Layout) box is appended after `esds` so iOS Safari /
/// QuickTime / AVFoundation render the correct multichannel layout
/// rather than defaulting to L+R downmix.
fn build_mp4a(info: &AudioInfo) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"mp4a");
    // SampleEntry header
    for _ in 0..6 {
        b.u8(0);
    } // reserved[6] = 0
    b.u16(1); // data_reference_index
    // AudioSampleEntry v0 body
    b.u32(0); // reserved (was version + revision_level in v0 QuickTime)
    b.u32(0); // reserved (vendor in v0 QuickTime)
    b.u16(info.channels); // channel_count (driven by demux)
    b.u16(16); // sample_size (bits)
    b.u16(0); // pre_defined
    b.u16(0); // reserved
    b.u32(info.sample_rate << 16); // samplerate 16.16 fixed-point
    // esds child (carries the AudioSpecificConfig verbatim)
    b.extend(&build_esds(info));
    // Apple Channel Layout (`chan`) box for multichannel AAC. Per
    // QuickTime File Format Spec §"Channel Layout Box" the box nests
    // *inside* the `mp4a` AudioSampleEntry alongside `esds`.
    if let Some(chan) = build_chan_box(info.channels) {
        b.extend(&chan);
    }
    b.finish()
}

/// Apple Channel Layout (`chan`) box for ≥3-channel audio. Per the QuickTime
/// File Format Specification, §"Channel Layout Box", and CoreAudioBaseTypes.h
/// (`AudioChannelLayout`):
///
///   - `mChannelLayoutTag` (u32 BE): one of the standard layout tags. The
///     low 16 bits carry the channel count and the high 16 bits identify
///     the layout. Returned by Apple's `kAudioChannelLayoutTag_*` macros.
///   - `mChannelBitmap` (u32 BE) = 0 — only used when the tag is
///     `kAudioChannelLayoutTag_UseChannelBitmap`.
///   - `mNumberChannelDescriptions` (u32 BE) = 0 — only used when the tag
///     is `kAudioChannelLayoutTag_UseChannelDescriptions`.
///
/// Total payload: 12 bytes. Box size: 20 bytes (8-byte header + 12-byte body).
///
/// Returns `None` for mono / stereo (Apple defaults to standard mono /
/// L+R already, no `chan` box needed). Returns `None` for unsupported
/// channel counts — caller's `with_audio` gate already restricts to the
/// supported set; this function uses `None` as a defence-in-depth.
///
/// Standard layouts emitted (channels in this order in the bitstream):
///   - 5.1 → `kAudioChannelLayoutTag_MPEG_5_1_C` = `(114 << 16) | 6`
///     = `0x00720006`. Channels: L, R, C, LFE, Ls, Rs.
///   - 7.1 → `kAudioChannelLayoutTag_MPEG_7_1_C` = `(127 << 16) | 8`
///     = `0x007F0008`. Channels: L, R, C, LFE, Ls, Rs, Lc, Rc.
///
/// 7.1 + Atmos and other extended / object-based layouts are NOT emitted
/// here (caller's `with_audio` gate already rejects them). Adding a wrong
/// `chan` tag is worse than omitting the box — Apple players would map
/// channels to the wrong speakers.
pub(crate) fn build_chan_box(channels: u16) -> Option<Vec<u8>> {
    let tag: u32 = match channels {
        1 | 2 => return None,    // Apple default is correct
        6 => (114u32 << 16) | 6, // kAudioChannelLayoutTag_MPEG_5_1_C
        7 => (127u32 << 16) | 8, // kAudioChannelLayoutTag_MPEG_7_1_C
        _ => return None,        // unsupported (gate already rejected)
    };
    let mut b = BoxBuilder::new(b"chan");
    b.u32(tag); // mChannelLayoutTag
    b.u32(0); // mChannelBitmap
    b.u32(0); // mNumberChannelDescriptions
    Some(b.finish())
}

/// `Opus` sample entry per RFC 7845 §4.4. Same generic AudioSampleEntry v0
/// layout as `mp4a` (per ISO/IEC 14496-12 §8.5.2.2) followed by the
/// Opus-Specific Box `dOps`.
///
/// 4-cc is `Opus` exactly — capital O lowercase pus, that spelling is
/// load-bearing per RFC 7845 §4.4 ("the four-character code shall be set
/// to 'Opus'"). Lowercase variants like `opus` will be rejected by
/// strict players (e.g. macOS / iOS AVFoundation).
///
/// `samplerate` field at the AudioSampleEntry level is set to
/// 48000 << 16 (16.16 fixed-point form of 48000) to match the
/// `InputSampleRate` we emit inside dOps. Apple's AVFoundation reads this
/// field; storing the source's nominal rate (e.g. 44100) would mismatch
/// the dOps body and confuse strict validators.
///
/// `channelcount` carries the actual decoded output channel count
/// (matches `OutputChannelCount` in dOps for ChannelMappingFamily=0).
fn build_opus_sample_entry(info: &AudioInfo) -> Vec<u8> {
    // RFC 7845 §4.4: 4-cc is exactly 'Opus' (capital O).
    let mut b = BoxBuilder::new(b"Opus");
    // SampleEntry header
    for _ in 0..6 {
        b.u8(0);
    } // reserved[6] = 0
    b.u16(1); // data_reference_index
    // AudioSampleEntry v0 body
    b.u32(0); // reserved
    b.u32(0); // reserved
    b.u16(info.channels); // channel_count
    b.u16(16); // sample_size (bits) — informational for Opus
    b.u16(0); // pre_defined
    b.u16(0); // reserved
    // Opus is internally always 48 kHz (RFC 6716). The sample-entry
    // samplerate is the playback / mdhd-aligned rate. Pin to 48000 << 16.
    b.u32(48_000u32 << 16); // samplerate 16.16 fixed-point = 48000
    // dOps child
    b.extend(&build_dops(info));
    b.finish()
}

/// `dOps` Opus-Specific Box per RFC 7845 §4.5.
///
/// Body layout (11 bytes minimum for ChannelMappingFamily=0):
///   - `Version` u8 = 0
///   - `OutputChannelCount` u8
///   - `PreSkip` u16 BE
///   - `InputSampleRate` u32 BE
///   - `OutputGain` i16 BE (Q8 dB; 0 = no gain)
///   - `ChannelMappingFamily` u8
///   - (when family != 0: StreamCount u8 + CoupledCount u8 + ChannelMapping[N])
///
/// Byte-order conversion: the source `codec_private` carries the OpusHead
/// body in **Ogg / WebM little-endian** convention (PreSkip / InputSampleRate
/// / OutputGain are LE) — that's what falls out of WebM/MKV `CodecPrivate`
/// directly, and what an Opus encoder library (libopusenc) emits when
/// asked for OpusHead. RFC 7845 §4.5 mandates **big-endian** for the same
/// fields inside `dOps`. We translate field-by-field rather than copying
/// bytes verbatim.
///
/// `Version`: OpusHead carries Version=1 (its own encoding); RFC 7845 §4.5
/// requires Version=0 in dOps (this is THE box version, not the Opus
/// stream version). We force-write 0 here regardless of what the input
/// `codec_private[0]` says.
fn build_dops(info: &AudioInfo) -> Vec<u8> {
    let p = &info.codec_private;
    debug_assert!(
        p.len() >= 11,
        "with_audio gate must enforce dOps minimum size"
    );

    // OpusHead → dOps numeric field translation.
    // Layout of input bytes (OpusHead, after the 8-byte 'OpusHead' magic
    // which the demuxer already strips):
    //   [0]    Version (u8) — OpusHead version, NOT the dOps version
    //   [1]    OutputChannelCount (u8)
    //   [2..4] PreSkip (u16 LE)
    //   [4..8] InputSampleRate (u32 LE)
    //   [8..10] OutputGain (i16 LE, Q8 dB)
    //   [10]   ChannelMappingFamily (u8)
    //   // Family != 0 trailer (Squad-28, RFC 7845 §5.1.1):
    //   [11]   StreamCount (u8)
    //   [12]   CoupledCount (u8)
    //   [13..13+N]  ChannelMapping (u8 per output channel)
    let output_channels = p[1];
    let pre_skip = u16::from_le_bytes([p[2], p[3]]);
    let input_sample_rate = u32::from_le_bytes([p[4], p[5], p[6], p[7]]);
    let output_gain = i16::from_le_bytes([p[8], p[9]]);
    let channel_mapping_family = p[10];

    let mut b = BoxBuilder::new(b"dOps");
    b.u8(0); // Version (RFC 7845 §4.5: MUST be 0)
    b.u8(output_channels); // OutputChannelCount
    b.u16(pre_skip); // PreSkip (BE — was LE in OpusHead)
    b.u32(input_sample_rate); // InputSampleRate (BE)
    // i16 output gain → wire u16 BE (two's complement preserved across the cast).
    b.u16(output_gain as u16); // OutputGain (BE Q8)
    b.u8(channel_mapping_family); // ChannelMappingFamily

    // ChannelMappingFamily != 0 → ChannelMappingTable follows
    // (RFC 7845 §5.1.1). For family 1 (Squad-28 multichannel) the
    // table is StreamCount + CoupledCount + ChannelMapping[N]. The
    // encoder packed these immediately after the 11-byte preamble in
    // its `extra_data()` output, so the demuxed `codec_private` buffer
    // already carries them in the correct order — we copy verbatim
    // (no endianness conversion: u8 fields).
    if channel_mapping_family != 0 {
        // with_audio's family-1 validation gate ensured codec_private
        // has the trailing bytes; this assert is just forward protection
        // against a future caller bypassing the gate.
        let trailer_len = 2 + output_channels as usize;
        debug_assert!(
            p.len() >= 11 + trailer_len,
            "family={channel_mapping_family} requires {trailer_len} more bytes after the 11-byte preamble; codec_private has {}",
            p.len()
        );
        b.u8(p[11]); // StreamCount
        b.u8(p[12]); // CoupledCount
        for i in 0..output_channels as usize {
            b.u8(p[13 + i]); // ChannelMapping[i]
        }
    }

    b.finish()
}

// ---- Squad-26: AC-3 / E-AC-3 sample entries + dac3 / dec3 boxes ----------
//
// Per ETSI TS 102 366 v1.4.1 Annex F:
//   §F.2 — AC-3 in MP4 / 3GP: 4cc 'ac-3' AudioSampleEntry + 'dac3' config box.
//   §F.4 — `dac3` body layout (3 bytes total payload, 11-byte total box):
//     fscod         2 bits   (0=48k 1=44.1k 2=32k)
//     bsid          5 bits   (=8 for AC-3 — verified from sync header)
//     bsmod         3 bits
//     acmod         3 bits
//     lfeon         1 bit
//     bit_rate_code 5 bits
//     reserved      5 bits   = 0
//   §F.5 — E-AC-3: 4cc 'ec-3' + 'dec3' config box.
//   §F.6 — `dec3` body: data_rate (13b) + num_ind_sub-1 (3b) followed by
//     N independent-substream descriptors (3 bytes each, plus 9-bit
//     chan_loc when num_dep_sub>0). Squad-26 emits the single-substream
//     case (5 bytes total payload, 13-byte box).
//
// Squad-26 hard-restricts to:
//   - AC-3 5.1 / stereo / mono (acmod 1, 2, 7 with optional LFE)
//   - E-AC-3 single independent substream (num_ind_sub=0 wire encoding,
//     num_dep_sub=0). Vanilla 5.1 is the dominant case in the wild.

/// `ac-3` AudioSampleEntry per ETSI TS 102 366 §F.2. Same generic
/// AudioSampleEntry v0 layout (per ISO/IEC 14496-12 §8.5.2.2) as `mp4a` /
/// `Opus` — 28-byte fixed body after the box header — followed by the
/// `dac3` Config Box.
///
/// 4cc is `ac-3` exactly (with the hyphen, ASCII bytes 0x61 0x63 0x2D
/// 0x33). NOT `ac3` — strict players reject the dehyphenated form.
///
/// `samplerate` field at the AudioSampleEntry level is set to
/// `info.sample_rate << 16`. AC-3 samples are 32 / 44.1 / 48 kHz.
///
/// `channelcount` carries the actual decoded output channel count
/// (acmod-derived) — informational; players use the dac3 body for the
/// authoritative channel layout.
fn build_ac3_sample_entry(info: &AudioInfo) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"ac-3");
    // SampleEntry header
    for _ in 0..6 {
        b.u8(0);
    } // reserved[6] = 0
    b.u16(1); // data_reference_index
    // AudioSampleEntry v0 body
    b.u32(0); // reserved
    b.u32(0); // reserved
    b.u16(info.channels); // channel_count (informational)
    b.u16(16); // sample_size (bits) — informational
    b.u16(0); // pre_defined
    b.u16(0); // reserved
    b.u32(info.sample_rate << 16); // samplerate 16.16 fixed-point
    b.extend(&build_dac3(info)); // dac3 child
    b.finish()
}

/// `ec-3` AudioSampleEntry per ETSI TS 102 366 §F.5. Mirrors `ac-3` with a
/// different 4cc and a `dec3` (rather than `dac3`) child config box.
fn build_ec3_sample_entry(info: &AudioInfo) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"ec-3");
    for _ in 0..6 {
        b.u8(0);
    }
    b.u16(1);
    b.u32(0);
    b.u32(0);
    b.u16(info.channels);
    b.u16(16);
    b.u16(0);
    b.u16(0);
    b.u32(info.sample_rate << 16);
    b.extend(&build_dec3(info));
    b.finish()
}

/// `dac3` AC-3 Config Box per ETSI TS 102 366 §F.4. Box header is 8 bytes;
/// payload is exactly 3 bytes (24 bits packed MSB-first). Total = 11 bytes.
///
/// Bit layout (all MSB-first within the 3-byte payload):
/// ```text
///   bit  0..2   fscod          (2 bits)
///   bit  2..7   bsid           (5 bits)
///   bit  7..10  bsmod          (3 bits)
///   bit 10..13  acmod          (3 bits)
///   bit 13..14  lfeon          (1 bit)
///   bit 14..19  bit_rate_code  (5 bits)
///   bit 19..24  reserved       (5 bits, must be 0)
/// ```
///
/// The 3 payload bytes carried in `info.codec_private` are emitted verbatim
/// — the demuxer side already serialised them per the spec, so this builder
/// is a thin wrapper. The 3-byte length contract is checked by `with_audio`.
fn build_dac3(info: &AudioInfo) -> Vec<u8> {
    debug_assert_eq!(
        info.codec_private.len(),
        3,
        "with_audio gate must enforce dac3 body == 3 bytes"
    );
    let mut b = BoxBuilder::new(b"dac3");
    b.extend(&info.codec_private);
    b.finish()
}

/// `dec3` E-AC-3 Config Box per ETSI TS 102 366 §F.6. Box header is 8 bytes;
/// payload is variable size depending on independent / dependent substream
/// count. For the single-independent-substream / no-dependent-substream
/// case (Squad-26's scope) the payload is 5 bytes:
///
/// ```text
///   bit  0..13   data_rate          (13 bits, kbps / 2)
///   bit 13..16   num_ind_sub - 1    (3 bits — 0 = 1 substream)
///   per independent substream:
///     bit 0..2    fscod            (2 bits)
///     bit 2..7    bsid             (5 bits, =16 for E-AC-3)
///     bit 7..8    reserved         (1 bit, =0)
///     bit 8..9    asvc             (1 bit)
///     bit 9..12   bsmod            (3 bits)
///     bit 12..15  acmod            (3 bits)
///     bit 15..16  lfeon            (1 bit)
///     bit 16..19  reserved         (3 bits, =0)
///     bit 19..23  num_dep_sub      (4 bits, =0 in Squad-26 scope)
///     // (if num_dep_sub > 0: chan_loc 9 bits — not emitted here)
/// ```
///
/// The body is carried in `info.codec_private` and emitted verbatim;
/// `with_audio` validates length ≥ 5. Demuxer-side construction of these
/// bytes happens in `demux::derive_dec3_from_eac3_sync`.
fn build_dec3(info: &AudioInfo) -> Vec<u8> {
    debug_assert!(
        info.codec_private.len() >= 5,
        "with_audio gate must enforce dec3 body >= 5 bytes"
    );
    let mut b = BoxBuilder::new(b"dec3");
    b.extend(&info.codec_private);
    b.finish()
}

/// Construct the 3-byte `dac3` body from a parsed AC-3 sync header. Used
/// by the demuxer (derive from first frame) and by tests.
///
/// Bit layout per ETSI TS 102 366 §F.4 (fscod 2 | bsid 5 | bsmod 3 |
/// acmod 3 | lfeon 1 | bit_rate_code 5 | reserved 5).
pub fn dac3_body_from_sync(s: &crate::ac3_sync::Ac3SyncInfo) -> [u8; 3] {
    let mut bw = MsbBitWriter::new();
    bw.put(2, s.fscod as u32);
    bw.put(5, s.bsid as u32);
    bw.put(3, s.bsmod as u32);
    bw.put(3, s.acmod as u32);
    bw.put(1, if s.lfeon { 1 } else { 0 });
    bw.put(5, s.bit_rate_code as u32);
    bw.put(5, 0); // reserved
    let bytes = bw.finish();
    // Exactly 24 bits = 3 bytes (compile-time invariant of the layout).
    [bytes[0], bytes[1], bytes[2]]
}

/// Construct the 5-byte single-substream `dec3` body from a parsed E-AC-3
/// sync header. Used by the demuxer (derive from first frame) and by tests.
///
/// `data_rate` is the source-frame nominal kbps / 2 per §F.6. Compute it
/// from the source: `data_rate = ceil((frame_size_bytes * 8 * sample_rate /
/// samples_per_frame) / 2 / 1000)`. We accept it as a parameter so the
/// caller can supply either the frame-derived value or a stored/best-known
/// value; for vanilla 5.1 48 kHz E-AC-3 at 384 kbps this is 192.
pub fn dec3_body_from_sync(s: &crate::ac3_sync::Eac3SyncInfo, data_rate_div2_kbps: u16) -> [u8; 5] {
    let mut bw = MsbBitWriter::new();
    // Header: data_rate (13b) + num_ind_sub - 1 (3b). num_ind_sub = 1 in
    // Squad-26's scope, so the wire field is 0.
    bw.put(13, (data_rate_div2_kbps & 0x1FFF) as u32);
    bw.put(3, 0); // num_ind_sub - 1 = 0
    // Per-independent-substream block (3 bytes for the no-dep-sub case).
    bw.put(2, s.fscod as u32);
    bw.put(5, 16); // bsid pinned to 16 per §F.6
    bw.put(1, 0); // reserved
    bw.put(1, 0); // asvc — Squad-26 doesn't carry alternate-stream signalling
    bw.put(3, s.bsmod as u32);
    bw.put(3, s.acmod as u32);
    bw.put(1, if s.lfeon { 1 } else { 0 });
    bw.put(3, 0); // reserved
    bw.put(4, 0); // num_dep_sub = 0 (Squad-26 scope)
    let bytes = bw.finish();
    debug_assert_eq!(bytes.len(), 5, "dec3 single-substream body must be 5 bytes");
    [bytes[0], bytes[1], bytes[2], bytes[3], bytes[4]]
}

/// MSB-first bit writer used to pack the dac3 / dec3 bodies. Keeps layout
/// math local to the box builders so the bit boundaries stay obvious in
/// review.
struct MsbBitWriter {
    bytes: Vec<u8>,
    bit_pos: usize,
}

impl MsbBitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            bit_pos: 0,
        }
    }
    fn put(&mut self, n: usize, v: u32) {
        debug_assert!(n <= 24);
        for i in (0..n).rev() {
            let bit = ((v >> i) & 0x01) as u8;
            if self.bit_pos.is_multiple_of(8) {
                self.bytes.push(0);
            }
            let byte_idx = self.bit_pos / 8;
            let bit_idx = 7 - (self.bit_pos % 8);
            self.bytes[byte_idx] |= bit << bit_idx;
            self.bit_pos += 1;
        }
    }
    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

/// Emit `esds` box = FullBox(v=0 f=0) + ES_Descriptor tree per 14496-1.
/// See task spec for layout — we materialise each child into a temp Vec
/// first to compute exact lengths, then wrap the parent descriptors in
/// variable-length headers via `write_descriptor_length`.
fn build_esds(info: &AudioInfo) -> Vec<u8> {
    // Innermost: DecoderSpecificInfo (tag 0x05) payload = ASC bytes verbatim.
    let asc_len = info.asc_bytes.len() as u32;
    let mut dsi = Vec::new();
    dsi.push(0x05u8);
    write_descriptor_length(&mut dsi, asc_len);
    dsi.extend_from_slice(&info.asc_bytes);

    // DecoderConfigDescriptor (tag 0x04): 13-byte fixed preamble + DSI.
    // Fields:
    //   objectTypeIndication u8 = 0x40 (MPEG-4 Audio)
    //   streamType u6 | upStream u1 | reserved u1 => (0x05 << 2) | 0x01 = 0x15
    //   bufferSizeDB u24 = 0
    //   maxBitrate u32 = 0
    //   avgBitrate u32 = 0
    let mut dcd_payload = Vec::new();
    dcd_payload.push(0x40); // AAC / MPEG-4 Audio
    dcd_payload.push((0x05 << 2) | 0x01); // AudioStream | upstream=1
    dcd_payload.extend_from_slice(&[0, 0, 0]); // bufferSizeDB
    dcd_payload.extend_from_slice(&0u32.to_be_bytes()); // maxBitrate
    dcd_payload.extend_from_slice(&0u32.to_be_bytes()); // avgBitrate
    dcd_payload.extend_from_slice(&dsi);
    let mut dcd = Vec::new();
    dcd.push(0x04);
    write_descriptor_length(&mut dcd, dcd_payload.len() as u32);
    dcd.extend_from_slice(&dcd_payload);

    // SLConfigDescriptor (tag 0x06): one byte payload = predefined=2 (MP4 reserved).
    let mut slc = Vec::new();
    slc.push(0x06);
    write_descriptor_length(&mut slc, 1);
    slc.push(0x02);

    // ES_Descriptor (tag 0x03): ES_ID u16=0 + flags u8=0 + DCD + SLC.
    let mut es_payload = Vec::new();
    es_payload.extend_from_slice(&0u16.to_be_bytes()); // ES_ID
    es_payload.push(0); // flags
    es_payload.extend_from_slice(&dcd);
    es_payload.extend_from_slice(&slc);
    let mut es = Vec::new();
    es.push(0x03);
    write_descriptor_length(&mut es, es_payload.len() as u32);
    es.extend_from_slice(&es_payload);

    // FullBox(0)
    let mut b = BoxBuilder::new(b"esds");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.extend(&es);
    b.finish()
}

/// Write a variable-length MPEG-4 descriptor length field. For len < 128
/// emits a single byte. For larger values emits a 4-byte continuation
/// sequence per ISO/IEC 14496-1 (high bit set on every byte but the last,
/// low 7 bits carry 7 bits of the length MSB-first).
///
/// Historical note: the `read_descriptor` peer in demux.rs caps at 4 bytes
/// of continuation, so we use 4 bytes consistently on the write side above
/// the 128 threshold — this keeps round-trip compatibility with our own
/// demuxer and is what ffmpeg / mp4box emit.
fn write_descriptor_length(buf: &mut Vec<u8>, len: u32) {
    if len < 128 {
        buf.push(len as u8);
        return;
    }
    buf.push(((len >> 21) & 0x7F) as u8 | 0x80);
    buf.push(((len >> 14) & 0x7F) as u8 | 0x80);
    buf.push(((len >> 7) & 0x7F) as u8 | 0x80);
    buf.push((len & 0x7F) as u8);
}

/// Audio stts: one entry per run of samples with identical durations.
/// AAC typically has uniform 1024-sample frames so this collapses to a
/// single (count, delta) entry, but we handle runs defensively — some
/// demuxed streams have a shorter tail sample.
fn build_audio_stts(durations: &[u32]) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"stts");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    // First pass: count runs.
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &d in durations {
        if let Some(last) = runs.last_mut()
            && last.1 == d
        {
            last.0 += 1;
            continue;
        }
        runs.push((1, d));
    }
    b.u32(runs.len() as u32);
    for (count, delta) in runs {
        b.u32(count);
        b.u32(delta);
    }
    b.finish()
}

fn build_minf(
    width: u32,
    height: u32,
    frame_duration: u32,
    sample_sizes: &[u32],
    keyframe_indices: &[u32],
    config_obus: &[u8],
    chunk_offsets: &[u64],
    samples_per_chunk: u32,
    use_co64: bool,
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let vmhd = build_vmhd();
    let dinf = build_dinf();
    let stbl = build_stbl(
        width,
        height,
        frame_duration,
        sample_sizes,
        keyframe_indices,
        config_obus,
        chunk_offsets,
        samples_per_chunk,
        use_co64,
        color_metadata,
    );

    let mut b = BoxBuilder::new(b"minf");
    b.extend(&vmhd);
    b.extend(&dinf);
    b.extend(&stbl);
    b.finish()
}

fn build_vmhd() -> Vec<u8> {
    let mut b = BoxBuilder::new(b"vmhd");
    b.u8(0);
    b.extend(&[0, 0, 0x01]); // flags (always 1)
    b.u16(0); // graphicsmode
    b.u16(0);
    b.u16(0);
    b.u16(0); // opcolor
    b.finish()
}

fn build_dinf() -> Vec<u8> {
    let mut dref = BoxBuilder::new(b"dref");
    dref.u8(0);
    dref.extend(&[0, 0, 0]);
    dref.u32(1); // entry_count
    let mut url = BoxBuilder::new(b"url ");
    url.u8(0);
    url.extend(&[0, 0, 0x01]); // self-contained
    dref.extend(&url.finish());

    let mut b = BoxBuilder::new(b"dinf");
    b.extend(&dref.finish());
    b.finish()
}

fn build_stbl(
    width: u32,
    height: u32,
    frame_duration: u32,
    sample_sizes: &[u32],
    keyframe_indices: &[u32],
    config_obus: &[u8],
    chunk_offsets: &[u64],
    samples_per_chunk: u32,
    use_co64: bool,
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let stsd = build_stsd(width, height, config_obus, color_metadata);
    let stts = build_stts(sample_sizes.len() as u32, frame_duration);
    let stsc = build_stsc(sample_sizes.len() as u32, samples_per_chunk);
    let stsz = build_stsz(sample_sizes);
    let chunk_offset_box = if use_co64 {
        build_co64(chunk_offsets)
    } else {
        build_stco(chunk_offsets)
    };
    let stss_box = if !keyframe_indices.is_empty() && keyframe_indices.len() < sample_sizes.len() {
        Some(build_stss(keyframe_indices))
    } else {
        None
    };

    let mut b = BoxBuilder::new(b"stbl");
    b.extend(&stsd);
    b.extend(&stts);
    if let Some(ss) = &stss_box {
        b.extend(ss);
    }
    b.extend(&stsc);
    b.extend(&stsz);
    b.extend(&chunk_offset_box);
    b.finish()
}

/// `stsd` wrapping a single, pre-built visual sample entry (`av01` / `avc1` /
/// `hvc1` — the caller builds the codec-appropriate one). The trailing params
/// are vestigial (the entry already carries width/height/colour) and kept only
/// so the threading call sites don't change.
fn build_stsd(
    _width: u32,
    _height: u32,
    video_sample_entry: &[u8],
    _color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"stsd");
    b.u8(0);
    b.extend(&[0, 0, 0]); // flags
    b.u32(1); // entry_count
    b.extend(video_sample_entry);
    b.finish()
}

/// AV1 visual sample entry per AV1-ISOBMFF v1.3.0 §2.2. Fourcc is `av01`
/// — there is no `hvc1`/`hev1`-style variant for AV1; the configOBU
/// transport mode is selected via flags inside `av1C` itself, not via a
/// separate sample entry name.
///
/// Children, in order:
/// 1. `av1C` — AV1CodecConfigurationRecord (REQUIRED).
/// 2. `colr` — nclx triple + full_range (REQUIRED for Apple, Squad-18).
/// 3. `mdcv` — Mastering Display Color Volume (HDR only, Squad-20).
/// 4. `clli` — Content Light Level Info (HDR only, Squad-20).
///
/// The HDR atoms `mdcv` and `clli` are emitted only when
/// `ColorMetadata.mastering_display` / `.content_light_level` are
/// `Some(_)`. AV1-ISOBMFF v1.3.0 §2.3.4 + §2.3.5 specify the order
/// `colr → mdcv → clli` inside the visual sample entry; players that
/// scan for `mdcv` / `clli` (browsers via Media Capabilities API,
/// AVFoundation) read the box-tree by 4cc, so order is recommended
/// but not load-bearing — we match the spec anyway.
pub(crate) fn build_av01(
    width: u32,
    height: u32,
    config_obus: &[u8],
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let av1c = build_av1c(config_obus);
    let colr = build_colr_nclx(color_metadata);
    let mdcv = color_metadata.mastering_display.as_ref().map(build_mdcv);
    let clli = color_metadata.content_light_level.as_ref().map(build_clli);
    let mut b = BoxBuilder::new(b"av01");
    // VisualSampleEntry
    for _ in 0..6 {
        b.u8(0);
    } // reserved[6]
    b.u16(1); // data_reference_index
    b.u16(0); // pre_defined
    b.u16(0); // reserved
    for _ in 0..3 {
        b.u32(0);
    } // pre_defined[3]
    b.u16(width as u16);
    b.u16(height as u16);
    b.u32(0x00480000); // horiz 72 dpi
    b.u32(0x00480000); // vert 72 dpi
    b.u32(0); // reserved
    b.u16(1); // frame_count (frames per sample)
    // compressorname: 1 length byte + 31 bytes
    b.u8(0);
    for _ in 0..31 {
        b.u8(0);
    }
    b.u16(0x0018); // depth
    b.u16(0xFFFF); // pre_defined
    b.extend(&av1c);
    b.extend(&colr);
    if let Some(mdcv) = &mdcv {
        b.extend(mdcv);
    }
    if let Some(clli) = &clli {
        b.extend(clli);
    }
    b.finish()
}

/// Write the 78-byte ISO 14496-12 `VisualSampleEntry` header (shared by
/// `av01` / `avc1` / `hvc1`) into a freshly-opened sample-entry box.
fn push_visual_sample_entry_header(b: &mut BoxBuilder, width: u32, height: u32) {
    for _ in 0..6 {
        b.u8(0);
    } // reserved[6]
    b.u16(1); // data_reference_index
    b.u16(0); // pre_defined
    b.u16(0); // reserved
    for _ in 0..3 {
        b.u32(0);
    } // pre_defined[3]
    b.u16(width as u16);
    b.u16(height as u16);
    b.u32(0x00480000); // horiz 72 dpi
    b.u32(0x00480000); // vert 72 dpi
    b.u32(0); // reserved
    b.u16(1); // frame_count
    b.u8(0);
    for _ in 0..31 {
        b.u8(0);
    } // compressorname
    b.u16(0x0018); // depth
    b.u16(0xFFFF); // pre_defined
}

/// Append `colr` + (HDR) `mdcv`/`clli` to a visual sample entry.
fn push_color_boxes(b: &mut BoxBuilder, color_metadata: &ColorMetadata) {
    b.extend(&build_colr_nclx(color_metadata));
    if let Some(md) = color_metadata.mastering_display.as_ref() {
        b.extend(&build_mdcv(md));
    }
    if let Some(cll) = color_metadata.content_light_level.as_ref() {
        b.extend(&build_clli(cll));
    }
}

/// Remove H.264/H.265 emulation-prevention bytes (`00 00 03` → `00 00`) so the
/// raw profile/tier/level fields can be read by byte offset. Returns the RBSP.
fn strip_emulation(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let n = data.len();
    let mut i = 0;
    while i < n {
        if i + 2 < n && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 3 {
            out.push(0);
            out.push(0);
            i += 3; // drop the 0x03; the following byte is handled next iter
        } else {
            out.push(data[i]);
            i += 1;
        }
    }
    out
}

/// H.264 `avc1` visual sample entry (avcC + colr [+ HDR atoms]).
pub(crate) fn build_avc1(
    width: u32,
    height: u32,
    avcc: &[u8],
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"avc1");
    push_visual_sample_entry_header(&mut b, width, height);
    b.extend(avcc);
    push_color_boxes(&mut b, color_metadata);
    b.finish()
}

/// H.265 `hvc1` visual sample entry (hvcC + colr [+ HDR atoms]).
pub(crate) fn build_hvc1(
    width: u32,
    height: u32,
    hvcc: &[u8],
    color_metadata: &ColorMetadata,
) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"hvc1");
    push_visual_sample_entry_header(&mut b, width, height);
    b.extend(hvcc);
    push_color_boxes(&mut b, color_metadata);
    b.finish()
}

/// AVCDecoderConfigurationRecord (`avcC`) per ISO 14496-15 §5.3.3.1. Profile /
/// compatibility / level come verbatim from the first SPS (NAL payload bytes
/// 1..4). 4-byte NAL length prefixes (`lengthSizeMinusOne = 3`).
pub(crate) fn build_avcc(sps: &[Vec<u8>], pps: &[Vec<u8>]) -> Vec<u8> {
    let first = sps.first().map(|s| s.as_slice()).unwrap_or(&[]);
    let (profile, compat, level) = if first.len() >= 4 {
        (first[1], first[2], first[3])
    } else {
        (0x64, 0x00, 0x1f) // High @ L3.1 fallback
    };
    let mut body = Vec::new();
    body.push(1); // configurationVersion
    body.push(profile);
    body.push(compat);
    body.push(level);
    body.push(0xFF); // reserved(6)=1 | lengthSizeMinusOne = 3
    body.push(0xE0 | (sps.len() as u8 & 0x1F)); // reserved(3)=1 | numOfSPS
    for s in sps {
        body.extend_from_slice(&(s.len() as u16).to_be_bytes());
        body.extend_from_slice(s);
    }
    body.push(pps.len() as u8); // numOfPPS
    for p in pps {
        body.extend_from_slice(&(p.len() as u16).to_be_bytes());
        body.extend_from_slice(p);
    }
    let mut b = BoxBuilder::new(b"avcC");
    b.extend(&body);
    b.finish()
}

/// HEVCDecoderConfigurationRecord (`hvcC`) per ISO 14496-15 §8.3.3.1.2. The
/// 12-byte general profile_tier_level is copied from the first SPS (RBSP bytes
/// 3..15 — after the 2-byte NAL header + the 1-byte vps_id/max_sub/nesting).
/// Chroma + bit depth are pinned to 4:2:0 8-bit (our SDR output). VPS/SPS/PPS
/// arrays follow. 4-byte NAL length prefixes.
pub(crate) fn build_hvcc(vps: &[Vec<u8>], sps: &[Vec<u8>], pps: &[Vec<u8>]) -> Vec<u8> {
    let mut ptl = [0u8; 12];
    if let Some(s) = sps.first() {
        let rbsp = strip_emulation(s);
        if rbsp.len() >= 15 {
            ptl.copy_from_slice(&rbsp[3..15]);
        } else {
            ptl[0] = 0x01; // Main profile
            ptl[11] = 123; // level 4.1
        }
    }
    let mut body = Vec::new();
    body.push(1); // configurationVersion
    body.extend_from_slice(&ptl); // [1..13] general PTL
    body.extend_from_slice(&[0xF0, 0x00]); // [13-14] reserved | min_spatial_segmentation_idc=0
    body.push(0xFC); // [15] reserved | parallelismType=0
    body.push(0xFC | 0x01); // [16] reserved | chromaFormat=1 (4:2:0)
    body.push(0xF8); // [17] reserved | bitDepthLumaMinus8=0
    body.push(0xF8); // [18] reserved | bitDepthChromaMinus8=0
    body.extend_from_slice(&[0, 0]); // [19-20] avgFrameRate=0
    body.push(0x0F); // [21] cfr=0 | numTemporalLayers=1 | tidNested=1 | lengthSizeMinusOne=3
    let arrays: [(u8, &[Vec<u8>]); 3] = [(32, vps), (33, sps), (34, pps)];
    let present: Vec<&(u8, &[Vec<u8>])> = arrays.iter().filter(|(_, v)| !v.is_empty()).collect();
    body.push(present.len() as u8); // numOfArrays
    for (nal_type, set) in present {
        body.push(0x80 | nal_type); // array_completeness=1 | reserved=0 | NAL_unit_type
        body.extend_from_slice(&(set.len() as u16).to_be_bytes());
        for nal in *set {
            body.extend_from_slice(&(nal.len() as u16).to_be_bytes());
            body.extend_from_slice(nal);
        }
    }
    let mut b = BoxBuilder::new(b"hvcC");
    b.extend(&body);
    b.finish()
}

/// Map the pipeline's `TransferFn` enum back into an H.273
/// `transfer_characteristics` u8 for the `colr nclx` writer. The
/// pipeline's enum is lossy — `Bt709` covers H.273 codes 1, 6, 14, 15 —
/// so we collapse to the canonical code (1 = BT.709) for the SDR family
/// and the spec-defined codes for the HDR transfers.
fn transfer_to_h273(transfer: codec::frame::TransferFn) -> u8 {
    use codec::frame::TransferFn;
    match transfer {
        TransferFn::Bt709 => 1,
        TransferFn::Bt470Bg => 4,
        TransferFn::Linear => 8,
        TransferFn::St2084 => 16,
        TransferFn::AribStdB67 => 18,
        // H.273 reserves 2 for "unspecified". Apple's player treats
        // unspecified as BT.709 limited, which is what the rest of this
        // code already assumes — so there's no behaviour change between
        // emitting 2 and emitting 1 here. Emit 2 to stay honest about
        // what the source told us.
        TransferFn::Unspecified => 2,
    }
}

/// Emit a `colr` box with `colour_type='nclx'` per ISO/IEC 14496-12 §12.1.5
/// and ICC's nclx subtype definition. Layout:
///
///   size u32 | 'colr' | colour_type[4] | colour_primaries u16
///   | transfer_characteristics u16 | matrix_coefficients u16
///   | full_range_flag(1) + reserved(7)
///
/// `nclx` is the right colour_type for video distribution (vs `nclc`
/// which is QuickTime-flavored or `rICC`/`prof` for embedded ICC
/// profiles). Apple's player and ffmpeg both honour it.
fn build_colr_nclx(color_metadata: &ColorMetadata) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"colr");
    b.extend(b"nclx");
    b.u16(color_metadata.colour_primaries as u16);
    b.u16(transfer_to_h273(color_metadata.transfer) as u16);
    b.u16(color_metadata.matrix_coefficients as u16);
    // full_range_flag is the high bit of a single packed byte; the low 7
    // bits are reserved-zero per ISO 23001-8.
    let full_range_byte: u8 = if color_metadata.full_range {
        0x80
    } else {
        0x00
    };
    b.u8(full_range_byte);
    b.finish()
}

/// Emit a `mdcv` (Mastering Display Color Volume) box per ISO/IEC
/// 14496-12 §12.1.6 / AV1-ISOBMFF v1.3.0 §2.3.4. Carries SMPTE ST 2086
/// metadata. Layout:
///
///   size u32 (=32) | 'mdcv' | display_primaries_R_x u16 | _R_y u16
///   | _G_x u16 | _G_y u16 | _B_x u16 | _B_y u16
///   | white_point_x u16 | white_point_y u16
///   | max_display_mastering_luminance u32
///   | min_display_mastering_luminance u32
///
/// Total payload = 8×2 + 2×4 = 24 bytes; with 8-byte header → 32 bytes.
///
/// Box type is `'mdcv'` per AV1-ISOBMFF / 14496-12 v6, NOT the older
/// `'SmDm'` from QuickTime-flavored MOV. Browsers + AVFoundation read
/// `'mdcv'`. The byte order is the standard u16/u32 BE everything else
/// in the file uses.
///
/// Field encoding follows HEVC SEI 137 (`mastering_display_colour_volume`):
///   - Chromaticities are u16 in increments of 0.00002 (so a value of
///     35400 ↔ x=0.708, the BT.2020 red primary).
///   - Luminances are u32 in increments of 0.0001 cd/m² (so 10_000_000
///     ↔ 1000 nits, the canonical HDR10 max).
///
/// We do not normalize/clamp here — the input struct carries spec-domain
/// integers already (Squad-21's probe is responsible for that conversion
/// from float chromaticities / nits).
fn build_mdcv(md: &codec::frame::MasteringDisplay) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"mdcv");
    b.u16(md.primaries_r_x);
    b.u16(md.primaries_r_y);
    b.u16(md.primaries_g_x);
    b.u16(md.primaries_g_y);
    b.u16(md.primaries_b_x);
    b.u16(md.primaries_b_y);
    b.u16(md.white_point_x);
    b.u16(md.white_point_y);
    b.u32(md.max_luminance);
    b.u32(md.min_luminance);
    b.finish()
}

/// Emit a `clli` (Content Light Level Information) box per ISO/IEC
/// 14496-12 §12.1.6 / AV1-ISOBMFF v1.3.0 §2.3.5. Carries CTA-861.3
/// metadata. Layout:
///
///   size u32 (=12) | 'clli' | max_content_light_level u16
///   | max_pic_average_light_level u16
///
/// Total payload = 4 bytes; with 8-byte header → 12 bytes.
///
/// Box type is `'clli'`, NOT `'CoLL'` (the older MOV variant). Both
/// fields are integer cd/m² (nits); MaxCLL is the peak pixel anywhere
/// in the stream, MaxFALL is the peak frame-average. The HDR10
/// reference values are typically MaxCLL ≈ 1000 nits / MaxFALL ≈
/// 400 nits, but we write whatever the source declared verbatim.
fn build_clli(cll: &codec::frame::ContentLightLevel) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"clli");
    b.u16(cll.max_cll);
    b.u16(cll.max_fall);
    b.finish()
}

fn build_av1c(config_obus: &[u8]) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"av1C");
    // marker=1, version=1 -> 0x81
    b.u8(0x81);
    // seq_profile=0, seq_level_idx_0=0 (default; parse from OBU if present)
    let (
        seq_profile,
        seq_level_idx_0,
        seq_tier_0,
        high_bitdepth,
        twelve_bit,
        monochrome,
        chroma_sub_x,
        chroma_sub_y,
        chroma_sample_position,
    ) = parse_seq_header_params(config_obus);
    b.u8(((seq_profile & 0x7) << 5) | (seq_level_idx_0 & 0x1F));
    let byte3 = ((seq_tier_0 & 0x1) << 7)
        | ((high_bitdepth as u8 & 0x1) << 6)
        | ((twelve_bit as u8 & 0x1) << 5)
        | ((monochrome as u8 & 0x1) << 4)
        | ((chroma_sub_x & 0x1) << 3)
        | ((chroma_sub_y & 0x1) << 2)
        | (chroma_sample_position & 0x3);
    b.u8(byte3);
    // initial_presentation_delay_present=0, reserved bits=0
    b.u8(0);
    // configOBUs
    b.extend(config_obus);
    b.finish()
}

fn build_stts(sample_count: u32, frame_duration: u32) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"stts");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(1); // entry_count
    b.u32(sample_count);
    b.u32(frame_duration);
    b.finish()
}

fn build_stss(keyframes: &[u32]) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"stss");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(keyframes.len() as u32);
    for &k in keyframes {
        b.u32(k);
    }
    b.finish()
}

/// Emit a `stsc` with run-length encoding. Full-size chunks of
/// `samples_per_chunk` are represented by one entry starting at chunk 1; if
/// the last chunk has a remainder (< samples_per_chunk), a second entry
/// records it. sample_description_index is always 1 because we emit a single
/// stsd entry (`av01`).
fn build_stsc(sample_count: u32, samples_per_chunk: u32) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"stsc");
    b.u8(0);
    b.extend(&[0, 0, 0]);

    let spc = samples_per_chunk.max(1);
    // Guard against sample_count=0 — the muxer bails before calling this, but
    // keep the expression total: empty tables still need a valid entry_count.
    if sample_count == 0 {
        b.u32(0);
        return b.finish();
    }

    let full_chunks = sample_count / spc;
    let remainder = sample_count % spc;

    if remainder == 0 {
        // Every chunk has spc samples → one entry covers everything.
        b.u32(1);
        b.u32(1); // first_chunk (1-based)
        b.u32(spc); // samples_per_chunk
        b.u32(1); // sample_description_index
    } else if full_chunks == 0 {
        // All samples fit in the final partial chunk → one entry (1, rem, 1).
        b.u32(1);
        b.u32(1);
        b.u32(remainder);
        b.u32(1);
    } else {
        // Full-size run (1 .. full_chunks), then a tail entry for the
        // remainder chunk at index full_chunks+1 (1-based).
        b.u32(2);
        b.u32(1);
        b.u32(spc);
        b.u32(1);
        b.u32(full_chunks + 1); // first_chunk of the tail (1-based)
        b.u32(remainder);
        b.u32(1);
    }
    b.finish()
}

fn build_stsz(sample_sizes: &[u32]) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"stsz");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(0); // sample_size (0 = varying)
    b.u32(sample_sizes.len() as u32); // sample_count
    for &s in sample_sizes {
        b.u32(s);
    }
    b.finish()
}

/// 32-bit chunk offset table. Caller must guarantee every offset fits in u32;
/// the muxer's co64-vs-stco decision does that upstream. Internal `as u32`
/// cast below is checked via `debug_assert` — `overflow-checks=false` in
/// release would otherwise silently wrap.
fn build_stco(chunk_offsets: &[u64]) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"stco");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(chunk_offsets.len() as u32);
    for &off in chunk_offsets {
        debug_assert!(
            off <= u32::MAX as u64,
            "stco offset exceeds u32; should be co64"
        );
        b.u32(off as u32);
    }
    b.finish()
}

/// 64-bit chunk offset table. Layout per ISO/IEC 14496-12:
/// `size u32be | 'co64' | version u8=0 | flags u8[3]=0 | entry_count u32be
/// | entries: u64be chunk_offset[entry_count]`.
fn build_co64(chunk_offsets: &[u64]) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"co64");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(chunk_offsets.len() as u32);
    for &off in chunk_offsets {
        b.u64(off);
    }
    b.finish()
}

/// Partition samples into chunks of size `samples_per_chunk` (last chunk
/// may be smaller), then emit an absolute file offset for each chunk's
/// first sample by walking `sample_sizes` with a running cursor that starts
/// at `first_sample_file_offset`.
///
/// Superseded by `plan_interleaved_layout` on the hot path — kept here for
/// the existing single-track unit tests that exercise the chunking math.
#[cfg(test)]
fn compute_chunk_offsets(
    first_sample_file_offset: u64,
    sample_sizes: &[u32],
    samples_per_chunk: u32,
) -> Vec<u64> {
    let spc = samples_per_chunk.max(1) as usize;
    let total = sample_sizes.len();
    if total == 0 {
        return Vec::new();
    }
    let chunk_count = (total + spc - 1) / spc;
    let mut offsets = Vec::with_capacity(chunk_count);
    let mut cursor = first_sample_file_offset;
    let mut sample_idx = 0usize;
    for _ in 0..chunk_count {
        offsets.push(cursor);
        let end = (sample_idx + spc).min(total);
        for &size in &sample_sizes[sample_idx..end] {
            cursor = cursor.saturating_add(size as u64);
        }
        sample_idx = end;
    }
    offsets
}

pub(crate) fn write_unity_matrix(b: &mut BoxBuilder) {
    b.u32(0x00010000);
    b.u32(0);
    b.u32(0);
    b.u32(0);
    b.u32(0x00010000);
    b.u32(0);
    b.u32(0);
    b.u32(0);
    b.u32(0x40000000);
}

pub(crate) struct BoxBuilder {
    buf: Vec<u8>,
}

impl BoxBuilder {
    pub(crate) fn new(box_type: &[u8; 4]) -> Self {
        let mut buf = Vec::with_capacity(64);
        buf.extend_from_slice(&[0, 0, 0, 0]); // size placeholder
        buf.extend_from_slice(box_type);
        Self { buf }
    }

    pub(crate) fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    pub(crate) fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }
    pub(crate) fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }
    pub(crate) fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }
    pub(crate) fn extend(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }

    /// Current byte length of the buffer (header + payload written so far).
    /// Used by the CMAF muxer to record the position of `trun.data_offset`
    /// so it can be patched once the moof's final size is known.
    pub(crate) fn current_len(&self) -> usize {
        self.buf.len()
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        let size = self.buf.len() as u32;
        self.buf[0..4].copy_from_slice(&size.to_be_bytes());
        self.buf
    }
}

/// Scan OBU stream for OBU_SEQUENCE_HEADER and return a re-emitted copy with
/// obu_has_size_field=1 (required for av1C configOBUs per AV1-ISOBMFF §2.3.3).
///
/// Requires the encoder to emit Low-Overhead-Bitstream (LOB) format with
/// obu_has_size_field set on every OBU — this is the case for rav1e and NVENC.
/// If has_size==0, bail rather than stuff frame data into configOBUs: without
/// a size field the parser can't know where one OBU ends and the next begins.
pub(crate) fn extract_sequence_header(data: &[u8]) -> Result<Vec<u8>> {
    let mut pos = 0;
    while pos < data.len() {
        let header_byte = data[pos];
        pos += 1;
        let obu_type = (header_byte >> 3) & 0x0F;
        let extension_flag = (header_byte >> 2) & 0x1;
        let has_size = (header_byte >> 1) & 0x1;
        if has_size == 0 {
            anyhow::bail!(
                "AV1 packet uses Annex-B style OBUs (obu_has_size_field=0); \
                 expected LOB format from the encoder"
            );
        }
        if extension_flag != 0 {
            if pos >= data.len() {
                anyhow::bail!("truncated OBU extension header");
            }
            pos += 1;
        }
        let (size64, size_len) = read_leb128(&data[pos..])?;
        let size = size64 as usize;
        pos += size_len;
        if pos + size > data.len() {
            anyhow::bail!("OBU payload extends past packet");
        }
        if obu_type == 1 {
            // Re-emit header with ext=0, has_size=1, no temporal/spatial ID.
            let header: u8 = (1 << 3) | (1 << 1);
            let mut out = Vec::with_capacity(1 + 8 + size);
            out.push(header);
            write_leb128(&mut out, size as u64);
            out.extend_from_slice(&data[pos..pos + size]);
            return Ok(out);
        }
        pos += size;
    }
    anyhow::bail!("no OBU_SEQUENCE_HEADER found in first packet")
}

fn read_leb128(data: &[u8]) -> Result<(u64, usize)> {
    let mut value: u64 = 0;
    let mut len = 0usize;
    for i in 0..8 {
        if i >= data.len() {
            anyhow::bail!("truncated leb128");
        }
        let byte = data[i];
        value |= ((byte & 0x7F) as u64) << (i * 7);
        len += 1;
        if (byte & 0x80) == 0 {
            return Ok((value, len));
        }
    }
    anyhow::bail!("leb128 too long")
}

fn write_leb128(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
            out.push(byte);
        } else {
            out.push(byte);
            return;
        }
    }
}

/// Parse AV1 sequence header OBU to extract parameters needed for av1C.
///
/// Returns `(seq_profile, seq_level_idx_0, seq_tier_0,
///          high_bitdepth, twelve_bit, monochrome,
///          chroma_subsampling_x, chroma_subsampling_y, chroma_sample_position)`.
///
/// Defaults match 8-bit 4:2:0 Main profile if parsing fails — the resulting
/// av1C will still be valid for typical rav1e output (profile 0 level 0).
fn parse_seq_header_params(obu: &[u8]) -> (u8, u8, u8, bool, bool, bool, u8, u8, u8) {
    if obu.len() < 2 {
        return (0, 0, 0, false, false, false, 1, 1, 0);
    }
    // Skip OBU header + leb128 size
    let mut pos = 1;
    if obu[0] & 0x02 != 0 {
        // has_size: parse leb128
        match read_leb128(&obu[pos..]) {
            Ok((_, len)) => pos += len,
            Err(_) => return (0, 0, 0, false, false, false, 1, 1, 0),
        }
    }
    if pos >= obu.len() {
        return (0, 0, 0, false, false, false, 1, 1, 0);
    }

    let mut br = BitReader::new(&obu[pos..]);
    let seq_profile = br.bits(3).unwrap_or(0) as u8;
    let _still_picture = br.bits(1).unwrap_or(0);
    let reduced_still_picture_header = br.bits(1).unwrap_or(0);

    let (seq_level_idx_0, seq_tier_0) = if reduced_still_picture_header != 0 {
        (br.bits(5).unwrap_or(0) as u8, 0)
    } else {
        let timing_info_present = br.bits(1).unwrap_or(0);
        if timing_info_present != 0 {
            let _num_units = br.bits(32);
            let _time_scale = br.bits(32);
            let equal_pts = br.bits(1).unwrap_or(0);
            if equal_pts != 0 {
                let _nticks = read_uvlc(&mut br);
            }
            let decoder_model_info_present = br.bits(1).unwrap_or(0);
            if decoder_model_info_present != 0 {
                let _bdlm1 = br.bits(5);
                let _nts = br.bits(32);
                let _brslm1 = br.bits(5);
                let _frpdlm1 = br.bits(5);
            }
        }
        let initial_display_delay_present = br.bits(1).unwrap_or(0);
        let operating_points_cnt_minus_1 = br.bits(5).unwrap_or(0);
        let mut level0 = 0u8;
        let mut tier0 = 0u8;
        for i in 0..=operating_points_cnt_minus_1 {
            let _operating_point_idc = br.bits(12).unwrap_or(0);
            let seq_level_idx_i = br.bits(5).unwrap_or(0) as u8;
            let seq_tier_i = if seq_level_idx_i > 7 {
                br.bits(1).unwrap_or(0) as u8
            } else {
                0
            };
            if i == 0 {
                level0 = seq_level_idx_i;
                tier0 = seq_tier_i;
            }
            // Decoder model / initial_display_delay skipping
            // decoder_model_info_present always 0 in our path above; skip its conditional fields.
            if initial_display_delay_present != 0 {
                let present = br.bits(1).unwrap_or(0);
                if present != 0 {
                    let _iddm1 = br.bits(4);
                }
            }
        }
        (level0, tier0)
    };

    let frame_width_bits_minus_1 = br.bits(4).unwrap_or(0);
    let frame_height_bits_minus_1 = br.bits(4).unwrap_or(0);
    let _max_frame_width_minus_1 = br.bits(frame_width_bits_minus_1 + 1);
    let _max_frame_height_minus_1 = br.bits(frame_height_bits_minus_1 + 1);

    if reduced_still_picture_header == 0 {
        let frame_id_numbers_present = br.bits(1).unwrap_or(0);
        if frame_id_numbers_present != 0 {
            let _delta_fid_len = br.bits(4);
            let _add_fid_len = br.bits(3);
        }
    }
    let _use_128x128 = br.bits(1);
    let _enable_filter_intra = br.bits(1);
    let _enable_intra_edge_filter = br.bits(1);
    if reduced_still_picture_header == 0 {
        let _enable_interintra = br.bits(1);
        let _enable_masked = br.bits(1);
        let _enable_warped = br.bits(1);
        let _enable_dual_filter = br.bits(1);
        let _enable_order_hint = br.bits(1);
        let enable_order_hint = _enable_order_hint.unwrap_or(0);
        if enable_order_hint != 0 {
            let _enable_jnt_comp = br.bits(1);
            let _enable_ref_frame_mvs = br.bits(1);
        }
        let seq_choose_screen_detection_tools = br.bits(1).unwrap_or(0);
        let seq_force_screen_content_tools = if seq_choose_screen_detection_tools != 0 {
            2
        } else {
            br.bits(1).unwrap_or(0)
        };
        if seq_force_screen_content_tools > 0 {
            let seq_choose_integer_mv = br.bits(1).unwrap_or(0);
            if seq_choose_integer_mv == 0 {
                let _seq_force_integer_mv = br.bits(1);
            }
        }
        if enable_order_hint != 0 {
            let _order_hint_bits_minus_1 = br.bits(3);
        }
    }
    let _enable_superres = br.bits(1);
    let _enable_cdef = br.bits(1);
    let _enable_restoration = br.bits(1);

    // color_config() per AV1 §5.5.2
    let high_bitdepth = br.bits(1).unwrap_or(0) != 0;
    let twelve_bit = if seq_profile == 2 && high_bitdepth {
        br.bits(1).unwrap_or(0) != 0
    } else {
        false
    };
    let monochrome = if seq_profile == 1 {
        false
    } else {
        br.bits(1).unwrap_or(0) != 0
    };
    let color_description_present = br.bits(1).unwrap_or(0) != 0;
    let (color_primaries, transfer_characteristics, matrix_coefficients) =
        if color_description_present {
            let cp = br.bits(8).unwrap_or(2) as u8;
            let tc = br.bits(8).unwrap_or(2) as u8;
            let mc = br.bits(8).unwrap_or(2) as u8;
            (cp, tc, mc)
        } else {
            (2u8, 2u8, 2u8) // CP_UNSPECIFIED / TC_UNSPECIFIED / MC_UNSPECIFIED
        };
    let (subsampling_x, subsampling_y, chroma_sample_position) = if monochrome {
        // color_range
        let _color_range = br.bits(1);
        (1u8, 1u8, 0u8)
    } else if color_primaries == 1 /* CP_BT_709 */
        && transfer_characteristics == 13 /* TC_SRGB */
        && matrix_coefficients == 0
    /* MC_IDENTITY */
    {
        // color_range is implicitly full (1), RGB 4:4:4
        (0u8, 0u8, 0u8)
    } else {
        let _color_range = br.bits(1);
        let (sx, sy) = if seq_profile == 0 {
            (1u8, 1u8)
        } else if seq_profile == 1 {
            (0u8, 0u8)
        } else {
            let bit_depth = if high_bitdepth {
                if twelve_bit { 12 } else { 10 }
            } else {
                8
            };
            if bit_depth == 12 {
                let sxb = br.bits(1).unwrap_or(1) as u8;
                let syb = if sxb != 0 {
                    br.bits(1).unwrap_or(1) as u8
                } else {
                    0
                };
                (sxb, syb)
            } else {
                (1u8, 0u8)
            }
        };
        let csp = if sx != 0 && sy != 0 {
            br.bits(2).unwrap_or(0) as u8
        } else {
            0u8
        };
        (sx, sy, csp)
    };
    // separate_uv_deltas follows but we don't emit it; parser state ends here.

    (
        seq_profile,
        seq_level_idx_0,
        seq_tier_0,
        high_bitdepth,
        twelve_bit,
        monochrome,
        subsampling_x,
        subsampling_y,
        chroma_sample_position,
    )
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn bits(&mut self, n: u32) -> Option<u32> {
        let mut v: u32 = 0;
        for _ in 0..n {
            if self.pos / 8 >= self.data.len() {
                return None;
            }
            let byte = self.data[self.pos / 8];
            let bit = (byte >> (7 - (self.pos % 8))) & 1;
            v = (v << 1) | bit as u32;
            self.pos += 1;
        }
        Some(v)
    }
}

fn read_uvlc(br: &mut BitReader) -> u32 {
    let mut leading_zeros = 0u32;
    while leading_zeros < 32 {
        match br.bits(1) {
            Some(0) => leading_zeros += 1,
            Some(_) => break,
            None => return 0,
        }
    }
    if leading_zeros >= 32 {
        return u32::MAX;
    }
    let value = br.bits(leading_zeros).unwrap_or(0);
    value + ((1u32 << leading_zeros) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ftyp_starts_with_size_and_type() {
        let ftyp = build_ftyp(VideoCodec::Av1);
        let size = u32::from_be_bytes([ftyp[0], ftyp[1], ftyp[2], ftyp[3]]);
        assert_eq!(size as usize, ftyp.len());
        assert_eq!(&ftyp[4..8], b"ftyp");
    }

    #[test]
    fn leb128_roundtrip() {
        let mut buf = Vec::new();
        write_leb128(&mut buf, 300);
        let (v, n) = read_leb128(&buf).unwrap();
        assert_eq!(v, 300);
        assert_eq!(n, buf.len());
    }

    #[test]
    fn box_builder_sizes_correctly() {
        let mut b = BoxBuilder::new(b"test");
        b.u32(0xDEADBEEF);
        let out = b.finish();
        assert_eq!(out.len(), 12);
        assert_eq!(&out[4..8], b"test");
        assert_eq!(u32::from_be_bytes([out[0], out[1], out[2], out[3]]), 12);
    }

    // ---- stsc chunk-run tests --------------------------------------------

    /// Parse a `stsc` box bytes → Vec<(first_chunk, samples_per_chunk, sdi)>.
    fn parse_stsc_entries(stsc: &[u8]) -> Vec<(u32, u32, u32)> {
        assert_eq!(&stsc[4..8], b"stsc");
        // size(4) type(4) ver(1) flags(3) count(4)
        let count = u32::from_be_bytes([stsc[12], stsc[13], stsc[14], stsc[15]]) as usize;
        let mut out = Vec::with_capacity(count);
        let mut p = 16usize;
        for _ in 0..count {
            let fc = u32::from_be_bytes([stsc[p], stsc[p + 1], stsc[p + 2], stsc[p + 3]]);
            let spc = u32::from_be_bytes([stsc[p + 4], stsc[p + 5], stsc[p + 6], stsc[p + 7]]);
            let sdi = u32::from_be_bytes([stsc[p + 8], stsc[p + 9], stsc[p + 10], stsc[p + 11]]);
            out.push((fc, spc, sdi));
            p += 12;
        }
        out
    }

    #[test]
    fn mux_stsc_emits_multiple_chunk_runs() {
        // 120 samples at spc=24 → 5 full chunks of 24, no remainder.
        let stsc = build_stsc(120, 24);
        let entries = parse_stsc_entries(&stsc);
        assert_eq!(entries, vec![(1, 24, 1)]);
    }

    #[test]
    fn mux_stsc_last_chunk_under_spc_emits_tail_entry() {
        // 121 samples at spc=24 → 5 full chunks + 1 tail of 1.
        let stsc = build_stsc(121, 24);
        let entries = parse_stsc_entries(&stsc);
        assert_eq!(entries, vec![(1, 24, 1), (6, 1, 1)]);
    }

    #[test]
    fn mux_stsc_all_under_spc_single_entry() {
        // 10 samples at spc=24 → one partial chunk.
        let stsc = build_stsc(10, 24);
        let entries = parse_stsc_entries(&stsc);
        assert_eq!(entries, vec![(1, 10, 1)]);
    }

    // ---- chunk offset computation ----------------------------------------

    #[test]
    fn compute_chunk_offsets_walks_sample_sizes() {
        let sizes = vec![100u32, 200, 300, 400, 500, 600, 700];
        let offs = compute_chunk_offsets(1000, &sizes, 3);
        // chunks: [0..3]=1000, [3..6]=1000+600=1600, [6..7]=1600+1500=3100
        assert_eq!(offs, vec![1000, 1600, 3100]);
    }

    #[test]
    fn compute_chunk_offsets_single_chunk() {
        let sizes = vec![10u32; 5];
        let offs = compute_chunk_offsets(42, &sizes, 120);
        assert_eq!(offs, vec![42]);
    }

    // ---- stco / co64 ------------------------------------------------------

    #[test]
    fn build_stco_emits_32bit_offsets() {
        let offs = vec![8u64, 1_000_000, u32::MAX as u64];
        let box_bytes = build_stco(&offs);
        assert_eq!(&box_bytes[4..8], b"stco");
        let count =
            u32::from_be_bytes([box_bytes[12], box_bytes[13], box_bytes[14], box_bytes[15]]);
        assert_eq!(count, 3);
        // 3 × 4 = 12 entry bytes. Header: 4 size + 4 type + 1 ver + 3 flags + 4 count = 16.
        assert_eq!(box_bytes.len(), 16 + 12);
        let last = u32::from_be_bytes([box_bytes[24], box_bytes[25], box_bytes[26], box_bytes[27]]);
        assert_eq!(last, u32::MAX);
    }

    #[test]
    fn build_co64_emits_64bit_offsets() {
        let big = (u32::MAX as u64) + 100;
        let offs = vec![8u64, big, big + 1_000_000];
        let box_bytes = build_co64(&offs);
        assert_eq!(&box_bytes[4..8], b"co64");
        let count =
            u32::from_be_bytes([box_bytes[12], box_bytes[13], box_bytes[14], box_bytes[15]]);
        assert_eq!(count, 3);
        // 3 × 8 = 24 entry bytes. Header = 16.
        assert_eq!(box_bytes.len(), 16 + 24);
        // Second entry: bytes 24..32.
        let got = u64::from_be_bytes([
            box_bytes[24],
            box_bytes[25],
            box_bytes[26],
            box_bytes[27],
            box_bytes[28],
            box_bytes[29],
            box_bytes[30],
            box_bytes[31],
        ]);
        assert_eq!(got, big);
    }

    #[test]
    fn build_co64_offsets_are_monotonic_and_be() {
        // Craft a descending payload input to guard against accidental
        // little-endian or re-sort bugs.
        let offs: Vec<u64> = (0..5)
            .map(|i| 10_000_000_000u64 + i as u64 * 4096)
            .collect();
        let box_bytes = build_co64(&offs);
        let mut prev = 0u64;
        for i in 0..5 {
            let p = 16 + i * 8;
            let v = u64::from_be_bytes([
                box_bytes[p],
                box_bytes[p + 1],
                box_bytes[p + 2],
                box_bytes[p + 3],
                box_bytes[p + 4],
                box_bytes[p + 5],
                box_bytes[p + 6],
                box_bytes[p + 7],
            ]);
            assert!(v > prev, "offsets not monotonic: {v} after {prev}");
            prev = v;
        }
    }

    // ---- moov-level stco vs co64 -----------------------------------------

    /// Find a 4-cc occurrence in a byte slice. Used to assert presence of
    /// `co64`/`stco` in built moov blobs. Returns None if absent.
    fn find_fourcc(data: &[u8], tag: &[u8; 4]) -> Option<usize> {
        data.windows(4).position(|w| w == tag)
    }

    #[test]
    fn moov_with_use_co64_true_emits_co64_not_stco() {
        let sample_sizes = vec![1000u32; 120];
        // Offsets span past u32::MAX — representative of a 5 GiB file.
        let chunk_offsets: Vec<u64> = (0..5)
            .map(|i| (u32::MAX as u64) + i * 1_000_000_000)
            .collect();
        // Minimal config_obus — content is opaque to stbl layout.
        let config_obus = vec![0x0Au8, 0x03, 0x00, 0x00, 0x00];
        let moov = build_moov(
            1920,
            1080,
            90_000,
            120 * 3750,
            3750,
            &sample_sizes,
            &[],
            &config_obus,
            &chunk_offsets,
            24,
            true,
        );
        assert!(find_fourcc(&moov, b"co64").is_some(), "co64 box missing");
        // NB: must check for standalone `stco` not a substring — `stco` can
        // appear in payload or other labels. Use exact 4-byte box-type match.
        assert!(
            find_fourcc(&moov, b"stco").is_none(),
            "stco present when co64 chosen"
        );
    }

    #[test]
    fn moov_with_use_co64_false_emits_stco_not_co64() {
        let sample_sizes = vec![1000u32; 120];
        let chunk_offsets: Vec<u64> = (0..5).map(|i| 1000 + i * 24_000).collect();
        let config_obus = vec![0x0Au8, 0x03, 0x00, 0x00, 0x00];
        let moov = build_moov(
            1920,
            1080,
            90_000,
            120 * 3750,
            3750,
            &sample_sizes,
            &[],
            &config_obus,
            &chunk_offsets,
            24,
            false,
        );
        assert!(find_fourcc(&moov, b"stco").is_some(), "stco box missing");
        assert!(
            find_fourcc(&moov, b"co64").is_none(),
            "co64 present when stco chosen"
        );
    }

    // ---- Apple-compat: ftyp brands ---------------------------------------

    /// AV1-ISOBMFF v1.3.0 §2.1 mandates `av01` in `compatible_brands`. Apple
    /// QuickTime / iOS Safari additionally need a structural ISOBMFF brand
    /// (`iso6` covers co64 / largesize from 14496-12 sixth edition). `mp42`
    /// is conventional for AAC parsing rules.
    #[test]
    fn ftyp_lists_av01_and_iso6_and_mp42_brands() {
        let ftyp = build_ftyp(VideoCodec::Av1);
        // major_brand at offset 8..12 (after size + 'ftyp')
        assert_eq!(&ftyp[8..12], b"iso6", "major_brand should be iso6");
        // After major(4) + minor(4) the compatible_brands list runs to end.
        let compat = &ftyp[16..];
        let brands: Vec<&[u8]> = compat.chunks_exact(4).collect();
        assert!(
            brands.contains(&b"av01".as_ref()),
            "compatible_brands must list av01 per AV1-ISOBMFF §2.1; got {:?}",
            brands
        );
        assert!(
            brands.contains(&b"iso6".as_ref()),
            "compatible_brands must list iso6 (14496-12 v6 — covers co64/largesize)"
        );
        assert!(
            brands.contains(&b"mp42".as_ref()),
            "compatible_brands should list mp42 for AAC parsing rules"
        );
    }

    // ---- Apple-compat: colr nclx atom ------------------------------------

    /// Find every occurrence of the 4-byte tag (used for assertions where
    /// the tag may legitimately appear inside payload too).
    fn count_fourcc_occurrences(data: &[u8], tag: &[u8; 4]) -> usize {
        data.windows(4).filter(|w| *w == tag).count()
    }

    #[test]
    fn av01_sample_entry_includes_colr_nclx_box() {
        let cm = ColorMetadata::default();
        let sample_sizes = vec![100u32; 30];
        let chunk_offsets: Vec<u64> = vec![1000];
        let config_obus = vec![0x0Au8, 0x03, 0x00, 0x00, 0x00];
        let _ = (&sample_sizes, &chunk_offsets);
        let moov = build_av01(1920, 1080, &config_obus, &cm);
        let colr_pos = find_fourcc(&moov, b"colr").expect("colr atom missing");
        // Body layout: [pos-4..pos] = size, [pos..pos+4] = 'colr',
        // [pos+4..pos+8] = colour_type, then 6 bytes nclx fields.
        assert_eq!(
            &moov[colr_pos + 4..colr_pos + 8],
            b"nclx",
            "colour_type must be 'nclx' per ISO/IEC 23001-8"
        );
        // colour_primaries (u16 BE) at +8..+10
        let cp = u16::from_be_bytes([moov[colr_pos + 8], moov[colr_pos + 9]]);
        assert_eq!(cp, 1, "default BT.709 colour_primaries=1");
        // transfer_characteristics at +10..+12
        let tc = u16::from_be_bytes([moov[colr_pos + 10], moov[colr_pos + 11]]);
        assert_eq!(tc, 1, "default BT.709 transfer_characteristics=1");
        // matrix_coefficients at +12..+14
        let mc = u16::from_be_bytes([moov[colr_pos + 12], moov[colr_pos + 13]]);
        assert_eq!(mc, 1, "default BT.709 matrix_coefficients=1");
        // full_range_flag is the high bit of the byte at +14
        let fr = moov[colr_pos + 14];
        assert_eq!(fr & 0x80, 0x00, "default limited-range full_range_flag=0");
    }

    #[test]
    fn colr_nclx_carries_hdr10_metadata() {
        // HDR10: BT.2020 NCL primaries (9), ST 2084 PQ transfer (16),
        // BT.2020 NCL matrix (9), limited range. This is the canonical
        // HDR10 nclx triple — Apple's player needs it to apply PQ tone
        // mapping correctly.
        let cm = ColorMetadata {
            transfer: codec::frame::TransferFn::St2084,
            matrix_coefficients: 9,
            colour_primaries: 9,
            full_range: false,
            ..ColorMetadata::default()
        };
        let colr = build_colr_nclx(&cm);
        assert_eq!(&colr[4..8], b"colr");
        assert_eq!(&colr[8..12], b"nclx");
        let cp = u16::from_be_bytes([colr[12], colr[13]]);
        let tc = u16::from_be_bytes([colr[14], colr[15]]);
        let mc = u16::from_be_bytes([colr[16], colr[17]]);
        let fr = colr[18];
        assert_eq!(cp, 9, "BT.2020 NCL primaries");
        assert_eq!(tc, 16, "ST 2084 PQ transfer");
        assert_eq!(mc, 9, "BT.2020 NCL matrix");
        assert_eq!(fr & 0x80, 0x00, "HDR10 typically signals limited range");
    }

    #[test]
    fn colr_nclx_full_range_sets_high_bit() {
        let cm = ColorMetadata {
            transfer: codec::frame::TransferFn::Bt709,
            matrix_coefficients: 1,
            colour_primaries: 1,
            full_range: true,
            ..ColorMetadata::default()
        };
        let colr = build_colr_nclx(&cm);
        assert_eq!(colr[18] & 0x80, 0x80, "full_range high bit must be set");
        // Low 7 bits are reserved-zero per ISO 23001-8.
        assert_eq!(colr[18] & 0x7F, 0x00, "reserved bits must be zero");
    }

    #[test]
    fn colr_nclx_box_size_matches_layout() {
        // Box: 4 size + 4 'colr' + 4 colour_type + 2 cp + 2 tc + 2 mc + 1 packed = 19 bytes.
        let colr = build_colr_nclx(&ColorMetadata::default());
        let size = u32::from_be_bytes([colr[0], colr[1], colr[2], colr[3]]) as usize;
        assert_eq!(
            size,
            colr.len(),
            "colr box size field must equal box length"
        );
        assert_eq!(size, 19, "colr nclx must be exactly 19 bytes");
    }

    /// Sanity: the `colr` atom must live inside the visual sample entry,
    /// not float at the moov / trak / stbl level. Players look for it
    /// nested inside `av01` (or `avc1`/`hvc1`) in `stsd`.
    #[test]
    fn colr_lives_inside_av01_sample_entry() {
        let cm = ColorMetadata::default();
        let sample_sizes = vec![100u32; 30];
        let chunk_offsets: Vec<u64> = vec![1000];
        let config_obus = vec![0x0Au8, 0x03, 0x00, 0x00, 0x00];
        let _ = (&sample_sizes, &chunk_offsets);
        let moov = build_av01(1920, 1080, &config_obus, &cm);
        let av01_pos = find_fourcc(&moov, b"av01").expect("av01 sample entry missing");
        let av01_size = u32::from_be_bytes([
            moov[av01_pos - 4],
            moov[av01_pos - 3],
            moov[av01_pos - 2],
            moov[av01_pos - 1],
        ]) as usize;
        let av01_end = av01_pos - 4 + av01_size;
        let colr_pos = find_fourcc(&moov, b"colr").expect("colr missing");
        assert!(
            colr_pos > av01_pos && colr_pos < av01_end,
            "colr must be nested inside av01 sample entry: av01@{}..{} colr@{}",
            av01_pos,
            av01_end,
            colr_pos
        );
        assert_eq!(
            count_fourcc_occurrences(&moov, b"colr"),
            1,
            "exactly one colr atom expected"
        );
    }

    // ---- mdat 64-bit largesize -------------------------------------------

    /// transfer_to_h273 should round-trip through the H.273 codes the
    /// pipeline knows about. The Bt709 enum variant collapses 4 H.273
    /// codes (1, 6, 14, 15) — we always emit the canonical 1 on write.
    #[test]
    fn transfer_to_h273_emits_canonical_codes() {
        use codec::frame::TransferFn;
        assert_eq!(transfer_to_h273(TransferFn::Bt709), 1);
        assert_eq!(transfer_to_h273(TransferFn::Bt470Bg), 4);
        assert_eq!(transfer_to_h273(TransferFn::Linear), 8);
        assert_eq!(transfer_to_h273(TransferFn::St2084), 16);
        assert_eq!(transfer_to_h273(TransferFn::AribStdB67), 18);
        assert_eq!(transfer_to_h273(TransferFn::Unspecified), 2);
    }

    // ---- HDR atoms: mdcv (Mastering Display Color Volume) ----------------

    /// HDR10-canonical mastering display values: BT.2020 primaries +
    /// D65 white point + 1000 nits / 0.0001 nits luminance, all in the
    /// HEVC SEI 137 / SMPTE ST 2086 spec-domain integer encoding.
    ///
    /// Cross-references for the wire numbers (so future reviewers can
    /// re-derive without chasing a spec PDF):
    ///   BT.2020 R primary  (0.708 , 0.292)  → (35400, 14600)
    ///   BT.2020 G primary  (0.170 , 0.797)  → ( 8500, 39850)
    ///   BT.2020 B primary  (0.131 , 0.046)  → ( 6550,  2300)
    ///   D65 white point    (0.3127, 0.3290) → (15635, 16450)
    ///   max luminance       1000 cd/m²      → 10_000_000  (0.0001 cd/m² steps)
    ///   min luminance       0.0001 cd/m²    →          1
    fn hdr10_mastering_display() -> codec::frame::MasteringDisplay {
        codec::frame::MasteringDisplay {
            primaries_r_x: 35400,
            primaries_r_y: 14600,
            primaries_g_x: 8500,
            primaries_g_y: 39850,
            primaries_b_x: 6550,
            primaries_b_y: 2300,
            white_point_x: 15635,
            white_point_y: 16450,
            max_luminance: 10_000_000,
            min_luminance: 1,
        }
    }

    /// 24-byte payload + 8-byte header = 32 bytes. Bytes laid out big-endian.
    /// Box-type is `'mdcv'` (NOT `'SmDm'`).
    #[test]
    fn mdcv_box_24_byte_payload_layout() {
        let md = hdr10_mastering_display();
        let mdcv = build_mdcv(&md);
        assert_eq!(
            mdcv.len(),
            32,
            "mdcv box must be exactly 32 bytes (8 header + 24 payload)"
        );
        let size = u32::from_be_bytes([mdcv[0], mdcv[1], mdcv[2], mdcv[3]]) as usize;
        assert_eq!(size, mdcv.len(), "size field must equal box length");
        assert_eq!(&mdcv[4..8], b"mdcv", "box type must be 'mdcv' (not 'SmDm')");
        // Body fields, all u16 BE except the trailing two u32s.
        let u16_at = |off: usize| u16::from_be_bytes([mdcv[off], mdcv[off + 1]]);
        let u32_at = |off: usize| {
            u32::from_be_bytes([mdcv[off], mdcv[off + 1], mdcv[off + 2], mdcv[off + 3]])
        };
        assert_eq!(u16_at(8), 35400, "primaries_r_x");
        assert_eq!(u16_at(10), 14600, "primaries_r_y");
        assert_eq!(u16_at(12), 8500, "primaries_g_x");
        assert_eq!(u16_at(14), 39850, "primaries_g_y");
        assert_eq!(u16_at(16), 6550, "primaries_b_x");
        assert_eq!(u16_at(18), 2300, "primaries_b_y");
        assert_eq!(u16_at(20), 15635, "white_point_x");
        assert_eq!(u16_at(22), 16450, "white_point_y");
        assert_eq!(u32_at(24), 10_000_000, "max_luminance (0.0001 cd/m² steps)");
        assert_eq!(u32_at(28), 1, "min_luminance");
    }

    /// 4-byte payload + 8-byte header = 12 bytes. Box-type is `'clli'`
    /// (NOT `'CoLL'`).
    #[test]
    fn clli_box_4_byte_payload_layout() {
        let cll = codec::frame::ContentLightLevel {
            max_cll: 1000,
            max_fall: 400,
        };
        let clli = build_clli(&cll);
        assert_eq!(
            clli.len(),
            12,
            "clli box must be exactly 12 bytes (8 header + 4 payload)"
        );
        let size = u32::from_be_bytes([clli[0], clli[1], clli[2], clli[3]]) as usize;
        assert_eq!(size, clli.len(), "size field must equal box length");
        assert_eq!(&clli[4..8], b"clli", "box type must be 'clli' (not 'CoLL')");
        let max_cll = u16::from_be_bytes([clli[8], clli[9]]);
        let max_fall = u16::from_be_bytes([clli[10], clli[11]]);
        assert_eq!(max_cll, 1000, "max_cll");
        assert_eq!(max_fall, 400, "max_fall");
    }

    /// When mastering_display is None, the av01 sample entry must omit
    /// the `mdcv` box entirely. SDR sources should produce a moov with
    /// no `mdcv` 4cc anywhere.
    #[test]
    fn mdcv_omitted_when_none() {
        let cm = ColorMetadata::default(); // None, None
        let sample_sizes = vec![100u32; 30];
        let chunk_offsets: Vec<u64> = vec![1000];
        let config_obus = vec![0x0Au8, 0x03, 0x00, 0x00, 0x00];
        let moov = build_moov_any(
            1920,
            1080,
            90_000,
            90_000,
            30 * 3000,
            30 * 3000,
            3000,
            &sample_sizes,
            &[],
            &config_obus,
            &chunk_offsets,
            30,
            None,
            &[],
            false,
            &cm,
        );
        assert!(
            find_fourcc(&moov, b"mdcv").is_none(),
            "SDR (mastering_display=None) moov must NOT contain mdcv box"
        );
    }

    /// When content_light_level is None, the av01 sample entry must omit
    /// the `clli` box entirely.
    #[test]
    fn clli_omitted_when_none() {
        let cm = ColorMetadata::default();
        let sample_sizes = vec![100u32; 30];
        let chunk_offsets: Vec<u64> = vec![1000];
        let config_obus = vec![0x0Au8, 0x03, 0x00, 0x00, 0x00];
        let moov = build_moov_any(
            1920,
            1080,
            90_000,
            90_000,
            30 * 3000,
            30 * 3000,
            3000,
            &sample_sizes,
            &[],
            &config_obus,
            &chunk_offsets,
            30,
            None,
            &[],
            false,
            &cm,
        );
        assert!(
            find_fourcc(&moov, b"clli").is_none(),
            "SDR (content_light_level=None) moov must NOT contain clli box"
        );
    }

    /// AV1-ISOBMFF v1.3.0 §2.3.4 + §2.3.5 prescribe the order
    /// `colr → mdcv → clli` inside the visual sample entry. Players
    /// scan by 4cc so order is recommended-not-required, but matching
    /// the spec keeps us defensible against strict validators
    /// (mp4parser, GPAC's mp4box -info).
    #[test]
    fn av01_sample_entry_emits_mdcv_and_clli_in_order() {
        let cm = ColorMetadata {
            transfer: codec::frame::TransferFn::St2084,
            matrix_coefficients: 9,
            colour_primaries: 9,
            full_range: false,
            mastering_display: Some(hdr10_mastering_display()),
            content_light_level: Some(codec::frame::ContentLightLevel {
                max_cll: 1000,
                max_fall: 400,
            }),
        };
        let sample_sizes = vec![100u32; 30];
        let chunk_offsets: Vec<u64> = vec![1000];
        let config_obus = vec![0x0Au8, 0x03, 0x00, 0x00, 0x00];
        let _ = (&sample_sizes, &chunk_offsets);
        let moov = build_av01(1920, 1080, &config_obus, &cm);
        let av01_pos = find_fourcc(&moov, b"av01").expect("av01 sample entry missing");
        let av01_size = u32::from_be_bytes([
            moov[av01_pos - 4],
            moov[av01_pos - 3],
            moov[av01_pos - 2],
            moov[av01_pos - 1],
        ]) as usize;
        let av01_end = av01_pos - 4 + av01_size;
        let av01_body = &moov[av01_pos..av01_end];
        let colr_rel = av01_body
            .windows(4)
            .position(|w| w == b"colr")
            .expect("colr nested in av01");
        let mdcv_rel = av01_body
            .windows(4)
            .position(|w| w == b"mdcv")
            .expect("mdcv nested in av01");
        let clli_rel = av01_body
            .windows(4)
            .position(|w| w == b"clli")
            .expect("clli nested in av01");
        assert!(
            colr_rel < mdcv_rel,
            "colr ({}) must precede mdcv ({})",
            colr_rel,
            mdcv_rel
        );
        assert!(
            mdcv_rel < clli_rel,
            "mdcv ({}) must precede clli ({})",
            mdcv_rel,
            clli_rel
        );
        // Exactly one of each, all under av01.
        assert_eq!(
            count_fourcc_occurrences(&moov, b"mdcv"),
            1,
            "exactly one mdcv expected"
        );
        assert_eq!(
            count_fourcc_occurrences(&moov, b"clli"),
            1,
            "exactly one clli expected"
        );
    }

    // ---- colr nclx HDR transfer-code coverage (Squad-18 verification) ----

    /// PQ transfer (HDR10) is H.273 transfer_characteristics = 16. Apple
    /// + browsers key off this code to apply the ST 2084 EOTF; emitting
    /// 1 (BT.709) here would render HDR10 as washed-out SDR.
    #[test]
    fn colr_handles_pq_transfer_code_16() {
        let cm = ColorMetadata {
            transfer: codec::frame::TransferFn::St2084,
            matrix_coefficients: 9,
            colour_primaries: 9,
            full_range: false,
            ..ColorMetadata::default()
        };
        let colr = build_colr_nclx(&cm);
        let tc = u16::from_be_bytes([colr[14], colr[15]]);
        assert_eq!(tc, 16, "PQ transfer must encode as H.273 code 16");
    }

    /// HLG transfer is H.273 transfer_characteristics = 18. Same role as
    /// PQ but for broadcast HDR; players that support HLG read 18 to
    /// activate the ARIB STD-B67 OETF.
    #[test]
    fn colr_handles_hlg_transfer_code_18() {
        let cm = ColorMetadata {
            transfer: codec::frame::TransferFn::AribStdB67,
            matrix_coefficients: 9,
            colour_primaries: 9,
            full_range: false,
            ..ColorMetadata::default()
        };
        let colr = build_colr_nclx(&cm);
        let tc = u16::from_be_bytes([colr[14], colr[15]]);
        assert_eq!(tc, 18, "HLG transfer must encode as H.273 code 18");
    }

    /// BT.2020 colour_primaries = 9, matrix_coefficients = 9 (NCL) or 10
    /// (CL). Both must round-trip verbatim — the pipeline preserves the
    /// raw u8 from the source SPS so the encode side can pick the right
    /// matrix back out.
    #[test]
    fn colr_bt2020_primaries_matrix() {
        // NCL variant (most common — matrix_coefficients = 9)
        let cm_ncl = ColorMetadata {
            transfer: codec::frame::TransferFn::St2084,
            matrix_coefficients: 9,
            colour_primaries: 9,
            full_range: false,
            ..ColorMetadata::default()
        };
        let colr_ncl = build_colr_nclx(&cm_ncl);
        let cp_ncl = u16::from_be_bytes([colr_ncl[12], colr_ncl[13]]);
        let mc_ncl = u16::from_be_bytes([colr_ncl[16], colr_ncl[17]]);
        assert_eq!(cp_ncl, 9, "BT.2020 colour_primaries must be 9");
        assert_eq!(mc_ncl, 9, "BT.2020 NCL matrix must be 9");

        // CL variant (matrix_coefficients = 10)
        let cm_cl = ColorMetadata {
            matrix_coefficients: 10,
            ..cm_ncl
        };
        let colr_cl = build_colr_nclx(&cm_cl);
        let mc_cl = u16::from_be_bytes([colr_cl[16], colr_cl[17]]);
        assert_eq!(
            mc_cl, 10,
            "BT.2020 CL matrix must be 10 (preserved verbatim)"
        );
    }

    // ---- Squad-23: Opus + dOps box layout (RFC 7845) ---------------------

    /// Standard OpusHead body for stereo @ 48 kHz with PreSkip = 312
    /// (the typical libopus encoder lookahead at 48 kHz). Output gain = 0,
    /// ChannelMappingFamily = 0 (stereo).
    ///
    /// Layout (post-magic body, 11 bytes; LE numeric fields per RFC 7845
    /// §5.1):
    ///   [0]    Version=1
    ///   [1]    OutputChannelCount=2
    ///   [2..4] PreSkip=312 LE → 38 01
    ///   [4..8] InputSampleRate=48000 LE → 80 BB 00 00
    ///   [8..10] OutputGain=0 LE → 00 00
    ///   [10]   ChannelMappingFamily=0
    fn opus_head_stereo_48k_preskip_312() -> Vec<u8> {
        let mut head = Vec::with_capacity(11);
        head.push(1u8); // Version
        head.push(2u8); // OutputChannelCount
        head.extend_from_slice(&312u16.to_le_bytes()); // PreSkip
        head.extend_from_slice(&48_000u32.to_le_bytes()); // InputSampleRate
        head.extend_from_slice(&0i16.to_le_bytes()); // OutputGain
        head.push(0u8); // ChannelMappingFamily
        head
    }

    fn opus_info_stereo_48k() -> AudioInfo {
        AudioInfo {
            codec: "opus".into(),
            sample_rate: 48_000,
            channels: 2,
            timescale: 48_000,
            asc_bytes: Vec::new(),
            codec_private: opus_head_stereo_48k_preskip_312(),
        }
    }

    /// `dOps` body layout per RFC 7845 §4.5: 11-byte minimum. Box wrapper
    /// adds 8-byte ISOBMFF header → total 19 bytes for ChannelMappingFamily=0.
    /// Numeric fields are big-endian (NOT the little-endian convention of
    /// the OpusHead source bytes).
    #[test]
    fn dops_box_11_byte_payload_layout() {
        let info = opus_info_stereo_48k();
        let dops = build_dops(&info);
        assert_eq!(
            dops.len(),
            19,
            "dOps must be exactly 19 bytes (8 header + 11 payload)"
        );
        let size = u32::from_be_bytes([dops[0], dops[1], dops[2], dops[3]]) as usize;
        assert_eq!(size, dops.len(), "size field must equal box length");
        assert_eq!(
            &dops[4..8],
            b"dOps",
            "box type must be 'dOps' (capital O lowercase ps)"
        );
        // Body fields, all BE per §4.5.
        assert_eq!(dops[8], 0, "Version (RFC 7845 §4.5: MUST be 0)");
        assert_eq!(dops[9], 2, "OutputChannelCount = stereo");
        let pre_skip = u16::from_be_bytes([dops[10], dops[11]]);
        assert_eq!(pre_skip, 312, "PreSkip = 312 (BE)");
        let input_sample_rate = u32::from_be_bytes([dops[12], dops[13], dops[14], dops[15]]);
        assert_eq!(input_sample_rate, 48_000, "InputSampleRate = 48000 (BE)");
        let output_gain = i16::from_be_bytes([dops[16], dops[17]]);
        assert_eq!(output_gain, 0, "OutputGain = 0 (Q8 dB, BE)");
        assert_eq!(dops[18], 0, "ChannelMappingFamily = 0 (mono/stereo)");
    }

    /// The byte-order conversion between OpusHead (LE) and dOps (BE) is
    /// the load-bearing piece — easy to mess up. PreSkip=312 in LE is
    /// `38 01`; in BE it must come back out as `01 38`.
    #[test]
    fn dops_byte_order_flipped_from_opushead() {
        let info = opus_info_stereo_48k();
        // Sanity check the input is in LE.
        assert_eq!(
            info.codec_private[2..4],
            [0x38, 0x01],
            "OpusHead PreSkip must be LE"
        );
        let dops = build_dops(&info);
        // PreSkip in dOps body = bytes 10..12 of the box (after 8-byte header).
        assert_eq!(
            dops[10..12],
            [0x01, 0x38],
            "dOps PreSkip must be BE — got {:02X?}",
            &dops[10..12]
        );
    }

    /// `Opus` sample entry per RFC 7845 §4.4. Same generic AudioSampleEntry
    /// preamble as `mp4a` (36 bytes including header) plus the dOps child.
    /// Total = 36 + 19 = 55 bytes for the minimum-channel-count case.
    /// 4-cc is `Opus` exactly (capital O).
    #[test]
    fn opus_sample_entry_size_and_fourcc() {
        let info = opus_info_stereo_48k();
        let entry = build_opus_sample_entry(&info);
        let size = u32::from_be_bytes([entry[0], entry[1], entry[2], entry[3]]) as usize;
        assert_eq!(size, entry.len(), "size field must equal box length");
        assert_eq!(&entry[4..8], b"Opus", "4-cc MUST be 'Opus' (capital O)");
        assert_ne!(&entry[4..8], b"opus", "lowercase 'opus' is non-conformant");
        // Total = 36 (sample entry preamble inc 8-byte header) + 19 (dOps) = 55.
        assert_eq!(
            entry.len(),
            55,
            "Opus sample entry should be 55 bytes for stereo + dOps minimum"
        );
    }

    /// AudioSampleEntry-level samplerate field inside `Opus` MUST be
    /// 48000 << 16 — RFC 7845 §3 mandates 48 kHz internally; emitting
    /// the source's nominal rate (e.g. 44100) would mismatch dOps and
    /// confuse strict validators.
    #[test]
    fn opus_sample_entry_samplerate_is_48000_q16() {
        let info = AudioInfo {
            // Source nominal sample_rate is 44100, but the sample-entry
            // and mdhd MUST report 48000.
            sample_rate: 44_100,
            ..opus_info_stereo_48k()
        };
        let entry = build_opus_sample_entry(&info);
        // Layout offsets inside the sample entry (after the 8-byte box header):
        //   reserved[6]+data_ref(2)=8, reserved2(8)=16, channelcount(2)=18,
        //   sample_size(2)=20, pre_def(2)=22, reserved3(2)=24,
        //   samplerate u32 16.16 at +24..+28.
        // So box-relative offset 8 + 24 = 32.
        let sr_q16 = u32::from_be_bytes([entry[32], entry[33], entry[34], entry[35]]);
        assert_eq!(
            sr_q16,
            48_000u32 << 16,
            "samplerate field MUST be 48000<<16 (Q16); got 0x{:08X}",
            sr_q16
        );
    }

    /// `dOps` must nest inside the `Opus` sample entry. The build_audio_stsd
    /// dispatcher routes Opus → build_opus_sample_entry → dOps child.
    #[test]
    fn dops_nests_inside_opus_sample_entry() {
        let info = opus_info_stereo_48k();
        let entry = build_opus_sample_entry(&info);
        let dops_pos = entry
            .windows(4)
            .position(|w| w == b"dOps")
            .expect("dOps child missing inside Opus sample entry");
        // dOps must come AFTER the 36-byte AudioSampleEntry preamble.
        assert!(
            dops_pos > 28,
            "dOps must come after the AudioSampleEntry preamble; got pos={}",
            dops_pos
        );
    }

    /// stsd dispatcher: AAC info → mp4a; Opus info → Opus. The dispatcher
    /// must NEVER produce mp4a for Opus or Opus for AAC.
    #[test]
    fn stsd_dispatcher_routes_codec_to_correct_sample_entry() {
        let aac = AudioInfo {
            codec: "aac".into(),
            sample_rate: 44_100,
            channels: 2,
            timescale: 44_100,
            asc_bytes: vec![0x12, 0x10],
            codec_private: Vec::new(),
        };
        let stsd_aac = build_audio_stsd(&aac);
        assert!(
            stsd_aac.windows(4).any(|w| w == b"mp4a"),
            "AAC stsd must contain mp4a"
        );
        assert!(
            !stsd_aac.windows(4).any(|w| w == b"Opus"),
            "AAC stsd must NOT contain Opus"
        );
        assert!(
            stsd_aac.windows(4).any(|w| w == b"esds"),
            "AAC stsd must contain esds"
        );

        let opus = opus_info_stereo_48k();
        let stsd_opus = build_audio_stsd(&opus);
        assert!(
            stsd_opus.windows(4).any(|w| w == b"Opus"),
            "Opus stsd must contain Opus"
        );
        assert!(
            !stsd_opus.windows(4).any(|w| w == b"mp4a"),
            "Opus stsd must NOT contain mp4a"
        );
        assert!(
            stsd_opus.windows(4).any(|w| w == b"dOps"),
            "Opus stsd must contain dOps"
        );
        assert!(
            !stsd_opus.windows(4).any(|w| w == b"esds"),
            "Opus stsd must NOT contain esds"
        );
    }

    /// Negative output gain (-3 dB Q8 = -768) round-trips correctly through
    /// the i16-as-u16 BE conversion.
    #[test]
    fn dops_handles_negative_output_gain() {
        let mut head = opus_head_stereo_48k_preskip_312();
        // OutputGain at offset 8..10. Set to -768 (i.e. -3 dB Q8).
        let gain: i16 = -768;
        head[8..10].copy_from_slice(&gain.to_le_bytes());
        let info = AudioInfo {
            codec_private: head,
            ..opus_info_stereo_48k()
        };
        let dops = build_dops(&info);
        let recovered = i16::from_be_bytes([dops[16], dops[17]]);
        assert_eq!(
            recovered, -768,
            "negative OutputGain must survive LE→BE roundtrip"
        );
    }

    /// PreSkip from the encoder's actual `OPUS_GET_LOOKAHEAD` (often
    /// non-default like 156, 312, 480) must round-trip verbatim — we
    /// don't normalize to 312.
    #[test]
    fn dops_preserves_arbitrary_preskip() {
        for &expected in &[0u16, 156, 312, 480, 1024, 65535] {
            let mut head = opus_head_stereo_48k_preskip_312();
            head[2..4].copy_from_slice(&expected.to_le_bytes());
            let info = AudioInfo {
                codec_private: head,
                ..opus_info_stereo_48k()
            };
            let dops = build_dops(&info);
            let got = u16::from_be_bytes([dops[10], dops[11]]);
            assert_eq!(got, expected, "PreSkip {} must survive LE→BE", expected);
        }
    }

    // ---- Squad-28: multichannel Opus dOps family=1 ----------------------

    /// Build an OpusHead body for an N-channel surround layout per
    /// RFC 7845 §5.1. Layout matches what Squad-28's
    /// `OpusEncoder::extra_data()` emits and what an MKV/WebM
    /// `CodecPrivate` carries verbatim. All multi-byte fields LE.
    fn opus_head_surround(
        channels: u8,
        pre_skip: u16,
        input_sample_rate: u32,
        streams: u8,
        coupled: u8,
        mapping: &[u8],
    ) -> Vec<u8> {
        assert_eq!(mapping.len(), channels as usize);
        let mut h = Vec::with_capacity(11 + 2 + channels as usize);
        h.push(1u8); // Version
        h.push(channels);
        h.extend_from_slice(&pre_skip.to_le_bytes());
        h.extend_from_slice(&input_sample_rate.to_le_bytes());
        h.extend_from_slice(&0i16.to_le_bytes()); // OutputGain
        h.push(1u8); // ChannelMappingFamily=1
        h.push(streams);
        h.push(coupled);
        h.extend_from_slice(mapping);
        h
    }

    fn opus_info_5_1() -> AudioInfo {
        // RFC 7845 §5.1.1.2 5.1 layout: streams=4, coupled=2,
        // mapping = [0, 4, 1, 2, 3, 5]. PreSkip=312 (typical libopus
        // lookahead).
        let cp = opus_head_surround(6, 312, 48_000, 4, 2, &[0, 4, 1, 2, 3, 5]);
        AudioInfo {
            codec: "opus".into(),
            sample_rate: 48_000,
            channels: 6,
            timescale: 48_000,
            asc_bytes: Vec::new(),
            codec_private: cp,
        }
    }

    /// 5.1 dOps box payload = 11 + 2 + 6 = 19 bytes; with the 8-byte
    /// box header the total is 27 bytes. All numeric fields BE inside
    /// the box; the trailing channel-mapping bytes are u8 each so no
    /// endianness conversion needed.
    #[test]
    fn dops_box_5_1_payload_is_19_bytes_total_27() {
        let info = opus_info_5_1();
        let dops = build_dops(&info);
        assert_eq!(
            dops.len(),
            27,
            "5.1 dOps box = 8 header + 19 payload = 27 bytes; got {}",
            dops.len()
        );
        let size = u32::from_be_bytes([dops[0], dops[1], dops[2], dops[3]]) as usize;
        assert_eq!(size, dops.len());
        assert_eq!(&dops[4..8], b"dOps");
        // Body
        assert_eq!(dops[8], 0, "Version");
        assert_eq!(dops[9], 6, "OutputChannelCount = 6 for 5.1");
        let pre_skip = u16::from_be_bytes([dops[10], dops[11]]);
        assert_eq!(pre_skip, 312);
        let isr = u32::from_be_bytes([dops[12], dops[13], dops[14], dops[15]]);
        assert_eq!(isr, 48_000);
        assert_eq!(i16::from_be_bytes([dops[16], dops[17]]), 0);
        assert_eq!(dops[18], 1, "ChannelMappingFamily = 1 for surround");
        assert_eq!(dops[19], 4, "StreamCount = 4 for 5.1");
        assert_eq!(dops[20], 2, "CoupledCount = 2 for 5.1");
        assert_eq!(
            &dops[21..27],
            &[0u8, 4, 1, 2, 3, 5][..],
            "ChannelMapping for 5.1"
        );
    }

    /// 7.1 layout: streams=5, coupled=3, mapping = [0, 6, 1, 2, 3, 4, 5, 7].
    /// dOps box = 8 header + 11 preamble + 2 stream/coupled + 8 mapping = 29 bytes.
    #[test]
    fn dops_box_7_1_payload_is_21_bytes_total_29() {
        let cp = opus_head_surround(8, 312, 48_000, 5, 3, &[0, 6, 1, 2, 3, 4, 5, 7]);
        let info = AudioInfo {
            codec: "opus".into(),
            sample_rate: 48_000,
            channels: 8,
            timescale: 48_000,
            asc_bytes: Vec::new(),
            codec_private: cp,
        };
        let dops = build_dops(&info);
        assert_eq!(dops.len(), 29);
        assert_eq!(dops[18], 1, "Family = 1");
        assert_eq!(dops[19], 5, "StreamCount = 5 for 7.1");
        assert_eq!(dops[20], 3, "CoupledCount = 3 for 7.1");
        assert_eq!(&dops[21..29], &[0u8, 6, 1, 2, 3, 4, 5, 7][..]);
    }

    /// Hex-dump the 5.1 dOps box for the deliverables report.
    #[test]
    fn dops_box_5_1_hex_dump() {
        let info = opus_info_5_1();
        let dops = build_dops(&info);
        let hex: String = dops.iter().map(|b| format!("{b:02x} ")).collect();
        println!("5.1 dOps box hex (27 bytes total): {}", hex.trim_end());
    }

    /// `Opus` sample entry containing a family-1 dOps for 5.1. Total
    /// size = 36 (sample-entry preamble) + 27 (5.1 dOps) = 63 bytes.
    #[test]
    fn opus_sample_entry_5_1_size_and_dops_nesting() {
        let info = opus_info_5_1();
        let entry = build_opus_sample_entry(&info);
        assert_eq!(
            entry.len(),
            36 + 27,
            "Opus sample entry for 5.1 = 36 + 27 = 63 bytes; got {}",
            entry.len()
        );
        // Sample-entry channel_count field is at offset 24 inside the
        // sample entry (after 8-byte box header + 6 reserved + 2 dri +
        // 8 reserved = 24).
        let entry_channels = u16::from_be_bytes([entry[24], entry[25]]);
        assert_eq!(
            entry_channels, 6,
            "channel_count in AudioSampleEntry must reflect 5.1"
        );
        // The dOps child should appear after the 36-byte preamble.
        assert!(entry[36..].windows(4).any(|w| w == b"dOps"));
        // Family byte inside the dOps child = entry[36 + 8 + 10] = entry[54].
        // (8-byte dOps box header + 11-byte preamble offset 10 = family).
        assert_eq!(
            entry[36 + 8 + 10],
            1,
            "dOps inside Opus sample entry must carry family=1 for 5.1"
        );
    }

    /// `with_audio()` family=1 validation: stream count + coupled +
    /// mapping must all be sane. Each negative case below is rejected
    /// loudly with a clear error message.
    #[test]
    fn with_audio_rejects_family_1_with_truncated_codec_private() {
        let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
        let mut info = opus_info_5_1();
        // Truncate so the channel-mapping table is missing.
        info.codec_private.truncate(13); // header + 2 stream/coupled, no mapping
        let err = match muxer.with_audio(info) {
            Ok(_) => panic!("truncated family=1 codec_private must reject"),
            Err(e) => e,
        };
        let msg = format!("{}", err);
        assert!(
            msg.contains("≥") && msg.contains("preamble"),
            "error message must explain the size requirement; got: {msg}"
        );
    }

    #[test]
    fn with_audio_rejects_family_1_with_zero_streams() {
        let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
        let mut info = opus_info_5_1();
        // Zero out StreamCount byte (offset 11).
        info.codec_private[11] = 0;
        let r = muxer.with_audio(info);
        assert!(r.is_err(), "StreamCount = 0 must reject");
    }

    #[test]
    fn with_audio_rejects_family_1_with_coupled_exceeding_streams() {
        let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
        let mut info = opus_info_5_1();
        // Make CoupledCount > StreamCount (offset 12 vs 11).
        info.codec_private[11] = 2;
        info.codec_private[12] = 5;
        let r = muxer.with_audio(info);
        assert!(r.is_err(), "CoupledCount > StreamCount must reject");
    }

    #[test]
    fn with_audio_rejects_family_1_with_mapping_index_out_of_range() {
        let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
        let mut info = opus_info_5_1();
        // Streams=4, coupled=2 → max valid mapping index = 5. Set first
        // mapping byte to 99 to force the out-of-range branch.
        info.codec_private[13] = 99;
        let r = muxer.with_audio(info);
        assert!(r.is_err(), "ChannelMapping out-of-range must reject");
    }

    #[test]
    fn with_audio_rejects_family_0_with_5_1_channels() {
        let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
        // Build a hand-crafted family-0 head but claim 6 channels.
        // Family 0 only supports 1..=2 channels per RFC 7845 §5.1.1.
        let mut head = Vec::with_capacity(11);
        head.push(1u8);
        head.push(6u8);
        head.extend_from_slice(&312u16.to_le_bytes());
        head.extend_from_slice(&48_000u32.to_le_bytes());
        head.extend_from_slice(&0i16.to_le_bytes());
        head.push(0u8); // family=0
        let info = AudioInfo {
            codec: "opus".into(),
            sample_rate: 48_000,
            channels: 6,
            timescale: 48_000,
            asc_bytes: Vec::new(),
            codec_private: head,
        };
        let r = muxer.with_audio(info);
        assert!(r.is_err(), "family=0 + 6 channels must reject");
    }

    #[test]
    fn with_audio_accepts_5_1_opus() {
        let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
        let info = opus_info_5_1();
        muxer
            .with_audio(info)
            .expect("5.1 Opus with valid family=1 trailer must accept");
    }

    #[test]
    fn with_audio_rejects_9_channel_opus() {
        let mut muxer = Av1Mp4Muxer::new(640, 480, 30.0).unwrap();
        // 9 channels has no defined family-1 layout.
        let mut head = Vec::with_capacity(11 + 2 + 9);
        head.push(1u8);
        head.push(9u8);
        head.extend_from_slice(&312u16.to_le_bytes());
        head.extend_from_slice(&48_000u32.to_le_bytes());
        head.extend_from_slice(&0i16.to_le_bytes());
        head.push(1u8); // family=1
        head.push(5);
        head.push(3);
        head.extend_from_slice(&[0u8, 1, 2, 3, 4, 5, 6, 7, 0]);
        let info = AudioInfo {
            codec: "opus".into(),
            sample_rate: 48_000,
            channels: 9,
            timescale: 48_000,
            asc_bytes: Vec::new(),
            codec_private: head,
        };
        let r = muxer.with_audio(info);
        assert!(
            r.is_err(),
            "9-channel Opus must reject (no family-1 layout above 8)"
        );
    }

    // ---- Squad-25: Apple `chan` (Channel Layout) box -----------------------

    /// Mono / stereo: no `chan` box (Apple's default layouts are correct).
    #[test]
    fn chan_box_omitted_for_mono_and_stereo() {
        assert!(build_chan_box(1).is_none(), "mono should not emit chan");
        assert!(build_chan_box(2).is_none(), "stereo should not emit chan");
    }

    /// Unsupported channel counts return None — defence-in-depth (the
    /// caller's `with_audio` gate already rejects them, so seeing 8/Atmos
    /// here means a code path bypassed that gate).
    #[test]
    fn chan_box_omitted_for_unsupported_counts() {
        for &c in &[0u16, 3, 4, 5, 8, 9, 16] {
            assert!(
                build_chan_box(c).is_none(),
                "channels={c} must not emit chan"
            );
        }
    }

    /// 5.1 → kAudioChannelLayoutTag_MPEG_5_1_C = (114 << 16) | 6 = 0x00720006.
    /// Body layout: tag u32 (4) | bitmap u32 (4) | num_descriptions u32 (4)
    /// = 12 bytes. Total box = 8-byte header + 12-byte body = 20 bytes.
    #[test]
    fn chan_box_5_1_layout_and_size() {
        let chan = build_chan_box(6).expect("5.1 must emit chan");
        assert_eq!(
            chan.len(),
            20,
            "5.1 chan box must be 20 bytes (8 header + 12 body)"
        );
        let size = u32::from_be_bytes([chan[0], chan[1], chan[2], chan[3]]);
        assert_eq!(
            size as usize,
            chan.len(),
            "size field must equal box length"
        );
        assert_eq!(&chan[4..8], b"chan", "fourcc must be 'chan'");
        let tag = u32::from_be_bytes([chan[8], chan[9], chan[10], chan[11]]);
        assert_eq!(
            tag, 0x00720006u32,
            "5.1 tag must be kAudioChannelLayoutTag_MPEG_5_1_C = 0x00720006; got 0x{tag:08X}"
        );
        let bitmap = u32::from_be_bytes([chan[12], chan[13], chan[14], chan[15]]);
        assert_eq!(bitmap, 0, "mChannelBitmap must be 0 for tag form");
        let ndescs = u32::from_be_bytes([chan[16], chan[17], chan[18], chan[19]]);
        assert_eq!(
            ndescs, 0,
            "mNumberChannelDescriptions must be 0 for tag form"
        );
    }

    /// 7.1 → kAudioChannelLayoutTag_MPEG_7_1_C = (127 << 16) | 8 = 0x007F0008.
    #[test]
    fn chan_box_7_1_layout_and_size() {
        let chan = build_chan_box(7).expect("7.1 must emit chan");
        assert_eq!(chan.len(), 20);
        let tag = u32::from_be_bytes([chan[8], chan[9], chan[10], chan[11]]);
        assert_eq!(
            tag, 0x007F0008u32,
            "7.1 tag must be kAudioChannelLayoutTag_MPEG_7_1_C = 0x007F0008; got 0x{tag:08X}"
        );
    }

    /// `chan` nests inside the `mp4a` AudioSampleEntry (alongside `esds`)
    /// per QuickTime File Format Spec. Multichannel mp4a should contain
    /// both an esds AND a chan child.
    #[test]
    fn chan_nests_inside_mp4a_for_5_1() {
        // 5.1 ASC: AOT=2 SFI=3 chan=6 → 0x11 0xB0.
        let info = AudioInfo {
            codec: "aac".into(),
            sample_rate: 48_000,
            channels: 6,
            timescale: 48_000,
            asc_bytes: vec![0x11, 0xB0],
            codec_private: Vec::new(),
        };
        let mp4a = build_mp4a(&info);
        assert_eq!(&mp4a[4..8], b"mp4a", "outer box must be mp4a");
        let chan_pos = mp4a
            .windows(4)
            .position(|w| w == b"chan")
            .expect("multichannel mp4a must contain chan child");
        let esds_pos = mp4a
            .windows(4)
            .position(|w| w == b"esds")
            .expect("mp4a must always contain esds child");
        // chan should come AFTER esds (we append chan last in build_mp4a).
        assert!(
            chan_pos > esds_pos,
            "chan should come after esds in mp4a (esds @ {}, chan @ {})",
            esds_pos,
            chan_pos
        );
    }

    /// Stereo mp4a must NOT carry a `chan` box — Apple's default L+R
    /// stereo layout is correct without one, and emitting a stereo `chan`
    /// would just bloat the output.
    #[test]
    fn chan_absent_from_stereo_mp4a() {
        let info = AudioInfo {
            codec: "aac".into(),
            sample_rate: 48_000,
            channels: 2,
            timescale: 48_000,
            asc_bytes: vec![0x11, 0x90],
            codec_private: Vec::new(),
        };
        let mp4a = build_mp4a(&info);
        assert!(
            mp4a.windows(4).all(|w| w != b"chan"),
            "stereo mp4a must not contain a chan box"
        );
    }

    // ---- Squad-26: AC-3 + E-AC-3 mux box layout (ETSI TS 102 366 §F) ----

    use crate::ac3_sync::{Ac3SyncInfo, Eac3SyncInfo};

    /// Canonical 5.1 384 kbps 48 kHz AC-3:
    ///   fscod=0, bsid=8, bsmod=0, acmod=7 (3/2), lfeon=1, bit_rate_code=14.
    fn ac3_sync_5_1_384k_48k() -> Ac3SyncInfo {
        Ac3SyncInfo {
            fscod: 0,
            bit_rate_code: 14,
            bsid: 8,
            bsmod: 0,
            acmod: 7,
            lfeon: true,
        }
    }

    fn ac3_info_5_1_384k() -> AudioInfo {
        let body = dac3_body_from_sync(&ac3_sync_5_1_384k_48k());
        AudioInfo::ac3(48_000, 6, body.to_vec())
    }

    /// Vanilla 5.1 E-AC-3 single independent substream, 48 kHz, 384 kbps.
    fn eac3_sync_5_1_48k() -> Eac3SyncInfo {
        Eac3SyncInfo {
            strmtyp: 0,
            substreamid: 0,
            // frmsiz arbitrary for box-layout tests; choose 191 → frame
            // size = 384 bytes which corresponds to 384 kbps @ 48 kHz / 1536
            // samples-per-frame.
            frmsiz: 191,
            fscod: 0,
            fscod2: 0,
            numblkscod: 3,
            acmod: 7,
            lfeon: true,
            bsid: 16,
            dialnorm: 0,
            bsmod: 0,
        }
    }

    fn eac3_info_5_1_384k() -> AudioInfo {
        // 384 kbps → data_rate field = 192 (the "kbps / 2" encoding).
        let body = dec3_body_from_sync(&eac3_sync_5_1_48k(), 192);
        AudioInfo::eac3(48_000, 6, body.to_vec())
    }

    /// `dac3` is exactly 11 bytes total (8-byte box header + 3-byte body).
    /// Body field positions per ETSI TS 102 366 §F.4: fscod 2b | bsid 5b |
    /// bsmod 3b | acmod 3b | lfeon 1b | bit_rate_code 5b | reserved 5b.
    #[test]
    fn dac3_box_3_byte_payload_layout() {
        let info = ac3_info_5_1_384k();
        let dac3 = build_dac3(&info);
        assert_eq!(dac3.len(), 11, "dac3 = 8-byte header + 3-byte body");
        let size = u32::from_be_bytes([dac3[0], dac3[1], dac3[2], dac3[3]]) as usize;
        assert_eq!(size, dac3.len(), "size field equals box length");
        assert_eq!(&dac3[4..8], b"dac3", "box type 'dac3'");
        // Body bit-extract (24 bits, MSB-first across 3 bytes 8..11).
        let raw = ((dac3[8] as u32) << 16) | ((dac3[9] as u32) << 8) | dac3[10] as u32;
        assert_eq!((raw >> 22) & 0x03, 0, "fscod = 0 (48 kHz)");
        assert_eq!((raw >> 17) & 0x1F, 8, "bsid = 8 (AC-3)");
        assert_eq!((raw >> 14) & 0x07, 0, "bsmod = 0");
        assert_eq!((raw >> 11) & 0x07, 7, "acmod = 7 (3/2 = 5.1 with LFE)");
        assert_eq!((raw >> 10) & 0x01, 1, "lfeon = 1");
        assert_eq!((raw >> 5) & 0x1F, 14, "bit_rate_code = 14 (= 384 kbps)");
        assert_eq!(raw & 0x1F, 0, "reserved 5 bits = 0");
    }

    /// `ac-3` AudioSampleEntry per ETSI TS 102 366 §F.2.
    /// Total = 36-byte sample-entry preamble + 11-byte dac3 = 47 bytes.
    /// 4cc is `ac-3` exactly (with the hyphen at byte index 6 = 0x2D).
    #[test]
    fn ac3_sample_entry_size_and_fourcc() {
        let info = ac3_info_5_1_384k();
        let entry = build_ac3_sample_entry(&info);
        let size = u32::from_be_bytes([entry[0], entry[1], entry[2], entry[3]]) as usize;
        assert_eq!(size, entry.len(), "size field equals box length");
        assert_eq!(&entry[4..8], b"ac-3", "4cc MUST be 'ac-3' (with hyphen)");
        // Reject the dehyphenated form
        assert_ne!(
            &entry[4..8],
            b"ac3\0",
            "4cc 'ac3' (3-char) is non-conformant"
        );
        assert_eq!(
            entry.len(),
            47,
            "ac-3 sample entry = 36 (preamble) + 11 (dac3)"
        );
        // dac3 must nest inside.
        let dac3_pos = entry
            .windows(4)
            .position(|w| w == b"dac3")
            .expect("dac3 child missing");
        assert!(
            dac3_pos > 28,
            "dac3 must come after AudioSampleEntry preamble"
        );
        // samplerate field at box-relative offset 8 + 24 = 32.
        let sr_q16 = u32::from_be_bytes([entry[32], entry[33], entry[34], entry[35]]);
        assert_eq!(sr_q16, 48_000u32 << 16, "samplerate = 48000 << 16 (Q16)");
    }

    /// `dec3` for a single independent substream (Squad-26's scope) is
    /// 13 bytes total = 8-byte box header + 5-byte body (no dependent
    /// substreams = no chan_loc tail). Body layout per ETSI TS 102 366
    /// §F.6.
    #[test]
    fn dec3_box_5_byte_payload_layout() {
        let info = eac3_info_5_1_384k();
        let dec3 = build_dec3(&info);
        assert_eq!(dec3.len(), 13, "dec3 = 8-byte header + 5-byte body");
        let size = u32::from_be_bytes([dec3[0], dec3[1], dec3[2], dec3[3]]) as usize;
        assert_eq!(size, dec3.len(), "size field equals box length");
        assert_eq!(&dec3[4..8], b"dec3", "box type 'dec3'");
        // Body header: data_rate(13) + num_ind_sub-1(3) packed in bytes 8..10.
        let header = ((dec3[8] as u16) << 8) | dec3[9] as u16;
        let data_rate = (header >> 3) & 0x1FFF;
        assert_eq!(data_rate, 192, "data_rate = 192 (= 384 kbps / 2)");
        let num_ind_sub_minus_1 = header & 0x07;
        assert_eq!(num_ind_sub_minus_1, 0, "single substream → field = 0");
        // Per-independent-substream block: bits 16..40 (3 bytes 10..13).
        // Layout shifts within the 24-bit window:
        //   bit 23..22 fscod
        //   bit 21..17 bsid (=16)
        //   bit 16     reserved
        //   bit 15     asvc
        //   bit 14..12 bsmod
        //   bit 11..9  acmod
        //   bit 8      lfeon
        //   bit 7..5   reserved
        //   bit 4..1   num_dep_sub (=0)
        //   bit 0      reserved
        let sub = ((dec3[10] as u32) << 16) | ((dec3[11] as u32) << 8) | dec3[12] as u32;
        assert_eq!((sub >> 22) & 0x03, 0, "fscod = 0 (48 kHz)");
        assert_eq!((sub >> 17) & 0x1F, 16, "bsid = 16 (E-AC-3 marker)");
        assert_eq!((sub >> 12) & 0x07, 0, "bsmod = 0");
        assert_eq!((sub >> 9) & 0x07, 7, "acmod = 7 (3/2 = 5.1 with LFE)");
        assert_eq!((sub >> 8) & 0x01, 1, "lfeon = 1");
        assert_eq!((sub >> 1) & 0x0F, 0, "num_dep_sub = 0 (single substream)");
    }

    /// `ec-3` AudioSampleEntry per ETSI TS 102 366 §F.5.
    /// Total = 36-byte sample-entry preamble + 13-byte dec3 = 49 bytes.
    #[test]
    fn ec3_sample_entry_size_and_fourcc() {
        let info = eac3_info_5_1_384k();
        let entry = build_ec3_sample_entry(&info);
        let size = u32::from_be_bytes([entry[0], entry[1], entry[2], entry[3]]) as usize;
        assert_eq!(size, entry.len(), "size field equals box length");
        assert_eq!(&entry[4..8], b"ec-3", "4cc MUST be 'ec-3' (with hyphen)");
        assert_eq!(
            entry.len(),
            49,
            "ec-3 sample entry = 36 (preamble) + 13 (dec3)"
        );
        let dec3_pos = entry
            .windows(4)
            .position(|w| w == b"dec3")
            .expect("dec3 child missing");
        assert!(
            dec3_pos > 28,
            "dec3 must come after AudioSampleEntry preamble"
        );
    }

    /// stsd dispatcher: ac3 info → ac-3 entry; eac3 info → ec-3 entry.
    /// Must NOT cross-pollinate with mp4a / Opus.
    #[test]
    fn stsd_dispatcher_routes_ac3_eac3() {
        let stsd_ac3 = build_audio_stsd(&ac3_info_5_1_384k());
        assert!(
            stsd_ac3.windows(4).any(|w| w == b"ac-3"),
            "AC-3 stsd has 'ac-3'"
        );
        assert!(
            stsd_ac3.windows(4).any(|w| w == b"dac3"),
            "AC-3 stsd has 'dac3'"
        );
        assert!(
            !stsd_ac3.windows(4).any(|w| w == b"mp4a"),
            "AC-3 stsd MUST NOT have mp4a"
        );
        assert!(
            !stsd_ac3.windows(4).any(|w| w == b"Opus"),
            "AC-3 stsd MUST NOT have Opus"
        );
        assert!(
            !stsd_ac3.windows(4).any(|w| w == b"esds"),
            "AC-3 stsd MUST NOT have esds"
        );

        let stsd_eac3 = build_audio_stsd(&eac3_info_5_1_384k());
        assert!(
            stsd_eac3.windows(4).any(|w| w == b"ec-3"),
            "E-AC-3 stsd has 'ec-3'"
        );
        assert!(
            stsd_eac3.windows(4).any(|w| w == b"dec3"),
            "E-AC-3 stsd has 'dec3'"
        );
        assert!(
            !stsd_eac3.windows(4).any(|w| w == b"mp4a"),
            "E-AC-3 stsd MUST NOT have mp4a"
        );
        assert!(
            !stsd_eac3.windows(4).any(|w| w == b"esds"),
            "E-AC-3 stsd MUST NOT have esds"
        );
        assert!(
            !stsd_eac3.windows(4).any(|w| w == b"dac3"),
            "E-AC-3 stsd MUST NOT have dac3"
        );
    }

    /// `with_audio` must accept a 5.1 AC-3 info and reject obvious shape
    /// errors (wrong dac3 body length, wrong sample rate).
    #[test]
    fn with_audio_accepts_ac3_5_1_and_rejects_bad_shape() {
        let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
        muxer
            .with_audio(ac3_info_5_1_384k())
            .expect("5.1 AC-3 must be accepted");

        // Wrong body length
        let mut muxer2 = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
        let mut bad = ac3_info_5_1_384k();
        bad.codec_private = vec![0u8; 2];
        let err = muxer2
            .with_audio(bad)
            .err()
            .expect("must reject 2-byte dac3");
        assert!(format!("{err:#}").contains("3 bytes"));

        // Wrong sample rate
        let mut muxer3 = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
        let bad_sr = AudioInfo {
            sample_rate: 22_050,
            timescale: 22_050,
            ..ac3_info_5_1_384k()
        };
        let err = muxer3
            .with_audio(bad_sr)
            .err()
            .expect("must reject 22050 for AC-3");
        assert!(format!("{err:#}").contains("32000"));
    }

    /// `with_audio` must accept a single-substream E-AC-3 info and reject
    /// an under-sized dec3 body.
    #[test]
    fn with_audio_accepts_eac3_5_1_and_rejects_short_dec3() {
        let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
        muxer
            .with_audio(eac3_info_5_1_384k())
            .expect("5.1 E-AC-3 must be accepted");

        let mut muxer2 = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
        let mut bad = eac3_info_5_1_384k();
        bad.codec_private = vec![0u8; 4];
        let err = muxer2
            .with_audio(bad)
            .err()
            .expect("must reject short dec3");
        assert!(format!("{err:#}").contains("≥5"));
    }

    /// AC-3 / E-AC-3 channel count gate: must reject >6.
    #[test]
    fn with_audio_rejects_ac3_more_than_6_channels() {
        let mut muxer = Av1Mp4Muxer::new(320, 240, 30.0).unwrap();
        let bad = AudioInfo {
            channels: 8,
            ..ac3_info_5_1_384k()
        };
        let err = muxer.with_audio(bad).err().expect("must reject 8 channels");
        assert!(format!("{err:#}").contains("1..=6"));
    }

    /// Round-trip: parse a synthetic 5.1 AC-3 sync header → derive dac3
    /// body → pack into an `ac-3` sample entry → walk the bytes back out
    /// and recover fscod / acmod / lfeon / bit_rate_code unchanged.
    #[test]
    fn ac3_sync_to_dac3_to_sample_entry_roundtrip() {
        let sync = ac3_sync_5_1_384k_48k();
        let body = dac3_body_from_sync(&sync);
        let info = AudioInfo::ac3(48_000, 6, body.to_vec());
        let entry = build_ac3_sample_entry(&info);
        // Find dac3 box body (8-byte box header inside the entry then 3
        // body bytes).
        let dac3_pos = entry.windows(4).position(|w| w == b"dac3").unwrap();
        let dac3_body_start = dac3_pos + 4;
        let raw = ((entry[dac3_body_start] as u32) << 16)
            | ((entry[dac3_body_start + 1] as u32) << 8)
            | entry[dac3_body_start + 2] as u32;
        assert_eq!((raw >> 22) & 0x03, sync.fscod as u32);
        assert_eq!((raw >> 17) & 0x1F, sync.bsid as u32);
        assert_eq!((raw >> 11) & 0x07, sync.acmod as u32);
        assert_eq!((raw >> 10) & 0x01, sync.lfeon as u32);
        assert_eq!((raw >> 5) & 0x1F, sync.bit_rate_code as u32);
    }
}
