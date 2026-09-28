# Configuring a transcode — the complete `OutputSpec` guide

Everything you can configure for a rivet job lives on one struct,
[`OutputSpec`](../crates/rivet/src/spec.rs). You **build** it (constructor +
chained `with_*` setters), optionally **validate** it, then **run** it. This page
documents every knob; for the internals see [pipeline & architecture](pipeline.md),
and for the CLI equivalents see the [CLI reference](cli.md).

```rust
use rivet::{OutputSpec, Rung, Quality, AudioCodecPolicy, EncodePolicy,
            ChunkSeamMode, PerceptualTarget, run_job_blocking, fn_sink};
use rivet::progress::RungProgress;
use std::sync::Arc;

// A 3-rung single-file ladder, fully specified.
let spec = OutputSpec::single_file(vec![
    Rung::new(1920, 1080).with_quality(Quality::crf(28)),
    Rung::new(1280, 720).with_quality(Quality::target(PerceptualTarget::Standard)),
    Rung::new(854, 480),                              // default quality
])
.with_audio(AudioCodecPolicy::Auto)                        // passthrough / transcode / drop
.with_max_frame_rate(30.0)                            // cap output cadence
.web_sdr()                                            // color preset: BT.709 8-bit SDR
.encode_policy(EncodePolicy::AllGpus)                 // chunk-encode across every GPU
.chunk_seam_mode(ChunkSeamMode::ParallelConstQp);     // keep seams quality-flat

spec.validate()?;                                     // fail fast on incoherent specs

let bytes = std::fs::read("input.mkv")?;
let sink = Arc::new(fn_sink(|p: RungProgress| {
    println!("{:<6} {:?} {:>5.1}%  {} frames", p.label, p.status, p.percent, p.frames_done);
}));
let out = run_job_blocking(&bytes, &spec, Some("out_dir".as_ref()), sink)?;
```

> **Just want one file in, one file out?** Skip the spec entirely:
> `rivet::transcode_file("in.mkv", "out.mp4")?` uses sensible defaults
> (source-resolution single rung, AAC/Opus passthrough, 8-bit SDR, all GPUs).

---

## 1. Construct — the output shape

| Constructor | Output |
|-------------|--------|
| `OutputSpec::single_file(rungs)` | One self-contained faststart **MP4** per rung (video + audio; AV1 by default — set `with_video_codec` for H.264/H.265). |
| `OutputSpec::hls(rungs, segment_seconds)` | A segmented **CMAF/HLS** package: `master.m3u8` + an audio rendition group + `video/<h>p/{init.mp4, seg-*.m4s, playlist.m3u8}` per rung, segment-aligned for clean ABR. |
| `OutputSpec::audio_only()` | The **audio alone** as one bare `.mp3` (`OutputMode::AudioOnly`, `Container::Mp3`, `Muxer::Mp3File`): no rungs, no video decoded. A `single_file` job whose input has no video becomes this by itself. See [§3](#3-audio--with_audioaudiocodecpolicy). |

`rungs` is a `Vec<Rung>` (next section). `segment_seconds` is the HLS target
segment length (segments still break on keyframes). The constructor wires the
matching `Container` + `Muxer` + `OutputMode` for you.

---

## 2. The ladder — rungs & quality

A [`Rung`](../crates/rivet/src/spec.rs) is one rendition: a target size + a
per-rung [`Quality`](../crates/rivet/src/spec.rs).

```rust
Rung::new(1280, 720)                       // auto label "720p", default quality
    .with_quality(Quality::crf(28))        // or .with_quality(Quality::target(..))
    .with_label("hd")                      // override the auto label
```

| `Rung` method | Effect |
|---------------|--------|
| `Rung::new(width, height)` | A rung at `width × height`; label auto-set to `"<short-side>p"`, default quality. |
| `.with_quality(Quality)` | Set the per-rung encoder quality. |
| `.with_label(impl Into<String>)` | Override the auto label. |
| `.short_side()` | The "p" number (`min(width, height)`). |

Public fields: `width`, `height`, `label`, `quality`.

### Quality

```rust
Quality::crf(28)                           // constant rate factor (lower = better)
Quality::target(PerceptualTarget::High)    // perceptual target instead of a CRF
```

| `Quality` field | Type | Meaning |
|-----------------|------|---------|
| `crf` | `Option<u8>` | Constant rate factor, encoder-native (rav1e/NVENC 0..=255). `None` → derive from `target`. |
| `speed_preset` | `Option<u8>` | Encoder-native speed preset. `None` → derive from `tier`. |
| `target` | `QualityTarget` | Perceptual target (used when `crf` is `None`). |
| `tier` | `SpeedTier` | Speed/efficiency tier (used when `speed_preset` is `None`). |
| `keyframe_interval` | `Option<u32>` | GOP length in frames. `None` → `2 × fps` (a 2-second GOP). |
| `overrides` | `EncodeOverrides` | Backend-agnostic per-rung knobs layered on the target/tier — a quality shift in libaom-CQ steps, tiles, reference frames, lookahead, B-frames, a bitrate and its buffer. Inert by default. |

`Quality::crf` / `Quality::target` are the two constructors; set the rest with
struct-update syntax, e.g. `Quality { tier: Speed::Archive, keyframe_interval:
Some(120), ..Quality::crf(30) }`, or `.with_overrides(EncodeOverrides { .. })`.

**VMAF as the target.** `QualityTarget::Vmaf(n)` aims every rung at a VMAF
score; `codec::encode::tuning` turns it into each backend's quantiser through
calibrated anchor tables, so `Vmaf(93)` means the same perceived quality on
NVENC, QSV, AMF and rav1e (`docs/av1-tuning-research.md` has the tables and
`docs/av1-tuning-methodology.md` how to re-calibrate for a new encoder). On
every surface it is the word `vmaf=93` — `--target`, `target=` on the socket,
`target` in the API/manifest, `target=vmaf=93` in the policy grammar. Whether a
target *delivers* its VMAF is measured, not assumed: [`bench/`](../bench/README.md)
scores a ladder against its source with libvmaf.

**GOP.** `OutputSpec::gop` (`with_gop`, CLI `--gop`, key `gop`) sets the
keyframe cadence for every rung — and, on the multi-GPU single-file path, the
chunk grid, since a chunk is a whole number of GOPs. For HLS the segment grid is
`segment_seconds`; a GOP shorter than the segment adds keyframes inside it, a
longer one is silently the segment. A rung's own `Quality::keyframe_interval`
wins over the spec-wide value.

**Bitrate rungs.** A rung can be coded to a rate instead of a quality:
`EncodeOverrides::bitrate` (bits per second) and `buffer_ms` (the coded
picture buffer, in milliseconds of that rate; one second unless named, `0`
declares none). On the
surfaces these are `--rung 1280x720@3M` (that rung), `--video-bitrate 3M`
(every rung without its own), `--video-buffer 1s`, and the policy grammar's
`bitrate=` / `buffer=` for a derived ladder. The precedence is the rung's own
`@RATE`, then the policy, then `--video-bitrate`. The native software H.264 /
H.265 encoder is the one that codes to a rate: `validate` refuses a rate
beside a CRF, under `--seam-mode constqp`, or on AV1, and a buffer without a
rate. The job refuses a bitrate rung whose encode pool is GPUs, before a
frame is decoded. A rung without a rate is the quality-target encode it
always was. What a buffer buys (the HLS `BANDWIDTH` it bounds) and what the
rate costs are measured in [codec-encode.md](codec-encode.md#bitrate-rungs-in-the-software-tier-measured).

### Per-rung policy — `with_rung_policy(RungPolicy)`

A ladder wants different knobs at different positions: softer going down (the
same quantizer at a quarter of the resolution is a far finer quantizer in
terms of what an eye can resolve), one tile below 4K, more reference frames.
Rather than hand-setting `overrides` on every rung, give the spec a
[`RungPolicy`](../crates/codec/src/encode/tuning/overrides.rs) and the engine
resolves it against each rung's position before encoding, layering the rung's
own `overrides` on top (the rung-specific knob wins; quality deltas
accumulate):

```rust
use codec::encode::tuning::RungPolicy;

// The measured recommendation: +2 steps softer per rung going down, no top
// bonus, one tile below 4K, three reference frames.
let spec = OutputSpec::hls(rungs, 4.0).with_rung_policy(RungPolicy::recommended());

// Or the text grammar (what `--encode-policy` and the settings key take):
let policy: RungPolicy = "qstep=2;top:q=-2;short<=2159:tiles=1x1;any:refs=3".parse()?;
```

The grammar: rules separated by `;`, each `selector:key=value,...`, later
wins; selectors `any`/`top`/`below_top`/`step=N`/`short<=N`/`short>=N`; keys
`q`, `tiles` (`CxR`), `gop`, `lookahead`, `bframes`, `refs`, `multipass`,
`grain`, `speed`, `target` (`vmaf=N` allowed), `aq`, `wp`, `cu_depth`,
`bitrate` (`3M`, `800k`), `buffer` (`1s`, `500ms`, `0`); `qstep=N` alone is the
compounding per-rung step. An empty policy — the default — changes nothing.
`bframes=N` is a non-pyramid run of N B pictures between anchors on NVENC and
the software H.264/H.265 tier (QSV maps it to `GopRefDist` but has not been
verified with it); the muxers carry the reorder as `ctts` / `trun` composition
offsets, so single-file, chunked and HLS output all take it.
[`LadderPolicy`](../crates/codec/src/encode/tuning/policy_grammar.rs) is the
recommendation as numbers, for tuning one of them.

- **`QualityTarget`** (re-exported as `PerceptualTarget`): `VisuallyLossless`,
  `High`, `Standard`, `Low`, `Vmaf(u8)` (target a specific VMAF score).
- **`SpeedTier`** (re-exported as `Speed`): `Draft` (fastest), `Standard`,
  `Archive` (slowest/most efficient).

### Auto ladder

Don't want to hand-write rungs? Derive a standard ABR ladder from the source:

```rust
let rungs = rivet::standard_ladder(source_w, source_h, /* max_short_side */ 1080);
let spec = OutputSpec::single_file(rungs);
```

It snaps to standard short sides (2160/1440/1080/720/480/360/240), preserves
aspect ratio, even-aligns dims, and caps the top rung.

---

## 3. Audio — `with_audio(AudioCodecPolicy)`

| `AudioCodecPolicy` | Behavior |
|---------------|----------|
| `Auto` *(default)* | Passthrough AAC / Opus / AC-3 / E-AC-3 / DTS verbatim, and MP3 into a single-file MP4; transcode the rest (Vorbis, MP2, PCM; MP3 for HLS) → Opus; drop what cannot be decoded. For `audio_only()` it means **MP3**: an MP3 source passes through, the rest is encoded. |
| `ForceOpus` | Always produce Opus (passthrough Opus, transcode everything else). |
| `ForceMp3` | Always produce **MP3** (passthrough MP3, encode everything else — CBR, stereo at most). Single-file MP4 and audio-only; refused for HLS. Encoding needs the `lame` feature (LAME, loaded at run time); `validate()` refuses it in a build without. |
| `ForceAac` | Always produce **AAC-LC** (passthrough AAC, encode everything else with rivet's own encoder — mono to 7.1, constant rate). The audio every browser and device plays, older iOS and Safari included (Opus in MP4 needs iOS / Safari 17). Single-file MP4 and HLS; refused for audio-only output. Needs no feature. |
| `Drop` | Video-only output. |
| `Flac` | Lossless FLAC: copy a FLAC source, encode anything decodable. Plays from MP4 in Chrome, Edge, Firefox and Safari; audio-only output is a native `.flac`. |
| `Alac` | Lossless ALAC: copy an ALAC source, encode anything decodable. Plays on Apple platforms and in Safari only; audio-only output is an `.m4a`. |

```rust
spec.with_audio(AudioCodecPolicy::ForceOpus)
    .with_audio_bitrate(320_000)
    .with_audio_channels(AudioChannels::Surround51)
```

Lossless output takes two more knobs — `with_audio_bit_depth(AudioBitDepth::{Source, Sixteen, TwentyFour})`
and, for FLAC, `with_flac_level(FlacLevel::{Fast, Default, Best})` — and an
audio-only job can be written as a native `.flac` or an `.m4a`:
`OutputSpec::audio_only_in(Container::{Mp3, Flac, M4a})`
(`OutputSpec::audio_only_container(policy)` is the default for a policy).
`spec.file_extension()` names the file a single-file or audio-only job
writes. See [lossless-audio.md](lossless-audio.md) for the rules and what
`validate()` refuses.

A source a forced codec cannot reach (an AAC track: there is no AAC decoder)
is passed through into an MP4 or HLS package with a warning, the handling
saying so — and refused for a bare `.mp3`, which cannot hold it. `ForceAac` on
an AAC source is simply a passthrough.

AAC output is an `mp4a` sample entry whose `esds` carries the
AudioSpecificConfig (object type 2, the channel configuration of ISO/IEC
13818-7 Table 42: 1 mono, 2 stereo, 3 3.0, 4 4.0, 5 5.0, 6 5.1, 7 7.1), with
`codecs` `mp4a.40.2` in the job output and the HLS master, and each HLS audio
rendition's `CHANNELS` its channel count. It is coded at 22.05 / 24 / 32 /
44.1 / 48 kHz: another source rate is resampled to the nearest in its family
(the 11.025 kHz family to 22.05 / 44.1, the rest to 24 / 32 / 48). The
encoder's one frame (1024 samples) of priming is hidden by the MP4 edit list,
so the track presents exactly the source's samples. The encoder is in-tree
and written from the standards ([decisions.md §26](decisions.md#26-the-aac-lc-encoder-is-written-here-from-the-standards));
AAC may be subject to patent licensing in some jurisdictions.

### Bitrate — `with_audio_bitrate(bps)`

For transcoded audio only. Omitted: Opus derives it from the channel layout
(64 kbps per uncoupled stream + 96 kbps per coupled pair — 64k mono, 96k
stereo, 160k 3.0, 256k 5.0, 320k 5.1, 352k 6.1, 416k 7.1); MP3 is 128k stereo,
64k mono. MP3 is constant bitrate on the MPEG-1 Layer III ladder (32k 40k 48k
56k 64k 80k 96k 112k 128k 160k 192k 224k 256k 320k); another rate is refused.
AAC defaults to 64k mono, 128k stereo, 384k 5.1 and 512k 7.1 (64k per main
channel for 3.0 / 4.0 / 5.0), and takes any rate from 8k per main channel up
to the ISO/IEC 13818-7 decoder buffer's ceiling (6144 bits per main channel
per frame: 288k a channel at 48 kHz, 144k at 24 kHz); a rate outside that for
the track's layout and coding rate fails the encode, saying the range.

### Channel layout — `with_audio_channels(AudioChannels)`

| `AudioChannels` | Settings word | Output |
|---|---|---|
| `Source` *(default)* | `source` | The source's layout wherever the codec carries it. Opus carries 1–8 channels; a layout it has no mapping for goes out in the narrowest one with a place for every speaker, the missing ones silent (2.1 → 5.1, 4.0 → 5.0 with the back centre in both surrounds). AAC carries the layouts of its channel configurations — mono, stereo, 3.0, 4.0, 5.0, 5.1, 7.1 — and the others the same way (quad → 5.0, 2.1 → 5.1, 6.1 → 7.1 with the back centre in both backs). MP3 carries two: a wider source is downmixed to stereo. A passthrough keeps its layout (a 5.1 AAC track stays 5.1, `channelConfiguration` 6). |
| `Mono` | `mono` | Downmix to one channel. |
| `Stereo` | `stereo` | Downmix to two. |
| `Surround51` | `5.1` | FL FR FC LFE BL BR (7.1 folds its side pair into the back pair). |
| `Surround71` | `7.1` | FL FR FC LFE BL BR SL SR. |

Downmixes are ITU-R BS.775 — centre and surrounds at −3 dB into the fronts,
the **LFE dropped** — normalised so no output can clip: 5.1 → stereo is
`L = 0.414·FL + 0.293·FC + 0.293·SL`, mono the stereo downmix folded at −3 dB
([`codec::audio::remix`](../crates/codec/src/audio/remix.rs)). **rivet never
upmixes**: asking for more channels than the source has is an error (a stereo
source with `Surround51` fails, rather than coming out as stereo or as six
channels made up from two). Asking for the width the source already has is a
passthrough. Changing the width of an AAC track is an error too, naming the
missing decoder.

The layout comes from the decoder, not the channel count: an AC-3 stream says
which of 4.0, quad(side) and 3.1 its four channels are (`acmod`), DTS the same
(`AMODE`), and a stream that changes layout mid-way is remixed to the output's.

### HLS: stereo fallback — `with_audio_stereo_fallback(true)`

Beside a surround audio rendition, a stereo downmix of it in the same audio
group — `EXT-X-MEDIA` entries with `CHANNELS="2"` (the group's `DEFAULT`) and
`CHANNELS="6"`, distinct `NAME`s, and every variant's `CODECS` listing each
codec the group holds — so a player on stereo hardware takes the stereo one
rather than downmixing itself. Nothing is added when the audio is stereo or
mono; a surround track that cannot be decoded (AAC) goes alone, and the job's
audio handling says why. `validate()` refuses the flag outside HLS.

### MP3 and audio-only output — `OutputSpec::audio_only()`

MP3 in a single-file MP4 is an `mp4a` sample entry with an `esds` of object
type 0x6B (0x69 at the MPEG-2 half rates), and the job reports its `codecs`
value as `mp3` (`JobOutput::audio_codecs`) — the spelling Chromium and Firefox
both accept; `mp4a.6B` is rejected by Firefox and `mp4a.40.34` by both.
Encoding resamples to 32 / 44.1 / 48 kHz where the source is not already one of
them (the 11.025 kHz family to 44.1, the rest to 48), and the encoder delay is
hidden by the MP4 edit list.

`audio_only()` writes the audio alone as a bare `.mp3`: an `Info` frame (frame
and byte counts, a seek table) and the frames. For an encode (and for an MP3
passthrough whose source's LAME tag stated them) the `Info` frame carries LAME's
encoder delay and end padding, so a gapless player — ffmpeg included — decodes
exactly the source's samples. `audio=opus` is refused (an `.mp3` cannot hold
Opus), as is a trim, and a splice. The job's one output is labelled `audio`
(width and height 0); the input is read by `container::streaming::demux_audio`,
which takes a video file's audio track, a bare MP3/MP2, an audio-only MP4 / M4A,
or an audio-only Matroska / WebM.

HLS with `ForceMp3` is refused: rivet's HLS is CMAF (fMP4), for which neither
the CMAF media profiles nor Apple's HLS authoring spec carry MP3.

---

## 4. Color & bit depth

Two orthogonal axes. Most callers use a **preset**; the low-level setters are
there when you need them.

```rust
spec.web_sdr()       // BT.709 8-bit SDR, tonemap any HDR source down (the default)
spec.hdr10()         // BT.2020 + PQ, 10-bit, no tonemap
spec.hlg()           // BT.2020 + HLG, 10-bit, no tonemap
spec.passthrough()   // keep the source's color + bit depth verbatim
```

Under the presets are exactly **two** methods:

| Method | Sets | Values |
|--------|------|--------|
| `with_color(ColorPolicy)` | **gamut + transfer + tonemap decision** | `TonemapToSdr` *(default)* · `Passthrough` · `Hdr10` · `Hlg` |
| `with_bit_depth(BitDepth)` | **bits per sample** | `Auto` *(default — follow the color policy)* · `EightBit` · `TenBit` |

There is intentionally **no** `with_gamut` / `with_transfer` / `with_color_space`
— `ColorPolicy` bundles them because only a few combinations are web-safe:

- **Gamut** = the color *primaries* (which colors are representable): **BT.709**
  (standard SDR) or **BT.2020** (wide, for HDR).
- **Transfer** = the *transfer function* / EOTF (the curve mapping stored values
  ↔ light, i.e. the brightness response): SDR **gamma** (~2.2/2.4), **PQ** (SMPTE
  ST 2084, absolute brightness — HDR10), or **HLG** (ARIB STD-B67, relative —
  broadcast HDR).

| `ColorPolicy` | Gamut | Transfer | Bit depth (with `Auto`) | Tonemap |
|---------------|-------|----------|:-----------------------:|:-------:|
| `TonemapToSdr` | BT.709 | gamma | 8-bit | HDR → SDR |
| `Passthrough`  | source | source | source | no |
| `Hdr10`        | BT.2020 | PQ | 10-bit | no |
| `Hlg`          | BT.2020 | HLG | 10-bit | no |

The on-disk pixel format follows from bit depth: 8-bit → `yuv420p`, 10-bit →
`yuv420p10le` (4:2:0). 10-bit and HDR need a 10-bit encoder **for the output
codec**: AV1 on `nvidia` / `amd` / `qsv` (the software AV1 tier,
`rav1e-fallback`, is 8-bit); H.265 on those or `h26x-fallback` (Main 10);
H.264 on `h26x-fallback` only (High 10 — no hardware backend has a 10-bit
H.264 encoder). `validate()` checks the spec's codec against this build and
refuses by name, saying which feature would serve it (see
[§9](#9-validate--validate)).
HDR is tagged in the container via `colr`/`mdcv`/`clli` atoms and, for
H.264 / H.265, in the SPS VUI and the HDR10 SEIs the encoders write.

The tags describe the picture *after* the policy, never the source: under
`TonemapToSdr` an 8-bit source whose matrix is BT.601 (SMPTE 170M / BT.470BG)
or BT.2020 is re-matrixed to BT.709 by the pump and comes out tagged
`matrix_coefficients` 1; its range, primaries and transfer are not converted
and keep the source's values. A 10-bit SDR source is not re-matrixed and
keeps every tag.

### A source that states no matrix

A source whose container and bitstream state no matrix — none at all, or `2`
("unspecified") — is read by its stored picture size, the rule
mpv's renderer (libplacebo, `pl_color_system_guess_ycbcr`) and DXVA2
(`DXVA2_VideoTransferMatrix_Unknown`) apply:

| Picture | Matrix | Primaries (when none are stated either) |
|---|---|---|
| **Standard definition**: narrower than 1280 **and** at most 576 lines (640x360, 720x480, 720x576, 1279x576) | **BT.601** (H.273 5 at 576 lines, 6 otherwise) | libplacebo's guess (`pl_color_primaries_guess`): BT.601-625 (5) at 576 lines, BT.601-525 (6) at 480 or 486, BT.709 otherwise |
| **High definition**: 1280 wide or more, or taller than 576 (720x577, 1280x576, 1280x720) | **BT.709** | BT.709 |

A standard-definition source read as BT.601 takes the tagged-BT.601 path above:
re-matrixed to BT.709 under `TonemapToSdr` and tagged `matrix_coefficients` 1,
its primaries carried as the tag. The transfer is BT.709's curve either way.
This is how ffmpeg (swscale takes BT.601 for any unstated matrix), mpv, DXVA2 and
Chrome / Firefox (below 720 lines) show such a source, so the output looks like
the source there. **VLC is the renderer that disagrees**: it takes BT.709 at any
size, so it showed the output of an untagged SD source right while the matrix
was left at BT.709, and now shows it off by what the matrix makes of the
picture. How much that is depends on the colour: on testsrc2's saturated bars
(640x360) the BT.601 renderers went from 24.4 dB to 38.5–40.4 dB (RGB PSNR of
the output's render against the source's) and VLC from 43.5 dB to 24.3 dB; on
low-saturation natural footage (foreman at 720x576), where the matrix alone
is 43.3 dB, the BT.601 renderers gain 0.7–0.8 dB (0.7–1.5 at 720x480) and
VLC loses 1.7 dB.

The bitstreams read are H.264 / H.265 (the SPS VUI), AV1 (the sequence
header's `color_config`), VP9 (a keyframe's `color_space`, a matrix only) and
MPEG-2 (the sequence display extension). A standard-definition stream that
states a BT.601 matrix and no primaries — all a VP9 stream can state — takes
the same primaries guess, as mpv takes them (a BT.709 or BT.2020 matrix names
its own primaries; a BT.601 one is guessed by size): left at BT.709, a 720x480
VP9 BT.601 stream came out 18 dB off against libplacebo's render of it. Other
codecs (VP8, ProRes) keep
BT.709: rivet does not read their bitstream colour, so it cannot tell a stream
that states nothing from one that states it only there.

## 5b. Output codec — `with_video_codec(...)`

`VideoCodecPolicy` (the video analogue of [`AudioCodecPolicy`](#3-audio--with_audioaudiocodecpolicy))
is `Av1` (default), `H264`, or `H265`. It resolves to the encoder/muxer's
low-level `VideoCodec` via `VideoCodecPolicy::codec()`.

```rust
use rivet::VideoCodecPolicy;

let spec = OutputSpec::single_file(rungs).with_video_codec(VideoCodecPolicy::H264);
```

**AV1** is the royalty-clean default (AV1 + Opus in MP4 = zero royalty exposure);
**H.264 / H.265** are for legacy-player compatibility and carry the
patent-licensing obligations AV1 was chosen to avoid. All three work for
single-file MP4 **and** CMAF/HLS — the muxer emits `av01`/`avc1`/`avc3`/`hvc1`/
`hev1` sample entries with the matching config box and `CODECS=` string.
**H.265 encodes 8- or 10-bit** (Main / Main 10 4:2:0) on NVENC + QSV — hardware-
validated on RTX 3090 and Intel Arc — and on the software tier, so
`with_bit_depth(TenBit)` / a HDR `ColorPolicy` works for H.265 too. **H.264 at
10 bits is the software tier's alone**: there is no hardware Hi10P profile on
NVENC (no `High 10` GUID), QSV (no `AVC High 10` in oneVPL) or AMF, so on a
build without `h26x-fallback` a 10-bit H.264 request is refused by `validate()`,
not down-converted. The encoder backend is chosen per GPU vendor: NVENC + QSV
encode H.264/H.265, and so does the software tier — the native `h26x` encoders
behind the `h26x-fallback` feature (`encode/h26x_sw.rs`) produce 8- and 10-bit
4:2:0 H.264 (High / High 10) and H.265 (Main / Main 10) on a host with no
capable silicon; AMF's H.264/H.265 path is in progress. The same string vocabulary
(`av1`/`h264`/`h265`) drives the CLI `--codec`, the `codec=` settings key, the
batch manifest `codec:`, and the HTTP `codec` field.

---

## 5. Frame rate — `with_max_frame_rate(fps)`

Cap the output cadence; the source cadence is otherwise preserved.

```rust
spec.with_max_frame_rate(30.0)   // never exceed 30 fps
```

---

## 6. Video filters — `with_filters(...)`

Per-frame transforms — geometry (crop, pad, flip, rotate, grayscale), an image
**overlay** (PNG logo/watermark with alpha), and colour (invert, brightness,
contrast, saturation) — applied to the decoded source **once**, before per-rung
scaling, so a filter applies to every rendition. `spec.filters` is a list of
`codec::filter::VideoFilter`:

```rust
spec.with_filters(vec![
    VideoFilter::Crop { w: 1920, h: 1080, x: None, y: None },
    VideoFilter::Overlay { image: "logo.png".into(), x: 24, y: 24 },
]);
// or parse the equivalent ffmpeg-style string form:
spec.with_filters(codec::filter::parse_chain("crop=1920:1080,overlay=logo.png:24:24")?);
```

See **[Video filters](filters/README.md)** for the full filter set, the string +
structured-object forms, and per-surface usage.

---

## 7. GPU selection — `encode_policy(...)`, `decode_policy(...)`

Which cards, and how the decode and the encode are laid across them. Two
questions, two enums, and each enum answers its question *whole* — so no two
settings can contradict each other ("pin decode to card 2" and "split the
decode across every card" are not both sayable).

| Method | Effect |
|--------|--------|
| `encode_policy(EncodePolicy)` | The encode plan (below). |
| `with_gpu_index(u32)` | Shorthand for `encode_policy(SingleGpu(Some(idx)))`. |
| `decode_policy(DecodePolicy)` | The decode plan (below). |

`EncodePolicy` — which cards encode, and how the work is laid across them:

| Variant | Meaning |
|---------|---------|
| `AllGpus` *(default)* | Every capable card, **ladder-scheduled**: one worker per card, each serving every rung and taking the next chunk of whichever rung is furthest behind. A card idles only when the whole job is out of work, and a ladder deeper than the GPU count still costs one decode. Measured faster than the pinned shape. |
| `PerRung` | Every capable card, each worker **pinned to its own rungs** (rung `i` to worker `i mod workers`) — "one rung, one GPU" when the ladder fits the pool. Predictable placement, and a rung's chunks all come off one card, at the cost of cards idling when their rungs are blocked. For benchmarking against `AllGpus`, and for hosts where placement matters more than throughput. |
| `SingleGpu(Option<u32>)` | One card — pinned to `Some(i)`, or the first with `None` — one encoder per rung, serial. Single-file output is seam-free by construction (there are no chunks); HLS runs one worker. |
| `Family(GpuFamily)` | Every card of one vendor (`GpuFamily::{Nvidia, Amd, Intel}`), ladder-scheduled — e.g. ignore an integrated GPU. |

`DecodePolicy` — which card(s) decode, and whether the decode is one pump or
split into ranges:

| Variant | Meaning |
|---------|---------|
| `Auto` *(default)* | Split the decode into one range per decode-capable card of the encode set, where the source allows (an un-spliced H.264/H.265 input whose keyframes fall on chunk boundaries — see [`plan_decode_ranges`](../crates/rivet/src/decode_pump.rs)); whole otherwise. The cards decode different stretches of the source at the same time; the numbering stays continuous across the join and the output is byte-identical to a whole-source decode. |
| `Whole` | One decoder for the whole source, on the first capable card of the encode set. What every job did before ranges existed; the control arm of any comparison. |
| `SpecificGpu(u32)` | One decoder pinned to that card (e.g. decode on an iGPU while the dGPUs encode). Never split — a split on one card is no split. |
| `FastestGpu` | Benchmark every decode-capable card on a short prefix of the input and put one decoder on the quickest. A no-op on single-GPU hosts. |
| `Ranges(usize)` | Split into up to this many ranges, round-robin over the capable cards. More ranges than cards is legal (several pumps share a card) and is how the split is exercised on a one-card host. |

Both apply to HLS and to multi-GPU single-file alike (single-file's unit is a
chunk of several GOPs stitched back into one MP4 — see
[§8](#8-chunk-seams--chunk_seam_modechunkseammode) for the seams). CLI:
`--encode all|per-rung|single|gpu:N|family:VENDOR`,
`--decode auto|whole|fastest|gpu:N|ranges:N`; settings keys `encode`,
`decode`; the older `--gpu` / `--single-gpu` / `--gpu-family` / `--decode-gpu`
still work as spellings of the same choices.

```rust
spec.encode_policy(EncodePolicy::Family(rivet::GpuFamily::Nvidia))
    .decode_policy(DecodePolicy::SpecificGpu(0));   // decode on GPU 0, encode on the NVIDIA cards

spec.encode_policy(EncodePolicy::PerRung)              // one rung, one GPU
    .decode_policy(DecodePolicy::Whole);              // one decoder, e.g. to A/B the split
```

---

## 8. Chunk seams — `chunk_seam_mode(ChunkSeamMode)`

Only relevant when **multiple GPUs** encode a **single file**: each rung is
chunked at GOP boundaries, encoded in parallel, and stitched. Each chunk is an
independent IDR-led GOP so it always plays, but per-chunk rate control can step
quality at the ~2 s seams. This knob governs that (chiefly for NVENC, which
otherwise runs VBR per chunk; AMD/QSV chunks are already constant-QP):

| `ChunkSeamMode` | Seams | Speed |
|-----------------|-------|-------|
| `Parallel` *(default)* | possible mild NVENC steps | fastest (all GPUs) |
| `ParallelConstQp` | flat (forced constant-QP, quality still tracks the target) | fast (all GPUs) |
| `Serial` | none (one encoder for the whole file) | slower; HLS still uses every GPU |

Single-GPU hosts, `--gpu`/`SingleGpu`, and HLS jobs are unaffected (HLS segments
are independent by design).

---

## 9. Validate — `validate()`

```rust
spec.validate()?;
```

Rejects incoherent specs before any work starts: no rungs, zero/odd dimensions,
container/muxer/mode mismatch, HDR with forced 8-bit, or 10-bit/HDR this build
cannot encode **for the spec's codec** — H.264 at 10 bits without
`h26x-fallback`, say, or AV1 at 10 bits with only software encoders compiled
in. The error names what the build has for that codec and which feature would
serve the request. A backend pinned by name with `TRANSCODE_ENCODER_BACKEND`
counts too, feature or no feature (`h26x` is built by name without
`h26x-fallback`), and the refusal names the pin when one is set. It checks the
build, not the silicon: an NVENC build accepts
10-bit AV1 and a card without AV1 encode refuses it when the encoder is built.
The per-codec answer is queryable at runtime via
`rivet::spec::CodecOutputCaps::of_this_build(codec)` and printed by
`rivet capabilities`.

---

## 10. Run it

| Function | Use |
|----------|-----|
| `rivet::transcode_file(input, output)` | One file → one file, default spec. Returns a `TranscodeOutcome`. |
| `rivet::transcode_bytes(&bytes, ..)` | The in-memory variant. |
| `rivet::run_job_blocking(&bytes, &spec, out_dir, sink)` | Run a full `OutputSpec` synchronously. `out_dir: Option<&Path>` (the HLS/multi-rung asset root; `None` = temp dir). Returns `JobOutput`. |
| `rivet::run_job(&bytes, &spec, out_dir, sink).await` | The async variant (drive from a Tokio runtime). |
| `rivet::probe_file(path)` / `probe_bytes(&bytes)` | Inspect without transcoding → `MediaInfo`. |

### Progress

Both `run_job*` take a `ProgressSink` that streams a uniform
[`RungProgress`](../crates/rivet/src/progress.rs) per rung — `label`, `status`
(`RungStatus`: `Pending` → `Running` → `Completed`/`Failed`), `percent`,
`frames_done`, segment + byte counters. Wire it however you like:

```rust
use std::sync::Arc;

// a closure
let sink = Arc::new(rivet::fn_sink(|p| println!("{} {:.0}%", p.label, p.percent)));

// or a Tokio channel (async)
let (tx, mut rx) = tokio::sync::mpsc::channel(64);
let sink = Arc::new(rivet::channel_sink(tx));
```

---

## Full method reference

| `OutputSpec` | Signature | Section |
|--------------|-----------|---------|
| `single_file` | `(Vec<Rung>) -> Self` | [1](#1-construct--the-output-shape) |
| `hls` | `(Vec<Rung>, f32) -> Self` | [1](#1-construct--the-output-shape) |
| `audio_only` | `() -> Self` | [3](#mp3-and-audio-only-output--outputspecaudio_only) |
| `with_audio` | `(AudioCodecPolicy) -> Self` | [3](#3-audio--with_audioaudiopolicy) |
| `audio_only_in` | `(Container) -> Self` | [3](#3-audio--with_audioaudiocodecpolicy) |
| `with_audio_bit_depth` | `(AudioBitDepth) -> Self` | [3](#3-audio--with_audioaudiocodecpolicy) |
| `with_flac_level` | `(FlacLevel) -> Self` | [3](#3-audio--with_audioaudiocodecpolicy) |
| `with_audio_bitrate` | `(u32) -> Self` | [3](#bitrate--with_audio_bitratebps) |
| `with_audio_channels` | `(AudioChannels) -> Self` | [3](#channel-layout--with_audio_channelsaudiochannels) |
| `with_audio_stereo_fallback` | `(bool) -> Self` | [3](#hls-stereo-fallback--with_audio_stereo_fallbacktrue) |
| `with_max_frame_rate` | `(f64) -> Self` | [5](#5-frame-rate--with_max_frame_ratefps) |
| `with_color` | `(ColorPolicy) -> Self` | [4](#4-color--bit-depth) |
| `with_bit_depth` | `(BitDepth) -> Self` | [4](#4-color--bit-depth) |
| `web_sdr` / `hdr10` / `hlg` / `passthrough` | `() -> Self` | [4](#4-color--bit-depth) |
| `with_gpu_index` | `(u32) -> Self` | [6](#6-gpu-selection) |
| `encode_policy` | `(EncodePolicy) -> Self` | [7](#7-gpu-selection--encode_policy-decode_policy) |
| `decode_policy` | `(DecodePolicy) -> Self` | [7](#7-gpu-selection--encode_policy-decode_policy) |
| `chunk_seam_mode` | `(ChunkSeamMode) -> Self` | [7](#7-chunk-seams--chunk_seam_modechunkseammode) |
| `with_rung_policy` | `(RungPolicy) -> Self` | [2](#per-rung-policy--with_rung_policyrungpolicy) |
| `validate` | `(&self) -> Result<()>` | [8](#8-validate--validate) |
| `tonemaps` | `(&self) -> bool` | (does this spec tonemap?) |
| `resolve_output` | `(ColorMetadata, PixelFormat) -> (ColorMetadata, PixelFormat)` | (resolve color/depth vs a source) |

All `OutputSpec` fields are `pub`, so anything above can also be set directly
(`spec.color = ColorPolicy::Hdr10;`): `mode`, `video_codec`, `audio`, `audio_bitrate`,
`audio_channels`, `audio_stereo_fallback`, `audio_filters`, `container`,
`muxer`, `rungs`, `max_frame_rate`, `gpu_index`, `encode_policy`, `decode_policy`,
`color`, `bit_depth`, `chunk_seam_mode`, `rung_policy`. The builders are the recommended path
(they keep linked fields — e.g. `gpu_index` and `encode_policy` — in sync).
