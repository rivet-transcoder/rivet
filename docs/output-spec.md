# Configuring a transcode — the complete `OutputSpec` guide

Everything you can configure for a rivet job lives on one struct,
[`OutputSpec`](../crates/rivet/src/spec/mod.rs). You **build** it (constructor +
chained `with_*` setters), optionally **validate** it, then **run** it. This page
documents every knob; for the internals see [pipeline & architecture](pipeline.md),
and for the CLI equivalents see the [CLI reference](cli.md).

```rust
use rivet::{OutputSpec, Rung, Quality, AudioCodecPolicy, EncodePolicy,
            run_job_blocking, fn_sink};
use rivet::spec::{ChunkSeamMode, PerceptualTarget};   // the rest of the spec types live here
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
> (one AV1 MP4 at the source resolution; AAC / Opus / AC-3 / E-AC-3 / DTS
> and MP3 passed through, the rest of the audio transcoded to Opus; 8-bit
> SDR, an HDR source tonemapped; one decoder and one encoder, serially).

---

## 1. Construct — the output shape

| Constructor | Output |
|-------------|--------|
| `OutputSpec::single_file(rungs)` | One self-contained file per rung (video + audio): a faststart **MP4** by default (AV1), or the codec's own file once `with_video_codec` names it — a **QuickTime movie** for ProRes, a **WebM** for VP8 / VP9 — or the file `with_container(Container::Mp4 \| Mov \| WebM)` names. |
| `OutputSpec::hls(rungs, segment_seconds)` | A segmented **CMAF/HLS** package: `master.m3u8` + an audio rendition group + `video/<h>p/{init.mp4, seg-*.m4s, playlist.m3u8}` per rung, segment-aligned for clean ABR. |
| `OutputSpec::audio_only()` | The **audio alone** as one bare `.mp3` (`OutputMode::AudioOnly`, `Container::Mp3`, `Muxer::Mp3File`): no rungs, no video decoded. A `single_file` job whose input has no video becomes this by itself. See [§3](#3-audio--with_audioaudiocodecpolicy). |
| `OutputSpec::audio_only_in(container)` | The audio alone in the file `container` names: `Container::Mp3`, `Container::Flac` (a native `.flac`) or `Container::M4a` (an audio-only MP4). |

`rungs` is a `Vec<Rung>` (next section). `segment_seconds` is the HLS target
segment length (segments still break on keyframes). The constructor wires the
matching `Container` + `Muxer` + `OutputMode` for you.

---

## 2. The ladder — rungs & quality

A [`Rung`](../crates/rivet/src/spec/rung.rs) is one rendition: a size — a **box**
the source is fitted into, see [below](#fitting-the-source-into-a-rung) — + a
per-rung [`Quality`](../crates/rivet/src/spec/rung.rs).

```rust
Rung::new(1280, 720)                       // auto label "720p", default quality
    .with_quality(Quality::crf(28))        // or .with_quality(Quality::target(..))
    .with_label("hd")                      // override the auto label
```

| `Rung` method | Effect |
|---------------|--------|
| `Rung::new(width, height)` | A rung with a `width × height` box; label auto-set to `"<short-side>p"`, default quality. |
| `.with_quality(Quality)` | Set the per-rung encoder quality. |
| `.with_label(impl Into<String>)` | Override the auto label. |
| `.with_fit(Fit)` / `.with_orientation(Orientation)` / `.with_upscale(bool)` | This rung's own fitting, over the spec's. |
| `.with_standard_rate()` | The rate the rung would have with none named anywhere (`WxH@standard`): a spec-wide rate does not reach it. See [bitrate rungs](#quality). |
| `.short_side()` | The "p" number (`min(width, height)`). |
| `.scale(&frame)` | This rung's frame from a decoded one: its `placement`, or a plain resize to `width x height` when it has none. |

Public fields: `width`, `height`, `label`, `quality`, `fit`, `orientation`,
`upscale` (the three `Option`s; `None` takes the spec's), `placement` (set by
the engine when it fits the rung) and `standard_rate`.

### Fitting the source into a rung

A rung's `width × height` is a **maximum box**, not the output size. Once the
source is probed, the engine replaces each rung's size with the size it
produces, re-derives an automatic label from it, and reports every requested
rung in `JobOutput::renditions` (the box asked for, the size produced). How
the picture meets the box is `OutputSpec::fit` (`with_fit`, `--fit`, settings
key `fit`), or a rung's own. The arithmetic is [`rivet::fit`](../crates/rivet/src/fit.rs)
(`place`, `fit_rungs`); `spec.with_rungs_fitted(source)` is the spec as the
engine runs it for a source shape, with what became of each requested rung.

| `Fit` | Output | Picture |
|---|---|---|
| `Contain` (default) | inside the box, the source's shape | whole |
| `Cover` | the box's shape | centre-cropped to fill it |
| `Pad` | exactly the box | whole, letterboxed or pillarboxed in black |
| `Stretch` | exactly the box | distorted to fill it — what every explicit rung did before fitting |

- **Orientation** (`OutputSpec::orientation`, `with_orientation`, `--orientation`): `Auto`
  (default) reads the box as long side × short side, so a 1920x1080 rung on a
  portrait source is 1080x1920. `Fixed` uses the box as written — a 9:16
  social rung that crops a landscape source is `1080x1920:cover:fixed`.
- **No upscaling** (`OutputSpec::upscale`, `with_upscale`, `--upscale`, default off): a source
  smaller than the box comes out at its own size, even-aligned. `cover`
  without upscale comes out in the box's shape at the largest size the source
  fills. Rungs that collapse onto the same output are merged — the first is
  kept and the others are reported with `duplicate_of`. Rungs asked for with
  the same box are never merged.
- **Non-square pixels**: the source's sample aspect ratio (MP4 `pasp`,
  Matroska display size, else the H.264/HEVC VUI or MPEG-2 sequence header)
  gives its display shape, and the output has square pixels: an anamorphic
  720x576 at 64:45 is fitted as the 1024x576 picture it is shown as.
- Sizes are always even (4:2:0). Labels are `<short side>p` of the output,
  made unique (`720p-2`) when two rungs land on the same short side.

On string surfaces a rung carries its own fitting after `:` —
`1080x1920:cover:fixed`, `1280x720@3M:pad`, `640x360:upscale`.

### Quality

```rust
Quality::crf(28)                           // constant rate factor (lower = better)
Quality::target(PerceptualTarget::High)    // perceptual target instead of a CRF
```

| `Quality` field | Type | Meaning |
|-----------------|------|---------|
| `crf` | `Option<u8>` | Constant rate factor, encoder-native (NVENC AV1 0..=63; the software AV1 and VP9 encoders take AV1's / libvpx's 0..=63 and multiply by 4). `None` → derive from `target`. |
| `speed_preset` | `Option<u8>` | Encoder-native speed preset. `None` → derive from `tier`. |
| `target` | `QualityTarget` | Perceptual target (used when `crf` is `None`). |
| `tier` | `SpeedTier` | Speed/efficiency tier (used when `speed_preset` is `None`). Settings key `video-speed` (`--video-speed`, `video_speed` in the API and the manifest) sets it for every rung; an `encode-policy` `speed=` word for one rung wins. |
| `keyframe_interval` | `Option<u32>` | GOP length in frames. `None` → `2 × fps` (a 2-second GOP). |
| `overrides` | `EncodeOverrides` | Backend-agnostic per-rung knobs layered on the target/tier — a quality shift in libaom-CQ steps, tiles, reference frames, lookahead, B-frames, a bitrate and its buffer. Inert by default. |

`Quality::crf` / `Quality::target` are the two constructors; set the rest with
struct-update syntax, e.g. `Quality { tier: Speed::Archive, keyframe_interval:
Some(120), ..Quality::crf(30) }`, or `.with_overrides(EncodeOverrides { .. })`.

**VMAF as the target.** `QualityTarget::Vmaf(n)` aims every rung at a VMAF
score; `codec::encode::tuning` turns it into each backend's quantiser through
calibrated anchor tables, so `Vmaf(93)` means the same perceived quality on
NVENC, QSV, AMF and the software AV1 encoder (`docs/av1-tuning-research.md` has the tables and
`docs/av1-tuning-methodology.md` how to re-calibrate for a new encoder). On
every surface it is the word `vmaf=93` — `--target`, `target=` on the socket,
`target` in the API/manifest, `target=vmaf=93` in the policy grammar. Whether a
target *delivers* its VMAF is measured, not assumed: [`bench/`](../bench/README.md)
scores a ladder against its source with libvmaf.

**GOP.** `OutputSpec::gop` (`with_gop(Some(frames))`, CLI `--gop`, key `gop`) sets the
keyframe cadence for every rung — and, on the multi-GPU single-file path, the
chunk grid, since a chunk is a whole number of GOPs. For HLS the segment grid is
`segment_seconds`; a GOP shorter than the segment adds keyframes inside it, a
longer one is silently the segment. A rung's own `Quality::keyframe_interval`
wins over the spec-wide value. A GOP can also be given in seconds of output
(`gop=1.5s`, `OutputSpec::gop_seconds`, `with_gop_seconds(Some(1.5))`; a
`gop` in frames wins over it, and `validate` refuses a non-positive one): it is made frames at the output
frame rate once that is known, rounded to the nearest frame exactly as the
two-second default is (`gop_frames_for_seconds`), so `gop=2s` is the
default, stated, and builds the same job as no `gop`.

**Bitrate rungs.** A rung can be coded to a rate instead of a quality:
`EncodeOverrides::bitrate` (bits per second) and `buffer_ms` (the coded
picture buffer, in milliseconds of that rate; one second unless named, `0`
declares none). On the
surfaces these are `--rung 1280x720@3M` (that rung), `--video-bitrate 3M`
(every rung without its own), `--video-buffer 1s`, and the policy grammar's
`bitrate=` / `buffer=` for a derived ladder. The precedence is the rung's own
`@RATE`, then the policy, then `--video-bitrate`. `WxH@standard` gives a rung
the rate it would have with none named anywhere, whatever `--video-bitrate`
or a policy `bitrate=` says: under `rate-mode=cbr` the default for its codec,
size and frame rate; otherwise no rate (its quality target).

A rate is an **average** rate unless the rung is constant-rate
(`EncodeOverrides::rate_mode = Some(RateMode::Constant)`; `rate-mode=cbr`
on the surfaces, `rate=cbr` in the policy grammar; `constant`, and
`average` / `abr` for the other, are spellings of the same two).

- **Average.** The software encoders code to an average rate — the native
  H.264 / H.265 encoder, and the software AV1, VP9, MPEG-2 and MPEG-4 Part 2
  encoders: `validate` refuses one beside a CRF or under `--seam-mode
  constqp`, and a buffer without a rate (the software AV1, VP9, MPEG-2 and
  MPEG-4 encoders refuse a coded picture buffer at all). The job
  refuses an average-rate rung whose encode pool is GPUs, before a frame is
  decoded.
- **Constant (CBR).** The rate is also the maximum and a buffer is declared
  (one second unless named). Coded by QSV, NVENC and AMF for every codec
  they encode, AV1 included, and by the native software H.264 / H.265
  encoder; not by the software AV1 or VP9 encoders, so an AV1 or VP9
  constant-rate job whose encoders are software is refused. `validate` refuses one beside a CRF, under
  `--seam-mode constqp`, or with `buffer=0`. A constant-rate rung with no
  rate of its own takes `--video-bitrate`, else the default for its codec,
  short side and output frame rate
  (`codec::encode::tuning::default_cbr_bitrate`), set once the frame rate
  is known (`OutputSpec::with_constant_rates_resolved`).

A rung without a rate is the quality-target encode it always was. What a
buffer buys (the HLS `BANDWIDTH` it bounds) and what the rate costs are
measured in [codec-encode.md](codec-encode.md#bitrate-rungs-in-the-software-tier-measured).
`spec.bitrate_rung()`, `average_rate_rung()` and `constant_rate_rung()`
name the first rung, with the policy resolved, coded to a rate of each kind.

### Per-rung policy — `with_rung_policy(RungPolicy)`

A ladder wants different knobs at different positions: softer going down (the
same quantizer at a quarter of the resolution is a far finer quantizer in
terms of what an eye can resolve), one tile below 4K, more reference frames.
Rather than hand-setting `overrides` on every rung, give the spec a
[`RungPolicy`](../crates/codec/src/encode/tuning/overrides.rs) and the engine
resolves it against each rung's position before encoding, layering the rung's
own `overrides` on top (the rung-specific knob wins; quality deltas
accumulate). `spec.with_rung_policy_resolved()` is that spec, with the
policy folded into the rungs and emptied:

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
`bitrate` (`3M`, `800k`), `buffer` (`1s`, `500ms`, `0`), `rate`
(`cbr` / `constant`, `average` / `abr`); `qstep=N` alone is the
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
  `Archive` (slowest/most efficient). On the string surfaces it is
  `video-speed=draft|standard|archive` for every rung (`standard` the
  default; an `encode-policy` `speed=` word wins for its rungs); the old
  `speed=` / `preset=` keys are refused, naming `video-speed`. Each encoder
  maps it onto its own presets: NVENC P5 / P6 / P7; the software AV1 encoder
  its motion search range (±8 / ±16 / ±32); VP9 the crate's speed and
  partitioning (`draft` fixed 32x32 partitions; `standard` fixed 16x16, about
  10 frames/s at 352x288 on one core; `archive` a rate-distortion partition
  search, about 1.7 frames/s).

### Auto ladder

Don't want to hand-write rungs? Derive a standard ABR ladder from the source:

```rust
let rungs = rivet::standard_ladder(source_w, source_h, /* max_short_side */ Some(1080));
let spec = OutputSpec::single_file(rungs);
```

It snaps to standard short sides (2160/1440/1080/720/480/360/240), preserves
aspect ratio, even-aligns dims, and caps the top rung (`None` is the default
cap, 1080; a higher one unlocks 1440 and 2160). A standard rung within 15%
below the source's short side is dropped and the source-size rung kept
(`ladder::SOURCE_SNAP_TOLERANCE`). `rivet::ladder::standard_ladder_with_quality` gives
every rung one `Quality`.

---

## 3. Audio — `with_audio(AudioCodecPolicy)`

| `AudioCodecPolicy` | Behavior |
|---------------|----------|
| `Auto` *(default)* | Passthrough AAC / Opus / AC-3 / E-AC-3 / DTS verbatim, and MP3 into a single-file MP4 (Opus and Vorbis into a WebM); transcode the rest (Vorbis, MP2, PCM, FLAC, ALAC; MP3 for HLS) → Opus; drop what cannot be decoded. For `audio_only()` it means **MP3**: an MP3 source passes through, the rest is encoded. |
| `ForceOpus` (`opus`) | Always produce Opus (passthrough Opus, transcode everything else; 1–8 channels). Single-file MP4 / MOV / WebM, HLS, an audio-only Ogg Opus file (the default file for it) or `.m4a`. |
| `ForceMp3` (`mp3`) | Always produce **MP3** (passthrough MP3, encode everything else — CBR, stereo at most). Single-file MP4 and audio-only; refused for HLS. |
| `ForceAac` (`aac`) | Always produce **AAC-LC** (passthrough AAC, encode everything else — mono to 7.1, constant rate). The audio every browser and device plays, older iOS and Safari included (Opus in MP4 needs iOS / Safari 17). Single-file MP4 / MOV, HLS and an audio-only `.m4a`. |
| `ForceHeAac` (`he-aac`) | Always produce **HE-AAC** (`mp4a.40.5`: an AAC-LC core at half the rate plus spectral band replication), at 32, 44.1 or 48 kHz, mono to 7.1 — for low rates (default 48k stereo). An HE-AAC source (v1 or v2) is kept. Where AAC goes. |
| `ForceHeAacV2` (`he-aacv2`) | Always produce **HE-AAC v2** (`mp4a.40.29`: HE-AAC with parametric stereo), stereo only (a wider source is downmixed; a mono one refused), 16–64k (default 32k). An HE-AAC v2 source is kept. Where AAC goes. |
| `ForceVorbis` (`vorbis`) | Always produce **Vorbis**, variable-rate by `audio_quality` (−1…10, default 5; no bitrate), 1–8 channels at the source's rate. A WebM, or an audio-only Ogg Vorbis file; refused for MP4, MOV and HLS, which have no Vorbis mapping. |
| `ForceAc3` (`ac3`) | Always produce **AC-3** (Dolby Digital): A/52's arrangements up to 5.1 (5.1 as 5.1(side), 7.1 downmixed), 48 / 44.1 / 32 kHz, Table 5.18's rates (default 192k stereo, 448k 5.1). Single-file MP4 / MOV (`ac-3` + `dac3`), HLS (`ac-3`) and an audio-only `.m4a`. |
| `ForceEac3` (`eac3`) | Always produce **E-AC-3** (Dolby Digital Plus): as AC-3, at 32k–6144k (default 192k stereo, 384k 5.1); `ec-3` + `dec3`. |
| `ForceDts` (`dts`) | Always produce **DTS** (the Coherent Acoustics core): the core's arrangements up to 5.1, 48 / 44.1 / 32 kHz, ETSI TS 102 114 Table 5-7's rates (default the full rate: 1536k at 48 kHz); `dtsc` + `ddts`. Single-file MP4 / MOV, HLS and an audio-only `.m4a`. |
| `Drop` | Video-only output. |
| `Flac` | Lossless FLAC: copy a FLAC source, encode anything decodable. Plays from MP4 in Chrome, Edge, Firefox and Safari; audio-only output is a native `.flac`. |
| `Alac` | Lossless ALAC: copy an ALAC source, encode anything decodable. Plays on Apple platforms and in Safari only; audio-only output is an `.m4a`. |

```rust
spec.with_audio(AudioCodecPolicy::ForceOpus)
    .with_audio_bitrate(320_000)
    .with_audio_channels(AudioChannels::Surround51)
```

Lossless output takes two more knobs — `with_audio_bit_depth(AudioBitDepth::{Source, Sixteen, TwentyFour})`
and, for FLAC, `with_flac_level(FlacLevel::{Fast, Default, Best})` — and Vorbis
takes `with_audio_quality(q)` (`audio_quality`, −1…10), which no other codec
does (Vorbis refuses a bitrate). An audio-only job can be written as a bare
`.mp3`, a native `.flac`, an `.m4a` or an Ogg file:
`OutputSpec::audio_only_in(Container::{Mp3, Flac, M4a, Ogg})`
(`OutputSpec::audio_only_container(policy)` is the default for a policy: a
`.flac` for FLAC, Ogg for Opus and Vorbis, an `.m4a` for ALAC, the AAC
profiles, AC-3, E-AC-3 and DTS, an `.mp3` otherwise). `spec.file_extension()`
names the file a single-file or audio-only job writes (`.opus` for Ogg Opus).
See [lossless-audio.md](lossless-audio.md) for the lossless rules.
`spec.audio_encode_codec()` is the codec a transcoded track becomes under the
spec: FLAC / ALAC when asked for, the forced codec of a `Force…` policy, MP3
for a bare `.mp3`, Opus otherwise.

`validate()` checks the audio against the file before any work: a codec the
file cannot hold is refused by name (`audio=vorbis` into an MP4, `audio=ac3`
into a WebM, anything but MP3 into a bare `.mp3`), and so is a bitrate the
codec does not code (an MP3 or AC-3 rate off its table, a DTS rate off Table
5-7, an AAC / HE-AAC / Opus / E-AC-3 rate outside its range, any bitrate for
Vorbis) and a layout it cannot carry (`audio-channels=5.1` with MP3 or HE-AAC
v2, `7.1` with AC-3, E-AC-3 or DTS).

`with_audio_filters(Vec<AudioFilter>)` sets an audio filter chain
(`channelmap`) run on the decoded PCM before the encoder; a filter forces the
track to be decoded and re-encoded, and `validate()` refuses one beside
`Drop`. See [audio filters](audio-filters.md).

AAC sources are decoded by rivet's own AAC decoder (the `crates/aac`
submodule; [decisions.md §26](decisions.md#26-aac-lc-is-encoded-and-decoded-here-from-the-standards)):
AAC-LC, HE-AAC and HE-AAC v2 in full, mono to 7.1 and program_config_element
layouts, so an AAC track can be downmixed, filtered, or transcoded to any
other codec, and it is still passed through untouched wherever nothing asks
for a change. `ForceAac` on an AAC source is simply a passthrough, as are
`ForceHeAac` on an HE-AAC one and `ForceHeAacV2` on an HE-AAC v2 one. A source a forced codec
cannot reach (an AAC object type the decoder refuses: Main, SSR, LTP) is
passed through into an MP4 or HLS package with a warning, the handling saying
so — and refused for a bare `.mp3`, which cannot hold it.

**HE-AAC** (and HE-AAC v2) decodes in full: spectral band replication at the
full rate, parametric stereo to two channels. The handling names the source
by its profile — `he-aac → opus (2ch)`, `he-aacv2 → …` — where an AAC-LC
decode reads `aac → opus (2ch)`. Decoding only the core (half the stream's
sample rate, a quarter of its full rate's bandwidth, HE-AAC v2's single core
channel) is still available, and reads `he-aac (lc core) → opus (2ch)`; that
wording is a contract, and a passthrough never contains it. What an HE-AAC
source becomes is `with_he_aac(HeAacPolicy)` (settings word `he-aac`):

| `HeAacPolicy` | Settings word | HE-AAC source |
|---|---|---|
| `Auto` *(default)* | `auto` | As any AAC track: passed through where the output can carry it and nothing asks for a change; decoded in full where the job needs PCM or another codec. |
| `Passthrough` | `passthrough` | Never decoded: passed through where the output can carry AAC (a codec change asked of it is not made, the handling saying why), and the job refused where it cannot (the error names the setting). |
| `Core` | `core` | Decoded as its AAC-LC core only, whenever it is decoded: the cheaper, narrower decode. |

Explicit signalling (object type 5 or 29 in the AudioSpecificConfig, or its
backward-compatible sync extension) and implicit signalling (SBR data in the
access units, found in the first one) are both recognised.

### Restricting decoders — `audio_decode_deny`

`with_audio_decode_deny(AudioDecodeDeny)` (settings word `audio-decode-deny`,
a comma list of `aac`, `ac3`, `alac`, `dts`, `eac3`, `flac`, `mp2`, `mp3`, `opus`, `pcm`, `vorbis`; empty or `none` restricts
nothing, an unknown name is a settings error) names source audio codecs that
may **not be decoded**. Each source codec spelling a decoder takes counts
under its name (`mp4a` as `aac`, `ec-3` as `eac3`, every `pcm_*` as `pcm`).
A denied track never reaches a decoder:

- Where the output can carry it as it is, it is **passed through**, packet
  for packet — also when another codec was asked of it (`opus`, `mp3`, `flac`
  or `alac` beside video), as for a codec with no decoder; the handling then
  reads e.g. `aac passthrough (opus requested; decoding aac is denied)`.
- Where the output needs its PCM — a downmix (`audio-channels`), an audio
  filter, a bare `.mp3` or native `.flac`, an output that cannot hold the
  codec — the job is **refused** when its audio is prepared, before any video
  is decoded or encoded, e.g. `decoding aac audio is denied by the audio-decode-deny setting; the output needs decoded audio (audio-channels=stereo of a 6-channel track)`.
  The parenthesis names what needed the decode: `audio filters: <chain>`,
  `audio-channels=<layout> of a <n>-channel track`, `an .mp3 file holds MP3`,
  `a native FLAC file holds FLAC`, or `<codec> output, which passing the
  <codec> track through cannot give`.
- With `aac` denied an HE-AAC source has no core to decode, so `he_aac` has
  nothing to choose: the track is passed through or refused as above.

AAC output is an `mp4a` sample entry whose `esds` carries the
AudioSpecificConfig (object type 2, the channel configuration of ISO/IEC
13818-7 Table 42: 1 mono, 2 stereo, 3 3.0, 4 4.0, 5 5.0, 6 5.1, 7 7.1), with
`codecs` `mp4a.40.2` in the job output and the HLS master, and each HLS audio
rendition's `CHANNELS` its channel count. It is coded at 22.05 / 24 / 32 /
44.1 / 48 kHz: another source rate is resampled to the nearest in its family
(the 11.025 kHz family to 22.05 / 44.1, the rest to 24 / 32 / 48). The
encoder's one frame (1024 samples) of priming is hidden by the MP4 edit list,
so the track presents exactly the source's samples. The encoder and decoder
are rivet's own, written from the standards ([decisions.md §26](decisions.md#26-aac-lc-is-encoded-and-decoded-here-from-the-standards));
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
passthrough. Changing the width of an AAC track decodes it (an HE-AAC track
in full, or as its core under `he-aac=core`, unless `he-aac=passthrough`
refuses that).

The layout comes from the decoder, not the channel count: an AC-3 stream says
which of 4.0, quad(side) and 3.1 its four channels are (`acmod`), DTS the same
(`AMODE`), and a stream that changes layout mid-way is remixed to the output's.

### HLS: stereo fallback — `with_audio_stereo_fallback(true)`

Beside a surround audio rendition, a stereo downmix of it in the same audio
group — `EXT-X-MEDIA` entries with `CHANNELS="2"` (the group's `DEFAULT`) and
`CHANNELS="6"`, distinct `NAME`s, and every variant's `CODECS` listing each
codec the group holds — so a player on stereo hardware takes the stereo one
rather than downmixing itself. Nothing is added when the audio is stereo or
mono; a surround track that cannot be decoded (an AAC object type the
decoder refuses, or HE-AAC under `he-aac=passthrough`) goes alone, and the
job's audio handling says why. `validate()` refuses the flag outside HLS.

### MP3 and audio-only output — `OutputSpec::audio_only()`

MP3 in a single-file MP4 is an `mp4a` sample entry with an `esds` of object
type 0x6B (0x69 at the MPEG-2 half rates), and the job reports its `codecs`
value as `mp3` (`JobOutput::audio_codecs`) — the spelling Chromium and Firefox
both accept; `mp4a.6B` is rejected by Firefox and `mp4a.40.34` by both.
Encoding resamples to 32 / 44.1 / 48 kHz where the source is not already one of
them (the 11.025 kHz family to 44.1, the rest to 48), and the encoder delay is
hidden by the MP4 edit list.

`audio_only()` writes the audio alone as a bare `.mp3`: an `Info` frame (frame
and byte counts, a seek table) and the frames. For an encode it is the
encoder's own tag frame, whose LAME-style extension (encoder `rivetmp3`)
carries the encoder delay and end padding; for an MP3 passthrough whose
source's tag stated them, the same fields under the source's encoder name —
so a gapless player decodes exactly the source's samples. Another codec is
refused for an `.mp3` (`audio-container` picks another file), as are a trim
and a splice. `audio=opus` and `audio=vorbis` write an Ogg file instead, whose
granule positions do the same for the pre-skip and the end; the AAC profiles,
AC-3, E-AC-3 and DTS an `.m4a`, whose edit list does. The job's one output is
labelled `audio` (width and height 0); the input is read by
`container::streaming::demux_audio`, which takes a video file's audio track, a
bare MP3/MP2, a native FLAC, an Ogg Opus / Vorbis file, an audio-only MP4 /
M4A, or an audio-only Matroska / WebM.

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
`av1-sw-fallback`, is 10-bit SDR: no HDR, as it writes no colour description
into the sequence header); H.265 on those or `h26x-fallback` (Main 10); VP9
10-bit SDR (profile 2) in every build;
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

`Hdr10` / `Hlg` on an SDR source map its picture into the HDR signal
(ITU-R BT.2408, `codec::colorspace::SdrToHdr`; `spec.sdr_to_hdr(&source)`
says which transfer), and an SDR source mapped into PQ is signalled with a
BT.709 / 203 cd/m² mastering display and content light level
(`SDR_IN_PQ_MASTERING_DISPLAY`, `SDR_IN_PQ_CONTENT_LIGHT_LEVEL`) unless it
carried its own. rivet does not convert between HDR transfers:
`spec.check_source_colour(&source)` refuses `hdr10` on an HLG source, `hlg`
on a PQ one, and an SDR source the mapping cannot take, before anything is
decoded.

`with_chroma_downsample(ChromaDownsample)` (settings key
`chroma-downsample`) picks the 4:4:4 → 4:2:0 filter for 4:4:4 sources:
`Box` (the default) or `Lanczos` (siting-correct Lanczos-2). It does nothing
to 4:2:0 or 4:2:2 sources.

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
single-file MP4 **and** CMAF/HLS — the muxer emits `av01`/`avc1`/`hvc1` sample
entries (`avc3`/`hev1` only where the parameter sets change mid-stream) with the
matching config box and `CODECS=` string.
**H.265 encodes 8- or 10-bit** (Main / Main 10 4:2:0) on NVENC, QSV and AMF —
hardware-validated on RTX 3090, Intel Arc and a Ryzen 9 9950X iGPU — and on
the software tier, so
`with_bit_depth(TenBit)` / a HDR `ColorPolicy` works for H.265 too. **H.264 at
10 bits is the software tier's alone**: there is no hardware Hi10P profile on
NVENC (no `High 10` GUID), QSV (no `AVC High 10` in oneVPL) or AMF, so on a
build without `h26x-fallback` a 10-bit H.264 request is refused by `validate()`,
not down-converted. The encoder backend is chosen per GPU vendor: NVENC, QSV
and AMF encode H.264/H.265, and so does the software tier — the native `h26x` encoders
behind the `h26x-fallback` feature (`encode/h26x_sw.rs`) produce 8- and 10-bit
4:2:0 H.264 (High / High 10) and H.265 (Main / Main 10) on a host with no
capable silicon. The same string vocabulary
(`av1`/`h264`/`h265`) drives the CLI `--codec`, the `codec=` settings key, the
batch manifest `codec:`, and the HTTP `codec` field.

### The other codecs: VP9, VP8, MPEG-2, MPEG-4 Part 2, ProRes

Every codec rivet decodes it can write too, with its own clean-room encoders
(software, in every build — no feature, no GPU):

```rust
use rivet::{Container, VideoCodecPolicy};
use rivet::spec::ProresProfile;

// A QuickTime movie of ProRes 422 HQ (the codec picks the file).
let master = OutputSpec::single_file(rungs.clone()).with_video_codec(VideoCodecPolicy::ProRes(ProresProfile::Hq));
// VP9 in WebM (its default file), or in an MP4.
let webm = OutputSpec::single_file(rungs.clone()).with_video_codec(VideoCodecPolicy::Vp9);
let vp9_mp4 = webm.clone().with_container(Container::Mp4);
// MPEG-2 in a QuickTime movie, with B pictures (two by default).
let dvdish = OutputSpec::single_file(rungs).with_video_codec(VideoCodecPolicy::Mpeg2).with_container(Container::Mov);
```

| Codec | Single file (default first) | HLS | Encoder | Depth / colour |
|---|---|---|---|---|
| AV1 (default) | MP4 | yes | NVENC / AMF / QSV, rivet's own (`crates/av1`, `av1-sw-fallback`) | 8- / 10-bit; HDR on the GPUs only |
| H.264 | MP4, QuickTime | yes | NVENC / AMF / QSV, h26x (`h26x-fallback`) | 8-bit; 10-bit + HDR in software |
| H.265 | MP4, QuickTime | yes | NVENC / AMF / QSV, h26x (`h26x-fallback`) | 8- / 10-bit + HDR |
| VP9 | WebM, MP4 | yes | rivet's own (`crates/vp9`), every build | profile 0 (8-bit) or 2 (10-bit, `--pixel-format 10bit`), 4:2:0, SDR |
| VP8 | WebM, MP4 | no | rivet's own (`crates/vp8`), every build | 8-bit 4:2:0, SDR |
| MPEG-2 | MP4, QuickTime | no | rivet's own (`crates/mpeg2`), every build | Main Profile, 8-bit 4:2:0, SDR |
| MPEG-4 Part 2 | MP4, QuickTime | no | rivet's own (`crates/mpeg4`), every build | Simple / Advanced Simple, 8-bit 4:2:0, SDR |
| ProRes (six profiles) | QuickTime only | no | rivet's own (`crates/prores`), every build | 4:2:2 / 4:4:4 from the 8- / 10-bit pipeline, HDR-tagged |

`with_video_codec` on a single-file spec still in its default MP4 moves it to
the codec's own file (`VideoCodecPolicy::default_container`); `with_container`
after it picks another. `validate()` refuses, by name and before anything is
decoded: a codec in a file that does not carry it (ProRes outside a `.mov`,
VP8 / VP9 in a QuickTime movie, MPEG-2 in WebM, …); VP8, MPEG-2, MPEG-4 or
ProRes as HLS (no CMAF binding); HDR for VP9, 10-bit or HDR for VP8 / MPEG-2 /
MPEG-4; a bitrate for VP8 / ProRes (a fixed quantiser; ProRes's rate is
its profile's); a constant rate or a coded picture buffer for VP9 / MPEG-2 /
MPEG-4 (they code an average rate); B frames for VP9 / VP8 / ProRes (ProRes is
intra-only); a crf for ProRes; sizes past MPEG-2's 4095x2800, MPEG-4's
8191x8191 or VP8's 16383x16383; non-Opus audio or `metadata_keep` in WebM.
A WebM carries Opus audio (copied, or encoded from anything decodable) and no
subtitles; a QuickTime movie takes the MP4 audio and `tx3g` subtitles. A
`crf` is the codec's own scale (VP9 / VP8 the libvpx 0-63 `cq-level`, MPEG-2
/ MPEG-4 their 1-31 codes); a quality target maps onto it. These codecs
encode on the serial path, one encoder per rung (never chunked across GPUs).
Settings keys: `codec`, `container` (`mp4`, `mov`, `webm`), `prores-profile`
(`proxy`, `lt`, `422`, `hq`, `4444`, `4444xq`).

---

## 5. Frame rate — `with_max_frame_rate(fps)`

Cap the output cadence; the source cadence is otherwise preserved.

```rust
spec.with_max_frame_rate(30.0)   // never exceed 30 fps
```

A cap below the source's rate **drops frames**, it does not retime them:
each output frame period gets the source frame showing at its start, counted
from the trim in-point, so the output keeps the source's duration and stays
in step with its audio (a 60 fps source capped at 30 keeps every other
frame). Frame counts, progress, HLS segment counts and chunk plans are in
output frames. Under a cap the decode is not split into ranges
(`DecodePolicy`), since a sample index no longer counts the output frames
before it. A cap at or above the source's rate changes nothing. Settings key
`max-fps` (`source` states the default).

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
container/muxer/mode mismatch, a non-positive HLS segment length or GOP in
seconds, `metadata_keep` on HLS, the audio knobs against each other and the
output ([§3](#3-audio--with_audioaudiocodecpolicy)), a rate a rung cannot be
coded to ([bitrate rungs](#quality)), HDR with forced 8-bit, or 10-bit/HDR this build
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
`rivet capabilities`. An audio-only spec is checked for its audio and its
file only. What only the source can settle (a 10-bit or HDR source kept by
`Auto` / `Passthrough`, an HDR transfer the colour policy cannot take) is
checked once the source is probed, before a frame is decoded.

---

## 10. Run it

| Function | Use |
|----------|-----|
| `rivet::transcode_file(input, output)` | One file → one file, default spec. Returns a `TranscodeOutcome`. |
| `rivet::transcode_bytes(&bytes)` | The in-memory variant; the output is `outcome.output_bytes`. |
| `rivet::run_job_blocking(&bytes, &spec, out_dir, sink)` | Run a full `OutputSpec` synchronously. `out_dir: Option<&Path>` (the HLS/multi-rung asset root; `None` = temp dir). Returns `JobOutput`. `run_job_blocking_owned` takes a `Bytes` the caller already owns, without a copy. |
| `rivet::run_job(input, &spec, out_dir, sink).await` | The async variant (drive from a Tokio runtime); `input` is a `bytes::Bytes`. |
| `rivet::run_splice_job(clips, &spec, out_dir, sink).await` / `run_splice_job_blocking` | Join several `Clip`s (each with its own range) into one output. Refuses audio-only output and `metadata_keep`. |
| `rivet::image::run_image_job(&bytes, &image_spec)` | A still-image job ([§11](#11-still-images--modeimage)). |
| `rivet::probe_file(path)` / `probe_bytes(&bytes)` | Inspect without transcoding → `MediaInfo` (its `sample_aspect` and `display_dims` give the source's display shape). |

`JobOutput::renditions` reports what fitting made of each requested rung
([§2](#fitting-the-source-into-a-rung)), and `JobOutput::hooks` the hooks'
report ([§15](#15-hooks--with_hookshooks)).

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
let sink = rivet::channel_sink(tx);   // already an Arc<dyn ProgressSink>
```

---

## 11. Still images — `mode=image`

*(the `image` feature)* A page is mostly pictures, and they are web media the
way video is: formats every browser decodes, at the sizes the layout asks for,
upright, in sRGB, and carrying nothing the uploader did not mean to publish.
An image job makes them, from a still image or from a video.

It is its own spec, [`rivet::image::ImageSpec`](../crates/rivet/src/image/mod.rs),
run by `rivet::image::run_image_job(&bytes, &spec)` (blocking; CPU-bound;
`run_image_job_with_hooks(&bytes, &spec, &hooks)` with [hooks](hooks.md)). The
string surfaces build it with `mode=image` and
`TranscodeSettings::into_image_spec`; `rivet image` is the CLI
([cli.md](cli.md#rivet-image)). `run_job` does not make images, and an image
knob on a video job is refused.

| Input | Read by |
|---|---|
| JPEG (EXIF orientation applied, ICC, CMYK / YCCK) | rivet's own `crates/jpeg` |
| PNG (every colour type and depth; `eXIf` orientation, `iCCP`) | rivet's own `crates/png` |
| GIF (first frame, composited), TIFF (first page; orientation, ICC, BigTIFF), BMP | rivet's own `crates/imagecodecs` |
| AVIF | rivet's HEIF reader → the AV1 decode dispatch (NVDEC / AMF / QSV, else rivet's own `av1` decoder, in every build) |
| WebP (a still, or an animation's first frame, composited; ICC) | rivet's own `crates/webp` |
| HEIC / HEIF | rivet's HEIF reader → the HEVC decode dispatch (GPU, else rivet's own `h26x`) |
| a video | the thumbnail path's decoder: the stills `frames-at` / `frames-count` pick |

| Output | Encoder | Notes |
|---|---|---|
| `avif` (default) | rivet's own AV1 encoder (`crates/av1`) in rivet's own HEIF writer (`avif.rs`) | 8-bit 4:2:0; alpha, when the picture has it, as an auxiliary item at 3/4 of the colour quantiser; `colr` nclx (BT.709 primaries, sRGB transfer, BT.601 matrix, full range); always sRGB (no ICC). Over 2048x2048 pixels, or wider than 4096, a `grid` of equal tiles of at most 2048 a side, encoded in parallel |
| `webp` | rivet's own `crates/webp` | lossy (VP8, through rivet-vp8) at the job's quality, default 80, or lossless (VP8L) with `image-lossless`; alpha kept; ICC kept with `image-keep-icc`; effort from `image-speed` |
| `jpeg` | rivet's own `crates/jpeg` | progressive, 4:2:0, optimised Huffman; transparency flattened onto white; ICC kept with `image-keep-icc` |
| `png` | rivet's own `crates/png` | RGB, or RGBA when the picture has transparency; DEFLATE level from `image-speed` |

| Setting (`key=value`) | Library | Meaning |
|---|---|---|
| `mode=image` | — | an image job |
| `image-format=avif,webp,jpeg,png` | `formats` | every rendition in each, in order. Default `avif` |
| `rung=WxH[:fit][:auto\|fixed][:upscale]` (repeatable) | `renditions` | boxes, fitted as [video rungs are](#fitting-the-source-into-a-rung) but to the pixel (`place_aligned(.., 1)`): a 641x481 photo in a larger box stays 641x481. None: one output at the picture's own size. No `@RATE` |
| `fit`, `orientation`, `upscale` | same | as for video. A rendition a small picture collapses onto another's output is made once (`ImageJobOutput::merged`) |
| `image-quality=1..100` / `image-quality=avif:60,webp:80,jpeg:82` | `quality` / `format_quality` | a bare number is every lossy format; `format:N` is that one (over a bare number); a lossy format not named keeps its default: AVIF 60, WebP 80, JPEG 82, so naming each at its default makes the same files as no `image-quality`. A bare number is refused when nothing lossy is made; a named format this job does not make does nothing; `png` and unknown formats are refused |
| `image-lossless=1` | `lossless` | WebP lossless; refused with AVIF or JPEG (PNG is lossless anyway) |
| `image-keep-icc=1` | `keep_icc` | keep the source's colour profile (PNG, JPEG and WebP carry it) rather than converting to sRGB |
| `image-speed=1..10` | `speed` | PNG's DEFLATE level and WebP's effort. PNG: 1→9, 2→8, 3→7, 4–6→6, 7→5, 8→4, 9→3, 10→1; default 6 (level 6). WebP (rivet-webp's effort, 0 fastest – 6 smallest): 1–2→6, 3–4→5, 5–6→4, 7–8→2, 9–10→0; default 6 gives 4, the codec's own default. On a photo-like 2048x2048 picture levels 1 / 3 / 6 / 9 took 0.25 / 0.34 / 1.12 / 1.59 s for 6.44 / 6.33 / 6.11 / 6.11 MB, and level 9 takes up to 8.5 s on other content, which is why it is not the default. AVIF and JPEG ignore it (until 2026-10-03 it was AVIF's effort) |
| `frames-at=1.5,10` / `frames-count=N` / `frames=poster` | `frames` | a video's stills: at these seconds, or N evenly spaced (the middles of N equal slices). Neither (or `frames=poster`, which states it): one frame 10% in, and a still image as it is. `frames-at` / `frames-count` are refused on a still image, and beside `frames=poster`; a time past the end is refused |
| `image-decode-deny=heic` | `decode_deny` | still-image inputs not to decode, refused as `decoding heic images is denied by the image-decode-deny setting`. Rides along on video jobs, ignored there |
| `metadata-keep=…` | `metadata_keep` | identifying source metadata written into every output as EXIF, as for video ([§14](#14-source-metadata--metadata_keep)). Default none |

What every output gets:

- **Upright**: EXIF orientation and HEIF `irot` / `imir` are applied.
- **No metadata**: outputs are encoded from pixels; EXIF, XMP, GPS and
  embedded thumbnails never reach them, unless `metadata-keep`
  (`ImageSpec::metadata_keep`) names a category: then a fresh EXIF block
  holding only that is written (a still from a video takes the video's).
  See [§14](#14-source-metadata--metadata_keep). The colour profile can be
  kept too (`image-keep-icc`).
- **sRGB**: a source with an ICC profile or a HEIF `nclx` naming other
  primaries is converted (moxcms). A profile that cannot be read leaves the
  pixels as they are, as a browser would show them.
- **Resampled** with rivet's own Lanczos-3 (`image/raster.rs`).

Every artifact is `<W>x<H>.<ext>` (`jpg` for JPEG), or `<W>x<H>-<nnn>.<ext>`
when a video gave several stills; a second rendition coming out the same size
another way is `<W>x<H>-2`. `ImageArtifact::rendition` says which requested
rendition it is, `frame` which still and at what time.

```rust
use rivet::image::{ImageFormat, ImageRendition, ImageSpec, run_image_job};

let spec = ImageSpec {
    formats: vec![ImageFormat::Avif, ImageFormat::Jpeg],
    renditions: vec![ImageRendition::new(1920, 1920), ImageRendition::new(640, 640)],
    ..ImageSpec::default()
};
let out = run_image_job(&bytes, &spec)?;
for a in &out.artifacts {
    std::fs::write(a.file_name(out.several_frames), &a.bytes)?;
}
```

**Limits.** Sources over 100 megapixels are refused from the header. Outputs
are at most 16384 a side. HEIF derived items other than `grid`
are refused, as are HEIF pictures coded with anything but AV1 or HEVC. HDR
stills are not tone-mapped. Why any of this is so: [decisions §28](decisions.md#28-still-images-are-web-media-and-get-the-webs-formats).

---

## 12. Subtitles — `with_subtitles(SubtitlePolicy)`

| `SubtitlePolicy` | Settings word | Carried |
|---|---|---|
| `All` *(default)* | `all` | every **text** track, in source order |
| `Drop` | `none` | no subtitle track |
| `Only(Vec<String>)` | `eng,deu` | the tracks in these languages, in list order (the first is the default HLS rendition). Codes match by language, not spelling (`en` = `eng`, `ger` = `deu`); a language no track has is logged, not an error |

A single-file MP4 gets one `tx3g` track per language, an HLS package one
segmented-WebVTT rendition per language. Bitmap subtitles (PGS, VobSub, DVB)
have no text form and are dropped with a warning under every policy. The
sources read, and what a trim does to the cues: [cli.md](cli.md#subtitles).

---

## 13. Trim — `with_trim(start, end)`

`with_trim(Some(2.0), Some(7.0))` keeps `[start, end)` seconds of the single
input (either bound `None` is open; `trim_start` / `trim_end` are the
fields), re-based to zero. Frames before the in-point are decoded and
dropped. Trimmed jobs take the serial encode path, and audio-only output
refuses a trim. CLI `--trim-start` / `--trim-end`. To join several clips,
each with its own range, use `run_splice_job`.

---

## 14. Source metadata — `metadata_keep`

By default an output carries none of the source's identifying metadata: the
muxers write no location, device, capture time or tags of the source's, and
with the device not kept a copied AAC or MP3 stream's encoder name is
cleared too (an AAC fill element's payload, MP3's ancillary bytes; the tag
frame keeps its delay and padding under rivet's own name, `rivetmp3`),
without changing the audio. `OutputSpec::metadata_keep` (a
[`container::metadata::Keep`](../crates/container/src/metadata/mod.rs); there
is no builder, set the field or use `Keep::parse`) names what to carry, per
category; settings key `metadata-keep` (`--metadata-keep`, manifest and HTTP
`metadata_keep`):

| Word | Keeps |
|---|---|
| `location` / `location:approximate` | GPS coordinates and place names / coordinates to two decimal places (about a kilometre), no altitude or place name |
| `capture_time` / `capture_time:date` | when it was recorded / the date only, time of day zeroed, no offset |
| `device` / `device:all` | make, model, software, lens / those plus serial numbers and owner name |
| `descriptive` | title, artist, copyright, comment, description, keywords, cover art |
| `all` / `none` | everything / nothing (the default) |

Words combine with commas (`location:approximate,descriptive`). What is kept
is read from the source and written into each single-file MP4, `.m4a`,
`.flac` (Vorbis comments) or `.mp3` (ID3v2), and into stills as EXIF
([§11](#11-still-images--modeimage)). HLS refuses it (`validate()`: a player
reads no file-level metadata from segments), and so does a splice, whose
clips can each say something different.

---

## 15. Hooks — `with_hooks(Hooks)`

`OutputSpec::hooks` is a [`rivet::hooks::Hooks`](../crates/rivet/src/hooks/mod.rs):
caller-supplied code run at fixed points of every job the spec drives — the
source bytes, the probe, decoded frames, the frames the encoders receive,
each artifact, and the end. Each hook can annotate the job's report
(`JobOutput::hooks`) or reject the job. Empty by default, which costs
nothing. The guide is [hooks.md](hooks.md); recipes are in the
[cookbook](hooks-cookbook.md), and [hooks-yolo.md](hooks-yolo.md) is a
complete vision-model integration.

```rust
use rivet::hooks::{Hooks, SourceDigest, DigestAlgorithm};

let spec = spec.with_hooks(
    Hooks::new().source("source-digest", SourceDigest::new(&[DigestAlgorithm::Sha256])),
);
```

---

## Full method reference

| `OutputSpec` | Signature | Section |
|--------------|-----------|---------|
| `single_file` | `(Vec<Rung>) -> Self` | [1](#1-construct--the-output-shape) |
| `hls` | `(Vec<Rung>, f32) -> Self` | [1](#1-construct--the-output-shape) |
| `audio_only` | `() -> Self` | [3](#mp3-and-audio-only-output--outputspecaudio_only) |
| `audio_only_in` | `(Container) -> Self` | [1](#1-construct--the-output-shape) |
| `audio_only_container` | `(AudioCodecPolicy) -> Container` | [3](#3-audio--with_audioaudiocodecpolicy) (the file a policy's audio-only output is, unless named) |
| `with_audio` | `(AudioCodecPolicy) -> Self` | [3](#3-audio--with_audioaudiocodecpolicy) |
| `with_audio_bit_depth` | `(AudioBitDepth) -> Self` | [3](#3-audio--with_audioaudiocodecpolicy) |
| `with_flac_level` | `(FlacLevel) -> Self` | [3](#3-audio--with_audioaudiocodecpolicy) |
| `with_audio_bitrate` | `(u32) -> Self` | [3](#bitrate--with_audio_bitratebps) |
| `with_audio_channels` | `(AudioChannels) -> Self` | [3](#channel-layout--with_audio_channelsaudiochannels) |
| `with_audio_stereo_fallback` | `(bool) -> Self` | [3](#hls-stereo-fallback--with_audio_stereo_fallbacktrue) |
| `with_he_aac` | `(HeAacPolicy) -> Self` | [3](#3-audio--with_audioaudiocodecpolicy) |
| `with_audio_decode_deny` | `(AudioDecodeDeny) -> Self` | [3](#restricting-decoders--audio_decode_deny) |
| `with_audio_filters` | `(Vec<AudioFilter>) -> Self` | [3](#3-audio--with_audioaudiocodecpolicy) |
| `audio_encode_codec` | `(&self) -> AudioCodec` | [3](#3-audio--with_audioaudiocodecpolicy) |
| `file_extension` | `(&self) -> &'static str` | [3](#3-audio--with_audioaudiocodecpolicy) |
| `with_subtitles` | `(SubtitlePolicy) -> Self` | [12](#12-subtitles--with_subtitlessubtitlepolicy) |
| `with_video_codec` | `(VideoCodecPolicy) -> Self` | [5b](#5b-output-codec--with_video_codec) |
| `with_max_frame_rate` | `(f64) -> Self` | [5](#5-frame-rate--with_max_frame_ratefps) |
| `with_color` | `(ColorPolicy) -> Self` | [4](#4-color--bit-depth) |
| `with_bit_depth` | `(BitDepth) -> Self` | [4](#4-color--bit-depth) |
| `with_chroma_downsample` | `(ChromaDownsample) -> Self` | [4](#4-color--bit-depth) |
| `web_sdr` / `hdr10` / `hlg` / `passthrough` | `(self) -> Self` | [4](#4-color--bit-depth) |
| `with_fit` / `with_orientation` / `with_upscale` | `(Fit)` / `(Orientation)` / `(bool) -> Self` | [2](#fitting-the-source-into-a-rung) |
| `with_rungs_fitted` | `(&self, SourceShape) -> (OutputSpec, Vec<FittedRung>)` | [2](#fitting-the-source-into-a-rung) |
| `with_filters` | `(Vec<VideoFilter>) -> Self` | [6](#6-video-filters--with_filters) |
| `with_trim` | `(Option<f64>, Option<f64>) -> Self` | [13](#13-trim--with_trimstart-end) |
| `with_gpu_index` | `(u32) -> Self` | [7](#7-gpu-selection--encode_policy-decode_policy) |
| `encode_policy` | `(EncodePolicy) -> Self` | [7](#7-gpu-selection--encode_policy-decode_policy) |
| `decode_policy` | `(DecodePolicy) -> Self` | [7](#7-gpu-selection--encode_policy-decode_policy) |
| `chunk_seam_mode` | `(ChunkSeamMode) -> Self` | [8](#8-chunk-seams--chunk_seam_modechunkseammode) |
| `with_rung_policy` | `(RungPolicy) -> Self` | [2](#per-rung-policy--with_rung_policyrungpolicy) |
| `with_gop` / `with_gop_seconds` | `(Option<u32>)` / `(Option<f64>) -> Self` | [2](#quality) |
| `gop_frames` | `(&self, f64) -> u32` | [2](#quality) (the GOP at a frame rate) |
| `with_gop_seconds_resolved` / `with_rung_policy_resolved` / `with_constant_rates_resolved` | `(&self, ..) -> OutputSpec` | [2](#quality) (the spec as the engine runs it) |
| `bitrate_rung` / `average_rate_rung` / `constant_rate_rung` | `(&self) -> Option<..>` | [2](#quality) |
| `with_hooks` | `(Hooks) -> Self` | [15](#15-hooks--with_hookshooks) |
| `validate` | `(&self) -> Result<()>` | [9](#9-validate--validate) |
| `check_source_colour` | `(&self, &ColorMetadata) -> Result<()>` | [4](#4-color--bit-depth) |
| `tonemaps` | `(&self) -> bool` | (does this spec tonemap?) |
| `sdr_to_hdr` | `(&self, &ColorMetadata) -> Option<TransferFn>` | [4](#4-color--bit-depth) |
| `resolve_output` | `(ColorMetadata, PixelFormat) -> (ColorMetadata, PixelFormat)` | (resolve color/depth vs a source: what the output is encoded and tagged as) |

Free items in `rivet::spec`: `gop_frames_for_seconds` and
`DEFAULT_GOP_SECONDS` (2.0), `sdr_into_hdr`, `encoder_input_format` (the
4:2:0 format the pump hands the encoder for a source format), the
`SDR_IN_PQ_*` constants, and the capability queries (`CodecOutputCaps`,
`every_codec_output_caps`, ...).

All `OutputSpec` fields are `pub`, so anything above can also be set directly
(`spec.color = ColorPolicy::Hdr10;`): `mode`, `video_codec`, `audio`, `audio_bitrate`,
`audio_channels`, `audio_stereo_fallback`, `audio_bit_depth`, `he_aac`,
`audio_decode_deny`, `metadata_keep`, `flac_level`, `subtitles`, `audio_filters`,
`container`, `muxer`, `rungs`, `fit`, `orientation`, `upscale`, `max_frame_rate`,
`gpu_index`, `encode_policy`, `decode_policy`, `gop`, `gop_seconds`, `rung_policy`,
`color`, `chroma_downsample`, `bit_depth`, `chunk_seam_mode`, `filters`,
`trim_start`, `trim_end`, `hooks`. The builders are the recommended path
(they keep linked fields — e.g. `gpu_index` and `encode_policy` — in sync).

## Stating the defaults

Every `key=value` setting a caller can leave out has a value that states what
leaving it out does, and builds exactly the same job, so a caller can name
every setting and leave nothing to an implicit default:

| Key | The default, stated |
|---|---|
| `audio-bitrate` | `standard` (the codec's rate for the output layout) |
| `video-bitrate` | `standard` (none: a `cbr` rung takes the default for its codec, size and frame rate); per rung `WxH@standard` |
| `gop` | `2s` (seconds are made frames at the output rate; `48` is still frames) |
| `max-fps` | `source` |
| `max-short-side` | `standard` (1080; there is no uncapped ladder) |
| `segment-seconds` | `4` |
| `bit-depth` / `audio-channels` / `audio-bit-depth` / `he-aac` | `auto` / `source` / `source` / `auto` |
| `subtitles` / `flac-compression` / `target` / `color` | `all` / `default` / `standard` / `sdr` |
| `fit` / `orientation` / `upscale` / `audio-stereo-fallback` | `contain` / `auto` / `false` / `false` |
| `audio-container` | `auto` (follows the codec) |
| `audio` / `codec` / `encode` / `decode` / `seam` / `chroma-downsample` | `auto` / `av1` / `all` / `auto` / `parallel` / `box` |
| `metadata-keep` / `audio-decode-deny` / `encode-policy` | `none` / `none` / `off` (`encode-policy=default` is the recommended policy, not the absence of one) |
| `ladder` | `false` |
| `image-quality` | `avif:60,webp:80,jpeg:82` |
| `image-format` / `image-speed` / `image-lossless` / `image-keep-icc` | `avif` / `6` / `false` / `false` |
| `frames` (image) | `poster` |

`mode=audio` still refuses a video word by name (`fit=contain` included), and
`mode=image` a video or audio one, except the words above that leave their key
unset (`standard`, `source`, `2s`), which carry nothing into the job.
