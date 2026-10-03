# Lossless audio: FLAC and ALAC

rivet reads and writes the two lossless audio formats that browsers and Apple
devices play: **FLAC** and **ALAC** (Apple Lossless). Both are implemented in
pure Rust in this workspace, decoders and encoders alike, with no codec
library underneath. The codecs are the `lossless` crate,
[`crates/lossless`](../crates/lossless/README.md): a git submodule, the
[rivet-lossless](https://github.com/safewords/rivet-lossless)
repository (published as `rivet-lossless`), where they are changed. The
`codec` crate re-exports it as `codec::audio::lossless` and adapts it to
its `AudioDecoder` / `AudioEncoder` traits.

- FLAC: [`flac/decode.rs`](../crates/lossless/src/flac/decode.rs),
  [`flac/encode.rs`](../crates/lossless/src/flac/encode.rs),
  [`flac/format.rs`](../crates/lossless/src/flac/format.rs) (STREAMINFO,
  metadata blocks, the CRCs)
- ALAC: [`alac/decode.rs`](../crates/lossless/src/alac/decode.rs),
  [`alac/encode.rs`](../crates/lossless/src/alac/encode.rs),
  [`alac/format.rs`](../crates/lossless/src/alac/format.rs) (the cookie, the
  Rice coder and adaptive predictor)
- Shared: [`bits.rs`](../crates/lossless/src/bits.rs) (bit I/O),
  [`lpc.rs`](../crates/lossless/src/lpc.rs) (LPC analysis),
  [`pcm.rs`](../crates/lossless/src/pcm.rs) (integer PCM ↔ f32),
  [`layout.rs`](../crates/lossless/src/layout.rs) (speaker names)
- Adapters in `codec`: [`audio/decode/flac.rs`](../crates/codec/src/audio/decode/flac.rs),
  [`audio/decode/alac.rs`](../crates/codec/src/audio/decode/alac.rs),
  [`audio/encode/flac.rs`](../crates/codec/src/audio/encode/flac.rs),
  [`audio/encode/alac.rs`](../crates/codec/src/audio/encode/alac.rs)
- Containers: [`demux/audio/lossless.rs`](../crates/container/src/demux/audio/lossless.rs),
  [`mux/lossless.rs`](../crates/container/src/mux/lossless.rs)

## Why a web-first engine carries lossless audio

The output is still web media. Both formats play in the browser from the
containers rivet already writes, and neither carries a royalty:

| | FLAC | ALAC |
|---|---|---|
| In MP4 | Chrome, Edge, Firefox 51+ (desktop) / 58+ (Android), Safari 11+ | Safari and every Apple platform, natively; **not** Chrome, Edge or Firefox |
| In HLS (fMP4) | Apple HLS (the HLS authoring specification lists FLAC with `CODECS="fLaC"`, fMP4 required); hls.js on the browsers above through MSE | Apple HLS (`CODECS="alac"`, fMP4 required); not hls.js on non-Apple browsers |
| Licence | Open format, royalty-free (RFC 9639) | Format and reference published by Apple under Apache 2.0 |

(Browser support as listed by MDN's audio codec guide, September 2026.)

The use cases are the ones where lossy audio is the wrong answer: masters and
archive copies, and lossless music delivery (the HLS switchable set Apple
defines pairs ALAC / FLAC renditions with AAC ones). Neither is the default:
`audio=auto` still passes AAC, Opus, AC-3, E-AC-3 and DTS through (and MP3 into
a single-file MP4) and transcodes the rest to Opus, and that includes FLAC and
ALAC sources, which it can decode.

For the widest reach pick FLAC; ALAC is the choice when the audience is Apple
devices. With video, both are allowed in MP4 and HLS; a player that cannot
decode the audio still plays the video.

## Settings

| Key | Values | |
|---|---|---|
| `audio` | `flac`, `alac` (beside `auto`, `opus`, `mp3`, `aac`, `drop`) | Encode the audio losslessly. A source already in that codec is **copied** (frames untouched; a FLAC copy's configuration is STREAMINFO alone, see [Containers](#containers)) unless a filter, `audio-channels` or a different bit depth asks for a re-encode. |
| `audio-bit-depth` | `source` (default), `16`, `24` | FLAC / ALAC only. `source` keeps 16 bits for a 16-bit (or shallower) or lossy source and 24 bits for anything deeper: a 20-bit source is carried exactly in 24; a 32-bit or float source is rounded to 24. `16` rounds deeper audio to the nearest step (no dither). |
| `flac-compression` | `fast`, `default` (default), `best` | FLAC only. `fast`: fixed predictors, Rice partitions to order 3. `default`: LPC to order 8, the order picked from the Levinson error estimate, partitions to order 6. `best`: every LPC order to 12 priced exactly, partitions to order 8. |
| `mode` | `audio` (beside `single`, `hls`) | The audio alone, as one file (`OutputMode::AudioOnly`): the video is never decoded, and the input need not have any — a native `.flac`, an `.m4a`, an `.mp3` or a Matroska audio file are all inputs, and a single-file job of such an input becomes this mode by itself. |
| `audio-container` | `auto` (default), `mp3`, `flac`, `mp4` | The file of an audio-only output. `auto` follows the codec: a native `.flac` for `audio=flac`, an `.m4a` for `audio=alac`, else an `.mp3`. `mp4` writes an `.m4a` for any codec the MP4 muxer takes (Opus and AAC included). |

The same keys are CLI flags (`--audio flac`, `--audio-bit-depth 24`,
`--flac-compression best`, `--mode audio`, `--audio-container mp4`), HTTP query
or JSON fields (`audio_bit_depth`, `flac_compression`, `audio_container`),
batch-manifest fields, and `#rivet` IPC header keys.

`audio-channels` (`source` / `mono` / `stereo` / `5.1` / `7.1`) applies to
lossless output as to Opus: a wider source is downmixed on the decoded PCM
before the encoder, and a narrower one is refused (rivet does not upmix).
With `source`, FLAC and ALAC keep the source's layout as it is, up to eight
channels.

### Validation errors

| Request | Refusal |
|---|---|
| `audio=flac` or `alac` with `audio-bitrate` | lossless has no bitrate to set |
| `audio-bit-depth` other than `source` without `audio=flac|alac` | applies to FLAC and ALAC output only |
| `flac-compression` other than `default` without `audio=flac` | applies to FLAC output only |
| `audio-container=flac` with anything but `audio=flac` | a native FLAC file holds FLAC only |
| `audio=flac|alac` into an `.mp3` (`audio-container=mp3`) | an `.mp3` cannot hold lossless audio |
| `audio-container` on a job with video | it names an audio-only output's file |

The audio-only mode's own rules apply as well: `audio=drop` and video knobs
(rungs, `crf`, …) are refused by name, and a trim is not available.

There is no transport-stream output in rivet, so there is no lossless-in-TS
case to refuse: the outputs are MP4, CMAF/HLS and, for audio alone, an
`.mp3`, a `.flac` or an `.m4a`.

A source rivet cannot decode (an AAC object type the decoder refuses, such as
AAC Main) under `audio=flac|alac` is passed through as it is, as `audio=opus`
does, and the job's audio handling says so (`aac passthrough (flac requested;
no aac decoder)`). AAC-LC, HE-AAC and HE-AAC v2 are decoded like any other
source (HE-AAC in full; as its AAC-LC core under `he-aac=core`, and not at
all under `he-aac=passthrough`). `audio-decode-deny` can forbid decoding
any source codec, `flac` and `alac` included: a denied track is passed through
where the output holds it and refused where the output needs its PCM (see
[output-spec.md](output-spec.md#restricting-decoders--audio_decode_deny)).

## Containers

| | Sample entry | Codec string | Notes |
|---|---|---|---|
| FLAC in MP4 / CMAF | `fLaC` + `dfLa` (FullBox: STREAMINFO alone, flagged last) | `fLaC` | per "Encapsulation of FLAC in ISO Base Media File Format" (xiph.org); `mdhd` timescale = sample rate; `samplerate` field 0 above 65535 Hz |
| ALAC in MP4 / CMAF | `alac` + `alac` (FullBox: the 24-byte `ALACSpecificConfig`) + `chan` past two channels | `alac` | `chan` names the ALAC layout (MPEG 3.0 B, 4.0 B, 5.0 D, 5.1 D, AAC 6.1, MPEG 7.1 B) |
| Native FLAC | `fLaC`, STREAMINFO, SEEKTABLE (a point every 10 s, on a frame start), VORBIS_COMMENT (an empty vendor string, no comments) | — | audio-only output; an encode's STREAMINFO is the encoder's, with the sample count and MD5 of what it coded; a copy's is the source's |
| Matroska (in) | `A_FLAC`, `A_ALAC` | — | CodecPrivate normalised to the MP4 forms; packet durations come from each frame's own sample count |

A FLAC source's other metadata blocks — Vorbis comments (its vendor string,
title, artist, date, …), pictures, application data, its seek table — are
read from `dfLa`, `A_FLAC` CodecPrivate or the native header, and not carried
into the output: whether copied or encoded, a FLAC track's configuration is
STREAMINFO alone, and a native `.flac` names no vendor (until 2026-09-29 a copy
kept every block and the writer named rivet). `metadata-keep` is the one way
tags reach the output: what it keeps is written into the native file's
`VORBIS_COMMENT` (or an `.m4a`'s `meta` keys); see
[output-spec.md §14](output-spec.md#14-source-metadata--metadata_keep). In the
library, `container::mux::write_native_flac_with_vendor` names a vendor.

CAF (Core Audio Format) is not read: rivet has no CAF demuxer, and ALAC
arrives in MP4/M4A or Matroska in practice. ALAC in CAF is unsupported.

Every lossless packet is one frame, and its duration is the frame's own
sample count. The encoders write 4096-sample frames.

## Channel order

The pipeline carries the WAVE channel order (`WAVEFORMATEXTENSIBLE`). FLAC's order for every count is
that order. ALAC leads with the centre channel, so the ALAC decoder and
encoder reorder:

| Channels | ALAC order | Native layout |
|---|---|---|
| 3 | C L R | 3.0: FL FR FC |
| 4 | C L R Cs | 4.0: FL FR FC BC |
| 5 | C L R Ls Rs | 5.0: FL FR FC BL BR |
| 6 | C L R Ls Rs LFE | 5.1: FL FR FC LFE BL BR |
| 7 | C L R Ls Rs Cs LFE | 6.1: FL FR FC LFE BC SL SR |
| 8 | C Lc Rc L R Ls Rs LFE | 7.1(wide): FL FR FC LFE BL BR FLC FRC |

The ALAC decoder reports four channels as 4.0, so a downmix of one weighs
the right speakers. The pipeline has no label for eight channels' front
left- and right-of-centre pair, so an eight-channel ALAC source is taken as
7.1, that pair in the SL / SR slots. On the way out, a lossless encode keeps
the samples in the slots they arrived in: a quad or 7.1 source written as
four- or eight-channel ALAC carries its channels in ALAC's 4.0 / 7.1(wide)
positions.

## Precision through the pipeline

Decoded audio travels as f32. An integer sample of `b` bits maps to
`s / 2^(b-1)` and back by the inverse, which is exact for every depth up to
24 bits (the f32 significand). So FLAC/ALAC → FLAC/ALAC at 24 bits or less is
bit-exact end to end; a 32-bit source keeps its top 24 bits. The decoders'
integer interfaces (`FlacDecoder::decode_int`, `AlacDecoder::decode_int`) are
exact at every depth, and the encoders' (`encode_int`) take any depth the
format has (FLAC 4–32, ALAC 16/20/24/32).

## Verification

[`crates/codec/tests/lossless_oracle.rs`](../crates/codec/tests/lossless_oracle.rs)
checks both codecs, through rivet's adapters, against independent
implementations used strictly as black boxes (their output is compared; their
source was never read); the `lossless` crate carries the same suite against
its own API as
[`crates/lossless/tests/oracle.rs`](../crates/lossless/tests/oracle.rs). The
references are `flac`, the Xiph.Org reference implementation's command-line
tool; `alacconvert`, built from Apple's open-source ALAC release
([macosforge/alac](https://github.com/macosforge/alac), pinned) by
`crates/lossless/tools/build-alacconvert.sh`; and MKVToolNix (`mkvmerge`,
`mkvextract`) to carry streams into and out of Matroska. `FLAC`, `MKVMERGE`,
`MKVEXTRACT` and `ALACCONVERT` name the binaries when they are not on `PATH`.
The tests skip when a tool is missing.

**Decode, bit-exact against the source PCM:**
- `flac` CLI streams at `-0`, `-3`, `-5`, `-8`, `-8 -l 32`, block sizes
  576 / 1152 / 4096, `--no-mid-side`; 8, 16, 24 and 32 bits; 22.05 / 32 / 44.1 /
  48 / 96 / 192 kHz; 1–8 channels; STREAMINFO MD5 checked.
- FLAC muxed into Matroska by `mkvmerge` and read by rivet's demuxer.
- Apple's ALAC encoder: 16, 24 and 32 bits, 1–8 channels, the magic cookie
  with the channel layout info Apple writes for more than two channels; and
  the same streams muxed into Matroska by `mkvmerge`, decoded through rivet.

**Encode, rivet → decoded by the reference, bit-exact:**
- FLAC: `flac -t` (MD5 verified) and `flac -d` on the native stream, and on
  the stream rivet muxed into MP4, taken back out by `mkvmerge` /
  `mkvextract`; all three effort levels; 16, 24 and 32 bits; 22.05–192 kHz;
  1, 2, 3, 6 and 8 channels.
- ALAC in rivet's MP4, taken back to CAF through `mkvmerge` / `mkvextract`,
  decoded by Apple's decoder: 16, 24 and 32 bits; 1–8 channels. Apple's tool
  does no channel reordering, so this also checks rivet's ALAC channel order
  against the orders Apple documents.
- rivet encode → rivet decode for every depth and layout, and the job engine's
  audio-only outputs end to end (`crates/rivet/src/job/lossless_tests.rs`).

CI (here and in the rivet-lossless repository) installs `flac` and
`mkvtoolnix`, builds `alacconvert`, and runs the oracle tests with
`RIVET_REQUIRE_LOSSLESS_ORACLES=1`, where a missing tool fails instead of
skipping.

**Size against the reference encoders** (10 s stereo at 44.1 kHz, % of the raw
PCM; `flac -5` and Apple's ALAC encoder at their defaults):

| Signal | rivet FLAC fast | default | best | `flac -5` | rivet ALAC | Apple ALAC |
|---|---|---|---|---|---|---|
| tones + noise, 16-bit | 69.8% | 68.7% | 68.4% | 69.2% | 68.8% | 69.0% |
| tones + noise, 24-bit | 79.3% | 78.5% | 78.3% | 78.9% | 78.9% | 79.4% |
| 1 kHz sine, 16-bit | 26.1% | 18.4% | 14.6% | 26.6% | 22.9% | 26.7% |
| brown noise, 16-bit | 62.6% | 62.6% | 62.6% | 63.6% | 63.3% | 63.4% |

**Known issue (ALAC encode):** on about 1 in 70 of the synthetic test
signals, a frame rivet's ALAC encoder writes decodes differently in Apple's
decoder than in rivet's — a few samples after a stretch of silence or of a
constant ends. The two halves agree with each other there, so they share the
departure from the format; the cases are an ignored test in
`crates/lossless/tests/oracle.rs` until it is fixed, and the oracle tests'
signals avoid them.

**Not verified here:** 20-bit ALAC against another implementation, either
way (`alacconvert` neither writes nor reads it; the round trips cover it);
ALAC in MP4 above 65535 Hz through `mkvmerge` (the sample entry's rate field
holds 0 there, which `mkvmerge` refuses; the codec itself is checked at those
rates); FLAC streams with variable block sizes from another encoder (`flac`
does not write them; the decoder handles the flag and the sample-numbered
header); playback in real browsers and on Apple devices, and of lossless HLS
renditions in players.

## Provenance

See [decisions.md §27](decisions.md#27-lossless-audio-is-clean-room-flac-and-alac):
written from RFC 9639, the published ALAC format description, and the
literature on linear prediction and Rice coding; no implementation's source
was consulted.
