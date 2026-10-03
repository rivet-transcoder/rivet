# Audio filters

Per-frame transforms applied to **decoded PCM**, between the decoder and the
audio encoder (Opus, AAC, MP3, FLAC or ALAC) — the audio counterpart to the [video filter chain](filters/README.md),
with the same two interchangeable forms and the same round-trip guarantee
(`parse_chain(&chain_to_string(c)) == c`).

Set one with `--audio-filter` (CLI), `audio-filter=` (IPC header),
`audio_filter` (HTTP / batch manifest).

## They only apply to transcoded audio

A filter needs PCM, so it only exists on the decode → re-encode path. A
**passthrough** track (AAC / Opus / AC-3 / E-AC-3 / DTS, MP3 into an MP4, or a
FLAC / ALAC source under `audio=flac|alac`, copied verbatim) never becomes PCM
unless something asks for it.

rivet handles that by treating an audio filter as an implicit request to
transcode: a track that *could* have been passed through is decoded and
re-encoded instead — to Opus under `audio=auto`, else to the codec `--audio`
names (to MP3 for a bare `.mp3`). If the source cannot be decoded — no decoder
for its codec, an AAC object type the decoder refuses (Main, SSR, LTP), a
codec named in `audio-decode-deny`, or HE-AAC under `he-aac=passthrough` — the
job fails with a message naming the codec and the reason, rather than quietly
emitting an unfiltered passthrough. What decodes today is
[below](#what-you-can-actually-filter-today).

## Catalog

| Filter | Effect |
|--------|--------|
| `channelmap` | Remap, reorder, or select channels. |

---

## `channelmap`

Matches `ffmpeg -filter:a channelmap=MAP[:LAYOUT]`.

```text
channelmap=FL-FL|FR-FR|FC-FC|LFE-LFE|SL-BL|SR-BR:5.1
channelmap=FR-FL|FL-FR:stereo          # swap the front pair
channelmap=FL-FL|FR-FR:stereo          # 5.1 → stereo, front pair only
channelmap=FR|FL:stereo                # positional form
```

```yaml
- channelmap:
    pairs: [[FR, FL], [FL, FR]]
    layout: stereo
```

**Map** — a `|`-separated list of `IN-OUT` pairs: "input channel *IN* becomes
output channel *OUT*". In the positional form (`FR|FL`) the *i*-th entry names
the input feeding output slot *i*, so a layout is required to say what slot *i*
is.

**Layout** — the optional second argument names the output. Omit it and the
layout is inferred from the output labels, in the order written.

Channels the map doesn't mention are **dropped**; an output channel that no pair
feeds is **silent** (not filled from a neighbour).

### Channel names

`FL` `FR` `FC` `LFE` `BL` `BR` `BC` `SL` `SR` — ffmpeg's names. `mono` is
accepted as a spelling of `FC`.

### Layout names

| Name | Channels, in order |
|------|--------------------|
| `mono` | FC |
| `stereo` | FL FR |
| `2.1` | FL FR LFE |
| `3.0` | FL FR FC |
| `4.0` | FL FR FC BC |
| `quad` | FL FR BL BR |
| `5.0` | FL FR FC BL BR |
| `5.0(side)` | FL FR FC SL SR |
| `5.1` | FL FR FC LFE BL BR |
| `5.1(side)` | FL FR FC LFE SL SR |
| `6.1` | FL FR FC LFE BC SL SR |
| `7.1` | FL FR FC LFE BL BR SL SR |
| `3.0(back)` | FL FR BC |
| `quad(side)` | FL FR SL SR |
| `3.1` | FL FR FC LFE |
| `4.1` | FL FR FC LFE BC |

A bare channel count also works (`6` = `5.1`), as does an explicit `FL+FR+FC`
spelling for anything unnamed.

These orders are **ffmpeg's native channel order** for the same names — the
order every decoder emits (AAC, AC-3, DTS, Vorbis, MP3, FLAC; ALAC's decoder
reorders into it) and every filter sees, so a
channel means the same thing at every stage of the pipeline. Opus's
channel-mapping family 1 (RFC 7845 §5.1.1.2) orders 5.1 differently — FL FC FR
RL RR LFE, the Vorbis order — and the Opus and Vorbis encoders permute into it
on the way in, their decoders out of it (`codec::audio::rfc7845_family1_order`);
before 2026-09-13 the Opus encoder fed the native order straight through, which
came out with FC/FR swapped and LFE/SL/SR rotated on any 5.1 source except
Vorbis (whose decoder then happened to emit the Vorbis order).

### How the *input* layout is decided

Containers rarely say more than "6 channels", and 6 channels is either `5.1`
(back surrounds) or `5.1(side)` — precisely the distinction that makes
`SL-BL|SR-BR` worth writing. So the map is read as evidence:

1. If the default layout for that channel count has every channel the map reads,
   use it.
2. Otherwise, look for the named layout of that width that does. Exactly one
   match is the answer.
3. No match, or more than one, is an error that says which channels are at fault.

So `channelmap=…|SL-BL|SR-BR:5.1` resolves the input as `5.1(side)` on a plain
6-channel track, and the relabel works. A map reading both `SL` *and* `BL` fits
no single layout and is rejected rather than guessed at.

### Validation happens up front

The encoder is built before the first frame arrives, so the output channel count
has to be known from the chain alone. That means a map reading a channel the
input can't have fails when the spec is built — not part-way through the audio.

---

## What you can actually filter today

The filter itself handles 1–8 channels, and the Opus encoder carries all of them
(mono/stereo on channel-mapping family 0, 3–8 on family 1 multistream), as do
AAC, FLAC and ALAC; MP3 carries at most stereo. The binding constraint is
upstream: what rivet decodes, and what the job allows it to decode
(`audio-decode-deny`, `he-aac`).

| Source | `--audio-filter` |
|--------|------------------|
| Vorbis (incl. multichannel) | ✅ |
| MP3 / MP2 | ✅ (stereo by nature) |
| Opus (incl. family-1 surround) | ✅ — the `crates/opus` multistream decoder |
| AC-3 / E-AC-3 (incl. 5.1) | ✅ — in-tree decoder ([codec-decode.md](codec-decode.md#ac-3--e-ac-3-decoder)); E-AC-3 7.1 decodes as its 5.1 core |
| DTS core | ✅ — in-tree decoder |
| PCM | ✅ |
| FLAC, ALAC (up to 8 channels) | ✅ — in-tree decoders ([lossless-audio.md](lossless-audio.md)) |
| AAC | ✅ — the `crates/aac` decoder, AAC-LC mono to 7.1 and PCE layouts, HE-AAC and HE-AAC v2 in full (as their AAC-LC core under `he-aac=core`; refused under `he-aac=passthrough`); AAC Main / SSR / LTP do not decode |

So a 5.1 **AAC**, **Vorbis**, **Opus**, **AC-3**, **E-AC-3**, **DTS**, **FLAC**
or **ALAC** source can be remapped and re-encoded. The AC-3 decoder emits channels in ffmpeg's native order for the
layout (5.1: FL FR FC LFE SL SR), which is what `channelmap` expects, and says
which layout that is.

## After the filters: the output layout

The chain runs first; then the frames are converted to the output layout
(`--audio-channels`, or what the codec carries — see
[output-spec.md](output-spec.md#channel-layout--with_audio_channelsaudiochannels)).
A `channelmap` that names its output layout (`…:5.1(side)`) is read as that
layout there, so a relabel and a downmix compose: `channelmap=…:5.1` then
`--audio-channels stereo` downmixes the relabelled 5.1.

## Related

- [`--audio-bitrate`](cli.md#rivet-transcode) — the target for transcoded
  lossy audio. Defaults to the encoder's value: Opus from the layout (64k mono,
  96k stereo, 320k for 5.1, 416k for 7.1), AAC 64k mono / 128k stereo / 384k
  5.1 / 512k 7.1, MP3 128k stereo / 64k mono. FLAC and ALAC take none.
- [`--audio-channels`](cli.md#rivet-transcode) — the output layout; downmixes,
  never upmixes.
- [`--audio`](cli.md#rivet-transcode) — the policy: `auto`, `opus`, `mp3`,
  `aac`, `flac`, `alac` or `drop`. `--audio drop` with a filter set is
  rejected as a contradiction.
- `--audio-decode-deny`, `--he-aac` — which sources may be decoded at all
  ([output-spec.md](output-spec.md#restricting-decoders--audio_decode_deny)).

Source: [`crates/codec/src/audio/filter/`](../crates/codec/src/audio/filter/).
