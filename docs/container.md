# The `container` crate

Clean-room demuxers (input) and muxers (output) for rivet — **no FFmpeg
dependency**. Every parser and writer in this crate is hand-rolled against the
relevant ISO / RFC / ETSI spec, so a default `rivet` build reads MP4 / MOV /
MKV / WebM / MPEG-TS / AVI and writes faststart MP4 or segmented CMAF/HLS
without linking a single line of libav. (FFmpeg is available as an *optional
decode backend* in the `codec` crate behind a feature flag — never for
containers.)

The crate sits at the two ends of the pipeline: **demux** turns container bytes
into codec-native video samples (Annex-B for H.264/HEVC, OBU for AV1) plus an
audio track, and **mux** packages encoded video + audio back into the output
container. The default output target is royalty-clean —
**AV1 video + Opus/AAC audio in MP4**, or the same in a **CMAF/HLS** package for
adaptive bitrate (ABR); **H.264 and H.265** output are also supported for
legacy-player compatibility. For how these pieces fit into the end-to-end job (demux →
decode-once pump → per-rung encode → mux), see
[the pipeline & architecture doc](pipeline.md); this document is the
container-crate companion — what each file does and *why*.

> Conventions in this doc: source links are relative to `docs/`
> (`../crates/container/src/...`); `file.rs:NN` cites a line. "Why (inferred)"
> marks a rationale not stated verbatim in the code.

---

## Module map

| File | Purpose |
|------|---------|
| [`lib.rs`](../crates/container/src/lib.rs) | Crate root + the shared `AudioInfo` mux-input type and `MkvColorInfo` / `MkvMasteringMetadata` extended-metadata carriers. |
| [`streaming.rs`](../crates/container/src/streaming.rs) | The `StreamingDemuxer` trait + `demux_streaming` magic-byte dispatch — one sample at a time, bounded peak RSS. |
| [`demux.rs`](../crates/container/src/demux.rs) | The materialize-all demuxers for MP4/MOV and MKV/WebM; container detection; audio extraction; ProRes fourcc routing; color-metadata plumbing. |
| [`ts.rs`](../crates/container/src/ts.rs) | MPEG-TS demux: PAT/PMT walk, PES reassembly, multi-program, AC-3/E-AC-3 audio, encrypted-stream guard, dimension + frame-rate recovery from the elementary stream. |
| [`avi.rs`](../crates/container/src/avi.rs) | AVI/RIFF demux + OpenDML 1.0 super-index for >1 GiB files. |
| [`annexb.rs`](../crates/container/src/annexb.rs) | AVCC/HVCC length-prefixed → Annex-B conversion + the `ParamSetTracker` that prepends SPS/PPS/VPS at the right sample. |
| [`mux.rs`](../crates/container/src/mux.rs) | The `Av1Mp4Muxer`: ISOBMFF box writers, faststart, audio interleave, co64/largesize auto-upgrade, Apple-compat `ftyp` brands, `colr`/`mdcv`/`clli` HDR atoms, `esds`/`mp4a`, Opus `dOps`, AC-3 `dac3` / E-AC-3 `dec3`. |
| [`cmaf.rs`](../crates/container/src/cmaf.rs) | Fragmented-MP4 / CMAF segment writers (`moof`/`mfhd`/`tfhd`/`tfdt`/`trun`), init segments, and the stateful `CmafVideoMuxer` / `CmafAudioMuxer`. |
| [`hls.rs`](../crates/container/src/hls.rs) | HLS playlist generation: `master.m3u8` + per-variant media playlist + shared audio rendition group. |
| [`aac_asc.rs`](../crates/container/src/aac_asc.rs) | AAC `AudioSpecificConfig` parse + implicit→explicit HE-AAC signaling rewrite. |
| [`ac3_sync.rs`](../crates/container/src/ac3_sync.rs) | AC-3 / E-AC-3 sync-frame / BSI parse → `dac3` / `dec3` config fields. |
| [`mp4_sanitize.rs`](../crates/container/src/mp4_sanitize.rs) | Lenient ISOBMFF box-size pre-pass so malformed files don't break the strict `mp4` crate. |

---

## Demuxers

rivet has **two demux surfaces over the same per-format parsers**: a
materialize-all path (`demux::demux` → a `DemuxResult` with `samples:
Vec<Vec<u8>>`) and a streaming path (`streaming::demux_streaming` → a
`Box<dyn StreamingDemuxer>` yielding one `Sample` at a time). The streaming path
is the one the production pipeline uses; the materialize-all path is retained as
a thin adapter (and for tests/benches).

### Streaming vs materialize-all

**What.** [`streaming::StreamingDemuxer`](../crates/container/src/streaming.rs:60)
is a pull-based trait: `header()` returns the parsed `DemuxHeader` (codec string
+ `StreamInfo`) immediately, `next_video_sample()` yields the next
[`Sample`](../crates/container/src/streaming.rs:51) (`Ok(None)` at EOF), and
`audio()` returns the one buffered audio track.
[`demux_streaming`](../crates/container/src/streaming.rs:80) magic-byte-detects
the container and dispatches to the per-format streaming reader (MP4, MKV, AVI,
TS).

**Why.** Per the module doc, the streaming shape "replaces the
materialize-everything-upfront `demux()` shape … nothing accumulates across
samples"
([`streaming.rs:1`](../crates/container/src/streaming.rs)). Peak heap from any
one `next_video_sample()` call is bounded by *that sample's size* plus the
reader's cursor state — not the whole file. For a 15-min 1080p60 source that is
the difference between a few MB and several GB of resident set. The decode pump
([`pipeline.md`](pipeline.md#1-demux)) consumes one sample, decodes it, and drops
it, so the demuxer never needs the whole stream in memory. The trait is `Send`
so the demuxer can live on the dedicated decode thread.

**Key types/functions.**
- [`DemuxHeader`](../crates/container/src/streaming.rs:29) — codec label + `StreamInfo`, available before any sample is pulled.
- [`Sample`](../crates/container/src/streaming.rs:51) — `data` (codec-native bitstream), `pts_ticks` (container timescale), `duration_ticks` (0 when the container records none — TS/AVI; the caller falls back to `1/frame_rate`).
- [`demux_streaming`](../crates/container/src/streaming.rs:80) + the module-private [`detect_container`](../crates/container/src/streaming.rs:94).
- Legacy adapter: [`demux::demux`](../crates/container/src/demux.rs:67) drains the iterator into a [`DemuxResult`](../crates/container/src/demux.rs:24).

**Notes / decisions.**
- `detect_container` is **deliberately duplicated** between `streaming.rs:94` and
  `demux.rs:80` rather than shared, "so the streaming dispatch doesn't reach into
  `demux::`'s private surface and so a future change to either path stays a
  one-file edit" ([`streaming.rs:90`](../crates/container/src/streaming.rs:90)).
  The two must stay in lock-step (both are tested to agree on every input).
- **Audio stays buffered** in both paths — it's a single slab populated at
  construction. Streaming audio was explicitly out of scope; passthrough audio
  is small relative to video, so the RSS win wasn't worth the complexity
  ([`streaming.rs:71`](../crates/container/src/streaming.rs:71)).

### MP4 / MOV (ISOBMFF)

**What.** [`demux_mp4`](../crates/container/src/demux.rs:111) /
[`demux_mp4_streaming_init`](../crates/container/src/demux.rs:3463) parse the
ISOBMFF box tree (via the `mp4` crate for the index, plus hand-written walks for
the bits the crate loses), pull the video track's samples, convert AVCC/HVCC
length-prefixed NALs to Annex-B, and surface the audio track. MOV shares the MP4
demuxer — same box tree — and `detect_container` returns `"mp4"` for `ftyp mp4*`,
`ftyp qt  `, and bare-`moov`/`mdat` MOVs alike
([`demux.rs:69`](../crates/container/src/demux.rs:69)).

**Why / decisions.**
- **ProRes fourcc routing.** [`prores_sample_entry_fourcc`](../crates/container/src/demux.rs:1972)
  byte-scans the `stsd` for the six Apple ProRes codes (`apco`/`apcs`/`apcn`/
  `apch`/`ap4h`/`ap4x`) and routes them to the unified `prores` codec label. This
  is a fallback used when the `mp4` crate reports `"unknown"`
  ([`demux.rs:146`](../crates/container/src/demux.rs:146)) — it recognises ProRes
  regardless of the strict crate's quirks.
- **Verbatim AAC ASC.** Audio extraction pulls the `AudioSpecificConfig` bytes
  straight out of the `esds` descriptor, *not* the `mp4` crate's rebuilt form, so
  HE-AAC / xHE-AAC signaling bits survive the copy
  ([`demux.rs:34`](../crates/container/src/demux.rs:34)).
- **Color metadata.** The demuxer reads the `colr` (and, on MKV,
  `MasteringMetadata` + MaxCLL/MaxFALL) so the source's
  `StreamInfo.color_metadata` carries primaries/transfer/matrix/range through to
  the muxer's HDR atoms.

### MKV / WebM (Matroska / EBML)

**What.** [`demux_mkv`](../crates/container/src/demux.rs:1366) /
[`demux_mkv_streaming_init`](../crates/container/src/demux.rs:3777) use the
`matroska-demuxer` crate for the cluster cursor and hand-rolled EBML walks for
the colour metadata the crate doesn't surface.
[`probe_mkv_color_info`](../crates/container/src/demux.rs:2736) returns the
extended [`MkvColorInfo`](../crates/container/src/lib.rs:161) (bits-per-channel,
chroma siting/subsampling, MaxCLL/MaxFALL, ST 2086 mastering chromaticities).

**Why (inferred).** The shared `StreamInfo` type in `codec` only carries the
core H.273-equivalent fields; `MkvColorInfo` /
[`MkvMasteringMetadata`](../crates/container/src/lib.rs:188) exist to carry the
*rest* "without requiring a breaking extension of the shared `StreamInfo` type"
([`lib.rs:148`](../crates/container/src/lib.rs:148)) — i.e. an additive carrier so
HDR signalling and future SEI passthrough have the data without an API churn
across crates. Both AAC and Opus audio carry through (`A_AAC`; `A_OPUS` →
`CodecPrivate` *is* the RFC 7845 OpusHead body, handed to the muxer verbatim).

### Shared audio-track shape

[`AudioTrack`](../crates/container/src/demux.rs:51) is the demuxer's output
contract: `codec`, `samples` (codec-native packets), `sample_rate`, `channels`,
`asc` (AAC only), `codec_private` (Opus/AC-3/E-AC-3), `timescale`, `durations`.
The muxer's input mirror is [`AudioInfo`](../crates/container/src/lib.rs:40) with
convenience constructors `aac_lc` / `opus` / `ac3` / `eac3`. Anything not in
{aac, opus, ac3, eac3} is rejected at `with_audio()` time — **no silent
degradation, no stubs** ([`lib.rs:42`](../crates/container/src/lib.rs:42)).

---

## MPEG-TS

**What.** [`demux_ts`](../crates/container/src/ts.rs:82) (materialize-all) and
the streaming init walk 188-byte (or 192-byte BDAV) TS packets, find the PAT
(PID 0), walk a PMT, pick the first video elementary stream, and reassemble PES
payloads into one sample per access unit. PTS is carried at the TS 90 kHz clock.

**Why TS is special.** Unlike MP4/MOV/MKV/AVI, **MPEG-TS has no container-level
track header** — there is no sample-entry box, no `BITMAPINFOHEADER`. Dimensions,
codec config, and timing all live *inside the elementary stream*. So the TS
demuxer has to do work the other demuxers get for free:

- **Dimension recovery.** [`detect_dims`](../crates/container/src/ts.rs:259)
  (from `codec::pixel_format`) parses the first sample's H.264/HEVC SPS or MPEG-2
  sequence header to recover `width`/`height`; on parse failure it falls back to
  `0` and logs a warn rather than fabricating a value
  ([`ts.rs:254`](../crates/container/src/ts.rs:254)).
- **Frame-rate inference.** [`estimate_frame_rate_from_ptses`](../crates/container/src/ts.rs:243)
  takes the **median of inter-PTS deltas** at 90 kHz. Why median and not
  `(samples-1)/duration`: the span-based calc was "off-by-one on boundary edge
  cases" ([`ts.rs:150`](../crates/container/src/ts.rs:150)) — median tolerates
  B-frame reorder and a stray boundary PTS without skewing. Both the streaming
  init and `demux_ts` share this path for consistency, with a span/count fallback
  and then a 30.0 last resort.

**Multi-program + audio (Squad-37).**
- The PAT walk surfaces *every* program with a default "first program" pick and a
  `select_program(program_number)` API for the others
  ([`ts.rs:11`](../crates/container/src/ts.rs:11)).
- Audio stream types: `0x0F` AAC-ADTS, `0x81` AC-3 (ATSC A/53), `0x87` E-AC-3
  (ATSC), and `0x06` PES-private *when* the ES descriptor loop carries a
  `registration_descriptor` tagged `"AC-3"` / `"EAC3"` (DVB / ETSI TS 101 154)
  ([`ts.rs:56`](../crates/container/src/ts.rs:56)). Random PES-private streams
  (DVB subtitles, teletext) are dropped silently.

**Encrypted-stream guard.** A scrambled packet (`transport_scrambling_control
!= 0`) on the active video PID trips a one-time typed warn and switches the
demuxer into a **drop-everything** mode
([`ts.rs:21`](../crates/container/src/ts.rs:21)). The rationale: previously the
bytes were skipped per-packet, which meant a *partial* scramble could still leak
garbled samples downstream. rivet doesn't carry CA (Conditional Access) tables,
so an encrypted stream can't be decrypted — dropping cleanly is the correct
behaviour.

**Not implemented (by decision):** PAT/PMT CRC validation (a mis-CRCed file is
already corrupt and surfaces downstream), multiple video streams per program
(first wins), CA descrambling.

---

## AVI / RIFF

**What.** [`demux_avi`](../crates/container/src/avi.rs:37) walks the RIFF tree:
`LIST hdrl` (→ `avih` + per-stream `LIST strl`) for the stream headers, and one
or more `LIST movi` for the sample chunks. It maps the stream handler/fourcc to a
codec label and emits per-frame samples in file (= display) order — AVI has no
container-layer B-frame reordering.

**Why OpenDML matters.** The classic AVI index (`avih.dwTotalFrames`, `idx1`
offsets) is **32-bit**, so it wraps for files past `2^32 / fps` frames — i.e.
anything over ~1 GiB / a couple hours. DivX/XviD muxers solve this with the
**OpenDML 1.0 super-index**: the file is split every ~1 GiB into a fresh
`RIFF AVIX` segment, each with its own `LIST movi`, indexed by an `indx`
super-index chunk that points at per-segment `ix##` standard indexes, and the
true frame count lives in `dmlh.dwTotalFrames` (a 64-bit-safe field in the
`LIST odml`) ([`avi.rs:11`](../crates/container/src/avi.rs:11)).

**Decisions.**
- Detection is at construction: presence of an `indx` chunk in the video
  stream's `strl` triggers the OpenDML precomputed-offset path; its absence falls
  back to the legacy single-`movi` cursor walk.
- `dmlh.dwTotalFrames` **supersedes** `avih.dwTotalFrames` for OpenDML files
  precisely because `avih` may have wrapped
  ([`avi.rs:105`](../crates/container/src/avi.rs:105)).
- The whole file is scanned for every `LIST movi` regardless of which RIFF
  segment it lives in ([`avi.rs:55`](../crates/container/src/avi.rs:55)).
- Out of scope (stated): AVI audio passthrough (usually MP3/AC-3, not AAC) and
  VBR index reconstruction — it trusts the `movi` sample order.

---

## Annex-B conversion

**What.** [`annexb.rs`](../crates/container/src/annexb.rs) converts the
length-prefixed NAL units that MP4 and MKV store (with parameter sets out-of-band
in an `avcC` / `hvcC` config box) into the **Annex-B** form decoders expect:
`00 00 00 01` start codes between NALs, with VPS/SPS/PPS prepended to the right
sample. [`parse_avcc`](../crates/container/src/annexb.rs:46) /
[`parse_hvcc`](../crates/container/src/annexb.rs:118) parse the config records;
[`length_prefixed_to_annexb_tracked`](../crates/container/src/annexb.rs:291) does
the per-sample conversion.

**Why a length-size field, not just 4 bytes.** The config record's
`lengthSizeMinusOne` can be 0/1/3 → 1/2/4 byte prefixes. Real MP4 streaming
profiles use length_size=2, so the recorded value is honored rather than
assumed ([`annexb.rs:9`](../crates/container/src/annexb.rs:9)).

**Why `ParamSetTracker` (and why ExoPlayer needs it).**
[`ParamSetTracker`](../crates/container/src/annexb.rs:181) is a per-stream state
machine that prepends only the parameter sets that haven't been emitted yet, *on
the first IRAP that lacks them*. It replaces an older
`prepend-on-sample-index==1` heuristic that broke two real cases
([`annexb.rs:166`](../crates/container/src/annexb.rs:166)):

1. **ExoPlayer open-GOP MP4** (#67/#68): sample 0 is SPS-only with a *non-IDR*
   slice. The decoder can't start mid-GOP without parameter sets at the next
   IRAP — but that IRAP carries only a slice NAL, so the stream stalls. The
   tracker prepends on the first IRAP that's missing parameter sets.
2. **avcC has SPS but PPS arrives inline late** — the tracker watches inline NAL
   types and prepends only the missing kind(s).

The fix is subtle: blindly prepending avcC SPS+PPS on sample 0 produced
`SPS PPS SPS slice`, and the decoder may discard the redundant second SPS and
try to start the GOP at a non-IDR slice, which fails
([`annexb.rs:275`](../crates/container/src/annexb.rs:275)). State is **per-stream**
(one tracker per `samples` iteration); sharing across streams would conflate
emission state. Both `demux_mp4` and `demux_mkv` use the tracked helper.

---

## The AV1 MP4 muxer

[`Av1Mp4Muxer`](../crates/container/src/mux.rs:37) is the single-file output
path: AV1 (default), H.264, or H.265 video + optional audio → one faststart MP4. It is the only mux output
besides CMAF/HLS, and it is where most of the crate's spec-conformance and
device-compat work lives.

### Spooled, RAM-bounded, faststart

**What.** The muxer streams the `mdat` payload to a tempfile while keeping only
small per-packet metadata (sizes, keyframe indices) in RAM
([`mux.rs:12`](../crates/container/src/mux.rs:12)).
[`finalize_to_file`](../crates/container/src/mux.rs:542) writes `ftyp` + `moov`
*first*, then streams the tempfile's `mdat` bytes into the output.

**Why.** Two goals at once. **Faststart** (moov before mdat) lets a player begin
playback after a short prefix download instead of seeking to the end for the
index — required for web playback. **Bounded RSS**: at 15-min 1080p60 the packet
metadata is ~700 KB while the actual payload (~500 MB/variant) never leaves disk
([`mux.rs:13`](../crates/container/src/mux.rs:13)). The two compose because the
`moov` (which references sample offsets) is computed from the cheap metadata, and
the bulky `mdat` is appended afterward.

### `co64` and `mdat largesize` auto-upgrade — handling >4 GiB

**What.** The muxer picks 64-bit forms automatically when sizes demand it:
- `use_co64` switches the chunk-offset table from `stco` (32-bit) to `co64`
  (64-bit) when the upper-bound file size exceeds `u32::MAX`
  ([`mux.rs:702`](../crates/container/src/mux.rs:702)).
- `use_largesize_mdat` switches the `mdat` header from the 8-byte short form to
  the ISOBMFF §4.2 16-byte `largesize` form (`size=1` sentinel + `'mdat'` +
  64-bit length) when payload + 8 would exceed `u32::MAX`
  ([`mux.rs:657`](../crates/container/src/mux.rs:657)).

**Why / gotcha.** A >4 GiB output can't address its samples with 32-bit offsets,
and an `mdat` over 4 GiB can't state its own size in the 32-bit field — both are
hard correctness failures for large transcodes. The subtlety is that the
largesize header grows 8 → 16 bytes, which shifts the first-sample file offset,
so **the `stco`/`co64` chunk offsets must account for the 16-byte header**
([`mux.rs:648`](../crates/container/src/mux.rs:648)); the two upgrades are
computed together. A `#[doc(hidden)]`
[`force_largesize_mdat_for_test`](../crates/container/src/mux.rs:137) exercises
the bit layout without crafting a 4 GiB tempfile — and it's a *regular* field,
not `#[cfg(test)]`-gated, so integration tests in `tests/` (which compile against
the release library) can flip it.

### Apple-compatible `ftyp` brands

**What.** [`build_ftyp`](../crates/container/src/mux.rs:1003) emits
`major_brand=iso6`, `minor_version=512`, and compatible brands `iso6` / `iso2` /
`av01` / `mp41` / `mp42`.

**Why each brand.**
- `av01` is **REQUIRED** by AV1-ISOBMFF v1.3.0 §2.1 — an AV1-bearing file SHALL
  list it ([`mux.rs:990`](../crates/container/src/mux.rs:990)).
- `iso6` (14496-12 6th ed.) covers `co64` / `mehd` v1 / largesize semantics —
  Apple's stack wants a structural ISOBMFF brand, and `major_brand=iso6` keeps a
  strict parser from rejecting a co64-bearing file that claims an older major
  brand like `mp41` (which predates co64) ([`mux.rs:1000`](../crates/container/src/mux.rs:1000)).
- `iso2` / `mp41` / `mp42` keep legacy parsers and AAC-parsing-rule players
  happy.

### `colr` / `mdcv` / `clli` HDR atoms

**What.** [`build_av01`](../crates/container/src/mux.rs:2050) builds the `av01`
visual sample entry with children, in spec order:
[`av1C`](../crates/container/src/mux.rs:2038) →
[`colr` (nclx)](../crates/container/src/mux.rs:2127) →
[`mdcv`](../crates/container/src/mux.rs:2170) →
[`clli`](../crates/container/src/mux.rs:2199).

**Why.**
- **`colr nclx`** carries primaries / transfer / matrix / full-range. Apple's
  QuickTime / iOS Safari **silently assume BT.709 limited-range when `colr` is
  absent**, which corrupts BT.2020 / HDR / wide-gamut clips
  ([`mux.rs:143`](../crates/container/src/mux.rs:143)). The default
  `ColorMetadata` is BT.709 SDR limited — correct for SDR — and real values
  arrive via `with_color`. `nclx` (not `nclc`/`rICC`/`prof`) is the right colour
  type for video distribution ([`mux.rs:2124`](../crates/container/src/mux.rs:2124)).
  Transfer functions map to H.273 codes via `transfer_to_h273` (PQ/ST2084 → 16,
  HLG/AribStdB67 → 18 — [`mux.rs:2106`](../crates/container/src/mux.rs:2106)).
- **`mdcv`** (Mastering Display Color Volume, ST 2086) and **`clli`** (Content
  Light Level, MaxCLL/MaxFALL) are emitted *only* when the source declared them
  (`ColorMetadata.mastering_display` / `.content_light_level` are `Some`). Per
  AV1-ISOBMFF v1.3.0 §2.3.4/§2.3.5 the order is `colr → mdcv → clli`; players
  scan by 4cc so order is recommended-not-load-bearing, but the muxer matches the
  spec anyway ([`mux.rs:2043`](../crates/container/src/mux.rs:2043)).

> Note the default rivet color policy **tonemaps HDR → 8-bit SDR BT.709**
> ([pipeline.md §6](pipeline.md#6-color--bit-depth)), so these HDR atoms are
> written when an HDR-preserving policy (`Hdr10`/`Hlg`/`Passthrough`) is selected
> and a 10-bit encoder is in the build.

### Audio interleave + per-codec sample entries

**What.** With audio present, `finalize_to_file` writes an **interleaved** `mdat`
that alternates ~1-second video and audio chunks, with each track's `stco`/`co64`
pointing at its chunk's first sample
([`mux.rs:537`](../crates/container/src/mux.rs:537)). The audio sample entry is
chosen by codec:

| Codec | Sample entry | Config box | Source |
|-------|--------------|-----------|--------|
| AAC-LC | `mp4a` | `esds` (ASC verbatim) + Apple `chan` for ≥3ch | [`build_audio_stsd`](../crates/container/src/mux.rs:1397) |
| Opus | `Opus` (capital O, RFC 7845 §4.4) | `dOps` (OpusHead body, LE→BE) | [`lib.rs:21`](../crates/container/src/lib.rs:21) |
| AC-3 | `ac-3` | `dac3` ([`dac3_body_from_sync`](../crates/container/src/mux.rs:1739)) | ETSI TS 102 366 §F.4 |
| E-AC-3 | `ec-3` | `dec3` ([`dec3_body_from_sync`](../crates/container/src/mux.rs:1761)) | §F.6 |

**Why ~1-second interleave (inferred).** Coarse interleave keeps both tracks
locally available to a player without forcing large read-ahead; finer
interleave bloats the chunk tables, coarser starves one track. **Why verbatim
config bytes:** the ASC / OpusHead / dac3 / dec3 payloads are passed through
untouched so the exact codec signalling (HE-AAC layers, Opus pre-skip,
Dolby BSI) survives — re-synthesising them risks losing bits Apple players
require.

---

## CMAF / HLS for ABR

The HLS output mode produces a **CMAF** package — fragmented MP4 broken into
segment-aligned chunks across the ladder — plus the HLS playlists that point at
it. See [pipeline.md §5](pipeline.md#5-output-modes) for how the multi-GPU engine
drives it.

### Fragmented-MP4 / CMAF writers

**What.** [`cmaf.rs`](../crates/container/src/cmaf.rs) writes the ISO 14496-12
§8.8 movie-fragment boxes (`moof`/`mfhd`/`traf`/`tfhd`/`tfdt`/`trun`) plus the
`mvex`/`mehd`/`trex` declarations that go in a CMAF init segment's `moov`. The
stateful segmenters are
[`CmafVideoMuxer`](../crates/container/src/cmaf.rs:1040) and
[`CmafAudioMuxer`](../crates/container/src/cmaf.rs:1366); each emits an `init.mp4`
plus `seg-NNNNN.m4s` files and a [`CmafTrackManifest`](../crates/container/src/cmaf.rs:997)
describing them.

**Why CMAF specifically.** CMAF (ISO 23000-19) constrains the general fragmented
model — exactly one track per fragment, one track per init segment, a small
mandatory box set ([`cmaf.rs:5`](../crates/container/src/cmaf.rs:5)). That
constraint is what lets hls.js / Safari do clean ABR: the renditions are
segment-aligned, so a player can switch bitrate at any segment boundary. Init
segments declare the CMAF brand (`cmfc` video / `cmfa` audio) alongside `iso6` /
`mp42` / `av01` so non-CMAF tools can still demux the boxes
([`cmaf.rs:19`](../crates/container/src/cmaf.rs:19)).

**Decisions / gotchas.**
- **`SampleFlags` packing** ([`cmaf.rs:89`](../crates/container/src/cmaf.rs:89))
  encodes per-sample sync/dependency bits per §8.8.3.1: a sync sample is
  `depends_on=2, non_sync=0`; a non-key sample is `depends_on=1, non_sync=1`. A
  helper packs the u32 so callers don't compose it by hand — getting it wrong
  makes a player treat every frame as a keyframe or vice-versa.
- The split into a **box-primitive layer** + higher-level segment composers
  exists so each box's byte layout can be unit-tested against the spec without
  driving a full encode ([`cmaf.rs:10`](../crates/container/src/cmaf.rs:10)).
- **Multi-GPU helper support.**
  [`CmafVideoMuxerOptions`](../crates/container/src/cmaf.rs:1066) lets a helper
  muxer start at a non-1 `first_segment_index` with the matching
  `first_segment_base_decode_time`, and skip writing `init.mp4`
  (`write_init_segment=false`), so segments produced on *different GPUs* for the
  same rung have byte-identical `tfdt` and filenames to a single-encoder run
  ([`cmaf.rs:1060`](../crates/container/src/cmaf.rs:1060)). This is what makes the
  reactive lease engine's cross-vendor helper dispatch safe at the container
  layer.
- The first AV1 packet's OBU stream MUST contain a sequence header; the muxer
  extracts it for `av1C` in the init segment, written lazily on first
  `flush_segment` ([`cmaf.rs:1029`](../crates/container/src/cmaf.rs:1029)).

### HLS playlists

**What.** [`write_hls_package`](../crates/container/src/hls.rs:139) emits a
`master.m3u8` (one `#EXT-X-STREAM-INF` per video rendition + one
`#EXT-X-MEDIA:TYPE=AUDIO` rendition-group entry), a per-rendition
`playlist.m3u8` (with `#EXT-X-MAP` → `init.mp4` and `#EXTINF` → `seg-*.m4s`), and
the shared `audio.m3u8`. Targets HLS protocol version 7 — the minimum that
supports `EXT-X-MAP` (fMP4 init) and `EXT-X-INDEPENDENT-SEGMENTS`
([`hls.rs:14`](../crates/container/src/hls.rs:14)).

**Why a shared audio rendition group.** Separating audio into its own rendition
group lets video variants switch bitrate *without re-downloading audio* — the ABR
win. The video variants are described by
[`VideoVariantSpec`](../crates/container/src/hls.rs:34).

**Gotcha — codec strings are load-bearing.** The `CODECS=` attribute MUST be
parsed from the *actual encoded bitstream* (via
`codec::codec_strings::av1_codec_string`), not composed from config — "a wrong
string causes hls.js / Safari to silently skip the variant"
([`hls.rs:19`](../crates/container/src/hls.rs:19)). `VIDEO-RANGE` is `PQ`/`HLG`
for HDR and omitted (not `=SDR`) for SDR, per HLS authoring guidance
([`hls.rs:73`](../crates/container/src/hls.rs:73)).

---

## Audio container glue

Three small, decoder-free modules turn raw audio config bytes into the
container-level boxes the muxer needs.

### AAC `AudioSpecificConfig`

**What.** [`parse_aac_asc`](../crates/container/src/aac_asc.rs:156) parses a
2..16-byte ASC into `{aot, sample_rate, channels, sbr_present, ps_present,
sbr_sample_rate, signaling}`. [`effective_output_channels`](../crates/container/src/aac_asc.rs:224)
applies the HE-AAC v2 Parametric Stereo upmix (1-ch core → 2-ch output).
[`upgrade_to_explicit_signaling`](../crates/container/src/aac_asc.rs:256)
rewrites an implicitly-signaled HE-AAC ASC into explicit form.

**Why explicit signaling matters.** With **implicit** signaling the ASC says only
`AOT=2` (LC) even though the bitstream carries SBR/PS — and **Apple Core
Audio / AVFoundation silently downgrade implicit HE-AAC to mono 22.05 kHz core**,
so listeners hear quiet, muffled audio
([`aac_asc.rs:17`](../crates/container/src/aac_asc.rs:17)). The explicit form
(leading `AOT=5` SBR + extension sample rate + inner `AOT=2`) is what Apple
players require to honour full HE-AAC output. The muxer **rejects** an
implicitly-signaled HE-AAC ASC rather than mux something Apple will silently
degrade ([`mux.rs:287`](../crates/container/src/mux.rs:287)).

**Scope.** No PCE (Programme Config Element) parsing — `channelConfiguration=0`
falls back to a sane default; only the GASpecificConfig prefix needed to locate
the SBR trailer is read ([`aac_asc.rs:48`](../crates/container/src/aac_asc.rs:48)).

### AC-3 / E-AC-3 sync parse

**What.** [`parse_sync_info`](../crates/container/src/ac3_sync.rs:113) walks the
AC-3 / E-AC-3 syncframe BSI (0x0B77 syncword) far enough to populate the MP4
config fields — [`Ac3SyncInfo`](../crates/container/src/ac3_sync.rs:26) /
[`Eac3SyncInfo`](../crates/container/src/ac3_sync.rs:52) — then helpers
([`channel_count`](../crates/container/src/ac3_sync.rs:262),
[`ac3_bit_rate_kbps`](../crates/container/src/ac3_sync.rs:280),
[`eac3_sample_rate_hz`](../crates/container/src/ac3_sync.rs:320)) derive the
`dac3` / `dec3` box body bytes (built in `mux.rs`).

**Why decoder-free.** Per task notes, "Do NOT introduce a Dolby decoder"
([`ac3_sync.rs:12`](../crates/container/src/ac3_sync.rs:12)) — AC-3/E-AC-3 audio
is **passthrough only**, so rivet parses just the BSI header to synthesise the
sample-entry config and copies the frames verbatim. No coefficient parsing, no
licensing exposure.

**Scope.** E-AC-3 extraction is the independent-substream subset (vanilla 5.1) —
dependent-substream fields are deferred as the dominant-case-first decision
([`ac3_sync.rs:47`](../crates/container/src/ac3_sync.rs:47)).

### Opus `dOps`

Opus needs no separate module: the demuxer surfaces the RFC 7845 OpusHead body
verbatim (MKV/WebM `CodecPrivate` *is* that body), and the muxer's `build_dops`
converts the LE OpusHead numeric fields to the BE ISOBMFF `dOps` convention and
pins the `mdhd` timescale to 48000 (Opus is internally always 48 kHz —
[`lib.rs:101`](../crates/container/src/lib.rs:101)).

---

## ISOBMFF box-size sanitizer

**What.** [`sanitize_isobmff_box_sizes`](../crates/container/src/mp4_sanitize.rs:110)
is a lenient pre-pass run before the strict `mp4` crate. It walks the box tree;
any time a child's advertised `size` exceeds the parent's remaining payload, it
rewrites the child's `size` to fit ([`mp4_sanitize.rs:1`](../crates/container/src/mp4_sanitize.rs)).

**Why.** Malformed encoders (older Apple QuickTime, some prosumer cameras, buggy
muxers) emit child boxes whose advertised size overruns the parent. The `mp4
0.14` crate (and most strict parsers) bail with
*"box contains a box with a larger size than it"* and the whole demux fails. The
sanitizer makes those files parseable while staying **byte-identical on every
well-formed file** — a clean MP4 hashes the same through it, only malformed files
mutate ([`mp4_sanitize.rs:18`](../crates/container/src/mp4_sanitize.rs:18)).

**Gotchas.** It only touches *header* bytes — leaf-payload corruption (e.g. a
malformed `esds`) is opaque to it. The `CONTAINER_FOURCCS` set
([`mp4_sanitize.rs:43`](../crates/container/src/mp4_sanitize.rs:43)) lists every
box the strict parser recurses into (including the visual/audio sample entries
that carry child boxes); extending the sanitizer's reach means adding to that set
when a future crate version recurses further. `size=0` ("extends to EOF") is left
untouched — strict parsers handle it correctly.

---

## Key decisions in the container crate

- **No FFmpeg for containers.** Every demuxer and muxer is hand-written against
  the spec, so the default build links no libav. This keeps the output narrow,
  predictable, and royalty-clean.
- **Streaming demux for bounded RSS.** One sample at a time; nothing accumulates
  across samples, so peak heap is a sample, not a file. Audio stays buffered (it's
  small).
- **AV1 + Opus/AAC in MP4 (or CMAF/HLS) is the default output; H.264/H.265 are
  also supported.** AV1 is the royalty-clean default; the muxer emits
  `av01`/`av1C` for AV1, `avc1`/`avc3` + `avcC` for H.264, and `hvc1`/`hev1` +
  `hvcC` for H.265 (legacy-player compatibility, at the cost of their
  patent-licensing obligations), with the `ftyp` brands, `colr`/HDR atoms, and
  faststart layout tuned to *just play* in browsers and on Apple devices.
- **Verbatim audio config bytes.** AAC ASC, Opus OpusHead, AC-3 `dac3`, E-AC-3
  `dec3` are passed through untouched so codec signalling survives passthrough.
  AC-3/E-AC-3 are parsed header-only (no Dolby decoder) — passthrough only.
- **`ParamSetTracker` over a sample-index heuristic** so ExoPlayer open-GOP MP4
  and late-inline-PPS streams start cleanly.
- **Automatic 64-bit upgrades** (`co64` + `mdat largesize`) so >4 GiB transcodes
  stay correct, with the chunk offsets computed against the grown header.
- **Apple-compat is explicit, not incidental.** `av01`+`iso6` brands, `colr
  nclx`, the `chan` box for multichannel AAC, explicit HE-AAC signaling, the
  capital-O `Opus` 4cc, and the box-size sanitizer all exist because a specific
  Apple/strict-parser behaviour breaks otherwise.
- **CMAF helper options** (`first_segment_index` / base decode time /
  `write_init_segment`) make cross-GPU, cross-vendor segment production produce a
  byte-identical package to a single-encoder run.
