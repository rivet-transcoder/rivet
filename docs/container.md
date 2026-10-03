# The `container` crate

Clean-room demuxers (input) and muxers (output) for rivet — **no FFmpeg
dependency**. Every parser and writer in this crate is hand-rolled against the
relevant ISO / RFC / ETSI spec, so every `rivet` build reads MP4 / MOV /
MKV / WebM / MPEG-TS / MPEG-PS / AVI and writes faststart MP4, QuickTime
movies, WebM or segmented CMAF/HLS without linking a single line of libav. That holds for the whole workspace,
in every build, not just this crate — see [No FFmpeg](../README.md#no-ffmpeg).

The crate sits at the two ends of the pipeline: **demux** turns container bytes
into codec-native video samples (Annex-B for H.264/HEVC, OBU for AV1) plus an
audio track, and **mux** packages encoded video + audio back into the output
container. The default output target is royalty-clean —
**AV1 video + Opus/AAC audio in MP4**, or the same in a **CMAF/HLS** package for
adaptive bitrate (ABR); **H.264 and H.265** output are also supported for
legacy-player compatibility, and VP9 / VP8 (WebM, MP4; VP9 in CMAF too),
MPEG-2 / MPEG-4 Part 2 (MP4, QuickTime) and ProRes (QuickTime) — see
[The other codecs](#the-other-codecs-sample-entries-quicktime-and-webm). For how these pieces fit into the end-to-end job (demux →
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
| [`sniff.rs`](../crates/container/src/sniff.rs) | `sniff_container` → `ContainerKind` (ISOBMFF, Matroska, AVI, MPEG-TS, MPEG-PS, native FLAC, bare MP3): the one magic-byte detector every dispatch reads. |
| [`ps.rs`](../crates/container/src/ps.rs) | MPEG program stream demux (`.mpg` / `.vob`): MPEG-1 system and MPEG-2 program streams, the first video (MPEG-1 / MPEG-2) and the first MPEG audio or DVD AC-3 sub-stream. |
| [`webm.rs`](../crates/container/src/webm.rs) | The WebM muxer: VP8 / VP9 with Opus or Vorbis audio, one Matroska file built in memory (SeekHead, Info, Tracks, a Cluster per key frame, Cues). |
| [`ogg.rs`](../crates/container/src/ogg.rs) | Ogg Opus (RFC 7845) and Ogg Vorbis files, read and written for audio-only output and input: the codec mappings over rivet-vorbis's RFC 3533 page reader and writer, granule positions as the presentation edit. |
| [`vpx.rs`](../crates/container/src/vpx.rs) | VP8 / VP9 frame headers, the `vpcC` record, the VP9 level, the `vp09.…` codecs string. |
| [`mpeg_es.rs`](../crates/container/src/mpeg_es.rs) | MPEG-1/2 and MPEG-4 Part 2 elementary streams: start codes, configuration headers, one access unit per picture. |
| [`streaming.rs`](../crates/container/src/streaming.rs) | The `StreamingDemuxer` trait + `demux_streaming` dispatch — one sample at a time, bounded peak RSS — and `demux_audio`, the audio of any input with or without video. |
| [`demux/`](../crates/container/src/demux/mod.rs) | The MP4/MOV (`mp4/`) and MKV/WebM (`mkv/`) demuxers, materialize-all and streaming; audio extraction for every container (`audio/`); colour and HDR metadata (`hdr.rs`); sample aspect ratio (`aspect.rs`); text subtitles. |
| [`ts/`](../crates/container/src/ts/mod.rs) | MPEG-TS demux: PAT/PMT walk, PES reassembly, multi-program, AAC / MPEG audio / AC-3 / E-AC-3 audio, encrypted-stream guard, dimension + frame-rate recovery from the elementary stream, the program clock and discontinuities. |
| [`avi/`](../crates/container/src/avi/mod.rs) | AVI/RIFF demux + OpenDML 1.0 super-index for >1 GiB files, dropped-frame pacing, the first audio stream. |
| [`annexb.rs`](../crates/container/src/annexb.rs) | AVCC/HVCC length-prefixed → Annex-B conversion + the `ParamSetTracker` that prepends SPS/PPS/VPS at the right sample. |
| [`edit.rs`](../crates/container/src/edit.rs) | Presentation edits: `VideoPresentation`, `AudioEdit`, `AudioGap`, the per-codec `AudioPreroll` and the audio packet cut. |
| [`mux/`](../crates/container/src/mux/mod.rs) | The `Av1Mp4Muxer`: ISOBMFF box writers (`boxes.rs`, `video_track.rs`, `audio_track.rs`, `sample_table.rs`), faststart, audio interleave, co64/largesize auto-upgrade, Apple-compat `ftyp` brands, `colr`/`mdcv`/`clli` HDR atoms, `esds`/`mp4a`, Opus `dOps`, AC-3 `dac3` / E-AC-3 `dec3`, DTS `ddts`; FLAC / ALAC entries and the `.m4a` / native `.flac` writers (`lossless.rs`). |
| [`nal_mux.rs`](../crates/container/src/nal_mux.rs) | H.264 / H.265 sample writing: parameter sets in or out of band, and whether a stream's sets stay fixed (`parameter_sets_fixed`). |
| [`reorder.rs`](../crates/container/src/reorder.rs) | Composition offsets (`ctts` / `trun`) from presentation timestamps. |
| [`cmaf/`](../crates/container/src/cmaf/mod.rs) | Fragmented-MP4 / CMAF segment writers (`moof`/`mfhd`/`tfhd`/`tfdt`/`trun`), init segments, the stateful `CmafVideoMuxer` / `CmafAudioMuxer`, and `settle_video_sample_entry`. |
| [`hls.rs`](../crates/container/src/hls.rs) | HLS playlist generation: `master.m3u8` + per-variant media playlist + shared audio rendition group. |
| [`webvtt.rs`](../crates/container/src/webvtt.rs) | Segmented WebVTT for HLS subtitle renditions. |
| [`aac_asc.rs`](../crates/container/src/aac_asc.rs) | AAC `AudioSpecificConfig` parse + implicit→explicit HE-AAC signaling rewrite. |
| [`ac3_sync.rs`](../crates/container/src/ac3_sync.rs) | AC-3 / E-AC-3 sync-frame / BSI parse → `dac3` / `dec3` config fields. |
| [`dts_sync.rs`](../crates/container/src/dts_sync.rs) | DTS core frame header parse → the `ddts` body. |
| [`mp3.rs`](../crates/container/src/mp3.rs) | MPEG audio frame headers, the frame walk, and the bare `.mp3` file read and written (Xing / Info, the LAME-style gapless extension). |
| [`metadata/`](../crates/container/src/metadata/mod.rs) | Identifying metadata (location, device, capture time, descriptive tags): read from any input, a kept subset written into an output, encoder names cleared from copied audio. |
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

**What.** [`streaming::StreamingDemuxer`](../crates/container/src/streaming.rs#L136)
is a pull-based trait: `header()` returns the parsed `DemuxHeader` (codec string
+ `StreamInfo`) immediately, `next_video_sample()` yields the next
[`Sample`](../crates/container/src/streaming.rs#L127) (`Ok(None)` at EOF), and
`audio()` returns the one buffered audio track.
[`demux_streaming`](../crates/container/src/streaming.rs#L205) magic-byte-detects
the container and dispatches to the per-format streaming reader (MP4, MKV, AVI,
TS); a native FLAC or bare MP3 input has no video and is refused there, to be
read by [`demux_audio`](../crates/container/src/streaming.rs#L249), which gives
the audio of any input, with or without video (the audio-only output mode).

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
- [`DemuxHeader`](../crates/container/src/streaming.rs#L29) — codec label + `StreamInfo`, the `timescale` of the samples' ticks, the container's `rotation_degrees`, and the `sample_aspect` (see [below](#sample-aspect-ratio)), available before any sample is pulled.
- [`Sample`](../crates/container/src/streaming.rs#L127) — `data` (codec-native bitstream), `pts_ticks` (container timescale), `duration_ticks` (0 when the container records none — TS/AVI; the caller falls back to `1/frame_rate`).
- [`demux_streaming`](../crates/container/src/streaming.rs#L205) (and `demux_streaming_shared`, over a refcounted `Bytes` so several demuxers of one input share one buffer) + the module-private [`detect_container`](../crates/container/src/streaming.rs#L302).
- Legacy adapter: [`demux::demux`](../crates/container/src/demux/mod.rs#L103) returns a [`DemuxResult`](../crates/container/src/demux/mod.rs#L45).

**Notes / decisions.**
- Both paths' `detect_container` (`streaming.rs:302`, `demux/mod.rs:117`) are
  [`sniff_container`](../crates/container/src/sniff.rs#L67)'s label, so no two
  dispatches can disagree about a file. ISOBMFF is a `ftyp`, `moov` or `mdat`
  box first; native FLAC (its `fLaC` marker, possibly after an ID3v2 tag) is
  tried before the MP3 sniff, which needs two frame headers that agree.
- **Audio stays buffered** in both paths — it's a single slab populated at
  construction. Streaming audio was explicitly out of scope; passthrough audio
  is small relative to video, so the RSS win wasn't worth the complexity
  ([`streaming.rs:147`](../crates/container/src/streaming.rs#L147)).

### MP4 / MOV (ISOBMFF)

**What.** [`demux_mp4`](../crates/container/src/demux/mp4/mod.rs#L58) /
[`demux_mp4_streaming_init`](../crates/container/src/demux/mp4/streaming.rs#L106) parse the
ISOBMFF box tree (via the `mp4` crate for the index, plus hand-written walks for
the bits the crate loses), pull the video track's samples, convert AVCC/HVCC
length-prefixed NALs to Annex-B, and surface the audio track. MOV shares the MP4
demuxer — same box tree — and `detect_container` returns `"mp4"` for `ftyp mp4*`,
`ftyp qt  `, and bare-`moov`/`mdat` MOVs alike
([`demux/mod.rs:103`](../crates/container/src/demux/mod.rs#L103)).

**Why / decisions.**
- **ProRes fourcc routing.** [`prores_sample_entry_fourcc`](../crates/container/src/demux/mp4/sample_entry.rs#L90)
  byte-scans the `stsd` for the six Apple ProRes codes (`apco`/`apcs`/`apcn`/
  `apch`/`ap4h`/`ap4x`) and routes them to the unified `prores` codec label. This
  is a fallback used when the `mp4` crate reports `"unknown"`
  ([`demux/mp4/mod.rs:93`](../crates/container/src/demux/mp4/mod.rs#L93)) — it recognises ProRes
  regardless of the strict crate's quirks.
- **Verbatim AAC ASC.** Audio extraction pulls the `AudioSpecificConfig` bytes
  straight out of the `esds` descriptor, *not* the `mp4` crate's rebuilt form, so
  HE-AAC / xHE-AAC signaling bits survive the copy
  ([`demux/audio/aac.rs`](../crates/container/src/demux/audio/aac.rs)).
- **Color metadata.** The demuxer reads the `colr` box (`nclx` / `nclc`:
  primaries / transfer / matrix / range) and the `mdcv` / `clli` HDR atoms, and
  when the file has no `colr` — ffmpeg's MP4 muxer writes none unless asked with
  `-movflags +write_colr` — it falls back to the H.264 / HEVC SPS VUI
  `colour_description` in the avcC / hvcC parameter sets
  ([`demux/hdr.rs`](../crates/container/src/demux/hdr.rs)). So `StreamInfo.color_metadata`
  carries the real transfer and an HDR MP4 is tonemapped exactly like the same
  clip remuxed to MKV (whose `Colour` element the MKV demuxer reads). Until
  2026-08-27 only `mdcv` / `clli` were read and every MP4 kept the SDR default
  transfer, so HDR MP4s went through untouched under an SDR tag.
  The bitstream fills what the container leaves unsaid, field by field, for
  AV1, VP9 and MPEG-2 too ([`demux/hdr.rs`](../crates/container/src/demux/hdr.rs)
  `header_colour`): the AV1 sequence header's `color_config` and its HDR10
  metadata OBUs (`METADATA_TYPE_HDR_MDCV` / `_HDR_CLL`, into the same fields
  SEI 137 / 144 fill, so they reach the output's `mdcv` / `clli` and SEIs), a
  VP9 keyframe's `color_space` and `color_range`, and the MPEG-2
  `sequence_display_extension()`.
  A stream that states no matrix anywhere (or states `2`) is
  BT.601 when its picture is standard definition — narrower than 1280 and at
  most 576 lines, libplacebo's and DXVA2's line — and BT.709 otherwise
  (`demux::hdr::default_unstated_sd_colour`; the table is in
  [output-spec.md](output-spec.md#a-source-that-states-no-matrix)).

### Edit lists (`edts` / `elst`)

An MP4 track's edit list says which of its samples are presented and when
(ISO/IEC 14496-12 §8.6.6). Three shapes are everywhere, and a transcode that
ignores them gets the start of the output wrong:

| Shape | Made by | Ignored, it gives |
|---|---|---|
| media edit with `media_time` past the first frame | `ffmpeg -ss T -i in.mp4 -c copy` (keeps the GOP before `T`, hides it) | the hidden frames at the start, the timeline shifted |
| audio media edit of 1024 / 2048 samples | every AAC encoder (priming) | audio late by ~21–43 ms |
| empty edit (`media_time = -1`) then media | `-itsoffset`, audio recorded after video | the late start lost, A/V offset by the delay |

[`demux::mp4::edit_list`](../crates/container/src/demux/mp4/edit_list.rs)
parses `elst` from the box bytes and reduces it to one media edit, optionally
after one empty edit. Anything else is refused by name: a rate other than 1
(slow motion, a dwell), a gap in the middle, two media segments, or no media
segment at all.
The streaming demuxer exposes the result as
`StreamingDemuxer::video_presentation()` and `::audio_edit()`
([`container::edit`](../crates/container/src/edit.rs)).

- **Video.** A decoder emits frames in display order, so an edit that starts
  at media time `t` hides the frames presented before `t`, and one that ends
  at `e` stops before the frames from `e` on. `info.total_frames` and
  `duration` then count the presented frames. The decode pump places each
  decoded frame by its absolute index: a hidden frame is dropped and a frame
  past the end ends the clip. A trim window counts presented frames, and
  range-parallel decode puts boundaries on presented indices after every
  hidden frame.
  One case cannot be read from timestamps. An H.264/HEVC track with no
  composition offsets on a stream that may reorder (an elementary stream
  remuxed with `-c copy`, like the WPP_C conformance stream) has decode-order
  timestamps. Its hidden *samples'* pictures land at scattered display
  positions: 0, 4 and 8 for WPP_C. The demuxer runs h26x's own decoder over
  the first GOPs to find them, the same pictures ffmpeg discards. Such a track
  whose edit also ends early is refused.
- **Audio.** Passthrough cuts whole packets outside the edit, keeping the
  codec's decoder preroll (`edit::AudioPreroll`): one packet for AAC / AC-3 /
  E-AC-3 / DTS (and the lossless codecs), three for MP3 (its bit reservoir
  reaches back up to 511 bytes), 80 ms for Opus. The output track's own `elst`
  then hides the rest, so the cut is exact to the sample, as `ffmpeg -c copy`
  writes it. A transcode, to any codec, trims the decoded PCM instead.
- **Delays.** `Av1Mp4Muxer::set_video_delay` / `set_audio_edit` write an empty
  edit. A CMAF rendition carries the delay in its first `tfdt`, and the
  audio's hidden samples in an `elst` in `init.mp4`.

A source with no edit list, or one that changes nothing (ffmpeg's B-frame
composition shift, whose `media_time` equals the first frame's presentation
time), takes none of these paths, and its output is unchanged byte for byte.

### MKV / WebM (Matroska / EBML)

**What.** [`demux_mkv`](../crates/container/src/demux/mkv/mod.rs#L34) /
[`demux_mkv_streaming_init`](../crates/container/src/demux/mkv/mod.rs#L370) use the
`matroska-demuxer` crate for the cluster cursor and hand-rolled EBML walks for
the colour metadata the crate doesn't surface.
[`probe_mkv_color_info`](../crates/container/src/demux/mkv/mod.rs#L776) returns the
extended [`MkvColorInfo`](../crates/container/src/lib.rs#L229) (bits-per-channel,
chroma siting/subsampling, MaxCLL/MaxFALL, ST 2086 mastering chromaticities).

**Why (inferred).** The shared `StreamInfo` type in `codec` only carries the
core H.273-equivalent fields; `MkvColorInfo` /
[`MkvMasteringMetadata`](../crates/container/src/lib.rs#L256) exist to carry the
*rest* "without requiring a breaking extension of the shared `StreamInfo` type"
([`lib.rs:222`](../crates/container/src/lib.rs#L222)) — i.e. an additive carrier so
HDR signalling and future SEI passthrough have the data without an API churn
across crates. The first audio track is read when its codec is `A_AAC`,
`A_OPUS` (`CodecPrivate` *is* the RFC 7845 OpusHead body, handed to the muxer
verbatim), `A_AC3`, `A_EAC3`, `A_DTS`, `A_VORBIS`, `A_MPEG/L1`–`L3`, `A_FLAC`
or `A_ALAC`; any other is dropped with a warning
([`demux/audio/mod.rs`](../crates/container/src/demux/audio/mod.rs)).

### Sample aspect ratio

`DemuxHeader::sample_aspect` is the shape of one stored sample, `(width,
height)` in lowest terms: `(1, 1)` for square pixels, `(64, 45)` for a 16:9 PAL
720x576. Every demuxer fills it, from the container first (an MP4 `pasp` box, a
Matroska `DisplayWidth` / `DisplayHeight` in pixels), else the stream (the
H.264 / HEVC SPS VUI `aspect_ratio_info`, an MPEG-2 sequence header's
`aspect_ratio_information`), else square; AVI reads none
([`demux/aspect.rs`](../crates/container/src/demux/aspect.rs)).
`upright_sample_aspect()` turns it with a quarter-turn rotation, and
`display_aspect()` is the picture's shape as shown — what an output sized to
the source has to keep, so a rung is fitted to it rather than to the stored
size. The muxers write no `pasp`: outputs have square samples.

### Shared audio-track shape

[`AudioTrack`](../crates/container/src/demux/mod.rs#L83) is the demuxer's output
contract: `codec`, `samples` (codec-native packets), `sample_rate`, `channels`,
`asc` (AAC only), `codec_private` (Opus: the OpusHead body; AC-3 / E-AC-3 /
DTS: the `dac3` / `dec3` / `ddts` body; FLAC: STREAMINFO; ALAC: the 24-byte
cookie; Vorbis: the three headers in Xiph lacing; a bare MP3: its tag's
encoder name), `timescale`, `durations`. The muxer's input mirror is
[`AudioInfo`](../crates/container/src/lib.rs) with convenience constructors
`aac_lc` / `opus` / `ac3` / `eac3` / `dts` / `mp3` / `flac` / `alac` /
`vorbis`, and `from_ac3_frame` / `from_dts_frame`, which describe an AC-3,
E-AC-3 or DTS track from its first frame (as the Matroska and TS demuxers do,
and as the job does for its own encodes). Vorbis is for the WebM and Ogg
writers: the MP4 muxer refuses it. Anything else is rejected at `with_audio()` time — **no
silent degradation, no stubs** ([`mux/mod.rs:461`](../crates/container/src/mux/mod.rs#L461),
`check_audio`, which a caller can also run before building a muxer).

### Lossless audio: FLAC and ALAC

[`demux/audio/lossless.rs`](../crates/container/src/demux/audio/lossless.rs)
reads `fLaC` + `dfLa` and `alac` + cookie sample entries, Matroska `A_FLAC` /
`A_ALAC`, and native `.flac` streams (the sniffer's `ContainerKind::Flac`,
recognised ahead of the MP3 sniff since both may open with an ID3v2 tag),
normalising the configuration to one form per codec (FLAC: STREAMINFO alone,
flagged last; ALAC: the 24-byte cookie) and timing every packet by its frame's
own sample count. The other FLAC metadata blocks — Vorbis comments with their
vendor string, pictures, application data, a seek table — describe the source
file, not the stream, so a copy does not carry them into the output
(`normalize_flac_blocks`; until 2026-09-29 a `dfLa` or `A_FLAC` copy kept
every block). A native stream is cut into frames at sync codes whose header
CRC-8 checks and whose preceding bytes pass the frame CRC-16;
`streaming::demux_audio` reads it for audio-only output.

[`mux/lossless.rs`](../crates/container/src/mux/lossless.rs) writes the two
sample entries (used by the MP4 muxer and CMAF alike), an audio-only
faststart MP4 (`write_audio_mp4`, `ftyp M4A `; it takes any codec
`check_audio` takes, so AAC and Opus `.m4a` files come from it too) and a
native FLAC stream (`write_native_flac`): STREAMINFO, a seek table with a
point every ten seconds, and a `VORBIS_COMMENT` with an empty vendor string
and no comments. `write_native_flac_with_vendor` names a vendor for a caller
that wants its files to. See [lossless-audio.md](lossless-audio.md).

---

### Codec mappings for the other decoders

The MP4 / MOV and Matroska demuxers name every codec rivet decodes, and hand
its decoder what it configures from:

| Container | Mapping | Note |
|---|---|---|
| MP4 / MOV | `vp08` → `vp8`; `vp09` → `vp9` | the `mp4` crate reads `vp09` only; `vp08` is found by the sample-entry walk |
| MP4 / MOV | `mp4v` → `mpeg4` (esds object type 0x20), `mpeg2` (0x60-0x65), `mpeg1` (0x6A) | the `esds` DecoderSpecificInfo (MPEG-4's VOL, MPEG-2's sequence header) goes ahead of the first sample when that has none of its own (`demux::mp4::prepend_config`) |
| MP4 / MOV | `apco` … `ap4x` → `prores` | as before |
| Matroska | `V_MPEG1` / `V_MPEG2` → `mpeg1` / `mpeg2`; `V_MPEG4/ISO/SP`, `/ASP`, `/AP` → `mpeg4` | `CodecPrivate` ahead of the first frame likewise |
| Matroska | `V_PRORES` → `prores` | Matroska stores a ProRes frame without its first eight bytes (size and `icpf`); they are restored |
| Matroska | `V_MS/VFW/FOURCC` → by the `BITMAPINFOHEADER`'s FourCC (`XVID`, `DIVX`, … → `mpeg4`) | the bytes after the 40-byte header are the configuration |
| MPEG-TS | stream type 0x01 → `mpeg1` | sized from its sequence header; the MPEG-2 decoder takes it |
| MPEG-PS | video `0xE0`-`0xEF` → `mpeg1` / `mpeg2` | see [MPEG-PS](#mpeg-ps-mpg--vob) |

## MPEG-PS (`.mpg` / `.vob`)

**What.** [`ps.rs`](../crates/container/src/ps.rs) reads a program stream
(a pack header `00 00 01 BA` first): packs, system headers and PES packets in
both the MPEG-1 and the MPEG-2 PES syntax. The first video stream becomes
`mpeg1` or `mpeg2` (by whether the sequence header has a sequence extension),
one sample per coded frame (a field pair joined), timed from the sequence
header's frame rate; the first audio it can carry — MPEG audio on
`0xC0`-`0xDF`, or AC-3 on a DVD `private_stream_1` sub-stream `0x80`-`0x87` —
is framed by the transport-stream reader's own code
(`ts::audio::{mpeg_audio_from_es, ac3_from_es}`) and placed against the video
by the first timestamps of each. Padding, the DVD navigation packets, subpictures,
LPCM and DTS are skipped.

**Why so simple.** A program stream's PES timestamps are sparse (one per PES,
often several pictures each) and an MPEG-2 decoder works from the stream, so
the frames are timed from the frame rate rather than reconstructed from the
PTS chain as the TS reader does; the whole file is read at construction, as
the TS reader does, since there is no index to seek by.

## MPEG-TS

**What.** [`demux_ts`](../crates/container/src/ts/mod.rs#L240) (materialize-all) and
the streaming init walk 188-byte (or 192-byte BDAV) TS packets, find the PAT
(PID 0), walk a PMT, pick the first video elementary stream, and reassemble PES
payloads into one sample per access unit. PTS is carried at the TS 90 kHz clock.

**Why TS is special.** Unlike MP4/MOV/MKV/AVI, **MPEG-TS has no container-level
track header** — there is no sample-entry box, no `BITMAPINFOHEADER`. Dimensions,
codec config, and timing all live *inside the elementary stream*. So the TS
demuxer has to do work the other demuxers get for free:

- **Dimension recovery.** `detect_dims` (from `frame::pixel_format`) parses the
  first SPS (H.264 / HEVC, from the colour window, since a stream cut mid-GOP
  has none in its first sample) or MPEG-2 sequence header to recover
  `width`/`height`; on parse failure it falls back to `0` and logs a warn
  rather than fabricating a value
  ([`ts/mod.rs:415`](../crates/container/src/ts/mod.rs#L415)).
- **Frame-rate inference.** [`estimate_frame_rate_from_ptses`](../crates/container/src/ts/framerate.rs#L22)
  takes the **median of inter-PTS deltas** at 90 kHz. Why median and not
  `(samples-1)/duration`: the span-based calc was "off-by-one on boundary edge
  cases" ([`ts/mod.rs:307`](../crates/container/src/ts/mod.rs#L307)) — median tolerates
  B-frame reorder and a stray boundary PTS without skewing. Both the streaming
  init and `demux_ts` share this path for consistency, with a span/count fallback
  and then a 30.0 last resort.
- **The program clock.** Every PES timestamp in a program counts one 90 kHz
  clock, so where the video's first picture sits against the audio's first frame
  is a fact of the source — 21 ms on an ffmpeg-muxed H.264 + AAC stream, a second
  on one cut mid-GOP. Both readers take the earliest first timestamp of the
  selected streams as the base and give each stream a late start past it
  ([`ts/clock.rs`](../crates/container/src/ts/clock.rs)): the video through
  `video_presentation` (its first *presented* picture — the IDR, or an HEVC
  IRAP's earlier RADL picture; the access units a mid-GOP cut opens with are
  dropped but stay on the clock), the audio through `audio_edit` (its first
  frame, placed by the first PES whose PTS belongs to a frame the reader kept).
  The outputs write them as they write any late start: an MP4 empty edit, the
  first CMAF `tfdt`. Timestamps are 33-bit and wrap every 26.5 hours; each is
  unwrapped against the one before it (and the audio's against the video's), so
  a start either side of the wrap, and a wrap inside the stream (frame rate,
  duration, `Sample::pts_ticks`), keep their order and distance.
- **The frame count.** A transport stream states no frame count, and the
  pipeline plans from one (HLS segments, the multi-GPU chunk grid, progress), so
  the streaming reader counts the frames a decoder makes
  ([`ts/pictures.rs`](../crates/container/src/ts/pictures.rs)): one per PES
  packet from the first one it keeps, less the RASL pictures of an HEVC stream's
  first IRAP; for an interlaced H.264 stream (`frame_mbs_only_flag` 0) one per
  frame picture and per pair of field pictures, read from the slice headers,
  since a field may ride in a PES of its own — and then the frame rate is read
  from the PTSes of the packets a frame starts in, not from every PES.
- **Discontinuities and holes.** A program's clock can start again mid-stream: a
  splice, or two recordings joined byte for byte. The PCR PID's
  `discontinuity_indicator` marks it, and a plain `cat` shows as a PCR that
  jumps back or more than ten seconds on; a stream's own PTS jumping as far cuts
  it too ([`ts/discontinuity.rs`](../crates/container/src/ts/discontinuity.rs)).
  The audio is then placed by its own timestamps against the video's pictures
  ([`ts/retime.rs`](../crates/container/src/ts/retime.rs)): a PTS plays where
  the output presents the picture nearest it. So audio PES lost in reception
  leave a hole kept as time — the packet before it lasts that much longer, and a
  track decoded to Opus gets that much silence (`StreamingDemuxer::audio_gaps`)
  — but only as far as the video has pictures across it: a dropout that took
  both streams closes up in both, as the output presents the video's frames one
  after another. Audio after a discontinuity plays against the pictures after
  it; audio overlapping what came before by more than half a frame is dropped.
  A stream with neither is left exactly as it was.

**Multi-program + audio (Squad-37).**
- The PAT walk surfaces *every* program with a default "first program" pick and a
  `select_program(program_number)` API for the others
  ([`ts/streaming.rs:386`](../crates/container/src/ts/streaming.rs#L386)).
- Audio stream types: `0x0F` AAC-ADTS, `0x03` / `0x04` MPEG-1 / MPEG-2 audio
  (labelled `mp3` or `mp2` by the frame headers' layer), `0x81` AC-3 (ATSC
  A/53), `0x87` E-AC-3 (ATSC), and `0x06` PES-private *when* the ES descriptor
  loop carries a `registration_descriptor` tagged `"AC-3"` / `"EAC3"` (DVB /
  ETSI TS 101 154) ([`ts/pat_pmt.rs`](../crates/container/src/ts/pat_pmt.rs)).
  Random PES-private streams (DVB subtitles, teletext) are dropped silently.

**Encrypted-stream guard.** A scrambled packet (`transport_scrambling_control
!= 0`) on the active video PID trips a one-time typed warn and switches the
demuxer into a **drop-everything** mode
([`ts/mod.rs:21`](../crates/container/src/ts/mod.rs#L21)). The rationale: previously the
bytes were skipped per-packet, which meant a *partial* scramble could still leak
garbled samples downstream. rivet doesn't carry CA (Conditional Access) tables,
so an encrypted stream can't be decrypted — dropping cleanly is the correct
behaviour.

**Not implemented (by decision):** PAT/PMT CRC validation (a mis-CRCed file is
already corrupt and surfaces downstream), multiple video streams per program
(first wins), CA descrambling.

---

## AVI / RIFF

**What.** [`demux_avi`](../crates/container/src/avi/mod.rs#L51) walks the RIFF tree:
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
`LIST odml`) ([`avi/mod.rs:12`](../crates/container/src/avi/mod.rs#L12)).

**Decisions.**
- Detection is at construction: presence of an `indx` chunk in the video
  stream's `strl` triggers the OpenDML precomputed-offset path; its absence falls
  back to the legacy single-`movi` cursor walk.
- `dmlh.dwTotalFrames` **supersedes** `avih.dwTotalFrames` for OpenDML files
  precisely because `avih` may have wrapped
  ([`avi/mod.rs:137`](../crates/container/src/avi/mod.rs#L137)).
- The whole file is scanned for every `LIST movi` regardless of which RIFF
  segment it lives in ([`avi/mod.rs:68`](../crates/container/src/avi/mod.rs#L68)).
- Out of scope (stated): VBR index reconstruction — it trusts the `movi`
  sample order.

**Dropped frames.** A video chunk is one `dwScale / dwRate` tick and an
empty one is a tick with no frame: either a dropped frame's slot (a 30 fps
stream on 1/30 with frames missing — ffmpeg writes one empty chunk per missing
frame) or just a tick of a time base finer than the frame rate (`-c copy` puts
a 30 fps stream on 1/600 or 1/1000, 19 or 32 empty chunks between frames). The
streaming reader tells the two apart by the median gap between frames
([`riff::frame_pacing`](../crates/container/src/avi/riff.rs)): a gap of about
`k` medians is `k` frame periods, the frame before it is shown for all `k`
(`StreamingDemuxer::frame_repeats`), the period is the span over the periods
counted, and the header's `frame_rate` / `total_frames` are that period's rate
and count. The decode pump (and the legacy `transcode_bytes`) repeat the frame
once a period, so the constant-rate output keeps each frame within half a
period of where ffmpeg shows it; a range-split decode is not planned for such
a source. A stream whose gaps are all one period (the `-c copy` case, a
29.97 fps stream's ±1-tick jitter) is read as before. Until 2026-09-18 the
frames were spread evenly at the average rate: 281 frames of a 300-period
stream came out at 28.1 fps, 200 ms off at the 7 s mark.

**Audio** ([`avi/audio.rs`](../crates/container/src/avi/audio.rs), since
2026-09-18; before, every AVI came out video-only). The first `auds` stream is
read with the timeline ffmpeg gives it: AVI stamps no packet, so a chunk's time
is its position — the stream starts `dwStart` units in (a late start becomes the
track's edit delay), a unit is `dwScale / dwRate` seconds, and a chunk spans
what ffmpeg's `get_duration` counts: bytes over the block size for a
constant-bitrate stream (`dwSampleSize > 0`: PCM, byte-run MP3), bytes over
`nBlockAlign` rounded up for one-frame-a-chunk streams (`dwSampleSize == 0`:
AAC, AC-3, VBR MP3). So the empty audio chunks ffmpeg's muxer writes take no
time. What the audio stage takes from it:

| `wFormatTag` | Track | Path |
|---|---|---|
| `0x0001` PCM 8/16/24/32-bit, `0x0003` float 32/64 (and `WAVE_FORMAT_EXTENSIBLE` with those sub-formats) | `pcm_u8` / `pcm_s16le` / `pcm_s24le` / `pcm_s32le` / `pcm_f32le` / `pcm_f64le` | decoded (`codec::audio::decode::pcm`) → Opus |
| `0x0055` MP3, `0x0050` MPEG Layer I/II | `mp3` (the `crates/mp3` decoder reads all three layers) | passthrough into a single-file MP4 at 16 kHz and up; else decoded → Opus |
| `0x2000` AC-3 / E-AC-3, one syncframe a chunk | `ac3` / `eac3`, `dac3` / `dec3` from the first frame | passthrough |
| `0x2001` DTS, one core frame a chunk | `dts`, `ddts` from the first frame | passthrough |
| `0x00FF` (and `0x706D`, `0x4143`, `0xA106`) AAC with the ASC in the WAVEFORMATEX extra bytes | `aac` | passthrough |
| anything else — ADPCM, A-law / µ-law, WMA, ADTS-framed AAC, AC-3 / DTS not stored a frame to a chunk | named (`wmav2`, `adpcm_ms`, `aac_adts`, `avi_audio_0x….`) with no packets | dropped, by name |

---

## Annex-B conversion

**What.** [`annexb.rs`](../crates/container/src/annexb.rs) converts the
length-prefixed NAL units that MP4 and MKV store (with parameter sets out-of-band
in an `avcC` / `hvcC` config box) into the **Annex-B** form decoders expect:
`00 00 00 01` start codes between NALs, with VPS/SPS/PPS prepended to the right
sample. [`parse_avcc`](../crates/container/src/annexb.rs#L46) /
[`parse_hvcc`](../crates/container/src/annexb.rs#L118) parse the config records;
[`length_prefixed_to_annexb_tracked`](../crates/container/src/annexb.rs#L291) does
the per-sample conversion.

**Why a length-size field, not just 4 bytes.** The config record's
`lengthSizeMinusOne` can be 0/1/3 → 1/2/4 byte prefixes. Real MP4 streaming
profiles use length_size=2, so the recorded value is honored rather than
assumed ([`annexb.rs:9`](../crates/container/src/annexb.rs#L9)).

**Why `ParamSetTracker` (and why ExoPlayer needs it).**
[`ParamSetTracker`](../crates/container/src/annexb.rs#L181) is a per-stream state
machine that prepends only the parameter sets that haven't been emitted yet, *on
the first IRAP that lacks them*. It replaces an older
`prepend-on-sample-index==1` heuristic that broke two real cases
([`annexb.rs:166`](../crates/container/src/annexb.rs#L166)):

1. **ExoPlayer open-GOP MP4** (#67/#68): sample 0 is SPS-only with a *non-IDR*
   slice. The decoder can't start mid-GOP without parameter sets at the next
   IRAP — but that IRAP carries only a slice NAL, so the stream stalls. The
   tracker prepends on the first IRAP that's missing parameter sets.
2. **avcC has SPS but PPS arrives inline late** — the tracker watches inline NAL
   types and prepends only the missing kind(s).

The fix is subtle: blindly prepending avcC SPS+PPS on sample 0 produced
`SPS PPS SPS slice`, and the decoder may discard the redundant second SPS and
try to start the GOP at a non-IDR slice, which fails
([`annexb.rs:275`](../crates/container/src/annexb.rs#L275)). State is **per-stream**
(one tracker per `samples` iteration); sharing across streams would conflate
emission state. Both `demux_mp4` and `demux_mkv` use the tracked helper.

---

## The AV1 MP4 muxer

[`Av1Mp4Muxer`](../crates/container/src/mux/mod.rs#L66) is the single-file output
path: AV1 (default), H.264, or H.265 video + optional audio → one faststart MP4. It is the only video mux output
besides CMAF/HLS, and it is where most of the crate's spec-conformance and
device-compat work lives. Audio-only output has its own writers: the `.m4a`
and native `.flac` ([lossless](#lossless-audio-flac-and-alac)) and the bare
`.mp3` ([MPEG audio](#mpeg-audio-mp3--mp2-and-the-bare-mp3)).

### Spooled, RAM-bounded, faststart

**What.** The muxer streams the `mdat` payload to a tempfile while keeping only
small per-packet metadata (sizes, keyframe indices) in RAM
([`mux/mod.rs:41`](../crates/container/src/mux/mod.rs#L41)).
[`finalize_to_file`](../crates/container/src/mux/mod.rs#L834) writes `ftyp` + `moov`
*first*, then streams the tempfile's `mdat` bytes into the output.

**Why.** Two goals at once. **Faststart** (moov before mdat) lets a player begin
playback after a short prefix download instead of seeking to the end for the
index — required for web playback. **Bounded RSS**: at 15-min 1080p60 the packet
metadata is ~700 KB while the actual payload (~500 MB/variant) never leaves disk
([`mux/mod.rs:42`](../crates/container/src/mux/mod.rs#L42)). The two compose because the
`moov` (which references sample offsets) is computed from the cheap metadata, and
the bulky `mdat` is appended afterward.

### The other codecs: sample entries, QuickTime and WebM

The MP4 muxer writes every codec rivet encodes, and two more files are written
beside it:

| Codec | MP4 sample entry | QuickTime (`set_quicktime`, `ftyp qt  `) | WebM (`webm::WebmMuxer`) | CMAF / HLS |
|---|---|---|---|---|
| VP9 | `vp09` + `vpcC` (profile, level from VP9 Annex A, depth, chroma, colour) | — | `V_VP9` | `vp09` init segment, `CODECS="vp09.PP.LL.DD.CC.cp.tc.mc.FF"` |
| VP8 | `vp08` + `vpcC` | — | `V_VP8` | refused |
| MPEG-2 | `mp4v` + `esds` (object type 0x61, the sequence header as DSI) | yes | — | refused |
| MPEG-4 Part 2 | `mp4v` + `esds` (object type 0x20, the VOS / VO / VOL as DSI) | yes | — | refused |
| ProRes | — (refused: a QuickTime codec) | `apco` / `apcs` / `apcn` / `apch` / `ap4h` / `ap4x`, `colr nclc`, `fiel` | — | refused |

VP8 / VP9 / MPEG-2 / MPEG-4 / ProRes samples are stored as the encoder wrote
them (no NAL repackaging); the configuration the sample entry needs is read
from the first packet (`vpx::VpxConfig::from_stream`,
`mpeg_es::{mpeg2_config, mpeg4_config}`). MPEG-2 / MPEG-4 B pictures get a
`ctts` like H.264's.

**WebM.** [`webm.rs`](../crates/container/src/webm.rs) writes the EBML header
(`DocType webm`) and one Segment: SeekHead (fixed-size positions), Info
(1 ms timestamps), Tracks (video track 1 with `DefaultDuration` and the
`Colour` element; the audio as track 2 — Opus with `CodecPrivate` the
`OpusHead`, `CodecDelay` the pre-skip, `SeekPreRoll` 80 ms, or Vorbis
(`A_VORBIS`) with `CodecPrivate` the three headers in Xiph lacing, its
`SamplingFrequency` the stream's), Clusters opening at every video key frame
(and at least every 5 s) holding `SimpleBlock`s in time order, and Cues for
every key frame. It is built in memory — single-file outputs are handed back
as bytes anyway — so every size and position is exact. Audio blocks are timed
by their packets' durations (a Vorbis packet's from its block sizes); WebM has
no end trim, so the last packet plays whole. Subtitles are not carried.

The Matroska demuxer gives a Vorbis track's packets their exact durations
(half the previous block plus half their own, from the setup header's modes)
rather than the block timestamps' milliseconds, so a Vorbis WebM passes
through into another WebM sample-exact.

### Composition offsets (`ctts`) for B pictures

**What.** An encoder hands the muxer its packets in **decode** order, and each
packet carries only the presentation timestamp of the picture it codes
(`EncodedPacket.pts`). [`reorder::composition_offsets`](../crates/container/src/reorder.rs)
ranks those timestamps: the sample whose `pts` is the `r`-th smallest is
presented at the `r`-th decode instant, so the `i`-th arrival's offset is
`DT(r) − DT(i)` on the muxer's own fixed-tick decode timeline. When any offset
is non-zero the track gets a **version 1** (signed) `ctts`; otherwise no table
is written at all and the file is byte-identical to one from a muxer that never
had the feature. The streaming demuxer applies the same table on the way back
in, so `demux_streaming` returns the presentation times that went in.

**Why no decode timestamp from the encoder.** There is then no second clock for
an encoder to get out of step with: the only input is the timestamp the frame
went in with, and the only way to lie is to put one frame's `pts` on another
frame's packet. Two things are refused rather than guessed: a duplicated `pts`
(a rank is undefined) and an offset that would not fit the 32-bit field. No B
pictures ⇒ every rank equals its index ⇒ no table.

### `co64` and `mdat largesize` auto-upgrade — handling >4 GiB

**What.** The muxer picks 64-bit forms automatically when sizes demand it:
- `use_co64` switches the chunk-offset table from `stco` (32-bit) to `co64`
  (64-bit) when the upper-bound file size exceeds `u32::MAX`
  ([`mux/mod.rs:1085`](../crates/container/src/mux/mod.rs#L1085)).
- `use_largesize_mdat` switches the `mdat` header from the 8-byte short form to
  the ISOBMFF §4.2 16-byte `largesize` form (`size=1` sentinel + `'mdat'` +
  64-bit length) when payload + 8 would exceed `u32::MAX`
  ([`mux/mod.rs:1032`](../crates/container/src/mux/mod.rs#L1032)).

**Why / gotcha.** A >4 GiB output can't address its samples with 32-bit offsets,
and an `mdat` over 4 GiB can't state its own size in the 32-bit field — both are
hard correctness failures for large transcodes. The subtlety is that the
largesize header grows 8 → 16 bytes, which shifts the first-sample file offset,
so **the `stco`/`co64` chunk offsets must account for the 16-byte header**
([`mux/mod.rs:1021`](../crates/container/src/mux/mod.rs#L1021)); the two upgrades are
computed together. A `#[doc(hidden)]`
[`force_largesize_mdat_for_test`](../crates/container/src/mux/mod.rs#L263) exercises
the bit layout without crafting a 4 GiB tempfile — and it's a *regular* field,
not `#[cfg(test)]`-gated, so integration tests in `tests/` (which compile against
the release library) can flip it.

### Apple-compatible `ftyp` brands

**What.** [`build_ftyp`](../crates/container/src/mux/boxes.rs#L81) emits
`major_brand=iso6`, `minor_version=512`, and compatible brands `iso6` / `iso2` /
the video codec's brand (`av01`, `avc1` or `hvc1` — `avc1` / `hvc1` also when
the sample entry is `avc3` / `hev1`) / `mp41` / `mp42`. The audio-only `.m4a`
writer's `ftyp` is `M4A `.

**Why each brand.**
- `av01` is **REQUIRED** by AV1-ISOBMFF v1.3.0 §2.1 — an AV1-bearing file SHALL
  list it ([`mux/boxes.rs:68`](../crates/container/src/mux/boxes.rs#L68)).
- `iso6` (14496-12 6th ed.) covers `co64` / `mehd` v1 / largesize semantics —
  Apple's stack wants a structural ISOBMFF brand, and `major_brand=iso6` keeps a
  strict parser from rejecting a co64-bearing file that claims an older major
  brand like `mp41` (which predates co64) ([`mux/boxes.rs:78`](../crates/container/src/mux/boxes.rs#L78)).
- `iso2` / `mp41` / `mp42` keep legacy parsers and AAC-parsing-rule players
  happy.

### `colr` / `mdcv` / `clli` HDR atoms

**What.** [`build_av01`](../crates/container/src/mux/video_track.rs#L339) builds the `av01`
visual sample entry with children, in spec order:
[`av1C`](../crates/container/src/mux/video_track.rs#L721) →
[`colr` (nclx)](../crates/container/src/mux/video_track.rs#L635) →
[`mdcv`](../crates/container/src/mux/video_track.rs#L685) →
[`clli`](../crates/container/src/mux/video_track.rs#L714).
The H.264 / H.265 entries (`build_avc1` / `build_hvc1`) put the same
`colr` / `mdcv` / `clli` after their `avcC` / `hvcC`.

**Why.**
- **`colr nclx`** carries primaries / transfer / matrix / full-range. Apple's
  QuickTime / iOS Safari **silently assume BT.709 limited-range when `colr` is
  absent**, which corrupts BT.2020 / HDR / wide-gamut clips
  ([`mux/mod.rs:269`](../crates/container/src/mux/mod.rs#L269)). The default
  `ColorMetadata` is BT.709 SDR limited — correct for SDR — and real values
  arrive via `set_color_metadata`. `nclx` (not `nclc`/`rICC`/`prof`) is the right colour
  type for video distribution ([`mux/video_track.rs:632`](../crates/container/src/mux/video_track.rs#L632)).
  Transfer functions map to H.273 codes via `transfer_to_h273` (PQ/ST2084 → 16,
  HLG/AribStdB67 → 18 — [`mux/video_track.rs:608`](../crates/container/src/mux/video_track.rs#L608)).
- **`mdcv`** (Mastering Display Color Volume, ST 2086) and **`clli`** (Content
  Light Level, MaxCLL/MaxFALL) are emitted *only* when the source declared them
  (`ColorMetadata.mastering_display` / `.content_light_level` are `Some`). Per
  AV1-ISOBMFF v1.3.0 §2.3.4/§2.3.5 the order is `colr → mdcv → clli`; players
  scan by 4cc so order is recommended-not-load-bearing, but the muxer matches the
  spec anyway ([`mux/video_track.rs:331`](../crates/container/src/mux/video_track.rs#L331)). The
  `mdcv` body is the HEVC SEI 137 payload byte for byte, so its primaries are
  in the SEI's order — **green, blue, red** — which is what this crate's own
  reader (`demux/hdr.rs`) and libavformat read; until 2026-09-13 the writer
  put red first, so ffprobe reported the green chromaticity as `red_x` and a
  file re-muxed through rivet came back with red and green swapped.

> Note the default rivet color policy **tonemaps HDR → 8-bit SDR BT.709**
> ([pipeline.md §6](pipeline.md#6-color--bit-depth)), so these HDR atoms are
> written when an HDR-preserving policy (`Hdr10`/`Hlg`/`Passthrough`) is selected
> and a 10-bit encoder is in the build.

### Audio interleave + per-codec sample entries

**What.** With audio present, `finalize_to_file` writes an **interleaved** `mdat`
that alternates ~1-second video and audio chunks, with each track's `stco`/`co64`
pointing at its chunk's first sample
([`mux/mod.rs:829`](../crates/container/src/mux/mod.rs#L829)). The audio sample entry is
chosen by codec:

| Codec | Sample entry | Config box | Source |
|-------|--------------|-----------|--------|
| AAC | `mp4a` | `esds` (ASC verbatim) + Apple `chan` for ≥3ch | [`build_audio_stsd`](../crates/container/src/mux/audio_track.rs#L123) |
| Opus | `Opus` (capital O, RFC 7845 §4.4) | `dOps` (OpusHead body, LE→BE) | [`lib.rs:32`](../crates/container/src/lib.rs#L32) |
| AC-3 | `ac-3` | `dac3` ([`dac3_body_from_sync`](../crates/container/src/mux/audio_track.rs#L535)) | ETSI TS 102 366 §F.4 |
| E-AC-3 | `ec-3` | `dec3` ([`dec3_body_from_sync`](../crates/container/src/mux/audio_track.rs#L557)) | §F.6 |
| DTS | `dtsc` | `ddts` | ETSI TS 102 114 |
| MP3 | `mp4a` | `esds`, object type 0x6B (0x69 at 16 / 22.05 / 24 kHz), no DecoderSpecificInfo | ISO/IEC 14496-14 §3.1.2 |
| FLAC | `fLaC` | `dfLa` (STREAMINFO) | xiph.org FLAC-in-ISOBMFF; [`mux/lossless.rs`](../crates/container/src/mux/lossless.rs) |
| ALAC | `alac` | `alac` (24-byte cookie) + `chan` for ≥3ch | Apple ALAC; [`mux/lossless.rs`](../crates/container/src/mux/lossless.rs) |

Channel counts: AAC, Opus, E-AC-3 (7.1 through a dependent substream), DTS,
FLAC and ALAC 1–8; AC-3 1–6; MP3 1–2.

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

**What.** [`cmaf/`](../crates/container/src/cmaf/mod.rs) writes the ISO 14496-12
§8.8 movie-fragment boxes (`moof`/`mfhd`/`traf`/`tfhd`/`tfdt`/`trun`) plus the
`mvex`/`mehd`/`trex` declarations that go in a CMAF init segment's `moov`. The
stateful segmenters are
[`CmafVideoMuxer`](../crates/container/src/cmaf/mod.rs#L283) and
[`CmafAudioMuxer`](../crates/container/src/cmaf/mod.rs#L734); each emits an `init.mp4`
plus `seg-NNNNN.m4s` files and a [`CmafTrackManifest`](../crates/container/src/cmaf/mod.rs#L180)
describing them.

**Why CMAF specifically.** CMAF (ISO 23000-19) constrains the general fragmented
model — exactly one track per fragment, one track per init segment, a small
mandatory box set ([`cmaf/mod.rs:6`](../crates/container/src/cmaf/mod.rs#L6)). That
constraint is what lets hls.js / Safari do clean ABR: the renditions are
segment-aligned, so a player can switch bitrate at any segment boundary. Init
segments declare the CMAF brand (`cmfc` video / `cmfa` audio) alongside `iso6` /
`iso2` / `mp42` and, for video, the codec's brand (`av01` / `avc1` / `hvc1`), so
non-CMAF tools can still demux the boxes
([`cmaf/init.rs`](../crates/container/src/cmaf/init.rs)).

**Decisions / gotchas.**
- **`SampleFlags` packing** ([`cmaf/mod.rs:105`](../crates/container/src/cmaf/mod.rs#L105))
  encodes per-sample sync/dependency bits per §8.8.3.1: a sync sample is
  `depends_on=2, non_sync=0`; a non-key sample is `depends_on=1, non_sync=1`. A
  helper packs the u32 so callers don't compose it by hand — getting it wrong
  makes a player treat every frame as a keyframe or vice-versa.
- The split into a **box-primitive layer** + higher-level segment composers
  exists so each box's byte layout can be unit-tested against the spec without
  driving a full encode ([`cmaf/mod.rs:10`](../crates/container/src/cmaf/mod.rs#L10)); the
  fragment boxes are in `cmaf/fragment.rs`, the init segment's in `cmaf/init.rs`.
- **B pictures.** The `trun` becomes version 1 with the signed
  `sample_composition_time_offset` column exactly when a segment holds a
  reordered sample (the same `reorder::composition_offsets` as the single-file
  muxer, ranked within the segment); without B pictures the column is absent and
  the box is what it always was. `tfdt` stays the first sample's decode time,
  and because a CMAF segment must stand alone its opening sync sample must also
  be its earliest-presented one — an offset there means a picture displayed
  before the IDR was coded after it (open GOP, or a reorder leaking across the
  boundary), and `flush_segment` refuses to write it.
- **Multi-GPU helper support.**
  [`CmafVideoMuxerOptions`](../crates/container/src/cmaf/mod.rs#L239) lets a helper
  muxer start at a non-1 `first_segment_index` with the matching
  `first_segment_base_decode_time`, and skip writing `init.mp4`
  (`write_init_segment=false`), so segments produced on *different GPUs* for the
  same rung have byte-identical `tfdt` and filenames to a single-encoder run
  ([`cmaf/mod.rs:239`](../crates/container/src/cmaf/mod.rs#L239)). This is what makes the
  reactive lease engine's cross-vendor helper dispatch safe at the container
  layer.
- The first AV1 packet's OBU stream MUST contain a sequence header; the muxer
  extracts it for `av1C` in the init segment, written lazily on first
  `flush_segment` ([`cmaf/mod.rs:272`](../crates/container/src/cmaf/mod.rs#L272)).
- **H.264 / H.265 sample entry.** The init segment is written as `avc1` /
  `hvc1` with the sets seen so far in `avcC` / `hvcC`, and every segment keeps
  identical copies in band too. Once a rendition's segments are all written,
  [`settle_video_sample_entry`](../crates/container/src/cmaf/settle.rs#L34)
  reads their in-band sets against the config box and rewrites the entry, in
  place, as `avc3` / `hev1` only when a segment carries a set the box does not
  (segments from encoders that disagree, e.g. multi-GPU helpers); HLS reads
  `CODECS` from the init segment after that, so the two agree. See
  [codec encode](codec-encode.md#sample-entries-avc1--hvc1-and-avc3--hev1-only-where-the-sets-change).

### HLS playlists

**What.** [`write_hls_package`](../crates/container/src/hls.rs#L167) emits a
`master.m3u8` (one `#EXT-X-STREAM-INF` per video rendition + one
`#EXT-X-MEDIA:TYPE=AUDIO` rendition-group entry), a per-rendition
`playlist.m3u8` (with `#EXT-X-MAP` → `init.mp4` and `#EXTINF` → `seg-*.m4s`), and
the shared `audio.m3u8`. Targets HLS protocol version 7 — the minimum that
supports `EXT-X-MAP` (fMP4 init) and `EXT-X-INDEPENDENT-SEGMENTS`
([`hls.rs:19`](../crates/container/src/hls.rs#L19)). Text subtitles add a
`SUBTITLES` group of segmented WebVTT renditions
([`webvtt.rs`](../crates/container/src/webvtt.rs)).

**Why a shared audio rendition group.** Separating audio into its own rendition
group lets video variants switch bitrate *without re-downloading audio* — the ABR
win. The group holds one rendition, or two when a surround track has a stereo
downmix beside it (`audio_stereo_fallback`): one `EXT-X-MEDIA` each, the first
`DEFAULT=YES`, `CHANNELS` on both, and each codec the group holds listed once
in every variant's `CODECS`. The audio codec string comes from the track
(`rivet::job::audio::audio_codec_string`) —
`mp4a.40.{AOT}` from the AAC config (`.2` LC, `.5` HE-AAC, `.29` HE-AAC v2),
`opus`, `ac-3`, `ec-3`, `dtsc`, and `fLaC` / `alac` for lossless audio (the
HLS authoring specification's values). The video variants are described by
[`VideoVariantSpec`](../crates/container/src/hls.rs#L38).

**Gotcha — codec strings are load-bearing.** The `CODECS=` attribute MUST be
parsed from the *actual encoded bitstream* (via
`codec::codec_strings::av1_codec_string`), not composed from config — "a wrong
string causes hls.js / Safari to silently skip the variant"
([`hls.rs:21`](../crates/container/src/hls.rs#L21)). `VIDEO-RANGE` is `PQ`/`HLG`
for HDR and omitted (not `=SDR`) for SDR, per HLS authoring guidance
([`hls.rs:77`](../crates/container/src/hls.rs#L77)).

**Gotcha — `ffmpeg -i master.m3u8` on a ladder prints `Invalid NAL unit size`.**
Reading a master playlist with two or more video variants, ffmpeg (8.1.1)
prints, once per variant it opens,

```
[NULL @ …] Invalid NAL unit size (2177 > 1676).
[NULL @ …] missing picture in access unit with size 1710
```

This is ffmpeg's HLS demuxer probing every variant, not a fault in the
package, and it is not ours to fix. Measured on 2026-09-14 against a two-rung
H.264 ladder (`--rung 640x360 --rung 320x180`, `avc3`, the entry HLS was
written with then; it is `avc1` now):

- Every rendition is well formed. Both `init.mp4` files carry `avc3` with
  `lengthSizeMinusOne = 3`; the first sample of each rendition is four-byte
  length-prefixed SPS (17 / 18 bytes), PPS (5), IDR (5032 / 10736). The
  master's `CODECS` (`avc3.640033`) and `RESOLUTION` match each rendition.
- Nothing is lost. Each variant mapped out of the master
  (`-map 0:v:0`, `-map 0:v:1`) decodes 120 frames with a `framemd5` identical
  to decoding that variant's own `playlist.m3u8`, which prints nothing.
- The message comes before decoding, from probing: `-map 0:a:0` (audio only)
  prints it for both video variants, and the `[NULL @ …]` context is a parser,
  not a decoder.
- ffmpeg's own packages do the same. A two-variant fMP4 HLS set written by
  ffmpeg (`-f hls -hls_segment_type fmp4 -var_stream_map …`) prints
  `Invalid NAL unit size (529 > 30)` with `avc3` (`-tag:v avc3`), with `avc1`,
  and with both variants at the same 640x360; a one-variant master — ffmpeg's
  or rivet's — prints nothing.

To check a package with ffmpeg, decode each rendition's `playlist.m3u8`, or map
one variant out of the master; `-v error` on either is silent for a good
package.

---

## Audio container glue

Small, decoder-free modules (`aac_asc`, `ac3_sync`, `dts_sync`, `mp3`, `ogg`)
turn raw audio config bytes and frame headers into the container-level boxes
the muxer needs.

### AAC `AudioSpecificConfig`

**What.** [`parse_aac_asc`](../crates/container/src/aac_asc.rs#L175) parses a
2..16-byte ASC into `{aot, sample_rate, channels, sbr_present, ps_present,
sbr_sample_rate, signaling}`. [`effective_output_channels`](../crates/container/src/aac_asc.rs#L599)
applies the HE-AAC v2 Parametric Stereo upmix (1-ch core → 2-ch output).
[`upgrade_to_explicit_signaling`](../crates/container/src/aac_asc.rs)
rewrites an implicitly-signaled HE-AAC ASC into explicit form. The two
explicit forms of ISO/IEC 14496-3 1.6.2.1 are read: the hierarchical one
(audio object type 5 or 29 first, *the core's* sampling frequency, the
channel configuration, then `extensionSamplingFrequencyIndex` — the SBR output
rate — and the core's own object type), and the backward-compatible one (the
core's plain AAC-LC configuration followed by the `0x2B7` SBR and `0x548` PS
sync extensions). Until 2026-10-03 the parser read the hierarchical form's
leading frequency as the SBR rate and halved it for the core, and took the
backward-compatible form for implicit signalling: an HE-AAC track then came
out with half its rate (a passthrough timed wrong), or refused by the muxer.

**Why explicit signaling matters.** With **implicit** signaling the ASC says only
`AOT=2` (LC) even though the bitstream carries SBR/PS — and **Apple Core
Audio / AVFoundation silently downgrade implicit HE-AAC to mono 22.05 kHz core**,
so listeners hear quiet, muffled audio
([`aac_asc.rs:20`](../crates/container/src/aac_asc.rs#L20)). The explicit form
(leading `AOT=5` SBR + extension sample rate + inner `AOT=2`) is what Apple
players require to honour full HE-AAC output.

**AAC-LC at the reduced rates.** An AAC-LC ASC at 24 kHz or less that says
nothing about SBR is the implicit shape, and a decoder may then look for SBR
in the access units — some play such a stream at twice its rate. The muxer
used to refuse every such ASC, which refused plain AAC-LC at 8 to 24 kHz
along with it (rivet's own encoder codes 8–16 kHz input at 22.05 / 24 kHz,
so every low-rate source failed). Until 2026-10-03; now the muxer takes AAC-LC
at every rate the ASC names, 7.35 to 96 kHz, writing the ASC verbatim (a
passthrough plays as its source did), and rivet's encoder ends its AAC-LC
ASC at 24 kHz or less with the backward-compatible sync extension saying
`sbrPresentFlag = 0`, which the parser reads as plain AAC-LC. The
AudioSampleEntry's 16.16 `samplerate` holds the rate, halved until it fits
16 bits at 88.2 and 96 kHz (it overflowed before); the ASC carries the real
rate.

**PCE.** `channelConfiguration=0` streams describe their layout with a
Programme Config Element (ISO/IEC 14496-3 Table 4.2) — ffmpeg writes 7.1 that
way, and anything with `-aac_pce 1`. [`parse_pce`](../crates/container/src/aac_asc.rs#L459)
reads it (from the ASC's GASpecificConfig, or from the head of an ADTS raw data
block in the TS demuxer), `ProgramConfig::channel_count` adds it up (CPEs count
two, LFEs one; coupling / associated-data elements none), and the TS path
re-serialises it into the ASC the MP4 needs — re-serialised rather than copied,
because the PCE's `byte_alignment()` is relative to its container's start and the
raw data block and the ASC pad differently. Before 2026-08-27 the TS
demuxer bailed on `channel_configuration=0` and the whole stream went video-only.

**`chan` tag.** The Apple `chan` box names the speakers the decoder feeds, in
its output order, and that comes from the ASC, not the channel count
(`aac_asc::speaker_order`, `AAC_LAYOUT_TAGS` in `mux/audio_track.rs`). Eight
channels are three layouts: `channelConfiguration` 7 is C Lc Rc L R Ls Rs LFE
(`MPEG_7_1_B`), 12 is C L R Ls Rs Rls Rrs LFE (`AAC_7_1_B`) and 14 is C L R Ls
Rs LFE Vhl Vhr (`AAC_7_1_C`); 11 is 6.1, C L R Ls Rs Cs LFE (`AAC_6_1`). A PCE
is read in the ISO arrangement — front from the centre out, then side, back
(a pair, then a centre), LFE — and a PCE laid out any other way gets no box.
ffmpeg's own encoder writes such PCEs (front pair before the centre, and for
5.1(side), 6.1 and 7.1(wide) a side single channel in place of an LFE); its
decoder names none of them either. Until 2026-09-18 the tag followed the
channel count, so every eight-channel stream was tagged as configuration 7 and
configurations 11, 12 and 14 counted as eleven, twelve and fourteen channels,
which the gate refused (the output went video-only). ffmpeg n8.1.1 does not
know `AAC_7_1_B` / `AAC_7_1_C` and reads no layout from them (its decoder
names the layout from the ASC). The gate (`check_audio`) takes every AAC
layout of one to eight channels — 3.0, 4.0 and 5.0 included — and refuses
22.2 (configuration 13).

### AC-3 / E-AC-3 sync parse

**What.** [`parse_sync_info`](../crates/container/src/ac3_sync.rs#L113) walks the
AC-3 / E-AC-3 syncframe BSI (0x0B77 syncword) far enough to populate the MP4
config fields — [`Ac3SyncInfo`](../crates/container/src/ac3_sync.rs#L26) /
[`Eac3SyncInfo`](../crates/container/src/ac3_sync.rs#L52) — then helpers
([`channel_count`](../crates/container/src/ac3_sync.rs#L262),
[`ac3_bit_rate_kbps`](../crates/container/src/ac3_sync.rs#L280),
[`eac3_sample_rate_hz`](../crates/container/src/ac3_sync.rs#L320)) derive the
`dac3` / `dec3` box body bytes (built in `mux/audio_track.rs`).

**Why decoder-free.** Per task notes, "Do NOT introduce a Dolby decoder"
([`ac3_sync.rs:13`](../crates/container/src/ac3_sync.rs#L13)) — this crate
parses just the BSI header to synthesise the sample-entry config and copies
the frames verbatim; no coefficient parsing. A job that needs the PCM (a
downmix, a filter, another codec asked of it) decodes it with the `codec`
crate's own AC-3 / E-AC-3 decoder
([codec-decode.md](codec-decode.md#ac-3--e-ac-3-decoder)); passthrough is
still the default.

**Dependent substreams.** An E-AC-3 access unit is independent substream 0's
syncframe and the dependent substreams after it (7.1: a 2/0 one on the back
surrounds); `parse_eac3_programme` reads it, and `dec3` names them
(`num_dep_sub`, and `chan_loc` for the locations they add, ETSI TS 102 366
F.6), so the channel count is the programme's (8 for 7.1). An MP4 or
Matroska sample holds the whole access unit; the TS demuxer joins a dependent
syncframe to the access unit before it. Until 2026-10-03 the `dec3` was the
independent substream's alone and the MP4 muxer refused more than six
channels.

### Opus `dOps`

Opus needs no separate module: the demuxer surfaces the RFC 7845 OpusHead body
verbatim (MKV/WebM `CodecPrivate` *is* that body), and the muxer's `build_dops`
converts the LE OpusHead numeric fields to the BE ISOBMFF `dOps` convention and
pins the `mdhd` timescale to 48000 (Opus is internally always 48 kHz).

### Ogg (Opus and Vorbis)

[`ogg`](../crates/container/src/ogg.rs) reads and writes Ogg files for the
audio-only path. The pages are the `crates/vorbis` crate's RFC 3533 reader and
writer (CRC-checked, resynchronising, packets across pages); the codec
mappings are here. Writing Opus (RFC 7845): an `OpusHead` page, an `OpusTags`
page (vendor `rivet`), then the packets, granule positions counting 48 kHz
samples from the start of the decoded stream with the pre-skip in, the last
page's ending the stream at the presented length. Writing Vorbis: the
identification header alone on the first page, the comment and setup headers
ending the second, then the packets, granules the decoded samples. Reading
takes the first Opus or Vorbis logical stream (others skipped; Theora video
refused by name), times each packet by its own duration (an Opus packet's from
its TOC, a Vorbis packet's from its block sizes), and turns the granule
positions into the track's presentation edit: the Opus pre-skip and end, or a
Vorbis stream's leading trim and end — the same edit an MP4 edit list states,
so a round trip through Ogg is sample-exact. `sniff_container` recognises
`OggS` as `ContainerKind::Ogg`, and `streaming::demux_audio` reads it as an
audio-only source.

---

### MPEG audio (MP3 / MP2) and the bare `.mp3`

[`mp3`](../crates/container/src/mp3.rs) parses MPEG audio frame headers for
every version (MPEG-1, MPEG-2, 2.5) and layer, and walks a stream frame by
frame, taking a header only when the next one sits where it says the frame
ends. MP3 arrives from five places: Matroska `A_MPEG/L1`–`L3`, an MP4 `mp4a` entry
with object type 0x69 / 0x6B or QuickTime's `.mp3` entry (these used to be
mistaken for AAC), a transport stream's PMT stream types 0x03 / 0x04, an AVI
`0x0055` / `0x0050` stream, and a bare `.mp3` / `.mp2` file, which `sniff_container` now recognises (an ID3v2
tag, or two agreeing headers) and `streaming::demux_audio` reads as an
audio-only source. A bare file's `Xing` / `Info` frame is skipped, and its
LAME-style extension's encoder delay and padding become the track's
presentation edit (the decoder's 529 samples added). The extension is taken
at its word when its CRC checks out, whatever encoder wrote it (rivet's own
signs `rivetmp3`), or when it names LAME or the `Lavf` / `Lavc` muxers.

For a job's own encode the `.mp3` opens with the encoder's own tag frame
(`mp3::Encoder::tag_frame`). For a passthrough, the writer (`mp3::write_file`)
puts an `Info` frame (`Xing` when the bitrate varies) in front of the frames:
frame and byte counts, a 100-entry seek table, and — when the source stated
its delay — the extension with the delay/padding pair and the tag CRC
(CRC-16/ARC over the frame up to it) that readers check. The tag frame takes
the stream's own bitrate when the tag fits in one of its frames, else the
smallest one it fits in.

A bare `.mp3` read as a source (`mp3::read_file`) keeps its tag's
9-character encoder name in the track's `codec_private`, so an MP3
passthrough into another `.mp3` writes the same delay and padding under the
same name again, rather than dropping them and leaving a player to play the
encoder delay as ~50 ms of silence. With the `device` metadata category not
kept (the default) the name written is rivet's own, `rivetmp3`, the tag CRC
marking the gapless fields valid ([below](#identifying-metadata)).

MP3 goes into an MP4 at the MPEG-1 and MPEG-2 rates (16 kHz and up; MPEG-2.5's
quarter rates have no object type), mono or stereo, in 1152-tick samples. Its
RFC 6381 string is `mp3`, not `mp4a.6B` or `mp4a.40.34`
(`mux::MP3_CODEC_STRING` says why); CMAF / HLS carries no MP3.

## Identifying metadata

[`metadata`](../crates/container/src/metadata/mod.rs) handles what says where
a file was made (location), what made it (device: make, model, software, lens,
serial numbers, owner), when (capture time) and what it is called
(descriptive: title, artist, copyright, comment, keywords, cover art, …).
Orientation, colour and codec headers are not metadata here.

- **Read.** [`metadata::read`](../crates/container/src/metadata/mod.rs#L619)
  finds it in any input this crate reads and in still images: QuickTime / MP4
  `udta` (`©xyz`, `©mak`, 3GPP `loci`, maker boxes), `meta` with `mdta` keys
  or an iTunes `ilst`, the `mvhd` / `tkhd` / `mdhd` times, a FLAC track's
  `dfLa`, and the timed metadata tracks cameras record (`mebx`, `gpmd`,
  `camm`, `tmcd`, `rtmd`); Matroska `Info` and `Tags`; EXIF (with GPS) and XMP
  in JPEG, PNG, WebP, TIFF and HEIF / AVIF items; FLAC Vorbis comments, ID3
  and RIFF `INFO`; and encoder names inside an audio track's first and last
  packets. It never fails, and an item it finds but cannot place in a
  category is listed in `Metadata::unclassified` rather than dropped. (The
  HEIF / AVIF *picture* reader for stills is not here: it is
  `rivet::image::heif`, see
  [output-spec.md §11](output-spec.md#11-still-images--modeimage).)
- **Keep.** [`Keep`](../crates/container/src/metadata/mod.rs#L178) says how
  much of each category an output keeps (`location` or approximate,
  `capture_time` or the date only, `device` with or without serial numbers
  and owner, `descriptive`); `Keep::parse` reads the `metadata-keep`
  setting's words. The default keeps nothing. `Metadata::kept` narrows what
  was read to that, and `Metadata::violations` lists what an output carries
  beyond a policy (for tests that a file is clean).
- **Write.** [`metadata::write`](../crates/container/src/metadata/write.rs)
  carries the kept subset into a single-file output where players read it:
  MP4 / M4A `moov/meta` `mdta` keys (`com.apple.quicktime.location.ISO6709`,
  `make`, `model`, `software`, `camera.lens_model`, `creationdate`, `title`,
  …) plus a `udta` `©xyz` location and the `mvhd` creation time, with the
  chunk offsets moved (`write::mp4`; a fragmented file is refused, so HLS
  output takes none); a native FLAC file's `VORBIS_COMMENT` (`write::flac`:
  `LOCATION`, `DATE` and the descriptive names, no vendor); an ID3v2.4 tag in
  front of an `.mp3` (`write::mp3`); and a fresh EXIF block for a still
  (`exif::build` + `write::still`: a JPEG `APP1`, a PNG `eXIf` chunk, a WebP
  `EXIF` chunk with a `VP8X` header, or an AVIF `Exif` item, orientation 1).
  Serial numbers and owner names have no standard place in a video or audio
  file and are written only into a still's EXIF.
- **Scrub.** [`metadata::scrub`](../crates/container/src/metadata/scrub.rs)
  clears an encoder's name from a copied stream without changing a bit of
  its audio: an AAC frame that opens with a fill element (where ffmpeg's
  encoder writes `Lavc…`) has that payload cleared (`aac_frame`), and MP3
  ancillary bytes — those no frame's main data takes, which LAME pads with
  its version — are zeroed (`mp3_frames`). The job does this to AAC and MP3
  passthrough unless `device` is kept.

Without `metadata-keep`, no output carries the source's location, device,
time or tags: the muxers write none of them. The setting itself, and where it
is refused (HLS, splice), is in
[output-spec.md §14](output-spec.md#14-source-metadata--metadata_keep).

## ISOBMFF box-size sanitizer

**What.** [`sanitize_isobmff_box_sizes`](../crates/container/src/mp4_sanitize.rs#L149)
is a lenient pre-pass run before the strict `mp4` crate. It walks the box tree;
any time a child's advertised `size` exceeds the parent's remaining payload, it
rewrites the child's `size` to fit ([`mp4_sanitize.rs:1`](../crates/container/src/mp4_sanitize.rs)).

**Why.** Malformed encoders (older Apple QuickTime, some prosumer cameras, buggy
muxers) emit child boxes whose advertised size overruns the parent. The `mp4
0.14` crate (and most strict parsers) bail with
*"box contains a box with a larger size than it"* and the whole demux fails. The
sanitizer makes those files parseable while staying **byte-identical on every
well-formed file** — a clean MP4 hashes the same through it, only malformed files
mutate ([`mp4_sanitize.rs:20`](../crates/container/src/mp4_sanitize.rs#L20)).

**Gotchas.** It only touches *header* bytes — leaf-payload corruption (e.g. a
malformed `esds`) is opaque to it. The `CONTAINER_FOURCCS` set
([`mp4_sanitize.rs:43`](../crates/container/src/mp4_sanitize.rs#L43)) lists every
box the strict parser recurses into (including the visual/audio sample entries
that carry child boxes); extending the sanitizer's reach means adding to that set
when a future crate version recurses further. `size=0` ("extends to EOF") is left
untouched — strict parsers handle it correctly.

---

## Key decisions in the container crate

- **No FFmpeg for containers.** Every demuxer and muxer is hand-written against
  the spec, so no build links libav. This keeps the output narrow,
  predictable, and royalty-clean.
- **Every codec rivet decodes, rivet can mux.** VP8 / VP9 in WebM and MP4,
  MPEG-2 / MPEG-4 Part 2 in MP4 and QuickTime, ProRes in QuickTime, VP9 in
  CMAF; the demuxers map the same codecs back (and MPEG-1 / MPEG-2 / MPEG-4 /
  ProRes in Matroska, MPEG-1 in TS, MPEG program streams), so every output is
  checked by reading it back here.
- **Streaming demux for bounded RSS.** One sample at a time; nothing accumulates
  across samples, so peak heap is a sample, not a file. Audio stays buffered (it's
  small).
- **AV1 + Opus/AAC in MP4 (or CMAF/HLS) is the default output; H.264/H.265 are
  also supported.** AV1 is the royalty-clean default; the muxer emits
  `av01`/`av1C` for AV1, `avc1` + `avcC` for H.264 (with the
  high-profile extension — chroma format and bit depths from the SPS — for
  every profile but Baseline / Main / Extended, as ISO/IEC 14496-15
  §5.3.3.1.2 and ffmpeg's writer have it; High 10 needs it), and `hvc1` +
  `hvcC` with complete arrays for H.265 (legacy-player compatibility, at the
  cost of their patent-licensing obligations), with the `ftyp` brands,
  `colr`/HDR atoms, and faststart layout tuned to *just play* in browsers and
  on Apple devices. `avc3` / `hev1` are written only where the parameter sets
  really change — see
  [codec encode](codec-encode.md#sample-entries-avc1--hvc1-and-avc3--hev1-only-where-the-sets-change).
- **Verbatim audio config bytes.** AAC ASC, Opus OpusHead, AC-3 `dac3`, E-AC-3
  `dec3` are passed through untouched so codec signalling survives passthrough.
  The FLAC configuration is the exception: a copy keeps STREAMINFO alone, since
  the other blocks are the source's tags, not the stream's.
  AC-3/E-AC-3 are parsed header-only here; decoding, when a job needs PCM, is
  the `codec` crate's ([codec-decode.md](codec-decode.md#ac-3--e-ac-3-decoder)).
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
