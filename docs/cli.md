# rivet CLI reference

> What happens under the hood for any of these commands — demux → decode-once
> pump → multi-GPU encode → mux — is in [pipeline & architecture](pipeline.md).

The `rivet` binary has these subcommands: [`transcode`](#rivet-transcode),
[`splice`](#rivet-splice), [`image`](#rivet-image) (feature `image`),
[`probe`](#rivet-probe), [`devices`](#rivet-devices),
[`capabilities`](#rivet-capabilities-alias-caps) (alias `caps`), [`pipe`](#rivet-pipe),
[`batch`](#rivet-batch) (feature `batch`), [`ipc`](#rivet-ipc) (feature `ipc`),
and [`serve`](#rivet-serve) (feature `server`). Build it with:

```sh
cargo build --release                     # GPU decode + GPU encode tiers
cargo build --release --features av1-sw-fallback,h26x-fallback  # + software AV1 / H.264 / H.265 encode
cargo build --release --features nvidia   # + NVENC AV1 encoder (Windows or Linux)
```

The binary is at `target/release/rivet`. Run `rivet --help` or
`rivet <command> --help` for generated usage at any time.

> rivet encodes **AV1** (default, royalty-clean), **H.264**, or **H.265** —
> select with `--codec av1|h264|h265`. The output container is MP4 (single file)
> or CMAF/HLS (segmented); all three codecs work in both. It also writes
> **VP9**, **VP8**, **MPEG-2**, **MPEG-4 Part 2** and **ProRes** with its own
> software encoders (`--codec vp9|vp8|mpeg2|mpeg4|prores[-profile]`), each in
> the files that carry it: WebM for VP8 / VP9, a QuickTime movie for ProRes,
> MP4 (or `--container mov`) for MPEG-2 / MPEG-4; VP9 as HLS too. See
> [the compatibility matrix](../README.md#compatibility-matrix) for codecs in.

> **One vocabulary, every surface.** Every worded value below — `--mode`,
> `--audio`, `--color`, `--bit-depth`, `--seam-mode`, `--encode`, `--decode`,
> `--gpu-family`, `--encode-policy`, … — means what
> [`rivet::settings`](../crates/rivet/src/settings.rs) says it means, and
> nothing else. The CLI hands each word to the same `TranscodeSettings` the
> [IPC socket](#rivet-ipc) reads as `key=value`, the [HTTP API](api.md) as
> query/JSON fields, and the [batch manifest](batch.md) as YAML/JSON keys —
> `--audio opus`, `audio=opus`, `?audio=opus` and `audio: opus` are one code
> path. The flag tables here therefore also document those keys.

---

## `rivet transcode`

```
rivet transcode <INPUT> [OPTIONS]
```

Transcodes `<INPUT>` (any supported container/codec) to AV1 (default), H.264, or
H.265 — pick with `--codec`.

### Arguments

| Argument | Description |
|----------|-------------|
| `<INPUT>` | Input media file. Container/codec is auto-detected. |

### Options

| Flag | Values / default | Description |
|------|------------------|-------------|
| `-o`, `--output <PATH>` | default `<input>.<codec>.<ext>` (`clip.av1.mp4`, `clip.prores.mov`, `clip.vp9.webm`) | Output file (single mode, one rung) or **directory** (multi-rung single mode, or HLS). |
| `--mode <MODE>` | `single` *(default)*, `hls`, `audio` | Output shape: one self-contained MP4 per rung, a CMAF/HLS package, or the audio alone as one file — `.mp3`, or `.flac` / `.m4a` as `--audio-container` says (no video decoded; `-o` defaults to `<input-stem>.mp3`, `.flac` or `.m4a`). A `single` job whose input has no video (a bare MP3, an M4A) is written as `audio` by itself. Still images are [`rivet image`](#rivet-image). |
| `--rung <WxH[@RATE][:FIT…]>` | repeatable | A ladder rung, e.g. `--rung 1920x1080 --rung 1280x720`. The size is a maximum box the source is fitted into (`--fit`). Omit for a single rung at the source resolution. `WxH@RATE` (`1280x720@3M`) codes that rung to a bitrate — see `--video-bitrate`; `WxH@standard` gives it the rate it would have with none named anywhere, whatever `--video-bitrate` says. A rung's own fitting follows after `:` — a fit, `auto`/`fixed`, `upscale`/`no-upscale`: `--rung 1080x1920:cover:fixed`. |
| `--fit <FIT>` | `contain` (default), `cover`, `pad`, `stretch` | How the source meets each rung's box: inside it keeping its shape; filling it and centre-cropping; inside it with black bars to exactly the box; or stretched to exactly the box (the pre-fitting behaviour). See [fitting](output-spec.md#fitting-the-source-into-a-rung). |
| `--orientation <auto\|fixed>` | default `auto` | `auto`: a box turns to the source's orientation (1920x1080 on a portrait source is 1080x1920). `fixed`: boxes are used as written. |
| `--upscale` | flag | Let a rung be larger than the source. Off by default: a smaller source comes out at its own size, and rungs that collapse onto the same size are merged (the summary lists them). |
| `--ladder` | flag | Auto-derive a standard ABR ladder from the source resolution (instead of `--rung`). |
| `--max-short-side <PIXELS\|standard>` | default `1080` | With `--ladder`, cap the tallest rung's short side. `standard` states the default. |
| `--segment-seconds <S>` | default `4.0` | HLS target segment length (segments still break on keyframes). |
| `--crf <N>` | encoder-native | Constant rate factor (lower = better quality). Names the quantiser directly; when set, `--target` is not consulted. |
| `--video-bitrate <BPS>` | e.g. `3M`, `800k`; `standard` states the default (none) | Code every rung that does not name its own (`--rung WxH@RATE`, or `bitrate=` in `--encode-policy`) to this bitrate rather than to `--target`: the encoder's rate controller picks a quantiser per picture to spend it. Who can code it depends on `--rate-mode`. An **average** rate (the default) is coded only by the software encoders — H.264 / H.265 (`h26x-fallback`), AV1 (`av1-sw-fallback`), and VP9, MPEG-2 and MPEG-4 Part 2 in every build: on a host whose encode pool is GPUs the job is refused before a frame is decoded, by name, saying how to reach the software pool. A **constant** rate (`--rate-mode cbr`) is coded by the GPU encoders (QSV, NVENC, AMF, AV1 included) and the software H.264 / H.265 encoder, not by the software AV1 or VP9 encoders, which refuse it (and a coded picture buffer) by name. A CRF or `--seam-mode constqp` beside a rate is refused. Measured in [codec-encode.md](codec-encode.md#bitrate-rungs-in-the-software-tier-measured). |
| `--video-buffer <DURATION>` | default `1s` for a bitrate rung; e.g. `500ms`; `0` for none | The coded picture buffer every bitrate rung declares (the stream's HRD) and keeps to. It bounds any stretch of the stream at the rate plus the buffer, which is what bounds an HLS segment's peak and so its `BANDWIDTH`. The unit is required. A `cbr` rung refuses `0`. |
| `--rate-mode <MODE>` | `average` *(default; also `abr`)*, `cbr` *(also `constant`)* | How every bitrate rung is coded. `cbr` is a constant rate, which is also the maximum within the declared buffer (`--video-buffer`). A `cbr` rung with no rate of its own takes `--video-bitrate`, else a default by codec, size and frame rate: H.264 at 30 fps is 16M for 2160p, 5M for 1080p, 3M for 720p, 1.2M for 480p, 0.8M for 360p; H.265 0.65x that, AV1 0.5x; more above 30 fps (720p60 H.264 is 4.5M). `--encode-policy` sets it per rung (`rate=cbr` / `rate=average`). |
| `--video-speed <TIER>` | `draft`, `standard` *(default)*, `archive` | Encoder effort for every rung, mapped by each encoder onto its own presets: NVENC P5 / P6 / P7; software VP9 from fixed 32x32 partitions (`draft`) through fixed 16x16 (`standard`, about 10 frames/s at 352x288 on one core) to a rate-distortion partition search (`archive`, about 1.7); the software AV1 encoder its motion search range (±8 / ±16 / ±32). An `--encode-policy` `speed=` word for one rung wins. Settings key `video-speed`. |
| `--target <T>` | `visually_lossless`, `high`, `standard` *(default)*, `low`, `vmaf=N` | Perceptual quality target for every rung. `vmaf=N` aims for a VMAF score — mapped to each backend's quantiser through the calibrated tables in `codec::encode::tuning`, so the same target means the same perceived quality on NVENC, QSV, AMF and rav1e. Measure it with [`bench/`](../bench/README.md). |
| `--gop <FRAMES\|SECONDSs>` (`--keyframe-interval`) | frames, or seconds (`2s`, `1.5s`) | GOP length for every rung (default: two seconds at the output rate, which `2s` states; seconds are made frames at the output rate, rounded). Single file: the keyframe cadence and, across GPUs, the chunk grid. HLS: the segment grid stays `--segment-seconds`; a shorter GOP adds keyframes inside each segment (for seeking); a longer one is silently the segment, since every segment opens on an IDR anyway. |
| `--audio <POLICY>` | `auto` *(default)*, `opus`, `mp3`, `aac`, `he-aac`, `he-aacv2`, `vorbis`, `ac3`, `eac3`, `dts`, `flac`, `alac`, `drop` | `auto`: passthrough AAC/Opus/AC-3/E-AC-3/DTS (and MP3 into a single-file MP4, Opus and Vorbis into a WebM), transcode the rest to Opus, drop what cannot be decoded; with `--mode audio` it means MP3. The rest force that codec (a source already in it is copied), every encoder rivet's own: `opus` (MP4 / MOV / WebM, HLS, an Ogg file); `mp3` (CBR; single-file or `--mode audio`, not HLS); `aac` (AAC-LC, mono to 7.1), `he-aac` (HE-AAC, SBR, 32 / 44.1 / 48 kHz, mono to 7.1) and `he-aacv2` (HE-AAC v2, parametric stereo, stereo) — single-file, HLS, or an `.m4a`; `vorbis` (WebM or an Ogg file; `--audio-quality`); `ac3`, `eac3` (Dolby Digital / Plus) and `dts` (DTS core), up to 5.1 (E-AC-3 up to 7.1) — single-file, HLS, or an `.m4a`. `flac` / `alac`: lossless, beside video in MP4 or HLS or alone as a `.flac` / `.m4a` — see [lossless audio](lossless-audio.md). `drop`: video only. A file that cannot hold the codec (a `.mp3` and `aac`, a WebM and `ac3`, an MP4 and `vorbis`) is refused by name. |
| `--audio-quality <Q>` | `-1` … `10` *(default 5)* | Vorbis quality (`--audio vorbis`, which takes no `--audio-bitrate`): about 66 kb/s stereo at `-1`, 141 at `4`, 173 at `6`, 234 at `10` for 44.1 kHz music. |
| `--audio-bit-depth <DEPTH>` | `source` *(default)*, `16`, `24` | Bit depth of `flac` / `alac` output. `source`: 16 for a 16-bit or lossy source, else 24. |
| `--flac-compression <LEVEL>` | `fast`, `default` *(default)*, `best` | FLAC compression effort. |
| `--he-aac <POLICY>` | `auto` *(default)*, `passthrough`, `core` | An HE-AAC (or HE-AAC v2) source, which rivet decodes in full: SBR at the full rate, parametric stereo to two channels. `auto`: as any AAC track — passed through where the output can carry it and nothing asks for a change, decoded in full otherwise. `passthrough`: never decoded (the job is refused where it would have to be). `core`: decoded as its AAC-LC core only (half the rate, lower bandwidth). See [output spec](output-spec.md#3-audio--with_audioaudiocodecpolicy). |
| `--audio-decode-deny <CODECS>` | comma list of `aac`, `ac3`, `alac`, `dts`, `eac3`, `flac`, `mp2`, `mp3`, `opus`, `pcm`, `vorbis` | Source audio codecs that may not be decoded (default: none). A denied track is never decoded: it is passed through where the output can carry it as it is (another codec asked of it is then not made, the handling saying why), and a job that needs its PCM (a downmix, an audio filter, a `.mp3` or `.flac` file, an output that cannot hold the codec) is refused before any work, naming the setting. With `aac` denied an HE-AAC source is passed through whatever `--he-aac` says. See [output spec](output-spec.md#restricting-decoders--audio_decode_deny). |
| `--audio-container <C>` | `auto` *(default)*, `mp3`, `flac`, `mp4`, `ogg` | The file `--mode audio` writes: `auto` follows the codec — `.flac` for `--audio flac`, an Ogg file (`.opus` / `.ogg`) for `opus` / `vorbis`, an `.m4a` for `alac`, `aac`, `he-aac`, `he-aacv2`, `ac3`, `eac3` and `dts`, else `.mp3`; `mp4` is an `.m4a` for any codec the MP4 muxer takes (Opus, MP3 and FLAC included); `ogg` an Ogg file for Opus or Vorbis. |
| `--audio-bitrate <BPS>` | e.g. `240k` | Target for **transcoded** audio. Omit (or `standard`) to derive it: Opus from the channel layout (64k mono, 96k stereo, 320k for 5.1, 416k for 7.1); AAC 64k mono, 128k stereo, 384k for 5.1, 512k for 7.1; HE-AAC 32k mono, 48k stereo; HE-AAC v2 32k; MP3 128k stereo / 64k mono, and an MP3 rate must be one of 32k 40k 48k 56k 64k 80k 96k 112k 128k 160k 192k 224k 256k 320k; AC-3 96k mono, 192k stereo, 448k 5.1, one of A/52 Table 5.18's rates (32k … 640k); E-AC-3 96k, 192k, 384k for 5.1, 512k for 7.1 (32k … 6144k); DTS the full rate (1536k at 48 kHz), one of ETSI TS 102 114 Table 5-7's (32k … 1536k). Not for Vorbis (`--audio-quality`). Ignored for passthrough tracks, which keep the bitrate they were authored at. |
| `--audio-channels <LAYOUT>` | `source` *(default)*, `mono`, `stereo`, `5.1`, `7.1` | Output channel layout. `source` keeps the source's where the codec carries it (MP3 and HE-AAC v2: stereo at most; AC-3 and DTS: 5.1 at most, as 5.1(side); E-AC-3 7.1). The others downmix (ITU-R BS.775, LFE dropped, normalised so nothing clips); asking for more channels than the source has is an error — rivet does not upmix. |
| `--audio-stereo-fallback` | off | HLS: beside a surround audio rendition, a stereo downmix of it in the same audio group (`CHANNELS="2"` and `"6"`), the group's default. |
| `--metadata-keep <CATEGORIES>` | none *(default)*; comma list of `location` or `location:approximate`, `capture_time` or `capture_time:date`, `device` or `device:all`, `descriptive`, `all`, `none` | The source's identifying metadata to carry into the output; by default none is written. `location:approximate` keeps two decimal places (about a kilometre; no altitude or place name); `capture_time:date` keeps the day only; `device` is make, model, software and lens, and `device:all` adds serial numbers and owner name. With the device not kept, a copied AAC or MP3 stream also loses the source encoder's name. Single-file and `--mode audio` output; HLS refuses it. |
| `--audio-filter <CHAIN>` | e.g. `channelmap=FL-FL\|FR-FR:stereo` | Audio filter chain applied to decoded PCM before the encoder — see [audio filters](audio-filters.md). Forces a decode/re-encode, so it can't be combined with a passthrough-only source codec. |
| `--subtitles <SELECTION>` | `all` *(default)*, `none`, `eng,deu` | Which of the source's **text** subtitle tracks to carry: every one, none, or a language list. Single file: a `tx3g` track per language. HLS: a WebVTT rendition per language. Bitmap subtitles (PGS / VobSub / DVB) are always dropped. See [Subtitles](#subtitles). |
| `--max-fps <FPS\|source>` | default `source` | Cap the output frame rate (source cadence otherwise preserved; frames over the cap are dropped, not retimed). `source` states the default: no cap. |
| `--color <POLICY>` | `sdr` *(default)*, `hdr10`, `hlg`, `passthrough` | Output color / tonemap policy — see [Color & bit depth](#color--bit-depth). |
| `--pixel-format <FMT>` | `auto` *(default)*, `8bit`, `10bit` | Output luma bit depth. |
| `--chroma-downsample <FILTER>` | `box` *(default)*, `lanczos` | 4:4:4 → 4:2:0 chroma filter for 4:4:4 sources — see [Color & bit depth](#color--bit-depth). |
| `--filter <CHAIN>` | e.g. `crop=1280:720,hflip` | Video filter chain applied before scaling — see [Video filters](filters/README.md). |
| `--trim-start <S>` | seconds | **Splice/trim:** keep from this time. The output is re-based to zero. Trimmed jobs take the serial encode path. |
| `--trim-end <S>` | seconds | **Splice/trim:** keep until this time. The kept range is `[start, end)`, exact at any frame rate. To *join* clips, use [`rivet splice`](#rivet-splice). |
| `--codec <CODEC>` | `av1` *(default)*, `h264`, `h265`, `vp9`, `vp8`, `mpeg2`, `mpeg4`, `prores` / `prores-proxy` / `-lt` / `-422` / `-hq` / `-4444` / `-4444xq` | Output video codec. `av1` is royalty-clean (the project default); `h264`/`h265` are for legacy-player compatibility (patent-licensing caveats). All three work for **single-file MP4 and CMAF/HLS**. H.264/H.265 are encoded on **NVENC** (validated on RTX 3090) + **QSV** (validated on Intel Arc); AMF H.264/H.265 is a follow-up. `vp9` / `vp8` / `mpeg2` / `mpeg4` / `prores` are encoded by rivet's own software encoders in every build; VP9 works for single files and HLS, the others for single files. See [output spec](output-spec.md#the-other-codecs-vp9-vp8-mpeg-2-mpeg-4-part-2-prores) for what each refuses. |
| `--container <C>` | `mp4`, `mov`, `webm`; default the codec's own | The file of a single-file output: `mov` (a QuickTime movie) is the default for ProRes and its only file; `webm` the default for VP8 / VP9 (Opus audio, no subtitles); `mp4` otherwise. A codec in a file that does not carry it is refused by name. |
| `--prores-profile <P>` | `proxy`, `lt`, `422` *(default)*, `hq`, `4444`, `4444xq` | The ProRes profile with `--codec prores` (`--codec prores-hq` says the same). |

### GPU selection

| Flag | Description |
|------|-------------|
| `--encode <PLAN>` | The encode plan — which cards, and how the work is laid across them, as one value so the halves cannot contradict: `all` *(default)* — every capable card, each worker serving every rung and taking the next chunk of whichever is furthest behind (a card idles only when the job is out of work); `per-rung` — every card, each pinned to its own rungs (one rung, one GPU when the ladder fits the pool; predictable placement, idle cards when a rung is blocked); `single` — one card, one encoder per rung, serial (single-file output is seam-free by construction); `gpu:N` — single, pinned to card N; `family:nvidia\|amd\|intel` — one vendor's cards, ladder-scheduled. |
| `--gpu <N>` / `--single-gpu` / `--gpu-family <VENDOR>` | Older spellings of `--encode gpu:N` / `single` / `family:VENDOR`. Still work; `--encode` wins when both are given. |
| `--decode <PLAN>` | The decode plan — which card(s), and whether the decode is one pump or split into ranges, as one value: `auto` *(default)* — cut an un-spliced H.264/H.265 source into one range per capable card at keyframes on chunk boundaries, one decode pump per card, each decoding its own stretch (whole where the source cannot be split); `whole` — one decoder for the whole source (the control arm of any comparison); `fastest` — benchmark every decode-capable card on a prefix of the input and put one decoder on the quickest; `gpu:N` — one decoder pinned to card N (e.g. an iGPU while the dGPUs encode); `ranges:N` — a range count (more than the cards is legal and is how the split is exercised on a one-card host). Output is byte-identical whichever you pick. `--decode-gpu N` still works and means `gpu:N`. |
| `--encode-policy <recommended\|off\|SPEC>` | Per-rung encoder knobs by ladder position. `recommended` (also `default`) is the measured ladder policy (+2 libaom-CQ steps softer per rung going down, no top bonus, one tile below 4K, three reference frames — about −20% storage on a five-rung ladder for a fraction of a VMAF point); `off` (also `none`) is none, the default; or the rule grammar, e.g. `qstep=2;top:q=-2;short<=2159:tiles=1x1;any:refs=3`, where `bitrate=`, `buffer=` and `rate=cbr` / `rate=average` set a rung's rate (`any:rate=cbr;top:bitrate=6M`) — see [output-spec.md](output-spec.md#per-rung-policy--with_rung_policyrungpolicy). |
| `--seam-mode <parallel\|constqp>` | Seam *quality* on the multi-GPU **single-file** path — how the chunks it stitches are rate-controlled. Nothing else: no seams at all is an encode plan (`--encode single`), not a seam mode. `serial` still parses as the older spelling of `--encode single`. |

See [GPU scheduling](../README.md#gpu-scheduling-the-rung-benefit) for how
`AllGpus` / `SingleGpu` / `Family` actually distribute work.

#### A pin nothing can serve is refused, by name

`family:VENDOR` and `gpu:N` name silicon. When nothing they name can encode the
job's codec in this build — the family is absent, the card index does not
exist, or the card is there but the build cannot drive it for that codec — the
job is **refused before a frame is decoded**, on every path (single-file
serial, chunked, HLS), and the error says what is present and what would work:

```
error: transcoding in.mp4: no encoder matches `--encode family:intel` for H.264 on this host: no Intel GPU is present. Present: NVIDIA GeForce RTX 3090 (gpu 0, NVIDIA, encodes H.264); AMD Radeon(TM) Graphics (gpu 1, AMD, cannot encode H.264 in this build). Fix: pin a card that can (`--encode family:nvidia` or `--encode gpu:0`) or drop the pin (`--encode all`, the default) to use them; to run on the software H.264 encoder (`h26x-fallback`) instead, drop the pin and hide the cards (`CUDA_VISIBLE_DEVICES=-1` hides NVIDIA), or build without the vendor features — the software pool takes the job only when no card can encode H.264 and none is pinned.
```

A pin never falls through: not to another vendor, and not to the software
encoders — on the serial path either, where a pinned card that fails to start
fails the job naming the vendor rather than sliding down the NVIDIA → AMD →
Intel → software chain. The **software pool** (the ladder on CPU leases, one
software encoder per slot) is what an *unpinned* plan (`all`, `per-rung`,
`single`) gets when no card can encode the codec in this build and a software
encoder is compiled in (`h26x-fallback` / `av1-sw-fallback`): hide the cards
(`CUDA_VISIBLE_DEVICES=-1` for NVIDIA) or build without `nvidia` / `amd` /
`qsv`. `TRANSCODE_ENCODER_BACKEND=h26x|av1|nvenc|amf|qsv` still pins a
backend by name on the serial path.

#### Chunk seams (`--seam-mode`)

When more than one GPU encodes a **single file**, each rung is chunked at GOP
boundaries, encoded in parallel, and the AV1 packets are stitched into one MP4.
Each chunk is an independent IDR-led GOP, so the result always plays — but each
chunk's rate control is independent, so quality can step at the ~2 s seams. AMD
(AMF) and Intel (QSV) chunks are constant-QP and already seam-flat; this knob
chiefly governs **NVENC** (which otherwise runs VBR per chunk):

| Mode | Seams | Speed | Notes |
|------|-------|-------|-------|
| `parallel` *(default)* | possible mild NVENC steps | fastest (all GPUs) | each chunk uses its encoder's normal rate control |
| `constqp` | flat | fast (all GPUs) | forces constant-QP; the QP is derived from the quality target, so quality still tracks it |

Wanting **no seams at all** is not a seam mode — it is one encoder per rung,
which is `--encode single` (or `gpu:N`); `--seam-mode serial` still parses as
that, for scripts that predate the split of the two questions.

(Single-GPU hosts, `--single-gpu`/`--gpu`, and HLS jobs are unaffected — HLS
segments are independent files by design.)

### Color & bit depth

The decode pump tonemaps only when the policy says so — it never decides on its
own:

| `--color` | Output | Bit depth | Needs |
|-----------|--------|-----------|-------|
| `sdr` *(default)* | tonemap HDR → SDR BT.709 | 8-bit | any encoder |
| `passthrough` | source color verbatim | source | 10-bit encoder if source is 10-bit |
| `hdr10` | BT.2020 + PQ | 10-bit | a 10-bit encoder (below) |
| `hlg` | BT.2020 + HLG | 10-bit | a 10-bit encoder (below) |

10-bit / HDR output needs a 10-bit encoder **for the output codec** in this
build, and `rivet transcode` checks that before anything is decoded:

| `--codec` | 10-bit / HDR with | 8-bit only on |
|-----------|-------------------|---------------|
| `av1` (Main) | `nvidia` (NVENC), `amd` (AMF), `qsv` (oneVPL P010), on a GPU with AV1 encode; 10-bit SDR (no HDR) also on the software tier, `av1` (`av1-sw-fallback`) | — |
| `h265` (Main 10) | `nvidia`, `amd`, `qsv`, or the software tier `h26x-fallback` | — |
| `h264` (High 10) | the software tier `h26x-fallback` only | NVENC, AMF, QSV (no hardware Hi10P encoder) |

It's 4:2:0 10-bit, HDR-tagged in the container (`colr`/`mdcv`/`clli`) and, for
H.264 / H.265, in the SPS VUI. A request the build can't produce for its codec
is refused by name, before the input is decoded — here on a `--features nvidia`
build:

```
$ rivet transcode in.mp4 -o out.mp4 --codec h264 --color hdr10
error: building output spec: invalid output spec: h264 at 10 bits (color=Hdr10, bit_depth=Auto) cannot be encoded: this build encodes h264 with nvenc (8-bit SDR). h264 at 10 bits needs the software tier (build with `h26x-fallback`); no hardware backend encodes h264 at 10 bits
```

What the **source** makes of the output is checked too, once the input is
probed and still before a frame is decoded. `--pixel-format auto` keeps a
10-bit source at 10 bits, and `--color passthrough` keeps an HDR source's
transfer, so a request that names neither can still need a 10-bit or HDR
encoder. Until 2026-09-14 such a job passed validation, started decoding and
failed building the encoder ("all 1 rung(s) failed"). Here on a
`--features nvidia` build, with a 10-bit SDR HEVC source:

```
$ rivet transcode clip10_hevc.mp4 -o out.mp4 --codec h264
error: transcoding clip10_hevc.mp4: invalid OutputSpec: h264 at 10 bits (color=TonemapToSdr, bit_depth=Auto) cannot be encoded: this build encodes h264 with nvenc (8-bit SDR). h264 at 10 bits needs the software tier (build with `h26x-fallback`); no hardware backend encodes h264 at 10 bits; the source is Yuv420p10le and bit_depth=Auto keeps its 10 bits: `--pixel-format 8bit` encodes it at 8 bits
```

A splice is checked against its first clip, which the output follows; an HDR
source under `--color passthrough` is told `--color sdr` tonemaps it.

On a build with both a card and the software tier (`--features
nvidia,h26x-fallback`), a 10-bit H.264 output passes that check — `h26x`
encodes it — and every path then gets an encoder that can take it:

- The **serial** single-file encoder (one card, and every splice) is built by
  the dispatcher, which falls back from NVENC, with no 10-bit H.264, to `h26x`.
- The **chunk-and-stitch** engine and the **HLS** ladder lease an encoder per
  chunk or segment, with no fallback, so their pool is built for the output's
  depth: a card whose encoder takes the codec only at 8 bits is left out, and
  software slots take its place. Until 2026-09-14 they leased the card and
  failed after decoding had started (`ladder worker 0 failed: creating encoder
  for segment: … NVENC on GPU 0 does not support 10-bit H264 encode`).
- A policy that pins the card (`--encode family:nvidia`, `--encode gpu:N`)
  gets neither fallback nor software, so it is refused before decoding, naming
  the format: `no encoder matches --encode family:nvidia for 10-bit H.264 on
  this host … drop the pin`.

A backend pinned by name counts as well: `TRANSCODE_ENCODER_BACKEND=h26x` builds
the software encoder with or without `h26x-fallback` (the feature only gates the
automatic fallback), so it makes `--codec h264|h265` at 10 bits valid on any
build; the pin applies to the serial single-file path. The check is against the build, not the card: a `--features nvidia` binary
accepts `--codec av1 --color hdr10`, and a GPU without AV1 encode (an RTX
30-series, say) then refuses it when the encoder is built. `rivet capabilities`
prints the per-codec answer.

12-bit sources (HEVC Main 12 / RExt 4:2:2 / 4:4:4 12-bit from the native decoder)
are accepted: they are narrowed to 10-bit with rounding (or to 8-bit for an
8-bit output), and 4:2:2 / 4:4:4 chroma is downsampled to 4:2:0. No encoder in the
tree takes more than 10 bits, so `--pixel-format auto` gives a 10-bit output for
a 12-bit source.

**`--chroma-downsample box|lanczos`** picks the 4:4:4 → 4:2:0 chroma filter for
4:4:4 sources (H.264 High 4:4:4, HEVC RExt 4:4:4, AV1 4:4:4): `box` (default) is the 2×2 average
and keeps outputs byte-identical to earlier releases; `lanczos` is a separable
Lanczos-2 sited where 4:2:0 decoders expect the chroma (co-sited horizontally,
midway vertically), measurably closer to the source after a round trip (numbers
in [codec-encode.md](codec-encode.md#why--the-avx2-runtime-dispatch-pattern)). The
same word is the `chroma-downsample` settings key on the IPC socket, the HTTP API,
and the batch manifest (`chroma_downsample:`). No effect on 4:2:0 / 4:2:2 sources.

### Audio

Four knobs beyond the `--audio` policy. The first three affect **transcoded**
audio only (a passthrough track is copied verbatim by definition):

- **`--audio-bitrate`** sets the encoder's target, ffmpeg-style (`240k`, `1.5M`,
  or a plain bits-per-second count). Omitted, Opus derives it from the channel
  layout — 64 kbps per uncoupled stream + 96 kbps per coupled pair, so 64k mono,
  96k stereo, 320k for 5.1, 416k for 7.1 — and MP3 takes 128k stereo / 64k mono.
  MP3 is constant bitrate on the MPEG-1 ladder (32k … 320k); another value is
  refused.
- **`--audio-channels`** sets the output layout: `source` (the default), `mono`,
  `stereo`, `5.1`, `7.1`. A narrower layout is a **downmix** by ITU-R BS.775
  (5.1 → stereo: `L = 0.414·FL + 0.293·FC + 0.293·SL`, the LFE dropped — the
  normalisation that keeps a full-scale centre from clipping), and asking for
  more channels than the source has is **an error**: rivet never upmixes.
  Asking for the width the source has is a passthrough. `source` keeps what
  the codec can carry: Opus 1–8 channels (a layout Opus has no mapping for,
  such as 2.1, goes out in the narrowest one with a place for every speaker,
  the missing ones silent), MP3 two at most (a surround source is downmixed to
  stereo).
- **`--audio-filter`** runs a chain over the decoded PCM before the layout
  conversion and the encoder — today `channelmap`, for remapping / reordering /
  selecting channels. Full reference: [audio filters](audio-filters.md).
- **`--audio-stereo-fallback`** (HLS) adds a stereo downmix rendition beside a
  surround one, in the same audio group.

Multichannel is carried end to end: 3–8 channels ride Opus's channel-mapping
family 1 (RFC 7845 §5.1.1.2). rivet decodes AAC, MP3, MP2, Vorbis, Opus, AC-3,
E-AC-3, DTS, FLAC, ALAC and PCM, so any of those can be downmixed or
re-encoded; a track nothing asks to change is passed through (a 5.1 AAC
source stays 5.1). HE-AAC and HE-AAC v2 decode in full (SBR, parametric
stereo); `--he-aac passthrough` keeps such a source undecoded and `--he-aac
core` decodes only its AAC-LC core.

Asking for `--audio drop` together with any of the knobs is rejected rather
than silently ignored.

### MP3

`--audio mp3` writes CBR MP3 with rivet's own encoder (the `crates/mp3`
submodule; every build, no feature, nothing to install). An MP3 source passes
through; anything else is decoded, downmixed to stereo at most, resampled to
32 / 44.1 / 48 kHz where it is not already one of them, and encoded. Into an
MP4 it is an `mp4a` entry (object type 0x6B) whose `codecs` value is `mp3`;
HLS refuses it, since CMAF has no MP3 profile. `--mode audio` writes the audio
alone as a bare `.mp3` behind the encoder's own `Info` frame, whose LAME-style
extension (encoder string `rivetmp3`) carries the encoder delay and padding,
so a gapless player presents exactly the source's samples:

```sh
rivet transcode talk.mkv --mode audio                 # -> talk.mp3, 128k stereo
rivet transcode talk.mkv -o talk.mp3 --mode audio --audio-channels mono --audio-bitrate 64k
rivet transcode episode.mp3 -o episode-copy.mp3       # no video: audio-only, MP3 passed through
```

### Other audio codecs

Every codec rivet reads it can write, each from the workspace's own encoder:

```sh
rivet transcode film.mkv -o film.mp4 --audio ac3                       # AC-3 5.1 at 448k (dac3)
rivet transcode film.mkv -o film.mp4 --audio eac3 --audio-bitrate 640k  # E-AC-3 (dec3)
rivet transcode film.mkv -o film.mp4 --audio dts                       # DTS core, 1536k (ddts)
rivet transcode talk.mkv -o hls/ --mode hls --audio he-aac --audio-bitrate 48k   # mp4a.40.5
rivet transcode talk.mkv -o talk.m4a --mode audio --audio he-aacv2     # mp4a.40.29, 32k stereo
rivet transcode clip.mkv -o clip.webm --codec vp9 --audio vorbis --audio-quality 6
rivet transcode song.flac -o song.ogg --mode audio --audio vorbis      # Ogg Vorbis
rivet transcode song.flac -o song.opus --mode audio --audio opus       # Ogg Opus
```

AC-3, E-AC-3 and DTS carry A/52's and the DTS core's arrangements up to 5.1:
5.1 goes out as 5.1(side), and for AC-3 and DTS 7.1 is downmixed to it.
E-AC-3 carries 7.1 as it is (FL FR FC LFE BL BR SL SR: a 5.1 independent
substream and a dependent one on the back surrounds, 512k by default). Vorbis has no MP4 or CMAF mapping, so it goes into
a WebM or an Ogg file only. An Ogg file is read as an input too.

### Subtitles

`--subtitles all` (the default) carries every **text** subtitle track the
source has — the `-c:s copy` equivalent. `--subtitles none` drops them, and a
language list such as `--subtitles eng,deu` keeps only those tracks, in that
order. Codes match by language, not spelling (`en` finds a track tagged `eng`,
`ger` finds `deu`), and a listed language the source lacks is logged, not an
error. `rivet probe` lists the tracks a file has.

| Source | Carried |
|--------|---------|
| Matroska `S_TEXT/UTF8` (SRT) | ✅ |
| Matroska `S_TEXT/ASS` / `S_TEXT/SSA` | ✅ (markup stripped) |
| Matroska `S_TEXT/WEBVTT` | ✅ (tags stripped) |
| MP4 `tx3g` (`mov_text`) | ✅ |
| MP4 `wvtt` (WebVTT in ISOBMFF) | ✅ (tags stripped) |
| PGS / VobSub / DVB (bitmap) | ❌ dropped with a warning — no text form exists |

Where they go depends on the output:

- **single** — one `tx3g` track per language (3GPP timed text, what ffmpeg calls
  `mov_text`, the only subtitle format MP4 natively holds), each with its own
  `mdhd` language code for the player's track picker.
- **hls** — one segmented-WebVTT rendition per language under
  `subs/<lang>/` (`seg-NNNNN.vtt` + `subtitles.m3u8`), listed in the master
  playlist as an `EXT-X-MEDIA:TYPE=SUBTITLES` group that every variant names
  with `SUBTITLES="subs"`. The subtitle segments sit on the video's segment
  grid — same boundaries, same `EXTINF` durations — and a cue that spans a
  boundary is repeated on both sides, per RFC 8216 §3.5. The first rendition
  is `DEFAULT=YES`.

Styling is not preserved: `tx3g` keeps style in side boxes keyed by byte range
rather than inline, so ASS override blocks (`{\an8}`), SRT/WebVTT tags
(`<i>`, `<font>`), and the ASS field prefix are stripped down to the text. Cue
timing is preserved; in `tx3g`, cue gaps become empty samples so the timeline
stays aligned.

Trims and splices carry them too: `--trim-start`/`--trim-end` clip the cues to
the kept window and re-base them to zero, and [`rivet splice`](#rivet-splice)
moves each clip's cues onto the joined timeline and merges tracks by language.

### Output layout

- **single** — one MP4 per rung. One rung → the `-o` file (faststart AV1 + audio;
  default `<input-stem>.av1.mp4`, whatever the codec). Multiple rungs (or
  `--ladder`) → `-o` is a directory (default `<input-stem>.av1/`) holding a
  `<label>.mp4` per rung.
- **audio** — one file at `-o` (default `<input>.mp3`, or `.flac` / `.m4a` for
  lossless audio, see `--audio-container`); no video.
- **hls** — `-o` is the asset root (default `<input-stem>.hls/`): `master.m3u8`, an `audio/` rendition group,
  and `video/<height>p/{init.mp4, seg-*.m4s, playlist.m3u8}` per rung,
  segment-aligned across the ladder for clean ABR.

### Examples

```sh
# Single MP4 at the source resolution
rivet transcode input.mkv -o output.mp4

# Explicit 3-rung ladder → a directory of MP4s
rivet transcode input.mkv -o out_dir/ --rung 1920x1080 --rung 1280x720 --rung 640x360

# Auto ABR ladder capped at 1080p short side
rivet transcode input.mkv -o out_dir/ --ladder --max-short-side 1080

# CMAF/HLS package, 4 s segments
rivet transcode input.mkv -o hls_dir/ --mode hls --ladder --segment-seconds 4

# Quality + audio + frame-rate knobs
rivet transcode input.mkv -o out.mp4 --crf 28 --audio opus --max-fps 30

# Lossless: FLAC beside the video, or the audio alone as a .flac / .m4a
rivet transcode input.mkv -o out.mp4 --audio flac
rivet transcode master.mkv -o master.flac --mode audio --audio flac --flac-compression best
rivet transcode album.flac -o album.m4a --mode audio --audio alac --audio-bit-depth 24

# Re-encode 5.1 audio to Opus at 240 kbps, re-tagging side surrounds as back
rivet transcode input.mkv -o out.mp4 --audio opus --audio-bitrate 240k \
  --audio-filter 'channelmap=FL-FL|FR-FR|FC-FC|LFE-LFE|SL-BL|SR-BR:5.1'

# Non-local-means denoise with explicit patch / research-window sizes
rivet transcode input.mkv -o out.mp4 --filter 'nlmeans=s=8:p=7:pc=5:r=9:rc=9'

# Temporal denoise (luma/chroma spatial, luma/chroma temporal strengths)
rivet transcode input.mkv -o out.mp4 --filter 'hqdn3d=4:3:6:4.5'

# Keep every text subtitle track (the default), only some languages, or none
rivet transcode input.mkv -o out.mp4 --subtitles eng,deu
rivet transcode input.mkv -o out.mp4 --subtitles none

# Pin to one GPU / one vendor / decode elsewhere
rivet transcode input.mkv -o out.mp4 --gpu 1
rivet transcode input.mkv -o out.mp4 --encode family:nvidia --decode gpu:0

# Benchmark decoders up front and decode on the fastest GPU (multi-GPU hosts)
rivet transcode input.mkv -o out.mp4 --decode fastest

# ProRes 422 HQ in a QuickTime movie; VP9 in WebM; MPEG-2 in a .mov
rivet transcode input.mkv --codec prores-hq             # -> input.prores.mov
rivet transcode input.mkv --codec vp9 -o out.webm
rivet transcode input.mkv --codec mpeg2 --container mov -o out.mov

# HDR10 passthrough (AV1 needs a GPU build with AV1 encode; see Color & bit depth)
rivet transcode input.mkv -o out.mp4 --color hdr10 --pixel-format 10bit

# Splice/trim: cut a single input to [2s, 7s)
rivet transcode input.mkv -o cut.mp4 --trim-start 2 --trim-end 7
```

---

## `rivet splice`

**Concatenate** (and per-clip **trim**) several inputs into one output. Clips are
joined in order; each is decoded with its own decoder, trimmed to its window,
and the kept frames are re-encoded into one continuous, zero-based timeline (the
muxer numbers frames by count, so the join is gap-free with no PTS rewriting).
Because everything is re-encoded to a uniform output, the inputs **may differ**
in codec, resolution, or color — output config follows the **first** clip. Audio
is trimmed per clip and concatenated to match. Outputs a single MP4
(`--mode single`, the default) or a CMAF/HLS package (`--mode hls`) — for HLS the
spliced frame stream feeds the same multi-GPU engine as a normal ladder, so
segments stay keyframe-aligned across the join.

```
rivet splice -o <OUTPUT> [OPTIONS] <CLIP>...
```

`<OUTPUT>` is a file for `--mode single`, or a directory for `--mode hls`. Each
`<CLIP>` is a path, or `PATH@START-END` to trim it (seconds, either side
optional). `@` is the separator so a Windows drive `C:\…` is unambiguous:

| Clip spec | Meaning |
|-----------|---------|
| `a.mp4` | the whole clip |
| `a.mp4@2-7` | seconds `[2, 7)` |
| `a.mp4@2-` | from 2 s to the end |
| `a.mp4@-7` | from the start to 7 s |

| Flag | Values / default | Description |
|------|------------------|-------------|
| `-o`, `--output <PATH>` | required | Output MP4 file (`single`) or directory (`hls`). |
| `--mode <MODE>` | `single` *(default)*, `hls` | Output shape: one MP4, or a CMAF/HLS package. (`audio` is refused for a splice.) |
| `--segment-seconds <S>` | default `4.0` | HLS target segment length (`--mode hls` only). |
| `--codec <CODEC>` | `av1` *(default)*, `h264`, `h265` | Output video codec (as for `transcode`). |
| `--crf <N>` | encoder-native | Constant rate factor. |
| `--target <TARGET>` | `standard` *(default)*, `visually_lossless`, `high`, `low`, `vmaf=N` | Perceptual quality target, as for `transcode`. |
| `--gop <N\|Ns>` | two seconds | GOP length in frames, or seconds (`2s`) (alias `--keyframe-interval`). |
| `--color <POLICY>` | `sdr` *(default)*, `hdr10`, `hlg`, `passthrough` | Output colour, as for `transcode`. The output follows the first clip; later clips are mapped into it. |
| `--pixel-format <DEPTH>` | `auto` *(default)*, `8bit`, `10bit` | Output bit depth, as for `transcode`. `8bit` is how a 10-bit first clip is joined into 8-bit H.264 on a build whose H.264 encoder is 8-bit — the remedy the depth refusal names. |
| `--chroma-downsample <FILTER>` | `box` *(default)*, `lanczos` | 4:4:4 → 4:2:0 chroma filter for 4:4:4 clips. |
| `--filter <CHAIN>` | none | Video filter chain applied to every clip before scaling, as for `transcode`. |
| `--video-bitrate <BPS>` / `--video-buffer <DURATION>` | e.g. `3M` / `500ms` | Code the output to a rate, with its coded picture buffer (1 s unless given), as for `transcode`. |
| `--rate-mode <MODE>` | `average` *(default)*, `cbr` | Average or constant rate, as for `transcode`. |
| `--video-speed <TIER>` | `standard` *(default)*, `draft`, `archive` | Encoder effort, as for `transcode`. |
| `--audio <POLICY>` | `auto` *(default)*, `opus`, `mp3`, `aac`, `flac`, `alac`, `drop` | Audio handling. |
| `--audio-bitrate <BPS>` | derived | Bitrate for transcoded audio (ignored for passthrough). |
| `--audio-channels <LAYOUT>` | `source` | Output channel layout, as for `transcode`. |
| `--audio-filter <CHAIN>` | none | Audio filter chain before the Opus encoder, as for `transcode`. |
| `--subtitles <SELECTION>` | `all` *(default)*, `none`, `eng,deu` | Subtitle tracks to carry, as for `transcode`. Each clip's cues are clipped to its trim window, moved onto the joined timeline, and merged by language. |
| `--decode <PLAN>` | `auto` *(default)*, `whole`, `fastest`, `gpu:N`, `ranges:N` | The decode plan, as for `transcode` (`--decode-gpu N` still works). |
| `--encode <PLAN>` | `all` *(default)*, `per-rung`, `single`, `gpu:N`, `family:VENDOR` | The encode plan, as for `transcode`. A splice always takes the serial encode path, so here it chooses the card. |

> Text subtitles come along: each clip's cues are trimmed with the clip,
> shifted by the length of the clips before it, and tracks join by language
> (clip 2's `eng` continues clip 1's `eng`; a language only some clips have
> is carried where it exists).

### Examples

```sh
# Join three clips end-to-end
rivet splice -o out.mp4 intro.mp4 body.mkv outro.mov

# Join with per-clip trims (first 5 s of A, then 10–20 s of B, then all of C)
rivet splice -o out.mp4 a.mp4@0-5 b.mp4@10-20 c.mp4 --codec h265

# A single trimmed clip is just a trim (same as transcode --trim-*)
rivet splice -o cut.mp4 a.mp4@2-7

# Concatenate straight into an HLS package
rivet splice -o out_hls/ --mode hls a.mp4 b.mp4 c.mp4 --codec h265
```

> The library equivalents are `rivet::run_splice_job(Vec<Clip>, &spec, …)` and
> `OutputSpec::with_trim(start, end)` for the single-input case.

---

## `rivet image`

*(the `image` feature)* Still images of a still image, or stills from a video:

```sh
rivet image <INPUT> -o <DIR> [--format avif,jpeg,png] [--rung WxH[:fit]]...
            [--fit contain|cover|pad|stretch] [--orientation auto|fixed] [--upscale]
            [--quality 1-100|FORMAT:N,...] [--lossless] [--keep-icc] [--speed 1-10]
            [--frames poster | --frames-at SECONDS,... | --frames-count N]
            [--image-decode-deny heic]
```

| Flag | Values / default | Description |
|------|------------------|-------------|
| `-o`, `--output <DIR>` | required | Output directory (created if missing). |
| `--format <FORMATS>` | `avif` *(default)*, `jpeg` (or `jpg`), `png`; `webp` is refused | Comma-separated; every size is made in each. `webp` is refused by name until rivet's own WebP codec (rivet-webp) lands: "WebP is not available in this build … Ask for avif, jpeg or png". |
| `--rung <WxH[:FIT…]>` | repeatable, or comma-separated | A box the picture is fitted into; a fit, `auto` / `fixed` and `upscale` / `no-upscale` may follow after `:`. No `@RATE`. None: one output at the picture's own size. |
| `--fit`, `--orientation`, `--upscale` | as for `transcode` | How each box is filled. |
| `--quality <Q>` | AVIF 60, JPEG 82 (WebP 80, once it lands) | 1–100 for the lossy formats: one for every format (`70`), one per format (`avif:60,jpeg:82`), or both (`70,jpeg:82`). |
| `--lossless` | flag | Lossless WebP (refused with AVIF or JPEG); accepted, but it only matters for WebP, which is pending. PNG is lossless anyway. |
| `--keep-icc` | flag | Keep the source's colour profile instead of converting to sRGB (PNG and JPEG carry it; AVIF is always converted, as rivet's AVIF writer writes no ICC). |
| `--speed <N>` | `6` | PNG compression effort, 1 (slowest, smallest) to 10 (fastest): DEFLATE level 9, 8, 7, 6, 6, 6, 5, 4, 3, 1 for 1 to 10, so the default is level 6. AVIF and JPEG ignore it. (Until 2026-10-03 it was AVIF's encoder effort.) |
| `--frames poster` | the default | States the default selection: a still image as it is, one frame 10% into a video. |
| `--frames-at <SECONDS>` | comma list | A video input: stills at these times. |
| `--frames-count <N>` | — | A video input: N evenly spaced stills. |
| `--image-decode-deny <FORMATS>` | e.g. `heic` | Still-image input formats not to decode. |

Inputs: JPEG, PNG, AVIF, GIF (first frame), TIFF, BMP, HEIC — or a video,
whose stills `--frames-at` / `--frames-count` pick (one frame 10% in without
either). Each `--rung` is a box, fitted as a video rung is but to the pixel;
without one, the output is the picture's own size. Files are `<W>x<H>.<ext>`,
or `<W>x<H>-<nnn>.<ext>` for several stills. Every output is upright, sRGB
(unless `--keep-icc`) and free of EXIF/XMP/GPS. A WebP input is refused by
name, as WebP output is, until rivet-webp lands. See
[output-spec.md §11](output-spec.md#11-still-images--modeimage).

```sh
rivet image photo.heic -o out --format avif,jpeg,png --rung 1920x1920 --rung 640x640
rivet image talk.mp4 -o stills --format jpeg --frames-count 12 --rung 320x320
```

## `rivet probe`

```
rivet probe <INPUT> [--json]
```

Inspect a file without transcoding. `--json` emits a machine-readable object
(`container`, `video_codec`, `width`, `height`, `frame_rate`, `duration`,
`pixel_format`, `audio` — `{codec, sample_rate, channels}` or `null` — and
`subtitles` — `[{codec, language, cues}]`); otherwise a human summary is
printed.

```sh
rivet probe input.mkv
rivet probe input.mkv --json
```

---

## `rivet devices`

```
rivet devices [--json]
```

List the GPUs rivet detects on this host — vendor, name, generation, VRAM, PCI
address, PCI BAR, which of AV1 / H.264 / H.265 each card can encode in this
build, and (NVIDIA only, via NVML) a live load snapshot (GPU / encoder /
decoder utilization, memory, temperature). `--json` emits
`{ "gpus": [ { index, vendor, name, generation, vram_mib, pci, av1_encode,
encode: { av1, h264, h265 }, pci_bar, load? } ] }`.

```sh
rivet devices
rivet devices --json
```

**PCI BAR** (Linux, discrete cards): whether the CPU can reach all of the
card's VRAM, which is whether Resizable BAR is in effect. Without it the
window is the PCI default, 256 MiB. Encode and decode don't need it, but
Intel's compute runtime won't expose an Arc card behind a small window, so
OpenCL, Level Zero and OpenVINO's GPU plugin can't use the card even though
QSV can. When the window is small, the line says what's in the way: the card
can resize it but the platform hasn't, the card offers no Resizable BAR to
this host, or this is a VM whose hypervisor may hide it.

```text
      PCI BAR    : small (256 MiB of 6144 MiB VRAM): the card supports Resizable BAR, but the platform hasn't enabled it; enable Above 4G Decoding and Resizable BAR in the firmware
                   Intel's compute runtime won't expose this card: no OpenCL, Level Zero or OpenVINO GPU (QSV is unaffected)
```

It's read from sysfs, without privileges. The sizes the card supports come
from the kernel's `resourceN_resize`, which exists since Linux 6.1. In
`--json`, `pci_bar` is `{ bar, bytes, vram_mib, full, resizable, max_bytes,
virtualised }`, or `null` (an integrated GPU, or not Linux). `full` is `null`
when the VRAM is unknown, and `resizable` is `null` on kernels older than 6.1,
which don't say.

This is **hardware inventory** — what's plugged in. What this *build* can actually
do with it is [`rivet capabilities`](#rivet-capabilities-alias-caps) (it depends on which
GPU feature the binary was compiled with).

## `rivet capabilities` (alias `caps`)

```
rivet capabilities [--json]
rivet caps [--json]
```

Report what this **build + host** can do:

- **Encode** — AV1 / H.264 / H.265 4:2:0: the compiled backends
  (`nvenc` / `amf` / `qsv` / `av1` / `h26x`), then **by codec** the bit depth
  (8 or 10) and whether HDR (PQ/HLG, BT.2020) is producible for each output
  codec, with each compiled backend's own answer: H.264 is 8-bit SDR on every
  hardware backend and 10-bit HDR only on `h26x`; AV1 is 10-bit SDR on the
  software `av1` encoder. The
  by-codec answer is what `rivet transcode` checks `--color` /
  `--pixel-format` against (`rivet::spec::CodecOutputCaps`). The last line,
  `every codec`, is what every output codec meets (the lowest depth, HDR only
  when every codec has it). `--json` carries the same numbers: `encode.by_codec`
  —
  `[{"codec","max_bit_depth","hdr","backends":[{"backend","max_bit_depth","hdr"}]}]`
  — and `encode.max_bit_depth` / `encode.hdr` for `every codec`. Until
  2026-09-14 the JSON fields were the union — the best codec's answer — and
  said 10-bit HDR on an `h26x-fallback`-only build, which has no AV1 encoder;
  the text report led with that union (`max depth` / `HDR` lines) until
  2026-09-18. Read `by_codec` for one codec's answer.
- **Decode** — a codec → backends table (which of the compiled decode
  backends decode `h264` / `hevc` / `vp8` / `vp9` / `av1` / `mpeg2` / `mpeg4`
  / `prores`; `--json` also lists the backends, `decode.backends`, in dispatch
  order: `nvdec`, `amf`, `qsv`, `h26x`, `prores`, `vp8`, `vp9`, `mpeg2`,
  `mpeg4`, `av1`, `openh264`, those compiled in). `h26x`
  (H.264 and HEVC), `prores`, `vp8`, `vp9`, `mpeg2` (MPEG-2 and MPEG-1 video),
  `mpeg4` (MPEG-4 Part 2) and `av1` are rivet's own software decoders and are
  in every build; `openh264` decodes H.264 only, when its feature is built. `prores` is the only backend that decodes ProRes.
- **Devices** — a one-line summary of the detected GPUs.

A backend only appears if its **feature was compiled in** (`--features nvidia`
etc.); the actual silicon (e.g. NVENC AV1 needs Ada+) is verified at encode time.

```sh
cargo build --release --features qsv
rivet capabilities            # Encode: qsv 10-bit HDR · Decode: h264/hevc/av1/vp9 → qsv
rivet caps --json
```

---

## `rivet pipe`

```
rivet pipe [--crf N] [--target T] [--gop FRAMES|SECONDSs]
           [--video-bitrate BPS] [--video-buffer DURATION] [--rate-mode average|cbr]
           [--video-speed draft|standard|archive]
           [--audio auto|opus|mp3|aac|he-aac|he-aacv2|vorbis|ac3|eac3|dts|flac|alac|drop] [--audio-bitrate BPS]
           [--audio-channels source|mono|stereo|5.1|7.1] [--audio-filter CHAIN]
           [--color sdr|hdr10|hlg|passthrough] [--bit-depth auto|8bit|10bit]
           [--chroma-downsample box|lanczos] [--max-fps FPS|source]
           [--width W] [--height H] [--fit FIT] [--orientation auto|fixed] [--upscale]
           [--gpu I] [--decode PLAN] [--encode PLAN] [--filter CHAIN]
```

Stream a transcode through standard I/O: read media from **stdin**, write the
AV1/MP4 to **stdout** (progress goes to stderr so stdout stays clean). With no
flags it's the zero-config transcode (`rivet::transcode_bytes`: source
resolution, AV1, audio passthrough, and `rivet transcode`'s default picture —
an HDR source tonemapped to 8-bit SDR, an SDR source at its own depth). A 10-bit
SDR source on a build whose only AV1 encoder is 8-bit is refused before
anything is decoded, naming the setting that narrows it: `--pixel-format 8bit`
(alias of `--bit-depth`), which sends the job through the job engine, as any
flag does. (Until 2026-09-18 it asked rav1e — then the software AV1 encoder, 8-bit
only — for 10-bit AV1 and failed with "no Av1 encoder available …
rebuild with `--features rav1e-fallback`". The software AV1 encoder that
replaced rav1e on 2026-10-03 codes 10-bit SDR itself.) The flags override per
job, each meaning what the [`transcode`](#rivet-transcode) flag of that name
means — `--width/--height` scale (a box, fitted as `--fit` says),
`--color/--bit-depth` set HDR/depth, `--crf/--target` set quality:

```sh
cat input.mkv | rivet pipe > output.mp4                       # defaults
cat input.mkv | rivet pipe --crf 28 --width 1280 --height 720 > out.mp4
ffmpeg -i src.mov -f matroska - | rivet pipe --color hdr10 | ./my-uploader
```

Single MP4 only — for an HLS package or a multi-rung ladder use
[`transcode`](#rivet-transcode) with a directory output, or the
[HTTP API](api.md).

## `rivet batch`

```
cargo build --release --features batch   # opt-in
rivet batch <MANIFEST> [--dry-run] [--stop-on-error]
```

Convert **many files in one run** from a YAML or JSON **manifest** — you list the
files (and how), rivet does them. Each job is an input (file or glob), an output,
and any transcode setting, on top of optional shared `defaults`. `--dry-run`
parses + expands globs + lists the planned jobs without converting; `--stop-on-error`
aborts on the first failure (default keeps going and exits non-zero if any failed).

```sh
rivet batch jobs.yaml --dry-run
rivet batch jobs.yaml
```

```yaml
output_dir: out
defaults: { crf: 28, color: sdr }
jobs:
  - input: in/a.mkv
    output: out/a.mp4
    crf: 24
  - input: "clips/*.mp4"   # glob -> one job per file -> out/<name>.mp4
    output: out/
```

**Full DSL reference: [batch.md](batch.md)** — every key, the output-path rules,
glob inputs, defaults merge, and JSON examples. A ready-to-edit manifest is in
[`examples/batch.yaml`](../examples/batch.yaml) / [`.json`](../examples/batch.json).

## `rivet ipc`

```
cargo build --release --features ipc   # opt-in; the subcommand only exists in an ipc build
rivet ipc --socket <PATH>
```

Run a **Unix-domain-socket** server (opt-in `ipc` feature; Unix only at runtime)
so a long-running application can stream jobs in and out without spawning a
process per file or going through HTTP. `rivet pipe` (stdin/stdout streaming) is
always available and needs no feature.
Bind a socket, then for **each connection**: the client optionally writes a
**settings header line**, then the input media, **half-closes** its write side
(signals end-of-input), and reads the transcoded AV1/MP4 back until EOF. One
thread per connection; the process-wide GPU pool serializes the actual GPU work,
so concurrent clients simply queue.

**Settings header** (optional): if the stream begins with `#rivet`, the first
line is parsed as space-separated `key=value` settings and stripped before
decode. The keys are the shared `TranscodeSettings` vocabulary — the same names
as the CLI flags (`mode` `rung` `fit` `orientation` `upscale` `ladder`
`max-short-side` `segment-seconds` `crf` `target` `gop` `video-bitrate`
`video-buffer` `rate-mode` `video-speed` `audio` `audio-bitrate` `audio-channels`
`audio-stereo-fallback` `audio-bit-depth` `he-aac` `audio-decode-deny`
`metadata-keep` `flac-compression` `audio-container` `audio-filter`
`subtitles` `color` `chroma-downsample` `bit-depth` `seam` `max-fps` `encode`
`decode` `gpu` `gpu-family` `single-gpu` `decode-gpu` `encode-policy` `width`
`height` `filter` `codec`; `rung` takes a comma list), with the same values and
the same meaning — a `#rivet encode=per-rung decode=whole` header is exactly
`--encode per-rung --decode whole`. Real container
magic bytes never start with `#rivet`, so a raw media stream without a header
just gets the defaults. (A single socket connection produces one MP4, so
`mode=hls`/multi-rung isn't supported here — use the HTTP API for that.)

```
#rivet crf=28 color=hdr10 width=1280 height=720\n
<media bytes…>
```

```sh
rivet ipc --socket /tmp/rivet.sock &
# any client that does write → shutdown(WR) → read works, e.g. socat (no header):
socat - UNIX-CONNECT:/tmp/rivet.sock < input.mkv > output.mp4
```

A minimal client with settings (Python):

```python
import socket
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect("/tmp/rivet.sock")
s.sendall(b"#rivet crf=28 width=1280 height=720\n")   # optional settings header
s.sendall(open("input.mkv", "rb").read())
s.shutdown(socket.SHUT_WR)                             # end-of-input
out = b"".join(iter(lambda: s.recv(65536), b""))
open("output.mp4", "wb").write(out)                    # AV1/MP4
```

Single MP4 per connection. On **Windows** `rivet ipc` is unavailable — use
[`rivet pipe`](#rivet-pipe) (stdin/stdout) or [`rivet serve`](#rivet-serve)
(HTTP).

---

## `rivet serve`

```
rivet serve [--addr <ADDR>]
```

Runs the HTTP transcode API (requires a `--features server` build). `--addr`
defaults to `127.0.0.1:8080`. See the [HTTP API reference](api.md) for endpoints.

```sh
cargo build --release --features server,nvidia
rivet serve --addr 0.0.0.0:8080
```

---

## Environment variables

| Variable | Effect |
|----------|--------|
| `RUST_LOG` | Log filter, e.g. `RUST_LOG=debug` or `RUST_LOG=rivet=info`. |
| `TRANSCODE_ENCODER_BACKEND` | Force an encoder backend on the serial single-file path: `nvenc` \| `amf` \| `qsv` \| `h26x` \| `av1` (`rav1e` is still accepted for `av1`) \| `prores` \| `vp8` \| `vp9` \| `mpeg2` \| `mpeg4`. |
| `RIVET_SOFTWARE_SLOTS` | Number of software encoder slots in the software pool (derived from the host by default; clamped to `1..=` the available parallelism). |
| `RIVET_FORCE_CHUNKED` | `1` runs the chunk-and-stitch engine on a one-GPU host, to exercise the chunked path (no speedup). |
| `RIVET_FILE_ROOT` | `rivet serve`: confine the JSON body's server-side `input.path` / `output.path` to this directory. |
| `LIBVA_MESSAGING_LEVEL` | rivet sets it to `0` (libva errors only) unless it is already set; set it yourself (e.g. `2`) to see libva's driver messages. |
| `DISABLE_NVDEC` | Skip NVDEC for every codec (fall through to the next decode tier). |
| `DISABLE_NVDEC_<CODEC>` | Skip NVDEC for one family, e.g. `DISABLE_NVDEC_AV1=1`. |
| `RIVET_AV1_DECODE_THREAD` | `0` makes the software AV1 decoder decode on the caller's thread instead of its own worker, which otherwise runs a few frames ahead so the rest of the pipeline overlaps the decode. |
| `RIVET_TEST_MEDIA` | Integration tests: directory of real media to run against. |
