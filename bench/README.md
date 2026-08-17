# Quality bench — VMAF against a source

Three scripts and one rule: **a ladder change is not a result until it has been
scored against a source.**

Everything here exists because reasoning about encoder settings from published
BD-rate figures has been wrong, in production, more than once. A ladder policy
tuned that way spent 23% of the largest rung buying 0.62 VMAF. A per-title
sweep certified a quality shift against a floor it then missed, and cost up to
13.6 VMAF per rung before anybody scored the output. Every number quoted in
this repository's docs — the recommended `RungPolicy`, the per-title floor,
the absorbed near-duplicate rung — was measured with this harness or its
predecessor.

The two halves of "VMAF as perception" live in two places on purpose:

- **Targeting** is in-process. `QualityTarget::Vmaf(n)` (CLI `--target
  vmaf=93`, policy grammar `target=vmaf=93`) maps a VMAF score to each
  backend's quantiser through calibrated tables
  (`codec::encode::tuning`, `docs/av1-tuning-research.md`). The in-process
  sweep (`rivet::per_title`, `codec::bench`) ranks candidates by SSIM, because
  scoring one clip's candidates against each other does not need VMAF's model
  and vendoring libvmaf into every worker is a cost nobody wants.
- **Measuring** is out-of-process, here, with libvmaf inside a container. It is
  the check on whether the tables, the floor and the policy are set sensibly.

## `generate-corpus.sh`

Four 20-second 1080p clips, one per content type a ladder has to cope with:

| clip | what it is | why it is in the set |
|---|---|---|
| `grain.mp4` | detail under heavy sensor-like noise | the hard end; the case AV1 film-grain synthesis exists for |
| `flat.mp4` | large solid regions, hard edges, no texture | the easy end; where per-title should find real headroom |
| `motion.mp4` | continuous rotation and zoom | high temporal energy, which starves inter prediction |
| `dark.mp4` | low luma with detail still in it | most easily ruined by an eager quantiser, and must not be mistaken for a blank frame |

At the same CRF these differ by roughly **2000×** in size — flat encodes to
about 260 KB and grain to over 500 MB. That spread is the point: a single
quality setting cannot be right for both ends of it, which is the entire premise
of per-title encoding. Measuring on one clip measures one point on that range.

Each clip **opens on a three-second fade from black**, deliberately. Sampling
that is a live bug class in both the in-process sweep and this harness — black
frames cost nothing to encode and score near-perfectly against themselves, so a
sample taken there reports that the content is free. The corpus should exercise
the defence, not assume it.

```sh
docker run --name corpusgen --entrypoint /bin/bash linuxserver/ffmpeg:latest \
  -c "$(cat generate-corpus.sh)"
docker cp corpusgen:/tmp/corpus ./corpus
```

Regenerated rather than committed: the sources are `lavfi` generators, so the
script is smaller than the output by six orders of magnitude and cannot go
stale against it.

## `score-ladder.sh`

Scores each encoded rung against the source it came from, with VMAF and SSIM.
Takes `rivet`'s output as it is — an HLS package (`video/<label>/init.mp4` +
`seg-*.m4s`) or a directory of single-file rungs (`<label>.mp4`).

Two things it does that are easy to get wrong, and that give wrong answers
quietly rather than loudly:

- **Every segment, in order.** Concatenating only `seg-00001.m4s` scores the
  opening seconds of a rung and calls it the ladder.
- **The rung is compared upscaled to source size.** A 240p rung is watched
  stretched to a screen, not in a 240-pixel-tall window. Comparing it against a
  240p reference flatters it by hiding exactly the detail the ladder traded
  away. The upscale is a measurement artifact — it exists for the comparison and
  is discarded; nothing above source is ever encoded or stored.

It also scores a window from the **middle**, chosen with `blackdetect` and
walked forward if the middle lands in a black stretch, for the fade reason
above. Clips shorter than one window are scored whole and it says so.

```sh
# expects source.mp4 and rungs/ next to the Dockerfile
docker build -t rivet-vmaf . && docker run --rm rivet-vmaf
```

## `run-ladder.sh`

The whole loop in one command: `rivet transcode` a clip into a ladder, then
score it. Everything after the label is handed to `rivet transcode`, so any
knob is a data point:

```sh
./run-ladder.sh corpus/flat.mp4 baseline
./run-ladder.sh corpus/flat.mp4 recommended --encode-policy recommended
./run-ladder.sh corpus/flat.mp4 vmaf93     --target vmaf=93
./run-ladder.sh corpus/flat.mp4 gop30      --gop 30
MODE=single ./run-ladder.sh corpus/grain.mp4 single-constqp --seam-mode constqp
```

`RIVET` (default `rivet` on `PATH`), `MODE` (`hls` default, or `single`) and
`SEGMENT_SECONDS` (default 4) are the environment knobs. Results land in
`result-<label>.txt`; compare two labels with `diff` or `paste`.

## What a result means

`bytes` is the rung's whole payload; `vmaf` and `ssim` are the mid-clip window
against the source. Read them together: −20% bytes for −0.5 VMAF on the top
rung is a good trade, +23% bytes for +0.6 VMAF is the trade the recommended
policy stopped making. Anything under about VMAF 93 on the top rung is
resolution- or content-limited and no amount of bitrate will fix it — check
that the ladder kept the source's resolution before spending bits.
