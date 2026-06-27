# rivet documentation

Reference pages. The top-level [README](../README.md) is the quick tour;
**[architecture.md](architecture.md) is the start-here map** of the codebase.

## Understand the system

| Page | What |
|------|------|
| [architecture.md](architecture.md) | **Start here** — the system map: the three crates, the transcode lifecycle, the two execution paths, and how the front-ends fit. |
| [decisions.md](decisions.md) | **The why** — the load-bearing design decisions (AV1-only output, no-FFmpeg/hand-rolled FFI, GPU scheduling, streaming, HDR→SDR, web-ready defaults) and their rationale. |
| [pipeline.md](pipeline.md) | The end-to-end data flow — demux → decode-once pump → per-rung scale → multi-GPU lease engine → mux, with diagrams + a code map. |

## Code references (what + why, per crate)

| Page | What |
|------|------|
| [codec-decode.md](codec-decode.md) | The `codec` crate, decode side: dispatch tiers, each GPU decoder, GPU detection, bitstream parsers, probe, HDR/SEI. |
| [codec-encode.md](codec-encode.md) | The `codec` crate, encode side: encoder dispatch, each HW backend, quality tuning, colorspace, tonemapping, audio. |
| [container.md](container.md) | The `container` crate: demuxers (streaming + per-format), Annex-B conversion, the AV1 MP4 muxer, CMAF/HLS, audio glue. |
| [engine.md](engine.md) | The `rivet` crate internals: the job engine, the reactive multi-GPU scheduler, progress, and the CLI/HTTP/IPC front-ends. |

## Use it

| Page | What |
|------|------|
| [output-spec.md](output-spec.md) | **Configuring a transcode** — the complete `OutputSpec` guide: every builder method, enum, and field, plus how to run a job. |
| [filters.md](filters.md) | **Video filters** — the filter set (crop / pad / flip / rotate / grayscale), the string + structured-object forms, and per-surface usage. |
| [batch.md](batch.md) | **Batch manifest DSL** — convert many files from one YAML/JSON file (`rivet batch`): the manifest shape, every key, glob inputs, output rules, and examples. |
| [cli.md](cli.md) | `rivet` CLI reference — every subcommand, flag, and environment variable, with examples. |
| [api.md](api.md) | HTTP transcode API (`rivet serve`) — endpoints, request bodies, the job lifecycle, and the OpenAPI / Swagger / Redoc docs. |
