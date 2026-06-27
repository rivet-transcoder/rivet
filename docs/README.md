# rivet documentation

Elaborative reference pages. The top-level [README](../README.md) has the quick
tour + concepts; these pages are the full details.

| Page | What |
|------|------|
| [output-spec.md](output-spec.md) | **Configuring a transcode** — the complete `OutputSpec` guide: every builder method, enum, and field (rungs/quality, audio, color/bit-depth, GPU policy, chunk seams), plus how to run a job. |
| [pipeline.md](pipeline.md) | **Pipeline & architecture** — the crate map and the full data flow: demux → decode-once pump → per-rung scale → the multi-GPU lease engine → mux, with a diagram and a code map. |
| [cli.md](cli.md) | `rivet` CLI reference — every subcommand, flag, and environment variable, with examples. |
| [api.md](api.md) | HTTP transcode API (`rivet serve`) — endpoints, the output-spec query params, the job lifecycle, and the OpenAPI / Swagger / Redoc docs. |

More pages (adding an encoder backend, the build/feature matrix) will land here
over time.
