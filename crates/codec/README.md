# rivet-codec

The codec layer of the **[rivet](https://crates.io/crates/rivet-transcoder)**
GPU video transcoder: GPU detection (including whether a discrete card's PCI BAR
covers its VRAM, `gpu::bar_report`), decode/encode dispatch (NVDEC/NVENC, AMF,
QSV, then software: `rivet-h26x` for H.264 / HEVC, and `rivet-av1` for AV1;
no FFmpeg), software decode and encode of ProRes,
VP8, VP9, MPEG-1 / MPEG-2 and MPEG-4 Part 2 through `rivet-prores`,
`rivet-vp8`, `rivet-vp9`, `rivet-mpeg2` and `rivet-mpeg4` (always compiled),
colorspace, HDR→SDR tonemapping, video and
audio filters, audio decode and encode through the workspace's own codecs
(Opus through `rivet-opus`, MP3 and MPEG audio through `rivet-mp3`, Vorbis
through `rivet-vorbis`, AAC-LC / HE-AAC through `rivet-aac`, AC-3 / E-AC-3
through `rivet-ac3`, DTS through `rivet-dts`, FLAC and ALAC through
`rivet-lossless`; and PCM decode — no third-party audio codec), and media
probing. Hand-rolled `dlopen`
FFI for every GPU codec vendor — no external wrapper crates; builds on Windows
+ Linux. The frame and stream types come from `rivet-frame` and are re-exported
at their old paths (`codec::frame::*`).

Published as `rivet-codec`; **imported as `codec`** (`use codec::…`). This is an
internal crate of the rivet project — see the
**[rivet-transcoder](https://crates.io/crates/rivet-transcoder)** crate and the
[repository](https://github.com/rivet-transcoder/rivet) for the full architecture and
documentation.

## License

Open Encoding Attribution License v1.0 — a source-available (not OSI open-source)
license, royalty-free, with a commercial-attribution requirement. See
[LICENSE.md](LICENSE.md) and [NOTICE](NOTICE).
