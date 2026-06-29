# rivet-container

The container layer of the **[rivet](https://crates.io/crates/rivet-transcoder)**
GPU video transcoder: clean-room demuxers (MP4/MOV, MKV/WebM, MPEG-TS, AVI —
streaming, low peak RSS) and muxers (faststart AV1 MP4, fragmented-MP4 CMAF, HLS
playlists). **No FFmpeg** — hand-written parsers and box writers.

Published as `rivet-container`; **imported as `container`** (`use container::…`).
This is an internal crate of the rivet project — see the
**[rivet-transcoder](https://crates.io/crates/rivet-transcoder)** crate and the
[repository](https://github.com/rivet-transcoder/rivet) for the full architecture and
documentation.

## License

Open Encoding Attribution License v1.0 — a source-available (not OSI open-source)
license, royalty-free, with a commercial-attribution requirement. See
[LICENSE.md](LICENSE.md) and [NOTICE](NOTICE).
