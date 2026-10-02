# rivet-container

The container layer of the **[rivet](https://crates.io/crates/rivet-transcoder)**
GPU video transcoder: clean-room demuxers (MP4/MOV, MKV/WebM, MPEG-TS, AVI, bare
MP3 and FLAC — streaming, low peak RSS) and muxers (faststart MP4 with AV1 /
H.264 / H.265 video, audio and subtitles; fragmented-MP4 CMAF; HLS playlists;
`.mp3`, `.flac` and `.m4a` files). It also reads identifying metadata
(location, device, capture time, descriptive) from those containers and from
still images, and writes a chosen subset of it (`metadata`). **No FFmpeg** —
hand-written parsers and box writers. It depends on `rivet-frame` rather than
`rivet-codec`.

Published as `rivet-container`; **imported as `container`** (`use container::…`).
This is an internal crate of the rivet project — see the
**[rivet-transcoder](https://crates.io/crates/rivet-transcoder)** crate and the
[repository](https://github.com/rivet-transcoder/rivet) for the full architecture and
documentation.

## License

Open Encoding Attribution License v1.0 — a source-available (not OSI open-source)
license, royalty-free, with a commercial-attribution requirement. See
[LICENSE.md](LICENSE.md) and [NOTICE](NOTICE).
