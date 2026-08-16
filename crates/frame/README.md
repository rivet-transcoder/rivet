# rivet-frame

The value types the **[rivet](https://crates.io/crates/rivet-transcoder)**
codec and container layers agree on: `StreamInfo`, `VideoFrame`,
`PixelFormat`, `ColorSpace`, `TransferFn`, `ColorMetadata` (+ HDR static
metadata) and `EncodedPacket`, plus `pixel_format` — the bitstream
introspection (AV1 sequence header, H.264/HEVC SPS, MPEG-2) the demuxers use
to learn a stream's pixel format. Imported as `frame`.

It exists so that `rivet-container` — the demuxers and muxers — does not have
to depend on `rivet-codec`, whose GPU FFI (`dlopen`), NVML and audio-codec
dependencies cannot build for `wasm32-unknown-unknown`. With the types here,
`rivet-container` builds for wasm and a browser can demux MP4/fMP4/MKV/TS
straight into a WebAssembly decoder.

`rivet-codec` re-exports everything at its old paths (`codec::frame::*`,
`codec::pixel_format::*`, `codec::encode::EncodedPacket`), so nothing that
already compiles changes.
