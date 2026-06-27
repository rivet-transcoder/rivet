# rivet-codec

The codec layer of the **[rivet](https://crates.io/crates/rivet-transcoder)**
GPU video transcoder: frame types, GPU decode/encode dispatch (NVDEC/NVENC, AMF,
QSV), colorspace, HDR→SDR tonemapping, audio, and media probing. Hand-rolled
`dlopen` FFI for every GPU vendor — no external wrapper crates; builds on Windows
+ Linux.

Published as `rivet-codec`; **imported as `codec`** (`use codec::…`). This is an
internal crate of the rivet project — see the
**[rivet-transcoder](https://crates.io/crates/rivet-transcoder)** crate and the
[repository](https://github.com/elyerinfox/rivet) for the full architecture and
documentation.

## License

Open Encoding Attribution License v1.0 — a source-available (not OSI open-source)
license, royalty-free, with a commercial-attribution requirement. See
[LICENSE.md](LICENSE.md) and [NOTICE](NOTICE).
