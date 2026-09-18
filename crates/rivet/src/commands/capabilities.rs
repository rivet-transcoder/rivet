//! Implementation of `rivet capabilities` / `rivet caps`.

use codec::encode::software_encode_available;
use codec::frame::VideoCodec;
use rivet::spec::{
    CodecOutputCaps, OUTPUT_CODECS, encode_backend_name, every_codec_output_caps,
    output_caps_label, output_codec_label,
};

pub(crate) fn run(json: bool) {
    let enc = codec::encode::encode_backends();
    let dec_backends = codec::decode::decode_backends();
    // Per output codec, over the compiled backends: what `OutputSpec::validate`
    // checks a job's `--color` / `--pixel-format` against.
    let by_codec: Vec<CodecOutputCaps> = OUTPUT_CODECS
        .iter()
        .map(|&c| CodecOutputCaps::of_this_build(c))
        .collect();
    let dec = codec::decode::decode_capabilities();
    let devices = codec::gpu::detect_gpus();

    if json {
        let enc_b = enc
            .iter()
            .map(|b| format!("\"{b}\""))
            .collect::<Vec<_>>()
            .join(",");
        let dec_b = dec_backends
            .iter()
            .map(|b| format!("\"{b}\""))
            .collect::<Vec<_>>()
            .join(",");
        let codecs = dec
            .iter()
            .map(|d| {
                let bs = d
                    .backends
                    .iter()
                    .map(|b| format!("\"{b}\""))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{{\"codec\":\"{}\",\"backends\":[{}]}}", d.codec, bs)
            })
            .collect::<Vec<_>>()
            .join(",");
        let plan = rivet::multigpu::host_software_pool_plan();
        // `max_bit_depth` / `hdr` beside `"codec":"av1"` are what every output
        // codec meets, not the best codec's answer; `by_codec` is the
        // per-codec answer.
        let every = every_codec_output_caps(&by_codec);
        println!(
            "{{\"encode\":{{\"codec\":\"av1\",\"backends\":[{}],\"max_bit_depth\":{},\"hdr\":{},\
             \"software\":{{\"av1\":{},\"h264\":{},\"h265\":{},\"slots\":{},\"threads\":{},\"parallelism\":{}}},\
             \"by_codec\":{}}},\
             \"decode\":{{\"backends\":[{}],\"codecs\":[{}]}},\"devices\":{}}}",
            enc_b,
            every.max_bit_depth,
            every.hdr,
            software_encode_available(VideoCodec::Av1),
            software_encode_available(VideoCodec::H264),
            software_encode_available(VideoCodec::H265),
            plan.slots,
            plan.threads,
            plan.parallelism,
            by_codec_json(&by_codec),
            dec_b,
            codecs,
            super::devices::devices_json(&devices)
        );
        return;
    }

    println!("rivet capabilities\n");
    print!("{}", encode_report(&enc, &by_codec));
    // The software tiers, and what a host with no usable encode silicon
    // gets from them: the ladder (HLS and chunked single-file) runs on
    // software leases — CPU shares — sized here.
    let yes_no = |b: bool| if b { "yes" } else { "no" };
    println!(
        "  software   : AV1 via rav1e: {} (`rav1e-fallback`) · H.264 / H.265 via h26x: {} (`h26x-fallback`)",
        yes_no(software_encode_available(VideoCodec::Av1)),
        yes_no(software_encode_available(VideoCodec::H264)),
    );
    let plan = rivet::multigpu::host_software_pool_plan();
    println!(
        "  CPU ladder : when no GPU can encode the codec, {} software slot(s) × {} thread(s) \
         ({} available; `{}` overrides the slot count)",
        plan.slots,
        plan.threads,
        plan.parallelism,
        rivet::multigpu::SOFTWARE_SLOTS_ENV,
    );

    println!("\nDecode — codec → backends:");
    if dec_backends.is_empty() {
        println!("  (none) build with a `nvidia` / `amd` / `qsv` / `rav1d-fallback` feature");
    } else {
        for d in &dec {
            let b = if d.backends.is_empty() {
                "—".to_string()
            } else {
                d.backends.join(", ")
            };
            println!("  {:<8} {}", d.codec, b);
        }
    }

    println!("\nDevices — {} detected:", devices.len());
    if devices.is_empty() {
        println!(
            "  (none) CPU-only host — only the software paths can run here: `rav1e-fallback` / \
             `rav1d-fallback` (AV1) and `h26x-fallback` (H.264 / H.265 encode; their decoders \
             are always in). This build encodes in software: AV1 {}, H.264 / H.265 {}.",
            yes_no(software_encode_available(VideoCodec::Av1)),
            yes_no(software_encode_available(VideoCodec::H264)),
        );
    } else {
        for dv in &devices {
            print!(
                "  [{}] {} {}",
                dv.index,
                codec::gpu::manufacturer_label(dv.vendor),
                dv.name
            );
            if dv.vram_mib > 0 {
                print!(" ({} MiB)", dv.vram_mib);
            }
            // Authoritative per-codec encode verdicts (the same probe the
            // encode pool uses to drop incapable cards) — so a pre-Ada NVIDIA
            // shows AV1 "no", and a build without the vendor feature shows
            // "no" for every codec: detected is not usable.
            println!(" · encode: {}", super::devices::encode_verdicts(dv));
        }
    }
}

/// The text report's encode section, down to the codec-agnostic line: the
/// compiled backends, then each output codec's depth and HDR with each
/// backend's own answer, then what every codec meets — the same numbers as
/// `--json`'s `encode.by_codec` and `encode.max_bit_depth` / `encode.hdr`.
///
/// There is no line for the best codec: a depth or HDR only some codec
/// reaches is not what `--color` / `--pixel-format` get for the codec a job
/// names, and the report used to lead with exactly that (10-bit HDR on an
/// `nvidia` build, whose H.264 is 8-bit SDR).
fn encode_report(enc: &[&str], by_codec: &[CodecOutputCaps]) -> String {
    let mut s = String::from("Encode — AV1 / H.264 / H.265 (4:2:0):\n");
    if enc.is_empty() {
        s.push_str(
            "  (none) build with a `nvidia` / `amd` / `qsv` feature, or `rav1e-fallback` \
             (software AV1) / `h26x-fallback` (software H.264 / H.265)\n",
        );
    } else {
        s.push_str(&format!("  backends   : {}\n", enc.join(", ")));
    }
    // What `--color` / `--pixel-format` are validated against for each `--codec`.
    s.push_str("  by codec   : what --color / --pixel-format are checked against (HDR: PQ / HLG, BT.2020, 10-bit)\n");
    for p in by_codec {
        s.push_str(&format!("    {:<5}: {}\n", output_codec_label(p.codec), by_codec_line(p)));
    }
    s.push_str(&format!(
        "  every codec: {} (what any --codec gets; `encode.max_bit_depth` / `encode.hdr` in --json)\n",
        output_caps_label(every_codec_output_caps(by_codec))
    ));
    s
}

/// One codec's text line: the build's answer, then each backend's.
fn by_codec_line(p: &CodecOutputCaps) -> String {
    if p.backends.is_empty() {
        return "no encoder in this build".to_string();
    }
    let each: Vec<String> = p
        .backends
        .iter()
        .map(|&(b, c)| format!("{} {}", encode_backend_name(b), output_caps_label(c)))
        .collect();
    format!("{} ({})", output_caps_label(p.caps), each.join(", "))
}

/// The `encode.by_codec` JSON array: each output codec's capabilities on this
/// build and the compiled backends behind them.
fn by_codec_json(by_codec: &[CodecOutputCaps]) -> String {
    let items: Vec<String> = by_codec
        .iter()
        .map(|p| {
            let backends: Vec<String> = p
                .backends
                .iter()
                .map(|&(b, c)| {
                    format!(
                        "{{\"backend\":\"{}\",\"max_bit_depth\":{},\"hdr\":{}}}",
                        encode_backend_name(b),
                        c.max_bit_depth,
                        c.hdr
                    )
                })
                .collect();
            format!(
                "{{\"codec\":\"{}\",\"max_bit_depth\":{},\"hdr\":{},\"backends\":[{}]}}",
                output_codec_label(p.codec),
                p.caps.max_bit_depth,
                p.caps.hdr,
                backends.join(",")
            )
        })
        .collect();
    format!("[{}]", items.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;
    use codec::encode::EncoderBackend;

    /// The per-codec report carries each backend's own answer, not the
    /// codec-agnostic one: H.264 on NVENC is 8-bit SDR even though NVENC is
    /// 10-bit HDR for the other codecs, and a backend that does not encode
    /// the codec is left out.
    #[test]
    fn by_codec_reports_each_backends_answer_for_the_codec() {
        let set = [
            EncoderBackend::Nvenc,
            EncoderBackend::Rav1e,
            EncoderBackend::H26x,
        ];
        let by_codec: Vec<CodecOutputCaps> = OUTPUT_CODECS
            .iter()
            .map(|&c| CodecOutputCaps::over(c, &set))
            .collect();
        assert_eq!(
            by_codec_json(&by_codec),
            "[{\"codec\":\"av1\",\"max_bit_depth\":10,\"hdr\":true,\"backends\":[\
             {\"backend\":\"nvenc\",\"max_bit_depth\":10,\"hdr\":true},\
             {\"backend\":\"rav1e\",\"max_bit_depth\":8,\"hdr\":false}]},\
             {\"codec\":\"h264\",\"max_bit_depth\":10,\"hdr\":true,\"backends\":[\
             {\"backend\":\"nvenc\",\"max_bit_depth\":8,\"hdr\":false},\
             {\"backend\":\"h26x\",\"max_bit_depth\":10,\"hdr\":true}]},\
             {\"codec\":\"h265\",\"max_bit_depth\":10,\"hdr\":true,\"backends\":[\
             {\"backend\":\"nvenc\",\"max_bit_depth\":10,\"hdr\":true},\
             {\"backend\":\"h26x\",\"max_bit_depth\":10,\"hdr\":true}]}]"
        );
        let h264_hw = CodecOutputCaps::over(VideoCodec::H264, &[EncoderBackend::Nvenc]);
        assert_eq!(by_codec_line(&h264_hw), "8-bit SDR (nvenc 8-bit SDR)");
        let av1_sw = CodecOutputCaps::over(VideoCodec::Av1, &[EncoderBackend::H26x]);
        assert_eq!(by_codec_line(&av1_sw), "no encoder in this build");
    }

    /// The text report says what `--json` says: per codec the numbers of
    /// `encode.by_codec`, and as its one codec-agnostic line the numbers of
    /// `encode.max_bit_depth` / `encode.hdr` (what every codec meets). It used
    /// to lead with the best codec's answer — "max depth 10-bit", "HDR yes" on
    /// an `nvidia` build whose H.264 is 8-bit SDR, and on an
    /// `h26x-fallback` build, which has no AV1 encoder at all.
    #[test]
    fn the_text_report_says_what_the_json_says() {
        let sets: [(&[&str], &[EncoderBackend]); 3] = [
            (&["nvenc"], &[EncoderBackend::Nvenc]),
            (&["h26x"], &[EncoderBackend::H26x]),
            (&["nvenc", "rav1e", "h26x"], &[EncoderBackend::Nvenc, EncoderBackend::Rav1e, EncoderBackend::H26x]),
        ];
        for (names, set) in sets {
            let by_codec: Vec<CodecOutputCaps> = OUTPUT_CODECS.iter().map(|&c| CodecOutputCaps::over(c, set)).collect();
            let text = encode_report(names, &by_codec);
            let json = by_codec_json(&by_codec);
            assert!(!text.contains("best") && !text.contains("max depth"), "{text}");
            for p in &by_codec {
                let label = output_codec_label(p.codec);
                assert!(
                    json.contains(&format!(
                        "{{\"codec\":\"{label}\",\"max_bit_depth\":{},\"hdr\":{},",
                        p.caps.max_bit_depth, p.caps.hdr
                    )),
                    "{json}"
                );
                let line = if p.backends.is_empty() {
                    format!("    {label:<5}: no encoder in this build\n")
                } else {
                    format!("    {label:<5}: {}-bit {} (", p.caps.max_bit_depth, if p.caps.hdr { "HDR" } else { "SDR" })
                };
                assert!(text.contains(&line), "{names:?}: no `{line}` in\n{text}");
            }
            let every = every_codec_output_caps(&by_codec);
            let line = format!("  every codec: {}-bit {} (", every.max_bit_depth, if every.hdr { "HDR" } else { "SDR" });
            assert!(text.contains(&line), "{names:?}: no `{line}` in\n{text}");
        }
        // The NVENC-only build, spelled out: AV1 and H.265 reach 10-bit HDR,
        // H.264 does not, so a job may count on 8-bit SDR whatever its codec.
        let by_codec: Vec<CodecOutputCaps> =
            OUTPUT_CODECS.iter().map(|&c| CodecOutputCaps::over(c, &[EncoderBackend::Nvenc])).collect();
        let text = encode_report(&["nvenc"], &by_codec);
        assert!(text.contains("    h264 : 8-bit SDR (nvenc 8-bit SDR)\n"), "{text}");
        assert!(text.contains("    h265 : 10-bit HDR (nvenc 10-bit HDR)\n"), "{text}");
        assert!(text.contains("  every codec: 8-bit SDR ("), "{text}");
    }
}
