//! Implementation of `rivet capabilities` / `rivet caps`.

use codec::encode::software_encode_available;
use codec::frame::VideoCodec;

pub(crate) fn run(json: bool) {
    let enc = codec::encode::encode_backends();
    let dec_backends = codec::decode::decode_backends();
    let caps = codec::encode::build_output_caps();
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
        println!(
            "{{\"encode\":{{\"codec\":\"av1\",\"backends\":[{}],\"max_bit_depth\":{},\"hdr\":{},\
             \"software\":{{\"av1\":{},\"h264\":{},\"h265\":{},\"slots\":{},\"threads\":{},\"parallelism\":{}}}}},\
             \"decode\":{{\"backends\":[{}],\"codecs\":[{}]}},\"devices\":{}}}",
            enc_b,
            caps.max_bit_depth,
            caps.hdr,
            software_encode_available(VideoCodec::Av1),
            software_encode_available(VideoCodec::H264),
            software_encode_available(VideoCodec::H265),
            plan.slots,
            plan.threads,
            plan.parallelism,
            dec_b,
            codecs,
            super::devices::devices_json(&devices)
        );
        return;
    }

    println!("rivet capabilities\n");
    println!("Encode — AV1 / H.264 / H.265 (4:2:0):");
    if enc.is_empty() {
        println!(
            "  (none) build with a `nvidia` / `amd` / `qsv` feature, or `rav1e-fallback` \
             (software AV1) / `h26x-fallback` (software H.264 / H.265)"
        );
    } else {
        println!("  backends   : {}", enc.join(", "));
        println!("  max depth  : {}-bit", caps.max_bit_depth);
        println!(
            "  HDR        : {}",
            if caps.hdr {
                "yes (PQ / HLG, BT.2020, 10-bit)"
            } else {
                "no"
            }
        );
    }
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
