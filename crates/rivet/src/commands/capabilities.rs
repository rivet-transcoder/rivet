//! Implementation of `rivet capabilities` / `rivet caps`.

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
        println!(
            "{{\"encode\":{{\"codec\":\"av1\",\"backends\":[{}],\"max_bit_depth\":{},\"hdr\":{}}},\
             \"decode\":{{\"backends\":[{}],\"codecs\":[{}]}},\"devices\":{}}}",
            enc_b,
            caps.max_bit_depth,
            caps.hdr,
            dec_b,
            codecs,
            super::devices::devices_json(&devices)
        );
        return;
    }

    println!("rivet capabilities\n");
    println!("Encode — AV1 (4:2:0):");
    if enc.is_empty() {
        println!("  (none) build with a `nvidia` / `amd` / `qsv` / `rav1e-fallback` feature");
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
            "  (none) CPU-only host — only the `rav1e-fallback` / `rav1d-fallback`              software AV1 paths can run here"
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
            // Authoritative AV1-encode verdict (the same probe the encode pool
            // uses to drop incapable cards) — so a pre-Ada NVIDIA shows "no".
            let av1 = if codec::encode::av1_encode_capable(dv) { "yes" } else { "no" };
            println!(" · AV1 encode: {av1}");
        }
    }
}
