//! Implementation of `rivet devices`.

pub(crate) fn run(json: bool) {
    let devices = codec::gpu::detect_gpus();
    if json {
        println!("{}", devices_json(&devices));
        return;
    }
    if devices.is_empty() {
        println!(
            "No GPUs detected (CPU-only host). GPU transcode needs a `nvidia` / `amd` / `qsv` \
             feature build with the matching hardware; `rav1e-fallback` provides software AV1."
        );
        return;
    }
    let util = codec::gpu::GpuUtilizationReader::new();
    println!("{} GPU(s) detected:\n", devices.len());
    for d in &devices {
        println!(
            "  [{}] {} {}",
            d.index,
            codec::gpu::manufacturer_label(d.vendor),
            d.name
        );
        println!("      generation : {}", d.generation);
        if d.vram_mib > 0 {
            println!("      VRAM       : {} MiB", d.vram_mib);
        }
        println!("      PCI        : {}", d.host_pci_address);
        println!(
            "      AV1 encode : {}",
            if codec::encode::av1_encode_capable(d) { "yes" } else { "no" }
        );
        // Live load is read via NVML — meaningful on NVIDIA only.
        if matches!(d.vendor, codec::gpu::GpuVendor::Nvidia) {
            let u = util.read(d);
            print!(
                "      load       : gpu {}% · enc {}% · dec {}% · mem {}/{} MiB",
                u.util_percent, u.encoder_percent, u.decoder_percent, u.mem_used_mib, u.mem_total_mib
            );
            if let Some(t) = u.temperature_c {
                print!(" · {t}°C");
            }
            println!();
        }
        println!();
    }
    println!("Run `rivet capabilities` for what this build can encode/decode.");
}

pub(crate) fn devices_json(devices: &[codec::gpu::GpuDevice]) -> String {
    let util = codec::gpu::GpuUtilizationReader::new();
    let items: Vec<String> = devices
        .iter()
        .map(|d| {
            let load = if matches!(d.vendor, codec::gpu::GpuVendor::Nvidia) {
                let u = util.read(d);
                let temp = u
                    .temperature_c
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "null".into());
                format!(
                    ",\"load\":{{\"gpu_percent\":{},\"encoder_percent\":{},\"decoder_percent\":{},\"mem_used_mib\":{},\"mem_total_mib\":{},\"temperature_c\":{}}}",
                    u.util_percent, u.encoder_percent, u.decoder_percent, u.mem_used_mib, u.mem_total_mib, temp
                )
            } else {
                String::new()
            };
            format!(
                "{{\"index\":{},\"vendor\":\"{}\",\"name\":\"{}\",\"generation\":\"{}\",\"vram_mib\":{},\"pci\":\"{}\",\"av1_encode\":{}{}}}",
                d.index,
                codec::gpu::manufacturer_label(d.vendor),
                super::esc(&d.name),
                super::esc(&d.generation),
                d.vram_mib,
                super::esc(&d.host_pci_address),
                codec::encode::av1_encode_capable(d),
                load
            )
        })
        .collect();
    format!("{{\"gpus\":[{}]}}", items.join(","))
}
