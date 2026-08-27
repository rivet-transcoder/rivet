//! Implementation of `rivet devices`.

use codec::frame::VideoCodec;

pub(crate) fn run(json: bool) {
    let devices = codec::gpu::detect_gpus();
    if json {
        println!("{}", devices_json(&devices));
        return;
    }
    if devices.is_empty() {
        println!(
            "No GPUs detected (CPU-only host). GPU transcode needs a `nvidia` / `amd` / `qsv` \
             feature build with the matching hardware; `rav1e-fallback` provides software AV1 and \
             `h26x-fallback` software H.264 / H.265, on which the ladder runs as software leases. \
             This build: {}.",
            software_summary()
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
        println!("      encode     : {}", encode_verdicts(d));
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
    if !devices.iter().any(|d| ENCODE_CODECS.iter().any(|&c| codec::encode::encode_capable(d, c))) {
        println!(
            "No detected GPU can encode in this build (detected is not usable: the vendor feature \
             may be off, or the silicon predates the codec). Software encode in this build: {}.",
            software_summary()
        );
        println!();
    }
    println!("Run `rivet capabilities` for what this build can encode/decode.");
}

/// The output codecs a device is asked about, in display order.
const ENCODE_CODECS: [VideoCodec; 3] = [VideoCodec::Av1, VideoCodec::H264, VideoCodec::H265];

/// `av1 no · h264 yes · h265 yes` — the per-codec encode verdicts for one
/// device, from the same probe the encode pool uses to drop incapable cards.
pub(crate) fn encode_verdicts(d: &codec::gpu::GpuDevice) -> String {
    ENCODE_CODECS
        .iter()
        .map(|&c| {
            format!(
                "{} {}",
                codec_label(c),
                if codec::encode::encode_capable(d, c) { "yes" } else { "no" }
            )
        })
        .collect::<Vec<_>>()
        .join(" · ")
}

/// Which codecs this build can encode with no silicon at all, and how the
/// ladder would divide this machine among software encoders.
pub(crate) fn software_summary() -> String {
    let plan = rivet::multigpu::host_software_pool_plan();
    format!(
        "AV1 {} (`rav1e-fallback`), H.264 / H.265 {} (`h26x-fallback`); {} software slot(s) × {} thread(s)",
        if codec::encode::software_encode_available(VideoCodec::Av1) { "yes" } else { "no" },
        if codec::encode::software_encode_available(VideoCodec::H264) { "yes" } else { "no" },
        plan.slots,
        plan.threads,
    )
}

fn codec_label(c: VideoCodec) -> &'static str {
    match c {
        VideoCodec::Av1 => "av1",
        VideoCodec::H264 => "h264",
        VideoCodec::H265 => "h265",
    }
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
                "{{\"index\":{},\"vendor\":\"{}\",\"name\":\"{}\",\"generation\":\"{}\",\"vram_mib\":{},\"pci\":\"{}\",\"av1_encode\":{},\
                 \"encode\":{{\"av1\":{},\"h264\":{},\"h265\":{}}}{}}}",
                d.index,
                codec::gpu::manufacturer_label(d.vendor),
                super::esc(&d.name),
                super::esc(&d.generation),
                d.vram_mib,
                super::esc(&d.host_pci_address),
                codec::encode::av1_encode_capable(d),
                codec::encode::encode_capable(d, VideoCodec::Av1),
                codec::encode::encode_capable(d, VideoCodec::H264),
                codec::encode::encode_capable(d, VideoCodec::H265),
                load
            )
        })
        .collect();
    format!("{{\"gpus\":[{}]}}", items.join(","))
}
