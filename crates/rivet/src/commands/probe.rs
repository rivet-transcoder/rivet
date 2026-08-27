//! Implementation of `rivet probe`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub(crate) fn run(input: PathBuf, json: bool) -> Result<()> {
    let info = rivet::probe_file(&input)
        .with_context(|| format!("probing {}", input.display()))?;
    if json {
        println!("{}", probe_json(&info));
    } else {
        print_probe(&input, &info);
    }
    Ok(())
}

fn print_probe(input: &Path, info: &rivet::MediaInfo) {
    println!("{}", input.display());
    println!("  container : {}", info.container);
    println!("  video     : {}", info.video_codec);
    println!("  dimensions: {}x{}", info.width, info.height);
    if info.rotation_degrees != 0 {
        println!(
            "  rotation:   {}° (stored {}x{}; the output is turned upright)",
            info.rotation_degrees, info.stored_width, info.stored_height
        );
    }
    println!("  frame rate: {:.3} fps", info.frame_rate);
    if info.duration > 0.0 {
        println!("  duration  : {:.3} s", info.duration);
    }
    println!("  pixel fmt : {}", info.pixel_format);
    match &info.audio {
        Some(a) => println!("  audio     : {} {} Hz {} ch", a.codec, a.sample_rate, a.channels),
        None => println!("  audio     : (none)"),
    }
    if info.subtitles.is_empty() {
        println!("  subtitles : (none)");
    } else {
        for s in &info.subtitles {
            println!("  subtitles : {} {} ({} cues)", s.language, s.codec, s.cues);
        }
    }
}

fn probe_json(info: &rivet::MediaInfo) -> String {
    let audio = match &info.audio {
        Some(a) => format!(
            "{{\"codec\":\"{}\",\"sample_rate\":{},\"channels\":{}}}",
            super::esc(&a.codec),
            a.sample_rate,
            a.channels
        ),
        None => "null".to_string(),
    };
    let subtitles = info
        .subtitles
        .iter()
        .map(|s| {
            format!(
                "{{\"codec\":\"{}\",\"language\":\"{}\",\"cues\":{}}}",
                super::esc(&s.codec),
                super::esc(&s.language),
                s.cues
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"container\":\"{}\",\"video_codec\":\"{}\",\"width\":{},\"height\":{},\"frame_rate\":{},\"duration\":{},\"pixel_format\":\"{}\",\"audio\":{},\"subtitles\":[{}]}}",
        super::esc(&info.container),
        super::esc(&info.video_codec),
        info.width,
        info.height,
        info.frame_rate,
        info.duration,
        super::esc(&info.pixel_format),
        audio,
        subtitles,
    )
}
