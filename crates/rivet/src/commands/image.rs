//! Implementation of `rivet image`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use rivet::TranscodeSettings;

use super::FitArgs;

/// `rivet image <input> -o <dir>`: still images of a still image, or stills
/// from a video.
#[derive(clap::Args, Debug)]
pub(crate) struct ImageArgs {
    /// Input: an image (JPEG, PNG, WebP, AVIF, GIF, TIFF, BMP, HEIC) or a
    /// video to take stills from.
    pub input: PathBuf,
    /// Output directory. Files are `<W>x<H>.<ext>`, or `<W>x<H>-<nnn>.<ext>`
    /// for several stills from a video.
    #[arg(short, long)]
    pub output: PathBuf,
    /// Output formats, comma-separated: `avif` (default), `webp`, `jpeg`,
    /// `png`. Every size is made in each.
    #[arg(long, value_name = "FORMATS", default_value = "avif")]
    pub format: String,
    /// Output sizes, each a box the picture is fitted into (repeatable, or
    /// comma-separated): `1920x1920`, `640x640:cover`. None is one output at
    /// the picture's own size.
    #[arg(long = "rung", value_name = "WxH")]
    pub rungs: Vec<String>,
    #[command(flatten)]
    pub fit: FitArgs,
    /// Quality for the lossy formats, 1–100 (defaults: AVIF 60, WebP 80,
    /// JPEG 82): one for every format (`70`), one per format
    /// (`avif:60,jpeg:82`), or both (`70,jpeg:82`).
    #[arg(long, value_name = "QUALITY")]
    pub quality: Option<String>,
    /// Lossless WebP (PNG is always lossless).
    #[arg(long)]
    pub lossless: bool,
    /// Keep the source's colour profile instead of converting to sRGB (PNG,
    /// JPEG and WebP carry it; AVIF is always converted).
    #[arg(long)]
    pub keep_icc: bool,
    /// Encoder effort, 1 (slowest, smallest) to 10 (fastest): the PNG
    /// DEFLATE level (6, the default, is level 6; 1 is level 9) and WebP's
    /// effort (0-6; 4 at the default).
    #[arg(long)]
    pub speed: Option<u8>,
    /// Which stills: `poster` (the default: an image input as it is, one
    /// frame 10% into a video). `--frames-at` / `--frames-count` choose
    /// others.
    #[arg(long, value_name = "poster", conflicts_with_all = ["frames_at", "frames_count"])]
    pub frames: Option<String>,
    /// A video input: take stills at these times, in seconds
    /// (comma-separated).
    #[arg(long = "frames-at", value_name = "SECONDS", conflicts_with = "frames_count")]
    pub frames_at: Option<String>,
    /// A video input: take this many stills, evenly spaced.
    #[arg(long = "frames-count", value_name = "N")]
    pub frames_count: Option<u32>,
    /// Still-image formats not to decode, e.g. `heic`.
    #[arg(long = "image-decode-deny", value_name = "FORMATS")]
    pub decode_deny: Option<String>,
}

pub(crate) fn run(args: ImageArgs) -> Result<()> {
    let mut settings = TranscodeSettings::default();
    settings.apply_kv("mode", "image")?;
    settings.apply_kv("image-format", &args.format).context("parsing --format")?;
    for rung in &args.rungs {
        settings.apply_kv("rung", rung).context("parsing --rung")?;
    }
    args.fit.apply(&mut settings)?;
    if let Some(q) = &args.quality {
        settings.apply_kv("image-quality", q).context("parsing --quality")?;
    }
    settings.image_lossless = args.lossless;
    settings.image_keep_icc = args.keep_icc;
    settings.image_speed = args.speed;
    if let Some(f) = &args.frames {
        settings.apply_kv("frames", f).context("parsing --frames")?;
    }
    if let Some(at) = &args.frames_at {
        settings.apply_kv("frames-at", at).context("parsing --frames-at")?;
    }
    if let Some(n) = args.frames_count {
        settings.apply_kv("frames-count", &n.to_string())?;
    }
    if let Some(deny) = &args.decode_deny {
        settings.apply_kv("image-decode-deny", deny).context("parsing --image-decode-deny")?;
    }
    let spec = settings.into_image_spec()?;

    let input = std::fs::read(&args.input).with_context(|| format!("reading {}", args.input.display()))?;
    let out = rivet::image::run_image_job(&bytes::Bytes::from(input), &spec)?;
    std::fs::create_dir_all(&args.output).with_context(|| format!("creating {}", args.output.display()))?;
    for m in &out.merged {
        eprintln!(
            "rendition {} comes out {}x{}, the same as rendition {}: made once",
            m.rendition, m.output.0, m.output.1, m.same_as
        );
    }
    for a in &out.artifacts {
        let name = a.file_name(out.several_frames);
        let path = args.output.join(&name);
        std::fs::write(&path, &a.bytes).with_context(|| format!("writing {}", path.display()))?;
        match a.frame {
            Some((_, t)) => println!("{name}  {}x{}  {} bytes  at {t:.3}s", a.width, a.height, a.bytes.len()),
            None => println!("{name}  {}x{}  {} bytes", a.width, a.height, a.bytes.len()),
        }
    }
    Ok(())
}
