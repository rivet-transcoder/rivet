//! Writes a synthetic H.264 MP4 test source, made entirely by this
//! workspace's own encoders and muxer (`tests/common/synth.rs`): the test
//! pattern (colour bars, a scrolling ramp, a moving box), optionally with
//! uniform noise over every sample, optionally with a 440 Hz stereo AAC
//! track. For CI jobs that hand the CLI a file.
//!
//! ```text
//! cargo run --release --example synth_clip -- OUT.mp4 [--size WxH] [--fps N]
//!     [--seconds S] [--noise N] [--bitrate BPS] [--audio]
//! ```
//!
//! Defaults: 320x240, 25 fps, 4 seconds, no noise, a fixed quantiser, no
//! audio.
//!
//! `synth_clip OUT.m4a --tones51-aac` instead writes the committed audio
//! fixture `tests/data/audio/tones_51_aac.m4a` (see `synth::tones_51_aac_m4a`).

#[path = "../tests/common/synth.rs"]
mod synth;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut out = None;
    let (mut w, mut h, mut fps, mut seconds, mut noise, mut bitrate, mut audio) = (320, 240, 25, 4.0, 0u8, 0u32, false);
    let mut tones = false;
    while let Some(a) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--size" => {
                let v = value();
                let (a, b) = v.split_once('x').expect("--size WxH");
                (w, h) = (a.parse().expect("width"), b.parse().expect("height"));
            }
            "--fps" => fps = value().parse().expect("--fps N"),
            "--seconds" => seconds = value().parse().expect("--seconds S"),
            "--noise" => noise = value().parse().expect("--noise N (0-255)"),
            "--bitrate" => bitrate = value().parse().expect("--bitrate BPS"),
            "--audio" => audio = true,
            "--tones51-aac" => tones = true,
            _ if out.is_none() && !a.starts_with("--") => out = Some(a),
            _ => panic!("unknown argument {a}"),
        }
    }
    let out = out.expect("usage: synth_clip OUT.mp4 [--size WxH] [--fps N] [--seconds S] [--noise N] [--bitrate BPS] [--audio]");
    if tones {
        let m4a = synth::tones_51_aac_m4a();
        std::fs::write(&out, &m4a).unwrap_or_else(|e| panic!("{out}: {e}"));
        println!("{out}: 5.1 AAC tones, {} bytes", m4a.len());
        return;
    }
    let mp4 = synth::clip(w, h, fps, seconds, noise, bitrate, audio);
    std::fs::write(&out, &mp4).unwrap_or_else(|e| panic!("{out}: {e}"));
    println!("{out}: {w}x{h} at {fps} fps, {seconds} s, {} bytes", mp4.len());
}
