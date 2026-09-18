//! Minimal AVI (RIFF) demuxer.
//!
//! Scope: read the video stream out of a one-video-track AVI file, map
//! the stream's handler / fourcc to one of the codec labels the
//! transcoder knows how to decode, and emit per-frame samples in the
//! order the file lays them down (presentation order — AVI does not
//! have B-frame reordering at the container layer, stream samples are
//! already display-order). The first audio stream is read too, with the
//! timeline ffmpeg gives it (see [`audio`]); further audio streams are
//! ignored, as MP4 and MKV ignore theirs.
//!
//! OpenDML 1.0 super-indexes (Squad-38, 2026-04-17): files >1 GiB use
//! multiple `LIST movi` chunks (one per ~1 GiB RIFF segment) plus an
//! `indx` superindex chunk per stream that points at per-LIST `ix##`
//! standard indexes. Detection happens at construction: presence of an
//! `indx` chunk inside the video stream's `LIST strl` triggers the
//! OpenDML path, which precomputes a sample chunk-offset list from the
//! `indx` → `ix##` chain and `next_video_sample()` consumes that. When
//! `indx` is absent we fall back to the legacy single-`movi` cursor walk.
//! `dmlh.dwTotalFrames` from the `LIST odml` (sibling of `strl` LISTs in
//! `hdrl`) supersedes `avih.dwTotalFrames` for OpenDML files because
//! `avih` is a 32-bit field and gets truncated for clips longer than
//! `2^32 / fps` frames.
//!
//! What's intentionally not supported:
//! - Audio formats rivet has no path for (ADPCM, A-law / µ-law, WMA,
//!   ADTS-framed AAC): the track is surfaced by name with no packets and
//!   the audio stage drops it saying so.
//! - Variable-bitrate index reconstruction. We trust the sample order
//!   in the `movi` LIST itself; `idx1` is only used as a fallback when
//!   `movi` is missing (which real-world files don't exhibit).

mod audio;
mod riff;
mod opendml;
mod streaming;

#[cfg(test)]
mod tests;

pub use streaming::AviStreamingDemuxer;
pub(crate) use streaming::demux_avi_streaming_init;

use anyhow::{Context, Result, bail};
use frame::{ColorSpace, PixelFormat, StreamInfo};
use crate::demux::DemuxResult;
use opendml::read_dmlh_total_frames;
use riff::{ascii, collect_movi_samples, find_video_stream, fourcc_to_codec,
           frames_per_second, scan_top_level_records};

pub(crate) fn demux_avi(data: &[u8]) -> Result<DemuxResult> {
    // RIFF header: "RIFF" u32-LE size "AVI ".  (Guaranteed present by
    // the dispatch in `demux::detect_container`, but re-validate in
    // case this is ever called directly.)
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"AVI " {
        bail!("not a RIFF/AVI file");
    }

    // The RIFF payload begins after the 12-byte header. From there we
    // see a sequence of LIST/chunk records and optionally further
    // top-level RIFF chunks (OpenDML 1.0 — multi-`movi` files split
    // every ~1 GiB into a fresh `RIFF AVIX` segment that itself
    // contains a `LIST movi`). The records we care about:
    //   LIST hdrl  -> avih + (LIST strl { strh + strf [+ indx] })*
    //                 + LIST odml { dmlh }
    //   LIST movi  -> stream sample chunks (##dc / ##db / ##wb)
    // For OpenDML the file is `RIFF AVI ` ... `RIFF AVIX` ... `RIFF AVIX` ...
    // We scan the entire file for every `LIST movi` regardless of
    // which RIFF segment it lives in.
    let mut hdrl: Option<(usize, usize)> = None;
    let mut movi_lists: Vec<(usize, usize)> = Vec::new();
    scan_top_level_records(data, &mut hdrl, &mut movi_lists);

    let (hdrl_start, hdrl_end) = hdrl.context("AVI: missing hdrl LIST")?;
    if movi_lists.is_empty() {
        bail!("AVI: missing movi LIST");
    }

    // Walk hdrl looking for the video stream's strl LIST. strl contains
    // strh (stream header: type, handler fourcc) and strf (stream
    // format: BITMAPINFOHEADER for video).
    let video = find_video_stream(&data[hdrl_start..hdrl_end])
        .context("AVI: no video stream found in hdrl")?;

    let codec = fourcc_to_codec(&video.handler)
        .or_else(|| fourcc_to_codec(&video.compression))
        .with_context(|| {
            format!(
                "AVI: unsupported video fourcc {:?}/{:?}",
                ascii(&video.handler),
                ascii(&video.compression)
            )
        })?;

    // Stream-id prefix for this video stream's sample chunks in movi.
    // Two ASCII digits (stream index, zero-padded) + 'd' for 'dc' / 'b'
    // for 'db'. E.g. stream 0 gives `00dc` (compressed DIB frame) or
    // `00db` (uncompressed DIB frame).
    let stream_idx = video.stream_index;
    let prefix = format!("{:02}", stream_idx);

    // Walk every movi LIST in file order (OpenDML splits one logical
    // movi across multiple RIFF AVIX segments). For each LIST, pull
    // every chunk whose fourcc starts with `<prefix>d` into samples in
    // order. Non-video chunks (audio `##wb`, JUNK, `rec ` LISTs for
    // OpenDML) are skipped, and so are empty video chunks: a dropped or
    // repeated frame's slot is not a frame.
    let mut samples: Vec<Vec<u8>> = Vec::new();
    let mut chunks = 0u64;
    for &(movi_start, movi_end) in &movi_lists {
        chunks += collect_movi_samples(&data[movi_start..movi_end], &prefix, &mut samples)?;
    }

    if samples.is_empty() {
        bail!(
            "AVI: movi LIST contained no video samples for stream {:02}",
            stream_idx
        );
    }

    // Length-prefixed H.264 (an avcC record in `strf`, as `-c copy` out of
    // an MP4 writes it) becomes Annex-B here, through the converter the MP4
    // and MKV demuxers use. An Annex-B stream is left exactly as it is.
    if let Some(lp) = riff::length_prefixed(&codec, &video.extradata) {
        let mut tracker = lp.tracker();
        for sample in samples.iter_mut() {
            let annexb = lp.to_annexb(sample, &mut tracker);
            *sample = annexb;
        }
    }

    // Empty chunks between the frames mean the header counts ticks, not
    // frames: the samples are the frames, and the rate follows them — the
    // stream's `chunks` ticks last as long as they did, over fewer frames.
    let frames = samples.len() as u64;
    let frame_rate = frames_per_second(video.frame_rate, chunks, frames);
    // Otherwise prefer dmlh.dwTotalFrames over the materialized sample count
    // when OpenDML is present — for >1 GiB files, dmlh is the spec-mandated
    // accurate count; avih.dwTotalFrames is u32 and may have wrapped.
    // Falling back to samples.len() preserves legacy behaviour for
    // single-`movi` files without an odml LIST.
    let total_frames = if frames < chunks {
        frames
    } else {
        read_dmlh_total_frames(&data[hdrl_start..hdrl_end]).unwrap_or(frames)
    };
    let duration = if frame_rate > 0.0 {
        total_frames as f64 / frame_rate
    } else {
        0.0
    };

    let info = StreamInfo {
        codec: codec.clone(),
        width: video.width,
        height: video.height,
        frame_rate,
        duration,
        // AVI's BITMAPINFOHEADER does not carry a spec-grade pixel
        // format — the fourcc implies 4:2:0 for the codecs we
        // actually support (MPEG-4 Part 2 / DivX / H.264 Baseline
        // in AVI). Downstream `pixel_format::detect` can refine
        // this once a codec-level parse runs.
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        color_metadata: Default::default(),
        total_frames,
        // AVI's dwRate/dwScale in strh gives a bps for audio, not
        // video. Real video bitrate requires codec-level inspection,
        // which we punt on. 0 means "unknown" to the pipeline — same
        // posture as the MKV demuxer.
        bitrate: 0,
    };

    // Refine pixel_format from the bitstream now that we have samples: the
    // first SPS, from the colour window for H.264 / HEVC.
    let head = crate::demux::hdr::colour_window(&codec, samples.iter().map(Vec::as_slice), "avi");
    let detected_pf = match &head {
        Some(head) if head.has_sps => {
            frame::pixel_format::detect(&codec, std::slice::from_ref(&head.annexb))
        }
        _ => frame::pixel_format::detect(&codec, &samples),
    };
    let mut info = StreamInfo {
        pixel_format: detected_pf,
        ..info
    };
    // AVI carries no colour description: the first SPS's VUI and the SEIs
    // beside it are the source's colour.
    crate::demux::hdr::resolve_source_colour(
        &mut info,
        Default::default(),
        &codec,
        &[],
        head.as_ref().map(|head| head.annexb.as_slice()),
        "avi",
    );

    let audio = audio::read_audio(data, &data[hdrl_start..hdrl_end], &movi_lists);
    let audio_edit = audio.as_ref().and_then(|a| a.edit);
    Ok(DemuxResult {
        codec,
        info,
        samples,
        audio: audio.map(|a| a.track),
        video_presentation: None,
        audio_edit,
    })
}
