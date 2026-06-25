//! HLS (HTTP Live Streaming) playlist generation for CMAF VOD output.
//!
//! Produces:
//!   - `master.m3u8` — the top-level multivariant playlist with one
//!     `#EXT-X-STREAM-INF` per video rendition and one
//!     `#EXT-X-MEDIA:TYPE=AUDIO` rendition group entry pointing at the
//!     shared audio playlist.
//!   - `<rendition>/playlist.m3u8` per video rendition — VOD media
//!     playlist with `#EXT-X-MAP` referring to the rendition's
//!     `init.mp4` and `#EXTINF` lines pointing at the
//!     `seg-NNNNN.m4s` files (relative URIs).
//!   - `<audio_dir>/audio.m3u8` — the shared audio media playlist.
//!
//! Spec: RFC 8216 (HLS) + Apple's HLS Authoring Spec for VOD content,
//! plus AV1-CMAF-HLS interoperability notes from hls.js's test suite.
//! We target HLS protocol version 7 — the minimum that supports
//! `EXT-X-MAP` (fMP4 init segment) and `EXT-X-INDEPENDENT-SEGMENTS`.
//!
//! Codec strings (the load-bearing `CODECS` attribute) are passed in
//! by the caller — they MUST be parsed from the actual encoded
//! bitstream by [`codec::codec_strings::av1_codec_string`], not
//! composed from a config file. A wrong string causes hls.js / Safari
//! to silently skip the variant.

use anyhow::{Context, Result};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::cmaf::CmafTrackManifest;

/// Description of one video rendition for the master playlist.
#[derive(Debug, Clone)]
pub struct VideoVariantSpec {
    /// Frame width in pixels (post-scaling, what `RESOLUTION=` reports).
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Source frame rate. `FRAME-RATE=` is formatted to 3 decimal
    /// places per Apple's authoring spec (e.g. 29.970, 60.000).
    pub frame_rate: f64,
    /// Average bitrate in bits per second. Goes in the
    /// `AVERAGE-BANDWIDTH=` attribute.
    pub average_bandwidth_bps: u32,
    /// Peak bitrate in bits per second — `BANDWIDTH=`. Per RFC 8216
    /// §4.3.4.2 this is the largest single-segment bitrate observed
    /// (or, for VBR encoders without per-segment metering, the
    /// rendition's nominal `max_bitrate`). Players use this for ABR
    /// switching headroom decisions.
    pub bandwidth_bps: u32,
    /// AV1 codec string for the video track. Parse from the encoded
    /// bitstream via `codec::codec_strings::av1_codec_string`. Joined
    /// with the audio codec string in `CODECS="..."`.
    pub codec_string: String,
    /// Optional SUPPLEMENTAL-CODECS attribute string. Per HLS-Authoring
    /// Spec §"Supplemental Codecs", this carries an enhanced codec
    /// signalling that AUGMENTS the `CODECS` attribute — e.g. the
    /// `dvh1.08.07/db4h` form for Dolby Vision Profile 8 over an HEVC
    /// base layer.
    ///
    /// For pure AV1 HDR there's no base+enhancement model (the
    /// bitstream IS the HDR content), so the canonical pattern when
    /// HDR encode lands (Squad-22 dep) will be parallel SDR + HDR
    /// renditions in the master rather than supplemental signalling
    /// on a single variant. This field exists so that future
    /// supplemental-codec patterns (Dolby Vision over AV1 if/when
    /// that becomes a thing, AV2, etc.) can be wired in without a
    /// schema change.
    ///
    /// Format when set: `"<codec>/<compat>[/<compat>...]"`. None
    /// today; field is plumbed for forward compat.
    pub supplemental_codecs: Option<String>,
    /// VIDEO-RANGE attribute on the STREAM-INF. Per HLS spec, allowed
    /// values: "SDR" (default, omitted when at SDR), "HLG", "PQ".
    /// Set to `Some("PQ")` for HDR10 sources, `Some("HLG")` for HLG.
    /// None for SDR (omitted from output — HLS authors recommend
    /// omitting the attribute when at SDR rather than emitting
    /// `VIDEO-RANGE=SDR` explicitly).
    pub video_range: Option<&'static str>,
    /// Relative directory under the asset root, e.g. `"video/1080p"`.
    /// The variant's `playlist.m3u8` URI in the master is
    /// `<relative_dir>/playlist.m3u8`.
    pub relative_dir: String,
    /// CMAF track manifest produced by the segmenter. Source for the
    /// `EXT-X-MAP` URI + per-segment `EXTINF` durations.
    pub manifest: CmafTrackManifest,
}

/// Description of one audio rendition. CMAF-HLS uses a separate
/// rendition group so video variants can switch bitrate without
/// touching the audio track. We currently emit exactly one audio
/// rendition (default English / undetermined-language); multi-track
/// audio is a future task.
#[derive(Debug, Clone)]
pub struct AudioVariantSpec {
    /// Codec string for the audio track — typically
    /// `AAC_LC_CODEC_STRING` (`mp4a.40.2`).
    pub codec_string: String,
    /// Channel count. Goes in `CHANNELS="<n>"` per RFC 8216 §4.3.4.2.
    pub channels: u16,
    /// Sample rate in Hz. Informational; not surfaced in the playlist
    /// directly but kept on the spec so the validator can verify the
    /// init segment matches.
    #[allow(dead_code)]
    pub sample_rate: u32,
    /// Relative directory under the asset root, e.g. `"audio"`.
    pub relative_dir: String,
    /// BCP-47 language tag — `"en"`, `"es"`, `"und"` for undetermined.
    pub language: String,
    /// Human-readable rendition name. Players use this in their UI.
    pub name: String,
    pub manifest: CmafTrackManifest,
}

/// Paths produced by [`write_hls_package`]. Useful for the integration
/// test + the wire-contract reporter that surfaces a manifest URL to
/// lewd.net.
#[derive(Debug, Clone)]
pub struct HlsManifestPaths {
    pub master_path: PathBuf,
    pub video_playlist_paths: Vec<PathBuf>,
    /// `None` when the source has no audio (video-only HLS package).
    /// Master playlist + video playlists exist; no audio rendition
    /// group in master, no `audio/audio.m3u8` on disk.
    pub audio_playlist_path: Option<PathBuf>,
}

/// Emit a complete CMAF-HLS playlist tree under `output_dir`.
///
/// `output_dir` is the asset's root (e.g. `output/<asset_id>`). The
/// CMAF segments referenced by `manifest` fields are NOT moved — they
/// stay where the segmenters wrote them (the manifest paths must
/// already be under `output_dir`).
///
/// `target_duration_seconds` is the value emitted in
/// `#EXT-X-TARGETDURATION` for every media playlist. Per RFC 8216
/// §4.3.3.1 it's an upper bound on `EXTINF` and must be rounded UP
/// to the nearest integer; pass the configured CMAF segment duration.
pub fn write_hls_package(
    output_dir: &Path,
    video_variants: &[VideoVariantSpec],
    audio: Option<&AudioVariantSpec>,
    target_duration_seconds: u32,
) -> Result<HlsManifestPaths> {
    fs::create_dir_all(output_dir)
        .with_context(|| format!("creating HLS output dir: {}", output_dir.display()))?;

    // Per-variant video playlists.
    let mut video_playlist_paths = Vec::with_capacity(video_variants.len());
    for v in video_variants {
        let dir = output_dir.join(&v.relative_dir);
        fs::create_dir_all(&dir)
            .with_context(|| format!("creating video variant dir: {}", dir.display()))?;
        let path = dir.join("playlist.m3u8");
        write_media_playlist(&path, &v.manifest, target_duration_seconds)
            .with_context(|| format!("writing video media playlist: {}", path.display()))?;
        video_playlist_paths.push(path);
    }

    // Audio playlist (optional — None for video-only sources).
    let audio_playlist_path = if let Some(audio) = audio {
        let audio_dir = output_dir.join(&audio.relative_dir);
        fs::create_dir_all(&audio_dir)
            .with_context(|| format!("creating audio variant dir: {}", audio_dir.display()))?;
        let path = audio_dir.join("audio.m3u8");
        write_media_playlist(&path, &audio.manifest, target_duration_seconds)
            .with_context(|| format!("writing audio media playlist: {}", path.display()))?;
        Some(path)
    } else {
        None
    };

    // Master playlist last so its existence is the "all done" signal
    // for any external watcher polling for the asset to appear.
    let master_path = output_dir.join("master.m3u8");
    write_master_playlist(&master_path, video_variants, audio)
        .with_context(|| format!("writing master playlist: {}", master_path.display()))?;

    Ok(HlsManifestPaths {
        master_path,
        video_playlist_paths,
        audio_playlist_path,
    })
}

/// Write a single media playlist file.
///
/// Format per RFC 8216 §4.3:
///   #EXTM3U
///   #EXT-X-VERSION:7
///   #EXT-X-TARGETDURATION:<rounded-up>
///   #EXT-X-PLAYLIST-TYPE:VOD
///   #EXT-X-MAP:URI="init.mp4"
///   #EXTINF:<exact_duration>,
///   seg-NNNNN.m4s
///   ...
///   #EXT-X-ENDLIST
///
/// The init/segment URIs are RELATIVE — same directory as the playlist
/// itself. CMAF muxers write into the variant's directory by
/// construction so this resolves cleanly without any path computation.
fn write_media_playlist(
    path: &Path,
    manifest: &CmafTrackManifest,
    target_duration_seconds: u32,
) -> Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "#EXTM3U")?;
    writeln!(w, "#EXT-X-VERSION:7")?;
    writeln!(w, "#EXT-X-TARGETDURATION:{}", target_duration_seconds)?;
    writeln!(w, "#EXT-X-PLAYLIST-TYPE:VOD")?;
    writeln!(
        w,
        "#EXT-X-MAP:URI=\"{}\"",
        manifest
            .init_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("init.mp4")
    )?;

    for seg in &manifest.segments {
        let dur = seg.duration_ticks as f64 / manifest.timescale as f64;
        // Apple HLS authoring requires 6 decimal places minimum for
        // EXTINF on VOD content so the cumulative duration matches
        // what playback computes. Trailing comma per RFC 8216 §4.3.2.1.
        writeln!(w, "#EXTINF:{:.6},", dur)?;
        let name = seg
            .path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("segment path has no filename"))?;
        writeln!(w, "{name}")?;
    }

    writeln!(w, "#EXT-X-ENDLIST")?;
    w.flush()?;
    Ok(())
}

/// Write the master (multivariant) playlist.
///
/// Format per RFC 8216 §4.3.4 + Apple HLS Authoring Spec:
///   #EXTM3U
///   #EXT-X-VERSION:7
///   #EXT-X-INDEPENDENT-SEGMENTS
///   #EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="aac",...,URI="audio/audio.m3u8"
///   #EXT-X-STREAM-INF:BANDWIDTH=...,RESOLUTION=...x...,CODECS="av01,...,mp4a.40.2",AUDIO="aac"
///   video/1080p/playlist.m3u8
///   ...
///
/// Variants are emitted in ascending bandwidth order — players with
/// limited ABR heuristics (older hls.js, Safari < 14) walk the list
/// linearly and pick the first variant that fits, so the order
/// matters in practice.
fn write_master_playlist(
    path: &Path,
    video_variants: &[VideoVariantSpec],
    audio: Option<&AudioVariantSpec>,
) -> Result<()> {
    let body = render_master_playlist_to_string(video_variants, audio);
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    w.write_all(body.as_bytes())?;
    w.flush()?;
    Ok(())
}

/// Render the master playlist as an in-memory string. Internal helper
/// for [`write_master_playlist`]. The transcoder no longer publishes a
/// `master.m3u8` to S3 (see commit 7197885 reverted by the follow-up
/// 2026-05-08 change): the backend builds the master document on
/// every viewer request so signed URLs + per-viewer permissions
/// (subscription tier, follower-only, paid-content bucket selection)
/// can be applied correctly. This function stays in tree because
/// `write_master_playlist` still produces an on-disk `master.m3u8`
/// that the orchestrator's pipeline tests rely on; it does NOT
/// participate in the production wire contract anymore.
fn render_master_playlist_to_string(
    video_variants: &[VideoVariantSpec],
    audio: Option<&AudioVariantSpec>,
) -> String {
    use std::fmt::Write;

    let mut out = String::with_capacity(256 + video_variants.len() * 192);
    let _ = writeln!(out, "#EXTM3U");
    let _ = writeln!(out, "#EXT-X-VERSION:7");
    let _ = writeln!(out, "#EXT-X-INDEPENDENT-SEGMENTS");
    let _ = writeln!(out);

    // Audio rendition group — only when source has an audio track.
    // For video-only sources we skip the EXT-X-MEDIA block AND drop
    // the AUDIO= attribute on each STREAM-INF. hls.js + native HLS
    // both handle the audio-less master cleanly.
    if let Some(audio) = audio {
        let _ = write!(out, "#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"aac\"");
        let _ = write!(out, ",NAME=\"{}\"", escape_attr(&audio.name));
        let _ = write!(out, ",DEFAULT=YES,AUTOSELECT=YES");
        let _ = write!(out, ",LANGUAGE=\"{}\"", escape_attr(&audio.language));
        let _ = write!(out, ",CHANNELS=\"{}\"", audio.channels);
        let _ = writeln!(out, ",URI=\"{}/audio.m3u8\"", audio.relative_dir);
        let _ = writeln!(out);
    }

    // Video variants ordered by ascending bandwidth.
    let mut sorted: Vec<&VideoVariantSpec> = video_variants.iter().collect();
    sorted.sort_by_key(|v| v.bandwidth_bps);

    for v in sorted {
        let _ = write!(out, "#EXT-X-STREAM-INF");
        let _ = write!(out, ":BANDWIDTH={}", v.bandwidth_bps);
        let _ = write!(out, ",AVERAGE-BANDWIDTH={}", v.average_bandwidth_bps);
        // CODECS is the failure mode if it's wrong — players silently
        // skip variants whose CODECS string they can't decode. The
        // string MUST come from bitstream parsing, never from config.
        // Audio-less sources drop the trailing `,mp4a.40.2` component.
        match audio {
            Some(audio) => {
                let _ = write!(out, ",CODECS=\"{},{}\"", v.codec_string, audio.codec_string);
            }
            None => {
                let _ = write!(out, ",CODECS=\"{}\"", v.codec_string);
            }
        }
        if let Some(supp) = v.supplemental_codecs.as_ref() {
            let _ = write!(out, ",SUPPLEMENTAL-CODECS=\"{}\"", supp);
        }
        if let Some(vr) = v.video_range {
            let _ = write!(out, ",VIDEO-RANGE={}", vr);
        }
        let _ = write!(out, ",RESOLUTION={}x{}", v.width, v.height);
        let _ = write!(out, ",FRAME-RATE={:.3}", v.frame_rate);
        if audio.is_some() {
            let _ = writeln!(out, ",AUDIO=\"aac\"");
        } else {
            let _ = writeln!(out);
        }
        let _ = writeln!(out, "{}/playlist.m3u8", v.relative_dir);
    }

    out
}

/// Escape characters that aren't legal inside an HLS attribute-value
/// quoted string. Per RFC 8216 §4.2 the quoted string MUST NOT
/// contain a literal `"`, line feed, or carriage return. We strip
/// rather than escape (HLS has no escape syntax for these).
fn escape_attr(s: &str) -> String {
    s.chars()
        .filter(|c| *c != '"' && *c != '\n' && *c != '\r')
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmaf::SegmentInfo;

    fn synth_manifest(timescale: u32, durations_ticks: &[u64]) -> CmafTrackManifest {
        let segments: Vec<SegmentInfo> = durations_ticks
            .iter()
            .enumerate()
            .map(|(i, &d)| SegmentInfo {
                sequence_number: (i + 1) as u32,
                path: PathBuf::from(format!("seg-{:05}.m4s", i + 1)),
                byte_size: 1024,
                duration_ticks: d,
            })
            .collect();
        CmafTrackManifest {
            init_path: PathBuf::from("init.mp4"),
            segments,
            timescale,
        }
    }

    #[test]
    fn media_playlist_includes_all_required_v7_tags() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("playlist.m3u8");
        let manifest = synth_manifest(30000, &[120_000, 120_000, 120_000]);
        write_media_playlist(&path, &manifest, 4).unwrap();
        let body = fs::read_to_string(&path).unwrap();
        assert!(body.starts_with("#EXTM3U\n"));
        assert!(body.contains("#EXT-X-VERSION:7\n"));
        assert!(body.contains("#EXT-X-TARGETDURATION:4\n"));
        assert!(body.contains("#EXT-X-PLAYLIST-TYPE:VOD\n"));
        assert!(body.contains("#EXT-X-MAP:URI=\"init.mp4\""));
        assert!(body.contains("#EXTINF:4.000000,"));
        assert!(body.contains("seg-00001.m4s\n"));
        assert!(body.contains("seg-00003.m4s\n"));
        assert!(body.trim_end().ends_with("#EXT-X-ENDLIST"));
    }

    #[test]
    fn media_playlist_uses_real_segment_durations_not_nominal() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("playlist.m3u8");
        // Three segments of slightly different durations (last one short).
        let manifest = synth_manifest(30000, &[120_000, 120_000, 87_500]);
        write_media_playlist(&path, &manifest, 4).unwrap();
        let body = fs::read_to_string(&path).unwrap();
        // 87500 / 30000 = 2.9166666...
        assert!(body.contains("#EXTINF:2.916667,"), "got: {body}");
    }

    #[test]
    fn master_playlist_orders_variants_by_ascending_bandwidth() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("master.m3u8");
        let video_manifest = synth_manifest(30000, &[120_000]);

        let v1080 = VideoVariantSpec {
            width: 1920,
            height: 1080,
            frame_rate: 30.0,
            average_bandwidth_bps: 3_000_000,
            bandwidth_bps: 4_500_000,
            codec_string: "av01.0.08M.08.0.001.001.001.0".into(),
            supplemental_codecs: None,
            video_range: None,
            relative_dir: "video/1080p".into(),
            manifest: video_manifest.clone(),
        };
        let v720 = VideoVariantSpec {
            width: 1280,
            height: 720,
            frame_rate: 30.0,
            average_bandwidth_bps: 1_600_000,
            bandwidth_bps: 2_400_000,
            codec_string: "av01.0.06M.08.0.001.001.001.0".into(),
            supplemental_codecs: None,
            video_range: None,
            relative_dir: "video/720p".into(),
            manifest: video_manifest.clone(),
        };
        let v480 = VideoVariantSpec {
            width: 854,
            height: 480,
            frame_rate: 30.0,
            average_bandwidth_bps: 800_000,
            bandwidth_bps: 1_200_000,
            codec_string: "av01.0.04M.08.0.001.001.001.0".into(),
            supplemental_codecs: None,
            video_range: None,
            relative_dir: "video/480p".into(),
            manifest: video_manifest.clone(),
        };

        let audio = AudioVariantSpec {
            codec_string: "mp4a.40.2".into(),
            channels: 2,
            sample_rate: 48000,
            relative_dir: "audio".into(),
            language: "und".into(),
            name: "Default".into(),
            manifest: synth_manifest(48000, &[192_000]),
        };

        // Pass them in REVERSE bandwidth order to verify sorting.
        write_master_playlist(&path, &[v1080, v720, v480], Some(&audio)).unwrap();
        let body = fs::read_to_string(&path).unwrap();

        // Find 480p, 720p, 1080p positions; assert ascending order.
        let p480 = body
            .find("video/480p/playlist.m3u8")
            .expect("480p variant present");
        let p720 = body
            .find("video/720p/playlist.m3u8")
            .expect("720p variant present");
        let p1080 = body
            .find("video/1080p/playlist.m3u8")
            .expect("1080p variant present");
        assert!(p480 < p720, "480p must come before 720p");
        assert!(p720 < p1080, "720p must come before 1080p");
    }

    #[test]
    fn master_playlist_emits_required_top_level_tags() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("master.m3u8");
        let video_manifest = synth_manifest(30000, &[120_000]);
        let v = VideoVariantSpec {
            width: 1920,
            height: 1080,
            frame_rate: 30.0,
            average_bandwidth_bps: 3_000_000,
            bandwidth_bps: 4_500_000,
            codec_string: "av01.0.08M.08.0.001.001.001.0".into(),
            supplemental_codecs: None,
            video_range: None,
            relative_dir: "video/1080p".into(),
            manifest: video_manifest,
        };
        let audio = AudioVariantSpec {
            codec_string: "mp4a.40.2".into(),
            channels: 2,
            sample_rate: 48000,
            relative_dir: "audio".into(),
            language: "und".into(),
            name: "Default".into(),
            manifest: synth_manifest(48000, &[192_000]),
        };
        write_master_playlist(&path, &[v], Some(&audio)).unwrap();
        let body = fs::read_to_string(&path).unwrap();

        assert!(body.starts_with("#EXTM3U"));
        assert!(body.contains("#EXT-X-VERSION:7"));
        assert!(body.contains("#EXT-X-INDEPENDENT-SEGMENTS"));
        assert!(body.contains("#EXT-X-MEDIA:TYPE=AUDIO"));
        assert!(body.contains("GROUP-ID=\"aac\""));
        assert!(body.contains("DEFAULT=YES"));
        assert!(body.contains("URI=\"audio/audio.m3u8\""));
        assert!(body.contains("#EXT-X-STREAM-INF"));
        assert!(body.contains("BANDWIDTH=4500000"));
        assert!(body.contains("AVERAGE-BANDWIDTH=3000000"));
        assert!(body.contains("CODECS=\"av01.0.08M.08.0.001.001.001.0,mp4a.40.2\""));
        assert!(body.contains("RESOLUTION=1920x1080"));
        assert!(body.contains("FRAME-RATE=30.000"));
        assert!(body.contains("AUDIO=\"aac\""));
    }

    #[test]
    fn write_hls_package_emits_full_directory_tree() {
        let dir = tempfile::tempdir().unwrap();
        let video_manifest = CmafTrackManifest {
            init_path: dir.path().join("video/1080p/init.mp4"),
            segments: vec![SegmentInfo {
                sequence_number: 1,
                path: dir.path().join("video/1080p/seg-00001.m4s"),
                byte_size: 1024,
                duration_ticks: 120_000,
            }],
            timescale: 30000,
        };
        let audio_manifest = CmafTrackManifest {
            init_path: dir.path().join("audio/init.mp4"),
            segments: vec![SegmentInfo {
                sequence_number: 1,
                path: dir.path().join("audio/seg-00001.m4s"),
                byte_size: 256,
                duration_ticks: 192_000,
            }],
            timescale: 48000,
        };

        let v = VideoVariantSpec {
            width: 1920,
            height: 1080,
            frame_rate: 30.0,
            average_bandwidth_bps: 3_000_000,
            bandwidth_bps: 4_500_000,
            codec_string: "av01.0.08M.08.0.001.001.001.0".into(),
            supplemental_codecs: None,
            video_range: None,
            relative_dir: "video/1080p".into(),
            manifest: video_manifest,
        };
        let a = AudioVariantSpec {
            codec_string: "mp4a.40.2".into(),
            channels: 2,
            sample_rate: 48000,
            relative_dir: "audio".into(),
            language: "und".into(),
            name: "Default".into(),
            manifest: audio_manifest,
        };

        let paths = write_hls_package(dir.path(), &[v], Some(&a), 4).unwrap();

        assert!(paths.master_path.exists());
        assert_eq!(paths.video_playlist_paths.len(), 1);
        assert!(paths.video_playlist_paths[0].exists());
        let audio_pl_path = paths.audio_playlist_path.expect("audio playlist set");
        assert!(audio_pl_path.exists());

        // Spot-check the audio playlist.
        let audio_pl = fs::read_to_string(&audio_pl_path).unwrap();
        assert!(audio_pl.contains("#EXT-X-MAP:URI=\"init.mp4\""));
        assert!(audio_pl.contains("#EXTINF:4.000000,"));
        assert!(audio_pl.contains("seg-00001.m4s"));
    }

    #[test]
    fn master_playlist_omits_audio_when_video_only() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("master.m3u8");
        let video_manifest = synth_manifest(30000, &[120_000]);
        let v = VideoVariantSpec {
            width: 1920,
            height: 1080,
            frame_rate: 30.0,
            average_bandwidth_bps: 3_000_000,
            bandwidth_bps: 4_500_000,
            codec_string: "av01.0.08M.08".into(),
            supplemental_codecs: None,
            video_range: None,
            relative_dir: "video/1080p".into(),
            manifest: video_manifest,
        };
        write_master_playlist(&path, &[v], None).unwrap();
        let body = fs::read_to_string(&path).unwrap();

        assert!(body.starts_with("#EXTM3U"));
        assert!(body.contains("#EXT-X-VERSION:7"));
        assert!(body.contains("#EXT-X-INDEPENDENT-SEGMENTS"));
        // No audio rendition group.
        assert!(!body.contains("#EXT-X-MEDIA:TYPE=AUDIO"), "got: {body}");
        // CODECS attr should NOT include the AAC component.
        assert!(body.contains("CODECS=\"av01.0.08M.08\""), "got: {body}");
        assert!(!body.contains("mp4a.40.2"), "got: {body}");
        // STREAM-INF should NOT have the AUDIO= attribute.
        assert!(!body.contains("AUDIO=\"aac\""), "got: {body}");
    }

    #[test]
    fn write_hls_package_video_only_emits_no_audio_dir() {
        let dir = tempfile::tempdir().unwrap();
        let video_manifest = CmafTrackManifest {
            init_path: dir.path().join("video/720p/init.mp4"),
            segments: vec![SegmentInfo {
                sequence_number: 1,
                path: dir.path().join("video/720p/seg-00001.m4s"),
                byte_size: 1024,
                duration_ticks: 120_000,
            }],
            timescale: 30000,
        };
        let v = VideoVariantSpec {
            width: 1280,
            height: 720,
            frame_rate: 30.0,
            average_bandwidth_bps: 1_600_000,
            bandwidth_bps: 2_400_000,
            codec_string: "av01.0.05M.08".into(),
            supplemental_codecs: None,
            video_range: None,
            relative_dir: "video/720p".into(),
            manifest: video_manifest,
        };
        let paths = write_hls_package(dir.path(), &[v], None, 4).unwrap();
        assert!(paths.master_path.exists());
        assert_eq!(paths.video_playlist_paths.len(), 1);
        assert!(paths.audio_playlist_path.is_none());
        assert!(
            !dir.path().join("audio").exists(),
            "no audio dir should be created"
        );
    }

    #[test]
    fn escape_attr_strips_disallowed_characters() {
        assert_eq!(escape_attr(r#"hello"world"#), "helloworld");
        assert_eq!(escape_attr("with\nnewline"), "withnewline");
        assert_eq!(escape_attr("with\rcarriage"), "withcarriage");
        assert_eq!(escape_attr("normal text"), "normal text");
    }
}
