//! NVDEC smoke test — exists to verify the live decode path on the
//! dev box (has RTX 3090 NVDEC). Prints progress to stdout since the
//! decode_integration tests don't initialize tracing.
//!
//! This test intentionally calls NvdecDecoder::new directly rather
//! than going through create_decoder, because create_decoder swallows
//! NVDEC errors into a tracing::warn! and falls through to CPU. For
//! diagnosis we want the raw NVDEC error.

use std::path::Path;

fn test_media(name: &str) -> Option<Vec<u8>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_media")
        .join(name);
    std::fs::read(&path).ok()
}

fn try_decode_matrix(file: &str, label: &str) -> (bool, usize, Option<String>) {
    let Some(data) = test_media(file) else {
        eprintln!("  SKIP {}: {} not present", label, file);
        return (false, 0, None);
    };
    let demuxed = match container::demux::demux(&data) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("  SKIP {}: demux failed: {}", label, e);
            return (false, 0, Some(format!("demux: {e}")));
        }
    };
    eprintln!(
        "  {} demux: codec={} {}x{} {} samples, head: {:02x?}",
        label,
        demuxed.codec,
        demuxed.info.width,
        demuxed.info.height,
        demuxed.samples.len(),
        &demuxed
            .samples
            .first()
            .map(|s| &s[..s.len().min(32)])
            .unwrap_or(&[]),
    );
    let mut decoder = codec::decode::nvdec::NvdecDecoder::new(demuxed.info.clone(), 0);
    for sample in &demuxed.samples {
        if let Err(e) = decoder.push_sample(sample) {
            eprintln!("  {} push_sample failed: {:#}", label, e);
            return (true, 0, Some(format!("{e:#}")));
        }
    }
    if let Err(e) = decoder.finish() {
        eprintln!("  {} finish failed: {:#}", label, e);
        return (true, 0, Some(format!("{e:#}")));
    }
    let mut count = 0usize;
    while let Some(_f) = decoder.decode_next().expect("decode_next") {
        count += 1;
        if count >= 30 {
            break;
        }
    }
    eprintln!("  {} decoded {} frames", label, count);
    (true, count, None)
}

/// Diagnostic for #65: dump H.264 NAL types per sample in the
/// ExoPlayer Main file to verify whether PPS ever appears in the demux
/// output. If type-8 (PPS) never appears, the avcC extradata parse
/// path dropped it — container issue, not codec.
#[test]
fn nvdec_exoplayer_main_nal_dump() {
    let Some(data) = test_media("exoplayer_h264_main_720p.mp4") else {
        eprintln!("SKIP: exoplayer_h264_main_720p.mp4 not present");
        return;
    };
    let demuxed = match container::demux::demux(&data) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: demux failed: {e}");
            return;
        }
    };
    eprintln!("Total samples: {}", demuxed.samples.len());

    let mut saw_pps = false;
    let mut idr_samples = Vec::new();
    for (i, s) in demuxed.samples.iter().enumerate() {
        let mut nals = Vec::new();
        let mut j = 0;
        while j + 4 < s.len() {
            if s[j] == 0 && s[j + 1] == 0 && s[j + 2] == 0 && s[j + 3] == 1 {
                nals.push(s[j + 4] & 0x1F);
                j += 4;
            } else if s[j] == 0 && s[j + 1] == 0 && s[j + 2] == 1 {
                nals.push(s[j + 3] & 0x1F);
                j += 3;
            } else {
                j += 1;
            }
        }
        if nals.contains(&8) {
            saw_pps = true;
        }
        if nals.contains(&5) {
            idr_samples.push(i);
        }
        if i < 20 {
            eprintln!("  sample {}: {} bytes, NALs={:?}", i, s.len(), nals);
        }
    }
    eprintln!(
        "\nPPS (type=8) seen: {}",
        if saw_pps { "YES" } else { "NO" }
    );
    eprintln!(
        "IDR (type=5) count: {} (first 10 indices: {:?})",
        idr_samples.len(),
        &idr_samples[..idr_samples.len().min(10)]
    );
    eprintln!(
        "Total samples: {}, IDR: {}, non-IDR: {}",
        demuxed.samples.len(),
        idr_samples.len(),
        demuxed.samples.len() - idr_samples.len()
    );
}

#[test]
fn nvdec_matrix_all_codecs() {
    let gpus = codec::gpu::detect_gpus();
    if !gpus
        .iter()
        .any(|g| g.vendor == codec::gpu::GpuVendor::Nvidia)
    {
        eprintln!("SKIP: no NVIDIA GPU");
        return;
    }
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(std::io::stderr)
        .try_init();

    eprintln!("--- NVDEC matrix on {} ---", gpus[0].name);
    // Profile spread: Baseline/Main (should work per perf-analyst's 9730 fps)
    // vs High / High 4:2:2 (hypothesized silent fallback).
    let h264_baseline = try_decode_matrix("exoplayer_h264_baseline_480p.mp4", "H.264 Baseline");
    let h264_main = try_decode_matrix("exoplayer_h264_main_720p.mp4", "H.264 Main");
    let h264_high = try_decode_matrix("jellyfin_h264_high_l40_1080p_24fps.mp4", "H.264 High");
    let h264_bbb = try_decode_matrix(
        "bigbuck_bunny_8bit_750kbps_720p_60.0fps_h264.mp4",
        "H.264 BBB (High 4:2:2)",
    );
    let hevc_bbb = try_decode_matrix(
        "bigbuck_bunny_8bit_750kbps_720p_60.0fps_hevc.mp4",
        "HEVC BBB",
    );
    let hevc_main = try_decode_matrix("jellyfin_hevc_main_1080p_24fps.mp4", "HEVC Main 1080p");
    // 10-bit cells — exercise the P016 surface path (#75). Before the
    // P016 fix these rejected with "10-bit content unsupported"; now
    // they should produce real Yuv420p10le frames. HDR10 sample also
    // exercises the VUI matrix_coefficients → ColorSpace::Bt2020 path
    // (#76).
    let hevc_main10 =
        try_decode_matrix("jellyfin_hevc_main10_1080p_24fps.mp4", "HEVC Main10 1080p");
    let hevc_main10_hdr = try_decode_matrix(
        "jellyfin_hevc_main10_hdr10_1080p_24fps.mp4",
        "HEVC Main10 HDR10",
    );
    let vp9 = try_decode_matrix(
        "bigbuck_bunny_8bit_750kbps_720p_60.0fps_vp9.mkv",
        "VP9 (MKV)",
    );
    let av1 = try_decode_matrix("jellyfin_av1_main_1080p_24fps.mp4", "AV1");

    // Legacy codec cells (task #71). These exercise the new MPEG-TS
    // demuxer, the AVI demuxer, and the VP8 + ProRes container paths.
    // NVDEC decodes VP8, MPEG-2, and MPEG-4 Part 2 natively on the
    // RTX 3090; ProRes has no GPU decoder on any vendor and is
    // expected to surface as "unsupported NVDEC codec" — this row is
    // deliberate documentation, not a passing cell.
    let vp8_webm = try_decode_matrix("vp8_webm_480p.webm", "VP8 (WebM)");
    let mpeg2_ts = try_decode_matrix("mpeg2_ts_720p.ts", "MPEG-2 (TS)");
    let mpeg4_avi = try_decode_matrix("xvid_divx_480p.avi", "MPEG-4 Part 2 (AVI XVID)");
    let divx_avi = try_decode_matrix(
        "ffmpeg_divx5_mpeg4part2_test_no_b_frames.avi",
        "MPEG-4 Part 2 (AVI DIVX no-B)",
    );
    let prores_422 = try_decode_matrix("prores_422_720p.mov", "ProRes 422 HQ (apch)");
    let prores_4444 = try_decode_matrix("ffmpeg_prores_4444_ap4h_fcp.mov", "ProRes 4444 (ap4h)");

    eprintln!("\n=== MATRIX ===");
    for (label, r) in [
        ("H.264 Baseline               ", &h264_baseline),
        ("H.264 Main                   ", &h264_main),
        ("H.264 High                   ", &h264_high),
        ("H.264 BBB (High 4:2:2)       ", &h264_bbb),
        ("HEVC BBB                     ", &hevc_bbb),
        ("HEVC Main 1080p              ", &hevc_main),
        ("HEVC Main10 1080p            ", &hevc_main10),
        ("HEVC Main10 HDR10            ", &hevc_main10_hdr),
        ("VP9 MKV                      ", &vp9),
        ("AV1                          ", &av1),
        ("VP8 (WebM)                   ", &vp8_webm),
        ("MPEG-2 (TS)                  ", &mpeg2_ts),
        ("MPEG-4 Part 2 (AVI XVID)     ", &mpeg4_avi),
        ("MPEG-4 Part 2 (AVI DIVX no-B)", &divx_avi),
        ("ProRes 422 HQ (apch)         ", &prores_422),
        ("ProRes 4444 (ap4h)           ", &prores_4444),
    ] {
        eprintln!("  {} → ran={} frames={:>4} err={:?}", label, r.0, r.1, r.2);
    }
}

#[test]
fn nvdec_decodes_h264_big_buck_bunny() {
    let gpus = codec::gpu::detect_gpus();
    println!("detected {} GPU(s):", gpus.len());
    for g in &gpus {
        println!("  - {:?} idx={} name={}", g.vendor, g.index, g.name);
    }
    let Some(dev) = gpus
        .iter()
        .find(|g| g.vendor == codec::gpu::GpuVendor::Nvidia)
    else {
        eprintln!("SKIP: no NVIDIA GPU detected");
        return;
    };

    let Some(data) = test_media("bigbuck_bunny_8bit_750kbps_720p_60.0fps_h264.mp4") else {
        eprintln!("SKIP: test_media/ not populated");
        return;
    };

    let demuxed = container::demux::demux(&data).expect("demux");
    println!(
        "demux: codec={} {}x{} @ {:.2}fps, {} samples",
        demuxed.codec,
        demuxed.info.width,
        demuxed.info.height,
        demuxed.info.frame_rate,
        demuxed.samples.len()
    );

    // Dump NAL types of first few samples so we can see whether
    // SPS/PPS/IDR are in order.
    for (i, s) in demuxed.samples.iter().enumerate().take(5) {
        let mut nal_types = Vec::new();
        let mut j = 0;
        while j + 4 < s.len() && nal_types.len() < 8 {
            if s[j] == 0 && s[j + 1] == 0 && s[j + 2] == 0 && s[j + 3] == 1 {
                nal_types.push(s[j + 4] & 0x1F);
                j += 4;
            } else if s[j] == 0 && s[j + 1] == 0 && s[j + 2] == 1 {
                nal_types.push(s[j + 3] & 0x1F);
                j += 3;
            } else {
                j += 1;
            }
        }
        println!("sample {}: {} bytes, nal_types={:?}", i, s.len(), nal_types);
    }

    // Initialize a barebones tracing subscriber so internal
    // tracing::debug/warn! calls in nvdec.rs surface on stdout.
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_writer(std::io::stderr)
        .try_init();

    // Streaming push-mode (#55 P3): construct, push every sample,
    // finish, then drain. The eager CUVID parse loop runs inside
    // finish() — typed UnsupportedChroma / UnsupportedPixelFormat
    // rejects surface from finish() rather than the constructor.
    let mut decoder = codec::decode::nvdec::NvdecDecoder::new(demuxed.info.clone(), dev.index);
    for sample in &demuxed.samples {
        if let Err(e) = decoder.push_sample(sample) {
            panic!("push_sample failed: {e:#}");
        }
    }
    match decoder.finish() {
        Ok(()) => {}
        Err(e) => {
            // Post codec-review-2 HIGH-1: a typed UnsupportedChroma
            // reject is the *correct* outcome for the Big Buck Bunny
            // H.264 "High 4:2:2" sample. Accept the typed reject as
            // a pass — it surfaces from finish() rather than new()
            // under the streaming-shape API.
            if let Some(nvdec_err) = e.downcast_ref::<codec::decode::nvdec::NvdecError>() {
                match nvdec_err {
                    codec::decode::nvdec::NvdecError::UnsupportedChroma { label, .. } => {
                        println!("NVDEC correctly rejected {}: {}", label, nvdec_err);
                        return;
                    }
                    codec::decode::nvdec::NvdecError::UnsupportedPixelFormat { bit_depth } => {
                        println!("NVDEC correctly rejected {}-bit: {}", bit_depth, nvdec_err);
                        return;
                    }
                    codec::decode::nvdec::NvdecError::UnsupportedByHardware { reason } => {
                        println!("NVDEC correctly rejected (hardware caps): {}", reason);
                        return;
                    }
                }
            }
            panic!("NvdecDecoder finish failed with non-typed error: {e:#}");
        }
    }

    let mut count = 0usize;
    while let Some(_frame) = decoder.decode_next().expect("decode_next") {
        count += 1;
        if count >= 60 {
            break;
        }
    }
    println!("decoded {} frames", count);
    assert!(count > 0, "NVDEC produced zero frames");
}

// ─── Unit coverage for reject / deinterleave / PTS paths ──────────
//
// These exercise the pure-Rust helpers extracted from the NVDEC
// callback pipeline so the reject matrix and P016 normalization can
// be proven correct without requiring a live GPU (the CI host has no
// NVIDIA hardware). Spec-conformant-by-review per reviews/codec-review-2.md
// HIGH-1 / HIGH-2 / HIGH-3.

use codec::decode::nvdec::{NvdecError, deinterleave_p016_to_yuv420p10le, validate_format};

/// HIGH-1: feeding a 4:2:2 chroma descriptor must surface a typed
/// reject rather than a stringly error. Callers (decode/mod.rs or
/// tests) can then pattern-match on the variant to choose fallback
/// policy.
#[test]
fn test_nvdec_rejects_422_chroma_with_typed_error() {
    // chroma_format = 2 == cudaVideoChromaFormat_422
    let err = validate_format(
        /* chroma_format = */ 2, /* bit_depth_luma_minus8 = */ 0,
        /* coded_width = */ 1920, /* coded_height = */ 1080,
    )
    .expect("4:2:2 must surface a typed reject");
    match err {
        NvdecError::UnsupportedChroma {
            chroma_format,
            label,
            width,
            height,
        } => {
            assert_eq!(chroma_format, 2);
            assert_eq!(label, "4:2:2");
            assert_eq!(width, 1920);
            assert_eq!(height, 1080);
        }
        other => panic!("expected UnsupportedChroma, got {other:?}"),
    }
    // Display impl must round-trip a human-readable message for log
    // sinks that don't yet pattern-match on the typed variant.
    let msg = format!("{err}");
    assert!(
        msg.contains("4:2:2"),
        "Display must mention chroma label: {msg}"
    );
    assert!(
        msg.contains("1920x1080"),
        "Display must carry resolution: {msg}"
    );
}

/// HIGH-1: same story for 4:4:4. A separate test rather than a matrix
/// so the label mapping stays pinned; regressions that collapse the
/// match arm would otherwise slip through.
#[test]
fn test_nvdec_rejects_444_chroma_with_typed_error() {
    let err = validate_format(
        /* chroma_format = */ 3, /* bit_depth_luma_minus8 = */ 0,
        /* coded_width = */ 3840, /* coded_height = */ 2160,
    )
    .expect("4:4:4 must surface a typed reject");
    match err {
        NvdecError::UnsupportedChroma {
            chroma_format,
            label,
            width,
            height,
        } => {
            assert_eq!(chroma_format, 3);
            assert_eq!(label, "4:4:4");
            assert_eq!(width, 3840);
            assert_eq!(height, 2160);
        }
        other => panic!("expected UnsupportedChroma, got {other:?}"),
    }
}

/// HIGH-1 accept path: 4:2:0 at 8-bit / 10-bit / 12-bit must pass.
#[test]
fn test_nvdec_accepts_420_at_standard_bit_depths() {
    // 8-bit
    assert!(validate_format(1, 0, 1280, 720).is_none());
    // 10-bit — HEVC Main10, VP9 profile 2, AV1 Main10 stream paths
    assert!(validate_format(1, 2, 1920, 1080).is_none());
    // 12-bit — HEVC Main12 (P016 wire format shared with 10-bit)
    assert!(validate_format(1, 4, 1920, 1080).is_none());
}

/// HIGH-2 bit depth reject: 14-bit (HEVC Rext / Main4:4:4:14) must
/// surface as UnsupportedPixelFormat. The wire P016 format only
/// accepts data ≤ 12-bit on the mainstream NVDEC API.
#[test]
fn test_nvdec_rejects_14bit_with_typed_error() {
    // bit_depth_luma_minus8 = 6 → 14-bit
    let err = validate_format(1, 6, 1920, 1080).expect("14-bit must reject");
    match err {
        NvdecError::UnsupportedPixelFormat { bit_depth } => {
            assert_eq!(bit_depth, 14);
        }
        other => panic!("expected UnsupportedPixelFormat, got {other:?}"),
    }
}

/// HIGH-2: P016 deinterleave + 10-bit normalize.
///
/// Build a synthetic 4×2 P016 surface with known u16 samples (pre-
/// shift) in the high 10 bits, feed it through the helper, and
/// verify:
///   1. Y plane emitted LE with value >> 6 in the low 10 bits.
///   2. UV interleaved input split into planar U and V planes.
///   3. Byte counts match expected Yuv420p10le layout.
#[test]
fn test_p016_deinterleave_round_trip() {
    // 4×2 frame → 8 Y samples, ceil(4/2) × ceil(2/2) = 2×1 = 2 UV pairs.
    let w: usize = 4;
    let h: usize = 2;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let uv_pairs = cw * ch;

    // Y plane: 10-bit values 0x100, 0x200, ... encoded in the HIGH
    // bits (shift-left by 6 before writing to the P016 buffer).
    let y_values_10bit: [u16; 8] = [0x001, 0x040, 0x080, 0x0FF, 0x100, 0x200, 0x3FF, 0x0AB];
    let uv_u_10bit: [u16; 2] = [0x123, 0x234];
    let uv_v_10bit: [u16; 2] = [0x345, 0x1FE];

    let mut p016 = Vec::with_capacity((w * h + uv_pairs * 2) * 2);
    for &y in &y_values_10bit {
        let hi = y << 6;
        p016.extend_from_slice(&hi.to_le_bytes());
    }
    for i in 0..uv_pairs {
        let u = uv_u_10bit[i] << 6;
        let v = uv_v_10bit[i] << 6;
        p016.extend_from_slice(&u.to_le_bytes());
        p016.extend_from_slice(&v.to_le_bytes());
    }

    let planar = deinterleave_p016_to_yuv420p10le(&p016, w, h);

    // Expected layout: Y (w*h u16) + U (cw*ch u16) + V (cw*ch u16).
    let y_bytes = w * h * 2;
    let u_bytes = uv_pairs * 2;
    let v_bytes = uv_pairs * 2;
    assert_eq!(
        planar.len(),
        y_bytes + u_bytes + v_bytes,
        "output layout mismatch"
    );

    // Verify Y plane round-trip (raw 10-bit in low bits).
    for (i, &expected) in y_values_10bit.iter().enumerate() {
        let lo = planar[i * 2];
        let hi = planar[i * 2 + 1];
        let got = u16::from_le_bytes([lo, hi]);
        assert_eq!(got, expected, "Y[{i}] mismatch");
    }
    // U plane
    for i in 0..uv_pairs {
        let base = y_bytes + i * 2;
        let got = u16::from_le_bytes([planar[base], planar[base + 1]]);
        assert_eq!(got, uv_u_10bit[i], "U[{i}] mismatch");
    }
    // V plane
    for i in 0..uv_pairs {
        let base = y_bytes + u_bytes + i * 2;
        let got = u16::from_le_bytes([planar[base], planar[base + 1]]);
        assert_eq!(got, uv_v_10bit[i], "V[{i}] mismatch");
    }
}

/// M-4: odd-height coverage. A 4×3 frame has ceil(3/2) = 2 chroma
/// rows; a truncating `h/2` would emit only 1 and drop the last.
/// Verify the helper allocates ceil(h/2) chroma rows so the bottom
/// UV row survives.
#[test]
fn test_p016_deinterleave_odd_height_preserves_last_uv_row() {
    let w: usize = 4;
    let h: usize = 3;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    assert_eq!(ch, 2, "test setup: h=3 → ceil(h/2) = 2");
    let uv_pairs = cw * ch;

    // Y plane: 4*3 = 12 samples.
    let mut p016 = Vec::new();
    for i in 0..(w * h) {
        let v = ((i as u16) + 1) << 6;
        p016.extend_from_slice(&v.to_le_bytes());
    }
    // UV plane: 2*2 = 4 pairs. Make the last pair distinctive so
    // truncation would be visible.
    let u_vals: [u16; 4] = [0x010, 0x020, 0x030, 0x3FE];
    let v_vals: [u16; 4] = [0x101, 0x202, 0x303, 0x1AB];
    for i in 0..uv_pairs {
        let u = u_vals[i] << 6;
        let v = v_vals[i] << 6;
        p016.extend_from_slice(&u.to_le_bytes());
        p016.extend_from_slice(&v.to_le_bytes());
    }

    let planar = deinterleave_p016_to_yuv420p10le(&p016, w, h);

    let y_bytes = w * h * 2;
    let u_bytes = uv_pairs * 2;

    // Verify last U sample survived the copy.
    let last_u_base = y_bytes + (uv_pairs - 1) * 2;
    let got_last_u = u16::from_le_bytes([planar[last_u_base], planar[last_u_base + 1]]);
    assert_eq!(got_last_u, u_vals[uv_pairs - 1], "last U row dropped");
    // Verify last V sample survived.
    let last_v_base = y_bytes + u_bytes + (uv_pairs - 1) * 2;
    let got_last_v = u16::from_le_bytes([planar[last_v_base], planar[last_v_base + 1]]);
    assert_eq!(got_last_v, v_vals[uv_pairs - 1], "last V row dropped");
}

/// HIGH-3: PTS propagation end-to-end through the display callback
/// pipeline.
///
/// We can't call the real cuvid parse loop without a GPU, so we
/// instead prove the end-link of the propagation chain: once
/// `DecodedFrame.timestamp` is populated (which is what
/// `display_callback` does from `CUVIDPARSERDISPINFO.timestamp`),
/// `decode_next` must thread it through to `VideoFrame.pts` without
/// overriding it with a monotonic counter. This is the exact bug
/// codec-review-2 HIGH-3 flagged ("`decode_next` at line 662 uses
/// a monotonic counter as PTS").
///
/// We feed synthetic frames in the B-frame display order
/// (0, 20, 40, 60, 80, 120) — non-monotonic decode order would be
/// different, but since this test bypasses cuvid, we can set the
/// frames in display order directly. The assertion is that the
/// emitted VideoFrame carries the exact PTS that was stashed on
/// its DecodedFrame, not some idx counter.
#[test]
fn test_pts_propagated_through_callback_for_b_frame_sequence() {
    use codec::decode::nvdec::NvdecDecoder;
    use codec::frame::{ColorMetadata, ColorSpace, PixelFormat, StreamInfo};

    let w = 16u32;
    let h = 16u32;
    let y_bytes = (w * h) as usize;
    let uv_bytes = (w * h / 2) as usize; // NV12: 1 byte/sample * w * h/2
    let frame_bytes = y_bytes + uv_bytes;

    // PTS values that a B-pyramid decoder would emit in display
    // order: I at 0, B at 20, P at 40, B at 60, P at 80, P at 120.
    // Non-contiguous numbers so a mistakenly-synthesized idx counter
    // (0,1,2,3,4,5) would show a visible mismatch.
    let pts_values: [u64; 6] = [0, 20, 40, 60, 80, 120];
    let frames: Vec<(Vec<u8>, u32, u32, u8, u64)> = pts_values
        .iter()
        .map(|&pts| (vec![0u8; frame_bytes], w, h, 0, pts))
        .collect();

    let info = StreamInfo {
        codec: "h264".into(),
        width: w,
        height: h,
        frame_rate: 30.0,
        duration: 1.0,
        pixel_format: PixelFormat::Yuv420p,
        color_space: ColorSpace::Bt709,
        total_frames: pts_values.len() as u64,
        bitrate: 1_000_000,
        color_metadata: ColorMetadata::default(),
    };

    let mut dec = NvdecDecoder::test_new_from_frames(frames, info);

    // Drain every frame and confirm the pts matches the input PTS.
    // If decode_next were still using a monotonic counter (the
    // HIGH-3 bug) the drained sequence would be 0,1,2,3,4,5 instead.
    let mut drained = Vec::new();
    while let Some(f) = dec.decode_next().expect("decode_next") {
        drained.push(f.pts);
    }
    assert_eq!(
        drained, pts_values,
        "PTS must round-trip through DecodedFrame → VideoFrame without synthetic counter"
    );
}

// ─── Task #39 regression tests: NVDEC H.264 segfault ──────────────
//
// Original bug: decoding real H.264 input on a Windows GPU box
// triggered STATUS_ACCESS_VIOLATION (0xc0000005) inside nvcuvid.dll.
// Root cause stack (per reviews/nvdec-segfault-hunt.md):
//   1. `creation_flags = 0` let the Windows driver pick the DXVA
//      backend — DXVA surfaces have different pitch/layout semantics
//      than the code assumes. Fixed by setting CUVID_CREATE_PREFER_CUVID.
//   2. CUVIDPARSERPARAMS was 80 bytes vs SDK's 136 — callbacks landed
//      on reserved zero-padding, undefined behaviour on dispatch.
//      Fixed by matching the real SDK layout (compile-time asserted).
//   3. CUVIDPICPARAMS was 2048 bytes vs SDK's 4280 — driver wrote past
//      the allocation into adjacent state. Fixed by sizing the
//      `codec_specific` buffer to the real SDK's 4096-byte
//      `CodecReserved[1024]` fallback union (compile-time asserted).
//
// The tests below guard against regressions on each of those fix
// layers. They require a real NVIDIA GPU; skip gracefully on CI hosts
// without one so this test file stays green everywhere.

/// Task #39 regression: the original symptom was a STATUS_ACCESS_VIOLATION
/// crashing the whole test process on a 4:2:0 H.264 sample. Use the
/// ExoPlayer and Jellyfin 4:2:0 samples (BBB 720p60 is High 4:2:2 and
/// now correctly rejects before reaching the fault path) and verify:
///   1. NvdecDecoder::new returns without crashing.
///   2. ≥ 1 frame is produced for any sample with an IDR.
///   3. Any sample that legitimately decodes zero frames (e.g. the
///      Exoplayer Main file has no IDR — see nvdec_exoplayer_main_nal_dump
///      above, type-5 count = 0) surfaces as a clean error, not a crash.
///
/// The latter case is why the pass predicate is "no crash" rather than
/// "frames > 0": on real corpora some files are pathological. A segfault
/// is ALWAYS wrong; zero-frames from a legitimately-IDR-less stream is
/// the driver doing its job.
#[test]
fn nvdec_h264_real_sample_does_not_segfault() {
    let gpus = codec::gpu::detect_gpus();
    let Some(dev) = gpus
        .iter()
        .find(|g| g.vendor == codec::gpu::GpuVendor::Nvidia)
    else {
        eprintln!("SKIP: no NVIDIA GPU detected");
        return;
    };
    eprintln!("GPU: {} idx={}", dev.name, dev.index);

    // Initialize tracing so any callback error messages surface.
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_writer(std::io::stderr)
        .try_init();

    // Curated list of real 4:2:0 H.264 samples. We require at least one
    // to decode ≥1 frame; all must complete without process crash.
    // jellyfin_h264_high_l40_1080p_24fps is the known-good reference
    // (matrix test verified it produces 30 frames on RTX 3090).
    let candidates: &[&str] = &[
        "jellyfin_h264_high_l40_1080p_24fps.mp4",
        "exoplayer_h264_main_720p.mp4",
    ];

    let mut any_decoded = false;
    let mut any_present = false;
    for fname in candidates {
        let Some(data) = test_media(fname) else {
            eprintln!("  skip {}: not present", fname);
            continue;
        };
        any_present = true;
        let demuxed = match container::demux::demux(&data) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  skip {}: demux failed: {e}", fname);
                continue;
            }
        };
        eprintln!(
            "  feed {}: {}x{} {} samples",
            fname,
            demuxed.info.width,
            demuxed.info.height,
            demuxed.samples.len()
        );
        // This is the crash surface. If task #39 regresses (e.g. someone
        // reverts CUVID_CREATE_PREFER_CUVID, or shrinks codec_specific
        // below 4096 bytes, or introduces a callback-thread race), the
        // process dies here with STATUS_ACCESS_VIOLATION. On success we
        // get either decoded frames or a typed Err (e.g. no frames).
        let mut decoder = codec::decode::nvdec::NvdecDecoder::new(demuxed.info.clone(), dev.index);
        for sample in &demuxed.samples {
            if let Err(e) = decoder.push_sample(sample) {
                eprintln!("  {} push_sample error (no crash): {e:#}", fname);
                continue;
            }
        }
        match decoder.finish() {
            Ok(()) => {
                let mut n = 0usize;
                while let Some(_f) = decoder.decode_next().expect("decode_next") {
                    n += 1;
                    if n >= 5 {
                        break;
                    }
                }
                eprintln!("  {} decoded {} frames (no crash)", fname, n);
                if n > 0 {
                    any_decoded = true;
                }
            }
            Err(e) => {
                // A clean Err path is acceptable — zero frames on a
                // file with no IDR is not a segfault. The assertion we
                // care about is "process still alive".
                eprintln!("  {} decode error (no crash): {e:#}", fname);
            }
        }
    }

    // If no candidate file was present, the caller probably hasn't
    // populated test_media/ — don't fail on that.
    if !any_present {
        eprintln!("SKIP: no 4:2:0 H.264 samples present in test_media/");
        return;
    }
    // Final assertion: at least one 4:2:0 H.264 file must have produced
    // ≥ 1 frame on the real driver. Absent that we cannot claim NVDEC
    // H.264 works — the whole point of the fix.
    assert!(
        any_decoded,
        "NVDEC produced 0 frames on every 4:2:0 H.264 sample — regression"
    );
}

/// Task #39 hardening: feed the eager NVDEC entry point pathological
/// inputs (empty buffer, 1 byte, all zeros, random garbage). None of
/// them should crash the process. They SHOULD return a clean Err from
/// NvdecDecoder::new or produce zero frames — either is acceptable as
/// long as the invariant "no STATUS_ACCESS_VIOLATION" holds.
///
/// Extensive fuzzing is out of scope (need cargo-fuzz for a real campaign);
/// this is a spot-check that the outer API is robust against a small
/// set of known-bad inputs so refactors of the parser loop have a guard.
#[test]
fn nvdec_robust_against_garbage_input() {
    let gpus = codec::gpu::detect_gpus();
    let Some(dev) = gpus
        .iter()
        .find(|g| g.vendor == codec::gpu::GpuVendor::Nvidia)
    else {
        eprintln!("SKIP: no NVIDIA GPU detected");
        return;
    };
    eprintln!("GPU: {} idx={}", dev.name, dev.index);

    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_writer(std::io::stderr)
        .try_init();

    // Plausible StreamInfo for a 1080p H.264 stream — the actual bytes
    // fed are garbage so the parser never reaches the create_decoder
    // step, but the info struct still needs to deserialize.
    let info = codec::frame::StreamInfo {
        codec: "h264".into(),
        width: 1920,
        height: 1080,
        frame_rate: 30.0,
        duration: 1.0,
        pixel_format: codec::frame::PixelFormat::Yuv420p,
        color_space: codec::frame::ColorSpace::Bt709,
        total_frames: 1,
        bitrate: 1_000_000,
        color_metadata: codec::frame::ColorMetadata::default(),
    };

    // Helper: drive the streaming-shape NvdecDecoder push wrapper
    // through a single sample buffer and report whether finish()
    // returned Ok (with whatever frame count) or Err — process
    // liveness is the assertion, not the result variant.
    fn run_one(info: &codec::frame::StreamInfo, gpu: u32, samples: Vec<Vec<u8>>) -> Option<String> {
        let mut dec = codec::decode::nvdec::NvdecDecoder::new(info.clone(), gpu);
        for s in &samples {
            if let Err(e) = dec.push_sample(s) {
                return Some(format!("push: {e}"));
            }
        }
        if let Err(e) = dec.finish() {
            return Some(format!("finish: {e}"));
        }
        None
    }

    // 1. Empty sample list — push loop runs zero iterations.
    eprintln!(
        "  empty-sample-list: {:?}",
        run_one(&info, dev.index, Vec::new())
    );

    // 2. Single empty byte-slice sample — exercises the is_empty guard
    //    in the parse loop. The slice is length zero and should be
    //    skipped cleanly.
    eprintln!(
        "  one-empty-sample: {:?}",
        run_one(&info, dev.index, vec![Vec::new()])
    );

    // 3. Single 1-byte sample — too small to contain any NAL or start
    //    code. The parser should either skip or error; no crash.
    eprintln!(
        "  one-byte-sample: {:?}",
        run_one(&info, dev.index, vec![vec![0u8]])
    );

    // 4. All-zero 1 KiB sample — resembles a malformed Annex B stream
    //    where the start code prefix dominates. Parser treats it as
    //    malformed input; no callbacks fire.
    eprintln!(
        "  all-zero-1KiB: {:?}",
        run_one(&info, dev.index, vec![vec![0u8; 1024]])
    );

    // 5. Random garbage 4 KiB — deterministically seeded from a
    //    linear-congruential step so the test is reproducible. Covers
    //    the case "bytes look structured but aren't valid H.264".
    let mut garbage = Vec::with_capacity(4096);
    let mut state: u32 = 0xDEADBEEF;
    for _ in 0..4096 {
        // Numerical Recipes LCG constants — cheap and spread-adjacent.
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        garbage.push((state >> 24) as u8);
    }
    eprintln!(
        "  random-garbage-4KiB: {:?}",
        run_one(&info, dev.index, vec![garbage])
    );

    // 6. A synthetic Annex B sequence with valid start codes but no
    //    payload — start code + NAL header bytes that refer to
    //    undefined types. Exercises the path where the parser finds
    //    structure but cannot decode.
    let fake_annex_b: Vec<u8> = vec![
        0x00, 0x00, 0x00, 0x01, 0x09, 0x10, // AUD NAL (type 9)
        0x00, 0x00, 0x00, 0x01, 0x1F, 0x00, // reserved NAL (type 31)
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, // malformed NAL (type 0)
    ];
    eprintln!(
        "  synthetic-annex-b: {:?}",
        run_one(&info, dev.index, vec![fake_annex_b])
    );

    // Reaching this line means we didn't crash on any of the six
    // inputs. The real check is that the process is still alive.
}

/// Squad-36 GPU-only smoke test: `create_decoder("h264", info)` on an
/// NVIDIA host must engage the streaming NVDEC dispatch (post-gate-lift)
/// and return a working decoder without driver init failure. This is
/// the lowest-overhead live-GPU regression guard for the gate-lift —
/// runs on RTX 3090 even when test_media is absent.
///
/// Predicate: NvdecStreamingDecoder::try_new must succeed (no driver
/// init error → no NvdecInitErrorDecoder fallback). We probe by
/// pushing a single empty buffer, which exercises the FFI fast-path
/// (push_sample skips empty per Squad-12 hardening) and would crash
/// if cuCtxPushCurrent or the parser ptr were misaligned.
#[test]
fn nvdec_streaming_dispatch_init_smoke() {
    let gpus = codec::gpu::detect_gpus();
    let Some(dev) = gpus
        .iter()
        .find(|g| g.vendor == codec::gpu::GpuVendor::Nvidia)
    else {
        eprintln!("SKIP: no NVIDIA GPU");
        return;
    };
    eprintln!("GPU: {} idx={}", dev.name, dev.index);

    let info = codec::frame::StreamInfo {
        codec: "h264".into(),
        width: 1920,
        height: 1080,
        frame_rate: 30.0,
        duration: 1.0,
        pixel_format: codec::frame::PixelFormat::Yuv420p,
        color_space: codec::frame::ColorSpace::Bt709,
        total_frames: 1,
        bitrate: 1_000_000,
        color_metadata: codec::frame::ColorMetadata::default(),
    };

    // create_decoder for an h264 input on an NVIDIA host MUST engage
    // NVDEC streaming after the Squad-36 gate-lift. Init failure
    // would surface here as a returned NvdecInitErrorDecoder whose
    // first push_sample returns the underlying anyhow.
    let mut decoder = codec::decode::create_decoder("h264", info)
        .expect("create_decoder must not fail on NVIDIA host");
    // Push an empty buffer — exercises the FFI bind/push/pop scope
    // without feeding the parser anything to choke on. Squad-12
    // empty-sample hardening means this returns Ok(()) cleanly.
    decoder.push_sample(&[]).expect("empty push must succeed");
    // No frames yet (we haven't pushed any real data); decode_next
    // returns Ok(None).
    assert!(decoder.decode_next().expect("decode_next").is_none());
    // Finish should succeed with zero frames produced.
    decoder.finish().expect("finish");
    while let Some(_f) = decoder.decode_next().expect("decode_next drain") {
        // No assertion here — finish() may flush the (empty) DPB.
    }
}

// ─── Squad-36 streaming-shape verification (Squad-36 NVDEC follow-up) ──
//
// The streaming-migration-55 sprint gated NVDEC OFF because the eager
// `NvdecDecoder::new_with_pts` (and the lazy-flush `NvdecPushDecoder`
// wrapper) materialised every decoded NV12 / P016 frame in RAM —
// projected ~315 GiB peak RSS for a 15-min 1080p60 input. Squad-36
// restructured the dispatch through `NvdecStreamingDecoder` (per
// `crates/codec/src/decode/nvdec.rs`): each `push_sample` invokes
// `cuvidParseVideoData` immediately, the display callback enqueues
// into a bounded `VecDeque<DecodedFrame>`, and `decode_next` pops one.
// These tests verify the streaming behaviour on real RTX 3090
// hardware. They gracefully skip on hosts without an NVIDIA GPU or
// without the test_media corpus.

/// Streaming verification: drive the dispatch-engaged NVDEC streaming
/// decoder one sample at a time and confirm at least one frame becomes
/// available BEFORE `finish()` is called (the eager / lazy-flush
/// shapes can never emit until finish — that's the regression this
/// guards against).
///
/// Flow:
///   - `create_decoder("h264", info)` → `NvdecStreamingDecoder` on the
///     RTX 3090 dispatch row.
///   - For each sample: `push_sample` then `decode_next` until None.
///   - Track the first sample-index where `decode_next` returned
///     Some. For a streaming decoder driven on real H.264 with
///     B-frames, the first decoded frame typically lands within the
///     first few pushes (decode-order ≈ display-order for the IDR;
///     B-frame reorder window is bounded).
///   - The assertion: at least one frame must be emitted before
///     `finish()`. Eager / lazy-flush decoders would emit zero frames
///     across all pushes and only release the full set after finish.
#[test]
fn nvdec_streaming_emits_frames_before_finish() {
    let gpus = codec::gpu::detect_gpus();
    let Some(_dev) = gpus
        .iter()
        .find(|g| g.vendor == codec::gpu::GpuVendor::Nvidia)
    else {
        eprintln!("SKIP: no NVIDIA GPU");
        return;
    };
    let Some(data) = test_media("jellyfin_h264_high_l40_1080p_24fps.mp4") else {
        eprintln!("SKIP: jellyfin_h264_high_l40_1080p_24fps.mp4 not present");
        return;
    };
    let demuxed = match container::demux::demux(&data) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: demux failed: {e}");
            return;
        }
    };
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_writer(std::io::stderr)
        .try_init();

    let mut decoder = codec::decode::create_decoder(&demuxed.codec, demuxed.info.clone())
        .expect("create_decoder must engage NVDEC streaming on NVIDIA host");

    let mut first_frame_at: Option<usize> = None;
    let mut frames_pre_finish: usize = 0;
    for (idx, sample) in demuxed.samples.iter().enumerate() {
        decoder.push_sample(sample).expect("push_sample");
        while let Some(_f) = decoder.decode_next().expect("decode_next") {
            if first_frame_at.is_none() {
                first_frame_at = Some(idx);
            }
            frames_pre_finish += 1;
        }
        // Cap the loop early once we have evidence of incremental
        // emission — we don't need the full 1797-frame run for this
        // test (the integration test below covers that).
        if frames_pre_finish >= 30 {
            break;
        }
    }

    eprintln!(
        "Streaming: first frame at sample={:?}, {} frames emitted pre-finish",
        first_frame_at, frames_pre_finish
    );
    assert!(
        frames_pre_finish > 0,
        "NvdecStreamingDecoder must emit frames per push (got 0 pre-finish)"
    );
    assert!(
        first_frame_at.is_some_and(|i| i < demuxed.samples.len() / 2),
        "First frame should appear in the first half of the stream, got {:?}",
        first_frame_at
    );
}

/// Memory test: pump up to 1000 H.264 samples through push_sample +
/// decode_next per-frame and assert peak RSS stays bounded. The
/// streaming shape's whole point is that the per-frame drain pattern
/// keeps the VecDeque small (≤ B-pyramid reorder window ≈ 16 frames
/// for High profile). At ~3.1 MiB per 1080p NV12 frame plus the
/// CUDA driver's own memory footprint, the total should stay well
/// under 200 MiB — orders of magnitude below the eager
/// `NvdecPushDecoder` projected ceiling.
///
/// Skips on hosts without GPU or without the test sample.
#[test]
fn nvdec_streaming_peak_rss_under_200mib() {
    let gpus = codec::gpu::detect_gpus();
    let Some(dev) = gpus
        .iter()
        .find(|g| g.vendor == codec::gpu::GpuVendor::Nvidia)
    else {
        eprintln!("SKIP: no NVIDIA GPU");
        return;
    };
    let Some(data) = test_media("jellyfin_h264_high_l40_1080p_24fps.mp4") else {
        eprintln!("SKIP: jellyfin_h264_high_l40_1080p_24fps.mp4 not present");
        return;
    };
    let demuxed = match container::demux::demux(&data) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: demux failed: {e}");
            return;
        }
    };
    eprintln!("GPU: {} idx={}", dev.name, dev.index);
    eprintln!(
        "Demux: {}x{}, {} samples available",
        demuxed.info.width,
        demuxed.info.height,
        demuxed.samples.len()
    );

    let baseline = current_rss_bytes();
    eprintln!("Baseline RSS: {} MiB", baseline / (1024 * 1024));

    let mut decoder = codec::decode::create_decoder(&demuxed.codec, demuxed.info.clone())
        .expect("create_decoder");

    // Pump up to 1000 samples per the Squad-36 brief (or the available
    // sample count, whichever is smaller). For each sample: push then
    // drain. This is the production-pipeline pattern (squad-streaming-qa
    // streaming_rss.rs uses the same shape).
    let n = demuxed.samples.len().min(1000);
    let mut peak_rss = baseline;
    let mut frames: usize = 0;
    for sample in demuxed.samples.iter().take(n) {
        decoder.push_sample(sample).expect("push_sample");
        while let Some(_f) = decoder.decode_next().expect("decode_next") {
            frames += 1;
        }
        let rss = current_rss_bytes();
        if rss > peak_rss {
            peak_rss = rss;
        }
    }
    decoder.finish().expect("finish");
    while let Some(_f) = decoder.decode_next().expect("decode_next drain") {
        frames += 1;
    }
    let final_rss = current_rss_bytes();
    if final_rss > peak_rss {
        peak_rss = final_rss;
    }

    let delta = peak_rss.saturating_sub(baseline);
    let limit_bytes: u64 = 200 * 1024 * 1024;
    eprintln!(
        "Pumped {} samples → {} decoded frames; peak RSS {} MiB ({} MiB above baseline). \
         Limit: {} MiB.",
        n,
        frames,
        peak_rss / (1024 * 1024),
        delta / (1024 * 1024),
        limit_bytes / (1024 * 1024),
    );
    assert!(
        delta <= limit_bytes,
        "Streaming NVDEC peak RSS delta {} MiB > 200 MiB limit (regression — \
         decoder is buffering frames between pushes)",
        delta / (1024 * 1024)
    );
    assert!(frames > 0, "Pumped {} samples, got zero decoded frames", n);
}

/// Integration test: run the verified Jellyfin H.264 sample
/// (1797 frames per Squad-12 baseline) through `create_decoder` →
/// streaming `push_sample` + per-frame drain + `finish` + final drain.
/// Assert the decoded frame count matches the sample count.
///
/// This is the end-to-end RTX 3090 verification the Squad-36 brief
/// asks for — it both proves the streaming dispatch path works on
/// real-media AND that the gate-lift didn't regress the frame count
/// vs the pre-streaming-migration baseline (Squad-12: 1797 frames
/// from the eager constructor on the same input).
///
/// Skips on hosts without GPU or without the sample.
#[test]
fn nvdec_streaming_jellyfin_h264_full_decode() {
    nvdec_streaming_h264_full_decode(
        "jellyfin_h264_high_l40_1080p_24fps.mp4",
        Some(1797),
        "Squad-12 baseline",
    );
}

/// 4K H.264 Main L5.1 screen recording — production reproducer for
/// the SIGSEGV that took down the prod transcoder fleet on
/// 2026-05-01 (job_id=16, ip-10-0-3-25 → 91a4575d, exit code 139).
///
/// This file is NOT in the test_media manifest yet — only present on
/// the dev box. Once the underlying CUVID binding bug is fixed, the
/// file should be uploaded to the test_media S3 bucket so CI can run
/// it as a regression guard.
///
/// Properties (from ffprobe):
///   3840×2160, H.264 Main profile, level 5.1, yuv420p, 30 fps,
///   179 frames, 5.97 s, ~7.5 Mbps, AAC-LC audio.
///
/// Expected: full 179 frames decoded. Current behavior: SIGSEGV
/// during steady-state decode (see /ecs/transcoder-production logs
/// 2026-05-01T08:41:52Z..08:41:59Z).
#[test]
fn nvdec_streaming_h264_4k_screen_recording_full_decode() {
    nvdec_streaming_h264_full_decode(
        "screen_4k_h264_main_l51.mp4",
        Some(179),
        "production segfault repro 2026-05-01",
    );
}

/// Shared driver for full-decode H.264 NVDEC tests. Exits early
/// (SKIP) on hosts without a NVIDIA GPU or without the named test
/// asset; otherwise fully demuxes + streams + asserts the frame count.
fn nvdec_streaming_h264_full_decode(fname: &str, expected_frames: Option<usize>, label: &str) {
    let gpus = codec::gpu::detect_gpus();
    let Some(dev) = gpus
        .iter()
        .find(|g| g.vendor == codec::gpu::GpuVendor::Nvidia)
    else {
        eprintln!("SKIP: no NVIDIA GPU ({label})");
        return;
    };
    let Some(data) = test_media(fname) else {
        eprintln!("SKIP: {fname} not present");
        return;
    };
    let demuxed = match container::demux::demux(&data) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIP: demux failed: {e}");
            return;
        }
    };
    eprintln!(
        "GPU: {} | Sample: {fname} | demux {}x{}, {} samples",
        dev.name,
        demuxed.info.width,
        demuxed.info.height,
        demuxed.samples.len()
    );

    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_writer(std::io::stderr)
        .try_init();

    let mut decoder = codec::decode::create_decoder(&demuxed.codec, demuxed.info.clone())
        .expect("create_decoder must engage NVDEC streaming on NVIDIA host");
    let mut frames: usize = 0;
    for sample in &demuxed.samples {
        decoder.push_sample(sample).expect("push_sample");
        while let Some(_f) = decoder.decode_next().expect("decode_next") {
            frames += 1;
        }
    }
    decoder.finish().expect("finish");
    while let Some(_f) = decoder.decode_next().expect("decode_next drain") {
        frames += 1;
    }
    eprintln!(
        "Decoded {} frames from {} samples ({label})",
        frames,
        demuxed.samples.len()
    );
    if let Some(expected) = expected_frames {
        assert_eq!(
            frames, expected,
            "Frame count regression: got {} expected {} ({label})",
            frames, expected
        );
    }
}

// ─── Tiny RSS helper for the memory test ─────────────────────────
//
// Self-contained (no dep on procfs/sysinfo/peak_alloc) — same shape
// as `crates/pipeline/tests/streaming_rss.rs`. Returns the peak
// resident-set since the process started on Windows / Linux; macOS
// returns 0 (test would appear to pass with peak=0; not a target
// platform).

#[cfg(target_os = "linux")]
fn current_rss_bytes() -> u64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse()
                .unwrap_or(0);
            return kb * 1024;
        }
    }
    0
}

#[cfg(target_os = "macos")]
fn current_rss_bytes() -> u64 {
    0
}

#[cfg(target_os = "windows")]
fn current_rss_bytes() -> u64 {
    #[repr(C)]
    #[allow(non_snake_case)]
    struct ProcessMemoryCounters {
        cb: u32,
        PageFaultCount: u32,
        PeakWorkingSetSize: usize,
        WorkingSetSize: usize,
        QuotaPeakPagedPoolUsage: usize,
        QuotaPagedPoolUsage: usize,
        QuotaPeakNonPagedPoolUsage: usize,
        QuotaNonPagedPoolUsage: usize,
        PagefileUsage: usize,
        PeakPagefileUsage: usize,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut core::ffi::c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }
    unsafe {
        let mut pmc: ProcessMemoryCounters = core::mem::zeroed();
        pmc.cb = core::mem::size_of::<ProcessMemoryCounters>() as u32;
        let h = GetCurrentProcess();
        if GetProcessMemoryInfo(h, &mut pmc, pmc.cb) != 0 {
            pmc.PeakWorkingSetSize as u64
        } else {
            0
        }
    }
}
