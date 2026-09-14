//! A source's colour is a property of the stream, whichever container carries
//! it: one real encode per demuxer × codec × case, each read through both the
//! whole-file demuxer (`demux`) and the streaming one (`demux_streaming`).
//!
//! The fixtures come from `tests/fixtures/colour/make_fixtures.sh` — x264 /
//! x265 two-frame 64x64 encodes with the colour in the bitstream and none in
//! the container (MP4 `colr` / `mdcv` / `clli` renamed `free`, the Matroska
//! `Colour` element voided, TS and AVI carrying none by nature):
//!
//! - `*_601`: SPS VUI `colour_primaries` 6, `transfer_characteristics` 6,
//!   `matrix_coefficients` 6, limited range.
//! - `*_pq`: SPS VUI 9 / 16 / 9, 10-bit, and the HDR10 static metadata only as
//!   SEI 137 (G 13250,34500 B 7500,3000 R 34000,16000 WP 15635,16450,
//!   L 40000000,50) and SEI 144 (1234, 567).

use frame::{ColorSpace, ContentLightLevel, MasteringDisplay, PixelFormat, StreamInfo, TransferFn};

macro_rules! fixture {
    ($name:literal) => {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/colour/",
            $name
        ))
    };
}

const MASTERING: MasteringDisplay = MasteringDisplay {
    primaries_r_x: 34000,
    primaries_r_y: 16000,
    primaries_g_x: 13250,
    primaries_g_y: 34500,
    primaries_b_x: 7500,
    primaries_b_y: 3000,
    white_point_x: 15635,
    white_point_y: 16450,
    max_luminance: 40_000_000,
    min_luminance: 50,
};
const CLL: ContentLightLevel = ContentLightLevel {
    max_cll: 1234,
    max_fall: 567,
};

/// The `StreamInfo` both readers produce for one file.
fn both_readers(data: &[u8]) -> [(&'static str, StreamInfo); 2] {
    let whole = crate::demux::demux(data).expect("whole-file demux").info;
    let streaming = crate::streaming::demux_streaming(data)
        .expect("streaming demux")
        .header()
        .info
        .clone();
    [("demux", whole), ("demux_streaming", streaming)]
}

fn assert_bt601_from_the_vui(name: &str, data: &[u8]) {
    for (reader, info) in both_readers(data) {
        let c = info.color_metadata;
        assert_eq!(
            (
                c.colour_primaries,
                c.transfer,
                c.matrix_coefficients,
                c.full_range
            ),
            (6, TransferFn::Bt709, 6, false),
            "{name} via {reader}: {c:?}"
        );
        assert_eq!(info.color_space, ColorSpace::Bt601, "{name} via {reader}");
        assert_eq!(info.pixel_format, PixelFormat::Yuv420p, "{name} via {reader}");
        assert_eq!(c.mastering_display, None, "{name} via {reader}");
        assert_eq!(c.content_light_level, None, "{name} via {reader}");
    }
}

fn assert_pq_from_the_vui_and_hdr10_from_the_seis(name: &str, data: &[u8]) {
    for (reader, info) in both_readers(data) {
        let c = info.color_metadata;
        assert_eq!(
            (
                c.colour_primaries,
                c.transfer,
                c.matrix_coefficients,
                c.full_range
            ),
            (9, TransferFn::St2084, 9, false),
            "{name} via {reader}: {c:?}"
        );
        assert_eq!(info.color_space, ColorSpace::Bt2020, "{name} via {reader}");
        // 10-bit already at open: the pipeline sizes its encoder from the header
        // before it pulls a sample, and an HDR source left at the 8-bit default
        // went out as 8-bit PQ under `--color passthrough`.
        assert_eq!(info.pixel_format, PixelFormat::Yuv420p10le, "{name} via {reader}");
        assert_eq!(c.mastering_display, Some(MASTERING), "{name} via {reader}");
        assert_eq!(c.content_light_level, Some(CLL), "{name} via {reader}");
    }
}

/// A transport stream cut two frames into a four-frame GOP: its first access
/// unit has no SPS. Asserted, so the case cannot pass on a fixture that no
/// longer starts mid-GOP; the dimensions come from the later SPS too (before
/// the colour window the streaming reader said 0x0).
fn assert_first_access_unit_has_no_sps(name: &str, data: &[u8]) {
    let codec = if name.starts_with("hevc") {
        "h265"
    } else {
        "h264"
    };
    let first = crate::demux::demux(data)
        .expect("whole-file demux")
        .samples
        .into_iter()
        .next()
        .expect("a sample");
    let head = crate::demux::hdr::colour_window(codec, [first.as_slice()], "t").expect("H.26x");
    assert!(
        !head.has_sps,
        "{name}: the first access unit carries an SPS, so the fixture no longer starts mid-GOP"
    );
    for (reader, info) in both_readers(data) {
        assert_eq!((info.width, info.height), (64, 64), "{name} via {reader}");
    }
}

fn assert_bt601_from_a_mid_gop_start(name: &str, data: &[u8]) {
    assert_first_access_unit_has_no_sps(name, data);
    assert_bt601_from_the_vui(name, data);
}

fn assert_pq_and_hdr10_from_a_mid_gop_start(name: &str, data: &[u8]) {
    assert_first_access_unit_has_no_sps(name, data);
    assert_pq_from_the_vui_and_hdr10_from_the_seis(name, data);
}

macro_rules! case {
    ($test:ident, $file:literal, $check:ident) => {
        #[test]
        fn $test() {
            $check($file, fixture!($file));
        }
    };
}

case!(
    mp4_h264_bt601_from_the_sps_vui,
    "h264_601.mp4",
    assert_bt601_from_the_vui
);
case!(
    mp4_hevc_bt601_from_the_sps_vui,
    "hevc_601.mp4",
    assert_bt601_from_the_vui
);
case!(
    mp4_h264_pq_from_the_vui_hdr10_from_the_seis,
    "h264_pq.mp4",
    assert_pq_from_the_vui_and_hdr10_from_the_seis
);
case!(
    mp4_hevc_pq_from_the_vui_hdr10_from_the_seis,
    "hevc_pq.mp4",
    assert_pq_from_the_vui_and_hdr10_from_the_seis
);

case!(
    mkv_h264_bt601_from_the_sps_vui,
    "h264_601.mkv",
    assert_bt601_from_the_vui
);
case!(
    mkv_hevc_bt601_from_the_sps_vui,
    "hevc_601.mkv",
    assert_bt601_from_the_vui
);
case!(
    mkv_h264_pq_from_the_vui_hdr10_from_the_seis,
    "h264_pq.mkv",
    assert_pq_from_the_vui_and_hdr10_from_the_seis
);
case!(
    mkv_hevc_pq_from_the_vui_hdr10_from_the_seis,
    "hevc_pq.mkv",
    assert_pq_from_the_vui_and_hdr10_from_the_seis
);

case!(
    ts_h264_bt601_from_the_sps_vui,
    "h264_601.ts",
    assert_bt601_from_the_vui
);
case!(
    ts_hevc_bt601_from_the_sps_vui,
    "hevc_601.ts",
    assert_bt601_from_the_vui
);
case!(
    ts_h264_pq_from_the_vui_hdr10_from_the_seis,
    "h264_pq.ts",
    assert_pq_from_the_vui_and_hdr10_from_the_seis
);
case!(
    ts_hevc_pq_from_the_vui_hdr10_from_the_seis,
    "hevc_pq.ts",
    assert_pq_from_the_vui_and_hdr10_from_the_seis
);

case!(
    ts_h264_bt601_from_a_mid_gop_start,
    "h264_601_midgop.ts",
    assert_bt601_from_a_mid_gop_start
);
case!(
    ts_hevc_pq_and_hdr10_from_a_mid_gop_start,
    "hevc_pq_midgop.ts",
    assert_pq_and_hdr10_from_a_mid_gop_start
);

case!(
    avi_h264_bt601_from_the_sps_vui,
    "h264_601.avi",
    assert_bt601_from_the_vui
);
case!(
    avi_h264_pq_from_the_vui_hdr10_from_the_seis,
    "h264_pq.avi",
    assert_pq_from_the_vui_and_hdr10_from_the_seis
);

/// A transport stream cut mid-GOP opens with access units no decoder can
/// decode. The streaming reader drops them, so its first sample is the
/// random-access point, and keeps the time they filled as the video's late
/// start; the whole-file reader, a plain sample list, still has them.
fn assert_starts_at_its_first_random_access_point(name: &str, data: &[u8]) {
    use crate::nal_mux::{NalMuxCodec, sample_is_keyframe};
    let codec = if name.starts_with("hevc") {
        NalMuxCodec::H265
    } else {
        NalMuxCodec::H264
    };
    let all = crate::demux::demux(data).expect("whole-file demux").samples;
    let lead = all
        .iter()
        .position(|s| sample_is_keyframe(s, codec))
        .expect("a random-access point");
    assert_eq!(
        lead, 2,
        "{name}: the fixture opens two frames before its IRAP"
    );

    let mut demuxer = crate::streaming::demux_streaming(data).expect("streaming demux");
    let p = demuxer
        .video_presentation()
        .cloned()
        .expect("a mid-GOP start is a late start");
    assert_eq!(
        (p.delay_ticks, p.delay_timescale, p.hidden.len()),
        (7200, 90_000, 0),
        "{name}: two frames at 25 fps, nothing hidden"
    );
    let mut samples = Vec::new();
    while let Some(s) = demuxer.next_video_sample().expect("sample") {
        samples.push(s.data);
    }
    assert!(
        sample_is_keyframe(&samples[0], codec),
        "{name}: the first sample is the IRAP"
    );
    assert_eq!(
        samples.len(),
        all.len() - lead,
        "{name}: every sample from it on"
    );
    assert_eq!(samples[0], all[lead], "{name}: the same bytes");
}

/// A stream that opens on its random-access point is left exactly as it was.
fn assert_nothing_dropped(name: &str, data: &[u8]) {
    let all = crate::demux::demux(data).expect("whole-file demux").samples;
    let mut demuxer = crate::streaming::demux_streaming(data).expect("streaming demux");
    assert!(demuxer.video_presentation().is_none(), "{name}");
    let mut n = 0;
    while demuxer.next_video_sample().expect("sample").is_some() {
        n += 1;
    }
    assert_eq!(n, all.len(), "{name}");
}

case!(
    ts_h264_mid_gop_starts_at_its_idr_and_keeps_the_time,
    "h264_601_midgop.ts",
    assert_starts_at_its_first_random_access_point
);
case!(
    ts_hevc_mid_gop_starts_at_its_irap_and_keeps_the_time,
    "hevc_pq_midgop.ts",
    assert_starts_at_its_first_random_access_point
);
case!(
    ts_h264_opening_on_an_idr_drops_nothing,
    "h264_601.ts",
    assert_nothing_dropped
);
case!(
    ts_hevc_opening_on_an_irap_drops_nothing,
    "hevc_pq.ts",
    assert_nothing_dropped
);
