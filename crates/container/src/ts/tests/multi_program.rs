use super::super::{demux_ts_streaming_init, STREAM_TYPE_H264, STREAM_TYPE_MPEG2_VIDEO};
use super::build_two_program_ts;
use crate::streaming::StreamingDemuxer;

#[test]
fn streaming_demuxer_lists_all_pat_programs() {
    let buf = build_two_program_ts();
    let dem = demux_ts_streaming_init(bytes::Bytes::from(buf.clone())).expect("init");
    let progs = dem.programs();
    assert_eq!(progs.len(), 2, "PAT advertised 2 programs");
    let nums: Vec<u16> = progs.iter().map(|p| p.program_number).collect();
    assert_eq!(nums, vec![1, 2]);
    assert_eq!(progs[0].pmt_pid, 0x100);
    assert_eq!(progs[1].pmt_pid, 0x101);
    // Program 1 → MPEG-2 on 0x200; program 2 → H.264 on 0x300.
    assert_eq!(progs[0].video_streams[0].pid, 0x200);
    assert_eq!(
        progs[0].video_streams[0].stream_type,
        STREAM_TYPE_MPEG2_VIDEO
    );
    assert_eq!(progs[1].video_streams[0].pid, 0x300);
    assert_eq!(progs[1].video_streams[0].stream_type, STREAM_TYPE_H264);
}

#[test]
fn streaming_demuxer_default_picks_first_program() {
    let buf = build_two_program_ts();
    let mut dem = demux_ts_streaming_init(bytes::Bytes::from(buf.clone())).expect("init");
    assert_eq!(dem.active_program_index(), 0);
    assert_eq!(dem.header().codec, "mpeg2", "program 1 is MPEG-2");
    // Drain — samples should be 0xAA-filled (program 1's bytes).
    let s = dem.next_video_sample().expect("sample").expect("some");
    assert!(
        s.data.iter().any(|&b| b == 0xAA),
        "program 1 sample should carry 0xAA"
    );
    assert!(
        !s.data.iter().any(|&b| b == 0xBB),
        "program 1 sample must not carry program 2's 0xBB"
    );
}

#[test]
fn streaming_demuxer_select_program_switches_active_streams() {
    let buf = build_two_program_ts();
    let mut dem = demux_ts_streaming_init(bytes::Bytes::from(buf.clone())).expect("init");
    dem.select_program(2).expect("switch to program 2");
    assert_eq!(dem.active_program_index(), 1);
    assert_eq!(dem.header().codec, "h264", "program 2 is H.264");
    let s = dem.next_video_sample().expect("sample").expect("some");
    assert!(
        s.data.iter().any(|&b| b == 0xBB),
        "program 2 sample should carry 0xBB"
    );
    assert!(
        !s.data.iter().any(|&b| b == 0xAA),
        "program 2 sample must not carry program 1's 0xAA"
    );
}

#[test]
fn streaming_demuxer_select_program_rejects_unknown_number() {
    let buf = build_two_program_ts();
    let mut dem = demux_ts_streaming_init(bytes::Bytes::from(buf.clone())).expect("init");
    assert!(
        dem.select_program(99).is_err(),
        "unknown program_number must error rather than silently no-op"
    );
}
