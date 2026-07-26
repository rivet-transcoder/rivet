use super::super::demux_ts_streaming_init;
use super::build_encrypted_ts;
use crate::streaming::StreamingDemuxer;

#[test]
fn streaming_demuxer_drops_video_when_active_pid_is_scrambled() {
    let buf = build_encrypted_ts();
    let mut dem = demux_ts_streaming_init(bytes::Bytes::from(buf.clone())).expect("init");
    // First call should hit the encrypted packet, latch the guard,
    // and return None. No samples should ever surface.
    let s = dem.next_video_sample().expect("call must not error");
    assert!(
        s.is_none(),
        "encrypted TS → next_video_sample returns None on first call"
    );
    // Subsequent calls remain None — the guard latches.
    let s2 = dem.next_video_sample().expect("call must not error");
    assert!(
        s2.is_none(),
        "encrypted TS → guard remains latched on subsequent calls"
    );
}
