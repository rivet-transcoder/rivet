use super::aac::esds_object_type;

/// An ES descriptor tree (the `esds` body after its FullBox preamble) whose
/// DecoderConfigDescriptor names `oti`, with an empty DecoderSpecificInfo.
fn esds_with_object_type(oti: u8) -> Vec<u8> {
    let dcd = {
        let mut d = vec![oti, 0x15]; // objectTypeIndication, streamType (audio)
        d.extend_from_slice(&[0, 0, 0]); // bufferSizeDB
        d.extend_from_slice(&[0, 0, 0, 0]); // maxBitrate
        d.extend_from_slice(&[0, 0, 0, 0]); // avgBitrate
        d.extend_from_slice(&[0x05, 0]); // DecoderSpecificInfo, empty
        d
    };
    let mut es = vec![0, 1, 0]; // ES_ID, flags (no optional fields)
    es.push(0x04);
    es.push(dcd.len() as u8);
    es.extend_from_slice(&dcd);
    let mut out = vec![0x03, es.len() as u8];
    out.extend_from_slice(&es);
    out
}

/// ffmpeg's MP4 muxer carries DTS under `mp4a` with the DTS object type
/// registrations; the descriptor walk must surface that byte so the track
/// is routed to the DTS path rather than parsed as an AAC config.
#[test]
fn esds_object_type_is_read_from_the_decoder_config_descriptor() {
    assert_eq!(esds_object_type(&esds_with_object_type(0xA9)), Some(0xA9), "DTS core");
    assert_eq!(esds_object_type(&esds_with_object_type(0xAB)), Some(0xAB), "DTS-HD MA");
    assert_eq!(esds_object_type(&esds_with_object_type(0x40)), Some(0x40), "MPEG-4 audio");
    // Not an ES_Descriptor at all.
    assert_eq!(esds_object_type(&[0x04, 2, 0xA9, 0x15]), None);
    assert_eq!(esds_object_type(&[]), None);
}
