//! MP3 in an MP4: the `mp4a` sample entry with an MPEG-audio `esds`.

use crate::AudioInfo;
use crate::mux::Av1Mp4Muxer;
use crate::mux::audio_track::build_audio_stsd;

/// The `esds` body inside an `stsd` built for `info`.
fn esds_of(stsd: &[u8]) -> &[u8] {
    let at = stsd.windows(4).position(|w| w == b"esds").expect("an esds box") - 4;
    let size = u32::from_be_bytes(stsd[at..at + 4].try_into().unwrap()) as usize;
    &stsd[at + 12..at + size]
}

#[test]
fn mp3_is_an_mp4a_entry_with_object_type_6b_and_no_decoder_config() {
    let stsd = build_audio_stsd(&AudioInfo::mp3(44_100, 2));
    assert_eq!(&stsd[20..24], b"mp4a", "the sample entry");
    // channelcount 2, samplesize 16, samplerate 44100 in 16.16.
    assert_eq!(&stsd[40..42], &[0, 2]);
    assert_eq!(u32::from_be_bytes(stsd[48..52].try_into().unwrap()), 44_100 << 16);
    let esds = esds_of(&stsd);
    // ES_Descriptor(0x03) { ES_ID 0, flags 0, DecoderConfigDescriptor(0x04)
    //   { OTI 0x6B, AudioStream (0x05 << 2 | 1), buffer/rates } , SLConfig(0x06) { 2 } }
    assert_eq!(esds[0], 0x03);
    let dcd = esds.windows(2).position(|w| w == [0x04, 13]).expect("a 13-byte DecoderConfigDescriptor");
    assert_eq!(esds[dcd + 2], 0x6B, "ISO/IEC 11172-3 audio");
    assert_eq!(esds[dcd + 3], 0x15, "AudioStream, upstream flag");
    assert_eq!(&esds[dcd + 15..], &[0x06, 1, 2], "no DecoderSpecificInfo: the SLConfigDescriptor follows the DCD");
    assert!(crate::mux::audio_track::build_chan_box(&[]).is_none());
}

#[test]
fn mpeg2_rates_take_object_type_69() {
    let stsd = build_audio_stsd(&AudioInfo::mp3(22_050, 1));
    let esds = esds_of(&stsd);
    let dcd = esds.windows(2).position(|w| w == [0x04, 13]).unwrap();
    assert_eq!(esds[dcd + 2], 0x69, "ISO/IEC 13818-3 audio");
    assert_eq!(crate::mux::mp3_object_type(48_000), 0x6B);
}

#[test]
fn the_muxer_takes_mono_and_stereo_mp3_and_refuses_the_rest() {
    assert!(Av1Mp4Muxer::check_audio(&AudioInfo::mp3(48_000, 2)).is_ok());
    assert!(Av1Mp4Muxer::check_audio(&AudioInfo::mp3(32_000, 1)).is_ok());
    assert!(Av1Mp4Muxer::check_audio(&AudioInfo::mp3(48_000, 6)).is_err(), "surround");
    assert!(Av1Mp4Muxer::check_audio(&AudioInfo::mp3(96_000, 2)).is_err(), "no MPEG audio rate");
}
