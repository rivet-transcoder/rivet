//! Container detection from the first bytes — the one answer every dispatch
//! in this crate, and every caller gating an upload, reads.
//!
//! # Structure, not brand
//!
//! ISO BMFF is recognised by the box that opens the file — `ftyp`, or a bare
//! `moov`/`mdat` for older MOV — and deliberately **not** by the `ftyp` major
//! brand. The brand space is open: every recorder vendor is free to mint one,
//! so a brand a sniffer has not heard of is not evidence of anything. A
//! 289 MB recording with major brand `nvr1` and compatible brands `isom` /
//! `mp42` was once rejected as "unrecognized format" by a brand-list sniffer,
//! while the demuxer beside it would have accepted the same bytes, because
//! it looks for the box and not the brand. The compatible-brands list exists
//! precisely so a reader can accept a file whose major brand it does not
//! know; anything that is ISO BMFF goes to the demuxer, which reads what is
//! actually in the file rather than what the first twelve bytes are named.
//!
//! This is the same test the demuxers dispatch on, shared so a gate and the
//! demuxer cannot drift into disagreeing about one file — which is what
//! happened.

/// A container family, as far as the first bytes can tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContainerKind {
    /// ISO base media file format: MP4, MOV, 3GP, CMAF — a `ftyp` box first,
    /// or a bare `moov` / `mdat` for older QuickTime files.
    IsoBmff,
    /// Matroska / WebM: the EBML magic.
    Matroska,
    /// RIFF AVI.
    Avi,
    /// MPEG transport stream: a `0x47` sync byte on the 188-byte grid.
    MpegTs,
    /// MPEG program stream (`.mpg`, `.vob`): a pack start code, `00 00 01 BA`.
    MpegPs,
    /// A bare MPEG audio file (`.mp3` / `.mp2`): an ID3v2 tag, or frame
    /// headers that agree with each other. Audio only.
    Mp3,
    /// A native FLAC stream: the `fLaC` marker, possibly after an ID3v2 tag.
    /// Audio only, so it is a source for the audio-only output mode.
    Flac,
    /// An Ogg file (`OggS` pages): Opus or Vorbis audio. Read for its audio
    /// alone ([`crate::ogg`]).
    Ogg,
    /// Nothing this crate demuxes.
    Unknown,
}

impl ContainerKind {
    /// The short label the demux dispatch and `probe` report: `"mp4"`,
    /// `"mkv"`, `"avi"`, `"ts"`, `"ps"`, `"mp3"`, `"flac"`, `"ogg"`, `"unknown"`.
    pub fn label(self) -> &'static str {
        match self {
            ContainerKind::IsoBmff => "mp4",
            ContainerKind::Matroska => "mkv",
            ContainerKind::Avi => "avi",
            ContainerKind::MpegTs => "ts",
            ContainerKind::MpegPs => "ps",
            ContainerKind::Mp3 => "mp3",
            ContainerKind::Flac => "flac",
            ContainerKind::Ogg => "ogg",
            ContainerKind::Unknown => "unknown",
        }
    }

    /// Whether this crate has a demuxer for it.
    pub fn is_known(self) -> bool {
        self != ContainerKind::Unknown
    }
}

/// Which container the bytes open with. Needs at least 12 bytes to say
/// anything; fewer is [`ContainerKind::Unknown`].
pub fn sniff_container(data: &[u8]) -> ContainerKind {
    if data.len() < 12 {
        return ContainerKind::Unknown;
    }
    // ISOBMFF: MP4 (`ftyp mp41`/`mp42`/`isom`/…) and MOV (`ftyp qt  `) both
    // land here. Older MOV files sometimes ship without a top-level `ftyp`
    // and lead with `moov` or `mdat` directly — accept those too. The brand
    // is not consulted; see the module docs.
    if matches!(&data[4..8], b"ftyp" | b"moov" | b"mdat") {
        return ContainerKind::IsoBmff;
    }
    // Matroska/WebM: EBML signature.
    if data[0] == 0x1A && data[1] == 0x45 && data[2] == 0xDF && data[3] == 0xA3 {
        return ContainerKind::Matroska;
    }
    // RIFF-based AVI: "RIFF" <size> "AVI ".
    if &data[..4] == b"RIFF" && &data[8..12] == b"AVI " {
        return ContainerKind::Avi;
    }
    // MPEG-TS: 0x47 sync byte at offset 0 AND at offset 188 (and 376 if we
    // have the bytes). A single 0x47 appears routinely in random payloads, so
    // require two confirming hits before committing.
    if data[0] == 0x47
        && data.len() > 188
        && data[188] == 0x47
        && (data.len() <= 376 || data[376] == 0x47)
    {
        return ContainerKind::MpegTs;
    }
    // Ogg: the capture pattern and stream structure version 0.
    if crate::ogg::sniff(data) {
        return ContainerKind::Ogg;
    }
    // MPEG program stream: a pack header opens it.
    if crate::ps::is_program_stream(data) {
        return ContainerKind::MpegPs;
    }
    // A FLAC stream may open with an ID3v2 tag too: its `fLaC` marker is
    // looked for before the MP3 sniff takes the tag as MPEG audio.
    if crate::demux::audio::lossless::native_flac_offset(data).is_some() {
        return ContainerKind::Flac;
    }
    // Last: an MPEG audio frame header is only two bytes of pattern, so it is
    // taken only when a second header sits where the first says it ends.
    if crate::mp3::sniff(data) {
        return ContainerKind::Mp3;
    }
    ContainerKind::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unfamiliar_ftyp_brand_is_still_iso_bmff() {
        // The production case: major brand `nvr1`, which no brand list knows.
        let mut data = vec![0x00, 0x00, 0x00, 0x20];
        data.extend_from_slice(b"ftyp");
        data.extend_from_slice(b"nvr1");
        data.extend_from_slice(&[0, 0, 0, 0]);
        data.extend_from_slice(b"isommp42");
        assert_eq!(sniff_container(&data), ContainerKind::IsoBmff);
    }

    #[test]
    fn a_bare_moov_or_mdat_is_iso_bmff_too() {
        for open in [b"moov", b"mdat"] {
            let mut data = vec![0x00, 0x00, 0x00, 0x08];
            data.extend_from_slice(open);
            data.extend_from_slice(&[0; 8]);
            assert_eq!(sniff_container(&data), ContainerKind::IsoBmff, "{open:?}");
        }
    }

    #[test]
    fn the_other_families_and_the_unknowns() {
        let mkv = [0x1A, 0x45, 0xDF, 0xA3, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(sniff_container(&mkv), ContainerKind::Matroska);

        let mut avi = Vec::from(*b"RIFF");
        avi.extend_from_slice(&[0, 0, 0, 0]);
        avi.extend_from_slice(b"AVI ");
        assert_eq!(sniff_container(&avi), ContainerKind::Avi);

        let mut ts = vec![0u8; 190];
        ts[0] = 0x47;
        ts[188] = 0x47;
        assert_eq!(sniff_container(&ts), ContainerKind::MpegTs);

        // A lone 0x47 is not a transport stream.
        let mut not_ts = vec![0u8; 190];
        not_ts[0] = 0x47;
        assert_eq!(sniff_container(&not_ts), ContainerKind::Unknown);

        assert_eq!(sniff_container(b"hello, this is plain text"), ContainerKind::Unknown);

        let mut mp3 = b"ID3\x04\x00\x00\x00\x00\x00\x00".to_vec();
        mp3.extend_from_slice(&[0xFF, 0xFB, 0x90, 0x44]);
        assert_eq!(sniff_container(&mp3), ContainerKind::Mp3);
        assert_eq!(ContainerKind::Mp3.label(), "mp3");
        assert_eq!(sniff_container(&[0u8; 4]), ContainerKind::Unknown, "too short to say");
        assert!(!ContainerKind::Unknown.is_known());
        assert_eq!(ContainerKind::IsoBmff.label(), "mp4");
    }
}
