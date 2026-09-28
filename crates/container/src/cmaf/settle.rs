//! The H.264 / H.265 sample entry of a CMAF rendition, settled once every
//! segment is written.
//!
//! [`CmafVideoMuxer`](super::CmafVideoMuxer) writes `init.mp4` when its first
//! segment is flushed, with the parameter sets it has seen by then in
//! `avcC` / `hvcC` under the `avc1` / `hvc1` sample entry, and keeps the sets
//! in band in every segment too. A rendition encoded by one encoder stays that
//! way: its sets are fixed, the in-band copies repeat the config box byte for
//! byte, and `avc1` / `hvc1` is what every player takes — Safari's `<video>`
//! element on iOS refuses `avc3`, and Apple's HLS authoring specification asks
//! for `hvc1`.
//!
//! A rendition whose segments came from several encoders (the multi-GPU
//! helpers, possibly other vendors) can carry sets the init segment does not:
//! the same id with other contents, or an id it never saw. Out of band, a
//! decoder would read those segments with the wrong sets. [`settle_video_sample_entry`]
//! reads every segment's in-band sets against the config box and, where any
//! differs, rewrites the entry as `avc3` / `hev1` — whose sets may change in
//! band — so each segment decodes with its own. Only the fourcc (and the
//! `hvcC` arrays' completeness bit) change: the init segment keeps its size.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};

use crate::nal_mux::{NalMuxCodec, ParamSetLedger};

/// Settle the video sample entry of the rendition whose init segment is at
/// `init_path`, given its media segments in order.
///
/// Returns the sample entry now in the init segment: `avc1` / `hvc1` when
/// every parameter set in the segments is one the config box holds, `avc3` /
/// `hev1` when one is not. `None` for an entry with no parameter sets to
/// settle (`av01`). The file is rewritten only when the entry changes.
pub fn settle_video_sample_entry(init_path: &Path, segments: &[PathBuf]) -> Result<Option<[u8; 4]>> {
    let mut init = std::fs::read(init_path)
        .with_context(|| format!("reading init segment {}", init_path.display()))?;
    let Some(entry) = video_sample_entry(&init) else {
        bail!("no sample entry in init segment {}", init_path.display());
    };
    let fourcc: [u8; 4] = init[entry.start + 4..entry.start + 8].try_into().unwrap();
    let codec = match &fourcc {
        b"avc1" | b"avc3" => NalMuxCodec::H264,
        b"hvc1" | b"hev1" => NalMuxCodec::H265,
        _ => return Ok(None),
    };
    // Visual sample entry: 8-byte box header + 78-byte VisualSampleEntry
    // header, then the config box (and colr etc.).
    let config_kind = match codec {
        NalMuxCodec::H264 => b"avcC",
        NalMuxCodec::H265 => b"hvcC",
    };
    let config = child_boxes(&init, entry.start + 8 + 78, entry.end)
        .find(|b| &init[b.start + 4..b.start + 8] == config_kind)
        .with_context(|| format!("{} box missing", String::from_utf8_lossy(config_kind)))?;
    let record = parse_config(&init[config.start + 8..config.end], codec)?;

    let mut ledger = ParamSetLedger::new(codec);
    for nal in &record.sets {
        ledger.seed(nal);
    }
    let mut fixed = true;
    'segments: for path in segments {
        let seg = std::fs::read(path).with_context(|| format!("reading segment {}", path.display()))?;
        for mdat in child_boxes(&seg, 0, seg.len()).filter(|b| &seg[b.start + 4..b.start + 8] == b"mdat") {
            let nals = LengthPrefixed { data: &seg[mdat.body..mdat.end], size: record.length_size };
            for nal in nals {
                let nal = nal.with_context(|| format!("walking the NAL units of {}", path.display()))?;
                if !ledger.describes(nal) {
                    fixed = false;
                    break 'segments;
                }
            }
        }
    }

    let settled: [u8; 4] = match (codec, fixed) {
        (NalMuxCodec::H264, true) => *b"avc1",
        (NalMuxCodec::H264, false) => *b"avc3",
        (NalMuxCodec::H265, true) => *b"hvc1",
        (NalMuxCodec::H265, false) => *b"hev1",
    };
    let before = init.clone();
    init[entry.start + 4..entry.start + 8].copy_from_slice(&settled);
    for at in record.array_headers {
        // array_completeness: every set of the kind is in the array, none in
        // band — hvc1's rule; hev1 may carry them in band.
        let byte = &mut init[config.start + 8 + at];
        *byte = if fixed { *byte | 0x80 } else { *byte & 0x7F };
    }
    if init != before {
        std::fs::write(init_path, &init)
            .with_context(|| format!("rewriting init segment {}", init_path.display()))?;
    }
    Ok(Some(settled))
}

/// A box's extent in a buffer: `start..end`, its body from `body`.
#[derive(Debug, Clone, Copy)]
struct BoxAt {
    start: usize,
    body: usize,
    end: usize,
}

/// The boxes laid end to end in `buf[from..to]`. Stops at the first box that
/// does not fit.
fn child_boxes(buf: &[u8], from: usize, to: usize) -> impl Iterator<Item = BoxAt> + '_ {
    let mut pos = from;
    std::iter::from_fn(move || {
        if pos + 8 > to {
            return None;
        }
        let size32 = u32::from_be_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        let (size, header) = match size32 {
            0 => (to - pos, 8),
            1 => {
                let large = buf.get(pos + 8..pos + 16)?;
                (u64::from_be_bytes(large.try_into().unwrap()) as usize, 16)
            }
            n => (n, 8),
        };
        if size < header || pos + size > to {
            return None;
        }
        let b = BoxAt { start: pos, body: pos + header, end: pos + size };
        pos += size;
        Some(b)
    })
}

/// The one visual sample entry of `moov/trak/mdia/minf/stbl/stsd`.
fn video_sample_entry(init: &[u8]) -> Option<BoxAt> {
    let mut at = BoxAt { start: 0, body: 0, end: init.len() };
    for kind in [b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"] {
        at = child_boxes(init, at.body, at.end).find(|b| &init[b.start + 4..b.start + 8] == kind)?;
    }
    // stsd: version/flags (4) + entry_count (4), then the entries.
    child_boxes(init, at.body + 8, at.end).next()
}

/// What the settle needs from an `avcC` / `hvcC` record.
struct ConfigRecord {
    /// NAL length-prefix size in the samples, bytes.
    length_size: usize,
    /// Every parameter set NAL unit the record holds.
    sets: Vec<Vec<u8>>,
    /// `hvcC` only: offset into the record body of each array's header byte
    /// (array_completeness | NAL_unit_type).
    array_headers: Vec<usize>,
}

fn parse_config(body: &[u8], codec: NalMuxCodec) -> Result<ConfigRecord> {
    let mut sets = Vec::new();
    let mut array_headers = Vec::new();
    let byte = |at: usize| body.get(at).copied().context("config record truncated");
    let mut take_nal = |at: &mut usize| -> Result<()> {
        let len = u16::from_be_bytes([byte(*at)?, byte(*at + 1)?]) as usize;
        let nal = body.get(*at + 2..*at + 2 + len).context("config record truncated")?;
        sets.push(nal.to_vec());
        *at += 2 + len;
        Ok(())
    };
    let length_size = match codec {
        NalMuxCodec::H264 => {
            // [4] lengthSizeMinusOne, [5] numOfSPS, SPS…, numOfPPS, PPS…
            let length_size = (byte(4)? & 0x03) as usize + 1;
            let mut at = 6;
            for _ in 0..(byte(5)? & 0x1F) {
                take_nal(&mut at)?;
            }
            let pps = byte(at)?;
            at += 1;
            for _ in 0..pps {
                take_nal(&mut at)?;
            }
            length_size
        }
        NalMuxCodec::H265 => {
            // [21] lengthSizeMinusOne, [22] numOfArrays, then each array:
            // completeness|type, numNalus (u16), NAL units.
            let length_size = (byte(21)? & 0x03) as usize + 1;
            let mut at = 23;
            for _ in 0..byte(22)? {
                array_headers.push(at);
                let n = u16::from_be_bytes([byte(at + 1)?, byte(at + 2)?]);
                at += 3;
                for _ in 0..n {
                    take_nal(&mut at)?;
                }
            }
            length_size
        }
    };
    Ok(ConfigRecord { length_size, sets, array_headers })
}

/// The NAL units of a length-prefixed buffer — a sample, or a whole `mdat`,
/// whose samples lie end to end.
struct LengthPrefixed<'a> {
    data: &'a [u8],
    size: usize,
}

impl<'a> Iterator for LengthPrefixed<'a> {
    type Item = Result<&'a [u8]>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.data.is_empty() {
            return None;
        }
        let Some(prefix) = self.data.get(..self.size) else {
            self.data = &[];
            return Some(Err(anyhow::anyhow!("NAL length prefix truncated")));
        };
        let len = prefix.iter().fold(0usize, |n, &b| n << 8 | b as usize);
        let Some(nal) = self.data.get(self.size..self.size + len) else {
            self.data = &[];
            return Some(Err(anyhow::anyhow!("NAL unit of {len} bytes overruns the mdat")));
        };
        self.data = &self.data[self.size + len..];
        Some(Ok(nal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmaf::{CmafVideoMuxer, CmafVideoMuxerOptions};
    use frame::{ColorMetadata, VideoCodec};

    fn au(nals: &[&[u8]]) -> Vec<u8> {
        let mut v = Vec::new();
        for n in nals {
            v.extend_from_slice(&[0, 0, 0, 1]);
            v.extend_from_slice(n);
        }
        v
    }

    fn entry_fourcc(init: &Path) -> [u8; 4] {
        let bytes = std::fs::read(init).unwrap();
        let e = video_sample_entry(&bytes).unwrap();
        bytes[e.start + 4..e.start + 8].try_into().unwrap()
    }

    const SPS: [u8; 5] = [0x67, 0x42, 0x00, 0x1e, 0xAA];
    /// `pic_parameter_set_id` 0, `seq_parameter_set_id` 0, then a byte that
    /// stands for the rest of the set.
    const PPS: [u8; 3] = [0x68, 0xCE, 0x3C];
    const PPS_CHANGED: [u8; 3] = [0x68, 0xCE, 0x38];
    const IDR: [u8; 4] = [0x65, 0x88, 0x11, 0x22];

    /// A rendition written by a primary muxer and a helper muxer (the
    /// multi-GPU split), each with its own encoder's parameter sets.
    fn two_encoder_rendition(helper_pps: &[u8]) -> (tempfile::TempDir, Vec<PathBuf>) {
        let dir = tempfile::tempdir().unwrap();
        let open = |first: u32, init: bool| {
            CmafVideoMuxer::new_with_codec_options(
                dir.path(),
                640,
                360,
                30000,
                ColorMetadata::default(),
                VideoCodec::H264,
                CmafVideoMuxerOptions {
                    first_segment_index: first,
                    write_init_segment: init,
                    ..CmafVideoMuxerOptions::default()
                },
            )
            .unwrap()
        };
        let mut primary = open(1, true);
        primary.add_packet(au(&[&SPS, &PPS, &IDR]), 1000, true, 0).unwrap();
        let a = primary.flush_segment().unwrap().unwrap();
        primary.finalize().unwrap();
        let mut helper = open(2, false);
        helper.add_packet(au(&[&SPS, helper_pps, &IDR]), 1000, true, 1).unwrap();
        let b = helper.flush_segment().unwrap().unwrap();
        helper.finalize().unwrap();
        (dir, vec![a.path, b.path])
    }

    #[test]
    fn one_set_across_every_segment_settles_on_avc1() {
        let (dir, segments) = two_encoder_rendition(&PPS);
        let init = dir.path().join("init.mp4");
        assert_eq!(&entry_fourcc(&init), b"avc1", "the muxer writes avc1");
        let before = std::fs::read(&init).unwrap();
        assert_eq!(settle_video_sample_entry(&init, &segments).unwrap(), Some(*b"avc1"));
        assert_eq!(std::fs::read(&init).unwrap(), before, "nothing to rewrite");
    }

    #[test]
    fn a_segment_with_other_sets_settles_on_avc3() {
        let (dir, segments) = two_encoder_rendition(&PPS_CHANGED);
        let init = dir.path().join("init.mp4");
        let size = std::fs::metadata(&init).unwrap().len();
        assert_eq!(settle_video_sample_entry(&init, &segments).unwrap(), Some(*b"avc3"));
        assert_eq!(&entry_fourcc(&init), b"avc3");
        assert_eq!(std::fs::metadata(&init).unwrap().len(), size, "only the fourcc changes");
        // The first segment alone agrees with the init segment.
        assert_eq!(settle_video_sample_entry(&init, &segments[..1]).unwrap(), Some(*b"avc1"));
    }

    #[test]
    fn hev1_clears_array_completeness_and_hvc1_sets_it() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = CmafVideoMuxer::new_with_codec_options(
            dir.path(),
            640,
            360,
            30000,
            ColorMetadata::default(),
            VideoCodec::H265,
            CmafVideoMuxerOptions::default(),
        )
        .unwrap();
        let vps = [0x40u8, 0x01, 0x0c];
        let sps = [0x42u8, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03];
        let pps = [0x44u8, 0x01, 0xc1];
        let idr = [0x26u8, 0x01, 0xaf];
        m.add_packet(au(&[&vps, &sps, &pps, &idr]), 1000, true, 0).unwrap();
        let seg = m.flush_segment().unwrap().unwrap();
        m.finalize().unwrap();
        let init = dir.path().join("init.mp4");
        let completeness = |init: &Path| {
            let bytes = std::fs::read(init).unwrap();
            let e = video_sample_entry(&bytes).unwrap();
            let c = child_boxes(&bytes, e.start + 86, e.end)
                .find(|b| &bytes[b.start + 4..b.start + 8] == b"hvcC")
                .unwrap();
            let rec = parse_config(&bytes[c.start + 8..c.end], NalMuxCodec::H265).unwrap();
            assert_eq!(rec.sets.len(), 3, "VPS, SPS and PPS arrays");
            rec.array_headers.iter().map(|&at| bytes[c.start + 8 + at] >> 7).collect::<Vec<_>>()
        };
        assert_eq!(&entry_fourcc(&init), b"hvc1");
        assert_eq!(completeness(&init), vec![1, 1, 1]);

        // A segment carrying a VPS the init segment never saw.
        let mut other = std::fs::read(&seg.path).unwrap();
        let at = other.windows(3).position(|w| w == vps).unwrap();
        other[at + 2] = 0x0d;
        let other_path = dir.path().join("other.m4s");
        std::fs::write(&other_path, other).unwrap();
        let segs = vec![seg.path.clone(), other_path];
        assert_eq!(settle_video_sample_entry(&init, &segs).unwrap(), Some(*b"hev1"));
        assert_eq!(completeness(&init), vec![0, 0, 0]);
        // Settling on the agreeing segment alone restores hvc1.
        assert_eq!(settle_video_sample_entry(&init, &segs[..1]).unwrap(), Some(*b"hvc1"));
        assert_eq!(completeness(&init), vec![1, 1, 1]);
    }

    #[test]
    fn av1_has_nothing_to_settle() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = CmafVideoMuxer::new(dir.path(), 64, 64, 30000, ColorMetadata::default()).unwrap();
        m.add_packet(vec![(1 << 3) | (1 << 1), 0x01, 0xAA], 1000, true, 0).unwrap();
        let seg = m.flush_segment().unwrap().unwrap();
        m.finalize().unwrap();
        assert_eq!(settle_video_sample_entry(&dir.path().join("init.mp4"), &[seg.path]).unwrap(), None);
    }
}
