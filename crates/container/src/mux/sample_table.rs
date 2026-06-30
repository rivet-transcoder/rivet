use super::boxes::BoxBuilder;
use crate::AudioInfo;

/// Audio build plan shared between sizing passes and the final moov emit.
/// Holds the post-flush AAC metadata plus the derived chunking policy.
pub(super) struct AudioBuildPlan {
    pub(super) info: AudioInfo,
    pub(super) sample_sizes: Vec<u32>,
    pub(super) durations: Vec<u32>,
    pub(super) total_duration_in_own_ts: u64,
    pub(super) total_duration_in_movie_ts: u64,
    pub(super) samples_per_chunk: u32,
}

/// One contiguous copy from one source tempfile to the output. The finalize
/// loop walks a Vec<InterleaveStep> and copies `bytes` from the chosen
/// track's tempfile into the output stream, which keeps peak RAM bounded.
#[derive(Debug, Clone, Copy)]
pub(super) struct InterleaveStep {
    pub(super) track: InterleaveTrack,
    pub(super) bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InterleaveTrack {
    Video,
    Audio,
}

pub(super) fn chunk_count_of(sample_count: usize, spc: u32) -> usize {
    if sample_count == 0 {
        return 0;
    }
    let spc = spc.max(1) as usize;
    sample_count.div_ceil(spc)
}

/// Compute chunk byte size arrays — one entry per chunk, summing sample
/// sizes inside each chunk.
fn chunk_byte_sizes(sample_sizes: &[u32], spc: u32) -> Vec<u64> {
    let spc = spc.max(1) as usize;
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < sample_sizes.len() {
        let end = (i + spc).min(sample_sizes.len());
        let mut total: u64 = 0;
        for &s in &sample_sizes[i..end] {
            total += s as u64;
        }
        out.push(total);
        i = end;
    }
    out
}

/// Plan the interleaved mdat layout + assign per-track chunk offsets.
/// Chunk-alternation: emit one video chunk then one audio chunk, repeating
/// until both are drained; tail chunks for whichever track has more chunks.
/// This gives ~1 s interleave granularity on both sides which matches the
/// spc policy (video: frame_rate fps / 1 chunk; audio: ~46 chunks/s worth).
pub(super) fn plan_interleaved_layout(
    first_sample_file_offset: u64,
    video_sample_sizes: &[u32],
    video_spc: u32,
    audio_plan: Option<&AudioBuildPlan>,
) -> (Vec<u64>, Vec<u64>, Vec<InterleaveStep>) {
    let video_chunks = chunk_byte_sizes(video_sample_sizes, video_spc);
    let audio_chunks = match audio_plan {
        Some(p) => chunk_byte_sizes(&p.sample_sizes, p.samples_per_chunk),
        None => Vec::new(),
    };

    let mut video_offsets: Vec<u64> = Vec::with_capacity(video_chunks.len());
    let mut audio_offsets: Vec<u64> = Vec::with_capacity(audio_chunks.len());
    let mut plan: Vec<InterleaveStep> = Vec::with_capacity(video_chunks.len() + audio_chunks.len());

    let mut cursor = first_sample_file_offset;
    let mut vi = 0usize;
    let mut ai = 0usize;
    loop {
        if vi < video_chunks.len() {
            video_offsets.push(cursor);
            let size = video_chunks[vi];
            plan.push(InterleaveStep {
                track: InterleaveTrack::Video,
                bytes: size,
            });
            cursor = cursor.saturating_add(size);
            vi += 1;
        }
        if ai < audio_chunks.len() {
            audio_offsets.push(cursor);
            let size = audio_chunks[ai];
            plan.push(InterleaveStep {
                track: InterleaveTrack::Audio,
                bytes: size,
            });
            cursor = cursor.saturating_add(size);
            ai += 1;
        }
        if vi >= video_chunks.len() && ai >= audio_chunks.len() {
            break;
        }
    }

    (video_offsets, audio_offsets, plan)
}

/// Emit a `stsc` with run-length encoding. Full-size chunks of
/// `samples_per_chunk` are represented by one entry starting at chunk 1; if
/// the last chunk has a remainder (< samples_per_chunk), a second entry
/// records it. sample_description_index is always 1 because we emit a single
/// stsd entry (`av01`).
pub(super) fn build_stsc(sample_count: u32, samples_per_chunk: u32) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"stsc");
    b.u8(0);
    b.extend(&[0, 0, 0]);

    let spc = samples_per_chunk.max(1);
    // Guard against sample_count=0 — the muxer bails before calling this, but
    // keep the expression total: empty tables still need a valid entry_count.
    if sample_count == 0 {
        b.u32(0);
        return b.finish();
    }

    let full_chunks = sample_count / spc;
    let remainder = sample_count % spc;

    if remainder == 0 {
        // Every chunk has spc samples → one entry covers everything.
        b.u32(1);
        b.u32(1); // first_chunk (1-based)
        b.u32(spc); // samples_per_chunk
        b.u32(1); // sample_description_index
    } else if full_chunks == 0 {
        // All samples fit in the final partial chunk → one entry (1, rem, 1).
        b.u32(1);
        b.u32(1);
        b.u32(remainder);
        b.u32(1);
    } else {
        // Full-size run (1 .. full_chunks), then a tail entry for the
        // remainder chunk at index full_chunks+1 (1-based).
        b.u32(2);
        b.u32(1);
        b.u32(spc);
        b.u32(1);
        b.u32(full_chunks + 1); // first_chunk of the tail (1-based)
        b.u32(remainder);
        b.u32(1);
    }
    b.finish()
}

pub(super) fn build_stsz(sample_sizes: &[u32]) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"stsz");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(0); // sample_size (0 = varying)
    b.u32(sample_sizes.len() as u32); // sample_count
    for &s in sample_sizes {
        b.u32(s);
    }
    b.finish()
}

/// 32-bit chunk offset table. Caller must guarantee every offset fits in u32;
/// the muxer's co64-vs-stco decision does that upstream. Internal `as u32`
/// cast below is checked via `debug_assert` — `overflow-checks=false` in
/// release would otherwise silently wrap.
pub(super) fn build_stco(chunk_offsets: &[u64]) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"stco");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(chunk_offsets.len() as u32);
    for &off in chunk_offsets {
        debug_assert!(
            off <= u32::MAX as u64,
            "stco offset exceeds u32; should be co64"
        );
        b.u32(off as u32);
    }
    b.finish()
}

/// 64-bit chunk offset table. Layout per ISO/IEC 14496-12:
/// `size u32be | 'co64' | version u8=0 | flags u8[3]=0 | entry_count u32be
/// | entries: u64be chunk_offset[entry_count]`.
pub(super) fn build_co64(chunk_offsets: &[u64]) -> Vec<u8> {
    let mut b = BoxBuilder::new(b"co64");
    b.u8(0);
    b.extend(&[0, 0, 0]);
    b.u32(chunk_offsets.len() as u32);
    for &off in chunk_offsets {
        b.u64(off);
    }
    b.finish()
}

/// Partition samples into chunks of size `samples_per_chunk` (last chunk
/// may be smaller), then emit an absolute file offset for each chunk's
/// first sample by walking `sample_sizes` with a running cursor that starts
/// at `first_sample_file_offset`.
///
/// Superseded by `plan_interleaved_layout` on the hot path — kept here for
/// the existing single-track unit tests that exercise the chunking math.
#[cfg(test)]
pub(super) fn compute_chunk_offsets(
    first_sample_file_offset: u64,
    sample_sizes: &[u32],
    samples_per_chunk: u32,
) -> Vec<u64> {
    let spc = samples_per_chunk.max(1) as usize;
    let total = sample_sizes.len();
    if total == 0 {
        return Vec::new();
    }
    let chunk_count = (total + spc - 1) / spc;
    let mut offsets = Vec::with_capacity(chunk_count);
    let mut cursor = first_sample_file_offset;
    let mut sample_idx = 0usize;
    for _ in 0..chunk_count {
        offsets.push(cursor);
        let end = (sample_idx + spc).min(total);
        for &size in &sample_sizes[sample_idx..end] {
            cursor = cursor.saturating_add(size as u64);
        }
        sample_idx = end;
    }
    offsets
}
