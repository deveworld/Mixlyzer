//! Flattened segment rows for the library database.
//!
//! The database stores a lossy projection of the analysis segments: adjacent
//! rows carrying the same value are merged, and the bar-phase reference
//! (`inizio`) is dropped because nothing queries it. These are the rows written
//! to `track_bpm_segments` and `track_key_segments`.

use crate::key::Key;
use crate::segments::{KeySegment, TempoSegment};

/// A row of `track_bpm_segments`.
#[derive(Debug, Clone, PartialEq)]
pub struct BpmSegmentRow {
    pub seq_index: usize,
    pub start_sec: f64,
    pub end_sec: f64,
    pub duration_sec: f64,
    pub bpm: f64,
    pub bpm_rounded: i64,
    pub time_signature: u8,
}

/// A row of `track_key_segments`.
#[derive(Debug, Clone, PartialEq)]
pub struct KeySegmentRow {
    pub seq_index: usize,
    pub start_sec: f64,
    pub end_sec: f64,
    pub duration_sec: f64,
    pub key_value: u8,
    pub key_label: String,
}

/// Segments shorter than this are dropped as degenerate.
const MIN_SEGMENT_DURATION: f64 = 1e-6;

/// Two BPM rows merge when their tempo and meter agree.
const BPM_MERGE_TOLERANCE: f64 = 1e-6;

/// Build the database rows for a track's tempo segments.
///
/// Consecutive segments with the same BPM and time signature are merged into
/// one row, so `seq_index` here is not the index of the analysis segment.
pub fn build_bpm_segments(segments: &[TempoSegment]) -> Vec<BpmSegmentRow> {
    let mut rows: Vec<BpmSegmentRow> = Vec::new();
    for seg in segments {
        let duration = seg.end - seg.start;
        if !duration.is_finite() || duration <= MIN_SEGMENT_DURATION {
            continue;
        }
        if let Some(last) = rows.last_mut() {
            let same_tempo = (last.bpm - seg.bpm).abs() < BPM_MERGE_TOLERANCE;
            if same_tempo && last.time_signature == seg.time_signature {
                last.end_sec = seg.end;
                last.duration_sec = last.end_sec - last.start_sec;
                continue;
            }
        }
        rows.push(BpmSegmentRow {
            seq_index: rows.len(),
            start_sec: seg.start,
            end_sec: seg.end,
            duration_sec: duration,
            bpm: seg.bpm,
            bpm_rounded: seg.bpm.round() as i64,
            time_signature: seg.time_signature,
        });
    }
    for (idx, row) in rows.iter_mut().enumerate() {
        row.seq_index = idx;
    }
    rows
}

/// Build the database rows for a track's key segments.
///
/// Consecutive segments with the same key are merged.
pub fn build_key_segments(segments: &[KeySegment]) -> Vec<KeySegmentRow> {
    let mut rows: Vec<KeySegmentRow> = Vec::new();
    for seg in segments {
        let duration = seg.end - seg.start;
        if !duration.is_finite() || duration <= MIN_SEGMENT_DURATION {
            continue;
        }
        if let Some(last) = rows.last_mut() {
            if last.key_value == seg.key.index() {
                last.end_sec = seg.end;
                last.duration_sec = last.end_sec - last.start_sec;
                continue;
            }
        }
        rows.push(KeySegmentRow {
            seq_index: rows.len(),
            start_sec: seg.start,
            end_sec: seg.end,
            duration_sec: duration,
            key_value: seg.key.index(),
            key_label: seg.key.camelot().to_string(),
        });
    }
    for (idx, row) in rows.iter_mut().enumerate() {
        row.seq_index = idx;
    }
    rows
}

/// Label written to `track_key_segments.key_label`.
pub fn key_value_to_label(key_value: Option<i64>) -> String {
    match key_value {
        Some(v) if (0..24).contains(&v) => Key::from_index(v).camelot().to_string(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key::Mode;

    #[test]
    fn merges_adjacent_rows_with_the_same_tempo() {
        let segs = vec![
            TempoSegment::new(0.0, 10.0, 128.0, 0.0, 4),
            TempoSegment::new(10.0, 20.0, 128.0, 10.0, 4),
            TempoSegment::new(20.0, 30.0, 140.0, 20.0, 4),
        ];
        let rows = build_bpm_segments(&segs);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].start_sec, 0.0);
        assert_eq!(rows[0].end_sec, 20.0);
        assert_eq!(rows[0].duration_sec, 20.0);
        assert_eq!(rows[1].bpm, 140.0);
        assert_eq!(rows[0].seq_index, 0);
        assert_eq!(rows[1].seq_index, 1);
    }

    #[test]
    fn a_meter_change_prevents_merging() {
        let segs = vec![
            TempoSegment::new(0.0, 10.0, 120.0, 0.0, 4),
            TempoSegment::new(10.0, 20.0, 120.0, 10.0, 3),
        ];
        assert_eq!(build_bpm_segments(&segs).len(), 2);
    }

    #[test]
    fn merging_ignores_time_gaps_between_segments() {
        // Same tempo either side of a gap still merges, spanning the hole.
        let segs = vec![
            TempoSegment::new(0.0, 10.0, 128.0, 0.0, 4),
            TempoSegment::new(50.0, 60.0, 128.0, 50.0, 4),
        ];
        let rows = build_bpm_segments(&segs);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].end_sec, 60.0);
        assert_eq!(rows[0].duration_sec, 60.0);
    }

    #[test]
    fn zero_length_segments_are_dropped() {
        let segs = vec![
            TempoSegment::new(0.0, 0.0, 128.0, 0.0, 4),
            TempoSegment::new(0.0, 10.0, 128.0, 0.0, 4),
        ];
        let rows = build_bpm_segments(&segs);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].end_sec, 10.0);
    }

    #[test]
    fn bpm_is_rounded_for_the_search_index() {
        let segs = vec![TempoSegment::new(0.0, 10.0, 127.6, 0.0, 4)];
        assert_eq!(build_bpm_segments(&segs)[0].bpm_rounded, 128);
    }

    #[test]
    fn key_rows_merge_and_carry_camelot_labels() {
        let segs = vec![
            KeySegment::new(0.0, 10.0, Key::new(9, Mode::Minor)),
            KeySegment::new(10.0, 20.0, Key::new(9, Mode::Minor)),
            KeySegment::new(20.0, 30.0, Key::new(0, Mode::Major)),
        ];
        let rows = build_key_segments(&segs);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].end_sec, 20.0);
        assert_eq!(rows[0].key_value, 21);
        assert_eq!(rows[0].key_label, "8A");
        assert_eq!(rows[1].key_label, "8B");
    }

    #[test]
    fn key_label_lookup_rejects_out_of_range_values() {
        assert_eq!(key_value_to_label(Some(0)), "8B");
        assert_eq!(key_value_to_label(Some(23)), "10A");
        assert_eq!(key_value_to_label(Some(24)), "");
        assert_eq!(key_value_to_label(Some(-1)), "");
        assert_eq!(key_value_to_label(None), "");
    }

    #[test]
    fn empty_input_produces_no_rows() {
        assert!(build_bpm_segments(&[]).is_empty());
        assert!(build_key_segments(&[]).is_empty());
    }
}
