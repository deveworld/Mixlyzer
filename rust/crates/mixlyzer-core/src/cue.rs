//! Cue points derived from the detected song structure.
//!
//! These are single points in time, distinct from JumpCUEs (which are pairs of
//! regions). They mark the places a DJ typically wants to drop into or out of:
//! the start of an interlude or outro, and the boundaries of a chorus run.
//!
//! Times are `f64` throughout. Python stores them as float32, whose resolution
//! at the two-hour mark is 0.49 ms — close enough to the 1 ms dedupe window
//! that distinct cues in a long mix silently merge into one.

use crate::phrase::{normalize, Phrase};

/// Two cues closer than this are treated as the same point.
const DEDUPE_EPSILON: f64 = 1e-3;

/// A single marked position in the track.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CuePoint {
    /// Position in the list, assigned after sorting and deduplication.
    pub id: usize,
    pub time_sec: f64,
    /// What the cue marks, e.g. `CHORUS_IN`. Merged cues join labels with `/`.
    pub label: String,
    /// Human-readable note. Merged cues join comments with ` / `.
    pub comment: String,
}

impl CuePoint {
    pub fn new(id: usize, time_sec: f64, label: impl Into<String>, comment: impl Into<String>) -> Self {
        Self {
            id,
            time_sec,
            label: label.into(),
            comment: comment.into(),
        }
    }
}

/// Sort, drop invalid entries, and merge cues that land on the same instant.
///
/// Labels and comments of merged cues are combined rather than discarded, so
/// a point that is both a chorus exit and an outro start says so.
pub fn dedupe(points: &[CuePoint]) -> Vec<CuePoint> {
    let mut rows: Vec<&CuePoint> = points
        .iter()
        .filter(|p| p.time_sec.is_finite() && p.time_sec >= 0.0)
        .collect();
    rows.sort_by(|a, b| {
        a.time_sec
            .partial_cmp(&b.time_sec)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.id.cmp(&b.id))
    });

    let mut merged: Vec<(f64, Vec<String>, Vec<String>)> = Vec::new();
    for point in rows {
        let label = point.label.trim();
        let comment = point.comment.trim();
        match merged.last_mut() {
            Some((time, labels, comments)) if (*time - point.time_sec).abs() <= DEDUPE_EPSILON => {
                push_unique(labels, label);
                push_unique(comments, comment);
            }
            _ => {
                let mut labels = Vec::new();
                let mut comments = Vec::new();
                push_unique(&mut labels, label);
                push_unique(&mut comments, comment);
                merged.push((point.time_sec, labels, comments));
            }
        }
    }

    merged
        .into_iter()
        .enumerate()
        .map(|(id, (time, labels, comments))| CuePoint {
            id,
            time_sec: time,
            label: labels.join("/"),
            comment: comments.join(" / "),
        })
        .collect()
}

fn push_unique(parts: &mut Vec<String>, value: &str) {
    let text = value.trim();
    if !text.is_empty() && !parts.iter().any(|p| p == text) {
        parts.push(text.to_string());
    }
}

/// Derive cue points from detected phrases.
///
/// Interludes and outros get a cue at their start. A run of consecutive
/// choruses gets an entry cue, an internal boundary and a pre-exit cue when it
/// is more than one phrase long, and an exit cue when something follows it.
pub fn from_phrases(phrases: &[Phrase]) -> Vec<CuePoint> {
    let rows = normalize(phrases);
    let upper: Vec<String> = rows
        .iter()
        .map(|p| p.label.trim().to_ascii_uppercase())
        .collect();

    let mut points: Vec<CuePoint> = Vec::new();
    fn add(points: &mut Vec<CuePoint>, time: f64, label: &str, comment: &str) {
        points.push(CuePoint::new(points.len(), time, label, comment));
    }

    let mut index = 0usize;
    while index < rows.len() {
        match upper[index].as_str() {
            "INTERLUDE" => add(&mut points, rows[index].start, "INTERLUDE", "Interlude start"),
            "OUTRO" => add(&mut points, rows[index].start, "OUTRO", "Outro start"),
            _ => {}
        }

        if upper[index] != "CHORUS" {
            index += 1;
            continue;
        }

        // Collect the whole run of consecutive choruses.
        let group_start = index;
        let mut group_end = index + 1;
        while group_end < rows.len() && upper[group_end] == "CHORUS" {
            group_end += 1;
        }
        let group = &rows[group_start..group_end];

        add(&mut points, group[0].start, "CHORUS_IN", "Chorus start");
        if group.len() >= 2 {
            add(&mut points, group[1].start, "CHORUS_NEXT", "Chorus internal boundary");
            add(
                &mut points,
                group[group.len() - 1].start,
                "CHORUS_PRE_OUT",
                "Chorus pre-exit boundary",
            );
        }
        if group_end < rows.len() {
            add(
                &mut points,
                group[group.len() - 1].end,
                "CHORUS_OUT",
                "Chorus exit",
            );
        }
        index = group_end;
    }

    dedupe(&points)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(start: f64, end: f64, label: &str) -> Phrase {
        Phrase::new(start, end, label)
    }

    fn labels(points: &[CuePoint]) -> Vec<&str> {
        points.iter().map(|c| c.label.as_str()).collect()
    }

    #[test]
    fn interlude_and_outro_get_a_start_cue() {
        let phrases = vec![
            p(0.0, 10.0, "INTRO"),
            p(10.0, 20.0, "INTERLUDE"),
            p(20.0, 30.0, "OUTRO"),
        ];
        let cues = from_phrases(&phrases);
        assert_eq!(labels(&cues), vec!["INTERLUDE", "OUTRO"]);
        assert_eq!(cues[0].time_sec, 10.0);
        assert_eq!(cues[1].time_sec, 20.0);
    }

    #[test]
    fn a_lone_chorus_gets_entry_and_exit_only() {
        let phrases = vec![
            p(0.0, 10.0, "VERSE"),
            p(10.0, 20.0, "CHORUS"),
            p(20.0, 30.0, "VERSE"),
        ];
        let cues = from_phrases(&phrases);
        assert_eq!(labels(&cues), vec!["CHORUS_IN", "CHORUS_OUT"]);
        assert_eq!(cues[0].time_sec, 10.0);
        assert_eq!(cues[1].time_sec, 20.0);
    }

    #[test]
    fn a_chorus_run_gets_internal_and_pre_exit_cues() {
        let phrases = vec![
            p(0.0, 10.0, "VERSE"),
            p(10.0, 20.0, "CHORUS"),
            p(20.0, 30.0, "CHORUS"),
            p(30.0, 40.0, "CHORUS"),
            p(40.0, 50.0, "OUTRO"),
        ];
        let cues = from_phrases(&phrases);
        // The chorus exit and the outro start are the same instant, so they
        // arrive as one cue carrying both labels.
        assert_eq!(
            labels(&cues),
            vec!["CHORUS_IN", "CHORUS_NEXT", "CHORUS_PRE_OUT", "CHORUS_OUT/OUTRO"]
        );
        assert_eq!(cues[0].time_sec, 10.0);
        assert_eq!(cues[1].time_sec, 20.0);
        assert_eq!(cues[2].time_sec, 30.0);
        assert_eq!(cues[3].time_sec, 40.0);
    }

    #[test]
    fn a_chorus_that_ends_the_track_has_no_exit_cue() {
        let phrases = vec![p(0.0, 10.0, "VERSE"), p(10.0, 20.0, "CHORUS")];
        assert_eq!(labels(&from_phrases(&phrases)), vec!["CHORUS_IN"]);
    }

    #[test]
    fn coincident_cues_merge_their_labels() {
        // A chorus exit that lands exactly where the outro begins.
        let phrases = vec![
            p(0.0, 10.0, "VERSE"),
            p(10.0, 20.0, "CHORUS"),
            p(20.0, 30.0, "OUTRO"),
        ];
        let cues = from_phrases(&phrases);
        let merged = cues.iter().find(|c| c.time_sec == 20.0).unwrap();
        assert_eq!(merged.label, "CHORUS_OUT/OUTRO");
        assert!(merged.comment.contains("Outro start"));
        assert!(merged.comment.contains("Chorus exit"));
    }

    #[test]
    fn ids_are_renumbered_contiguously_after_merging() {
        let phrases = vec![
            p(0.0, 10.0, "VERSE"),
            p(10.0, 20.0, "CHORUS"),
            p(20.0, 30.0, "OUTRO"),
        ];
        let cues = from_phrases(&phrases);
        for (index, cue) in cues.iter().enumerate() {
            assert_eq!(cue.id, index);
        }
    }

    /// float32 resolution at two hours is 0.49 ms, inside the 1 ms dedupe
    /// window, so Python drops the second of these two cues entirely.
    #[test]
    fn cues_two_hours_in_stay_distinct() {
        let points = vec![
            CuePoint::new(0, 7200.0, "A", ""),
            CuePoint::new(1, 7200.002, "B", ""),
        ];
        let out = dedupe(&points);
        assert_eq!(out.len(), 2, "2 ms apart should not merge");
        assert_ne!(out[0].time_sec, out[1].time_sec);
    }

    #[test]
    fn cues_within_the_epsilon_do_merge() {
        let points = vec![
            CuePoint::new(0, 10.0, "A", "first"),
            CuePoint::new(1, 10.0005, "B", "second"),
        ];
        let out = dedupe(&points);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].label, "A/B");
        assert_eq!(out[0].comment, "first / second");
    }

    #[test]
    fn duplicate_labels_are_not_repeated_when_merging() {
        let points = vec![
            CuePoint::new(0, 5.0, "CHORUS_IN", "Chorus start"),
            CuePoint::new(1, 5.0, "CHORUS_IN", "Chorus start"),
        ];
        let out = dedupe(&points);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].label, "CHORUS_IN");
        assert_eq!(out[0].comment, "Chorus start");
    }

    #[test]
    fn negative_and_non_finite_times_are_dropped() {
        let points = vec![
            CuePoint::new(0, -1.0, "A", ""),
            CuePoint::new(1, f64::NAN, "B", ""),
            CuePoint::new(2, 5.0, "C", ""),
        ];
        let out = dedupe(&points);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].label, "C");
    }

    #[test]
    fn no_phrases_means_no_cues() {
        assert!(from_phrases(&[]).is_empty());
    }

    #[test]
    fn cues_come_out_in_time_order() {
        let phrases = vec![
            p(0.0, 10.0, "INTRO"),
            p(10.0, 20.0, "CHORUS"),
            p(20.0, 30.0, "CHORUS"),
            p(30.0, 40.0, "INTERLUDE"),
            p(40.0, 50.0, "CHORUS"),
            p(50.0, 60.0, "OUTRO"),
        ];
        let cues = from_phrases(&phrases);
        assert!(
            cues.windows(2).all(|w| w[0].time_sec <= w[1].time_sec),
            "cues must be sorted: {:?}",
            cues.iter().map(|c| c.time_sec).collect::<Vec<_>>()
        );
    }
}
