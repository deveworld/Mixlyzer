//! Reconstructing the beat and bar grid the phrase features are measured on.
//!
//! The detector is only allowed to see what a DJ library would already have:
//! beat times and tempo segments. Everything bar-shaped is rebuilt from those,
//! by anchoring each segment's stated downbeat to the nearest actual beat and
//! counting the meter out from there. Nothing about the song's structure is
//! read from the analysis file.

use mixlyzer_core::TempoSegment;

use crate::error::PhraseError;
use crate::features::matrix::median;

/// A tempo segment reduced to what the grid needs: when it runs, where its
/// downbeat is, and how many beats there are to a bar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeterSegment {
    pub start_sec: f64,
    pub end_sec: f64,
    pub inizio_sec: f64,
    pub meter: usize,
}

/// The beat and bar grid, in the shape the feature extractor consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictorGrid {
    pub beat_times_sec: Vec<f64>,
    /// One more entry than there are beats: the trailing edge of the last beat.
    pub beat_edges_sec: Vec<f64>,
    pub meter_segments: Vec<MeterSegment>,
    pub downbeat_mask: Vec<bool>,
    /// Position of each beat inside its bar, counting from 0.
    pub beat_in_bar: Vec<usize>,
    /// Which bar each beat belongs to.
    pub bar_index_of_beat: Vec<usize>,
    pub bar_starts_beat: Vec<usize>,
    pub bar_ends_beat: Vec<usize>,
    pub bar_starts_sec: Vec<f64>,
    pub bar_ends_sec: Vec<f64>,
    pub bar_meters: Vec<usize>,
    /// Bars that do not start on a downbeat or do not hold a full meter.
    pub bar_is_partial: Vec<bool>,
}

impl PredictorGrid {
    pub fn n_beats(&self) -> usize {
        self.beat_times_sec.len()
    }

    pub fn n_bars(&self) -> usize {
        self.bar_starts_beat.len()
    }
}

/// Reject a beat array that cannot describe a playable grid.
fn validate_beats(beats: &[f64]) -> Result<(), PhraseError> {
    if beats.len() < 8 {
        return Err(PhraseError::BeatGridTooShort);
    }
    if !beats.iter().all(|b| b.is_finite()) || beats[0] < 0.0 {
        return Err(PhraseError::BeatTimesNotFinite);
    }
    for index in 1..beats.len() {
        if beats[index] <= beats[index - 1] {
            return Err(PhraseError::BeatsNotIncreasing { index });
        }
    }
    Ok(())
}

/// Keep the segments that describe a real span with a real meter, in time order.
///
/// Malformed rows are dropped rather than rejected, mirroring the Python:
/// exported tempo arrays routinely contain zero-length rows at tempo changes.
fn parse_meter_segments(segments: &[TempoSegment]) -> Result<Vec<MeterSegment>, PhraseError> {
    let mut parsed: Vec<MeterSegment> = segments
        .iter()
        .filter(|s| s.start.is_finite() && s.end.is_finite() && s.inizio.is_finite())
        .filter(|s| s.end > s.start && s.time_signature >= 1)
        .map(|s| MeterSegment {
            start_sec: s.start,
            end_sec: s.end,
            inizio_sec: s.inizio,
            meter: usize::from(s.time_signature),
        })
        .collect();
    if parsed.is_empty() {
        return Err(PhraseError::NoTempoSegments);
    }
    parsed.sort_by(|a, b| {
        a.start_sec
            .total_cmp(&b.start_sec)
            .then(a.end_sec.total_cmp(&b.end_sec))
    });
    Ok(parsed)
}

/// Median inter-beat interval, the natural unit for grid tolerances.
fn median_ibi(beats: &[f64]) -> f64 {
    let gaps: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
    median(&gaps)
}

/// Where the last beat ends, since no later beat marks it.
fn estimate_final_beat_edge(beats: &[f64], segment_end: f64) -> f64 {
    let last = beats[beats.len() - 1];
    let ibi = median_ibi(beats);
    let mut estimated = last + ibi;
    if segment_end.is_finite() && segment_end > last {
        estimated = estimated.min(segment_end);
    }
    if estimated <= last + 1e-5 {
        estimated = last + ibi;
    }
    estimated
}

/// Build the grid from beat times and tempo segments.
pub fn build_predictor_grid(
    beats: &[f64],
    segments: &[TempoSegment],
) -> Result<PredictorGrid, PhraseError> {
    validate_beats(beats)?;
    let segments = parse_meter_segments(segments)?;
    let n_beats = beats.len();

    // 0.48 of a beat is just under half, so a downbeat can never be ambiguous
    // between two neighbouring beats; the 80 ms floor covers very fast tempos.
    let tolerance = (0.48 * median_ibi(beats)).max(0.08);

    let mut downbeat_mask = vec![false; n_beats];
    let mut beat_meter = vec![0usize; n_beats];
    let mut beat_anchor = vec![usize::MAX; n_beats];

    for segment in &segments {
        let anchor = (0..n_beats)
            .min_by(|a, b| {
                (beats[*a] - segment.inizio_sec)
                    .abs()
                    .total_cmp(&(beats[*b] - segment.inizio_sec).abs())
            })
            .unwrap_or(0);
        let error = (beats[anchor] - segment.inizio_sec).abs();
        if error > tolerance {
            return Err(PhraseError::DownbeatNotOnBeat {
                inizio: segment.inizio_sec,
                error,
                tolerance,
            });
        }

        let covered: Vec<usize> = (0..n_beats)
            .filter(|i| {
                beats[*i] >= segment.start_sec - tolerance
                    && beats[*i] < segment.end_sec + tolerance
            })
            .collect();
        for index in covered {
            beat_meter[index] = segment.meter;
            beat_anchor[index] = anchor;
            let phase = (index as isize - anchor as isize).rem_euclid(segment.meter as isize);
            if phase == 0 {
                downbeat_mask[index] = true;
            }
        }
    }

    // Beats outside every segment inherit the meter and phase of the nearest
    // covered beat: the tempo map is allowed to have gaps, the grid is not.
    let covered_indices: Vec<usize> = (0..n_beats).filter(|i| beat_meter[*i] > 0).collect();
    if covered_indices.is_empty() {
        return Err(PhraseError::SegmentsMissTheGrid);
    }
    for index in 0..n_beats {
        if beat_meter[index] > 0 {
            continue;
        }
        let nearest = *covered_indices
            .iter()
            .min_by_key(|c| c.abs_diff(index))
            .expect("covered_indices is non-empty");
        beat_meter[index] = beat_meter[nearest];
        beat_anchor[index] = beat_anchor[nearest];
        let phase =
            (index as isize - beat_anchor[index] as isize).rem_euclid(beat_meter[index] as isize);
        if phase == 0 {
            downbeat_mask[index] = true;
        }
    }

    // Bars run from one downbeat to the next; beat 0 always opens a bar, even
    // when it is mid-bar, so no beat is left outside the grid.
    let mut bar_starts: Vec<usize> = std::iter::once(0)
        .chain((0..n_beats).filter(|i| downbeat_mask[*i]))
        .collect();
    bar_starts.sort_unstable();
    bar_starts.dedup();
    bar_starts.retain(|s| *s < n_beats);
    let bar_ends: Vec<usize> = bar_starts
        .iter()
        .skip(1)
        .copied()
        .chain(std::iter::once(n_beats))
        .collect();

    let mut beat_in_bar = vec![0usize; n_beats];
    let mut bar_index_of_beat = vec![0usize; n_beats];
    let mut bar_meters = Vec::with_capacity(bar_starts.len());
    let mut bar_is_partial = Vec::with_capacity(bar_starts.len());
    for (bar, (start, end)) in bar_starts.iter().zip(&bar_ends).enumerate() {
        for beat in *start..*end {
            bar_index_of_beat[beat] = bar;
            beat_in_bar[beat] = beat - start;
        }
        let meters: Vec<f64> = (*start..*end).map(|b| beat_meter[b] as f64).collect();
        let meter = median(&meters) as usize;
        bar_meters.push(meter);
        bar_is_partial.push(!downbeat_mask[*start] || (end - start) != meter);
    }

    let final_edge = estimate_final_beat_edge(
        beats,
        segments
            .iter()
            .map(|s| s.end_sec)
            .fold(f64::NEG_INFINITY, f64::max),
    );
    let mut beat_edges = beats.to_vec();
    beat_edges.push(final_edge);

    let bar_starts_sec = bar_starts.iter().map(|b| beats[*b]).collect();
    let bar_ends_sec = bar_ends.iter().map(|b| beat_edges[*b]).collect();

    Ok(PredictorGrid {
        beat_times_sec: beats.to_vec(),
        beat_edges_sec: beat_edges,
        meter_segments: segments,
        downbeat_mask,
        beat_in_bar,
        bar_index_of_beat,
        bar_starts_beat: bar_starts,
        bar_ends_beat: bar_ends,
        bar_starts_sec,
        bar_ends_sec,
        bar_meters,
        bar_is_partial,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steady_beats(count: usize, period: f64) -> Vec<f64> {
        (0..count).map(|i| i as f64 * period).collect()
    }

    fn four_four(end: f64, inizio: f64) -> Vec<TempoSegment> {
        vec![TempoSegment::new(0.0, end, 120.0, inizio, 4)]
    }

    #[test]
    fn every_fourth_beat_is_a_downbeat_in_four_four() {
        let beats = steady_beats(32, 0.5);
        let grid = build_predictor_grid(&beats, &four_four(16.0, 0.0)).unwrap();
        assert_eq!(grid.n_beats(), 32);
        assert_eq!(grid.n_bars(), 8);
        for (index, is_downbeat) in grid.downbeat_mask.iter().enumerate() {
            assert_eq!(*is_downbeat, index % 4 == 0, "beat {index}");
        }
        assert!(grid.bar_meters.iter().all(|m| *m == 4));
        assert!(!grid.bar_is_partial.iter().any(|p| *p));
    }

    #[test]
    fn the_downbeat_anchor_shifts_the_whole_phase() {
        let beats = steady_beats(32, 0.5);
        let grid = build_predictor_grid(&beats, &four_four(16.0, 1.0)).unwrap();
        for (index, is_downbeat) in grid.downbeat_mask.iter().enumerate() {
            assert_eq!(*is_downbeat, index % 4 == 2, "beat {index}");
        }
        assert!(grid.bar_is_partial[0], "the opening two beats are a part bar");
    }

    #[test]
    fn a_downbeat_far_from_any_beat_is_an_error_not_a_silent_snap() {
        let beats = steady_beats(32, 0.5);
        let err = build_predictor_grid(&beats, &four_four(16.0, 0.25)).unwrap_err();
        match err {
            PhraseError::DownbeatNotOnBeat { error, tolerance, .. } => {
                assert!(error > tolerance);
            }
            other => panic!("expected DownbeatNotOnBeat, got {other:?}"),
        }
    }

    #[test]
    fn a_downbeat_inside_the_tolerance_snaps_to_the_nearest_beat() {
        let beats = steady_beats(32, 0.5);
        // 0.48 * 0.5 = 0.24s of slack; 0.2 is inside it.
        let grid = build_predictor_grid(&beats, &four_four(16.0, 4.2)).unwrap();
        assert!(grid.downbeat_mask[8], "4.0s is beat 8");
    }

    #[test]
    fn fewer_than_eight_beats_is_rejected() {
        let beats = steady_beats(7, 0.5);
        assert!(matches!(
            build_predictor_grid(&beats, &four_four(4.0, 0.0)),
            Err(PhraseError::BeatGridTooShort)
        ));
    }

    #[test]
    fn beats_that_do_not_advance_are_rejected_with_the_offending_index() {
        let mut beats = steady_beats(16, 0.5);
        beats[9] = beats[8];
        match build_predictor_grid(&beats, &four_four(8.0, 0.0)) {
            Err(PhraseError::BeatsNotIncreasing { index }) => assert_eq!(index, 9),
            other => panic!("expected BeatsNotIncreasing, got {other:?}"),
        }
    }

    #[test]
    fn no_usable_tempo_segment_is_reported() {
        let beats = steady_beats(16, 0.5);
        assert!(matches!(
            build_predictor_grid(&beats, &[]),
            Err(PhraseError::NoTempoSegments)
        ));
        // A zero-length row carries no meter information and is dropped.
        let empty = vec![TempoSegment::new(4.0, 4.0, 120.0, 4.0, 4)];
        assert!(matches!(
            build_predictor_grid(&beats, &empty),
            Err(PhraseError::NoTempoSegments)
        ));
    }

    #[test]
    fn a_meter_change_mid_track_is_honoured_by_each_segment() {
        let beats = steady_beats(24, 0.5);
        let segments = vec![
            TempoSegment::new(0.0, 6.0, 120.0, 0.0, 4),
            TempoSegment::new(6.0, 12.0, 120.0, 6.0, 3),
        ];
        let grid = build_predictor_grid(&beats, &segments).unwrap();
        assert!(grid.downbeat_mask[0] && grid.downbeat_mask[4] && grid.downbeat_mask[8]);
        // From beat 12 (6.0s) onwards the bar is three beats long.
        assert!(grid.downbeat_mask[12] && grid.downbeat_mask[15] && grid.downbeat_mask[18]);
        assert!(!grid.downbeat_mask[16]);
    }

    #[test]
    fn beats_beyond_the_last_segment_inherit_the_nearest_meter() {
        let beats = steady_beats(32, 0.5);
        // The segments stop at 4s but the beats run to 15.5s.
        let grid = build_predictor_grid(&beats, &four_four(4.0, 0.0)).unwrap();
        assert!(grid.bar_meters.iter().all(|m| *m == 4));
        assert!(grid.downbeat_mask[28], "the phase carries on past the segment");
    }

    #[test]
    fn the_last_beat_gets_an_edge_a_beat_wide() {
        let beats = steady_beats(16, 0.5);
        let grid = build_predictor_grid(&beats, &four_four(20.0, 0.0)).unwrap();
        assert_eq!(grid.beat_edges_sec.len(), 17);
        assert!((grid.beat_edges_sec[16] - 8.0).abs() < 1e-9);
    }

    #[test]
    fn bars_tile_the_beats_without_gaps_or_overlap() {
        let beats = steady_beats(30, 0.5);
        let grid = build_predictor_grid(&beats, &four_four(15.0, 1.0)).unwrap();
        assert_eq!(grid.bar_starts_beat[0], 0);
        assert_eq!(*grid.bar_ends_beat.last().unwrap(), 30);
        for pair in 0..grid.n_bars() - 1 {
            assert_eq!(grid.bar_ends_beat[pair], grid.bar_starts_beat[pair + 1]);
        }
    }
}
