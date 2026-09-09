//! The beatgrid: beat times plus the tempo segments that give them bar structure.
//!
//! # One definition of a downbeat
//!
//! The Python implementation grew two different ways to decide where bars
//! start. `core.beat_geometry.build_downbeats_from_segments` walks forward in
//! *seconds*, adding `ts * 60 / bpm` per bar, and feeds the beatgrid view;
//! `core.beat_geometry.downbeat_beat_indices` counts in *beat indices* and
//! feeds the playhead label and the metronome. On a perfectly uniform grid the
//! two agree, so the divergence is invisible in a unit test built from
//! `arange`. On a detected grid with ordinary beat jitter they do not: over an
//! hour at 128 BPM with 4 ms of jitter the two disagree by 145 ms on average
//! and 360 ms at worst, so the red bar lines and the "bar.beat" readout drift
//! visibly apart.
//!
//! Here the beat-index computation is the only definition. [`Beatgrid::downbeat_times`]
//! is derived from [`Beatgrid::downbeat_indices`] by looking the beats up, so
//! the two can never disagree. [`synthesize_downbeat_times`] exists only for
//! the case where there are no detected beats at all.

use crate::segments::{TempoSegment, DEFAULT_TIME_SIGNATURE};

/// A musical position: which bar, and which beat inside that bar.
///
/// Both are 1-based, matching how DJ software displays them. Before the first
/// downbeat the bar is 0 and `beat` counts the beats remaining until the first
/// bar starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BarBeat {
    pub bar: usize,
    pub beat: usize,
}

impl std::fmt::Display for BarBeat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.bar, self.beat)
    }
}

/// Beat times together with the tempo segments that organise them into bars.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Beatgrid {
    beats: Vec<f64>,
    segments: Vec<TempoSegment>,
}

impl Beatgrid {
    /// Build a grid, dropping non-finite beats and sorting what remains.
    pub fn new(beats: impl IntoIterator<Item = f64>, segments: Vec<TempoSegment>) -> Self {
        let mut beats: Vec<f64> = beats.into_iter().filter(|b| b.is_finite()).collect();
        beats.sort_by(|a, b| a.partial_cmp(b).expect("non-finite beats filtered out"));
        Self { beats, segments }
    }

    pub fn beats(&self) -> &[f64] {
        &self.beats
    }

    pub fn segments(&self) -> &[TempoSegment] {
        &self.segments
    }

    pub fn is_empty(&self) -> bool {
        self.beats.is_empty()
    }

    pub fn len(&self) -> usize {
        self.beats.len()
    }

    /// Index of the last beat at or before `time`, or `None` before the first beat.
    pub fn beat_index_at(&self, time: f64) -> Option<usize> {
        let pos = self.partition_point_right(time);
        if pos == 0 {
            None
        } else {
            Some(pos - 1)
        }
    }

    /// Index of the beat closest to `time`, or `None` when there are no beats.
    pub fn nearest_beat_index(&self, time: f64) -> Option<usize> {
        if self.beats.is_empty() {
            return None;
        }
        let pos = self.partition_point_left(time);
        let candidates = [pos.saturating_sub(1), pos.min(self.beats.len() - 1)];
        candidates
            .into_iter()
            .min_by(|&a, &b| {
                let da = (self.beats[a] - time).abs();
                let db = (self.beats[b] - time).abs();
                da.partial_cmp(&db)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            })
    }

    /// Indices of the beats that start a bar, sorted and deduplicated.
    ///
    /// Within each tempo segment a downbeat falls every `time_signature` beats,
    /// counted from the beat nearest that segment's `inizio`. Because the walk
    /// is over integer indices, the result cannot drift away from the beats
    /// however long the track is.
    ///
    /// With no usable tempo segments this assumes a 4/4 grid anchored on the
    /// first beat, as the Python code does.
    pub fn downbeat_indices(&self) -> Vec<usize> {
        if self.beats.is_empty() {
            return Vec::new();
        }
        if self.segments.is_empty() {
            return (0..self.beats.len())
                .step_by(usize::from(DEFAULT_TIME_SIGNATURE))
                .collect();
        }

        let mut out: Vec<usize> = Vec::new();
        for seg in &self.segments {
            let lo = self.partition_point_left(seg.start);
            let hi = self.partition_point_left(seg.end);
            if hi <= lo {
                continue;
            }
            let Some(reference) = self.nearest_beat_index(seg.inizio) else {
                continue;
            };
            let ts = usize::from(seg.time_signature.max(1));
            // First index at or after `lo` that is congruent to `reference` mod ts.
            let offset = (reference as i64 - lo as i64).rem_euclid(ts as i64) as usize;
            let mut j = lo + offset;
            while j < hi {
                out.push(j);
                j += ts;
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Times of the bar starts, derived from [`Self::downbeat_indices`].
    ///
    /// Always a subset of the beat times, so a bar line is always drawn on a
    /// real beat.
    pub fn downbeat_times(&self) -> Vec<f64> {
        self.downbeat_indices()
            .into_iter()
            .map(|i| self.beats[i])
            .collect()
    }

    /// Musical position at `time`, or `None` without beat or downbeat data.
    pub fn bar_beat_at(&self, time: f64) -> Option<BarBeat> {
        self.bar_beat_with_downbeats(time, &self.downbeat_indices())
    }

    /// Same as [`Self::bar_beat_at`] but reusing precomputed downbeat indices.
    ///
    /// The playhead calls this once per frame, so it should not recompute the
    /// downbeats every time.
    pub fn bar_beat_with_downbeats(&self, time: f64, downbeats: &[usize]) -> Option<BarBeat> {
        if self.beats.is_empty() || downbeats.is_empty() {
            return None;
        }
        let current = match self.beat_index_at(time) {
            Some(idx) => idx as i64,
            // Before the first beat: report how many beats until bar 1.
            None => {
                return Some(BarBeat {
                    bar: 0,
                    beat: downbeats[0] + 1,
                })
            }
        };
        let bar = downbeats.partition_point(|&d| (d as i64) <= current);
        if bar == 0 {
            let remaining = (downbeats[0] as i64 - current).max(0) as usize;
            return Some(BarBeat { bar: 0, beat: remaining });
        }
        let beat_in_bar = (current - downbeats[bar - 1] as i64 + 1).max(1) as usize;
        Some(BarBeat { bar, beat: beat_in_bar })
    }

    /// `"bar.beat"`, or an empty string when the position is unknown.
    pub fn bar_beat_label(&self, time: f64) -> String {
        self.bar_beat_at(time)
            .map(|p| p.to_string())
            .unwrap_or_default()
    }

    /// Index of the tempo segment covering `time`.
    pub fn segment_at(&self, time: f64) -> Option<usize> {
        self.segments.iter().position(|s| s.contains(time))
    }

    /// Number of beats in `[from, to)`.
    pub fn beats_between(&self, from: f64, to: f64) -> usize {
        if to <= from {
            return 0;
        }
        self.partition_point_left(to) - self.partition_point_left(from)
    }

    /// Count of beats strictly before `time` (`searchsorted(..., side="left")`).
    fn partition_point_left(&self, time: f64) -> usize {
        self.beats.partition_point(|&b| b < time)
    }

    /// Count of beats at or before `time` (`searchsorted(..., side="right")`).
    fn partition_point_right(&self, time: f64) -> usize {
        self.beats.partition_point(|&b| b <= time)
    }
}

/// Downbeat times synthesised from tempo alone, for when no beats were detected.
///
/// This walks forward in seconds and therefore accumulates error, which is
/// exactly why [`Beatgrid::downbeat_times`] does not use it. Reach for it only
/// as a fallback when there is no beat array to index into.
pub fn synthesize_downbeat_times(segments: &[TempoSegment]) -> Vec<f64> {
    let mut out: Vec<f64> = Vec::new();
    for seg in segments {
        let bar = seg.bar_period();
        if !bar.is_finite() || bar <= 0.0 {
            continue;
        }
        let mut t = seg.inizio.max(0.0);
        let stop = seg.end.max(t);
        while t <= stop + 1e-6 {
            out.push(t);
            t += bar;
        }
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_grid(bpm: f64, beats: usize, ts: u8) -> Beatgrid {
        let period = 60.0 / bpm;
        let times: Vec<f64> = (0..beats).map(|i| i as f64 * period).collect();
        let seg = TempoSegment::new(0.0, times[beats - 1] + period, bpm, 0.0, ts);
        Beatgrid::new(times, vec![seg])
    }

    #[test]
    fn downbeats_land_every_time_signature_beats() {
        let grid = uniform_grid(120.0, 16, 4);
        assert_eq!(grid.downbeat_indices(), vec![0, 4, 8, 12]);
        let waltz = uniform_grid(120.0, 12, 3);
        assert_eq!(waltz.downbeat_indices(), vec![0, 3, 6, 9]);
    }

    #[test]
    fn downbeat_times_are_always_real_beats() {
        let grid = uniform_grid(128.0, 64, 4);
        let beats = grid.beats();
        for t in grid.downbeat_times() {
            assert!(
                beats.iter().any(|b| (b - t).abs() < 1e-12),
                "downbeat {t} is not on a beat"
            );
        }
    }

    /// The regression this module exists to prevent. A jittered grid is what
    /// real detection produces; the seconds-accumulating definition drifts
    /// hundreds of milliseconds away from the beats over an hour, while the
    /// index-based one stays exact by construction.
    #[test]
    fn downbeats_do_not_drift_on_a_jittered_grid() {
        let bpm = 128.0;
        let period = 60.0 / bpm;
        let n = 7680; // one hour at 128 BPM
        let mut times = Vec::with_capacity(n);
        let mut t = 0.0;
        // Deterministic pseudo-jitter, a few milliseconds either way.
        for i in 0..n {
            let jitter = ((i as f64 * 12.9898).sin() * 43758.5453).fract() * 0.008 - 0.004;
            times.push(t);
            t += period + jitter;
        }
        let seg = TempoSegment::new(0.0, *times.last().unwrap() + period, bpm, times[0], 4);
        let grid = Beatgrid::new(times.clone(), vec![seg]);

        let index_based = grid.downbeat_times();
        for (k, t) in index_based.iter().enumerate() {
            let expected = times[k * 4];
            assert!(
                (t - expected).abs() < 1e-12,
                "bar {k} landed at {t}, not on beat {}",
                expected
            );
        }

        // The seconds-accumulating fallback is the one that drifts; show that
        // it really does, so this test documents the difference rather than
        // asserting both are fine.
        let accumulated = synthesize_downbeat_times(grid.segments());
        let pairs = index_based.len().min(accumulated.len());
        let worst = (0..pairs)
            .map(|i| (index_based[i] - accumulated[i]).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst > 0.05,
            "expected the accumulating definition to drift, saw only {worst}s"
        );
    }

    #[test]
    fn bar_beat_counts_from_one() {
        let grid = uniform_grid(120.0, 16, 4);
        let p = 0.5; // beat period
        assert_eq!(grid.bar_beat_at(0.0).unwrap(), BarBeat { bar: 1, beat: 1 });
        assert_eq!(grid.bar_beat_at(p).unwrap(), BarBeat { bar: 1, beat: 2 });
        assert_eq!(grid.bar_beat_at(3.0 * p).unwrap(), BarBeat { bar: 1, beat: 4 });
        assert_eq!(grid.bar_beat_at(4.0 * p).unwrap(), BarBeat { bar: 2, beat: 1 });
        assert_eq!(grid.bar_beat_label(4.0 * p), "2.1");
    }

    #[test]
    fn before_the_first_beat_reports_bar_zero() {
        let grid = Beatgrid::new(
            vec![1.0, 1.5, 2.0, 2.5],
            vec![TempoSegment::new(0.0, 3.0, 120.0, 1.0, 4)],
        );
        let pos = grid.bar_beat_at(0.1).unwrap();
        assert_eq!(pos.bar, 0);
    }

    #[test]
    fn empty_grid_has_no_position() {
        let grid = Beatgrid::default();
        assert!(grid.bar_beat_at(1.0).is_none());
        assert!(grid.downbeat_indices().is_empty());
        assert!(grid.downbeat_times().is_empty());
        assert_eq!(grid.bar_beat_label(1.0), "");
    }

    #[test]
    fn without_segments_assumes_four_four_from_the_first_beat() {
        let times: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let grid = Beatgrid::new(times, Vec::new());
        assert_eq!(grid.downbeat_indices(), vec![0, 4, 8]);
    }

    #[test]
    fn tempo_change_restarts_the_bar_count_at_its_own_inizio() {
        // 8 beats at 120 BPM, then 8 beats at 140 BPM whose bar starts on the
        // 9th beat overall.
        let mut times: Vec<f64> = (0..8).map(|i| i as f64 * 0.5).collect();
        let switch = 4.0;
        for i in 0..8 {
            times.push(switch + i as f64 * (60.0 / 140.0));
        }
        let segs = vec![
            TempoSegment::new(0.0, switch, 120.0, 0.0, 4),
            TempoSegment::new(switch, times[15] + 0.4, 140.0, switch, 4),
        ];
        let grid = Beatgrid::new(times, segs);
        assert_eq!(grid.downbeat_indices(), vec![0, 4, 8, 12]);
    }

    #[test]
    fn nearest_beat_prefers_the_closer_side() {
        let grid = Beatgrid::new(vec![0.0, 1.0, 2.0], Vec::new());
        assert_eq!(grid.nearest_beat_index(0.4), Some(0));
        assert_eq!(grid.nearest_beat_index(0.6), Some(1));
        assert_eq!(grid.nearest_beat_index(-5.0), Some(0));
        assert_eq!(grid.nearest_beat_index(99.0), Some(2));
    }

    #[test]
    fn beats_between_counts_a_half_open_range() {
        let grid = uniform_grid(120.0, 16, 4);
        assert_eq!(grid.beats_between(0.0, 2.0), 4);
        assert_eq!(grid.beats_between(2.0, 2.0), 0);
        assert_eq!(grid.beats_between(5.0, 1.0), 0, "reversed range is empty");
    }

    #[test]
    fn non_finite_beats_are_dropped_and_order_is_restored() {
        let grid = Beatgrid::new(vec![2.0, f64::NAN, 0.0, 1.0, f64::INFINITY], Vec::new());
        assert_eq!(grid.beats(), &[0.0, 1.0, 2.0]);
    }
}
