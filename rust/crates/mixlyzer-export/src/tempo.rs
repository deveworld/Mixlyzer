//! `TEMPO` elements: the beatgrid Rekordbox draws over the waveform.
//!
//! Rekordbox describes a beatgrid as a list of anchors. Each anchor says
//! "at time `Inizio` there is a beat, the tempo from here is `Bpm`, the bar has
//! `Metro` beats, and this beat is number `Battito` within its bar". One
//! [`TempoSegment`] therefore becomes exactly one anchor.

use mixlyzer_core::TempoSegment;

use crate::xml::Element;

/// How close to a whole beat an offset must be to count as landing on one.
///
/// Beat periods are irrational in decimal (1/3 s at 180 bpm), so an offset that
/// is mathematically 3 arrives as 3.0000000000000004.
const BEAT_SNAP_EPSILON: f64 = 1e-9;

/// One `TEMPO` anchor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TempoEntry {
    /// Time of the first grid beat at or after the segment start, in seconds.
    pub inizio: f64,
    /// Tempo from this anchor onwards.
    pub bpm: f64,
    /// Beats per bar; written as `Metro="{time_signature}/4"`.
    pub time_signature: u8,
    /// Which beat of the bar [`Self::inizio`] is, 1-based.
    pub battito: u8,
}

impl TempoEntry {
    /// Derive the anchor for one segment, or `None` when the segment cannot
    /// describe a grid (non-finite or negative start, non-positive tempo).
    ///
    /// The anchor beat is found by stepping from the segment's downbeat
    /// (`inizio`) in whole beats until reaching the segment start — the same
    /// `ceil((start - downbeat) / period)` the Python `_tempo_entries_for_xml`
    /// performs. `Battito` then counts that many beats around the bar, so a
    /// segment whose start *is* its downbeat anchors on beat 1.
    pub fn from_segment(segment: &TempoSegment) -> Option<Self> {
        if !segment.start.is_finite() || segment.start < 0.0 || !segment.bpm.is_finite() || segment.bpm <= 0.0
        {
            return None;
        }
        let period = segment.beat_period();
        let downbeat = if segment.inizio.is_finite() && segment.inizio >= 0.0 {
            segment.inizio
        } else {
            // `TempoSegment::from_row` already repairs this; a hand-built
            // segment can still carry a degenerate downbeat, and Python raises
            // `TypeError` on `np.isfinite(None)` rather than coping.
            segment.start
        };

        let offset = (segment.start - downbeat) / period;
        // `beats` counts whole beats from the downbeat to the anchor. Python
        // recovers the same number as `round((downbeat - first_beat) / period)`
        // and then walks it backwards around the bar; counting forward here is
        // the same arithmetic without the double negation.
        //
        // A segment usually starts exactly on a grid beat, and dividing two
        // doubles lands a hair either side of the integer: `ceil` alone would
        // push that anchor a whole beat late (and its `Battito` with it), which
        // is what Python does. Snap to the integer when we are within a
        // rounding error of one.
        let beats = if offset.is_finite() {
            let snapped = if (offset - offset.round()).abs() <= BEAT_SNAP_EPSILON {
                offset.round()
            } else {
                offset.ceil()
            };
            snapped.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
        } else {
            0
        };
        let first_beat = downbeat + f64::from(beats) * period;
        let time_signature = segment.time_signature.max(1);
        let battito = beats.rem_euclid(i32::from(time_signature)) + 1;

        Some(Self {
            inizio: first_beat,
            bpm: segment.bpm,
            time_signature,
            battito: battito as u8,
        })
    }

    /// Render as a `TEMPO` element.
    pub fn to_element(self) -> Element {
        Element::new("TEMPO")
            .attr("Inizio", format!("{:.3}", self.inizio))
            .attr("Bpm", format!("{:.2}", self.bpm))
            .attr("Metro", format!("{}/4", self.time_signature))
            .attr("Battito", self.battito.to_string())
    }
}

/// Build the anchors for a track, sorted by time.
///
/// When no segment yields an anchor, a single flat-grid anchor at 0 s is
/// emitted for `fallback_bpm` so the track still has a usable grid. Unlike
/// Python, a non-positive fallback produces no anchor at all: `Bpm="0.00"`
/// makes Rekordbox treat the grid as corrupt, which is worse than no grid.
pub fn tempo_entries(segments: &[TempoSegment], fallback_bpm: f64) -> Vec<TempoEntry> {
    let mut entries: Vec<TempoEntry> = segments.iter().filter_map(TempoEntry::from_segment).collect();
    if entries.is_empty() && fallback_bpm.is_finite() && fallback_bpm > 0.0 {
        entries.push(TempoEntry {
            inizio: 0.0,
            bpm: fallback_bpm,
            time_signature: mixlyzer_core::segments::DEFAULT_TIME_SIGNATURE,
            battito: 1,
        });
    }
    entries.sort_by(|a, b| a.inizio.total_cmp(&b.inizio));
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(start: f64, end: f64, bpm: f64, inizio: f64, ts: u8) -> TempoSegment {
        TempoSegment::new(start, end, bpm, inizio, ts)
    }

    /// 120 bpm is a 0.5 s beat: the anchor is the first beat at or after the
    /// start, and `Battito` counts it around the bar (n=0 -> 1, n=1 -> 2, ...).
    #[test]
    fn battito_counts_beats_from_the_downbeat_one_based() {
        for (n, expected) in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 1), (5, 2)] {
            let start = f64::from(n) * 0.5;
            let entry = TempoEntry::from_segment(&segment(start, start + 10.0, 120.0, 0.0, 4)).unwrap();
            assert_eq!(entry.battito, expected, "n={n}");
            assert!((entry.inizio - start).abs() < 1e-9, "n={n}");
        }
    }

    #[test]
    fn the_anchor_is_the_first_grid_beat_at_or_after_the_start() {
        // Start falls between beats 2 and 3 of the grid, so beat 3 anchors.
        let entry = TempoEntry::from_segment(&segment(1.2, 20.0, 120.0, 0.0, 4)).unwrap();
        assert!((entry.inizio - 1.5).abs() < 1e-9);
        assert_eq!(entry.battito, 4);
    }

    #[test]
    fn a_downbeat_after_the_start_still_anchors_on_a_grid_beat() {
        // Downbeat is later than the segment start: stepping back by whole
        // beats lands before the start, so `ceil` gives a negative count.
        let entry = TempoEntry::from_segment(&segment(0.0, 10.0, 120.0, 1.25, 4)).unwrap();
        assert!((entry.inizio - 0.25).abs() < 1e-9);
        assert_eq!(entry.battito, 3);
    }

    /// 180 bpm has a 1/3 s beat, so `(1.0 - 0.0) / (60.0 / 180.0)` evaluates to
    /// 3.0000000000000004 and Python's bare `ceil` anchors on beat 5 instead of
    /// beat 4, one beat late.
    #[test]
    fn grid_anchors_are_not_pushed_a_beat_late_by_floating_point_noise() {
        let entry = TempoEntry::from_segment(&segment(1.0, 9.0, 180.0, 0.0, 4)).unwrap();
        assert!((entry.inizio - 1.0).abs() < 1e-9, "anchored at {}", entry.inizio);
        assert_eq!(entry.battito, 4);
    }

    #[test]
    fn non_four_four_meters_wrap_battito_at_their_own_bar_length() {
        let entry = TempoEntry::from_segment(&segment(1.0, 9.0, 180.0, 0.0, 3)).unwrap();
        // 180 bpm -> 1/3 s beat; start 1.0 s is beat 3 after the downbeat.
        assert_eq!(entry.time_signature, 3);
        assert_eq!(entry.battito, 1);
        assert_eq!(entry.to_element().attribute("Metro"), Some("3/4"));
    }

    /// Python calls `np.isfinite(None)` on a missing `inizio` and dies with a
    /// `TypeError`; `TempoSegment::from_row` substitutes the segment start.
    #[test]
    fn a_segment_with_a_degenerate_inizio_still_exports() {
        let from_row = TempoSegment::from_row(&[4.0, 8.0, 128.0, f64::NAN, 4.0]).unwrap();
        let entry = TempoEntry::from_segment(&from_row).unwrap();
        assert!((entry.inizio - 4.0).abs() < 1e-9);
        assert_eq!(entry.battito, 1);

        // A hand-built segment carrying NaN directly is repaired here too.
        let direct = TempoEntry::from_segment(&segment(4.0, 8.0, 128.0, f64::NAN, 4)).unwrap();
        assert_eq!(direct.inizio, 4.0);
        assert_eq!(direct.battito, 1);

        let negative = TempoSegment::from_row(&[4.0, 8.0, 128.0, -3.0, 4.0]).unwrap();
        assert_eq!(TempoEntry::from_segment(&negative).unwrap().inizio, 4.0);
    }

    #[test]
    fn invalid_segments_are_skipped() {
        assert!(TempoEntry::from_segment(&segment(-1.0, 5.0, 120.0, 0.0, 4)).is_none());
        assert!(TempoEntry::from_segment(&segment(f64::NAN, 5.0, 120.0, 0.0, 4)).is_none());
        assert!(TempoEntry::from_segment(&segment(0.0, 5.0, 0.0, 0.0, 4)).is_none());
        assert!(TempoEntry::from_segment(&segment(0.0, 5.0, f64::NAN, 0.0, 4)).is_none());
    }

    #[test]
    fn entries_come_out_sorted_by_time() {
        let segments = [
            segment(60.0, 90.0, 130.0, 60.0, 4),
            segment(0.0, 30.0, 120.0, 0.0, 4),
            segment(30.0, 60.0, 125.0, 30.0, 4),
        ];
        let times: Vec<f64> = tempo_entries(&segments, 0.0).iter().map(|e| e.inizio).collect();
        assert_eq!(times, vec![0.0, 30.0, 60.0]);
    }

    #[test]
    fn with_no_segments_a_flat_grid_falls_back_to_the_average_bpm() {
        let entries = tempo_entries(&[], 128.0);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].inizio, 0.0);
        assert_eq!(entries[0].bpm, 128.0);
        assert_eq!(entries[0].battito, 1);
    }

    /// Deliberately unlike Python, which writes `Bpm="0.00"` in this case.
    #[test]
    fn with_no_segments_and_no_tempo_no_anchor_is_emitted() {
        assert!(tempo_entries(&[], 0.0).is_empty());
        assert!(tempo_entries(&[], f64::NAN).is_empty());
    }

    #[test]
    fn the_element_formats_bpm_to_two_places_and_time_to_three() {
        let entry = TempoEntry::from_segment(&segment(1.0 / 3.0, 9.0, 128.456, 1.0 / 3.0, 4)).unwrap();
        let element = entry.to_element();
        assert_eq!(element.name(), "TEMPO");
        assert_eq!(element.attribute("Inizio"), Some("0.333"));
        assert_eq!(element.attribute("Bpm"), Some("128.46"));
        assert_eq!(element.attribute("Battito"), Some("1"));
    }
}
