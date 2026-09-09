//! Tempo and key segments: the analysis output that describes how a track's
//! tempo and key change over time.

use crate::key::{Key, Mode};

/// Default time-signature numerator when a segment does not carry one.
pub const DEFAULT_TIME_SIGNATURE: u8 = 4;

/// One stretch of constant tempo.
///
/// This is the Rust form of a row of the `tempo_segments` array, which the
/// Python code stores as `[start, end, bpm, inizio, ts_num]` float32. Legacy
/// libraries hold 3- or 4-column rows; [`TempoSegment::from_row`] accepts all
/// three widths.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TempoSegment {
    /// Segment start, in seconds from the beginning of the track.
    pub start: f64,
    /// Segment end, in seconds. Exclusive.
    pub end: f64,
    /// Tempo in beats per minute. Always finite and positive for a valid segment.
    pub bpm: f64,
    /// Reference downbeat (bar start) inside the segment, in seconds.
    ///
    /// Named `inizio` after the Rekordbox XML attribute it feeds.
    pub inizio: f64,
    /// Beats per bar (the time-signature numerator).
    pub time_signature: u8,
}

impl TempoSegment {
    pub fn new(start: f64, end: f64, bpm: f64, inizio: f64, time_signature: u8) -> Self {
        Self {
            start,
            end,
            bpm,
            inizio,
            time_signature: if time_signature >= 1 {
                time_signature
            } else {
                DEFAULT_TIME_SIGNATURE
            },
        }
    }

    /// Parse one row of a stored tempo array, tolerating 3, 4 and 5 columns.
    ///
    /// Returns `None` for rows that cannot describe a real segment: too few
    /// columns, non-finite times, or a non-positive tempo. The Python readers
    /// scatter these checks across `beat_geometry`, `linear_segments` and
    /// `rekordbox`, and `_tempo_entries_for_xml` misses the `inizio` case
    /// badly enough to raise `TypeError` on `np.isfinite(None)`.
    pub fn from_row(row: &[f64]) -> Option<Self> {
        if row.len() < 3 {
            return None;
        }
        let (start, end, bpm) = (row[0], row[1], row[2]);
        if !start.is_finite() || !end.is_finite() || !bpm.is_finite() || bpm <= 0.0 {
            return None;
        }
        let inizio = match row.get(3) {
            Some(v) if v.is_finite() && *v >= 0.0 => *v,
            _ => start,
        };
        let time_signature = match row.get(4) {
            Some(v) if v.is_finite() && *v >= 1.0 => v.round() as u8,
            _ => DEFAULT_TIME_SIGNATURE,
        };
        Some(Self::new(start, end, bpm, inizio, time_signature))
    }

    /// Serialise back to the 5-column row layout used on disk.
    pub fn to_row(self) -> [f64; 5] {
        [
            self.start,
            self.end,
            self.bpm,
            self.inizio,
            f64::from(self.time_signature),
        ]
    }

    /// Length of the segment in seconds. Never negative.
    pub fn duration(self) -> f64 {
        (self.end - self.start).max(0.0)
    }

    /// Seconds between consecutive beats.
    pub fn beat_period(self) -> f64 {
        60.0 / self.bpm
    }

    /// Seconds between consecutive downbeats (one bar).
    pub fn bar_period(self) -> f64 {
        self.beat_period() * f64::from(self.time_signature)
    }

    /// Whether `time` falls inside `[start, end)`.
    pub fn contains(self, time: f64) -> bool {
        time >= self.start && time < self.end
    }
}

/// Parse a flat `[start, end, bpm, inizio, ts]`-style buffer into segments.
///
/// `width` is the number of columns per row. Rows that fail validation are
/// skipped rather than aborting the whole parse.
pub fn tempo_segments_from_flat(flat: &[f64], width: usize) -> Vec<TempoSegment> {
    if width < 3 || flat.is_empty() {
        return Vec::new();
    }
    flat.chunks_exact(width)
        .filter_map(TempoSegment::from_row)
        .collect()
}

/// One stretch of constant musical key.
///
/// Stored by Python as `[pitch, mode_flag, start, end]` — note the column order
/// differs from [`TempoSegment`], where the times come first.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct KeySegment {
    pub start: f64,
    pub end: f64,
    pub key: Key,
}

impl KeySegment {
    pub fn new(start: f64, end: f64, key: Key) -> Self {
        Self { start, end, key }
    }

    /// Parse one `[pitch, mode_flag, start, end]` row.
    pub fn from_row(row: &[f64]) -> Option<Self> {
        if row.len() < 4 {
            return None;
        }
        let (pitch, mode_flag, start, end) = (row[0], row[1], row[2], row[3]);
        if !pitch.is_finite() || !start.is_finite() || !end.is_finite() {
            return None;
        }
        let mode = if mode_flag.is_finite() && mode_flag.round() as i64 == 1 {
            Mode::Minor
        } else {
            Mode::Major
        };
        let pitch_class = (pitch.round() as i64).rem_euclid(12) as u8;
        Some(Self::new(start, end, Key::new(pitch_class, mode)))
    }

    pub fn to_row(self) -> [f64; 4] {
        [
            f64::from(self.key.pitch_class()),
            f64::from(self.key.mode().flag()),
            self.start,
            self.end,
        ]
    }

    pub fn duration(self) -> f64 {
        (self.end - self.start).max(0.0)
    }
}

/// Parse a flat `[pitch, mode, start, end]` buffer into key segments.
pub fn key_segments_from_flat(flat: &[f64], width: usize) -> Vec<KeySegment> {
    if width < 4 || flat.is_empty() {
        return Vec::new();
    }
    flat.chunks_exact(width)
        .filter_map(KeySegment::from_row)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_five_column_rows() {
        let seg = TempoSegment::from_row(&[0.0, 60.0, 128.0, 0.5, 3.0]).unwrap();
        assert_eq!(seg.start, 0.0);
        assert_eq!(seg.end, 60.0);
        assert_eq!(seg.bpm, 128.0);
        assert_eq!(seg.inizio, 0.5);
        assert_eq!(seg.time_signature, 3);
    }

    #[test]
    fn legacy_rows_default_inizio_and_time_signature() {
        let three = TempoSegment::from_row(&[10.0, 20.0, 120.0]).unwrap();
        assert_eq!(three.inizio, 10.0, "inizio defaults to the segment start");
        assert_eq!(three.time_signature, DEFAULT_TIME_SIGNATURE);

        let four = TempoSegment::from_row(&[10.0, 20.0, 120.0, 11.0]).unwrap();
        assert_eq!(four.inizio, 11.0);
        assert_eq!(four.time_signature, DEFAULT_TIME_SIGNATURE);
    }

    /// `third_party/rekordbox.py:245` calls `np.isfinite(None)` for these rows
    /// and raises `TypeError`, aborting export and Rekordbox sync entirely.
    #[test]
    fn non_finite_or_negative_inizio_falls_back_to_start() {
        for bad in [f64::NAN, f64::INFINITY, -1.0] {
            let seg = TempoSegment::from_row(&[5.0, 10.0, 120.0, bad]).unwrap();
            assert_eq!(seg.inizio, 5.0, "inizio {bad} should fall back to start");
        }
    }

    #[test]
    fn rejects_unusable_rows() {
        assert!(TempoSegment::from_row(&[0.0, 1.0]).is_none(), "too few columns");
        assert!(TempoSegment::from_row(&[f64::NAN, 1.0, 120.0]).is_none());
        assert!(TempoSegment::from_row(&[0.0, 1.0, 0.0]).is_none(), "zero bpm");
        assert!(TempoSegment::from_row(&[0.0, 1.0, -120.0]).is_none(), "negative bpm");
    }

    #[test]
    fn zero_time_signature_falls_back_to_four() {
        let seg = TempoSegment::from_row(&[0.0, 1.0, 120.0, 0.0, 0.0]).unwrap();
        assert_eq!(seg.time_signature, DEFAULT_TIME_SIGNATURE);
    }

    #[test]
    fn row_round_trips() {
        let seg = TempoSegment::new(1.0, 2.0, 130.5, 1.25, 3);
        assert_eq!(TempoSegment::from_row(&seg.to_row()).unwrap(), seg);
    }

    #[test]
    fn periods_follow_bpm_and_meter() {
        let seg = TempoSegment::new(0.0, 60.0, 120.0, 0.0, 4);
        assert!((seg.beat_period() - 0.5).abs() < 1e-12);
        assert!((seg.bar_period() - 2.0).abs() < 1e-12);
        let waltz = TempoSegment::new(0.0, 60.0, 120.0, 0.0, 3);
        assert!((waltz.bar_period() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn flat_buffer_parses_and_skips_bad_rows() {
        let flat = [
            0.0, 10.0, 120.0, 0.0, 4.0, // good
            10.0, 20.0, 0.0, 0.0, 4.0, // zero bpm -> skipped
            20.0, 30.0, 128.0, 20.5, 4.0, // good
        ];
        let segs = tempo_segments_from_flat(&flat, 5);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[1].bpm, 128.0);
    }

    #[test]
    fn key_rows_use_pitch_mode_start_end_order() {
        let seg = KeySegment::from_row(&[9.0, 1.0, 0.0, 30.0]).unwrap();
        assert_eq!(seg.key.index(), 21, "A minor is index 21");
        assert_eq!(seg.key.camelot(), "8A");
        assert_eq!(seg.start, 0.0);
        assert_eq!(seg.end, 30.0);
        assert_eq!(KeySegment::from_row(&seg.to_row()).unwrap(), seg);
    }

    #[test]
    fn key_row_mode_flag_other_than_one_is_major() {
        let seg = KeySegment::from_row(&[0.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(seg.key.mode(), Mode::Major);
        let nan_mode = KeySegment::from_row(&[0.0, f64::NAN, 0.0, 1.0]).unwrap();
        assert_eq!(nan_mode.key.mode(), Mode::Major);
    }
}
