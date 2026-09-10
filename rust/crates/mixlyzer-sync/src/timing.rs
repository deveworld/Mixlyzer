//! Turning a sample index into a playhead time.
//!
//! Some DJ programs expose the playhead as a sample index rather than seconds.
//! Converting needs the track's total sample count, which comes from one of two
//! places, matching `total_sample_count_source` in the config:
//!
//! * `reference_sample_rate` — assume a fixed rate and derive the total from
//!   the track's duration. Right when every track in the library was decoded at
//!   the same rate.
//! * `file` — use the total sample count stored with the track.
//!
//! Every division here is guarded. Python guards the zero cases but not the
//! nonsense ones: `round(nan * rate)` raises `ValueError` out of the poll loop,
//! which the caller turns into "Memory Sync read failed and has been disabled".

/// Where the total sample count comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TotalSampleSource {
    /// Derive it from the duration and a configured rate.
    ReferenceSampleRate,
    /// Use the count stored with the track.
    File,
}

/// What the engine needs to know about a track to convert a sample index.
///
/// Implemented by the host application over its library database.
pub trait TrackInfo {
    /// Track length in seconds, by normalised path.
    fn duration_sec(&self, normalized_path: &str) -> Option<f64>;

    /// Total decoded samples, by normalised path.
    fn total_samples(&self, normalized_path: &str) -> Option<i64>;
}

impl<T: TrackInfo + ?Sized> TrackInfo for Box<T> {
    fn duration_sec(&self, normalized_path: &str) -> Option<f64> {
        (**self).duration_sec(normalized_path)
    }

    fn total_samples(&self, normalized_path: &str) -> Option<i64> {
        (**self).total_samples(normalized_path)
    }
}

/// Total sample count for a track, or `0.0` when it cannot be determined.
pub fn total_samples_for(
    source: TotalSampleSource,
    duration_sec: f64,
    reference_sample_rate: u32,
    stored_total_samples: Option<i64>,
) -> f64 {
    match source {
        TotalSampleSource::ReferenceSampleRate => {
            let rate = f64::from(reference_sample_rate);
            if !duration_sec.is_finite() || duration_sec <= 0.0 || rate <= 0.0 {
                // A zero reference rate is a misconfiguration. Python hides it
                // behind `max(1.0, ...)` and reports a playhead that is wrong
                // by whatever the real rate is.
                return 0.0;
            }
            (duration_sec * rate).round()
        }
        TotalSampleSource::File => match stored_total_samples {
            Some(total) if total > 0 => total as f64,
            _ => 0.0,
        },
    }
}

/// Convert a sample index to seconds.
///
/// Returns `0.0` whenever the inputs cannot produce a meaningful time, so a
/// missing duration parks the playhead at the start instead of dividing by
/// zero. The index is clamped into the track.
pub fn sample_index_to_time(sample_index: f64, total_samples: f64, duration_sec: f64) -> f64 {
    if !sample_index.is_finite()
        || !total_samples.is_finite()
        || !duration_sec.is_finite()
        || total_samples <= 0.0
        || duration_sec <= 0.0
    {
        return 0.0;
    }
    let clamped = sample_index.clamp(0.0, total_samples);
    (clamped / total_samples) * duration_sec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sample_index_maps_onto_the_duration() {
        assert_eq!(sample_index_to_time(22_050.0, 44_100.0, 2.0), 1.0);
        assert_eq!(sample_index_to_time(0.0, 44_100.0, 2.0), 0.0);
    }

    #[test]
    fn the_index_is_clamped_into_the_track() {
        assert_eq!(sample_index_to_time(1e12, 44_100.0, 2.0), 2.0);
        assert_eq!(sample_index_to_time(-5.0, 44_100.0, 2.0), 0.0);
    }

    #[test]
    fn a_zero_total_never_divides() {
        assert_eq!(sample_index_to_time(1000.0, 0.0, 2.0), 0.0);
        assert_eq!(sample_index_to_time(1000.0, 44_100.0, 0.0), 0.0);
    }

    #[test]
    fn non_finite_inputs_park_the_playhead_at_zero() {
        assert_eq!(sample_index_to_time(f64::NAN, 44_100.0, 2.0), 0.0);
        assert_eq!(sample_index_to_time(1.0, f64::INFINITY, 2.0), 0.0);
        assert_eq!(sample_index_to_time(1.0, 44_100.0, f64::NAN), 0.0);
    }

    #[test]
    fn the_reference_rate_derives_a_total_from_the_duration() {
        assert_eq!(
            total_samples_for(TotalSampleSource::ReferenceSampleRate, 2.0, 44_100, None),
            88_200.0
        );
    }

    /// Python's `max(1.0, ...)` silently substitutes a different rate.
    #[test]
    fn a_zero_reference_rate_yields_no_total_rather_than_a_wrong_one() {
        assert_eq!(
            total_samples_for(TotalSampleSource::ReferenceSampleRate, 2.0, 0, None),
            0.0
        );
        assert_eq!(
            sample_index_to_time(
                1_000.0,
                total_samples_for(TotalSampleSource::ReferenceSampleRate, 2.0, 0, None),
                2.0
            ),
            0.0
        );
    }

    #[test]
    fn a_zero_or_missing_duration_yields_no_total() {
        assert_eq!(
            total_samples_for(TotalSampleSource::ReferenceSampleRate, 0.0, 44_100, None),
            0.0
        );
        assert_eq!(
            total_samples_for(
                TotalSampleSource::ReferenceSampleRate,
                f64::NAN,
                44_100,
                None
            ),
            0.0
        );
    }

    #[test]
    fn the_file_source_uses_the_stored_count() {
        assert_eq!(
            total_samples_for(TotalSampleSource::File, 2.0, 44_100, Some(96_000)),
            96_000.0
        );
        assert_eq!(
            total_samples_for(TotalSampleSource::File, 2.0, 44_100, None),
            0.0
        );
        assert_eq!(
            total_samples_for(TotalSampleSource::File, 2.0, 44_100, Some(-1)),
            0.0
        );
    }

    #[test]
    fn the_source_name_matches_the_python_config() {
        let json = "\"reference_sample_rate\"";
        let parsed: TotalSampleSource = serde_json::from_str(json).unwrap();
        assert_eq!(parsed, TotalSampleSource::ReferenceSampleRate);
        assert_eq!(
            serde_json::to_string(&TotalSampleSource::File).unwrap(),
            "\"file\""
        );
    }
}
