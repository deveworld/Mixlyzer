//! Two-stage gradient-boosted phrase (song-structure) detection.
//!
//! Given decoded audio, the beat times and the tempo map, this crate says where
//! the intro ends, where the chorus starts, and so on. It is a port of the
//! Python `analyzer_core.cue_and_phrase` pipeline and runs the same shipped
//! model, `assets/weights/phrase_analyzer.npz`.
//!
//! ```no_run
//! use mixlyzer_phrase::{detect_phrases, PhraseModel, PhraseOptions};
//! # fn main() -> Result<(), mixlyzer_phrase::PhraseError> {
//! # let (samples, beats, tempo): (Vec<f32>, Vec<f64>, Vec<mixlyzer_core::TempoSegment>) = Default::default();
//! let model = PhraseModel::load("assets/weights/phrase_analyzer.npz")?;
//! let options = PhraseOptions::new(model);
//! let phrases = detect_phrases(&samples, 22_050, &beats, &tempo, &options)?;
//! # Ok(()) }
//! ```
//!
//! # Fidelity to the Python original
//!
//! The models split on raw librosa feature values, so a feature that is close
//! but not identical silently degrades the output into something that still
//! looks like a song structure. Every frame-rate feature here is therefore a
//! deliberate reimplementation of a specific librosa routine, checked against
//! it by `rust/parity/phrase_parity.py`, and the ensemble runtime is checked
//! against the Python one by `tests/gbm_reference.rs`.
//!
//! Two things are deliberately *not* the same:
//!
//! * Audio must already be mono at the model's sample rate. The Python entry
//!   point resamples with `soxr_hq`, which has no equivalent here; rather than
//!   substitute a different resampler and quietly move every feature value,
//!   this crate reports [`PhraseError::SampleRateMismatch`]. Callers can use
//!   `mixlyzer_dsp::decode_for_analysis` to decode at the right rate.
//! * The end-state prior is off by default. See
//!   [`PhraseOptions::end_state_weight`].

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod boundary;
pub mod error;
pub mod features;
pub mod gbm;
pub mod grid;
pub mod label;
pub mod model;
pub mod npz;
pub mod testsig;

pub use error::PhraseError;
pub use grid::{build_predictor_grid, MeterSegment, PredictorGrid};
pub use model::{find_default_model, PhraseModel, DEFAULT_MODEL_RELATIVE_PATH};

use mixlyzer_core::{Phrase, TempoSegment};

/// Fewer beats than this and there is not enough context to place a boundary.
///
/// The boundary features look 16 beats either side and the edge guard blanks
/// 8 beats at each end, so a shorter track has no beat that could be chosen.
pub const MIN_BEATS_FOR_DETECTION: usize = 17;

/// How to run the detector.
#[derive(Debug, Clone)]
pub struct PhraseOptions {
    /// The loaded weights. Everything else defaults from the artifact.
    pub model: PhraseModel,

    /// Weight on the final segment's end-state prior. **Defaults to 0.**
    ///
    /// The shipped transition matrix gives P(END | SILENCE) = 0.863 against
    /// P(END | OUTRO) = 0.0026 — a 5.8 nat gap, and 7.2 nats once the
    /// OUTRO→SILENCE transition is counted too. That is an artifact of the
    /// training annotations, which end each track with a trailing silence
    /// segment; the production boundary detector never emits one, so at
    /// inference the prior has nothing legitimate to reward. Applied at full
    /// weight it relabels the last phrase of an ordinary fade-out as SILENCE,
    /// overriding a label classifier that is often 80% confident otherwise.
    ///
    /// Zero means the last segment is labelled on its own evidence, like every
    /// other segment. Set this to `1.0` to reproduce the Python behaviour
    /// exactly, including that bias.
    pub end_state_weight: f64,
}

impl PhraseOptions {
    /// Options with the model's own settings and the end-state prior disabled.
    pub fn new(model: PhraseModel) -> Self {
        Self {
            model,
            end_state_weight: 0.0,
        }
    }

    /// Options that reproduce the Python pipeline bit for bit, end-state bias
    /// included. Useful for parity checking, not recommended for production.
    pub fn matching_python(model: PhraseModel) -> Self {
        let transition_weight = model.settings.transition_weight;
        Self {
            model,
            end_state_weight: transition_weight,
        }
    }
}

/// Detect the phrase structure of one track.
///
/// `samples` must be mono at the model's sample rate (22.05 kHz for the shipped
/// weights). `beats` are beat onset times in seconds, strictly increasing.
/// `tempo_segments` supply the meter and the reference downbeat of each span.
///
/// Returns an empty vector — not an error — for a track with fewer than
/// [`MIN_BEATS_FOR_DETECTION`] beats: that is a short track, not a failure.
pub fn detect_phrases(
    samples: &[f32],
    sample_rate: u32,
    beats: &[f64],
    tempo_segments: &[TempoSegment],
    options: &PhraseOptions,
) -> Result<Vec<Phrase>, PhraseError> {
    let model = &options.model;
    let settings = &model.settings;
    if sample_rate != settings.sample_rate {
        return Err(PhraseError::SampleRateMismatch {
            found: sample_rate,
            expected: settings.sample_rate,
        });
    }

    let grid = build_predictor_grid(beats, tempo_segments)?;
    if grid.n_beats() < MIN_BEATS_FOR_DETECTION {
        return Ok(Vec::new());
    }

    let acoustic = features::song::extract_song_features(samples, settings, &grid)?;
    let feature_z = boundary::feature_z(&acoustic.stacked());
    let n_beats = feature_z.rows();

    let context = boundary::grid_context(&grid, n_beats);
    let boundary_features = boundary::boundary_feature_matrix(
        &feature_z,
        &settings.boundary_context_beats,
        &context,
    );
    let valid = boundary::valid_mask(n_beats, settings.edge_beats);
    let probability = boundary::boundary_probability(&model.boundary, &boundary_features, &valid);
    let raw_bounds = boundary::pick_boundaries(
        &model.boundary,
        &boundary_features,
        &valid,
        &probability,
        settings.min_distance_beats,
        settings.max_boundaries,
    );
    let bounds = boundary::refine_boundaries(
        &raw_bounds,
        &probability,
        &valid,
        &grid.downbeat_mask,
        settings,
    );

    let segment_features = label::segment_features(&feature_z, &bounds);
    let log_probability = label::label_log_probabilities(&model.label, &segment_features);
    let labels = label::decode_labels(
        model,
        &log_probability,
        &bounds,
        &label::LabelOptions {
            label_weight: settings.label_weight,
            transition_weight: settings.transition_weight,
            length_weight: settings.length_weight,
            end_state_weight: options.end_state_weight,
        },
    );

    Ok(assemble(&grid, &acoustic, &bounds, &labels))
}

/// Turn beat-index segments into timed phrases, dropping degenerate ones.
fn assemble(
    grid: &PredictorGrid,
    acoustic: &features::song::AcousticFeatures,
    bounds: &[usize],
    labels: &[String],
) -> Vec<Phrase> {
    let beats = &grid.beat_times_sec;
    // The final segment runs to the end of the audio, or a beat past the last
    // beat if the beat grid outlasts the file.
    let mut duration = acoustic.audio_duration_sec;
    if beats.len() >= 2 {
        let gaps: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
        duration = duration.max(beats[beats.len() - 1] + features::matrix::median(&gaps));
    }

    bounds
        .windows(2)
        .zip(labels)
        .filter_map(|(pair, label)| {
            let start = beats.get(pair[0]).copied().unwrap_or(duration);
            let end = beats.get(pair[1]).copied().unwrap_or(duration);
            if end - start <= mixlyzer_core::phrase::MIN_PHRASE_DURATION {
                return None;
            }
            Some(Phrase::new(start, end, label.clone()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> PhraseModel {
        let path = find_default_model(env!("CARGO_MANIFEST_DIR"))
            .expect("the shipped weights should be findable");
        PhraseModel::load(path).expect("load weights")
    }

    fn steady_beats(count: usize, period: f64) -> Vec<f64> {
        (0..count).map(|i| i as f64 * period).collect()
    }

    #[test]
    fn a_short_track_yields_no_phrases_rather_than_an_error() {
        let options = PhraseOptions::new(model());
        let beats = steady_beats(16, 0.5);
        let segments = vec![TempoSegment::new(0.0, 8.0, 120.0, 0.0, 4)];
        let samples = testsig::parity_signal(22_050, 9.0);
        let phrases = detect_phrases(&samples, 22_050, &beats, &segments, &options).unwrap();
        assert!(phrases.is_empty(), "16 beats is below the 17-beat floor");
    }

    #[test]
    fn a_sample_rate_the_model_was_not_trained_on_is_refused() {
        let options = PhraseOptions::new(model());
        let beats = steady_beats(64, 0.5);
        let segments = vec![TempoSegment::new(0.0, 32.0, 120.0, 0.0, 4)];
        let samples = testsig::parity_signal(44_100, 33.0);
        assert!(matches!(
            detect_phrases(&samples, 44_100, &beats, &segments, &options),
            Err(PhraseError::SampleRateMismatch { .. })
        ));
    }

    #[test]
    fn a_missing_tempo_map_is_an_error_not_a_guess() {
        let options = PhraseOptions::new(model());
        let beats = steady_beats(64, 0.5);
        let samples = testsig::parity_signal(22_050, 33.0);
        assert!(matches!(
            detect_phrases(&samples, 22_050, &beats, &[], &options),
            Err(PhraseError::NoTempoSegments)
        ));
    }

    #[test]
    fn matching_python_turns_the_end_state_prior_back_on() {
        let strict = PhraseOptions::matching_python(model());
        assert_eq!(strict.end_state_weight, strict.model.settings.transition_weight);
        assert_eq!(PhraseOptions::new(model()).end_state_weight, 0.0);
    }
}
