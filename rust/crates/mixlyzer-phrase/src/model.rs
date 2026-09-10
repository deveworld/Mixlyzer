//! The shipped phrase model: two ensembles plus the label decoder's priors.

use std::path::{Path, PathBuf};

use crate::error::PhraseError;
use crate::gbm::HistGradientBoosting;
use crate::npz::{self, Npz};

/// The only artifact layout this crate understands.
pub const NPZ_FORMAT: &str = "mixlyzer_phrase_weight_v1";

/// Where the desktop build keeps the shipped weights, relative to the repo root.
pub const DEFAULT_MODEL_RELATIVE_PATH: &str = "assets/weights/phrase_analyzer.npz";

/// The training-time settings the detector has to reproduce at inference.
///
/// These come out of the artifact rather than being hard-coded, because a
/// retrained model may use different context windows or a different edge guard,
/// and reading them from the file is what keeps the two in step.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSettings {
    pub sample_rate: u32,
    pub hop_length: usize,
    pub n_fft: usize,
    pub n_mels: usize,
    pub n_mfcc: usize,
    /// Half-widths, in beats, of the left/right context windows compared at
    /// each candidate boundary.
    pub boundary_context_beats: Vec<usize>,
    /// Beats at each end of the track that can never be a boundary.
    pub edge_beats: usize,
    /// Non-maximum suppression radius between accepted boundaries.
    pub min_distance_beats: usize,
    /// Cap on how many boundaries survive suppression, if any.
    pub max_boundaries: Option<usize>,
    /// How far the refinement pass may move a boundary.
    pub boundary_refine_window_beats: usize,
    /// Phrase lengths the refinement prefers, in beats.
    pub boundary_lengths_beats: Vec<usize>,
    pub boundary_length_weight: f64,
    pub boundary_shift_penalty: f64,
    pub boundary_downbeat_bonus: f64,
    pub label_weight: f64,
    pub transition_weight: f64,
    pub length_weight: f64,
}

impl Default for ModelSettings {
    fn default() -> Self {
        Self {
            sample_rate: 22_050,
            hop_length: 512,
            n_fft: 2048,
            n_mels: 48,
            n_mfcc: 20,
            boundary_context_beats: vec![1, 2, 4, 8, 16],
            edge_beats: 8,
            min_distance_beats: 16,
            max_boundaries: None,
            boundary_refine_window_beats: 8,
            boundary_lengths_beats: vec![16, 32, 64, 128],
            boundary_length_weight: 0.45,
            boundary_shift_penalty: 0.015,
            boundary_downbeat_bonus: 0.35,
            label_weight: 1.0,
            transition_weight: 1.0,
            length_weight: 0.0,
        }
    }
}

impl ModelSettings {
    fn from_json(text: &str) -> Self {
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap_or(serde_json::Value::Null);
        let mut out = Self::default();
        let get = |key: &str| parsed.get(key).cloned().unwrap_or(serde_json::Value::Null);
        let number = |key: &str, fallback: f64| get(key).as_f64().unwrap_or(fallback);
        let list = |key: &str, fallback: &[usize]| -> Vec<usize> {
            match get(key) {
                serde_json::Value::Array(items) => items
                    .iter()
                    .filter_map(serde_json::Value::as_u64)
                    .map(|v| v as usize)
                    .collect(),
                _ => fallback.to_vec(),
            }
        };

        out.sample_rate = number("sr", f64::from(out.sample_rate)) as u32;
        out.hop_length = number("hop_length", out.hop_length as f64) as usize;
        out.n_fft = number("n_fft", out.n_fft as f64) as usize;
        out.n_mels = number("n_mels", out.n_mels as f64) as usize;
        out.n_mfcc = number("n_mfcc", out.n_mfcc as f64) as usize;
        out.boundary_context_beats = list("boundary_context_beats", &out.boundary_context_beats);
        out.edge_beats = number("edge_beats", out.edge_beats as f64) as usize;
        out.min_distance_beats = number("min_distance_beats", out.min_distance_beats as f64) as usize;
        out.max_boundaries = get("max_boundaries").as_u64().map(|v| v as usize);
        out.boundary_refine_window_beats =
            number("boundary_refine_window_beats", out.boundary_refine_window_beats as f64) as usize;
        out.boundary_lengths_beats = list("boundary_lengths_beats", &out.boundary_lengths_beats);
        out.boundary_length_weight = number("boundary_length_weight", out.boundary_length_weight);
        out.boundary_shift_penalty = number("boundary_shift_penalty", out.boundary_shift_penalty);
        out.boundary_downbeat_bonus = number("boundary_downbeat_bonus", out.boundary_downbeat_bonus);
        out.label_weight = number("label_weight", out.label_weight);
        out.transition_weight = number("transition_weight", out.transition_weight);
        out.length_weight = number("length_weight", out.length_weight);
        out
    }
}

/// A loaded two-stage phrase model.
#[derive(Debug, Clone)]
pub struct PhraseModel {
    pub settings: ModelSettings,
    pub boundary: HistGradientBoosting,
    pub label: HistGradientBoosting,
    /// Segment labels, in the order the label ensemble scores them.
    pub labels: Vec<String>,
    /// `(n_labels + 2) x (n_labels + 2)` **log** transition probabilities; the
    /// two extra states are START (row `n_labels`) and END (column
    /// `n_labels + 1`).
    pub transition: Vec<Vec<f64>>,
    /// Mean of log segment length, per label.
    pub length_mu: Vec<f64>,
    /// Standard deviation of log segment length, per label.
    pub length_sigma: Vec<f64>,
}

impl PhraseModel {
    /// Read a model from a `.npz` on disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, PhraseError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| PhraseError::ModelIo {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_bytes(&bytes)
    }

    /// Read a model from an in-memory `.npz`, e.g. one embedded in the binary.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PhraseError> {
        let archive = npz::parse(bytes)?;
        Self::from_npz(&archive)
    }

    fn from_npz(archive: &Npz) -> Result<Self, PhraseError> {
        let format = archive
            .get("format")
            .ok_or_else(|| PhraseError::MissingModelField("format".into()))?
            .to_scalar_string("format")?;
        if format != NPZ_FORMAT {
            return Err(PhraseError::UnsupportedModelFormat {
                found: format,
                expected: NPZ_FORMAT.to_string(),
            });
        }

        let settings_json = archive
            .get("settings_json")
            .ok_or_else(|| PhraseError::MissingModelField("settings_json".into()))?
            .to_scalar_string("settings_json")?;
        let settings = ModelSettings::from_json(&settings_json);

        let labels = archive
            .get("label_labels")
            .ok_or_else(|| PhraseError::MissingModelField("label_labels".into()))?
            .to_strings("label_labels")?;
        let n = labels.len();

        let flat = archive
            .get("label_transition")
            .ok_or_else(|| PhraseError::MissingModelField("label_transition".into()))?
            .to_f64("label_transition")?;
        if flat.len() != (n + 2) * (n + 2) {
            return Err(PhraseError::ModelField {
                field: "label_transition".into(),
                reason: format!("has {} entries; {n} labels need {}", flat.len(), (n + 2) * (n + 2)),
            });
        }
        let transition: Vec<Vec<f64>> = flat.chunks(n + 2).map(<[f64]>::to_vec).collect();

        let length_mu = archive
            .get("label_length_mu")
            .ok_or_else(|| PhraseError::MissingModelField("label_length_mu".into()))?
            .to_f64("label_length_mu")?;
        let length_sigma = archive
            .get("label_length_sigma")
            .ok_or_else(|| PhraseError::MissingModelField("label_length_sigma".into()))?
            .to_f64("label_length_sigma")?;

        Ok(Self {
            settings,
            boundary: HistGradientBoosting::from_npz(archive, "boundary")?,
            label: HistGradientBoosting::from_npz(archive, "label")?,
            labels,
            transition,
            length_mu,
            length_sigma,
        })
    }

    /// Index of the START state in [`Self::transition`].
    pub fn start_state(&self) -> usize {
        self.labels.len()
    }

    /// Index of the END state in [`Self::transition`].
    pub fn end_state(&self) -> usize {
        self.labels.len() + 1
    }
}

/// Best guess at where the shipped weights live, walking up from `start`.
///
/// A library cannot know how the caller lays out its tree, so this only tries
/// the repository convention and reports failure plainly rather than guessing.
pub fn find_default_model(start: impl AsRef<Path>) -> Option<PathBuf> {
    let mut dir = Some(start.as_ref());
    while let Some(current) = dir {
        let candidate = current.join(DEFAULT_MODEL_RELATIVE_PATH);
        if candidate.is_file() {
            return Some(candidate);
        }
        dir = current.parent();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_fall_back_to_the_production_defaults_when_json_is_unusable() {
        let settings = ModelSettings::from_json("not json at all");
        assert_eq!(settings, ModelSettings::default());
    }

    #[test]
    fn a_null_max_boundaries_means_no_cap() {
        let settings = ModelSettings::from_json(r#"{"max_boundaries": null}"#);
        assert_eq!(settings.max_boundaries, None);
        let capped = ModelSettings::from_json(r#"{"max_boundaries": 12}"#);
        assert_eq!(capped.max_boundaries, Some(12));
    }

    #[test]
    fn settings_are_read_from_the_artifact_not_assumed() {
        let settings = ModelSettings::from_json(
            r#"{"sr": 44100, "edge_beats": 4, "boundary_context_beats": [2, 3]}"#,
        );
        assert_eq!(settings.sample_rate, 44_100);
        assert_eq!(settings.edge_beats, 4);
        assert_eq!(settings.boundary_context_beats, vec![2, 3]);
    }

    #[test]
    fn a_missing_model_file_reports_the_path() {
        let err = PhraseModel::load("/nonexistent/phrase_analyzer.npz").unwrap_err();
        assert!(matches!(err, PhraseError::ModelIo { .. }));
        assert!(err.to_string().contains("/nonexistent/phrase_analyzer.npz"));
    }

    #[test]
    fn an_empty_model_file_is_reported_as_malformed() {
        assert!(matches!(
            PhraseModel::from_bytes(&[]),
            Err(PhraseError::MalformedModel(_))
        ));
    }
}
