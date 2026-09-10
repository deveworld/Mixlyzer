//! Naming each segment: per-segment features, then Viterbi over the labels.
//!
//! The label ensemble scores each segment on its own; the Viterbi pass then
//! picks the sequence that also reads like a song, using the transition matrix
//! learned from the training annotations (a chorus follows a verse, an intro
//! does not follow an outro).

use crate::features::matrix::{mean, std, Mat};
use crate::gbm::HistGradientBoosting;
use crate::model::PhraseModel;

/// Beats at each end of a segment that form its "head" and "tail" summaries.
const EDGE_BEATS: usize = 4;

/// Options for the label decoder that are not baked into the model artifact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelOptions {
    /// Weight on the per-segment classifier scores.
    pub label_weight: f64,
    /// Weight on the label-to-label transition log-probabilities.
    pub transition_weight: f64,
    /// Weight on the per-label log-length prior.
    pub length_weight: f64,
    /// Weight on the prior for the *final* segment's label.
    ///
    /// See [`crate::PhraseOptions::end_state_weight`] for why this defaults
    /// to zero rather than to `transition_weight`.
    pub end_state_weight: f64,
}

/// Build the label ensemble's input: one 537-column row per segment.
pub fn segment_features(feature_z: &Mat, bounds: &[usize]) -> Mat {
    let n = feature_z.rows();
    let d = feature_z.cols();
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(bounds.len().saturating_sub(1));
    for pair in bounds.windows(2) {
        let start = pair[0].min(n.saturating_sub(1));
        let end = pair[1].clamp(start + 1, n);
        let length = end - start;
        let edge = EDGE_BEATS.min(length).max(1);

        let mut row = Vec::with_capacity(4 * d + 5);
        let mut spreads = Vec::with_capacity(d);
        let mut maxima = Vec::with_capacity(d);
        let mut shifts = Vec::with_capacity(d);
        for c in 0..d {
            let block: Vec<f64> = (start..end).map(|r| feature_z.get(r, c)).collect();
            row.push(mean(&block));
            spreads.push(std(&block));
            maxima.push(block.iter().copied().fold(f64::NEG_INFINITY, f64::max));
            let head = mean(&block[..edge]);
            let tail = mean(&block[block.len() - edge..]);
            shifts.push(tail - head);
        }
        row.extend(spreads);
        row.extend(maxima);
        row.extend(shifts);
        // Where the segment sits and how long it is, in track-relative terms.
        let scale = n.max(1) as f64;
        row.push((length.max(1) as f64).ln());
        row.push(length as f64 / scale);
        row.push(start as f64 / scale);
        row.push(end as f64 / scale);
        row.push(0.5 * (start + end) as f64 / scale);
        for value in row.iter_mut() {
            if value.is_nan() {
                *value = 0.0;
            } else if value.is_infinite() {
                *value = if *value > 0.0 { f64::MAX } else { f64::MIN };
            }
        }
        rows.push(row);
    }
    Mat::from_rows(rows)
}

/// Log P(label | segment) for every segment, floored so a zero is survivable.
pub fn label_log_probabilities(model: &HistGradientBoosting, features: &Mat) -> Mat {
    let mut out = Mat::zeros(features.rows(), model.n_classes());
    for r in 0..features.rows() {
        let row: Vec<f64> = features.row(r).to_vec();
        for (c, p) in model.predict_proba_row(&row).into_iter().enumerate() {
            out.set(r, c, p.clamp(1e-7, 1.0).ln());
        }
    }
    out
}

/// Log-likelihood of a segment length under each label's length prior.
fn length_log_likelihood(mu: &[f64], sigma: &[f64], length: usize) -> Vec<f64> {
    let log_length = (length.max(1) as f64).ln();
    (0..mu.len())
        .map(|i| {
            // The floor stops a label that only ever appeared at one length
            // from becoming a hard constraint.
            let s = sigma.get(i).copied().unwrap_or(0.25).max(0.25);
            let z = (log_length - mu[i]) / s;
            -0.5 * z * z - s.ln()
        })
        .collect()
}

/// Viterbi over the label sequence, returning one label name per segment.
pub fn decode_labels(
    model: &PhraseModel,
    log_probability: &Mat,
    bounds: &[usize],
    options: &LabelOptions,
) -> Vec<String> {
    let segments = log_probability.rows();
    let n_labels = model.labels.len();
    if segments == 0 || n_labels == 0 {
        return Vec::new();
    }

    let mut emit = Mat::zeros(segments, n_labels);
    for s in 0..segments {
        let length = bounds
            .get(s + 1)
            .zip(bounds.get(s))
            .map_or(1, |(end, start)| end.saturating_sub(*start));
        let length_ll = if options.length_weight != 0.0 {
            length_log_likelihood(&model.length_mu, &model.length_sigma, length)
        } else {
            vec![0.0; n_labels]
        };
        for (l, length_term) in length_ll.iter().enumerate() {
            emit.set(
                s,
                l,
                options.label_weight * log_probability.get(s, l)
                    + options.length_weight * length_term,
            );
        }
    }

    let start_state = model.start_state();
    let end_state = model.end_state();
    let mut dp = Mat::zeros(segments, n_labels);
    let mut back = vec![vec![0usize; n_labels]; segments];
    for l in 0..n_labels {
        dp.set(
            0,
            l,
            emit.get(0, l) + options.transition_weight * model.transition[start_state][l],
        );
    }
    for (s, choices) in back.iter_mut().enumerate().skip(1) {
        for (l, choice) in choices.iter_mut().enumerate() {
            let mut best = f64::NEG_INFINITY;
            let mut best_from = 0usize;
            for from in 0..n_labels {
                let value =
                    dp.get(s - 1, from) + options.transition_weight * model.transition[from][l];
                if value > best {
                    best = value;
                    best_from = from;
                }
            }
            *choice = best_from;
            dp.set(s, l, emit.get(s, l) + best);
        }
    }

    let mut best = f64::NEG_INFINITY;
    let mut label = 0usize;
    for l in 0..n_labels {
        let value = dp.get(segments - 1, l) + options.end_state_weight * model.transition[l][end_state];
        if value > best {
            best = value;
            label = l;
        }
    }

    let mut out = vec![label];
    for s in (1..segments).rev() {
        label = back[s][label];
        out.push(label);
    }
    out.reverse();
    out.into_iter().map(|l| model.labels[l].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gbm::HistGradientBoosting;
    use crate::npz::NpyArray;
    use std::collections::HashMap;

    /// A model with the shipped transition matrix's shape and its END bias,
    /// but a trivial ensemble, so the decoder can be tested on its own.
    fn toy_model(labels: &[&str], transition: Vec<Vec<f64>>) -> PhraseModel {
        let n = labels.len();
        let mut archive: HashMap<String, NpyArray> = HashMap::new();
        archive.insert("t_classes".into(), NpyArray::Str(vec!["0".into()]));
        archive.insert("t_baseline".into(), NpyArray::F64(vec![0.0]));
        archive.insert("t_tree_classes".into(), NpyArray::I64(vec![]));
        archive.insert("t_tree_offsets".into(), NpyArray::I64(vec![0]));
        archive.insert("t_node_value".into(), NpyArray::F64(vec![]));
        archive.insert("t_node_feature_idx".into(), NpyArray::I64(vec![]));
        archive.insert("t_node_num_threshold".into(), NpyArray::F64(vec![]));
        archive.insert("t_node_missing_go_to_left".into(), NpyArray::U8(vec![]));
        archive.insert("t_node_left".into(), NpyArray::I64(vec![]));
        archive.insert("t_node_right".into(), NpyArray::I64(vec![]));
        archive.insert("t_node_is_leaf".into(), NpyArray::U8(vec![]));
        let stub = HistGradientBoosting::from_npz(&archive, "t").unwrap();
        PhraseModel {
            settings: crate::model::ModelSettings::default(),
            boundary: stub.clone(),
            label: stub,
            labels: labels.iter().map(|s| (*s).to_string()).collect(),
            transition,
            length_mu: vec![3.4; n],
            length_sigma: vec![0.4; n],
        }
    }

    /// Uniform transitions except for the END column, which carries the bias
    /// the shipped model has: P(END|SILENCE)=0.863 against P(END|OUTRO)=0.0026.
    fn outro_silence_model() -> PhraseModel {
        let labels = ["OUTRO", "SILENCE"];
        let n = labels.len();
        let mut transition = vec![vec![-0.7; n + 2]; n + 2];
        transition[0][n + 1] = -5.9402; // OUTRO -> END
        transition[1][n + 1] = -0.1468; // SILENCE -> END
        transition[0][1] = -0.4015; // OUTRO -> SILENCE
        toy_model(&labels, transition)
    }

    fn options(end_state_weight: f64) -> LabelOptions {
        LabelOptions {
            label_weight: 1.0,
            transition_weight: 1.0,
            length_weight: 0.0,
            end_state_weight,
        }
    }

    /// A single segment the classifier is fairly sure is an outro.
    fn outro_scores() -> Mat {
        Mat::from_rows(vec![vec![(0.80f64).ln(), (0.20f64).ln()]])
    }

    #[test]
    fn a_final_outro_is_not_rewritten_to_silence_by_default() {
        let model = outro_silence_model();
        let labels = decode_labels(&model, &outro_scores(), &[0, 64], &options(0.0));
        assert_eq!(labels, vec!["OUTRO"]);
    }

    /// The bias this guards against: the shipped end-state column is worth
    /// 5.8 nats, far more than the classifier's own 1.4-nat preference.
    #[test]
    fn the_end_state_prior_at_full_weight_would_override_the_classifier() {
        let model = outro_silence_model();
        let labels = decode_labels(&model, &outro_scores(), &[0, 64], &options(1.0));
        assert_eq!(
            labels,
            vec!["SILENCE"],
            "documents why end_state_weight defaults to 0"
        );
    }

    #[test]
    fn the_start_prior_still_applies_with_the_end_prior_off() {
        let labels = ["INTRO", "CHORUS"];
        let mut transition = vec![vec![-0.7; 4]; 4];
        transition[2][0] = -0.05; // START -> INTRO is overwhelmingly likely
        transition[2][1] = -6.0;
        let model = toy_model(&labels, transition);
        // The classifier marginally prefers CHORUS; the start prior wins.
        let scores = Mat::from_rows(vec![vec![(0.45f64).ln(), (0.55f64).ln()]]);
        assert_eq!(
            decode_labels(&model, &scores, &[0, 32], &options(0.0)),
            vec!["INTRO"]
        );
    }

    #[test]
    fn transitions_shape_the_sequence_not_just_each_segment() {
        let labels = ["VERSE", "CHORUS"];
        let mut transition = vec![vec![-0.7; 4]; 4];
        // Repeating a label is heavily penalised, so the run must alternate.
        transition[0][0] = -8.0;
        transition[1][1] = -8.0;
        let model = toy_model(&labels, transition);
        let flat = (0.5f64).ln();
        let scores = Mat::from_rows(vec![vec![flat; 2], vec![flat; 2], vec![flat; 2]]);
        let decoded = decode_labels(&model, &scores, &[0, 16, 32, 48], &options(0.0));
        assert_ne!(decoded[0], decoded[1]);
        assert_ne!(decoded[1], decoded[2]);
    }

    #[test]
    fn no_segments_decodes_to_no_labels() {
        let model = outro_silence_model();
        assert!(decode_labels(&model, &Mat::zeros(0, 2), &[0], &options(0.0)).is_empty());
    }

    #[test]
    fn segment_features_have_the_expected_537_columns() {
        let feature_z = Mat::zeros(64, 133);
        let features = segment_features(&feature_z, &[0, 32, 64]);
        assert_eq!(features.rows(), 2);
        assert_eq!(features.cols(), 4 * 133 + 5);
    }

    #[test]
    fn a_one_beat_segment_still_produces_a_head_and_a_tail() {
        let mut feature_z = Mat::zeros(8, 2);
        for r in 0..8 {
            feature_z.set(r, 0, r as f64);
        }
        let features = segment_features(&feature_z, &[0, 1, 8]);
        assert_eq!(features.rows(), 2);
        assert!(features.as_slice().iter().all(|v| v.is_finite()));
        // With one beat, tail and head are the same value, so the shift is 0.
        assert_eq!(features.get(0, 6), 0.0);
    }

    #[test]
    fn the_length_prior_favours_the_length_a_label_usually_has() {
        let mu = [(32.0f64).ln()];
        let sigma = [0.4];
        let near = length_log_likelihood(&mu, &sigma, 32)[0];
        let far = length_log_likelihood(&mu, &sigma, 4)[0];
        assert!(near > far);
    }
}
