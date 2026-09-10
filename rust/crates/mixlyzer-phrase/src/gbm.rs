//! Inference for an exported scikit-learn `HistGradientBoostingClassifier`.
//!
//! The training code flattens every tree in the ensemble into one long run of
//! node arrays, with `tree_offsets` marking where each tree starts. Prediction
//! is then just a walk down `left`/`right` from node 0 of each tree, adding the
//! leaf value into that tree's class channel.
//!
//! This has to agree with the Python runtime to the last bit, not merely
//! closely: a tree threshold is an arbitrary float, and a decision that goes
//! the other way changes the leaf, not the last decimal of the answer.

use crate::error::PhraseError;
use crate::npz::Npz;

/// A flattened gradient-boosted ensemble, ready for prediction.
#[derive(Debug, Clone)]
pub struct HistGradientBoosting {
    /// Class labels, in the order the raw score columns are laid out.
    pub classes: Vec<String>,
    /// Per-class starting score, before any tree contributes.
    baseline: Vec<f64>,
    /// Which class channel each tree adds into.
    tree_classes: Vec<usize>,
    /// `tree_offsets[i]..tree_offsets[i+1]` are tree `i`'s nodes.
    tree_offsets: Vec<usize>,
    value: Vec<f64>,
    feature_idx: Vec<usize>,
    threshold: Vec<f64>,
    missing_go_to_left: Vec<bool>,
    left: Vec<usize>,
    right: Vec<usize>,
    is_leaf: Vec<bool>,
}

fn field<'a>(archive: &'a Npz, name: &str) -> Result<&'a crate::npz::NpyArray, PhraseError> {
    archive
        .get(name)
        .ok_or_else(|| PhraseError::MissingModelField(name.to_string()))
}

impl HistGradientBoosting {
    /// Load the ensemble stored under `prefix` (`"boundary"` or `"label"`).
    pub fn from_npz(archive: &Npz, prefix: &str) -> Result<Self, PhraseError> {
        let name = |suffix: &str| format!("{prefix}_{suffix}");

        let classes = field(archive, &name("classes"))?.to_strings(&name("classes"))?;
        let baseline = field(archive, &name("baseline"))?.to_f64(&name("baseline"))?;
        let tree_classes =
            field(archive, &name("tree_classes"))?.to_usize(&name("tree_classes"))?;
        let tree_offsets =
            field(archive, &name("tree_offsets"))?.to_usize(&name("tree_offsets"))?;
        let value = field(archive, &name("node_value"))?.to_f64(&name("node_value"))?;
        let feature_idx =
            field(archive, &name("node_feature_idx"))?.to_usize(&name("node_feature_idx"))?;
        let threshold =
            field(archive, &name("node_num_threshold"))?.to_f64(&name("node_num_threshold"))?;
        let missing_go_to_left = field(archive, &name("node_missing_go_to_left"))?
            .to_bool(&name("node_missing_go_to_left"))?;
        let left = field(archive, &name("node_left"))?.to_usize(&name("node_left"))?;
        let right = field(archive, &name("node_right"))?.to_usize(&name("node_right"))?;
        let is_leaf = field(archive, &name("node_is_leaf"))?.to_bool(&name("node_is_leaf"))?;

        let nodes = value.len();
        for (label, len) in [
            ("node_feature_idx", feature_idx.len()),
            ("node_num_threshold", threshold.len()),
            ("node_missing_go_to_left", missing_go_to_left.len()),
            ("node_left", left.len()),
            ("node_right", right.len()),
            ("node_is_leaf", is_leaf.len()),
        ] {
            if len != nodes {
                return Err(PhraseError::ModelField {
                    field: name(label),
                    reason: format!("has {len} nodes but node_value has {nodes}"),
                });
            }
        }
        if tree_offsets.len() != tree_classes.len() + 1 {
            return Err(PhraseError::ModelField {
                field: name("tree_offsets"),
                reason: format!(
                    "has {} entries; {} trees need {}",
                    tree_offsets.len(),
                    tree_classes.len(),
                    tree_classes.len() + 1
                ),
            });
        }
        if tree_offsets.last().copied().unwrap_or(0) > nodes {
            return Err(PhraseError::ModelField {
                field: name("tree_offsets"),
                reason: "runs past the end of the node arrays".into(),
            });
        }
        if let Some(bad) = tree_classes.iter().find(|c| **c >= baseline.len()) {
            return Err(PhraseError::ModelField {
                field: name("tree_classes"),
                reason: format!("class {bad} has no baseline entry"),
            });
        }

        Ok(Self {
            classes,
            baseline,
            tree_classes,
            tree_offsets,
            value,
            feature_idx,
            threshold,
            missing_go_to_left,
            left,
            right,
            is_leaf,
        })
    }

    /// Number of raw score columns: 1 for binary, one per class otherwise.
    pub fn n_raw(&self) -> usize {
        self.baseline.len()
    }

    /// Number of classes the probabilities span.
    pub fn n_classes(&self) -> usize {
        if self.n_raw() == 1 {
            2
        } else {
            self.n_raw()
        }
    }

    /// Walk one tree for one sample and return its leaf value.
    fn predict_tree(&self, row: &[f64], start: usize, end: usize) -> f64 {
        let mut node = start;
        // A malformed model could describe a cycle; the node budget makes that
        // a wrong answer rather than a hang.
        for _ in 0..(end - start) + 1 {
            if self.is_leaf[node] {
                return self.value[node];
            }
            let column = self.feature_idx[node];
            let sample = row.get(column).copied().unwrap_or(f64::NAN);
            node = start
                + if sample.is_nan() {
                    if self.missing_go_to_left[node] {
                        self.left[node]
                    } else {
                        self.right[node]
                    }
                } else if sample <= self.threshold[node] {
                    self.left[node]
                } else {
                    self.right[node]
                };
            if node >= end {
                return 0.0;
            }
        }
        0.0
    }

    /// Raw (pre-link) scores for one sample, one per class channel.
    pub fn raw_predict_row(&self, row: &[f64]) -> Vec<f64> {
        let mut raw = self.baseline.clone();
        for (index, class) in self.tree_classes.iter().enumerate() {
            let start = self.tree_offsets[index];
            let end = self.tree_offsets[index + 1];
            raw[*class] += self.predict_tree(row, start, end);
        }
        raw
    }

    /// Class probabilities for one sample: sigmoid for binary, softmax above.
    pub fn predict_proba_row(&self, row: &[f64]) -> Vec<f64> {
        // The clamps are the Python runtime's, and they change the answer for
        // saturated scores, so they are part of the contract rather than a
        // defensive extra.
        const CLAMP: f64 = 50.0;
        const EPS: f64 = 1e-9;
        let raw = self.raw_predict_row(row);
        if raw.len() == 1 {
            let p1 = 1.0 / (1.0 + (-raw[0].clamp(-CLAMP, CLAMP)).exp());
            return vec![1.0 - p1, p1];
        }
        let max = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = raw
            .iter()
            .map(|v| (v - max).clamp(-CLAMP, CLAMP).exp())
            .collect();
        let total = exp.iter().sum::<f64>().max(EPS);
        exp.into_iter().map(|v| v / total).collect()
    }

    /// Predicted class index: `raw > 0` for binary, argmax above.
    pub fn predict_row(&self, row: &[f64]) -> usize {
        let raw = self.raw_predict_row(row);
        if raw.len() == 1 {
            return usize::from(raw[0] > 0.0);
        }
        raw.iter()
            .enumerate()
            .max_by(|(ai, a), (bi, b)| a.total_cmp(b).then(bi.cmp(ai)))
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    /// The predicted class *label*, as stored in the model.
    pub fn predict_label(&self, row: &[f64]) -> &str {
        let index = self.predict_row(row);
        self.classes.get(index).map_or("", String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npz::NpyArray;
    use std::collections::HashMap;

    /// A two-node stump: split on feature 0 at 0.5, leaves -1 and +1.
    fn stump() -> Npz {
        let mut archive: Npz = HashMap::new();
        archive.insert("m_classes".into(), NpyArray::Str(vec!["0".into(), "1".into()]));
        archive.insert("m_baseline".into(), NpyArray::F64(vec![0.0]));
        archive.insert("m_tree_classes".into(), NpyArray::I64(vec![0]));
        archive.insert("m_tree_offsets".into(), NpyArray::I64(vec![0, 3]));
        archive.insert("m_node_value".into(), NpyArray::F64(vec![0.0, -1.0, 1.0]));
        archive.insert("m_node_feature_idx".into(), NpyArray::I64(vec![0, 0, 0]));
        archive.insert("m_node_num_threshold".into(), NpyArray::F64(vec![0.5, 0.0, 0.0]));
        archive.insert("m_node_missing_go_to_left".into(), NpyArray::U8(vec![1, 0, 0]));
        archive.insert("m_node_left".into(), NpyArray::I64(vec![1, 0, 0]));
        archive.insert("m_node_right".into(), NpyArray::I64(vec![2, 0, 0]));
        archive.insert("m_node_is_leaf".into(), NpyArray::U8(vec![0, 1, 1]));
        archive
    }

    #[test]
    fn a_stump_splits_at_its_threshold_with_the_boundary_going_left() {
        let model = HistGradientBoosting::from_npz(&stump(), "m").unwrap();
        assert_eq!(model.predict_row(&[0.0]), 0);
        assert_eq!(model.predict_row(&[0.5]), 0, "<= threshold goes left");
        assert_eq!(model.predict_row(&[0.51]), 1);
    }

    #[test]
    fn a_missing_value_follows_the_flag_rather_than_the_threshold() {
        let model = HistGradientBoosting::from_npz(&stump(), "m").unwrap();
        assert_eq!(model.predict_row(&[f64::NAN]), 0, "missing_go_to_left is set");
    }

    #[test]
    fn binary_probabilities_are_the_sigmoid_of_the_raw_score() {
        let model = HistGradientBoosting::from_npz(&stump(), "m").unwrap();
        let p = model.predict_proba_row(&[1.0]);
        let expected = 1.0 / (1.0 + (-1.0f64).exp());
        assert!((p[1] - expected).abs() < 1e-12);
        assert!((p[0] + p[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mismatched_node_arrays_are_reported_rather_than_indexed_past() {
        let mut archive = stump();
        archive.insert("m_node_left".into(), NpyArray::I64(vec![1, 0]));
        let err = HistGradientBoosting::from_npz(&archive, "m").unwrap_err();
        assert!(matches!(err, PhraseError::ModelField { .. }));
    }

    #[test]
    fn a_missing_array_names_the_field() {
        let mut archive = stump();
        archive.remove("m_baseline");
        let err = HistGradientBoosting::from_npz(&archive, "m").unwrap_err();
        assert!(matches!(err, PhraseError::MissingModelField(f) if f == "m_baseline"));
    }
}
