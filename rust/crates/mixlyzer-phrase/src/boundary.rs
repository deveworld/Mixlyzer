//! Deciding which beats start a new phrase.
//!
//! Three stages, in order. A 1761-column feature vector per beat describes how
//! different the music is on either side of it, at five context widths. The
//! boundary ensemble then makes a *hard* call per beat — `predict`, not a
//! threshold on the probability — and each run of consecutive positives
//! collapses to its most confident beat. Finally a dynamic program is allowed
//! to slide each surviving boundary by up to eight beats, trading a small
//! penalty for landing on a downbeat and for phrase lengths near a power of two.

use crate::features::matrix::{mean, median, std, Mat};
use crate::grid::PredictorGrid;
use crate::model::ModelSettings;
use crate::gbm::HistGradientBoosting;

/// The floor the Python uses inside the boundary feature matrix.
const EPS: f64 = 1e-9;

/// Median-centre and MAD-scale each row, then transpose to `(beats, features)`.
///
/// Standardising per feature rather than per beat is what lets one ensemble
/// work across tracks of wildly different loudness and instrumentation.
pub fn feature_z(stacked: &Mat) -> Mat {
    let mut out = Mat::zeros(stacked.cols(), stacked.rows());
    for r in 0..stacked.rows() {
        let values = stacked.row(r);
        let centre = median(values);
        let deviations: Vec<f64> = values.iter().map(|v| (v - centre).abs()).collect();
        let mad = 1.4826 * median(&deviations);
        let scale = if mad > 1e-8 {
            mad
        } else {
            let spread = std(values);
            if spread > 1e-8 {
                spread
            } else {
                1.0
            }
        };
        for (c, value) in values.iter().enumerate() {
            out.set(c, r, (value - centre) / scale);
        }
    }
    out
}

/// Median-centre and MAD-scale each *column*, in place.
fn standardize_columns(mat: &mut Mat) {
    for c in 0..mat.cols() {
        let column = mat.column(c);
        let centre = median(&column);
        let deviations: Vec<f64> = column.iter().map(|v| (v - centre).abs()).collect();
        let mad = 1.4826 * median(&deviations);
        let scale = if mad > 1e-8 {
            mad
        } else {
            let spread = std(&column);
            if spread > 1e-8 {
                spread
            } else {
                1.0
            }
        };
        for r in 0..mat.rows() {
            mat.set(r, c, (mat.get(r, c) - centre) / scale);
        }
    }
}

/// Where each beat sits in the bar and in the hypermetre, as 17 columns.
///
/// Phrase edges are overwhelmingly on downbeats and on bars whose index is a
/// multiple of 2, 4, 8 or 16, so this gives the ensemble the metrical position
/// directly instead of making it infer one from the acoustics.
pub fn grid_context(grid: &PredictorGrid, n_beats: usize) -> Mat {
    if grid.beat_in_bar.len() != n_beats || grid.downbeat_mask.len() != n_beats {
        return Mat::zeros(n_beats, 0);
    }
    let meters: Vec<f64> = (0..n_beats)
        .map(|b| {
            grid.bar_meters
                .get(grid.bar_index_of_beat[b])
                .map_or(4.0, |m| (*m as f64).max(1.0))
        })
        .collect();

    let downbeats: Vec<usize> = (0..n_beats).filter(|b| grid.downbeat_mask[*b]).collect();
    let distance: Vec<f64> = (0..n_beats)
        .map(|b| {
            let previous = downbeats
                .iter()
                .rev()
                .find(|d| **d <= b)
                .map_or(n_beats as f64, |d| (b - d) as f64);
            let next = downbeats
                .iter()
                .find(|d| **d >= b)
                .map_or(n_beats as f64, |d| (d - b) as f64);
            previous.min(next) / meters[b].max(1.0)
        })
        .collect();

    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(17);
    rows.push((0..n_beats).map(|b| f64::from(u8::from(grid.downbeat_mask[b]))).collect());
    rows.push((0..n_beats).map(|b| f64::from(u8::from(grid.beat_in_bar[b] == 1))).collect());
    let phase: Vec<f64> = (0..n_beats)
        .map(|b| (grid.beat_in_bar[b] as f64 % meters[b]) / meters[b].max(1.0))
        .collect();
    rows.push(phase.iter().map(|p| (2.0 * std::f64::consts::PI * p).sin()).collect());
    rows.push(phase.iter().map(|p| (2.0 * std::f64::consts::PI * p).cos()).collect());
    rows.push(distance);
    for period in [2.0, 4.0, 8.0, 16.0] {
        let bar_pos: Vec<f64> = (0..n_beats).map(|b| grid.bar_index_of_beat[b] as f64).collect();
        rows.push(bar_pos.iter().map(|p| f64::from(u8::from(p % period == 0.0))).collect());
        rows.push(bar_pos.iter().map(|p| (2.0 * std::f64::consts::PI * p / period).sin()).collect());
        rows.push(bar_pos.iter().map(|p| (2.0 * std::f64::consts::PI * p / period).cos()).collect());
    }

    let mut out = Mat::zeros(n_beats, rows.len());
    for (c, values) in rows.iter().enumerate() {
        for (r, value) in values.iter().enumerate() {
            out.set(r, c, *value);
        }
    }
    standardize_columns(&mut out);
    out
}

/// Build the boundary ensemble's input, one row per beat.
pub fn boundary_feature_matrix(feature_z: &Mat, windows: &[usize], context: &Mat) -> Mat {
    let n = feature_z.rows();
    let d = feature_z.cols();
    let mut wins: Vec<usize> = windows.iter().map(|w| (*w).max(1)).collect();
    wins.sort_unstable();
    wins.dedup();
    let context = if context.rows() == n { Some(context) } else { None };

    let global_std: Vec<f64> = (0..d).map(|c| std(&feature_z.column(c)) + EPS).collect();

    let width = 3 * d + 2 * d * wins.len() + 3 * wins.len() + context.map_or(0, Mat::cols);
    let mut out = Mat::zeros(n, width);

    // Column means and standard deviations over `x[from..to]`, reused a lot.
    let block_stats = |from: usize, to: usize| -> (Vec<f64>, Vec<f64>) {
        let mut means = vec![0.0; d];
        let mut spreads = vec![0.0; d];
        let count = (to - from) as f64;
        for c in 0..d {
            let mut sum = 0.0;
            for r in from..to {
                sum += feature_z.get(r, c);
            }
            let m = sum / count;
            let mut variance = 0.0;
            for r in from..to {
                let delta = feature_z.get(r, c) - m;
                variance += delta * delta;
            }
            means[c] = m;
            spreads[c] = (variance / count).sqrt();
        }
        (means, spreads)
    };

    for b in 0..n {
        let mut column = 0usize;
        let push = |out: &mut Mat, value: f64, column: &mut usize| {
            out.set(b, *column, value);
            *column += 1;
        };
        for c in 0..d {
            push(&mut out, feature_z.get(b, c), &mut column);
        }
        // The step across this beat. `next` is the beat itself, not the one
        // after it, so this is a backward difference despite the Python names.
        let previous = b.saturating_sub(1);
        for c in 0..d {
            push(&mut out, feature_z.get(b, c) - feature_z.get(previous, c), &mut column);
        }
        for c in 0..d {
            push(
                &mut out,
                (feature_z.get(b, c) - feature_z.get(previous, c)).abs(),
                &mut column,
            );
        }

        let mut scalars: Vec<f64> = Vec::with_capacity(3 * wins.len());
        for w in &wins {
            let (left_from, left_to) = if b == 0 { (0, 1) } else { (b.saturating_sub(*w), b) };
            let (right_from, right_to) = (b, n.min(b + w));
            let (left_mean, left_std) = block_stats(left_from, left_to);
            let (right_mean, right_std) = block_stats(right_from, right_to);

            let diff: Vec<f64> = (0..d).map(|c| right_mean[c] - left_mean[c]).collect();
            for value in &diff {
                push(&mut out, *value, &mut column);
            }
            for value in &diff {
                push(&mut out, value.abs(), &mut column);
            }
            let absdiff: Vec<f64> = diff.iter().map(|v| v.abs()).collect();
            let scaled: f64 = (0..d)
                .map(|c| (diff[c] / global_std[c]).powi(2))
                .sum::<f64>()
                .sqrt();
            let spread_change: Vec<f64> =
                (0..d).map(|c| right_std[c] - left_std[c]).collect();
            scalars.push(mean(&absdiff));
            scalars.push(scaled / (d as f64).sqrt());
            scalars.push(mean(&spread_change));
        }
        for value in scalars {
            push(&mut out, value, &mut column);
        }
        if let Some(context) = context {
            for c in 0..context.cols() {
                push(&mut out, context.get(b, c), &mut column);
            }
        }
    }

    standardize_columns(&mut out);
    // `np.nan_to_num`: a constant column standardises to 0/0, and the trees
    // must see a number rather than a missing value there.
    for value in out.as_mut_slice() {
        if value.is_nan() {
            *value = 0.0;
        } else if value.is_infinite() {
            *value = if *value > 0.0 { f64::MAX } else { f64::MIN };
        }
    }
    out
}

/// Beats that may be a boundary: never the first or last `edge_beats`.
pub fn valid_mask(n: usize, edge_beats: usize) -> Vec<bool> {
    let mut valid = vec![true; n];
    let edge = edge_beats.min(n / 2);
    for slot in valid.iter_mut().take(edge) {
        *slot = false;
    }
    if edge > 0 {
        for slot in valid.iter_mut().skip(n - edge) {
            *slot = false;
        }
    }
    valid
}

/// P(boundary) per beat, zeroed outside the valid range and at both ends.
pub fn boundary_probability(
    model: &HistGradientBoosting,
    features: &Mat,
    valid: &[bool],
) -> Vec<f64> {
    let mut p: Vec<f64> = (0..features.rows())
        .map(|r| {
            let row: Vec<f64> = features.row(r).to_vec();
            model.predict_proba_row(&row)[1].clamp(1e-6, 1.0 - 1e-6)
        })
        .collect();
    for (index, value) in p.iter_mut().enumerate() {
        if !valid[index] {
            *value = 0.0;
        }
    }
    if let Some(first) = p.first_mut() {
        *first = 0.0;
    }
    if let Some(last) = p.last_mut() {
        *last = 0.0;
    }
    p
}

/// Pick boundaries from the ensemble's hard calls, with non-maximum suppression.
///
/// Returns beat indices including the implicit `0` and `n` bookends, so
/// consecutive pairs are the segments.
pub fn pick_boundaries(
    model: &HistGradientBoosting,
    features: &Mat,
    valid: &[bool],
    probability: &[f64],
    min_distance_beats: usize,
    max_boundaries: Option<usize>,
) -> Vec<usize> {
    let n = features.rows();
    let mut hard: Vec<bool> = (0..n)
        .map(|r| {
            let row: Vec<f64> = features.row(r).to_vec();
            let index = model.predict_row(&row);
            // The stored classes are the strings "0" and "1"; the decision is
            // on the label's value, not on its position.
            model
                .classes
                .get(index)
                .and_then(|c| c.parse::<i64>().ok())
                .map_or(index > 0, |value| value > 0)
        })
        .collect();
    for (index, value) in hard.iter_mut().enumerate() {
        if !valid[index] {
            *value = false;
        }
    }
    if let Some(first) = hard.first_mut() {
        *first = false;
    }
    if let Some(last) = hard.last_mut() {
        *last = false;
    }

    // One candidate per run of positives: the most confident beat in it.
    let mut candidates: Vec<usize> = Vec::new();
    let mut run_start: Option<usize> = None;
    // The trailing `false` closes a run that reaches the last beat.
    for (index, inside) in hard.iter().copied().chain(std::iter::once(false)).enumerate() {
        match (inside, run_start) {
            (true, None) => run_start = Some(index),
            (false, Some(start)) => {
                // The first beat achieving the maximum, matching Python's `max`.
                let mut best = start;
                for beat in start + 1..index {
                    if probability[beat] > probability[best] {
                        best = beat;
                    }
                }
                candidates.push(best);
                run_start = None;
            }
            _ => {}
        }
    }

    // Greedy suppression from the most confident candidate down.
    candidates.sort_by(|a, b| probability[*b].total_cmp(&probability[*a]));
    let mut selected: Vec<usize> = Vec::new();
    for beat in candidates {
        if selected
            .iter()
            .all(|prev| beat.abs_diff(*prev) >= min_distance_beats)
        {
            selected.push(beat);
            if max_boundaries.is_some_and(|cap| selected.len() >= cap) {
                break;
            }
        }
    }
    selected.sort_unstable();

    let mut bounds = vec![0usize];
    bounds.extend(selected);
    bounds.push(n);
    bounds
}

/// How well a phrase of `length` beats matches the preferred lengths.
///
/// A log-domain distance, so 24 beats is penalised the same whether it is read
/// as a long 16 or a short 32.
pub fn length_prior(length: usize, targets: &[usize]) -> f64 {
    if targets.is_empty() {
        return 0.0;
    }
    let length = (length.max(1) as f64).ln();
    let best = targets
        .iter()
        .map(|t| {
            let z = length - ((*t).max(1) as f64).ln();
            z * z
        })
        .fold(f64::INFINITY, f64::min);
    -0.5 * best
}

/// Slide each boundary within a window to a jointly better set of positions.
///
/// The ensemble is good at spotting *that* something changed and vague about
/// exactly which beat; this pass buys back that precision using the fact that
/// phrases are near-power-of-two lengths and start on downbeats. It is a proper
/// dynamic program rather than per-boundary greed, so lengths stay consistent.
pub fn refine_boundaries(
    raw: &[usize],
    probability: &[f64],
    valid: &[bool],
    downbeats: &[bool],
    settings: &ModelSettings,
) -> Vec<usize> {
    let window = settings.boundary_refine_window_beats;
    if raw.len() <= 2 || window == 0 {
        return raw.to_vec();
    }
    let n = raw[raw.len() - 1];
    if probability.len() != n || valid.len() != n {
        return raw.to_vec();
    }
    let downbeats: Vec<bool> = if downbeats.len() == n {
        downbeats.to_vec()
    } else {
        vec![false; n]
    };
    let targets = &settings.boundary_lengths_beats;
    let length_weight = settings.boundary_length_weight;

    let mut candidate_sets: Vec<Vec<usize>> = Vec::new();
    let mut emissions: Vec<Vec<f64>> = Vec::new();
    for centre in &raw[1..raw.len() - 1] {
        let lo = centre.saturating_sub(window).max(1);
        let hi = (centre + window).min(n - 1);
        let mut candidates: Vec<usize> = if hi >= lo {
            (lo..=hi).filter(|c| valid[*c]).collect()
        } else {
            Vec::new()
        };
        if candidates.is_empty() {
            candidates.push(*centre);
        }
        let scores: Vec<f64> = candidates
            .iter()
            .map(|c| {
                probability[*c].clamp(1e-7, 1.0).ln()
                    - settings.boundary_shift_penalty * (*c as f64 - *centre as f64).abs()
                    + settings.boundary_downbeat_bonus * f64::from(u8::from(downbeats[*c]))
            })
            .collect();
        candidate_sets.push(candidates);
        emissions.push(scores);
    }

    let mut dp: Vec<Vec<f64>> = Vec::with_capacity(candidate_sets.len());
    let mut back: Vec<Vec<isize>> = Vec::with_capacity(candidate_sets.len());
    for (index, candidates) in candidate_sets.iter().enumerate() {
        let mut scores = vec![f64::NEG_INFINITY; candidates.len()];
        let mut choice = vec![-1isize; candidates.len()];
        if index == 0 {
            for (j, beat) in candidates.iter().enumerate() {
                scores[j] = emissions[index][j] + length_weight * length_prior(*beat, targets);
            }
        } else {
            let previous = &candidate_sets[index - 1];
            for (j, beat) in candidates.iter().enumerate() {
                let mut best = f64::NEG_INFINITY;
                let mut best_index = -1isize;
                for (k, earlier) in previous.iter().enumerate() {
                    if beat <= earlier {
                        continue;
                    }
                    let value = dp[index - 1][k]
                        + length_weight * length_prior(beat - earlier, targets);
                    if value > best {
                        best = value;
                        best_index = k as isize;
                    }
                }
                if best_index >= 0 {
                    choice[j] = best_index;
                    scores[j] = emissions[index][j] + best;
                }
            }
        }
        dp.push(scores);
        back.push(choice);
    }

    let last_candidates = candidate_sets.last().expect("at least one boundary");
    let mut best = f64::NEG_INFINITY;
    let mut choice = -1isize;
    for (j, beat) in last_candidates.iter().enumerate() {
        if *beat >= n || !dp[dp.len() - 1][j].is_finite() {
            continue;
        }
        let value = dp[dp.len() - 1][j] + length_weight * length_prior(n - beat, targets);
        if value > best {
            best = value;
            choice = j as isize;
        }
    }
    if choice < 0 {
        return raw.to_vec();
    }

    let mut out = vec![last_candidates[choice as usize]];
    let mut current = choice;
    for index in (1..candidate_sets.len()).rev() {
        current = back[index][current as usize];
        if current < 0 {
            return raw.to_vec();
        }
        out.push(candidate_sets[index - 1][current as usize]);
    }
    out.reverse();

    let mut refined = vec![0usize];
    refined.extend(out);
    refined.push(n);
    if refined.windows(2).any(|w| w[1] <= w[0]) {
        return raw.to_vec();
    }
    refined
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_edge_guard_blanks_both_ends_and_never_the_whole_track() {
        let valid = valid_mask(40, 8);
        assert!(!valid[7] && valid[8] && valid[31] && !valid[32]);
        // A guard wider than half the track would leave nothing; it is clamped.
        let valid = valid_mask(10, 40);
        assert_eq!(valid.iter().filter(|v| **v).count(), 0);
        assert_eq!(valid.len(), 10);
    }

    #[test]
    fn the_length_prior_peaks_exactly_on_a_target_length() {
        let targets = [16, 32, 64, 128];
        assert!((length_prior(32, &targets)).abs() < 1e-12);
        assert!(length_prior(24, &targets) < 0.0);
        assert!(
            length_prior(32, &targets) > length_prior(40, &targets),
            "closer to a target must score higher"
        );
    }

    #[test]
    fn the_length_prior_is_symmetric_in_log_space() {
        let targets = [32];
        let half = length_prior(16, &targets);
        let double = length_prior(64, &targets);
        assert!((half - double).abs() < 1e-12);
    }

    /// A synthetic boundary picker, so suppression can be tested without a model.
    fn suppress(candidates: &[(usize, f64)], min_distance: usize, cap: Option<usize>) -> Vec<usize> {
        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut selected: Vec<usize> = Vec::new();
        for (beat, _) in sorted {
            if selected.iter().all(|p| beat.abs_diff(*p) >= min_distance) {
                selected.push(beat);
                if cap.is_some_and(|c| selected.len() >= c) {
                    break;
                }
            }
        }
        selected.sort_unstable();
        selected
    }

    #[test]
    fn suppression_keeps_the_more_confident_of_two_close_candidates() {
        let selected = suppress(&[(20, 0.6), (28, 0.9), (60, 0.5)], 16, None);
        assert_eq!(selected, vec![28, 60]);
    }

    #[test]
    fn suppression_stops_at_the_cap() {
        let selected = suppress(&[(20, 0.6), (60, 0.9), (100, 0.7)], 16, Some(2));
        assert_eq!(selected, vec![60, 100]);
    }

    fn refine_settings() -> ModelSettings {
        ModelSettings::default()
    }

    #[test]
    fn refinement_pulls_a_boundary_onto_a_nearby_downbeat() {
        let n = 128;
        let mut probability = vec![0.01; n];
        // The model is most confident at beat 62, but 64 is a downbeat and
        // makes both phrases exactly 64 beats long.
        probability[62] = 0.9;
        probability[64] = 0.8;
        let valid = valid_mask(n, 8);
        let downbeats: Vec<bool> = (0..n).map(|b| b % 4 == 0).collect();
        let refined = refine_boundaries(&[0, 62, n], &probability, &valid, &downbeats, &refine_settings());
        assert_eq!(refined, vec![0, 64, n]);
    }

    #[test]
    fn refinement_leaves_a_confident_boundary_where_it_is() {
        let n = 128;
        let mut probability = vec![0.001; n];
        probability[64] = 0.99;
        let valid = valid_mask(n, 8);
        let downbeats: Vec<bool> = (0..n).map(|b| b % 4 == 0).collect();
        let refined = refine_boundaries(&[0, 64, n], &probability, &valid, &downbeats, &refine_settings());
        assert_eq!(refined, vec![0, 64, n]);
    }

    #[test]
    fn refinement_never_reorders_or_collapses_boundaries() {
        let n = 200;
        let probability = vec![0.5; n];
        let valid = valid_mask(n, 8);
        let downbeats: Vec<bool> = (0..n).map(|b| b % 4 == 0).collect();
        let refined = refine_boundaries(
            &[0, 40, 44, 120, n],
            &probability,
            &valid,
            &downbeats,
            &refine_settings(),
        );
        assert!(refined.windows(2).all(|w| w[1] > w[0]), "got {refined:?}");
        assert_eq!(refined.len(), 5);
    }

    #[test]
    fn refinement_with_no_interior_boundary_is_a_no_op() {
        let probability = vec![0.5; 64];
        let valid = valid_mask(64, 8);
        let downbeats = vec![false; 64];
        assert_eq!(
            refine_boundaries(&[0, 64], &probability, &valid, &downbeats, &refine_settings()),
            vec![0, 64]
        );
    }

    #[test]
    fn refinement_cannot_move_a_boundary_beyond_its_window() {
        let n = 128;
        let mut probability = vec![0.01; n];
        // Far more confident 36 beats away, but the window is only 8 wide.
        probability[100] = 0.99;
        let valid = valid_mask(n, 8);
        let downbeats = vec![false; n];
        let mut settings = refine_settings();
        settings.boundary_shift_penalty = 0.0;
        settings.boundary_length_weight = 0.0;
        let refined = refine_boundaries(&[0, 64, n], &probability, &valid, &downbeats, &settings);
        assert!(
            (56..=72).contains(&refined[1]),
            "moved to {} from 64 with a window of 8",
            refined[1]
        );
    }

    #[test]
    fn refinement_prefers_the_more_confident_beat_inside_the_window() {
        let n = 128;
        let mut probability = vec![0.01; n];
        probability[70] = 0.99;
        let valid = valid_mask(n, 8);
        let downbeats = vec![false; n];
        let mut settings = refine_settings();
        settings.boundary_length_weight = 0.0;
        let refined = refine_boundaries(&[0, 64, n], &probability, &valid, &downbeats, &settings);
        assert_eq!(refined[1], 70);
    }
}
