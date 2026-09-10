//! Chroma extraction and key detection.
//!
//! A chromagram folds the spectrum onto the twelve pitch classes; correlating
//! it against key profiles says which key the music is in, and a Viterbi pass
//! over time turns per-frame opinions into stretches of stable key with clean
//! boundaries.
//!
//! # Both modes are tracked
//!
//! The Python implementation scores only the twelve **major** templates, then
//! decides one mode for the entire track by majority vote and rewrites every
//! segment into it. A track that genuinely moves between relative major and
//! minor cannot be represented at all, which is at odds with what the app
//! advertises. Here all 24 keys are states in the same decode, so a modal
//! change is just another transition.

use mixlyzer_core::key::{Key, Mode};
use mixlyzer_core::segments::KeySegment;

use std::sync::Arc;

use rustfft::{num_complex::Complex, Fft, FftPlanner};

/// Krumhansl-Kessler major profile, the classic probe-tone ratings.
const MAJOR_PROFILE: [f64; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];

/// Krumhansl-Kessler minor profile.
const MINOR_PROFILE: [f64; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// A chromagram: twelve pitch-class energies per frame.
#[derive(Debug, Clone, PartialEq)]
pub struct Chroma {
    /// Row-major, `frames * 12`.
    values: Vec<f64>,
    frames: usize,
    /// Time of each frame, in seconds.
    times: Vec<f64>,
}

impl Chroma {
    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn is_empty(&self) -> bool {
        self.frames == 0
    }

    /// The twelve values of one frame.
    pub fn frame(&self, index: usize) -> &[f64] {
        &self.values[index * 12..(index + 1) * 12]
    }

    /// Time of frame `index`.
    pub fn time(&self, index: usize) -> f64 {
        self.times.get(index).copied().unwrap_or(0.0)
    }

    /// Average chroma over the whole piece, normalised to sum to one.
    pub fn mean(&self) -> [f64; 12] {
        let mut out = [0.0f64; 12];
        for frame in 0..self.frames {
            for (pitch, value) in self.frame(frame).iter().enumerate() {
                out[pitch] += value;
            }
        }
        let total: f64 = out.iter().sum();
        if total > 0.0 {
            for value in out.iter_mut() {
                *value /= total;
            }
        }
        out
    }
}

/// How chroma is extracted.
#[derive(Debug, Clone, Copy)]
pub struct ChromaOptions {
    pub n_fft: usize,
    pub hop: usize,
    /// Lowest frequency to fold in. Below this, pitch is unreliable.
    pub fmin: f64,
    /// Highest frequency to fold in.
    pub fmax: f64,
}

impl Default for ChromaOptions {
    fn default() -> Self {
        Self {
            n_fft: 4096,
            hop: 2048,
            fmin: 65.0,   // C2
            fmax: 2100.0, // C7
        }
    }
}

/// Fold a signal's spectrum onto the twelve pitch classes, frame by frame.
pub fn chroma(signal: &[f32], sample_rate: u32, options: ChromaOptions) -> Chroma {
    let n_fft = options.n_fft.max(64).next_power_of_two();
    let hop = options.hop.max(1);
    if signal.len() < n_fft || sample_rate == 0 {
        return Chroma {
            values: Vec::new(),
            frames: 0,
            times: Vec::new(),
        };
    }

    let window: Vec<f64> = (0..n_fft)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / n_fft as f64;
            0.5 * (1.0 - phase.cos())
        })
        .collect();

    // Precompute which pitch class each bin belongs to, and how strongly.
    // Bins outside the usable range contribute nothing.
    let bin_hz = f64::from(sample_rate) / n_fft as f64;
    let bins = n_fft / 2 + 1;
    let mut bin_pitch: Vec<Option<(usize, f64)>> = Vec::with_capacity(bins);
    for bin in 0..bins {
        let freq = bin as f64 * bin_hz;
        if freq < options.fmin || freq > options.fmax {
            bin_pitch.push(None);
            continue;
        }
        // MIDI note number, then fold to a pitch class.
        let midi = 69.0 + 12.0 * (freq / 440.0).log2();
        let nearest = midi.round();
        let cents_off = (midi - nearest).abs();
        // Weight by how close the bin is to a real semitone centre, so energy
        // smeared between two semitones does not vote at full strength.
        let weight = (1.0 - 2.0 * cents_off).max(0.0);
        if weight <= 0.0 {
            bin_pitch.push(None);
        } else {
            let pitch_class = (nearest as i64).rem_euclid(12) as usize;
            bin_pitch.push(Some((pitch_class, weight)));
        }
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(n_fft);
    let frame_count = (signal.len() - n_fft) / hop + 1;

    let mut values = Vec::with_capacity(frame_count * 12);
    let mut times = Vec::with_capacity(frame_count);
    let mut scratch = vec![Complex::new(0.0, 0.0); n_fft];

    for frame in 0..frame_count {
        let start = frame * hop;
        for (i, slot) in scratch.iter_mut().enumerate() {
            *slot = Complex::new(f64::from(signal[start + i]) * window[i], 0.0);
        }
        fft.process(&mut scratch);

        let mut row = [0.0f64; 12];
        for (bin, mapping) in bin_pitch.iter().enumerate() {
            if let Some((pitch_class, weight)) = mapping {
                row[*pitch_class] += scratch[bin].norm() * weight;
            }
        }
        let total: f64 = row.iter().sum();
        if total > 0.0 {
            for value in row.iter_mut() {
                *value /= total;
            }
        }
        values.extend_from_slice(&row);
        // The window's centre is the honest timestamp for a pitch measurement.
        times.push((start as f64 + n_fft as f64 / 2.0) / f64::from(sample_rate));
    }

    Chroma {
        values,
        frames: frame_count,
        times,
    }
}

/// Average chroma between consecutive beats, one row per beat.
///
/// Chords change on beats far more often than between them, so averaging
/// within a beat removes noise without blurring a real change.
pub fn beat_synchronous(chroma: &Chroma, beats: &[f64]) -> Chroma {
    if chroma.is_empty() || beats.len() < 2 {
        return chroma.clone();
    }
    let mut values = Vec::with_capacity((beats.len() - 1) * 12);
    let mut times = Vec::with_capacity(beats.len() - 1);

    for pair in beats.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        let mut row = [0.0f64; 12];
        let mut count = 0usize;
        for frame in 0..chroma.frames() {
            let t = chroma.time(frame);
            if t >= start && t < end {
                for (pitch, value) in chroma.frame(frame).iter().enumerate() {
                    row[pitch] += value;
                }
                count += 1;
            }
        }
        if count > 0 {
            let total: f64 = row.iter().sum();
            if total > 0.0 {
                for value in row.iter_mut() {
                    *value /= total;
                }
            }
        }
        values.extend_from_slice(&row);
        times.push(start);
    }

    Chroma {
        frames: times.len(),
        values,
        times,
    }
}

/// How key detection is tuned.
#[derive(Debug, Clone, Copy)]
pub struct KeyOptions {
    /// Probability of staying in the same key from one frame to the next.
    ///
    /// High, because keys last for tens of seconds; the cost of a low value is
    /// a key track that flickers on every passing chord.
    pub self_transition: f64,
    /// Relative weight of a move to a harmonically close key.
    pub near_transition: f64,
    /// Relative weight of a move to any other key.
    pub far_transition: f64,
}

impl Default for KeyOptions {
    fn default() -> Self {
        Self {
            self_transition: 0.98,
            near_transition: 0.015,
            far_transition: 0.005,
        }
    }
}

/// Score every key against one chroma frame, as log likelihoods.
fn frame_log_likelihoods(frame: &[f64]) -> [f64; 24] {
    let mut scores = [0.0f64; 24];
    for (index, score) in scores.iter_mut().enumerate() {
        let key = Key::from_index(index as i64);
        let profile = match key.mode() {
            Mode::Major => &MAJOR_PROFILE,
            Mode::Minor => &MINOR_PROFILE,
        };
        let tonic = usize::from(key.pitch_class());
        // Correlate the frame against the profile rotated to this tonic.
        let mut dot = 0.0;
        let mut profile_norm = 0.0;
        let mut frame_norm = 0.0;
        for step in 0..12 {
            let weight = profile[step];
            let observed = frame[(tonic + step) % 12];
            dot += weight * observed;
            profile_norm += weight * weight;
            frame_norm += observed * observed;
        }
        let denom = (profile_norm * frame_norm).sqrt();
        let correlation = if denom > 1e-12 { dot / denom } else { 0.0 };
        // Sharpen so the decode has something to prefer; a raw cosine over
        // these profiles spans only a narrow band.
        *score = (correlation.max(1e-6) * 8.0).ln();
    }
    scores
}

/// Log transition weight from `from` to `to`.
fn transition_log(from: usize, to: usize, options: KeyOptions) -> f64 {
    if from == to {
        return options.self_transition.max(1e-12).ln();
    }
    let source = Key::from_index(from as i64);
    let target = Key::from_index(to as i64);
    let weight = if source.harmonic_neighbours().contains(&target) {
        options.near_transition
    } else {
        options.far_transition
    };
    weight.max(1e-12).ln()
}

/// Decode the most likely key path through a chromagram.
///
/// Returns one key per chroma frame.
pub fn decode_key_path(chroma: &Chroma, options: KeyOptions) -> Vec<Key> {
    if chroma.is_empty() {
        return Vec::new();
    }
    let frames = chroma.frames();
    let mut costs = frame_log_likelihoods(chroma.frame(0));
    let mut backpointers: Vec<[u8; 24]> = Vec::with_capacity(frames);

    for frame in 1..frames {
        let emission = frame_log_likelihoods(chroma.frame(frame));
        let mut next = [f64::NEG_INFINITY; 24];
        let mut chosen = [0u8; 24];
        for to in 0..24 {
            let (best_from, best_cost) = (0..24)
                .map(|from| (from, costs[from] + transition_log(from, to, options)))
                .fold((0usize, f64::NEG_INFINITY), |best, candidate| {
                    if candidate.1 > best.1 {
                        candidate
                    } else {
                        best
                    }
                });
            next[to] = best_cost + emission[to];
            chosen[to] = best_from as u8;
        }
        costs = next;
        backpointers.push(chosen);
    }

    let mut state = (0..24)
        .fold((0usize, f64::NEG_INFINITY), |best, index| {
            if costs[index] > best.1 {
                (index, costs[index])
            } else {
                best
            }
        })
        .0;

    let mut path = vec![Key::from_index(state as i64); frames];
    for (frame, slot) in path.iter_mut().enumerate().take(frames - 1).rev() {
        state = usize::from(backpointers[frame][state]);
        *slot = Key::from_index(state as i64);
    }
    path
}

/// Turn a per-frame key path into contiguous segments.
pub fn segments_from_path(chroma: &Chroma, path: &[Key], duration_sec: f64) -> Vec<KeySegment> {
    if path.is_empty() {
        return Vec::new();
    }
    let mut segments: Vec<KeySegment> = Vec::new();
    let mut run_start = chroma.time(0);
    let mut current = path[0];

    for (index, key) in path.iter().enumerate().skip(1) {
        if *key != current {
            let boundary = chroma.time(index);
            if boundary > run_start {
                segments.push(KeySegment::new(run_start, boundary, current));
            }
            run_start = boundary;
            current = *key;
        }
    }
    let end = duration_sec.max(chroma.time(path.len() - 1));
    if end > run_start {
        segments.push(KeySegment::new(run_start, end, current));
    }
    segments
}

/// The single key that best describes the whole piece.
pub fn overall_key(chroma: &Chroma) -> Option<Key> {
    if chroma.is_empty() {
        return None;
    }
    let mean = chroma.mean();
    let scores = frame_log_likelihoods(&mean);
    let best = (0..24).fold((0usize, f64::NEG_INFINITY), |best, index| {
        if scores[index] > best.1 {
            (index, scores[index])
        } else {
            best
        }
    });
    Some(Key::from_index(best.0 as i64))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 22_050;

    /// Sum of sine partials for one MIDI note.
    fn note(midi: f64, seconds: f64, amplitude: f64) -> Vec<f32> {
        let freq = 440.0 * 2f64.powf((midi - 69.0) / 12.0);
        let n = (seconds * f64::from(RATE)) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(RATE);
                let fundamental = (2.0 * std::f64::consts::PI * freq * t).sin();
                let octave = 0.4 * (2.0 * std::f64::consts::PI * freq * 2.0 * t).sin();
                let fifth = 0.2 * (2.0 * std::f64::consts::PI * freq * 3.0 * t).sin();
                (amplitude * (fundamental + octave + fifth)) as f32
            })
            .collect()
    }

    /// Mix several notes into one chord.
    fn chord(midis: &[f64], seconds: f64) -> Vec<f32> {
        let voices: Vec<Vec<f32>> = midis.iter().map(|m| note(*m, seconds, 0.3)).collect();
        let len = voices[0].len();
        (0..len)
            .map(|i| voices.iter().map(|v| v[i]).sum::<f32>())
            .collect()
    }

    #[test]
    fn a_signal_shorter_than_the_window_gives_no_chroma() {
        let c = chroma(&vec![0.1f32; 100], RATE, ChromaOptions::default());
        assert!(c.is_empty());
        assert_eq!(c.frames(), 0);
    }

    #[test]
    fn every_frame_is_normalized_to_sum_to_one() {
        let signal = chord(&[60.0, 64.0, 67.0], 2.0);
        let c = chroma(&signal, RATE, ChromaOptions::default());
        assert!(c.frames() > 0);
        for frame in 0..c.frames() {
            let total: f64 = c.frame(frame).iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-9,
                "frame {frame} summed to {total}"
            );
        }
    }

    #[test]
    fn a_single_note_lights_up_its_own_pitch_class() {
        // MIDI 60 is C, pitch class 0.
        let signal = note(60.0, 2.0, 0.5);
        let c = chroma(&signal, RATE, ChromaOptions::default());
        let mean = c.mean();
        let brightest = mean
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(brightest, 0, "C should dominate, saw {mean:?}");
    }

    #[test]
    fn a_transposed_note_moves_the_bright_pitch_class() {
        // MIDI 62 is D, pitch class 2.
        let signal = note(62.0, 2.0, 0.5);
        let c = chroma(&signal, RATE, ChromaOptions::default());
        let mean = c.mean();
        let brightest = mean
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(brightest, 2, "D should dominate, saw {mean:?}");
    }

    #[test]
    fn silence_produces_flat_chroma_without_dividing_by_zero() {
        let c = chroma(
            &vec![0.0f32; RATE as usize * 2],
            RATE,
            ChromaOptions::default(),
        );
        assert!(c.frames() > 0);
        for frame in 0..c.frames() {
            assert!(c.frame(frame).iter().all(|v| *v == 0.0));
        }
        assert!(c.mean().iter().all(|v| *v == 0.0));
    }

    #[test]
    fn a_c_major_triad_is_detected_as_c_major() {
        // C4 E4 G4.
        let signal = chord(&[60.0, 64.0, 67.0], 4.0);
        let c = chroma(&signal, RATE, ChromaOptions::default());
        let key = overall_key(&c).unwrap();
        assert_eq!(key.mode(), Mode::Major, "detected {}", key.classical());
        assert_eq!(key.pitch_class(), 0, "detected {}", key.classical());
    }

    /// The case Python cannot express: it scores only major templates and then
    /// forces one mode across the whole track.
    #[test]
    fn an_a_minor_triad_is_detected_as_a_minor_not_its_relative_major() {
        // A3 C4 E4.
        let signal = chord(&[57.0, 60.0, 64.0], 4.0);
        let c = chroma(&signal, RATE, ChromaOptions::default());
        let key = overall_key(&c).unwrap();
        assert_eq!(key.mode(), Mode::Minor, "detected {}", key.classical());
        assert_eq!(key.pitch_class(), 9, "detected {}", key.classical());
    }

    #[test]
    fn the_key_path_has_one_entry_per_frame() {
        let signal = chord(&[60.0, 64.0, 67.0], 3.0);
        let c = chroma(&signal, RATE, ChromaOptions::default());
        let path = decode_key_path(&c, KeyOptions::default());
        assert_eq!(path.len(), c.frames());
    }

    #[test]
    fn a_steady_piece_decodes_to_one_key_throughout() {
        let signal = chord(&[60.0, 64.0, 67.0], 6.0);
        let c = chroma(&signal, RATE, ChromaOptions::default());
        let path = decode_key_path(&c, KeyOptions::default());
        let first = path[0];
        assert!(
            path.iter().all(|k| *k == first),
            "a steady chord should not wander between keys"
        );
    }

    #[test]
    fn segments_cover_the_track_and_do_not_overlap() {
        let signal = chord(&[60.0, 64.0, 67.0], 6.0);
        let c = chroma(&signal, RATE, ChromaOptions::default());
        let path = decode_key_path(&c, KeyOptions::default());
        let segments = segments_from_path(&c, &path, 6.0);
        assert!(!segments.is_empty());
        for pair in segments.windows(2) {
            assert!(pair[0].end <= pair[1].start + 1e-9);
        }
        assert!((segments.last().unwrap().end - 6.0).abs() < 0.2);
        assert!(segments.iter().all(|s| s.duration() > 0.0));
    }

    #[test]
    fn an_empty_chromagram_yields_no_path_no_segments_and_no_key() {
        let empty = Chroma {
            values: Vec::new(),
            frames: 0,
            times: Vec::new(),
        };
        assert!(decode_key_path(&empty, KeyOptions::default()).is_empty());
        assert!(segments_from_path(&empty, &[], 10.0).is_empty());
        assert!(overall_key(&empty).is_none());
    }

    #[test]
    fn beat_synchronous_chroma_has_one_row_per_beat_interval() {
        let signal = chord(&[60.0, 64.0, 67.0], 4.0);
        let c = chroma(&signal, RATE, ChromaOptions::default());
        let beats: Vec<f64> = (0..9).map(|i| i as f64 * 0.5).collect();
        let synced = beat_synchronous(&c, &beats);
        assert_eq!(synced.frames(), beats.len() - 1);
        assert!((synced.time(0) - beats[0]).abs() < 1e-9);
    }

    #[test]
    fn beat_synchronous_chroma_falls_back_when_there_are_too_few_beats() {
        let signal = chord(&[60.0, 64.0, 67.0], 2.0);
        let c = chroma(&signal, RATE, ChromaOptions::default());
        assert_eq!(beat_synchronous(&c, &[]).frames(), c.frames());
        assert_eq!(beat_synchronous(&c, &[1.0]).frames(), c.frames());
    }

    #[test]
    fn staying_put_is_cheaper_than_moving_and_near_beats_far() {
        let options = KeyOptions::default();
        let stay = transition_log(0, 0, options);
        // 8B -> 8A is the relative minor, a near move; 8B -> 2A is not.
        let near_target = usize::from(Key::from_index(0).relative().index());
        let near = transition_log(0, near_target, options);
        let far = transition_log(0, 6, options);
        assert!(stay > near, "staying should be preferred");
        assert!(near > far, "a harmonic neighbour should beat a distant key");
    }

    #[test]
    fn every_key_scores_against_a_chroma_frame() {
        let frame = [1.0f64 / 12.0; 12];
        let scores = frame_log_likelihoods(&frame);
        assert_eq!(scores.len(), 24);
        assert!(scores.iter().all(|s| s.is_finite()));
    }
}
