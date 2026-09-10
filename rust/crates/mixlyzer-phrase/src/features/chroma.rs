//! Chromagram, tuning estimation and tonnetz, as librosa computes them.
//!
//! `chroma_stft` is the one feature here with hidden state: with no `tuning`
//! argument librosa first runs `estimate_tuning`, which peak-picks the
//! spectrogram with parabolic interpolation and histograms the fractional
//! semitone offsets. Skipping that and assuming A=440 shifts every chroma bin
//! whenever the material is not exactly in concert pitch.

use super::matrix::{median, Mat};
use super::stft::fft_frequencies;

const N_CHROMA: usize = 12;

/// Octaves above `A440 / 16`, i.e. `librosa.hz_to_octs`.
fn hz_to_octs(hz: f64, tuning: f64) -> f64 {
    let a440 = 440.0 * 2.0f64.powf(tuning / N_CHROMA as f64);
    (hz / (a440 / 16.0)).log2()
}

/// Second-order accurate central differences, i.e. `np.gradient` along a slice.
fn gradient(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let mut out = vec![0.0; n];
    out[0] = values[1] - values[0];
    out[n - 1] = values[n - 1] - values[n - 2];
    for i in 1..n - 1 {
        out[i] = 0.5 * (values[i + 1] - values[i - 1]);
    }
    out
}

/// Sub-bin peak offset from fitting a parabola through each triple.
///
/// Matches librosa's `_pi_stencil`, including its guard: when the curvature
/// `a` is no larger than the slope `b` the fit is not a peak, and the offset
/// is reported as zero rather than extrapolating wildly.
fn parabolic_shift(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![0.0; n];
    for i in 1..n.saturating_sub(1) {
        let a = values[i + 1] + values[i - 1] - 2.0 * values[i];
        let b = 0.5 * (values[i + 1] - values[i - 1]);
        out[i] = if b.abs() >= a.abs() { 0.0 } else { -b / a };
    }
    out
}

/// `librosa.util.localmax` along a slice: strictly above the left neighbour,
/// at least equal to the right one. Index 0 is never a local maximum.
fn localmax(values: &[f64]) -> Vec<bool> {
    let n = values.len();
    let mut out = vec![false; n];
    for i in 1..n.saturating_sub(1) {
        out[i] = values[i] > values[i - 1] && values[i] >= values[i + 1];
    }
    if n >= 2 {
        out[n - 1] = values[n - 1] > values[n - 2];
    }
    out
}

/// Deviation from equal temperament, in fractions of a semitone.
///
/// This is `librosa.estimate_tuning(S=power)`: peak-pick each frame, keep the
/// peaks at or above the median magnitude, and take the mode of their
/// fractional semitone offsets over a 100-bin histogram.
pub fn estimate_tuning(power: &Mat, sample_rate: f64, n_fft: usize) -> f64 {
    const FMIN: f64 = 150.0;
    const FMAX: f64 = 4000.0;
    const THRESHOLD: f64 = 0.1;

    let freqs = fft_frequencies(sample_rate, n_fft);
    let fmax = FMAX.min(sample_rate / 2.0);

    let mut pitches: Vec<f64> = Vec::new();
    let mut mags: Vec<f64> = Vec::new();
    for frame in 0..power.cols() {
        let column = power.column(frame);
        let grad = gradient(&column);
        let shift = parabolic_shift(&column);
        let peak = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ref_value = THRESHOLD * peak;
        let gated: Vec<f64> = column
            .iter()
            .map(|v| if *v > ref_value { *v } else { 0.0 })
            .collect();
        let maxima = localmax(&gated);
        for bin in 0..column.len() {
            if !maxima[bin] || freqs[bin] < FMIN || freqs[bin] >= fmax {
                continue;
            }
            pitches.push((bin as f64 + shift[bin]) * sample_rate / n_fft as f64);
            mags.push(column[bin] + 0.5 * grad[bin] * shift[bin]);
        }
    }

    if pitches.is_empty() {
        return 0.0;
    }
    // Only peaks at or above the median magnitude vote, so a handful of loud
    // partials do not get outvoted by the noise floor.
    let threshold = median(&mags);
    let voters: Vec<f64> = pitches
        .iter()
        .zip(&mags)
        .filter(|(pitch, mag)| **pitch > 0.0 && **mag >= threshold)
        .map(|(pitch, _)| *pitch)
        .collect();
    if voters.is_empty() {
        return 0.0;
    }
    pitch_tuning(&voters)
}

/// Mode of the fractional-semitone residuals, on librosa's 0.01 grid.
fn pitch_tuning(frequencies: &[f64]) -> f64 {
    const BINS: usize = 100;
    let mut counts = [0usize; BINS];
    for freq in frequencies {
        let mut residual = (N_CHROMA as f64 * hz_to_octs(*freq, 0.0)).rem_euclid(1.0);
        if residual >= 0.5 {
            residual -= 1.0;
        }
        // np.histogram over linspace(-0.5, 0.5, 101): the last bin is closed.
        let position = (residual + 0.5) * BINS as f64;
        let index = (position.floor() as isize).clamp(0, BINS as isize - 1) as usize;
        counts[index] += 1;
    }
    let best = counts
        .iter()
        .enumerate()
        .max_by_key(|(index, count)| (**count, std::cmp::Reverse(*index)))
        .map(|(index, _)| index)
        .unwrap_or(0);
    -0.5 + best as f64 / BINS as f64
}

/// `librosa.filters.chroma`: `(12, n_fft/2 + 1)` Gaussian pitch-class weights.
pub fn chroma_filterbank(sample_rate: f64, n_fft: usize, tuning: f64) -> Mat {
    // Frequencies run over the *full* FFT here, not just the real half; the
    // basis is truncated only at the very end.
    let mut frqbins = Vec::with_capacity(n_fft);
    let first = N_CHROMA as f64 * hz_to_octs(sample_rate / n_fft as f64, tuning);
    frqbins.push(first - 1.5 * N_CHROMA as f64);
    for k in 1..n_fft {
        frqbins.push(N_CHROMA as f64 * hz_to_octs(k as f64 * sample_rate / n_fft as f64, tuning));
    }

    let mut binwidth = Vec::with_capacity(n_fft);
    for k in 0..n_fft - 1 {
        binwidth.push((frqbins[k + 1] - frqbins[k]).max(1.0));
    }
    binwidth.push(1.0);

    let half = (N_CHROMA as f64 / 2.0).round();
    let mut wts = Mat::zeros(N_CHROMA, n_fft);
    for c in 0..N_CHROMA {
        for k in 0..n_fft {
            let d = (frqbins[k] - c as f64 + half + 10.0 * N_CHROMA as f64)
                .rem_euclid(N_CHROMA as f64)
                - half;
            let x = 2.0 * d / binwidth[k];
            wts.set(c, k, (-0.5 * x * x).exp());
        }
    }

    // L2-normalise each frequency column, then taper by an octave-wide
    // Gaussian centred on octave 5, which is what `octwidth=2` does.
    for (k, frq) in frqbins.iter().enumerate() {
        let norm: f64 = (0..N_CHROMA).map(|c| wts.get(c, k).powi(2)).sum::<f64>().sqrt();
        let norm = if norm < f64::from(f32::MIN_POSITIVE) { 1.0 } else { norm };
        let taper = {
            let z = (frq / N_CHROMA as f64 - 5.0) / 2.0;
            (-0.5 * z * z).exp()
        };
        for c in 0..N_CHROMA {
            wts.set(c, k, wts.get(c, k) / norm * taper);
        }
    }

    // `base_c=True`: roll so that row 0 is C rather than A.
    let bins = n_fft / 2 + 1;
    let mut out = Mat::zeros(N_CHROMA, bins);
    for c in 0..N_CHROMA {
        let source = (c + 3) % N_CHROMA;
        for k in 0..bins {
            out.set(c, k, f64::from(wts.get(source, k) as f32));
        }
    }
    out
}

/// Normalise each column to unit `norm`-norm, as `librosa.util.normalize`.
fn normalize_columns(mat: &mut Mat, norm: f64) {
    for c in 0..mat.cols() {
        let length: f64 = (0..mat.rows())
            .map(|r| mat.get(r, c).abs().powf(norm))
            .sum::<f64>()
            .powf(1.0 / norm);
        // librosa leaves a below-threshold column untouched rather than
        // dividing by something close to zero.
        let length = if length < f64::from(f32::MIN_POSITIVE) { 1.0 } else { length };
        for r in 0..mat.rows() {
            mat.set(r, c, mat.get(r, c) / length);
        }
    }
}

/// `librosa.feature.chroma_stft(S=power, norm=2)`.
pub fn chroma_stft(power: &Mat, sample_rate: f64, n_fft: usize, tuning: f64) -> Mat {
    let basis = chroma_filterbank(sample_rate, n_fft, tuning);
    let mut out = Mat::zeros(N_CHROMA, power.cols());
    for c in 0..N_CHROMA {
        let weights = basis.row(c);
        for frame in 0..power.cols() {
            let mut acc = 0.0;
            for (bin, weight) in weights.iter().enumerate() {
                acc += weight * power.get(bin, frame);
            }
            out.set(c, frame, acc);
        }
    }
    normalize_columns(&mut out, 2.0);
    out
}

/// `librosa.feature.tonnetz(chroma=...)`: six tonal-centroid dimensions.
pub fn tonnetz(chroma: &Mat) -> Mat {
    let scale = [7.0 / 6.0, 7.0 / 6.0, 1.5, 1.5, 2.0 / 3.0, 2.0 / 3.0];
    let radius = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5];
    let rows = chroma.rows();
    let mut phi = Mat::zeros(6, rows);
    for p in 0..6 {
        for c in 0..rows {
            let dim = 12.0 * c as f64 / rows as f64;
            let mut v = scale[p] * dim;
            if p % 2 == 0 {
                v -= 0.5;
            }
            phi.set(p, c, radius[p] * (std::f64::consts::PI * v).cos());
        }
    }

    let mut normalized = chroma.clone();
    normalize_columns(&mut normalized, 1.0);

    let mut out = Mat::zeros(6, chroma.cols());
    for p in 0..6 {
        for frame in 0..chroma.cols() {
            let mut acc = 0.0;
            for c in 0..rows {
                acc += phi.get(p, c) * normalized.get(c, frame);
            }
            out.set(p, frame, acc);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_concert_pitch_tone_reports_near_zero_detuning() {
        // A440 sits between FFT bins, so the parabolic peak fit on a
        // Hann-windowed magnitude lands a couple of cents off; anything
        // larger would mean the octave mapping itself is wrong.
        let n_fft = 2048;
        let sr = 22_050.0;
        let signal: Vec<f32> = (0..22_050)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sr).sin() as f32)
            .collect();
        let mag = super::super::stft::stft_magnitude(&signal, n_fft, 512);
        let tuning = estimate_tuning(&mag.map(|v| v * v), sr, n_fft);
        assert!(tuning.abs() <= 0.05, "tuning was {tuning}");
    }

    #[test]
    fn chroma_columns_are_unit_length() {
        let mut power = Mat::zeros(1025, 3);
        for frame in 0..3 {
            power.set(41, frame, 1.0);
            power.set(82, frame, 0.5);
        }
        let chroma = chroma_stft(&power, 22_050.0, 2048, 0.0);
        for frame in 0..3 {
            let norm: f64 = (0..12).map(|c| chroma.get(c, frame).powi(2)).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn tonnetz_of_a_flat_chroma_is_near_zero() {
        let mut chroma = Mat::zeros(12, 1);
        for c in 0..12 {
            chroma.set(c, 0, 1.0);
        }
        let ton = tonnetz(&chroma);
        for p in 0..6 {
            assert!(ton.get(p, 0).abs() < 1e-12, "dimension {p}");
        }
    }
}
