//! Short-time Fourier transform, matching `librosa.stft` bit-for-bit in method.
//!
//! The model was trained on librosa's numbers, so the conventions here are not
//! free choices: a *periodic* Hann window (what `scipy.signal.get_window`
//! returns with `fftbins=True`), centring by zero-padding `n_fft / 2` samples
//! on both ends (librosa 0.10 changed the default `pad_mode` from `reflect` to
//! `constant`), and a frame count of `1 + len(y) / hop`.

use std::sync::Arc;

use rustfft::{num_complex::Complex, Fft, FftPlanner};

use super::matrix::Mat;

/// Periodic Hann window of length `n`.
///
/// Periodic, not symmetric: `w[i] = 0.5 - 0.5 cos(2 pi i / n)`. The symmetric
/// variant divides by `n - 1` instead and would leave a slow, systematic tilt
/// across every spectrum.
pub fn hann_periodic(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
        .collect()
}

/// FFT bin centre frequencies, i.e. `librosa.fft_frequencies`.
pub fn fft_frequencies(sample_rate: f64, n_fft: usize) -> Vec<f64> {
    let bins = n_fft / 2 + 1;
    (0..bins)
        .map(|k| k as f64 * sample_rate / n_fft as f64)
        .collect()
}

/// Frame start times in seconds, i.e. `librosa.frames_to_time`.
pub fn frames_to_time(n_frames: usize, sample_rate: f64, hop: usize) -> Vec<f64> {
    (0..n_frames)
        .map(|i| i as f64 * hop as f64 / sample_rate)
        .collect()
}

/// Number of STFT frames librosa produces for `len` samples with centring on.
pub fn n_frames(len: usize, hop: usize) -> usize {
    1 + len / hop
}

/// Magnitude spectrogram `|stft(signal)|`, shaped `(n_fft/2 + 1, frames)`.
///
/// Only the magnitude is returned because nothing downstream of here uses
/// phase; the phrase features are all built from `|S|` or `|S|^2`.
pub fn stft_magnitude(signal: &[f32], n_fft: usize, hop: usize) -> Mat {
    let pad = n_fft / 2;
    let mut padded = vec![0.0f64; signal.len() + 2 * pad];
    for (slot, sample) in padded[pad..pad + signal.len()].iter_mut().zip(signal) {
        *slot = f64::from(*sample);
    }

    let frames = n_frames(signal.len(), hop);
    let bins = n_fft / 2 + 1;
    let window = hann_periodic(n_fft);

    let mut planner = FftPlanner::<f64>::new();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(n_fft);
    let mut scratch = vec![Complex::new(0.0, 0.0); n_fft];

    let mut out = Mat::zeros(bins, frames);
    for frame in 0..frames {
        let start = frame * hop;
        for (i, slot) in scratch.iter_mut().enumerate() {
            let sample = padded.get(start + i).copied().unwrap_or(0.0);
            *slot = Complex::new(sample * window[i], 0.0);
        }
        fft.process(&mut scratch);
        for (bin, value) in scratch.iter().take(bins).enumerate() {
            out.set(bin, frame, value.norm());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_is_periodic_so_the_last_sample_is_not_zero() {
        let w = hann_periodic(8);
        assert!((w[0]).abs() < 1e-15);
        assert!(w[7] > 0.1, "a symmetric window would end at exactly 0");
        assert!((w[4] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn frame_count_matches_librosa_centring() {
        assert_eq!(n_frames(22050, 512), 1 + 22050 / 512);
        assert_eq!(n_frames(0, 512), 1);
    }

    #[test]
    fn a_pure_tone_peaks_in_its_own_bin() {
        let sr = 22050.0;
        let freq = 22050.0 / 2048.0 * 100.0;
        let signal: Vec<f32> = (0..22050)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr).sin() as f32)
            .collect();
        let mag = stft_magnitude(&signal, 2048, 512);
        let middle = mag.cols() / 2;
        let peak = (0..mag.rows())
            .max_by(|a, b| mag.get(*a, middle).total_cmp(&mag.get(*b, middle)))
            .unwrap();
        assert_eq!(peak, 100);
    }
}
