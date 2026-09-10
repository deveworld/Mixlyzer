//! Scalar spectral descriptors and spectral contrast, as librosa computes them.

use super::matrix::Mat;
use super::mel::{power_to_db, DbRef};
use super::stft::fft_frequencies;

/// Round half to even, i.e. `np.rint`. Rust's `f64::round` goes half away from
/// zero, which picks a different bin count on exact `.5` band sizes.
fn rint(value: f64) -> f64 {
    let nearest = value.round();
    if (value - value.trunc()).abs() == 0.5 && nearest % 2.0 != 0.0 {
        nearest - value.signum()
    } else {
        nearest
    }
}

/// Column sums, used by the L1 normalisation the shape descriptors share.
fn column_l1(mat: &Mat, frame: usize) -> f64 {
    let sum: f64 = (0..mat.rows()).map(|r| mat.get(r, frame).abs()).sum();
    // Matches `librosa.util.normalize`, which divides by 1 rather than by a
    // denormal when a frame is silent.
    if sum < f64::from(f32::MIN_POSITIVE) {
        1.0
    } else {
        sum
    }
}

/// `librosa.feature.spectral_centroid(S=magnitude)`, in hertz.
pub fn spectral_centroid(magnitude: &Mat, sample_rate: f64, n_fft: usize) -> Vec<f64> {
    let freqs = fft_frequencies(sample_rate, n_fft);
    (0..magnitude.cols())
        .map(|frame| {
            let norm = column_l1(magnitude, frame);
            (0..magnitude.rows())
                .map(|r| freqs[r] * magnitude.get(r, frame) / norm)
                .sum()
        })
        .collect()
}

/// `librosa.feature.spectral_bandwidth(S=magnitude)` with `p=2`, in hertz.
pub fn spectral_bandwidth(magnitude: &Mat, sample_rate: f64, n_fft: usize) -> Vec<f64> {
    let freqs = fft_frequencies(sample_rate, n_fft);
    let centroid = spectral_centroid(magnitude, sample_rate, n_fft);
    (0..magnitude.cols())
        .map(|frame| {
            let norm = column_l1(magnitude, frame);
            let acc: f64 = (0..magnitude.rows())
                .map(|r| {
                    let deviation = (freqs[r] - centroid[frame]).abs();
                    magnitude.get(r, frame) / norm * deviation * deviation
                })
                .sum();
            acc.sqrt()
        })
        .collect()
}

/// `librosa.feature.spectral_flatness(S=power)` with librosa's `power=2`.
///
/// `structure.py` hands this the *power* spectrogram, and librosa then raises
/// it to the power of two again, so the ratio is over `|S|^4`. That is the
/// number the model was trained on, odd as it looks.
pub fn spectral_flatness(power: &Mat) -> Vec<f64> {
    const AMIN: f64 = 1e-10;
    (0..power.cols())
        .map(|frame| {
            let rows = power.rows() as f64;
            let mut log_sum = 0.0;
            let mut arithmetic = 0.0;
            for r in 0..power.rows() {
                let value = power.get(r, frame);
                let thresholded = (value * value).max(AMIN);
                log_sum += thresholded.ln();
                arithmetic += thresholded;
            }
            (log_sum / rows).exp() / (arithmetic / rows)
        })
        .collect()
}

/// `librosa.feature.spectral_rolloff(S=magnitude)`, in hertz.
pub fn spectral_rolloff(
    magnitude: &Mat,
    sample_rate: f64,
    n_fft: usize,
    roll_percent: f64,
) -> Vec<f64> {
    let freqs = fft_frequencies(sample_rate, n_fft);
    (0..magnitude.cols())
        .map(|frame| {
            let total: f64 = (0..magnitude.rows()).map(|r| magnitude.get(r, frame)).sum();
            let threshold = roll_percent * total;
            let mut running = 0.0;
            for (r, freq) in freqs.iter().enumerate().take(magnitude.rows()) {
                running += magnitude.get(r, frame);
                if running >= threshold {
                    return *freq;
                }
            }
            freqs[magnitude.rows() - 1]
        })
        .collect()
}

/// `librosa.feature.rms(S=magnitude, frame_length=n_fft)`.
///
/// The DC and Nyquist bins are halved because they appear once in the
/// two-sided spectrum while every other bin appears twice.
pub fn rms_from_spectrum(magnitude: &Mat, frame_length: usize) -> Vec<f64> {
    let last = magnitude.rows() - 1;
    (0..magnitude.cols())
        .map(|frame| {
            let mut acc = 0.0;
            for r in 0..magnitude.rows() {
                let value = magnitude.get(r, frame);
                let mut squared = value * value;
                if r == 0 || (r == last && frame_length % 2 == 0) {
                    squared *= 0.5;
                }
                acc += squared;
            }
            (2.0 * acc / (frame_length as f64 * frame_length as f64)).sqrt()
        })
        .collect()
}

/// `librosa.feature.rms(y=signal, frame_length, hop_length, center=True)`.
pub fn rms_from_signal(signal: &[f32], frame_length: usize, hop: usize) -> Vec<f64> {
    let pad = frame_length / 2;
    let mut padded = vec![0.0f64; signal.len() + 2 * pad];
    for (slot, sample) in padded[pad..pad + signal.len()].iter_mut().zip(signal) {
        *slot = f64::from(*sample);
    }
    let frames = super::stft::n_frames(signal.len(), hop);
    (0..frames)
        .map(|frame| {
            let start = frame * hop;
            let mut acc = 0.0;
            for i in 0..frame_length {
                let value = padded.get(start + i).copied().unwrap_or(0.0);
                acc += value * value;
            }
            (acc / frame_length as f64).sqrt()
        })
        .collect()
}

/// `librosa.feature.spectral_contrast(S=magnitude, fmin=200, quantile=0.02)`.
///
/// Each octave band's peak-minus-valley, in decibels. The band edges and the
/// off-by-one bin fixes below are librosa's, quirks included: every band after
/// the first reaches one bin below its nominal start, the last band runs to
/// Nyquist, and every band but the last drops its top bin.
pub fn spectral_contrast(magnitude: &Mat, sample_rate: f64, n_fft: usize, n_bands: usize) -> Mat {
    const FMIN: f64 = 200.0;
    const QUANTILE: f64 = 0.02;
    let freqs = fft_frequencies(sample_rate, n_fft);

    let mut octa = vec![0.0; n_bands + 2];
    for k in 0..=n_bands {
        octa[k + 1] = FMIN * 2.0f64.powi(k as i32);
    }

    let mut valley = Mat::zeros(n_bands + 1, magnitude.cols());
    let mut peak = Mat::zeros(n_bands + 1, magnitude.cols());
    for k in 0..=n_bands {
        let (low, high) = (octa[k], octa[k + 1]);
        let mut band: Vec<usize> = (0..magnitude.rows())
            .filter(|r| freqs[*r] >= low && freqs[*r] <= high)
            .collect();
        if band.is_empty() {
            continue;
        }
        if k > 0 && band[0] > 0 {
            band.insert(0, band[0] - 1);
        }
        if k == n_bands {
            let last = *band.last().unwrap();
            band.extend(last + 1..magnitude.rows());
        }
        // librosa sizes the quantile from the band *before* dropping the top
        // bin, and rounds half to even. Both matter: at four bands the third
        // band lands on exactly 1.5 bins, where the two conventions differ.
        let count = rint(QUANTILE * band.len() as f64).max(1.0) as usize;
        if k < n_bands {
            band.pop();
        }
        if band.is_empty() {
            continue;
        }
        let count = count.min(band.len());

        for frame in 0..magnitude.cols() {
            let mut values: Vec<f64> = band.iter().map(|r| magnitude.get(*r, frame)).collect();
            values.sort_by(f64::total_cmp);
            let low_mean: f64 = values[..count].iter().sum::<f64>() / count as f64;
            let high_mean: f64 = values[values.len() - count..].iter().sum::<f64>() / count as f64;
            valley.set(k, frame, low_mean);
            peak.set(k, frame, high_mean);
        }
    }

    let peak_db = power_to_db(&peak, DbRef::Value(1.0));
    let valley_db = power_to_db(&valley, DbRef::Value(1.0));
    let mut out = Mat::zeros(n_bands + 1, magnitude.cols());
    for i in 0..out.as_slice().len() {
        out.as_mut_slice()[i] = peak_db.as_slice()[i] - valley_db.as_slice()[i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_bin(bin: usize, value: f64) -> Mat {
        let mut mat = Mat::zeros(1025, 1);
        mat.set(bin, 0, value);
        mat
    }

    #[test]
    fn centroid_of_a_single_bin_is_that_bin_frequency() {
        let freqs = fft_frequencies(22_050.0, 2048);
        let centroid = spectral_centroid(&single_bin(100, 3.0), 22_050.0, 2048);
        assert!((centroid[0] - freqs[100]).abs() < 1e-9);
    }

    #[test]
    fn bandwidth_of_a_single_bin_is_zero() {
        let bandwidth = spectral_bandwidth(&single_bin(100, 3.0), 22_050.0, 2048);
        assert!(bandwidth[0].abs() < 1e-9);
    }

    #[test]
    fn rolloff_returns_the_first_bin_reaching_the_energy_share() {
        let mut mat = Mat::zeros(1025, 1);
        for r in 0..10 {
            mat.set(r, 0, 1.0);
        }
        let freqs = fft_frequencies(22_050.0, 2048);
        let rolloff = spectral_rolloff(&mat, 22_050.0, 2048, 0.85);
        assert!((rolloff[0] - freqs[8]).abs() < 1e-9, "85% of 10 units lands in bin 8");
    }

    #[test]
    fn an_empty_frame_does_not_divide_by_zero() {
        let mat = Mat::zeros(1025, 1);
        assert_eq!(spectral_centroid(&mat, 22_050.0, 2048)[0], 0.0);
        assert_eq!(spectral_bandwidth(&mat, 22_050.0, 2048)[0], 0.0);
    }
}
