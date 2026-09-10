//! Mel filterbank, decibel conversion and MFCCs, following librosa exactly.
//!
//! Two details here decide whether the ported features are usable. librosa's
//! default mel scale is *Slaney*, not HTK: linear below 1 kHz, logarithmic
//! above, and each filter is scaled by `2 / (f[i+2] - f[i])` so bands carry
//! roughly equal energy. And `power_to_db` clamps at `top_db` below the
//! **global** maximum of the whole spectrogram, not per frame — a per-frame
//! floor would flatten every quiet bar into the same value.

use super::matrix::Mat;

/// Slaney mel scale: linear at `200/3` Hz per mel up to 1 kHz, log above.
pub fn hz_to_mel(hz: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = 6.4f64.ln() / 27.0;
    if hz >= MIN_LOG_HZ {
        min_log_mel + (hz / MIN_LOG_HZ).ln() / logstep
    } else {
        hz / F_SP
    }
}

pub fn mel_to_hz(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        MIN_LOG_HZ * (logstep * (mel - min_log_mel)).exp()
    } else {
        F_SP * mel
    }
}

/// `librosa.filters.mel`: `(n_mels, n_fft/2 + 1)` triangular weights.
///
/// The weights are rounded to `f32` because librosa builds this basis as a
/// float32 array; keeping the extra precision would put a systematic offset
/// between our mel spectrum and the one the model was trained on.
pub fn mel_filterbank(sample_rate: f64, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Mat {
    let bins = n_fft / 2 + 1;
    let fft_freqs = super::stft::fft_frequencies(sample_rate, n_fft);

    let min_mel = hz_to_mel(fmin);
    let max_mel = hz_to_mel(fmax);
    let mel_f: Vec<f64> = (0..n_mels + 2)
        .map(|i| {
            let mel = min_mel + (max_mel - min_mel) * i as f64 / (n_mels + 1) as f64;
            mel_to_hz(mel)
        })
        .collect();

    let mut weights = Mat::zeros(n_mels, bins);
    for band in 0..n_mels {
        let fdiff_lo = mel_f[band + 1] - mel_f[band];
        let fdiff_hi = mel_f[band + 2] - mel_f[band + 1];
        // Slaney normalisation: constant energy per band rather than per bin.
        let enorm = 2.0 / (mel_f[band + 2] - mel_f[band]);
        for (bin, freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - mel_f[band]) / fdiff_lo;
            let upper = (mel_f[band + 2] - freq) / fdiff_hi;
            let value = lower.min(upper).max(0.0) * enorm;
            weights.set(band, bin, f64::from(value as f32));
        }
    }
    weights
}

/// Apply a mel basis to a power spectrogram, i.e. `librosa.feature.melspectrogram(S=...)`.
pub fn melspectrogram(power: &Mat, basis: &Mat) -> Mat {
    let mut out = Mat::zeros(basis.rows(), power.cols());
    for band in 0..basis.rows() {
        let weights = basis.row(band);
        for frame in 0..power.cols() {
            let mut acc = 0.0;
            for (bin, weight) in weights.iter().enumerate() {
                if *weight != 0.0 {
                    acc += weight * power.get(bin, frame);
                }
            }
            out.set(band, frame, acc);
        }
    }
    out
}

/// Reference level for [`power_to_db`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DbRef {
    /// `ref=np.max`: the loudest cell in the whole spectrogram becomes 0 dB.
    Max,
    /// `ref=<value>`: a fixed reference, librosa's default being 1.0.
    Value(f64),
}

/// `librosa.power_to_db` with `amin=1e-10` and `top_db=80`.
pub fn power_to_db(power: &Mat, reference: DbRef) -> Mat {
    const AMIN: f64 = 1e-10;
    const TOP_DB: f64 = 80.0;
    let ref_value = match reference {
        DbRef::Max => power.max(),
        DbRef::Value(v) => v.abs(),
    };
    let offset = 10.0 * ref_value.max(AMIN).log10();
    let mut out = power.map(|v| 10.0 * v.max(AMIN).log10() - offset);
    // The floor is global, so it has to be found after the whole array is in
    // decibels rather than while mapping.
    let floor = out.max() - TOP_DB;
    for value in out.as_mut_slice() {
        *value = value.max(floor);
    }
    out
}

/// Type-II DCT with orthonormal scaling along the row axis, then truncated.
///
/// This is `scipy.fft.dct(S, axis=-2, type=2, norm='ortho')[:n_out]`, which is
/// what `librosa.feature.mfcc` calls when handed a decibel mel spectrogram.
pub fn dct_ortho_rows(input: &Mat, n_out: usize) -> Mat {
    let n = input.rows();
    let n_out = n_out.min(n);
    let mut out = Mat::zeros(n_out, input.cols());
    let scale0 = (1.0 / n as f64).sqrt();
    let scale = (2.0 / n as f64).sqrt();
    for k in 0..n_out {
        let factor = if k == 0 { scale0 } else { scale };
        for frame in 0..input.cols() {
            let mut acc = 0.0;
            for (row, _) in (0..n).enumerate() {
                let angle = std::f64::consts::PI * (2.0 * row as f64 + 1.0) * k as f64
                    / (2.0 * n as f64);
                acc += input.get(row, frame) * angle.cos();
            }
            out.set(k, frame, factor * acc);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_scale_is_linear_below_a_kilohertz_and_round_trips() {
        assert!((hz_to_mel(500.0) - 500.0 / (200.0 / 3.0)).abs() < 1e-12);
        for hz in [0.0, 30.0, 440.0, 999.0, 1000.0, 4000.0, 11025.0] {
            assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < 1e-9, "{hz}");
        }
    }

    #[test]
    fn power_to_db_floors_against_the_global_maximum() {
        let mat = Mat::from_rows(vec![vec![1.0, 1e-30], vec![1e-30, 1e-30]]);
        let db = power_to_db(&mat, DbRef::Max);
        assert!((db.get(0, 0) - 0.0).abs() < 1e-12);
        assert!((db.get(0, 1) - -80.0).abs() < 1e-12, "clamped 80 dB below the peak");
    }

    #[test]
    fn dct_of_a_constant_column_puts_everything_in_the_first_coefficient() {
        let mat = Mat::from_rows(vec![vec![2.0], vec![2.0], vec![2.0], vec![2.0]]);
        let out = dct_ortho_rows(&mat, 4);
        assert!((out.get(0, 0) - 4.0).abs() < 1e-12);
        for k in 1..4 {
            assert!(out.get(k, 0).abs() < 1e-12);
        }
    }
}
