//! Onset strength, as `librosa.onset.onset_strength(S=mel_db)`.
//!
//! The alignment is the whole subtlety. librosa differences consecutive frames,
//! which shortens the curve by one, then left-pads by `lag + n_fft / (2 * hop)`
//! zeros and truncates back to the original frame count. With librosa's default
//! `n_fft=2048` and `hop_length=512` that is three leading zeros, so the value
//! at frame `i` describes the step from frame `i - 3` to frame `i - 2`.

use super::matrix::Mat;

/// Zeros at the start of the curve, from librosa's `lag` plus centring shift.
const PAD_FRAMES: usize = 3;

/// Mean positive frame-to-frame increase across the mel bands.
pub fn onset_strength(mel_db: &Mat) -> Vec<f64> {
    let frames = mel_db.cols();
    let mut out = vec![0.0; frames];
    if frames < 2 || mel_db.rows() == 0 {
        return out;
    }
    for (target, slot) in out.iter_mut().enumerate().skip(PAD_FRAMES) {
        let current = target - PAD_FRAMES + 1;
        let previous = target - PAD_FRAMES;
        let sum: f64 = (0..mel_db.rows())
            .map(|band| (mel_db.get(band, current) - mel_db.get(band, previous)).max(0.0))
            .sum();
        *slot = sum / mel_db.rows() as f64;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_three_frames_are_always_zero() {
        let mut mel = Mat::zeros(4, 10);
        for band in 0..4 {
            for frame in 0..10 {
                mel.set(band, frame, frame as f64);
            }
        }
        let onset = onset_strength(&mel);
        assert_eq!(&onset[..3], &[0.0, 0.0, 0.0]);
        assert!((onset[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_falling_spectrum_produces_no_onset() {
        let mut mel = Mat::zeros(4, 8);
        for band in 0..4 {
            for frame in 0..8 {
                mel.set(band, frame, -(frame as f64));
            }
        }
        assert!(onset_strength(&mel).iter().all(|v| *v == 0.0));
    }
}
