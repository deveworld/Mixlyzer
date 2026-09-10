//! Harmonic/percussive source separation, as `librosa.decompose.hpss`.
//!
//! Harmonic content is smooth along time and spiky along frequency; percussive
//! content is the other way round. Median-filtering the magnitude spectrogram
//! in each direction gives two rough estimates, and a soft Wiener-style mask
//! built from them splits the original.

use super::matrix::{median, Mat};

/// librosa's default median-filter length, in frames and in bins.
const KERNEL: usize = 31;

/// Reflect an index into `0..len`, matching `scipy.ndimage`'s `mode="reflect"`.
///
/// That mode is `(d c b a | a b c d | d c b a)`: the edge sample is repeated,
/// unlike NumPy's `reflect`, which does not.
fn reflect_index(i: isize, len: usize) -> usize {
    let n = len as isize;
    if n == 1 {
        return 0;
    }
    let period = 2 * n;
    let mut k = i.rem_euclid(period);
    if k >= n {
        k = period - 1 - k;
    }
    k as usize
}

/// Median filter of length [`KERNEL`] along the time axis of each row.
fn median_filter_time(input: &Mat) -> Mat {
    let half = (KERNEL / 2) as isize;
    let mut out = Mat::zeros(input.rows(), input.cols());
    let mut window = vec![0.0; KERNEL];
    for r in 0..input.rows() {
        let row = input.row(r);
        for c in 0..input.cols() {
            for (k, slot) in window.iter_mut().enumerate() {
                *slot = row[reflect_index(c as isize - half + k as isize, input.cols())];
            }
            out.set(r, c, median(&window));
        }
    }
    out
}

/// Median filter of length [`KERNEL`] along the frequency axis of each column.
fn median_filter_frequency(input: &Mat) -> Mat {
    let half = (KERNEL / 2) as isize;
    let mut out = Mat::zeros(input.rows(), input.cols());
    let mut window = vec![0.0; KERNEL];
    for c in 0..input.cols() {
        for r in 0..input.rows() {
            for (k, slot) in window.iter_mut().enumerate() {
                *slot = input.get(reflect_index(r as isize - half + k as isize, input.rows()), c);
            }
            out.set(r, c, median(&window));
        }
    }
    out
}

/// `librosa.util.softmask(x, x_ref, power=2, split_zeros=True)`.
///
/// The scaling by `z` before squaring is librosa's, and it is load-bearing:
/// without it the squares of large magnitudes overflow `f32` on loud material.
fn softmask(x: f64, x_ref: f64) -> f64 {
    // librosa runs this on float32 arrays, so the "both cells are zero" test is
    // against the smallest normal float32, not the f64 one.
    const TINY_F32: f64 = 1.175_494_35e-38;
    let z = x.max(x_ref);
    if z < TINY_F32 {
        return 0.5;
    }
    let a = (x / z) * (x / z);
    let b = (x_ref / z) * (x_ref / z);
    a / (a + b)
}

/// Split a magnitude spectrogram into `(harmonic, percussive)` magnitudes.
pub fn hpss(magnitude: &Mat) -> (Mat, Mat) {
    let harm = median_filter_time(magnitude);
    let perc = median_filter_frequency(magnitude);
    let mut harmonic = Mat::zeros(magnitude.rows(), magnitude.cols());
    let mut percussive = Mat::zeros(magnitude.rows(), magnitude.cols());
    for (i, value) in magnitude.as_slice().iter().enumerate() {
        let h = harm.as_slice()[i];
        let p = perc.as_slice()[i];
        harmonic.as_mut_slice()[i] = value * softmask(h, p);
        percussive.as_mut_slice()[i] = value * softmask(p, h);
    }
    (harmonic, percussive)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_repeats_the_edge_sample_like_scipy() {
        assert_eq!(reflect_index(-1, 4), 0);
        assert_eq!(reflect_index(-2, 4), 1);
        assert_eq!(reflect_index(4, 4), 3);
        assert_eq!(reflect_index(5, 4), 2);
        assert_eq!(reflect_index(2, 4), 2);
    }

    #[test]
    fn the_two_masks_sum_to_the_original_magnitude() {
        let mut mat = Mat::zeros(40, 40);
        for r in 0..40 {
            for c in 0..40 {
                mat.set(r, c, ((r * 7 + c * 3) % 11) as f64);
            }
        }
        let (h, p) = hpss(&mat);
        for i in 0..mat.as_slice().len() {
            let sum = h.as_slice()[i] + p.as_slice()[i];
            assert!((sum - mat.as_slice()[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn a_steady_tone_lands_almost_entirely_in_the_harmonic_part() {
        let mut mat = Mat::zeros(64, 64);
        for c in 0..64 {
            mat.set(20, c, 1.0);
        }
        let (h, p) = hpss(&mat);
        assert!(h.get(20, 32) > 0.9, "got {}", h.get(20, 32));
        assert!(p.get(20, 32) < 0.1);
    }
}
