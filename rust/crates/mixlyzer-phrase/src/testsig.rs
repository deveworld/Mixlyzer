//! Deterministic synthetic audio, shared by the tests and the parity harness.
//!
//! The parity harness has to feed *exactly* the same samples to Rust and to
//! librosa. Rather than trust two random number generators to agree, the Rust
//! side generates the signal and ships it to Python inside the dump, and this
//! module is the single definition of what it contains.

#![doc(hidden)]

/// A 32-bit xorshift, so the "noise" is identical on every platform and run.
#[derive(Debug, Clone)]
pub struct Xorshift(u32);

impl Xorshift {
    pub fn new(seed: u32) -> Self {
        Self(if seed == 0 { 0x1234_5678 } else { seed })
    }

    /// Next value in `[-1, 1)`.
    pub fn next_unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        f64::from(self.0) / f64::from(u32::MAX) * 2.0 - 1.0
    }
}

/// A short signal with bass, a chord, hats and a level change part-way through.
///
/// It exercises every feature family: harmonic content for chroma and tonnetz,
/// broadband transients for onset strength and percussive HPSS, and a loudness
/// step so the standardised features are not all zero.
pub fn parity_signal(sample_rate: u32, seconds: f64) -> Vec<f32> {
    let n = (seconds * f64::from(sample_rate)) as usize;
    let mut rng = Xorshift::new(0x9E37_79B9);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / f64::from(sample_rate);
        let two_pi = 2.0 * std::f64::consts::PI;
        let bass = 0.45 * (two_pi * 55.0 * t).sin();
        let chord = 0.18 * (two_pi * 220.0 * t).sin()
            + 0.14 * (two_pi * 277.18 * t).sin()
            + 0.11 * (two_pi * 329.63 * t).sin();
        // A hat every eighth of a second, decaying over 30 ms.
        let since_hat = (t * 8.0).fract() / 8.0;
        let hat = 0.30 * (-since_hat / 0.03).exp() * rng.next_unit();
        // Kick transient on every half second.
        let since_kick = (t * 2.0).fract() / 2.0;
        let kick = 0.5 * (-since_kick / 0.08).exp() * (two_pi * 60.0 * since_kick).sin();
        let level = if t > seconds * 0.5 { 1.0 } else { 0.45 };
        out.push(((bass + chord + hat + kick) * level * 0.5) as f32);
    }
    out
}

/// A longer signal with an obvious structural change every `section` seconds.
///
/// Used by the end-to-end test: the detector should find boundaries somewhere
/// near the section edges rather than scattering them at random.
pub fn structured_signal(sample_rate: u32, seconds: f64, section: f64) -> Vec<f32> {
    let n = (seconds * f64::from(sample_rate)) as usize;
    let mut rng = Xorshift::new(0x0BAD_C0DE);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / f64::from(sample_rate);
        let two_pi = 2.0 * std::f64::consts::PI;
        let section_index = (t / section) as usize;
        let loud = section_index % 2 == 1;
        let root = if loud { 110.0 } else { 82.41 };
        let bass = 0.5 * (two_pi * root * t).sin();
        let chord = if loud {
            0.25 * (two_pi * root * 4.0 * t).sin() + 0.2 * (two_pi * root * 5.0 * t).sin()
        } else {
            0.1 * (two_pi * root * 3.0 * t).sin()
        };
        let since_hat = (t * 8.0).fract() / 8.0;
        let hat_gain = if loud { 0.35 } else { 0.08 };
        let hat = hat_gain * (-since_hat / 0.02).exp() * rng.next_unit();
        let since_kick = (t * 2.0).fract() / 2.0;
        let kick = 0.6 * (-since_kick / 0.06).exp() * (two_pi * 55.0 * since_kick).sin();
        let level = if loud { 1.0 } else { 0.4 };
        out.push(((bass + chord + hat + kick) * level * 0.45) as f32);
    }
    out
}

/// A pseudo-random feature matrix for exercising the model runtime.
///
/// Every 97th cell is NaN so the missing-value branch of the tree walk is
/// covered too. `rust/parity/gbm_reference.py` reimplements this exactly, and
/// the fixture it writes carries a checksum so the two cannot drift apart
/// unnoticed.
pub fn random_matrix(seed: u32, rows: usize, cols: usize) -> Vec<Vec<f64>> {
    let mut rng = Xorshift::new(seed);
    let mut cell = 0usize;
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    let value = 3.0 * rng.next_unit();
                    cell += 1;
                    if cell % 97 == 0 {
                        f64::NAN
                    } else {
                        value
                    }
                })
                .collect()
        })
        .collect()
}

/// Sum of the finite cells, used to prove two generators agree.
pub fn matrix_checksum(rows: &[Vec<f64>]) -> f64 {
    rows.iter()
        .flatten()
        .filter(|v| v.is_finite())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_parity_signal_is_reproducible_and_bounded() {
        let a = parity_signal(22_050, 0.5);
        let b = parity_signal(22_050, 0.5);
        assert_eq!(a, b);
        assert!(a.iter().all(|v| v.abs() <= 1.0));
        assert_eq!(a.len(), 11_025);
    }
}
