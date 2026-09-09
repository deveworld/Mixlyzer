//! The onset detection function: how much the spectrum changes, frame by frame.
//!
//! Everything downstream of this — tempo, phase, the beatgrid — is a reading of
//! this one curve, so it is worth computing carefully. The method is the usual
//! one: a short-time Fourier transform, mapped onto a mel scale so that a kick
//! and a hi-hat contribute comparably, compressed logarithmically, then summed
//! over the frame-to-frame increases.

use std::sync::Arc;

use rustfft::{num_complex::Complex, Fft, FftPlanner};

/// Where in the analysis window a transient sits when the flux peaks.
///
/// The window tapers, so a transient entering at the very edge contributes
/// almost nothing; the energy climbs fastest as the transient crosses the
/// three-quarter point of a Hann window. Reporting the raw frame start would
/// therefore place every onset about `0.75 * n_fft` samples early — 70 ms at
/// the default settings, which is a seventh of a beat at 120 BPM and enough to
/// put the whole beatgrid visibly off the music.
const WINDOW_GROUP_DELAY: f64 = 0.75;

/// Onset strength over time, on a fixed frame grid.
#[derive(Debug, Clone, PartialEq)]
pub struct OnsetEnvelope {
    /// One non-negative strength value per frame.
    pub strength: Vec<f64>,
    /// Hop between frames, in samples.
    pub hop: usize,
    /// Analysis window length, in samples. Sets the group delay above.
    pub n_fft: usize,
    pub sample_rate: u32,
}

impl OnsetEnvelope {
    pub fn len(&self) -> usize {
        self.strength.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strength.is_empty()
    }

    /// Seconds between consecutive frames.
    pub fn frame_duration(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.hop as f64 / f64::from(self.sample_rate)
        }
    }

    /// Delay between a frame's nominal start and the transient it reports on.
    ///
    /// Anything that converts a frame index back into a track time has to add
    /// this, or every derived time lands early by the same amount.
    pub fn group_delay(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            WINDOW_GROUP_DELAY * self.n_fft as f64 / f64::from(self.sample_rate)
        }
    }

    /// Time of the transient that frame `index` reports on.
    pub fn frame_time(&self, index: usize) -> f64 {
        index as f64 * self.frame_duration() + self.group_delay()
    }

    /// Frame reporting on a given time.
    pub fn frame_at(&self, time: f64) -> usize {
        let step = self.frame_duration();
        if step <= 0.0 {
            return 0;
        }
        let shifted = ((time - self.group_delay()) / step).round().max(0.0);
        (shifted as usize).min(self.strength.len().saturating_sub(1))
    }

    /// Whether there is any onset energy at all.
    ///
    /// Worth checking before tempo estimation: digital silence has no tempo,
    /// and the Python pipeline discovers this only when its eigendecomposition
    /// throws an opaque ArpackError.
    pub fn has_energy(&self) -> bool {
        self.strength.iter().any(|v| *v > 1e-9)
    }

    /// Mean-remove and scale to unit standard deviation.
    ///
    /// Autocorrelation of a curve with a large DC term is dominated by that
    /// term rather than by its periodicity.
    pub fn normalized(&self) -> Vec<f64> {
        let n = self.strength.len();
        if n == 0 {
            return Vec::new();
        }
        let mean = self.strength.iter().sum::<f64>() / n as f64;
        let variance = self
            .strength
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            / n as f64;
        let scale = variance.sqrt();
        if scale < 1e-12 {
            return vec![0.0; n];
        }
        self.strength.iter().map(|v| (v - mean) / scale).collect()
    }
}

/// How the onset envelope is computed.
#[derive(Debug, Clone, Copy)]
pub struct OnsetOptions {
    /// FFT size. Must be a power of two for the transform to be fast.
    pub n_fft: usize,
    /// Hop between analysis frames, in samples.
    pub hop: usize,
    /// Number of mel bands.
    pub n_mels: usize,
    /// Lowest mel band edge.
    pub fmin: f64,
    /// Highest mel band edge; clamped to Nyquist.
    pub fmax: f64,
}

impl Default for OnsetOptions {
    fn default() -> Self {
        Self {
            n_fft: 2048,
            hop: 128,
            n_mels: 64,
            fmin: 30.0,
            fmax: 11_025.0,
        }
    }
}

/// Compute the onset envelope of `signal`.
pub fn compute(signal: &[f32], sample_rate: u32, options: OnsetOptions) -> OnsetEnvelope {
    let hop = options.hop.max(1);
    let n_fft = options.n_fft.max(16).next_power_of_two();

    if signal.len() < n_fft || sample_rate == 0 {
        return OnsetEnvelope {
            strength: Vec::new(),
            hop,
            n_fft,
            sample_rate,
        };
    }

    let spectrogram = mel_spectrogram(signal, sample_rate, n_fft, hop, &options);
    let strength = spectral_flux(&spectrogram);

    OnsetEnvelope {
        strength,
        hop,
        n_fft,
        sample_rate,
    }
}

/// Log-compressed mel spectrogram, indexed `[frame][mel_band]`.
fn mel_spectrogram(
    signal: &[f32],
    sample_rate: u32,
    n_fft: usize,
    hop: usize,
    options: &OnsetOptions,
) -> Vec<Vec<f64>> {
    let window = hann_window(n_fft);
    let filters = mel_filterbank(
        options.n_mels,
        n_fft,
        f64::from(sample_rate),
        options.fmin,
        options.fmax,
    );

    let mut planner = FftPlanner::<f64>::new();
    let fft: Arc<dyn Fft<f64>> = planner.plan_fft_forward(n_fft);

    let frame_count = (signal.len() - n_fft) / hop + 1;
    let bins = n_fft / 2 + 1;
    let mut scratch = vec![Complex::new(0.0, 0.0); n_fft];
    let mut power = vec![0.0f64; bins];
    let mut out = Vec::with_capacity(frame_count);

    for frame in 0..frame_count {
        let start = frame * hop;
        for (i, slot) in scratch.iter_mut().enumerate() {
            *slot = Complex::new(f64::from(signal[start + i]) * window[i], 0.0);
        }
        fft.process(&mut scratch);
        for (bin, slot) in power.iter_mut().enumerate() {
            *slot = scratch[bin].norm_sqr();
        }
        // Log compression: without it, one loud bass note swamps every other
        // band and the flux tracks that note instead of the rhythm.
        let row = filters
            .iter()
            .map(|weights| {
                let energy: f64 = weights
                    .iter()
                    .map(|(bin, weight)| power[*bin] * weight)
                    .sum();
                (1.0 + energy).ln()
            })
            .collect();
        out.push(row);
    }
    out
}

/// Sum of the positive frame-to-frame increases in each band.
fn spectral_flux(spectrogram: &[Vec<f64>]) -> Vec<f64> {
    if spectrogram.len() < 2 {
        return vec![0.0; spectrogram.len()];
    }
    let mut out = Vec::with_capacity(spectrogram.len());
    out.push(0.0); // nothing to difference against on the first frame
    for pair in spectrogram.windows(2) {
        let (previous, current) = (&pair[0], &pair[1]);
        let flux: f64 = current
            .iter()
            .zip(previous)
            .map(|(now, before)| (now - before).max(0.0))
            .sum();
        out.push(flux);
    }
    out
}

/// Periodic Hann window.
fn hann_window(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| {
            let phase = 2.0 * std::f64::consts::PI * i as f64 / size as f64;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

/// Triangular mel filters as `(bin, weight)` pairs, one list per band.
fn mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: f64,
    fmin: f64,
    fmax: f64,
) -> Vec<Vec<(usize, f64)>> {
    let nyquist = sample_rate * 0.5;
    let fmin = fmin.max(0.0);
    let fmax = fmax.min(nyquist).max(fmin + 1.0);
    let bins = n_fft / 2 + 1;

    let mel_lo = hz_to_mel(fmin);
    let mel_hi = hz_to_mel(fmax);
    // n_mels bands need n_mels + 2 edges: each band spans three of them.
    let edges: Vec<f64> = (0..n_mels + 2)
        .map(|i| {
            let mel = mel_lo + (mel_hi - mel_lo) * i as f64 / (n_mels + 1) as f64;
            mel_to_hz(mel)
        })
        .collect();

    let bin_hz = sample_rate / n_fft as f64;
    let mut filters = Vec::with_capacity(n_mels);
    for band in 0..n_mels {
        let (lo, center, hi) = (edges[band], edges[band + 1], edges[band + 2]);
        let mut weights = Vec::new();
        for bin in 0..bins {
            let freq = bin as f64 * bin_hz;
            let weight = if freq >= lo && freq <= center && center > lo {
                (freq - lo) / (center - lo)
            } else if freq > center && freq <= hi && hi > center {
                (hi - freq) / (hi - center)
            } else {
                0.0
            };
            if weight > 0.0 {
                weights.push((bin, weight));
            }
        }
        filters.push(weights);
    }
    filters
}

/// The HTK mel scale, which is what librosa uses by default.
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Clicks at a fixed interval, the simplest thing with a real tempo.
    fn click_track(bpm: f64, seconds: f64, rate: u32) -> Vec<f32> {
        let n = (seconds * f64::from(rate)) as usize;
        let period = (60.0 / bpm * f64::from(rate)) as usize;
        let mut signal = vec![0.0f32; n];
        let mut at = 0usize;
        while at < n {
            for offset in 0..64.min(n - at) {
                let decay = 1.0 - offset as f32 / 64.0;
                signal[at + offset] = decay * if offset % 2 == 0 { 1.0 } else { -1.0 };
            }
            at += period;
        }
        signal
    }

    #[test]
    fn a_signal_shorter_than_the_window_gives_nothing() {
        let env = compute(&vec![0.1f32; 100], 22_050, OnsetOptions::default());
        assert!(env.is_empty());
        assert!(!env.has_energy());
    }

    #[test]
    fn silence_has_no_onset_energy() {
        let env = compute(&vec![0.0f32; 22_050], 22_050, OnsetOptions::default());
        assert!(!env.is_empty(), "frames should still be produced");
        assert!(!env.has_energy(), "but none of them should carry energy");
    }

    #[test]
    fn strength_is_never_negative() {
        let signal = click_track(128.0, 4.0, 22_050);
        let env = compute(&signal, 22_050, OnsetOptions::default());
        assert!(env.strength.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn clicks_produce_peaks_at_the_click_positions() {
        let bpm = 120.0;
        let rate = 22_050;
        let signal = click_track(bpm, 8.0, rate);
        let env = compute(&signal, rate, OnsetOptions::default());
        assert!(env.has_energy());

        // The largest peaks should sit a beat period apart.
        let period_frames = (60.0 / bpm) / env.frame_duration();
        let mut peaks: Vec<usize> = Vec::new();
        let threshold = env.strength.iter().cloned().fold(0.0f64, f64::max) * 0.5;
        for (i, value) in env.strength.iter().enumerate() {
            if *value > threshold
                && peaks
                    .last()
                    .is_none_or(|last| i - last > period_frames as usize / 2)
            {
                peaks.push(i);
            }
        }
        assert!(peaks.len() >= 4, "expected several peaks, found {}", peaks.len());
        for pair in peaks.windows(2) {
            let gap = (pair[1] - pair[0]) as f64;
            assert!(
                (gap - period_frames).abs() < period_frames * 0.25,
                "peak gap {gap} frames is not near the beat period {period_frames}"
            );
        }
    }

    #[test]
    fn frame_timing_follows_the_hop() {
        let env = OnsetEnvelope {
            strength: vec![0.0; 100],
            hop: 128,
            n_fft: 2048,
            sample_rate: 22_050,
        };
        let expected = 128.0 / 22_050.0;
        assert!((env.frame_duration() - expected).abs() < 1e-12);
        // frame_time and frame_at must be inverses of each other.
        for frame in [0usize, 1, 10, 99] {
            assert_eq!(env.frame_at(env.frame_time(frame)), frame);
        }
    }

    #[test]
    fn frame_lookup_is_clamped_to_the_available_frames() {
        let env = OnsetEnvelope {
            strength: vec![0.0; 10],
            hop: 128,
            n_fft: 2048,
            sample_rate: 22_050,
        };
        assert_eq!(env.frame_at(-5.0), 0);
        assert_eq!(env.frame_at(1e6), 9);
    }

    #[test]
    fn normalization_centres_and_scales() {
        let env = OnsetEnvelope {
            strength: vec![1.0, 2.0, 3.0, 4.0],
            hop: 128,
            n_fft: 2048,
            sample_rate: 22_050,
        };
        let norm = env.normalized();
        let mean = norm.iter().sum::<f64>() / norm.len() as f64;
        assert!(mean.abs() < 1e-12, "mean should vanish, got {mean}");
        let var = norm.iter().map(|v| v * v).sum::<f64>() / norm.len() as f64;
        assert!((var - 1.0).abs() < 1e-9, "variance should be 1, got {var}");
    }

    #[test]
    fn normalizing_a_flat_curve_gives_zeros_rather_than_dividing_by_zero() {
        let env = OnsetEnvelope {
            strength: vec![5.0; 16],
            hop: 128,
            n_fft: 2048,
            sample_rate: 22_050,
        };
        assert!(env.normalized().iter().all(|v| *v == 0.0));
    }

    /// The reported onset time must land on the transient, not on the frame
    /// where the analysis window first brushed against it.
    #[test]
    fn reported_onset_times_land_on_the_clicks() {
        let rate = 22_050u32;
        let bpm = 120.0;
        let signal = click_track(bpm, 8.0, rate);
        let env = compute(&signal, rate, OnsetOptions::default());

        let threshold = env.strength.iter().cloned().fold(0.0f64, f64::max) * 0.5;
        let period = 60.0 / bpm;
        let mut checked = 0;
        let mut previous: Option<usize> = None;
        for (frame, value) in env.strength.iter().enumerate() {
            if *value <= threshold {
                continue;
            }
            if previous.is_some_and(|p| frame - p < 20) {
                continue;
            }
            previous = Some(frame);
            let time = env.frame_time(frame);
            let offset = (time / period).fract();
            let error = offset.min(1.0 - offset) * period;
            assert!(
                error < 0.02,
                "onset reported at {time:.4}s is {error:.4}s off the click grid"
            );
            checked += 1;
        }
        assert!(checked >= 4, "expected several onsets, checked {checked}");
    }

    #[test]
    fn the_mel_scale_round_trips() {
        for hz in [0.0, 100.0, 1_000.0, 8_000.0, 11_025.0] {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!((back - hz).abs() < 1e-6, "{hz} Hz round-tripped to {back}");
        }
    }

    #[test]
    fn the_filterbank_has_one_entry_per_band_and_covers_the_spectrum() {
        let filters = mel_filterbank(32, 2048, 22_050.0, 30.0, 11_025.0);
        assert_eq!(filters.len(), 32);
        assert!(
            filters.iter().all(|f| !f.is_empty()),
            "every band should touch at least one bin"
        );
        // Bands ascend: the first band's bins sit below the last band's.
        let first_max = filters[0].iter().map(|(b, _)| *b).max().unwrap();
        let last_min = filters[31].iter().map(|(b, _)| *b).min().unwrap();
        assert!(first_max < last_min);
    }

    #[test]
    fn filter_weights_are_a_triangle_peaking_at_one() {
        let filters = mel_filterbank(8, 1024, 22_050.0, 30.0, 11_025.0);
        for band in &filters {
            let peak = band.iter().map(|(_, w)| *w).fold(0.0f64, f64::max);
            assert!(peak <= 1.0 + 1e-9 && peak > 0.3, "peak weight {peak}");
        }
    }

    #[test]
    fn the_hann_window_is_zero_at_the_edges_and_one_in_the_middle() {
        let window = hann_window(64);
        assert!(window[0].abs() < 1e-12);
        assert!((window[32] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn flux_ignores_decreases() {
        // Rising then falling: only the rise should register.
        let spectrogram = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![0.0, 0.0]];
        let flux = spectral_flux(&spectrogram);
        assert_eq!(flux[0], 0.0);
        assert!((flux[1] - 2.0).abs() < 1e-12);
        assert_eq!(flux[2], 0.0, "a fall must not count as an onset");
    }
}
