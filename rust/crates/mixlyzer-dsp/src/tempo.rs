//! Tempo and beat-phase estimation from the onset envelope.
//!
//! The method is autocorrelation to find the beat period, then a fine sweep to
//! refine it, then a phase search to decide where the beats actually fall.
//!
//! Two things are done differently from the Python implementation, both of
//! them bugs there rather than choices.
//!
//! The first is an indexing error. Python slices the autocorrelation to the
//! searched lag range and then indexes that slice with unsliced lag values
//! minus the range start — subtracting the offset a second time. With the
//! default 110-220 BPM range the negative indices wrap around and merely rank
//! the candidates by unrelated values; narrow the range in the settings, as
//! `bpm_min=120, bpm_max=140`, and it raises `IndexError` and the track fails
//! to analyse at all.
//!
//! The second is a bias. Python prefers integer tempos by penalising any
//! candidate whose distance from `int(bpm)` exceeds 0.1, but truncation makes
//! that window asymmetric: 128.05 is unpenalised while 127.95 is halved, so
//! the search is pulled upward. Rounding makes the preference symmetric.

use mixlyzer_core::segments::TempoSegment;

use crate::error::AnalysisError;
use crate::onset::OnsetEnvelope;

/// A tempo estimate together with where its beats land.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TempoEstimate {
    pub bpm: f64,
    /// Time of the first beat, in seconds.
    pub phase_sec: f64,
    /// How strongly the onset envelope agrees with this grid, `0..=1`.
    pub confidence: f64,
}

impl TempoEstimate {
    pub fn beat_period(&self) -> f64 {
        60.0 / self.bpm
    }

    /// Beat times covering `[0, duration_sec]`.
    pub fn beats(&self, duration_sec: f64) -> Vec<f64> {
        let period = self.beat_period();
        if !period.is_finite() || period <= 0.0 || duration_sec <= 0.0 {
            return Vec::new();
        }
        // Step back to the first beat at or after zero, so a phase found in
        // the middle of the track still describes the whole grid.
        let first = self.phase_sec - (self.phase_sec / period).floor() * period;
        let mut beats = Vec::with_capacity((duration_sec / period) as usize + 2);
        let mut t = first;
        while t <= duration_sec + 1e-9 {
            beats.push(t);
            t += period;
        }
        beats
    }
}

/// Knobs for the tempo search.
#[derive(Debug, Clone, Copy)]
pub struct TempoOptions {
    /// Lowest tempo to consider.
    pub bpm_min: f64,
    /// Highest tempo to consider.
    pub bpm_max: f64,
    /// How many autocorrelation peaks to refine.
    pub candidates: usize,
    /// Strength of the pull towards whole-number tempos, `0..1`.
    pub integer_bias: f64,
}

impl Default for TempoOptions {
    fn default() -> Self {
        Self {
            bpm_min: 110.0,
            bpm_max: 220.0,
            candidates: 10,
            integer_bias: 0.15,
        }
    }
}

impl TempoOptions {
    /// The range, ordered and never degenerate.
    fn range(&self) -> (f64, f64) {
        let lo = self.bpm_min.min(self.bpm_max).max(20.0);
        let hi = self.bpm_max.max(self.bpm_min).min(400.0);
        (lo, hi.max(lo + 1.0))
    }
}

/// Estimate a single tempo and phase for the whole envelope.
pub fn estimate(
    envelope: &OnsetEnvelope,
    options: TempoOptions,
) -> Result<TempoEstimate, AnalysisError> {
    if !envelope.has_energy() {
        return Err(AnalysisError::Silent);
    }
    let frame_dur = envelope.frame_duration();
    if frame_dur <= 0.0 {
        return Err(AnalysisError::Silent);
    }

    let signal = envelope.normalized();
    let (bpm_lo, bpm_hi) = options.range();

    // A lag is a number of frames; convert the tempo bounds into that space.
    // A slower tempo means a longer period, so bpm_hi gives the *smallest* lag.
    let min_lag = ((60.0 / bpm_hi) / frame_dur).floor().max(1.0) as usize;
    let max_lag = ((60.0 / bpm_lo) / frame_dur).ceil() as usize;
    let max_lag = max_lag.min(signal.len().saturating_sub(1));

    if max_lag <= min_lag || max_lag - min_lag < 2 {
        return Err(AnalysisError::TempoRangeTooNarrow {
            lo: bpm_lo,
            hi: bpm_hi,
            hop: envelope.hop,
        });
    }

    let autocorr = autocorrelate(&signal);

    // Peaks within the searched band. `lag` is an absolute lag, and
    // `autocorr[lag]` is indexed by that same absolute lag — the offset is
    // never subtracted, which is the Python bug this module documents.
    let mut candidates: Vec<(usize, f64)> = Vec::new();
    for lag in (min_lag + 1)..max_lag {
        let value = autocorr[lag];
        if value > autocorr[lag - 1] && value >= autocorr[lag + 1] && value > 0.0 {
            candidates.push((lag, value));
        }
    }
    // A band with no interior peak still has a best lag; use it rather than
    // giving up, which is what makes very short excerpts analysable.
    if candidates.is_empty() {
        let best = (min_lag..=max_lag)
            .max_by(|a, b| {
                autocorr[*a]
                    .partial_cmp(&autocorr[*b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or(AnalysisError::Silent)?;
        candidates.push((best, autocorr[best]));
    }

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(options.candidates.max(1));

    let mut best: Option<TempoEstimate> = None;
    let mut best_score = f64::NEG_INFINITY;
    for (lag, _) in candidates {
        let coarse_bpm = 60.0 / (lag as f64 * frame_dur);
        let (bpm, score) = refine_tempo(&signal, frame_dur, coarse_bpm, bpm_lo, bpm_hi, options);
        if score > best_score {
            let phase_frames = locate_phase(&signal, frame_dur, bpm);
            // Frame indices carry the analysis window's latency; add it back so
            // the phase is a real position in the track rather than a position
            // in the onset envelope.
            let phase_sec = phase_frames * frame_dur + envelope.group_delay();
            best_score = score;
            best = Some(TempoEstimate {
                bpm,
                phase_sec: phase_sec.max(0.0),
                confidence: phase_confidence(&envelope.strength, frame_dur, bpm),
            });
        }
    }

    best.ok_or(AnalysisError::Silent)
}

/// Search around `coarse_bpm` for the tempo the envelope agrees with most.
///
/// Three passes, each an order of magnitude finer than the last.
fn refine_tempo(
    signal: &[f64],
    frame_dur: f64,
    coarse_bpm: f64,
    bpm_lo: f64,
    bpm_hi: f64,
    options: TempoOptions,
) -> (f64, f64) {
    let mut best_bpm = coarse_bpm.clamp(bpm_lo, bpm_hi);
    let mut best_score = fold_score(signal, frame_dur, best_bpm, options.integer_bias);

    for (span, step) in [(5.0f64, 0.05f64), (0.1, 0.01), (0.02, 0.001)] {
        let center = best_bpm;
        let steps = (2.0 * span / step).round() as i64;
        for i in 0..=steps {
            let bpm = center - span + step * i as f64;
            if bpm < bpm_lo || bpm > bpm_hi {
                continue;
            }
            let score = fold_score(signal, frame_dur, bpm, options.integer_bias);
            if score > best_score {
                best_score = score;
                best_bpm = bpm;
            }
        }
    }
    (best_bpm, best_score)
}

/// How peaked the envelope looks when folded onto one beat period.
///
/// A correct tempo concentrates the onsets into one place in the fold; a wrong
/// one smears them out.
fn fold_score(signal: &[f64], frame_dur: f64, bpm: f64, integer_bias: f64) -> f64 {
    let period_frames = (60.0 / bpm) / frame_dur;
    if !period_frames.is_finite() || period_frames < 2.0 {
        return f64::NEG_INFINITY;
    }
    let bins = period_frames.round().max(2.0) as usize;
    let mut histogram = vec![0.0f64; bins];
    for (index, value) in signal.iter().enumerate() {
        let phase = (index as f64 / period_frames).fract();
        let bin = ((phase * bins as f64) as usize).min(bins - 1);
        histogram[bin] += value;
    }

    let mut sorted = histogram.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    let peak = sorted[sorted.len() - 1];
    let mut score = peak - median;

    if integer_bias > 0.0 && !is_near_integer_tempo(bpm) {
        score *= 1.0 - integer_bias.clamp(0.0, 1.0);
    }
    score
}

/// How far from a whole number a tempo may sit and still count as "round".
const INTEGER_TOLERANCE_BPM: f64 = 0.1;

/// Whether `bpm` is close enough to a whole number to escape the penalty.
///
/// Rounding, not truncation. Python tests `abs(bpm - int(bpm)) > 0.1`, which
/// puts the whole tolerance window above the integer: 128.05 passes but 127.95
/// is penalised, so the search is pulled upward for no musical reason.
fn is_near_integer_tempo(bpm: f64) -> bool {
    (bpm - bpm.round()).abs() <= INTEGER_TOLERANCE_BPM
}

/// Fold the signal onto one beat period, summing into `bins` phase buckets.
fn phase_histogram(signal: &[f64], period_frames: f64, bins: usize) -> Vec<f64> {
    let mut histogram = vec![0.0f64; bins];
    for (index, value) in signal.iter().enumerate() {
        let phase = (index as f64 / period_frames).fract();
        let bin = ((phase * bins as f64) as usize).min(bins - 1);
        histogram[bin] += value;
    }
    histogram
}

/// Where in the beat period the onsets pile up, measured in frames.
fn locate_phase(signal: &[f64], frame_dur: f64, bpm: f64) -> f64 {
    let period_frames = (60.0 / bpm) / frame_dur;
    if !period_frames.is_finite() || period_frames < 2.0 {
        return 0.0;
    }
    let bins = period_frames.round().max(2.0) as usize;
    let histogram = phase_histogram(signal, period_frames, bins);

    let (best_bin, best_value) = histogram
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, v)| (i, *v))
        .unwrap_or((0, 0.0));

    // Parabolic interpolation across the winning bin's neighbours, so the phase
    // is not quantised to whole frames.
    let left = histogram[(best_bin + bins - 1) % bins];
    let right = histogram[(best_bin + 1) % bins];
    let denom = left - 2.0 * best_value + right;
    let offset = if denom.abs() > 1e-12 {
        (0.5 * (left - right) / denom).clamp(-0.5, 0.5)
    } else {
        0.0
    };

    (best_bin as f64 + offset) * period_frames / bins as f64
}

/// How concentrated the onsets are at one phase, `0..=1`.
///
/// Measured on the raw envelope rather than the mean-removed one used for
/// autocorrelation: a curve centred on zero sums to nothing, so a share-of-total
/// measure taken from it is meaningless.
fn phase_confidence(strength: &[f64], frame_dur: f64, bpm: f64) -> f64 {
    let period_frames = (60.0 / bpm) / frame_dur;
    if !period_frames.is_finite() || period_frames < 2.0 {
        return 0.0;
    }
    let bins = period_frames.round().max(2.0) as usize;
    let histogram = phase_histogram(strength, period_frames, bins);
    let total: f64 = histogram.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let peak = histogram.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // Compare the winning bin's share against what a flat distribution would
    // give it, so the score does not drift with the number of bins.
    let uniform = 1.0 / bins as f64;
    ((peak / total - uniform) / (1.0 - uniform)).clamp(0.0, 1.0)
}

/// Autocorrelation via FFT, normalised so lag zero is 1.
fn autocorrelate(signal: &[f64]) -> Vec<f64> {
    use rustfft::{num_complex::Complex, FftPlanner};

    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    // Zero-pad to twice the length so the circular correlation the FFT
    // computes matches the linear one we want.
    let size = (2 * n).next_power_of_two();
    let mut buffer: Vec<Complex<f64>> = signal
        .iter()
        .map(|v| Complex::new(*v, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)).take(size - n))
        .collect();

    let mut planner = FftPlanner::<f64>::new();
    planner.plan_fft_forward(size).process(&mut buffer);
    for slot in buffer.iter_mut() {
        *slot = Complex::new(slot.norm_sqr(), 0.0);
    }
    planner.plan_fft_inverse(size).process(&mut buffer);

    let zero = buffer[0].re;
    let scale = if zero.abs() > 1e-12 { 1.0 / zero } else { 0.0 };
    buffer.iter().take(n).map(|c| c.re * scale).collect()
}

/// Estimate a tempo for each window of the track, then merge equal neighbours.
///
/// This is the dynamic-tempo path: tracks that speed up, slow down or change
/// feel get one segment per stretch rather than one average tempo that fits
/// none of it.
pub fn estimate_segments(
    envelope: &OnsetEnvelope,
    options: TempoOptions,
    window_sec: f64,
    duration_sec: f64,
) -> Result<Vec<TempoSegment>, AnalysisError> {
    if !envelope.has_energy() {
        return Err(AnalysisError::Silent);
    }
    let frame_dur = envelope.frame_duration();
    let window_frames = ((window_sec / frame_dur).round() as usize).max(64);

    // Too short to window: one segment for the whole track.
    if envelope.len() <= window_frames {
        let estimate = estimate(envelope, options)?;
        return Ok(vec![TempoSegment::new(
            0.0,
            duration_sec,
            estimate.bpm,
            estimate.phase_sec,
            4,
        )]);
    }

    let step = window_frames / 2;
    let mut windows: Vec<(f64, f64, f64, f64)> = Vec::new(); // start, end, bpm, phase
    let mut start_frame = 0usize;
    while start_frame + window_frames <= envelope.len() {
        let slice = OnsetEnvelope {
            strength: envelope.strength[start_frame..start_frame + window_frames].to_vec(),
            hop: envelope.hop,
            n_fft: envelope.n_fft,
            sample_rate: envelope.sample_rate,
        };
        if slice.has_energy() {
            if let Ok(estimate) = estimate(&slice, options) {
                let start = start_frame as f64 * frame_dur;
                let end = (start_frame + window_frames) as f64 * frame_dur;
                windows.push((start, end, estimate.bpm, start + estimate.phase_sec));
            }
        }
        start_frame += step;
    }

    if windows.is_empty() {
        let estimate = estimate(envelope, options)?;
        return Ok(vec![TempoSegment::new(
            0.0,
            duration_sec,
            estimate.bpm,
            estimate.phase_sec,
            4,
        )]);
    }

    // Merge neighbouring windows whose tempo agrees. The tolerance is
    // deliberately loose: the fine sweep resolves to 0.001 BPM, and treating
    // that resolution as a real tempo change would shred the track into
    // hundreds of segments.
    const MERGE_TOLERANCE_BPM: f64 = 0.5;
    let mut segments: Vec<TempoSegment> = Vec::new();
    for (start, end, bpm, phase) in windows {
        match segments.last_mut() {
            Some(last) if (last.bpm - bpm).abs() <= MERGE_TOLERANCE_BPM => {
                last.end = end.max(last.end);
            }
            _ => segments.push(TempoSegment::new(start, end, bpm, phase, 4)),
        }
    }

    // Close the gaps left by the half-window step, and extend to the ends.
    if let Some(first) = segments.first_mut() {
        first.start = 0.0;
    }
    for index in 1..segments.len() {
        let boundary = segments[index].start;
        segments[index - 1].end = boundary;
    }
    if let Some(last) = segments.last_mut() {
        last.end = duration_sec.max(last.end);
    }
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onset::{self, OnsetOptions};

    const RATE: u32 = 22_050;

    fn click_track(bpm: f64, seconds: f64) -> Vec<f32> {
        let n = (seconds * f64::from(RATE)) as usize;
        let period = 60.0 / bpm * f64::from(RATE);
        let mut signal = vec![0.0f32; n];
        let mut beat = 0usize;
        loop {
            let at = (beat as f64 * period).round() as usize;
            if at >= n {
                break;
            }
            // A short noisy burst: broadband, so every mel band sees it.
            for offset in 0..128.min(n - at) {
                let decay = (1.0 - offset as f32 / 128.0).powi(2);
                let noise = if (at + offset) % 3 == 0 { 1.0 } else { -0.7 };
                signal[at + offset] = decay * noise;
            }
            beat += 1;
        }
        signal
    }

    fn envelope_of(signal: &[f32]) -> OnsetEnvelope {
        onset::compute(signal, RATE, OnsetOptions::default())
    }

    #[test]
    fn recovers_the_tempo_of_a_click_track() {
        for bpm in [120.0, 128.0, 140.0, 174.0] {
            let envelope = envelope_of(&click_track(bpm, 20.0));
            let estimate = estimate(&envelope, TempoOptions::default()).unwrap();
            assert!(
                (estimate.bpm - bpm).abs() < 1.0,
                "expected {bpm} BPM, estimated {}",
                estimate.bpm
            );
        }
    }

    /// The narrow-range case that raises `IndexError` in the Python.
    #[test]
    fn a_narrow_tempo_range_still_works() {
        let envelope = envelope_of(&click_track(128.0, 20.0));
        let options = TempoOptions {
            bpm_min: 120.0,
            bpm_max: 140.0,
            ..TempoOptions::default()
        };
        let estimate = estimate(&envelope, options).unwrap();
        assert!((estimate.bpm - 128.0).abs() < 1.0, "got {}", estimate.bpm);
    }

    #[test]
    fn a_tempo_outside_the_range_is_not_returned() {
        let envelope = envelope_of(&click_track(128.0, 20.0));
        let options = TempoOptions {
            bpm_min: 150.0,
            bpm_max: 180.0,
            ..TempoOptions::default()
        };
        let estimate = estimate(&envelope, options).unwrap();
        assert!(
            estimate.bpm >= 150.0 && estimate.bpm <= 180.0,
            "estimate {} escaped the requested range",
            estimate.bpm
        );
    }

    #[test]
    fn silence_is_reported_as_silence_not_as_a_crash() {
        let envelope = envelope_of(&vec![0.0f32; RATE as usize * 5]);
        assert!(matches!(
            estimate(&envelope, TempoOptions::default()),
            Err(AnalysisError::Silent)
        ));
    }

    #[test]
    fn an_impossibly_narrow_range_is_reported_as_such() {
        let envelope = envelope_of(&click_track(128.0, 20.0));
        // A one-BPM-wide band high up maps to fewer than two distinct lags.
        let options = TempoOptions {
            bpm_min: 399.0,
            bpm_max: 400.0,
            ..TempoOptions::default()
        };
        assert!(matches!(
            estimate(&envelope, options),
            Err(AnalysisError::TempoRangeTooNarrow { .. })
        ));
    }

    #[test]
    fn the_phase_lands_on_the_clicks() {
        let bpm = 120.0;
        let envelope = envelope_of(&click_track(bpm, 20.0));
        let estimate = estimate(&envelope, TempoOptions::default()).unwrap();
        let period = 60.0 / bpm;
        // Clicks start at t=0, so the phase should be near a whole period.
        let distance = (estimate.phase_sec / period).fract();
        let error = distance.min(1.0 - distance) * period;
        assert!(
            error < 0.05,
            "phase {} is {error:.3}s off the click grid",
            estimate.phase_sec
        );
    }

    #[test]
    fn confidence_is_higher_for_a_clean_grid_than_for_noise() {
        let clean = envelope_of(&click_track(128.0, 20.0));
        let clean_estimate = estimate(&clean, TempoOptions::default()).unwrap();

        // Deterministic pseudo-noise: no periodicity to lock onto.
        let noise: Vec<f32> = (0..RATE as usize * 20)
            .map(|i| ((i as f64 * 12.9898).sin() * 43758.5453).fract() as f32 * 2.0 - 1.0)
            .collect();
        let noisy = envelope_of(&noise);
        let noisy_estimate = estimate(&noisy, TempoOptions::default()).unwrap();

        assert!(
            clean_estimate.confidence > noisy_estimate.confidence,
            "clean {} should beat noise {}",
            clean_estimate.confidence,
            noisy_estimate.confidence
        );
    }

    #[test]
    fn beats_tile_the_track_at_the_beat_period() {
        let estimate = TempoEstimate {
            bpm: 120.0,
            phase_sec: 0.25,
            confidence: 1.0,
        };
        let beats = estimate.beats(10.0);
        assert!(!beats.is_empty());
        assert!((beats[0] - 0.25).abs() < 1e-9);
        for pair in beats.windows(2) {
            assert!((pair[1] - pair[0] - 0.5).abs() < 1e-9);
        }
        assert!(*beats.last().unwrap() <= 10.0 + 1e-9);
    }

    #[test]
    fn a_phase_past_the_first_beat_still_describes_the_whole_grid() {
        // A phase found deep in the track must be wound back to the start.
        let estimate = TempoEstimate {
            bpm: 120.0,
            phase_sec: 30.25,
            confidence: 1.0,
        };
        let beats = estimate.beats(10.0);
        assert!((beats[0] - 0.25).abs() < 1e-9, "first beat at {}", beats[0]);
    }

    #[test]
    fn a_zero_length_track_has_no_beats() {
        let estimate = TempoEstimate {
            bpm: 120.0,
            phase_sec: 0.0,
            confidence: 1.0,
        };
        assert!(estimate.beats(0.0).is_empty());
    }

    /// Truncation makes Python's integer preference asymmetric, pulling the
    /// search upward; rounding treats both sides alike.
    #[test]
    fn the_integer_preference_is_symmetric() {
        for delta in [0.0, 0.01, 0.05, 0.099] {
            assert!(
                is_near_integer_tempo(128.0 - delta),
                "128 - {delta} should count as a round tempo"
            );
            assert!(
                is_near_integer_tempo(128.0 + delta),
                "128 + {delta} should count as a round tempo"
            );
        }
        for delta in [0.11, 0.25, 0.5] {
            assert!(!is_near_integer_tempo(128.0 - delta));
            assert!(!is_near_integer_tempo(128.0 + delta));
        }
    }

    #[test]
    fn a_tempo_off_the_integer_grid_scores_lower_than_one_on_it() {
        // A signal whose period is exactly 128 BPM at the default frame rate.
        let frame_dur = 128.0 / f64::from(RATE);
        let period_frames = (60.0 / 128.0) / frame_dur;
        let signal: Vec<f64> = (0..4000)
            .map(|i| {
                let phase = (i as f64 / period_frames).fract();
                if phase < 0.05 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let on_grid = fold_score(&signal, frame_dur, 128.0, 0.15);
        let off_grid = fold_score(&signal, frame_dur, 128.5, 0.15);
        assert!(
            off_grid < on_grid,
            "128.5 scored {off_grid}, which should be below 128.0 at {on_grid}"
        );
    }

    #[test]
    fn autocorrelation_peaks_at_the_period() {
        let period = 16usize;
        let signal: Vec<f64> = (0..1024)
            .map(|i| if i % period == 0 { 1.0 } else { 0.0 })
            .collect();
        let acf = autocorrelate(&signal);
        assert!((acf[0] - 1.0).abs() < 1e-9, "lag zero should be 1");
        // The period and its multiples should stand above their neighbours.
        for multiple in 1..4 {
            let lag = period * multiple;
            assert!(
                acf[lag] > acf[lag - 1] && acf[lag] > acf[lag + 1],
                "no peak at lag {lag}"
            );
        }
    }

    #[test]
    fn autocorrelation_of_nothing_is_nothing() {
        assert!(autocorrelate(&[]).is_empty());
    }

    #[test]
    fn segmentation_of_a_steady_track_yields_one_segment() {
        let envelope = envelope_of(&click_track(128.0, 30.0));
        let segments = estimate_segments(&envelope, TempoOptions::default(), 5.0, 30.0).unwrap();
        assert_eq!(segments.len(), 1, "a steady tempo should not be split");
        assert!((segments[0].bpm - 128.0).abs() < 1.0);
        assert_eq!(segments[0].start, 0.0);
        assert!((segments[0].end - 30.0).abs() < 1e-9);
    }

    #[test]
    fn segments_cover_the_track_without_gaps() {
        let mut signal = click_track(120.0, 15.0);
        signal.extend(click_track(150.0, 15.0));
        let envelope = envelope_of(&signal);
        let segments = estimate_segments(&envelope, TempoOptions::default(), 5.0, 30.0).unwrap();
        assert!(!segments.is_empty());
        assert_eq!(segments[0].start, 0.0);
        for pair in segments.windows(2) {
            assert!(
                (pair[0].end - pair[1].start).abs() < 1e-9,
                "gap between {:?} and {:?}",
                pair[0],
                pair[1]
            );
        }
        assert!((segments.last().unwrap().end - 30.0).abs() < 1e-6);
    }

    #[test]
    fn segmentation_of_silence_reports_silence() {
        let envelope = envelope_of(&vec![0.0f32; RATE as usize * 10]);
        assert!(matches!(
            estimate_segments(&envelope, TempoOptions::default(), 5.0, 10.0),
            Err(AnalysisError::Silent)
        ));
    }

    #[test]
    fn a_track_shorter_than_the_window_gets_one_segment() {
        let envelope = envelope_of(&click_track(128.0, 6.0));
        let segments = estimate_segments(&envelope, TempoOptions::default(), 30.0, 6.0).unwrap();
        assert_eq!(segments.len(), 1);
        assert!((segments[0].end - 6.0).abs() < 1e-9);
    }
}
