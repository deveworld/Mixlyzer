//! Beat-synchronous feature extraction: the 133 numbers that describe a beat.
//!
//! This is the port of `extract_song_features`. Frame-rate features are
//! computed once over the whole track, then collapsed onto the beat grid with
//! the reducer each one was trained with — median for the mel bands, sum for
//! onset strength, mean for everything else. The four families (timbre,
//! harmony, rhythm, texture) are stacked in a fixed order, because the models
//! address them by column index and nothing records what column 91 was.

use crate::error::PhraseError;
use crate::grid::PredictorGrid;
use crate::model::ModelSettings;

use super::matrix::{mean, median, std, Mat};
use super::{chroma, hpss, mel, onset, spectral, stft};

/// The floor the Python uses before every logarithm in this file.
const EPS: f64 = 1e-10;

/// Onset-profile resolution inside a beat, from `FeatureConfig`.
const BEAT_SUBDIVISIONS: usize = 8;

/// The lowest mel band edge, from `FeatureConfig`.
const FMIN: f64 = 30.0;

/// How frame-rate values are collapsed onto a beat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reducer {
    Mean,
    Median,
    Sum,
}

/// Beat-synchronous features, one column per beat.
#[derive(Debug, Clone)]
pub struct AcousticFeatures {
    /// Mel bands, MFCCs and spectral contrast.
    pub timbre: Mat,
    /// Chroma and tonnetz.
    pub harmony: Mat,
    /// Within-beat onset profiles and the three onset sums.
    pub rhythm: Mat,
    /// Loudness, band balance, spectral shape and the vocal proxy.
    pub texture: Mat,
    pub audio_duration_sec: f64,
}

impl AcousticFeatures {
    /// All four families stacked, in the order the models expect.
    pub fn stacked(&self) -> Mat {
        Mat::vstack(&[&self.timbre, &self.harmony, &self.rhythm, &self.texture])
    }
}

/// First index whose value is at least `target`, i.e. `np.searchsorted(side="left")`.
fn searchsorted(sorted: &[f64], target: f64) -> usize {
    sorted.partition_point(|v| *v < target)
}

/// Collapse frame-rate rows onto beat intervals.
///
/// A beat shorter than one hop can contain no frames at all; rather than emit a
/// zero, librosa's caller falls back to the single frame nearest the middle of
/// the beat, which is what the empty-interval branch does here.
fn sync_to_intervals(values: &Mat, frame_times: &[f64], edges: &[f64], reducer: Reducer) -> Mat {
    let intervals = edges.len() - 1;
    let mut out = Mat::zeros(values.rows(), intervals);
    let mut buffer: Vec<f64> = Vec::new();
    for index in 0..intervals {
        let lo = searchsorted(frame_times, edges[index]);
        let hi = searchsorted(frame_times, edges[index + 1]);
        let (from, to) = if hi <= lo {
            let middle = 0.5 * (edges[index] + edges[index + 1]);
            let nearest = searchsorted(frame_times, middle).min(frame_times.len().saturating_sub(1));
            (nearest, nearest + 1)
        } else {
            (lo, hi)
        };
        for row in 0..values.rows() {
            buffer.clear();
            buffer.extend_from_slice(&values.row(row)[from..to.min(values.cols())]);
            let value = match reducer {
                Reducer::Mean => mean(&buffer),
                Reducer::Median => median(&buffer),
                Reducer::Sum => buffer.iter().sum(),
            };
            out.set(row, index, value);
        }
    }
    out
}

/// The same, for a single row of frame-rate values.
fn sync_scalar(values: &[f64], frame_times: &[f64], edges: &[f64], reducer: Reducer) -> Vec<f64> {
    let mat = Mat::from_rows(vec![values.to_vec()]);
    sync_to_intervals(&mat, frame_times, edges, reducer).row(0).to_vec()
}

/// Onset energy distributed over eight slots inside each beat, per envelope.
///
/// Normalised within the beat, so this describes *where* the hits fall rather
/// than how loud they are — the loudness is already in the texture family.
fn subdivision_profiles(envelopes: &[&[f64]], frame_times: &[f64], edges: &[f64]) -> Mat {
    let n_beats = edges.len() - 1;
    let mut out = Mat::zeros(envelopes.len() * BEAT_SUBDIVISIONS, n_beats);
    for beat in 0..n_beats {
        let (start, end) = (edges[beat], edges[beat + 1]);
        let sub_edge = |k: usize| start + (end - start) * k as f64 / BEAT_SUBDIVISIONS as f64;
        for (channel, envelope) in envelopes.iter().enumerate() {
            let mut total = 0.0;
            let base = channel * BEAT_SUBDIVISIONS;
            for sub in 0..BEAT_SUBDIVISIONS {
                let lo = searchsorted(frame_times, sub_edge(sub));
                let hi = searchsorted(frame_times, sub_edge(sub + 1));
                let value = if hi <= lo {
                    let middle = 0.5 * (sub_edge(sub) + sub_edge(sub + 1));
                    let nearest =
                        searchsorted(frame_times, middle).min(frame_times.len().saturating_sub(1));
                    envelope.get(nearest).copied().unwrap_or(0.0)
                } else {
                    envelope[lo..hi.min(envelope.len())].iter().sum()
                };
                out.set(base + sub, beat, value);
                total += value;
            }
            let normalizer = total + EPS;
            for sub in 0..BEAT_SUBDIVISIONS {
                out.set(base + sub, beat, out.get(base + sub, beat) / normalizer);
            }
        }
    }
    out
}

/// Mean power across the bins in `[low, high)`.
fn band_energy(power: &Mat, frequencies: &[f64], low: f64, high: f64) -> Vec<f64> {
    let bins: Vec<usize> = (0..power.rows())
        .filter(|r| frequencies[*r] >= low && frequencies[*r] < high)
        .collect();
    if bins.is_empty() {
        return vec![0.0; power.cols()];
    }
    (0..power.cols())
        .map(|frame| bins.iter().map(|r| power.get(*r, frame)).sum::<f64>() / bins.len() as f64)
        .collect()
}

/// How peaky the harmonic spectrum is in the range a lead voice occupies.
fn pitch_salience(harmonic: &Mat, frequencies: &[f64]) -> Vec<f64> {
    let bins: Vec<usize> = (0..harmonic.rows())
        .filter(|r| frequencies[*r] >= 90.0 && frequencies[*r] <= 1400.0)
        .collect();
    if bins.is_empty() {
        return vec![0.0; harmonic.cols()];
    }
    (0..harmonic.cols())
        .map(|frame| {
            let mut peak = f64::NEG_INFINITY;
            let mut total = 0.0;
            for r in &bins {
                let value = harmonic.get(*r, frame);
                peak = peak.max(value);
                total += value;
            }
            let average = total / bins.len() as f64;
            (peak / average.max(EPS)).ln_1p()
        })
        .collect()
}

/// Median-centred, MAD-scaled values, falling back to the standard deviation.
fn robust_z(values: &[f64]) -> Vec<f64> {
    let centre = median(values);
    let deviations: Vec<f64> = values.iter().map(|v| (v - centre).abs()).collect();
    let mut scale = 1.4826 * median(&deviations);
    if scale <= 1e-8 {
        scale = std(values);
    }
    if scale <= 1e-8 {
        scale = 1.0;
    }
    values.iter().map(|v| (v - centre) / scale).collect()
}

fn log_floor(values: &[f64]) -> Vec<f64> {
    values.iter().map(|v| v.max(EPS).ln()).collect()
}

fn row(values: Vec<f64>) -> Mat {
    Mat::from_rows(vec![values])
}

/// Extract the beat-synchronous features for one track.
///
/// `samples` must already be at `settings.sample_rate`; see [`PhraseError::EmptyAudio`]
/// and the caller in [`crate::detect_phrases`] for how a mismatch is handled.
pub fn extract_song_features(
    samples: &[f32],
    settings: &ModelSettings,
    grid: &PredictorGrid,
) -> Result<AcousticFeatures, PhraseError> {
    if samples.is_empty() {
        return Err(PhraseError::EmptyAudio);
    }
    let sr = f64::from(settings.sample_rate);
    let n_fft = settings.n_fft;
    let hop = settings.hop_length;
    let duration_sec = samples.len() as f64 / sr;

    let beats = &grid.beat_times_sec;
    let gaps: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
    let ibi = median(&gaps);
    let tolerance = (2.0 * ibi).max(1.0);
    if beats[beats.len() - 1] > duration_sec + tolerance {
        return Err(PhraseError::GridPastAudio {
            beats_end: beats[beats.len() - 1],
            audio_end: duration_sec,
        });
    }

    let magnitude = stft::stft_magnitude(samples, n_fft, hop);
    let power = magnitude.map(|v| v * v);
    let (harmonic_mag, percussive_mag) = hpss::hpss(&magnitude);
    let harmonic_power = harmonic_mag.map(|v| v * v);
    let percussive_power = percussive_mag.map(|v| v * v);

    let frame_times = stft::frames_to_time(magnitude.cols(), sr, hop);
    let frequencies = stft::fft_frequencies(sr, n_fft);

    // `_parse_feature_config` pins fmax to Nyquist rather than the 11025 Hz
    // default, so the mel bands cover the whole spectrum at any sample rate.
    let fmax = 0.5 * sr;
    let basis = mel::mel_filterbank(sr, n_fft, settings.n_mels, FMIN, fmax);
    let mel_power = mel::melspectrogram(&power, &basis);
    let mel_db = mel::power_to_db(&mel_power, mel::DbRef::Max);
    let mfcc = mel::dct_ortho_rows(&mel_db, settings.n_mfcc);

    let tuning = chroma::estimate_tuning(&harmonic_power, sr, n_fft);
    let chromagram = chroma::chroma_stft(&harmonic_power, sr, n_fft, tuning);
    let tonnetz = chroma::tonnetz(&chromagram);

    // The band count is chosen so no band's upper edge crosses Nyquist.
    let contrast_bands = (((0.5 * sr / 200.0).log2().floor() as i64) - 1).clamp(1, 6) as usize;
    let contrast = spectral::spectral_contrast(&magnitude, sr, n_fft, contrast_bands);

    let centroid = spectral::spectral_centroid(&magnitude, sr, n_fft);
    let bandwidth = spectral::spectral_bandwidth(&magnitude, sr, n_fft);
    let flatness = spectral::spectral_flatness(&power);
    let rolloff = spectral::spectral_rolloff(&magnitude, sr, n_fft, 0.85);

    let rms = spectral::rms_from_spectrum(&magnitude, n_fft);
    let harmonic_rms = spectral::rms_from_spectrum(&harmonic_mag, n_fft);
    let percussive_rms = spectral::rms_from_spectrum(&percussive_mag, n_fft);
    // Mono in, so the side channel is silent by construction; this reproduces
    // the Python path for a mono decode exactly rather than approximating it.
    let side_rms = vec![0.0; magnitude.cols()];
    let mid_rms = spectral::rms_from_signal(samples, n_fft, hop);

    let low = band_energy(&power, &frequencies, 30.0, 180.0);
    let low_mid = band_energy(&power, &frequencies, 180.0, 800.0);
    let mid_band = band_energy(&power, &frequencies, 800.0, 4000.0);
    let high = band_energy(&power, &frequencies, 4000.0, 11_000.0);
    let total_energy: Vec<f64> = (0..power.cols())
        .map(|frame| (0..power.rows()).map(|r| power.get(r, frame)).sum::<f64>() / power.rows() as f64)
        .collect();
    let salience = pitch_salience(&harmonic_mag, &frequencies);

    let onset_full = onset::onset_strength(&mel_db);
    let percussive_mel = mel::melspectrogram(&percussive_power, &basis);
    let onset_percussive =
        onset::onset_strength(&mel::power_to_db(&percussive_mel, mel::DbRef::Max));
    let high_cut = 5000.0f64.min(0.45 * sr);
    let high_bins: Vec<usize> = (0..power.rows())
        .filter(|r| frequencies[*r] >= high_cut)
        .collect();
    let high_log: Vec<f64> = (0..power.cols())
        .map(|frame| {
            if high_bins.is_empty() {
                return 0.0;
            }
            let sum: f64 = high_bins.iter().map(|r| power.get(*r, frame)).sum();
            (sum / high_bins.len() as f64).ln_1p()
        })
        .collect();
    let onset_high: Vec<f64> = (0..high_log.len())
        .map(|i| {
            if i == 0 {
                0.0
            } else {
                (high_log[i] - high_log[i - 1]).max(0.0)
            }
        })
        .collect();

    // The trailing beat edge is clamped into the audio: a grid that runs a
    // beat past the file would otherwise sync the last beat against nothing.
    let mut edges = grid.beat_edges_sec.clone();
    let last = edges.len() - 1;
    edges[last] = edges[last].max(beats[beats.len() - 1]).min(duration_sec);
    if edges[last] <= edges[last - 1] {
        edges[last] = edges[last - 1] + ibi;
    }

    let beat_mel = sync_to_intervals(&mel_db, &frame_times, &edges, Reducer::Median);
    let beat_mfcc = sync_to_intervals(&mfcc, &frame_times, &edges, Reducer::Mean);
    let beat_chroma = sync_to_intervals(&chromagram, &frame_times, &edges, Reducer::Mean);
    let beat_tonnetz = sync_to_intervals(&tonnetz, &frame_times, &edges, Reducer::Mean);
    let beat_contrast = sync_to_intervals(&contrast, &frame_times, &edges, Reducer::Mean);
    let beat_rhythm = subdivision_profiles(
        &[&onset_full, &onset_percussive, &onset_high],
        &frame_times,
        &edges,
    );

    let scalar = |values: &[f64], reducer: Reducer| sync_scalar(values, &frame_times, &edges, reducer);

    let log_rms = log_floor(&scalar(&rms, Reducer::Mean));
    let harmonic_log_rms = log_floor(&scalar(&harmonic_rms, Reducer::Mean));
    let percussive_log_rms = log_floor(&scalar(&percussive_rms, Reducer::Mean));
    let low_log = log_floor(&scalar(&low, Reducer::Mean));
    let low_mid_log = log_floor(&scalar(&low_mid, Reducer::Mean));
    let mid_log = log_floor(&scalar(&mid_band, Reducer::Mean));
    let high_log_energy = log_floor(&scalar(&high, Reducer::Mean));
    let total_log = log_floor(&scalar(&total_energy, Reducer::Mean));
    let hp_ratio: Vec<f64> = harmonic_log_rms
        .iter()
        .zip(&percussive_log_rms)
        .map(|(h, p)| h - p)
        .collect();
    let side_sync = scalar(&side_rms, Reducer::Mean);
    let mid_sync = scalar(&mid_rms, Reducer::Mean);
    let stereo_width: Vec<f64> = side_sync
        .iter()
        .zip(&mid_sync)
        .map(|(s, m)| (s.max(EPS) / m.max(EPS)).ln())
        .collect();
    let nyquist = 0.5 * sr;
    let centroid_norm: Vec<f64> = scalar(&centroid, Reducer::Mean).iter().map(|v| v / nyquist).collect();
    let bandwidth_norm: Vec<f64> = scalar(&bandwidth, Reducer::Mean).iter().map(|v| v / nyquist).collect();
    let rolloff_norm: Vec<f64> = scalar(&rolloff, Reducer::Mean).iter().map(|v| v / nyquist).collect();
    let flatness_sync = scalar(&flatness, Reducer::Mean);
    let salience_sync = scalar(&salience, Reducer::Mean);
    let onset_full_sync = scalar(&onset_full, Reducer::Sum);
    let onset_percussive_sync = scalar(&onset_percussive, Reducer::Sum);
    let onset_high_sync = scalar(&onset_high, Reducer::Sum);

    // A hand-tuned stand-in for "is someone singing": bright, pitched, more
    // harmonic than percussive, and not noise-like.
    let mid_ratio: Vec<f64> = mid_log
        .iter()
        .zip(&total_log)
        .map(|(m, t)| (m - t).exp())
        .collect();
    let z_salience = robust_z(&salience_sync);
    let z_mid_ratio = robust_z(&mid_ratio);
    let z_hp = robust_z(&hp_ratio);
    let z_flatness = robust_z(&flatness_sync);
    let z_onset_percussive = robust_z(&onset_percussive_sync);
    let vocal_proxy: Vec<f64> = (0..z_salience.len())
        .map(|i| {
            let logit = 0.85 * z_salience[i] + 0.70 * z_mid_ratio[i] + 0.45 * z_hp[i]
                - 0.35 * z_flatness[i]
                - 0.20 * z_onset_percussive[i];
            1.0 / (1.0 + (-logit).exp())
        })
        .collect();

    Ok(AcousticFeatures {
        timbre: Mat::vstack(&[&beat_mel, &beat_mfcc, &beat_contrast]),
        harmony: Mat::vstack(&[&beat_chroma, &beat_tonnetz]),
        rhythm: Mat::vstack(&[
            &beat_rhythm,
            &row(onset_full_sync),
            &row(onset_percussive_sync),
            &row(onset_high_sync),
        ]),
        texture: Mat::vstack(&[
            &row(log_rms),
            &row(harmonic_log_rms),
            &row(percussive_log_rms),
            &row(low_log),
            &row(low_mid_log),
            &row(mid_log),
            &row(high_log_energy),
            &row(hp_ratio),
            &row(centroid_norm),
            &row(bandwidth_norm),
            &row(rolloff_norm),
            &row(flatness_sync),
            &row(stereo_width),
            &row(salience_sync),
            &row(vocal_proxy),
        ]),
        audio_duration_sec: duration_sec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsig;
    use mixlyzer_core::TempoSegment;

    fn grid_for(seconds: f64, period: f64) -> PredictorGrid {
        let beats: Vec<f64> = (0..(seconds / period) as usize).map(|i| i as f64 * period).collect();
        let segments = vec![TempoSegment::new(0.0, seconds, 60.0 / period, 0.0, 4)];
        crate::grid::build_predictor_grid(&beats, &segments).unwrap()
    }

    #[test]
    fn the_stacked_feature_vector_is_the_expected_133_dimensions() {
        let settings = ModelSettings::default();
        let grid = grid_for(20.0, 0.5);
        let samples = testsig::parity_signal(settings.sample_rate, 20.0);
        let features = extract_song_features(&samples, &settings, &grid).unwrap();
        assert_eq!(features.timbre.rows(), 73, "48 mel + 20 mfcc + 5 contrast");
        assert_eq!(features.harmony.rows(), 18, "12 chroma + 6 tonnetz");
        assert_eq!(features.rhythm.rows(), 27, "24 subdivisions + 3 onset sums");
        assert_eq!(features.texture.rows(), 15);
        assert_eq!(features.stacked().rows(), 133);
        assert_eq!(features.stacked().cols(), grid.n_beats());
    }

    #[test]
    fn every_feature_is_finite_for_ordinary_audio() {
        let settings = ModelSettings::default();
        let grid = grid_for(12.0, 0.5);
        let samples = testsig::structured_signal(settings.sample_rate, 12.0, 4.0);
        let features = extract_song_features(&samples, &settings, &grid).unwrap();
        assert!(features.stacked().as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn digital_silence_produces_finite_features_rather_than_nan() {
        let settings = ModelSettings::default();
        let grid = grid_for(12.0, 0.5);
        let samples = vec![0.0f32; 12 * 22_050];
        let features = extract_song_features(&samples, &settings, &grid).unwrap();
        assert!(features.stacked().as_slice().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn a_beat_grid_running_past_the_audio_is_rejected() {
        let settings = ModelSettings::default();
        let grid = grid_for(60.0, 0.5);
        let samples = testsig::parity_signal(settings.sample_rate, 5.0);
        assert!(matches!(
            extract_song_features(&samples, &settings, &grid),
            Err(PhraseError::GridPastAudio { .. })
        ));
    }

    #[test]
    fn empty_audio_is_rejected() {
        let settings = ModelSettings::default();
        let grid = grid_for(12.0, 0.5);
        assert!(matches!(
            extract_song_features(&[], &settings, &grid),
            Err(PhraseError::EmptyAudio)
        ));
    }

    #[test]
    fn the_subdivision_profile_of_each_beat_sums_to_one() {
        let frame_times: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let envelope: Vec<f64> = (0..100).map(|i| (i % 7) as f64).collect();
        let edges = vec![0.0, 0.25, 0.5, 0.75];
        let profiles = subdivision_profiles(&[&envelope], &frame_times, &edges);
        for beat in 0..3 {
            let total: f64 = (0..BEAT_SUBDIVISIONS).map(|s| profiles.get(s, beat)).sum();
            assert!((total - 1.0).abs() < 1e-6, "beat {beat} summed to {total}");
        }
    }

    #[test]
    fn robust_z_falls_back_when_the_median_deviation_is_zero() {
        // A constant run has zero MAD; without the fallback this divides by 0.
        let z = robust_z(&[5.0, 5.0, 5.0, 5.0]);
        assert!(z.iter().all(|v| v.abs() < 1e-12));
        // Mostly-constant: MAD is 0 but the standard deviation is not.
        let z = robust_z(&[5.0, 5.0, 5.0, 9.0]);
        assert!(z.iter().all(|v| v.is_finite()));
    }
}
