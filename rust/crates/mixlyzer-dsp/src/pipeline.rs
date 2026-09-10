//! The whole analysis, from a file on disk to a musical description.
//!
//! Stages run in the order each one's inputs become available: decode, then
//! envelopes and onsets from the samples, then tempo from the onsets, then key
//! from the samples and the beats. Every stage that can fail says which one it
//! was, so a track that cannot be analysed reports a reason rather than a
//! traceback from somewhere inside the numerics.

use mixlyzer_core::beatgrid::Beatgrid;
use mixlyzer_core::config::AnalysisConfig;
use mixlyzer_core::jumpcue::JumpCueGraph;
use mixlyzer_core::key::Key;
use mixlyzer_core::segments::{KeySegment, TempoSegment};

use crate::envelope::{self, Band, Envelopes};
use crate::error::AnalysisError;
use crate::jumpcue_detect::{self, JumpCueOptions};
use crate::key as key_stage;
use crate::onset::{self, OnsetOptions};
use crate::tempo::{self, TempoOptions};

/// Everything the analysis found out about a track.
#[derive(Debug, Clone)]
pub struct Analysis {
    pub duration_sec: f64,
    /// Sample rate the analysis ran at, not the file's own rate.
    pub analysis_sample_rate: u32,
    /// Beats and the tempo structure that organises them into bars.
    pub beatgrid: Beatgrid,
    /// Single representative tempo, for display and for sorting a library.
    pub tempo_global: f64,
    pub key_segments: Vec<KeySegment>,
    /// One key for the whole track.
    pub overall_key: Option<Key>,
    /// Band energies for drawing the waveform.
    pub envelopes: Envelopes,
    /// How strongly the onsets agreed with the chosen grid, `0..=1`.
    pub beat_confidence: f64,
    /// Regions of the track that sound alike, and the jumps between them.
    pub jump_cues: JumpCueGraph,
}

impl Analysis {
    pub fn beats(&self) -> &[f64] {
        self.beatgrid.beats()
    }

    pub fn tempo_segments(&self) -> &[TempoSegment] {
        self.beatgrid.segments()
    }

    /// Times of the bar starts.
    pub fn downbeats(&self) -> Vec<f64> {
        self.beatgrid.downbeat_times()
    }
}

/// Shortest track the tempo stage can say anything useful about.
///
/// Below a few seconds there is not enough of the onset envelope to
/// autocorrelate: the estimate would be an artefact of the window, not a tempo.
const MIN_ANALYSIS_SECONDS: f64 = 4.0;

/// Analyse already-decoded mono samples.
pub fn analyze_samples(
    samples: &[f32],
    sample_rate: u32,
    config: &AnalysisConfig,
) -> Result<Analysis, AnalysisError> {
    let duration_sec = if sample_rate == 0 {
        0.0
    } else {
        samples.len() as f64 / f64::from(sample_rate)
    };
    if duration_sec < MIN_ANALYSIS_SECONDS {
        return Err(AnalysisError::TooShort {
            duration_sec,
            needed_sec: MIN_ANALYSIS_SECONDS,
        });
    }

    let envelopes = envelope::compute(
        samples,
        sample_rate,
        config.env_hop_samples(),
        [
            Band::new(config.env_lo.0, config.env_lo.1),
            Band::new(config.env_mid.0, config.env_mid.1),
            Band::new(config.env_hi.0, config.env_hi.1),
        ],
        config.env_order,
    );

    let onsets = onset::compute(
        samples,
        sample_rate,
        OnsetOptions {
            hop: config.bpm_hop_length,
            ..OnsetOptions::default()
        },
    );

    let (bpm_min, bpm_max) = config.bpm_range();
    let tempo_options = TempoOptions {
        bpm_min,
        bpm_max,
        ..TempoOptions::default()
    };

    let segments = if config.bpm_dynamic {
        tempo::estimate_segments(
            &onsets,
            tempo_options,
            config.bpm_win_length as f64 / 1000.0,
            duration_sec,
        )?
    } else {
        let estimate = tempo::estimate(&onsets, tempo_options)?;
        vec![TempoSegment::new(
            0.0,
            duration_sec,
            estimate.bpm,
            estimate.phase_sec,
            4,
        )]
    };

    let beat_confidence = tempo::estimate(&onsets, tempo_options)
        .map(|estimate| estimate.confidence)
        .unwrap_or(0.0);

    let beats = beats_from_segments(&segments, duration_sec, config.beatgrid_offset_sec());
    let beatgrid = Beatgrid::new(beats, segments);

    // The representative tempo is the one covering the most time, not the mean:
    // a track that spends four minutes at 128 and ten seconds at 90 is a 128
    // BPM track, and averaging would call it neither.
    let tempo_global = dominant_tempo(beatgrid.segments()).unwrap_or(0.0);

    let chroma = key_stage::chroma(samples, sample_rate, key_stage::ChromaOptions::default());
    let synced = key_stage::beat_synchronous(&chroma, beatgrid.beats());
    let path = key_stage::decode_key_path(&synced, key_stage::KeyOptions::default());
    let key_segments = key_stage::segments_from_path(&synced, &path, duration_sec);
    let overall_key = key_stage::overall_key(&chroma);

    // A track with nothing that repeats has no jump cues; that is an empty
    // graph rather than a failure, so it never costs the rest of the analysis.
    let jump_cues = jumpcue_detect::detect(
        samples,
        sample_rate,
        beatgrid.beats(),
        JumpCueOptions::default(),
    )?;

    Ok(Analysis {
        duration_sec,
        analysis_sample_rate: sample_rate,
        beatgrid,
        tempo_global,
        key_segments,
        overall_key,
        envelopes,
        beat_confidence,
        jump_cues,
    })
}

/// Decode a file and analyse it.
pub fn analyze_file(
    path: impl AsRef<std::path::Path>,
    config: &AnalysisConfig,
) -> Result<Analysis, AnalysisError> {
    let rate = config.analysis_samp_rate;
    let samples = crate::decode::decode_for_analysis(path, rate)?;
    analyze_samples(&samples, rate, config)
}

/// Lay beats across every segment, each from its own downbeat at its own tempo.
///
/// Beats are generated per segment rather than by resampling one global grid,
/// so a tempo change lands exactly on the segment boundary instead of drifting
/// through it.
fn beats_from_segments(segments: &[TempoSegment], duration_sec: f64, offset_sec: f64) -> Vec<f64> {
    let mut beats: Vec<f64> = Vec::new();
    for segment in segments {
        let period = segment.beat_period();
        if !period.is_finite() || period <= 0.0 {
            continue;
        }
        // Wind the reference downbeat back to the segment start so the first
        // beat of the segment is included.
        let anchor = segment.inizio;
        let steps = ((segment.start - anchor) / period).floor();
        let mut t = anchor + steps * period;
        while t < segment.start - 1e-9 {
            t += period;
        }
        while t < segment.end - 1e-9 {
            let shifted = t + offset_sec;
            if shifted >= 0.0 && shifted <= duration_sec {
                beats.push(shifted);
            }
            t += period;
        }
    }
    beats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Two segments can each contribute a beat at their shared boundary.
    beats.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    beats
}

/// The tempo that covers the most time.
fn dominant_tempo(segments: &[TempoSegment]) -> Option<f64> {
    segments
        .iter()
        .max_by(|a, b| {
            a.duration()
                .partial_cmp(&b.duration())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|segment| segment.bpm)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 22_050;

    fn config() -> AnalysisConfig {
        AnalysisConfig {
            analysis_samp_rate: RATE,
            ..AnalysisConfig::default()
        }
    }

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
            for offset in 0..128.min(n - at) {
                let decay = (1.0 - offset as f32 / 128.0).powi(2);
                let noise = if (at + offset) % 3 == 0 { 1.0 } else { -0.7 };
                signal[at + offset] = decay * noise;
            }
            beat += 1;
        }
        signal
    }

    #[test]
    fn a_click_track_analyses_to_its_own_tempo() {
        let analysis = analyze_samples(&click_track(128.0, 30.0), RATE, &config()).unwrap();
        assert!(
            (analysis.tempo_global - 128.0).abs() < 1.0,
            "estimated {} BPM",
            analysis.tempo_global
        );
        assert!((analysis.duration_sec - 30.0).abs() < 0.01);
        assert_eq!(analysis.analysis_sample_rate, RATE);
    }

    #[test]
    fn the_beatgrid_spans_the_track_at_the_beat_period() {
        let analysis = analyze_samples(&click_track(120.0, 30.0), RATE, &config()).unwrap();
        let beats = analysis.beats();
        assert!(beats.len() > 50, "only {} beats", beats.len());
        assert!(beats[0] >= 0.0);
        assert!(*beats.last().unwrap() <= analysis.duration_sec + 1e-6);
        let gaps: Vec<f64> = beats.windows(2).map(|w| w[1] - w[0]).collect();
        let mean_gap = gaps.iter().sum::<f64>() / gaps.len() as f64;
        assert!(
            (mean_gap - 0.5).abs() < 0.02,
            "mean beat gap {mean_gap}s, expected 0.5s"
        );
    }

    #[test]
    fn beats_land_on_the_clicks() {
        let bpm = 120.0;
        let analysis = analyze_samples(&click_track(bpm, 30.0), RATE, &config()).unwrap();
        let period = 60.0 / bpm;
        // Every beat should sit near a multiple of the click period.
        let worst = analysis
            .beats()
            .iter()
            .map(|beat| {
                let offset = (beat / period).fract();
                offset.min(1.0 - offset) * period
            })
            .fold(0.0f64, f64::max);
        assert!(worst < 0.05, "worst beat was {worst:.4}s off the clicks");
    }

    #[test]
    fn downbeats_are_a_subset_of_the_beats_every_four_beats() {
        let analysis = analyze_samples(&click_track(128.0, 30.0), RATE, &config()).unwrap();
        let beats = analysis.beats();
        let downbeats = analysis.downbeats();
        assert!(!downbeats.is_empty());
        for downbeat in &downbeats {
            assert!(
                beats.iter().any(|b| (b - downbeat).abs() < 1e-9),
                "downbeat {downbeat} is not on a beat"
            );
        }
        // Four beats to a bar by default.
        assert!(
            (beats.len() as f64 / downbeats.len() as f64 - 4.0).abs() < 0.5,
            "{} beats over {} bars",
            beats.len(),
            downbeats.len()
        );
    }

    #[test]
    fn a_short_track_is_reported_as_too_short() {
        let err = analyze_samples(&click_track(128.0, 1.0), RATE, &config()).unwrap_err();
        assert!(matches!(err, AnalysisError::TooShort { .. }));
        assert!(err.to_string().contains("shorter than"));
    }

    #[test]
    fn silence_is_reported_as_silence() {
        let silence = vec![0.0f32; RATE as usize * 10];
        assert!(matches!(
            analyze_samples(&silence, RATE, &config()),
            Err(AnalysisError::Silent)
        ));
    }

    #[test]
    fn a_static_tempo_run_produces_exactly_one_segment() {
        let cfg = AnalysisConfig {
            bpm_dynamic: false,
            ..config()
        };
        let analysis = analyze_samples(&click_track(128.0, 30.0), RATE, &cfg).unwrap();
        assert_eq!(analysis.tempo_segments().len(), 1);
        assert_eq!(analysis.tempo_segments()[0].start, 0.0);
    }

    #[test]
    fn the_grid_offset_shifts_every_beat() {
        let base = analyze_samples(&click_track(120.0, 20.0), RATE, &config()).unwrap();
        let shifted_cfg = AnalysisConfig {
            beatgrid_offset_msec: 50.0,
            ..config()
        };
        let shifted = analyze_samples(&click_track(120.0, 20.0), RATE, &shifted_cfg).unwrap();
        let common = base.beats().len().min(shifted.beats().len()) - 1;
        let deltas: Vec<f64> = (1..common)
            .map(|i| shifted.beats()[i] - base.beats()[i])
            .collect();
        let mean = deltas.iter().sum::<f64>() / deltas.len() as f64;
        assert!(
            (mean - 0.05).abs() < 0.01,
            "expected a 50 ms shift, measured {mean:.4}s"
        );
    }

    #[test]
    fn envelopes_are_produced_alongside_the_grid() {
        let analysis = analyze_samples(&click_track(128.0, 20.0), RATE, &config()).unwrap();
        assert!(!analysis.envelopes.is_empty());
        assert_eq!(analysis.envelopes.len(), analysis.envelopes.min.len());
        assert!(analysis.envelopes.low.iter().any(|v| *v > 0.0));
    }

    #[test]
    fn confidence_is_reported_and_bounded() {
        let analysis = analyze_samples(&click_track(128.0, 20.0), RATE, &config()).unwrap();
        assert!((0.0..=1.0).contains(&analysis.beat_confidence));
        assert!(
            analysis.beat_confidence > 0.0,
            "a clean click track should not score zero"
        );
    }

    #[test]
    fn key_segments_come_back_for_tonal_material() {
        // A sustained C major triad under a click track.
        let clicks = click_track(120.0, 20.0);
        let mut signal = Vec::with_capacity(clicks.len());
        for (i, click) in clicks.iter().enumerate() {
            let t = i as f64 / f64::from(RATE);
            let tone: f64 = [261.63, 329.63, 392.0]
                .iter()
                .map(|f| (2.0 * std::f64::consts::PI * f * t).sin() * 0.2)
                .sum();
            signal.push(click * 0.5 + tone as f32);
        }
        let analysis = analyze_samples(&signal, RATE, &config()).unwrap();
        assert!(!analysis.key_segments.is_empty());
        assert!(analysis.overall_key.is_some());
        assert!(analysis
            .key_segments
            .iter()
            .all(|segment| segment.duration() > 0.0));
    }

    #[test]
    fn beats_generated_per_segment_meet_exactly_at_the_boundary() {
        let segments = vec![
            TempoSegment::new(0.0, 10.0, 120.0, 0.0, 4),
            TempoSegment::new(10.0, 20.0, 140.0, 10.0, 4),
        ];
        let beats = beats_from_segments(&segments, 20.0, 0.0);
        assert!(beats.windows(2).all(|w| w[1] > w[0]), "beats must ascend");
        // No duplicate at the shared boundary.
        assert!(
            beats.windows(2).all(|w| (w[1] - w[0]) > 1e-6),
            "a beat was emitted twice at the segment boundary"
        );
        // The first segment's beats run at 0.5s, the second's at ~0.4286s.
        let first_gap = beats[1] - beats[0];
        let last_gap = beats[beats.len() - 1] - beats[beats.len() - 2];
        assert!((first_gap - 0.5).abs() < 1e-6);
        assert!((last_gap - 60.0 / 140.0).abs() < 1e-6);
    }

    #[test]
    fn beats_outside_the_track_are_dropped() {
        let segments = vec![TempoSegment::new(0.0, 10.0, 120.0, 0.0, 4)];
        let beats = beats_from_segments(&segments, 5.0, 0.0);
        assert!(beats.iter().all(|b| *b <= 5.0));
    }

    #[test]
    fn a_negative_offset_does_not_produce_beats_before_zero() {
        let segments = vec![TempoSegment::new(0.0, 10.0, 120.0, 0.0, 4)];
        let beats = beats_from_segments(&segments, 10.0, -0.2);
        assert!(beats.iter().all(|b| *b >= 0.0));
    }

    #[test]
    fn the_representative_tempo_is_the_one_that_lasts_longest() {
        let segments = vec![
            TempoSegment::new(0.0, 10.0, 90.0, 0.0, 4),
            TempoSegment::new(10.0, 250.0, 128.0, 10.0, 4),
        ];
        assert_eq!(dominant_tempo(&segments), Some(128.0));
        assert_eq!(dominant_tempo(&[]), None);
    }

    /// One plucked note per beat, so the track has both a beat to find and a
    /// spectrum that says which section is playing.
    fn pitched_track(pitches: &[f64], beat_sec: f64) -> Vec<f32> {
        let period = (beat_sec * f64::from(RATE)) as usize;
        let mut samples = vec![0.0f32; pitches.len() * period];
        for (beat, hz) in pitches.iter().enumerate() {
            for offset in 0..period {
                let t = offset as f64 / f64::from(RATE);
                let decay = (-6.0 * t / beat_sec).exp();
                let tone = (2.0 * std::f64::consts::PI * hz * t).sin() * decay;
                let click = if offset < 64 {
                    1.0 - offset as f64 / 64.0
                } else {
                    0.0
                };
                samples[beat * period + offset] = (0.7 * tone + 0.5 * click) as f32;
            }
        }
        samples
    }

    #[test]
    fn a_returning_section_becomes_a_pair_of_jump_cues() {
        // Eight bars of A, eight of B, then A again, at 120 BPM.
        let section_a: Vec<f64> = (0..32)
            .map(|i: usize| {
                let mixed = (i as u64).wrapping_mul(2_654_435_761) ^ 0x9E37_79B9;
                110.0 * 2f64.powf((mixed >> 11) as f64 % 36.0 / 12.0)
            })
            .collect();
        let section_b: Vec<f64> = section_a.iter().map(|hz| hz * 2.0).collect();
        let pitches: Vec<f64> = section_a
            .iter()
            .chain(&section_b)
            .chain(&section_a)
            .copied()
            .collect();
        let analysis = analyze_samples(&pitched_track(&pitches, 0.5), RATE, &config()).unwrap();

        let cues = analysis.jump_cues.cues();
        assert_eq!(
            cues.len(),
            2,
            "expected the two ends of the A repeat: {cues:#?}"
        );
        assert!(analysis.jump_cues.validate_labels().is_ok());
        assert_eq!(analysis.jump_cues.components().len(), 1);
        assert!(
            (cues[0].point - 0.0).abs() < 18.0
                && (cues[1].point - cues[0].point - 32.0).abs() < 3.0,
            "cues at {} and {} do not describe a 32s repeat",
            cues[0].point,
            cues[1].point
        );
    }

    #[test]
    fn a_track_with_nothing_that_repeats_has_no_jump_cues() {
        let analysis = analyze_samples(&click_track(128.0, 30.0), RATE, &config()).unwrap();
        // Every beat of a click track is the same beat, so no stretch of it is
        // a repeat of any other in particular.
        assert!(analysis.jump_cues.is_empty());
        assert!(analysis.jump_cues.links().is_empty());
    }

    #[test]
    fn a_missing_file_surfaces_as_a_decode_error() {
        let err = analyze_file("/nonexistent/track.flac", &config()).unwrap_err();
        assert!(matches!(err, AnalysisError::Decode(_)));
    }
}
