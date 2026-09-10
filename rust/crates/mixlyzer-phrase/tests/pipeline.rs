//! End-to-end behaviour of the detector, against the shipped weights.

use mixlyzer_core::TempoSegment;
use mixlyzer_phrase::features::matrix::Mat;
use mixlyzer_phrase::label::{decode_labels, LabelOptions};
use mixlyzer_phrase::{detect_phrases, testsig, PhraseError, PhraseModel, PhraseOptions};

fn model() -> PhraseModel {
    let path = mixlyzer_phrase::find_default_model(env!("CARGO_MANIFEST_DIR"))
        .expect("the shipped weights should be findable from the crate directory");
    PhraseModel::load(path).expect("load the shipped weights")
}

fn beats_at(bpm: f64, seconds: f64) -> Vec<f64> {
    let period = 60.0 / bpm;
    (0..(seconds / period) as usize).map(|i| i as f64 * period).collect()
}

/// Log-probabilities the label ensemble might plausibly produce, given a
/// confident guess for each segment.
fn scores(model: &PhraseModel, guesses: &[(&str, f64)]) -> Mat {
    let n = model.labels.len();
    let mut mat = Mat::zeros(guesses.len(), n);
    for (row, (label, confidence)) in guesses.iter().enumerate() {
        let rest = (1.0 - confidence) / (n - 1) as f64;
        for c in 0..n {
            let p = if model.labels[c] == *label { *confidence } else { rest };
            mat.set(row, c, p.clamp(1e-7, 1.0).ln());
        }
    }
    mat
}

fn options(end_state_weight: f64) -> LabelOptions {
    LabelOptions {
        label_weight: 1.0,
        transition_weight: 1.0,
        length_weight: 0.0,
        end_state_weight,
    }
}

/// The correction this port makes on purpose, pinned against the real weights.
///
/// The shipped transition matrix ends a track in SILENCE with probability
/// 0.863 and in OUTRO with probability 0.0026, because every training
/// annotation carries a trailing silence segment that the production boundary
/// detector never emits. Left at full weight the prior is worth up to 7.26
/// nats and simply overrules the label ensemble on the final segment.
#[test]
fn a_final_outro_survives_with_the_default_end_state_weight() {
    let model = model();
    let logp = scores(&model, &[("INTRO", 0.9), ("CHORUS", 0.9), ("OUTRO", 0.72)]);
    let bounds = [0usize, 32, 96, 160];

    let decoded = decode_labels(&model, &logp, &bounds, &options(0.0));
    assert_eq!(decoded, vec!["INTRO", "CHORUS", "OUTRO"]);
}

#[test]
fn the_end_state_prior_at_full_weight_rewrites_the_final_outro_to_silence() {
    let model = model();
    let logp = scores(&model, &[("INTRO", 0.9), ("CHORUS", 0.9), ("OUTRO", 0.72)]);
    let bounds = [0usize, 32, 96, 160];

    let decoded = decode_labels(&model, &logp, &bounds, &options(1.0));
    assert_eq!(
        decoded.last().map(String::as_str),
        Some("SILENCE"),
        "this is the Python behaviour the default deliberately avoids"
    );
    assert_ne!(
        decoded[1], "CHORUS",
        "the joint decode drags the middle segment along with it too"
    );
}

#[test]
fn the_end_state_gap_in_the_shipped_weights_is_as_documented() {
    let model = model();
    let index = |name: &str| model.labels.iter().position(|l| l == name).expect(name);
    let end = model.end_state();
    let outro = model.transition[index("OUTRO")][end];
    let silence = model.transition[index("SILENCE")][end];
    assert!((outro.exp() - 0.0026).abs() < 5e-4, "P(END|OUTRO) = {}", outro.exp());
    assert!((silence.exp() - 0.863).abs() < 5e-3, "P(END|SILENCE) = {}", silence.exp());
    // The worst case adds the transition into the final label to the gap.
    let worst = (0..model.labels.len())
        .map(|from| {
            silence - outro + model.transition[from][index("SILENCE")]
                - model.transition[from][index("OUTRO")]
        })
        .fold(f64::NEG_INFINITY, f64::max);
    assert!((worst - 7.26).abs() < 0.05, "worst-case swing was {worst} nats");
}

#[test]
fn structured_audio_produces_phrases_that_tile_the_track_in_order() {
    let bpm = 120.0;
    let seconds = 32.0;
    let beats = beats_at(bpm, seconds);
    let tempo = vec![TempoSegment::new(0.0, seconds, bpm, 0.0, 4)];
    let samples = testsig::structured_signal(22_050, seconds, 8.0);
    let options = PhraseOptions::new(model());

    let phrases = detect_phrases(&samples, 22_050, &beats, &tempo, &options).unwrap();
    assert!(!phrases.is_empty(), "64 beats of obvious structure should segment");
    assert!(
        phrases.windows(2).all(|w| w[0].end <= w[1].start),
        "phrases must not overlap: {phrases:?}"
    );
    assert!(phrases.windows(2).all(|w| (w[0].end - w[1].start).abs() < 1e-9));
    assert_eq!(phrases[0].start, 0.0, "the first phrase starts at beat 0");
    assert!(phrases.iter().all(|p| p.duration() > 0.0));
    assert!(
        phrases
            .iter()
            .all(|p| mixlyzer_core::phrase::PHRASE_LABELS.contains(&p.label.as_str())),
        "every label must be one the rest of the app knows: {phrases:?}"
    );
}

#[test]
fn the_detector_finds_boundaries_near_the_real_section_changes() {
    let bpm = 120.0;
    let seconds = 48.0;
    let section = 8.0;
    let beats = beats_at(bpm, seconds);
    let tempo = vec![TempoSegment::new(0.0, seconds, bpm, 0.0, 4)];
    let samples = testsig::structured_signal(22_050, seconds, section);
    let options = PhraseOptions::new(model());

    let phrases = detect_phrases(&samples, 22_050, &beats, &tempo, &options).unwrap();
    // Every interior boundary should sit within a beat of a section change.
    for phrase in phrases.iter().skip(1) {
        let offset = (phrase.start / section).round() * section;
        assert!(
            (phrase.start - offset).abs() <= 60.0 / bpm,
            "boundary at {:.3}s is not near a section change",
            phrase.start
        );
    }
}

#[test]
fn a_track_whose_beat_grid_outruns_the_audio_is_refused() {
    let beats = beats_at(120.0, 120.0);
    let tempo = vec![TempoSegment::new(0.0, 120.0, 120.0, 0.0, 4)];
    let samples = testsig::structured_signal(22_050, 10.0, 4.0);
    let options = PhraseOptions::new(model());
    assert!(matches!(
        detect_phrases(&samples, 22_050, &beats, &tempo, &options),
        Err(PhraseError::GridPastAudio { .. })
    ));
}

#[test]
fn a_downbeat_that_matches_no_beat_is_refused_before_any_audio_is_touched() {
    let bpm = 120.0;
    let beats = beats_at(bpm, 32.0);
    // 0.25s is half a beat away from every beat at 120 BPM.
    let tempo = vec![TempoSegment::new(0.0, 32.0, bpm, 0.25, 4)];
    let options = PhraseOptions::new(model());
    assert!(matches!(
        detect_phrases(&[], 22_050, &beats, &tempo, &options),
        Err(PhraseError::DownbeatNotOnBeat { .. })
    ));
}
