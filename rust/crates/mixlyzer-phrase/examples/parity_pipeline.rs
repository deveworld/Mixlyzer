//! Dump every stage of the detector as JSON, for `rust/parity/pipeline_parity.py`.
//!
//! Usage: `cargo run --release -p mixlyzer-phrase --example parity_pipeline -- <out.json>`
//!
//! Where `parity_dump` checks the frame-rate features against librosa, this
//! checks everything built on top of them against the Python detector: the beat
//! grid, the standardised beat features, the 1761-column boundary matrix, the
//! per-beat probabilities, the boundaries before and after refinement, the
//! per-segment features, the label log-probabilities and the final phrases.

use mixlyzer_core::TempoSegment;
use mixlyzer_phrase::features::matrix::Mat;
use mixlyzer_phrase::{boundary, features, grid, label, testsig, PhraseModel};
use serde_json::{json, Map, Value};

fn mat(m: &Mat) -> Value {
    json!({ "shape": [m.rows(), m.cols()], "data": m.as_slice() })
}

fn main() {
    let sample_rate = 22_050u32;
    let variant = std::env::args().nth(2).unwrap_or_else(|| "regular".to_string());
    // "regular" puts every section change exactly on a 16-beat boundary, which
    // the detector can hit without refinement. "irregular" uses an off-grid
    // tempo, a section length that is not a power of two and a downbeat that
    // is not beat zero, so the DP refinement has to do real work.
    let (seconds, bpm, section, inizio_beat) = match variant.as_str() {
        "irregular" => (52.0, 137.0, 6.3, 2usize),
        _ => (48.0, 120.0, 8.0, 0usize),
    };
    let period = 60.0 / bpm;

    let signal = testsig::structured_signal(sample_rate, seconds, section);
    let beats: Vec<f64> = (0..(seconds / period) as usize)
        .map(|i| i as f64 * period)
        .collect();
    let tempo = vec![TempoSegment::new(
        0.0,
        seconds,
        bpm,
        beats[inizio_beat],
        4,
    )];

    let path = mixlyzer_phrase::find_default_model(env!("CARGO_MANIFEST_DIR"))
        .expect("the shipped weights should be findable");
    let model = PhraseModel::load(&path).expect("load model");
    let settings = &model.settings;

    let grid = grid::build_predictor_grid(&beats, &tempo).expect("build grid");
    let acoustic =
        features::song::extract_song_features(&signal, settings, &grid).expect("features");
    let feature_z = boundary::feature_z(&acoustic.stacked());
    let context = boundary::grid_context(&grid, feature_z.rows());
    let boundary_features =
        boundary::boundary_feature_matrix(&feature_z, &settings.boundary_context_beats, &context);
    let valid = boundary::valid_mask(feature_z.rows(), settings.edge_beats);
    let probability = boundary::boundary_probability(&model.boundary, &boundary_features, &valid);
    let raw_bounds = boundary::pick_boundaries(
        &model.boundary,
        &boundary_features,
        &valid,
        &probability,
        settings.min_distance_beats,
        settings.max_boundaries,
    );
    let refined = boundary::refine_boundaries(
        &raw_bounds,
        &probability,
        &valid,
        &grid.downbeat_mask,
        settings,
    );
    let segment_features = label::segment_features(&feature_z, &refined);
    let log_probability = label::label_log_probabilities(&model.label, &segment_features);

    let python_options = label::LabelOptions {
        label_weight: settings.label_weight,
        transition_weight: settings.transition_weight,
        length_weight: settings.length_weight,
        // The Python decoder applies the end-state prior at full weight; this
        // dump has to reproduce that to be comparable.
        end_state_weight: settings.transition_weight,
    };
    let labels = label::decode_labels(&model, &log_probability, &refined, &python_options);

    let mut default_options = python_options;
    default_options.end_state_weight = 0.0;
    let labels_default =
        label::decode_labels(&model, &log_probability, &refined, &default_options);

    let mut out = Map::new();
    out.insert("sample_rate".into(), json!(sample_rate));
    out.insert("seconds".into(), json!(seconds));
    out.insert("bpm".into(), json!(bpm));
    out.insert("variant".into(), json!(variant));
    out.insert("signal".into(), json!(signal));
    out.insert("beats".into(), json!(beats));
    out.insert(
        "tempo_segments".into(),
        json!([[0.0, seconds, bpm, beats[inizio_beat], 4.0]]),
    );
    out.insert("downbeat_mask".into(), json!(grid.downbeat_mask));
    out.insert("beat_in_bar".into(), json!(grid.beat_in_bar));
    out.insert("bar_index_of_beat".into(), json!(grid.bar_index_of_beat));
    out.insert("beat_edges_sec".into(), json!(grid.beat_edges_sec));
    out.insert("family_timbre".into(), mat(&acoustic.timbre));
    out.insert("family_harmony".into(), mat(&acoustic.harmony));
    out.insert("family_rhythm".into(), mat(&acoustic.rhythm));
    out.insert("family_texture".into(), mat(&acoustic.texture));
    out.insert("feature_z".into(), mat(&feature_z));
    out.insert("grid_context".into(), mat(&context));
    out.insert("boundary_features".into(), mat(&boundary_features));
    out.insert("boundary_probability".into(), json!(probability));
    out.insert("raw_bounds".into(), json!(raw_bounds));
    out.insert("refined_bounds".into(), json!(refined));
    out.insert("segment_features".into(), mat(&segment_features));
    out.insert("label_log_probability".into(), mat(&log_probability));
    out.insert("labels".into(), json!(labels));
    out.insert("labels_default_options".into(), json!(labels_default));

    // The audio above rarely makes the refinement pass move anything, so probe
    // the dynamic program directly with a curve built to need it: confident
    // peaks a few beats off the downbeat grid, and phrase lengths that only
    // become powers of two after the shift.
    let probe_n = 160usize;
    let probe_downbeats: Vec<bool> = (0..probe_n).map(|b| b % 4 == 0).collect();
    let probe_valid = boundary::valid_mask(probe_n, settings.edge_beats);
    let probe_probability: Vec<f64> = (0..probe_n)
        .map(|b| match b {
            30 => 0.71,
            32 => 0.66,
            35 => 0.55,
            61 => 0.60,
            64 => 0.58,
            96 => 0.80,
            123 => 0.52,
            128 => 0.49,
            _ => 0.02 + 0.01 * ((b % 7) as f64),
        })
        .collect();
    let probe_raw = vec![0usize, 30, 61, 96, 123, probe_n];
    let probe_refined = boundary::refine_boundaries(
        &probe_raw,
        &probe_probability,
        &probe_valid,
        &probe_downbeats,
        settings,
    );
    out.insert("probe_probability".into(), json!(probe_probability));
    out.insert("probe_valid".into(), json!(probe_valid));
    out.insert("probe_downbeats".into(), json!(probe_downbeats));
    out.insert("probe_raw".into(), json!(probe_raw));
    out.insert("probe_refined".into(), json!(probe_refined));

    let text = serde_json::to_string(&Value::Object(out)).expect("serialise dump");
    match std::env::args().nth(1) {
        Some(path) => std::fs::write(&path, text).expect("write dump"),
        None => println!("{text}"),
    }
}
