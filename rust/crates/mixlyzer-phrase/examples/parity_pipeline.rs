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
    let seconds = 48.0;
    let bpm = 120.0;
    let period = 60.0 / bpm;
    let sample_rate = 22_050u32;

    let signal = testsig::structured_signal(sample_rate, seconds, 8.0);
    let beats: Vec<f64> = (0..(seconds / period) as usize)
        .map(|i| i as f64 * period)
        .collect();
    let tempo = vec![TempoSegment::new(0.0, seconds, bpm, 0.0, 4)];

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
    out.insert("signal".into(), json!(signal));
    out.insert("beats".into(), json!(beats));
    out.insert(
        "tempo_segments".into(),
        json!([[0.0, seconds, bpm, 0.0, 4.0]]),
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

    let text = serde_json::to_string(&Value::Object(out)).expect("serialise dump");
    match std::env::args().nth(1) {
        Some(path) => std::fs::write(&path, text).expect("write dump"),
        None => println!("{text}"),
    }
}
