//! Dump the pure-logic results as JSON, for comparison against the Python
//! implementation this crate replaces.
//!
//! Run with `cargo run -p mixlyzer-core --example parity_dump`; the companion
//! script `rust/parity/compare.py` produces the same structure from the Python
//! modules and diffs the two. Fields where the two are meant to disagree are
//! listed in the script rather than silently omitted here.

use mixlyzer_core::{
    beatgrid::Beatgrid,
    cue, key::Key, linear, phrase,
    phrase::Phrase,
    segments::TempoSegment,
};
use serde_json::{json, Value};

fn key_table() -> Value {
    let rows: Vec<Value> = (0..24)
        .map(|index| {
            let key = Key::from_index(index);
            json!({
                "index": index,
                "camelot": key.camelot(),
                "classical": key.classical(),
                "neighbours": key
                    .harmonic_neighbours()
                    .iter()
                    .map(|k| k.index())
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    Value::Array(rows)
}

fn bpm_segment_rows() -> Value {
    // Same tempo either side of a gap, a meter change, and a degenerate row.
    let segments = vec![
        TempoSegment::new(0.0, 10.0, 128.0, 0.0, 4),
        TempoSegment::new(10.0, 20.0, 128.0, 10.0, 4),
        TempoSegment::new(20.0, 20.0, 128.0, 20.0, 4),
        TempoSegment::new(20.0, 30.0, 128.0, 20.0, 3),
        TempoSegment::new(30.0, 45.0, 174.4, 30.0, 4),
    ];
    let rows: Vec<Value> = linear::build_bpm_segments(&segments)
        .into_iter()
        .map(|row| {
            json!({
                "seq_index": row.seq_index,
                "start_sec": row.start_sec,
                "end_sec": row.end_sec,
                "duration_sec": row.duration_sec,
                "bpm": row.bpm,
                "bpm_rounded": row.bpm_rounded,
                "time_signature": row.time_signature,
            })
        })
        .collect();
    Value::Array(rows)
}

fn phrase_fixture() -> Vec<Phrase> {
    vec![
        Phrase::new(0.0, 8.0, "INTRO"),
        Phrase::new(8.0, 24.0, "VERSE"),
        Phrase::new(24.0, 32.0, "FILL_IN"),
        Phrase::new(32.0, 48.0, "CHORUS"),
        Phrase::new(48.0, 64.0, "CHORUS"),
        Phrase::new(64.0, 80.0, "VERSE"),
        Phrase::new(80.0, 96.0, "INTERLUDE"),
        Phrase::new(96.0, 112.0, "CHORUS"),
        Phrase::new(112.0, 128.0, "OUTRO"),
    ]
}

fn phrase_results() -> Value {
    let phrases = phrase_fixture();
    json!({
        "numbered": phrase::numbered_labels(&phrases),
        "abbreviated": phrase::abbreviated_labels(&phrases),
        "merged_fills": phrase::merge_fills_for_display(&phrases)
            .iter()
            .map(|p| json!({"start": p.start, "end": p.end, "label": p.label}))
            .collect::<Vec<_>>(),
        "colors": phrase::PHRASE_LABELS
            .iter()
            .map(|label| {
                let (r, g, b) = phrase::phrase_color(label);
                json!({"label": label, "rgb": [r, g, b]})
            })
            .collect::<Vec<_>>(),
    })
}

fn cue_points() -> Value {
    let rows: Vec<Value> = cue::from_phrases(&phrase_fixture())
        .into_iter()
        .map(|point| {
            json!({
                "id": point.id,
                "time_sec": point.time_sec,
                "label": point.label,
                "comment": point.comment,
            })
        })
        .collect();
    Value::Array(rows)
}

fn downbeats() -> Value {
    // A tempo change part way through, with the second segment's bar phase
    // deliberately offset from the first.
    let period = 60.0 / 120.0;
    let mut beats: Vec<f64> = (0..32).map(|i| i as f64 * period).collect();
    let switch = 32.0 * period;
    let second = 60.0 / 140.0;
    beats.extend((0..32).map(|i| switch + i as f64 * second));

    let segments = vec![
        TempoSegment::new(0.0, switch, 120.0, 0.0, 4),
        TempoSegment::new(switch, switch + 32.0 * second, 140.0, switch + second, 4),
    ];
    let grid = Beatgrid::new(beats, segments);
    let probes = [0.0f64, 1.0, 4.5, 9.0, 20.0];
    let labels: Vec<String> = probes.iter().map(|t| grid.bar_beat_label(*t)).collect();
    json!({
        "downbeat_indices": grid.downbeat_indices(),
        "bar_beat_probe_times": probes,
        "bar_beat_labels": labels,
    })
}

fn main() {
    let document = json!({
        "keys": key_table(),
        "bpm_segment_rows": bpm_segment_rows(),
        "phrases": phrase_results(),
        "cue_points": cue_points(),
        "beatgrid": downbeats(),
    });
    println!("{}", serde_json::to_string_pretty(&document).unwrap());
}
