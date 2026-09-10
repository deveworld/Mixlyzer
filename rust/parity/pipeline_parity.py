#!/usr/bin/env python3
"""Check the whole Rust detector against the Python one, stage by stage.

Run from the repository root:

    .venv/bin/python rust/parity/pipeline_parity.py

`phrase_parity.py` proves the frame-rate features match librosa. This proves
everything built on top of them matches `analyzer_core.cue_and_phrase`: the
beat grid, the four feature families, the standardised beat matrix, the
1761-column boundary matrix, the per-beat probabilities, the boundaries before
and after DP refinement, the per-segment features, the label
log-probabilities, and the labels themselves.

The boundary indices and the labels are discrete: they match exactly or the
port is wrong. The float stages carry the float32 error inherited from
librosa, amplified by the standardisation steps (dividing by a MAD that is
itself uncertain), so they get an allclose-style tolerance instead.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# (absolute, relative) tolerance per stage, and why.
TOLERANCE = {
    "family_timbre": (1e-3, 1e-5, "mel/MFCC/contrast synced to beats"),
    "family_harmony": (1e-4, 1e-5, "chroma and tonnetz synced to beats"),
    "family_rhythm": (1e-4, 1e-5, "onset profiles, normalised within each beat"),
    "family_texture": (1e-4, 1e-5, "log energies and spectral shape"),
    "feature_z": (1e-3, 1e-4, "each family row divided by its own MAD"),
    "grid_context": (1e-9, 1e-9, "derived from beat indices, so exact"),
    # A handful of the 1761 columns are near-constant across beats, so their
    # MAD is tiny and dividing by it amplifies librosa's float32 noise by ~1e6.
    # Those columns reach values in the thousands; the tolerance has to be
    # relative to that, not to a standardised feature's usual +/-3.
    "boundary_features": (1e-2, 1e-3, "differences of means, then standardised per column"),
    "boundary_probability": (1e-6, 1e-6, "sigmoid of a sum of 360 leaf values"),
    "segment_features": (1e-2, 1e-4, "block statistics of feature_z"),
    "label_log_probability": (1e-6, 1e-6, "softmax over 360 trees per class"),
}


VARIANTS = ("regular", "irregular")


def rust_dump(variant: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "pipeline.json"
        subprocess.run(
            [
                "cargo", "run", "-q", "--release", "-p", "mixlyzer-phrase",
                "--example", "parity_pipeline", "--", str(out), variant,
            ],
            cwd=REPO / "rust",
            check=True,
        )
        return json.loads(out.read_text())


def as_matrix(entry) -> np.ndarray:
    if isinstance(entry, list):
        return np.asarray(entry, dtype=np.float64).reshape(1, -1)
    rows, cols = entry["shape"]
    return np.asarray(entry["data"], dtype=np.float64).reshape(rows, cols)


def python_stages(dump: dict) -> dict[str, np.ndarray]:
    from analyzer_core.cue_and_phrase import phrase_analyzer as pa
    from analyzer_core.cue_and_phrase import structure

    model_path = REPO / "assets/weights/phrase_analyzer.npz"
    artifact = pa.load_two_stage_model(str(model_path.resolve()))
    settings = dict(artifact["settings"])

    y = np.asarray(dump["signal"], dtype=np.float32)
    beats = np.asarray(dump["beats"], dtype=np.float64)
    tempo = np.asarray(dump["tempo_segments"], dtype=np.float64)

    grid = structure.build_predictor_grid(beats, tempo)
    acoustic = structure.extract_song_features(
        None,
        grid,
        pa._parse_feature_config(settings),
        audio_array=y,
        audio_sr=int(dump["sample_rate"]),
    )
    feature_z = pa._feature_z_from_acoustic(acoustic)
    context = pa._grid_context_feature_matrix(feature_z, grid)
    boundary_features = pa._boundary_feature_matrix(
        feature_z, settings.get("boundary_context_beats", (1, 2, 4, 8, 16)), context
    )
    valid = pa._valid_mask(feature_z.shape[0], int(settings.get("edge_beats", 8)))
    clf = artifact["boundary_clf"]
    probability = pa._predict_boundary_probability(clf, boundary_features, valid)
    raw_bounds = pa._pick_boundaries_direct(
        clf,
        boundary_features,
        valid,
        probability,
        min_distance_beats=int(settings.get("min_distance_beats", 16)),
        max_boundaries=settings.get("max_boundaries"),
    )
    refined = pa._refine_boundaries(
        raw_bounds,
        probability,
        valid,
        np.asarray(grid.downbeat_mask, dtype=bool),
        window_beats=int(settings.get("boundary_refine_window_beats", 8)),
        target_lengths_beats=settings.get("boundary_lengths_beats", (16, 32, 64, 128)),
        length_weight=float(settings.get("boundary_length_weight", 0.45)),
        shift_penalty=float(settings.get("boundary_shift_penalty", 0.015)),
        downbeat_bonus=float(settings.get("boundary_downbeat_bonus", 0.35)),
    )
    probe_refined = pa._refine_boundaries(
        np.asarray(dump["probe_raw"], dtype=np.int32),
        np.asarray(dump["probe_probability"], dtype=np.float64),
        np.asarray(dump["probe_valid"], dtype=bool),
        np.asarray(dump["probe_downbeats"], dtype=bool),
        window_beats=int(settings.get("boundary_refine_window_beats", 8)),
        target_lengths_beats=settings.get("boundary_lengths_beats", (16, 32, 64, 128)),
        length_weight=float(settings.get("boundary_length_weight", 0.45)),
        shift_penalty=float(settings.get("boundary_shift_penalty", 0.015)),
        downbeat_bonus=float(settings.get("boundary_downbeat_bonus", 0.35)),
    )

    segment_features = pa._segment_features(feature_z, refined)
    log_probability = pa._label_logp(artifact["label_clf"], feature_z, refined)
    labels = pa._decode_labels(
        list(artifact["label_labels"]),
        np.asarray(artifact["label_transition"], dtype=np.float64),
        np.asarray(artifact["label_length_mu"], dtype=np.float64),
        np.asarray(artifact["label_length_sigma"], dtype=np.float64),
        log_probability,
        refined,
        label_weight=float(settings.get("label_weight", 1.0)),
        transition_weight=float(settings.get("transition_weight", 1.0)),
        length_weight=float(settings.get("length_weight", 0.0)),
    )

    return {
        "downbeat_mask": np.asarray(grid.downbeat_mask, dtype=bool),
        "beat_in_bar": np.asarray(grid.beat_in_bar, dtype=np.int64),
        "bar_index_of_beat": np.asarray(grid.bar_index_of_beat, dtype=np.int64),
        "beat_edges_sec": np.asarray(grid.beat_edges_sec, dtype=np.float64),
        "family_timbre": acoustic.family_beat["timbre"],
        "family_harmony": acoustic.family_beat["harmony"],
        "family_rhythm": acoustic.family_beat["rhythm"],
        "family_texture": acoustic.family_beat["texture"],
        "feature_z": feature_z,
        "grid_context": context,
        "boundary_features": boundary_features,
        "boundary_probability": probability,
        "raw_bounds": raw_bounds,
        "refined_bounds": refined,
        "probe_refined": probe_refined,
        "segment_features": segment_features,
        "label_log_probability": log_probability,
        "labels": labels,
    }


def compare(name: str, rust: np.ndarray, expected: np.ndarray) -> tuple[bool, str]:
    expected = np.atleast_2d(np.asarray(expected, dtype=np.float64))
    if rust.shape != expected.shape:
        return False, f"shape {rust.shape} != {expected.shape}"
    abs_tol, rel_tol, reason = TOLERANCE[name]
    diff = np.abs(rust - expected)
    allowed = abs_tol + rel_tol * np.abs(expected)
    bad = diff > allowed
    max_abs = float(np.max(diff)) if diff.size else 0.0
    detail = f"max_abs={max_abs:.2e} tol=({abs_tol:.0e},{rel_tol:.0e})"
    if bad.any():
        worst = np.unravel_index(int(np.argmax(diff - allowed)), diff.shape)
        detail += f" fails={int(bad.sum())}/{bad.size} worst@{tuple(int(x) for x in worst)}"
        detail += f" rust={rust[worst]:.9g} python={expected[worst]:.9g}"
        return False, f"{detail}  [{reason}]"
    return True, f"{detail}  [{reason}]"


def main() -> int:
    failures = 0
    for variant in VARIANTS:
        failures += check(variant)
    print(f"\n{failures} stage(s) outside tolerance")
    return 1 if failures else 0


def check(variant: str) -> int:
    dump = rust_dump(variant)
    expected = python_stages(dump)
    failures = 0
    print(f"\n######## signal: {variant} ########")

    def report(ok: bool, name: str, detail: str) -> None:
        nonlocal failures
        print(f"  {'ok  ' if ok else 'FAIL'} {name:24s} {detail}")
        failures += 0 if ok else 1

    print("=== grid (must be exact) ===")
    for name, cast in (
        ("downbeat_mask", bool),
        ("beat_in_bar", np.int64),
        ("bar_index_of_beat", np.int64),
    ):
        got = np.asarray(dump[name], dtype=cast)
        report(np.array_equal(got, expected[name]), name, f"{got.size} beats")
    edges_ok = np.allclose(
        np.asarray(dump["beat_edges_sec"]), expected["beat_edges_sec"], atol=1e-12
    )
    report(edges_ok, "beat_edges_sec", "beat interval edges")

    print("=== features and model inputs ===")
    for name in TOLERANCE:
        ok, detail = compare(name, as_matrix(dump[name]), expected[name])
        report(ok, name, detail)

    print("=== decisions (must be exact) ===")
    probe_raw = [int(v) for v in dump["probe_raw"]]
    probe_got = [int(v) for v in dump["probe_refined"]]
    probe_want = [int(v) for v in expected["probe_refined"]]
    report(
        probe_got == probe_want,
        "probe_refined",
        f"rust={probe_got} python={probe_want} (from {probe_raw})",
    )
    if probe_got == probe_raw:
        print("  WARN probe_refined            the DP left every boundary alone; "
              "this case is not exercising it")
    for name in ("raw_bounds", "refined_bounds"):
        got = [int(v) for v in dump[name]]
        want = [int(v) for v in expected[name]]
        report(got == want, name, f"rust={got} python={want}")
    got_labels = list(dump["labels"])
    want_labels = list(expected["labels"])
    report(got_labels == want_labels, "labels", f"rust={got_labels} python={want_labels}")

    print("=== the deliberate correction ===")
    default_labels = list(dump["labels_default_options"])
    note = "same as Python here" if default_labels == want_labels else "differs from Python"
    print(f"  note end_state_weight=0    {default_labels} ({note})")
    return failures


if __name__ == "__main__":
    sys.exit(main())
