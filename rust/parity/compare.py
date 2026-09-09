"""Compare the Rust core against the Python implementation it replaces.

Run from the repository root:

    rust/parity/compare.py

It builds the same fixtures through both implementations and reports every
field that differs. Differences that are intended are listed in EXPECTED below
with the reason; anything else is a regression in the port.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Fields where the two implementations are meant to disagree, and why.
EXPECTED = {
    "phrases.abbreviated": (
        "Python abbreviates by first letter, so INTRO/INTERLUDE both become 'I' "
        "and BRIDGE/BREAK_CHORUS both become 'B'. The Rust port uses an explicit "
        "table (IL, BC) so overview labels stay distinguishable."
    ),
}


def rust_dump() -> dict:
    result = subprocess.run(
        ["cargo", "run", "-q", "-p", "mixlyzer-core", "--example", "parity_dump"],
        cwd=REPO / "rust",
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def python_dump() -> dict:
    from core.beat_geometry import bar_beat_label, downbeat_beat_indices
    from core.linear_segments import build_bpm_segments, harmonic_compatible_keys
    from utils.cue_points import build_phrase_cue_points
    from utils.labels import idx_to_labels
    from utils.phrases import (
        PHRASE_LABELS,
        abbreviate_phrase_labels,
        merge_fill_phrases_for_display,
        number_phrase_labels,
        phrase_color,
    )
    import numpy as np

    keys = []
    for index in range(24):
        camelot, classical = idx_to_labels(index)
        keys.append(
            {
                "index": index,
                "camelot": camelot,
                "classical": classical,
                "neighbours": sorted(harmonic_compatible_keys(index)),
            }
        )

    tempo = np.asarray(
        [
            [0.0, 10.0, 128.0, 0.0, 4],
            [10.0, 20.0, 128.0, 10.0, 4],
            [20.0, 20.0, 128.0, 20.0, 4],
            [20.0, 30.0, 128.0, 20.0, 3],
            [30.0, 45.0, 174.4, 30.0, 4],
        ],
        dtype=float,
    )
    bpm_rows = [
        {
            "seq_index": row["seq_index"],
            "start_sec": row["start_sec"],
            "end_sec": row["end_sec"],
            "duration_sec": row["duration_sec"],
            "bpm": row["bpm"],
            "bpm_rounded": row["bpm_rounded"],
            "time_signature": row["time_signature"],
        }
        for row in build_bpm_segments(tempo)
    ]

    phrases = [
        {"start": 0.0, "end": 8.0, "label": "INTRO"},
        {"start": 8.0, "end": 24.0, "label": "VERSE"},
        {"start": 24.0, "end": 32.0, "label": "FILL_IN"},
        {"start": 32.0, "end": 48.0, "label": "CHORUS"},
        {"start": 48.0, "end": 64.0, "label": "CHORUS"},
        {"start": 64.0, "end": 80.0, "label": "VERSE"},
        {"start": 80.0, "end": 96.0, "label": "INTERLUDE"},
        {"start": 96.0, "end": 112.0, "label": "CHORUS"},
        {"start": 112.0, "end": 128.0, "label": "OUTRO"},
    ]
    phrase_block = {
        "numbered": number_phrase_labels(phrases),
        "abbreviated": abbreviate_phrase_labels(phrases),
        "merged_fills": [
            {"start": p["start"], "end": p["end"], "label": p["label"]}
            for p in merge_fill_phrases_for_display(phrases)
        ],
        "colors": [
            {"label": label, "rgb": list(phrase_color(label))} for label in PHRASE_LABELS
        ],
    }

    cue_points = [
        {
            "id": point["id"],
            "time_sec": point["time_sec"],
            "label": point["label"],
            "comment": point["comment"],
        }
        for point in build_phrase_cue_points(phrases)
    ]

    period = 60.0 / 120.0
    beats = [i * period for i in range(32)]
    switch = 32 * period
    second = 60.0 / 140.0
    beats += [switch + i * second for i in range(32)]
    segments = np.asarray(
        [
            [0.0, switch, 120.0, 0.0, 4],
            [switch, switch + 32 * second, 140.0, switch + second, 4],
        ],
        dtype=float,
    )
    beats_arr = np.asarray(beats, dtype=float)
    probes = [0.0, 1.0, 4.5, 9.0, 20.0]
    beatgrid = {
        "downbeat_indices": [int(i) for i in downbeat_beat_indices(beats_arr, segments)],
        "bar_beat_probe_times": probes,
        "bar_beat_labels": [bar_beat_label(beats_arr, t, segments) for t in probes],
    }

    return {
        "keys": keys,
        "bpm_segment_rows": bpm_rows,
        "phrases": phrase_block,
        "cue_points": cue_points,
        "beatgrid": beatgrid,
    }


def diff(path: str, left, right, out: list) -> None:
    """Collect every leaf where the two documents disagree."""
    if isinstance(left, dict) and isinstance(right, dict):
        for name in sorted(set(left) | set(right)):
            diff(f"{path}.{name}" if path else name, left.get(name), right.get(name), out)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            out.append((path, f"{len(left)} items", f"{len(right)} items"))
            return
        for index, (a, b) in enumerate(zip(left, right)):
            diff(f"{path}[{index}]", a, b, out)
        return
    if isinstance(left, float) or isinstance(right, float):
        try:
            if abs(float(left) - float(right)) <= 1e-9:
                return
        except (TypeError, ValueError):
            pass
    if left != right:
        out.append((path, left, right))


def main() -> int:
    rust = rust_dump()
    python = python_dump()

    differences: list = []
    diff("", rust, python, differences)

    unexpected = []
    accounted = []
    for path, rust_value, python_value in differences:
        reason = next((why for prefix, why in EXPECTED.items() if path.startswith(prefix)), None)
        (accounted if reason else unexpected).append((path, rust_value, python_value, reason))

    print(f"compared {len(rust)} sections")
    if accounted:
        print(f"\n{len(accounted)} intended difference(s):")
        seen = set()
        for path, rust_value, python_value, reason in accounted:
            print(f"  {path}: rust={rust_value!r} python={python_value!r}")
            if reason not in seen:
                print(f"    reason: {reason}")
                seen.add(reason)
    if unexpected:
        print(f"\n{len(unexpected)} UNEXPECTED difference(s):")
        for path, rust_value, python_value, _ in unexpected:
            print(f"  {path}: rust={rust_value!r} python={python_value!r}")
        return 1
    print("\nno unexpected differences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
