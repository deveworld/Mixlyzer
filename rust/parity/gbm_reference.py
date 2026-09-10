#!/usr/bin/env python3
"""Regenerate the golden reference for the Rust gradient-boosting runtime.

Run from the repository root:

    .venv/bin/python rust/parity/gbm_reference.py

It feeds a pseudo-random feature matrix through the *Python*
``NumpyHistGradientBoostingClassifier`` loaded from the shipped weights and
writes its raw scores, probabilities and predictions to

    rust/crates/mixlyzer-phrase/tests/data/gbm_reference.json

``gbm_matches_the_python_runtime`` in the Rust test suite then reproduces the
same matrix and checks it lands on the same numbers. This part of the port
needs no audio at all, so it should agree to floating-point noise; if it does
not, the tree walk itself is wrong and nothing downstream can be trusted.

The feature matrix is generated rather than stored (1761 columns of it would
dwarf the answers), so the fixture carries a checksum: if the Rust and Python
generators ever drift apart, the Rust test fails on the checksum rather than
silently comparing two different inputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SEED = 0x1357_9BDF
ROWS = 48
OUT = REPO / "rust/crates/mixlyzer-phrase/tests/data/gbm_reference.json"


class Xorshift:
    """The generator in ``mixlyzer_phrase::testsig``, digit for digit."""

    MASK = 0xFFFF_FFFF

    def __init__(self, seed: int) -> None:
        self.state = seed if seed else 0x1234_5678

    def next_unit(self) -> float:
        state = self.state
        state ^= (state << 13) & self.MASK
        state ^= state >> 17
        state ^= (state << 5) & self.MASK
        self.state = state
        return state / 0xFFFF_FFFF * 2.0 - 1.0


def random_matrix(seed: int, rows: int, cols: int) -> np.ndarray:
    rng = Xorshift(seed)
    out = np.empty((rows, cols), dtype=np.float64)
    cell = 0
    for r in range(rows):
        for c in range(cols):
            value = 3.0 * rng.next_unit()
            cell += 1
            out[r, c] = np.nan if cell % 97 == 0 else value
    return out


def main() -> int:
    from analyzer_core.cue_and_phrase.phrase_analyzer import load_two_stage_model

    model_path = REPO / "assets/weights/phrase_analyzer.npz"
    artifact = load_two_stage_model(str(model_path.resolve()))

    payload: dict[str, object] = {"seed": SEED, "rows": ROWS}
    for stage in ("boundary", "label"):
        clf = artifact[f"{stage}_clf"]
        cols = int(clf.node_feature_idx.max()) + 1
        # The trees never split on a column past the last one they use, so the
        # matrix only has to be that wide for the walk to be fully exercised.
        x = random_matrix(SEED, ROWS, cols)
        finite = x[np.isfinite(x)]
        payload[f"{stage}_columns"] = cols
        payload[f"{stage}_checksum"] = float(finite.sum())
        payload[f"{stage}_raw"] = clf.raw_predict(x).tolist()
        payload[f"{stage}_proba"] = clf.predict_proba(x).tolist()
        payload[f"{stage}_predict"] = [str(v) for v in clf.predict(x)]
        payload[f"{stage}_classes"] = [str(v) for v in clf.classes_]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {OUT.relative_to(REPO)}")
    for stage in ("boundary", "label"):
        print(f"  {stage}: {payload[f'{stage}_columns']} columns, "
              f"checksum {payload[f'{stage}_checksum']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
