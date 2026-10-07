"""Replay acceptance: every finished attempt of the committed held-out report must be reproduced exactly by the replay.

  python -m eval.compare_replay --replay out/heldout_m/replay --results results/heldout_batch1_floor.json
Exit 0 only if every finished episode is present in the replay and its scores, steps and fallback count are identical.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

FIELDS = ("q_t", "coverage", "s_t", "steps_used", "n_fallback")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args(argv)
    rep = json.loads(pathlib.Path(a.results).read_text())
    done = [r for r in rep["attempts"] if r.get("status") in ("ok", "fallback")]
    if len(done) != rep["cells_expected"] or len({(r["scene"], r["task"], r["seed"], r["policy"]) for r in done}) != len(done):
        print(f"the committed report does not hold exactly one finished attempt per cell ({len(done)} of {rep['cells_expected']})")
        return 1
    bad = []
    for r in done:
        f = pathlib.Path(a.replay) / r["scene"] / r["task"] / f"s{r['seed']}" / r["policy"] / "metrics.json"
        if not f.exists():
            bad.append((f"{r['scene']}/{r['task']}/s{r['seed']}/{r['policy']}", "missing"))
            continue
        m = json.loads(f.read_text())
        diff = {k: (r.get(k), m.get(k)) for k in FIELDS if r.get(k) != m.get(k)}
        if diff:
            bad.append((f"{r['scene']}/{r['task']}/s{r['seed']}/{r['policy']}", diff))
    print(f"{len(done) - len(bad)} of {len(done)} episodes reproduced exactly")
    for cell, why in bad[:20]:
        print("  mismatch", cell, why)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
