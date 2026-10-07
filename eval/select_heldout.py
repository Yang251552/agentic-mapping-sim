"""Held-out scene selection (frozen rule, eval/prereg.json): reads only val screening metadata, never an episode.

A val scene is eligible when, for every preregistered seed, the screening found a start (2 m from task objects) whose
floor holds objects of every task's categories and whose navigable area lies in the tuning scenes' range. Malformed
rows stop the selection (a non-val scene id, a repeated scene and seed, a loading error); a row whose `passes` is
not true or without a valid start is not eligible. Eligible scenes are ordered by sha256(salt + scene id); the first n are the
held-out batch. Fewer than n eligible scenes stops.

  python -m eval.select_heldout --prereg eval/prereg.json --screen out/screen_val.jsonl --out out/heldout.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re

SPLIT_ID = {"val": re.compile(r"^008\d\d-[A-Za-z0-9]+$"),        # HM3D val scenes are 00800-00899
            "train": re.compile(r"^00[0-7]\d\d-[A-Za-z0-9]+$")}   # train scenes are 00000-00799
VAL = SPLIT_ID["val"]


def valid_start(v) -> bool:
    return isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v)


def select(rows: list, seeds: list, area: list, salt: str, n: int, split: str = "val", exclude=()) -> list:
    """`split` names the HM3D split the batch comes from; `exclude` lists scenes that may not be held out (batch 2 on the
    train split, user 10-07: every train scene screened or run before it)."""
    by, ids, exclude = {}, SPLIT_ID[split], set(exclude)
    for r in rows:
        if not ids.match(str(r.get("scene"))):
            raise SystemExit(f"held-out selection reads {split} scenes only: {r.get('scene')}")
        if r["scene"] in exclude:
            raise SystemExit(f"{r['scene']} was used before this batch and cannot be held out")
        if "error" in r:
            raise SystemExit(f"screening could not load {r['scene']} seed {r['seed']}: {str(r['error'])[:120]}")
        if r["seed"] in by.get(r["scene"], {}):
            raise SystemExit(f"duplicate screening row for {r['scene']} seed {r['seed']}")
        by.setdefault(r["scene"], {})[r["seed"]] = r
    good = lambda r: (r.get("passes") is True and "error" not in r and isinstance(r.get("nav_area_m2"), (int, float))
                      and area[0] <= r["nav_area_m2"] <= area[1] and valid_start(r.get("start_xyz")))
    ok = [s for s, rs in by.items() if set(rs) == set(seeds) and all(good(r) for r in rs.values())]
    ok.sort(key=lambda s: hashlib.sha256(f"{salt}{s}".encode()).hexdigest())
    if len(ok) < n:
        raise SystemExit(f"only {len(ok)} eligible held-out scenes, need {n}")
    return ok[:n]


def excluded(rule: dict) -> list:
    return open(rule["exclude_file"]).read().split() if rule.get("exclude_file") else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--screen", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rule = json.load(open(a.prereg))["heldout"]
    rows = [json.loads(l) for l in open(a.screen) if l.strip()]
    pick = select(rows, rule["seeds"], rule["area_window_m2"], rule["salt"], rule["n_scenes"], rule.get("split", "val"), excluded(rule))
    open(a.out, "w").write("\n".join(pick) + "\n")
    print(f"{len(pick)} held-out scenes:", " ".join(pick))


if __name__ == "__main__":
    main()
