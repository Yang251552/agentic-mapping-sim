"""Redesign round 3: apply eval/stage3_rule.json (arm L3 against the logged arm L1 on the 12 fresh train scenes).

  python -m eval.stage3 --l3 out/fresh_l3 --l1 runs/aws/amap-dev-1791349086-14812/out/fresh_l1 \
      --scenes runs/aws/amap-dev-1791349086-14812/out/fresh12.txt --config configs/fresh_l3.json
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from eval.stage1 import check

TASKS, SEEDS = ("T_wet", "T_sleep"), (1, 2)


def first(d: pathlib.Path, union: set):
    for e in json.loads((d / "confident.json").read_text()):
        if e["true"] in union:
            return e["true"]
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--l3", required=True)
    ap.add_argument("--l1", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    scenes = [l.strip() for l in open(a.scenes) if l.strip()]
    if len(scenes) != 12 or len(set(scenes)) != 12:
        raise SystemExit("the decision set must be 12 distinct scenes")
    cfg = json.loads(pathlib.Path(a.config).read_text())
    wet, sleep = (set(cfg["tasks"][t]["categories"]) for t in TASKS)
    L3, L1 = pathlib.Path(a.l3), pathlib.Path(a.l1)
    m = {}
    for s in scenes:
        for t in TASKS:
            for sd in SEEDS:
                for tag, root, p in (("L3", L3, "llm"), ("L1", L1, "llm"), ("b1", L1, "b1")):
                    f = root / s / t / f"s{sd}" / p / "metrics.json"
                    if not f.exists():
                        raise SystemExit(f"missing cell {f}")
                    m[(tag, s, t, sd)] = json.loads(f.read_text())
                    check(m[(tag, s, t, sd)], f)
    rep = {}
    for tag, root in (("L3", L3), ("L1", L1)):
        g = [round(m[(tag, s, t, sd)]["q_t"] - m[("b1", s, t, sd)]["q_t"], 6) for s in scenes for t in TASKS for sd in SEEDS]
        f2 = [(first(root / s / "T_wet" / f"s{sd}" / "llm", wet | sleep), first(root / s / "T_sleep" / f"s{sd}" / "llm", wet | sleep))
              for s in scenes for sd in SEEDS]
        rep[tag] = {"lead": sum(x >= 0.10 for x in g), "lag": sum(x <= -0.10 for x in g), "mean_gain": round(float(np.mean(g)), 4),
                    "f2_pass": sum(w in wet and z in sleep for w, z in f2), "f2_anti": sum(w in sleep and z in wet for w, z in f2),
                    "fallback_episodes": sum(m[(tag, s, t, sd)]["n_fallback"] > 0 for s in scenes for t in TASKS for sd in SEEDS)}
    delta = float(np.mean([m[("L3", s, t, sd)]["q_t"] - m[("L1", s, t, sd)]["q_t"] for s in scenes for t in TASKS for sd in SEEDS]))
    r3, r1 = rep["L3"], rep["L1"]
    rep["delta_vs_L1"] = round(delta, 4)
    rep["R"] = {"R1": r3["f2_pass"] > r1["f2_pass"] and r3["f2_pass"] >= r3["f2_anti"], "R2": r3["lead"] >= 6 and r3["lag"] <= 2,
                "R3": delta >= -0.01, "R4": r3["fallback_episodes"] == 0}
    rep["decision"] = "adopt v7 for held-out batch 2" if all(rep["R"].values()) else "do not adopt v7: stop and report to the author"
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
