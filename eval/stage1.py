"""Redesign decision boot: apply eval/stage1_rule.json to the two arms. Reads metrics.json only; every cell of every
policy must be present (a missing cell stops with an error, never a verdict).

  python -m eval.stage1 --l1 out/fresh_l1 --l2 out/fresh_l2 --scenes out/fresh12.txt
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

TASKS, SEEDS, POLS = ("T_wet", "T_sleep"), (1, 2), ("b1", "b2", "refg", "llm")


def ok01(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1


def check(m: dict, f) -> None:
    """Every value the rule reads must be present and valid: no default fills a gap."""
    nf = m.get("n_fallback")
    good = (all(ok01(m.get(k)) for k in ("q_t", "coverage", "s_t")) and isinstance(nf, int) and not isinstance(nf, bool) and nf >= 0
            and all(ok01(m.get("by_task", {}).get(t, {}).get("q_t")) for t in TASKS))
    if not good:
        raise SystemExit(f"invalid or incomplete metrics: {f}")


def load(root: pathlib.Path, scenes) -> dict:
    out, missing = {}, []
    for s in scenes:
        for t in TASKS:
            for sd in SEEDS:
                for p in POLS:
                    f = root / s / t / f"s{sd}" / p / "metrics.json"
                    if f.exists():
                        m = json.loads(f.read_text())
                        check(m, f)
                        out[(s, t, sd, p)] = m
                    else:
                        missing.append(str(f))
    if missing:
        raise SystemExit(f"{len(missing)} missing cells, e.g. {missing[0]}")
    return out


def arm(m: dict, scenes) -> dict:
    for k, v in m.items():
        check(v, k)
    cells = [(s, t, sd) for s in scenes for t in TASKS for sd in SEEDS]
    other = {"T_wet": "T_sleep", "T_sleep": "T_wet"}
    rng = np.random.default_rng(0)
    b1_mean = float(np.mean([m[c + ("b1",)]["q_t"] for c in cells]))   # unrounded: the rule reads this, the report rounds
    rep = {"_b1_mean": b1_mean, "b1_mean_q": round(b1_mean, 4),
           "b1_median_q": round(float(np.median([m[c + ("b1",)]["q_t"] for c in cells])), 4)}
    for p in ("llm", "b2", "refg"):
        g = {c: round(m[c + (p,)]["q_t"] - m[c + ("b1",)]["q_t"], 6) for c in cells}
        sw = {(s, t, sd): round(m[(s, other[t], sd, p)]["by_task"][t]["q_t"] - m[(s, t, sd, "b1")]["q_t"], 6) for s, t, sd in cells}
        lead, lag = sum(v >= 0.10 for v in g.values()), sum(v <= -0.10 for v in g.values())
        slead, slag = sum(v >= 0.10 for v in sw.values()), sum(v <= -0.10 for v in sw.values())
        sm = np.array([np.mean([g[(s, t, sd)] for t in TASKS for sd in SEEDS]) for s in scenes])
        bs = rng.choice(sm, (10000, len(sm))).mean(1)
        mean = float(np.mean(list(g.values())))
        rep[p] = {"lead": lead, "lag": lag, "net": lead - lag, "swap_lead": slead, "swap_lag": slag, "swap_net": slead - slag,
                  "_mean_gain": mean, "mean_gain": round(mean, 4),
                  "ci95_scene": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                  "by_task": {t: [sum(g[c] >= 0.10 for c in cells if c[1] == t), sum(g[c] <= -0.10 for c in cells if c[1] == t)] for t in TASKS},
                  "cov_half": round(float(np.mean([(m[c + (p,)]["coverage"] - m[c + ("b1",)]["coverage"]) / 2 for c in cells])), 4),
                  "st_half": round(float(np.mean([(m[c + (p,)]["s_t"] - m[c + ("b1",)]["s_t"]) / 2 for c in cells])), 4),
                  "fallback_episodes": sum(m[c + (p,)].get("n_fallback", 0) > 0 for c in cells)}
    L = rep["llm"]
    rep["Q"] = {"Q1": L["net"] - L["swap_net"] >= 4, "Q2": L["lead"] >= 6 and L["lead"] >= 2 * L["swap_lead"],
                "Q3": L["lag"] <= 6, "Q4": L["_mean_gain"] >= -1e-12, "Q5": L["fallback_episodes"] == 0}   # gains carry 6 decimals
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1", required=True)
    ap.add_argument("--l2", required=True)
    ap.add_argument("--scenes", required=True)
    a = ap.parse_args(argv)
    scenes = [l.strip() for l in open(a.scenes) if l.strip()]
    if len(scenes) != 12 or len(set(scenes)) != 12:
        raise SystemExit("the decision set must be 12 distinct scenes")
    r = {"L1": arm(load(pathlib.Path(a.l1), scenes), scenes), "L2": arm(load(pathlib.Path(a.l2), scenes), scenes)}
    n = len(scenes) * len(TASKS) * len(SEEDS)
    r["L2"]["Q"]["Q6"] = abs(r["L2"]["_b1_mean"] - r["L1"]["_b1_mean"]) <= 0.02 + 1e-9 and r["L2"]["b2"]["swap_lead"] < 0.15 * n
    r["L1"]["Q"]["Q6"] = True
    ok = [k for k in ("L1", "L2") if all(r[k]["Q"].values())]
    pool = ok or ["L1", "L2"]
    pick = max(pool, key=lambda k: (r[k]["llm"]["net"], k == "L1"))
    r["decision"] = {"qualifying": ok, "pick": pick, "outcome": "qualified" if ok else "no arm qualified: freeze the pick, run held-out once, report the null"}
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
