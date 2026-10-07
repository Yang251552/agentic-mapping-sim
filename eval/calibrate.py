"""Phase 2 calibration on tuning scenes only (never on held-out scenes).

  python -m eval.calibrate budget RUN_ROOT            # B1 curves from one long run per cell -> budget B
  python -m eval.calibrate checks RUN_ROOT --budget B  # S4a (B2 task-swap null) and S4b (REF - B1 headroom)
  python -m eval.calibrate window LONG_ROOT             # S4a / S4b at every step of the window from long b1/b2/ref/refg runs
  python -m eval.calibrate b2grid GRID_ROOT            # B2 weights: grid point with the highest median own-task Q_T
  python -m eval.calibrate decompose AT_B --budget B --b1-root B1_LONG  # each difference vs B1 split into coverage and S_T halves

Budget rule (PROJECT_SPEC): the fixed step count at which the median B1 Q_T over (scene, task, seed) cells falls in
[0.3, 0.6]; among those, the one closest to 0.45 (rounded to curve_every). B1 reads neither the task nor the steps
left, so its trajectory does not depend on B and one long run gives Q_T at every B.
S4a: share of (scene, seed, T) cells where Q_T(B2 run under the other task) - Q_T(B1) >= 0.10; >= 15% triggers S4a.
S4b: median over cells of Q_T(REF) - Q_T(B1); < 0.15 triggers S4b.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics


def cells(root: pathlib.Path, policy: str):
    for m in sorted(root.glob(f"*/*/s*/{policy}/metrics.json")):
        scene, task, seed = m.parts[-5], m.parts[-4], m.parts[-3]
        yield (scene, task, seed), m.parent


def regular(points: list, top: int, every: int = 10) -> dict:
    """Curve points on the grid every, 2 every, ..., top: the value at step s is the last record at or before s. Curves
    written before 10-07 can miss a grid step (an executor bump landed on it) and runs can end early (no candidates)."""
    pts = sorted(points, key=lambda p: p["step"])
    out, i = {}, -1
    for s in range(every, top + 1, every):
        while i + 1 < len(pts) and pts[i + 1]["step"] <= s:
            i += 1
        if i >= 0:
            out[s] = pts[i]
    return out


def points(d: pathlib.Path) -> list:
    """A run's curve plus its final state: a run that ended early (no candidates) between curve steps, or before the
    first one (a trapped start ends at step 8 with an empty curve), keeps that final state for every later step."""
    pts = json.loads((d / "curve.json").read_text())
    m = json.loads((d / "metrics.json").read_text())
    if not pts or pts[-1]["step"] < m["steps_used"]:
        pts.append({"step": m["steps_used"], **{k: m[k] for k in ("q_t", "coverage", "s_t", "n_task_objects", "n_confident_correct",
                                                                    "floor_coverage") if k in m}, "by_task": m.get("by_task", {})})
    return pts


def _top(root: pathlib.Path, policies) -> int:
    return max((p["step"] for pol in policies for _, d in cells(root, pol) for p in points(d)), default=0)


def b1_curves(root: pathlib.Path) -> dict:
    """B1 Q_T by step for each cell on a regular grid; a run that ended early keeps its last value."""
    top = _top(root, ("b1",))
    curves = {k: {s: p["q_t"] for s, p in regular(points(d), top).items()} for k, d in cells(root, "b1")}
    if not curves:
        raise SystemExit(f"no b1 runs under {root}")
    return curves


def budget(root: pathlib.Path, lo=0.3, hi=0.6, target=0.45) -> dict:
    curves = b1_curves(root)
    steps = sorted(set.intersection(*(set(c) for c in curves.values())))
    med = {s: statistics.median(c[s] for c in curves.values()) for s in steps}
    ok = [s for s in steps if lo <= med[s] <= hi]
    pick = min(ok, key=lambda s: (abs(med[s] - target), s)) if ok else None
    return {"n_cells": len(curves), "budget": pick, "median_q_t_at_budget": med.get(pick),
            "window": [min(ok), max(ok)] if ok else None, "median_curve": {s: round(med[s], 4) for s in steps[::5]}}


def checks(root: pathlib.Path, B: int, b1_root: pathlib.Path = None, b2_root: pathlib.Path = None) -> dict:
    """B1 does not read the budget, so with b1_root its value at B is read off the long-run curve."""
    by = {}
    if b1_root is not None:
        for (scene, task, seed), c in b1_curves(b1_root).items():
            by[("b1", scene, task, seed)] = {"q_t": c[B]}
    for pol in (("b2", "ref") if b1_root is not None else ("b1", "b2", "ref")):
        for k, d in cells(b2_root if (pol == "b2" and b2_root) else root, pol):
            m = json.loads((d / "metrics.json").read_text())
            by[(pol, *k)] = m
    if b2_root is not None and not any(k[0] == "b2" for k in by):
        raise SystemExit(f"no b2 runs under --b2-root {b2_root}")
    tasks = sorted({k[2] for k in by})
    swap, head = [], []
    for (pol, scene, task, seed), m in by.items():
        if pol != "b1":
            continue
        other = [t for t in tasks if t != task]
        for t2 in other:
            b2 = by.get(("b2", scene, t2, seed))
            if b2:
                swap.append(round(b2["by_task"][task]["q_t"] - m["q_t"], 6) >= 0.10)  # rounded like compare(): 0.6 - 0.5 is < 0.10 in floats
        ref = by.get(("ref", scene, task, seed))
        if ref:
            head.append(ref["q_t"] - m["q_t"])
    early = {}  # trap detector: runs that ran out of candidates before the budget (should be rare and policy-independent)
    for (pol, scene, task, seed), m in by.items():
        if m.get("end_reason") == "no_candidates" and m.get("steps_used", B) < B:
            early.setdefault(pol, []).append(f"{scene}/{task}@{m['steps_used']}")
    s4a = sum(swap) / len(swap) if swap else None
    s4b = statistics.median(head) if head else None
    return {"budget": B, "s4a_null_success_rate": s4a, "s4a_n": len(swap), "s4a_triggered": s4a is not None and s4a >= 0.15,
            "s4b_median_headroom": s4b, "s4b_n": len(head), "s4b_triggered": s4b is not None and s4b < 0.15,
            "ended_early_no_candidates": early}


def compare(root: pathlib.Path, B: int, b1_root: pathlib.Path, policy="llm") -> dict:
    """Tuning-scene read of the single-run criterion: Q_T(policy) - Q_T(B1) >= 0.10 at the budget (B1 off its curve)."""
    b1 = {k: c[B] for k, c in b1_curves(b1_root).items()}
    rows = []
    for k, d in cells(root, policy):
        m = json.loads((d / "metrics.json").read_text())
        rows.append({"cell": "/".join(k), "q_t": m["q_t"], "b1": b1.get(k), "diff": round(m["q_t"] - b1[k], 4) if k in b1 else None,
                     "fallback": m.get("n_fallback", 0), "end": m.get("end_reason")})
    diffs = [r["diff"] for r in rows if r["diff"] is not None]
    return {"policy": policy, "budget": B, "n": len(rows), "successes": sum(d >= 0.10 for d in diffs),
            "median_diff": statistics.median(diffs) if diffs else None, "rows": rows}


def decompose(root: pathlib.Path, B: int, b1_root: pathlib.Path) -> dict:
    """Q_T = (coverage + S_T) / 2, so every difference against B1 splits into a coverage half and a label half.
    Rows: b2swap (the S4a null: B2 run under the other task, scored on this one), llm, ref, b2 (own task)."""
    b1 = {}  # full curve point (q_t, coverage, s_t) at B; a run that ended early keeps its last point
    for k, d in cells(b1_root, "b1"):
        pts = {p["step"]: p for p in points(d)}
        b1[k] = pts.get(B) or pts[max(pts)]
    tasks = sorted({k[1] for k in b1})
    out = {"budget": B, "rows": {}}
    for pol in ("b2swap", "llm", "ref", "b2"):
        rows = []
        for (scene, task, seed), base in b1.items():
            if pol == "b2swap":
                t2 = next(t for t in tasks if t != task)
                m = root / scene / t2 / seed / "b2" / "metrics.json"
                if not m.exists():
                    continue
                a = json.loads(m.read_text())["by_task"][task]
            else:
                m = root / scene / task / seed / pol / "metrics.json"
                if not m.exists():
                    continue
                a = json.loads(m.read_text())
            rows.append({"cell": f"{scene}/{task}/{seed}", "d_q_t": round(a["q_t"] - base["q_t"], 4),
                         "d_cov_half": round((a["coverage"] - base["coverage"]) / 2, 4), "d_s_t_half": round((a["s_t"] - base["s_t"]) / 2, 4),
                         "n_task_objects": a["n_task_objects"], "one_object_q_t": round(1 / (2 * a["n_task_objects"]), 4) if a["n_task_objects"] else None})
        succ = [r for r in rows if r["d_q_t"] >= 0.10]
        out["rows"][pol] = {"n": len(rows), "successes": len(succ),
                            "successes_label_driven": sum(r["d_s_t_half"] > r["d_cov_half"] for r in succ),
                            "median_d_cov_half": statistics.median(r["d_cov_half"] for r in rows) if rows else None,
                            "median_d_s_t_half": statistics.median(r["d_s_t_half"] for r in rows) if rows else None, "cells": rows}
    return out


def long_curves(root: pathlib.Path, policy: str, top: int = 0) -> dict:
    """Curve points on a regular grid up to `top` (default: the longest run of this policy) for each cell."""
    top = max(top, _top(root, (policy,)))
    return {k: regular(points(d), top) for k, d in cells(root, policy)}


def window(root: pathlib.Path, ref_policies=("ref", "refg")) -> dict:
    """S4a and S4b at every step of the budget window, from long runs of b1, b2, b0 and the REF variants. Next to S4a:
    how often B1 is ahead of the null by >= 0.10, the null's floor coverage minus B1's, and the S4a-style rate for the
    random B0, so a low S4a from a null that merely stopped differing from B1 shows up. B1, B2 and REF read
    neither the task's step budget nor the steps left, so a long run's prefix is the run at any shorter budget.
    The budget itself still comes from the B1 rule; the other rows are information, not a menu."""
    top = _top(root, ("b1", "b2", "b0") + tuple(ref_policies))
    b1, b2, b0 = (long_curves(root, p, top) for p in ("b1", "b2", "b0"))
    refs = {p: long_curves(root, p, top) for p in ref_policies}
    allk = set(b1) | set(b2) | set(b0) | {k for cur in refs.values() for k in cur}   # a cell any policy has, all must have
    missing = [f"{p}:{'/'.join(k)}" for p, cur in (("b1", b1), ("b2", b2), ("b0", b0), *refs.items()) for k in sorted(allk) if not cur.get(k)]
    if not b1 or missing:  # a missing or empty cell would silently shrink a rate's denominator
        raise SystemExit("window: incomplete long runs: " + (", ".join(missing) if missing else "no b1 runs"))
    pick = budget(root)
    every = (b1, b2, b0, *refs.values())
    steps = sorted(set.intersection(*(set(c) for cur in every for c in cur.values())))  # steps where every cell has a value
    tasks = sorted({k[1] for k in b1})
    rows = []
    for B in steps:
        med = statistics.median(c[B]["q_t"] for c in b1.values())
        if not 0.3 <= med <= 0.6:
            continue
        swap, mirror, fcov, rand, head = [], [], [], [], {p: [] for p in refs}
        for (scene, task, seed), c in b1.items():
            for t2 in tasks:
                o = b2.get((scene, t2, seed))
                if t2 != task and o and B in o:
                    d = round(o[B]["by_task"][task]["q_t"] - c[B]["q_t"], 6)
                    swap.append(d >= 0.10)
                    mirror.append(d <= -0.10)   # a null that never differs from B1 would pass S4a for the wrong reason
                    fcov.append(o[B]["floor_coverage"] - c[B]["floor_coverage"])
            o = b0.get((scene, task, seed))
            if o and B in o:
                rand.append(round(o[B]["q_t"] - c[B]["q_t"], 6) >= 0.10)
            for p_, r in refs.items():
                o = r.get((scene, task, seed))
                if o and B in o:
                    head[p_].append(o[B]["q_t"] - c[B]["q_t"])
        frac = lambda v: round(sum(v) / len(v), 4) if v else None
        rows.append({"B": B, "b1_median": round(med, 4), "s4a": frac(swap), "s4a_n": len(swap), "b1_ahead_of_null": frac(mirror),
                     "null_floor_cov_minus_b1": round(statistics.median(fcov), 4) if fcov else None, "b0_ahead_of_b1": frac(rand),
                     **{f"s4b_{p_}": round(statistics.median(v), 4) if v else None for p_, v in head.items()},
                     **{f"ahead_{p_}": frac([round(x, 6) >= 0.10 for x in v]) for p_, v in head.items()}})  # success-rate form
    if pick["budget"] is not None and pick["budget"] not in steps:
        raise SystemExit(f"window: some cell has no value at the rule's budget {pick['budget']}")
    return {"rule_budget": pick["budget"], "window": pick["window"], "n_cells": len(b1), "rows": rows}


B2_LAMS = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5)


def b2lam(root: pathlib.Path, B: int) -> dict:
    """Evidence-gated B2 (rule fixed 10-07 03:27): the lambda with the highest median own-task Q_T at step B over the
    tuning cells; ties -> higher mean, then the larger lambda (closer to B1). Every grid point must cover the same cells."""
    rows, keys = [], {}
    for lam in B2_LAMS:
        cur = long_curves(root / f"lam{lam}", "b2", top=B)   # a run that ended early keeps its last value up to B
        if any(B not in c for c in cur.values()):
            raise SystemExit(f"lambda grid incomplete: a run under lam{lam} has no value at step {B}")
        keys[lam] = set(cur)
        q = [c[B]["q_t"] for c in cur.values()]
        rows.append({"lam": lam, "n": len(q), "median": statistics.median(q) if q else None, "mean": sum(q) / len(q) if q else None})
    want = set().union(*keys.values())
    if not want or any(k != want for k in keys.values()):
        raise SystemExit(f"lambda grid incomplete under {root} at step {B}")
    pick = max(rows, key=lambda r: (r["median"], r["mean"], r["lam"]))
    for r in rows:
        r["median_q_t"], r["mean_q_t"] = round(r.pop("median"), 6), round(r.pop("mean"), 6)
    return {"pick": pick["lam"], "dir": f"lam{pick['lam']}", "step": B, "n_cells": len(want), "rows": rows}


B2_GRID = [(a, b) for a in (0.5, 1.0, 2.0) for b in (0.0, 0.1, 0.3)]
B2_DEFAULT = (1.0, 0.1)


def b2grid(root: pathlib.Path) -> dict:
    """Pick the B2 weights (rule fixed before any grid result, EXPERIMENT_LOG 10-06 22:20): highest median own-task Q_T
    over the tuning cells at B; ties -> higher mean, then the point nearest the default (1.0, 0.1)."""
    rows, keys = [], {}
    for a, b in B2_GRID:
        got = {k: json.loads((d / "metrics.json").read_text())["q_t"] for k, d in cells(root / f"a{a}_b{b}", "b2")}
        keys[(a, b)] = set(got)
        q = list(got.values())
        rows.append({"a": a, "b": b, "n": len(q), "median": statistics.median(q) if q else None, "mean": sum(q) / len(q) if q else None})
    want = set().union(*keys.values())
    if not want or any(k != want for k in keys.values()):  # every point must cover the same cells, or nothing is picked
        raise SystemExit(f"b2 grid incomplete under {root}: " + ", ".join(f"a{a}_b{b}: {len(k)}/{len(want)}" for (a, b), k in keys.items()))
    pick = max(rows, key=lambda r: (r["median"], r["mean"], -abs(r["a"] - B2_DEFAULT[0]) - abs(r["b"] - B2_DEFAULT[1])))  # ties on exact values
    for r in rows:
        r["median_q_t"], r["mean_q_t"] = round(r.pop("median"), 6), round(r.pop("mean"), 6)
    return {"pick": {"a": pick["a"], "b": pick["b"]}, "dir": f"a{pick['a']}_b{pick['b']}", "n_cells": len(want), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["budget", "checks", "compare", "decompose", "b2grid", "window", "b2lam"])
    ap.add_argument("root")
    ap.add_argument("--budget", type=int)
    ap.add_argument("--b1-root", help="checks: read B1 at the budget off these long B1 runs")
    ap.add_argument("--b2-root", help="checks: take the B2 runs from here (the selected grid point)")
    ap.add_argument("--refs", default="ref,refg", help="window: REF variants to read (comma list)")
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    if a.mode == "budget":
        out = budget(root)
    elif a.mode == "checks":
        out = checks(root, a.budget, a.b1_root and pathlib.Path(a.b1_root), a.b2_root and pathlib.Path(a.b2_root))
    elif a.mode == "b2grid":
        out = b2grid(root)
    elif a.mode == "b2lam":
        out = b2lam(root, a.budget)
    elif a.mode == "window":
        out = window(root, tuple(a.refs.split(",")))
    elif a.mode == "compare":
        out = compare(root, a.budget, pathlib.Path(a.b1_root))
    else:
        out = decompose(root, a.budget, pathlib.Path(a.b1_root))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
