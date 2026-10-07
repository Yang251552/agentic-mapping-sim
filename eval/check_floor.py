"""Showcase-floor judgment for the held-out batch, exactly as preregistered in eval/prereg.json (success, showcase_floor,
predictions, gif_rule). Reads the run artifacts only; judges F1-F5 and reports the statistics and predictions.

  python -m eval.check_floor --root out/heldout_m --prereg eval/prereg.json --tests-log out/tests_heldout.log --out out/floor.json

Malformed or mismatched inputs (a grid that is not the preregistered batch, artifacts that do not belong to their run
row, values outside [0, 1]) stop with an error instead of a verdict. Otherwise exit 0 when every floor item passes and
1 when one does not; the report is printed (and written to --out) either way.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics

import numpy as np

from amap.run import canonical, sha

LEAD = 0.10
META_TESTS = ("tests/test_habitat.py::test_metamorphic_unobserved_categories_are_invisible",    # M: shuffled unseen categories
              "tests/test_episode.py::test_metamorphic_unobserved_changes_are_invisible")      # L core: unseen geometry, walls, rooms


def check_grid(pre: dict, cfg: dict):
    """The grid must be the preregistered batch: the frozen config's axes, n_scenes scenes, the rule's seeds, no repeats."""
    g, rule = pre["grid"], pre["heldout"]
    axes = (g["scenes"], g["tasks"], g["seeds"], g["policies"])
    bad = (sha(canonical(cfg)) != pre["config_sha256"] or any(not x or len(set(x)) != len(x) for x in axes)
           or g["cells"] != len(axes[0]) * len(axes[1]) * len(axes[2]) * len(axes[3])
           or (g["scenes"], g["tasks"], g["seeds"], g["policies"]) != (cfg["scenes"], list(cfg["tasks"]), cfg["seeds"], cfg["policies"])
           or len(g["scenes"]) != rule["n_scenes"] or g["seeds"] != rule["seeds"] or set(g["tasks"]) != {"T_wet", "T_sleep"}
           or not {"llm", "b1"} <= set(g["policies"]))
    if bad:
        raise SystemExit("the grid or config is not the preregistered batch")


def finite01(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1


def check_metrics(k, m: dict, row: dict, pre: dict, tasks):
    """Artifacts must belong to the finished row they are judged by (a later run into the same directory replaces them)."""
    nf = m.get("n_fallback")
    ok = (m.get("config_sha256") == pre["config_sha256"] and m.get("mode") == "live" and finite01(m.get("q_t"))
          and m["q_t"] == row.get("q_t") and isinstance(nf, int) and not isinstance(nf, bool) and nf >= 0
          and nf == row.get("n_fallback") and not isinstance(row.get("n_fallback"), bool)
          and (row["status"] == "fallback") == (nf > 0)
          and all(finite01(m.get(f, {}).get(t, {}).get("q_t")) for f in ("by_task", "by_task_mid") for t in tasks)
          and m["by_task"][k[1]]["q_t"] == m["q_t"]   # the episode's own task score is its headline score
          and all(m["by_task"][t]["q_t"] == (row.get("q_t_by_task") or {}).get(t) for t in tasks)   # and every task score,
          and all(m["by_task_mid"][t]["q_t"] == (row.get("q_t_by_task_mid") or {}).get(t) for t in tasks))   # final and at B/2, is the row's
    if not ok:
        raise SystemExit(f"invalid or mismatched artifacts for {'/'.join(map(str, k))}")


INFRA_ERR = re.compile(r"^(timeout|TimeoutError|ConnectionResetError|ConnectionError|RemoteDisconnected|IncompleteRead|URLError|InfraError)\b")


def infra(r: dict) -> str:
    """S7: an API timeout or dropped connection is an infrastructure abort (one rerun allowed) even when, as in the held-out
    batch on 10-07 (Python 3.9: socket.timeout is not a TimeoutError), it escaped the retry loop and was logged as a crash."""
    api = r.get("policy") == "llm"   # only the llm calls an API (post-run review of batch 1: b1 / b0 / b2 crashes are never exempt)
    return "aborted_infra" if api and r.get("status") == "crashed" and INFRA_ERR.match(str(r.get("error", ""))) else r.get("status")


def load(root: pathlib.Path, pre: dict) -> tuple:
    """One finished live row per grid cell. Allowed per cell: [finished] or [aborted_infra, finished] (one infra rerun,
    PROJECT_SPEC S7); anything else (a second finished run, a crash, a second rerun) is irregular and listed."""
    g = pre["grid"]
    rows, attempts, irregular = {}, [], []
    grid = {(s, t, sd, p) for s in g["scenes"] for t in g["tasks"] for sd in g["seeds"] for p in g["policies"]}
    for line in (root / "runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        k = (r.get("scene"), r.get("task"), r.get("seed"), r.get("policy"))
        attempts.append(r)
        if r.get("config_sha256") != pre["config_sha256"] or r.get("mode") != "live" or k not in grid:
            irregular.append({"cell": "/".join(map(str, k)), "status": r.get("status"), "why": "row of another config or mode, or outside the grid"})
            continue
        rows.setdefault(k, []).append(r)
    cells, missing = {}, []
    for s in g["scenes"]:
        for t in g["tasks"]:
            for sd in g["seeds"]:
                for p in g["policies"]:
                    k = (s, t, sd, p)
                    pattern = [infra(r) for r in rows.get(k, [])]
                    if pattern in (["ok"], ["fallback"], ["aborted_infra", "ok"], ["aborted_infra", "fallback"]):
                        d = root / s / t / f"s{sd}" / p
                        m = json.loads((d / "metrics.json").read_text())
                        check_metrics(k, m, rows[k][-1], pre, g["tasks"])
                        cells[k] = {"dir": d, "m": m, "row": rows[k][-1]}
                    elif not pattern or pattern == ["aborted_infra"]:
                        missing.append({"cell": "/".join(map(str, k)), "status": pattern or "no row"})
                    else:
                        irregular.append({"cell": "/".join(map(str, k)), "status": pattern})
    return cells, missing, irregular, attempts


def gain(cells, s, t, sd, p="llm"):
    return round(cells[(s, t, sd, p)]["m"]["q_t"] - cells[(s, t, sd, "b1")]["m"]["q_t"], 6)


def swapped(cells, s, t, sd, p, tasks):
    other = next(x for x in tasks if x != t)
    return round(cells[(s, other, sd, p)]["m"]["by_task"][t]["q_t"] - cells[(s, t, sd, "b1")]["m"]["q_t"], 6)


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def wilcoxon(x):
    from scipy.stats import wilcoxon as w
    return None if not any(x) else round(float(w(x, alternative="greater").pvalue), 4)


def first_task_object(cells, s, t, sd, cats):
    for e in json.loads((cells[(s, t, sd, "llm")]["dir"] / "confident.json").read_text()):
        if e["true"] in cats:
            return e["true"]
    return None


def revisit_events(cells, s, t, sd):
    out = []
    for line in (cells[(s, t, sd, "llm")]["dir"] / "decisions.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d["kind"] != "revisit" or d["fallback"] or d["n_frontiers"] < 1 or "after" not in d:
            continue
        b, a = d["before"], d["after"]
        if not b or len(a) != len(b) or not all(finite01(o.get("conf")) and isinstance(o.get("correct"), bool) for o in a + b):
            raise SystemExit(f"invalid revisit record in {s}/{t}/s{sd} at step {d.get('step')}")
        conf = lambda c: statistics.mean(o["conf"] for o in c)
        share = lambda c: sum(o["correct"] for o in c) / len(c)
        if conf(a) > conf(b) and share(a) > share(b):
            out.append({"scene": s, "task": t, "seed": sd, "step": d["step"]})
    return out


def judge(root: pathlib.Path, pre: dict, tests_log: str = None) -> dict:
    g, cfg = pre["grid"], json.loads(pathlib.Path(pre["config"]).read_text())
    check_grid(pre, cfg)
    tasks, seeds, scenes = g["tasks"], g["seeds"], g["scenes"]
    cells, missing, irregular, attempts = load(root, pre)
    rep = {"cells_expected": g["cells"], "cells_found": len(cells), "missing": missing, "irregular": irregular, "attempts": attempts}
    f = {"F5_grid": not missing and not irregular and len(cells) == g["cells"]}
    if not f["F5_grid"]:   # nothing else is judged on a partial or irregular grid
        return {**rep, "floor": f, "all_pass": False}
    ok = lambda s, t, sd: cells[(s, t, sd, "llm")]["m"].get("n_fallback", 0) == 0 and gain(cells, s, t, sd) >= LEAD
    wins = [(s, t, sd) for s in scenes for t in tasks for sd in seeds if ok(s, t, sd)]
    f["F1_successes"] = len(wins) >= 3 and len({w[0] for w in wins}) >= 3 and {w[1] for w in wins} == set(tasks)
    rep["successes"] = [list(w) for w in wins]
    wet, sleep = (set(cfg["tasks"][t]["categories"]) for t in ("T_wet", "T_sleep"))
    adapt = [[s, sd] for s in scenes for sd in seeds
             if first_task_object(cells, s, "T_wet", sd, wet | sleep) in wet and first_task_object(cells, s, "T_sleep", sd, wet | sleep) in sleep]
    f["F2_task_adaptation"], rep["adaptation_scene_seeds"] = bool(adapt), adapt
    rev = [e for s in scenes for t in tasks for sd in seeds for e in revisit_events(cells, s, t, sd)]
    f["F3_revisit"], rep["revisit_events"] = bool(rev), rev
    text = pathlib.Path(tests_log).read_text() if tests_log else ""   # pytest -rA output; no log, no pass
    f["F4_meta_transform"] = (all(re.search(rf"^PASSED {re.escape(t)}(\[[^\]]*\])?\s*$", text, re.M) for t in META_TESTS)
                              and not re.search(r"^(FAILED|ERROR)\b", text, re.M))
    rep["floor"], rep["all_pass"] = f, all(f.values())

    per = {p: [statistics.mean(gain(cells, s, t, sd, p) for t in tasks for sd in seeds) for s in scenes] for p in g["policies"] if p != "b1"}
    rep["stats"] = {p: {"mean_diff_vs_b1": round(float(np.mean(x)), 4), "ci95": boot_ci(x), "wilcoxon_p": wilcoxon(x)} for p, x in per.items()}
    cnt = lambda p, sign: sum(sign * gain(cells, s, t, sd, p) >= LEAD for s in scenes for t in tasks for sd in seeds)
    nul = lambda p: sum(swapped(cells, s, t, sd, p, tasks) >= LEAD for s in scenes for t in tasks for sd in seeds)
    n = len(scenes) * len(tasks) * len(seeds)
    rep["lead_lag_vs_b1"] = {p: [cnt(p, 1), cnt(p, -1)] for p in per}
    rep["llm_successes"] = f"{len(wins)}/{n}"
    rep["task_swap_null_successes"] = {p: nul(p) for p in ("llm", "b2") if p in g["policies"]}
    mid = lambda p, s, t, sd: cells[(s, t, sd, p)]["m"]["by_task_mid"]
    def s_h(p):
        out = []
        for s in scenes:
            dirs = [round(statistics.mean(mid(p, s, t, sd)[t]["q_t"] - mid(p, s, next(x for x in tasks if x != t), sd)[t]["q_t"] for sd in seeds), 6) for t in tasks]
            out.append(dirs)
        return out
    rep["S_h"] = {p: {"mean": round(float(np.mean([np.mean(d) for d in v])), 4), "ci95": boot_ci([np.mean(d) for d in v]),
                      "both_directions_ge_0.10": f"{sum(min(d) >= LEAD for d in v)}/{len(scenes)}"} for p in ("llm", "b1") for v in [s_h(p)]}
    P = rep["lead_lag_vs_b1"]
    rep["predictions"] = {"P1": P["llm"][0] > P["llm"][1],                                     # eval/prereg.json predictions
                          "P2": len(wins) <= rep["task_swap_null_successes"]["llm"] + 3,
                          "P3": P["llm"][1] <= 6}
    pairs = [(s, sd) for s in scenes for sd in seeds]
    both = [(s, sd) for s, sd in pairs if all(ok(s, t, sd) for t in tasks)]
    if both:
        g2 = {p: round(statistics.mean(gain(cells, p[0], t, p[1]) for t in tasks), 6) for p in both}
        med = sorted(g2.values())[(len(g2) - 1) // 2]   # the median gain (lower middle for an even count)
        rep["gif"] = {"pair": list(next(p for p in pairs if g2.get(p) == med)), "rule": "median gain, both tasks succeed"}
    else:
        one = [(s, sd) for s, sd in pairs if any(ok(s, t, sd) for t in tasks)]
        rep["gif"] = {"pair": list(one[0]) if one else None, "rule": "first pair in held-out order with a success; label every panel"}
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--prereg", default="eval/prereg.json")
    ap.add_argument("--tests-log")
    ap.add_argument("--out", help="write the report here (it lists every attempt of the batch, the results JSON of the spec)")
    a = ap.parse_args(argv)
    rep = judge(pathlib.Path(a.root), json.loads(pathlib.Path(a.prereg).read_text()), a.tests_log)
    text = json.dumps(rep, indent=1)
    if a.out:
        pathlib.Path(a.out).write_text(text + "\n")
    print(text)
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
