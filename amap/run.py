"""One command: run every (scene, task, seed, policy) cell of a config and write the anchor output for each run
(map, path, decision log, metrics) plus a summary of all attempts.

  python -m amap.run --config configs/dev_l.json            # replay: LLM answers from the committed cache only
  python -m amap.run --config configs/dev_l.json --live     # calls the API on cache misses; appends <out>/runs.jsonl

Replay writes to <out>/replay/ and never appends runs.jsonl (a replay is not a new attempt); a missing cache key
exits non-zero. Exit code 0 only if every cell finished.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

from .contracts import CATEGORIES, GridSpec
from . import llm as llm_mod
from . import policies as P
from .episode import run_episode


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def make_backend(cfg, scene, seed):
    grid = GridSpec(cfg["grid"]["n"], cfg["grid"]["cell"])
    avoid = sorted({c for t in cfg["tasks"].values() for c in t["categories"]})
    s = cfg["sensor"]
    if cfg["backend"] == "procthor":
        if s.get("confusion", "uniform") != "uniform" or s.get("dist", "median") != "median":
            raise NotImplementedError("the L backend implements only the uniform noise model and the median distance")
        from .lworld import ProcthorBackend, load_tune
        house = load_tune(cfg["data"])[scene]
        return ProcthorBackend(house, grid, seed, fov_deg=s["fov_deg"], range_m=s["range_m"], p_bins=tuple(s["p_bins"]),
                               avoid_categories=avoid)
    if cfg["backend"] == "habitat":
        from .habitat_env import HabitatBackend
        return HabitatBackend(cfg, scene, grid, seed, avoid_categories=avoid)
    raise ValueError(cfg["backend"])


def make_policy(name, cfg, task, client, system_prompt):
    if name == "llm":
        return P.LLMPolicy(client, cfg["llm"]["model"], system_prompt, cfg["llm"].get("max_tokens", 300), cfg["llm"].get("render", "v5"))
    if name == "b1":
        return P.B1()
    if name == "b0":
        return P.B0()
    if name == "b2":
        return P.B2(task["categories"], **cfg.get("b2", {}))
    if name == "ref":
        return P.REF(task["categories"])
    if name == "refg":
        return P.REF(task["categories"], geodesic=True)
    raise ValueError(name)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def write_run(d: pathlib.Path, res: dict, record: bool, stamp: dict = None):
    d.mkdir(parents=True, exist_ok=True)
    amap = res["amap"]
    objs = amap.objects_summary()
    np.savez_compressed(d / "map.npz", occ=amap.occ, hit=amap._hit, miss=amap._miss,
                        obj_rc=np.array([o[3] for o in objs], dtype=np.int32).reshape(-1, 2),
                        obj_posterior=np.array([amap.posterior(o[0]) for o in objs]).reshape(-1, len(CATEGORIES)))
    (d / "trajectory.json").write_text(json.dumps(res["trajectory"]))
    with open(d / "decisions.jsonl", "w") as f:
        for r in res["decisions"]:
            f.write(canonical(r) + "\n")
    (d / "metrics.json").write_text(json.dumps({**(stamp or {}), **res["metrics"], "steps_used": res["steps_used"], "end_reason": res["end_reason"],
                                                "n_decisions": len(res["decisions"]), "n_fallback": res["n_fallback"],
                                                "by_task": res["metrics_by_task"], "by_task_mid": res["metrics_by_task_mid"]}, indent=1))
    (d / "curve.json").write_text(json.dumps(res["curve"]))
    (d / "confident.json").write_text(json.dumps(res["confident"]))
    try:
        from .viz import save_map_png, save_frames
        save_map_png(d / "map.png", res)
        if record:
            save_frames(d / "frames.npz", res)
    except ImportError:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--live", action="store_true", help="call the LLM API on cache misses (default: replay only)")
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--policies", nargs="*")
    ap.add_argument("--record", action="store_true", help="also save per-decision map snapshots for the GIF")
    ap.add_argument("--budget", type=int, help="calibration only (Phase 2): override the step budget")
    ap.add_argument("--out", help="calibration only: override the output directory")
    a = ap.parse_args(argv)

    cfg = json.loads(pathlib.Path(a.config).read_text())
    if cfg.get("prereg") and (a.budget is not None or a.out is not None):
        print("REFUSED: --budget/--out are for calibration; a preregistered config runs as frozen", file=sys.stderr)
        return 3
    if cfg.get("prereg"):   # a frozen config may be run in slices (one policy per process), never outside its own grid
        for flag in ("scenes", "tasks", "seeds", "policies"):
            given = getattr(a, flag)
            if given is None:
                continue
            extra = set(given) - set(cfg[flag])
            if extra or not given or len(set(given)) != len(given):
                print(f"REFUSED: --{flag} {given} is empty, repeats a value or leaves the preregistered config", file=sys.stderr)
                return 3
    if a.budget is not None:
        cfg["budget"] = a.budget
    if a.out is not None:
        cfg["out"] = a.out
    system_prompt = pathlib.Path(cfg["llm"]["prompt"]).read_text()
    cfg_sha, prompt_sha = sha(canonical(cfg)), sha(system_prompt)
    pre = {}
    if cfg.get("prereg"):
        pre = json.loads(pathlib.Path(cfg["prereg"]).read_text())
        prior_file = cfg["sensor"].get("prior_file")
        prior_sha = sha(pathlib.Path(prior_file).read_text()) if prior_file else None   # the prior shapes the noise model
        code = pre.get("code_sha256") or {}
        # live runs (new attempts) must use the frozen code; a replay is checked by its outputs instead (send-line acceptance
        # runs on the published snapshot, whose internal comments are stripped)
        code_ok = not a.live or (bool(code) and all(h and pathlib.Path(f).exists() and sha(pathlib.Path(f).read_text()) == h for f, h in code.items()))
        if pre.get("config_sha256") != cfg_sha or pre.get("prompt_sha256") != prompt_sha or pre.get("prior_sha256") != prior_sha or not code_ok:
            print("REFUSED: config, prompt, prior or evaluator-code hash differs from the preregistration", file=sys.stderr)
            return 3
    out = pathlib.Path(cfg["out"]) / ("" if a.live else "replay")
    client = llm_mod.LLMClient(cfg["llm"]["cache"], live=a.live, max_usd=cfg["llm"].get("max_usd", 4.0))
    scenes = a.scenes or cfg["scenes"]
    tasks = a.tasks or list(cfg["tasks"])
    seeds = a.seeds or cfg["seeds"]
    pols = a.policies or cfg["policies"]
    commit = git_commit()
    rows, failed = [], 0
    for scene in scenes:
        for seed in seeds:
            for tname in tasks:
                task = {"name": tname, **cfg["tasks"][tname]}
                for pname in pols:
                    cell = {"scene": scene, "task": tname, "seed": seed, "policy": pname}
                    t0 = time.time()
                    row = {**cell, "config": cfg["name"], "config_sha256": cfg_sha, "prompt_sha256": prompt_sha,
                           "commit": commit, "mode": "live" if a.live else "replay"}
                    try:
                        backend = make_backend(cfg, scene, seed)
                        starts = pre.get("heldout", {}).get("starts")
                        rec = starts.get(scene, {}).get(str(seed)) if starts else None
                        got = [round(float(v), 3) for v in getattr(backend, "start_xyz", [])]
                        if starts and rec != got:   # once starts are preregistered, the seeded start must be the recorded one
                            raise RuntimeError(f"start {got} differs from the preregistered {rec}")
                        res = run_episode(backend, make_policy(pname, cfg, task, client, system_prompt), cfg, task, seed, record=a.record)
                        write_run(out / scene / tname / f"s{seed}" / pname, res, a.record,
                                  {"config_sha256": cfg_sha, "mode": row["mode"]})   # binds the artifacts to the run row
                        row.update(status="fallback" if res["n_fallback"] else "ok", **res["metrics"],
                                   steps_used=res["steps_used"], n_decisions=len(res["decisions"]), n_fallback=res["n_fallback"],
                                   q_t_by_task={t: m["q_t"] for t, m in res["metrics_by_task"].items()},
                                   q_t_by_task_mid={t: m["q_t"] for t, m in res["metrics_by_task_mid"].items()})
                    except llm_mod.CacheMiss as e:
                        print(f"{cell}: {e}", file=sys.stderr)
                        return 2
                    except (llm_mod.InfraError, llm_mod.BudgetExceeded) as e:
                        row.update(status="aborted_infra", error=str(e)); failed += 1
                    except Exception as e:  # a crash is an attempt too: listed, not hidden
                        row.update(status="crashed", error=f"{type(e).__name__}: {e}"); failed += 1
                        if not a.live:
                            raise
                    row["wall_s"] = round(time.time() - t0, 2)
                    rows.append(row)
                    stop = cfg.get("prereg") and row["status"] not in ("ok", "fallback")   # a frozen batch stops at its first failure
                    print(json.dumps({k: row.get(k) for k in ("scene", "task", "seed", "policy", "status", "q_t", "coverage", "s_t", "steps_used", "wall_s")}), flush=True)
                    if a.live:
                        out.mkdir(parents=True, exist_ok=True)
                        with open(out / "runs.jsonl", "a") as f:
                            f.write(canonical(row) + "\n")
                    if stop:
                        print(f"STOPPED: {row['status']} in a preregistered batch: {row.get('error')}", file=sys.stderr)
                        return 1
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"config": cfg["name"], "config_sha256": cfg_sha, "prompt_sha256": prompt_sha,
                                                   "runs": rows}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
