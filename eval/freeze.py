"""Freeze step: fill the derived fields of eval/prereg.json. It never changes the hand-written parts.

  python -m eval.freeze --prereg eval/prereg.json          # part 1: hashes of the prompt, prior, code and config template
  python -m eval.freeze --prereg eval/prereg.json --screen out/screen_val.jsonl --heldout out/heldout.txt
                                                            # part 2: selected scenes, starts, config hash
  python -m eval.freeze --prereg eval/prereg.json --verify [--screen out/screen_val.jsonl]
                                                            # check every recorded hash (and the selection) against the files

Part 2 first checks the part-1 hashes (a file changed since part 1 stops it: re-freezing is a separate, explicit step),
re-derives the selection from the val screening rows and the preregistered rule (a heldout.txt that disagrees is an
error), checks that the config's grid is the preregistered one, and only then writes the scenes into the config (so the
config hash covers them) and the hash into the preregistration. Run it before the first held-out episode; the runner
refuses a config, prompt, prior or evaluator file whose hash differs from what is recorded here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from amap.run import canonical, sha
from eval.select_heldout import excluded, select

SUMMARY = ("tasks", "seeds", "policies", "budget", "tau", "sensor", "b2", "llm", "grid", "decision_every", "lookaround_steps",
           "candidates", "habitat", "state", "start_min_dist_m", "region_radius_m", "curve_every")


def required_code(pre: dict) -> list:
    """Every file a live held-out run or its judgement executes: the whole amap package and the selection, screening and
    floor scripts (the runner refuses a live run when any of them differs). `code_files` overrides it (tests only)."""
    return pre.get("code_files") or (sorted(str(f) for f in pathlib.Path("amap").glob("*.py"))
                                     + ["eval/check_floor.py", "eval/screen_scenes.py", "eval/select_heldout.py"])


def file_sha(path: str) -> str:
    return sha(pathlib.Path(path).read_text())


def template_sha(cfg: dict) -> str:
    """The config without its scene list: fixed at part 1, so nothing but the selected scenes can change at part 2."""
    return sha(canonical({**cfg, "scenes": []}))


HELDOUT_FIELDS = ("name", "prereg", "data", "out", "scenes", "seeds", "policies")


def check_arm(pre: dict, cfg: dict):
    """The held-out config is the arm the decision boot selected, apart from the held-out fields and the cache path."""
    if "arm_config" not in pre:
        return
    arm = json.loads(pathlib.Path(pre["arm_config"]).read_text())
    strip = lambda c: {k: ({**v, "cache": None} if k == "llm" else v) for k, v in c.items() if k not in HELDOUT_FIELDS}
    if strip(arm) != strip(cfg):
        diff = sorted(k for k in set(strip(arm)) | set(strip(cfg)) if strip(arm).get(k) != strip(cfg).get(k))
        raise SystemExit(f"{pre['config']} differs from the selected arm {pre['arm_config']} in {diff}")


def part1(pre: dict, cfg: dict) -> dict:
    return {"prompt_sha256": file_sha(cfg["llm"]["prompt"]), "prior_sha256": file_sha(cfg["sensor"]["prior_file"]),
            "code_sha256": {f: file_sha(f) for f in required_code(pre)}, "config_template_sha256": template_sha(cfg)}


def selection(pre: dict, cfg: dict, screen: str) -> tuple:
    """(picked scenes, starts, screen hash) from the screening rows under the preregistered rule; checks the config's grid."""
    rule = pre["heldout"]
    rows = [json.loads(l) for l in open(screen) if l.strip()]
    picked = select(rows, rule["seeds"], rule["area_window_m2"], rule["salt"], rule["n_scenes"], rule.get("split", "val"), excluded(rule))
    if cfg["seeds"] != rule["seeds"]:
        raise SystemExit(f"config seeds {cfg['seeds']} differ from the preregistered seeds {rule['seeds']}")
    by = {(r["scene"], r["seed"]): r for r in rows}
    starts = {s: {str(sd): by[(s, sd)]["start_xyz"] for sd in rule["seeds"]} for s in picked}
    return picked, starts, hashlib.sha256(pathlib.Path(screen).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--screen")
    ap.add_argument("--heldout")
    ap.add_argument("--verify", action="store_true", help="check the recorded hashes against the files and write nothing")
    a = ap.parse_args(argv)
    pre_path = pathlib.Path(a.prereg)
    pre = json.loads(pre_path.read_text())
    cfg_path = pathlib.Path(pre["config"])
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("prereg") != a.prereg:
        raise SystemExit(f"{cfg_path} points at {cfg.get('prereg')}, not {a.prereg}")
    check_arm(pre, cfg)
    want = part1(pre, cfg)

    if a.verify:   # every recorded hash must match its file; the config hash and selection once part 2 has recorded them
        if pre.get("config_sha256"):
            want["config_sha256"] = sha(canonical(cfg))
        bad = [k for k, v in want.items() if pre.get(k) != v]
        if "config_summary" in pre and pre["config_summary"] != {k: cfg[k] for k in SUMMARY}:
            bad.append("config_summary")
        if a.screen:
            picked, starts, h = selection(pre, cfg, a.screen)
            rule = pre["heldout"]
            bad += [k for k, v in (("scenes", picked), ("starts", starts), ("screen_sha256", h)) if rule.get(k) != v]
            bad += ["config scenes"] if cfg["scenes"] != picked else []
        if bad:
            raise SystemExit(f"hash mismatch against the preregistration: {bad}")
        print("preregistration hashes match" + (" (selection re-derived)" if a.screen else ""))
        return

    if a.screen or a.heldout:
        if not (a.screen and a.heldout):
            raise SystemExit("part 2 needs both --screen and --heldout")
        changed = [k for k, v in want.items() if pre.get(k) != v]
        if changed:
            raise SystemExit(f"part 1 is not in place or a frozen file changed since: {changed}")
        missing = [k for k in SUMMARY if k not in cfg]
        if missing:
            raise SystemExit(f"config lacks {missing}")
        picked, starts, h = selection(pre, cfg, a.screen)
        listed = [l.strip() for l in open(a.heldout) if l.strip()]
        if listed != picked:
            raise SystemExit(f"{a.heldout} disagrees with the selection rule applied to {a.screen}")
        cfg["scenes"] = picked
        grid = {"scenes": picked, "tasks": list(cfg["tasks"]), "seeds": cfg["seeds"], "policies": cfg["policies"],
                "cells": len(picked) * len(cfg["tasks"]) * len(cfg["seeds"]) * len(cfg["policies"])}
        pre["heldout"].update(scenes=picked, starts=starts, screen_sha256=h)
        pre.update(config_sha256=sha(canonical(cfg)), config_summary={k: cfg[k] for k in SUMMARY}, grid=grid)
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")   # every check above passed: now write both files
    else:
        pre.update(want)
    pre_path.write_text(json.dumps(pre, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: pre.get(k) for k in ("config_sha256", "prompt_sha256", "prior_sha256", "code_sha256")}, indent=1))


if __name__ == "__main__":
    main()
