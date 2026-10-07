"""Freeze package: held-out selection, the freeze step, and the showcase-floor judge (synthetic artifacts, seconds, no API key).
Run: python -m pytest tests/test_freeze.py -q
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from amap import run as run_mod
from eval import check_floor, freeze
from eval.select_heldout import select

SALT = "am-sp-m-heldout-v1:"


def rows(scenes, seeds=(1001, 1002), area=60.0, passes=True):
    return [{"scene": s, "seed": sd, "nav_area_m2": area, "passes": passes, "start_xyz": [1.0, 0.0, 2.0]} for s in scenes for sd in seeds]


def test_select_orders_by_salted_hash_and_applies_the_area_window_and_both_seeds():
    scenes = [f"008{i:02d}-x" for i in range(10)]
    got = select(rows(scenes), [1001, 1002], [26.4, 242.5], SALT, 4)
    assert got == sorted(scenes, key=lambda s: hashlib.sha256(f"{SALT}{s}".encode()).hexdigest())[:4]
    r = rows(scenes) + []
    r[0]["nav_area_m2"] = 10.0                                    # one seed outside the window drops that scene
    r[3]["passes"] = False                                        # one seed failing the screening drops that scene
    got = select(r, [1001, 1002], [26.4, 242.5], SALT, 8)
    assert scenes[0] not in got and scenes[1] not in got and len(got) == 8
    with pytest.raises(SystemExit):                               # fewer than n eligible: the batch stops
        select(r, [1001, 1002], [26.4, 242.5], SALT, 9)
    with pytest.raises(SystemExit):                               # a scene screened for one seed only is not eligible
        select([x for x in rows(scenes) if x["seed"] == 1001], [1001, 1002], [26.4, 242.5], SALT, 1)


def make_freeze_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, text in (("prompt.txt", "p"), ("prior.json", '{"prior": [1]}'), ("metrics.py", "m"), ("sel.py", "s"), ("floor.py", "f")):
        pathlib.Path(name).write_text(text)
    cfg = {"name": "h", "prereg": "prereg.json", "scenes": [], "seeds": [1001, 1002], "policies": ["llm", "b1"], "tasks": {"T_wet": {}, "T_sleep": {}},
           "budget": 120, "tau": 0.7, "sensor": {"prior_file": "prior.json"}, "b2": {}, "llm": {"prompt": "prompt.txt"}, "grid": {}, "decision_every": 40,
           "lookaround_steps": 4, "candidates": {}, "habitat": {}, "state": {}, "start_min_dist_m": 2.0, "region_radius_m": 1.0, "curve_every": 10}
    pathlib.Path("cfg.json").write_text(json.dumps(cfg))
    pre = {"config": "cfg.json", "code_files": ["metrics.py", "sel.py", "floor.py"], "code_sha256": {},
           "heldout": {"salt": SALT, "n_scenes": 2, "seeds": [1001, 1002], "area_window_m2": [26.4, 242.5]}}
    pathlib.Path("prereg.json").write_text(json.dumps(pre))
    scenes = ["00800-a", "00801-b", "00802-c"]
    pathlib.Path("screen.jsonl").write_text("\n".join(json.dumps(r) for r in rows(scenes)) + "\n")
    return scenes


def test_freeze_fills_hashes_then_the_selection_and_the_runner_hash_matches(tmp_path, monkeypatch):
    scenes = make_freeze_inputs(tmp_path, monkeypatch)
    freeze.main(["--prereg", "prereg.json"])                       # part 1
    pre = json.loads(pathlib.Path("prereg.json").read_text())
    assert pre["prompt_sha256"] == run_mod.sha("p") and pre["code_sha256"]["floor.py"] == run_mod.sha("f")
    assert "config_sha256" not in pre                              # part 1 leaves the config hash for part 2
    picked = select(rows(scenes), [1001, 1002], [26.4, 242.5], SALT, 2)
    pathlib.Path("heldout.txt").write_text("\n".join(picked) + "\n")
    freeze.main(["--prereg", "prereg.json", "--screen", "screen.jsonl", "--heldout", "heldout.txt"])   # part 2
    pre, cfg = json.loads(pathlib.Path("prereg.json").read_text()), json.loads(pathlib.Path("cfg.json").read_text())
    assert cfg["scenes"] == picked and pre["heldout"]["scenes"] == picked
    assert pre["config_sha256"] == run_mod.sha(run_mod.canonical(cfg))   # what the runner will compute
    freeze.main(["--prereg", "prereg.json", "--verify"])
    freeze.main(["--prereg", "prereg.json", "--verify", "--screen", "screen.jsonl"])   # selection, starts and screen hash re-derived
    pathlib.Path("screen2.jsonl").write_text(pathlib.Path("screen.jsonl").read_text() + "\n")
    with pytest.raises(SystemExit):                                       # same rows, different bytes: the recorded hash catches it
        freeze.main(["--prereg", "prereg.json", "--verify", "--screen", "screen2.jsonl"])
    pathlib.Path("metrics.py").write_text("changed")                     # an evaluator file edited after the freeze is caught
    with pytest.raises(SystemExit):
        freeze.main(["--prereg", "prereg.json", "--verify"])
    assert pre["heldout"]["starts"][picked[0]]["1001"] == [1.0, 0.0, 2.0] and pre["grid"]["cells"] == 2 * 2 * 2 * 2


def test_freeze_rejects_a_selection_that_the_rule_does_not_give(tmp_path, monkeypatch):
    scenes = make_freeze_inputs(tmp_path, monkeypatch)
    freeze.main(["--prereg", "prereg.json"])
    wrong = [s for s in scenes if s not in select(rows(scenes), [1001, 1002], [26.4, 242.5], SALT, 2)]
    pathlib.Path("heldout.txt").write_text("\n".join(wrong) + "\n")
    with pytest.raises(SystemExit):
        freeze.main(["--prereg", "prereg.json", "--screen", "screen.jsonl", "--heldout", "heldout.txt"])


# ---- check_floor on synthetic artifacts -------------------------------------------------------------------------------

SC, TASKS, SEEDS, POLS = ["00800-a", "00801-b", "00802-c"], ["T_wet", "T_sleep"], [1001], ["llm", "b1", "b0", "b2"]


H = None   # the fixture config's hash, set by fixture()


def write_cell(root, s, t, sd, p, q, other_q=None, fallback=0, conf=None, decisions=None, status=None):
    d = root / s / t / f"s{sd}" / p
    d.mkdir(parents=True, exist_ok=True)
    other = next(x for x in TASKS if x != t)
    oq = q if other_q is None else other_q
    by = {t: {"q_t": q}, other: {"q_t": oq}}
    (d / "metrics.json").write_text(json.dumps({"config_sha256": H, "mode": "live", "q_t": q, "n_fallback": fallback, "by_task": by, "by_task_mid": by}))
    (d / "confident.json").write_text(json.dumps(conf or []))
    (d / "decisions.jsonl").write_text("\n".join(json.dumps(x) for x in (decisions or [])))
    row = {"scene": s, "task": t, "seed": sd, "policy": p, "status": status or ("fallback" if fallback else "ok"), "config_sha256": H,
           "mode": "live", "q_t": q, "n_fallback": fallback, "q_t_by_task": {k: v["q_t"] for k, v in by.items()},
           "q_t_by_task_mid": {k: v["q_t"] for k, v in by.items()}}
    with open(root / "runs.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


REVISIT = {"kind": "revisit", "fallback": False, "n_frontiers": 1, "step": 40,
           "before": [{"conf": 0.4, "correct": False}, {"conf": 0.5, "correct": True}],
           "after": [{"conf": 0.8, "correct": True}, {"conf": 0.9, "correct": True}]}
TESTS_LOG = "".join(f"PASSED {t}\n" for t in check_floor.META_TESTS) + "5 passed in 3s\n"


def fixture(tmp_path, llm_q=0.6, fallback=0, revisit=REVISIT, skip=None):
    global H
    root = tmp_path / "out"
    root.mkdir(parents=True)
    cfg = {"scenes": SC, "seeds": SEEDS, "policies": POLS, "tasks": {"T_wet": {"categories": ["sink", "toilet"]}, "T_sleep": {"categories": ["bed"]}}}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    H = run_mod.sha(run_mod.canonical(cfg))
    pre = {"config": str(tmp_path / "cfg.json"), "config_sha256": H, "heldout": {"n_scenes": 3, "seeds": SEEDS},
           "grid": {"scenes": SC, "tasks": TASKS, "seeds": SEEDS, "policies": POLS, "cells": 24}}
    for s in SC:
        for t in TASKS:
            for sd in SEEDS:
                first = {"T_wet": "sink", "T_sleep": "bed"}[t]
                conf = [{"step": 5, "oid": 1, "label": "x", "true": "chair"}, {"step": 9, "oid": 2, "label": first, "true": first}]
                for p in POLS:
                    if (s, t, p) == skip:
                        continue
                    q = {"llm": llm_q, "b1": 0.3, "b0": 0.2, "b2": 0.4}[p]
                    write_cell(root, s, t, sd, p, q, other_q=0.3, fallback=fallback if p == "llm" else 0, conf=conf if p == "llm" else None,
                               decisions=[revisit] if p == "llm" and revisit else None)
    return root, pre


def test_check_floor_passes_a_batch_that_meets_every_item(tmp_path):
    root, pre = fixture(tmp_path)
    log = tmp_path / "tests.log"
    log.write_text(TESTS_LOG)
    rep = check_floor.judge(root, pre, str(log))
    assert rep["all_pass"] and all(rep["floor"].values()) and len(rep["successes"]) == 6
    assert rep["lead_lag_vs_b1"]["b2"] == [6, 0] and rep["predictions"]["P1"] and rep["predictions"]["P3"]
    assert rep["gif"]["pair"][0] == "00800-a" and rep["task_swap_null_successes"]["llm"] == 0   # equal gains: the earliest; swapped scores sit at 0.3 = b1
    for bad in ("0 passed in 1s\n", "5 passed in 3s\n", TESTS_LOG.replace("PASSED tests/test_habitat", "SKIPPED tests/test_habitat"),
                TESTS_LOG + "FAILED tests/test_core_map.py::test_x\n"):
        log.write_text(bad)                                                          # both meta-transform tests must be listed as PASSED
        assert not check_floor.judge(root, pre, str(log))["floor"]["F4_meta_transform"]


def test_check_floor_fails_on_a_missing_cell_and_does_not_judge_the_rest(tmp_path):
    root, pre = fixture(tmp_path, skip=("00801-b", "T_sleep", "b0"))
    rep = check_floor.judge(root, pre, None)
    assert not rep["all_pass"] and not rep["floor"]["F5_grid"] and len(rep["missing"]) == 1 and "F1_successes" not in rep["floor"]


def test_check_floor_does_not_count_fallback_episodes_or_unqualified_revisits_or_a_missing_test_log(tmp_path):
    root, pre = fixture(tmp_path / "a", fallback=2)
    assert not check_floor.judge(root, pre, None)["floor"]["F1_successes"]
    root, pre = fixture(tmp_path / "b", revisit={**REVISIT, "n_frontiers": 0})        # a revisit with no frontier left does not count
    rep = check_floor.judge(root, pre, None)
    assert not rep["floor"]["F3_revisit"] and not rep["floor"]["F4_meta_transform"]   # no tests log: not a pass
    root, pre = fixture(tmp_path / "c", revisit={**REVISIT, "after": [{"conf": 0.9, "correct": False}, {"conf": 0.9, "correct": False}]})
    assert not check_floor.judge(root, pre, None)["floor"]["F3_revisit"]              # confidence up but correctness down


def test_check_floor_stops_on_artifacts_that_do_not_belong_to_their_row_or_a_grid_that_is_not_the_batch(tmp_path):
    root, pre = fixture(tmp_path / "a")
    m = root / "00800-a" / "T_wet" / "s1001" / "llm" / "metrics.json"
    m.write_text(m.read_text().replace('"q_t": 0.6', '"q_t": 0.9', 1))             # another run overwrote the artifacts
    with pytest.raises(SystemExit):
        check_floor.judge(root, pre, None)
    root, pre = fixture(tmp_path / "b")
    pre["grid"] = {**pre["grid"], "seeds": [1001, 1002], "cells": 48}              # the grid disagrees with the frozen config
    with pytest.raises(SystemExit):
        check_floor.judge(root, pre, None)


def test_check_floor_lists_a_rerun_of_a_finished_cell_as_irregular(tmp_path):
    root, pre = fixture(tmp_path)
    write_cell(root, "00800-a", "T_wet", 1001, "llm", 0.6)                        # a second finished run of the same cell
    rep = check_floor.judge(root, pre, None)
    assert not rep["all_pass"] and rep["irregular"] and not rep["floor"]["F5_grid"]


def test_selection_and_freeze_reject_malformed_screening_rows_and_a_changed_part1_file(tmp_path, monkeypatch):
    scenes = make_freeze_inputs(tmp_path, monkeypatch)
    good = rows(scenes)
    for bad in ([*good, good[0]],                                                 # a repeated (scene, seed)
                [*good[:-1], {**good[-1], "scene": "00999-z"}],                    # not a val scene id
                [*good[:-1], {**good[-1], "error": "GPU unavailable"}]):           # a loading error stops the selection
        with pytest.raises(SystemExit):
            select(bad, [1001, 1002], [26.4, 242.5], SALT, 2)
    for field, val in (("passes", "true"), ("start_xyz", None), ("start_xyz", [1, 2]), ("start_xyz", [1, float("nan"), 2])):
        r = [{**x, field: val} if x["scene"] == scenes[0] else x for x in good]    # that scene drops out of the eligible set
        assert scenes[0] not in select(r, [1001, 1002], [26.4, 242.5], SALT, 2)
    freeze.main(["--prereg", "prereg.json"])
    picked = select(good, [1001, 1002], [26.4, 242.5], SALT, 2)
    pathlib.Path("heldout.txt").write_text("\n".join(picked) + "\n")
    pathlib.Path("prompt.txt").write_text("changed after part 1")
    with pytest.raises(SystemExit):                                              # part 2 never re-freezes part 1 silently
        freeze.main(["--prereg", "prereg.json", "--screen", "screen.jsonl", "--heldout", "heldout.txt"])
    assert json.loads(pathlib.Path("cfg.json").read_text())["scenes"] == []      # and nothing was written


def test_freeze_refuses_a_config_whose_seeds_differ_from_the_rule(tmp_path, monkeypatch):
    make_freeze_inputs(tmp_path, monkeypatch)
    cfg = json.loads(pathlib.Path("cfg.json").read_text())
    pathlib.Path("cfg.json").write_text(json.dumps({**cfg, "seeds": [1001]}))
    freeze.main(["--prereg", "prereg.json"])
    pathlib.Path("heldout.txt").write_text("\n".join(select(rows(["00800-a", "00801-b", "00802-c"]), [1001, 1002], [26.4, 242.5], SALT, 2)) + "\n")
    with pytest.raises(SystemExit):
        freeze.main(["--prereg", "prereg.json", "--screen", "screen.jsonl", "--heldout", "heldout.txt"])


def test_part2_refuses_a_config_changed_since_part1_and_verify_checks_the_summary(tmp_path, monkeypatch):
    scenes = make_freeze_inputs(tmp_path, monkeypatch)
    freeze.main(["--prereg", "prereg.json"])
    pathlib.Path("heldout.txt").write_text("\n".join(select(rows(scenes), [1001, 1002], [26.4, 242.5], SALT, 2)) + "\n")
    cfg = json.loads(pathlib.Path("cfg.json").read_text())
    pathlib.Path("cfg.json").write_text(json.dumps({**cfg, "budget": 121}))        # drift between the two parts
    with pytest.raises(SystemExit):
        freeze.main(["--prereg", "prereg.json", "--screen", "screen.jsonl", "--heldout", "heldout.txt"])
    pathlib.Path("cfg.json").write_text(json.dumps(cfg))
    freeze.main(["--prereg", "prereg.json", "--screen", "screen.jsonl", "--heldout", "heldout.txt"])
    pre = json.loads(pathlib.Path("prereg.json").read_text())
    pathlib.Path("prereg.json").write_text(json.dumps({**pre, "config_summary": {**pre["config_summary"], "budget": 999}}))
    with pytest.raises(SystemExit):
        freeze.main(["--prereg", "prereg.json", "--verify"])


def test_check_floor_treats_an_api_timeout_crash_as_an_infra_abort_with_one_rerun(tmp_path):
    root, pre = fixture(tmp_path / "a", skip=("00800-a", "T_wet", "llm"))
    with open(root / "runs.jsonl", "a") as f:
        f.write(json.dumps({"scene": "00800-a", "task": "T_wet", "seed": 1001, "policy": "llm", "status": "crashed",
                            "error": "timeout: The read operation timed out", "config_sha256": H, "mode": "live"}) + "\n")
    write_cell(root, "00800-a", "T_wet", 1001, "llm", 0.6, other_q=0.3, conf=[{"step": 9, "oid": 2, "label": "sink", "true": "sink"}])
    assert check_floor.judge(root, pre, None)["floor"]["F5_grid"]                   # S7: one rerun after an infra abort
    root, pre = fixture(tmp_path / "b", skip=("00800-a", "T_wet", "llm"))
    with open(root / "runs.jsonl", "a") as f:
        f.write(json.dumps({"scene": "00800-a", "task": "T_wet", "seed": 1001, "policy": "llm", "status": "crashed",
                            "error": "KeyError: 'x'", "config_sha256": H, "mode": "live"}) + "\n")
    write_cell(root, "00800-a", "T_wet", 1001, "llm", 0.6, other_q=0.3)
    assert not check_floor.judge(root, pre, None)["floor"]["F5_grid"]               # any other crash stays irregular


def test_train_split_selection_with_an_exclusion_list():
    scenes = [f"001{i:02d}-t" for i in range(8)]
    got = select(rows(scenes), [1001, 1002], [26.4, 242.5], SALT, 3, "train", exclude=["00999-x"])
    assert got == sorted(scenes, key=lambda s: hashlib.sha256(f"{SALT}{s}".encode()).hexdigest())[:3]
    with pytest.raises(SystemExit):                                              # a used scene may not be held out
        select(rows(scenes), [1001, 1002], [26.4, 242.5], SALT, 3, "train", exclude=[scenes[0]])
    with pytest.raises(SystemExit):                                              # a val id is not a train scene
        select(rows(["00800-a"] + scenes), [1001, 1002], [26.4, 242.5], SALT, 3, "train")


def test_check_floor_exempts_only_llm_api_timeouts_and_binds_every_task_score_to_the_row(tmp_path):
    root, pre = fixture(tmp_path / "a", skip=("00800-a", "T_wet", "b1"))
    with open(root / "runs.jsonl", "a") as f:                                    # b1 calls no API: its timeout is a crash
        f.write(json.dumps({"scene": "00800-a", "task": "T_wet", "seed": 1001, "policy": "b1", "status": "crashed",
                            "error": "TimeoutError: navigation worker timed out", "config_sha256": H, "mode": "live"}) + "\n")
    write_cell(root, "00800-a", "T_wet", 1001, "b1", 0.3, other_q=0.3)
    assert not check_floor.judge(root, pre, None)["floor"]["F5_grid"]
    root, pre = fixture(tmp_path / "b")
    m = root / "00800-a" / "T_wet" / "s1001" / "llm" / "metrics.json"
    d = json.loads(m.read_text()); d["by_task"]["T_sleep"]["q_t"] = 0.99; m.write_text(json.dumps(d))
    with pytest.raises(SystemExit):                                              # the other task's score was altered
        check_floor.judge(root, pre, None)


def test_compare_replay_needs_every_finished_episode_reproduced_exactly(tmp_path):
    from eval import compare_replay
    root, pre = fixture(tmp_path / "a")
    rows = [json.loads(l) for l in (root / "runs.jsonl").read_text().splitlines()]
    for r in rows:                                                              # what the runner records per row
        m = json.loads((root / r["scene"] / r["task"] / f"s{r['seed']}" / r["policy"] / "metrics.json").read_text())
        r.update(coverage=m.get("coverage"), s_t=m.get("s_t"), steps_used=m.get("steps_used"))
    res = tmp_path / "res.json"
    res.write_text(json.dumps({"cells_expected": 24, "attempts": rows}))
    assert compare_replay.main(["--replay", str(root), "--results", str(res)]) == 0
    m = root / "00800-a" / "T_wet" / "s1001" / "b1" / "metrics.json"
    d = json.loads(m.read_text()); d["q_t"] = 0.31; m.write_text(json.dumps(d))
    assert compare_replay.main(["--replay", str(root), "--results", str(res)]) == 1   # one score differs
