"""End-to-end checks on the L backend (seconds, no API key): truth firewall (metamorphic), cache replay, fallback,
preregistration refusal. Run: python -m pytest tests/test_episode.py -q
"""
from __future__ import annotations

import copy
import json
import pathlib

import numpy as np
import pytest

from amap import llm as llm_mod
from amap import lworld, policies, run as run_mod
from amap.contracts import UNKNOWN, GridSpec
from amap.episode import run_episode

CFG = json.loads(pathlib.Path("configs/dev_l.json").read_text())
HOUSES = lworld.load_tune("data/procthor/tune.jsonl")
TASK = {"name": "T_wet", **CFG["tasks"]["T_wet"]}
AVOID = sorted({c for t in CFG["tasks"].values() for c in t["categories"]})


def backend(house, seed=1, start_world=None):
    s = CFG["sensor"]
    return lworld.ProcthorBackend(house, GridSpec(CFG["grid"]["n"], CFG["grid"]["cell"]), seed, fov_deg=s["fov_deg"],
                                  range_m=s["range_m"], p_bins=tuple(s["p_bins"]), avoid_categories=AVOID, start_world=start_world)


def episode(be, policy, budget=150, **cand):
    return run_episode(be, policy, {**CFG, "budget": budget, "candidates": {**CFG["candidates"], **cand}}, TASK, 1)


def touched_rooms(house, be, known) -> set:
    """Rooms with any observed cell within ~0.4 m of their polygon (door thresholds count as touched)."""
    n, cell = be.grid.n, be.grid.cell
    i0, j0 = be.start_world
    out = set()
    for r, c in zip(*np.nonzero(known)):
        x, z = (c + j0 - n // 2 + 0.5) * cell, (r + i0 - n // 2 + 0.5) * cell
        for dx, dz in ((0, 0), (0.4, 0), (-0.4, 0), (0, 0.4), (0, -0.4)):
            rid = lworld.room_of(house, x + dx, z + dz)
            if rid is not None:
                out.add(rid)
    return out


def fingerprint(res):
    return [d["state"] for d in res["decisions"]], res["trajectory"]


@pytest.mark.parametrize("scene", ["train-00516", "train-02808", "train-05485", "train-08169", "train-06478", "train-09533"])
def test_metamorphic_unobserved_changes_are_invisible(scene):
    house = HOUSES[scene]
    be = backend(house)
    base = episode(be, policies.B1(), budget=60)
    seen = touched_rooms(house, be, base["amap"].occ != UNKNOWN)
    hidden = [r["id"] for r in house["rooms"] if r["id"] not in seen]
    if not hidden:
        pytest.skip(f"{scene}: every room touched within the budget")
    rid = hidden[0]
    for name, mod in (("drop_room", lambda h: lworld.drop_room(h, rid)),
                      ("move_objects", lambda h: lworld.move_objects(h, rid, 0.3, -0.2)),
                      ("relabel_objects", lambda h: lworld.relabel_objects(h, rid, "Toilet"))):
        h2 = mod(copy.deepcopy(house))
        res = episode(backend(h2, start_world=be.start_world), policies.B1(), budget=60)
        assert fingerprint(res) == fingerprint(base), f"{scene}/{name}: unobserved change leaked into input or route"


@pytest.mark.parametrize("scene", ["train-00516", "train-02808", "train-05485"])
def test_metamorphic_with_object_viewpoints_and_a_revisiting_policy(scene):
    """The object-anchored inspect candidates (redesign 10-07) read the agent map only: B0 takes revisits too."""
    house = HOUSES[scene]
    be = backend(house)
    kw = {"revisit_mode": "object", "k_revisit": 3}
    base = episode(be, policies.B0(), budget=80, **kw)
    seen = touched_rooms(house, be, base["amap"].occ != UNKNOWN)
    hidden = [r["id"] for r in house["rooms"] if r["id"] not in seen]
    if not hidden:
        pytest.skip(f"{scene}: every room touched within the budget")
    for mod in (lambda h: lworld.drop_room(h, hidden[0]), lambda h: lworld.relabel_objects(h, hidden[0], "Toilet")):
        res = episode(backend(mod(copy.deepcopy(house)), start_world=be.start_world), policies.B0(), budget=80, **kw)
        assert fingerprint(res) == fingerprint(base)


class FakeAPI:
    """Stands in for the HTTP call: picks the last candidate id from the enum (or an invalid id)."""

    def __init__(self, invalid=False):
        self.invalid, self.calls = invalid, 0

    def __call__(self, body):
        self.calls += 1
        ids = body["tools"][0]["function"]["parameters"]["properties"]["candidate_id"]["enum"]
        args = {"candidate_id": "zz" if self.invalid else ids[-1], "reason": "test", "predicted_beyond": "bathroom"}
        return {"choices": [{"message": {"tool_calls": [{"function": {"arguments": json.dumps(args)}}]}}],
                "model": body["model"], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}, 0.01


def llm_policy(cache, live, fake=None):
    c = llm_mod.LLMClient(cache, live=live, max_usd=1.0)
    if fake is not None:
        c._post = fake
    return policies.LLMPolicy(c, CFG["llm"]["model"], "system", 300)


def test_replay_is_byte_identical_and_never_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod, "spent_usd", lambda *_: 0.0)
    cache = tmp_path / "c.jsonl"
    be = backend(HOUSES["train-03126"])
    fake = FakeAPI()
    live = episode(be, llm_policy(cache, True, fake), budget=120)
    assert fake.calls > 0
    replay = episode(backend(HOUSES["train-03126"]), llm_policy(cache, False), budget=120)
    assert fingerprint(replay) == fingerprint(live)
    assert [d["cid"] for d in replay["decisions"]] == [d["cid"] for d in live["decisions"]]


def test_replay_missing_key_fails_loudly(tmp_path):
    with pytest.raises(llm_mod.CacheMiss):
        episode(backend(HOUSES["train-03126"]), llm_policy(tmp_path / "empty.jsonl", False), budget=60)


def test_invalid_tool_calls_fall_back_to_b1_and_are_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod, "spent_usd", lambda *_: 0.0)
    fake = FakeAPI(invalid=True)
    res = episode(backend(HOUSES["train-03126"]), llm_policy(tmp_path / "c.jsonl", True, fake), budget=60)
    assert res["n_fallback"] == len(res["decisions"]) > 0
    assert fake.calls == 3 * len(res["decisions"])  # first try + 2 retries per decision


def test_runner_refuses_a_config_that_differs_from_prereg(tmp_path):
    cfg = {**CFG, "out": str(tmp_path / "out"), "prereg": str(tmp_path / "prereg.json"), "scenes": ["train-03126"]}
    (tmp_path / "prereg.json").write_text(json.dumps({"config_sha256": "0" * 64, "prompt_sha256": "0" * 64}))
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    assert run_mod.main(["--config", str(tmp_path / "cfg.json"), "--policies", "b1"]) == 3


def test_runner_refuses_a_changed_prior_file_and_runs_when_everything_matches(tmp_path):
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"prior": [1.0] * 3}))
    cfg = {**CFG, "out": str(tmp_path / "out"), "prereg": str(tmp_path / "prereg.json"), "scenes": ["train-03126"],
           "sensor": {**CFG["sensor"], "prior_file": str(prior)}}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    good = {"config_sha256": run_mod.sha(run_mod.canonical(cfg)), "prompt_sha256": run_mod.sha(pathlib.Path(cfg["llm"]["prompt"]).read_text()),
            "prior_sha256": run_mod.sha(prior.read_text())}
    (tmp_path / "prereg.json").write_text(json.dumps({**good, "prior_sha256": "0" * 64}))
    assert run_mod.main(["--config", str(tmp_path / "cfg.json"), "--policies", "b1"]) == 3      # prior differs
    (tmp_path / "prereg.json").write_text(json.dumps(good))
    assert run_mod.main(["--config", str(tmp_path / "cfg.json"), "--policies", "b1", "--budget", "40"]) == 3   # no overrides
    assert run_mod.main(["--config", str(tmp_path / "cfg.json"), "--policies", "b1"]) == 0      # all hashes match: runs
    for flag, val in (("--policies", "b0"), ("--scenes", "train-00000"), ("--seeds", "9"), ("--tasks", "T_other")):
        assert run_mod.main(["--config", str(tmp_path / "cfg.json"), flag, val]) == 3, flag   # outside the frozen grid
    assert run_mod.main(["--config", str(tmp_path / "cfg.json"), "--policies", "b1", "--tasks", "T_wet"]) == 0   # a slice is fine


def test_runner_refuses_a_start_that_differs_from_the_preregistered_one(tmp_path, monkeypatch):
    cfg = {**CFG, "out": str(tmp_path / "out"), "prereg": str(tmp_path / "prereg.json"), "scenes": ["train-03126"]}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg))
    good = {"config_sha256": run_mod.sha(run_mod.canonical(cfg)), "prompt_sha256": run_mod.sha(pathlib.Path(cfg["llm"]["prompt"]).read_text()),
            "prior_sha256": None}
    real = run_mod.make_backend

    def with_start(*a):
        be = real(*a)
        be.start_xyz = [1.0, 0.0, 2.0]
        return be
    monkeypatch.setattr(run_mod, "make_backend", with_start)
    for start, ok in (([1.0, 0.0, 2.0], True), ([1.0, 0.0, 2.5], False)):
        (tmp_path / "prereg.json").write_text(json.dumps({**good, "heldout": {"starts": {"train-03126": {"1": start}}}}))
        if ok:
            assert run_mod.main(["--config", str(tmp_path / "cfg.json"), "--policies", "b1", "--tasks", "T_wet"]) == 0
        else:
            with pytest.raises(RuntimeError, match="preregistered"):
                run_mod.main(["--config", str(tmp_path / "cfg.json"), "--policies", "b1", "--tasks", "T_wet"])


def test_a_socket_read_timeout_is_retried_not_a_crash(tmp_path, monkeypatch):
    """Held-out 10-07: on Python 3.9 socket.timeout is not a TimeoutError and escaped the retry loop."""
    import io
    import socket
    monkeypatch.setattr(llm_mod, "secret", lambda name: "k")
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    calls = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise socket.timeout("The read operation timed out")
        return Resp(json.dumps({"choices": [{"message": {}}]}).encode())
    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", urlopen)
    resp, _ = llm_mod.LLMClient(tmp_path / "c.jsonl", live=True)._post({"model": "m"})
    assert len(calls) == 2 and "choices" in resp
