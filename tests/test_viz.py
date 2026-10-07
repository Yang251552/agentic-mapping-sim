"""amap.viz on synthetic runs (no backend needed): map.png geometry, frames.npz round trip, a multi-frame GIF.
Run: python -m pytest tests/test_viz.py -q"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from amap import viz
from amap.agent_map import AgentMap
from amap.candidates import Candidate
from amap.contracts import CAT, CATEGORIES, Detection, GridSpec, Observation, UNKNOWN

N, CELL = 40, 0.5


def fake_res(n_frames=3, long_reason=False):
    amap = AgentMap(GridSpec(N, CELL), len(CATEGORIES), (0.8, 0.65, 0.55), 0.7)
    free = np.array([(r, c) for r in range(15, 25) for c in range(14, 30)])
    occ = np.array([(14, c) for c in range(14, 30)] + [(r, 30) for r in range(14, 25)])
    dets = [Detection(1, (18, 22), 0, CAT["sink"]), Detection(2, (21, 17), 2, CAT["bed"])]
    snaps = []
    for k in range(n_frames):
        lim = 18 + 4 * k   # more columns known each decision
        amap.integrate(Observation(free[free[:, 1] <= lim], occ[occ[:, 1] <= lim], dets if k else []))
        snaps.append(amap.occ.copy())
    objs = amap.objects_summary()
    why = "visit the nearest door because " + "the map beyond it is still unknown and the task needs more rooms " * 6
    frames = []
    for k, s in enumerate(snaps):
        cands = [Candidate("f1", "frontier", (20, 20 + 2 * k), 5, 6, (), ((20, 20), (21, 20))),
                 Candidate("r1", "revisit", (19, 21), 8, 1, (0,), ())]
        frames.append({"step": 20 * k + 4, "occ": s, "rc": (20, 19 + k), "cands": cands, "cid": "r1" if k == 1 else "f1",
                       "nearest": "f1", "reason": why if long_reason else f"go {k}", "objects": objs if k else []})
    traj = [[i + 1, 20, 18 + i // 6, 0] for i in range(20 * n_frames)]
    return {"amap": amap, "frames": frames, "trajectory": traj, "steps_used": traj[-1][0], "end_reason": "budget"}


def write_run(d, res, q_end):
    d.mkdir(parents=True)
    viz.save_frames(d / "frames.npz", res)
    (d / "trajectory.json").write_text(json.dumps(res["trajectory"]))
    (d / "curve.json").write_text(json.dumps([{"step": s, "q_t": min(1.0, q_end * s / 60)} for s in range(10, 61, 10)]))
    (d / "metrics.json").write_text(json.dumps({"steps_used": res["steps_used"]}))


def test_map_png_is_known_area_plus_margin(tmp_path):
    res = fake_res()
    viz.save_map_png(tmp_path / "m.png", res)
    rows, cols = np.nonzero(res["amap"].occ != UNKNOWN)
    m = int(round(viz.MARGIN_M / CELL))
    h, w = rows.max() - rows.min() + 1 + 2 * m, cols.max() - cols.min() + 1 + 2 * m
    assert Image.open(tmp_path / "m.png").size == (w * viz.PX, h * viz.PX)


def test_frames_round_trip(tmp_path):
    res = fake_res(long_reason=True)
    viz.save_frames(tmp_path / "f.npz", res)
    frames, info = viz.load_frames(tmp_path / "f.npz")
    assert len(frames) == len(res["frames"]) + 1 and frames[-1]["final"] and not frames[0]["final"]
    r0, c0 = info["offset"]
    for f, g in zip(res["frames"], frames):
        h, w = g["occ"].shape
        assert g["occ"].dtype == np.int8 and (f["occ"][r0:r0 + h, c0:c0 + w] == g["occ"]).all()
        assert (g["step"], g["cid"], g["nearest"], g["reason"], g["rc"]) == (f["step"], f["cid"], f["nearest"], f["reason"], list(f["rc"]))
        assert g["cands"] == [[c.cid, c.kind, list(c.target), len(c.cells), list(c.objects)] for c in f["cands"]]
        assert g["objects"] == [[o, k, p, list(rc)] for o, k, p, rc in f["objects"]]
    assert info["cell"] == CELL and info["start"] == [N // 2, N // 2]


def test_gif_has_one_frame_per_decision_of_the_longer_run(tmp_path):
    a, b = fake_res(4, long_reason=True), fake_res(3)
    write_run(tmp_path / "l", a, 0.8)
    write_run(tmp_path / "r", b, 0.6)
    write_run(tmp_path / "lb", a, 0.5)
    write_run(tmp_path / "rb", b, 0.3)
    out = tmp_path / "out" / "demo.gif"
    info = viz.make_gif(out, tmp_path / "l", tmp_path / "r", tmp_path / "lb", tmp_path / "rb", budget=80, ms_per_frame=100)
    im = Image.open(out)
    assert im.n_frames == 5 == info["frames"]      # 4 decisions + the final frame of the longer run; the shorter one holds
    assert im.info["loop"] == 0 and out.stat().st_size < 8e6
    assert im.size[0] > im.size[1] * 0.8
    im.seek(0)
    first = im.convert("RGB")
    im.seek(2)
    assert np.any(np.asarray(first) != np.asarray(im.convert("RGB")))   # the frames actually differ
