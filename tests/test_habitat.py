"""M backend checks; run on the AWS instance inside the habitat env (skipped elsewhere).
  AMAP_HM3D=<data root> AMAP_SCENE=<tuning scene> python -m pytest tests/test_habitat.py -q -s
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

habitat_sim = pytest.importorskip("habitat_sim")
if not os.environ.get("AMAP_HM3D") or not os.environ.get("AMAP_SCENE"):
    pytest.skip("needs AMAP_HM3D and AMAP_SCENE", allow_module_level=True)

from amap import policies  # noqa: E402
from amap.contracts import CATEGORIES, FREE, GridSpec  # noqa: E402
from amap.episode import run_episode  # noqa: E402
from amap.habitat_env import HabitatBackend  # noqa: E402

CFG = json.loads(pathlib.Path(os.environ.get("AMAP_TEST_CFG", "configs/dev_m.json")).read_text())  # e.g. a tuning config's noise model
SCENE = os.environ["AMAP_SCENE"]
GRID = GridSpec(CFG["grid"]["n"], CFG["grid"]["cell"])
AVOID = sorted({c for t in CFG["tasks"].values() for c in t["categories"]})
TASK = {"name": "T_wet", **CFG["tasks"]["T_wet"]}


def backend(**kw):
    return HabitatBackend(CFG, SCENE, GRID, 1, avoid_categories=AVOID, **kw)


def test_render_is_deterministic():
    be = backend()
    a, b = be.render(GRID.start, 0), be.render(GRID.start, 0)
    for k in ("rgb", "depth", "semantic"):
        assert np.array_equal(a[k], b[k]), k
    a = {k: v.copy() for k, v in a.items()}
    start = be.start_xyz
    be.close()  # real reload: the shared simulator is released and the scene loaded again
    be2 = backend(start_xyz=start)
    c = be2.render(GRID.start, 0)
    for k in ("rgb", "depth", "semantic"):
        assert np.array_equal(a[k], c[k]), f"{k} after reload"


def test_scan_agrees_with_navmesh():
    be = backend()
    free = set()
    n_det = 0
    for h in range(4):
        o = be.observe(GRID.start, h)
        free |= set(map(tuple, o.free.tolist()))
        n_det += len(o.detections)
    near = [rc for rc in free if np.hypot(rc[0] - GRID.start[0], rc[1] - GRID.start[1]) * GRID.cell <= 3.0]
    share = np.mean([be.traversable(rc) for rc in near])
    print(f"\n{SCENE}: {len(free)} free cells from the start, {share:.2f} of those within 3 m navigable, {n_det} detections, "
          f"{len(be.gt.objects)} GT objects on the start floor, {int(be.nav.sum())} navigable cells")
    assert len(near) > 50 and share > 0.6 and n_det > 0


def test_metamorphic_unobserved_categories_are_invisible():
    be = backend()
    for budget in (150, 80, 40, 20):   # a small floor can be fully seen in 150 steps: shorten until something stays unobserved
        base = run_episode(be, policies.B1(), {**CFG, "budget": budget}, TASK, 1)
        seen = set(base["amap"]._id_of)
        override = {sid: (c + 1) % len(CATEGORIES) for sid, c in be.cat_of.items() if be.key_of[sid] not in seen}
        if override:
            break
    print(f"\nrelabelled {len(override)} unobserved instances, {len(seen)} observed, budget {budget}")
    assert override, "no unobserved instance to relabel: the test would pass without transforming anything"
    be2 = backend(category_override=override, start_xyz=be.start_xyz)
    res = run_episode(be2, policies.B1(), {**CFG, "budget": budget}, TASK, 1)
    assert [d["state"] for d in res["decisions"]] == [d["state"] for d in base["decisions"]]
    assert res["trajectory"] == base["trajectory"]
