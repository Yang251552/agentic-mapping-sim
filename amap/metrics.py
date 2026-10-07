"""Evaluator (reads ground truth; nothing here is visible to the agent).

R_T = start-floor cells within 1.0 m of a ground-truth object whose category is in C_T; coverage = share of R_T the
agent has observed; S_T = share of C_T objects whose posterior argmax is the true category with confidence >= tau;
Q_T = (coverage + S_T) / 2. Unobserved counts as 0.
"""
from __future__ import annotations

import numpy as np

from .contracts import UNKNOWN


def task_region(gt, grid, cat_ids, radius_m=1.0) -> np.ndarray:
    n = grid.n
    rr, cc = np.mgrid[0:n, 0:n]
    mask = np.zeros((n, n), bool)
    for o in gt.objects:
        if o.category in cat_ids:
            mask |= (rr - o.rc[0]) ** 2 + (cc - o.rc[1]) ** 2 <= (radius_m / grid.cell) ** 2
    return mask & gt.floor


def score(amap, gt, key_to_oid, cat_ids, tau, region) -> dict:
    known = amap.occ != UNKNOWN
    cov = float(known[region].mean()) if region.any() else 0.0
    objs = [o for o in gt.objects if o.category in cat_ids]
    ok = 0
    for o in objs:
        oid = key_to_oid.get(o.key)
        if oid is not None:
            cat, conf = amap.label(oid)
            ok += int(cat == o.category and conf >= tau)
    s_t = ok / len(objs) if objs else 0.0
    floor_cov = float(known[gt.floor].mean()) if gt.floor.any() else 0.0
    return {"q_t": round((cov + s_t) / 2, 6), "coverage": round(cov, 6), "s_t": round(s_t, 6),
            "n_task_objects": len(objs), "n_confident_correct": ok, "floor_coverage": round(floor_cov, 6)}
