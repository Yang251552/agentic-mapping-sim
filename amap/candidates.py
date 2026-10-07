"""Stage 4: candidate targets shared by the LLM agent and every baseline, computed on the agent map only.

Truth firewall: reads AgentMap.occ, AgentMap.grid and the AgentMap object API; UNKNOWN cells block reachability
and line of sight. Deterministic: row-major seeds and ties, integer centroid distances, an explicit total sort
order, Python ints in every output field.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from amap.agent_map import AgentMap
from amap.contracts import FREE, UNKNOWN
from amap.planner import bfs


@dataclass(frozen=True)
class Candidate:
    cid: str
    kind: str                                   # "frontier" | "revisit"
    target: tuple[int, int]
    steps: int
    size: int                                   # frontier cluster cells, or number of objects
    objects: tuple[int, ...] = ()               # agent object ids (revisit only)
    cells: tuple[tuple[int, int], ...] = ()     # frontier cluster cells, row-major (frontier only)


def frontier_mask(occ: np.ndarray) -> np.ndarray:
    """FREE cells with at least one UNKNOWN 4-neighbour (outside the grid counts as not UNKNOWN)."""
    unk = occ == UNKNOWN
    nb = np.zeros_like(unk)
    nb[1:] |= unk[:-1]
    nb[:-1] |= unk[1:]
    nb[:, 1:] |= unk[:, :-1]
    nb[:, :-1] |= unk[:, 1:]
    return (occ == FREE) & nb


def _clusters8(mask: np.ndarray) -> list[np.ndarray]:
    """8-connected components of mask as (m, 2) row-major sorted cell arrays, ordered by first cell."""
    n0, n1 = mask.shape
    W = n1 + 2
    pad = np.zeros((n0 + 2, W), dtype=np.uint8)
    pad[1:-1, 1:-1] = mask
    todo = bytearray(pad.tobytes())
    offs = (1, W + 1, W, W - 1, -1, -W - 1, -W, -W + 1)
    out = []
    for s in np.flatnonzero(pad).tolist():
        if not todo[s]:
            continue
        todo[s] = 0
        stack, members = [s], []
        while stack:
            i = stack.pop()
            members.append(i)
            for o in offs:
                j = i + o
                if todo[j]:
                    todo[j] = 0
                    stack.append(j)
        idx = np.sort(np.asarray(members, dtype=np.int64))
        out.append(np.stack([idx // W - 1, idx % W - 1], axis=1))
    return out


def _segments_clear(p0: np.ndarray, end: np.ndarray, blk: np.ndarray) -> np.ndarray:
    """(m,) bool: segment p0[i] -> end (cell units) touches no closed unit square centred on a blk cell."""
    if len(blk) == 0:
        return np.ones(len(p0), dtype=bool)
    a = p0[:, None, :]                       # (m, 1, 2)
    d = (end - p0)[:, None, :]               # (m, 1, 2)
    lo, hi = blk[None] - 0.5, blk[None] + 0.5  # (1, k, 2)
    flat = d == 0
    inside = (a >= lo) & (a <= hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - a) / d, (hi - a) / d
    tmin = np.where(flat, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2))
    tmax = np.where(flat, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2))
    hit = np.maximum(tmin.max(-1), 0.0) <= np.minimum(tmax.min(-1), 1.0)   # (m, k) slab test, t in [0, 1]
    return ~hit.any(1)


def _link_components(rc: np.ndarray, link_cells2: float) -> list[list[int]]:
    """Single-linkage groups (indices into rc), each sorted, ordered by smallest member."""
    d2 = ((rc[:, None, :] - rc[None, :, :]) ** 2).sum(-1)
    adj = d2 <= link_cells2
    seen = [False] * len(rc)
    groups = []
    for s in range(len(rc)):
        if seen[s]:
            continue
        seen[s] = True
        stack, g = [s], []
        while stack:
            i = stack.pop()
            g.append(i)
            for j in np.flatnonzero(adj[i]).tolist():
                if not seen[j]:
                    seen[j] = True
                    stack.append(j)
        groups.append(sorted(g))
    return groups


def _window(c, R, shape):
    n0, n1 = shape
    r0, r1 = max(int(np.floor(c[0] - R)) - 1, 0), min(int(np.ceil(c[0] + R)) + 1, n0 - 1)
    c0, c1 = max(int(np.floor(c[1] - R)) - 1, 0), min(int(np.ceil(c[1] + R)) + 1, n1 - 1)
    rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    return rr.ravel(), cc.ravel()


def seen_from(free: np.ndarray, vps: np.ndarray, obj, radius_cells: float, clearance_cells: float) -> np.ndarray:
    """(m,) bool: a straight look from each viewpoint (m, 2 cells) to the object crosses no non-FREE cell of the agent
    map, except those within clearance_cells of the object (its own footprint)."""
    rr, cc = _window(obj, radius_cells, free.shape)
    blk = ~free[rr, cc] & (((rr - obj[0]) ** 2 + (cc - obj[1]) ** 2) > clearance_cells ** 2)
    return _segments_clear(np.asarray(vps, np.float64).reshape(-1, 2), np.asarray(obj, np.float64),
                           np.stack([rr[blk], cc[blk]], 1).astype(np.float64))


def _object_viewpoints(amap, elig, dist, free, revisit_radius, los_clearance) -> list:
    """revisit_mode "object" (redesign 10-07, shared by every policy): one viewpoint per eligible object, the cheapest
    reachable FREE cell strictly within revisit_radius of the object (distance bin 0) with line of sight to it; every
    eligible object within that radius of a chosen viewpoint, with line of sight from it, is listed (the look-around there
    settles it). Objects with the same viewpoint form one candidate."""
    cell = amap.grid.cell
    R, R2, Cc = revisit_radius / cell, (revisit_radius / cell) ** 2, los_clearance / cell
    window = lambda c: _window(c, R, free.shape)
    seen = lambda vps, obj: seen_from(free, vps, obj, R, Cc)

    best = {}
    for oid, rc in elig:
        rr, cc = window(rc)
        vp = (((rr - rc[0]) ** 2 + (cc - rc[1]) ** 2) < R2) & free[rr, cc] & (dist[rr, cc] >= 0)
        if not vp.any():
            continue
        vr, vc, vd = rr[vp], cc[vp], dist[rr[vp], cc[vp]]
        order = np.lexsort((vc, vr, vd))   # fewest steps, then row-major
        vr, vc, vd = vr[order], vc[order], vd[order]
        ok = seen(np.stack([vr, vc], 1), rc)
        if ok.any():
            i = int(np.argmax(ok))
            best[oid] = (int(vd[i]), (int(vr[i]), int(vc[i])))
    rows = []
    for steps, target in sorted(set(best.values())):
        objs = tuple(oid for oid, rc in elig
                     if (rc[0] - target[0]) ** 2 + (rc[1] - target[1]) ** 2 < R2 and bool(seen(np.array([target]), rc)[0]))
        rows.append((steps, 1, target, objs, (), len(objs)))
    return rows


def generate(amap: AgentMap, agent_rc, *, k: int = 8, min_frontier_cells: int = 2, revisit_radius: float = 1.5,
             cluster_link: float = 1.5, los_clearance: float = 0.5, revisit_mode: str = "cluster",
             k_revisit: int = None) -> list[Candidate]:
    if revisit_mode not in ("cluster", "object") or (k_revisit is not None and (isinstance(k_revisit, bool) or not isinstance(k_revisit, int) or k_revisit < 0)):
        raise ValueError(f"bad candidate settings: revisit_mode={revisit_mode!r}, k_revisit={k_revisit!r}")
    if not (np.isfinite(revisit_radius) and revisit_radius > 0 and np.isfinite(los_clearance) and los_clearance >= 0):
        raise ValueError(f"bad candidate settings: revisit_radius={revisit_radius!r}, los_clearance={los_clearance!r}")
    occ = amap.occ
    n0, n1 = occ.shape
    cell = amap.grid.cell
    free = occ == FREE
    dist = bfs(free, agent_rc)
    rows = []   # (steps, kind_rank, target, objects, cells, size)

    for cl in _clusters8(frontier_mask(occ) & ~amap.dead):
        if len(cl) < min_frontier_cells:
            continue
        dd = dist[cl[:, 0], cl[:, 1]]
        ok = dd >= 0
        if not ok.any():
            continue
        m = len(cl)
        sr, sc = int(cl[:, 0].sum()), int(cl[:, 1].sum())
        e2 = (m * cl[:, 0] - sr) ** 2 + (m * cl[:, 1] - sc) ** 2   # exact integer (m * distance to centroid)^2
        e2 = np.where(ok, e2, np.iinfo(np.int64).max)
        i = int(np.argmin(e2))                                     # first minimum = row-major tie-break
        cells = tuple((int(r), int(c)) for r, c in cl)
        rows.append((int(dd[i]), 0, cells[i], (), cells, m))

    elig = [(oid, rc) for oid, _, _, rc in amap.objects_summary() if amap.eligible_revisit(oid)]
    if elig and revisit_mode == "object":
        rows += _object_viewpoints(amap, elig, dist, free, revisit_radius, los_clearance)
    elif elig:
        orc = np.array([rc for _, rc in elig], dtype=np.float64)
        R2 = (revisit_radius / cell) ** 2
        C2 = (los_clearance / cell) ** 2
        for g in _link_components(orc, (cluster_link / cell) ** 2):
            ctr = orc[g].mean(0)
            r0 = max(int(np.floor(ctr[0] - revisit_radius / cell)) - 1, 0)
            r1 = min(int(np.ceil(ctr[0] + revisit_radius / cell)) + 1, n0 - 1)
            c0 = max(int(np.floor(ctr[1] - revisit_radius / cell)) - 1, 0)
            c1 = min(int(np.ceil(ctr[1] + revisit_radius / cell)) + 1, n1 - 1)
            rr, cc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
            rr, cc = rr.ravel(), cc.ravel()
            e2 = (rr - ctr[0]) ** 2 + (cc - ctr[1]) ** 2
            dd = dist[rr, cc]
            vp = (e2 < R2) & free[rr, cc] & (dd >= 0)   # strict: matches distance bin 0 = [0, 1.5) m
            if not vp.any():
                continue
            blk = ~free[rr, cc] & (e2 > C2)                # UNKNOWN or OCC outside the clearance disc
            vr, vc, vd = rr[vp], cc[vp], dd[vp]
            order = np.lexsort((vc, vr, vd))               # fewest steps, then row-major
            vr, vc, vd = vr[order], vc[order], vd[order]
            clear = _segments_clear(np.stack([vr, vc], 1).astype(np.float64), ctr,
                                    np.stack([rr[blk], cc[blk]], 1).astype(np.float64))
            if not clear.any():
                continue
            i = int(np.argmax(clear))
            objs = tuple(int(elig[j][0]) for j in g)
            rows.append((int(vd[i]), 1, (int(vr[i]), int(vc[i])), objs, (), len(objs)))

    rows.sort(key=lambda t: t[:5])
    if k_revisit is None:
        kept = rows[:k]
    else:   # the cap keeps the nearest frontiers (B1's choice is always listed) and at most k_revisit revisits
        fr = [r for r in rows if r[1] == 0]
        rev = [r for r in rows if r[1] == 1][:min(k_revisit, k - 1 if fr else k)]   # a frontier keeps at least one slot
        kept = sorted(fr[:k - len(rev)] + rev, key=lambda t: t[:5])
    out, nf, nr = [], 0, 0
    for steps, rank, target, objs, cells, size in kept:
        if rank == 0:
            nf += 1
            out.append(Candidate(f"f{nf}", "frontier", target, steps, size, (), cells))
        else:
            nr += 1
            out.append(Candidate(f"r{nr}", "revisit", target, steps, size, objs, ()))
    return out


def still_valid(c: Candidate, amap: AgentMap) -> bool:
    if c.kind == "frontier":
        cl = np.asarray(c.cells, dtype=np.int64).reshape(-1, 2)
        return bool((frontier_mask(amap.occ) & ~amap.dead)[cl[:, 0], cl[:, 1]].any())
    return any(amap.eligible_revisit(o) for o in c.objects)
