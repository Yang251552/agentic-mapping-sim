"""Stage 3/4/6 shared core: AgentMap fusion, frontier/revisit candidates, BFS planner. Runs in seconds."""
from __future__ import annotations

import time

import numpy as np
import pytest

from amap.agent_map import AgentMap
from amap.candidates import generate, still_valid
from amap.contracts import CATEGORIES, FREE, OCC, UNKNOWN, Detection, GridSpec, Observation, stable_key
from amap.planner import bfs, path
from amap.sensor import DEFAULT_P_BINS, noisy_label

NC = len(CATEGORIES)


def obs(free=(), occ=(), dets=()):
    return Observation(np.array(free, dtype=np.int64).reshape(-1, 2), np.array(occ, dtype=np.int64).reshape(-1, 2),
                       list(dets))


def new_map(n=40, cell=0.25, tau=0.7):
    return AgentMap(GridSpec(n, cell), NC, DEFAULT_P_BINS, tau)


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


# ---- stage 3: occupancy and fusion ---------------------------------------------------------------------------
def test_occupancy_majority_vote_bumps_permanent_outside_ignored():
    m = new_map(n=10)
    m.integrate(obs(free=[(1, 1), (-1, 3), (10, 0)], occ=[(3, 3), (5, 99)]))
    assert m.occ[1, 1] == FREE and m.occ[3, 3] == OCC
    m.integrate(obs(free=[(3, 3)]))
    assert m.occ[3, 3] == FREE                     # 1 vs 1: a doorway glimpsed once as occupied reopens
    m.integrate(obs(occ=[(3, 3)]))
    m.integrate(obs(occ=[(3, 3)]))
    assert m.occ[3, 3] == OCC                      # majority occupied
    m.mark_occupied((1, 1))
    m.mark_occupied((-5, 0))
    for _ in range(5):
        m.integrate(obs(free=[(1, 1)]))
    assert m.occ[1, 1] == OCC                      # an executor bump is permanent
    assert (m.occ == UNKNOWN).sum() == 100 - 2


def test_agent_ids_hide_sensor_key():
    m = new_map()
    k1, k2 = stable_key("house/obj_a"), stable_key("house/obj_b")
    m.integrate(obs(dets=[Detection(k2, (5, 5), 2, 3), Detection(k1, (6, 6), 2, 4)]))
    m.integrate(obs(dets=[Detection(k2, (5, 6), 1, 3)]))
    summ = m.objects_summary()
    assert [s[0] for s in summ] == [0, 1] and summ[0][3] == (5, 6) and summ[1][3] == (6, 6)
    assert not any(k in s for s in summ for k in (k1, k2))


def test_posterior_after_one_observation_equals_p_bin():
    for b, p in enumerate(DEFAULT_P_BINS):
        m = new_map()
        m.integrate(obs(dets=[Detection(stable_key(f"x{b}"), (3, 3), b, 7)]))
        post = m.posterior(0)
        assert abs(post[7] - p) < 1e-12 and abs(post.sum() - 1) < 1e-12
        assert np.allclose(np.delete(post, 7), (1 - p) / (NC - 1), rtol=0, atol=1e-12)
        assert m.label(0) == (7, post[7])


def test_close_observation_reduces_class_uncertainty():
    far_p, close_p, far_h, close_h = [], [], [], []
    for seed in range(25):
        for i in range(20):                        # 500 (seed, key) pairs
            key = stable_key(f"scene{seed}/obj{i}")
            true = int(np.random.default_rng([seed, i]).integers(NC))
            m = new_map()
            m.integrate(obs(dets=[Detection(key, (4, 4), 2, noisy_label(seed, key, 2, true, NC))]))
            far_p.append(m.posterior(0)[true])
            far_h.append(entropy(m.posterior(0)))
            m.integrate(obs(dets=[Detection(key, (4, 4), 0, noisy_label(seed, key, 0, true, NC))]))
            close_p.append(m.posterior(0)[true])
            close_h.append(entropy(m.posterior(0)))
    print(f"\nmean P(true): bin2 {np.mean(far_p):.3f} -> bin2+bin0 {np.mean(close_p):.3f}; "
          f"mean entropy {np.mean(far_h):.3f} -> {np.mean(close_h):.3f} nats (n={len(far_p)})")
    assert np.mean(close_p) > np.mean(far_p)
    assert np.mean(close_h) < np.mean(far_h)


def test_repeated_same_bin_observation_adds_no_evidence():
    m = new_map()
    key = stable_key("house/obj_7")
    d = Detection(key, (8, 8), 1, 5)
    m.integrate(obs(dets=[d]))
    before = m.posterior(0).tobytes()
    m.integrate(obs(dets=[d]))
    assert m.posterior(0).tobytes() == before
    m.integrate(obs(dets=[Detection(key, (9, 7), 1, 5)]))   # same bin, other position
    assert m.posterior(0).tobytes() == before
    assert m.objects_summary()[0][3] == (9, 7)              # position estimate still updates


# ---- stage 6: planner ----------------------------------------------------------------------------------------
MAZE = ["S.#..",
        "#.#.#",
        "....G",
        ".##.."]


def parse(rows):
    a = np.array([list(r) for r in rows])
    return a != "#", tuple(int(v) for v in np.argwhere(a == "S")[0]), tuple(int(v) for v in np.argwhere(a == "G")[0])


def test_bfs_and_path_on_maze():
    P, s, g = parse(MAZE)
    d = bfs(P, s)
    expect = np.array([[0, 1, -1, 4, 4],        # 8-connected, a diagonal needs one passable orthogonal neighbour
                       [-1, 1, -1, 3, -1],
                       [2, 2, 2, 3, 4],
                       [3, -1, -1, 3, 4]])
    assert d.dtype == np.int32 and (d == expect).all()
    p = path(P, s, g)
    assert len(p) == d[g] and p[-1] == g and s not in p
    for a, b in zip([s] + p[:-1], p):
        dr, dc = b[0] - a[0], b[1] - a[1]
        assert max(abs(dr), abs(dc)) == 1 and P[b]
        assert dr == 0 or dc == 0 or P[a[0] + dr, a[1]] or P[a[0], a[1] + dc]   # no squeezing between blocked cells
    assert p == path(P, s, g) and p == path(P, s, g, dist=d)
    assert path(P, s, s) == [] and path(P, s, (0, 2)) is None


# ---- stage 4: candidates -------------------------------------------------------------------------------------
def room_map():
    """Walled room rows 5..15 x cols 5..20 with a 6-cell top doorway (row 4, cols 8..13), a 2-cell right gap
    (col 21, rows 10..11), a free patch behind the bottom wall (OCC-separated) and one behind UNKNOWN (col 22)."""
    m = new_map(n=30)
    m.occ[4:17, 4:22] = OCC
    m.occ[5:16, 5:21] = FREE
    m.occ[4, 8:14] = FREE
    m.occ[10:12, 21] = FREE
    m.occ[17:19, 6:15] = FREE          # touches the OCC wall row 16: unreachable
    m.occ[6:9, 23:26] = FREE           # beyond UNKNOWN col 22: unreachable
    return m


def test_frontier_clusters_on_hand_built_map():
    m = room_map()
    c = generate(m, (10, 12), min_frontier_cells=5)
    assert [x.cid for x in c] == ["f1"]
    f = c[0]
    assert f.kind == "frontier" and f.cells == tuple((4, j) for j in range(8, 14)) and f.size == 6
    assert f.target == (4, 10) and f.steps == 6          # centroid (4, 10.5): row-major tie -> (4, 10); 8-connected
    c2 = generate(m, (10, 12))                            # default 2 cells: the narrow right gap counts too
    assert [(x.cid, x.cells, x.steps) for x in c2] == [("f1", f.cells, 6), ("f2", ((10, 21), (11, 21)), 9)]
    assert still_valid(f, m)
    m.occ[0:4, 7:15] = FREE                               # explore beyond the doorway: old cluster no longer frontier
    assert not still_valid(f, m)


def revisit_map(hole: bool):
    m = new_map(n=40, tau=0.99)
    m.occ[10:31, 10:31] = FREE
    m.occ[10:31, 17] = UNKNOWN                            # unobserved column between agent side and object
    if hole:
        m.occ[20, 17] = FREE
    m.integrate(obs(occ=[(20, 20)], dets=[Detection(stable_key("o"), (20, 20), 2, 3)]))
    return m


def test_revisit_viewpoint_needs_reachability_and_line_of_sight():
    m = revisit_map(hole=False)
    assert m.eligible_revisit(0)
    # left side is reachable but every segment crosses the UNKNOWN column; right side has LOS but is unreachable
    assert [x for x in generate(m, (20, 12)) if x.kind == "revisit"] == []
    m = revisit_map(hole=True)
    r = [x for x in generate(m, (20, 12)) if x.kind == "revisit"]
    assert len(r) == 1 and r[0].objects == (0,) and r[0].size == 1
    assert r[0].target == (20, 15) and r[0].steps == 3   # (20, 14) is exactly 1.5 m away: outside bin 0
    assert still_valid(r[0], m)
    m.integrate(obs(dets=[Detection(stable_key("o"), (20, 20), 0, 3)]))
    assert not m.eligible_revisit(0) and not still_valid(r[0], m)
    assert [x for x in generate(m, (20, 12)) if x.kind == "revisit"] == []


def test_bin0_seen_object_never_revisited():
    m = new_map(n=40, tau=0.99)
    m.occ[10:31, 10:31] = FREE
    m.integrate(obs(dets=[Detection(stable_key("a"), (20, 20), 0, 1), Detection(stable_key("b"), (25, 25), 2, 1)]))
    assert m.label(0)[1] < m.tau and not m.eligible_revisit(0) and m.eligible_revisit(1)
    r = [x for x in generate(m, (12, 12)) if x.kind == "revisit"]
    assert [x.objects for x in r] == [(1,)]


def house(seed=0, n=400, cell=0.25, n_obj=80):
    """~60 x 60 m explored region (240 x 240 cells) with walls, doors, unknown pockets and ~n_obj objects."""
    rng = np.random.default_rng(seed)
    m = AgentMap(GridSpec(n, cell), NC, DEFAULT_P_BINS, 0.7)
    lo, hi = n // 2 - 120, n // 2 + 120
    m.occ[lo:hi, lo:hi] = FREE
    for w in range(lo + 15, hi, 30):                      # interior walls every 7.5 m with doorways; start mid-room
        m.occ[w, lo:hi] = OCC
        m.occ[lo:hi, w] = OCC
        for d in range(lo + 5, hi, 30):
            m.occ[w, d:d + 4] = FREE
            m.occ[d:d + 4, w] = FREE
    for _ in range(25):                                   # unobserved pockets -> interior frontiers
        r, c = rng.integers(lo + 5, hi - 15, 2)
        m.occ[r:r + rng.integers(3, 12), c:c + rng.integers(3, 12)] = UNKNOWN
    cand = np.argwhere(m.occ == FREE)
    cand = cand[(cand != m.grid.start).any(1)]
    pick = cand[rng.choice(len(cand), n_obj, replace=False)]
    dets = [Detection(stable_key(f"obj{i}"), (int(r), int(c)), int(rng.integers(1, 3)), int(rng.integers(NC)))
            for i, (r, c) in enumerate(pick)]
    m.integrate(obs(occ=pick, dets=dets))
    return m


def test_generate_deterministic_and_fast():
    m = house()
    start = m.grid.start
    a = generate(m, start)
    assert a == generate(m, start) and len(a) == 8
    assert {x.kind for x in generate(m, start, k=200)} == {"frontier", "revisit"}
    assert [x.steps for x in a] == sorted(x.steps for x in a)
    d = bfs(m.occ == FREE, start)
    for x in generate(m, start, k=200):
        assert d[x.target] == x.steps and m.occ[x.target] == FREE
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        generate(m, start)
        ts.append(time.perf_counter() - t)
    t = time.perf_counter()
    bfs(m.occ == FREE, start)
    tb = time.perf_counter() - t
    print(f"\ngenerate() on 400x400, {int((m.occ == FREE).sum())} free cells, {m.n_objects} objects: "
          f"median {np.median(ts) * 1e3:.1f} ms (min {min(ts) * 1e3:.1f}); bfs {tb * 1e3:.1f} ms")
    assert np.median(ts) < 0.5   # target < 150 ms; loose bound so a slow CI box does not flake


def test_revisit_line_of_sight_property():
    """Independent check: densely sample every chosen viewpoint's segment; no UNKNOWN/OCC outside the clearance."""
    checked = 0
    for seed in range(5):
        m = house(seed)
        cell = m.grid.cell
        pos = {oid: rc for oid, _, _, rc in m.objects_summary()}
        d = bfs(m.occ == FREE, m.grid.start)
        for x in generate(m, m.grid.start, k=500):
            if x.kind != "revisit":
                continue
            checked += 1
            ctr = np.mean([pos[o] for o in x.objects], axis=0)
            assert d[x.target] >= 0 and np.hypot(*(np.array(x.target) - ctr)) * cell < 1.5
            for t in np.linspace(0, 1, 400):
                p = np.array(x.target) + t * (ctr - np.array(x.target))
                rc = tuple(np.floor(p + 0.5).astype(int))
                if np.hypot(*(np.array(rc) - ctr)) * cell > 0.5:
                    assert m.occ[rc] == FREE, (seed, x, rc)
    assert checked >= 100


def test_diagonal_never_squeezes_between_two_blocked_cells():
    P = np.array([[1, 0], [0, 1]], dtype=bool)
    assert bfs(P, (0, 0))[1, 1] == -1 and path(P, (0, 0), (1, 1)) is None
    P[0, 1] = True
    assert bfs(P, (0, 0))[1, 1] == 1 and path(P, (0, 0), (1, 1)) == [(1, 1)]


def test_revisit_retires_only_members_near_the_viewpoint():
    from amap.episode import revisit_retired
    m = new_map(n=40, cell=0.25)
    m.integrate(obs(dets=[Detection(1, (10, 10), 2, 0), Detection(2, (10, 14), 2, 0), Detection(3, (10, 30), 2, 0)]))
    r = 1.5 / 0.25  # 6 cells
    assert revisit_retired(m, (0, 1, 2), (10, 10), r) == {0, 1}       # 0 and 4 cells away; 20 cells is not
    assert revisit_retired(m, (2,), (10, 10), r) == {2}               # nothing near: the nearest is still retired
    assert revisit_retired(m, (1, 2), (10, 4), r) == {1}              # 10 vs 26 cells: nearest only


def test_frequency_preserving_noise_model():
    """Kernel: zero diagonal, rows sum to 1, pi K = pi; generator draws follow K; one look leaves the reported class at
    exactly p_bin for every class, so only a near look clears tau; fusion matches hand-computed Bayes."""
    from amap.sensor import frequency_preserving_kernel, single_look_posteriors
    rng = np.random.default_rng(7)
    pi = rng.dirichlet(np.ones(NC) * 0.6)
    pi = np.minimum(pi, 0.4); pi /= pi.sum()
    K = frequency_preserving_kernel(pi)
    assert np.allclose(np.diag(K), 0) and np.allclose(K.sum(1), 1) and np.allclose(pi @ K, pi, atol=1e-10)
    post = single_look_posteriors(pi, K, (0.9, 0.65, 0.55))
    assert np.allclose(post, np.array([0.9, 0.65, 0.55])[:, None])
    # generator: wrong labels of one true class follow its kernel row
    cdf, c = np.cumsum(K, axis=1), int(np.argmin(pi))
    labs = np.array([noisy_label(3, k, 1, c, NC, (0.9, 0.65, 0.55), cdf) for k in range(20000)])
    wrong = labs[labs != c]
    assert abs(len(wrong) / len(labs) - 0.35) < 0.02
    freq = np.bincount(wrong, minlength=NC) / len(wrong)
    assert np.abs(freq - K[c]).max() < 0.02 and freq[c] == 0
    assert noisy_label(3, 11, 1, c, NC, (0.9, 0.65, 0.55), cdf) == noisy_label(3, 11, 1, c, NC, (0.9, 0.65, 0.55), cdf)
    # agent: one mid look, then one near look, against hand-computed Bayes
    m = AgentMap(GridSpec(40, 0.25), NC, (0.9, 0.65, 0.55), 0.7, pi, K)
    y = int(np.argmax(pi))
    m.integrate(obs(dets=[Detection(5, (10, 10), 1, y)]))
    cat, conf = m.label(0)
    assert cat == y and abs(conf - 0.65) < 1e-9 and m.eligible_revisit(0)
    m.integrate(obs(dets=[Detection(5, (10, 10), 0, c)]))
    L1, L0 = 0.65 * np.eye(NC) + 0.35 * K, 0.9 * np.eye(NC) + 0.1 * K
    want = pi * L1[:, y] * L0[:, c]
    assert np.allclose(m.posterior(0), want / want.sum())


def test_curve_reads_last_record_at_or_before_each_grid_step():
    from eval.calibrate import regular
    pts = [{"step": 10, "q": 1}, {"step": 30, "q": 3}, {"step": 47, "q": 4}]   # 20 skipped by a bump; run ended at 47
    r = regular(pts, top=70)
    assert [r[s]["q"] for s in (10, 20, 30, 40, 50, 70)] == [1, 1, 3, 3, 4, 4]
    assert regular([{"step": 20, "q": 2}], top=30).keys() == {20, 30}   # nothing recorded yet at 10


def test_b2_evidence_gated_switches_only_for_task_evidence_worth_the_detour():
    from amap.policies import B2, render_user
    ds = {"task": "t", "steps_left": 50, "pose": {"x": "+0.0", "y": "+0.0", "heading": "+x"}, "n_objects_seen": 2, "recent": [],
          "objects": [{"id": "o0", "label": "sink", "conf": "0.65", "x": "+1.0", "y": "+0.0"},
                      {"id": "o1", "label": "chair", "conf": "0.90", "x": "+0.0", "y": "+1.0"}],
          "candidates": [{"id": "f1", "kind": "frontier", "steps": 3, "extra": 0, "x": "+0.6", "y": "+0.0", "opening_m": "0.4", "near": [{"id": "o1", "dist_m": "1.0"}]},
                         {"id": "f2", "kind": "frontier", "steps": 5, "extra": 2, "x": "+1.0", "y": "+0.0", "opening_m": "3.0", "near": []},
                         {"id": "r1", "kind": "revisit", "steps": 9, "extra": 6, "x": "+1.8", "y": "+0.0", "objects": ["o0"]}]}
    assert B2(["sink"], lam=0.05).decide(ds, None, {}).cid == "r1"      # 0.65 - 0.05 * (6 + 4) = 0.15 > 0
    assert B2(["sink"], lam=0.10).decide(ds, None, {}).cid == "f1"      # 0.65 - 0.10 * 10 < 0: stay with B1
    assert B2(["toilet"], lam=0.01).decide(ds, None, {}).cid == "f1"    # no evidence: never the wider f2
    text = render_user(ds)
    assert "(+0 vs nearest frontier)" in text and "(+6 vs nearest frontier)" in text


def test_heldout_selection_rule():
    import hashlib
    from eval.select_heldout import select
    rows = []
    for i in range(6):
        s = f"0080{i}-x"
        for sd in (1001, 1002):
            rows.append({"scene": s, "seed": sd, "passes": not (i == 1 and sd == 1002), "nav_area_m2": 300.0 if i == 2 else 60.0,
                         "start_xyz": [0.0, 0.0, 0.0]})
    rows.append({"scene": "00806-y", "seed": 1001, "passes": True, "nav_area_m2": 60.0, "start_xyz": [0.0, 0.0, 0.0]})   # one seed missing: not eligible
    got = select(rows, [1001, 1002], [39.6, 242.5], "salt:", 2)
    eligible = sorted(["00800-x", "00803-x", "00804-x", "00805-x"], key=lambda s: hashlib.sha256(f"salt:{s}".encode()).hexdigest())
    assert got == eligible[:2]
    with pytest.raises(SystemExit):
        select(rows, [1001, 1002], [39.6, 242.5], "salt:", 5)


def test_object_viewpoints_list_only_objects_in_near_range_and_keep_the_nearest_frontier():
    m = revisit_map(hole=True)                            # one object: same viewpoint as the cluster rule
    r = [x for x in generate(m, (20, 12), revisit_mode="object", k_revisit=3) if x.kind == "revisit"]
    assert len(r) == 1 and r[0].objects == (0,) and r[0].target == (20, 15) and r[0].steps == 3
    m = house()
    start, cell = m.grid.start, m.grid.cell
    a = generate(m, start, revisit_mode="object", k_revisit=3)
    assert a == generate(m, start, revisit_mode="object", k_revisit=3) and len(a) == 8
    rev = [x for x in a if x.kind == "revisit"]
    assert 0 < len(rev) <= 3 and [x.steps for x in a] == sorted(x.steps for x in a)
    rc = {oid: orc for oid, _, _, orc in m.objects_summary()}
    for x in generate(m, start, k=200, revisit_mode="object"):
        if x.kind == "revisit":                          # every listed object is settled by the look-around there
            assert x.objects and all(((rc[o][0] - x.target[0]) ** 2 + (rc[o][1] - x.target[1]) ** 2) * cell ** 2 < 1.5 ** 2 for o in x.objects)
    nearest = min((x for x in generate(m, start, k=200) if x.kind == "frontier"), key=lambda x: (x.steps, x.target))
    assert any(x.kind == "frontier" and x.target == nearest.target for x in a)   # B1's choice is never crowded out


def test_v6_resolves_lists_only_objects_in_near_range_with_line_of_sight():
    from amap.policies import decision_state
    m = new_map(n=40)
    m.occ[:] = FREE
    m.occ[:35, 20] = OCC                                  # a wall between the two objects (review of the v6 rendering)
    m.integrate(obs(dets=[Detection(1, (20, 17), 2, 10), Detection(2, (20, 23), 2, 10)]))
    cs = generate(m, (20, 18), k=50)
    ds = decision_state("t", 50, m.grid, (20, 18), 0, m, cs, [], task_categories=["sink"])
    rev = [d for d in ds["candidates"] if d["kind"] == "revisit"]
    assert rev and all("o1" not in d["resolves"] for d in rev if (20, 18) == next(c.target for c in cs if c.cid == d["id"]))


def test_v7_lists_counting_objects_crowded_out_of_a_frontiers_near_list():
    from amap.policies import render_user_v6, render_user_v7
    o = lambda i, lab, x: {"id": f"o{i}", "label": lab, "conf": "0.55", "x": f"{x:+.1f}", "y": "+0.0"}
    ds = {"task": "t", "task_categories": ["sink"], "steps_left": 50, "pose": {"x": "+0.0", "y": "+0.0", "heading": "+x"},
          "objects": [o(1, "chair", 1.0), o(2, "chair", 1.1), o(3, "chair", 1.2), o(4, "sink", 2.5), o(5, "sink", 6.0)],
          "n_objects_seen": 5, "recent": [],
          "candidates": [{"id": "f1", "kind": "frontier", "steps": 5, "extra": 0, "x": "+1.0", "y": "+0.0", "opening_m": "1.0",
                          "near": [{"id": "o1", "dist_m": "0.0"}, {"id": "o2", "dist_m": "0.1"}, {"id": "o3", "dist_m": "0.2"}]}]}
    v6, v7 = render_user_v6(ds), render_user_v7(ds)
    assert "o4 sink" not in v6.split("Candidates")[1] and "other objects that count within 2 m: o4 sink 0.55" in v7
    assert "o5" not in v7.split("Candidates")[1]                       # beyond 2 m of the frontier: not listed
    assert v7.replace("; other objects that count within 2 m: o4 sink 0.55", "") == v6
