"""Stages 2-7 loop for one episode: sense -> fuse -> candidates -> decide -> move along the agent-map path.

Step accounting is identical for every policy: one move or one bump = 1 step (with one observation after a move,
facing the move direction); arriving at a target = a 4-step look-around; the episode starts with a look-around.
A new decision is made on arrival, when the target stops being valid or unreachable, and every `decision_every`
steps. Evaluator values (Q_T curve, label correctness) are computed here from the backend's ground truth but are
written only to the logs, never into the DecisionState.
"""
from __future__ import annotations

from .agent_map import AgentMap
from .candidates import frontier_mask, generate, still_valid
from .contracts import CATEGORIES, FREE, STEPS
from .metrics import score, task_region
from .planner import path as plan_path
from .policies import canonical_state, decision_state
from .sensor import noise_model


def _heading(a, b) -> int:
    dr, dc = b[0] - a[0], b[1] - a[1]
    if dr and dc:  # diagonal move: face along its x component
        dr = 0
    return STEPS.index((dr, dc))


def revisit_retired(amap, objects, at_rc, radius_cells) -> set:
    """Cluster members within radius_cells of the viewpoint (strict, like the viewpoint rule); none -> the nearest one."""
    rc = {oid: orc for oid, _, _, orc in amap.objects_summary()}
    d2 = {o: (rc[o][0] - at_rc[0]) ** 2 + (rc[o][1] - at_rc[1]) ** 2 for o in objects}
    near = {o for o, d in d2.items() if d < radius_cells ** 2}
    return near or {min(d2, key=lambda o: (d2[o], o))}


def run_episode(backend, policy, cfg: dict, task: dict, seed: int, record=False) -> dict:
    grid, gt = backend.grid, backend.gt
    tau, budget = cfg["tau"], cfg["budget"]
    prior, kernel = noise_model(cfg["sensor"])  # the same train-split prior and wrong-label kernel the sensor draws from
    amap = AgentMap(grid, len(CATEGORIES), tuple(cfg["sensor"]["p_bins"]), tau, prior, kernel)
    key_to_oid = amap._id_of  # sensor key -> agent id: evaluator and REF only, never in a DecisionState
    cat_ids = {CATEGORIES.index(c) for c in task["categories"]}
    region = task_region(gt, grid, cat_ids, cfg.get("region_radius_m", 1.0))
    true_cat = {o.key: o.category for o in gt.objects}
    # every task's score for every run: the task-swap null scores a run made under T' against T, and S_h needs t = B/2
    all_tasks = {t: ({CATEGORIES.index(c) for c in v["categories"]}) for t, v in cfg["tasks"].items()}
    all_regions = {t: task_region(gt, grid, ids, cfg.get("region_radius_m", 1.0)) for t, ids in all_tasks.items()}
    mid = {}
    oid_to_key = {}

    st = {"rc": grid.start, "heading": 0, "steps": 0}
    traj, decisions, history, curve, confident, frames = [], [], [], [], [], []
    seen_conf = set()

    def evaluate():
        return score(amap, gt, key_to_oid, cat_ids, tau, region)

    def evaluate_all():
        return {t: score(amap, gt, key_to_oid, ids, tau, all_regions[t]) for t, ids in all_tasks.items()}

    def after_sense():
        st["steps"] += 1
        for k, oid in key_to_oid.items():
            oid_to_key[oid] = k
            if oid in seen_conf:
                continue
            cat, conf = amap.label(oid)
            if conf >= tau:
                seen_conf.add(oid)
                confident.append({"step": st["steps"], "oid": oid, "label": CATEGORIES[cat],
                                  "true": CATEGORIES[true_cat[k]] if k in true_cat else None})
        checkpoint()
        traj.append([st["steps"], st["rc"][0], st["rc"][1], st["heading"]])

    def checkpoint():
        """After every step, a sensing step or an executor bump: a bump that lands on B/2 or on a curve step must not
        skip that record (before 10-07 it did, leaving gaps in the curves and S_h read off the final map)."""
        if st["steps"] == budget // 2:
            mid.update(evaluate_all())
        if st["steps"] % cfg.get("curve_every", 10) == 0:
            curve.append({"step": st["steps"], **evaluate(), "by_task": evaluate_all()})

    def sense(h):
        st["heading"] = h
        amap.integrate(backend.observe(st["rc"], h))
        after_sense()

    def look_around():
        h0 = st["heading"]
        for i in range(cfg.get("lookaround_steps", 4)):
            if st["steps"] >= budget:
                return
            sense((h0 + i + 1) % 4)

    def settle(cand):
        """After the look-around at a target: what is still open there cannot be resolved from here."""
        if cand.kind == "revisit":
            # Only members the look-around could have seen up close count as tried: a single-linkage cluster can stretch
            # far past the viewpoint, and marking all of it tried dropped 41-52% of members with no new evidence (M 10-06).
            # The nearest member is always marked, so every revisit retires at least one object (no livelock).
            amap.tried.update(revisit_retired(amap, cand.objects, st["rc"], cfg["candidates"].get("revisit_radius", 1.5) / grid.cell))
        else:
            r, c = st["rc"]
            rad = int(round(cfg.get("dead_frontier_m", 0.4) / grid.cell))  # small: a wide radius can kill a doorway
            win = (slice(max(0, r - rad), r + rad + 1), slice(max(0, c - rad), c + rad + 1))
            amap.dead[win] |= frontier_mask(amap.occ)[win]

    def cluster_state(oids):
        out = []
        for oid in oids:
            cat, conf = amap.label(oid)
            k = oid_to_key.get(oid)
            out.append({"oid": oid, "label": CATEGORIES[cat], "conf": round(float(conf), 6),
                        "correct": bool(k in true_cat and cat == true_cat[k])})
        return out

    look_around()
    end_reason = "budget"
    while st["steps"] < budget:
        cands = generate(amap, st["rc"], **cfg["candidates"])
        if not cands:
            end_reason = "no_candidates"
            break
        ds = decision_state(task["sentence"], budget - st["steps"], grid, st["rc"], st["heading"], amap, cands, history,
                            max_objects=cfg.get("state", {}).get("max_objects", 60),
                            task_categories=task["categories"] if cfg.get("llm", {}).get("render") in ("v6", "v7") else None,
                            look_m=cfg["candidates"].get("revisit_radius", 1.5), los_m=cfg["candidates"].get("los_clearance", 0.5))
        ctx = {"seed": seed, "decision_index": len(decisions), "gt": gt, "amap": amap, "tau": tau, "key_to_oid": key_to_oid,
               "region": region, "nav": backend.nav}  # gt, key_to_oid, region, nav: REF only (S4b), never in the DecisionState
        ch = policy.decide(ds, cands, ctx)
        cand = next(c for c in cands if c.cid == ch.cid)
        rec = {"step": st["steps"], "state": canonical_state(ds), "cid": ch.cid, "kind": cand.kind, "target": list(cand.target),
               "reason": ch.reason, "predicted_beyond": ch.predicted_beyond, "fallback": ch.fallback,
               "n_frontiers": sum(c.kind == "frontier" for c in cands), "meta": ch.meta}
        if cand.kind == "revisit":
            rec["before"] = cluster_state(cand.objects)
        if record:
            frames.append({"step": st["steps"], "occ": amap.occ.copy(), "rc": st["rc"], "heading": st["heading"], "cands": cands, "cid": ch.cid,
                           "fallback": ch.fallback,
                           "nearest": min(cands, key=lambda c: c.steps).cid, "reason": ch.reason,
                           "objects": amap.objects_summary()})
        decisions.append(rec)
        history.append({"step": st["steps"], "cid": ch.cid, "kind": cand.kind, "reason": ch.reason[:200]})

        since, route = 0, []
        while st["steps"] < budget:
            if st["rc"] == cand.target:
                look_around()
                settle(cand)
                break
            if since >= cfg["decision_every"] or not still_valid(cand, amap):
                break
            if not route or any(amap.occ[r] != FREE for r in route):
                route = plan_path(amap.occ == FREE, st["rc"], cand.target)
                if not route:
                    break
            nxt = route[0]
            if backend.traversable(nxt):
                route.pop(0)
                h = _heading(st["rc"], nxt)
                st["rc"] = nxt
                sense(h)
            else:
                amap.mark_occupied(nxt)
                route = []
                st["steps"] += 1
                checkpoint()
                traj.append([st["steps"], st["rc"][0], st["rc"][1], st["heading"]])
            since += 1
        if cand.kind == "revisit":
            rec["after"] = cluster_state(cand.objects)

    final = evaluate()
    if not mid:  # episode ended before B/2: the map stays as it is
        mid.update(evaluate_all())
    return {"budget": budget, "metrics": final, "metrics_by_task": evaluate_all(), "metrics_by_task_mid": mid, "steps_used": st["steps"], "end_reason": end_reason, "trajectory": traj,
            "decisions": decisions, "curve": curve, "confident": confident, "frames": frames, "amap": amap,
            "n_fallback": sum(d["fallback"] for d in decisions)}

