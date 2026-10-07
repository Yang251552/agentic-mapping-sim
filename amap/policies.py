"""Stage 5: the decision makers. All of them see the same DecisionState and the same candidate list; only REF
(S4b headroom reference, never a comparison) is handed ground truth.

The DecisionState is built here from the agent map alone (agent-frame coordinates, agent ids, fixed float
formatting); its canonical JSON is what the metamorphic test compares and what the LLM prompt is rendered from.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .candidates import seen_from
from .contracts import CATEGORIES, FREE, stable_key
from .planner import bfs_multi
from . import llm as llm_mod

HEADING_NAMES = ("+x", "+y", "-x", "-y")


def f1(v: float) -> str:
    return f"{v:+.1f}"


@dataclass
class Choice:
    cid: str
    reason: str = ""
    predicted_beyond: str = ""
    fallback: bool = False
    meta: dict = field(default_factory=dict)


def decision_state(task_sentence, steps_left, grid, agent_rc, heading, amap, cands, history, near_m=2.0, max_objects=60,
                   task_categories=None, look_m=1.5, los_m=0.5) -> dict:
    """Task-agnostic summary rules, identical for every policy (S6 prompt budget): objects confidently labelled with the
    catch-all class are left out; at most `max_objects` are listed, those a candidate refers to first, then the nearest.
    A revisit also lists which of its objects lie within near range (look_m) of its viewpoint with a clear line of sight on
    the agent map (los_m: the candidate rule's clearance): only those can be settled by the look-around there. task_categories (user 10-07: the LLM gets the same table as B2) is passed through as is."""
    x, y = grid.to_xy(agent_rc)
    obj_rc = {oid: rc for oid, _, _, rc in amap.objects_summary()}
    free = amap.occ == FREE
    catch_all = CATEGORIES.index("objects")
    objs = []
    for oid, cat, conf, rc in amap.objects_summary():
        if cat == catch_all and conf >= amap.tau:
            continue
        ox, oy = grid.to_xy(rc)
        objs.append({"id": f"o{oid}", "label": CATEGORIES[cat], "conf": f"{conf:.2f}", "x": f1(ox), "y": f1(oy),
                     "_d": float(np.hypot(ox - x, oy - y))})
    out, referenced = [], set()
    fr = [c.steps for c in cands if c.kind == "frontier"]
    base = min(fr) if fr else min((c.steps for c in cands), default=0)   # the nearest frontier (B1's target)
    for c in cands:
        tx, ty = grid.to_xy(c.target)
        d = {"id": c.cid, "kind": c.kind, "steps": int(c.steps), "extra": int(c.steps) - int(base), "x": f1(tx), "y": f1(ty)}
        if c.kind == "frontier":
            d["opening_m"] = f"{c.size * grid.cell:.1f}"
            near = []
            for o in objs:
                dist = float(np.hypot(float(o["x"]) - tx, float(o["y"]) - ty))
                if dist <= near_m:
                    near.append((dist, o["id"]))
            d["near"] = [{"id": oid, "dist_m": f"{dist:.1f}"} for dist, oid in sorted(near)[:3]]
            referenced.update(n["id"] for n in d["near"])
        else:
            d["objects"] = [f"o{oid}" for oid in c.objects]
            d["resolves"] = [f"o{oid}" for oid in c.objects
                             if np.hypot(obj_rc[oid][0] - c.target[0], obj_rc[oid][1] - c.target[1]) * grid.cell < look_m
                             and bool(seen_from(free, [c.target], obj_rc[oid], look_m / grid.cell, los_m / grid.cell)[0])]
            referenced.update(d["objects"])
        out.append(d)
    by_id = {o["id"]: o for o in objs}
    for oid in referenced - set(by_id):  # a revisit cluster may hold a confident catch-all object: keep it listed
        i = int(oid[1:])
        cat, conf = amap.label(i)
        ox, oy = grid.to_xy(amap.objects_summary()[i][3])
        by_id[oid] = {"id": oid, "label": CATEGORIES[cat], "conf": f"{conf:.2f}", "x": f1(ox), "y": f1(oy), "_d": 0.0}
    keep = sorted(by_id.values(), key=lambda o: (o["id"] not in referenced, o["_d"], int(o["id"][1:])))[:max(max_objects, len(referenced))]
    keep.sort(key=lambda o: int(o["id"][1:]))
    listed = [{k: v for k, v in o.items() if k != "_d"} for o in keep]
    extra = {"task_categories": list(task_categories)} if task_categories is not None else {}
    return {"task": task_sentence, **extra, "steps_left": int(steps_left), "pose": {"x": f1(x), "y": f1(y), "heading": HEADING_NAMES[heading]},
            "objects": listed, "n_objects_seen": amap.n_objects, "candidates": out, "recent": history[-3:]}


def render_user(ds: dict) -> str:
    """Plain-text view of the DecisionState for the LLM (same information, fewer tokens than JSON)."""
    ob = {o["id"]: o for o in ds["objects"]}
    lines = [f"Task: {ds['task']}", f"Steps left: {ds['steps_left']}",
             f"Pose: ({ds['pose']['x']}, {ds['pose']['y']}) m from the start, facing {ds['pose']['heading']}",
             f"Objects seen ({len(ds['objects'])} of {ds['n_objects_seen']} listed: confidently generic clutter omitted, "
             "then nearest first; id: label confidence @ x, y m):"]
    lines += [f"{o['id']}: {o['label']} {o['conf']} @ ({o['x']}, {o['y']})" for o in ds["objects"]] or ["(none yet)"]
    lines.append("Candidates (id: kind, path steps (extra steps over the nearest frontier), target x, y m, detail):")
    for c in ds["candidates"]:
        head = f"{c['id']}: {c['kind']}, {c['steps']} steps ({c['extra']:+d} vs nearest frontier), @ ({c['x']}, {c['y']})"
        if c["kind"] == "frontier":
            near = ", ".join(f"{n['id']} {ob[n['id']]['label']} {ob[n['id']]['conf']} ({n['dist_m']} m)" for n in c["near"])
            lines.append(f"{head}, opening {c['opening_m']} m" + (f", near: {near}" if near else ""))
        else:
            lines.append(f"{head}, look closer at: " + ", ".join(f"{i} {ob[i]['label']} {ob[i]['conf']}" for i in c["objects"]))
    lines.append("Recent decisions:")
    lines += [f"step {h['step']}: {h['cid']} ({h['kind']}) - {h['reason']}" for h in ds["recent"]] or ["(none yet)"]
    return "\n".join(lines)


def render_user_v6(ds: dict) -> str:
    """v6 view (user 10-07): the task's scored categories are stated (the table B2 has) and objects of those categories are
    marked; a frontier opens unexplored space, a revisit maps none and lists only the objects its look-around can settle
    (within near range of the viewpoint): cluster members farther away stay as uncertain as before."""
    cats = ds.get("task_categories", [])
    ob = {o["id"]: o for o in ds["objects"]}
    tag = lambda o: f"{o['id']} {o['label']} {o['conf']}" + (" [counts]" if o["label"] in cats else "")
    lines = [f"Task: {ds['task']}", "Objects that count for this task: " + (", ".join(cats) or "(not given)"),
             f"Steps left: {ds['steps_left']}",
             f"Pose: ({ds['pose']['x']}, {ds['pose']['y']}) m from the start, facing {ds['pose']['heading']}",
             f"Objects seen ({len(ds['objects'])} of {ds['n_objects_seen']} listed: confidently generic clutter omitted, "
             "then nearest first; id: label confidence @ x, y m; [counts] = a category that counts):"]
    lines += [f"{tag(o)} @ ({o['x']}, {o['y']})" for o in ds["objects"]] or ["(none yet)"]
    lines.append("Candidates (id: kind, path steps (extra steps over the nearest frontier), target x, y m, detail):")
    for c in ds["candidates"]:
        head = f"{c['id']}: {c['kind']}, {c['steps']} steps ({c['extra']:+d} vs nearest frontier), @ ({c['x']}, {c['y']})"
        if c["kind"] == "frontier":
            near = ", ".join(f"{tag(ob[n['id']])} ({n['dist_m']} m)" for n in c["near"])
            lines.append(f"{head}, opens unexplored space, opening {c['opening_m']} m" + (f", near: {near}" if near else ""))
        else:
            res = c.get("resolves", [])
            far = len(c["objects"]) - len(res)
            lines.append(f"{head}, maps no new area; the look-around there settles: " + (", ".join(tag(ob[i]) for i in res) or "nothing")
                         + (f" ({far} other cluster members are too far to settle)" if far else ""))
    lines.append("Recent decisions:")
    lines += [f"step {h['step']}: {h['cid']} ({h['kind']}) - {h['reason']}" for h in ds["recent"]] or ["(none yet)"]
    return "\n".join(lines)


def render_user_v7(ds: dict, cue_m: float = 2.0) -> str:
    """v7 (redesign round 3, 10-07): the v6 view; a frontier also lists every [counts] object within cue_m of its target
    (the shared `near` list keeps only the 3 nearest objects, so a counting object could be crowded out by clutter)."""
    cats = ds.get("task_categories", [])
    lines = render_user_v6(ds).split("\n")
    for i, line in enumerate(lines):
        c = next((c for c in ds["candidates"] if c["kind"] == "frontier" and line.startswith(f"{c['id']}: frontier,")), None)
        if c is None:
            continue
        listed = {n["id"] for n in c["near"]}
        cues = [o for o in ds["objects"] if o["label"] in cats and o["id"] not in listed
                and np.hypot(float(o["x"]) - float(c["x"]), float(o["y"]) - float(c["y"])) <= cue_m]
        if cues:
            lines[i] = line + "; other objects that count within 2 m: " + ", ".join(f"{o['id']} {o['label']} {o['conf']}" for o in cues)
    return "\n".join(lines)


RENDER = {"v5": render_user, "v6": render_user_v6, "v7": render_user_v7}


def canonical_state(ds: dict) -> str:
    return json.dumps(ds, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class B1:
    """Nearest frontier; once frontiers run out, the nearest revisit candidate. Candidates arrive sorted by steps."""
    name = "b1"

    def decide(self, ds, cands, ctx) -> Choice:
        fr = [c for c in cands if c.kind == "frontier"]
        return Choice((fr or cands)[0].cid, reason="nearest frontier" if fr else "nearest revisit")


class B0:
    name = "b0"

    def decide(self, ds, cands, ctx) -> Choice:
        rng = np.random.default_rng([ctx["seed"], ctx["decision_index"], stable_key("b0")])
        return Choice(cands[int(rng.integers(len(cands)))].cid, reason="random")


class B2:
    """Same-information scripted scorer with the experimenter's task->category table.
    lam set (evidence-gated, user 10-07): start from the nearest frontier (B1's choice) and switch only to a candidate
    with task evidence, i.e. objects labelled in C_T near the frontier or in the revisit cluster; score = sum of their
    confidences - lam * (extra steps over the nearest frontier, + 4 for a revisit taken instead of a frontier); ties stay with B1.
    lam None (before 10-07): frontier value = sum of confidences of nearby objects labelled in C_T + b * opening (m);
    revisit value = number of cluster objects labelled in C_T; score = a * value - steps / 100."""
    name = "b2"

    def __init__(self, task_cats, a=1.0, b=0.1, lam=None):
        self.task_cats, self.a, self.b, self.lam = set(task_cats), a, b, lam

    def _gated(self, ds) -> Choice:
        ob = {o["id"]: o for o in ds["objects"]}
        cs = ds["candidates"]
        f0 = min([c for c in cs if c["kind"] == "frontier"] or cs, key=lambda c: c["steps"])   # first minimum = B1's choice

        def evidence(c):
            ids = [n["id"] for n in c["near"]] if c["kind"] == "frontier" else c["objects"]
            return sum(float(ob[i]["conf"]) for i in ids if ob[i]["label"] in self.task_cats)
        best, best_s = f0["id"], evidence(f0)
        for c in cs:
            e = evidence(c)
            if c["id"] == f0["id"] or e <= 0:
                continue
            look = 4 if c["kind"] == "revisit" and f0["kind"] == "frontier" else 0   # a revisit instead of exploring
            s = e - self.lam * (max(0, c["steps"] - f0["steps"]) + look)
            if s > best_s + 1e-9:   # float noise must not turn a tie into a switch
                best, best_s = c["id"], s
        return Choice(best, reason=f"evidence {best_s:.3f}" if best != f0["id"] else "nearest frontier")

    def decide(self, ds, cands, ctx) -> Choice:
        if self.lam is not None:
            return self._gated(ds)
        ob = {o["id"]: o for o in ds["objects"]}
        best, best_s = None, None
        for c in ds["candidates"]:
            if c["kind"] == "frontier":
                v = sum(float(ob[n["id"]]["conf"]) for n in c["near"] if ob[n["id"]]["label"] in self.task_cats) + self.b * float(c["opening_m"])
            else:
                v = sum(1.0 for i in c["objects"] if ob[i]["label"] in self.task_cats)
            s = self.a * v - c["steps"] / 100.0
            if best_s is None or s > best_s:
                best, best_s = c["id"], s
        return Choice(best, reason=f"score {best_s:.3f}")


class REF:
    """Reads ground truth (S4b headroom only, never a comparison): heads for the candidate minimising path steps plus the
    distance to the nearest piece of unfinished task work, i.e. an unobserved cell of the task region R_T or a task
    object not yet labelled correctly and confidently. geodesic=False: straight-line distance in cells (can point
    through a wall); geodesic=True ("refg"): steps over the true navigable cells to a cell within near range of that work."""
    name = "ref"

    def __init__(self, task_cats, geodesic=False):
        self.task_cats = {CATEGORIES.index(c) for c in task_cats}
        self.geodesic = geodesic
        self.name = "refg" if geodesic else "ref"

    def decide(self, ds, cands, ctx) -> Choice:
        if self.geodesic:
            return self._decide_geodesic(cands, ctx)
        gt, amap, tau, key_to_oid = ctx["gt"], ctx["amap"], ctx["tau"], ctx["key_to_oid"]
        todo = []
        for o in gt.objects:
            if o.category not in self.task_cats:
                continue
            oid = key_to_oid.get(o.key)
            if oid is not None:
                cat, conf = amap.label(oid)
                if cat == o.category and conf >= tau:
                    continue
            todo.append(o.rc)
        region = ctx.get("region")
        if region is not None:
            todo.extend(map(tuple, np.argwhere(region & (amap.occ == -1))[::4]))  # every 4th unseen R_T cell is enough
        if not todo:
            return B1().decide(ds, cands, ctx)
        pts = np.array(todo)
        score = [c.steps + float(np.min(np.hypot(pts[:, 0] - c.target[0], pts[:, 1] - c.target[1]))) for c in cands]
        return Choice(cands[int(np.argmin(score))].cid, reason="oracle")

    def _decide_geodesic(self, cands, ctx) -> Choice:
        gt, amap, tau, key_to_oid, nav = ctx["gt"], ctx["amap"], ctx["tau"], ctx["key_to_oid"], ctx["nav"]
        cell, n = amap.grid.cell, nav.shape[0]
        R = int(np.ceil(1.5 / cell))
        disk = [(dr, dc) for dr in range(-R, R + 1) for dc in range(-R, R + 1) if (dr * dr + dc * dc) * cell * cell < 1.5 ** 2]
        src = set()
        for o in gt.objects:  # unfinished objects: any navigable cell within near range (bin 0) of the object
            if o.category not in self.task_cats:
                continue
            oid = key_to_oid.get(o.key)
            if oid is not None:
                cat, conf = amap.label(oid)
                if (cat == o.category and conf >= tau) or amap.near_seen(oid):  # done, or a near look already decided it
                    continue
            src.update((o.rc[0] + dr, o.rc[1] + dc) for dr, dc in disk
                       if 0 <= o.rc[0] + dr < n and 0 <= o.rc[1] + dc < n and nav[o.rc[0] + dr, o.rc[1] + dc])
        region = ctx.get("region")
        if region is not None:
            src.update(map(tuple, np.argwhere(region & (amap.occ == -1) & nav)))
        if not src:
            return B1().decide(None, cands, ctx)
        D = bfs_multi(nav, sorted(src))

        def to_work(t):  # the target is free on the agent map; read the nearest navigable value around it
            vals = [D[r, c] for r in range(t[0] - 1, t[0] + 2) for c in range(t[1] - 1, t[1] + 2)
                    if 0 <= r < n and 0 <= c < n and D[r, c] >= 0]
            return min(vals) if vals else 10 ** 6
        work = [to_work(c.target) for c in cands]
        if min(work) >= 10 ** 6:   # no candidate connects to unfinished work on the true map
            return B1().decide(None, cands, ctx)
        score = [c.steps + w for c, w in zip(cands, work)]
        return Choice(cands[int(np.argmin(score))].cid, reason="oracle-geodesic")


class LLMPolicy:
    """Forced go_to tool call. An invalid call is retried <= 2 times (each retry is its own cache entry); still invalid
    -> this step uses the B1 rule and is flagged fallback. Infra errors propagate (episode -> aborted_infra)."""
    name = "llm"

    def __init__(self, client: llm_mod.LLMClient, model: str, system_prompt: str, max_tokens=300, render="v5"):
        self.client, self.model, self.system, self.max_tokens, self.render = client, model, system_prompt, max_tokens, RENDER[render]

    def decide(self, ds, cands, ctx) -> Choice:
        ids = [c.cid for c in cands]
        body = llm_mod.request_body(self.model, self.system, self.render(ds), ids, self.max_tokens)
        usage = []
        for attempt in range(3):
            rec = self.client.call(body, attempt)
            usage.append({"usage": rec.get("usage"), "latency_s": rec.get("latency_s"), "cost_usd": rec.get("cost_usd")})
            a = rec.get("args")
            if isinstance(a, dict) and a.get("candidate_id") in ids:
                return Choice(a["candidate_id"], a.get("reason", ""), a.get("predicted_beyond", ""), meta={"calls": usage})
        c = B1().decide(ds, cands, ctx)
        return Choice(c.cid, c.reason, fallback=True, meta={"calls": usage})
