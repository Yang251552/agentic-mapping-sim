"""L backend (stages 1-2): a ProcTHOR-10K house rasterised to a 2D occupancy world with a 2D ray-cast sensor.

World raster: anchored at the absolute world origin (x=0, z=0), so adding or removing a room never shifts the rest
of the house. World cell (i, j) covers z in [i*cell, (i+1)*cell), x in [j*cell, (j+1)*cell). Agent frame: agent +x
is world +x, agent +y is world +z, and the start is the centre of agent cell (n//2, n//2).

Cell codes: VOID (outside every room: opaque, not traversable, never reported), FLOOR, FURN (object footprint:
blocks movement, not sight), WALL (room polygon edges, one cell thick, opaque; interior door openings carved out).

Door openings: holePolygon x offsets are measured along wall0 from the first floor vertex of wall0's `polygon`
(not from the min-sorted corner in the wall id; checked on the vendored houses: with the id corner, paintings and a
TV would hang inside door openings). Exterior doors (room0 == room1) stay closed.

Object footprints are hard-coded disks per category (the house JSON has no bounding boxes), restricted to the
non-wall floor cells of the object's own room; an object whose disk has no such cell (e.g. a painting hanging
0.02 m off the wall) gets the nearest one, so it is seen from its own side of the wall only.
"""
from __future__ import annotations

import copy
import json
import math
from collections import deque
from pathlib import Path

import numpy as np

from amap import sensor
from amap.contracts import (CAT, CATEGORIES, Backend, Detection, GridSpec, GroundTruth, GTObject, Observation,
                            stable_key)

VOID, FLOOR, FURN, WALL = 0, 1, 2, 3

_TYPES = {
    "chair": ("Chair",), "table": ("DiningTable", "Desk", "SideTable", "CoffeeTable", "TVStand"),
    "picture": ("Painting",), "cushion": ("Pillow",), "sofa": ("Sofa",), "seating": ("ArmChair", "Ottoman"),
    "bed": ("Bed",), "chest_of_drawers": ("Dresser",), "plant": ("HousePlant",), "sink": ("Sink",),
    "toilet": ("Toilet",), "stool": ("Stool",), "tv_monitor": ("Television",), "counter": ("CounterTop",),
    "lighting": ("FloorLamp", "DeskLamp"), "shelving": ("ShelvingUnit",),
    "appliances": ("Fridge", "Microwave", "Toaster", "CoffeeMachine", "WashingMachine", "ClothesDryer"),
    "gym_equipment": ("Dumbbell",),
}
TYPE_TO_CAT = {t: cat for cat, types in _TYPES.items() for t in types}  # any other ProcTHOR type -> "objects"

# Footprint radius (m) per category: hard-coded guesses, the house JSON carries no object extents.
RADIUS = {"bed": 0.9, "sofa": 0.8, "table": 0.5, "counter": 0.5, "chest_of_drawers": 0.4, "appliances": 0.4,
          "shelving": 0.4, "toilet": 0.35, "sink": 0.3, "picture": 0.15}
DEFAULT_RADIUS = 0.25
START_CLEARANCE_M = 2.0
ROOT = Path(__file__).resolve().parents[1]


def category_of(obj_id: str) -> str:
    return TYPE_TO_CAT.get(obj_id.split("|")[0], "objects")


def load_tune(path="data/procthor/tune.jsonl") -> dict:
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    with open(p) as f:
        return {d["scene"]: d["house"] for d in map(json.loads, f)}


# ---------------------------------------------------------------- geometry helpers

def _poly(room) -> list:
    return [(p["x"], p["z"]) for p in room["floorPolygon"]]


def _inside(poly, X, Z):
    """Even-odd point-in-polygon, vectorised over arrays X, Z."""
    c = np.zeros(np.broadcast(X, Z).shape, bool)
    for (x1, z1), (x2, z2) in zip(poly, poly[1:] + poly[:1]):
        if z1 == z2:
            continue
        c ^= ((z1 > Z) != (z2 > Z)) & (X < x1 + (Z - z1) * (x2 - x1) / (z2 - z1))
    return c


def room_of(house: dict, x: float, z: float):
    """Id of the first room whose floor polygon contains (x, z), else None."""
    for r in house["rooms"]:
        if _inside(_poly(r), np.float64(x), np.float64(z)):
            return r["id"]
    return None


def _seg_cells(p, q, cell):
    """4-connected world cells (i, j) crossed by segment p->q (points are (x, z))."""
    (x0, z0), (x1, z1) = p, q
    t = np.linspace(0.0, 1.0, max(1, math.ceil(math.hypot(x1 - x0, z1 - z0) / (cell / 8))) + 1)
    i = np.floor((z0 + t * (z1 - z0)) / cell).astype(np.int64)
    j = np.floor((x0 + t * (x1 - x0)) / cell).astype(np.int64)
    diag = (np.diff(i) != 0) & (np.diff(j) != 0)  # a diagonal step would let a ray slip through: fill the corner
    return np.unique(np.stack([np.r_[i, i[1:][diag]], np.r_[j, j[:-1][diag]]], 1), axis=0)


def _label4(mask):
    """4-connected components of a bool mask: (labels with -1 off-mask, sizes per label)."""
    lab = np.full(mask.shape, -1, np.int32)
    H, W = mask.shape
    sizes = []
    for i0, j0 in np.argwhere(mask):
        if lab[i0, j0] >= 0:
            continue
        k = len(sizes)
        lab[i0, j0] = k
        q, n = deque([(i0, j0)]), 0
        while q:
            i, j = q.popleft()
            n += 1
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < H and 0 <= b < W and mask[a, b] and lab[a, b] < 0:
                    lab[a, b] = k
                    q.append((a, b))
        sizes.append(n)
    return lab, np.array(sizes, np.int64)


# ---------------------------------------------------------------- backend

class ProcthorBackend(Backend):
    def __init__(self, house: dict, grid: GridSpec, seed: int, *, fov_deg=90.0, range_m=4.0,
                 p_bins=sensor.DEFAULT_P_BINS, avoid_categories=(), start_world=None):
        assert set(avoid_categories) <= set(CATEGORIES), f"unknown avoid_categories {avoid_categories}"
        self.grid, self.seed, self.p_bins, self.range_m = grid, seed, tuple(p_bins), float(range_m)
        cell = grid.cell
        rooms = [_poly(r) for r in house["rooms"]]
        xmax = max(x for p in rooms for x, _ in p)
        zmax = max(z for p in rooms for _, z in p)
        H, W = int(zmax // cell) + 2, int(xmax // cell) + 2
        Zc, Xc = (np.arange(H)[:, None] + 0.5) * cell, (np.arange(W)[None, :] + 0.5) * cell

        room_ix = np.full((H, W), -1, np.int32)
        for k, p in enumerate(rooms):
            room_ix[(room_ix < 0) & _inside(p, Xc, Zc)] = k
        floor = room_ix >= 0

        def clip(c):
            return c[(c[:, 0] >= 0) & (c[:, 0] < H) & (c[:, 1] >= 0) & (c[:, 1] < W)]

        edges = [(p[k], p[(k + 1) % len(p)]) for p in rooms for k in range(len(p))]
        edge_cells = [clip(_seg_cells(a, b, cell)) for a, b in edges]
        wall = np.zeros((H, W), bool)
        for c in edge_cells:
            wall[c[:, 0], c[:, 1]] = True

        walls_by_id = {w["id"]: w for w in house.get("walls", [])}
        for d in house.get("doors", []):
            if d["room0"] == d["room1"]:
                continue  # exterior door: keep the wall closed
            w = walls_by_id.get(d["wall0"])
            if w is not None:
                O, E = [(p["x"], p["z"]) for p in w["polygon"] if abs(p["y"]) < 1e-6][:2]
            else:  # ponytail: wall entry missing, fall back to the segment encoded in the id
                x0, z0, x1, z1 = map(float, d["wall0"].split("|")[2:6])
                O, E = (x0, z0), (x1, z1)
            L = math.hypot(E[0] - O[0], E[1] - O[1])
            D = ((E[0] - O[0]) / L, (E[1] - O[1]) / L)
            hx = sorted(h["x"] for h in d["holePolygon"][:2])
            a, b = max(hx[0], 0.0), min(hx[1], L)
            door = clip(_seg_cells((O[0] + D[0] * a, O[1] + D[1] * a), (O[0] + D[0] * b, O[1] + D[1] * b), cell))
            keep = np.ones(len(door), bool)
            for (p, q), c in zip(edges, edge_cells):  # never open cells of walls crossing this wall's line
                dx, dz = q[0] - p[0], q[1] - p[1]
                collinear = (abs(D[0] * dz - D[1] * dx) < 1e-6 * math.hypot(dx, dz)
                             and abs(D[0] * (p[1] - O[1]) - D[1] * (p[0] - O[0])) < 1e-3)
                if not collinear:
                    keep &= ~np.isin(door[:, 0] * W + door[:, 1], c[:, 0] * W + c[:, 1])
            door = door[keep]
            wall[door[:, 0], door[:, 1]] = False
            floor[door[:, 0], door[:, 1]] = True
        floor &= ~wall

        # objects: (key, category index, x, z, footprint world cells)
        self._objects = []
        furn = np.zeros((H, W), bool)
        for o in house.get("objects", []):
            x, z = o["position"]["x"], o["position"]["z"]
            cat = category_of(o["id"])
            r = RADIUS.get(cat, DEFAULT_RADIUS)
            rid = next((k for k, p in enumerate(rooms) if _inside(p, np.float64(x), np.float64(z))), None)
            valid = floor if rid is None else floor & (room_ix == rid)
            reach = int(math.ceil(r / cell)) + 3
            i0, j0 = math.floor(z / cell), math.floor(x / cell)
            ii, jj = np.mgrid[max(i0 - reach, 0):min(i0 + reach + 1, H), max(j0 - reach, 0):min(j0 + reach + 1, W)]
            d2 = ((jj + 0.5) * cell - x) ** 2 + ((ii + 0.5) * cell - z) ** 2
            ok = valid[ii, jj]
            sel = ok & ((d2 <= r * r) | ((ii == i0) & (jj == j0)))
            if not sel.any() and ok.any():
                sel = ok & (d2 == d2[ok].min())
            fp = np.stack([ii[sel], jj[sel]], 1)
            furn[fp[:, 0], fp[:, 1]] = True
            self._objects.append((stable_key(o["id"]), CAT[cat], x, z, fp))

        code = np.zeros((H, W), np.int8)
        code[floor] = FLOOR
        code[floor & furn] = FURN
        code[wall] = WALL
        self.world = code  # evaluator/tests only (truth firewall)

        if start_world is None:
            trav = code == FLOOR
            ok = trav.copy()
            for _, c, x, z, _ in self._objects:
                if CATEGORIES[c] in avoid_categories:
                    ok &= (Xc - x) ** 2 + (Zc - z) ** 2 >= START_CLEARANCE_M ** 2
            lab, sizes = _label4(trav)
            cands = np.argwhere(ok & (lab == int(np.argmax(sizes))))
            if len(cands) == 0:
                raise ValueError("no start cell satisfies the clearance constraint")
            k = int(np.random.default_rng([seed, stable_key("start")]).integers(len(cands)))
            start_world = tuple(int(v) for v in cands[k])
        self.start_world = tuple(int(v) for v in start_world)

        # agent window, padded by P void cells so rays never index out of bounds
        n = grid.n
        self._oi, self._oj = self.start_world[0] - n // 2, self.start_world[1] - n // 2
        P = self._P = int(math.ceil(self.range_m / cell)) + 2
        pad = np.zeros((n + 2 * P, n + 2 * P), np.int8)
        wi0, wi1 = max(self._oi, 0), min(self._oi + n, H)
        wj0, wj1 = max(self._oj, 0), min(self._oj + n, W)
        if wi0 < wi1 and wj0 < wj1:
            pad[P + wi0 - self._oi:P + wi1 - self._oi, P + wj0 - self._oj:P + wj1 - self._oj] = code[wi0:wi1, wj0:wj1]
        self._pad = pad
        self.cells = pad[P:P + n, P:P + n]  # (n, n) codes in the agent frame; evaluator/tests only

        gt_objs, fp_lin, fp_obj = [], [], []
        self._win_objects = []  # (key, cat, x, z, rc)
        for key, c, x, z, fp in self._objects:
            rc = (math.floor(z / cell) - self._oi, math.floor(x / cell) - self._oj)
            if not grid.inside(rc):
                continue
            k = len(self._win_objects)
            self._win_objects.append((key, c, x, z, rc))
            gt_objs.append(GTObject(key, c, rc))
            a = fp - (self._oi, self._oj)
            a = a[(a[:, 0] >= 0) & (a[:, 0] < n) & (a[:, 1] >= 0) & (a[:, 1] < n)]
            fp_lin.append(a[:, 0] * n + a[:, 1])
            fp_obj.append(np.full(len(a), k))
        self._fp_lin = np.concatenate(fp_lin) if fp_lin else np.zeros(0, np.int64)
        self._fp_obj = np.concatenate(fp_obj) if fp_obj else np.zeros(0, np.int64)
        self.gt = GroundTruth(objects=gt_objs, floor=(self.cells == FLOOR) | (self.cells == FURN))
        self.nav = self.cells == FLOOR  # executor mask (and REF-geodesic only)

        # ray fan per heading: angular step 0.5*cell/range, march step 0.25*cell (all offsets in agent cells)
        fov = math.radians(fov_deg)
        K = int(math.ceil(fov / (0.5 * cell / self.range_m))) + 1
        t = np.arange(1, int(self.range_m / (0.25 * cell)) + 1) * (0.25 * cell)
        self._rays = []
        for h in range(4):
            th = h * math.pi / 2 + np.linspace(-fov / 2, fov / 2, K)
            dr = np.floor(np.sin(th)[:, None] * t[None, :] / cell + 0.5).astype(np.int64)
            dc = np.floor(np.cos(th)[:, None] * t[None, :] / cell + 0.5).astype(np.int64)
            self._rays.append((dr, dc))

    def observe(self, rc, heading):
        assert self.grid.inside(rc), rc  # out-of-window indices would wrap around silently
        n, P = self.grid.n, self._P
        r, c = int(rc[0]), int(rc[1])
        dr, dc = self._rays[heading]
        codes = self._pad[r + P + dr, c + P + dc]
        block = (codes == WALL) | (codes == VOID)
        first = np.where(block.any(1), block.argmax(1), codes.shape[1])[:, None]
        idx = np.arange(codes.shape[1])[None, :]
        lin = (r + dr) * n + (c + dc)
        free = np.unique(np.r_[lin[(idx < first) & (codes == FLOOR)], r * n + c])
        occ = np.unique(lin[((idx < first) & (codes == FURN)) | ((idx == first) & (codes == WALL))])

        dets = []
        if len(self._fp_lin):
            hit = np.isin(self._fp_lin, np.r_[free, occ])
            cell = self.grid.cell
            ax, az = (c + self._oj + 0.5) * cell, (r + self._oi + 0.5) * cell
            for k in np.unique(self._fp_obj[hit]):
                key, cat, x, z, orc = self._win_objects[k]
                b = sensor.dist_bin(math.hypot(x - ax, z - az))
                lab = sensor.noisy_label(self.seed, key, b, cat, len(CATEGORIES), self.p_bins)
                dets.append(Detection(key, orc, b, lab))
            dets.sort(key=lambda d: d.key)
        return Observation(np.stack(np.divmod(free, n), 1), np.stack(np.divmod(occ, n), 1), dets)

    def traversable(self, rc):
        return self.grid.inside(rc) and int(self.cells[rc[0], rc[1]]) == FLOOR


# ---------------------------------------------------------------- metamorphic helpers (return deep copies)

def _in_room(house, room_id):
    poly = _poly({r["id"]: r for r in house["rooms"]}[room_id])
    return lambda o: bool(_inside(poly, np.float64(o["position"]["x"]), np.float64(o["position"]["z"])))


def drop_room(house: dict, room_id: str) -> dict:
    h = copy.deepcopy(house)
    inside = _in_room(h, room_id)
    h["objects"] = [o for o in h["objects"] if not inside(o)]
    h["rooms"] = [r for r in h["rooms"] if r["id"] != room_id]
    h["walls"] = [w for w in h.get("walls", []) if w.get("roomId") != room_id]
    h["doors"] = [d for d in h.get("doors", []) if room_id not in (d["room0"], d["room1"])]
    return h


def move_objects(house: dict, room_id: str, dx: float, dz: float) -> dict:
    h = copy.deepcopy(house)
    inside = _in_room(h, room_id)
    for o in [o for o in h["objects"] if inside(o)]:
        o["position"]["x"] += dx
        o["position"]["z"] += dz
    return h


def relabel_objects(house: dict, room_id: str, new_type: str) -> dict:
    """Rewrites the type prefix of the ids (so their stable_key, hence their noise draws, change too)."""
    h = copy.deepcopy(house)
    inside = _in_room(h, room_id)
    for o in h["objects"]:
        if inside(o):
            o["id"] = new_type + o["id"][o["id"].index("|"):]
    return h
