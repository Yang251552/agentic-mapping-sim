"""L backend (amap/lworld.py) checks. Run: python -m pytest tests/test_lworld.py -q  (-s prints the coverage table)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from amap.contracts import CAT, GridSpec
from amap.lworld import (FLOOR, FURN, WALL, ProcthorBackend, _inside, _label4, _poly, category_of, drop_room,
                         load_tune, room_of)

G = GridSpec(400, 0.2)
HOUSES = load_tune()


def world_xz(b, rc):
    """World (x, z) of the centre of agent cell rc."""
    n, cell = b.grid.n, b.grid.cell
    return ((rc[1] + b.start_world[1] - n // 2 + 0.5) * cell, (rc[0] + b.start_world[0] - n // 2 + 0.5) * cell)


def door_rooms(house, room_id):
    """Rooms reachable from room_id through interior doors."""
    seen, stack = set(), [room_id]
    while stack:
        r = stack.pop()
        if r not in seen:
            seen.add(r)
            stack += [d["room1"] if d["room0"] == r else d["room0"] for d in house["doors"]
                      if r in (d["room0"], d["room1"]) and d["room0"] != d["room1"]]
    return seen


def poly_dist(poly, x, z):
    if _inside(poly, np.float64(x), np.float64(z)):
        return 0.0
    best = math.inf
    for (x1, z1), (x2, z2) in zip(poly, poly[1:] + poly[:1]):
        t = max(0.0, min(1.0, ((x - x1) * (x2 - x1) + (z - z1) * (z2 - z1)) / ((x2 - x1) ** 2 + (z2 - z1) ** 2)))
        best = min(best, math.hypot(x - x1 - t * (x2 - x1), z - z1 - t * (z2 - z1)))
    return best


def same_obs(o1, o2):
    return all(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
               for a, b in ((o1.free, o2.free), (o1.occupied, o2.occupied))) and o1.detections == o2.detections


def test_start_component_covers_house():
    rows, bad = [], []
    for scene, house in HOUSES.items():
        b = ProcthorBackend(house, G, 0)
        trav = b.cells == FLOOR
        lab, _ = _label4(trav)
        comp = lab == lab[G.start]
        r, c = np.nonzero(trav)
        x, z = world_xz(b, (r, c))
        reach_rooms = door_rooms(house, room_of(house, *world_xz(b, G.start)))
        in_reach = np.zeros(len(r), bool)
        for room in house["rooms"]:
            if room["id"] in reach_rooms:
                in_reach |= _inside(_poly(room), x, z)
        on = comp[r, c]
        f_all, f_reach = on.mean(), on[in_reach].mean()
        connected = len(reach_rooms) == len(house["rooms"])
        rows.append(f"{scene}: start component = {f_all:.3f} of all traversable, {f_reach:.3f} of door-reachable "
                    f"rooms ({len(reach_rooms)}/{len(house['rooms'])} rooms door-connected)")
        assert not on[~in_reach].any(), f"{scene}: start component leaks into rooms with no door path"
        if f_reach < 0.9 or (connected and f_all < 0.9):
            bad.append(rows[-1])
    print("\n" + "\n".join(rows))
    assert not bad, bad


def two_rooms(door: bool) -> dict:
    """Room A = [0, 4.1] x [0, 4], room B = [4.1, 8] x [0, 4] sharing the wall x = 4.1; a chair in B."""
    polys = {"room|1": [(0, 0), (4.1, 0), (4.1, 4), (0, 4)], "room|2": [(4.1, 0), (8, 0), (8, 4), (4.1, 4)]}
    rooms = [{"id": k, "roomType": "LivingRoom", "floorPolygon": [{"x": x, "y": 0, "z": z} for x, z in p]}
             for k, p in polys.items()]
    walls = [{"id": f"wall|{k[5:]}|{p[i][0]:.2f}|{p[i][1]:.2f}|{p[i - 3][0]:.2f}|{p[i - 3][1]:.2f}", "roomId": k,
              "polygon": [{"x": q[0], "y": y, "z": q[1]} for y in (0, 2.5) for q in (p[i], p[i - 3])]}
             for k, p in polys.items() for i in range(4)]
    doors = [{"id": "door|1|2", "room0": "room|1", "room1": "room|2", "wall0": "wall|1|4.10|0.00|4.10|4.00",
              "wall1": "wall|2|4.10|4.00|4.10|0.00",
              "holePolygon": [{"x": 1.5, "y": 0, "z": 0}, {"x": 2.5, "y": 2.0, "z": 0}]}] if door else []
    objects = [{"id": "Chair|2|0", "position": {"x": 6.0, "y": 0.4, "z": 2.0}}]
    return {"rooms": rooms, "walls": walls, "doors": doors, "objects": objects}


def test_wall_blocks_sight_and_door_opens_it():
    seen_b, chair_seen = {}, {}
    for door in (False, True):
        b = ProcthorBackend(two_rooms(door), G, 0, start_world=(10, 10))  # (x, z) = (2.1, 2.1) in room A
        assert any(w["id"] == "wall|1|4.10|0.00|4.10|4.00" for w in two_rooms(door)["walls"])
        seen_b[door] = chair_seen[door] = 0
        for rc in np.argwhere(b.cells == FLOOR):
            if world_xz(b, rc)[0] > 4.0:
                continue  # stand in room A only
            for h in range(4):
                o = b.observe(tuple(rc), h)
                cells = np.r_[o.free, o.occupied]
                seen_b[door] += int((world_xz(b, (cells[:, 0], cells[:, 1]))[0] > 4.2).sum())
                chair_seen[door] += len(o.detections)
    assert seen_b[False] == 0 and chair_seen[False] == 0, "room B seen through a solid wall"
    assert seen_b[True] > 0 and chair_seen[True] > 0, "room B not visible through the door"


def test_labels_deterministic():
    house = HOUSES["train-05485"]
    b = ProcthorBackend(house, G, 3)
    cells = np.argwhere(b.cells == FLOOR)
    cells = cells[np.random.default_rng(0).choice(len(cells), 60, replace=False)]
    calls = [(tuple(rc), h) for rc in cells for h in range(4)]

    def labels(backend, order):
        out = {}
        for rc, h in order:
            for d in backend.observe(rc, h).detections:
                assert out.setdefault((d.key, d.dist_bin), d.label) == d.label
        return out

    first = labels(b, calls)
    assert len(first) >= 20
    assert labels(ProcthorBackend(house, G, 3, start_world=b.start_world), calls[::-1]) == first
    other = labels(ProcthorBackend(house, G, 4, start_world=b.start_world), calls)
    assert other.keys() == first.keys() and any(other[k] != first[k] for k in first)


def test_traversable():
    b = ProcthorBackend(HOUSES["train-00516"], G, 0)
    for code, want in ((WALL, False), (FURN, False), (FLOOR, True)):
        cells = np.argwhere(b.cells == code)
        assert len(cells) > 0
        assert all(b.traversable(tuple(rc)) == want for rc in cells)
    assert not b.traversable((-1, 0)) and not b.traversable((G.n, G.n // 2))


def test_start_clearance_and_replay():
    avoid = ("toilet", "bed")
    for scene, house in HOUSES.items():
        b = ProcthorBackend(house, G, 7, avoid_categories=avoid)
        assert b.traversable(G.start), scene
        sx, sz = world_xz(b, G.start)
        for o in house["objects"]:
            if category_of(o["id"]) in avoid:
                assert math.hypot(o["position"]["x"] - sx, o["position"]["z"] - sz) >= 2.0, (scene, o["id"])
        b2 = ProcthorBackend(house, G, 7, start_world=b.start_world)
        assert b2.gt.floor.tobytes() == b.gt.floor.tobytes() and b2.cells.tobytes() == b.cells.tobytes()
        assert b2.gt.objects == b.gt.objects
        assert {CAT[category_of(o["id"])] for o in house["objects"]} == {g.category for g in b.gt.objects}


def test_drop_far_room_keeps_start_room_observations():
    """Up to 2 far rooms per house (corner-moving drops first): observations from start-room cells > range + 0.5 m
    from the dropped room, all 4 headings, are byte-identical. Dropping a room on the origin side would shift a
    raster anchored at the house's min corner; the absolute-origin anchoring must not care."""
    rng_m = 4.0
    tested, corner_moved = 0, 0

    def corner(rooms):
        return min(x for r in rooms for x, _ in _poly(r)), min(z for r in rooms for _, z in _poly(r))

    for scene, house in HOUSES.items():
        b = ProcthorBackend(house, G, 0)
        base = {}
        start_room = room_of(house, *world_xz(b, G.start))
        r, c = np.nonzero(b.cells == FLOOR)
        x, z = world_xz(b, (r, c))
        ins = _inside(_poly(next(q for q in house["rooms"] if q["id"] == start_room)), x, z)
        cells = list(zip(r[ins][::10].tolist(), c[ins][::10].tolist()))
        far = [q["id"] for q in house["rooms"]
               if q["id"] != start_room and poly_dist(_poly(q), *world_xz(b, G.start)) > rng_m + 1.0]
        moved = {rid: corner([q for q in house["rooms"] if q["id"] != rid]) != corner(house["rooms"]) for rid in far}
        for rid in sorted(far, key=lambda rid: not moved[rid])[:2]:
            dropped = drop_room(house, rid)
            assert len(dropped["rooms"]) == len(house["rooms"]) - 1
            poly = _poly(next(q for q in house["rooms"] if q["id"] == rid))
            b2 = ProcthorBackend(dropped, G, 0, start_world=b.start_world)
            for rc in [rc for rc in cells if poly_dist(poly, *world_xz(b, rc)) > rng_m + 0.5] + [G.start]:
                for h in range(4):
                    if (rc, h) not in base:
                        base[rc, h] = b.observe(rc, h)
                    assert same_obs(base[rc, h], b2.observe(rc, h)), (scene, rid, rc, h)
            tested += 1
            corner_moved += corner(dropped["rooms"]) != corner(house["rooms"])
    print(f"\ndrop_room: {tested} drops compared, {corner_moved} of them moved the house's min corner")
    assert tested >= 5
    assert corner_moved, "no drop moved the house's min corner: the absolute-origin anchoring was not exercised"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
