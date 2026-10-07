"""Stage 6 path planner: 8-connected BFS on the agent's own map (passable = occ == FREE; UNKNOWN blocks).

Every move, straight or diagonal, costs one step. A diagonal move needs one of its two orthogonal neighbours
passable (no squeezing between two blocked cells); without diagonals, corridors that run at an angle on a 0.2 m
grid are often connected only at corners and their frontiers became unreachable (M calibration 10-06 20:30).
Deterministic: neighbours are expanded in MOVES order, and `path` walks back from the goal taking the first legal
MOVES-order neighbour one step closer to src. The src cell is always the BFS root, passable or not.
"""
from __future__ import annotations

import numpy as np

from amap.contracts import STEPS

DIAG = ((1, 1), (1, -1), (-1, -1), (-1, 1))
MOVES = STEPS + DIAG


def bfs(passable: np.ndarray, src) -> np.ndarray:
    """(n0, n1) int32 step distances from src over passable cells; -1 where unreachable."""
    return bfs_multi(passable, [src])


def bfs_multi(passable: np.ndarray, sources) -> np.ndarray:
    """Step distances to the nearest of several sources (each a BFS root, passable or not); same moves as bfs."""
    n0, n1 = passable.shape
    W = n1 + 2
    pad = np.zeros((n0 + 2, W), dtype=np.uint8)
    pad[1:-1, 1:-1] = passable
    pas = pad.tobytes()                # immutable passability for the diagonal corner rule
    avail = bytearray(pas)             # 1 = passable and not yet reached; the zero border needs no bounds check
    roots, seen = [], set()
    for src in sources:
        r, c = int(src[0]), int(src[1])
        if not (0 <= r < n0 and 0 <= c < n1):
            raise ValueError(f"src {src} outside grid {passable.shape}")
        s = (r + 1) * W + c + 1
        if s not in seen:
            seen.add(s)
            avail[s] = 0
            roots.append(s)
    if not roots:
        return np.full((n0, n1), -1, dtype=np.int32)
    orth = (1, W, -1, -W)                                      # STEPS order: +c, +r, -c, -r
    diag = ((W + 1, W, 1), (W - 1, W, -1), (-W - 1, -W, -1), (-W + 1, -W, 1))  # DIAG order, with its two orthogonals
    order, counts, frontier = list(roots), [len(roots)], list(roots)
    while frontier:
        nxt = []
        app = nxt.append
        for i in frontier:
            for o in orth:
                j = i + o
                if avail[j]:
                    avail[j] = 0
                    app(j)
            for o, a1, a2 in diag:
                j = i + o
                if avail[j] and (pas[i + a1] or pas[i + a2]):
                    avail[j] = 0
                    app(j)
        if nxt:
            order.extend(nxt)
            counts.append(len(nxt))
        frontier = nxt
    dist = np.full(pad.size, -1, dtype=np.int32)
    dist[np.asarray(order, dtype=np.int64)] = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    return np.ascontiguousarray(dist.reshape(pad.shape)[1:-1, 1:-1])


def path(passable: np.ndarray, src, goal, dist: np.ndarray | None = None) -> list[tuple[int, int]] | None:
    """Shortest 8-connected route, src excluded, goal included; [] if goal == src, None if unreachable.

    `dist` may be a precomputed bfs(passable, src) to skip the search."""
    src = (int(src[0]), int(src[1]))
    goal = (int(goal[0]), int(goal[1]))
    if goal == src:
        return []
    d = bfs(passable, src) if dist is None else dist
    n0, n1 = d.shape
    if not (0 <= goal[0] < n0 and 0 <= goal[1] < n1) or d[goal] < 0:
        return None
    r, c = goal
    k = int(d[goal])
    out = [goal]
    def ok(rr, cc, dr, dc):  # the move (rr, cc) -> (r, c) obeys the corner rule
        return dr == 0 or dc == 0 or bool(passable[rr + dr, cc]) or bool(passable[rr, cc + dc])

    while k > 1:
        for dr, dc in MOVES:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n0 and 0 <= cc < n1 and d[rr, cc] == k - 1 and ok(rr, cc, -dr, -dc):
                r, c = rr, cc
                break
        out.append((r, c))
        k -= 1
    out.reverse()
    return out
