"""Data contracts between the environment backends (stages 1-2) and the shared core (stages 3-7).

Grid geometry. Every backend and the core share one fixed n x n agent grid of square cells (`cell` metres)
whose origin is the episode start: the start is the centre of cell (n//2, n//2), and cell (r, c) has its centre
at agent-frame metres x = (c - n//2) * cell, y = (r - n//2) * cell. Heading h in 0..3 faces +x, +y, -x, -y.
Agent-frame coordinates are the only coordinates the core, the policies and the LLM ever see; the grid size is
fixed by config, never by the environment's extent (otherwise an unseen wall would change the agent's input).

Truth firewall. Ground truth (object categories, unobserved geometry, the navmesh) lives only inside a backend.
The core receives `Observation`s; the executor asks `Backend.traversable`; the evaluator reads `Backend.gt`.
Stages 4-6 (candidates, decision, path planning) must work from `AgentMap` alone.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

UNKNOWN, FREE, OCC = -1, 0, 1
STEPS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # (dr, dc) for heading 0..3: +x is +c, +y is +r

# Label space shared by both backends: the mpcat40 object classes (structural classes such as wall, floor,
# door, window, stairs, ceiling removed). A wrong label is drawn uniformly from the other classes.
CATEGORIES = (
    "chair", "table", "picture", "cabinet", "cushion", "sofa", "bed", "curtain", "chest_of_drawers", "plant",
    "sink", "toilet", "stool", "towel", "mirror", "tv_monitor", "shower", "bathtub", "counter", "fireplace",
    "lighting", "shelving", "gym_equipment", "seating", "furniture", "appliances", "clothes", "objects",
    "wardrobe",   # 10-07: split out of cabinet so that "store clothes" can name it (appended: earlier indices unchanged)
)
CAT = {name: i for i, name in enumerate(CATEGORIES)}


def stable_key(text: str) -> int:
    """Non-negative 60-bit int from an instance's string id; seeds the noise RNG (never Python's hash())."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)


@dataclass(frozen=True)
class GridSpec:
    n: int
    cell: float

    @property
    def start(self) -> tuple[int, int]:
        return (self.n // 2, self.n // 2)

    def to_xy(self, rc) -> tuple[float, float]:
        return ((rc[1] - self.n // 2) * self.cell, (rc[0] - self.n // 2) * self.cell)

    def to_rc(self, xy) -> tuple[int, int]:
        return (self.n // 2 + int(np.floor(xy[1] / self.cell + 0.5)), self.n // 2 + int(np.floor(xy[0] / self.cell + 0.5)))

    def inside(self, rc) -> bool:
        return 0 <= rc[0] < self.n and 0 <= rc[1] < self.n


@dataclass(frozen=True)
class Detection:
    key: int              # sensor-side instance key (stable_key of the instance id): association inside AgentMap
    rc: tuple[int, int]   # cell of the object's position estimate
    dist_bin: int         # 0: [0, 1.5) m, 1: [1.5, 3) m, 2: [3, range] m
    label: int            # noisy category index into CATEGORIES


@dataclass
class Observation:
    free: np.ndarray                                  # (k, 2) int cells seen free
    occupied: np.ndarray                              # (m, 2) int cells seen occupied
    detections: list[Detection] = field(default_factory=list)


@dataclass(frozen=True)
class GTObject:
    key: int
    category: int
    rc: tuple[int, int]


@dataclass
class GroundTruth:
    """Evaluator-only. Never passed to AgentMap, candidates, planner or policies (REF excepted, S4b only)."""
    objects: list[GTObject]       # start floor only
    floor: np.ndarray             # (n, n) bool: cells on the start floor (free or furniture), the R_T universe


class Backend:
    """What a backend provides. L: amap.lworld.ProcthorBackend. M: amap.habitat_env (Phase 1b)."""
    grid: GridSpec
    gt: GroundTruth
    nav: np.ndarray       # (n, n) bool true navigable cells: executor and REF only, never the agent

    def observe(self, rc: tuple[int, int], heading: int) -> Observation:
        """Sense from cell rc facing heading (90 deg FOV, `range` m); the shared noise model is applied here."""
        raise NotImplementedError

    def traversable(self, rc: tuple[int, int]) -> bool:
        """Executor only: may the agent stand in this cell?"""
        raise NotImplementedError
