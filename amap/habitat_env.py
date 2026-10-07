"""M backend (stages 1-2): habitat-sim renders HM3D-Semantics; depth -> 2D scan, semantic instances -> detections.

Runs only inside the pinned habitat-sim headless env on a Linux GPU instance. Hard-coded / simulated parts (README):
semantics come from the scene annotation plus the shared noise model (no detector, perfect data association via
instance ids); depth and occupancy are noise-free; the pose is ground truth; only the start floor is used; the
navmesh answers only the executor's "can I stand here" (`traversable`), never the planner.

Frames: habitat world is y-up; agent +x = world +x, agent +y = world +z; the start is the centre of agent cell
(n//2, n//2). Heading h faces agent STEPS[h], i.e. world +x, +z, -x, -z.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

from .contracts import CAT, CATEGORIES, Backend, Detection, GridSpec, GroundTruth, GTObject, Observation, stable_key
from .sensor import DEFAULT_P_BINS, dist_bin, noise_model, noisy_label

# HM3D-Sem has ~1600 free-text category names and no shipped mpcat40 table (S3b). Task classes use exact name
# lists so that e.g. "toilet paper" or "bed sheet" never count as a task object; other names go through keyword
# rules into the context vocabulary; structural names are not objects at all.
EXACT = {
    "sink": {"sink", "bath sink", "kitchen sink", "vessel sink", "sink/basin", "bathroom sink"},
    "toilet": {"toilet"},
    "shower": {"shower", "shower cabin", "shower stall", "showerhead", "shower floor", "shower tray", "shower base", "shower cabinet"},
    "bathtub": {"bathtub", "bath", "bath tub", "shower tub", "spa bathtub"},
    "counter": {"kitchen counter", "countertop", "counter", "bathroom counter", "washbasin counter"},
    "bed": {"bed", "bunk bed", "bedframe"},
    "chest_of_drawers": {"chest of drawers", "dresser", "drawers", "chest of drawer", "drawers for clothes"},
    "wardrobe": {"wardrobe", "closet", "armoire", "walk-in closet", "clothes closet"},
}
EXACT_OF = {name: cat for cat, names in EXACT.items() for name in names}
STRUCTURAL = re.compile(r"\b(wall|floor|ceiling|door|window|stair|stairs|step|railing|rail|beam|column|pillar|frame|unknown|"
                        r"baseboard|molding|moulding|trim|roof|ledge|vent|outlet|switch|pipe|void|misc|object)s?\b")
KEYWORDS = (  # first match wins; never maps to a task class
    ("chair", r"\bchairs?\b|armchair"), ("sofa", r"\bsofa|couch"), ("seating", r"\bbench|ottoman|pouf|seat\b"),
    ("lighting", r"lamp|light|chandelier|sconce"),
    ("table", r"\btables?\b|\bdesk\b|nightstand"), ("picture", r"picture|painting|poster|photo|\bart\b"),
    ("cabinet", r"cabinet|wardrobe|closet|cupboard"), ("cushion", r"cushion|pillow"), ("curtain", r"curtain|blind"),
    ("plant", r"plant|flower"), ("stool", r"stool"), ("towel", r"towel"), ("mirror", r"mirror"),
    ("tv_monitor", r"\btv\b|television|monitor|screen"), ("fireplace", r"fireplace"),
    ("shelving", r"shel(f|ves|ving)|bookcase"),
    ("appliances", r"applianc|refrigerator|fridge|oven|stove|microwave|washing machine|washer|dryer|dishwasher|kettle|toaster"),
    ("clothes", r"cloth|jacket|shirt|coat\b"), ("gym_equipment", r"treadmill|exercise|dumbbell|weights"),
)
KEYWORD_RE = [(c, re.compile(p)) for c, p in KEYWORDS]


def map_category(raw: str):
    """HM3D raw category name -> CATEGORIES index, or None for structural names (not an object)."""
    n = (raw or "").strip().lower()
    if n in EXACT_OF:
        return CAT[EXACT_OF[n]]
    if not n or STRUCTURAL.search(n):
        return None
    for cat, rx in KEYWORD_RE:
        if rx.search(n):
            return CAT[cat]
    return CAT["objects"]


YAW = (-np.pi / 2, np.pi, np.pi / 2, 0.0)

# One GL context per process: habitat-sim crashes when a second Simulator exists or the first is closed while another
# lives (10-06 18:5x). Backends of the same scene share one cached simulator (the agent pose is set on every render).
_SIM = {"key": None, "sim": None}


def close_sim():
    if _SIM["sim"] is not None:
        _SIM["sim"].close()
    _SIM.update(key=None, sim=None)  # heading 0..3 -> rotation about +y so the camera (-z) faces +x, +z, -x, -z


def find_scene(data_root: str, scene: str):
    if "$" in data_root:
        raise ValueError(f"unresolved environment variable in data path {data_root!r} (set AMAP_HM3D)")
    glbs = glob.glob(os.path.join(data_root, "**", scene, "*.basis.glb"), recursive=True)
    cfgs = sorted(glob.glob(os.path.join(data_root, "**", "*annotated*.scene_dataset_config.json"), recursive=True))
    if not glbs or not cfgs:
        raise FileNotFoundError(f"{scene}: glb {len(glbs)}, annotated dataset configs {len(cfgs)} under {data_root}")
    return glbs[0], cfgs[0]


class HabitatBackend(Backend):
    def __init__(self, cfg: dict, scene: str, grid: GridSpec, seed: int, *, avoid_categories=(), category_override=None,
                 start_xyz=None):
        import habitat_sim
        from habitat_sim.utils.common import quat_from_angle_axis

        self._hs, self._quat = habitat_sim, quat_from_angle_axis
        s, m = cfg["sensor"], cfg.get("habitat", {})
        self.grid, self.seed, self.scene = grid, seed, scene
        self.range_m, self.p_bins = float(s["range_m"]), tuple(s.get("p_bins", DEFAULT_P_BINS))
        _, kernel = noise_model(s)
        self.kernel_cdf = None if kernel is None else np.cumsum(kernel, axis=1)
        self.res, self.cam_h = int(m.get("res", 256)), float(m.get("camera_height", 0.88))
        self.band = tuple(m.get("obstacle_band", (0.15, 1.5)))
        self.min_px = int(m.get("min_pixels", 50))
        # distance that picks the noise bin: "median" = to the median back-projected pixel (the map position);
        # "near" = a low percentile of the instance's per-pixel distances, i.e. its nearest visible part
        self.dist_mode, self.dist_pct = s.get("dist", "median"), float(s.get("dist_pct", 10.0))
        if self.dist_mode not in ("median", "near"):
            raise ValueError(f"sensor.dist must be 'median' or 'near', got {self.dist_mode!r}")
        self.start_min_dist = float(cfg.get("start_min_dist_m", 2.0))  # spec default; S4b may only tighten it
        glb, dcfg = find_scene(os.path.expandvars(cfg["data"]), scene)
        self.sim = self._make_sim(glb, dcfg, float(s["fov_deg"]))
        pf = self.sim.pathfinder

        # semantic objects: instance id -> category index (structural names dropped)
        self.cat_of, self.center_of, self.raw_of = {}, {}, {}
        for o in self.sim.semantic_scene.objects:
            if o is None:
                continue
            raw = o.category.name() if o.category else ""
            c = map_category(raw)
            if c is not None:
                sid = int(o.semantic_id)
                self.cat_of[sid] = c
                self.raw_of[sid] = raw.strip().lower()  # screening statistics only
                self.center_of[sid] = np.array(o.aabb.center(), dtype=np.float64)  # Magnum Range3D: center() is a method
        if category_override:
            self.cat_of.update(category_override)
        self.key_of = {sid: stable_key(f"{scene}:{sid}") for sid in self.cat_of}

        # start: seeded navigable point on a large island, >= start_min_dist_m from every avoid-category object on that floor;
        # with start_max_task_geo_m = D (S4b start rule, user 10-07): for every task, at least one of its objects has its
        # centre within D in straight line AND the navmesh point it snaps to within D in path length, so that no (scene,
        # task) cell is one where nobody can reach the task region within the budget. (Both conditions are the rule as
        # run in boot ten, fixed before any of its results; the straight-line one is not only a speed-up.)
        avoid = {CAT[c] for c in avoid_categories}
        max_geo = cfg.get("start_max_task_geo_m")
        task_cats = [{CAT[c] for c in t["categories"]} for t in cfg["tasks"].values()]
        snapped = {}

        def geo(p, sid):  # path distance on the navmesh from p to the point next to the object
            if sid not in snapped:
                snapped[sid] = np.array(pf.snap_point(self.center_of[sid].astype(np.float32)), dtype=np.float32)
            q = snapped[sid]
            if not np.all(np.isfinite(q)):
                return np.inf
            sp = habitat_sim.ShortestPath()
            sp.requested_start, sp.requested_end = p.astype(np.float32), q
            return float(sp.geodesic_distance) if pf.find_path(sp) else np.inf

        def reaches_every_task(p):
            for cats in task_cats:
                objs = [sid for sid, c in self.cat_of.items() if c in cats and self._on_floor(sid, p[1])
                        and np.hypot(*(self.center_of[sid][[0, 2]] - p[[0, 2]])) <= max_geo]  # centre within D, straight line
                if not any(geo(p, sid) <= max_geo for sid in objs):
                    return False
            return True

        if start_xyz is None:
            pf.seed(int(seed))
            for _ in range(5000):
                p = np.array(pf.get_random_navigable_point(), dtype=np.float64)
                if not np.all(np.isfinite(p)) or pf.island_area(pf.get_island(p.astype(np.float32))) < 20.0:
                    continue
                near = [sid for sid, c in self.cat_of.items() if c in avoid and self._on_floor(sid, p[1])
                        and np.hypot(*(self.center_of[sid][[0, 2]] - p[[0, 2]])) < self.start_min_dist]
                if not near and (max_geo is None or reaches_every_task(p)):
                    start_xyz = p
                    break
            else:
                raise RuntimeError(f"{scene}: no start found")
        self.start_xyz = np.asarray(start_xyz, dtype=np.float64)
        self.y0 = float(self.start_xyz[1])
        self.island = pf.get_island(self.start_xyz.astype(np.float32))

        # executor mask: cells whose centre is navigable on the start island at the start height
        n = grid.n
        self.nav = np.zeros((n, n), bool)
        lo, hi = (np.array(b) for b in pf.get_bounds())
        c_lo, c_hi = self._rc_of(lo[0], lo[2]), self._rc_of(hi[0], hi[2])
        for r in range(max(0, min(c_lo[0], c_hi[0])), min(n, max(c_lo[0], c_hi[0]) + 1)):
            for c in range(max(0, min(c_lo[1], c_hi[1])), min(n, max(c_lo[1], c_hi[1]) + 1)):
                x, z = self._xz_of((r, c))
                pt = np.array([x, self.y0, z], dtype=np.float32)
                if pf.is_navigable(pt, 0.5) and pf.get_island(pt) == self.island:
                    self.nav[r, c] = True

        objs = []
        for sid, c in self.cat_of.items():
            if self._on_floor(sid, self.y0):
                rc = self._rc_of(self.center_of[sid][0], self.center_of[sid][2])
                if grid.inside(rc):
                    objs.append(GTObject(self.key_of[sid], c, rc))
        self.gt = GroundTruth(sorted(objs, key=lambda o: o.key), self.nav.copy())

        f = (self.res / 2) / np.tan(np.radians(float(s["fov_deg"])) / 2)
        u = (np.arange(self.res) + 0.5 - self.res / 2) / f
        self._xn, self._yn = np.meshgrid(u, -u)  # camera frame: x right, y up, looking along -z
        self.last_rgb = None

    # ---- geometry ------------------------------------------------------------------------------------------
    def _rc_of(self, x, z):
        return self.grid.to_rc((x - self.start_xyz[0], z - self.start_xyz[2]))

    def _xz_of(self, rc):
        dx, dz = self.grid.to_xy(rc)
        return self.start_xyz[0] + dx, self.start_xyz[2] + dz

    def _on_floor(self, sid, y0):
        y = self.center_of[sid][1]
        return y0 - 0.5 <= y <= y0 + 2.0

    def _make_sim(self, glb, dcfg, fov):
        key = (glb, dcfg, fov, self.res, self.cam_h)
        if _SIM["key"] == key:
            return _SIM["sim"]
        close_sim()
        hs = self._hs
        b = hs.SimulatorConfiguration()
        b.scene_dataset_config_file, b.scene_id, b.enable_physics, b.gpu_device_id = dcfg, glb, False, 0
        specs = []
        for uuid, st in (("rgb", hs.SensorType.COLOR), ("depth", hs.SensorType.DEPTH), ("semantic", hs.SensorType.SEMANTIC)):
            sp = hs.CameraSensorSpec()
            sp.uuid, sp.sensor_type, sp.resolution, sp.position, sp.hfov = uuid, st, [self.res, self.res], [0.0, self.cam_h, 0.0], fov
            specs.append(sp)
        _SIM.update(key=key, sim=hs.Simulator(hs.Configuration(b, [hs.agent.AgentConfiguration(sensor_specifications=specs)])))
        return _SIM["sim"]

    def render(self, rc, heading):
        x, z = self._xz_of(rc)
        st = self._hs.AgentState()
        st.position = np.array([x, self.y0, z], dtype=np.float32)
        st.rotation = self._quat(float(YAW[heading]), np.array([0.0, 1.0, 0.0]))
        self.sim.get_agent(0).set_state(st)
        return self.sim.get_sensor_observations()

    # ---- Backend -------------------------------------------------------------------------------------------
    def traversable(self, rc) -> bool:
        return self.grid.inside(rc) and bool(self.nav[rc[0], rc[1]])

    def observe(self, rc, heading) -> Observation:
        o = self.render(rc, heading)
        self.last_rgb = o["rgb"][..., :3]
        d = o["depth"].astype(np.float64)
        sem = o["semantic"].astype(np.int64)
        th = YAW[heading]
        cx, cz = self._xz_of(rc)
        xc, yc, zc = self._xn * d, self._yn * d, -d
        X = cx + np.cos(th) * xc + np.sin(th) * zc
        Z = cz - np.sin(th) * xc + np.cos(th) * zc
        H = self.cam_h + yc  # height above the start floor
        hd = np.hypot(X - cx, Z - cz)
        valid = (d > 0) & np.isfinite(d) & (hd <= self.range_m)
        obst = valid & (H >= self.band[0]) & (H <= self.band[1])
        floor = valid & (H < self.band[0])

        # 2D scan per image column: free up to the nearest obstacle (or the farthest floor seen), capped at range
        inf = np.where(obst, hd, np.inf).min(axis=0)
        far_floor = np.where(floor, hd, 0.0).max(axis=0)
        extent = np.minimum(np.minimum(inf, far_floor + self.grid.cell), self.range_m)
        az = th - np.arctan(self._xn[0])  # yaw of each column's ray: camera ray (x, 0, -1) = forward of yaw th - atan(x)
        step = self.grid.cell / 2
        free = []
        for j in range(self.res):
            if extent[j] <= 0:
                continue
            t = np.arange(0.0, extent[j] - 1e-6, step)
            dirx, dirz = -np.sin(az[j]), -np.cos(az[j])  # forward (-z) rotated by az about +y
            free.append(np.stack([cz + t * dirz, cx + t * dirx], axis=1))
        free_rc = self._cells(np.concatenate(free) if free else np.zeros((0, 2)))
        occ_rc = self._cells(np.stack([Z[obst], X[obst]], axis=1))
        free_rc = np.array(sorted(set(map(tuple, free_rc)) - set(map(tuple, occ_rc))), dtype=np.int64).reshape(-1, 2)

        dets = []
        ids, counts = np.unique(sem[valid], return_counts=True)
        for sid, cnt in zip(ids.tolist(), counts.tolist()):
            if cnt < self.min_px or sid not in self.cat_of or not self._on_floor(sid, self.y0):
                continue
            px = valid & (sem == sid)
            ox, oz = float(np.median(X[px])), float(np.median(Z[px]))
            d_obj = float(np.percentile(hd[px], self.dist_pct)) if self.dist_mode == "near" else float(np.hypot(ox - cx, oz - cz))
            b = dist_bin(d_obj)
            key = self.key_of[sid]
            dets.append(Detection(key, self._rc_of(ox, oz), b,
                                  noisy_label(self.seed, key, b, self.cat_of[sid], len(CATEGORIES), self.p_bins, self.kernel_cdf)))
        dets.sort(key=lambda t: t.key)
        return Observation(free_rc, occ_rc, dets)

    def _cells(self, zx: np.ndarray) -> np.ndarray:
        if len(zx) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        n, cell = self.grid.n, self.grid.cell
        r = n // 2 + np.floor((zx[:, 0] - self.start_xyz[2]) / cell + 0.5).astype(np.int64)
        c = n // 2 + np.floor((zx[:, 1] - self.start_xyz[0]) / cell + 0.5).astype(np.int64)
        rc = np.unique(np.stack([r, c], axis=1), axis=0)
        return rc[(rc[:, 0] >= 0) & (rc[:, 0] < n) & (rc[:, 1] >= 0) & (rc[:, 1] < n)]

    def close(self):
        """Releases the shared simulator (the next backend reloads the scene)."""
        close_sim()
