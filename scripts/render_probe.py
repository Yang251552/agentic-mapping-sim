"""Phase 1b probe (runs on the AWS instance inside the pinned habitat env): load HM3D-Sem tuning scenes, render
depth + semantic + RGB, check byte-identical re-renders, and dump what the M adapter needs to know
(dataset paths, semantic categories, object counts, navmesh heights). Writes only to --out.

  python scripts/render_probe.py --data /opt/dlami/nvme/amap/hm3d --out out/
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import time
import traceback

import numpy as np


def make_sim(habitat_sim, cfg_file, scene, res=256):
    b = habitat_sim.SimulatorConfiguration()
    b.scene_dataset_config_file = cfg_file
    b.scene_id = scene
    b.enable_physics = False
    b.gpu_device_id = 0
    specs = []
    for uuid, st in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH),
                     ("semantic", habitat_sim.SensorType.SEMANTIC)):
        s = habitat_sim.CameraSensorSpec()
        s.uuid, s.sensor_type, s.resolution, s.position, s.hfov = uuid, st, [res, res], [0.0, 0.88, 0.0], 90.0
        specs.append(s)
    a = habitat_sim.agent.AgentConfiguration(sensor_specifications=specs)
    return habitat_sim.Simulator(habitat_sim.Configuration(b, [a]))


def render(habitat_sim, sim, pos, yaw):
    st = habitat_sim.AgentState()
    st.position = np.asarray(pos, dtype=np.float32)
    st.rotation = np.quaternion(np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0)
    sim.get_agent(0).set_state(st)
    return sim.get_sensor_observations()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-scenes", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import habitat_sim
    import quaternion  # noqa: F401  (registers np.quaternion; ships with habitat-sim)
    from PIL import Image

    report = {"habitat_sim": habitat_sim.__version__, "numpy": np.__version__, "scenes": []}
    cfgs = sorted(glob.glob(os.path.join(a.data, "**", "*.scene_dataset_config.json"), recursive=True))
    report["dataset_configs"] = cfgs
    for c in cfgs:
        with open(c) as f:
            report.setdefault("dataset_config_text", {})[c] = f.read()[:4000]
    sem_cfg = next((c for c in cfgs if "annotated" in os.path.basename(c)), cfgs[0] if cfgs else "")
    glbs = sorted(g for g in glob.glob(os.path.join(a.data, "**", "*.basis.glb"), recursive=True))
    report["n_basis_glb"] = len(glbs)
    for glb in glbs[: a.max_scenes]:
        sc = {"scene": glb}
        sim = None
        try:
            t0 = time.time()
            sim = make_sim(habitat_sim, sem_cfg, glb)
            sc["load_s"] = round(time.time() - t0, 2)
            pf = sim.pathfinder
            sc["navmesh_loaded"] = bool(pf.is_loaded)
            sc["navigable_area_m2"] = round(float(pf.navigable_area), 1)
            pf.seed(1)
            pts = np.array([pf.get_random_navigable_point() for _ in range(400)])
            sc["nav_height_hist"] = np.histogram(pts[:, 1], bins=8)[0].tolist()
            sc["nav_height_range"] = [round(float(pts[:, 1].min()), 2), round(float(pts[:, 1].max()), 2)]
            ss = sim.semantic_scene
            objs = list(ss.objects)
            sc["n_objects"] = len(objs)
            sc["n_levels"] = len(ss.levels)
            sc["n_regions"] = len(ss.regions)
            cats = collections.Counter(o.category.name() if o.category else "None" for o in objs if o is not None)
            sc["top_categories"] = cats.most_common(60)
            sc["object_examples"] = [{"id": o.id, "semantic_id": int(o.semantic_id), "cat": o.category.name() if o.category else None,
                                      "center": [round(float(x), 2) for x in o.aabb.center()], "size": [round(float(x), 2) for x in o.aabb.size()]}
                                     for o in objs[:15] if o is not None]
            p = pts[0]
            t0 = time.time()
            o1 = render(habitat_sim, sim, p, 0.3)
            o2 = render(habitat_sim, sim, p, 0.3)
            n = 50
            for i in range(n):
                render(habitat_sim, sim, p, 0.3 + i * 0.1)
            sc["render_ms"] = round((time.time() - t0) / (n + 2) * 1000, 2)
            sc["identical_rerender"] = {k: bool(np.array_equal(o1[k], o2[k])) for k in ("rgb", "depth", "semantic")}
            sim.close()
            sim = None
            sim = make_sim(habitat_sim, sem_cfg, glb)
            o3 = render(habitat_sim, sim, p, 0.3)
            sc["identical_after_reload"] = {k: bool(np.array_equal(o1[k], o3[k])) for k in ("rgb", "depth", "semantic")}
            sem = o1["semantic"]
            ids, cnt = np.unique(sem, return_counts=True)
            sc["semantic_ids_in_view"] = {int(i): int(c) for i, c in zip(ids, cnt)}
            sc["depth_range"] = [float(o1["depth"].min()), float(o1["depth"].max())]
            sc["dtypes"] = {k: str(o1[k].dtype) + str(o1[k].shape) for k in ("rgb", "depth", "semantic")}
            tag = os.path.basename(glb).split(".")[0]
            Image.fromarray(o1["rgb"][..., :3]).save(os.path.join(a.out, f"{tag}_rgb.png"))
            d = np.clip(o1["depth"] / 5.0 * 255, 0, 255).astype(np.uint8)
            Image.fromarray(d).save(os.path.join(a.out, f"{tag}_depth.png"))
            Image.fromarray(((sem.astype(np.int64) * 2654435761) % 255).astype(np.uint8)).save(os.path.join(a.out, f"{tag}_sem.png"))
        except Exception:
            sc["error"] = traceback.format_exc()[-3000:]
        finally:
            if sim is not None:
                sim.close()  # an open simulator left behind breaks the next GL context (crash 10-06 17:38)
        report["scenes"].append(sc)
        print(json.dumps({k: v for k, v in sc.items() if k in ("scene", "load_s", "n_objects", "render_ms", "identical_rerender", "identical_after_reload", "error")}), flush=True)
    with open(os.path.join(a.out, "render_probe.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
