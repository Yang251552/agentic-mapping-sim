"""Scene screening from metadata only (no episode is run): for each scene and seed, the start-floor navigable area and
the number of task-category objects on the start floor. Used to pick tuning scenes and, at freeze time, held-out scenes.

  python -m eval.screen_scenes --config configs/tune_m.json --scenes A B C --seeds 1 --out out/screen.jsonl
Rows go to --out (habitat-sim prints renderer banners on stdout, so stdout is not a data channel).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from amap.contracts import CATEGORIES, GridSpec
from amap.run import make_backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--out", required=True)
    ap.add_argument("--names", action="store_true", help="also count raw category names on the start floor and mapped "
                    "categories on all floors (train scenes only: category-set and prior statistics)")
    a = ap.parse_args()
    out = open(a.out, "w")
    cfg = json.load(open(a.config))
    grid = GridSpec(cfg["grid"]["n"], cfg["grid"]["cell"])
    for scene in a.scenes:
        for seed in a.seeds:
            row = {"scene": scene, "seed": seed}
            try:
                be = make_backend(cfg, scene, seed)
                cats = [CATEGORIES[o.category] for o in be.gt.objects]
                row["nav_area_m2"] = round(float(be.gt.floor.sum()) * grid.cell ** 2, 1)
                for t, v in cfg["tasks"].items():
                    row[f"n_{t}"] = sum(c in v["categories"] for c in cats)
                row["n_objects"] = len(cats)
                if hasattr(be, "start_xyz"):   # the seeded start, recorded for the preregistration
                    row["start_xyz"] = [round(float(v), 3) for v in be.start_xyz]
                if a.names:
                    row["cat_all_floors"] = dict(Counter(CATEGORIES[c] for c in be.cat_of.values()))
                    row["names_start_floor"] = dict(Counter(be.raw_of[sid] for sid in be.cat_of if be._on_floor(sid, be.y0)))
                row["passes"] = all(row[f"n_{t}"] > 0 for t in cfg["tasks"])
            except Exception as e:  # a scene that cannot be loaded is listed, not hidden
                row["error"] = f"{type(e).__name__}: {e}"
                row["passes"] = False
            out.write(json.dumps(row) + "\n"); out.flush()
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
