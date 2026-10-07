"""Category prior from train-split screening rows (all floors, mapped categories), add-one smoothed, in CATEGORIES order.
Only train scenes may go in: the prior shapes the noise model and the agent's belief for every run.

  python scripts/category_prior.py out/screen_train20.jsonl out/prior_train.json out/tune12.txt   # third arg: scenes to leave out
"""
import json
import sys
from collections import Counter

from amap.contracts import CATEGORIES

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
skip = set(open(sys.argv[3]).read().split()) if len(sys.argv) > 3 else set()  # e.g. the tuning scenes: keep them out
rows = [r for r in rows if "cat_all_floors" in r and r["scene"] not in skip]
if not rows or any(not r["scene"][:5].isdigit() or int(r["scene"][:5]) >= 800 for r in rows):
    sys.exit("need screening rows with --names from HM3D train scenes only (00000-00799)")
rows = list({r["scene"]: r for r in reversed(rows)}.values())  # one row per scene (screening writes one per seed)
cnt = Counter()
for r in rows:
    cnt.update(r["cat_all_floors"])
tot = sum(cnt.values())
prior = [(cnt[c] + 1) / (tot + len(CATEGORIES)) for c in CATEGORIES]
json.dump({"source": "HM3D-Sem v0.2 train split, all floors, mapped categories, add-one smoothed",
           "scenes": sorted({r["scene"] for r in rows}), "n_objects": tot, "counts": {c: cnt[c] for c in CATEGORIES},
           "categories": list(CATEGORIES), "prior": [round(p, 8) for p in prior]}, open(sys.argv[2], "w"), indent=1)
print(f"prior from {len(rows)} scenes, {tot} objects; top: " + ", ".join(f"{c} {cnt[c] / tot:.3f}" for c, _ in cnt.most_common(5)))
