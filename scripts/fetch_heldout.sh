#!/usr/bin/env bash
# Fetch the 12 held-out HM3D-Semantics val scenes of configs/heldout_m.json with your own Matterport API token
# (HM3D_TOKEN_ID, HM3D_TOKEN_SECRET in the environment; request access at https://matterport.com/habitat-matterport-3d-research-dataset).
# HM3D is licensed for non-commercial academic use (Matterport End User License Agreement for Academic Use of Model Data).
#   source env.sh; AMAP_HM3D=/data/hm3d bash scripts/fetch_heldout.sh
set -euo pipefail
: "${PY:?source env.sh first}" "${AMAP_HM3D:?set AMAP_HM3D to a data directory}"
SCENES=($("$PY" -c "import json; print(' '.join(json.load(open('configs/heldout_m.json'))['scenes']))"))
[ "${#SCENES[@]}" -eq 12 ] || { echo "configs/heldout_m.json must list 12 scenes"; exit 1; }
D=$AMAP_HM3D/val; mkdir -p "$D"
"$PY" scripts/hm3d_fetch.py all semantic-configs --split val --dest "$D"
"$PY" scripts/hm3d_fetch.py get semantic-annots --split val --scenes "${SCENES[@]}" --dest "$D"
"$PY" scripts/hm3d_fetch.py get habitat --split val --scenes "${SCENES[@]}" --dest "$D"
echo "fetched ${#SCENES[@]} scenes into $D"
