#!/usr/bin/env bash
# One command: replay held-out batch 1 (12 scenes x 2 tasks x 2 seeds x {llm, b1, b0, b2} = 192 episodes) without an
# OpenAI key. Every LLM answer comes from llm_cache/heldout_m.jsonl; Habitat re-renders every observation on the GPU.
# Outputs per episode: out/heldout_m/replay/<scene>/<task>/s<seed>/<policy>/{map.png, map.npz, trajectory.json,
# decisions.jsonl, metrics.json}. Exit 0 only if every episode matches results/heldout_batch1_floor.json.
#   source env.sh; AMAP_HM3D=/data/hm3d bash scripts/replay_heldout.sh
set -uo pipefail
: "${PY:?source env.sh first}" "${AMAP_HM3D:?set AMAP_HM3D to the data directory of scripts/fetch_heldout.sh}"
export AMAP_HM3D PYTHONPATH=.
mkdir -p out; F=out/replay.failed; rm -f "$F"
for p in llm b1 b0 b2; do   # one process per policy (a slice of the frozen grid); replay never calls the API
  { env -u OPENAI_API_KEY "$PY" -m amap.run --config configs/heldout_m.json --policies $p > out/replay_$p.log 2>&1 || echo $p >> "$F"; } &
done; wait
[ -e "$F" ] && { echo "replay failed for: $(tr '\n' ' ' < "$F") (see out/replay_<policy>.log)"; exit 1; }
"$PY" -m eval.compare_replay --replay out/heldout_m/replay --results results/heldout_batch1_floor.json
