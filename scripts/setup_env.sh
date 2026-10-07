#!/usr/bin/env bash
# Pinned headless habitat-sim environment (Linux x86_64 with an NVIDIA GPU and driver; tested on an AWS g4dn.xlarge,
# Deep Learning Base OSS Nvidia Driver GPU AMI, Ubuntu 22.04). Writes env.sh with PY= (the environment's python).
#   bash scripts/setup_env.sh [install dir, default ./.mf]
set -euo pipefail
MF=$(realpath -m "${1:-.mf}")
curl -fsSL -o /tmp/mf.sh https://github.com/conda-forge/miniforge/releases/download/26.7.2-0/Miniforge3-Linux-x86_64.sh
echo "281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05  /tmp/mf.sh" | sha256sum -c -
bash /tmp/mf.sh -b -p "$MF" > /dev/null
"$MF/bin/conda" create -y -q -n hab -c conda-forge -c aihabitat python=3.9 numpy=1.26.4 pillow pytest \
  "habitat-sim=0.3.3=py3.9_headless_linux_acbe6f4922e68145e401e55c30f9dfea460a3f24"
PY=$MF/envs/hab/bin/python
$PY -c "import habitat_sim, numpy; print('habitat_sim', habitat_sim.__version__, 'numpy', numpy.__version__)"
echo "export PY=$PY" > env.sh
echo "environment ready: source env.sh"
