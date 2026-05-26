#!/usr/bin/env bash
# Run inside: screen -S Exp
#   source /mnt/storage/RSystemsBenchmarking/envs/39myenv/bin/activate
#   ./run_ablation_sweep.sh
set -euo pipefail
cd "$(dirname "$0")"
exec /mnt/storage/RSystemsBenchmarking/envs/39myenv/bin/python batch_retrieve_ablation_sweep.py "$@"
