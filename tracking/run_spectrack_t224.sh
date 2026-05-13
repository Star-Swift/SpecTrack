#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CONFIG="${CONFIG:-spectrack_t224_must_ihmoe}"
DATASET="${DATASET:-MUSTHSI}"
GPU="${GPU:-0}"
THREADS="${THREADS:-4}"
EPOCH="${EPOCH:-50}"
MODE="${1:-train}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS="${SPECTRACK_OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${SPECTRACK_MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${SPECTRACK_OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${SPECTRACK_NUMEXPR_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

check_setup() {
  CONFIG_TO_CHECK="${CONFIG}" "${PYTHON}" - <<'PY'
import os
from pathlib import Path
from lib.train.admin.environment import env_settings as train_env
from lib.test.evaluation.environment import env_settings as test_env

config = os.environ["CONFIG_TO_CHECK"]
cfg_file = Path("experiments/spectrack") / f"{config}.yaml"
tr = train_env()
te = test_env()
checks = {
    "config": cfg_file.is_file(),
    "musthsi_train": Path(tr.musthsi_dir, "train", "list.txt").is_file(),
    "musthsi_test": Path(te.musthsi_path, "test", "list.txt").is_file(),
    "pretrained_tiny": Path("pretrained/itpn/fast_itpn_tiny_1600e_1k.pt").is_file(),
}
for name, ok in checks.items():
    print(f"{name}: {ok}")
if not all(checks.values()):
    raise SystemExit(1)
PY
}

require_cuda() {
  "${PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; train/eval needs a CUDA GPU.")
print("cuda:", torch.cuda.get_device_name(0))
PY
}

case "${MODE}" in
  check)
    check_setup
    ;;
  train)
    require_cuda
    "${PYTHON}" tracking/train.py --script spectrack --config "${CONFIG}" --save_dir . --mode single
    ;;
  eval)
    require_cuda
    "${PYTHON}" test.py spectrack "${CONFIG}" --dataset_name "${DATASET}" --threads "${THREADS}" --num_gpus 1 --epoch "${EPOCH}"
    ;;
  *)
    echo "Usage: bash tracking/run_spectrack_t224.sh [check|train|eval]" >&2
    exit 2
    ;;
esac
