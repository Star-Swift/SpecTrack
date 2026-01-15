#!/usr/bin/env bash
set -euo pipefail

# 脚本在 tracking/ 下，但 checkpoints/test/test.py 都在仓库根目录下
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PARAM="${PARAM:-sutrack_l224_must_ihmoe}"   # 可通过环境变量或 --param 覆盖，方便 CI/自动化
DATASET="${DATASET:-MUSTHSI}"
GPU="${GPU:-0}"
THREADS="${THREADS:-8}"
FORCE=0
DO_TEST=1
DO_EVAL=1
CKPT_DIR=""
LOG_DATE="$(date +%F)"
# 让日志文件名包含 param，避免多任务并行时日志混淆
LOG_FILE="$REPO_ROOT/testval_${PARAM}_${LOG_DATE}.txt"

usage() {
  cat <<'EOF'
Usage:
  bash testval.sh --param <config_name> [--dataset MUSTHSI] [--gpu 0] [--threads 20] [--force]
                  [--ckpt_dir /path/to/checkpoints] [--test_only|--eval_only]

Notes:
  - 自动遍历 CKPT_DIR/SUTRACK_ep*.pth.tar
  - 通过环境变量 SUTRACK_CHECKPOINT 覆盖权重路径（无需改源码）
  - 结果默认目录：test/tracking_results/sutrack/<param>/epoch_<N>
  - 日志追加写入：testval_YYYY-MM-DD.txt
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --param) PARAM="$2"; shift 2;;
    --dataset|--dataset_name) DATASET="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --force) FORCE=1; shift 1;;
    --ckpt_dir) CKPT_DIR="$2"; shift 2;;
    --test_only) DO_EVAL=0; shift 1;;
    --eval_only) DO_TEST=0; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

if [[ -z "$CKPT_DIR" ]]; then
  CKPT_DIR="$REPO_ROOT/checkpoints/train/sutrack/${PARAM}"
fi

if [[ ! -d "$CKPT_DIR" ]]; then
  echo "[FATAL] CKPT_DIR not found: $CKPT_DIR" | tee -a "$LOG_FILE"
  exit 1
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "[RUN] $(date '+%F %T')" | tee -a "$LOG_FILE"
echo "[CFG] param=$PARAM dataset=$DATASET gpu=$GPU threads=$THREADS ckpt_dir=$CKPT_DIR force=$FORCE test=$DO_TEST eval=$DO_EVAL" | tee -a "$LOG_FILE"

shopt -s nullglob
CKPTS=( "$CKPT_DIR"/SUTRACK_ep*.pth.tar )
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "[FATAL] No checkpoints matched: $CKPT_DIR/SUTRACK_ep*.pth.tar" | tee -a "$LOG_FILE"
  exit 1
fi

for ckpt in "${CKPTS[@]}"; do
  bn="$(basename "$ckpt")"
  ep="$(echo "$bn" | sed -n 's/.*_ep\([0-9]\+\)\.pth\.tar/\1/p')"
  if [[ -z "$ep" ]]; then
    echo "[WARN] Skip unrecognized ckpt name: $bn" | tee -a "$LOG_FILE"
    continue
  fi

  # 去掉前导 0，避免 bash 当成八进制
  ep_num=$((10#$ep))
  results_dir="$REPO_ROOT/test/tracking_results/sutrack/${PARAM}/epoch_${ep_num}"

  # 如果不是强制模式，且结果目录存在且非空
  if [[ $FORCE -eq 0 && -d "$results_dir" && -n "$(ls -A "$results_dir" 2>/dev/null || true)" ]]; then
    # 如果是 eval_only 模式，即使结果存在也要继续（为了跑下面的 eval）
    if [[ $DO_TEST -eq 1 ]]; then
      echo "[SKIP] epoch_${ep_num} results exist: $results_dir" | tee -a "$LOG_FILE"
      continue
    fi
  fi

  echo "------------------------------------------------------------" | tee -a "$LOG_FILE"
  echo "[CKPT] $ckpt" | tee -a "$LOG_FILE"
  echo "[EPOCH] $ep_num" | tee -a "$LOG_FILE"

  if [[ $DO_TEST -eq 1 ]]; then
    export SUTRACK_CHECKPOINT="$ckpt"
    echo "[TEST] CUDA_VISIBLE_DEVICES=$GPU python test.py sutrack $PARAM --dataset_name $DATASET --threads $THREADS --num_gpus 1 --epoch $ep_num" | tee -a "$LOG_FILE"
    CUDA_VISIBLE_DEVICES="$GPU" python test.py sutrack "$PARAM" \
      --dataset_name "$DATASET" --threads "$THREADS" --num_gpus 1 --epoch "$ep_num" 2>&1 | tee -a "$LOG_FILE"
  fi

  if [[ $DO_EVAL -eq 1 ]]; then
    echo "[EVAL] python tracking/analysis_results.py --dataset_name $DATASET --tracker_param $PARAM --results_dir $results_dir --epoch $ep_num" | tee -a "$LOG_FILE"
    python tracking/analysis_results.py --dataset_name "$DATASET" --tracker_param "$PARAM" --results_dir "$results_dir" --epoch "$ep_num" --force_evaluation 1 2>&1 | tee -a "$LOG_FILE"
  fi
done

echo "[DONE] Log: $LOG_FILE" | tee -a "$LOG_FILE"
