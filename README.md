# SpecTrack

SpecTrack is a multispectral single object tracking framework with Spectral Prompt Guided Adaptive Experts.

This repository is prepared as a compact source release. It keeps the code, the three paper-version configurations, and the three matching launch scripts. Datasets, checkpoints, pretrained backbones, logs, result files, generated tables, and local machine settings are intentionally excluded.

## Repository Layout

```text
experiments/spectrack/
  spectrack_t224_must_ihmoe.yaml
  spectrack_b224_must_ihmoe.yaml
  spectrack_l224_must_ihmoe.yaml

lib/
  config/spectrack/       SpecTrack configuration
  models/spectrack/       SpecTrack model implementation
  train/                  training datasets, actors, trainers, and utilities
  test/                   tracking parameters, trackers, evaluation utilities

tracking/
  run_spectrack_t224.sh   paper T224 launcher
  run_spectrack_b224.sh   paper B224 launcher
  run_spectrack_l224.sh   paper L224 launcher
  train.py                training entry used by the launchers
  create_default_local_file.py

test.py                   evaluation entry
install.sh                dependency helper from the original training setup
```

## Environment

Create a Python environment and install dependencies. `install.sh` keeps the original dependency baseline, including an old `torch==1.11.0+cu113` install command at the top. On newer GPUs, install a PyTorch build that matches your CUDA driver and comment out or update that torch block before running the rest of the helper.

```bash
conda create -n spectrack python=3.8 -y
conda activate spectrack

# Example only. Choose the torch command for your CUDA runtime.
pip install torch torchvision torchaudio

# Then install the remaining dependencies. On newer GPUs, update or skip the
# torch install block at the top of install.sh before running it.
bash install.sh
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

If you already have a working virtual environment in the project as `.venv`, the launch scripts will use it by default. You can override it with `PYTHON=/path/to/python`.

## Data And Weights

Create local path files after placing datasets and pretrained backbones on the machine:

```bash
python tracking/create_default_local_file.py \
  --workspace_dir . \
  --data_dir ./data \
  --save_dir .
```

Expected local structure:

```text
data/
  MUSTHSI/
    train/list.txt
    test/list.txt

pretrained/
  itpn/
    fast_itpn_tiny_1600e_1k.pt
    fast_itpn_base_clipl_e1600.pt
    fast_itpn_large_1600e_1k.pt
```

Generated local path files are ignored by Git:

```text
lib/train/admin/local.py
lib/test/evaluation/local.py
```

## Quick Check

Before training, verify the config, dataset paths, and pretrained files:

```bash
bash tracking/run_spectrack_t224.sh check
bash tracking/run_spectrack_b224.sh check
bash tracking/run_spectrack_l224.sh check
```

The scripts use GPU `0` by default. Override common runtime options through environment variables:

```bash
GPU=1 THREADS=8 bash tracking/run_spectrack_b224.sh eval
PYTHON=/path/to/python bash tracking/run_spectrack_t224.sh train
```

Thread-related variables are also supported:

```bash
SPECTRACK_OMP_NUM_THREADS=4 \
SPECTRACK_MKL_NUM_THREADS=4 \
SPECTRACK_OPENBLAS_NUM_THREADS=4 \
SPECTRACK_NUMEXPR_NUM_THREADS=4 \
bash tracking/run_spectrack_t224.sh train
```

## Training

Run one of the three paper-version scripts:

```bash
bash tracking/run_spectrack_t224.sh train
bash tracking/run_spectrack_b224.sh train
bash tracking/run_spectrack_l224.sh train
```

Checkpoints are saved under:

```text
checkpoints/train/spectrack/<config>/SPECTRACK_epXXXX.pth.tar
```

For example:

```text
checkpoints/train/spectrack/spectrack_b224_must_ihmoe/SPECTRACK_ep0050.pth.tar
```

## Evaluation

Evaluate a trained checkpoint with the matching script:

```bash
EPOCH=50 bash tracking/run_spectrack_b224.sh eval
```

Useful overrides:

```bash
DATASET=MUSTHSI EPOCH=50 THREADS=8 GPU=0 bash tracking/run_spectrack_b224.sh eval
```

If the checkpoint is stored outside the default checkpoint directory, pass it through `SPECTRACK_CHECKPOINT`:

```bash
SPECTRACK_CHECKPOINT=/path/to/SPECTRACK_ep0050.pth.tar \
EPOCH=50 \
bash tracking/run_spectrack_b224.sh eval
```

Results are written under `test/tracking_results/` and are ignored by Git.

## Local Smoke Test

The current code path was checked on a CUDA GPU with:

```bash
python -m compileall -q lib tracking test.py
bash tracking/run_spectrack_t224.sh check
bash tracking/run_spectrack_b224.sh check
bash tracking/run_spectrack_l224.sh check
```

Additional runtime checks completed locally:

```text
T224 model construction on CUDA: passed
T224 one-epoch smoke training: passed
T224 one-sequence MUSTHSI evaluation: passed
```

## Notes

Only the three paper-version launchers and YAML configurations are kept for release. Runtime outputs, local datasets, pretrained weights, and checkpoints are excluded through `.gitignore`.
