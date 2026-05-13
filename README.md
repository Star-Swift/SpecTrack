# SpecTrack

Compact code release for SpecTrack, a multispectral single object tracking framework with Spectral Prompt Guided Adaptive Experts.

This folder is prepared for GitHub source release only. It intentionally excludes datasets, checkpoints, pretrained weights, logs, tracking results, experiment tables, generated figures, and local caches.

## Repository Layout

    lib/                  Core model, training, testing, datasets, and utility code
    tracking/             Local path setup, training entry, and the three paper-version launchers
    experiments/spectrack/  YAML configuration files for T224, B224, and L224
    install.sh            Environment installation helper
    test.py               Top-level test entry

## Basic Setup

    conda create -n spectrack python=3.8
    conda activate spectrack
    bash install.sh
    export PYTHONPATH=<repo-root>:<existing-pythonpath>

Create local path files before training or evaluation:

    python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .

Place datasets under data/ and pretrained backbones under pretrained/. These directories are ignored by Git.

## Paper Version Scripts

The release keeps three paper-version launchers:

    bash tracking/run_spectrack_t224.sh check
    bash tracking/run_spectrack_t224.sh train
    bash tracking/run_spectrack_b224.sh train
    bash tracking/run_spectrack_l224.sh train

Each script also supports evaluation when a matching checkpoint exists:

    bash tracking/run_spectrack_b224.sh eval

## Release Note

The original project directory contained experiment logs, paper tables, checkpoints, and generated diagnostic artifacts. They are not part of this open-source copy.
