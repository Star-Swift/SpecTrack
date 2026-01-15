from pathlib import Path
from typing import List

def collect_hsi_images(root_path: Path) -> List[Path]:
    img_list = []
    for seq_path in root_path.iterdir():
        if not seq_path.is_dir():
            continue
        hsi_path = seq_path / seq_path.name / "HSI"
        if not hsi_path.exists():
            continue
        for img_path in hsi_path.iterdir():
            if img_path.suffix.lower() == ".png":
                img_list.append(img_path)
    return img_list

import numpy as np
from lib.train.data.image_loader import hsi_loader

def compute_channel_stats(img_path):
    img_arr = hsi_loader(img_path)
    stats = {}
    for c in range(img_arr.shape[2]):
        channel = img_arr[:, :, c]
        stats[f"channel_{c}_mean"] = float(channel.mean())
        stats[f"channel_{c}_std"] = float(channel.std())
    return stats

import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def compute_dataset_channel_stats(
    img_list,
    num_workers: int = None
):
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for stats in tqdm(
            executor.map(compute_channel_stats, img_list),
            total=len(img_list),
            desc="Computing channel stats"
        ):
            results.append(stats)
    return pd.DataFrame(results)

if __name__ == "__main__":
    from pathlib import Path

    hot2022_path = Path("/root/autodl-tmp/hot2022")
    train_path = hot2022_path / "train"
    test_path = hot2022_path / "test"

    # -------- train --------
    train_imgs = collect_hsi_images(train_path)
    train_stats = compute_dataset_channel_stats(
        train_imgs,
        num_workers=20  # ⚠️ 建议 = CPU 核心数或略小
    )
    train_stats.to_parquet("hot2022_train_channel_stats.parquet", index=False)

    # -------- test --------
    test_imgs = collect_hsi_images(test_path)
    test_stats = compute_dataset_channel_stats(
        test_imgs,
        num_workers=20  # ⚠️ 建议 = CPU 核心数或略小
    )
    test_stats.to_parquet("hot2022_test_channel_stats.parquet", index=False)
