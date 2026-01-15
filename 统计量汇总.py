import pandas as pd
import re

def compute_dataset_channel_statistics(parquet_path: str):
    """
    计算数据集级别的各通道统计量
    """
    df = pd.read_parquet(parquet_path)

    channel_stats = []

    # 找所有通道编号
    channel_ids = sorted({
        int(re.findall(r"channel_(\d+)_", col)[0])
        for col in df.columns
        if col.startswith("channel_") and "_mean" in col
    })

    for c in channel_ids:
        mean_col = f"channel_{c}_mean"
        std_col  = f"channel_{c}_std"

        df[f"channel_{c}_x2"] = df[std_col]**2 + df[mean_col]**2

        mean_x2 = df[f"channel_{c}_x2"].mean()

        std = (mean_x2 - df[mean_col].mean() ** 2) ** 0.5

        print(f"Channel {c}: std_mean={df[std_col].mean():.4f}, std={std:.4f}")
            

        stats = {
            "channel": c,
            "mean": df[mean_col].mean(),
            "std":  std,
        }
        channel_stats.append(stats)

    return pd.DataFrame(channel_stats)

if __name__ == "__main__":

    train_stats = compute_dataset_channel_statistics(
        "hot2022_train_channel_stats.parquet"
    )
    test_stats = compute_dataset_channel_statistics(
        "hot2022_test_channel_stats.parquet"
    )

    print("Train channel statistics:")
    print(train_stats)

    print("\nTest channel statistics:")
    print(test_stats)

    train_stats.to_csv("hot2022_train_channel_summary.csv", index=False)
    test_stats.to_csv("hot2022_test_channel_summary.csv", index=False)
    for data in [train_stats["mean"], train_stats["std"], test_stats["mean"], test_stats["std"]]:
        print("="*20)
        for i in range(16):
            print(f"  - {data[i]/255:.6f}")
        
