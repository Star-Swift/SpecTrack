import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np

data_path = Path('/root/autodl-tmp/SUTrack-moe/moeeval')

full_data = []

for pkl_file in tqdm(list(data_path.glob('routing_info_*.pkl'))):
    with open(pkl_file, 'rb') as f:
        routing_info = pickle.load(f)
    full_data.extend(routing_info)

print(f'Total routing entries collected: {len(full_data)}')

# 统计每个块中被选择的专家
import numpy as np
import pandas as pd

# 假设 full_data 已存在
# full_data: list[num_layer][num_block][dict]

rows = []

num_blocks = len(full_data[0])

for i in range(num_blocks):
    full_top_k_indices = []

    for data in full_data:
        top_k_indices = data[i]['top_k_indices'][0]
        full_top_k_indices.extend(list(top_k_indices))

    # 统计专家出现次数
    unique_experts, counts = np.unique(full_top_k_indices, return_counts=True)
    expert_count_dict = dict(zip(unique_experts, counts))

    # 构造一行数据（block + 各专家次数）
    row = {"block": i}
    for expert_id, count in expert_count_dict.items():
        row[f"expert_{expert_id}"] = count

    rows.append(row)

# 转为 DataFrame
df = pd.DataFrame(rows)

# 缺失专家填 0
df = df.fillna(0).astype(int)

# 保存为 Excel
output_path = "expert_selection_statistics.xlsx"
df.to_excel(output_path, index=False)

print(f"统计结果已保存到 {output_path}")


