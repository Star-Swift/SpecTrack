import pickle
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def load_pickle_file(pkl_file):
    """加载单个pickle文件"""
    try:
        with open(pkl_file, 'rb') as f:
            routing_info = pickle.load(f)
        return routing_info
    except Exception as e:
        print(f"Error loading {pkl_file}: {e}")
        return []

data_path = Path('/root/autodl-tmp/SUTrack-moe/moeeval')

# 获取所有pickle文件
pkl_files = list(data_path.glob('routing_info_*.pkl'))
print(f"Found {len(pkl_files)} files to process")

full_data = []

# 使用map函数并行处理
with ThreadPoolExecutor(max_workers=8) as executor:
    # 使用map保持顺序，但会等待最慢的任务
    results = list(tqdm(executor.map(load_pickle_file, pkl_files), 
                       total=len(pkl_files), 
                       desc="Loading files"))
    
    # 合并结果
    for routing_info in results:
        full_data.extend(routing_info)

print(f'Total routing entries collected: {len(full_data)}')