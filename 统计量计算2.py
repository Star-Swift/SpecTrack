import argparse
import importlib
import torch
from typing import Tuple, List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread
from pathlib import Path
import os
import csv

# 原始默认导入
from lib.train.dataset.hot2022 import HOT2022
from lib.train.data.image_loader import hsi_loader

def dynamic_import(path: str):
	"""导入形如 package.module.Class 或 package.module.attr 的字符串，返回对象"""
	module_path, _, attr = path.rpartition('.')
	if not module_path:
		raise ImportError(f"invalid import path: {path}")
	mod = importlib.import_module(module_path)
	return getattr(mod, attr)

def get_frame_path_for_seq(seq_path, frame_id):
	seq_path = Path(seq_path)
	name_list = [p for p in seq_path.iterdir() if p.is_file() and p.suffix == '.png']
	name_list.sort(key=lambda p: int(re.search(r'\d+', p.stem).group()))
	return os.path.join(seq_path, name_list[frame_id])

def batch_to_tensor_chw(imgs: List, device: torch.device, num_channels: int) -> torch.Tensor:
	"""批量转换图像到 tensor，返回 (N, C, H, W)"""
	valid_imgs = []
	for img in imgs:
		if img is None:
			continue
		try:
			t = torch.as_tensor(img, dtype=torch.float64)
			if t.ndim == 3 and t.shape[-1] <= 32 and t.shape[0] > t.shape[-1]:
				t = t.permute(2, 0, 1)  # HWC -> CHW
			elif t.ndim == 2:
				t = t.unsqueeze(0)
			c, h, w = t.shape
			if c != num_channels:
				if c < num_channels:
					pad = torch.zeros((num_channels - c, h, w), dtype=t.dtype)
					t = torch.cat([t, pad], dim=0)
				else:
					t = t[:num_channels, :, :]
			valid_imgs.append(t)
		except:
			continue
	if not valid_imgs:
		return None
	batch = torch.stack(valid_imgs, dim=0).to(device)
	return batch

def compute_stats_streaming(dataset, num_channels: int, device: torch.device, 
                            num_workers: int = 32, batch_size: int = 64, 
                            prefetch_batches: int = 3) -> Tuple[torch.Tensor, torch.Tensor, int]:
	"""
	流式处理：使用生产者-消费者模式，边读边算，限制内存占用。
	prefetch_batches: 预读取的批次数，控制内存占用
	"""
	sums = torch.zeros(num_channels, dtype=torch.float64, device=device)
	sumsqs = torch.zeros(num_channels, dtype=torch.float64, device=device)
	total_pixels = 0
	skipped = 0

	num_seqs = len(dataset.sequence_list)
	print(f"共 {num_seqs} 个序列, 使用 {num_workers} 个工作线程, 预读取 {prefetch_batches} 批")

	# 收集所有任务
	all_tasks = []
	for seq_id in range(num_seqs):
		seq_path = dataset._get_sequence_path(seq_id)
		seq_info = dataset.get_sequence_info(seq_id)
		num_frames = seq_info['bbox'].shape[0]
		for frame_id in range(num_frames):
			all_tasks.append((seq_path, frame_id))

	total_frames = len(all_tasks)
	print(f"总共 {total_frames} 帧待处理")

	image_loader = dataset.image_loader

	# 数据队列，限制大小以控制内存
	data_queue = Queue(maxsize=prefetch_batches)
	
	# 分批任务列表
	batch_tasks_list = []
	for batch_start in range(0, total_frames, batch_size):
		batch_end = min(batch_start + batch_size, total_frames)
		batch_tasks_list.append(all_tasks[batch_start:batch_end])

	def load_frame(task):
		seq_path, frame_id = task
		try:
			frame_path = get_frame_path_for_seq(seq_path, frame_id)
			return image_loader(frame_path)
		except:
			return None

	def producer():
		"""生产者：多线程读取数据放入队列"""
		with ThreadPoolExecutor(max_workers=num_workers) as executor:
			for batch_tasks in batch_tasks_list:
				imgs = list(executor.map(load_frame, batch_tasks))
				data_queue.put((imgs, len(batch_tasks)))
		# 发送结束信号
		data_queue.put(None)

	# 启动生产者线程
	producer_thread = Thread(target=producer, daemon=True)
	producer_thread.start()

	# 消费者：主线程处理数据
	pbar = tqdm(total=len(batch_tasks_list), desc="批次处理")
	while True:
		item = data_queue.get()
		if item is None:
			break
		imgs, batch_len = item
		
		batch_tensor = batch_to_tensor_chw(imgs, device, num_channels)
		# 立即释放原始数据
		del imgs
		
		if batch_tensor is not None:
			n, c, h, w = batch_tensor.shape
			pixels_per_img = h * w
			flatten = batch_tensor.reshape(n, c, -1)
			sums += flatten.sum(dim=(0, 2))
			sumsqs += (flatten * flatten).sum(dim=(0, 2))
			total_pixels += n * pixels_per_img
			del flatten, batch_tensor
		else:
			skipped += batch_len
		
		pbar.update(1)
	
	pbar.close()
	producer_thread.join()

	if skipped > 0:
		print(f"跳过了 {skipped} 个无效帧")

	if total_pixels == 0:
		return torch.zeros(num_channels), torch.zeros(num_channels), 0

	means = (sums / total_pixels).cpu()
	vars_ = (sumsqs / total_pixels - (sums / total_pixels) ** 2).cpu()
	vars_ = torch.clamp(vars_, min=0)
	stds = torch.sqrt(vars_)
	return means, stds, total_pixels

def format_array(arr: torch.Tensor) -> str:
	return ', '.join([f"{float(x):.6f}" for x in arr.tolist()])

def main():
	parser = argparse.ArgumentParser(description="计算 HOT2022 等多光谱数据集每通道均值与标准差（流式处理版）")
	parser.add_argument('--root', type=str, default='/root/autodl-tmp/hot2022', help='数据集根目录')
	parser.add_argument('--dataset', type=str, default='lib.train.dataset.hot2022.HOT2022', help='数据集类完整路径')
	parser.add_argument('--image_loader', type=str, default='lib.train.data.image_loader.hsi_loader', help='image_loader 导入路径')
	parser.add_argument('--num_channels', type=int, default=8, help='通道数，默认 8')
	parser.add_argument('--split_train', type=str, default='train')
	parser.add_argument('--split_test', type=str, default='test')
	parser.add_argument('--device', type=str, default='cpu', help='计算设备，默认 cpu')
	parser.add_argument('--num_workers', type=int, default=32, help='并行工作线程数')
	parser.add_argument('--batch_size', type=int, default=64, help='每批处理的帧数')
	parser.add_argument('--prefetch', type=int, default=3, help='预读取批次数（控制内存占用）')
	args = parser.parse_args()

	# 动态加载
	try:
		DatasetClass = dynamic_import(args.dataset)
	except Exception as e:
		print(f"加载数据集类失败({e})，使用默认 HOT2022")
		DatasetClass = HOT2022

	try:
		ImageLoader = dynamic_import(args.image_loader)
	except Exception as e:
		print(f"加载 image_loader 失败({e})，使用默认 msi_loader")
		ImageLoader = hsi_loader

	device = torch.device(args.device)
	print(f"使用设备: {device}")
	print(f"并行线程: {args.num_workers}, 批大小: {args.batch_size}, 预读取批次: {args.prefetch}")

	# 实例化数据集
	print(f"\n=== 加载训练集 ({args.split_train}) ===")
	train_ds = DatasetClass(root=args.root, image_loader=ImageLoader, split=args.split_train)
	print(f"=== 加载测试集 ({args.split_test}) ===")
	test_ds = DatasetClass(root=args.root, image_loader=ImageLoader, split=args.split_test)

	# 流式统计
	print("\n=== 统计训练集 ===")
	train_mean, train_std, train_pixels = compute_stats_streaming(
		train_ds, args.num_channels, device, args.num_workers, args.batch_size, args.prefetch)
	
	print("\n=== 统计测试集 ===")
	test_mean, test_std, test_pixels = compute_stats_streaming(
		test_ds, args.num_channels, device, args.num_workers, args.batch_size, args.prefetch)

	# 合并统计量
	train_sums = train_mean.to(device) * train_pixels
	test_sums = test_mean.to(device) * test_pixels
	train_sumsqs = (train_std.to(device) ** 2 + train_mean.to(device) ** 2) * train_pixels
	test_sumsqs = (test_std.to(device) ** 2 + test_mean.to(device) ** 2) * test_pixels

	total_pixels = train_pixels + test_pixels
	if total_pixels == 0:
		print("没有样本，退出。")
		return

	total_sums = train_sums + test_sums
	total_sumsqs = train_sumsqs + test_sumsqs
	total_mean = (total_sums / total_pixels).cpu()
	total_var = (total_sumsqs / total_pixels - (total_sums / total_pixels) ** 2).cpu()
	total_var = torch.clamp(total_var, min=0)
	total_std = torch.sqrt(total_var)

	# 保存结果
	dataset_name = getattr(DatasetClass, '__name__', args.dataset.split('.')[-1])
	out_path = f"{dataset_name}.txt"
	with open(out_path, 'w', encoding='utf-8') as f:
		f.write(f"dataset: {dataset_name}\n")
		f.write(f"root: {args.root}\n")
		f.write(f"num_channels: {args.num_channels}\n")
		f.write(f"train_sequences: {len(train_ds.sequence_list)}, train_pixels: {train_pixels}\n")
		f.write(f"test_sequences: {len(test_ds.sequence_list)}, test_pixels: {test_pixels}\n")
		f.write("\n")
		f.write("train_mean: " + format_array(train_mean) + "\n")
		f.write("train_std:  " + format_array(train_std) + "\n\n")
		f.write("test_mean:  " + format_array(test_mean) + "\n")
		f.write("test_std:   " + format_array(test_std) + "\n\n")
		f.write("all_mean:   " + format_array(total_mean) + "\n")
		f.write("all_std:    " + format_array(total_std) + "\n")

	print(f"\n统计完成，结果已保存到 {out_path}")
	print(f"\ntrain_mean: {format_array(train_mean)}")
	print(f"train_std:  {format_array(train_std)}")
	print(f"test_mean:  {format_array(test_mean)}")
	print(f"test_std:   {format_array(test_std)}")
	print(f"all_mean:   {format_array(total_mean)}")
	print(f"all_std:    {format_array(total_std)}")

if __name__ == '__main__':
	main()