from pathlib import Path

import os
import cv2
import numpy as np
from typing import List, Tuple
import re
from pathlib import Path
import sys
from lib.train.data.image_loader import msi_loader

def merge_videos_side_by_side(video1_path, video2_path, output_path):
    """
    将两个视频左右合并
    
    参数:
        video1_path: 第一个视频路径
        video2_path: 第二个视频路径  
        output_path: 输出视频路径
    """
    # 打开两个视频文件
    cap1 = cv2.VideoCapture(str(video1_path))
    cap2 = cv2.VideoCapture(str(video2_path))
    
    # 检查视频是否成功打开
    if not cap1.isOpened() or not cap2.isOpened():
        print("错误: 无法打开视频文件")
        return None
    
    # 获取视频属性[1](@ref)
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 使用第一个视频的帧率[1](@ref)
    fps = fps1
    # 计算合并后的分辨率
    total_width = width1 + width2
    total_height = max(height1, height2)
    
    # 创建视频写入对象[1](@ref)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (total_width, total_height))
    
    print(f"开始合并视频: {Path(video1_path).name} 和 {Path(video2_path).name}")
    print(f"输出分辨率: {total_width}x{total_height}, 帧率: {fps}fps")
    
    frame_count = 0
    while True:
        # 读取两个视频的帧[1](@ref)
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        # 如果任一视频结束则退出循环
        if not ret1 or not ret2:
            break
        
        # 调整帧高度保持一致[7](@ref)
        if height1 != total_height:
            frame1 = cv2.resize(frame1, (width1, total_height))
        if height2 != total_height:
            frame2 = cv2.resize(frame2, (width2, total_height))
        
        # 水平拼接帧[1,6](@ref)
        merged_frame = np.hstack([frame1, frame2])
        
        # 写入合并后的帧
        out.write(merged_frame)
        frame_count += 1
        
        # 每处理50帧显示进度
        if frame_count % 50 == 0:
            print(f"已处理 {frame_count} 帧...")
    
    # 释放资源[1](@ref)
    cap1.release()
    cap2.release()
    out.release()
    
    print(f"视频合并完成: {output_path}")
    print(f"总帧数: {frame_count}")
    
    return output_path

def create_video_from_images(
    image_folder: str|Path, 
    output_video_path: str|Path, 
    data: List[Tuple[np.ndarray, str]] | None, 
    fps: int = 30
) -> None:
    """
    从图片文件夹创建视频，并在每帧上绘制矩形框

    参数:
        image_folder: 包含图片的文件夹路径
        output_video_path: 输出视频文件路径
        data: 列表，每个元素是元组 (bboxes, color)
              bboxes: np.ndarray，通常形状为 (frame_count, 4)，每行对应一帧的一个框。
                      也可为一维长度>=4，表示对所有帧使用同一 bbox。
              color: str, 十六进制颜色值，如 "#FF0000"
              或者 data 可以为 None（表示不绘制任何框）
        fps: 视频帧率，默认30
    """
    
    # 1. 获取所有图片文件并按数字顺序排序
    image_files = []
    for file in os.listdir(image_folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
            image_files.append(file)
    
    if not image_files:
        print(f"错误: 文件夹 {image_folder} 中没有找到图片文件")
        return
    
    # 按文件名中的数字排序
    def extract_number(filename):
        # 从文件名中提取所有数字
        numbers = re.findall(r'\d+', filename)
        return int(numbers[0]) if numbers else 0
    
    image_files.sort(key=extract_number)
    
    # 2. 读取第一张图片获取视频尺寸
    first_image_path = os.path.join(image_folder, image_files[0])
    first_frame = cv2.imread(first_image_path)
    if first_frame is None:
        print(f"错误: 无法读取图片 {first_image_path}")
        return
    
    height, width = first_frame.shape[:2]
    frame_count = len(image_files)
    
    # 说明: data 为若干轨迹，每个轨迹的 ndarray 的每一行对应视频的一帧。
    # 如果某轨迹只有一行（shape 一维或 1x4），则视为对所有帧使用同一 bbox。
    
    # 3. 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4格式
    # 或者使用其他编码器：
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # avi格式
    # fourcc = cv2.VideoWriter_fourcc(*'H264')  # h264编码
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"错误: 无法创建视频文件 {output_video_path}")
        return
    
    # 将十六进制颜色转换为BGR格式（独立函数）
    def hex_to_bgr(hex_color):
        hex_color = str(hex_color).lstrip('#')
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (b, g, r)  # OpenCV使用BGR顺序
        else:
            return (255, 255, 255)  # 默认白色
    
    # 5. 处理每一张图片：按帧从每条轨迹取对应行并绘制
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_folder, img_file)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"警告: 无法读取图片 {img_file}，跳过")
            # 使用前一帧或空白帧
            if idx > 0:
                frame = last_frame.copy()
            else:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            last_frame = frame.copy()
        
        # 如果 data 为 None，则不绘制任何框
        frame_items = []  # list of (bbox(4,), color_str)
        if data is not None:
            for arr, color_str in data:
                if arr is None:
                    continue
                arr_np = np.asarray(arr)
                if arr_np.size == 0:
                    continue
                # 1D 情况：单个 bbox，重复到所有帧
                if arr_np.ndim == 1:
                    if arr_np.shape[0] >= 4:
                        bbox = arr_np[:4]
                        frame_items.append((bbox, color_str))
                    else:
                        continue
                # 2D 情况：每行对应一帧
                elif arr_np.ndim == 2:
                    if idx < arr_np.shape[0]:
                        row = arr_np[idx]
                        # 跳过全 NaN 行或不足 4 列的行
                        if np.isnan(row).all():
                            continue
                        if row.size >= 4:
                            bbox = row[:4]
                            frame_items.append((bbox, color_str))
                        else:
                            continue
                    else:
                        # 轨迹在此帧没有数据
                        continue
                else:
                    # 非预期维度，跳过
                    continue
        
        # 绘制所有边界框（如果有）
        for bbox, color_str in frame_items:
            x1, y1, x2, y2 = bbox[:4]
            
            color_bgr = hex_to_bgr(color_str)
            # 绘制矩形
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_bgr, 2)
        
        # 写入视频帧
        out.write(frame)
        
        # 显示进度
        if (idx + 1) % 10 == 0 or idx + 1 == frame_count:
            print(f"进度: {idx + 1}/{frame_count} ({100*(idx+1)/frame_count:.1f}%)")
    
    # 6. 释放资源
    out.release()
    print(f"视频已保存到: {output_video_path}")
    print(f"视频信息: {width}x{height}, {frame_count}帧, {fps}fps")
    
    return output_video_path

if __name__ == "__main__":
    if len(sys.argv) == 4:
        result_path = Path(sys.argv[1])
        dataset_path = Path(sys.argv[2])
        dataset_name = sys.argv[3]
    else:
        result1_path = Path("/root/autodl-tmp/SUTrack-moe/test/tracking_results/sutrack/sutrack_b224_msi_ihmoe/epoch_30")
        result2_path = Path("/root/autodl-tmp/SUTrack-moe/test/tracking_results/sutrack/sutrack_b224_msi_ihmoe/epoch_40")
        # result3_path = Path("/workspace/SUTrack-main/RGBE_workspace/results/VisEvent/sutrack_sutrack_l224")
        # result4_path = Path("/workspace/SUTrack-main/RGBE_workspace/results/VisEvent/sutrack_sutrack_b384")
        # result5_path = Path("/workspace/SUTrack-main/RGBE_workspace/results/VisEvent/sutrack_sutrack_l384")
        dataset_path = Path("/root/autodl-tmp/SUTrack-moe/data/MSITrack/test")
        dataset_name = "MSITrack"
        output_video_path = Path("video/MSITrack")
        output_video_path.mkdir(parents=True, exist_ok=True)
        temp_path = Path("video/MSITrack/temp")
        temp_path.mkdir(parents=True, exist_ok=True)
    
    k_random = np.random.choice(np.arange(1, 500), size=5, replace=False)
    # 创建有框视频
    k = 0
    for result in result1_path.iterdir():
        if k+1 not in k_random:
            k += 1
            continue
        odata_path = dataset_path / result.stem / "groundtruth.txt"
        infrared_path = dataset_path / result.stem 
        # rgb_path = dataset_path / result.stem / "vis_imgs"

        infrared_video_path = output_video_path / "temp"/ f"{result.stem}_infrared.mp4"
        rgb_video_path = output_video_path / "temp" /f"{result.stem}_rgb.mp4"
        merge_video_path = output_video_path / f"{result.stem}_box.mp4"

        # 其他模型结果
        result1 = result1_path / (result.stem + ".txt")
        result2 = result2_path / (result.stem + ".txt")
        # result3 = result3_path / (result.stem + ".txt")
        # result4 = result4_path / (result.stem + ".txt")
        # result5 = result5_path / (result.stem + ".txt")
        # if not result2.exists() or not result3.exists() or not result4.exists(): # or not result5.exists():
        #     k_random = k_random[k_random != (k+1)]
        #     continue

        # 获取视频
        box_data = [(np.genfromtxt(odata_path, delimiter=','), "#FF0000"), # 真值-红色
                    (np.genfromtxt(result1, delimiter=','), "#FFFF00"),    # t224-黄色
                    (np.genfromtxt(result2, delimiter=','), "#00FF00"),]   # b224-绿色
                    # (np.genfromtxt(result3, delimiter=','), "#0000FF"),   # l224-蓝色
                    # (np.genfromtxt(result4, delimiter=','), "#00FFFF"),   # b384-青色
                    # (np.genfromtxt(result5, delimiter=','), "#FF00FF")]   # l384-品红色
        if dataset_name in ["MSITrack"]:
            for data_item in box_data:
                for line in data_item[0]:
                    line[2] = line[2] + line[0]
                    line[3] = line[3] + line[1]
        create_video_from_images(image_folder=infrared_path, output_video_path=infrared_video_path, data=box_data, fps=30)
        create_video_from_images(image_folder=rgb_path, output_video_path=rgb_video_path, data=box_data, fps=30)
        merge_video_path = merge_videos_side_by_side(rgb_video_path, infrared_video_path, merge_video_path)

        k += 1
        # if k >= 10:
        #     break


    # 创建无框视频
    k = 0
    for result in result_path.iterdir():
        if k+1 not in k_random:
            k += 1
            continue
        odata_path = dataset_path / result.stem / "groundtruth.txt"
        infrared_path = dataset_path / result.stem / "event_imgs"
        rgb_path = dataset_path / result.stem / "vis_imgs"

        infrared_video_path = output_video_path / "temp"/ f"{result.stem}_infrared.mp4"
        rgb_video_path = output_video_path / "temp" /f"{result.stem}_rgb.mp4"
        merge_video_path = output_video_path / f"{result.stem}_ori.mp4"

        # 获取视频
        box_data = None
        create_video_from_images(image_folder=infrared_path, output_video_path=infrared_video_path, data=box_data, fps=30)
        create_video_from_images(image_folder=rgb_path, output_video_path=rgb_video_path, data=box_data, fps=30)
        merge_video_path = merge_videos_side_by_side(rgb_video_path, infrared_video_path, merge_video_path)

        k += 1
        # if k >= 10:
        #     break
