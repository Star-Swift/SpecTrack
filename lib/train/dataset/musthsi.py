import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data.image_loader import must_loader
from lib.train.admin import env_settings


class MUSTHSI(BaseVideoDataset):
    def __init__(self, root=None, image_loader=must_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - 数据集路径
            image_loader - 数据读取方式,因为是npy,因此用must_loader
            split - train和test分别对应两个不同的文件夹
            seq_ids - 可以设置id号来选择使用文件夹内指定序列,默认为None,表示使用全部序列
            data_fraction - 一个小于1的数,例如0.8,表示从全部序列中随机选择80%使用
        """
        # 如果root为None，则从env_settings中获取MUSTHSI数据集的路径
        root = env_settings().musthsi_dir if root is None else root
        # 调用父类的构造函数
        super().__init__('MUSTHSI', root, image_loader)

        # 拼接数据集的根目录和split（如 'train' 或 'test'）
        self.root = os.path.join(self.root, split)
        # 获取序列列表
        self.sequence_list = self._get_sequence_list()

        # 如果未指定seq_ids，则使用所有序列
        if seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        # 根据seq_ids筛选序列列表
        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        # 如果指定了data_fraction，则随机采样一部分序列
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def get_name(self):
        # 返回数据集的名称
        return 'musthsi'

    def has_class_info(self):
        # 返回数据集是否包含类别信息
        return True
    
    def has_out_of_view_info(self):
        # 返回数据集是否包含目标出视野信息
        return True

    def has_occlusion_info(self):
        # 返回数据集是否包含遮挡信息
        return True

    def _get_sequence_list(self):
        # 从list.txt文件中读取序列列表
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list
    
    def _read_bb_anno(self, seq_path):
        # 读取边界框标注文件
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        # 使用pandas读取csv格式的标注，并转换为torch.tensor
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        # 根据序列ID获取序列的完整路径
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        # 获取单个序列的信息（边界框、有效性、可见性）
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        # 判断边界框的宽和高是否大于0，来确定是否有效
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        # 将valid转换为byte类型作为visible
        visible = valid.byte()
        
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        # 获取指定帧的图像路径
        # full_occlusion.txt 包含帧的文件名
        with open(os.path.join(seq_path, "full_occlusion.txt")) as f:
            # 直接按行读取，并取逗号前的内容作为文件名
            name_list = [line.strip().split(',')[0] for line in f]
        
        # 拼接成完整的图像路径，生成一个包含 _img1 的基础路径
        name_path = os.path.join(seq_path, name_list[frame_id] + '.npy')
        ## 拼接成完整的图像路径，生成一个包含 _img1 的基础路径
        # name_path = os.path.join(seq_path, 'HSIData', name_list[frame_id] + '_img1.npy')
        
        return name_path

    def _get_frame(self, seq_path, frame_id):
        # 使用指定的图像加载器加载单个帧
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        # 获取一个序列中的多个帧及其标注
        seq_path = self._get_sequence_path(seq_id)
        # 从序列路径中解析出类别名称
        class_name = seq_path.split('/')[-1].split('-')[1]

        # 加载所有请求的帧图像
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        # 如果没有提供标注，则获取该序列的标注
        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # 提取指定帧的标注
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
            
        # 构建目标的元数据
        object_meta = OrderedDict({'object_class_name': class_name,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
