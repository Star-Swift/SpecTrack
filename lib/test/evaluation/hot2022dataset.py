import os
import csv
import numpy as np
from pathlib import Path
import re
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList


class HOT2022Dataset(BaseDataset):
    """HOT2022 dataset loader for evaluation."""
    def __init__(self, split='test'):
        super().__init__()
        base_dir = getattr(self.env_settings, 'hot2022_path', None) or getattr(self.env_settings, 'hot2022_dir', None)
        if base_dir is None:
            raise ValueError("HOT2022 path is not set in env settings.")
        self.split = split
        self.base_path = os.path.join(base_dir, split) if split else base_dir
        self.sequence_list = self._get_sequence_list()

    def _get_sequence_list(self):
        test_path = Path(self.base_path)
        dir_list = [p.name for p in test_path.iterdir() if p.is_dir()]
        return dir_list

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def __len__(self):
        return len(self.sequence_list)

    def _construct_sequence(self, sequence_name):
        seq_path = os.path.join(self.base_path, sequence_name, sequence_name, "HSI")
        anno_path = os.path.join(seq_path, "groundtruth_rect.txt")
        ground_truth_rect = np.loadtxt(anno_path, dtype=np.float64).reshape(-1, 4)

        seq_path = Path(seq_path)
        frames_list = [p for p in seq_path.iterdir() if p.is_file() and p.suffix == '.png']
        frames_list.sort(key=lambda p: int(re.search(r'\d+', p.stem).group()))
        frames_list = [str(p) for p in frames_list]

        target_visible = np.ones(len(frames_list), dtype=np.bool_)
        obj_class = self._infer_class_name(sequence_name)

        return Sequence(sequence_name, frames_list, 'HOT2022', ground_truth_rect,
                        object_class=obj_class, target_visible=target_visible)

    @staticmethod
    def _infer_class_name(sequence_name):
        parts = sequence_name.split('-')
        return parts[1] if len(parts) > 1 else sequence_name
