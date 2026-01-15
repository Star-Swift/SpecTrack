import os
import csv
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList


class MUSTHSIDataset(BaseDataset):
    """MUSTHSI dataset loader for evaluation."""
    def __init__(self, split='test'):
        super().__init__()
        base_dir = getattr(self.env_settings, 'musthsi_path', None) or getattr(self.env_settings, 'musthsi_dir', None)
        if base_dir is None:
            raise ValueError("MUSTHSI path is not set in env settings.")
        self.split = split
        self.base_path = os.path.join(base_dir, split) if split else base_dir
        self.sequence_list = self._get_sequence_list()

    def _get_sequence_list(self):
        list_file = os.path.join(self.base_path, 'list.txt')
        with open(list_file, newline='') as f:
            return [row[0] for row in csv.reader(f) if row]

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def __len__(self):
        return len(self.sequence_list)

    def _construct_sequence(self, sequence_name):
        seq_path = os.path.join(self.base_path, sequence_name)
        anno_path = os.path.join(seq_path, "groundtruth.txt")
        ground_truth_rect = np.loadtxt(anno_path, delimiter=',', dtype=np.float64).reshape(-1, 4)

        frame_names = self._load_frame_names(seq_path, len(ground_truth_rect))
        frames_list = [os.path.join(seq_path, f"{name}.npy") for name in frame_names]

        target_visible = np.ones(len(frames_list), dtype=np.bool_)
        obj_class = self._infer_class_name(sequence_name)

        return Sequence(sequence_name, frames_list, 'musthsi', ground_truth_rect,
                        object_class=obj_class, target_visible=target_visible)

    def _load_frame_names(self, seq_path, expected_len):
        occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
        if os.path.isfile(occlusion_file):
            with open(occlusion_file) as f:
                names = [line.strip().split(',')[0] for line in f if line.strip()]
            if len(names) == expected_len:
                return names
        # fallback to numeric naming if full_occlusion missing/mismatched
        return [f"{i:06d}" for i in range(expected_len)]

    @staticmethod
    def _infer_class_name(sequence_name):
        parts = sequence_name.split('-')
        return parts[1] if len(parts) > 1 else sequence_name
