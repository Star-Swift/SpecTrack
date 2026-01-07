from pathlib import Path
import numpy as np
from PIL import Image
import sys
from tqdm import tqdm

from lib.train.data.image_loader import hsi_loader, msi_loader, must_loader

# data_path = Path("/root/autodl-tmp/hot2022/train")

# png_files = list(data_path.glob('*'))

# for png in tqdm(png_files):
#     png_path = png / png.name / "HSI" / "0001.png"
#     im = hsi_loader(png_path)
#     print(im.shape)

# data_path = Path("/root/autodl-tmp/MSITrack/train")

# png_files = list(data_path.glob('*'))

# for png in tqdm(png_files):
#     png_path = list(png.glob('*.mat'))[0]
#     im = msi_loader(png_path)
#     print(im.shape)

# data_path = Path("/root/autodl-tmp/MUSTHSI/train")

# png_files = list(data_path.glob('*'))

# for png in tqdm(png_files):
#     png_path = list(png.glob('*.npy'))[0]
#     im = must_loader(png_path)
#     print(im.shape)