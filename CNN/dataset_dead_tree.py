#!/usr/bin/env python3
# dataset_dead_tree.py
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 512  # 与 notebook 保持一致

train_tfms = A.Compose([
    # ① 先把短边 resize 到 ≥512，保持长宽比
    A.SmallestMaxSize(max_size=IMG_SIZE),
    # ② 如果仍有一边 <512，用 PadIfNeeded 补零
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE,
                  border_mode=0, value=0, mask_value=0),
    # ③ 再做随机裁剪到 512×512
    A.RandomCrop(height=IMG_SIZE, width=IMG_SIZE),
    # 其他随机增强
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                       rotate_limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    ToTensorV2(transpose_mask=True),
])

val_tfms = A.Compose([
    A.SmallestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE,
                  border_mode=0, value=0, mask_value=0),
    A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE),
    ToTensorV2(transpose_mask=True),
])




class DeadTreeDataset(Dataset):
    def __init__(self, stems, root_dir, aug=None):
        self.stems = stems
        self.root  = Path(root_dir)
        self.aug   = aug

    def __len__(self): return len(self.stems)

    def _load_np(self, stem):
        rgb = np.array(Image.open(self.root/"RGB_images"/f"RGB_{stem}.png").convert("RGB"))
        nrg = np.array(Image.open(self.root/"NRG_images"/f"NRG_{stem}.png").convert("RGB"))
        nir = nrg[...,0:1]
        img = np.concatenate([nir, rgb], axis=-1)
        mask = np.array(Image.open(self.root/"masks"/f"mask_{stem}.png").convert("L"))
        mask = (mask > 127).astype(np.uint8)
        return img, mask

    def __getitem__(self, idx):
        img, mask = self._load_np(self.stems[idx])
        if self.aug:
            out = self.aug(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        img = img.float()/255.0           # (4,H,W)
        mask = mask.unsqueeze(0).float()  # (1,H,W)
        return img, mask
