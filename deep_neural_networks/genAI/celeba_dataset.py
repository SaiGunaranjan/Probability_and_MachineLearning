# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 21:14:26 2025

@author: Sai Gunaranjan
"""

from torch.utils.data import Dataset
import torch
import os
from PIL import Image


class CelebADatasetWithAttributes(Dataset):
    def __init__(self, img_dir, attr_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        # Read attribute file
        with open(attr_path, "r") as f:
            lines = f.readlines()

        # Skip the first line (number of images)
        # Second line is attribute names (optional)
        self.attr_names = lines[1].strip().split()

        # Each line: [filename, attr1, attr2, ..., attr40]
        self.data = []
        for line in lines[2:]:
            parts = line.strip().split()
            filename = parts[0]
            attrs = [int(a) for a in parts[1:]]
            # Convert -1 to 0 for binary attributes
            attrs = [(a + 1) // 2 for a in attrs]
            self.data.append((filename, torch.tensor(attrs, dtype=torch.float)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, attrs = self.data[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, attrs



class CelebADataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image