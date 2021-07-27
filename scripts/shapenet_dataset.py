
import json
import cv2
import scipy.io as sio
import numpy as np
import scripts.config as config
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import sys
from PIL import Image

DATASET_DIR = "/home/matthew/datasets/ShapeNet256/blenderRenderPreprocess"
CATEGORIES = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]
CATEGORY_CODES = ["02691156", "02834778", "04530566", "02876657", "02924116", "02958343", "03001627", "04379243", "03790512", "04256520", "04468005", "03211117"]


class ShapeNetDataset(Dataset):

    def __init__(self, size, cat_ls=None):
        self.size = size
        self.data = []
        self.labels = []
        for cat in (CATEGORY_CODES if cat_ls is None else [CATEGORY_CODES[CATEGORIES.index(name)] for name in cat_ls]):
            self.append_samples(cat)

    def append_samples(self, cat):
        path = os.path.join(DATASET_DIR, cat)
        img_list = os.listdir(path)
        for img in img_list:
            f = open(os.path.join(path, img, "view.txt"), "r")
            for i, ann_str in enumerate(f.readlines()):
                self.data.append((cat, os.path.join(path, img, "render_{}.png".format(i))))
                ann = [float(a) for a in ann_str.split(" ")]
                ann[0] = 360+ann[0] if ann[0] < 0 else ann[0]
                self.labels.append(np.array(as_cartesian([1, ann[1], ann[0]])))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cat, img_name = self.data[idx]
        img = Image.open(img_name).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor()
        ])
        return transform(img), torch.from_numpy(self.labels[idx]).float()


def as_cartesian(rthetaphi):
    r = 1
    theta = np.deg2rad(90-rthetaphi[1])
    phi = np.deg2rad(rthetaphi[2])
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )
    return [x,y,z]

