
import json
import cv2
import scipy.io as sio
import numpy as np
import os

import torch
from torch.utils.data import DataLoader, Dataset

DATASET_TRAIN_DIR = "imagenet_128_128_stretched_train/"
DATASET_VAL_DIR = "imagenet_128_128_stretched_val/"
CATEGORIES = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]
PASCAL_DIR = "~/pascal3d-analysis/PASCAL3D+_release1.1/"


class PascalDataset(Dataset):

    def __init__(self, train=True):
        self.data = []
        self.labels = []

        dataset_dir = DATASET_TRAIN_DIR if train else DATASET_VAL_DIR
        annotation_dict = json.load(open(dataset_dir+"annotation.json","r"))
        for cat in CATEGORIES:
            for img_name,ann in annotation_dict[cat].items():
                self.data.append(dataset_dir + cat + "_imagenet/{}.png".format(img_name))
                self.labels.append(np.array(ann))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx])
        img = np.moveaxis(img, -1, 0)
        img = img / 255.0
        out = torch.from_numpy(img).float()
        print(torch.from_numpy(self.labels[idx]).float())
        return out, torch.from_numpy(self.labels[idx]).float()

