
import json
import cv2
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset

DATASET_TRAIN_DIR = "pascal3d_imagenet_512_320_train/"
DATASET_VAL_DIR = "pascal3d_imagenet_512_320_val/"
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
                img = cv2.imread(dataset_dir+cat+"_imagenet/{}.png".format(img_name))
                img = np.moveaxis(img, -1, 0)
                img = img/255.0
                self.data.append(img)
                self.labels.append(np.array(ann))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

