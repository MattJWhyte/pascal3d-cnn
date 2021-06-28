
import json
import cv2
import scipy.io as sio
import numpy as np
import scripts.config as config
import torch
from torch.utils.data import DataLoader, Dataset
import sys

DATASET_TRAIN_DIR = "imagenet_128_128_stretched_train/"
DATASET_VAL_DIR = "imagenet_128_128_stretched_val/"
CATEGORIES = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]


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
        print(out.shape)
        return out, torch.from_numpy(self.labels[idx]).float()


class RawPascalDataset(Dataset):

    def __init__(self, size, train=True):
        self.data = []
        self.labels = []
        self.size = size
        tag = "train" if train else "val"
        for cat in CATEGORIES:
            with open("{}/Image_sets/{}_imagenet_{}.txt".format(config.PASCAL_DIR, cat, tag), "r") as f:
                for img_name in f.readlines():
                    img_name = img_name.replace("\n", "")
                    ann = get_image_annotations(
                        "{}/Annotations/{}_imagenet/{}.mat".format(config.PASCAL_DIR, cat, img_name), cat)[0]
                    az = ann["viewpoint"]["azimuth"]
                    el = ann["viewpoint"]["elevation"]
                    coords = as_cartesian([1, el, az])
                    self.labels.append(np.array(coords))
                    self.data.append((cat, img_name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cat,img_name = self.data[idx]
        img = cv2.imread("{}/Images/{}_imagenet/{}.JPEG".format(config.PASCAL_DIR, cat, img_name))
        img = cv2.resize(img, self.size)
        print(img.shape)
        img = np.moveaxis(img, -1, 0)
        cv2.imwrite("test.png", img)
        sys.exit()
        print(img.shape)
        img = img / 255.0
        out = torch.from_numpy(img).float()
        print(out.shape)
        return out, torch.from_numpy(self.labels[idx]).float()


def as_cartesian(rthetaphi):
    r = 1
    theta = np.deg2rad(90-rthetaphi[1])
    phi = np.deg2rad(rthetaphi[2])
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )
    return [x,y,z]


# Get annotation information for all objects of specified category in image
def get_image_annotations(img, category):
    mat = sio.loadmat(img)
    objs = mat["record"][0, 0]["objects"]
    annotation_list = []
    for i in range(objs.shape[1]): # Loop over objects in image
        o = objs[0, i]
        if len(o.dtype) > 0:
            if o["class"][0] != category:
                continue
            annotation_dict = {
                "truncated": o["truncated"][0, 0],
                "occluded": o["occluded"][0, 0],
                "difficult": o["difficult"][0, 0],
                "class": o["class"][0]
            }
            vp = o["viewpoint"]
            if vp.size > 0:
                vp_annotation = {}
                for annotation in vp.dtype.fields.keys():
                    vp_annotation[annotation] = vp[annotation][0, 0][0, 0]
                annotation_dict["viewpoint"] = vp_annotation
                annotation_list.append(annotation_dict)
            else:
                print("WARNING : no viewpoint set for annotation")
                continue
    return annotation_list

