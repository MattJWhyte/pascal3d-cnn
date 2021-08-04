
import json
import cv2
import scipy.io as sio
import numpy as np
import scripts.config as config
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import sys
from PIL import Image

DATASET_TRAIN_DIR = "imagenet_128_128_stretched_train/"
DATASET_VAL_DIR = "imagenet_128_128_stretched_val/"
CATEGORIES = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]


class RawPascalDataset(Dataset):

    def __init__(self, size, train=True, cat_ls=None):
        self.data = []
        self.labels = []
        self.size = size
        self.cat_idx = []
        tag = "train" if train else "val"
        out = ""
        for cat in (CATEGORIES if cat_ls is None else cat_ls):
            with open("{}/Image_sets/{}_imagenet_{}.txt".format(config.PASCAL_DIR, cat, tag), "r") as f:
                for img_name in f.readlines():
                    img_name = img_name.replace("\n", "")
                    ann = get_image_annotations(
                        "{}/Annotations/{}_imagenet/{}.mat".format(config.PASCAL_DIR, cat, img_name), cat)[0]
                    if ann["viewpoint"]["num_anchor"] == 0:
                        continue
                    az = ann["viewpoint"]["azimuth"]
                    el = ann["viewpoint"]["elevation"]
                    bb = ann["bbox"]
                    coords = as_cartesian([1, el, az])
                    self.labels.append(np.array(coords))
                    self.data.append((cat, img_name, bb))
                    out += img_name + "\n"
            self.cat_idx.append(len(self.labels))
        self.cat_idx = np.array(self.cat_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cat,img_name,bb = self.data[idx]
        img = Image.open("{}/Images/{}_imagenet/{}.JPEG".format(config.PASCAL_DIR, cat, img_name)).convert('RGB')
        img = img.crop(bb[0], bb[1], bb[2], bb[3])
        transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor()
        ])
        t_img = transform(img)
        #rthetaphi = "_".join([str(s) for s in self.labels[idx].tolist()])
        #save_image(t_img, "pascal_imgs/{}@{}.png".format(img_name.replace("/", "_"), rthetaphi))
        return t_img, torch.from_numpy(self.labels[idx]).float()


def distance_elevation_azimuth(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    theta = np.abs(90-np.rad2deg(np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))))
    if z < 0:
        theta *= -1.0
    phi = np.rad2deg(np.arctan2(y,x))
    if phi < 0.0:
        phi += 360.0
    return [np.sqrt(x**2+y**2+z**2), theta, phi]


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
                "class": o["class"][0],
                "bbox": o["bbox"][0]
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

