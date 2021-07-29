
import json
import cv2
import scipy.io as sio
import numpy as np
import scripts.config as config
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import os
import sys
from PIL import Image
from torchvision.datasets import LSUN
import numpy.random as rand

DATASET_DIR = "/home/matthew/datasets/ShapeNet256/blenderRenderPreprocess"
SUN_DIR = "/home/matthew/datasets/SUN397/"
CATEGORIES = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train", "tvmonitor"]
CATEGORY_CODES = ["02691156", "02834778", "04530566", "02876657", "02924116", "02958343", "03001627", "04379243", "03790512", "04256520", "04468005", "03211117"]

LSUN_DATASET = None

class ShapeNetDataset(Dataset):

    def __init__(self, size, cat_ls=None):
        self.size = size
        self.data = []
        self.labels = []
        for cat in (CATEGORY_CODES if cat_ls is None else [CATEGORY_CODES[CATEGORIES.index(name)] for name in cat_ls]):
            self.append_samples(cat)
        self.sun = []
        with open(os.path.join(SUN_DIR,"ClassName.txt"), "r") as f:
            groups = f.readlines()
            for g in groups:
                g = g.replace("\n", "")[1:] # Remove leading forward slash to prevent it being treated as absolute path
                img_list = [os.path.join(SUN_DIR,g,img_name) for img_name in os.listdir(os.path.join(SUN_DIR,g))]
                self.sun += img_list


    def append_samples(self, cat):
        path = os.path.join(DATASET_DIR, cat)
        img_list = os.listdir(path)
        for img in img_list:
            f = open(os.path.join(path, img, "view.txt"), "r")
            for i, ann_str in enumerate(f.readlines()):
                self.data.append((cat, os.path.join(path, img, "render_{}.png".format(i))))
                ann = [float(a) for a in ann_str.split(" ")]
                el = ann[1]
                az = 360+ann[0] if ann[0] < 0 else ann[0]
                coords = np.array(as_cartesian([1, el, az]))
                self.labels.append(coords)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cat, img_name = self.data[idx]
        obj_img = Image.open(img_name).convert('RGBA')
        r_idx = int(rand.uniform() * len(self.sun))
        img_path = self.sun[r_idx]
        back_img = Image.open(img_path).convert('RGBA')
        transform = transforms.Compose([
            transforms.Resize(self.size),
        ])
        t_obj_img = transform(obj_img)
        t_back_img = transform(back_img)

        t_back_img.paste(t_obj_img, (0,0), t_obj_img)
        t_back_img = t_back_img.convert("RGB")

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        t_img = transform(t_back_img)
        #save_image(t_img, "imgs/{}.png".format(img_name.replace("/", "_")))
        ''''
        obj_img = Image.open(img_name).convert('RGBA')
        transform = transforms.Compose([
            transforms.Resize(self.size),
        ])
        t_obj_img = transform(obj_img)

        r_idx = int(rand.uniform()*len(self.sun))
        img_path = self.sun[r_idx]
        print(img_path)
        back_img = Image.open(img_path).convert('RGBA')
        t_back_img = transform(back_img)

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                alpha = t_obj_img.getpixel((i,j))[3]
                a = np.array(t_obj_img.getpixel((i,j))[:3])
                t_obj_img.putpixel((i,j),tuple([int(np.round(x)) for x in (alpha*np.array(t_obj_img.getpixel((i,j))[:3]) + (1-alpha)*np.array(t_back_img.getpixel((i,j)))).tolist()]))

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        t_img = transform(t_back_img)
        '''
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
    rtp = distance_elevation_azimuth(np.array([x,y,z]))
    if np.abs(rthetaphi[0]-rtp[0]) > 0.005:
        print("BIG R PROBLEM")
    if np.abs(rthetaphi[1] - rtp[1]) > 0.005:
        print("BIG THETA PROBLEM")
    if np.abs(rthetaphi[2] - rtp[2]) > 0.005:
        print("BIG PHI PROBLEM")

    return [x,y,z]

