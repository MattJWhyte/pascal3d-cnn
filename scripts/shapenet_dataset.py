
import json
import cv2
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scripts.config as config
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import os
import sys
from PIL import Image
from torchvision.transforms import RandomResizedCrop, ColorJitter
import numpy.random as rand
from scipy.stats import vonmises

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
        self.elevations = []
        self.azimuths = []
        for cat in (CATEGORY_CODES if cat_ls is None else [CATEGORY_CODES[CATEGORIES.index(name)] for name in cat_ls]):
            self.append_samples(cat)
        self.sun = []
        '''
        with open(os.path.join(SUN_DIR,"ClassName.txt"), "r") as f:
            groups = f.readlines()
            for g in groups:
                g = g.replace("\n", "")[1:] # Remove leading forward slash to prevent it being treated as absolute path
                img_list = [os.path.join(SUN_DIR,g,img_name) for img_name in os.listdir(os.path.join(SUN_DIR,g))]
                self.sun += img_list
        if not os.path.exists(os.path.join(SUN_DIR,"temp")):
            os.mkdir(os.path.join(SUN_DIR,"temp"))'''

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
                self.elevations.append(el)
                self.azimuths.append(az)
                self.labels.append(coords)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cat, img_name = self.data[idx]
        if os.path.exists(os.path.join(SUN_DIR,"temp",img_name.replace("/","-"))):
            img = Image.open(os.path.join(SUN_DIR,"temp",img_name.replace("/","-"))).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize(self.size),
                transforms.ToTensor(),
                RandomResizedCrop(self.size, scale=(0.4, 1.0), ratio=(0.66, 1.5)),
                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
            ])
            t_img = transform(img)
            save_image(t_img, "test.png")
        else:
            obj_img = Image.open(img_name).convert('RGBA')
            r_idx = int(rand.uniform() * len(self.sun))
            img_path = self.sun[r_idx]
            back_img = Image.open(img_path).convert('RGBA')
            transform = transforms.Compose([
                transforms.Resize(self.size),
            ])
            t_obj_img = transform(obj_img)
            t_back_img = transform(back_img)

            t_back_img.paste(t_obj_img, (0, 0), t_obj_img)
            t_back_img = t_back_img.convert("RGB")

            t_back_img.save(os.path.join(SUN_DIR,"temp",img_name.replace("/","-")))
            transform = transforms.Compose([
                transforms.ToTensor(),
                RandomResizedCrop(self.size, scale=(0.4, 1.0), ratio=(0.66, 1.5)),
                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
            ])
            t_img = transform(t_back_img)
        return t_img, torch.from_numpy(self.labels[idx]).float()


class VMBiasedShapeNetDataset(ShapeNetDataset):

    def __init__(self, size, kappa, loc, cat_ls=None):
        super().__init__(size, cat_ls)
        self.sort_idx = np.argsort(np.array(self.azimuths))
        sorted_az = [self.azimuths[i] for i in self.sort_idx]
        sample_ls = vonmises.rvs(kappa, loc=loc, size=len(self))
        sample_ls = sample_ls*180.0/np.pi
        sample_ls = (sample_ls + 360) % 360
        self.sample_idx = []
        for s in sample_ls:
            idx = binary_search(sorted_az, s)
            self.sample_idx.append(self.sort_idx[idx])
        a = [np.deg2rad(self.azimuths[idx]) for idx in self.sample_idx]
        f = plt.figure()
        ax = f.add_subplot(projection="polar")
        ax.hist(a)
        plt.savefig("dist.png")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return super().__getitem__(self.sample_idx[idx])


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


def angle_dist(a, b):
    clockwise_angle = np.abs(b-a)
    anticlockwise_angle = np.abs( min(a,b) + (360.0 - max(a,b)) )
    return min(clockwise_angle,anticlockwise_angle)


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



def binary_search(ls, x):
    a = 0
    b = len(ls)
    while b-a > 1:
        i = (a+b)//2
        if ls[i] < x:
            a = i
        elif ls[i] > x:
            b = i
        else:
            return i
    if b == len(ls):
        b = 0
    d_a = angle_dist(ls[a],x)
    d_b = angle_dist(ls[b],x)
    if d_a < d_b:
        return a
    return b


a = VMBiasedShapeNetDataset((224,224), 1.0, np.pi)

