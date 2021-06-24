import os

import numpy as np
from dataset import PascalDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from network import *
import torch

# Python file for all evaluation metrics / graphics of nn against ground truth


def distance_elevation_azimuth(xyz):
    x,y,z = xyz
    theta = np.abs(90-np.rad2deg(np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))))
    if z < 0:
        theta *= -1.0
    phi = np.rad2deg(np.arctan2(y,x))
    if phi < 0.0:
        phi += 360.0
    return [np.sqrt(x**2+y**2+z**2), theta, phi]

# Return angle between vectors
def get_angle(y,target):
    y = y.numpy()
    target = target.detach().numpy()
    y_norm = np.linalg.norm(y, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    norm = y_norm * target_norm.T
    return np.rad2deg(np.arccos(np.diag((y @ target.T)) / norm)).tolist()


# Count number of times the angle between predicted label and target is < 30 degrees
def thirty_deg_accuracy(y, target):
    y = y.numpy()
    target = target.detach().numpy()
    y_norm = np.linalg.norm(y, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    norm = y_norm * target_norm.T
    theta = np.arccos(np.diag((y @ target.T)) / norm)
    size = float(theta.shape[0])
    return np.count_nonzero(theta < np.deg2rad(30.0)) / size


def evaluate_model(pth, net):
    # Load model
    nt = net()
    nt.load(pth)
    nt.eval()
    nt.to('cuda' if torch.cuda.is_available() else "cpu")
    dset = PascalDataset(train=False)
    dataloader = DataLoader(dset, batch_size=48)
    n = len(dset)
    acc = 0.0
    ct = 0.0
    theta = []
    for X, target in dataloader:
        X = X.to('cuda' if torch.cuda.is_available() else "cpu")
        ct += 1
        y = nt(X).detach().cpu()
        k = thirty_deg_accuracy(y, target)
        theta += get_angle(y, target)
        X.detach()
        del X
        torch.cuda.empty_cache()
        acc += k
    return acc/ct, np.median(np.array(theta))


def predict_model(pth, net):
    # Load model
    nt = net()
    nt.load(pth)
    nt.eval()
    nt.to('cuda' if torch.cuda.is_available() else "cpu")
    train_dset = PascalDataset()
    test_dset = PascalDataset(train=False)

    train_pred_az = []
    train_target_az = []
    train_pred_el = []
    train_target_el = []

    if not os.path.exists("results"):
        os.mkdir("results")

    if not os.path.exists("results/predictions"):
        os.mkdir("results/predictions")

    for i in range(100):
        y = nt(train_dset[i][0].unsqueeze(0).to('cuda' if torch.cuda.is_available() else "cpu")).detach().cpu()
        img_name = train_dset.data[i].replace("/", "-")
        target = train_dset[i][1].unsqueeze(0)
        theta = get_angle(y, target)
        y = y.numpy()
        target = target.numpy()
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot([0,y[0,0]], [0,y[0,1]], [0,y[0,2]], "k-", label="Pred")
        ax.plot([0, target[0,0]], [0, target[0,1]], [0, target[0,2]], "r-", label="Target")
        plt.legend()
        plt.title("Angle = {}".format(theta))
        plt.savefig("results/predictions/"+img_name)
        plt.close()

    for i in range(100):
        y = nt(test_dset[i][0].unsqueeze(0).to('cuda' if torch.cuda.is_available() else "cpu")).detach().cpu()
        img_name = test_dset.data[i].replace("/", "-")
        target = test_dset[i][1].unsqueeze(0)
        theta = get_angle(y, target)
        y = y.numpy()
        target = target.numpy()
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot([0,y[0]], [0,y[1]], [0,y[2]], "k-", label="Pred")
        ax.plot([0, target[0]], [0, target[1]], [0, target[2]], "r-", label="Target")
        plt.legend()
        plt.title("Angle = {}".format(theta))
        plt.savefig("results/predictions/"+img_name)
        plt.clf()


# predict_model("models/pascal3d-vp-cnn-net1.pth", Net4)
