import os

import numpy as np
from dataset import PascalDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from network import *
import torch

# Python file for all evaluation metrics / graphics of nn against ground truth


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

    if os.path.exists("results"):
        os.mkdir("results")

    if os.path.exists("results/predictions"):
        os.mkdir("results/predictions")

    for i in range(len(train_dset)):
        y = nt(train_dset[i][0].unsqueeze(0).to('cuda' if torch.cuda.is_available() else "cpu")).detach().cpu().numpy()
        img_name = train_dset.labels[i].replace("/", "-")
        target = train_dset[i][1].numpy()
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot([0,y[0]], [0,y[1]], [0,y[2]], "k-", label="Pred")
        ax.plot([0, target[0]], [0, target[1]], [0, target[2]], "r-", label="Target")
        plt.legend()
        plt.savefig("results/predictions/"+img_name)
        plt.clf()

    for i in range(len(test_dset)):
        y = nt(test_dset[i][0].unsqueeze().to('cuda' if torch.cuda.is_available() else "cpu")).detach().cpu().numpy()
        img_name = test_dset.labels[i].replace("/", "-")
        target = test_dset[i][1].numpy()
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot([0,y[0]], [0,y[1]], [0,y[2]], "k-", label="Pred")
        ax.plot([0, target[0]], [0, target[1]], [0, target[2]], "r-", label="Target")
        plt.legend()
        plt.savefig("results/predictions/"+img_name)
        plt.clf()


predict_model("models/pascal3d-vp-cnn-net1.pth", Net4)
