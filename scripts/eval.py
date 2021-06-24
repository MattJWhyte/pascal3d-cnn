import os

import numpy as np
from scripts.dataset import PascalDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scripts.network import *
import torch

# Python file for all evaluation metrics / graphics of nn against ground truth


def distance_elevation_azimuth(xyz):
    x = xyz[0,0]
    y = xyz[0,1]
    z = xyz[0,2]
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

    test_pred_az = []
    test_target_az = []
    test_pred_el = []
    test_target_el = []

    if not os.path.exists("results"):
        os.mkdir("results")

    if not os.path.exists("results/predictions"):
        os.mkdir("results/predictions")

    for i in range(len(train_dset)):
        y = nt(train_dset[i][0].unsqueeze(0).to('cuda' if torch.cuda.is_available() else "cpu")).detach().cpu()
        img_name = train_dset.data[i].replace("/", "-")
        target = train_dset[i][1].unsqueeze(0)
        theta = get_angle(y, target)
        y = y.numpy()
        target = target.numpy()

        _, pred_el, pred_az = distance_elevation_azimuth(y)
        _, target_el, target_az = distance_elevation_azimuth(target)
        train_pred_el.append(pred_el)
        train_target_el.append(target_el)
        train_pred_az.append(pred_az)
        train_target_az.append(target_az)

        print("Train {}".format(i))
        '''
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot([0,y[0,0]], [0,y[0,1]], [0,y[0,2]], "k-", label="Pred")
        ax.plot([0, target[0,0]], [0, target[0,1]], [0, target[0,2]], "r-", label="Target")
        plt.legend()
        plt.title("Angle = {}".format(theta))
        plt.savefig("results/predictions/"+img_name)
        plt.close()'''

    for i in range(len(test_dset)):
        y = nt(test_dset[i][0].unsqueeze(0).to('cuda' if torch.cuda.is_available() else "cpu")).detach().cpu()
        img_name = test_dset.data[i].replace("/", "-")
        target = test_dset[i][1].unsqueeze(0)
        theta = get_angle(y, target)
        y = y.numpy()
        target = target.numpy()

        _, pred_el, pred_az = distance_elevation_azimuth(y)
        _, target_el, target_az = distance_elevation_azimuth(target)
        test_pred_el.append(pred_el)
        test_target_el.append(target_el)
        test_pred_az.append(pred_az)
        test_target_az.append(target_az)

        print("Test {}".format(i))

        '''
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot([0,y[0,0]], [0,y[0,1]], [0,y[0,2]], "k-", label="Pred")
        ax.plot([0, target[0,0]], [0, target[0,1]], [0, target[0,2]], "r-", label="Target")
        plt.legend()
        plt.title("Angle = {}".format(theta))
        plt.savefig("results/predictions/"+img_name)
        plt.close()'''

    l = [(train_pred_el,train_target_el),(train_pred_az,train_target_az),(test_pred_el,test_target_el),
         (test_pred_az,test_target_az)]
    t = ["Train Elevation","Train Azimuth","Test Elevation","Test Azimuth"]
    for i in range(4):
        plt.figure()
        plt.scatter(l[i][1], l[i][0])
        low = -50 if i % 2 == 0 else 0
        rn = 180 if i % 2 == 0 else 360
        plt.xlim(low, low+rn)
        plt.ylim(low, low+rn)
        plt.xlabel("Target")
        plt.ylabel("Pred")
        plt.savefig("results/" + t[i].lower().replace(" ", "_") + ".png")



