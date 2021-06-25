import os

import numpy as np
from scripts.dataset import PascalDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scripts.network import *
import torch

# Python file for all evaluation metrics / graphics of nn against ground truth


def distance_elevation_azimuth(xyz):
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
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


def predict_model(pth, net, net_name):
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

    test_acc = 0.0
    train_acc = 0.0

    if not os.path.exists("results"):
        os.mkdir("results")

    if not os.path.exists("results/"+net_name):
        os.mkdir("results/"+net_name)

    pt = "results/" + net_name + "/"

    for i in range(len(train_dset)):
        y = nt(train_dset[i][0].unsqueeze(0).to('cuda' if torch.cuda.is_available() else "cpu")).detach().cpu()
        img_name = train_dset.data[i].replace("/", "-")
        target = train_dset[i][1].unsqueeze(0)
        theta = get_angle(y, target)
        train_acc += thirty_deg_accuracy(y, target)
        y = y.numpy()
        target = target.numpy()

        _, pred_el, pred_az = distance_elevation_azimuth(y)
        _, target_el, target_az = distance_elevation_azimuth(target)
        train_pred_el.append(pred_el)
        train_target_el.append(target_el)
        train_pred_az.append(pred_az)
        train_target_az.append(target_az)
        '''
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot([0,y[0,0]], [0,y[0,1]], [0,y[0,2]], "k-", label="Pred")
        ax.plot([0, target[0,0]], [0, target[0,1]], [0, target[0,2]], "r-", label="Target")
        plt.legend()
        plt.title("Angle = {}".format(theta))
        plt.savefig("results/predictions/"+img_name)
        plt.close()'''

    print("TRAIN ACCURACY: {}".format(train_acc/len(train_dset)))

    for i in range(len(test_dset)):
        y = nt(test_dset[i][0].unsqueeze(0).to('cuda' if torch.cuda.is_available() else "cpu")).detach().cpu()
        img_name = test_dset.data[i].replace("/", "-")
        target = test_dset[i][1].unsqueeze(0)
        theta = get_angle(y, target)
        test_acc += thirty_deg_accuracy(y, target)
        y = y.numpy()
        target = target.numpy()

        _, pred_el, pred_az = distance_elevation_azimuth(y)
        _, target_el, target_az = distance_elevation_azimuth(target)
        test_pred_el.append(pred_el)
        test_target_el.append(target_el)
        test_pred_az.append(pred_az)
        test_target_az.append(target_az)

        '''
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot([0,y[0,0]], [0,y[0,1]], [0,y[0,2]], "k-", label="Pred")
        ax.plot([0, target[0,0]], [0, target[0,1]], [0, target[0,2]], "r-", label="Target")
        plt.legend()
        plt.title("Angle = {}".format(theta))
        plt.savefig("results/predictions/"+img_name)
        plt.close()'''

    print("TEST ACCURACY: {}".format(test_acc / len(test_dset)))

    cmap = cm.get_cmap('hsv')

    rgba = cmap(0.5)

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1, projection="polar")
    train_el_diff = [train_pred_el[i]-train_target_el[i] for i in range(len(train_pred_el))]
    ax2.scatter(train_pred_az, train_el_diff, cmap='hsv', c=[t/360.0 for t in train_target_az], s=2)
    plt.savefig(pt + "train.png")
    plt.close(fig)

    fig = plt.figure()
    ax2 = fig.add_subplot(1, 1, 1, projection="polar")
    test_el_diff = [test_pred_el[i] - test_target_el[i] for i in range(len(test_pred_el))]
    ax2.scatter(test_pred_az, test_target_az, c=[cmap(t/360.0) for t in test_target_az], s=2)
    plt.savefig(pt + "test.png")
    plt.close(fig)

    l = [(train_pred_el,train_target_el),(train_pred_az,train_target_az),(test_pred_el,test_target_el),
         (test_pred_az,test_target_az)]
    t = ["Train Elevation","Train Azimuth","Test Elevation","Test Azimuth"]
    for i in range(4):
        plt.figure()
        plt.scatter([x-180 for x in l[i][1]], [x-180 for x in l[i][0]], s=5)
        low = -100 if i % 2 == 0 else -180
        rn = 200 if i % 2 == 0 else 360
        plt.xlim(low, low+rn)
        plt.ylim(low, low+rn)
        plt.xlabel("Target")
        plt.ylabel("Pred")
        plt.savefig(pt + t[i].lower().replace(" ", "_") + ".png")



