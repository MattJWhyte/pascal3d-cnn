
import numpy as np
import scripts.network
from scripts.dataset import PascalDataset
from torch.utils.data import DataLoader
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


def evaluate_model(pth):
    # Load model
    nt = scripts.network.Net1()
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

