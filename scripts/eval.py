
import numpy as np
import network
from dataset import PascalDataset

# Python file for all evaluation metrics / graphics of nn against ground truth


# Return angle between vectors
def get_angle(x,y):
    return np.arccos(np.dot(x,y))


# Count number of times the angle between predicted label and target is < 30 degrees
def thirty_deg_accuracy(y, target):
    y = y.numpy()
    target = target.numpy()
    y_norm = np.linalg.norm(y, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    norm = y_norm * target_norm.T
    theta = np.arccos(np.diag((y @ target.T)) / norm)
    size = float(theta.shape[0])
    return np.count_nonzero(theta < np.deg2rad(30.0)) / size


def evaluate_model(pth):
    # Load model
    nt = network.Net1()
    nt.load(pth)
    nt.eval()
    dset = PascalDataset(train=False)
    n = len(dset)
    acc = 0.0
    for i in range(0,n):
        X, target = dset[i]
        y = nt(X)
        k = thirty_deg_accuracy(y, target)
        print(k)
        acc += k
    return acc/n


print(evaluate_model("../models/test-model.pth"))
