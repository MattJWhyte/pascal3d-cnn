import sys

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scripts.network import *
from scripts.dataset import RawPascalDataset
from scripts.shapenet_dataset import ShapeNetDataset, CATEGORIES
from scripts.eval import thirty_deg_accuracy_vector_full, distance_elevation_azimuth, get_angle
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 64
MAX_EPOCH = 50


def epoch(dataloader, model, loss_fn, train_loss_ls, train_acc_ls, test_loss_ls, test_acc_ls, optimizer=None):
    correct = 0.0
    avg_dev = 0.0
    min_dev = np.pi
    epoch_loss = 0.0
    ct = 0.0

    istrain = optimizer is not None
    torch.autograd.set_grad_enabled(istrain)
    model = model.train() if istrain else model.eval()

    for batch, (X, y) in enumerate(dataloader):
        ct += 1
        X, y = X.to(device), y.to(device)

        if batch == 1:
            break

        # Compute prediction and loss
        pred = model(X)
        pred = pred/pred.norm(dim=1, keepdim=True)
        loss = loss_fn(pred, y)

        if istrain:
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred = pred.detach().cpu()
        y = y.detach().cpu()
        k,mu,m = thirty_deg_accuracy_vector_full(pred, y)

        if m < min_dev:
            min_dev = m
        avg_dev += mu

        correct += np.count_nonzero(k)

        epoch_loss += loss.item()

    epoch_loss /= ct
    avg_dev /= ct
    correct /= (ct*float(BATCH_SIZE))
    if istrain:
        train_loss_ls.append(epoch_loss)
        train_acc_ls.append(correct)
    else:
        test_loss_ls.append(epoch_loss)
        test_acc_ls.append(correct)


def train(train_dataloader, test_dataloader, name):

    train_loss_ls = []
    train_acc_ls = []
    test_loss_ls = []
    test_acc_ls = []

    model = vgg_pose()
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    best_loss = 999999
    epochs_without_improvement = 0
    best_train_acc = 0.0
    best_val_acc = 0.0

    for t in range(MAX_EPOCH):
        epoch(train_dataloader, model, loss_fn, train_loss_ls, train_acc_ls, test_loss_ls, test_acc_ls, optimizer)
        epoch(test_dataloader, model, loss_fn, train_loss_ls, train_acc_ls, test_loss_ls, test_acc_ls)

        plt.plot([i for i in range(1, t + 2)], train_acc_ls, 'r-', label="Train acc.")
        plt.plot([i for i in range(1, t + 2)], train_loss_ls, 'r--', label="Train loss")
        plt.plot([i for i in range(1, t + 2)], test_acc_ls, 'b-', label="Test acc.")
        plt.plot([i for i in range(1, t + 2)], test_loss_ls, 'b--', label="Test loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.savefig("temp-training.png")
        plt.title(name)
        plt.clf()

        current_loss = test_loss_ls[-1]
        if current_loss <= best_loss:
            model.save("models/pascal3d-vp-cnn-vgg_pose-{}.pth".format(name))
            best_loss = current_loss
            best_train_acc = train_acc_ls[-1]
            best_val_acc = test_acc_ls[-1]
            epochs_without_improvement = 0
        elif epochs_without_improvement > 5:
            return best_train_acc, best_val_acc
        else:
            epochs_without_improvement += 1

    return best_train_acc, best_val_acc


if __name__ == "__main__":
    for cat in CATEGORIES:
        print("TRAINING {}".format(cat))
        shapenet_train_set = ShapeNetDataset((224, 224), cat_ls=[cat])
        pascal_set = RawPascalDataset((224, 224), train=True, cat_ls=[cat])
        n = len(pascal_set)
        pascal_train_set, pascal_val_set = torch.utils.data.random_split(pascal_set, (int(n*0.8), n - int(n*0.8)))
        pascal_train_dataloader = DataLoader(pascal_train_set, batch_size=BATCH_SIZE, shuffle=True)
        shapenet_train_dataloader = DataLoader(shapenet_train_set, batch_size=BATCH_SIZE, shuffle=True)
        pascal_val_dataloader = DataLoader(pascal_val_set, batch_size=BATCH_SIZE)
        print("ShapeNet - ", end="")
        train_acc, test_acc = train(shapenet_train_dataloader, pascal_val_dataloader, "{}-{}".format(cat, "shapenet"))
        print("Train {}% Val {}%".format(np.round(train_acc,2),np.round(test_acc,2)))
        print("Pascal - ", end="")
        train_acc, test_acc = train(pascal_train_dataloader, pascal_val_dataloader, "{}-{}".format(cat, "pascal"))
        print("Train {}% Val {}%".format(np.round(train_acc, 2), np.round(test_acc, 2)))
        print("--------------------------------")
