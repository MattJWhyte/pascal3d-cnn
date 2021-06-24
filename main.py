
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from scripts.network import *
from scripts.dataset import PascalDataset
from scripts.eval import thirty_deg_accuracy, get_angle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loss_ls = []
train_acc_ls = []
train_loss_test_ls = []
train_acc_test_ls = []
test_loss_ls = []
test_acc_ls = []

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    correct = 0.0
    train_loss = 0.0
    ct = 0.0

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        ct += 1
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        if optimizer is not None:
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred = pred.detach().cpu()
        y = y.detach().cpu()
        k = thirty_deg_accuracy(pred, y)
        correct += k

        train_loss += loss.item()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= ct
    correct /= ct
    print("Train loss: {}".format(train_loss))
    print("Train accuracy: {}".format(correct))
    train_loss_ls.append(train_loss)
    train_acc_ls.append(correct)


def train_loop_test(dataloader, model, loss_fn, optimizer=None):
    size = len(dataloader.dataset)
    correct = 0.0
    train_loss = 0.0
    ct = 0.0

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        ct += 1
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        pred = pred.detach().cpu()
        y = y.detach().cpu()
        k = thirty_deg_accuracy(pred, y)
        correct += k

        train_loss += loss.item()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= ct
    correct /= ct
    print("Train loss: {}".format(train_loss))
    print("Train accuracy: {}".format(correct))
    train_loss_test_ls.append(train_loss)
    train_acc_test_ls.append(correct)


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct, theta = 0.0, 0.0, []
    ct = 0.0

    model.eval()

    with torch.no_grad():
        for X, y in dataloader:
            ct += 1
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            pred = pred.detach().cpu()
            y = y.detach().cpu()
            k = thirty_deg_accuracy(pred, y)
            theta += get_angle(y, pred)
            correct += k

    test_loss /= ct
    print(f"Test loss: {test_loss:>8f} \n")
    print("Test accuracy: {}".format(correct/ct))
    print("Median angle: {}".format(np.median(np.array(theta))))
    print("Mean angle: {}".format(np.mean(np.array(theta))))
    test_loss_ls.append(test_loss)
    test_acc_ls.append(correct/ct)


train_set = PascalDataset()
val_set = PascalDataset(train=False)

train_dataloader = DataLoader(train_set, batch_size=128, shuffle=True)
train_no_shuffle_dataloader = DataLoader(train_set, batch_size=128, shuffle=True)
test_dataloader = DataLoader(val_set, batch_size=128, shuffle=True)

model = Net2()
model.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)

epochs = 1000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    train_loop_test(test_dataloader, model, loss_fn)
    test_loop(test_dataloader, model, loss_fn)
    model.save("models/pascal3d-vp-cnn-net2.pth")
    plt.plot([i for i in range(1,t+2)], train_acc_ls, 'r-', label="Train acc.")
    plt.plot([i for i in range(1, t + 2)], train_loss_ls, 'r--', label="Train loss")
    plt.plot([i for i in range(1, t + 2)], test_acc_ls, 'b-', label="Test acc.")
    plt.plot([i for i in range(1, t + 2)], test_loss_ls, 'b--', label="Test loss")
    plt.plot([i for i in range(1, t + 2)], train_acc_test_ls, 'g-', label="Test in train acc.")
    plt.plot([i for i in range(1, t + 2)], train_loss_test_ls, 'g--', label="Test in train loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.savefig("training.png")
    plt.clf()

