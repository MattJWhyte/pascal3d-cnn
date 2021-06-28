import sys

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scripts.network import *
from scripts.dataset import PascalDataset, RawPascalDataset
from scripts.eval import thirty_deg_accuracy, get_angle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loss_ls = []
train_acc_ls = []
test_loss_ls = []
test_acc_ls = []

def epoch(dataloader, model, loss_fn, optimizer=None):
    size = len(dataloader.dataset)
    correct = 0.0
    epoch_loss = 0.0
    ct = 0.0

    istrain = optimizer is not None
    torch.autograd.set_grad_enabled(istrain)
    model = model.train() if istrain else model.eval()

    for batch, (X, y) in enumerate(dataloader):
        ct += 1
        X, y = X.to(device), y.to(device)

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
        k = thirty_deg_accuracy(pred, y)
        correct += k

        epoch_loss += loss.item()

        if istrain and batch % 20 == 0:
            torchvision.utils.save_image(X[0], 'sample.png')
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    epoch_loss /= ct
    correct /= ct
    if istrain:
        print("Train loss: {}".format(epoch_loss))
        print("Train accuracy: {}".format(correct))
        train_loss_ls.append(epoch_loss)
        train_acc_ls.append(correct)
    else:
        print("Test loss: {}".format(epoch_loss))
        print("Test accuracy: {}".format(correct))
        test_loss_ls.append(epoch_loss)
        test_acc_ls.append(correct)


# Get CLI args
model_name = sys.argv[1].lower()
width = int(sys.argv[2])
height = int(sys.argv[3])

train_set = RawPascalDataset((width,height))
val_set = RawPascalDataset((width,height), train=False)

train_dataloader = DataLoader(train_set, batch_size=96, shuffle=True)
test_dataloader = DataLoader(val_set, batch_size=96)

model = MODEL[model_name]()
model.to(device)
name = type(model).__name__.lower()

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 1000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    epoch(train_dataloader, model, loss_fn, optimizer)
    epoch(test_dataloader, model, loss_fn)
#    model.save("models/pascal3d-vp-cnn-"+name+".pth")
    plt.plot([i for i in range(1,t+2)], train_acc_ls, 'r-', label="Train acc.")
    plt.plot([i for i in range(1, t + 2)], train_loss_ls, 'r--', label="Train loss")
    plt.plot([i for i in range(1, t + 2)], test_acc_ls, 'b-', label="Test acc.")
    plt.plot([i for i in range(1, t + 2)], test_loss_ls, 'b--', label="Test loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.savefig("training-"+name+".png")
    plt.clf()

