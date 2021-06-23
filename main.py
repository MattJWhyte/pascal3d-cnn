
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from scripts.network import Net1, Net2
from scripts.dataset import PascalDataset
from scripts.eval import thirty_deg_accuracy, get_angle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct, theta = 0.0, 0.0, []
    ct = 0.0

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

            torch.cuda.empty_cache()

    test_loss /= ct
    print(f"Avg loss: {test_loss:>8f} \n")
    print("Accuracy: {}".format(correct/ct))
    print("Median angle: {}".format(np.median(np.array(theta))))
    print("Mean angle: {}".format(np.mean(np.array(theta))))
    model.save("models/test-model.pth")


train_set = PascalDataset()
val_set = PascalDataset(train=False)

train_dataloader = DataLoader(train_set, batch_size=128)
test_dataloader = DataLoader(val_set, batch_size=128)

model = Net2()
model.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    model.save("models/pascal3d-vp-cnn-net1.pth")
print("Done!")