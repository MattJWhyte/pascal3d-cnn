
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from scripts.network import Net1
from scripts.dataset import PascalDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        X.detach()
        del X
        y.detach()
        del y
        torch.cuda.empty_cache()


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            X.detach()
            del X
            y.detach()
            del y
            torch.cuda.empty_cache()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    print(f"Avg loss: {test_loss:>8f} \n")

    model.save("models/test-model.pth")


train_set = PascalDataset()
val_set = PascalDataset(train=False)

train_dataloader = DataLoader(train_set, batch_size=48)
test_dataloader = DataLoader(val_set, batch_size=48)

model = Net1()
model.to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    model.save("models/pascal3d-vp-cnn-net1.pth")
print("Done!")