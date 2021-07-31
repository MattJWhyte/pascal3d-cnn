import sys

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scripts.network import *
from scripts.dataset import RawPascalDataset
from scripts.shapenet_dataset import ShapeNetDataset
from scripts.eval import thirty_deg_accuracy_vector_full, distance_elevation_azimuth
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loss_ls = []
train_acc_ls = []
test_loss_ls = []
test_acc_ls = []


def epoch(dataloader, model, loss_fn, optimizer=None):
    size = len(dataloader.dataset)
    correct = 0.0
    avg_dev = 0.0
    min_dev = np.pi
    epoch_loss = 0.0
    ct = 0.0

    istrain = optimizer is not None
    torch.autograd.set_grad_enabled(istrain)
    model = model.train() if istrain else model.eval()

    bin_ct = np.array([0.0 for _ in range(24)])
    bin_acc = [0.0 for _ in range(24)]

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
        k,mu,m = thirty_deg_accuracy_vector_full(pred, y)

        if m < min_dev:
            min_dev = m
        avg_dev += mu

        #_,_,e = distance_elevation_azimuth(y.numpy())
        #e = (e % 15).astype(int)

        #print(np.max(e))

        #bin_ct += np.histogram(e, bins= np.array([0.5] + [i+1.5 for i in range(24)]))[0]

        correct += np.count_nonzero(k)

        epoch_loss += loss.item()

        if istrain and batch % 20 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        if not istrain and batch % 10 == 0:
            b = batch
            if not os.path.exists("predictions/sample-{}".format(b)):
                os.mkdir("predictions/sample-{}".format(b))
            torchvision.utils.save_image(X[0], "predictions/sample-{}/img.png".format(b))
            ln = []
            ln.append("Predicted : {}\n".format(str(pred.numpy()[0].tolist())))
            ln.append("\t\t{}\n".format(str(distance_elevation_azimuth(pred.numpy()[0]))))
            ln.append("Target : {}\n".format(str(y.numpy()[0].tolist())))
            ln.append("\t\t{}\n".format(str(distance_elevation_azimuth(y.numpy()[0]))))
            with open("predictions/sample-{}/info.txt".format(b), "w") as f:
                f.writelines(ln)

    f = plt.figure()
    ax = f.add_subplot(projection='polar')
    ax.plot([15.0*i*np.pi/180.0 for i in range(24)], bin_ct)
    plt.savefig("train-dist.png")
    plt.close()

    epoch_loss /= ct
    avg_dev /= ct
    correct /= float(len(dataloader.dataset))
    if istrain:
        print("Train loss: {}".format(epoch_loss))
        print("Train accuracy: {}".format(correct))
        print("Train avg dev: {}".format(avg_dev))
        print("Train min dev: {}".format(min_dev))
        train_loss_ls.append(epoch_loss)
        train_acc_ls.append(correct)
    else:
        print("Test loss: {}".format(epoch_loss))
        print("Test accuracy: {}".format(correct))
        print("Test avg dev: {}".format(avg_dev))
        print("Test min dev: {}".format(min_dev))
        test_loss_ls.append(epoch_loss)
        test_acc_ls.append(correct)


# Get CLI args
model_name = sys.argv[1].lower()
width = int(sys.argv[2])
height = int(sys.argv[3])
comment = ""
if len(sys.argv) > 4:
    comment = "-" + sys.argv[4]

print("Saving model to 'models/pascal3d-vp-cnn-"+model_name+comment+".pth'")

train_set = ShapeNetDataset((width,height), cat_ls=["aeroplane"])
val_set = RawPascalDataset((width,height), train=False, cat_ls=["aeroplane"])

'''
st = ShapeNetDataset((width,height), cat_ls=["aeroplane"])
train_len = int(len(st)*0.7)
test_len = len(st)-train_len
train_set, val_set = torch.utils.data.random_split(st, [train_len, test_len])'''

train_dataloader = DataLoader(train_set, batch_size=96, shuffle=True)
test_dataloader = DataLoader(val_set, batch_size=96)

model = MODEL[model_name]()
model.to(device)
name = type(model).__name__.lower()

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    epoch(train_dataloader, model, loss_fn, optimizer)
    epoch(test_dataloader, model, loss_fn)
    model.save("models/pascal3d-vp-cnn-"+name+comment+".pth")
    plt.plot([i for i in range(1,t+2)], train_acc_ls, 'r-', label="Train acc.")
    plt.plot([i for i in range(1, t + 2)], train_loss_ls, 'r--', label="Train loss")
    plt.plot([i for i in range(1, t + 2)], test_acc_ls, 'b-', label="Test acc.")
    plt.plot([i for i in range(1, t + 2)], test_loss_ls, 'b--', label="Test loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.savefig("training-"+name+".png")
    plt.clf()

