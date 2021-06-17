
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 11)
        self.conv2 = nn.Conv2d(16, 16, 11)
        self.conv3 = nn.Conv2d(16, 16, 11)
        self.conv4 = nn.Conv2d(16, 8, 11)
        self.conv5 = nn.Conv2d(8, 8, 7)
        self.conv6 = nn.Conv2d(8, 8, 5)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(14 * 10 * 6, 144)  # 5*5 from image dimension
        self.fc2 = nn.Linear(144, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16,3)

    def forward(self, x):
        # C1
        x = F.relu(self.conv1(x))
        # C2
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # C3
        x = F.relu(self.conv3(x))
        # C4
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        # C5
        x = F.relu(self.conv5(x))
        # C6
        x = F.max_pool2d(F.relu(self.conv6(x)), 2)

        x = torch.flatten(x, 1)    # flatten all dimensions except the batch dimension
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc3(x)

        return x