
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        # Input 128 x 128
        self.conv1 = nn.Conv2d(3, 32, 7)
        self.conv2 = nn.Conv2d(32, 64, 7)
        self.conv3 = nn.Conv2d(64, 64, 5, stride=(2,2))
        self.conv4 = nn.Conv2d(64, 128, 3, stride=(2, 2))
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 256, 3)

        # More channels , smaller convolutions
        # Use stride for convolutions instead of max pool
        # Downsample image more before model (128)

        # No more than 1024
        self.fc1 = nn.Linear(8 * 8 * 256, 128)  # 5*5 from image dimension
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 3)

    def forward(self, x):
        # C1
        x = F.relu(self.conv1(x))
        # C2
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # C3
        x = F.relu(self.conv3(x))
        # C4
        x = F.relu(self.conv4(x))
        # C5
        x = F.relu(self.conv5(x))
        # C6
        x = F.relu(self.conv6(x))

        # Try get it to column vec

        x = torch.flatten(x, 1)    # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return x

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))

