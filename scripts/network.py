
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        # Input 128 x 128
        self.conv1 = nn.Conv2d(3, 32, 5, stride=(2,2))
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=(2,2))
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=(2,2))
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.conv7 = nn.Conv2d(256, 256, 3, stride=(2,2))

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(256)

        # More channels , smaller convolutions
        # Use stride for convolutions instead of max pool
        # Downsample image more before model (128)

        # No more than 1024
        self.fc1 = nn.Linear(4 * 4 * 256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 3)

        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.fc3_bn = nn.BatchNorm1d(64)
        self.fc4_bn = nn.BatchNorm1d(32)


    def forward(self, x):
        # C1
        x = F.relu(self.bn1(self.conv1(x)))
        # C2
        x = F.relu(self.bn2(self.conv2(x)))
        # C3
        x = F.relu(self.bn3(self.conv3(x)))
        # C4
        x = F.relu(self.bn4(self.conv4(x)))
        # C5
        x = F.relu(self.bn5(self.conv5(x)))
        # C6
        x = F.relu(self.bn6(self.conv6(x)))
        # C7
        x = F.relu(self.bn7(self.conv7(x)))

        # Try get it to column vec

        x = torch.flatten(x, 1)    # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = F.relu(self.fc4_bn(self.fc4(x)))

        return self.fc5(x)

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))


class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        # Input 128 x 128
        self.conv1 = nn.Conv2d(3, 32, 21, stride=(2,2))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 23, stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(128)

        # More channels , smaller convolutions
        # Use stride for convolutions instead of max pool
        # Downsample image more before model (128)

        # No more than 1024
        self.fc1 = nn.Linear(6 * 6 * 128, 128)  # 5*5 from image dimension
        self.bn4 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        # C1
        x = F.relu(self.bn1(self.conv1(x)))
        # C2
        x = F.relu(self.bn2(self.conv2(x)))
        # C3
        x = F.relu(self.bn3(self.conv3(x)))

        # Try get it to column vec

        x = torch.flatten(x, 1)    # flatten all dimensions except the batch dimension
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))

        return self.fc3(x)

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))


class Net3(nn.Module):

    def __init__(self):
        super(Net3, self).__init__()
        # Input 128 x 128
        self.conv1 = nn.Conv2d(3, 4, 13, stride=(2,2))  # 58 x 58
        self.bn1 = nn.BatchNorm2d(4, track_running_stats=False)
        self.conv2 = nn.Conv2d(4, 8, 11, stride=(3, 3))  # 16 x 16
        self.bn2 = nn.BatchNorm2d(8, track_running_stats=False)
        self.conv3 = nn.Conv2d(8, 16, 11, stride=(2, 2))  # 3 x 3
        self.bn3 = nn.BatchNorm2d(16, track_running_stats=False)

        # More channels , smaller convolutions
        # Use stride for convolutions instead of max pool
        # Downsample image more before model (128)

        # No more than 1024
        self.fc1 = nn.Linear(3 * 3 * 16, 16)  # 5*5 from image dimension
        self.bn5 = nn.BatchNorm1d(16, track_running_stats=False)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        # C1
        x = F.relu(self.bn1(self.conv1(x)))
        # C2
        x = F.relu(self.bn2(self.conv2(x)))
        # C3
        x = F.relu(self.bn3(self.conv3(x)))

        # Try get it to column vec

        x = torch.flatten(x, 1)    # flatten all dimensions except the batch dimension
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.fc2(x)

        return x

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))


class Net4(nn.Module):

    def __init__(self):
        super(Net4, self).__init__()
        # Input 128 x 128
        self.conv1 = nn.Conv2d(3, 32, 3, stride=(2,2), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=(2, 2), padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=(2, 2), padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, 3, stride=(2, 2), padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        # More channels , smaller convolutions
        # Use stride for convolutions instead of max pool
        # Downsample image more before model (128)

        # No more than 1024
        self.fc1 = nn.Linear(3 * 3 * 32, 128)  # 5*5 from image dimension
        self.bn6 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        # C1
        x = F.relu(self.bn1(self.conv1(x)))
        # C2
        x = F.relu(self.bn2(self.conv2(x)))
        # C3
        x = F.relu(self.bn3(self.conv3(x)))
        # C4
        x = F.relu(self.bn4(self.conv4(x)))
        # C5
        x = F.relu(self.bn5(self.conv5(x)))
        # Try get it to column vec

        x = torch.flatten(x, 1)    # flatten all dimensions except the batch dimension
        x = F.silu(self.bn6(self.fc1(x)))
        x = F.silu(self.fc2(x))

        return x

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))


class Net5(nn.Module):

    def __init__(self):
        super(Net5, self).__init__()
        # Input 128 x 128
        self.conv1 = nn.Conv2d(3, 32, 3, stride=(2,2), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=(2, 2), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=(2, 2), padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 3, stride=(2, 2), padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        # More channels , smaller convolutions
        # Use stride for convolutions instead of max pool
        # Downsample image more before model (128)

        # No more than 1024
        self.fc1 = nn.Linear(3 * 3 * 256, 128)  # 5*5 from image dimension
        self.bn6 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        # C1
        x = F.relu(self.bn1(self.conv1(x)))
        # C2
        x = F.relu(self.bn2(self.conv2(x)))
        # C3
        x = F.relu(self.bn3(self.conv3(x)))
        # C4
        x = F.relu(self.bn4(self.conv4(x)))
        # C5
        x = F.relu(self.bn5(self.conv5(x)))
        # Try get it to column vec

        x = torch.flatten(x, 1)    # flatten all dimensions except the batch dimension
        x = F.silu(self.bn6(self.fc1(x)))
        x = F.silu(self.fc2(x))

        return x

    def save(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))