from torch import nn

from src.services.arch.base import AerofoilBaseNN


class Aerofoil3BN1FC(AerofoilBaseNN):
    def __init__(self, num_channel=3):
        super(Aerofoil3BN1FC, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channel, 10, 3, 1, 1),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, 3, 1, 1),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 40, 3, 1, 0),
            nn.BatchNorm2d(40),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(9000, 3)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4_flat = f3.view(f3.size(0), -1)

        return self.fc1(f4_flat)


class Aerofoil3BN2FC(AerofoilBaseNN):
    def __init__(self, num_channel=3):
        super(Aerofoil3BN2FC, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channel, 16, 5, 1, 1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5,),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
        )

        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out
