from torch import nn

from src.services.arch.base import AerofoilBaseNN


class Aerofoil4BN1FC(AerofoilBaseNN):
    def __init__(self, num_channel=3):
        super(Aerofoil4BN1FC, self).__init__()

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
            nn.Conv2d(20, 40, 3, 1, 1),
            nn.BatchNorm2d(40),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(40, 60, 3, 1, 1),
            nn.BatchNorm2d(60),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(3840, 3)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv2(f2)
        f4 = self.conv2(f3)
        f5_flat = f4.view(f4.size(0), -1)

        return self.fc1(f5_flat)
