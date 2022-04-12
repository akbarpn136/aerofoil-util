from torch import nn

from src.services.arch.base import AerofoilBaseNN


class Aerofoil2BN1FC(AerofoilBaseNN):
    def __init__(self, num_channel=3):
        super(Aerofoil2BN1FC, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channel, 20, 5, 1, 0),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 40, 5, 1, 0),
            nn.BatchNorm2d(40),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(1000, 3)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3_flat = f2.view(f2.size(0), -1)

        return self.fc1(f3_flat)
