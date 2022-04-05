from torch import nn

from src.services.arch.base import AerofoilBaseNN


class Aerofoil4BNDO2FC(AerofoilBaseNN):
    def __init__(self, num_channel=3):
        super(Aerofoil4BNDO2FC, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channel, 10, 13),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, 7),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 40, 7),
            nn.BatchNorm2d(40),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(40, 80, 5),
            nn.BatchNorm2d(80),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(720, 400),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Linear(400, 3)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        f4_flat = f4.view(f4.size(0), -1)
        f5 = self.fc1(f4_flat)

        return self.fc2(f5)
