from torch import nn

from src.services.arch.base import AerofoilBaseNN


class Aerofoil2Relu1FC(AerofoilBaseNN):
    def __init__(self, num_channel=3):
        super(Aerofoil2Relu1FC, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channel, 40, 5, 1, 0),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(40, 80, 5, 1, 0),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(2000, 3)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3_flat = f2.view(f2.size(0), -1)

        return self.fc1(f3_flat)


class Aerofoil2BN1FC(AerofoilBaseNN):
    def __init__(self, num_channel=3):
        super(Aerofoil2BN1FC, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channel, 40, 5, 1, 0),
            nn.BatchNorm2d(40),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(40, 80, 5, 1, 0),
            nn.BatchNorm2d(80),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(2000, 3)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3_flat = f2.view(f2.size(0), -1)

        return self.fc1(f3_flat)


class Aerofoil2BN3FC(AerofoilBaseNN):
    def __init__(self):
        super(Aerofoil2BN1FC, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=3)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
