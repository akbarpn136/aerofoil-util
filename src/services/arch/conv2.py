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
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)

        return self.fc1(out)


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
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)

        return self.fc1(out)


class Aerofoil2BN2FC(AerofoilBaseNN):
    def __init__(self, num_channel=1):
        super(Aerofoil2BN2FC, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channel, 32, 15),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 15),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 15),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 15),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64*11*11, 128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out


class Aerofoil2BN3FC(AerofoilBaseNN):
    def __init__(self, num_channel=1):
        super(Aerofoil2BN3FC, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channel, 32, 17),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4, 4)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 15),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64*7*7, 600),
            nn.Dropout(0.25)
        )
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
