from torch import nn


class AerofoilNN(nn.Module):
    def __init__(self, device):
        super(AerofoilNN, self).__init__()

        self.lossList = []
        self.valid_lossList = []
        self.device = device

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 10, 13),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, 7),
            # nn.Dropout2d(0.5),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 40, 7),
            nn.BatchNorm2d(40),
            # nn.Dropout2d(0.5),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(40, 80, 5),
            nn.BatchNorm2d(80),
            # nn.Dropout2d(0.5),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(720, 400),
            nn.ReLU(),
            # nn.Dropout(0.5)
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

    def fit(self, criterion, optimizer, train_loader, valid_loader, epochs=100):
        for epoch in range(epochs):
            loss_sum_train = 0
            loss_sum_validation = 0

            for _, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                output = self(images)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum_train += loss.item()

            self.lossList.append(loss_sum_train)

            for _, (images, labels) in enumerate(valid_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                output_valid = self(images)
                loss = criterion(output_valid, labels)
                loss_sum_validation += loss.item()

            self.valid_lossList.append(loss_sum_validation)

            print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {loss_sum_train} | Validation Loss: {loss_sum_validation}")