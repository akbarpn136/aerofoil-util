import torch
from torch import nn
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split

from src.libs.arch import AerofoilNN
from src.libs.collection import AerofoilForceDataset

if __name__ == '__main__':
    batch_size = 21
    num_epochs = 30
    learning_rate = 0.00001

    dataset = AerofoilForceDataset(
        "payload/airfoil.csv",
        "payload",
        transform=transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
        ])
    )

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    valid_size = dataset_size - train_size
    train_dataset, valid_dataset = random_split(dataset, (train_size, valid_size))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AerofoilNN(dev).to(dev)
    loss_func = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), learning_rate)

    model.fit(loss_func, optim, train_loader, valid_loader, epochs=30)

    torch.save(model.state_dict(), "aerocnn.pt")

    plt.plot(model.lossList, label="Train Loss")
    plt.plot(model.valid_lossList, label="Valid Loss")
    plt.legend()
    plt.show()
