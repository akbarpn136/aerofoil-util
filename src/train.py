import torch
from torch import nn
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split

from src.services.arch.conv5bn1fc import Aerofoil5BN1FC
from src.services.collection import AerofoilForceDataset

if __name__ == '__main__':
    batch_size = 41
    num_channel = 3
    num_epochs = 1600
    learning_rate = 0.00001

    dataset = AerofoilForceDataset(
        "out.csv",
        "out",
        transform=transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if num_channel == 3
            else transforms.Normalize((0.5,), (0.5,)),
        ]),
        num_channel=num_channel
    )
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    valid_size = dataset_size - train_size
    train_dataset, valid_dataset = random_split(dataset, (train_size, valid_size))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Aerofoil5BN1FC(num_channel=num_channel).to(dev)
    loss_func = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), learning_rate)

    model.fit(loss_func, optim, train_loader, valid_loader, device=dev, epochs=num_epochs)
    torch.save(model.state_dict(), "aerofoil_mesh_4Conv_BN_2FC.pt")
    plt.plot(model.lossList[1:], label="Train Loss")
    plt.plot(model.valid_lossList[1:], label="Valid Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()
