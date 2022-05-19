import sys
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm


class AerofoilBaseNN(nn.Module):
    def __init__(self):
        super(AerofoilBaseNN, self).__init__()
        torch.manual_seed(2022)

        self.lossList = []
        self.valid_lossList = []

    def fit(self,
            criterion,
            optimizer,
            train_loader,
            valid_loader,
            device,
            epochs=100):
        for epoch in range(epochs):
            loss_sum_train = 0
            loss_sum_validation = 0

            self.train()
            for _, (images, labels) in enumerate(tqdm(train_loader, desc="Train step", file=sys.stdout)):
                images = images.to(device)
                labels = labels.to(device)

                output = self(images)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum_train += loss.item()

            self.lossList.append(loss_sum_train)

            self.eval()
            for _, (images, labels) in enumerate(tqdm(valid_loader, desc="Valid step", file=sys.stdout)):
                images = images.to(device)
                labels = labels.to(device)
                output_valid = self(images)
                loss = criterion(output_valid, labels)
                loss_sum_validation += loss.item()

            self.valid_lossList.append(loss_sum_validation)

            print(
                f"Epoch {epoch + 1}/{epochs}: Train Loss: {loss_sum_train} | Validation Loss: {loss_sum_validation}\n"
            )

        combine = np.vstack((self.lossList, self.valid_lossList)).T
        df = pd.DataFrame(combine, columns=["train_loss", "valid_loss"])
        df.to_csv("train.csv")
