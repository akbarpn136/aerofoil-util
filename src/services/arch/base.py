import sys
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torchvision import models


class AerofoilBaseNN(nn.Module):
    def __init__(self, tf=False, dev=None):
        super(AerofoilBaseNN, self).__init__()
        torch.manual_seed(2022)

        self.tf = tf
        self.dev = dev
        self.lossList = []
        self.valid_lossList = []
        modal = None

        if self.tf == True:
            self.mdl = models.vgg11_bn(pretrained=True)

            for param in self.mdl.parameters():
                param.requires_grad = False

            self.mdl.classifier[6] = nn.Linear(4096, 3)
            self.mdl.to(self.dev)

    def fit(self,
            criterion,
            optimizer,
            train_loader,
            valid_loader,
            epochs=100):
        for epoch in range(epochs):
            loss_sum_train = 0
            loss_sum_validation = 0

            self.train()
            for _, (images, labels) in enumerate(tqdm(train_loader, desc="Train step", file=sys.stdout, colour="blue")):
                images = images.to(self.dev)
                labels = labels.to(self.dev)

                if self.tf == True:
                    output = self.mdl(images)
                else:
                    mdl = self.to(self.dev)
                    output = mdl(images)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum_train += loss.item()

            self.lossList.append(loss_sum_train)

            self.eval()
            for _, (images, labels) in enumerate(tqdm(valid_loader, desc="Valid step", file=sys.stdout, colour="green")):
                images = images.to(self.dev)
                labels = labels.to(self.dev)

                if self.tf == True:
                    output_valid = self.mdl(images)
                else:
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
