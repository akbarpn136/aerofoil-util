import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class AerofoilForceDataset(Dataset):
    def __init__(self, csvfile, imgdir, transform=None):
        self.df = pd.read_csv(csvfile)
        self.imgdir = imgdir
        self.transform = transform

    def __len__(self):
        return self.df.index.size

    def __getitem__(self, item):
        img = cv2.imread(f"{self.imgdir}/{self.df.iloc[item]['img']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels = self.df.iloc[item][["cl", "cd", "cm"]].to_numpy(dtype="float32")
        labels = torch.from_numpy(labels)

        if self.transform:
            img = self.transform(img)

        return img, labels
