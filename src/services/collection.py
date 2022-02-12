import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class AerofoilForceDataset(Dataset):
    def __init__(self, csvfile, imgdir, transform=None):
        self.df = pd.read_csv(csvfile)
        self.imgdir = imgdir
        self.transform = transform

    def __len__(self):
        return self.df.index.size

    def __getitem__(self, item):
        img = Image.open(f"{self.imgdir}/{self.df.iloc[item]['img']}")
        img = img.convert("RGB")

        labels = self.df.iloc[item][["cl", "cd", "cm"]].to_numpy(dtype="float32")
        labels = torch.from_numpy(labels)

        if self.transform:
            img = self.transform(img)

        return img, labels
