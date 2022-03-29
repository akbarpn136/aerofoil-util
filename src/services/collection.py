import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class AerofoilForceDataset(Dataset):
    def __init__(self, csvfile, imgdir, transform=None, num_channel=3):
        self.df = pd.read_csv(csvfile)
        self.imgdir = imgdir
        self.transform = transform
        self.num_channel = num_channel

    def __len__(self):
        return self.df.index.size

    def __getitem__(self, item):
        img = Image.open(f"{self.imgdir}/{self.df.iloc[item]['img']}")

        if self.num_channel == 3:
            img = img.convert("RGB")
        else:
            img = img.convert("L")

        labels = self.df.iloc[item][["cl", "cd", "cm"]].to_numpy(dtype="float32")
        labels = denorm(labels, 0.5, 0.5)
        labels = torch.from_numpy(labels)

        if self.transform:
            img = self.transform(img)

        return img, labels


def denorm(x, mean, std, normalize=True):
    if normalize:
        xx = (x - mean) / std

    else:
        xx = x * std + mean

    return xx
