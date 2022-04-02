import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms

from src.services.arch import AerofoilNN

if __name__ == "__main__":
    airfoilname = "NACA2024"
    kind = "binary"
    num_channel = 1
    all_files = glob.glob(f"../out/{airfoilname}_{kind}*.jpg")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AerofoilNN(num_channel=num_channel).to(dev)
    model.load_state_dict(
        torch.load("../aerofoil_binary_4Conv_BN_2FC.pt", map_location=dev)
    )
    model.eval()

    arr = np.empty((0, 4), int)
    for filename in all_files:
        split = filename.split("_")[-1]
        angle = split.replace(".jpg", "")
        angle = int(angle)
        img = Image.open(filename)

        if num_channel == 3:
            img = img.convert("RGB")
        else:
            img = img.convert("L")

        transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if num_channel == 3
            else transforms.Normalize((0.5,), (0.5,))
        ])

        img = transform(img).unsqueeze(0)
        pred = model(img.to(dev))
        if torch.cuda.is_available():
            pred = pred.detach().cpu().numpy() * 0.5 + 0.5  # Denormalize
        else:
            pred = pred.detach().numpy() * 0.5 + 0.5  # Denormalize

        arr = np.append(
            arr,
            np.array([
                [angle, pred[0, 0], pred[0, 1], pred[0, 2]]
            ]),
            axis=0
        )

    df = pd.DataFrame(arr)
    df.columns = ["alpha", "cl", "cd", "cm"]
    df["name"] = airfoilname
    df = df.sort_values("alpha")

    print(f"Saving prediction result")
    df.to_csv("../prediction.csv", index=False)
