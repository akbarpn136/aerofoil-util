import glob
import torch
import typer
import numpy as np
import pandas as pd
from PIL import Image
from services.arch import AerofoilNN
from torchvision.transforms import transforms

from . import app


@app.command()
def predict(
        airfoilname: str = typer.Argument(
            ...,
            help="Airfoil name code such as naca2412.",
            metavar="airfoilname"
        )
):
    """
    Function to predict the airfoil coeficient aerodynamic
    (Cl, Cd and Cm) with varying angle of attack.
    """
    path = f"out/{airfoilname}"
    all_files = glob.glob(f"{path}/*.jpg")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AerofoilNN().to(dev)
    model.load_state_dict(
        torch.load("aerocnn.pt", map_location=dev)
    )
    model.eval()

    arr = np.empty((0, 4), int)
    for filename in all_files:
        split = filename.split("_")[-1]
        angle = split.replace(".jpg", "")
        angle = int(angle)
        img = Image.open(filename)

        transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        img = transform(img).unsqueeze(0)
        pred = model(img.to(dev))
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

    typer.secho(f"Saving prediction result", fg=typer.colors.MAGENTA)
    df.to_csv("prediction.csv", index=False)
