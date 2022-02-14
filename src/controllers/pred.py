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
    images = []
    angles = []
    path = f"out/{airfoilname}"
    all_files = glob.glob(f"{path}/*.jpg")

    for filename in all_files:
        split = filename.split("_")[-1]
        angle = split.replace(".jpg", "")
        angle = int(angle)
        angles.append(angle)
        img = Image.open(filename)
        img = np.asarray(img)

        images.append(img)

    images = np.array(images, dtype=np.float32)
    images = images.reshape((
        images.shape[0],
        3,
        images.shape[1],
        images.shape[2]
    ))
    images = torch.from_numpy(images)

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = transform(images)

    with torch.no_grad():
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AerofoilNN(None).to(dev)
        model.load_state_dict(
            torch.load("aerocnn.pt", map_location=dev)
        )

        res = model(img.to(dev))
        df = pd.DataFrame(res)
        df.columns = ["cl", "cd", "cm"]
        df = (df * 0.5) + 0.5
        df["sudut"] = 0
        df = df.sort_values("sudut")

        df.to_csv("prediction.csv")

        typer.secho(f"Saving prediction result", fg=typer.colors.MAGENTA)
