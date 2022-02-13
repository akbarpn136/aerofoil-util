import glob
import torch
import typer
import numpy as np
from PIL import Image
from torchvision.transforms import transforms

from . import app
from services.arch import AerofoilNN
from services.collection import AerofoilForceDataset


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
    path = f"out/{airfoilname}"
    all_files = glob.glob(f"{path}/*.jpg")

    for filename in all_files:
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
    ])

    images = transform(images)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AerofoilNN(dev).to(dev)
    model.load_state_dict(
        torch.load("aerocnn.pt", map_location=dev)
    )
    model.eval()

    res = model(images)

    print(res)
    print(res.shape)

    typer.secho(f"Prediction result in {path}", fg=typer.colors.MAGENTA)
