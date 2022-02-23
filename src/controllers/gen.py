import os
import typer
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count

from . import app
from services.mesh import meshing
from services.image import rendering


@app.command()
def generate(
        kind: str = typer.Option(
            "binary",
            "--kind",
            "-k",
            help="Airfoil geometry representation: binary, mesh or sdf.",
        ),
        angle_start: int = typer.Option(
            0,
            "--start",
            "-st",
            help="Start from angle of attack e.g. -10.",
        ),
        angle_stop: int = typer.Option(
            0,
            "--stop",
            "-sp",
            help="Stop to angle of attack e.g. 10.",
        ),
        resolution: int = typer.Option(
            1024,
            "--res",
            "-rs",
            help="Airfoil image resolution in pixel.",
        ),
        re: int = typer.Option(
            500000,
            "--re",
            "-r",
            help="Reynold number.",
        ),
        ma: float = typer.Option(
            0.0,
            "--ma",
            "-m",
            help="Mach number.",
        ),
        path: str = typer.Option(
            "foil",
            "--path",
            "-p",
            help="Path to airfoil.dat files",
        ),
        filename: str = typer.Option(
            "out.csv",
            "--filename",
            "-f",
            help="Filename for output csv file after using get command",
        )
):
    """
    Function to generate airfoil geometry representation
    such as binary, mesh-like or SDF (Signed Distance Fields).
    """
    df = pd.read_csv(f"{filename}")
    names = df["name"].unique()
    angles = range(angle_start, angle_stop + 1)
    resolution = [resolution]
    kind = [kind]
    re = [re]
    ma = [ma]
    path = [path]
    paramlist = list(itertools.product(names, angles, resolution, kind, re, ma, path))
    path = "out"
    isExist = os.path.exists(path)

    if not isExist:
        typer.secho("Creating new folder", fg=typer.colors.YELLOW)
        os.makedirs(path)

    typer.secho(f"Generating airfoil image. It might take time ⏳", fg=typer.colors.CYAN)

    with Pool(processes=cpu_count()) as pool:
        results = [x for x in tqdm(pool.imap(to_img, paramlist),
                                   total=len(paramlist))]
        results = np.array(results)

        for i in tqdm(range(results.shape[0])):
            fig, ax = plt.subplots()
            plt.margins(x=0, y=0)
            plt.axis("off")
            ax.set_box_aspect(1)
            plt.tight_layout()

            if kind == "binary":
                colormap = "gray"
            elif kind == "sdf":
                colormap = "jet"
            else:
                colormap = "gray"

            plt.imshow(results[i], cmap=plt.get_cmap(colormap))

            plt.savefig(f"out/coba{i}.jpg", bbox_inches="tight", pad_inches=0, dpi=34.7)
            plt.close("all")

    typer.secho("Rendering done.", fg=typer.colors.GREEN)


def to_img(payload):
    name = payload[0]
    angle = payload[1]
    resolution = payload[2]
    kind = payload[3]
    re = payload[4]
    ma = payload[5]
    path = payload[6]

    try:
        df = pd.read_csv(f"{path}/{name}.dat", delim_whitespace=True, header=None, skiprows=1)
        df.columns = ["x", "y"]

        val = df.loc[0, "x"]
        val = int(round(val, 0))
        if val != 1:
            first_idx = df.loc[df.x == int(round(1, 0))].index.values
            first_half = df.iloc[:first_idx[0] + 1]
            second_half = df.iloc[first_idx[0] + 2:]

            # Reverse rows from first_half
            first_half = first_half.loc[::-1]
            df = pd.concat([first_half, second_half], axis=0, ignore_index=True)

        airfoil_name = name.replace(" ", "")
        if kind != "mesh":
            rendering(name, angle, df.to_dict("records"), resolution, kind, re, ma)
        elif kind == "mesh":
            meshing(name, angle, df.to_numpy(), kind, re, ma)
        else:
            typer.secho("Invalid kind. Only binary, mesh or sdf available.", fg=typer.colors.RED)
            typer.Abort()
    except FileNotFoundError as err:
        typer.secho(f"{err}", fg=typer.colors.RED)

        raise typer.Abort()
