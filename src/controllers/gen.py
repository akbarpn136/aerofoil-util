import time
import glob
import typer
import pandas as pd

from . import app
from services.image import rendering
from services.mesh import meshing


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
):
    """
    Function to generate airfoil geometry representation
    such as binary, mesh-like or SDF (Signed Distance Fields).
    """
    all_files = glob.glob(f"{path}/*.dat")
    for filename in all_files:
        airfoilname = filename.replace("foil", "") \
            .replace("\\", "") \
            .replace("/", "") \
            .replace(".dat", "")\
            .replace(" ", "")

        try:
            df = pd.read_csv(filename, delim_whitespace=True, header=None, skiprows=1)
            df.columns = ["x", "y"]

            val = df.loc[0, "x"]
            val = int(round(val, 0))
            if val == 1:
                typer.secho(f"Using Selig format for airfoil {airfoilname}", fg=typer.colors.CYAN)
            else:
                first_idx = df.loc[df.x == int(round(1, 0))].index.values
                first_half = df.iloc[:first_idx[0] + 1]
                second_half = df.iloc[first_idx[0] + 2:]

                # Reverse rows from first_half
                first_half = first_half.loc[::-1]
                df = pd.concat([first_half, second_half], axis=0, ignore_index=True)
                typer.secho(f"Using Lednicer format for airfoil {airfoilname}", fg=typer.colors.CYAN)

            typer.secho(f"Rendering {kind} airfoil {airfoilname}", fg=typer.colors.CYAN)
            start = time.time()
            if kind != "mesh":
                rendering(airfoilname, df.to_dict("records"), resolution, kind, angle_start, angle_stop, re, ma)
            elif kind == "mesh":
                meshing(airfoilname, df.to_numpy(), kind, angle_start, angle_stop, re, ma)
            else:
                typer.secho("Invalid kind. Only binary, mesh or sdf available.", fg=typer.colors.RED)
                typer.Abort()

            typer.secho(f"Rendering done. Took {round(time.time() - start, 1)} s", fg=typer.colors.GREEN)

        except FileNotFoundError as err:
            typer.secho(f"{err}", fg=typer.colors.RED)

            raise typer.Abort()
