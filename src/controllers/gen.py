import os
import typer
import pandas as pd

from . import app


@app.command()
def generate(
        airfoilname: str = typer.Argument(
            ...,
            help="Airfoil name code such as naca2412",
            metavar="airfoilname"
        ),
        filename: str = typer.Argument(
            ...,
            help="Airfoil coordinate file in csv without a header from current directory",
            metavar="filename"
        ),
        angle_start: int = typer.Option(
            0,
            "--start",
            help="Start from angle of attack e.g. -10",
        ),
        angle_stop: int = typer.Option(
            1,
            "--stop",
            help="Stop to angle of attack e.g. 10",
        ),
):
    """
    Function to generate airfoil geometry representation
    such as binary, mesh-like or SDF (Signed Distance Fields).
    """
    try:
        df = pd.read_csv(filename, delim_whitespace=True, header=None)
        df.columns = ["x", "y"]

        val = df.loc[0, "x"]
        val = int(round(val, 0))
        if val == 1:
            typer.secho(f"Using Selig format for airfoil {airfoilname}", fg=typer.colors.CYAN)
            coord_points = df.to_dict("records")
        else:
            first_idx = df.loc[df.x == int(round(1, 0))].index.values
            first_half = df.iloc[:first_idx[0] + 1]
            second_half = df.iloc[first_idx[0] + 2:]

            # Reverse rows from first_half
            first_half = first_half.loc[::-1]
            stack = pd.concat([first_half, second_half], axis=0, ignore_index=True)
            coord_points = stack.to_dict("records")
            typer.secho(f"Using Lednicer format for airfoil {airfoilname}", fg=typer.colors.CYAN)

        # Generate many SDF for airfoil using multiprocessing
        # mpool = multiprocessing.Pool()
        # first_arg = partial(work_on_sdf, points=coord_points, airfoil=airfoil)
        # mpool.map(first_arg, alpha)

    except FileNotFoundError as err:
        typer.secho(f"{err}", fg=typer.colors.RED)

        raise typer.Abort()

