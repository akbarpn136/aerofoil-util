import os
import typer

from controllers import app


@app.command()
def generate(
        airfoilname: str = typer.Argument(
            ...,
            help="Airfoil name code such as naca2412",
            metavar="airfoilname"
        ),
        filename: str = typer.Argument(
            ...,
            help="Airfoil coordinate file in csv from current directory",
            metavar="filename"
        ),
):
    """
    Function to generate airfoil geometry representation
    such as binary, mesh-like or SDF (Signed Distance Fields).
    """
    typer.secho(f"{os.getcwd()}", fg=typer.colors.MAGENTA)
