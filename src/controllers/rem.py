import os
import shutil
import typer
import pandas as pd

from . import app


@app.command()
def clean(
        path: str = typer.Option(
            "out",
            "--path",
            "-p",
            help="Path to airfoil representation (binary or sdf) after using get command",
        ),
        filename: str = typer.Option(
            "out.csv",
            "--filename",
            "-f",
            help="Filename for output csv file after using get command",
        ),
):
    """
    Function to remove unused airfoil geometry representation.
    """
    df = pd.read_csv(f"{filename}")
    images = df["img"].tolist()
    isExist = os.path.exists("tmp")

    if not isExist:
        os.makedirs("tmp")

    for img in images:
        try:
            os.rename(f"{path}/{img}", f"tmp/{img}")
        except FileNotFoundError:
            pass

    try:
        shutil.rmtree("out")
        os.rename("tmp", f"{path}")
    except FileNotFoundError:
        typer.secho("File not found. Skipped", fg=typer.colors.RED)
    except OSError as err:
        typer.secho(f"{err.strerror}", fg=typer.colors.RED)
