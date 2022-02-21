import os
import glob
import time
import typer
import pandas as pd

from . import app


@app.command()
def get(
        path: str = typer.Option(
            "aero",
            help="Path where aerodyanmic data is saved from XFOIL.",
        ),
        kind: str = typer.Option(
            "sdf",
            "--kind",
            "-k",
            help="Airfoil geometry representation: binary, mesh or sdf.",
        ),
):
    """
    Function to get data airfoil aerodynamic
    extracted from XFOIL.
    """
    start = time.time()
    payload = []
    all_files = glob.glob(f"{path}/*.txt")
    for filename in all_files:
        df = _process(path, filename, kind)
        payload.append(df)

    dt = pd.concat(payload, axis=0)
    filename = "out.csv"
    files_present = glob.glob(filename)

    if len(files_present) != 0:
        os.remove(filename)

    dt.to_csv(filename, index=False)
    typer.secho(f"Successfully save data. Took {time.time() - start} s", fg=typer.colors.GREEN)


def _process(path, filename, kind):
    file = filename.replace(path, "") \
        .replace("\\", "") \
        .replace("/", "")

    split = file.split("_")
    dat_name = split[0]
    name = dat_name.replace(" ", "")
    re = split[2].replace("Re", "")
    re = int(float(re) * 10 ** 6)
    ma = split[3].replace("M", "")
    ma = float(ma)
    image = f"{name}_{kind}_{re}_{ma}"

    df = pd.read_fwf(
        filename,
        skiprows=11,
        header=None,
        names=["alpha", "cl", "cd", "cdp", "cm", "TopXtr", "BotXtr", "Cpmin", "Chinge", "XCp"]
    )

    df = df[["alpha", "cl", "cd", "cm"]]
    df["name"] = dat_name
    df["re"] = re
    df["ma"] = ma
    df["img"] = df.apply(lambda row: f"{image}_{int(row['alpha'])}.jpg", axis=1)

    return df
