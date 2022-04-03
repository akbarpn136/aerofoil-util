import os
import itertools
import pandas as pd
from mpire import WorkerPool

from src.services.mesh import meshing_ogrid, meshing_unstructured
from src.services.image import rendering


def to_img(*payload):
    name = payload[0]
    angle = payload[1]
    resol = payload[2]
    kn = payload[3]
    rey = payload[4]
    mac = payload[5]
    pat = payload[6]

    try:
        ddf = pd.read_csv(f"{pat}/{name}.dat", delim_whitespace=True, header=None, skiprows=1)
        ddf.columns = ["x", "y"]

        val = ddf.loc[0, "x"]
        val = int(round(val, 0))
        if val != 1:
            first_idx = ddf.loc[ddf.x == int(round(1, 0))].index.values
            first_half = ddf.iloc[:first_idx[0] + 1]
            second_half = ddf.iloc[first_idx[0] + 2:]

            # Reverse rows from first_half
            first_half = first_half.loc[::-1]
            ddf = pd.concat([first_half, second_half], axis=0, ignore_index=True)

        if kn != "mesh":
            rendering(name, angle, ddf.to_dict("records"), resol, kn, rey, mac)
        elif kn == "mesh":
            meshing_ogrid(name, angle, ddf.to_numpy(), kn, rey, mac)
        else:
            print("Invalid kind. Only binary, mesh or sdf available.")

    except FileNotFoundError as err:
        print(f"{err}")


if __name__ == "__main__":
    kind = "mesh"
    path = "../out"
    foil = "../foil"
    filename = "../out.csv"
    ma = 0.0
    re = 500000
    resolution = 1024
    angle_start = -20
    angle_stop = 20

    df = pd.read_csv(f"{filename}")
    names = df["name"].unique()
    angles = range(angle_start, angle_stop + 1)
    resolution = [resolution]
    knd = [kind]
    re = [re]
    ma = [ma]
    pth = [foil]
    paramlist = list(itertools.product(names, angles, resolution, knd, re, ma, pth))
    isExist = os.path.exists(path)

    if not isExist:
        print("Creating new folder")
        os.makedirs(path)

    with WorkerPool(n_jobs=os.cpu_count()) as pool:
        pool.map(to_img, paramlist, progress_bar=True)
