import os
import itertools
import pandas as pd
from mpire import WorkerPool
from src.controllers.gen import to_img


if __name__ == "__main__":
    kind = "sdf"
    path = "../out"
    foil = "../foil"
    filename = "../out.csv"
    ma = 0.0
    re = 500000
    resolution = 1024
    angle_start = -14
    angle_stop = 14

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
