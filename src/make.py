import os
import csv
import numpy as np
import pandas as pd
from geomdl import BSpline
from geomdl import utilities


def genfoil_curve(id, coords, degree=2, sample=100):
    name = "Aerofoil"
    curve = BSpline.Curve()

    # Set up the Bezier curve
    curve.degree = degree
    curve.ctrlpts = coords

    # Auto-generate knot vector
    curve.knotvector = utilities.generate_knot_vector(
        curve.degree,
        len(curve.ctrlpts)
    )

    # Set evaluation delta
    curve.sample_size = sample

    with open(f"foil/{name}{5000 + id}.dat", "w") as f:
        f.write(f"{name}{5000 + id}\n")

        write = csv.writer(f, delimiter=" ", lineterminator="\n")
        write.writerows(curve.evalpts)


def airfoil_bezier(offset=0):
    df = pd.read_csv('make.csv')

    for idx, row in df.iterrows():
        coords = (
            (row['x0'], row['y0']),
            (row['x1'], row['y1']),
            (row['x2'], row['y2']),
            (row['x3'], row['y3']),
            (row['x4'], row['y4']),
            (row['x5'], row['y5']),
            (row['x6'], row['y6']),
            (row['x7'], row['y7']),
            (row['x8'], row['y8']),
            (row['x9'], row['y9']),
            (row['x10'], row['y10']),
            (row['x11'], row['y11']),
            (row['x12'], row['y12']),
        )

        genfoil_curve(idx + offset, coords, 3)

    cur = os.getcwd()
    foils_dir = os.path.join(cur, 'foil')

    for foil in os.listdir(foils_dir):
        if foil.endswith(".dat"):
            fname = os.path.join(foils_dir, foil)
            df = pd.read_csv(fname, skiprows=1,
                             sep=' ',
                             header=None,
                             names=['x', 'y'])

            df['x'] = (df['x'] - df['x'].min()) / \
                (df['x'].max() - df['x'].min())

            df.columns = [foil.replace('.dat', ''), '']
            df.to_csv(fname, index=False, sep=' ')


if __name__ == "__main__":
    airfoil_bezier(offset=40)
