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


def airfoil_bezier(num=10):
    up11 = np.linspace(0.01698, 0.02, num, dtype=np.float16)
    up12 = np.linspace(0.05805, 0.06, num, dtype=np.float16)
    up13 = np.linspace(0.12363, 0.15, num, dtype=np.float16)
    up14 = np.linspace(0.07339, 0.1, num, dtype=np.float16)
    up15 = np.linspace(0.01515, 0.02, num, dtype=np.float16)
    up21 = np.linspace(0.02, 0.03, num, dtype=np.float16)
    up22 = np.linspace(0.07, 0.1, num, dtype=np.float16)
    up23 = np.linspace(0.14, 0.15, num, dtype=np.float16)
    up24 = np.linspace(0.08, 0.1, num, dtype=np.float16)
    up25 = np.linspace(0.02, 0.03, num, dtype=np.float16)

    low11 = np.linspace(-0.01213, -0.02, num, dtype=np.float16)
    low12 = np.linspace(-0.01975, -0.03, num, dtype=np.float16)
    low13 = np.linspace(0.00720, 0.02, num, dtype=np.float16)
    low14 = np.linspace(0.02549, 0.05, num, dtype=np.float16)
    low15 = np.linspace(0.00952, 0.01, num, dtype=np.float16)
    low21 = np.linspace(-0.02, -0.03, num, dtype=np.float16)
    low22 = np.linspace(-0.023, -0.033, num, dtype=np.float16)
    low23 = np.linspace(0.01, -0.01, num, dtype=np.float16)
    low24 = np.linspace(0.04, 0.01, num, dtype=np.float16)
    low25 = np.linspace(0.015, -0.01, num, dtype=np.float16)

    for idx in range(num):
        coord_upper1 = (
            (1.000, 0),
            (0.95916, up11[idx]),
            (0.77901, up12[idx]),
            (0.39420, up13[idx]),
            (0.07526, up14[idx]),
            (0.00185, up15[idx]),
            (0.000, 0.000),
            (0.00077, -0.01213),
            (0.04964, -0.01975),
            (0.24755, 0.00720),
            (0.50960, 0.02549),
            (0.88475, 0.00952),
            (1.000, 0),
        )

        coord_upper2 = (
            (1.000, 0),
            (0.95916, up21[idx]),
            (0.77901, up22[idx]),
            (0.39420, up23[idx]),
            (0.07526, up24[idx]),
            (0.00185, up25[idx]),
            (0.000, 0.000),
            (0.00077, -0.01213),
            (0.04964, -0.01975),
            (0.24755, 0.00720),
            (0.50960, 0.02549),
            (0.88475, 0.00952),
            (1.000, 0),
        )

        genfoil_curve(idx, coord_upper1, 3)
        genfoil_curve(idx + num, coord_upper2, 3)

    for idx in range(num):
        coord_lower1 = (
            (1.000, 0),
            (0.95916, 0.01698),
            (0.77901, 0.05805),
            (0.39420, 0.12363),
            (0.07526, 0.07339),
            (0.00185, 0.01515),
            (0.000, 0.000),
            (0.00077, low11[idx]),
            (0.04964, low12[idx]),
            (0.24755, low13[idx]),
            (0.50960, low14[idx]),
            (0.88475, low15[idx]),
            (1.000, 0),
        )

        coord_lower2 = (
            (1.000, 0),
            (0.95916, 0.01698),
            (0.77901, 0.05805),
            (0.39420, 0.12363),
            (0.07526, 0.07339),
            (0.00185, 0.01515),
            (0.000, 0.000),
            (0.00077, low21[idx]),
            (0.04964, low22[idx]),
            (0.24755, low23[idx]),
            (0.50960, low24[idx]),
            (0.88475, low25[idx]),
            (1.000, 0),
        )

        genfoil_curve(idx + num + num, coord_lower1, 3)
        genfoil_curve(idx + num + num + num, coord_lower2, 3)


def bump(x, location, magnitude, n=100):
    width = 2
    sf = 1  # Scale factor

    y = np.zeros(len(x))
    m = np.log(0.5) / np.log(location)

    for i in range(len(x)):
        for j in range(len(m)):
            y[i] += sf * magnitude[j] * np.sin(np.pi * x[i] ** m[j]) ** width

    return y


def airfoil_bump():
    n = 46  # Number of points
    x = np.linspace(0, 1, n, endpoint=True, dtype=np.float16)
    # p = pathlib.Path(__file__)
    # my_data = np.genfromtxt(f"{p.parent}/Aerofoil1000.dat",
    #                         delimiter=",")

    # Bump locations
    loc = np.array([
        [0.15, 0.18, 0.37, 0.6],
        [0.11, 0.18, 0.35, 0.58],
        [0.18, 0.2, 0.38, 0.66],
        [0.17, 0.25, 0.32, 0.55],
        [0.12, 0.18, 0.41, 0.7],
        [0.13, 0.15, 0.33, 0.66],
        [0.14, 0.2, 0.45, 0.75],
        [0.16, 0.23, 0.34, 0.56],
        [0.13, 0.26, 0.41, 0.64],
        [0.15, 0.3, 0.43, 0.73],
    ], dtype=np.float16)

    nodes = 15
    bm_upper1 = np.linspace(0.01, 0.03, nodes, dtype=np.float16)
    bm_upper2 = np.linspace(0.03, 0.04, nodes, dtype=np.float16)
    bm_upper3 = np.linspace(0.04, 0.09, nodes, dtype=np.float16)
    bm_upper4 = np.linspace(0.02, 0.05, nodes, dtype=np.float16)

    foils = []
    for i in range(loc.shape[0]):
        for j in range(nodes):
            bm = [bm_upper1[j], bm_upper2[j], bm_upper3[j], bm_upper4[j]]

            upper = bump(x, loc[i], bm, n)
            up = np.flip(
                np.vstack((x, upper)).T,
                axis=0
            )
            up = up[1:, :]

            lower = bump(
                x,
                np.array((0.08, 0.11, 0.28, 0.52), dtype=np.float16),
                np.array((-0.007, -0.015, 0.01, 0.018), dtype=np.float16)
            )
            low = np.vstack((x, lower)).T
            low = low[1:-1, :]

            foil = np.vstack((up, low))
            foils.append(foil)

    for i in range(len(foils)):
        np.savetxt(
            f"foil/Aerofoil{i + 1000}.dat",
            foils[i],
            header=f"Aerofoil{i + 1000}",
            comments=""
        )


def airfoil_bezier_io(offset=0):
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
    airfoil_bezier_io(offset=40)
