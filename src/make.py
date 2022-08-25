import pathlib
import bezier
import itertools
import numpy as np
from matplotlib import pyplot as plt


def airfoil_bezier(num=100):
    rng = np.random.default_rng(2022)

    for i in range(num):
        randpoints = np.array([
            [1, 0.00],  # trailing edge (top)
            [0.76, rng.random.uniform(0.04, 0.1)],
            [0.52, rng.random.uniform(0.1, 0.2)],
            [0.25, rng.random.uniform(0.1, 0.2)],
            [0.1, rng.random.uniform(0, 0.1)],
            [0, rng.random.uniform(0, 0.06)],  # leading edge (top)
            [0, -1 * rng.random.uniform(0, 0.06)],  # leading edge (bottom)
            [0.15, -1 * rng.random.uniform(0, 0.1)],
            [0.37, -1 * rng.random.uniform(0, 0.1)],
            [0.69, rng.random.uniform(-0.04, 0.04)],
            [1, 0.00],  # trailing edge (bottom)
        ])

        curve = bezier.Curve(randpoints.T, degree=randpoints.shape[0] - 1)
        eval_range = np.linspace(0, 1, 100)
        result = curve.evaluate_multi(eval_range)
        np.savetxt(f"foil/Aerofoil{i + 1}.dat", result.T,
                   header=f"Aerofoil{i + 1}", comments="")


def bump(x, location, magnitude, n=100):
    width = 2
    sf = 1  # Scale factor

    y = np.zeros(len(x))
    m = np.log(0.5) / np.log(location)

    for i in range(len(x)):
        for j in range(len(m)):
            y[i] += sf * magnitude[j] * np.sin(np.pi * x[i] ** m[j]) ** width

    return y


if __name__ == "__main__":
    n = 100  # Number of points
    x = np.linspace(0, 1, n, endpoint=True)
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
    ])

    nodes = 15
    bm_upper1 = np.linspace(0.01, 0.03, nodes)
    bm_upper2 = np.linspace(0.03, 0.04, nodes)
    bm_upper3 = np.linspace(0.04, 0.09, nodes)
    bm_upper4 = np.linspace(0.02, 0.05, nodes)

    for i in range(loc.shape[0]):
        for j in range(nodes):
            bm = [bm_upper1[j], bm_upper2[j], bm_upper3[j], bm_upper4[j]]
            upper = bump(x, loc[i], bm, n)

            plt.plot(x, upper)

    plt.show()
