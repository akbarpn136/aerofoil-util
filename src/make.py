import bezier
import numpy as np
from matplotlib import pyplot as plt


def airfoil(num=100):
    for i in range(num):
        randpoints = np.array([
            [1, 0.00],  # trailing edge (top)
            [0.76, np.random.uniform(0.04, 0.1)],
            [0.52, np.random.uniform(0.1, 0.2)],
            [0.25, np.random.uniform(0.1, 0.2)],
            [0.1, np.random.uniform(0, 0.1)],
            [0, np.random.uniform(0, 0.06)],  # leading edge (top)
            [0, -1 * np.random.uniform(0, 0.06)],  # leading edge (bottom)
            [0.15, -1 * np.random.uniform(0, 0.1)],
            [0.37, -1 * np.random.uniform(0, 0.1)],
            [0.69, np.random.uniform(-0.04, 0.04)],
            [1, 0.00],  # trailing edge (bottom)
        ])

        curve = bezier.Curve(randpoints.T, degree=randpoints.shape[0] - 1)
        eval_range = np.linspace(0, 1, 100)
        result = curve.evaluate_multi(eval_range)
        np.savetxt(f"foil/aerofoil{i + 1}.dat", result.T, header=f"Aerofoil{i + 1}", comments="")


if __name__ == "__main__":
    airfoil()
