import bezier
import numpy as np
from matplotlib import pyplot as plt


def airfoil():
    points = np.array([
        [1, 0.00],  # trailing edge (top)
        [0.76, 0.08],
        [0.52, 0.125],
        [0.25, 0.12],
        [0.1, 0.08],
        [0, 0.03],  # leading edge (top)
        [0, -0.03],  # leading edge (bottom)
        [0.15, -0.08],
        [0.37, -0.01],
        [0.69, 0.04],
        [1, -0.00],  # trailing edge (bottom)
    ])

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.tight_layout()
    ax.axes.axis("off")
    ax.axes.set_aspect("equal", "datalim")
    curve = bezier.Curve(points.T, degree=points.shape[0] - 1)
    curve.plot(100, ax=ax)
    plt.show()
