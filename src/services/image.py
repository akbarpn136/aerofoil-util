import cv2
import skfmm
import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def rotating(draw, angle):
    center = (draw.shape[0] // 2, draw.shape[1] // 2)
    rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-angle, scale=1)
    rotated = cv2.warpAffine(
        src=draw,
        M=rotate_matrix,
        dsize=(draw.shape[1], draw.shape[0]),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1
    )

    return rotated


def rendering(name, angle, points, resolution, kind, re, ma):
    airfoil_image_name = f"{name}_{kind}_{re}_{ma}_{angle}"
    dimension = 2  # Image has 2 dimensions shape
    padding = 110
    offset_y = resolution // 2
    phi = -1 * np.ones((resolution, resolution, 1), dtype="uint8")
    airfoils = np.empty((0, dimension), int)

    for point in points:
        airfoils = np.append(
            airfoils,
            np.array([[
                round(resolution * point["x"]),
                round(resolution * point["y"] + offset_y)
            ]]),
            axis=0
        )

    airfoils = airfoils.reshape((-1, 1, 2))
    cv2.fillPoly(phi, [airfoils], (255, 255, 255), lineType=cv2.LINE_AA)
    phi = cv2.flip(phi, 0)
    phi = cv2.copyMakeBorder(phi, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[-1, -1, -1])
    phi = rotating(phi, angle)

    colormap = "gray"
    if kind == "sdf":
        colormap = "jet"
        phi = skfmm.distance(phi, dx=1, order=2)

    matplotlib.use("Agg")
    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.imshow(phi, cmap=plt.get_cmap(colormap))
    ax.axes.axis("off")
    fig.savefig(f"../out/{airfoil_image_name.replace(' ', '')}.jpg", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
