import os

import cv2
import numpy as np
import skfmm
import typer
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


def rendering(name, points, resolution, kind, start, stop, re, ma):
    # Image has 2 dimensions shape
    dimension = 2

    for angle in range(start, stop + 1):
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
        phi = rotating(phi, angle)

        fig, ax = plt.subplots()
        plt.margins(x=0, y=0)
        plt.axis("off")
        ax.set_box_aspect(1)
        plt.tight_layout()

        if kind == "binary":
            colormap = "gray"
        elif kind == "sdf":
            colormap = "plasma"
            phi = skfmm.distance(phi, dx=1, order=2)
        else:
            colormap = "gray"

        plt.imshow(phi, cmap=plt.get_cmap(colormap))

        path = f"out/{name}"
        isExist = os.path.exists(path)

        if not isExist:
            typer.secho("Creating new folder", fg=typer.colors.YELLOW)
            os.makedirs(path)

        plt.savefig(f"{path}/{kind}_{re}_{ma}_{angle}.jpg", bbox_inches="tight", pad_inches=0)
        plt.close("all")
