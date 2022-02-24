import cv2
import skfmm
import numpy as np
import pandas as pd
from plotly import express as px


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
    airfoil_image_name = f"{name}_{kind}_{re}_{ma}_{angle}.csv"
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

    if kind == "sdf":
        phi = skfmm.distance(phi, dx=1, order=2)

    fig = px.imshow(phi, color_continuous_scale="turbo")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(t=0, b=0, l=0, r=0),
        width=128, height=128
    )
    fig.write_image(f"out/{airfoil_image_name.replace(' ', '')}.jpg")
