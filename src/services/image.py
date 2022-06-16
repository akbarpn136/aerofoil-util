import os
import cv2
import skfmm
import matplotlib
import numpy as np
from scipy.io import wavfile
from PIL import Image, ImageDraw, ImageOps
from matplotlib import pyplot as plt

from src.services.mesh import rotate_around
from src.services.spectro import SpectroGraphic


def rotating(draw, angle):
    center = (draw.shape[0] // 2, draw.shape[1] // 2)
    rotate_matrix = cv2.getRotationMatrix2D(
        center=center, angle=-angle, scale=1)
    rotated = cv2.warpAffine(
        src=draw,
        M=rotate_matrix,
        dsize=(draw.shape[1], draw.shape[0]),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1
    )

    return rotated


def rendering_sdf(name, angle, points, resolution, kind, re, ma):
    airfoil_image_name = f"{name}_{kind}_{re}_{ma}_{angle}.jpg"
    airfoil_image_name = airfoil_image_name.replace(' ', '')

    colormap = "jet"
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
    phi = cv2.copyMakeBorder(phi, padding, padding, padding,
                             padding, cv2.BORDER_CONSTANT, value=[-1, -1, -1])
    phi = rotating(phi, angle)
    phi = skfmm.distance(phi, dx=1, order=2)

    matplotlib.use("Agg")
    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.imshow(phi, cmap=plt.get_cmap(colormap))
    ax.axes.axis("off")
    fig.savefig(f"out/{airfoil_image_name}", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def rendering_binary(name, angle, points, kind, re, ma):
    airfoil_image_name = f"{name}_{kind}_{re}_{ma}_{angle}.jpg"
    airfoil_image_name = airfoil_image_name.replace(" ", "")
    px = points[:, 0]
    py = points[:, 1]

    matplotlib.use("Agg")
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.tight_layout()
    ax.plot(px, py)
    ax.fill(px, py, "white", zorder=10)
    ax.axes.axis("off")
    ax.axes.set_aspect("equal", "datalim")
    fig.savefig(f"out/{airfoil_image_name}", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    img = Image.open(f"out/{airfoil_image_name}")
    img = img.rotate(-1 * angle)
    img = img.resize((146, 146))
    img.save(f"out/{airfoil_image_name}")


def rendering_spectro(name, angle, points, kind, re, ma):
    airfoil_image_name = f"{name}_{kind}_{re}_{ma}_{angle}"
    airfoil_image_name = airfoil_image_name.replace(" ", "")
    px = points[:, 0]
    py = points[:, 1]
    fl = f"out/{airfoil_image_name}.jpg"
    au = f"out/{airfoil_image_name}.wav"

    matplotlib.use("Agg")
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.tight_layout()
    ax.plot(px, py)
    ax.fill(px, py, "white", zorder=10)
    ax.axes.axis("off")
    ax.axes.set_aspect("equal", "datalim")
    fig.savefig(fl, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    img = Image.open(fl)
    img = img.rotate(-1 * angle)
    img = img.resize((146, 146))
    img.save(fl)

    sg = SpectroGraphic(path=fl, duration=1, height=35)
    sg.save(wav_file=au)
    Fs, aud = wavfile.read(au)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.specgram(aud, Fs=Fs, cmap=plt.get_cmap("viridis"))
    ax.axes.axis("off")
    fig.savefig(fl, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    img = Image.open(fl)
    img = img.resize((128, 128))
    img.save(fl)

    os.remove(au)

def rendering_stack(name, angle, points, kind, re, ma):
    airfoil_image_name = f"{name}_{kind}_{re}_{ma}_{angle}.jpg"
    airfoil_image_name = airfoil_image_name.replace(" ", "")
    resolution = 8192
    offset = resolution // 2
    divider = 16
    im = Image.new("RGB", (resolution // divider,
                   resolution // divider), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    cmap = matplotlib.cm.get_cmap("jet_r")

    im_re = Image.new("RGB", (resolution // divider,
                   resolution // divider), (0, 0, 0))
    draw_re = ImageDraw.Draw(im_re)

    if re == 100000:
        cmap_re = matplotlib.cm.get_cmap("Set3")
    elif re == 200000:
        cmap_re = matplotlib.cm.get_cmap("tab20b")
    elif re == 500000:
        cmap_re = matplotlib.cm.get_cmap("tab20c")
    else:
        cmap_re =  matplotlib.cm.get_cmap("Paired")

    dt = rotate_around(points, np.radians(angle))
    dt[:, 1] *= -1

    _draw_airfoil(dt, draw, cmap, resolution, offset, divider)
    _draw_airfoil(dt, draw_re, cmap_re, resolution, offset, divider)

    im = im.resize((78, 78))
    im_re = im_re.resize((78, 78))

    img = Image.blend(im, im_re, 0.5)
    img.save(f"out/{airfoil_image_name}", quality="maximum")

def _draw_airfoil(dt, draw, cmap, resolution, offset, divider):
    for scale in range(resolution, 0, -64):
        rgba = cmap(scale / resolution)
        pts = dt - dt.mean(axis=0)
        pts *= scale
        pts += dt.mean(axis=0) + offset // divider
        pts = np.round(pts).astype(int)
        pts = tuple(map(tuple, pts))

        draw.polygon(
            pts,
            fill=(
                int(255 * rgba[0]),
                int(255 * rgba[1]),
                int(255 * rgba[2])
            ),
            # width=3
        )
