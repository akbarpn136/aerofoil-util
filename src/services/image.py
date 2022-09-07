import os
import cv2
import skfmm
import matplotlib
import numpy as np
from scipy.io import wavfile
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm, pyplot as plt

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


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def rendering_sdf(name, angle, points, resolution, kind, re, ma):
    airfoil_image_name = f"{name}_{kind}_{re}_{ma}_{angle}.jpg"
    airfoil_image_name = airfoil_image_name.replace(' ', '')

    padding = 110
    phi = -1 * np.ones((resolution, resolution, 1), dtype="uint8")
    points = points * 512
    points[:, 0] += 256
    points[:, 1] += 512

    cv2.fillPoly(phi, [points.astype(np.int)], (255, 255, 255), lineType=cv2.LINE_AA)
    phi = cv2.flip(phi, 0)
    phi = cv2.copyMakeBorder(phi, padding, padding, padding,
                             padding, cv2.BORDER_CONSTANT, value=[-1, -1, -1])
    phi = rotating(phi, angle)
    phi = skfmm.distance(phi, dx=1, order=2)
    im = Image.fromarray(np.uint8(cm.jet(normalize_data(phi)) * 255))
    im = im.convert("RGB")
    im = im.resize((78, 78))

    im_re = Image.new("RGB", (128, 128), (128, 128, 128))
    draw_re = ImageDraw.Draw(im_re)
    font = ImageFont.truetype("arial.ttf", 21)
    draw_re.text((30, 78), str(re), font=font, fill=(255, 255, 255, 64))

    im_re = im_re.resize((78, 78))
    img = Image.blend(im, im_re, 0.5)

    img.save(f"out/{airfoil_image_name}", quality="maximum")


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
    np.seterr(divide = "ignore")
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

    sg = SpectroGraphic(path=fl, duration=20, height=35)
    sg.save(wav_file=au)
    Fs, aud = wavfile.read(au)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.specgram(aud, Fs=Fs, cmap=plt.get_cmap("plasma"))
    ax.axes.axis("off")
    fig.savefig(fl, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    img = Image.open(fl)
    img = img.resize((78, 78))
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

    if re == 300000:
        cmap_re = matplotlib.cm.get_cmap("Set3")
    elif re == 400000:
        cmap_re = matplotlib.cm.get_cmap("tab20b")
    elif re == 500000:
        cmap_re = matplotlib.cm.get_cmap("tab20c")
    else:
        cmap_re = matplotlib.cm.get_cmap("Paired")

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
