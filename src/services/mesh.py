import pygmsh
import matplotlib
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def rotate_around(pts, radians, origin=(0, 0)):
    origin = np.array(origin)
    rot = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)]
    ])
    adjusted = pts - origin
    res = origin + np.dot(adjusted, rot)

    return res


def meshing_unstructured(name, angle, points, kind, re, ma):
    airfoil_image_name = f"{name}_{kind}_{re}_{ma}_{angle}.jpg"
    airfoil_image_name = airfoil_image_name.replace(" ", "")

    with pygmsh.geo.Geometry() as geom:
        hole = geom.add_polygon(
            points,
            make_surface=False
        )

        geom.add_circle(np.mean(points, axis=0), 1.0, mesh_size=0.25, holes=[hole.curve_loop])

        mesh = geom.generate_mesh()
        x = mesh.points[:, 0]
        y = mesh.points[:, 1]
        tri = mesh.cells[1].data

        matplotlib.use("Agg")
        plt.style.use("dark_background")
        fig, ax = plt.subplots()
        fig.tight_layout()
        Cmap = np.linspace(0, 1, len(tri))
        ax.tripcolor(x, y, tri, facecolors=Cmap, edgecolors="#4E4E4E", cmap=plt.get_cmap("jet"))
        ax.axes.axis("off")
        ax.axes.margins(x=0, y=0)
        ax.axes.set_box_aspect(1)
        fig.savefig(f"../out/{airfoil_image_name.replace(' ', '')}", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        img = Image.open(f"../out/{airfoil_image_name}")
        img = img.rotate(-1 * angle)
        img = img.resize((146, 146))
        img.save(f"../out/{airfoil_image_name}")


def scale01(z):  # --- transform z to [0 ..1]
    return (z - np.min(z)) / (np.max(z) - np.min(z))


def fStretch(my):
    # control Parameters
    yStretch = 0.5
    yOffset = 2.95
    iy = np.linspace(0, 1, my)
    sy = scale01(np.exp((yStretch * (yOffset + iy)) ** 2))  # build streched

    return sy


def coord(a, b, xi):
    return a + xi * (b - a)


def meshing_ogrid(name, angle, points, kind, re, ma):
    airfoil_image_name = f"{name}_{kind}_{re}_{ma}_{angle}.jpg"
    airfoil_image_name = airfoil_image_name.replace(" ", "")

    # --- outer circle, north boundary ------
    R = 1.25
    px = points[:, 0]
    py = points[:, 1]
    nn = px.size
    cx = np.mean(px)
    cy = np.mean(py)
    phi = np.linspace(0.0, 2.0 * np.pi, nn)
    Rtx = R * np.cos(phi) + cx
    Rty = R * np.sin(phi) + cy

    mx = px.size  # number of points in x-direction
    my = 10  # number of points in y-direction

    # --- initialize the 2D arrays of coordinates ---------
    xk = np.zeros((mx, my))
    yk = np.zeros((mx, my))

    yEta = fStretch(my)  # strechting in y-direction

    # --- assemble the coord arrays ------------------
    for t in range(0, mx):
        xk[t, :] = coord(px[t], Rtx[t], yEta)
        yk[t, :] = coord(py[t], Rty[t], yEta)

    matplotlib.use("Agg")
    plt.style.use("dark_background")
    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.pcolormesh(xk, yk, xk, edgecolors="#4E4E4E", linewidths=0.1, cmap=plt.get_cmap("jet"))
    ax.axes.axis("off")
    ax.axes.margins(x=0, y=0)
    ax.axes.set_box_aspect(1)
    fig.savefig(f"../out/{airfoil_image_name}", bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    img = Image.open(f"../out/{airfoil_image_name}")
    img = img.rotate(-1 * angle)
    img = img.resize((146, 146))
    img.save(f"../out/{airfoil_image_name}")
