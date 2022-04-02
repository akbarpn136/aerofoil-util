import pygmsh
import matplotlib
import numpy as np
import matplotlib.colors as mclr
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

    with pygmsh.geo.Geometry() as geom:
        hole = geom.add_polygon(
            rotate_around(
                points,
                np.radians(angle),
                origin=np.mean(points, axis=0)
            ),
            make_surface=False
        )

        geom.add_polygon(
            [
                [-0.1, 0.6],
                [-0.1, -0.6],
                [1.1, -0.6],
                [1.1, 0.6],
            ],
            mesh_size=0.1,
            holes=[hole.curve_loop]
        )

        mesh = geom.generate_mesh()
        x = mesh.points[:, 0]
        y = mesh.points[:, 1]
        tri = mesh.cells[1].data

        matplotlib.use("Agg")
        plt.style.use("dark_background")
        fig, ax = plt.subplots()
        fig.tight_layout()
        ax.triplot(x, y, tri, color="white")
        ax.axes.axis("off")
        ax.axes.margins(x=0, y=0)
        ax.axes.set_box_aspect(1)
        fig.savefig(f"../out/{airfoil_image_name.replace(' ', '')}", bbox_inches="tight", pad_inches=0)
        plt.close(fig)


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

    # --- outer circle, north boundary ------
    R = 1.25
    px = points[:, 0]
    py = points[:, 1]
    nn = px.size
    cx = np.mean(px)
    cy = np.mean(py)
    # phi = np.linspace(0.02,1.98*np.pi, nn) #for demonstration only, (0.0,2.0*np.pi, nn) else
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
    Cmap = mclr.ListedColormap(["white", "white"])
    ax.pcolormesh(xk, yk, np.zeros_like(xk), edgecolors="#4E4E4E", linewidths=0.1, cmap=Cmap)
    ax.axes.axis("off")
    ax.axes.margins(x=0, y=0)
    ax.axes.set_box_aspect(1)
    fig.savefig(f"../out/{airfoil_image_name.replace(' ', '')}", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
