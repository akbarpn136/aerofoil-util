import pygmsh
import matplotlib
import numpy as np
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


def meshing_ogrid(name, angle, points, kind, re, ma):
    airfoil_image_name = f"{name}_{kind}_{re}_{ma}_{angle}.jpg"
