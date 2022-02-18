import os
import typer
import pygmsh
import numpy as np
from functools import partial
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count


def rotate_around(pts, radians, origin=(0, 0)):
    origin = np.array(origin)
    rot = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)]
    ])
    adjusted = pts - origin
    res = origin + np.dot(adjusted, rot)

    return res


def to_mesh(name, angle, points, kind, re, ma):
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
        # z = mesh.points[:, 2]
        tri = mesh.cells[1].data

        plt.style.use("dark_background")
        _, ax = plt.subplots()
        plt.triplot(x, y, tri, color="white")
        plt.margins(x=0, y=0)
        plt.axis("off")
        ax.set_box_aspect(1)
        plt.tight_layout()
        plt.savefig("naca.jpg", bbox_inches="tight", pad_inches=0)
        plt.savefig(f"out/{name}_{kind}_{re}_{ma}_{angle}.jpg", bbox_inches="tight", pad_inches=0)
        plt.close("all")


def meshing(name, points, kind, start, stop, re, ma):
    path = "out"
    isExist = os.path.exists(path)

    if not isExist:
        typer.secho("Creating new folder", fg=typer.colors.YELLOW)
        os.makedirs(path)

    with Pool(cpu_count()) as pool:
        partial_func = partial(
            to_mesh,
            name=name,
            points=points,
            kind=kind,
            re=re,
            ma=ma
        )
        pool.map(partial_func, range(start, stop + 1))
