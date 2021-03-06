from __future__ import annotations

from random import shuffle

import axi
import click
import numpy as np
from axi import Drawing
from shapely.affinity import rotate
from shapely.geometry import (
    LineString,
    MultiLineString,
    Polygon,
    GeometryCollection,
    Point,
)

rng = np.random.default_rng()


def nearest_point(p: Point, c: GeometryCollection) -> Point:
    if isinstance(c, Point):
        return c
    points = []
    if isinstance(c, LineString):
        points += [Point(coord) for coord in c.coords]
    else:
        for g in c.geoms:
            if isinstance(g, Point):
                points.append(g)
            else:
                for x in g.coords:
                    points.append(Point(x))
    best = None
    best_dist = float("inf")
    for elem in points:
        if (dist := elem.distance(p)) < best_dist:
            best_dist = dist
            best = elem
    return best


def radial_drawing(
    width: float,
    height: float,
    shapes: list[Polygon],
    n_lines: int,
) -> list[Drawing]:
    boundary = LineString([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    drawings = []
    for i, shape in enumerate(shapes):
        obstacles = GeometryCollection([boundary, *[x for x in shapes if x != shape]])
        lines = []
        center = np.array(shape.centroid.coords[0])
        angle_offset = i * 2 * np.pi / n_lines / len(shapes)
        for angle in np.linspace(
            angle_offset, 2 * np.pi + angle_offset, n_lines, endpoint=False
        ):
            velocity = 1000 * np.array([np.cos(angle), np.sin(angle)])
            ray = LineString([center, center + velocity])
            start = shape.exterior.intersection(ray)
            end = nearest_point(start, obstacles.intersection(ray))
            if None in (start, end) or start.distance(end) < 1e-5:
                continue
            try:
                lines.append([start.coords[0], end.coords[0]])
            except NotImplementedError:
                print("uh oh")
        drawings.append(Drawing(lines))
    return drawings


def n_gon(n: int, x: float, y: float, r: float) -> Polygon:
    return Polygon(
        [
            (x + r * np.cos(angle), y + r * np.sin(angle))
            for angle in np.linspace(0, 2 * np.pi, n, endpoint=False)
        ]
    )


def x_shape(
    x: float, y: float, r_inner: float, r_outer: float, angle_offset: float
) -> Polygon:
    outer_angles = [
        np.pi / 4 - angle_offset,
        np.pi / 4 + angle_offset,
        3 * np.pi / 4 - angle_offset,
        3 * np.pi / 4 + angle_offset,
        5 * np.pi / 4 - angle_offset,
        5 * np.pi / 4 + angle_offset,
        7 * np.pi / 4 - angle_offset,
        7 * np.pi / 4 + angle_offset,
    ]
    outer_points = [
        (x + r_outer * np.cos(a), y + r_outer * np.sin(a)) for a in outer_angles
    ]
    inner_angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    inner_points = [
        (x + r_inner * np.cos(a), y + r_inner * np.sin(a)) for a in inner_angles
    ]
    return Polygon(
        [
            outer_points[0],
            outer_points[1],
            inner_points[1],
            outer_points[2],
            outer_points[3],
            inner_points[2],
            outer_points[4],
            outer_points[5],
            inner_points[3],
            outer_points[6],
            outer_points[7],
            inner_points[0],
        ]
    )


def random_n_gons(
    width: float,
    height: float,
    min_r: float,
    max_r: float,
    shapes: int,
    separation: float,
    margin: float,
) -> list[Polygon]:
    out: list[Polygon] = []
    sides = rng.choice([3, 4, 6, 200], shapes, False)
    while len(out) < shapes:
        r = rng.uniform(min_r, max_r)
        shape = n_gon(
            sides[len(out)],
            rng.uniform(margin + r, width - r - margin),
            rng.uniform(margin + r, height - r - margin),
            r,
        )
        shape = rotate(shape, rng.uniform(0, 360), "centroid")
        good = True
        for other in out:
            if other.distance(shape) < separation:
                good = False
                break
        if good:
            out.append(shape)
    return out


@click.command()
@click.option("-t", "--test", is_flag=True)
@click.option("-w", "--width", prompt=True, type=float, default=7.87)
@click.option("-h", "--height", prompt=True, type=float, default=7.87)
@click.option("-m", "--margin", prompt=True, type=float, default=0.5)
@click.option("-s", "--spokes", prompt=True, type=int, default=100)
@click.option("-n", "--emitters", prompt=True, type=int)
@click.option("-minr", "--min_radius", prompt=True, type=float, default=0.5)
@click.option("-maxr", "--max_radius", prompt=True, type=float, default=1.5)
@click.option("-sep", "--separation", prompt=True, type=float, default=0.5)
def main(
    test: bool,
    width: float,
    height: float,
    margin: float,
    spokes: int,
    emitters: int,
    min_radius: float,
    max_radius: float,
    separation: float,
):
    dw, dh = width - 2 * margin, height - 2 * margin
    emitters = random_n_gons(
        dw, dh, min_radius, max_radius, emitters, separation, margin
    )
    drawings = radial_drawing(dw, dh, emitters, spokes)
    drawings = [drawing.repeat().sort_paths() for drawing in drawings]
    drawings = Drawing.multi_scale_to_fit(drawings, width, height, margin)

    if test or axi.device.find_port() is None:
        im = Drawing.render_layers(drawings, bounds=(0, 0, width, height))
        im.write_to_png("suns.png")
    else:
        axi.draw_layers(drawings)


if __name__ == "__main__":
    main()
