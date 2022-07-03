from __future__ import annotations

from random import shuffle

import axi
import click
import numpy as np
from axi import Drawing
from shapely.geometry import LineString, MultiLineString, Polygon, GeometryCollection

rng = np.random.default_rng()


class GrowthLine:
    def __init__(self, anchor: tuple[float, float], speed: float, angle: float):
        self.anchor = np.array(anchor)
        self.endpoint = np.array(anchor)
        self.velocity = speed * np.array([np.cos(angle), np.sin(angle)])

    @property
    def linestring(self) -> LineString:
        return LineString([self.anchor, self.endpoint])

    @property
    def next_linestring(self) -> LineString:
        return LineString([self.anchor, self.endpoint + self.velocity])

    def grow(self) -> None:
        self.endpoint += self.velocity


class RadialSun:
    def __init__(self, center: tuple[float, float], spokes: int, center_radius: float):
        center = np.array(center)
        offset = rng.uniform(0, 2 * np.pi / spokes)
        self.spokes = [
            GrowthLine(
                center
                + center_radius
                * np.array([np.cos(angle + offset), np.sin(angle + offset)]),
                0.01,
                angle,
            )
            for angle in np.linspace(0, 2 * np.pi, spokes, endpoint=False)
        ]

    @property
    def linestrings(self) -> list[LineString]:
        return [spoke.linestring for spoke in self.spokes]

    @property
    def central_disc(self) -> Polygon:
        return Polygon([spoke.anchor for spoke in self.spokes])

    def grow(self, obstacle: MultiLineString, sep: float) -> bool:
        growth = False
        for spoke in self.spokes:
            if spoke.next_linestring.distance(obstacle) >= sep:
                spoke.grow()
                growth = True
        return growth

    @property
    def drawing(self) -> Drawing:
        return Drawing([(spoke.anchor, spoke.endpoint) for spoke in self.spokes])


def radial_sun_drawing(
    width: float, height: float, suns: int, spokes: int, radius: float
) -> list[Drawing]:
    boundary = LineString([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])
    r = min(width, height) / 4
    centers = [
        (width / 2 + r * np.cos(angle), height / 2 + r * np.sin(angle))
        for angle in np.linspace(0, 2 * np.pi, suns, endpoint=False)
    ]
    suns = [RadialSun(center, spokes, radius) for center in centers]
    obstacles = {
        sun: GeometryCollection([boundary, *[x.central_disc for x in suns if x != sun]])
        for sun in suns
    }
    while True:
        growth = False
        for sun in suns:
            growth = growth or sun.grow(obstacles[sun], 0.01)
        if not growth:
            break
    drawings = [sun.drawing for sun in suns]
    return drawings


@click.command()
@click.option("-t", "--test", is_flag=True)
@click.option("-w", "--width", prompt=True, type=float, default=8)
@click.option("-h", "--height", prompt=True, type=float, default=8)
@click.option("-m", "--margin", prompt=True, type=float, default=0.5)
@click.option("-s", "--spokes", prompt=True, type=int, default=100)
@click.option("-n", "--suns", prompt=True, type=int)
@click.option("-r", "--radius", prompt=True, type=float, default=1.0)
def main(
    test: bool,
    width: float,
    height: float,
    margin: float,
    spokes: int,
    suns: int,
    radius: float,
):
    drawings = radial_sun_drawing(
        width - 2 * margin, height - 2 * margin, suns, spokes, radius
    )
    drawings = Drawing.multi_scale_to_fit(drawings, width, height, margin)

    if test or axi.device.find_port() is None:
        im = Drawing.render_layers(drawings, bounds=(0, 0, 8, 8))
        im.write_to_png("suns.png")
    else:
        axi.draw_layers(drawings)


if __name__ == "__main__":
    main()
