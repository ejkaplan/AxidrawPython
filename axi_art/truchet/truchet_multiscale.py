from random import choice
from typing import NamedTuple

import axi
import click
import numpy as np
from axi import Drawing
from axi.paths import Path
from shapely.geometry import MultiLineString, Polygon, LineString


class Box(NamedTuple):
    r: int
    c: int
    size: int


class Grid:
    def __init__(self, width: int, height: int):
        self.grid = np.zeros((height, width))
        self._boxes = []

    @property
    def width(self) -> int:
        return self.grid.shape[1]

    @property
    def height(self) -> int:
        return self.grid.shape[0]

    @property
    def boxes(self) -> list[Box]:
        out = self._boxes.copy()
        singletons = np.where(self.grid == 0)
        return out + [Box(*coord, 1) for coord in zip(*singletons)]

    def randomly_merge(self, size: int) -> bool:
        legal_placements = [
            (r, c)
            for r in range(self.height - size + 1)
            for c in range(self.width - size + 1)
        ]
        legal_placements = list(
            filter(
                lambda x: np.all(
                    self.grid[x[0] : x[0] + size, x[1] : x[1] + size] == 0
                ),
                legal_placements,
            )
        )
        if len(legal_placements) == 0:
            return False
        coord = choice(legal_placements)
        self.grid[coord[0] : coord[0] + size, coord[1] : coord[1] + size] = 1
        self._boxes.append(Box(coord[0], coord[1], size))
        return True


def render(grid: Grid, p_turn: float) -> list[Drawing]:
    out = Drawing()
    highlights = Drawing()
    for box in grid.boxes:
        if np.random.random() < p_turn:
            tile = truchet_corner_circles(box.size, 2 * box.size)
            highlight = truchet_corner_circle_highlights(box.size, 2 * box.size)
        else:
            tile = truchet_crossed_lines(box.size, 2 * box.size)
            highlight = truchet_crossed_lines_highlights(box.size, 2 * box.size)
        tile = tile.translate(-box.size / 2, -box.size / 2)
        angle = np.random.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        tile = tile.rotate(angle)
        tile = tile.translate(box.c + box.size / 2, box.r + box.size / 2)
        out.add(tile)
        highlight = highlight.translate(-box.size / 2, -box.size / 2)
        highlight = highlight.rotate(angle)
        highlight = highlight.translate(box.c + box.size / 2, box.r + box.size / 2)
        highlights.add(highlight)
    return [out, highlights]


def arc(x: float, y: float, r: float, start_angle=0, end_angle=2 * np.pi) -> Path:
    return [
        (x + r * np.cos(theta), y + r * np.sin(theta))
        for theta in np.linspace(start_angle, end_angle, 32)
    ]


def truchet_corner_circles(size: float, n_circles: int) -> Drawing:
    paths = []
    r = np.linspace(size / (n_circles * 2), size - size / (n_circles * 2), n_circles)
    circle_mask = Polygon(arc(0, 0, size - size / (n_circles * 2)))
    for i in range(n_circles):
        paths.append(arc(0, 0, r[i], 0, np.pi / 2))
        under_arc = arc(size, size, r[i], np.pi, 3 * np.pi / 2)
        shapely_under_arc = LineString(under_arc)
        diffed = shapely_under_arc.difference(circle_mask)
        if isinstance(diffed, LineString):
            paths.append(diffed.coords)
        elif isinstance(diffed, MultiLineString):
            paths.extend([line.coords for line in diffed.geoms])
    return Drawing(paths)


def truchet_corner_circle_highlights(size: float, n_circles: int) -> Drawing:
    paths = []
    r = np.linspace(size / n_circles, size, n_circles)
    circle_mask = Polygon(arc(0, 0, size - size / (n_circles * 2)))
    for i in range(0, n_circles - 1, 2):
        paths.append(arc(0, 0, r[i], 0, np.pi / 2))
        under_arc = arc(size, size, r[i], np.pi, 3 * np.pi / 2)
        shapely_under_arc = LineString(under_arc)
        diffed = shapely_under_arc.difference(circle_mask)
        if isinstance(diffed, LineString):
            paths.append(diffed.coords)
        elif isinstance(diffed, MultiLineString):
            paths.extend([line.coords for line in diffed.geoms])
    return Drawing(paths)


def truchet_crossed_lines(size: float, n_lines: int) -> Drawing:
    paths = []
    positions = np.linspace(size / (n_lines * 2), size - size / (n_lines * 2), n_lines)
    horizontal_rect_mask = Polygon(
        [
            (0, size / (n_lines * 2)),
            (size, size / (n_lines * 2)),
            (size, size - size / (n_lines * 2)),
            (0, size - size / (n_lines * 2)),
        ]
    )
    for i in range(n_lines):
        paths.append([(0, positions[i]), (size, positions[i])])
        line_to_cut = LineString([(positions[i], 0), (positions[i], size)])
        diffed = line_to_cut.difference(horizontal_rect_mask)
        if isinstance(diffed, LineString):
            paths.append(diffed.coords)
        elif isinstance(diffed, MultiLineString):
            paths.extend([line.coords for line in diffed.geoms])
    return Drawing(paths)


def truchet_crossed_lines_highlights(size: float, n_lines: int) -> Drawing:
    paths = []
    positions = np.linspace(size / n_lines, size, n_lines)
    horizontal_rect_mask = Polygon(
        [
            (0, size / (n_lines * 2)),
            (size, size / (n_lines * 2)),
            (size, size - size / (n_lines * 2)),
            (0, size - size / (n_lines * 2)),
        ]
    )
    for i in range(0, n_lines - 1, 2):
        paths.append([(0, positions[i]), (size, positions[i])])
        line_to_cut = LineString([(positions[i], 0), (positions[i], size)])
        diffed = line_to_cut.difference(horizontal_rect_mask)
        if isinstance(diffed, LineString):
            paths.append(diffed.coords)
        elif isinstance(diffed, MultiLineString):
            paths.extend([line.coords for line in diffed.geoms])
    return Drawing(paths)


def make_grid(width: int, height: int, max_block_size: int):
    grid = Grid(width, height)
    available_sizes = list(range(2, max_block_size + 1))
    while len(available_sizes) > 0:
        size = np.random.choice(available_sizes)
        if not grid.randomly_merge(size):
            available_sizes.remove(size)
    return grid


@click.command()
@click.option("-t", "--test", is_flag=True)
@click.option("-w", "--width", prompt=True, type=float)
@click.option("-h", "--height", prompt=True, type=float)
@click.option("-m", "--margin", prompt=True, type=float)
@click.option("-r", "--rows", prompt=True, type=int)
@click.option("-b", "--max-block-size", prompt=True, type=int)
@click.option("-p", "--prob_turn", prompt=True, type=float)
def main(
    test: bool,
    width: float,
    height: float,
    margin: float,
    rows: int,
    max_block_size: int,
    prob_turn: float,
):
    grid = make_grid(rows, round(rows * height / width), max_block_size)
    layers = render(grid, prob_turn)
    layers = Drawing.multi_scale_to_fit(list(layers), width, height, padding=margin)
    layers = [layer.join_paths(0.05).sort_paths() for layer in layers]
    if test or axi.device.find_port() is None:
        im = Drawing.render_layers(layers, bounds=(0, 0, width, height))
        im.write_to_png("truchet.png")
    else:
        axi.draw_layers(layers)


if __name__ == "__main__":
    main()
