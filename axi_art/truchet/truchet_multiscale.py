from random import choice
from typing import NamedTuple

import axi
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
        legal_placements = [(r, c) for r in range(self.height - size + 1) for c in range(self.width - size + 1)]
        legal_placements = list(filter(lambda x: np.all(self.grid[x[0]:x[0] + size, x[1]:x[1] + size] == 0),
                                       legal_placements))
        if len(legal_placements) == 0:
            return False
        coord = choice(legal_placements)
        self.grid[coord[0]:coord[0] + size, coord[1]:coord[1] + size] = 1
        self._boxes.append(Box(coord[0], coord[1], size))
        return True

    def render(self) -> Drawing:
        out = Drawing()
        for box in self.boxes:
            if np.random.random() < 0.2:
                tile = truchet_corner_circles(box.size, 2 * box.size)
            else:
                tile = truchet_crossed_lines(box.size, 2 * box.size)
            tile = tile.translate(-box.size / 2, -box.size / 2)
            tile = tile.rotate(np.random.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2]))
            tile = tile.translate(box.c + box.size / 2, box.r + box.size / 2)
            out.add(tile)
            # out.add(Drawing([[(box.c, box.r), (box.c + box.size, box.r),
            #                   (box.c + box.size, box.r + box.size), (box.c + box.size, box.r)]]))
        return out


def arc(x: float, y: float, r: float, start_angle=0, end_angle=2 * np.pi) -> Path:
    return [(x + r * np.cos(theta), y + r * np.sin(theta)) for theta in np.linspace(start_angle, end_angle, 32)]


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
            paths.extend([line.coords for line in diffed])
    return Drawing(paths)


def truchet_crossed_lines(size: float, n_lines: int) -> Drawing:
    paths = []
    positions = np.linspace(size / (n_lines * 2), size - size / (n_lines * 2), n_lines)
    horizontal_rect_mask = Polygon([(0, size / (n_lines * 2)),
                                    (size, size / (n_lines * 2)),
                                    (size, size - size / (n_lines * 2)),
                                    (0, size - size / (n_lines * 2))])
    for i in range(n_lines):
        paths.append([(0, positions[i]), (size, positions[i])])
        line_to_cut = LineString([(positions[i], 0), (positions[i], size)])
        diffed = line_to_cut.difference(horizontal_rect_mask)
        if isinstance(diffed, LineString):
            paths.append(diffed.coords)
        elif isinstance(diffed, MultiLineString):
            paths.extend([line.coords for line in diffed])
    return Drawing(paths)


def make_grid(width: int, height: int, size_dict: dict[int, int]):
    grid = Grid(width, height)
    for size in sorted(size_dict.keys(), reverse=True):
        for _ in range(size_dict[size]):
            if not grid.randomly_merge(size):
                break
    return grid


TEST = True


def main():
    grid = make_grid(60, 46, {3: 100, 2: 100})
    drawing = grid.render()
    drawing = drawing.scale_to_fit(11, 8.5, 0.5)
    drawing = drawing.center(11, 8.5)
    if TEST or axi.device.find_port() is None:
        im = drawing.render(bounds=(0, 0, 11, 8.5))
        im.write_to_png('truchet.png')
    else:
        axi.draw(drawing)


if __name__ == '__main__':
    main()
