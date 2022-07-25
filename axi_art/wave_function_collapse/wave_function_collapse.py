from __future__ import annotations

from collections import deque
from copy import copy
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Generator, Optional

import axi
import numpy as np
from axi import Drawing
from tqdm import tqdm

rng = np.random.default_rng()


class TileSet:
    def __init__(self, tile_size: tuple[float, float]):
        self.tile_size = tile_size
        self.tiles: list[Tile] = []
        self.id_iter = count()
        self.adjacency_rules: np.ndarray | None = None

    def make_tile(self, drawing: Drawing, edges: list[Any]) -> Tile:
        tile = Tile(drawing, edges, self.tile_size, next(self.id_iter), self)
        self.tiles.append(tile)
        return tile

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, i: int) -> Tile:
        return self.tiles[i]

    def make_adjacency_rules(self) -> None:
        self.adjacency_rules = np.zeros(
            (4, len(self.tiles), len(self.tiles)), dtype=bool
        )
        for i, tile in enumerate(self.tiles):
            for j, neighbor in enumerate(self.tiles):
                for d in range(4):
                    self.adjacency_rules[d, i, j] = (
                        tile.edges[d] == neighbor.edges[(d + 2) % 4]
                    )


@dataclass
class Tile:
    drawing: Drawing
    edges: list[Any]
    size: tuple[float, float]
    id: int
    parent: TileSet

    def rotate(self, n: int):
        d = self.drawing.rotate(np.pi / 2 * n, (self.size[0] / 2, self.size[1] / 2))
        edges = self.edges[4 - n :] + self.edges[: 4 - n]
        return self.parent.make_tile(d, edges)


@dataclass
class Grid:
    grid: np.ndarray = field(init=False)
    tile_set: TileSet
    size: tuple[int, int]

    def __post_init__(self):
        self.grid = np.ones((*self.size, len(self.tile_set)), dtype=bool)

    def __copy__(self) -> Grid:
        g = Grid(self.tile_set, self.size)
        g.grid = self.grid.copy()
        return g

    def reduce_cell(self, r: int, c: int) -> bool:
        changed = False
        neighbors = {0: (r, c + 1), 1: (r + 1, c), 2: (r, c - 1), 3: (r - 1, c)}
        neighbors = {
            edge: coord
            for edge, coord in neighbors.items()
            if coord[0] in range(self.size[0]) and coord[1] in range(self.size[1])
        }
        for d, coord in neighbors.items():
            neighbor_tiles = self.grid[coord[0], coord[1], :]
            allowed = np.any(
                self.tile_set.adjacency_rules[(d + 2) % 4, neighbor_tiles, :], axis=0
            )
            possible_old = self.grid[r, c]
            self.grid[r, c] = np.logical_and(possible_old, allowed)
            changed = changed or np.any(self.grid[r, c] != possible_old)
        return changed

    def reduce_grid(self, start: tuple[int, int]) -> None:
        frontier = deque([start])
        while frontier:
            r, c = frontier.pop()
            if self.reduce_cell(r, c):
                neighbors = [(r, c + 1), (r + 1, c), (r, c - 1), (r - 1, c)]
                neighbors = [
                    coord
                    for coord in neighbors
                    if coord[0] in range(self.size[0])
                    and coord[1] in range(self.size[1])
                    and coord not in frontier
                ]
                frontier.extend(neighbors)

    def count_unfinished(self) -> int:
        option_count = np.sum(self.grid, axis=2)
        if np.any(option_count == 0):
            return -1
        return np.count_nonzero(option_count != 1)

    def guess(self) -> Generator[Grid, None, None]:
        coords = [
            (r, c)
            for r in range(self.size[0])
            for c in range(self.size[1])
            if np.sum(self.grid[r, c]) > 1
        ]
        coords.sort(key=lambda x: np.sum(self.grid[x]))
        for coord in coords:
            options_order = np.where(self.grid[coord])[0]
            rng.shuffle(options_order)
            for option in options_order:
                new_grid = copy(self)
                new_options = np.zeros(len(self.tile_set), dtype=bool)
                new_options[option] = True
                new_grid.grid[coord] = new_options
                new_grid.reduce_grid(coord)
                yield new_grid


def solve_grid(grid: Grid) -> Optional[Grid]:
    total_cells = grid.size[0] * grid.size[1]
    if grid.count_unfinished() == 0:
        return grid
    frontier = deque([grid.guess()])
    t = tqdm(total=total_cells)
    while len(frontier) > 0:
        try:
            grid = next(frontier[-1])
            if (unfinished := grid.count_unfinished()) == -1:
                continue
            t.n = total_cells - unfinished
            t.refresh()
            if unfinished == 0:
                return grid
            frontier.append(grid.guess())
        except StopIteration:
            frontier.pop()
    t.close()


def draw_grid(grid: Grid) -> Drawing:
    out = Drawing()
    for r in range(grid.size[0]):
        for c in range(grid.size[1]):
            idx = np.where(grid.grid[r, c])[0][0]
            tile = grid.tile_set[idx]
            out.add(tile.drawing.translate(c * tile.size[0], r * tile.size[1]))
    return out


def main():
    test = True
    tileset = TileSet((1, 1))
    L_drawing = Drawing([[(0, 0.5), (0.5, 0.5), (0.5, 0)]])
    tile0 = tileset.make_tile(L_drawing, [0, 0, 1, 1])
    for i in range(1, 4):
        tile0.rotate(i)
    straight_drawing = Drawing([[(0, 0.5), (1, 0.5)]])
    tile1 = tileset.make_tile(straight_drawing, [1, 0, 1, 0])
    tile1.rotate(1)
    tileset.make_adjacency_rules()
    grid = Grid(tileset, (30, 30))
    grid = solve_grid(grid)
    d = draw_grid(grid).scale_to_fit(8, 8, 0.5).center(8, 8).join_paths(0.01)
    drawings = [d]
    if test or axi.device.find_port() is None:
        im = Drawing.render_layers(drawings, bounds=(0, 0, 8, 8))
        im.write_to_png("test.png")
    else:
        axi.draw_layers(drawings)


if __name__ == "__main__":
    main()
