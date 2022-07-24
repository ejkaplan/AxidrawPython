from __future__ import annotations

from copy import deepcopy, copy
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from random import shuffle, seed
from typing import Any, Iterator, Generator, Optional

import axi
import numpy as np
from axi import Drawing


@dataclass
class TileSet:
    tile_size: tuple[float, float]
    tiles: list[Tile] = field(default_factory=list)
    id_iter: Iterator[int] = field(default_factory=count)

    def make_tile(self, drawing: Drawing, edges: list[Any]) -> Tile:
        return Tile(drawing, edges, self.tile_size, next(self.id_iter), self)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, i: int) -> Tile:
        return self.tiles[i]


@dataclass
class Tile:
    drawing: Drawing
    edges: list[Any]
    size: tuple[float, float]
    id: int
    parent: TileSet
    neighbors: tuple[list[Tile], list[Tile], list[Tile], list[Tile]] = field(
        default_factory=lambda: [[], [], [], []]
    )

    def __post_init__(self):
        other: Tile
        for other in self.parent.tiles:
            for direction in range(4):
                opp = (direction + 2) % 4
                if self.edges[direction] == other.edges[opp]:
                    self.neighbors[direction].append(other.id)
                    other.neighbors[opp].append(self.id)
        self.parent.tiles.append(self)

    def rotate(self, n: int):
        d = self.drawing.rotate(np.pi / 2 * n, (self.size[0] / 2, self.size[1] / 2))
        edges = self.edges[4 - n :] + self.edges[: 4 - n]
        return self.parent.make_tile(d, edges)


class Status(Enum):
    INVALID = 0
    VALID = 1
    DONE = 2


@dataclass
class Grid:
    grid: list[list[set[int]]] = field(init=False)
    tile_set: TileSet
    size: tuple[int, int]

    def __post_init__(self):
        self.grid = [
            [set(range(len(self.tile_set))) for c in range(self.size[1])]
            for r in range(self.size[0])
        ]

    def __copy__(self) -> Grid:
        g = Grid(self.tile_set, self.size)
        g.grid = deepcopy(self.grid)
        return g

    def __getitem__(self, coord: tuple[int, int]) -> set[int]:
        r, c = coord
        return self.grid[r][c]

    def __setitem__(self, coord: tuple[int, int], new_options: set[int]) -> None:
        r, c = coord
        self.grid[r][c] = new_options

    def reduce_cell(self, r: int, c: int) -> bool:
        changed = False
        neighbors = {0: (r, c + 1), 1: (r + 1, c), 2: (r, c - 1), 3: (r - 1, c)}
        neighbors = {
            edge: coord
            for edge, coord in neighbors.items()
            if coord[0] in range(self.size[0]) and coord[1] in range(self.size[1])
        }
        new_options = copy(self[r, c])
        for edge, coord in neighbors.items():
            cell_options = set()
            opp_edge = (edge + 2) % 4
            for edge_tile in self[coord]:
                cell_options |= set(self.tile_set[edge_tile].neighbors[opp_edge])
            new_options &= cell_options
        if new_options != self[r, c]:
            self[r, c] = new_options
            changed = True
        return changed

    def reduce_grid(self) -> None:
        # TODO: Propagate changes outwards from the changed cell instead of redoing the whole grid every time.
        while True:
            changed = False
            for r in range(self.size[0]):
                for c in range(self.size[1]):
                    changed = changed or self.reduce_cell(r, c)
            if not changed:
                break

    def status(self) -> Status:
        done = True
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                if (n := len(self[r, c])) == 0:
                    return Status.INVALID
                elif n > 1:
                    done = False
        return Status.DONE if done else Status.VALID

    def guess(self) -> Generator[Grid, None, None]:
        coords = [(r, c) for r in range(self.size[0]) for c in range(self.size[1])]
        coords = [coord for coord in coords if len(self[coord]) > 1]
        coords.sort(key=lambda x: len(self[x]))
        for coord in coords:
            options_order = list(self[coord])
            shuffle(options_order)
            for option in options_order:
                new_grid = copy(self)
                new_grid[coord] = {option}
                new_grid.reduce_grid()
                yield new_grid


def solve_grid(grid: Grid) -> Optional[Grid]:
    if grid.status() == Status.DONE:
        return grid
    frontier = [grid.guess()]
    while len(frontier) > 0:
        try:
            grid = next(frontier[-1])
            if (status := grid.status()) == Status.DONE:
                return grid
            elif status == Status.VALID:
                frontier.append(grid.guess())
        except StopIteration:
            frontier.pop()


def draw_grid(grid: Grid) -> Drawing:
    assert grid.status() == Status.DONE
    out = Drawing()
    for r in range(grid.size[0]):
        for c in range(grid.size[1]):
            tile = grid.tile_set[next(iter(grid[r, c]))]
            out.add(tile.drawing.translate(c * tile.size[0], r * tile.size[1]))
    return out


def main():
    test = True
    tileset = TileSet((1, 1))
    # L_drawing = Drawing([[(0, 0.5), (0.5, 0.5), (0.5, 0)]])
    # tile = tileset.make_tile(L_drawing, [0, 0, 1, 1])
    # for i in range(1, 4):
    #     tile.rotate(i)
    tileset.make_tile(Drawing(), [0, 0, 0, 0])
    T_drawing = Drawing([[(0, 0.5), (1, 0.5)], [(0.5, 0.5), (0.5, 1)]])
    tile = tileset.make_tile(T_drawing, [1, 1, 1, 0])
    for i in range(1, 4):
        tile.rotate(i)
    grid = Grid(tileset, (30, 30))
    grid = solve_grid(grid)
    d = draw_grid(grid).scale_to_fit(8, 8, 0.5).center(8, 8)
    print(len(d.paths))
    d = d.join_paths(0.1)
    print(len(d.paths))
    drawings = [d]
    if test or axi.device.find_port() is None:
        im = Drawing.render_layers(drawings, bounds=(0, 0, 8, 8))
        im.write_to_png("test.png")
    else:
        axi.draw_layers(drawings)


if __name__ == "__main__":
    main()
