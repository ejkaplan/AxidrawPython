from __future__ import annotations

from collections import deque
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

import numpy as np
from axi import Drawing
from tqdm import tqdm


class TileSet:
    def __init__(self, tile_size: tuple[float, float]):
        self.tile_size = tile_size
        self.tiles: list[Tile] = []
        self.adjacency_rules: np.ndarray | None = None

    def make_tile(
        self,
        drawings: dict[int, Drawing],
        edges: list[Any],
        rotations: int = 1,
        mirror: bool = False,
        weight: float = 1.0,
    ) -> list[Tile]:
        if weight <= 0:
            weight = 1e-10
        tile = Tile(drawings, edges, self.tile_size, self, weight)
        created = []
        for i in range(rotations):
            rotated = tile.rotate(i)
            self.tiles.append(rotated)
            created.append(rotated)
        if mirror:
            mirrored = tile.mirror()
            for i in range(rotations):
                rotated = mirrored.rotate(i)
                self.tiles.append(rotated)
                created.append(rotated)
        return created

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
    drawing_layers: dict[int, Drawing]
    edges: list[Any]
    size: tuple[float, float]
    parent: TileSet
    weight: float = 1.0

    def __eq__(self, other: Tile):
        if self.drawing_layers.keys() != other.drawing_layers.keys():
            return False
        for layer, drawing in self.drawing_layers.items():
            if other.drawing_layers[layer] != drawing:
                return False
        return (
            self.parent == other.parent
            and self.weight == other.weight
            and self.size == other.size
            and self.edges == other.edges
        )

    def rotate(self, n: int):
        new_layers = {
            layer: d.rotate(np.pi / 2 * n, (self.size[0] / 2, self.size[1] / 2))
            for layer, d in self.drawing_layers.items()
        }
        new_edges = self.edges[4 - n :] + self.edges[: 4 - n]
        return Tile(new_layers, new_edges, self.size, self.parent, self.weight)

    def mirror(self):
        new_layers = {
            layer: d.scale(-1, 1).translate(self.size[0], 0)
            for layer, d in self.drawing_layers.items()
        }
        new_edges = [self.edges[2], self.edges[1], self.edges[0], self.edges[3]]
        return Tile(new_layers, new_edges, self.size, self.parent, self.weight)


def entropy(probs: np.ndarray) -> float:
    probs = probs[probs != 0]
    return -np.sum(probs * np.log2(probs))


@dataclass
class Grid:
    grid: np.ndarray = field(init=False)
    probs: np.ndarray = field(init=False)
    entropy: np.ndarray = field(init=False)
    tile_set: TileSet
    size: tuple[int, int]

    def __post_init__(self):
        self.grid = np.ones((*self.size, len(self.tile_set)), dtype=bool)
        self.probs = np.array(
            [
                [[t.weight for t in self.tile_set.tiles] for _ in range(self.size[1])]
                for __ in range(self.size[0])
            ]
        )
        cell_sums = np.sum(self.probs, axis=2)
        self.probs /= cell_sums[:, :, np.newaxis]
        e = entropy(self.probs[0, 0])
        self.entropy = np.ones(self.size) * e

    def recalculate_probs(self, r: int, c: int):
        if not np.any(self.grid[r, c]):
            return
        self.probs[r, c] *= self.grid[r, c]
        self.probs[r, c] /= np.sum(self.probs[r, c])
        self.entropy[r, c] = entropy(self.probs[r, c])

    def __copy__(self) -> Grid:
        g = Grid(self.tile_set, self.size)
        g.grid = self.grid.copy()
        g.probs = self.probs.copy()
        return g

    def reduce_cell(self, r: int, c: int) -> bool:
        changed = False
        neighbors = self.get_neighbors((r, c))
        for d, coord in neighbors.items():
            neighbor_tiles = self.grid[coord[0], coord[1], :]
            allowed = np.any(
                self.tile_set.adjacency_rules[(d + 2) % 4, neighbor_tiles, :], axis=0
            )
            possible_old = self.grid[r, c].copy()
            self.grid[r, c] = np.logical_and(self.grid[r, c], allowed)
            neighbor_allowed = np.any(
                self.tile_set.adjacency_rules[d, :, neighbor_tiles], axis=0
            )
            self.grid[r, c] = np.logical_and(self.grid[r, c], neighbor_allowed)
            changed = changed or np.any(self.grid[r, c] != possible_old)
        if changed:
            self.recalculate_probs(r, c)
        return changed

    def get_neighbors(self, coord: tuple[int, int]) -> dict[int, tuple[int, int]]:
        r, c = coord
        neighbors = {
            0: (r, (c + 1) % self.size[1]),
            1: ((r + 1) % self.size[0], c),
            2: (r, (c - 1) % self.size[1]),
            3: ((r - 1) % self.size[0], c),
        }
        neighbors = {
            edge: coord
            for edge, coord in neighbors.items()
            if coord[0] in range(self.size[0]) and coord[1] in range(self.size[1])
        }
        return neighbors

    def reduce_grid(self, start: tuple[int, int]) -> None:
        frontier = set(self.get_neighbors(start).values())
        while frontier:
            r, c = frontier.pop()
            if self.reduce_cell(r, c):
                neighbors = set(
                    coord
                    for coord in self.get_neighbors((r, c)).values()
                    if coord not in frontier
                )
                frontier |= neighbors

    def count_unfinished(self) -> int:
        option_count = np.sum(self.grid, axis=2)
        if np.any(option_count == 0):
            return -1
        return np.count_nonzero(option_count != 1)

    def guess(self, rng: np.random.Generator) -> Generator[Grid, None, None]:
        coords = [
            (r, c)
            for r in range(self.size[0])
            for c in range(self.size[1])
            if np.sum(self.grid[r, c]) > 1
        ]
        rng.shuffle(coords)
        coords.sort(key=lambda x: self.entropy[x])
        for coord in coords:
            tiles_available = int(np.sum(self.grid[coord]))
            options_order = rng.choice(
                range(len(self.tile_set)),
                tiles_available,
                replace=False,
                p=self.probs[coord],
            )
            for option in options_order:
                new_grid = copy(self)
                new_options = np.zeros(len(self.tile_set), dtype=bool)
                new_options[option] = True
                new_grid.grid[coord] = new_options
                new_grid.reduce_grid(coord)
                if new_grid.count_unfinished() == -1:
                    continue
                yield new_grid

    def __str__(self):
        out = ""
        for r in range(self.size[0]):
            for c in range(self.size[1]):
                options = self.grid[r, c]
                n = np.sum(options)
                if n == 0:
                    out += "X  "
                elif n == 1:
                    out += f"{np.where(options)[0][0]:<3}"
                else:
                    out += "?  "
            out += "\n"
        return out


def solve_grid(grid: Grid, rng: np.random.Generator) -> Optional[Grid]:
    grid.tile_set.make_adjacency_rules()
    total_cells = grid.size[0] * grid.size[1]
    if grid.count_unfinished() == 0:
        return grid
    frontier = deque([grid.guess(rng)])
    t = tqdm(total=total_cells)
    attempts = 0
    while len(frontier) > 0:
        try:
            grid = next(frontier[-1])
            unfinished = grid.count_unfinished()
            t.n = total_cells - unfinished
            t.refresh()
            if unfinished == 0:
                return grid
            frontier.append(grid.guess(rng))
        except StopIteration:
            frontier.pop()
        attempts += 1
    t.close()


def draw_grid(grid: Grid) -> list[Drawing]:
    out: list[Drawing] = []
    for r in range(grid.size[0]):
        for c in range(grid.size[1]):
            idx = np.where(grid.grid[r, c])[0][0]
            tile = grid.tile_set[idx]
            for layer, drawing in tile.drawing_layers.items():
                while len(out) < layer + 1:
                    out.append(Drawing())
                out[layer].add(drawing.translate(c * tile.size[0], r * tile.size[1]))
    return out
