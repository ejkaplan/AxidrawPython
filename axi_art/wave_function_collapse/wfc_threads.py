from itertools import permutations

import axi
import numpy as np
from axi import Drawing

from axi_art.wave_function_collapse.wave_function_collapse import (
    TileSet,
    Grid,
    solve_grid,
    draw_grid,
)


def make_tileset(colors: int) -> TileSet:
    tileset = TileSet((1, 1))
    circle_turn = Drawing(
        [
            [
                (0.5 * np.cos(theta), 0.5 * np.sin(theta))
                for theta in np.linspace(0, np.pi / 2, 50)
            ]
        ]
    )
    horizontal = Drawing([[(0, 0.5), (1, 0.5)]])
    t_junction = Drawing()
    t_junction.add(horizontal)
    t_junction.add(circle_turn)
    under = Drawing([[(0.5, 0), (0.5, 0.3)], [(0.5, 0.7), (0.5, 1)]])
    dead_end = Drawing(
        [
            [(0, 0.5), (0.5, 0.5)],
            [
                (0.5 + 0.15 * np.cos(theta), 0.5 + 0.15 * np.sin(theta))
                for theta in np.linspace(0, 2 * np.pi, 50)
            ],
        ]
    )
    for c in range(colors):
        tileset.make_tile({c: circle_turn}, [-1, -1, c, c], 4)
        tileset.make_tile({c: horizontal}, [c, -1, c, -1], 2)
        tileset.make_tile({c: dead_end}, [-1, -1, c, -1], 4, weight=0)
        tileset.make_tile({c: t_junction}, [c, -1, c, c], 4, True, weight=0.1)
    for c0, c1 in permutations(range(colors), 2):
        tileset.make_tile({c0: horizontal, c1: under}, [c0, c1, c0, c1], 2)
    # blank tile
    tileset.make_tile(dict(), [-1, -1, -1, -1], weight=10)
    return tileset


def make_drawings(
    rng: np.random.Generator, colors: int, rows: int, cols: int
) -> list[Drawing]:
    tileset = make_tileset(colors)
    grid = Grid(tileset, (rows, cols))
    grid = solve_grid(grid, rng)
    drawings = draw_grid(grid)
    drawings = Drawing.multi_scale_to_fit(drawings, 8, 8, 0.5)
    drawings = [d.sort_paths() for d in drawings]  # TODO: Fix join paths
    # drawings = [d.join_paths(0.01).sort_paths() for d in drawings]
    return drawings


def main():
    test = True
    rng = np.random.default_rng()
    drawings = make_drawings(rng, 3, 30, 30)
    if test or axi.device.find_port() is None:
        im = Drawing.render_layers(drawings)
        im.write_to_png("test.png")
    else:
        axi.draw_layers(drawings)


if __name__ == "__main__":
    main()
