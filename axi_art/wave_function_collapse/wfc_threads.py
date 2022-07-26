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
    turn = Drawing([[(0, 0.5), (0.5, 0.5), (0.5, 0)]])
    circle_turn = Drawing(
        [
            [
                (0.5 * np.cos(theta), 0.5 * np.sin(theta))
                for theta in np.linspace(0, np.pi / 2, 50)
            ]
        ]
    )
    horizontal = Drawing([[(0, 0.5), (1, 0.5)]])
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
        t = tileset.make_tile({c: circle_turn}, [-1, -1, c, c])
        for i in range(1, 4):
            t.rotate(i)
        t = tileset.make_tile({c: horizontal}, [c, -1, c, -1])
        t.rotate(1)
        t = tileset.make_tile({c: dead_end}, [-1, -1, c, -1])
        for i in range(1, 4):
            t.rotate(i)
    for c0, c1 in permutations(range(colors), 2):
        t = tileset.make_tile({c0: horizontal, c1: under}, [c0, c1, c0, c1])
        t.rotate(1)
        # t = tileset.make_tile(
        #     {c0: circle_turn, c1: circle_turn.rotate(np.pi, (0.5, 0.5))},
        #     [c1, c1, c0, c0],
        # )
        # t.rotate(1)
    # blank tile
    # tileset.make_tile(dict(), [-1, -1, -1, -1])
    return tileset


def make_drawings(
    rng: np.random.Generator, colors: int, rows: int, cols: int
) -> list[Drawing]:
    while True:
        try:
            tileset = make_tileset(colors)
            grid = Grid(tileset, (rows, cols))
            grid = solve_grid(grid, rng)
            drawings = draw_grid(grid)
            drawings = Drawing.multi_scale_to_fit(drawings, 8, 8, 0.5)
            drawings = [d.join_paths(0.01).sort_paths() for d in drawings]
            break
        except ZeroDivisionError:
            pass
    return drawings


def main():
    test = True
    rng = np.random.default_rng()
    drawings = make_drawings(rng, 5, 30, 30)
    if test or axi.device.find_port() is None:
        im = Drawing.render_layers(drawings)
        im.write_to_png("test.png")
    else:
        axi.draw_layers(drawings)


if __name__ == "__main__":
    main()
