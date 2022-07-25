from itertools import combinations

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
    horizontal = Drawing([[(0, 0.5), (1, 0.5)]])
    vertical = Drawing([[(0.5, 0), (0.5, 1)]])
    for c in range(colors):
        t = tileset.make_tile({c: turn}, [-1, -1, c, c])
        for i in range(1, 4):
            t.rotate(i)
        t = tileset.make_tile({c: horizontal}, [c, -1, c, -1])
        t.rotate(1)
    tileset.make_tile(dict(), [-1, -1, -1, -1])
    for c0, c1 in combinations(range(colors), 2):
        t = tileset.make_tile({c0: horizontal, c1: vertical}, [c0, c1, c0, c1])
        t.rotate(1)
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
            drawings = [d.sort_paths() for d in drawings]
            break
        except ZeroDivisionError:
            pass
    return drawings


def main():
    test = True
    rng = np.random.default_rng()
    drawings = make_drawings(rng, 2, 20, 20)
    if test or axi.device.find_port() is None:
        im = Drawing.render_layers(drawings)
        im.write_to_png("test.png")
    else:
        axi.draw_layers(drawings)


if __name__ == "__main__":
    main()
