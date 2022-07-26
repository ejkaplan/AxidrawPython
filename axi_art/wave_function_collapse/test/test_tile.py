import numpy as np
from axi import Drawing

from axi_art.wave_function_collapse.wave_function_collapse import TileSet, Grid


def test_adjacency():
    ts = TileSet((1, 1))
    ts.make_tile({0: Drawing()}, [0, 0, 1, 1])  # angle
    ts.make_tile({0: Drawing()}, [1, 0, 1, 0])  # horizontal
    ts.make_adjacency_rules()
    print(ts.adjacency_rules)
    target_adjacency = np.array(
        [
            [[False, False], [True, True]],
            [[False, True], [False, True]],
            [[False, True], [False, True]],
            [[False, False], [True, True]],
        ]
    )
    assert np.all(ts.adjacency_rules == target_adjacency)


def test_reduce_cell():
    ts = TileSet((1, 1))
    ts.make_tile({0: Drawing()}, [0, 0, 1, 1])  # angle
    ts.make_tile({0: Drawing()}, [0, 1, 0, 1])  # vertical
    ts.make_tile({0: Drawing()}, [1, 0, 1, 0])  # horizontal
    ts.make_adjacency_rules()
    grid = Grid(ts, (1, 2))
    grid.grid[0, 0] = [True, False, False]
    grid.reduce_cell(0, 1)
    assert np.all(grid.grid == [[[True, False, False], [False, True, False]]])


def test_reduce_cell_impossible():
    ts = TileSet((1, 1))
    ts.make_tile({0: Drawing()}, [0, 0, 1, 1])  # angle
    ts.make_tile({0: Drawing()}, [1, 0, 1, 0])  # horizontal
    ts.make_adjacency_rules()
    grid = Grid(ts, (1, 2))
    grid.grid[0, 0] = [True, False]
    grid.reduce_cell(0, 1)
    assert np.all(grid.grid == [[[True, False], [False, False]]])


def test_grid_status():
    ts = TileSet((1, 1))
    ts.make_tile({0: Drawing()}, [0, 0, 1, 1])  # angle
    ts.make_tile({0: Drawing()}, [0, 1, 0, 1])  # vertical
    ts.make_tile({0: Drawing()}, [1, 0, 1, 0])  # horizontal
    ts.make_adjacency_rules()
    grid = Grid(ts, (1, 2))
    assert grid.count_unfinished() == 2
    grid.grid[0, 0] = [True, False, False]
    assert grid.count_unfinished() == 1
    grid.grid[0, 0, 0] = False
    assert grid.count_unfinished() == -1
