from copy import copy

from axi import Drawing

from axi_art.wave_function_collapse.wave_function_collapse import Tile, TileSet, Grid


def test_rotate():
    tile_set = TileSet((5, 5))
    t0 = tile_set.make_tile(Drawing(), [0, 1, 2, 3])
    t1 = t0.rotate(1)
    assert t1.edges == [3, 0, 1, 2]
    t2 = t0.rotate(2)
    assert t2.edges == [2, 3, 0, 1]
    t3 = t0.rotate(3)
    assert t3.edges == [1, 2, 3, 0]
    assert t0.neighbors == [[2], [2], [2], [2]]
    assert t1.neighbors == [[3], [3], [3], [3]]
    assert t2.neighbors == [[0], [0], [0], [0]]
    assert t3.neighbors == [[1], [1], [1], [1]]


def test_neighbors():
    tile_set = TileSet((5, 5))
    t0 = tile_set.make_tile(Drawing(), [1, 0, 2, 0])
    t1 = tile_set.make_tile(Drawing(), [2, 2, 1, 0])
    assert t0.neighbors == [[1], [1], [1], []]
    assert t1.neighbors == [[0], [], [0], [0]]


def test_grid_setup():
    tile_set = TileSet((5, 5))
    tile_set.make_tile(Drawing(), [1, 0, 2, 0])
    tile_set.make_tile(Drawing(), [2, 2, 1, 0])
    g = Grid(tile_set, (3, 3))
    assert g[0, 0] == {0, 1}


def test_grid_cell_reduce():
    tile_set = TileSet((5, 5))
    tile_set.make_tile(Drawing(), [1, 0, 2, 0])
    tile_set.make_tile(Drawing(), [2, 2, 1, 0])
    g = Grid(tile_set, (3, 3))
    assert not g.reduce_cell(0, 0)
    g[1, 0].remove(0)
    g[0, 1].remove(0)
    assert g.reduce_cell(0, 0)
    assert g[0, 0] == {0}


def test_grid_copy():
    tile_set = TileSet((5, 5))
    tile_set.make_tile(Drawing(), [1, 0, 2, 0])
    tile_set.make_tile(Drawing(), [2, 2, 1, 0])
    g0 = Grid(tile_set, (3, 3))
    g1 = copy(g0)
    assert g1.tile_set == g0.tile_set
    assert g1.size == g0.size
    assert g1.grid == g0.grid
    g0[0, 0] = {0}
    assert g1[0, 0] == {0, 1}


