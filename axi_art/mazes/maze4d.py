from __future__ import annotations

import math
import random

import axi
import click

from axi_art.utils import offset_paths

coord = tuple[int, int, int, int]
DIRECTIONS = [(-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, -1),
              (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]


def reverse(direction):
    return tuple([-d for d in direction])


def diff(dir0, dir1):
    return tuple([(dir1[i] - dir0[i]) for i in range(len(dir0))])


class Cell:
    def __init__(self, x, y, z, w):
        self.coords = (x, y, z, w)
        self.neighbors = {direction: None for direction in DIRECTIONS}
        self.walls = {direction: True for direction in DIRECTIONS}

    def add_neighbor(self, other: Cell):
        d = diff(self.coords, other.coords)
        if d in self.neighbors:
            self.neighbors[d] = other
            other.neighbors[reverse(d)] = self
        else:
            raise ValueError("Tried to set non-adjacent cells as neighbors")

    def remove_wall(self, direction: tuple[int, int]):
        self.walls[direction] = False
        self.neighbors[direction].walls[reverse(direction)] = False

    def get_neighbor_directions(self):
        return [x for x in self.neighbors if self.neighbors[x] is not None]

    def __repr__(self):
        return f"Cell@({self.coords}"


def make_grid(width, height, depth, dimensions):
    cells = {(x, y, z, w): Cell(x, y, z, w)
             for x in range(width)
             for y in range(height)
             for z in range(depth)
             for w in range(dimensions)}
    for loc in cells:
        if loc[0] + 1 < width:
            cells[loc].add_neighbor(cells[(loc[0] + 1, loc[1], loc[2], loc[3])])
        if loc[1] + 1 < height:
            cells[loc].add_neighbor(cells[(loc[0], loc[1] + 1, loc[2], loc[3])])
        if loc[2] + 1 < depth:
            cells[loc].add_neighbor(cells[(loc[0], loc[1], loc[2] + 1, loc[3])])
        if loc[3] + 1 < dimensions:
            cells[loc].add_neighbor(cells[(loc[0], loc[1], loc[2], loc[3] + 1)])
    return cells


def get_bias(d, dir_bias):
    nonzero = [i for i, e in enumerate(d) if e != 0][0]
    return dir_bias[nonzero]


def make_maze(width: int, height: int, depth: int, dimensions: int, p_random: float = 0.0, dir_bias=(1, 1, 1, 1)):
    cells = make_grid(width, height, depth, dimensions)
    frontier = [random.choice(list(cells.values()))]
    visited = {frontier[0]}
    while len(frontier) > 0:
        if random.random() < p_random:
            curr = frontier.pop(random.randrange(len(frontier)))
        else:
            curr = frontier.pop(-1)
        neighbor_directions = [n for n in curr.get_neighbor_directions() if curr.neighbors[n] not in visited]
        if len(neighbor_directions) == 0:
            continue
        biases = [get_bias(d, dir_bias) for d in neighbor_directions]
        expansion_direction = random.choices(neighbor_directions, weights=biases)[0]
        curr.remove_wall(expansion_direction)
        next_cell = curr.neighbors[expansion_direction]
        visited.add(next_cell)
        frontier.append(curr)
        frontier.append(next_cell)
    return cells


def bfs(cells, start):
    visited = []
    frontier = [start]
    while len(frontier) > 0:
        curr = frontier.pop(0)
        visited.append(curr)
        cell = cells.get(curr)
        neighbors = [cell.neighbors[n].coords for n in cell.get_neighbor_directions() if
                     not cell.walls[n] and cell.neighbors[n].coords not in visited + frontier]
        frontier += neighbors
    return visited


def circle(x, y, r):
    path = []
    for i in range(5):
        angle = math.tau * i / 4
        path.append((x + r * math.cos(angle), y + r * math.sin(angle)))
    return path


def make_2d_slice_paths(cells, z, w, bounds, endpoints=None):
    if endpoints is None:
        endpoints = []
    paths = []
    door_margin = 1 / 8
    # Render the Maze
    for x in range(bounds[0]):
        for y in range(bounds[1]):
            cell = cells[(x, y, z, w)]
            if cell.walls[(-1, 0, 0, 0)]:
                paths.append([(x, y), (x, y + 1)])  # WEST WALL
            else:
                paths += [[(x, y), (x, y + door_margin)], [(x, y + 1 - door_margin), (x, y + 1)]]  # WEST WALL W/ DOOR
            if cell.walls[(0, -1, 0, 0)]:
                paths.append([(x, y), (x + 1, y)])  # NORTH WALL
            else:
                paths += [[(x, y), (x + door_margin, y)], [(x + 1 - door_margin, y), (x + 1, y)]]  # WEST WALL W/ DOOR
            if not cell.walls[(0, 0, -1, 0)]:
                paths.append([(x + 0.4, y + 0.3), (x + 0.5, y + 0.1), (x + 0.6, y + 0.3)])  # up arrow
            if not cell.walls[(0, 0, 1, 0)]:
                paths.append([(x + 0.4, y + 0.7), (x + 0.5, y + 0.9), (x + 0.6, y + 0.7)])  # down arrow
            if not cell.walls[(0, 0, 0, -1)]:
                paths.append([(x + 0.3, y + 0.4), (x + 0.1, y + 0.5), (x + 0.3, y + 0.6)])  # left arrow
            if not cell.walls[(0, 0, 0, 1)]:
                paths.append([(x + 0.7, y + 0.4), (x + 0.9, y + 0.5), (x + 0.7, y + 0.6)])  # right arrow
            if (x, y, z, w) in endpoints:
                paths.append(circle(x + 0.5, y + 0.5, 0.1))
    # Render the two missing walls
    paths.append([(bounds[0], 0), (bounds[0], bounds[1]), (0, bounds[1])])
    return paths


def manhattan(start, end):
    return sum([abs(start[i] - end[i]) for i in range(len(start))])


def backtrack(prev, cell):
    path = []
    while cell is not None:
        path.insert(0, cell)
        cell = prev[cell]
    return path


def astar(cells, start, end):
    prev = {start: None}
    g_score = {start: 0}
    f_score = {start: manhattan(start, end)}
    open_set = {start}
    while len(open_set) > 0:
        curr = min(open_set, key=lambda x: f_score[x])
        if curr == end:
            return backtrack(prev, curr)
        open_set.remove(curr)
        cell = cells[curr]
        for neighbor in [cell.neighbors[d].coords for d in cell.get_neighbor_directions() if not cell.walls[d]]:
            tentative_g_score = g_score[curr] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                prev[neighbor] = curr
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + manhattan(neighbor, end)
                open_set.add(neighbor)


@click.command()
@click.option('-t', '--test', is_flag=True)
@click.option('-w', '--width', prompt=True, type=float)
@click.option('-h', '--height', prompt=True, type=float)
@click.option('-m', '--margin', prompt=True, type=float)
@click.option('-r', '--rows', prompt=True, type=int)
@click.option('-c', '--cols', prompt=True, type=int)
@click.option('-mr', '--meta-rows', prompt=True, type=int)
@click.option('-mc', '--meta-cols', prompt=True, type=int)
@click.option('-rb', '--row-bias', prompt=True, type=float, default=1)
@click.option('-cb', '--col-bias', prompt=True, type=float, default=1)
@click.option('-mrb', '--meta-row-bias', prompt=True, type=float, default=1)
@click.option('-mcb', '--meta-col-bias', prompt=True, type=float, default=1)
@click.option('-r', '--random_pickup_prob', prompt=True, type=float, default=0)
def main(test: bool, width: float, height: float, margin: float,
         rows: int, cols: int, meta_rows: int, meta_cols: int,
         row_bias: float, col_bias: float, meta_row_bias: float, meta_col_bias: float,
         random_pickup_prob: float):
    bounds = (rows, cols, meta_rows, meta_cols)
    cells = make_maze(*bounds, p_random=random_pickup_prob, dir_bias=(row_bias, col_bias, meta_row_bias, meta_col_bias))
    end_a = bfs(cells, (0, 0, 0, 0))[-1]
    end_b = bfs(cells, end_a)[-1]
    paths = []
    for floor in range(bounds[2]):
        for dimension in range(bounds[3]):
            submaze = make_2d_slice_paths(cells, floor, dimension, bounds, endpoints=[end_a, end_b])
            submaze = offset_paths(submaze, dimension * (bounds[0] + 1), floor * (bounds[1] + 1))
            paths += submaze
    drawing = axi.Drawing(paths).scale_to_fit(width, height, margin).join_paths(0.03).sort_paths().center(width, height)
    if test or axi.device.find_port() is None:
        im = drawing.render(bounds=(0, 0, width, height))
        im.write_to_png('maze_4d.png')
    else:
        axi.draw(drawing)


if __name__ == "__main__":
    main()
