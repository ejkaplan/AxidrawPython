from __future__ import annotations
import random
import axi
from utils import merge_paths, offset_paths

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


def make_maze(width: int, height: int, depth: int, dimensions: int, p_random: float = 0.0):
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
        expansion_direction = random.choice(neighbor_directions)
        curr.remove_wall(expansion_direction)
        next_cell = curr.neighbors[expansion_direction]
        visited.add(next_cell)
        frontier.append(curr)
        frontier.append(next_cell)
    return cells


def make_2d_slice_paths(cells, z, w, bounds):
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
    # Render the two missing walls
    paths.append([(bounds[0], 0), (bounds[0], bounds[1]), (0, bounds[1])])
    return paths


def main():
    bounds = (4, 4, 4, 4)
    cells = make_maze(*bounds, 0.1)
    paths = []
    for floor in range(bounds[2]):
        for dimension in range(bounds[3]):
            submaze = make_2d_slice_paths(cells, floor, dimension, bounds)
            submaze = offset_paths(submaze, dimension * (bounds[0] + 1), floor * (bounds[1] + 1))
            paths += submaze
    paths = merge_paths(paths)
    drawing = axi.Drawing(paths).scale_to_fit(11, 8.5, 1).sort_paths()
    drawing = drawing.center(11, 8.5)
    if axi.device.find_port() is None:
        im = drawing.render()
        im.write_to_png('out.png')
    else:
        axi.draw(drawing)


if __name__ == "__main__":
    main()
