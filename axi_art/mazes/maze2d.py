from __future__ import annotations
import random
import axi

from axi_art.utils import merge_paths

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
coord = tuple[int, int]


def reverse_direction(direction):
    return -direction[0], -direction[1]


class Cell:
    def __init__(self, r, c):
        self.coords = (r, c)
        self.neighbors = {direction: None for direction in DIRECTIONS}
        self.walls = {direction: True for direction in DIRECTIONS}

    def add_neighbor(self, other: Cell):
        diff = (other.coords[0] - self.coords[0], other.coords[1] - self.coords[1])
        if diff in self.neighbors:
            self.neighbors[diff] = other
            other.neighbors[reverse_direction(diff)] = self
        else:
            raise ValueError("Tried to set non-adjacent cells as neighbors")

    def remove_wall(self, direction: coord):
        self.walls[direction] = False
        self.neighbors[direction].walls[reverse_direction(direction)] = False

    def get_neighbor_directions(self):
        return [x for x in self.neighbors if self.neighbors[x] is not None]

    def __repr__(self):
        return f"Cell@({self.coords}"


def make_grid(rows: int, cols: int) -> list[list[Cell]]:
    cells = [[Cell(r, c) for c in range(cols)] for r in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                cells[r][c].add_neighbor(cells[r + 1][c])
            if c + 1 < cols:
                cells[r][c].add_neighbor(cells[r][c + 1])
    return cells


def make_maze(rows: int, cols: int, p_random: float = 0) -> list[list[Cell]]:
    cells = make_grid(rows, cols)
    start = random.choice(random.choice(cells))
    frontier = [start]
    visited = {start}
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


def make_paths(cells: list[list[Cell]]) -> list[list[coord]]:
    paths = []
    # Render the Maze
    for row in cells:
        for cell in row:
            r, c = cell.coords
            if cell.walls[(0, -1)]:
                paths.append([(c, r), (c, r + 1)])  # NORTH WALL
            if cell.walls[(-1, 0)]:
                paths.append([(c, r), (c + 1, r)])  # WEST WALL
    # Render the two missing walls
    rows = len(cells)
    cols = len(cells[0])
    paths.append([(cols, 0), (cols, rows), (0, rows)])
    return paths


TEST = False


def main():
    rows = 40
    cols = round(rows * 11 / 8.5)
    cells = make_maze(rows, cols, 0.1)
    paths = make_paths(cells)
    paths = merge_paths(paths)
    drawing = axi.Drawing(paths).scale_to_fit(11, 8.5, 1).sort_paths()
    drawing = drawing.center(11, 8.5)
    if TEST or axi.device.find_port() is None:
        im = drawing.render()
        im.write_to_png('maze.png')
    else:
        axi.draw(drawing)


if __name__ == "__main__":
    main()
