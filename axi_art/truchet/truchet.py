import axi
import random
import numpy as np


def horizontal_lines():
    paths = []
    # horizontal lines
    for i in range(4):
        y = 1 / 8 + i / 4
        paths.append([(0, y), (1, y)])
    # horizontal lines
    for i in range(4):
        x = 1 / 8 + i / 4
        paths.append([(x, 0), (x, 1 / 8)])
        paths.append([(x, 7 / 8), (x, 1)])
    return axi.Drawing(paths)


def diagonal_lines():
    paths = [
        [(0, 1 / 8), (1 / 8, 0)],
        [(0, 3 / 8), (3 / 8, 0)],
        [(0, 5 / 8), (5 / 8, 0)],
        [(0, 7 / 8), (7 / 8, 0)],
        [(1 / 8, 1), (1, 1 / 8)],
        [(3 / 8, 1), (1, 3 / 8)],
        [(5 / 8, 1), (1, 5 / 8)],
        [(7 / 8, 1), (1, 7 / 8)]
    ]
    return axi.Drawing(paths)


def chevrons():
    paths = [
        [(0, 3 / 8), (1 / 8, 4 / 8), (0, 5 / 8)],
        [(0, 1 / 8), (3 / 8, 4 / 8), (0, 7 / 8)],
        [(1 / 8, 0), (5 / 8, 4 / 8), (1 / 8, 1)],
        [(3 / 8, 0), (7 / 8, 4 / 8), (3 / 8, 1)],
        [(5 / 8, 0), (1, 3 / 8)],
        [(1, 5 / 8), (5 / 8, 1)],
        [(7 / 8, 0), (1, 1 / 8)],
        [(1, 7 / 8), (7 / 8, 1)]
    ]
    return axi.Drawing(paths)


def corner_circles(samples_per_circle=32):
    paths = []
    # Draw the circles centered on (0, 0)
    for i in range(4):
        radius = 1 / 8 + i / 4
        thetas = np.linspace(0, np.pi / 2, num=samples_per_circle)
        x = radius * np.cos(thetas)
        y = radius * np.sin(thetas)
        path = list(zip(x, y))
        paths.append(path)
    # Draw the circles centered on (1, 1)
    for i in range(4):
        radius = 1 / 8 + i / 4
        thetas = np.linspace(np.pi, 3 * np.pi / 2, num=samples_per_circle)
        x = 1 + radius * np.cos(thetas)
        y = 1 + radius * np.sin(thetas)
        points = np.stack(arrays=[x, y])
        path = []
        for point_idx in range(samples_per_circle):
            dist = np.linalg.norm(points[:, point_idx])
            if dist < 7 / 8:
                if len(path) > 1:
                    paths.append(path)
                path = []
            else:
                path.append(tuple(points[:, point_idx]))
        if len(path) > 1:
            paths.append(path)
    return axi.Drawing(paths)


def edge_circles(samples_per_circle=32):
    paths = []
    # Draw the circles centered on (0, 0.5)
    for i in range(2):
        radius = 1 / 8 + i / 4
        thetas = np.linspace(-np.pi / 2, np.pi / 2, num=samples_per_circle)
        x = radius * np.cos(thetas)
        y = 0.5 + radius * np.sin(thetas)
        path = list(zip(x, y))
        paths.append(path)
    # Draw the circles centered on (1, 0.5)
    for i in range(2):
        radius = 1 / 8 + i / 4
        thetas = np.linspace(np.pi / 2, 3 * np.pi / 2, num=samples_per_circle)
        x = 1 + radius * np.cos(thetas)
        y = 0.5 + radius * np.sin(thetas)
        path = list(zip(x, y))
        paths.append(path)
    # draw the lines down the middle
    for i in range(4):
        path = []
        x = 1 / 8 + i / 4
        for j in range(samples_per_circle):
            y = j / samples_per_circle
            point = np.array([x, y])
            dist = min(np.linalg.norm(point - [0, 0.5]), np.linalg.norm(point - [1, 0.5]))
            if dist < 3 / 8:
                if len(path) > 1:
                    paths.append(path)
                path = []
            else:
                path.append((x, y))
        if len(path) > 1:
            paths.append(path)
    return axi.Drawing(paths)


def truchet_tiles(rows, cols):
    tiles = ([corner_circles, horizontal_lines, edge_circles, chevrons, diagonal_lines], [1, 0, 0, 0, 0])
    out = axi.Drawing()
    for x in range(cols):
        for y in range(rows):
            tile = random.choices(tiles[0], weights=tiles[1])[0]()
            tile = tile.translate(-0.5, -0.5)
            tile = tile.rotate(random.choice([0, 0.25, 0.5, 0.75]) * 2 * np.pi)
            tile = tile.translate(x, y)
            out.add(tile)
    return out


TEST = False


def main():
    drawing = truchet_tiles(15, 20).sort_paths().join_paths(0.01)
    drawing = drawing.scale_to_fit(11, 8.5, 0.5)
    drawing = drawing.center(11, 8.5)
    if TEST or axi.device.find_port() is None:
        im = drawing.render(bounds=(0, 0, 11, 8.5))
        im.write_to_png('truchet.png')
    else:
        axi.draw(drawing)


if __name__ == "__main__":
    main()
