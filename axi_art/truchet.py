import axi
import random
import numpy as np


def lines():
    paths = []
    # horizontal lines
    for i in range(4):
        y = 1 / 8 + i / 4
        paths.append([(0, y), (1, y)])
    # horizontal lines
    for i in range(4):
        x = 1 / 8 + i / 4
        paths.append([(x, 0), (x, 1/8)])
        paths.append([(x, 7/8), (x, 1)])
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


def corner_circles(samples_per_segment=128):
    paths = []
    # Draw the circles centered on (0, 0)
    for i in range(4):
        radius = 1 / 8 + i / 4
        thetas = np.linspace(0, np.pi / 2, num=samples_per_segment)
        x = radius * np.cos(thetas)
        y = radius * np.sin(thetas)
        path = list(zip(x, y))
        paths.append(path)
    # Draw the circles centered on (1, 1)
    for i in range(4):
        radius = 1 / 8 + i / 4
        thetas = np.linspace(np.pi, 3 * np.pi / 2, num=samples_per_segment)
        x = 1 + radius * np.cos(thetas)
        y = 1 + radius * np.sin(thetas)
        points = np.stack(arrays=[x, y])
        path = []
        for point_idx in range(samples_per_segment):
            dist = np.linalg.norm(points[:, point_idx])
            if dist < 7 / 8:
                if len(path) > 0:
                    paths.append(path)
                    path = []
            else:
                path.append(tuple(points[:, point_idx]))
        paths.append(path)
    return axi.Drawing(paths)


def truchet_tiles(rows, cols):
    tiles = ([corner_circles, chevrons, lines], [1, 0, 3])
    out = axi.Drawing()
    for x in range(cols):
        for y in range(rows):
            tile = random.choices(tiles[0], weights=tiles[1])[0]()
            tile = tile.translate(-0.5, -0.5)
            tile = tile.rotate(random.choice([0, 1 / 4, 1 / 2, 3 / 4]) * 360)
            tile = tile.translate(x, y)
            out.add(tile)
    return out


def main():
    drawing = truchet_tiles(20, 20)
    drawing = drawing.scale_to_fit(11, 8.5, 1).sort_paths()
    drawing = drawing.join_paths(0.03)
    drawing = drawing.center(11, 8.5)
    if axi.device.find_port() is None:
        im = drawing.render()
        im.write_to_png('truchet.png')
    else:
        axi.draw(drawing)


if __name__ == "__main__":
    main()
