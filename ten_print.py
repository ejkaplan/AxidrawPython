import axi
import random
from utils import merge_paths


def tile(x, y, w, h):
    points = []
    if random.random() < 0.5:
        points.append((x, y))
        points.append((x + w, y + h))
    else:
        points.append((x + w, y))
        points.append((x, y + h))
    return points


def ten_print(rows, cols):
    paths = []
    for y in range(cols):
        for x in range(rows):
            paths.append(tile(x, y, 1, 1))
    return paths


def main():
    paths = ten_print(30, 20)
    paths = merge_paths(paths)
    drawing = axi.Drawing(paths).scale_to_fit(11, 8.5, 1).sort_paths()
    drawing = drawing.center(11, 8.5)
    im = drawing.render()
    # im.write_to_png('out.png')
    axi.draw(drawing)


if __name__ == "__main__":
    main()
