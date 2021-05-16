import axi
import random


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
    paths = [[(0, 0), (0, rows), (cols, rows), (cols, 0), (0, 0)]]
    for y in range(rows):
        for x in range(cols):
            paths.append(tile(x, y, 1, 1))
    return paths


TEST = False


def main():
    paths = ten_print(30, 40)
    drawing = axi.Drawing(paths).scale_to_fit(11, 8.5, 1).sort_paths()
    drawing = drawing.join_paths(0.03)
    drawing = drawing.center(11, 8.5)
    if TEST or axi.device.find_port() is None:
        im = drawing.render()
        im.write_to_png('10print.png')
    else:
        axi.draw(drawing)


if __name__ == "__main__":
    main()
