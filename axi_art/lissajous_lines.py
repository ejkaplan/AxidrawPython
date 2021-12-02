import axi
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

PAPER_WIDTH = 5.5
PAPER_HEIGHT = 4.25
PAPER_MARGIN = 0.5


def lissajous(lo, hi, samples):
    while True:
        x_coef, y_coef = np.random.uniform(1, 5, 2)
        x_off, y_off = np.random.uniform(-100, 100, 2)
        if np.abs(x_coef - y_coef) > 0.5:
            break
    w = PAPER_WIDTH - 2 * PAPER_MARGIN
    h = PAPER_HEIGHT - 2 * PAPER_MARGIN
    return [(w * np.cos(x_coef * theta + x_off), h * np.sin(y_coef * theta + y_off)) for theta in
            np.linspace(lo, hi, samples)]


def lissajous_lines(lo, hi, samples):
    curves = [lissajous(lo, hi, samples), lissajous(lo, hi, samples)]
    paths = [[curves[0][i], curves[1][i]] for i in range(samples)] * 2
    return axi.Drawing(paths)


def main():
    fig, ax = plt.subplots(figsize=(PAPER_WIDTH, PAPER_HEIGHT))
    ax.set_aspect('equal')
    axi.device.MAX_VELOCITY = 3
    while True:
        drawing = lissajous_lines(0, 4, 300).sort_paths().join_paths(0.01)
        drawing = drawing.scale_to_fit(PAPER_WIDTH, PAPER_HEIGHT, PAPER_MARGIN)
        drawing = drawing.center(PAPER_WIDTH, PAPER_HEIGHT)
        im = drawing.render()
        im.write_to_png('lissajous.png')
        img = mpimg.imread('lissajous.png')
        plt.imshow(img)
        plt.show()
        if input("Are you satisfied with this image? (y/n) ").lower()[0] == 'y':
            break
    if axi.device.find_port() is not None:
        axi.draw(drawing)


if __name__ == "__main__":
    main()
