import axi
import axi.device

import numpy as np
import math
from PIL import Image
import requests

from axi_art.utils import map_range


def grid_interpolate(pixels, x, y):
    x_off = x - math.floor(x)
    y_off = y - math.floor(y)
    ul = (1 - x_off) * (1 - y_off) * pixels[math.floor(y), math.floor(x)]
    dl = (1 - x_off) * y_off * pixels[math.ceil(y), math.floor(x)]
    ur = x_off * (1 - y_off) * pixels[math.floor(y), math.ceil(x)]
    dr = x_off * y_off * pixels[math.ceil(y), math.ceil(x)]
    return sum([ul, dr, ur, dl])


def im_line(pixels, y, width, samples, amp, waves_per_line):
    path = []
    freq = 2 * np.pi * waves_per_line / width
    for i in range(samples):
        x = map_range(i, 0, samples, 0, width - 1)
        lum = grid_interpolate(pixels, x, y)
        y_off = np.sin(freq * x) * map_range(lum, 0, 255, amp, 0)
        path.append((x, y + y_off))
    return path


def im_lines(lines, pixels, samples, waves_per_line):
    height, width = pixels.shape
    amp = 0.5 * height / lines
    paths = []
    for i in range(lines):
        y = map_range(i, 0, lines, 0, height)
        paths.append(im_line(pixels, y, width, samples, amp, waves_per_line))
    return paths


def main():
    axi.device.MAX_VELOCITY = 2
    url = (
        "https://pbs.twimg.com/profile_images/1309133913953099776/PEgTVuQB_400x400.jpg"
    )
    img = Image.open(requests.get(url, stream=True).raw).convert("L")
    pixels = np.asarray(img)
    paths = im_lines(100, pixels, 1000, 150)
    drawing = axi.Drawing(paths).scale_to_fit(11, 8.5, 0).sort_paths()
    drawing = drawing.join_paths(0.03).simplify_paths(0.001)
    drawing = drawing.scale_to_fit(11, 8.5, 0.5).center(11, 8.5).sort_paths()
    if axi.device.find_port() is None:
        im = drawing.render()
        im.write_to_png("image_lines.png")
    else:
        axi.draw(drawing)


if __name__ == "__main__":
    main()
