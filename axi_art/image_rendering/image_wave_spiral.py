import axi
import axi.device

import numpy as np
import math
from PIL import Image
import requests

from axi_art.utils import map_range


def grid_interpolate(pixels, x, y, channel=None):
    x_off = x - math.floor(x)
    y_off = y - math.floor(y)
    if channel is None:
        ul = (1 - x_off) * (1 - y_off) * pixels[math.floor(y), math.floor(x)]
        dl = (1 - x_off) * y_off * pixels[math.ceil(y), math.floor(x)]
        ur = x_off * (1 - y_off) * pixels[math.floor(y), math.ceil(x)]
        dr = x_off * y_off * pixels[math.ceil(y), math.ceil(x)]
    else:
        ul = (1 - x_off) * (1 - y_off) * pixels[math.floor(y), math.floor(x), channel]
        dl = (1 - x_off) * y_off * pixels[math.ceil(y), math.floor(x), channel]
        ur = x_off * (1 - y_off) * pixels[math.floor(y), math.ceil(x), channel]
        dr = x_off * y_off * pixels[math.ceil(y), math.ceil(x), channel]
    return sum([ul, dr, ur, dl])


def im_spiral(pixels, n_points, segment_length, amp, channel):
    # TODO: Make the arguments more intuitive
    def spiral_sample(n, k):
        return np.sqrt(2) * np.sqrt(-1 + np.sqrt(1 + (k ** 2) * (n ** 2)))

    radius = min(pixels.shape[:2]) // 2
    theta_max = spiral_sample(n_points, segment_length)
    spiral_gap = map_range(2 * np.pi, 0, theta_max, 0, radius)
    path = []
    for i in range(n_points):
        theta = spiral_sample(i, segment_length) + np.pi
        r = map_range(theta, 0, theta_max, 0, radius)
        x, y = r * np.cos(theta) + pixels.shape[1] // 2, r * np.sin(theta) + pixels.shape[0] // 2
        x, y = int(x), int(y)
        # r += map_range(grid_interpolate(pixels, x, y, channel), 0, 255, amp, 0) * spiral_gap * np.cos(i)
        try:
            color = pixels[y, x, channel]
        except IndexError:
            pass
        r_offset = map_range(color, 0, 255, amp, 0) * spiral_gap * np.cos(i + channel * 2 * np.pi / 3)
        r += r_offset
        x, y = r * np.cos(theta), r * np.sin(theta)
        path.append((x, y))
    return path


def main():
    axi.device.MAX_VELOCITY = 1.5
    url = "https://mykindofmeeple.com/wp-content/uploads/2019/01/many-meeples-1602-27042020.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert('CMYK')
    pixels = np.asarray(img)
    paths = [im_spiral(pixels, 50000, 1, 0.5, 0)]  # Plot in this order: Yellow (2), Magenta (1), Cyan (0)
    drawing = axi.Drawing(paths).scale_to_fit(11, 8.5, 1)
    drawing = drawing.scale_to_fit(8.5, 5.5, 0.5).center(8.5, 5.5)
    if axi.device.find_port() is None:
        im = drawing.render()
        im.write_to_png('image_lines.png')
    else:
        axi.draw(drawing)


if __name__ == "__main__":
    main()
