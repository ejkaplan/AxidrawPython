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


def im_spiral(pixels, n_points, segment_length, amp):
    # TODO: Make the arguments more intuitive
    def spiral_sample(n, k):
        return np.sqrt(2) * np.sqrt(-1 + np.sqrt(1 + (k ** 2) * (n ** 2)))
    radius = (min(pixels.shape)-1) // 2
    theta_max = spiral_sample(n_points, segment_length)
    spiral_gap = map_range(2*np.pi, 0, theta_max, 0, radius)
    path = []
    for i in range(n_points):
        theta = spiral_sample(i, segment_length)
        r = map_range(theta, 0, theta_max, 0, radius)
        x, y = r * np.cos(theta) + pixels.shape[1]/2, r * np.sin(theta) + pixels.shape[0]/2
        r += map_range(grid_interpolate(pixels, x, y), 0, 255, amp, 0) * spiral_gap * np.cos(i)
        x, y = r * np.cos(theta), r * np.sin(theta)
        path.append((x, y))
    return path


def main():
    axi.device.MAX_VELOCITY = 2
    url = "https://pbs.twimg.com/profile_images/1309133913953099776/PEgTVuQB_400x400.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert('L')
    pixels = np.asarray(img)
    paths = [im_spiral(pixels, 150000, 1, 1.1)]
    drawing = axi.Drawing(paths).scale_to_fit(11, 8.5, 0)
    drawing = drawing.scale_to_fit(11, 8.5, 0.5).center(11, 8.5)
    if axi.device.find_port() is None:
        im = drawing.render()
        im.write_to_png('image_lines.png')
    else:
        axi.draw(drawing)


if __name__ == "__main__":
    main()
