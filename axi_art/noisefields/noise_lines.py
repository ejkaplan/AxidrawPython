from opensimplex import OpenSimplex
import axi
import random
import axi.device
import math

from axi_art.utils import map_range, Font, vertical_stack

seed = random.getrandbits(64)
noise = OpenSimplex(seed=seed)
random.seed(seed)


def noise_octaves(x, y, octaves, persistence):
    total = 0
    frequency = 1
    amplitude = 1
    max_value = 0
    for i in range(octaves):
        total += noise.noise2d(x * frequency, y * frequency) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= 2
    return 2 * total / max_value


def noise_line(y, width, amp, samples, x_noise_scale, y_noise_scale, octaves, persistence):
    path = []
    for i in range(samples):
        theta = map_range(i, 0, samples, -math.pi, math.pi)
        x_amp = map_range(math.cos(theta), -1, 1, 0, amp)
        x = map_range(i, 0, samples, 0, width)
        path.append((x, y + x_amp * noise_octaves(x * x_noise_scale, y * y_noise_scale, octaves, persistence)))
    return path


def noise_field(lines, width, height, samples, amp, x_noise_scale, y_noise_scale, octaves, persistence):
    paths = []
    for i in range(lines):
        y = map_range(i, 0, lines, 0, height)
        paths.append(noise_line(y, width, amp, samples, x_noise_scale, y_noise_scale, octaves, persistence))
    return paths


def occlude(paths, lookahead=None):
    if lookahead is None:
        lookahead = len(paths)
    out = []
    for i in range(len(paths) - 1):
        path = []
        for j in range(len(paths[i])):
            path_point = paths[i][j]
            point_ok = True
            for other in range(i + 1, min(lookahead + i + 1, len(paths))):
                below = paths[other][j]
                if path_point[1] > below[1]:
                    point_ok = False
                    break
            if point_ok:
                path.append(path_point)
            else:
                if len(path) > 1:
                    out.append(path)
                path = []
        if len(path) > 1:
            out.append(path)
    out.append(paths[-1])
    return out


def main():
    axi.device.MAX_VELOCITY = 2

    amp = random.gauss(0.75, 0.1)
    x_noise_scale = random.gauss(0.7, 0.05)
    y_noise_scale = random.gauss(0.7, 0.05)
    octaves = 4
    persistence = random.gauss(0.3, 0.1)
    print(amp, x_noise_scale, y_noise_scale, octaves, persistence)
    paths = noise_field(lines=100, width=12, height=8.5, samples=1000, amp=amp,
                        x_noise_scale=x_noise_scale, y_noise_scale=y_noise_scale, octaves=octaves,
                        persistence=persistence)
    paths = occlude(paths)
    drawing = axi.Drawing(paths).scale_to_fit(11, 8.5, 0).sort_paths()
    drawing = drawing.join_paths(0.03).simplify_paths(0.001)
    f = Font(axi.FUTURAL, 10)
    text_drawing = f.text(str(seed)).scale_to_fit(11, 0.1)
    drawing = vertical_stack([drawing, text_drawing], 0.2, False)
    drawing = drawing.scale_to_fit(11, 8.5, 0.5).center(11, 8.5).sort_paths()
    if axi.device.find_port() is None:
        im = drawing.render()
        im.write_to_png('noise_lines.png')
    else:
        axi.draw(drawing)


if __name__ == "__main__":
    main()
