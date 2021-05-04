from opensimplex import OpenSimplex
import axi
import random
import axi.device

from AxidrawPython.utils import map_range, Font, vertical_stack

seed = random.getrandbits(64)
noise = OpenSimplex(seed=seed)


def noise_line(y, width, amp, samples, noise_scale):
    path = []
    for i in range(samples):
        x = map_range(i, 0, samples, 0, width)
        if i < samples / 2:
            x_amp = map_range(i, 0, samples/2, 0, amp)
        else:
            x_amp = map_range(i, samples/2, samples, amp, 0)
        path.append((x, y + x_amp * noise.noise2d(noise_scale * x, y)))
    return path


def noise_field(lines, width, height, samples, amp, noise_scale):
    paths = []
    for i in range(lines):
        y = map_range(i, 0, lines, 0, height)
        paths.append(noise_line(y, width, amp, samples, noise_scale))
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
    paths = noise_field(100, 12, 8.5, 1000, 1.2, 0.7)
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
