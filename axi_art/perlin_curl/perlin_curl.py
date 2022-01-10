from typing import Tuple, Union, Optional

import axi
import numpy as np
from axi import Drawing
from perlin_numpy import generate_perlin_noise_2d
from shapely.geometry import LineString, MultiLineString, Point, LinearRing
from tqdm import tqdm


def circle_pack(width: int, height: int, min_separation: float) -> list[np.ndarray]:
    outline = LinearRing([(0, 0), (0, width), (height, width), (height, 0)])
    distances = np.array([[Point(x, y).distance(outline) for y in range(height)] for x in range(width)])
    centers = []
    while len(options := np.argwhere(distances > min_separation)) > 0:
        center = options[np.random.randint(options.shape[0])]
        centers.append(center)
        i, j = np.indices(distances.shape, sparse=True)
        new_distances = np.maximum(0, np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2))
        distances = np.minimum(distances, new_distances)
    return centers


def point_in_bounds(point: np.ndarray, low_r: float, low_c: float, high_r: float, high_c: float) -> bool:
    assert point.shape == (2,)
    return low_r < round(point[0]) < high_r and low_c < round(point[1]) < high_c


def curl_noise(shape: Tuple[int, int], res: Tuple[int, int]) -> np.ndarray:
    shape = (shape[0], shape[1])
    noise = generate_perlin_noise_2d(shape, res)
    dx = (noise[1:-1, 2:] - noise[1:-1, :-2]) / 2
    dy = (noise[2:, 1:-1] - noise[:-2, 1:-1]) / 2
    return np.dstack((dx, -dy))


def grid_render_vector_field(field: np.ndarray, line_length: float) -> Drawing:
    assert field.ndim == 3
    assert field.shape[2] == 2
    lines = []
    for x in range(field.shape[1]):
        for y in range(field.shape[0]):
            vy, vx = line_length * field[y, x]
            line = LineString([(x, y), (x + vx, y + vy)])
            if len(line.coords) == 2:
                lines.append(list(line.coords))
    return Drawing(lines)


def single_linestring_through_field(field: np.ndarray, line_length: int,
                                    start_pos: Union[np.ndarray, tuple[float, float]],
                                    speed_mult: float, separation_dist: float,
                                    linestring_in_progress: Optional[MultiLineString]) -> Optional[LineString]:
    if isinstance(start_pos, tuple):
        start_pos = np.ndarray(start_pos)
    if isinstance(start_pos, np.ndarray):
        assert start_pos.shape == (2,)
    points = np.zeros((line_length, 2))
    points[0] = start_pos
    for i in range(1, line_length):
        prev_point = points[i - 1]
        if not point_in_bounds(prev_point, 0, 0, *field.shape[:2]):
            break
        next_point = prev_point + speed_mult * field[round(prev_point[0]), round(prev_point[1])]
        if linestring_in_progress is not None and linestring_in_progress.distance(Point(next_point)) < separation_dist:
            break
        points[i] = next_point
    points = points[~np.all(points == 0, axis=1)]
    if points.shape[0] >= 2:
        return LineString(points)


def line_strings_through_field(field: np.ndarray, line_length: int, line_separation: float) -> MultiLineString:
    line_starts = circle_pack(field.shape[0], field.shape[1], line_separation)
    line_strings = None
    for line_start in tqdm(line_starts):
        line = single_linestring_through_field(field, line_length,
                                               line_start,
                                               4, 1, line_strings)
        if line is not None:
            if line_strings is None:
                line_strings = MultiLineString([line])
            else:
                line_strings = line_strings.union(line)
    return line_strings


def assign_to_layers(paths: MultiLineString, n_layers: int, width: float, height: float,
                     color_cohesion: float = 1) -> list[Drawing]:
    layers = [[] for _ in range(n_layers)]
    centers = [Point([width, height] * np.random.random(2)) for _ in range(n_layers)]
    for path in paths:
        dists = np.array([(1 / path.distance(center)) ** color_cohesion for center in centers])
        dists /= np.sum(dists)
        layer = np.random.choice(n_layers, p=dists)
        layers[layer].append(path.coords)
    return [Drawing(layer) for layer in layers]


def main(test: bool):
    curl = curl_noise((200, 200), (4, 4))
    lines = line_strings_through_field(curl, 200, 4)
    layers = assign_to_layers(lines, 3, 200, 200, 2)
    layers = Drawing.multi_scale_to_fit(layers, 10, 10, 0.5)
    layers = [layer.sort_paths() for layer in layers]
    if test or axi.device.find_port() is None:
        im = Drawing.render_layers(layers, bounds=(0, 0, 10, 10))
        im.write_to_png('curl_preview.png')
        im.finish()
    else:
        axi.draw_layers(layers)


if __name__ == '__main__':
    main(True)
