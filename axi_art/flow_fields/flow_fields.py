from typing import Tuple, Union, Optional, Callable

import numpy as np
from axi import Drawing
from perlin_numpy import generate_perlin_noise_2d
from shapely.geometry import LineString, MultiLineString, Point, LinearRing
from tqdm import tqdm


def circle_pack(width: int, height: int, min_separation: float) -> list[np.ndarray]:
    outline = LinearRing([(0, 0), (0, width), (height, width), (height, 0)])
    distances = np.array(
        [[Point(x, y).distance(outline) for y in range(height)] for x in range(width)]
    )
    centers = []
    while len(options := np.argwhere(distances > min_separation)) > 0:
        center = options[np.random.randint(options.shape[0])]
        centers.append(center)
        i, j = np.indices(distances.shape, sparse=True)
        new_distances = np.maximum(
            0, np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
        )
        distances = np.minimum(distances, new_distances)
    return centers


def point_in_bounds(
    point: np.ndarray, low_r: float, low_c: float, high_r: float, high_c: float
) -> bool:
    assert point.shape == (2,)
    return low_r < round(point[0]) < high_r and low_c < round(point[1]) < high_c


def curl_noise(shape: Tuple[int, int], res: Tuple[int, int]) -> np.ndarray:
    shape = (shape[0], shape[1])
    noise = generate_perlin_noise_2d(shape, res)
    dx = (noise[1:-1, 2:] - noise[1:-1, :-2]) / 2
    dy = (noise[2:, 1:-1] - noise[:-2, 1:-1]) / 2
    return np.dstack((dx, -dy))


def blend_vector_fields(
    field_a: np.ndarray, field_b: np.ndarray, f_blend: Callable[[int, int], float]
) -> np.ndarray:
    """
    Blend two vector fields into a single vector field
    Args:
        field_a: The first field to be blended as 2d grid of velocities
        field_b: The second field to be blended
        f_blend: A function mapping coordinates to percentages, where 0 is the first vector field and 1 is the second

    Returns: The new vector field
    """
    if field_a.shape[0] < field_b.shape[0]:
        field_b = field_b[: field_a.shape[0], :, :]
    elif field_a.shape[0] > field_b.shape[0]:
        field_a = field_a[: field_b.shape[0], :, :]
    if field_a.shape[1] < field_b.shape[1]:
        field_b = field_b[:, : field_a.shape[1], :]
    elif field_a.shape[1] > field_b.shape[1]:
        field_a = field_a[:, : field_b.shape[1], :]
    blend = np.array(
        [
            [f_blend(c, r) for c in range(field_a.shape[1])]
            for r in range(field_a.shape[0])
        ]
    )[:, :, None]
    return (1 - blend) * field_a + blend * field_b


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


def single_linestring_through_field(
    field: np.ndarray,
    line_length: int,
    start_pos: Union[np.ndarray, tuple[float, float]],
    speed_mult: float,
    separation_dist: float,
    linestring_in_progress: Optional[MultiLineString],
) -> Optional[LineString]:
    if isinstance(start_pos, tuple):
        start_pos = np.ndarray(start_pos)
    if isinstance(start_pos, np.ndarray):
        assert start_pos.shape == (2,)
    points = np.zeros((line_length, 2))
    points[0] = start_pos
    for i in range(1, line_length):
        prev_point = points[i - 1]
        next_point = (
            prev_point + speed_mult * field[round(prev_point[0]), round(prev_point[1])]
        )
        if not point_in_bounds(next_point, 0, 0, *field.shape[:2]):
            break
        if (
            linestring_in_progress is not None
            and linestring_in_progress.distance(Point(next_point)) < separation_dist
        ):
            break
        points[i] = next_point
    points = points[~np.all(points == 0, axis=1)]
    if points.shape[0] >= 2:
        return LineString(points)


def line_strings_through_field(
    field: np.ndarray, line_length: int, line_separation: float
) -> MultiLineString:
    line_starts = circle_pack(field.shape[0], field.shape[1], line_separation * 0.8)
    line_strings = None
    consecutive_failures = 0
    for line_start in tqdm(line_starts):
        line = single_linestring_through_field(
            field, line_length, line_start, 4, 1, line_strings
        )
        if line is not None:
            consecutive_failures = 0
            if line_strings is None:
                line_strings = MultiLineString([line])
            else:
                line_strings = line_strings.union(line)
        else:
            consecutive_failures += 1
            if consecutive_failures > 0.05 * len(line_starts):
                break
    return line_strings


def assign_to_layers(
    paths: MultiLineString,
    n_layers: int,
    width: float,
    height: float,
    color_cohesion: float = 1,
) -> list[Drawing]:
    layers = [[] for _ in range(n_layers)]
    centers = [Point([width, height] * np.random.random(2)) for _ in range(n_layers)]
    for path in paths.geoms:
        dists = np.array(
            [(1 / path.distance(center)) ** color_cohesion for center in centers]
        )
        dists /= np.sum(dists)
        layer = np.random.choice(n_layers, p=dists)
        layers[layer].append(path.coords)
    return [Drawing(layer) for layer in layers]


def derive_grid_shape(
    width: float, height: float, margin: float, grid_res: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    grid_shape_x = round((width - 2 * margin) * 40)
    grid_shape_y = round((height - 2 * margin) * 40)
    if width > height:
        grid_res_x, grid_res_y = grid_res, round(grid_shape_y * grid_res / grid_shape_x)
    else:
        grid_res_x, grid_res_y = round(grid_shape_x * grid_res / grid_shape_y), grid_res
    grid_shape_x = grid_shape_x // grid_res_x * grid_res_x
    grid_shape_y = grid_shape_y // grid_res_y * grid_res_y
    return (grid_shape_x, grid_shape_y), (grid_res_x, grid_res_y)


def render_flow_field(
    field: np.ndarray,
    line_length: int,
    line_separation: float,
    n_colors: int,
    color_cohesion: float,
) -> list[Drawing]:
    assert field.ndim == 3
    assert field.shape[2] == 2
    lines = line_strings_through_field(field, line_length, line_separation)
    layers = assign_to_layers(
        lines, n_colors, field.shape[0], field.shape[1], color_cohesion
    )
    return layers
