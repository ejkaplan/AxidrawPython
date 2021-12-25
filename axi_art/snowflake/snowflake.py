from __future__ import annotations

from dataclasses import dataclass
import axi
import numpy as np
from axi import Drawing
from scipy.spatial import KDTree
from tqdm import tqdm

TEST = False
path_list = list[list[tuple[float, float]]]


@dataclass
class Particle:
    pos: np.ndarray
    heading: float
    children: list[Particle] = None

    def __post_init__(self):
        self.children = []

    def update(self, speed: float):
        vel = np.array([np.cos(self.heading), np.sin(self.heading)])
        self.pos += speed * vel

    def make_paths(self) -> path_list:
        paths = []
        for child in self.children:
            paths.append([tuple(self.pos), tuple(child.pos)])
            paths += child.make_paths()
        return paths


@dataclass
class DLA:
    root: Particle = None
    kdtree: KDTree = None
    particles: list[Particle] = None

    def __post_init__(self):
        self.root = Particle(np.array([0, 0]), 0)
        self.kdtree = KDTree(self.root.pos[None, :])
        self.particles = [self.root]

    def add_particle(self, particle: Particle) -> None:
        new_data = np.concatenate((self.kdtree.data, particle.pos[None, :]), axis=0)
        self.particles.append(particle)
        self.kdtree = KDTree(new_data)

    def distance(self, particle: Particle) -> tuple[float, Particle]:
        dd, ii = self.kdtree.query(particle.pos[None, :])
        return dd[0], self.particles[ii[0]]


def snowflake(n_points, attach_radius, outer_radius) -> Drawing:
    dla = DLA()
    for _ in tqdm(range(n_points), ncols=100):
        theta = np.random.normal(0, scale=0.25)
        if theta < 0:
            theta *= -1
        theta += np.pi/6
        reverse_theta = (theta + np.pi) % (2 * np.pi)
        particle = Particle(outer_radius * np.array([np.cos(theta), np.sin(theta)]),
                            reverse_theta)
        dist, neighbor = dla.distance(particle)
        if dist <= attach_radius:
            continue
        while True:
            particle.update(max(0.5*attach_radius, dist/10))
            dist, neighbor = dla.distance(particle)
            if dist <= attach_radius:
                neighbor.children.append(particle)
                dla.add_particle(particle)
                break
            elif dist > outer_radius:
                break
    paths = dla.root.make_paths()
    wedge_drawing = axi.Drawing(paths)
    wedge_drawing.add(wedge_drawing.scale(1, -1))
    drawing = Drawing()
    for i in range(6):
        drawing.add(wedge_drawing.rotate(i * 360 / 6))
    return drawing


def main():
    axi.device.MAX_VELOCITY = 2
    drawing = snowflake(2000, 1, 100000)
    drawing = drawing.scale_to_fit(11, 8.5, 1).sort_paths()
    drawing = drawing.join_paths(0.001)
    drawing = drawing.center(11, 8.5)

    if TEST or axi.device.find_port() is None:
        im = drawing.render()
        im.write_to_png('snowflake_preview.png')
    else:
        axi.draw(drawing)


if __name__ == '__main__':
    main()
