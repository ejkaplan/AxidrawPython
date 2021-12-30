from __future__ import annotations

from dataclasses import dataclass

import axi
import numpy as np
from axi import Drawing, Font
from scipy.spatial import KDTree
from tqdm import tqdm

import winsound

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


def snowflake(n_points: int, attach_radius: float, outer_radius: float, symmetry: int) -> Drawing:
    dla = DLA()
    for _ in tqdm(range(n_points), ncols=100):
        theta = abs(np.random.normal(0, scale=0.25))
        theta += np.pi / symmetry
        reverse_theta = (theta + np.pi) % (2 * np.pi)
        particle = Particle(outer_radius * np.array([np.cos(theta), np.sin(theta)]),
                            reverse_theta)
        dist, neighbor = dla.distance(particle)
        if dist <= attach_radius:
            continue
        while True:
            particle.update(max(0.5 * attach_radius, dist / 10))
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
    for i in range(symmetry):
        drawing.add(wedge_drawing.rotate(i * 360 / symmetry))
    return drawing


def notify():
    winsound.Beep(523, 500)
    winsound.Beep(659, 500)
    winsound.Beep(784, 500)


TEST = False
WIDTH = 7
HEIGHT = 5
MARGIN = 0.5
N_POINTS = 150


def main():
    n_drawings = int(input("How many drawings would you like? "))
    for i in range(n_drawings):
        print(f"Generating drawing {i+1} of {n_drawings}.")
        seed = np.random.randint(0, 2 ** 31)
        np.random.seed(seed)
        axi.device.MAX_VELOCITY = 2
        drawing = snowflake(N_POINTS, 1, 100000, 6)
        drawing = drawing.scale_to_fit(WIDTH, HEIGHT, MARGIN).sort_paths()
        drawing = drawing.join_paths(0.001)
        f = Font(axi.FUTURAL, 10)
        text_drawing = f.text(f'{seed:0>10}-{N_POINTS}').translate(0.5, HEIGHT - 0.5)
        drawing = drawing.center(WIDTH, HEIGHT)
        drawing.add(text_drawing)

        notify()
        input("Press enter when you're ready to draw!")
        if TEST or axi.device.find_port() is None:
            im = drawing.render(bounds=(0, 0, WIDTH, HEIGHT))
            im.write_to_png('snowflake_preview.png')
            im.finish()
        else:
            axi.draw(drawing)


if __name__ == '__main__':
    main()
