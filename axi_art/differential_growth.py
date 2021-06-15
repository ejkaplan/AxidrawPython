import numpy as np
from scipy.spatial import KDTree
from matplotlib import pyplot as plt


def circle_points(r, n):
    """Returns a circle of points as a list of tuples"""
    return [np.array([r * np.cos(theta), r * np.sin(theta)]) for theta in np.linspace(0, 2 * np.pi, n + 1)[:-1]]


def clamp_forces(forces, magnitude):
    for i, force in enumerate(forces):
        mag = np.linalg.norm(force)
        if mag > magnitude:
            forces[i] = magnitude * force / mag


def attractive_force(points, coef, forces=None):
    """Each node tries to get closer to its 2 connected neighbors"""
    if forces is None:
        forces = [np.zeros(2) for _ in range(len(points))]
    for i, point in enumerate(points):
        for neighbor in [points[(i - 1) % len(points)], points[(i + 1) % len(points)]]:
            d = np.linalg.norm(neighbor - point)
            forces[i] += (neighbor - point) * (coef / d)
    return forces


def repulsive_force(points, coef, repel_dist, forces=None):
    if forces is None:
        forces = [np.zeros(2) for _ in range(len(points))]
    tree = KDTree(points)
    repellors = tree.query_ball_point(points, r=repel_dist)
    for i, point in enumerate(points):
        for j in repellors[i]:
            if i == j:
                continue
            rpoint = points[j]
            d = np.linalg.norm(point - rpoint)
            forces[i] += (point - rpoint) * (coef / d)
    return forces


def alignment_force(points, coef, forces=None):
    if forces is None:
        forces = [np.zeros(2) for _ in range(len(points))]
    for i, point in enumerate(points):
        midpoint = (points[(i - 1) % len(points)] + points[(i + 1) % len(points)]) / 2
        if np.linalg.norm(midpoint - point) > 0.0001:
            d = np.linalg.norm(midpoint - point)
            forces[i] += (midpoint - point) * (coef / d)
    return forces


def grow(points):
    i = np.random.randint(len(points))
    point_a, point_b = points[i], points[(i + 1) % len(points)]
    new_point = (point_a + point_b) / 2
    points.insert(i + 1, new_point)


def main():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    points = circle_points(1, 40)
    for x in range(1000):
        if x % 10 == 0:
            print(x)
        grow(points)
        forces = attractive_force(points, 0.05)
        repulsive_force(points, 0.06, 10, forces=forces)
        alignment_force(points, 0.02, forces=forces)
        clamp_forces(forces, 0.02)
        for i in range(len(points)):
            points[i] += forces[i]
    plt.plot([point[0] for point in points] + [points[0][0]], [point[1] for point in points] + [points[0][1]])

    plt.show()


if __name__ == "__main__":
    main()
