import numpy as np
from scipy.spatial import KDTree
from matplotlib import pyplot as plt


def circle_points(r, n):
    """Returns a circle of points as a list of tuples"""
    return [np.array([r * np.cos(theta), r * np.sin(theta)]) for theta in np.linspace(0, 2 * np.pi, n)]


def distances(points):
    """Returns an array of the distances between pairs of points, where the 0th element is the distance between the 0th
    and 1th points"""
    return [np.linalg.norm((points[i][0] - points[(i + 1) % len(points)][0],
                            points[i][1] - points[(i + 1) % len(points)][1]))
            for i in range(len(points))]


def main():
    points = circle_points(1, 10)
    print(distances(points))
    plt.plot([point[0] for point in points], [point[1] for point in points])
    plt.show()


if __name__ == "__main__":
    main()
