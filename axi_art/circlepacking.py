import numpy as np
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

x_bounds = [-10, 10]
y_bounds = [-10, 10]
min_radius = 0.1
max_radius = 2

circles = []
n_tries = 5000

for i in range(n_tries):
    cx, cy = np.random.uniform(*x_bounds), np.random.uniform(*y_bounds)
    c = np.array([cx, cy])

    r = max_radius

    for ci, ri in circles:
        dist = np.linalg.norm(c - ci)
        largest_r = dist - ri
        largest_r = np.clip(largest_r, 0, largest_r)
        r = min(r, largest_r)

    if r >= min_radius:
        circles.append((c, r))

blue_patches = []
red_patches = []

for c, r in circles:

    if np.random.random() > 0.5:
        blue_patches.append(mpatches.Circle(c, r, fill=None, edgecolor='blue'))
    else:
        red_patches.append(mpatches.Circle(c, r, fill=None, edgecolor='red'))

fig, ax = plt.subplots(figsize=(10, 10))

plt.grid(False)
plt.axis('off')
ax.set_aspect('equal')

ax.set_xlim(x_bounds)
ax.set_ylim(y_bounds)

collection = PatchCollection(blue_patches + red_patches, match_original=True)
ax.add_collection(collection)

plt.show()
