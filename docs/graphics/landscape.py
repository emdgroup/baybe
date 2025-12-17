"""A 3D plot showing an optimization landscape."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

N_POINTS = 500
COLORS = ["#1a0033", "#3498db", "#316E91FF", "#f39c12", "#9e2a1d"]
N_COLOR_BINS = 256


def chebfun2(x, y):
    """https://www.chebfun.org/docs/guide/guide12.html"""  # noqa
    return (
        3 * (1 - x) ** 2.0 * np.exp(-(x**2) - (y + 1) ** 2)
        - 10 * (x / 5 - x**3 - y**5) * np.exp(-(x**2) - y**2)
        - 1 / 3 * np.exp(-((x + 1) ** 2) - y**2)
    )


x = np.linspace(-3, 3, N_POINTS)
y = np.linspace(-3, 3, N_POINTS)
X, Y = np.meshgrid(x, y)
Z = chebfun2(X, Y)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
fancy_cmap = LinearSegmentedColormap.from_list("fancy_logo", COLORS, N=N_COLOR_BINS)
surf = ax.plot_surface(X, Y, Z, cmap=fancy_cmap, edgecolor="k", linewidth=0.1)
ax.view_init(elev=20, azim=-120)
ax.set_box_aspect([1, 1, 0.5])
plt.axis("off")
plt.tight_layout()
plt.show()
