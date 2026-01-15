"""A 3D plot showing an optimization landscape."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

N_POINTS = 500
COLORS = ["#1a0033", "#3498db", "#316E91", "#f39c12", "#9e2a1d"]
# Colors:
# 1a0033 – dark indigo
# 3498db – light blue
# 316E91 – teal blue
# f39c12 – amber
# 9e2a1d – brick red
N_COLOR_BINS = 256


def chebfun2(x, y):
    """Chebfun test function for 2D optimization landscapes."""
    # https://www.chebfun.org/docs/guide/guide12.html
    return (
        3 * (1 - x) ** 2.0 * np.exp(-(x**2) - (y + 1) ** 2)
        - 10 * (x / 5 - x**3 - y**5) * np.exp(-(x**2) - y**2)
        - 1 / 3 * np.exp(-((x + 1) ** 2) - y**2)
    )


# Generate function data
x = np.linspace(-3, 3, N_POINTS)
y = np.linspace(-3, 3, N_POINTS)
X, Y = np.meshgrid(x, y)
Z = chebfun2(X, Y)

# 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
fancy_cmap = LinearSegmentedColormap.from_list("fancy_logo", COLORS, N=N_COLOR_BINS)
surf = ax.plot_surface(X, Y, Z, cmap=fancy_cmap, edgecolor="k", linewidth=0.1)
ax.view_init(elev=20, azim=-120)
ax.set_box_aspect([1, 1, 0.5])

# Remove tick marks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Make axis lines light gray
ax.xaxis.line.set_color("gray")
ax.yaxis.line.set_color("gray")
ax.zaxis.line.set_color("gray")

# Set axis line width
ax.xaxis.line.set_linewidth(5)
ax.yaxis.line.set_linewidth(5)
ax.zaxis.line.set_linewidth(5)

# Set axis line caps to round
ax.xaxis.line.set_solid_capstyle("round")
ax.yaxis.line.set_solid_capstyle("round")
ax.zaxis.line.set_solid_capstyle("round")

# Set pane alpha to 0 to make them transparent
ax.xaxis.pane.set_alpha(0)
ax.yaxis.pane.set_alpha(0)
ax.zaxis.pane.set_alpha(0)

plt.tight_layout()
plt.savefig("landscape.svg", transparent=True)
