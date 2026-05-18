"""A 3D plot showing an optimization landscape."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

N_POINTS = 50  # Reduce for SVG size
COLORS = ["#1a0033", "#3498db", "#316E91", "#f39c12", "#9e2a1d"]
"""
Colors:
1a0033 – dark indigo
3498db – light blue
316E91 – teal blue
f39c12 – amber
9e2a1d – brick red
"""
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
fig = plt.figure(figsize=(2.5, 2))
ax = fig.add_subplot(111, projection="3d")
fancy_cmap = LinearSegmentedColormap.from_list("fancy_logo", COLORS, N=N_COLOR_BINS)
surf = ax.plot_surface(
    X,
    Y,
    Z,
    cmap=fancy_cmap,
    linewidth=0.0,  # Remove edge lines for SVG
)
ax.view_init(elev=20, azim=-120)
ax.set_box_aspect([1, 1, 0.5])

# Option used for figure without axes
plt.axis("off")

plt.tight_layout(pad=0)
plt.savefig(
    "landscape.svg",
    transparent=True,
    dpi=150,
    # Crops the image
    bbox_inches=fig.subplots_adjust(left=-0.1, right=1.05, top=3, bottom=-2),
    pad_inches=0,
)


def optimize_svg(svg_file):
    """Optimize the SVG file."""
    import shutil
    import subprocess

    # Check if svgo is available
    svgo_path = shutil.which("svgo")

    if svgo_path is None:
        print(
            "svgo for file size optimization is not installed. "
            + "Install it with: brew install svgo"
        )
        return False

    # Run svgo to optimize the SVG
    try:
        subprocess.run([svgo_path, svg_file, "-o", svg_file], check=True)
        print(f"SVG optimized: {svg_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error optimizing SVG: {e}")
        return False


# After saving the SVG, optimize it
optimize_svg("landscape.svg")
