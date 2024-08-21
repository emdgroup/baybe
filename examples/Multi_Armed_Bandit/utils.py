from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import quad
from scipy.stats import beta, rv_continuous

from baybe.utils.plotting import get_themes


def max_rv_distribution(
    dist_params: list[Iterable[float]], dist: rv_continuous
) -> list[float]:
    """Calculate the distribution of being the maximum RV in a set of independt RVs."""
    res = []
    for i, params in enumerate(dist_params):

        def integrand(x):
            product = dist.pdf(x, *params)
            for j, other_params in enumerate(dist_params):
                if j != i:
                    product *= dist.cdf(x, *other_params)
            return product

        probability, _ = quad(integrand, dist.a, dist.b)
        res.append(probability)
    return res


def create_mab_plot(
    n_iterations,
    n_arms,
    mc_regret_mean,
    mc_regret_std,
    mc_estimated_means_squared_error_mean,
    mc_estimated_means_squared_error_std,
    max_rv_distribution_over_iterations,
    posterior_params,
    real_means,
    estimated_means,
    acqf,
    base_name,
):
    """Create multi armed bandit plots for multiple mc runs."""
    COLORS = ["blue", "red", "green", "aqua", "pink"]
    themes, _ = get_themes()

    for theme_name in themes:
        if theme_name == "dark":
            plt.style.use("dark_background")
        else:
            plt.style.use("default")
        fig, [top_left_ax, middle_ax, bottom_ax] = plt.subplots(
            3,
            height_ratios=(0.25, 0.25, 0.5),
        )
        fig.set_figheight(7)
        top_right_ax = top_left_ax.twinx()

        def animate(i):
            """Animate one frame."""
            bottom_ax.clear()
            bottom_ax.set_xlim(-0.05, 1.05)
            bottom_ax.set_ylim(-0.1, 8)
            top_left_ax.clear()
            top_right_ax.clear()
            middle_ax.clear()
            middle_ax.set_xlim(0, n_iterations)
            middle_ax.set_ylim(0, 1)
            top_left_ax.set_xlim(0, n_iterations)
            top_left_ax.set_ylim(0, (mc_regret_mean + mc_regret_std).max() + 0.05)
            top_right_ax.set_ylim(
                0,
                (
                    mc_estimated_means_squared_error_mean
                    + mc_estimated_means_squared_error_std
                ).max()
                + 0.5,
            )
            current_params = posterior_params[i].T
            current_means = estimated_means[i]

            for j in range(n_arms):
                if current_means[j]:
                    bottom_ax.vlines(
                        x=current_means[j],
                        ymin=0,
                        ymax=0.5,
                        color=COLORS[j],
                        label=f"Est. {j}",
                        linestyle="dashed",
                    )
                x = np.arange(0, 1, 0.01)
                bottom_ax.plot(
                    x,
                    beta.pdf(x, *current_params[j]),
                    color=COLORS[j],
                    label=f"Dist. {j}",
                )
                bottom_ax.scatter(
                    [real_means[j]],
                    [0],
                    color=COLORS[j],
                    label=f"Real {j}",
                )
            x = list(range(i))
            top_left_ax.plot(
                x,
                mc_regret_mean[:i],
                color="red",
                label="Regret",
            )
            top_left_ax.fill_between(
                x,
                mc_regret_mean[:i] + mc_regret_std[:i],
                mc_regret_mean[:i] - mc_regret_std[:i],
                facecolor="red",
                alpha=0.2,
            )
            top_right_ax.plot(
                x,
                mc_estimated_means_squared_error_mean[:i],
                color="blue",
                label="MSE",
            )
            top_right_ax.fill_between(
                x,
                mc_estimated_means_squared_error_mean[:i]
                + mc_estimated_means_squared_error_std[:i],
                mc_estimated_means_squared_error_mean[:i]
                - mc_estimated_means_squared_error_std[:i],
                facecolor="blue",
                alpha=0.2,
            )
            middle_ax.stackplot(
                x,
                *max_rv_distribution_over_iterations[:i].T,
                labels=[f"Arm {i+1}" for i in range(n_arms)],
                colors=COLORS,
            )
            middle_ax.set_xlabel("Iterations")
            middle_ax.set_ylabel(r"$P(X_i > X_{-i})$")

            bottom_ax.set_xlabel("Win Rate")
            bottom_ax.set_ylabel("Density")
            top_left_ax.set_ylabel("Regret")
            top_right_ax.set_ylabel("MSE")
            top_right_ax.yaxis.set_label_position("right")

            bottom_ax.legend(loc="upper left", ncol=n_arms, prop={"size": 8})
            top_left_ax.legend(loc="upper left", ncol=1, prop={"size": 8})
            top_right_ax.legend(loc="upper right", ncol=1, prop={"size": 8})
            plt.tight_layout()

        ani = FuncAnimation(
            fig,
            animate,
            interval=40,
            blit=False,
            repeat=False,
            frames=n_iterations - 1,
        )
        fig.suptitle(
            f"Optimizing a Bernoulli Multi-Armed Bandit with {acqf.__class__.__name__}"
        )
        output_path = Path(
            ".", f"{base_name}_{acqf.__class__.__name__}_{theme_name}.gif"
        )
        ani.save(
            output_path,
            dpi=300,
            writer=PillowWriter(fps=15),
        )
