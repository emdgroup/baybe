"""# Top-Fraction Pareto Ranking (TFPR)

The purpose of this dashboard is to explore how the settings of the `TFPRObjective`
affect the recommendations of the `BotorchRecommender`, using synthetic quadratic
targets with a known Pareto set.
"""  # noqa: D415

import time
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from attrs import define, field
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from baybe import Settings
from baybe.objectives import ParetoObjective, TFPRObjective
from baybe.parameters import NumericalDiscreteParameter
from baybe.recommenders import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget

# Names of the two input parameters spanning the square [-1, 1]^2
PARAMETER_NAMES = ["x0", "x1"]

# The target optima lie on a regular polygon around the origin. Radius and starting
# angle are chosen such that two targets reproduce the centers of the Pareto example.
CENTER_RADIUS = np.sqrt(0.5)
CENTER_START_ANGLE = 1.25 * np.pi

# Maximum integer weight accepted by the TFPR objective
MAX_WEIGHT = 10

# Resolution of the grid used for drawing contour lines
N_CONTOUR_POINTS = 100

# Number of points sampled from the Pareto set for visualizing its image
N_PARETO_SAMPLES = 2000

# Number of Pareto set samples drawn in the parallel coordinates plot
N_PARALLEL_LINES = 300

# Factor applied to the number of candidates when scaling up the grid
SCALE_UP_FACTOR = 10

PARETO_EXAMPLE_URL = (
    "https://emdgroup.github.io/baybe/stable/examples/Multi_Target/pareto.html"
)

TFPR_MODEL = "TFPR (surrogate)"
TFPR_TRUTH = "TFPR (ground truth)"
QNEHVI = "qLogNEHVI"

MARKER_STYLES = {
    TFPR_MODEL: {"marker": "o", "color": "tab:purple", "markersize": 6},
    TFPR_TRUTH: {
        "marker": "o",
        "markerfacecolor": "none",
        "markeredgecolor": "tab:purple",
        "markeredgewidth": 1.5,
        "markersize": 12,
    },
    QNEHVI: {"marker": "s", "color": "tab:green", "markersize": 6},
}

TRAINING_STYLE = {
    "marker": "o",
    "color": "0.7",
    "markersize": 3,
    "linestyle": "none",
    "label": "training",
}

LINE_STYLES = {
    TFPR_MODEL: {"color": "tab:purple", "linewidth": 1.5},
    TFPR_TRUTH: {"color": "tab:purple", "linewidth": 1.5, "linestyle": "--"},
    QNEHVI: {"color": "tab:green", "linewidth": 1.5},
}


@define(frozen=True, eq=False)
class Problem:
    """A synthetic multi-target problem on a discretized parameter grid."""

    centers: np.ndarray = field()
    """The parameter values at which the individual targets attain their optimum."""

    resolution: int = field()
    """The number of grid values per parameter."""

    n_training_points: int = field()
    """The number of training points."""

    seed: int = field()
    """The random seed used for sampling the training points and fitting models."""

    targets: tuple[NumericalTarget, ...] = field()
    """The targets of the problem."""

    searchspace: SearchSpace = field()
    """The discrete search space containing the grid points."""

    values: np.ndarray = field()
    """The true target values of all candidates."""

    measurements: pd.DataFrame = field()
    """The training data."""

    @property
    def candidates(self) -> pd.DataFrame:
        """The candidates in experimental representation."""
        return self.searchspace.discrete.exp_rep


def make_centers(n_targets: int) -> np.ndarray:
    """Place the target optima on a regular polygon."""
    if n_targets == 2:
        # Use the exact centers of the Pareto example since the floating-point errors
        # of the polygon construction would break exact ties between candidates
        return np.array([[-0.5, -0.5], [0.5, 0.5]])
    angles = CENTER_START_ANGLE + 2 * np.pi * np.arange(n_targets) / n_targets
    return CENTER_RADIUS * np.c_[np.cos(angles), np.sin(angles)]


def make_targets(n_targets: int) -> tuple[NumericalTarget, ...]:
    """Create targets alternating between maximization and minimization."""
    return tuple(
        NumericalTarget(f"y{i}", minimize=i % 2 == 1) for i in range(n_targets)
    )


def evaluate(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Evaluate the target functions at the given parameter values.

    As in the Pareto example, even targets are negative squared distances to their
    center (maximized) and odd targets are scaled and shifted squared distances
    (minimized).
    """
    squared_distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    is_minimized = np.arange(len(centers)) % 2 == 1
    return np.where(is_minimized, 100 * squared_distances + 1000, -squared_distances)


def evaluate_recommendations(
    recommendations: pd.DataFrame, centers: np.ndarray
) -> np.ndarray:
    """Evaluate the target functions at the recommended parameter values."""
    return evaluate(recommendations[PARAMETER_NAMES].to_numpy(), centers)


def sample_convex_polygon(vertices: np.ndarray, n_samples: int) -> np.ndarray:
    """Sample points uniformly from a convex polygon with ordered vertices."""
    rng = np.random.default_rng(0)
    if len(vertices) == 2:
        fractions = rng.uniform(size=(n_samples, 1))
        return vertices[0] + fractions * (vertices[1] - vertices[0])

    # Fan triangulation from the first vertex, sampled proportionally to area
    a, b, c = vertices[0], vertices[1:-1], vertices[2:]
    ab, ac = b - a, c - a
    areas = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    triangles = rng.choice(len(areas), size=n_samples, p=areas / areas.sum())
    r1 = np.sqrt(rng.uniform(size=(n_samples, 1)))
    r2 = rng.uniform(size=(n_samples, 1))
    return a + r1 * (1 - r2) * ab[triangles] + r1 * r2 * ac[triangles]


def make_problem(
    n_targets: int, resolution: int, n_training_points: int, seed: int
) -> Problem:
    """Create the problem and sample training data from the candidate grid."""
    centers = make_centers(n_targets)
    targets = make_targets(n_targets)
    grid = np.linspace(-1, 1, resolution)
    searchspace = SearchSpace.from_product(
        [NumericalDiscreteParameter(name, grid) for name in PARAMETER_NAMES]
    )
    candidates = searchspace.discrete.exp_rep
    values = evaluate(candidates.to_numpy(), centers)

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(candidates), size=n_training_points, replace=False)
    measurements = candidates.iloc[indices].reset_index(drop=True)
    measurements[[t.name for t in targets]] = values[indices]

    return Problem(
        centers=centers,
        resolution=resolution,
        n_training_points=n_training_points,
        seed=seed,
        targets=targets,
        searchspace=searchspace,
        values=values,
        measurements=measurements,
    )


def recommend_tfpr(
    problem: Problem, objective: TFPRObjective, batch_size: int
) -> tuple[pd.DataFrame, float]:
    """Recommend candidates via TFPR on the surrogate posterior."""
    with Settings(random_seed=problem.seed):
        start = time.perf_counter()
        recommendations = BotorchRecommender().recommend(
            batch_size=batch_size,
            searchspace=problem.searchspace,
            objective=objective,
            measurements=problem.measurements,
        )
        elapsed = time.perf_counter() - start
    return recommendations, elapsed


def rank_ground_truth(
    problem: Problem, objective: TFPRObjective, batch_size: int
) -> pd.DataFrame:
    """Select the top candidates by TFPR fitness computed from the true values."""
    stats = {}
    for i, target in enumerate(problem.targets):
        stats[f"{target.name}_mean"] = problem.values[:, i]
        stats[f"{target.name}_std"] = np.zeros(len(problem.candidates))
    fitness = objective.compute_fitness(
        pd.DataFrame(stats, index=problem.candidates.index)
    )
    order = np.argsort(-fitness.to_numpy(), kind="stable")[:batch_size]
    return problem.candidates.iloc[order]


@st.cache_data(show_spinner="Running qLogNEHVI ...")
def recommend_qnehvi(
    n_targets: int, resolution: int, n_training_points: int, seed: int, batch_size: int
) -> tuple[pd.DataFrame, float]:
    """Recommend candidates via a Pareto objective (cached per problem setting)."""
    problem = make_problem(n_targets, resolution, n_training_points, seed)
    with Settings(random_seed=seed):
        start = time.perf_counter()
        recommendations = BotorchRecommender().recommend(
            batch_size=batch_size,
            searchspace=problem.searchspace,
            objective=ParetoObjective(problem.targets),
            measurements=problem.measurements,
        )
        elapsed = time.perf_counter() - start
    return recommendations, elapsed


def make_recommendations(
    problem: Problem, objective: TFPRObjective, batch_size: int, compare: bool
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """Collect the recommendations and runtimes of all requested methods."""
    recommendations: dict[str, pd.DataFrame] = {}
    runtimes: dict[str, float] = {}
    recommendations[TFPR_MODEL], runtimes[TFPR_MODEL] = recommend_tfpr(
        problem, objective, batch_size
    )
    recommendations[TFPR_TRUTH] = rank_ground_truth(problem, objective, batch_size)
    if compare:
        recommendations[QNEHVI], runtimes[QNEHVI] = recommend_qnehvi(
            len(problem.targets),
            problem.resolution,
            problem.n_training_points,
            problem.seed,
            batch_size,
        )
    return recommendations, runtimes


def problem_sidebar() -> tuple[int, int, int]:
    """Create the sidebar controls for the problem setup."""
    seed = int(
        st.sidebar.number_input(
            "Random seed", min_value=0, max_value=2**32 - 1, value=42, step=1
        )
    )
    n_training_points = st.sidebar.slider("Number of training points", 2, 100, 30)
    resolution = st.sidebar.slider(
        "Grid values per parameter",
        11,
        81,
        15,
        help="The parameter square is discretized into a grid of candidates since "
        "the current TFPR implementation supports discrete search spaces only.",
    )
    return seed, n_training_points, resolution


def tfpr_sidebar(
    targets: Sequence[NumericalTarget], weights: Sequence[int]
) -> TFPRObjective:
    """Create the sidebar controls for the remaining TFPR settings."""
    dominance_threshold = st.sidebar.slider(
        "Dominance threshold",
        0.5,
        0.99,
        0.5,
        0.01,
        help="Share of the total weight a candidate must secure in a pairwise "
        "comparison to dominate the other candidate. Won targets count with their "
        "full weight, tied targets with half of it. The default of 0.5 reproduces "
        "the original TFPR rule.",
    )
    optimism_lambda = st.sidebar.slider(
        "Optimism lambda",
        0.0,
        3.0,
        0.0,
        0.1,
        help="Multiple of the posterior standard deviation added in each target's "
        "favorable direction.",
    )
    with st.sidebar.expander("Tolerances", expanded=len(targets) <= 2):
        tolerances = [
            st.slider(
                f"Tolerance {t.name}",
                0.0,
                0.2,
                0.0,
                0.005,
                format="%.3f",
                help="Relative tolerance within which two values are considered tied.",
            )
            for t in targets
        ]
    automatic_top_fraction = st.sidebar.checkbox(
        "Automatic top fraction",
        value=True,
        help="Use the size-dependent rule, which keeps all candidates up to 5000 "
        "candidates.",
    )
    top_fraction = (
        None
        if automatic_top_fraction
        else st.sidebar.slider("Top fraction", 0.05, 1.0, 1.0, 0.05)
    )

    try:
        return TFPRObjective(
            targets,
            weights=weights,
            tolerances=tolerances,
            optimism_lambda=optimism_lambda,
            top_fraction=top_fraction,
            dominance_threshold=dominance_threshold,
        )
    except ValueError as ex:
        st.error(str(ex))
        st.stop()


def recommendation_sidebar(
    compare_default: bool, compare_help: str
) -> tuple[int, bool]:
    """Create the sidebar controls for the recommendation step."""
    st.sidebar.markdown("# Recommendations")
    batch_size = st.sidebar.slider("Number of recommendations", 1, 30, 30)
    compare = st.sidebar.toggle(
        f"Compare with {QNEHVI}", value=compare_default, help=compare_help
    )
    return batch_size, compare


def show_runtimes(problem: Problem, runtimes: dict[str, float]) -> None:
    """Display the number of candidates and the runtimes of the methods."""
    columns = st.columns(1 + len(runtimes))
    columns[0].metric("Candidates", f"{len(problem.candidates):,}")
    for column, (label, seconds) in zip(columns[1:], runtimes.items()):
        column.metric(f"{label} runtime", f"{seconds:.2f} s")
    st.caption(
        "Runtimes cover the full recommendation call including surrogate fitting. "
        f"{QNEHVI} results are cached per problem setting, so the displayed runtime "
        "refers to their first computation."
    )


def make_tie_interval(
    values: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the lower and upper ends of the values tied with the given values.

    Two values ``a`` and ``b`` are tied if ``|a - b| <= tolerance * max(|a|, |b|)``.
    For a fixed ``a`` and a tolerance below 1, this holds for all ``b`` between
    ``a * (1 - tolerance)`` and ``a / (1 - tolerance)``.
    """
    endpoints = np.stack([values * (1 - tolerance), values / (1 - tolerance)])
    return endpoints.min(axis=0), endpoints.max(axis=0)


def plot_tie_bands(problem: Problem, tolerances: Sequence[float]) -> Figure:
    """Plot the tie band width across the value range of each target."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, (target, tolerance) in enumerate(zip(problem.targets, tolerances)):
        if tolerance == 0:
            continue
        values = np.sort(problem.values[:, i])
        span = values[-1] - values[0]
        position = (values - values[0]) / span
        if target.minimize:
            position = 1 - position
        lower, upper = make_tie_interval(values, tolerance)
        ax.plot(
            position,
            100 * (upper - lower) / span,
            label=f"{target.name} (tolerance {tolerance:g})",
        )
    ax.set(
        xlabel="Position within the candidates' value range (0 = worst, 1 = best)",
        ylabel="Tie band width\n(% of value range)",
        title="Tie Bands Across Each Target's Value Range",
    )
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    return fig


def show_tie_bands(problem: Problem, tolerances: Sequence[float]) -> None:
    """Display how the relative tolerances translate into tie bands."""
    st.subheader("Tolerances")
    st.markdown(
        "TFPR considers two values `a` and `b` tied if "
        "`|a - b| <= tolerance * max(|a|, |b|)`. The tie band therefore scales with "
        "the magnitude of the values rather than with their spread. In this problem, "
        "the maximized targets peak at 0 while the minimized targets carry an "
        "offset of 1000, so identical tolerances produce very different tie bands. "
        "The bands shown are computed from the true target values. The recommender "
        "compares the posterior means shifted by the optimism term instead, whose "
        "magnitudes and hence tie bands differ accordingly."
    )
    if not any(tolerances):
        st.caption("Set a tolerance above zero to see the resulting tie bands.")
        return
    fig = plot_tie_bands(problem, tolerances)
    st.pyplot(fig)
    plt.close(fig)


def show_objective(objective: TFPRObjective) -> None:
    """Display the configured objective."""
    with st.expander("Objective"):
        st.code(str(objective), language=None)


def plot_two_targets(
    problem: Problem,
    recommendations: dict[str, pd.DataFrame],
    tolerances: Sequence[float],
) -> Figure:
    """Plot the two-target problem in parameter and target space."""
    center_0, center_1 = problem.centers
    name_0, name_1 = (t.name for t in problem.targets)
    training = problem.measurements

    mesh = np.linspace(-1, 1, N_CONTOUR_POINTS)
    x0_mesh, x1_mesh = np.meshgrid(mesh, mesh)
    mesh_values = evaluate(np.c_[x0_mesh.ravel(), x1_mesh.ravel()], problem.centers)
    front_x = np.linspace(center_0, center_1)
    front_y = evaluate(front_x, problem.centers)

    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(14, 7))

    # Parameter space
    for i, color in enumerate(["tab:red", "tab:blue"]):
        ax_x.contour(
            x0_mesh,
            x1_mesh,
            mesh_values[:, i].reshape(x0_mesh.shape),
            colors=color,
            alpha=0.2,
        )
    ax_x.plot(*front_x.T, "k", label="Pareto front")
    ax_x.plot(*training[PARAMETER_NAMES].to_numpy().T, **TRAINING_STYLE)
    for label, recs in recommendations.items():
        ax_x.plot(
            *recs[PARAMETER_NAMES].to_numpy().T,
            linestyle="none",
            label=label,
            **MARKER_STYLES[label],
        )
    ax_x.set(
        xlabel=PARAMETER_NAMES[0],
        ylabel=PARAMETER_NAMES[1],
        title="Parameter Space",
        xlim=(-1, 1),
        ylim=(-1, 1),
        aspect="equal",
    )
    ax_x.legend(loc="upper left")

    # Target space
    ax_y.plot(*problem.values.T, ".", color="0.9", ms=2, label="candidates", zorder=0)
    ax_y.plot(*front_y.T, "k", label="Pareto front")
    ax_y.plot(training[name_0], training[name_1], **TRAINING_STYLE)
    for label, recs in recommendations.items():
        outcomes = evaluate_recommendations(recs, problem.centers)
        ax_y.plot(*outcomes.T, linestyle="none", label=label, **MARKER_STYLES[label])
    if any(tolerances):
        outcomes = evaluate_recommendations(
            recommendations[TFPR_TRUTH], problem.centers
        )
        errors = []
        for values, tolerance in zip(outcomes.T, tolerances):
            lower, upper = make_tie_interval(values, tolerance)
            errors.append(np.stack([values - lower, upper - values]))
        ax_y.errorbar(
            *outcomes.T,
            xerr=errors[0],
            yerr=errors[1],
            fmt="none",
            ecolor="tab:purple",
            alpha=0.5,
            capsize=3,
            label=f"{TFPR_TRUTH} tie bands",
        )
    ax_y.set(xlabel=name_0, ylabel=name_1, title="Target Space")
    ax_y.legend(loc="upper left")

    fig.tight_layout()
    return fig


def plot_many_targets(
    problem: Problem, recommendations: dict[str, pd.DataFrame]
) -> Figure:
    """Plot the many-target problem in parameter space, PCA space and as lines."""
    centers = problem.centers
    target_names = [t.name for t in problem.targets]
    is_minimized = np.array([t.minimize for t in problem.targets])
    training = problem.measurements
    outcomes = {
        label: evaluate_recommendations(recs, centers)
        for label, recs in recommendations.items()
    }

    # The Pareto set is the convex hull of the target optima
    pareto_x = sample_convex_polygon(centers, N_PARETO_SAMPLES)
    pareto_y = evaluate(pareto_x, centers)

    fig = plt.figure(figsize=(14, 12))
    grid = fig.add_gridspec(2, 2)
    ax_x = fig.add_subplot(grid[0, 0])
    ax_pca = fig.add_subplot(grid[0, 1])
    ax_par = fig.add_subplot(grid[1, :])

    # Parameter space
    ax_x.fill(*centers.T, color="0.85", label="Pareto set")
    ax_x.plot(*np.r_[centers, centers[:1]].T, "k", linewidth=1)
    for name, center in zip(target_names, centers):
        ax_x.plot(*center, "k*", ms=10)
        ax_x.annotate(name, center, xytext=(5, 5), textcoords="offset points")
    ax_x.plot(*training[PARAMETER_NAMES].to_numpy().T, **TRAINING_STYLE)
    for label, recs in recommendations.items():
        ax_x.plot(
            *recs[PARAMETER_NAMES].to_numpy().T,
            linestyle="none",
            label=label,
            **MARKER_STYLES[label],
        )
    ax_x.set(
        xlabel=PARAMETER_NAMES[0],
        ylabel=PARAMETER_NAMES[1],
        title="Parameter Space",
        xlim=(-1, 1),
        ylim=(-1, 1),
        aspect="equal",
    )
    ax_x.legend(loc="upper left")

    # Target space via PCA of the standardized target values
    mean = problem.values.mean(axis=0)
    std = problem.values.std(axis=0)
    pca = PCA(n_components=2).fit((problem.values - mean) / std)

    def project(values: np.ndarray) -> np.ndarray:
        return pca.transform((values - mean) / std)

    ax_pca.plot(*project(problem.values).T, ".", color="0.9", ms=2, zorder=0)
    ax_pca.plot(*project(pareto_y).T, ".", color="0.3", ms=2, label="Pareto front")
    training_values = training[target_names].to_numpy()
    ax_pca.plot(*project(training_values).T, **TRAINING_STYLE)
    for label, values in outcomes.items():
        ax_pca.plot(
            *project(values).T, linestyle="none", label=label, **MARKER_STYLES[label]
        )
    variance = 100 * pca.explained_variance_ratio_
    ax_pca.set(
        xlabel=f"PC1 ({variance[0]:.0f}% variance)",
        ylabel=f"PC2 ({variance[1]:.0f}% variance)",
        title="Target Space (PCA of Standardized Values)",
    )
    ax_pca.legend(loc="upper left")

    # Parallel coordinates with each axis scaled to the candidates' value range
    lower = problem.values.min(axis=0)
    upper = problem.values.max(axis=0)

    def normalize(values: np.ndarray) -> np.ndarray:
        scaled = (values - lower) / (upper - lower)
        return np.where(is_minimized, 1 - scaled, scaled)

    positions = np.arange(len(target_names))
    for position in positions:
        ax_par.axvline(position, color="k", linewidth=0.5)
    lines = ax_par.plot(
        positions, normalize(pareto_y[:N_PARALLEL_LINES]).T, color="0.8", lw=0.5
    )
    lines[0].set_label("Pareto front")
    for label, values in outcomes.items():
        lines = ax_par.plot(positions, normalize(values).T, **LINE_STYLES[label])
        lines[0].set_label(label)
    ax_par.set_xticks(positions, target_names)
    ax_par.set(
        ylabel="Scaled value (1 = best candidate)",
        title="Target Space (Parallel Coordinates)",
        ylim=(-0.02, 1.02),
    )
    ax_par.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

    fig.tight_layout()
    return fig


def two_target_page() -> None:
    """Render the two-target page based on the Pareto example."""
    st.title("TFPR on the Pareto Example")
    st.markdown(
        f"This page turns the [Pareto example]({PARETO_EXAMPLE_URL}) into an "
        "interactive comparison for the `TFPRObjective`. Two TFPR selections are "
        f"shown: **{TFPR_MODEL}** is what the recommender picks based on the "
        f"surrogate posterior, and **{TFPR_TRUTH}** is what TFPR picks when ranking "
        "the true target values, i.e. with a perfect model. Training points are "
        "shown in light gray."
    )

    st.sidebar.markdown("# Problem")
    seed, n_training_points, resolution = problem_sidebar()
    problem = make_problem(2, resolution, n_training_points, seed)

    st.sidebar.markdown("# TFPR")
    # NOTE: Integer options are used since tuple values would turn the slider into a
    #   range slider.
    weight_y1 = st.sidebar.select_slider(
        "Weights (y0 : y1)",
        options=list(range(MAX_WEIGHT + 1)),
        value=MAX_WEIGHT // 2,
        format_func=lambda s: f"{MAX_WEIGHT - s} : {s}",
        help="Left puts all weight on y0, right puts all weight on y1.",
    )
    objective = tfpr_sidebar(problem.targets, (MAX_WEIGHT - weight_y1, weight_y1))
    batch_size, compare = recommendation_sidebar(
        compare_default=True,
        compare_help="Also recommend with a ParetoObjective, which uses the "
        "hypervolume-based qLogNEHVI acquisition function on the same candidates.",
    )

    recommendations, runtimes = make_recommendations(
        problem, objective, batch_size, compare
    )
    show_runtimes(problem, runtimes)
    fig = plot_two_targets(problem, recommendations, objective.tolerances)
    st.pyplot(fig)
    plt.close(fig)
    show_tie_bands(problem, objective.tolerances)
    show_objective(objective)


def many_target_page() -> None:
    """Render the page with additional experiments on many targets."""
    st.title("Extras: Many Targets")
    st.markdown(
        "The same construction with up to six quadratic targets whose optima lie on "
        "a regular polygon. The Pareto set is the polygon itself (the convex hull of "
        "the optima). Since the target space cannot be drawn directly, it is shown "
        "via PCA of the standardized target values and via parallel coordinates. "
        f"The **{TFPR_MODEL}** and **{TFPR_TRUTH}** selections are defined as on the "
        "two-target page."
    )

    st.sidebar.markdown("# Problem")
    n_targets = st.sidebar.slider("Number of targets", 2, 6, 6)
    seed, n_training_points, resolution = problem_sidebar()
    scale_up = st.sidebar.toggle(
        f"{SCALE_UP_FACTOR}x candidates",
        help=f"Multiply the number of candidates by {SCALE_UP_FACTOR} to compare "
        "the runtimes of the methods.",
    )
    if scale_up:
        resolution = round(resolution * np.sqrt(SCALE_UP_FACTOR))
    problem = make_problem(n_targets, resolution, n_training_points, seed)

    st.sidebar.markdown("# TFPR")
    with st.sidebar.expander("Weights", expanded=True):
        weights = [
            st.slider(f"Weight {t.name}", 0, MAX_WEIGHT, 1) for t in problem.targets
        ]
    objective = tfpr_sidebar(problem.targets, weights)
    batch_size, compare = recommendation_sidebar(
        compare_default=False,
        compare_help="Also recommend with a ParetoObjective, which uses the "
        "hypervolume-based qLogNEHVI acquisition function on the same candidates. "
        "This gets very slow with many targets: with six targets on a 41 x 41 "
        "grid, a local test took about 45 s for one recommendation and about "
        "13 min for ten. The runtime grows further with the number of candidates.",
    )

    recommendations, runtimes = make_recommendations(
        problem, objective, batch_size, compare
    )
    show_runtimes(problem, runtimes)
    fig = plot_many_targets(problem, recommendations)
    st.pyplot(fig)
    plt.close(fig)
    show_tie_bands(problem, objective.tolerances)
    show_objective(objective)


def main():
    """Create the streamlit dashboard."""
    st.set_page_config(page_title="TFPR", layout="wide")
    page = st.navigation(
        [
            st.Page(two_target_page, title="Two targets", default=True),
            st.Page(many_target_page, title="Extras: many targets"),
        ]
    )
    page.run()


if __name__ == "__main__":
    main()
