"""
Test to validate if the initial strategies work as expected
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from baybe.core import BayBE
from baybe.parameters import NumericDiscrete
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.strategy import Strategy
from baybe.targets import NumericalTarget, Objective

DIMENSIONS = 3
POINTS_PER_DIM = 10


def hartmann6(x1, x2, x3, x4, x5, x6):
    """
    self.bounds = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
    self.min = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    self.fmin = -3.32237
    """
    # pylint: disable=invalid-name
    sd = 0

    xx = np.array([x1, x2, x3, x4, x5, x6])
    if len(xx.shape) == 1:
        xx = xx.reshape((1, 6))

    assert xx.shape[1] == 6

    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    P = 1e-4 * np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )

    outer = 0
    for ii in range(4):
        inner = 0
        for jj in range(6):
            xj = xx[0, jj]
            Aij = A[ii, jj]
            Pij = P[ii, jj]
            inner = inner + Aij * (xj - Pij) ** 2

        new = alpha[ii] * np.exp(-inner)
        outer = outer + new

    y = -outer

    if sd == 0:
        noise = 0
    else:
        noise = np.random.normal(0, sd, 1)

    return y + noise


def hartmann3(x1, x2, x3):
    """
    bounds = np.array([[0, 1], [0, 1], [0, 1]])
    min = np.array([0.114614, 0.555649, 0.852547])
    fmin = -3.86278
    """
    # pylint: disable=invalid-name

    sd = 0

    xx = np.array([x1, x2, x3])
    if len(xx.shape) == 1:
        xx = xx.reshape((1, 3))

    assert xx.shape[1] == 3

    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [3, 10, 30],
            [0.1, 10, 35],
            [3, 10, 30],
            [0.1, 10, 35],
        ]
    )
    P = 1e-4 * np.array(
        [
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828],
        ]
    )

    outer = 0
    for ii in range(4):
        inner = 0
        for jj in range(3):
            xj = xx[0, jj]
            Aij = A[ii, jj]
            Pij = P[ii, jj]
            inner = inner + Aij * (xj - Pij) ** 2

        new = alpha[ii] * np.exp(-inner)
        outer = outer + new

    y = -outer

    if sd == 0:
        noise = 0
    else:
        noise = np.random.normal(0, sd, 1)

    return y + noise


scenarios = {
    name: BayBE(
        searchspace=SearchSpace.create(
            parameters=[
                NumericDiscrete(
                    name=f"x_{k+1}",
                    values=list(np.linspace(0, 1, POINTS_PER_DIM)),
                    tolerance=0.01,
                )
                for k in range(DIMENSIONS)
            ],
        ),
        strategy=Strategy(
            initial_recommender_cls=rec,
            allow_repeated_recommendations=False,
            allow_recommending_already_measured=False,
        ),
        objective=Objective(
            mode="SINGLE",
            targets=[NumericalTarget(name="Target", mode="MIN")],
        ),
        numerical_measurements_must_be_within_tolerance=True,
    )
    for name, rec in [
        ("Random", "RANDOM"),
        ("Farthest Point Sampling", "FPS"),
        ("KMEANS Clustering", "CLUSTERING_KMEANS"),
    ]
}

results = simulate_scenarios(
    scenarios=scenarios,
    lookup=hartmann3,
    n_exp_iterations=10,
    n_mc_iterations=5,
    batch_quantity=5,
)

print(results)

sns.lineplot(data=results, x="Num_Experiments", y="Target_CumBest", hue="Variant")
plt.gcf().set_size_inches(24, 8)
plt.savefig("./run_analytical.png")
