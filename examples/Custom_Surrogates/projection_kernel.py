# # Gaussian Processes in High-Dimensional Spaces

# A common challenge when applying Gaussian process models to high-dimensional spaces is
# to avoid model overfitting, which can easily happen when the hyperparameters of the
# model are not chosen with sufficient care {cite:p}`pmlr-v235-hvarfner24a`. While there
# exist various strategies to mitigate this issue, a simple yet effective approach is
# to model the data on a lower-dimensional subspace of the original parameter space.
# By reducing the effective dimensionality, this technique improves sample efficiency
# and reduces the risk of overfitting. Here, we demonstrate how to use BayBE's
# {class}`~baybe.kernels.composite.ProjectionKernel` to implement this idea.


# ## Imports

import os
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from baybe.kernels import MaternKernel, ProjectionKernel
from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe.surrogates.base import Surrogate
from baybe.surrogates.gaussian_process.kernel_factory import ProjectionKernelFactory
from baybe.surrogates.gaussian_process.presets.default import DefaultKernelFactory
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import arrays_to_dataframes
from baybe.utils.random import set_random_seed

# ## Settings

# Before we start, let us define some general settings for this example:

# Simulation Settings
SMOKE_TEST = "SMOKE_TEST" in os.environ
N_MC = 1 if SMOKE_TEST else 10

# Data Settings
N_TRAIN_DATA = [2] if SMOKE_TEST else [20, 50, 100, 200]
N_TEST_DATA = 100

# Problem Settings
N_DIMENSIONS_TOTAL = 3 if SMOKE_TEST else 20
N_DIMENSIONS_SUBSPACE = 2

# Projection Kernel Settings
N_PROJECTIONS = 2  # Dimensionality of the assumed subspace
N_MATRICES = 1 if SMOKE_TEST else 3  # Number of projection matrices to use

set_random_seed(1337)

# ## The Scenario

# To keep things simple, we consider a single target that needs to be modeled:

objective = NumericalTarget("t").to_objective()

# We consider a situation where the available regression parameters span a rather
# high-dimensional space:

parameters = [
    NumericalContinuousParameter(f"p{i}", (-1, 1)) for i in range(N_DIMENSIONS_TOTAL)
]
searchspace = SearchSpace.from_product(parameters)

# However, we assume that the actual relationship between the regression parameters and
# the target can be expressed via a low-dimensional function that is embedded into the
# higher-dimensional parameter space. To benchmark the proposed approach, let us
# generate this ground truth subspace by randomly drawing some unit-length vectors
# spanning it:

subspace = np.random.randn(N_DIMENSIONS_TOTAL, N_DIMENSIONS_SUBSPACE)
subspace /= np.linalg.norm(subspace, axis=0, keepdims=True)


# As a last component to describe the modeling scenario, we define the corresponding
# ground truth function that operates on this subspace:


@arrays_to_dataframes(searchspace.parameter_names, ["t"])
def low_dimensional_function(array: np.ndarray, /) -> np.ndarray:
    """The low-dimensional ground truth function.

    Computes the sigmoid transformation of the norm of its input.
    """
    assert array.shape[-1] == N_DIMENSIONS_TOTAL
    return expit(np.linalg.norm(array @ subspace, axis=1, keepdims=True))


# ## Subspace Modeling

# Now, let us turn to the modeling side. First, we define a little helper that lets
# us easily train and evaluate different surrogate models:


def predict(
    model: Surrogate, train_data: pd.DataFrame, test_data: pd.DataFrame
) -> pd.DataFrame:
    """Fit and evaluate a surrogate model."""
    model.fit(searchspace, objective, train_data)
    return model.posterior_stats(test_data[list(searchspace.parameter_names)])


# We compare the following models:
# - **Vanilla:**
#   The vanilla Gaussian process model operating on the **full** parameter space.
# - **Learned Projection:**
#   A Gaussian process model operating on **learned** low-dimensional subspaces.
# - **Ideal Projection:**
#   A Gaussian process model operating on the **ground truth** low-dimensional subspace.

models: dict[str, Surrogate] = {
    "Vanilla": GaussianProcessSurrogate(),
    "Learned Projection": GaussianProcessSurrogate(
        ProjectionKernelFactory(
            n_projections=N_PROJECTIONS,
            n_matrices=N_MATRICES,
            initialization="PLS",
            kernel_or_factory=DefaultKernelFactory(),
            learn_projection=True,
        )
    ),
    "Ideal Projection": GaussianProcessSurrogate(
        ProjectionKernel(
            MaternKernel(),
            projection_matrix=subspace,
            learn_projection=False,
        )
    ),
}

# We can now evaluate the regression performance of these models for different training
# data set sizes. The entire process is repeated for several Monte Carlo iterations:

metrics: list[dict[str, Any]] = []
for mc in range(N_MC):
    test_data = searchspace.continuous.sample_uniform(N_TEST_DATA)
    test_data["t"] = low_dimensional_function(test_data)
    for n_train_data in N_TRAIN_DATA:
        train_data = searchspace.continuous.sample_uniform(n_train_data)
        train_data["t"] = low_dimensional_function(train_data)
        for model_name, model in models.items():
            predictions = predict(model, train_data, test_data)
            r2 = r2_score(test_data["t"], predictions["t_mean"])
            correlation = spearmanr(test_data["t"], predictions["t_mean"]).correlation
            metrics.append(
                {
                    "mc_iteration": mc,
                    "model": model_name,
                    "n_train_data": n_train_data,
                    "rank_correlation": correlation,
                    "r2_score": r2,
                }
            )

df = pd.DataFrame.from_records(metrics)

# ## Results

# Finally, we visualize the resulting regression metrics. As expected, when operating on
# the ground truth subspace, the Gaussian process can easily capture the structure of
# the data, even for relatively small training set sizes. More importantly, even when
# learning the subspace from the training data itself (the realistic scenario where the
# ground truth is unknown), the projection approach identifies patterns in the data
# significantly earlier compared to the vanilla approach. This effect is more prominent
# for the underlying ordering of the target data (as reflected by the rank correlation)
# than for the actual prediction values. Nevertheless, when applied in a Bayesian
# optimization context, rank correlation is oftentimes the more important metric, since
# data-driven optimization heavily relies on correctly ranking candidates.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# fmt: off
sns.lineplot(df, x="n_train_data", y="rank_correlation", hue="model", ax=ax1)
ax1.set_title("Spearman Correlation")
ax1.set_xlabel("Number of Training Points")
ax1.set_ylabel("Rank Correlation")
# fmt: on

# fmt: off
sns.lineplot(df, x="n_train_data", y="r2_score", hue="model", ax=ax2)
ax2.set_title("R² Score")
ax2.set_xlabel("Number of Training Points")
ax2.set_ylabel("R² Score")
# fmt: on

plt.tight_layout()
if not SMOKE_TEST:
    plt.savefig("projection_kernel.svg")
plt.show()

# ```{bibliography}
# :filter: docname in docnames
# ```
