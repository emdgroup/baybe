"""
Compatability test for surrogate models and botorch wrapping
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from baybe.acquisition import debotorchize
from baybe.parameters import NumericDiscrete
from baybe.recommender import MarginalRankingRecommender
from baybe.searchspace import SearchSpace
from baybe.surrogate import (
    BayesianLinearModel,
    GaussianProcessModel,
    NGBoostModel,
    RandomForestModel,
)

from baybe.utils import to_tensor
from botorch.acquisition import (
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
)

# Borrow function configuration from surrogate model streamlit
LOWER_PARAM_LIMIT = -10
UPPER_PARAM_LIMIT = 10

FUNCTION_AMP = 10
FUNCTION_BIAS = 0


# Construct a 2D sin test function
def sin_2d(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Sinusoid test function."""
    out = (
        FUNCTION_AMP
        * (
            np.sin(
                (arr1 - LOWER_PARAM_LIMIT)
                / (UPPER_PARAM_LIMIT - LOWER_PARAM_LIMIT)
                * 2
                * np.pi
            )
            + np.sin(
                (arr2 - LOWER_PARAM_LIMIT)
                / (UPPER_PARAM_LIMIT - LOWER_PARAM_LIMIT)
                * 2
                * np.pi
            )
        )
        + FUNCTION_BIAS
    )
    return out


# Create search space
N_PARAMETER_VALUES = 100

searchspace = pd.DataFrame(
    {
        "x1": np.linspace(LOWER_PARAM_LIMIT, UPPER_PARAM_LIMIT, N_PARAMETER_VALUES),
        "x2": np.linspace(LOWER_PARAM_LIMIT, UPPER_PARAM_LIMIT, N_PARAMETER_VALUES),
    }
)
Searchspace = SearchSpace(
    [
        NumericDiscrete(name="x1", values=searchspace["x1"].to_list()),
        NumericDiscrete(name="x2", values=searchspace["x2"].to_list()),
    ],
)

# Create y
targets = pd.DataFrame(sin_2d(arr1=searchspace["x1"], arr2=searchspace["x2"]))

N_TRAIN_POINTS = 10

# Randomly select some points for training
train_idx = np.random.choice(range(len(searchspace)), N_TRAIN_POINTS, replace=False)
train_x = to_tensor(searchspace.loc[train_idx])
train_y = to_tensor(targets.loc[train_idx])

# Printouts
# print(searchspace)
# print(targets)
# plt.plot(targets)
# plt.show()
# print(train_x)
# print(train_y)

# Select testing points if needed; mostly just use search space
# n_testing_points=10
# test_idx = np.random.choice(range(len(searchspace)), N_TRAIN_POINTS, replace=False)
# test_x = to_tensor(searchspace.loc[test_idx])

# Get best_f for acquisition function
best_f = train_y.max()

# All available surrogate models
surrogate_models = {
    "Gaussian Process": GaussianProcessModel,
    "Random Forest": RandomForestModel,
    "Natural Gradient Boosting": NGBoostModel,
    "Bayesian Linear Regression": BayesianLinearModel,
}

for name, mdl in surrogate_models.items():
    print("Model:", name)

    # Define and train model
    model = mdl(Searchspace)
    model.fit(train_x, train_y)

    # Define candidates with q=1; can also expand both t-batch & q-batch artificially
    candidates = to_tensor(searchspace).unsqueeze(
        1
    )  # .unsqueeze(0).unsqueeze(0).expand((5,2,100,5,2))

    # See mean, covar explicitly (Confirm size)
    # mean, covar = model.posterior(candidates)
    # print(candidates.size())
    # print(mean.size())
    # print(covar.size())

    # Define available acquisition functions
    acquisition_functions = [
        ExpectedImprovement,
        PosteriorMean,
        ProbabilityOfImprovement,
    ]

    # Get recommendations based on different acqf
    for acqf in acquisition_functions:
        acqf = debotorchize(acqf)(model, best_f)

        recommender = MarginalRankingRecommender(Searchspace, acqf)

        idxs = recommender.recommend(3)

        print(idxs)
