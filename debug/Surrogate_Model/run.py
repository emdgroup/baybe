from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional, Type, Union

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from botorch.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
)

from baybe.acquisition import debotorchize
from baybe.recommender import MarginalRankingRecommender
from baybe.surrogate import (
    SurrogateModel,
    GaussianProcessModel, 
    RandomForestModel, 
    NGBoostModel, 
    BayesianLinearModel
)

from baybe.utils import to_tensor

# Borrow function configuration from surrogate model streamlit
lower_parameter_limit = -10
upper_parameter_limit = 10

function_amplitude = 10
function_bias = 0

# Construct a 2D sin test function
def sin_2d(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Sinusoid test function."""
    out = (
        function_amplitude
        * (
            np.sin(
                (arr1 - lower_parameter_limit)
                / (upper_parameter_limit - lower_parameter_limit)
                * 2
                * np.pi
            ) +
            np.sin(
                (arr2 - lower_parameter_limit)
                / (upper_parameter_limit - lower_parameter_limit)
                * 2
                * np.pi
            )
        )
        + function_bias
    )
    return out

# Create search space
N_PARAMETER_VALUES = 100

searchspace = pd.DataFrame(
    {
        "x1": np.linspace(lower_parameter_limit, upper_parameter_limit, N_PARAMETER_VALUES),
        "x2": np.linspace(lower_parameter_limit, upper_parameter_limit, N_PARAMETER_VALUES),
    }
)

# Create Y
targets = pd.DataFrame(sin_2d(arr1=searchspace["x1"], arr2=searchspace["x2"]))

n_training_points=10

# Randomly select some points for training
train_idx = np.random.choice(range(len(searchspace)), n_training_points, replace=False)
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
# test_idx = np.random.choice(range(len(searchspace)), n_training_points, replace=False)
# test_x = to_tensor(searchspace.loc[test_idx])

# Get best_f for acquisition function
best_f = train_y.max()

# All available surrogate models
surrogate_models = {
    "Gaussian Process": GaussianProcessModel,
    "Random Forest": RandomForestModel,
    "Natural Gradient Boosting": NGBoostModel,
    "Bayesian Linear Regression": BayesianLinearModel
}

for name, mdl in surrogate_models.items():
    print("Model:", name)

    # Define and train model
    model = mdl(searchspace)
    model.fit(train_x, train_y)

    # Define candidates with q=1; can also expand both t-batch & q-batch artificially
    candidates = to_tensor(searchspace).unsqueeze(1)#.unsqueeze(0).unsqueeze(0).expand((5,2,100,5,2))

    # See mean, covar explicitly (Confirm size)
    # mean, covar = model.posterior(candidates)
    # print(candidates.size())
    # print(mean.size())
    # print(covar.size())

    # Define available aquisition functions
    acquisition_functions = [
        ExpectedImprovement,
        PosteriorMean,
        ProbabilityOfImprovement
    ]

    # Get recommendations based on different acqf 
    for acqf in acquisition_functions:
        acqf = debotorchize(acqf)(model, best_f)

        recommender = MarginalRankingRecommender(acqf)

        idxs = recommender.recommend(searchspace, 1)

        print(idxs)


