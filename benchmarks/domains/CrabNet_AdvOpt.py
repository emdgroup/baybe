"""Crabnet hyperparameter function with 20 continuous and 3 categorical input. 
https://huggingface.co/spaces/AccelerationConsortium/crabnet-hyperparameter"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas import DataFrame

from baybe.campaign import Campaign
from baybe.parameters import NumericalContinuousParameter, CategoricalParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget, TargetMode
from baybe.constraints import ContinuousLinearInequalityConstraint
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.simulation import simulate_scenarios
from baybe.utils.random import set_random_seed
from benchmarks.definition import (
    Benchmark, 
    ConvergenceExperimentSettings,
)
from gradio_client import Client

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D

# Initialize the client
client = Client("AccelerationConsortium/crabnet-hyperparameter")


# Define the function to evaluate
def _lookup(c1, c2, c3, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, 
            x11, x12, x13, x14, x15, x16, x17, x18, x19, x20):
    """Optimization function with 20 continuous variables and 3 categorical parameters."""
    # Assertion checks for the continuous parameters
    try:
        for xi, name in zip(
            [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, 
             x11, x12, x13, x14, x15, x16, x17, x18, x19, x20],
            ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10",
             "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20"]
        ):
            assert 0.0 <= xi <= 1.0, f"{name} must be in the range [0.0, 1.0]."
    except AssertionError as e:
        raise ValueError(f"Assertion failed for continuous variables: {e}")

    # Assertion checks for the categorical parameters
    try:
        assert c1 in ["c1_0", "c1_1"], f"Invalid value for c1: {c1}."
        assert c2 in ["c2_0", "c2_1"], f"Invalid value for c2: {c2}."
        assert c3 in ["c3_0", "c3_1", "c3_2"], f"Invalid value for c3: {c3}."
    except AssertionError as e:
        raise ValueError(f"Assertion failed for categorical variables: {e}")

    # Execute the prediction via the Gradio client
    result = client.predict(
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, 
        x11, x12, x13, x14, x15, x16, x17, x18, x19, x20,  # Continuous variables
        c1, c2, c3,  # Categorical variables
        0.5,  # Fidelity
        api_name="/predict",
    )
    return result['data'][0][0]  # Return y1 value



def advopt(settings: ConvergenceExperimentSettings) -> DataFrame: 
    """Crabnet hyperparameter function with 20 continuous and 3 categorical input."""
    # Define the continuous and categorical parameters
    parameters = [
        NumericalContinuousParameter(name="x1", bounds=(0.0, 1.0)), 
        NumericalContinuousParameter(name="x2", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x3", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x4", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x5", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x6", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x7", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x8", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x9", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x10", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x11", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x12", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x13", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x14", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x15", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x16", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x17", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x18", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x19", bounds=(0.0, 1.0)),
        NumericalContinuousParameter(name="x20", bounds=(0.0, 1.0)),

        CategoricalParameter(name='c1', values=['c1_0', 'c1_1'], encoding="OHE"),
        CategoricalParameter(name='c2', values=['c2_0', 'c2_1'], encoding="OHE"),
        CategoricalParameter(name='c3', values=['c3_0', 'c3_1', 'c3_2'], encoding="OHE"),
    ]
    # define constraints
    constraints = [
        ContinuousLinearInequalityConstraint(parameters=["x19", "x20"], coefficients=[-1.0, 1.0], rhs=0.0),
        ContinuousLinearInequalityConstraint(parameters=["x6", "x15"], coefficients=[-1.0, -1.0], rhs=-1.0), 
    ]

    # Define the objective and search space
    objective = NumericalTarget(name="target", mode=TargetMode.MIN).to_objective()
    search_space = SearchSpace.from_product(parameters=parameters, constraints=constraints)

    # Define the scenarios
    scenarios: dict[str, Campaign] = {
        "Random Recommender": Campaign(
            searchspace=search_space,
            recommender=RandomRecommender(),
            objective=objective,
        ),
        "Default Recommender": Campaign(
            searchspace=search_space,
            objective=objective,
        ),
    }

    # Simulate the scenarios
    return simulate_scenarios(
        scenarios,
        _lookup,
        batch_size=settings.batch_size,
        n_doe_iterations=settings.n_doe_iterations,
        n_mc_iterations=settings.n_mc_iterations,
        impute_mode="error",
    )

#%%
benchmark_config = ConvergenceExperimentSettings(
    batch_size=1,
    n_doe_iterations=10,
    n_mc_iterations=5,
)

# Define the benchmark
crabnet_advopt_benchmark = Benchmark(
    function=advopt,
    best_possible_result=None,
    settings=benchmark_config,
    optimal_function_inputs=None,
)


# ---------------- to do
# reasonable visulization of the benchmark (eg. data frame)
# help understand the benchmark

if __name__ == "__main__":
    #  Describe the benchmark task
    print("CrabNet optimization is a minimization task by tuning 20 continuous and 3 categorical hyperparameters.")
    print("The continuous hyperparameters include number of attention layers, learning rate, step size of epochs, etc.")
    print("Details can be found in Table 1 of Baird, S. G.; Liu, M.; Sparks, T. D. High-Dimensional Bayesian Optimization of 23 Hyperparameters over 100 Iterations for an Attention-Based Network to Predict Materials Property: A Case Study on CrabNet Using Ax Platform and SAASBO. Computational Materials Science 2022, 211, 111505.")
    print("The categorical hyperparameters include boolean values for bias residual network, loss function, and elemental feature vector.")
    print("The continuous hyperparameters are normalized to range [0.0, 1.0], while the categorical hyperparameters are one-hot encoded.")
    print("")
    print("The objective is to minimize y1, RMSE, of the CrabNet hyperparameter function. If y1 is greater than 0.2, the result is coonsider bad.")

