"""This file contains temporary tests for the target issue."""

import itertools

import numpy as np
import pandas as pd

from baybe import Campaign
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.recommenders import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget

DIMENSION = 4
BOUNDS = (-1, 1)

parameters = [
    NumericalContinuousParameter(name=f"x_{k}", bounds=BOUNDS) for k in range(DIMENSION)
]
searchspace = SearchSpace.from_product(parameters=parameters)


def sum_of_squares(df: pd.DataFrame, /) -> pd.DataFrame:
    """Sum of squares."""
    return (df[[p.name for p in parameters]] ** 2).sum(axis=1).to_frame("Target")


def sum(df: pd.DataFrame, /) -> pd.DataFrame:
    """Sum."""
    return (df[[p.name for p in parameters]]).sum(axis=1).to_frame("Target")


def max(df: pd.DataFrame, /) -> pd.DataFrame:
    """Max."""
    return (df[[p.name for p in parameters]]).max(axis=1).to_frame("Target")


TARGET_BOUNDS_DICT = {
    "sum": (-DIMENSION, DIMENSION),
    "sum_of_squares": (0, DIMENSION),
    "max": (-1, 1),
}

df_list = [pd.DataFrame({f"x_{i}": np.linspace(-1, 1, 6)}) for i in range(DIMENSION)]

# Create a list of DataFrames with their single columns converted to lists
data_lists = [df.iloc[:, 0].tolist() for df in df_list]

# Generate the Cartesian product
measurements = pd.DataFrame(
    list(itertools.product(*data_lists)),
    columns=[df.columns[0] for df in df_list],
)

for blackbox, mode in itertools.product([sum_of_squares, sum, max], ["MAX", "MIN"]):
    print(f"Using black-box function {blackbox.__name__} and mode {mode}")
    TARGET_BOUNDS = TARGET_BOUNDS_DICT[blackbox.__name__]

    target_with_bounds = NumericalTarget(
        name="Target", mode=mode, bounds=TARGET_BOUNDS, transformation="LINEAR"
    )
    target_without_bounds = NumericalTarget(name="Target", mode=mode)

    bounds_campaign = Campaign(
        searchspace=searchspace,
        objective=target_with_bounds.to_objective(),
    )

    no_bounds_campaign = Campaign(
        searchspace=searchspace,
        objective=target_without_bounds.to_objective(),
    )

    measurements_w_target = pd.concat([measurements, blackbox(measurements)], axis=1)

    bounds_campaign.add_measurements(measurements_w_target)
    no_bounds_campaign.add_measurements(measurements_w_target)
    print(f"Using target bounds of {TARGET_BOUNDS}")
    bounds_rec = bounds_campaign.recommend(1)
    print(pd.concat([bounds_rec, blackbox(bounds_rec)], axis=1))

    print("Using no target bounds")
    no_bounds_rec = no_bounds_campaign.recommend(1)
    print(pd.concat([no_bounds_rec, blackbox(no_bounds_rec)], axis=1))
    print("")


searchspace = NumericalContinuousParameter("p", [0, 1]).to_searchspace()
objective = NumericalTarget("t", "MIN", (0, 1)).to_objective()
recommender = BotorchRecommender()
measurements = pd.DataFrame({"p": np.linspace(0, 1, 100), "t": np.linspace(0, 1, 100)})
rec = recommender.recommend(1, searchspace, objective, measurements)
print(rec)
