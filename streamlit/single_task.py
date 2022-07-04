"""
Streamlit dashboard to compare DoE strategies on a single task.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from baybe.strategy import GaussianProcessStrategy
from botorch.models.transforms import Standardize
from botorch.test_functions import Hartmann
from tqdm import trange

# create logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# set plotting style
plt.style.use("seaborn")

# streamlit sidebar
st.sidebar.markdown("# Progress")
st.sidebar.markdown("Monte Carlo runs")
progress_bar_monte_carlo = st.sidebar.progress(1.0)
st.sidebar.markdown("Experiment evaluation")
progress_bar_experiments = st.sidebar.progress(1.0)
st.sidebar.markdown("---")
st.sidebar.markdown("# Settings")
objective_name = st.sidebar.selectbox("Objective function", ["Quadratic", "Hartmann"])
min_n_dims = 1 if objective_name == "Quadratic" else 3
n_dims = int(
    st.sidebar.number_input(
        "Number of dimensions", min_value=min_n_dims, value=min_n_dims
    )
)
n_points_per_dim = int(
    st.sidebar.number_input("Points per dimension", min_value=2, value=5)
)
n_monte_carlo = int(st.sidebar.number_input("Monte Carlo runs", min_value=1, value=1))
if objective_name == "Quadratic":
    objective_function = lambda x: x.square().sum(dim=-1)  # noqa: E731
else:
    objective_function = Hartmann(dim=n_dims)

# create design points
grid = torch.linspace(-1, 1, n_points_per_dim)
X = torch.meshgrid(*[grid] * n_dims)
points = torch.vstack([X_i.ravel() for X_i in X]).T

# standardize target values
# TODO: to be removed once the computational data representations are used
target_transform = Standardize(1)
targets = target_transform(objective_function(points).unsqueeze(1))[0]

# construct input and target dataframes
features = pd.DataFrame(points, columns=[f"Param_{i}" for i in range(n_dims)])
targets = pd.DataFrame(targets, columns=["Target"])

# list to store the experimentation sequences
sequences = []

# run several Monte Carlo experiments
for mc in trange(n_monte_carlo, desc="Monte Carlo runs"):

    progress_bar_monte_carlo.progress(mc / n_monte_carlo)
    sequence = []

    # create a fresh model instance
    model = GaussianProcessStrategy()

    # evaluate all experiments
    for i_experiment in range(len(points)):

        progress_bar_experiments.progress(i_experiment / len(points))

        # extract the current training data and candidate experiments
        train_x = features.loc[sequence]
        train_y = targets.loc[sequence]
        candidates = features.drop(sequence)

        # train the model
        model.set_training_data(train_x, train_y)

        # get the next experiment recommendation from the model
        new_loc = model.recommend(candidates)
        sequence.append(new_loc)

    progress_bar_experiments.progress(1.0)
    sequences.append(sequence)

progress_bar_monte_carlo.progress(1.0)


def plot_stats(arr: np.ndarray, color: str, label: str):
    """Plots the mean and quantiles of the running optimum from several experiments."""
    cumulative_min = np.minimum.accumulate(arr, axis=-1)
    q = np.quantile(cumulative_min, [0.05, 0.95], axis=0)
    plt.fill_between(
        np.arange(arr.shape[-1]), *q, alpha=0.1, color=color, edgecolor=None
    )
    plt.plot(cumulative_min.mean(axis=0), color=color, label=label)


# sort the targets according to the experimentation sequences
random = np.asarray([np.random.permutation(targets.values[:, 0]) for _ in range(1000)])
doe = np.asarray([targets.loc[sequence].iloc[:, 0] for sequence in sequences])

# plot the results
fig = plt.figure()
plot_stats(random, "tab:blue", "Random")
plot_stats(doe, "tab:red", "Gaussian process")
plt.legend()
plt.xlabel("Number of conducted experiments")
plt.ylabel("Running optimum")
st.pyplot(fig)
