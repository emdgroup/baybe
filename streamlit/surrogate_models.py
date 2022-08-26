"""
The purpose of this script is to compare different surrogate models and verify that
their predictions are invariant to changes in scale of the underlying target function.

This means that the shown function approximation should always appear visually the same
when the input and output scales are changed.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydantic
import streamlit as st
import torch

from baybe.surrogate import GaussianProcessModel
from baybe.utils import to_tensor

# fix issue with streamlit and pydantic
# https://github.com/streamlit/streamlit/issues/3218
pydantic.class_validators._FUNCS.clear()  # pylint: disable=protected-access

# basic settings
plt.style.use("seaborn")

# show docstring in dashboard
st.info(__doc__)


def sin(arr: np.ndarray) -> np.ndarray:
    """Sinusoid test function."""
    out = (
        function_amplitude
        * np.sin(
            (arr - lower_parameter_limit)
            / (upper_parameter_limit - lower_parameter_limit)
            * 2
            * np.pi
        )
        + function_bias
    )
    return out


# define all available test functions
test_functions = {
    "Sine": sin,
}

# define all available surrogate models
surrogate_models = {
    "Gaussian Process": GaussianProcessModel,
}

# simulation parameters
N_PARAMETER_VALUES = 1000
random_seed = int(st.sidebar.number_input("Random seed", value=1337))
function_name = st.sidebar.selectbox("Test function", list(test_functions.keys()))
surrogate_name = st.sidebar.selectbox("Surrogate model", list(surrogate_models.keys()))
n_training_points = st.sidebar.slider("Number of training points", 2, 20, 5)
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    The plot should remain static (except for the axis labels) when changing the
    following parameters.
    """
)
upper_parameter_limit = st.sidebar.slider("Upper parameter limit", 0.0, 100.0, 1.0)
lower_parameter_limit = st.sidebar.slider("Lower parameter limit", -100.0, 0.0, 0.0)
function_amplitude = st.sidebar.slider("Function amplitude", 1.0, 100.0, 1.0)
function_bias = st.sidebar.slider("Function bias", -100.0, 100.0, 0.0)

# fix the chosen random seed
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# select the test function and the surrogate model class
fun = test_functions[function_name]
surrogate_model_cls = surrogate_models[surrogate_name]

# define the search space and corresponding target values
searchspace = pd.DataFrame(
    np.linspace(lower_parameter_limit, upper_parameter_limit, N_PARAMETER_VALUES)
)
targets = pd.DataFrame(fun(searchspace))

# randomly select the specified number of training data points
train_idx = np.random.choice(range(len(searchspace)), n_training_points, replace=False)
train_x = searchspace.loc[train_idx]
train_y = targets.loc[train_idx]

# create the surrogate model, train it, and get its predictions
surrogate_model = surrogate_model_cls(searchspace)
surrogate_model.fit(train_x, train_y)
mvn = surrogate_model.model.posterior(to_tensor(searchspace))

# visualize the test function, training points, and model predictions
fig = plt.figure()
plt.plot(searchspace, targets, color="tab:blue", label="Test function")
plt.plot(train_x, train_y, "o", color="tab:blue")
mean = mvn.mean.detach().numpy()[:, 0]
std = mvn.variance.sqrt().detach().numpy()[:, 0]
plt.plot(searchspace, mean, color="tab:red", label="Surrogate model")
plt.fill_between(
    searchspace.values[:, 0], mean - std, mean + std, alpha=0.2, color="tab:red"
)
plt.legend()
st.pyplot(fig)
