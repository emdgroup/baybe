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

from baybe.surrogate import SurrogateModel
from baybe.utils import to_tensor
from funcy import rpartial

# define constants
N_PARAMETER_VALUES = 1000


def cubic(
    x: np.ndarray, x_min: float, x_max: float, amplitude: float, bias: float
) -> np.ndarray:
    """Cubic test function."""
    out = amplitude * np.power((x - x_min) / (x_max - x_min), 3) + bias
    return out


def sin(
    x: np.ndarray, x_min: float, x_max: float, amplitude: float, bias: float
) -> np.ndarray:
    """Sinusoid test function."""
    out = amplitude * np.sin((x - x_min) / (x_max - x_min) * 2 * np.pi) + bias
    return out


def main():
    """Creates the streamlit dashboard."""

    # fix issue with streamlit and pydantic
    # https://github.com/streamlit/streamlit/issues/3218
    pydantic.class_validators._FUNCS.clear()  # pylint: disable=protected-access

    # basic settings
    plt.style.use("seaborn")

    # show docstring in dashboard
    st.info(__doc__)

    # define all available test functions
    test_functions = {"Sine": sin, "Cubic": cubic}

    # collect all available surrogate models
    surrogate_models = SurrogateModel.SUBCLASSES

    # simulation parameters
    random_seed = int(st.sidebar.number_input("Random seed", value=1337))
    function_name = st.sidebar.selectbox("Test function", list(test_functions.keys()))
    surrogate_name = st.sidebar.selectbox(
        "Surrogate model", list(surrogate_models.keys())
    )
    n_training_points = st.sidebar.slider("Number of training points", 1, 20, 5)
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
    fun = rpartial(
        test_functions[function_name],
        lower_parameter_limit,
        upper_parameter_limit,
        function_amplitude,
        function_bias,
    )
    surrogate_model_cls = surrogate_models[surrogate_name]

    # define the search space and corresponding target values
    # TODO [searchspace]: The pandas dataframe here is only needed because the current
    #  GP implementation looks for MORDRED features in the column names.
    #   --> Remove the dataframe once the `Searchspace` class has been introduced
    searchspace = pd.DataFrame(
        {
            "param": np.linspace(
                lower_parameter_limit, upper_parameter_limit, N_PARAMETER_VALUES
            )
        }
    )
    targets = pd.DataFrame(fun(searchspace))

    # randomly select the specified number of training data points
    train_idx = np.random.choice(
        range(len(searchspace)), n_training_points, replace=False
    )
    train_x = to_tensor(searchspace.loc[train_idx])
    train_y = to_tensor(targets.loc[train_idx])

    # create the surrogate model, train it, and get its predictions
    surrogate_model = surrogate_model_cls(searchspace)
    surrogate_model.fit(train_x, train_y)

    # create the mean and standard deviation predictions for the entire search space
    test_x = to_tensor(searchspace)
    mean, covar = surrogate_model.posterior(test_x)
    mean = mean.detach().numpy()
    std = covar.diag().sqrt().detach().numpy()

    # visualize the test function, training points, and model predictions
    fig = plt.figure()
    plt.plot(searchspace, targets, color="tab:blue", label="Test function")
    plt.plot(train_x, train_y, "o", color="tab:blue")
    plt.plot(searchspace, mean, color="tab:red", label="Surrogate model")
    plt.fill_between(
        searchspace.values[:, 0], mean - std, mean + std, alpha=0.2, color="tab:red"
    )
    plt.legend()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
