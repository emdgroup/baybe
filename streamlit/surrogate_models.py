"""Compare different surrogate model and verify their predictions.

The purpose of this script is to compare different surrogate models and verify that
their predictions are invariant to changes in scale of the underlying target function.

This means that the shown function approximation should always appear visually the same
when the input and output scales are changed.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf_discrete
from funcy import rpartial

import streamlit as st
from baybe.acquisition import debotorchize
from baybe.parameters import NumericalDiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.surrogates import get_available_surrogates

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


def constant(
    x: np.ndarray, x_min: float, x_max: float, amplitude: float, bias: float
) -> np.ndarray:
    """Constant test function."""
    out = np.full(x.shape, fill_value=bias)
    return out


def linear(
    x: np.ndarray, x_min: float, x_max: float, amplitude: float, bias: float
) -> np.ndarray:
    """Linear test function."""
    out = amplitude * np.linspace(0, 1, len(x)) + bias
    return out


def main():
    """Create the streamlit dashboard."""
    # basic settings
    plt.style.use("seaborn-v0_8-paper")  # 'seaborn' is depreciated since matplotlib 3.6

    # show docstring in dashboard
    st.info(__doc__)

    # define all available test functions
    test_functions = {
        "Sine": sin,
        "Constant": constant,
        "Linear": linear,
        "Cubic": cubic,
    }

    # collect all available surrogate models
    surrogate_model_classes = {
        surr.__name__: surr for surr in get_available_surrogates()
    }

    # simulation parameters
    random_seed = int(st.sidebar.number_input("Random seed", value=1337))
    function_name = st.sidebar.selectbox("Test function", list(test_functions.keys()))
    surrogate_name = st.sidebar.selectbox(
        "Surrogate model", list(surrogate_model_classes.keys())
    )
    n_training_points = st.sidebar.slider("Number of training points", 1, 20, 5)
    n_recommendations = st.sidebar.slider("Number of recommendations", 1, 20, 5)
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

    # create the input grid and corresponding target values
    test_x = torch.linspace(
        lower_parameter_limit, upper_parameter_limit, N_PARAMETER_VALUES
    )
    test_y = torch.from_numpy(fun(test_x.numpy()))

    # randomly select the specified number of training data points
    train_idx = np.random.choice(
        range(N_PARAMETER_VALUES), n_training_points, replace=False
    )
    train_x = test_x[train_idx]
    train_y = test_y[train_idx]

    # create the searchspace object
    param = NumericalDiscreteParameter(name="param", values=test_x.numpy().tolist())
    searchspace = SearchSpace.from_product(parameters=[param])

    # create the surrogate model, train it, and get its predictions
    surrogate_model = surrogate_model_classes[surrogate_name]()
    surrogate_model.fit(searchspace, train_x.unsqueeze(-1), train_y.unsqueeze(-1))

    # recommend next experiments
    # TODO: use BayBE recommender and add widgets for recommender selection
    best_f = train_y.max().item()
    acqf = debotorchize(qExpectedImprovement)(surrogate_model, best_f)
    recommendatations = optimize_acqf_discrete(
        acqf, q=n_recommendations, choices=test_x.unsqueeze(-1)
    )[0]

    # create the mean and standard deviation predictions for the entire search space
    mean, covar = surrogate_model.posterior(test_x.unsqueeze(-1))
    mean = mean.detach().numpy()
    std = covar.diag().sqrt().detach().numpy()

    # visualize the test function, training points, model predictions, recommendations
    fig = plt.figure()
    plt.plot(test_x, test_y, color="tab:blue", label="Test function")
    plt.plot(train_x, train_y, "o", color="tab:blue")
    plt.plot(test_x, mean, color="tab:red", label="Surrogate model")
    plt.fill_between(test_x, mean - std, mean + std, alpha=0.2, color="tab:red")
    plt.vlines(
        recommendatations, *plt.gca().get_ylim(), color="k", label="Recommendations"
    )
    plt.legend()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
