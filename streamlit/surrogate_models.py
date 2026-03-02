"""# Surrogate Models

The purpose of this script is to compare different surrogate models and verify that
their predictions are invariant to changes in location/scale of the underlying target
function.

This means that the displayed function approximation should always look the same when
the input and output locations/scales are changed.
"""  # noqa: D415

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from baybe import active_settings
from baybe.acquisition import qLogExpectedImprovement
from baybe.acquisition.base import AcquisitionFunction
from baybe.exceptions import IncompatibleSurrogateError
from baybe.parameters import NumericalDiscreteParameter, TaskParameter
from baybe.recommenders import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.surrogates import CustomONNXSurrogate, GaussianProcessSurrogate
from baybe.surrogates.base import Surrogate
from baybe.surrogates.gaussian_process.presets import GaussianProcessPreset
from baybe.targets import NumericalTarget
from baybe.utils.basic import get_subclasses

# Number of values used for the input parameter
N_PARAMETER_VALUES = 200


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
    out = amplitude * x + bias
    return out


def main():
    """Create the streamlit dashboard."""
    # Show module docstring in dashboard
    st.info(__doc__)

    # Collect all available test functions
    test_functions = {
        "Sine": sin,
        "Constant": constant,
        "Linear": linear,
        "Cubic": cubic,
    }

    # Collect all available surrogate models
    surrogate_model_classes = {
        cls.__name__: cls
        for cls in get_subclasses(Surrogate)
        if not issubclass(cls, CustomONNXSurrogate)
    }
    surrogate_model_names = list(surrogate_model_classes.keys())

    # Collect all available acquisition functions
    acquisition_function_classes = {
        cls.__name__: cls for cls in get_subclasses(AcquisitionFunction)
    }
    acquisition_function_names = list(acquisition_function_classes.keys())

    # >>>>> Sidebar options >>>>>
    # Domain
    st.sidebar.markdown("# Domain")
    st_enable_multitask = st.sidebar.toggle("Multi-task")
    st_n_tasks = 2 if st_enable_multitask else 1
    st_random_seed = int(st.sidebar.number_input("Random seed", value=1337))
    st_function_name = st.sidebar.selectbox(
        "Test function", list(test_functions.keys())
    )
    st_minimize = st.sidebar.checkbox("Minimize")

    # Training data
    st.sidebar.markdown("---")
    st.sidebar.markdown("# Training Data")
    st_n_training_points = st.sidebar.slider(
        "Number of training points",
        0 if st_enable_multitask else 1,
        20,
        0 if st_enable_multitask else 5,
    )
    if st_enable_multitask:
        st_n_training_points_other = st.sidebar.slider(
            "Number of off-task training points", 0, 20, 5
        )
        st_offtask_scale = st.sidebar.slider("Off-task scale factor", -20.0, 20.0, 1.0)
        st_offtask_offset_factor = st.sidebar.slider(
            "Off-task offset", -100.0, 100.0, 0.0
        )

    # Model
    st.sidebar.markdown("---")
    st.sidebar.markdown("# Model")
    st_surrogate_name = st.sidebar.selectbox(
        "Surrogate model",
        surrogate_model_names,
        surrogate_model_names.index(GaussianProcessSurrogate.__name__),
    )

    st_gp_preset = None
    st_transfer_learning = False
    if st_surrogate_name == GaussianProcessSurrogate.__name__:
        preset_names = [preset.value for preset in GaussianProcessPreset]
        st_gp_preset = st.sidebar.selectbox(
            "GP Preset",
            preset_names,
            index=preset_names.index(GaussianProcessPreset.BAYBE.value),
        )
        if st_enable_multitask:
            st_transfer_learning = st.sidebar.checkbox("Transfer learning", value=True)
    st_acqf_name = st.sidebar.selectbox(
        "Acquisition function",
        acquisition_function_names,
        acquisition_function_names.index(qLogExpectedImprovement.__name__),
    )
    st_n_recommendations = st.sidebar.slider("Number of recommendations", 1, 20, 5)

    # Surrogate consistency validation
    st.sidebar.markdown("---")
    st.sidebar.markdown("# Validation")
    st.sidebar.markdown(
        """
        When scaling is implemented correctly, the plot should remain static (except for
        the axis labels) when changing the following parameters:
        """
    )
    st_upper_parameter_limit = st.sidebar.slider(
        "Upper parameter limit", 0.0, 100.0, 1.0
    )
    st_lower_parameter_limit = st.sidebar.slider(
        "Lower parameter limit", -100.0, 0.0, 0.0
    )
    st_function_amplitude = st.sidebar.slider("Function amplitude", 1.0, 100.0, 1.0)
    st_function_bias = st.sidebar.slider("Function bias", -100.0, 100.0, 0.0)
    # <<<<< Sidebar options <<<<<

    # Derived settings
    st_use_separate_gps = st_n_tasks > 1 and not st_transfer_learning

    # Set the chosen random seed
    active_settings.random_seed = st_random_seed

    # Construct the specific test function
    fun = partial(
        test_functions[st_function_name],
        x_min=st_lower_parameter_limit,
        x_max=st_upper_parameter_limit,
        amplitude=st_function_amplitude,
        bias=st_function_bias,
    )

    # Generate task-specific transforms (scale and offset for each task)
    task_names = ["on-task", "off-task"] if st_n_tasks > 1 else ["on-task"]
    task_transforms = {}
    for task_idx in range(st_n_tasks):
        task_name = task_names[task_idx]
        if task_idx == 0:
            # On-task: use original function values
            task_transforms[task_name] = {"scale": 1.0, "offset": 0.0}
        else:
            # Off-task: use user-specified scale and offset
            scale = st_offtask_scale
            offset = st_offtask_offset_factor * st_function_amplitude
            task_transforms[task_name] = {"scale": scale, "offset": offset}

    # Create training data
    measurements_list = []
    for task_idx in range(st_n_tasks):
        task_name = task_names[task_idx]
        transform = task_transforms[task_name]
        n_points = st_n_training_points if task_idx == 0 else st_n_training_points_other
        train_x = np.random.uniform(
            st_lower_parameter_limit, st_upper_parameter_limit, n_points
        )
        task_measurements = pd.DataFrame(
            {
                "x": train_x,
                "task": task_name,
                "y": fun(train_x) * transform["scale"] + transform["offset"],
            }
        )
        measurements_list.append(task_measurements)
    measurements = pd.concat(measurements_list, ignore_index=True)

    # Create the plotting grid and corresponding target values
    test_x = np.linspace(
        st_lower_parameter_limit, st_upper_parameter_limit, N_PARAMETER_VALUES
    )
    candidates_list = []
    test_ys = {}
    for task_idx in range(st_n_tasks):
        task_name = task_names[task_idx]
        transform = task_transforms[task_name]
        test_ys[task_name] = fun(test_x) * transform["scale"] + transform["offset"]
        task_candidates = pd.DataFrame(
            {"x": test_x, "task": task_name, "y": test_ys[task_name]}
        )
        candidates_list.append(task_candidates)
    candidates = pd.concat(candidates_list, ignore_index=True)

    # Create the searchspace and objective
    parameters = [
        NumericalDiscreteParameter(
            name="x",
            values=np.linspace(
                st_lower_parameter_limit, st_upper_parameter_limit, N_PARAMETER_VALUES
            ),
        )
    ]
    if st_transfer_learning:
        parameters.append(
            TaskParameter(
                name="task",
                values=task_names,
                active_values=["on-task"],
            )
        )
    searchspace = SearchSpace.from_product(parameters=parameters)
    objective = NumericalTarget(name="y", minimize=st_minimize).to_objective()

    # Create the acquisition function
    acqf_cls = acquisition_function_classes[st_acqf_name]
    try:
        acqf = acqf_cls(maximize=not st_minimize)
    except TypeError:
        acqf = acqf_cls()

    def make_surrogate():
        if st_surrogate_name == GaussianProcessSurrogate.__name__:
            assert st_gp_preset is not None
            return GaussianProcessSurrogate.from_preset(
                preset=GaussianProcessPreset[st_gp_preset]
            )
        return surrogate_model_classes[st_surrogate_name]()

    if st_use_separate_gps:
        # One independent GP per task, each trained without the task column
        stats_by_task = {}
        for task_name in task_names:
            task_meas = measurements[measurements["task"] == task_name][
                ["x", "y"]
            ].reset_index(drop=True)
            task_recommender = BotorchRecommender(
                surrogate_model=make_surrogate(),
                acquisition_function=acqf,
            )
            if task_name == "on-task":
                try:
                    recommendations = task_recommender.recommend(
                        st_n_recommendations, searchspace, objective, task_meas
                    )
                except IncompatibleSurrogateError:
                    st.error(
                        f"You requested {st_n_recommendations} recommendations but "
                        f"the selected surrogate class does not support recommending "
                        f"more than one candidate at a time."
                    )
                    st.stop()
            task_surrogate = task_recommender.get_surrogate(
                searchspace, objective, task_meas
            )
            stats_by_task[task_name] = task_surrogate.posterior_stats(
                pd.DataFrame({"x": test_x})
            )
    else:
        # Single recommender (single task, or multi-task with transfer learning)
        recommender = BotorchRecommender(
            surrogate_model=make_surrogate(),
            acquisition_function=acqf,
        )
        try:
            recommendations = recommender.recommend(
                st_n_recommendations, searchspace, objective, measurements
            )
        except IncompatibleSurrogateError:
            st.error(
                f"You requested {st_n_recommendations} recommendations but the "
                f"selected surrogate class does not support recommending more than "
                f"one candidate at a time."
            )
            st.stop()
        surrogate = recommender.get_surrogate(searchspace, objective, measurements)
        stats = surrogate.posterior_stats(candidates)

    # Visualize the test function, training points, model predictions, recommendations
    if st_n_tasks > 1:
        cols = st.columns(st_n_tasks)

    for task_idx in range(st_n_tasks):
        task_name = task_names[task_idx]
        task_mask = candidates["task"] == task_name if st_n_tasks > 1 else slice(None)

        if st_use_separate_gps:
            task_stats = stats_by_task[task_name]
            mean = task_stats["y_mean"].values
            std = task_stats["y_std"].values
        elif st_n_tasks > 1:
            mean = stats["y_mean"][task_mask].values
            std = stats["y_std"][task_mask].values
        else:
            mean = stats["y_mean"].values
            std = stats["y_std"].values

        test_y = test_ys[task_name]
        train_mask = (
            measurements["task"] == task_name if st_n_tasks > 1 else slice(None)
        )
        train_y = measurements[train_mask]["y"].values
        task_train_x = measurements[train_mask]["x"].values

        fig = plt.figure()
        plt.plot(test_x, test_y, color="tab:blue", label="Test function")
        plt.plot(task_train_x, train_y, "o", color="tab:blue")
        plt.plot(test_x, mean, color="tab:red", label="Surrogate model")
        plt.fill_between(test_x, mean - std, mean + std, alpha=0.2, color="tab:red")
        if task_name == "on-task":
            plt.vlines(
                recommendations["x"] if st_n_tasks > 1 else recommendations,
                *plt.gca().get_ylim(),
                color="k",
                label="Recommendations",
            )
        plt.legend()
        if st_n_tasks > 1:
            plt.title(task_name.capitalize())
            with cols[task_idx]:
                st.pyplot(fig)
        else:
            st.pyplot(fig)


if __name__ == "__main__":
    main()
