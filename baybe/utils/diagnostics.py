"""Diagnostics utilities."""

import numbers
import warnings

import numpy as np
import pandas as pd
import shap

from baybe import Campaign
from baybe.utils.dataframe import to_tensor


def explainer(
    campaign: Campaign,
    explainer_class: shap.Explainer = shap.KernelExplainer,
    computational_representation: bool = False,
    **kwargs,
) -> shap.Explainer:
    """Create an explainer for the provided campaign.

    Args:
        campaign: The campaign to be explained.
        explainer_class: The explainer to be used. Default is shap.KernelExplainer.
        computational_representation: Whether to compute the Shapley values
            in  computational or experimental searchspace.
            Default is False.
        **kwargs: Additional keyword arguments to be passed to the explainer.

    Returns:
        The explainer for the provided campaign.

    Raises:
        ValueError: If no measurements have been provided yet.
    """
    if campaign.measurements.empty:
        raise ValueError("No measurements have been provided yet.")

    data = campaign.measurements[[p.name for p in campaign.parameters]].copy()

    if computational_representation:
        data = campaign.searchspace.transform(data)

        def model(x):
            tensor = to_tensor(x)
            output = campaign.get_surrogate()._posterior_comp(tensor).mean

            return output.detach().numpy()
    else:

        def model(x):
            df = pd.DataFrame(x, columns=data.columns)
            output = campaign.get_surrogate().posterior(df).mean

            return output.detach().numpy()

    shap_explainer = explainer_class(model, data, **kwargs)
    return shap_explainer


def explanation(
    campaign: Campaign,
    data: np.ndarray = None,
    explainer_class: shap.Explainer = shap.KernelExplainer,
    computational_representation: bool = False,
    **kwargs,
) -> shap.Explanation:
    """Compute the Shapley values for the provided campaign and data.

    Args:
        campaign: The campaign to be explained.
        data: The data to be explained.
            Default is None which uses the campaign's measurements.
        explainer_class: The explainer to be used.
            Default is shap.KernelExplainer.
        computational_representation: Whether to compute the Shapley values
            in computational or experimental searchspace.
            Default is False.
        **kwargs: Additional keyword arguments to be passed to the explainer.

    Returns:
        The Shapley values for the provided campaign.

    Raises:
        ValueError: If the provided data does not have the same amount of parameters
            as previously provided to the explainer.
    """
    is_shap_explainer = not explainer_class.__module__.startswith(
        "shap.explainers.other."
    )

    if not is_shap_explainer and not computational_representation:
        raise ValueError(
            "Experimental representation is not "
            "supported for non-Kernel SHAP explainer."
        )

    explainer_obj = explainer(
        campaign,
        explainer_class=explainer_class,
        computational_representation=computational_representation,
        **kwargs,
    )

    if data is None:
        if isinstance(explainer_obj.data, np.ndarray):
            data = explainer_obj.data
        else:
            data = explainer_obj.data.data
    elif computational_representation:
        data = campaign.searchspace.transform(data)

    if not is_shap_explainer:
        """Return attributions for non-SHAP explainers."""
        if explainer_class.__module__.endswith("maple"):
            """Aditional argument for maple to increase comparability to SHAP."""
            attributions = explainer_obj.attributions(data, multiply_by_input=True)[0]
        else:
            attributions = explainer_obj.attributions(data)[0]
        if computational_representation:
            feature_names = campaign.searchspace.comp_rep_columns
        else:
            feature_names = campaign.searchspace.parameter_names
        explanations = shap.Explanation(
            values=attributions,
            base_values=data,
            data=data,
        )
        explanations.feature_names = list(feature_names)
        return explanations

    if data.shape[1] != explainer_obj.data.data.shape[1]:
        raise ValueError(
            "The provided data does not have the same amount "
            "of parameters as the shap explainer background."
        )
    else:
        shap_explanations = explainer_obj(data)[:, :, 0]

    return shap_explanations


def shap_plot_beeswarm(explanation: shap.Explanation, **kwargs) -> None:
    """Plot the Shapley values using a beeswarm plot."""
    shap.plots.beeswarm(explanation, **kwargs)


def shap_plot_waterfall(explanation: shap.Explanation, **kwargs) -> None:
    """Plot the Shapley values using a waterfall plot."""
    shap.plots.waterfall(explanation, **kwargs)


def shap_plot_bar(explanation: shap.Explanation, **kwargs) -> None:
    """Plot the Shapley values using a bar plot."""
    shap.plots.bar(explanation, **kwargs)


def shap_plot_scatter(explanation: shap.Explanation | memoryview, **kwargs) -> None:
    """Plot the Shapley values using a scatter plot while leaving out string values.

    Args:
        explanation: The Shapley values to be plotted.
        **kwargs: Additional keyword arguments to be passed to the scatter plot.

    Raises:
        ValueError: If the provided explanation object does not match the
            required types.
    """
    if isinstance(explanation, memoryview):
        data = explanation.obj
    elif isinstance(explanation, shap.Explanation):
        data = explanation.data.data.obj
    else:
        raise ValueError("The provided explanation argument is not of a valid type.")

    def is_not_numeric_column(col):
        return np.array([not isinstance(v, numbers.Number) for v in col]).any()

    if data.ndim == 1:
        if is_not_numeric_column(data):
            warnings.warn(
                "Cannot plot scatter plot for the provided "
                "explanation as it contains non-numeric values."
            )
        else:
            shap.plots.scatter(explanation, **kwargs)
    else:
        number_enum = [i for i, x in enumerate(data[1]) if not isinstance(x, str)]
        if len(number_enum) < len(explanation.feature_names):
            warnings.warn(
                "Cannot plot SHAP scatter plot for all "
                "parameters as some contain non-numeric values."
            )
        shap.plots.scatter(explanation[:, number_enum], **kwargs)
