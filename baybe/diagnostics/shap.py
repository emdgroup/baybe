"""SHAP utilities."""

import numbers
import warnings

import numpy as np
import pandas as pd

from baybe import Campaign
from baybe._optional.diagnostics import shap
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
        NotImplementedError: If the provided explainer does not support
            the campaign surrogate.
        TypeError: If the provided explainer does not support the campaign surrogate.
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

    if (
        campaign.searchspace.type != "CONTINUOUS"
        and not computational_representation
        and not explainer_class == shap.KernelExplainer
    ):
        raise NotImplementedError(
            "Only KernelExplainer is supported for non-continous searchspaces."
        )

    try:
        shap_explainer = explainer_class(model, data, **kwargs)
    except shap.utils._exceptions.InvalidModelError:
        raise TypeError(
            "The selected explainer class does not support the campaign surrogate."
        )
    return shap_explainer


def explanation(
    campaign: Campaign,
    data: pd.DataFrame | None = None,
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
        ValueError: If the provided explainer does not support experimental
            representation.
        NotImplementedError: If the provided explainer does not support
            the campaign surrogate.
        ValueError: If the provided data does not have the same amount of parameters
            as the campaign.
    """
    is_shap_explainer = not explainer_class.__module__.startswith(
        "shap.explainers.other."
    )

    if not is_shap_explainer and not computational_representation:
        raise ValueError(
            "Experimental representation is not "
            "supported for non-Kernel SHAP explainer."
        )

    try:
        explainer_obj = explainer(
            campaign,
            explainer_class=explainer_class,
            computational_representation=computational_representation,
            **kwargs,
        )
    except NotImplementedError:
        warnings.warn(
            "The provided Explainer does not support experimental representation. "
            "Switching to computational representation. "
            "Otherwise consider using a different explainer (e.g. KernelExplainer)."
        )
        return explanation(
            campaign,
            data=data,
            explainer_class=explainer_class,
            computational_representation=True,
            **kwargs,
        )

    if data is None:
        data = campaign.measurements[[p.name for p in campaign.parameters]].copy()
    elif set(campaign.searchspace.parameter_names) != set(data.columns.values):
        raise ValueError(
            "The provided data does not have the same amount of parameters "
            "as specified for the campaign."
        )
    if computational_representation:
        data = campaign.searchspace.transform(pd.DataFrame(data))

    """Get background data depending on the explainer."""
    bg_data = getattr(explainer_obj, "data", getattr(explainer_obj, "masker", None))
    bg_data = getattr(bg_data, "data", bg_data)

    # Type checking for mypy
    bg_data = bg_data if isinstance(bg_data, pd.DataFrame) else pd.DataFrame(bg_data)
    assert isinstance(data, pd.DataFrame)

    if not bg_data.shape[1] == data.shape[1]:
        raise ValueError(
            "The provided data does not have the same amount of "
            "parameters as the shap explainer background."
        )

    if not is_shap_explainer:
        """Return attributions for non-SHAP explainers."""
        if explainer_class.__module__.endswith("maple"):
            """Additional argument for maple to increase comparability to SHAP."""
            attributions = explainer_obj.attributions(
                np.array(data), multiply_by_input=True
            )[0]
        else:
            attributions = explainer_obj.attributions(np.array(data))[0]
        if computational_representation:
            feature_names = campaign.searchspace.comp_rep_columns
        else:
            feature_names = campaign.searchspace.parameter_names
        explanations = shap.Explanation(
            values=attributions,
            base_values=np.array(data),
            data=np.array(data),
        )
        explanations.feature_names = list(feature_names)
        return explanations

    shap_explanations = explainer_obj(np.array(data))
    if len(shap_explanations.shape) == 2:
        return shap_explanations
    else:
        shap_explanations = explainer_obj(data)[:, :, 0]

    return shap_explanations


def plot_shap_scatter(explanation: shap.Explanation, **kwargs) -> None:
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

    if np.ndim(data) == 1:
        if is_not_numeric_column(data):
            warnings.warn(
                "Cannot plot scatter plot for the provided "
                "explanation as it contains non-numeric values."
            )
        else:
            shap.plots.scatter(explanation, **kwargs)
    else:
        # Type checking for mypy
        assert isinstance(data, np.ndarray)

        number_enum = [i for i, x in enumerate(data[1]) if not isinstance(x, str)]
        if len(number_enum) < len(explanation.feature_names):
            warnings.warn(
                "Cannot plot SHAP scatter plot for all "
                "parameters as some contain non-numeric values."
            )
        shap.plots.scatter(explanation[:, number_enum], **kwargs)
