"""Diagnostics utilities."""

import pandas as pd
import shap
import torch

from baybe import Campaign
from baybe.utils.dataframe import to_tensor


def shapley_values(
    campaign: Campaign,
    explainer: callable = shap.KernelExplainer,
    computational_representation: bool = False,
) -> shap.Explanation:
    """Compute the Shapley values for the provided campaign and data.

    Args:
        campaign: The campaign to be explained.
        explainer: The explainer to be used. Default is shap.KernelExplainer.
        computational_representation: Whether to compute the Shapley values
            in  computational or experimental searchspace.
            Default is False.

    Returns:
        The Shapley values for the provided campaign and data.

    Raises:
        ValueError: If no measurements have been provided yet.
    """
    if campaign.measurements.empty:
        raise ValueError("No measurements have been provided yet.")

    data = campaign.measurements[[p.name for p in campaign.parameters]]

    if computational_representation:
        data = campaign.searchspace.transform(data)

        def model(x):
            df = pd.DataFrame(x, columns=data.columns)

            tensor = to_tensor(df)

            output = campaign.get_surrogate()._posterior_comp(tensor).mean

            if isinstance(output, torch.Tensor):
                return output.detach().numpy()

            return output
    else:

        def model(x):
            df = pd.DataFrame(x, columns=data.columns)

            output = campaign.get_surrogate().posterior(df).mean

            if isinstance(output, torch.Tensor):
                return output.detach().numpy()

            return output

    explain = explainer(model, data)
    shap_values = explain(data)
    return shap_values[:, :, 0]


def shapley_plot_beeswarm(explaination: shap.Explainer) -> None:
    """Plot the Shapley values using a beeswarm plot."""
    shap.plots.beeswarm(explaination)
