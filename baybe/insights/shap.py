"""SHAP insights."""

from __future__ import annotations

import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from attrs import define, field
from attrs.validators import instance_of

from baybe import Campaign
from baybe._optional.insights import shap
from baybe.objectives.base import Objective
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.searchspace import SearchSpace
from baybe.surrogates.base import Surrogate
from baybe.utils.dataframe import to_tensor

_DEFAULT_EXPLAINER_CLS = "KernelExplainer"
SHAP_EXPLAINERS = {
    "AdditiveExplainer",
    "ExactExplainer",
    "KernelExplainer",
    "PartitionExplainer",
    "PermutationExplainer",
}
NON_SHAP_EXPLAINERS = {"LimeTabular", "Maple"}
EXPLAINERS = {*SHAP_EXPLAINERS, *NON_SHAP_EXPLAINERS}
SHAP_PLOTS = {"bar", "beeswarm", "heatmap", "scatter"}


def _get_explainer_cls(name: str) -> type[shap.Explainer]:
    """Retrieve the explainer class reference by name."""
    if name in SHAP_EXPLAINERS:
        return getattr(shap.explainers, name)
    if name in NON_SHAP_EXPLAINERS:
        return getattr(shap.explainers.other, name)
    raise ValueError(f"Unknown SHAP explainer class '{name}'.")


def is_shap_explainer(explainer: shap.Explainer) -> bool:
    """Whether the explainer is a SHAP explainer or not (e.g. MAPLE, LIME)."""
    return type(explainer).__name__ in SHAP_EXPLAINERS


def _make_explainer(
    surrogate: Surrogate,
    data: pd.DataFrame,
    explainer_cls: type[shap.Explainer] | str = _DEFAULT_EXPLAINER_CLS,
    use_comp_rep: bool = False,
    **kwargs,
) -> shap.Explainer:
    """Create a SHAP explainer.

    Args:
        surrogate: The surrogate to be explained.
        data: The background data set.
        explainer_cls: The SHAP explainer class that is used to generate the
            explanation.
        use_comp_rep: Whether to analyze the model in computational representation
                (experimental representation otherwise).
        **kwargs: Additional keyword arguments to be passed to the explainer.

    Returns:
        shap.Explainer: The created explainer object.

    Raises:
        ValueError: If the provided background data set is empty.
        TypeError: If the provided explainer class does not
            support the campaign surrogate.
    """
    if data.empty:
        raise ValueError("The provided background data set is empty.")

    if isinstance(explainer_cls, str):
        explainer_cls = _get_explainer_cls(explainer_cls)

    import torch

    if use_comp_rep:

        def model(x: npt.ArrayLike) -> np.ndarray:
            tensor = to_tensor(x)
            with torch.no_grad():
                output = surrogate._posterior_comp(tensor).mean
            return output.numpy()

    else:

        def model(x: npt.ArrayLike) -> np.ndarray:
            df = pd.DataFrame(x, columns=data.columns)
            with torch.no_grad():
                output = surrogate.posterior(df).mean
            return output.numpy()

    # Handle special settings
    if "Lime" in explainer_cls.__name__:
        # Lime default mode is otherwise set to 'classification'
        kwargs["mode"] = "regression"

    try:
        shap_explainer = explainer_cls(model, data, **kwargs)

        # Explain first two data points to ensure that the explainer is working
        if is_shap_explainer(shap_explainer):
            shap_explainer(data.iloc[0:1])
    except shap.utils._exceptions.InvalidModelError:
        raise TypeError(
            f"The selected explainer class {explainer_cls} does not support the "
            f"provided surrogate model."
        )
    except TypeError as e:
        if "not supported for the input types" in str(e) and not use_comp_rep:
            raise NotImplementedError(
                f"The selected explainer class {explainer_cls} does not support "
                f"the experimental representation. Switch to computational "
                f"representation or use a different explainer (e.g. the default "
                f"shap.KernelExplainer)."
            )
        else:
            raise e
    return shap_explainer


@define
class SHAPInsight:
    """Class for SHAP-based feature importance insights.

    This also supports LIME and MAPLE explainers via ways provided by the shap module.
    """

    explainer: shap.Explainer = field(validator=instance_of(shap.Explainer))
    """The explainer instance."""

    background_data: pd.DataFrame = field(validator=instance_of(pd.DataFrame))
    """The background data set used to build the explainer."""

    @property
    def uses_shap_explainer(self) -> bool:
        """Whether the explainer is a SHAP explainer or not (e.g. MAPLE, LIME)."""
        return is_shap_explainer(self.explainer)

    @classmethod
    def from_surrogate(
        cls,
        surrogate: Surrogate,
        data: pd.DataFrame,
        explainer_cls: type[shap.Explainer] | str = _DEFAULT_EXPLAINER_CLS,
        use_comp_rep: bool = False,
    ):
        """Create a SHAP insight from a surrogate model."""
        explainer = _make_explainer(surrogate, data, explainer_cls, use_comp_rep)
        return cls(explainer, data)

    @classmethod
    def from_campaign(
        cls,
        campaign: Campaign,
        explainer_cls: type[shap.Explainer] | str = _DEFAULT_EXPLAINER_CLS,
        use_comp_rep: bool = False,
    ) -> SHAPInsight:
        """Create a SHAP insight from a campaign.

        Args:
            campaign: The campaign which holds the recommender and model.
            explainer_cls: The SHAP explainer class that is used to generate the
                explanation.
            use_comp_rep:
                Whether to analyze the model in computational representation
                (experimental representation otherwise).

        Returns:
            The SHAP insight object.
        """
        data = campaign.measurements[[p.name for p in campaign.parameters]].copy()
        background_data = campaign.searchspace.transform(data) if use_comp_rep else data

        return cls.from_surrogate(
            campaign.get_surrogate(),
            background_data,
            explainer_cls=explainer_cls,
            use_comp_rep=use_comp_rep,
        )

    @classmethod
    def from_recommender(
        cls,
        recommender: BayesianRecommender,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        explainer_cls: type[shap.Explainer] | str = "KernelExplainer",
        use_comp_rep: bool = False,
    ) -> SHAPInsight:
        """Create a SHAP insight from a recommender.

        Args:
            recommender: The model-based recommender.
            searchspace: The searchspace for the recommender.
            objective: The objective for the recommender.
            measurements: The background data set for Explainer.
                This is used the measurement data set for the recommender.
            explainer_cls: The explainer class.
            use_comp_rep:
                Whether to analyze the model in computational representation
                (experimental representation otherwise).

        Returns:
            The SHAP insight object.
        """
        surrogate_model = recommender.get_surrogate(
            searchspace, objective, measurements
        )

        return cls.from_surrogate(
            surrogate_model,
            searchspace.transform(measurements) if use_comp_rep else measurements,
            explainer_cls=explainer_cls,
            use_comp_rep=use_comp_rep,
        )

    def explain(self, df: pd.DataFrame | None = None, /) -> shap.Explanation:
        """Compute the Shapley values based on the chosen explainer and data set.

        Args:
            df: The data set for which the Shapley values should be computed.
                By default, the background data of the explainer is used.

        Returns:
            shap.Explanation: The computed Shapley values.

        Raises:
            ValueError: If the provided data set does not have the same amount of
                parameters as the SHAP explainer background
        """
        if df is None:
            df = self.background_data

        if not self.background_data.shape[1] == df.shape[1]:
            raise ValueError(
                "The provided data does not have the same amount of "
                "parameters as the shap explainer background."
            )

        if not self.uses_shap_explainer:
            # Return attributions for non-SHAP explainers
            if self.explainer.__module__.endswith("maple"):
                # Additional argument for maple to increase comparability to SHAP
                attributions = self.explainer.attributions(df, multiply_by_input=True)[
                    0
                ]
            else:
                attributions = self.explainer.attributions(df)[0]

            explanations = shap.Explanation(
                values=attributions,
                base_values=self.explainer.model(self.background_data).mean(),
                data=df,
                feature_names=df.columns.values,
            )
            return explanations
        else:
            explanations = self.explainer(df)

        # Reduce dimensionality of explanations to 2D in case
        # a 3D explanation is returned. This is the case for
        # some explainers even if only one output is present.
        if len(explanations.shape) == 2:
            return explanations
        if len(explanations.shape) == 3:
            return explanations[:, :, 0]
        raise RuntimeError(
            f"The explanation obtained for {self.__class__.__name__} has an unexpected "
            f"invalid dimensionality of {len(explanations.shape)}."
        )

    def plot(
        self,
        plot_type: Literal["bar", "beeswarm", "force", "heatmap", "scatter"],
        df: pd.DataFrame | None = None,
        /,
        *,
        show: bool = True,
        **kwargs: dict,
    ) -> plt.Axes:
        """Plot the Shapley values using the provided plot type.

        Args:
            plot_type: The type of plot to be created.
            df: The data for which the Shapley values shall be plotted.
                By default, the background data of the explainer is used.
            show: Boolean flag determining if the plot shall be rendered.
            **kwargs: Additional keyword arguments to be passed to the plot function.

        Returns:
            The plot object.

        Raises:
            ValueError: If the provided plot type is not supported.
        """
        if df is None:
            df = self.background_data

        if plot_type == "scatter":
            return self._plot_shap_scatter(df, show=show, **kwargs)

        if plot_type not in SHAP_PLOTS:
            raise ValueError(
                f"Invalid plot type: '{plot_type}'. "
                f"Available options: {SHAP_PLOTS}."
            )
        plot_func = getattr(shap.plots, plot_type)

        return plot_func(self.explain(df), show=show, **kwargs)

    def _plot_shap_scatter(
        self, df: pd.DataFrame | None = None, /, *, show: bool = True, **kwargs: dict
    ) -> plt.Axes:
        """Plot the Shapley values as scatter plot while leaving out non-numeric values.

        Args:
            df: The data for which the Shapley values shall be plotted.
                By default, the background data of the explainer is used.
            show: Boolean flag determining if the plot shall be rendered.
            **kwargs: Additional keyword arguments to be passed to the plot function.

        Returns:
            The plot object.

        Raises:
            ValueError: If no plot can be created because of non-numeric data.
        """
        if df is None:
            df = self.background_data

        df_numeric = df.select_dtypes("number")
        numeric_idx = df.columns.get_indexer(df_numeric.columns)
        if df_numeric.empty:
            raise ValueError(
                "No SHAP scatter plot can be created since all features contain "
                "non-numeric values."
            )
        if non_numeric_cols := set(df.columns) - set(df_numeric.columns):
            warnings.warn(
                f"The following features are excluded from the SHAP scatter plot "
                f"because their contain non-numeric values: {non_numeric_cols}",
                UserWarning,
            )
        return shap.plots.scatter(self.explain(df)[:, numeric_idx], show=show, **kwargs)
