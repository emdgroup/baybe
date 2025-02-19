"""SHAP insights."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from attrs import define, field
from attrs.validators import instance_of
from shap import KernelExplainer

from baybe import Campaign
from baybe._optional.insights import shap
from baybe.exceptions import IncompatibleExplainerError
from baybe.objectives.base import Objective
from baybe.recommenders.base import RecommenderProtocol
from baybe.searchspace import SearchSpace
from baybe.surrogates.base import Surrogate, SurrogateProtocol
from baybe.utils.dataframe import to_tensor

_DEFAULT_EXPLAINER_CLS = "KernelExplainer"
SHAP_EXPLAINERS = {
    "AdditiveExplainer",
    "ExactExplainer",
    "KernelExplainer",
    "PartitionExplainer",
    "PermutationExplainer",
}
"""Supported SHAP explainer types for :class:`baybe.insights.shap.SHAPInsight`"""

NON_SHAP_EXPLAINERS = {"LimeTabular", "Maple"}
"""Supported non-SHAP explainer types for :class:`baybe.insights.shap.SHAPInsight`"""

EXPLAINERS = SHAP_EXPLAINERS | NON_SHAP_EXPLAINERS
"""Supported explainer types for :class:`baybe.insights.shap.SHAPInsight`"""

SHAP_PLOTS = {"bar", "beeswarm", "force", "heatmap", "scatter"}
"""Supported plot types for :meth:`baybe.insights.shap.SHAPInsight.plot`"""


def _get_explainer_cls(name: str) -> type[shap.Explainer]:
    """Retrieve the explainer class reference by name."""
    if name in SHAP_EXPLAINERS:
        return getattr(shap.explainers, name)
    if name in NON_SHAP_EXPLAINERS:
        return getattr(shap.explainers.other, name)
    raise ValueError(f"Unknown SHAP explainer class '{name}'.")


def is_shap_explainer(explainer: shap.Explainer) -> bool:
    """Indicate if the given explainer is a SHAP explainer or not (e.g. MAPLE, LIME)."""
    return type(explainer).__name__ in SHAP_EXPLAINERS


def make_explainer_for_surrogate(
    surrogate: Surrogate,
    data: pd.DataFrame,
    explainer_cls: type[shap.Explainer] | str = _DEFAULT_EXPLAINER_CLS,
    *,
    use_comp_rep: bool = False,
) -> shap.Explainer:
    """Create a SHAP explainer for a given surrogate model.

    Args:
        surrogate: The surrogate model to be explained.
        data: The background data set.
        explainer_cls: The SHAP explainer class for generating the explanation.
        use_comp_rep: Boolean flag specifying whether to explain the model's
            experimental or computational representation.

    Returns:
        The created explainer object.

    Raises:
        ValueError: If the provided background data set is empty.
        TypeError: If the provided explainer class is incompatible with the surrogate.
    """
    if data.empty:
        raise ValueError("The provided background data set is empty.")

    if isinstance(explainer_cls, str):
        explainer_cls = _get_explainer_cls(explainer_cls)

    if not (
        data.select_dtypes(exclude="number").empty
        or issubclass(explainer_cls, KernelExplainer)
    ):
        raise IncompatibleExplainerError(
            f"The selected explainer class '{explainer_cls.__name__}' does not support "
            f"categorical data. Switch to computational representation or use "
            f"'{KernelExplainer.__name__}'."
        )

    import torch

    if use_comp_rep:

        def model(x: np.ndarray) -> np.ndarray:
            tensor = to_tensor(x)
            with torch.no_grad():
                output = surrogate._posterior_comp(tensor).mean
            return output.numpy()

    else:

        def model(x: np.ndarray) -> np.ndarray:
            df = pd.DataFrame(x, columns=data.columns)
            with torch.no_grad():
                output = surrogate.posterior(df).mean
            return output.numpy()

    # Handle special settings: Lime default mode is otherwise set to "classification"
    kwargs = {"mode": "regression"} if explainer_cls.__name__ == "LimeTabular" else {}

    return explainer_cls(model, data, **kwargs)


@define
class SHAPInsight:
    """Class for SHAP-based feature importance insights.

    Also supports LIME and MAPLE explainers via the ``shap`` package.
    """

    explainer: shap.Explainer = field(validator=instance_of(shap.Explainer))
    """The explainer instance."""

    background_data: pd.DataFrame = field(validator=instance_of(pd.DataFrame))
    """The background data set used by the explainer."""

    @explainer.validator
    def _validate_explainer(self, _, explainer: shap.Explainer) -> None:
        """Validate the explainer type."""
        if (name := explainer.__class__.__name__) not in EXPLAINERS:
            raise ValueError(
                f"The given explainer type must be one of {EXPLAINERS}. "
                f"Given: '{name}'."
            )

    @property
    def uses_shap_explainer(self) -> bool:
        """Indicates if a SHAP explainer is used or not (e.g. MAPLE, LIME)."""
        return is_shap_explainer(self.explainer)

    @classmethod
    def from_surrogate(
        cls,
        surrogate: SurrogateProtocol,
        data: pd.DataFrame,
        explainer_cls: type[shap.Explainer] | str = _DEFAULT_EXPLAINER_CLS,
        *,
        use_comp_rep: bool = False,
    ):
        """Create a SHAP insight from a surrogate.

        For details, see :func:`make_explainer_for_surrogate`.
        """
        if not isinstance(surrogate, Surrogate):
            raise ValueError(
                f"'{cls.__name__}.{cls.from_surrogate.__name__}' only accepts "
                f"surrogate models of type '{Surrogate.__name__}' or its subclasses."
            )

        explainer = make_explainer_for_surrogate(
            surrogate, data, explainer_cls, use_comp_rep=use_comp_rep
        )
        return cls(explainer, data)

    @classmethod
    def from_campaign(
        cls,
        campaign: Campaign,
        explainer_cls: type[shap.Explainer] | str = _DEFAULT_EXPLAINER_CLS,
        *,
        use_comp_rep: bool = False,
    ) -> SHAPInsight:
        """Create a SHAP insight from a campaign.

        Uses the measurements of the campaign as background data.

        Args:
            campaign: A campaign using a surrogate-based recommender.
            explainer_cls: See :func:`make_explainer_for_surrogate`.
            use_comp_rep: See :func:`make_explainer_for_surrogate`.

        Returns:
            The SHAP insight object.

        Raises:
            ValueError: If the campaign does not contain any measurements.
        """
        if campaign.measurements.empty:
            raise ValueError("The campaign does not contain any measurements.")
        data = campaign.measurements[[p.name for p in campaign.parameters]]
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
        recommender: RecommenderProtocol,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        explainer_cls: type[shap.Explainer] | str = "KernelExplainer",
        *,
        use_comp_rep: bool = False,
    ) -> SHAPInsight:
        """Create a SHAP insight from a recommender.

        Uses the provided measurements to train the surrogate and as background data for
        the explainer.

        Args:
            recommender: A recommender using a surrogate model.
            searchspace: The searchspace for the recommender.
            objective: The objective for the recommender.
            measurements: The measurements for training the surrogate and the explainer.
            explainer_cls: See :func:`make_explainer_for_surrogate`.
            use_comp_rep: See :func:`make_explainer_for_surrogate`.

        Returns:
            The SHAP insight object.

        Raises:
            TypeError: If the recommender has no ``get_surrogate`` method.
        """
        if not hasattr(recommender, "get_surrogate"):
            raise TypeError(
                f"The provided recommender does not provide a surrogate model. "
                f"'{cls.__name__}' needs a surrogate model and thus only works with "
                f"model-based recommenders."
            )

        surrogate_model = recommender.get_surrogate(
            searchspace, objective, measurements
        )

        data = measurements[[p.name for p in searchspace.parameters]]

        return cls.from_surrogate(
            surrogate_model,
            searchspace.transform(data) if use_comp_rep else data,
            explainer_cls=explainer_cls,
            use_comp_rep=use_comp_rep,
        )

    def explain(self, data: pd.DataFrame | None = None, /) -> shap.Explanation:
        """Compute a Shapley explanation for a given data set.

        Args:
            data: The dataframe for which the Shapley values are to be computed.
                By default, the background data set of the explainer is used.

        Returns:
            The computed Shapley explanation.

        Raises:
            ValueError: If the columns of the given dataframe cannot be aligned with the
                columns of the explainer background dataframe.
        """
        if data is None:
            data = self.background_data
        elif set(self.background_data.columns) != set(data.columns):
            raise ValueError(
                "The provided dataframe must have the same column names as used by "
                "the explainer object."
            )

        # Align columns with background data
        df_aligned = data[self.background_data.columns]

        if not self.uses_shap_explainer:
            # Return attributions for non-SHAP explainers
            if self.explainer.__module__.endswith("maple"):
                # Additional argument for maple to increase comparability to SHAP
                attributions = self.explainer.attributions(
                    df_aligned, multiply_by_input=True
                )[0]
            else:
                attributions = self.explainer.attributions(df_aligned)[0]

            explanations = shap.Explanation(
                values=attributions,
                base_values=self.explainer.model(self.background_data).mean(),
                data=df_aligned.values,
                feature_names=df_aligned.columns.values,
            )
        else:
            explanations = self.explainer(df_aligned)

        # Permute explanation object data according to input column order
        # (`base_values` can be a scalar or vector)
        # TODO: https://github.com/shap/shap/issues/3958
        idx = self.background_data.columns.get_indexer(data.columns)
        for attr in ["values", "data", "base_values"]:
            try:
                setattr(explanations, attr, getattr(explanations, attr)[:, idx])
            except IndexError as ex:
                if not (
                    isinstance(explanations.base_values, float)
                    or explanations.base_values.shape[1] == 1
                ):
                    raise TypeError("Unexpected explanation format.") from ex
        explanations.feature_names = [explanations.feature_names[i] for i in idx]

        # Reduce dimensionality of explanations to 2D in case
        # a 3D explanation is returned. This is the case for
        # some explainers even if only one output is present.
        if len(explanations.shape) == 2:
            return explanations
        if len(explanations.shape) == 3:
            return explanations[:, :, 0]
        raise RuntimeError(
            f"The explanation obtained for '{self.__class__.__name__}' has an "
            f"unexpected dimensionality of {len(explanations.shape)}."
        )

    def plot(
        self,
        plot_type: Literal["bar", "beeswarm", "force", "heatmap", "scatter"],
        data: pd.DataFrame | None = None,
        /,
        *,
        show: bool = True,
        explanation_index: int | None = None,
        **kwargs: Any,
    ) -> plt.Axes:
        """Plot the Shapley values using the provided plot type.

        Args:
            plot_type: The type of plot to be created.
            data: See :meth:`explain`.
            show: Boolean flag determining if the plot is to be rendered.
            explanation_index: Positional index of the data point that should be
                explained. Only relevant for plot types that can only handle a single
                data point.
            **kwargs: Additional keyword arguments passed to the plot function.

        Returns:
            The plot object.

        Raises:
            ValueError: If the provided plot type is not supported.
        """
        if data is None:
            data = self.background_data

        # Use custom scatter plot function to ignore non-numeric features
        if plot_type == "scatter":
            return self._plot_shap_scatter(data, show=show, **kwargs)

        if plot_type not in SHAP_PLOTS:
            raise ValueError(
                f"Invalid plot type: '{plot_type}'. "
                f"Available options: {SHAP_PLOTS}."
            )
        plot_func = getattr(shap.plots, plot_type)

        # Handle plot types that only explain a single data point
        if plot_type == "force":
            if explanation_index is None:
                warnings.warn(
                    f"When using plot type '{plot_type}', an 'explanation_index' must "
                    f"be chosen to identify a single data point that should be "
                    f"explained. Choosing the first entry at position 0."
                )
                explanation_index = 0
            toplot = self.explain(data.iloc[[explanation_index]])
            kwargs["matplotlib"] = True
        else:
            toplot = self.explain(data)

        return plot_func(toplot, show=show, **kwargs)

    def _plot_shap_scatter(
        self, data: pd.DataFrame | None = None, /, *, show: bool = True, **kwargs: Any
    ) -> plt.Axes:
        """Plot the Shapley values as scatter plot, ignoring non-numeric features.

        For details, see :meth:`explain`.
        """
        if data is None:
            data = self.background_data

        df_numeric = data.select_dtypes("number")
        numeric_idx = data.columns.get_indexer(df_numeric.columns)
        if df_numeric.empty:
            raise ValueError(
                "No SHAP scatter plot can be created since all features contain "
                "non-numeric values."
            )
        if non_numeric_cols := set(data.columns) - set(df_numeric.columns):
            warnings.warn(
                f"The following features are excluded from the SHAP scatter plot "
                f"because they contain non-numeric values: {non_numeric_cols}",
                UserWarning,
            )
        return shap.plots.scatter(
            self.explain(data)[:, numeric_idx], show=show, **kwargs
        )
