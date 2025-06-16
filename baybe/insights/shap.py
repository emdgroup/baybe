"""SHAP insights."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from attr.validators import deep_iterable
from attrs import define, field
from attrs.validators import instance_of
from shap import KernelExplainer

from baybe import Campaign
from baybe._optional.insights import shap
from baybe.exceptions import IncompatibleExplainerError, NoMeasurementsError
from baybe.objectives.base import Objective
from baybe.recommenders.base import RecommenderProtocol
from baybe.searchspace import SearchSpace
from baybe.surrogates import CompositeSurrogate
from baybe.surrogates.base import Surrogate, SurrogateProtocol
from baybe.utils.basic import is_all_instance
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

SHAP_PLOTS = {"bar", "beeswarm", "force", "heatmap", "scatter", "waterfall"}
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

    explainers: tuple[shap.Explainer, ...] = field(
        converter=tuple,
        validator=deep_iterable(
            member_validator=instance_of(shap.Explainer),
        ),
    )
    """The explainer instances."""

    background_data: pd.DataFrame = field(validator=instance_of(pd.DataFrame))
    """The background data set used by the explainer."""

    @explainers.validator
    def _validate_explainers(self, _, explainers: tuple[shap.Explainer, ...]) -> None:
        """Validate the explainer type."""
        for explainer in explainers:
            if (name := explainer.__class__.__name__) not in EXPLAINERS:
                raise ValueError(
                    f"The given explainer type must be one of {EXPLAINERS}. "
                    f"Given: '{name}'."
                )

    @classmethod
    def from_surrogate(
        cls,
        surrogate: SurrogateProtocol,
        data: pd.DataFrame,
        explainer_cls: type[shap.Explainer] | str = _DEFAULT_EXPLAINER_CLS,
        *,
        use_comp_rep: bool = False,
    ) -> SHAPInsight:
        """Create a SHAP insight from a surrogate.

        For details, see :func:`make_explainer_for_surrogate`.
        """
        if isinstance(surrogate, Surrogate):
            single_output_surrogates = (surrogate,)
        elif isinstance(surrogate, CompositeSurrogate):
            single_output_surrogates = surrogate._surrogates_flat  # type:ignore[assignment]
            if not is_all_instance(single_output_surrogates, Surrogate):
                raise TypeError(
                    f"'{cls.__name__}.{cls.from_surrogate.__name__}' only supports "
                    f"'{CompositeSurrogate.__name__}' if it is composed only of models "
                    f"of type '{Surrogate.__name__}'."
                )
        else:
            raise ValueError(
                f"'{cls.__name__}.{cls.from_surrogate.__name__}' only accepts "
                f"surrogate models derived from '{Surrogate.__name__}' or "
                f"{CompositeSurrogate.__name__}."
            )

        explainers = tuple(
            make_explainer_for_surrogate(
                s, data, explainer_cls, use_comp_rep=use_comp_rep
            )
            for s in single_output_surrogates
        )
        return cls(explainers, data)

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
            raise NoMeasurementsError("The campaign does not contain any measurements.")
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

    def explain_target(
        self, target_index: int, data: pd.DataFrame | None = None, /
    ) -> shap.Explanation:
        """Compute Shapley explanations for a given data set for a single-target.

        Args:
            target_index: The index of the target for which the explanation is created.
            data: The dataframe for which the Shapley values are to be computed.
                By default, the background data set of the explainer is used.

        Returns:
            The computed Shapley explanation.

        Raises:
            ValueError: If not all the columns of the explainer background dataframe
                are present in the given data.
        """
        if data is None:
            data = self.background_data
        elif not set(self.background_data.columns).issubset(data.columns):
            raise ValueError(
                "The provided dataframe must contain all columns that were used for "
                "the background data."
            )

        # Align columns with background data
        df_aligned = data[self.background_data.columns]

        explainer = self.explainers[target_index]
        if not is_shap_explainer(explainer):
            # Return attributions for non-SHAP explainers
            if explainer.__module__.endswith("maple"):
                # Additional argument for maple to increase comparability to SHAP
                attributions = explainer.attributions(
                    df_aligned, multiply_by_input=True
                )[0]
            else:
                attributions = explainer.attributions(df_aligned)[0]

            explanation = shap.Explanation(
                values=attributions,
                base_values=explainer.model(self.background_data).mean(),
                data=df_aligned.values,
                feature_names=df_aligned.columns.values,
            )
        else:
            explanation = explainer(df_aligned)

        # Permute explanation object data according to input column order
        # (`base_values` can be a scalar or vector)
        # TODO: https://github.com/shap/shap/issues/3958
        idx = self.background_data.columns.get_indexer(data.columns)
        idx = idx[idx != -1]  # Additional columns in data are ignored.
        for attr in ["values", "data", "base_values"]:
            try:
                setattr(explanation, attr, getattr(explanation, attr)[:, idx])
            except IndexError as ex:
                if not (
                    isinstance(explanation.base_values, float)
                    or explanation.base_values.shape[1] == 1
                ):
                    raise TypeError("Unexpected explanation format.") from ex
        explanation.feature_names = [explanation.feature_names[i] for i in idx]

        # Reduce dimensionality of explanations to 2D in case
        # a 3D explanation is returned. This is the case for
        # some explainers even if only one output is present.
        if len(explanation.shape) == 3:
            if explanation.shape[2] == 1:
                # Some explainers have a third dimension corresponding to the
                # number of model outputs (in this implementation always 1).
                explanation = explanation[:, :, 0]
            else:
                # Some explainers (like ``AdditiveExplainer``) have a third
                # dimension corresponding to feature interactions. The total shap
                # value is obtained by summing over them.
                explanation = explanation.sum(axis=2)
        elif len(explanation.shape) != 2:
            raise RuntimeError(
                f"The explanation obtained for '{self.__class__.__name__}' has an "
                f"unexpected dimensionality of {len(explanation.shape)}."
            )

        return explanation

    def explain(
        self, data: pd.DataFrame | None = None, /
    ) -> tuple[shap.Explanation, ...]:
        """Compute Shapley explanations for a given data set for all targets.

        Args:
            data: The dataframe for which the Shapley values are to be computed.
                By default, the background data set of the explainer is used.

        Returns:
            The computed Shapley explanations.
        """
        return tuple(self.explain_target(k, data) for k in range(len(self.explainers)))

    def plot(
        self,
        plot_type: Literal[
            "bar", "beeswarm", "force", "heatmap", "scatter", "waterfall"
        ],
        data: pd.DataFrame | None = None,
        /,
        *,
        show: bool = True,
        explanation_index: int | None = None,
        target_index: int | None = None,
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
            target_index: The index of the target for which the plot is created. Only
                required for multi-output objectives.
            **kwargs: Additional keyword arguments passed to the plot function.

        Returns:
            The plot object.

        Raises:
            ValueError: If the provided plot type is not supported.
            ValueError: If the target index is not specified for multi-output
                situations.
        """
        if data is None:
            data = self.background_data
        if target_index is None:
            if len(self.explainers) > 1:
                raise ValueError(
                    "The 'target_index' must be specified for multi-output scenarios."
                )
            target_index = 0

        # Use custom scatter plot function to ignore non-numeric features
        if plot_type == "scatter":
            return self._plot_shap_scatter(data, target_index, show=show, **kwargs)

        if plot_type not in SHAP_PLOTS:
            raise ValueError(
                f"Invalid plot type: '{plot_type}'. Available options: {SHAP_PLOTS}."
            )
        plot_func = getattr(shap.plots, plot_type)

        # Handle plot types that only explain a single data point
        if plot_type in ["force", "waterfall"]:
            if explanation_index is None:
                warnings.warn(
                    f"When using plot type '{plot_type}', an 'explanation_index' must "
                    f"be chosen to identify a single data point that should be "
                    f"explained. Choosing the first entry at position 0."
                )
                explanation_index = 0

            toplot = self.explain_target(target_index, data.iloc[[explanation_index]])
            toplot = toplot[0]

            if plot_type == "force":
                kwargs["matplotlib"] = True
        else:
            toplot = self.explain_target(target_index, data)

        return plot_func(toplot, show=show, **kwargs)

    def _plot_shap_scatter(
        self,
        data: pd.DataFrame | None = None,
        target_index: int = 0,
        /,
        *,
        show: bool = True,
        **kwargs: Any,
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
            self.explain_target(target_index, data)[:, numeric_idx], show=show, **kwargs
        )
