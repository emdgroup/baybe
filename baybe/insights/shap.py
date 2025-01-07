"""SHAP insights."""

from __future__ import annotations

import inspect
import numbers
import warnings

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


def _get_explainer_maps() -> (
    tuple[dict[str, type[shap.Explainer]], dict[str, type[shap.Explainer]]]
):
    """Get maps for SHAP and non-SHAP explainers.

    Returns:
        The maps for SHAP and non-SHAP explainers.
    """
    EXCLUDED_EXPLAINER_KEYWORDS = [
        "Tree",
        "GPU",
        "Gradient",
        "Sampling",
        "Deep",
        "Linear",
    ]

    def _has_required_init_parameters(cls):
        """Check if non-shap initializer has required standard parameters."""
        REQUIRED_PARAMETERS = ["self", "model", "data"]

        init_signature = inspect.signature(cls.__init__)
        parameters = list(init_signature.parameters.keys())

        return parameters[:3] == REQUIRED_PARAMETERS

    shap_explainers = {
        cls_name: getattr(shap.explainers, cls_name)
        for cls_name in shap.explainers.__all__
        if all(x not in cls_name for x in EXCLUDED_EXPLAINER_KEYWORDS)
    }

    non_shap_explainers = {
        cls_name: explainer
        for cls_name in shap.explainers.other.__all__
        if _has_required_init_parameters(
            explainer := getattr(shap.explainers.other, cls_name)
        )
        and all(x not in cls_name for x in EXCLUDED_EXPLAINER_KEYWORDS)
    }

    return shap_explainers, non_shap_explainers


SHAP_EXPLAINERS, NON_SHAP_EXPLAINERS = _get_explainer_maps()
ALL_EXPLAINERS = SHAP_EXPLAINERS | NON_SHAP_EXPLAINERS
SUPPORTED_SHAP_PLOTS = {
    "bar",
    "beeswarm",
    "heatmap",
    "scatter",
}


def _convert_explainer_cls(x: type[shap.Explainer] | str) -> type[shap.Explainer]:
    """Get an explainer class from an explainer class name (with class passthrough)."""
    return ALL_EXPLAINERS[x] if isinstance(x, str) else x


def is_shap_explainer(explainer_cls: type[shap.Explainer]) -> bool:
    """Whether the explainer is a SHAP explainer or not (e.g. MAPLE, LIME)."""
    return not explainer_cls.__module__.startswith("shap.explainers.other.")


def _make_explainer(
    surrogate: Surrogate,
    data: pd.DataFrame,
    explainer_cls: type[shap.Explainer] | str = shap.KernelExplainer,
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

    explainer_cls = _convert_explainer_cls(explainer_cls)

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
        if is_shap_explainer(explainer_cls):
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
        return is_shap_explainer(type(self.explainer))

    @classmethod
    def from_surrogate(
        cls,
        surrogate: Surrogate,
        data: pd.DataFrame,
        explainer_cls: type[shap.Explainer] | str = "KernelExplainer",
        use_comp_rep: bool = False,
    ):
        """Create a SHAP insight from a surrogate model."""
        explainer = _make_explainer(surrogate, data, explainer_cls, use_comp_rep)
        return cls(explainer, data)

    @classmethod
    def from_campaign(
        cls,
        campaign: Campaign,
        explainer_cls: type[shap.Explainer] | str = "KernelExplainer",
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

        Raises:
            ValueError: If the campaign does not contain any measurements.
        """
        if campaign.measurements.empty:
            raise ValueError(
                f"The campaign does not contain any measurements. A '{cls.__name__}' "
                f"assumes there is mandatory background data in the form of "
                f"measurements as part of the campaign."
            )
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

    def explain(self, df: pd.DataFrame, /) -> shap.Explanation:
        """Compute the Shapley values based on the chosen explainer and data set.

        Args:
            df: The data set for which the Shapley values should be computed.

        Returns:
            shap.Explanation: The computed Shapley values.

        Raises:
            ValueError: If the provided data set does not have the same amount of
                parameters as the SHAP explainer background
        """
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
        self, df: pd.DataFrame, /, plot_type: str, **kwargs: dict
    ) -> None | plt.Axes:
        """Plot the Shapley values using the provided plot type.

        Args:
            df: The data for which the Shapley values shall be plotted.
            plot_type: The type of plot to be created. Supported types are:
                "bar", "beeswarm", "force", "heatmap", "scatter".
            **kwargs: Additional keyword arguments to be passed to the plot function.

        Returns:
            None | plt.Axes: The plot object if 'show' is set to False.

        Raises:
            ValueError: If the provided plot type is not supported.
        """
        # Extract the 'show' argument from the kwargs
        show = kwargs.pop("show", True)

        plot = None

        # Special case for scatter plot
        if plot_type == "scatter":
            plot = self._plot_shap_scatter(df, show=show, **kwargs)
            if not show:
                return plot
            return None

        # Cases for all other plots
        plot_func = getattr(shap.plots, plot_type, None)
        if (
            (plot_type not in SUPPORTED_SHAP_PLOTS)
            or (plot_func is None)
            or (not callable(plot_func))
        ):
            raise ValueError(
                f"Invalid plot type: '{plot_type}'. Available options: "
                f"{SUPPORTED_SHAP_PLOTS}."
            )

        plot = plot_func(self.explain(df), show=show, **kwargs)
        if not show:
            return plot
        return None

    def _plot_shap_scatter(
        self, df: pd.DataFrame, /, show: bool = True, **kwargs: dict
    ) -> None | plt.Axes:
        """Plot the Shapley values as scatter plot while leaving out string values.

        Args:
            df: The data for which the Shapley values shall be plotted.
            show: Whether to call plt.show() after plotting or not.
            **kwargs: Additional keyword arguments to be passed to the plot function.

        Returns:
            None | plt.Axes: The plot object if 'show' is set to False.
        """
        plot = None

        def is_not_numeric_column(col):
            return np.array([not isinstance(v, numbers.Number) for v in col]).any()

        if np.ndim(self.background_data) == 1:
            if is_not_numeric_column(self.background_data):
                warnings.warn(
                    "Cannot plot scatter plot for the provided "
                    "explanation as it contains non-numeric values."
                )
            else:
                plot = shap.plots.scatter(self.explain(df), show=show, **kwargs)
        else:
            # Type checking for mypy
            assert isinstance(self.background_data, pd.DataFrame)

            mask = self.background_data.iloc[0].apply(lambda x: not isinstance(x, str))
            number_enum = np.where(mask)[0].tolist()

            if len(number_enum) < len(self.background_data.iloc[0]):
                warnings.warn(
                    "Cannot plot SHAP scatter plot for all parameters as some contain "
                    "non-numeric values."
                )
            plot = shap.plots.scatter(
                self.explain(df)[:, number_enum], show=show, **kwargs
            )

        if not show:
            return plot
        return None
