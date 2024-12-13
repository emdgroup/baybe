"""SHAP utilities."""

import inspect
import numbers
import warnings
from typing import Any, override

import numpy as np
import pandas as pd

from baybe import Campaign
from baybe._optional.insights import shap
from baybe.insights.base import Insight
from baybe.objectives.base import Objective
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.searchspace import SearchSpace
from baybe.utils.dataframe import to_tensor


class SHAPInsight(Insight):
    """Base class for all SHAP insights."""

    DEFAULT_SHAP_PLOTS = {
        "bar",
        "scatter",
        "heatmap",
        "force",
        "beeswarm",
    }

    @staticmethod
    def _get_explainer_maps() -> (
        tuple[dict[str, type[shap.Explainer]], dict[str, type[shap.Explainer]]]
    ):
        """Get explainer maps for SHAP and non-SHAP explainers.

        Returns:
            tuple[dict[str, type[shap.Explainer]], dict[str, type[shap.Explainer]]]:
              The explainer maps for SHAP and non-SHAP explainers.
        """
        EXCLUDED_EXPLAINER_KEYWORDS = ["Tree", "GPU", "Gradient", "Sampling", "Deep"]

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

    ALL_EXPLAINERS = {**SHAP_EXPLAINERS, **NON_SHAP_EXPLAINERS}

    def __init__(
        self,
        surrogate_model,
        bg_data: pd.DataFrame,
        explained_data: pd.DataFrame | None = None,
        explainer_class: shap.Explainer | str = "KernelExplainer",
        computational_representation: bool = False,
    ):
        super().__init__(surrogate_model)
        self._computational_representation = computational_representation
        explainer_cls = (
            explainer_class
            if not isinstance(explainer_class, str)
            or explainer_class not in self.ALL_EXPLAINERS
            else self.ALL_EXPLAINERS[explainer_class]
        )
        self._is_shap_explainer = not explainer_cls.__module__.startswith(
            "shap.explainers.other."
        )
        self._bg_data = bg_data
        self._explained_data = explained_data
        self.explainer = self._init_explainer(bg_data, explainer_cls)  # type: ignore[arg-type]
        self._explanation = None

    @override
    @classmethod
    def from_campaign(
        cls,
        campaign: Campaign,
        explained_data: pd.DataFrame | None = None,
        explainer_class: shap.Explainer | str = "KernelExplainer",
        computational_representation: bool = False,
    ):
        """Create a SHAP insight from a campaign.

        Args:
            campaign: The campaign to be used for the SHAP insight.
            explained_data: The data set to be explained. If None,
                all measurements from the campaign are used.
            explainer_class: The explainer class to be used for the computation.
            computational_representation:
                Whether to use the computational representation.

        Returns:
            SHAPInsight: The SHAP insight object.

        Raises:
            ValueError: If the campaign does not contain any measurements.
        """
        if campaign.measurements.empty:
            raise ValueError("The campaign does not contain any measurements.")
        data = campaign.measurements[[p.name for p in campaign.parameters]].copy()
        return cls(
            campaign.get_surrogate(),
            bg_data=campaign.searchspace.transform(data)
            if computational_representation
            else data,
            explainer_class=explainer_class,
            computational_representation=computational_representation,
            explained_data=explained_data,
        )

    @override
    @classmethod
    def from_recommender(
        cls,
        recommender: BayesianRecommender,
        searchspace: SearchSpace,
        objective: Objective,
        bg_data: pd.DataFrame,
        explained_data: pd.DataFrame | None = None,
        explainer_class: shap.Explainer | str = "KernelExplainer",
        computational_representation: bool = False,
    ):
        """Create a SHAP insight from a recommender.

        Args:
            recommender: The recommender to be used for the SHAP insight.
            searchspace: The searchspace for the recommender.
            objective: The objective for the recommender.
            bg_data: The background data set for Explainer.
                This is also the measurement data set for the recommender.
            explained_data: The data set to be explained. If None,
                the background data set is used.
            explainer_class: The explainer class.
            computational_representation:
                Whether to use the computational representation.

        Returns:
            SHAPInsight: The SHAP insight object.

        Raises:
            ValueError: If the recommender has not implemented a "get_surrogate" method.
        """
        if not hasattr(recommender, "get_surrogate"):
            raise ValueError(
                "The provided recommender does not provide a surrogate model."
            )
        surrogate_model = recommender.get_surrogate(searchspace, objective, bg_data)

        return cls(
            surrogate_model,
            bg_data=searchspace.transform(bg_data)
            if computational_representation
            else bg_data,
            explained_data=explained_data,
            explainer_class=explainer_class,
            computational_representation=computational_representation,
        )

    def _init_explainer(
        self,
        data: pd.DataFrame,
        explainer_class: type[shap.Explainer] = shap.KernelExplainer,
        **kwargs,
    ) -> shap.Explainer:
        """Create an explainer for the provided campaign.

        Args:
            data: The background data set.
            explainer_class: The explainer class to be used.
            **kwargs: Additional keyword arguments to be passed to the explainer.

        Returns:
            shap.Explainer: The created explainer object.

        Raises:
            NotImplementedError: If the provided explainer class does
                not support the experimental representation.
            ValueError: If the provided background data set is empty.
            TypeError: If the provided explainer class does not
                support the campaign surrogate.
        """
        if not self._is_shap_explainer and not self._computational_representation:
            raise NotImplementedError(
                "Experimental representation is not "
                "supported for non-Kernel SHAP explainer."
            )

        if data.empty:
            raise ValueError("The provided background data set is empty.")

        if self._computational_representation:

            def model(x):
                tensor = to_tensor(x)
                output = self.surrogate._posterior_comp(tensor).mean

                return output.detach().numpy()
        else:

            def model(x):
                df = pd.DataFrame(x, columns=data.columns)
                output = self.surrogate.posterior(df).mean

                return output.detach().numpy()

        try:
            shap_explainer = explainer_class(model, data, **kwargs)
            """Explain first two data points to ensure that the explainer is working."""
            if self._is_shap_explainer:
                shap_explainer(self._bg_data.iloc[0:1])
        except shap.utils._exceptions.InvalidModelError:
            raise TypeError(
                "The selected explainer class does not support the campaign surrogate."
            )
        except TypeError as e:
            if (
                "not supported for the input types" in str(e)
                and not self._computational_representation
            ):
                raise NotImplementedError(
                    "The selected explainer class does not support experimental "
                    "representation. Switch to computational representation "
                    "or use a different explainer "
                    "(e.g. the default shap.KernelExplainer)."
                )
        return shap_explainer

    def _init_explanation(
        self,
        data: pd.DataFrame | None = None,
    ) -> shap.Explanation:
        """Compute the Shapley values based on the chosen explainer and data set.

        Args:
            data: The data set for which the Shapley values should be computed.

        Returns:
            shap.Explanation: The computed Shapley values.

        Raises:
            ValueError: If the provided data set does not have the same amount of
                parameters as the SHAP explainer background
        """
        if data is None:
            data = self._bg_data
        elif not self._bg_data.shape[1] == data.shape[1]:
            raise ValueError(
                "The provided data does not have the same amount of "
                "parameters as the shap explainer background."
            )

        # Type checking for mypy
        assert isinstance(data, pd.DataFrame)

        if not self._is_shap_explainer:
            """Return attributions for non-SHAP explainers."""
            if self.explainer.__module__.endswith("maple"):
                """Additional argument for maple to increase comparability to SHAP."""
                attributions = self.explainer.attributions(
                    data, multiply_by_input=True
                )[0]
            else:
                attributions = self.explainer.attributions(data)[0]
            explanations = shap.Explanation(
                values=attributions,
                base_values=self.explainer.model(self._bg_data).mean(),
                data=data,
                feature_names=data.columns.values,
            )
            return explanations
        else:
            explanations = self.explainer(data)

        """Ensure that the explanation object is of the correct dimensionality."""
        if len(explanations.shape) == 2:
            return explanations
        if len(explanations.shape) == 3:
            return explanations[:, :, 0]
        raise ValueError(
            "The Explanation has an invalid "
            f"dimensionality of {len(explanations.shape)}."
        )

    @property
    def explanation(self) -> shap.Explanation:
        """Get the SHAP explanation object. Uses lazy evaluation.

        Returns:
        shap.Explanation: The SHAP explanation object.
        """
        if self._explanation is None:
            self._explanation = self._init_explanation()

        return self._explanation

    def plot(self, plot_type: str, **kwargs) -> None:
        """Plot the Shapley values using the provided plot type.

        Args:
            plot_type: The type of plot to be created. Supported types are:
                "bar", "scatter", "heatmap", "force", "beeswarm".
            **kwargs: Additional keyword arguments to be passed to the plot function.

        Raises:
            ValueError: If the provided plot type is not supported
        """
        if plot_type == "scatter":
            self._plot_shap_scatter(**kwargs)
            return None

        plot = getattr(shap.plots, plot_type, None)
        if (
            plot is None
            or not callable(plot)
            or plot_type not in self.DEFAULT_SHAP_PLOTS
        ):
            raise ValueError(f"Invalid plot type: {plot_type}")

        plot(self.explanation, **kwargs)

    def _plot_shap_scatter(self, **kwargs: Any) -> None:
        """Plot the Shapley values as scatter plot while leaving out string values.

        Args:
            **kwargs: Additional keyword arguments to be passed to the plot function.
        """

        def is_not_numeric_column(col):
            return np.array([not isinstance(v, numbers.Number) for v in col]).any()

        if np.ndim(self._bg_data) == 1:
            if is_not_numeric_column(self._bg_data):
                warnings.warn(
                    "Cannot plot scatter plot for the provided "
                    "explanation as it contains non-numeric values."
                )
            else:
                shap.plots.scatter(self.explanation)
        else:
            # Type checking for mypy
            assert isinstance(self._bg_data, pd.DataFrame)

            mask = self._bg_data.iloc[0].apply(lambda x: not isinstance(x, str))
            number_enum = np.where(mask)[0].tolist()

            if len(number_enum) < len(self._bg_data.iloc[0]):
                warnings.warn(
                    "Cannot plot SHAP scatter plot for all "
                    "parameters as some contain non-numeric values."
                )
            shap.plots.scatter(self.explanation[:, number_enum], **kwargs)
