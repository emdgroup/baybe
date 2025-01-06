"""SHAP insights."""

from __future__ import annotations

import inspect
import numbers
import warnings
from typing import Any

import numpy as np
import pandas as pd
from attrs import Factory, define, field
from attrs.validators import instance_of, optional
from typing_extensions import override

from baybe import Campaign
from baybe._optional.insights import shap
from baybe.insights.base import Insight
from baybe.objectives.base import Objective
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.searchspace import SearchSpace
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
    "force",
    "heatmap",
    "scatter",
}


@define
class SHAPInsight(Insight):
    """Class for SHAP-based feature importance insights.

    This also supports LIME and MAPLE explainers via ways provided by the shap module.
    """

    background_data: pd.DataFrame = field(validator=instance_of(pd.DataFrame))
    """The background data set used to build the explainer."""

    explained_data: pd.DataFrame | None = field(
        default=None, validator=optional(instance_of(pd.DataFrame))
    )
    """The data for which a SHAP explanation is generated."""

    # FIXME[typing]: https://github.com/python/mypy/issues/10998
    explainer_cls: type[shap.Explainer] = field(  # type: ignore[assignment]
        default="KernelExplainer",
        converter=lambda x: ALL_EXPLAINERS[x] if isinstance(x, str) else x,
    )
    """The SHAP explainer class that is used to generate the explanation.

    Some non-SHAP explainers, like MAPLE and LIME, are also supported if they are
    available via 'shap.explainers.other'.
    """

    use_comp_rep: bool = field(default=False, validator=instance_of(bool))
    """Flag for toggling in which representation the insight should be provided."""

    _explainer: shap.Explainer | None = field(
        default=Factory(
            lambda self: self._init_explainer(self.background_data, self.explainer_cls),
            takes_self=True,
        ),
        init=False,
    )
    """The explainer generated from the model and background data."""

    _explanation: shap.Explanation | None = field(default=None, init=False)
    """The explanation generated."""

    @property
    def uses_shap_explainer(self) -> bool:
        """Whether the explainer is a SHAP explainer or not (e.g. MAPLE, LIME)."""
        return not self.explainer_cls.__module__.startswith("shap.explainers.other.")

    @override
    @classmethod
    def from_campaign(
        cls,
        campaign: Campaign,
        explained_data: pd.DataFrame | None = None,
        explainer_cls: type[shap.Explainer] | str = "KernelExplainer",
        use_comp_rep: bool = False,
    ) -> SHAPInsight:
        """Create a SHAP insight from a campaign.

        Args:
            campaign: The campaign which holds the recommender and model.
            explained_data: The data set to be explained. If None, all measurements
                from the campaign are used.
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

        return cls(
            campaign.get_surrogate(),
            background_data=background_data,
            explainer_cls=explainer_cls,
            use_comp_rep=use_comp_rep,
            explained_data=explained_data,
        )

    @override
    @classmethod
    def from_recommender(
        cls,
        recommender: BayesianRecommender,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        explained_data: pd.DataFrame | None = None,
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
            explained_data: The data set to be explained. If None,
                the background data set is used.
            explainer_cls: The explainer class.
            use_comp_rep:
                Whether to analyze the model in computational representation
                (experimental representation otherwise).

        Returns:
            The SHAP insight object.

        Raises:
            ValueError: If the recommender has not implemented a "get_surrogate" method.
        """
        if not hasattr(recommender, "get_surrogate"):
            raise ValueError(
                f"The provided recommender does not provide a surrogate model. A "
                f"'{cls.__name__}' needs a surrogate model and thus only works with "
                f"model-based recommenders."
            )
        surrogate_model = recommender.get_surrogate(
            searchspace, objective, measurements
        )

        return cls(
            surrogate_model,
            background_data=searchspace.transform(measurements)
            if use_comp_rep
            else measurements,
            explained_data=explained_data,
            explainer_cls=explainer_cls,
            use_comp_rep=use_comp_rep,
        )

    def _init_explainer(
        self,
        background_data: pd.DataFrame,
        explainer_cls: type[shap.Explainer] = shap.KernelExplainer,
        **kwargs,
    ) -> shap.Explainer:
        """Create a SHAP explainer.

        Args:
            background_data: The background data set.
            explainer_cls: The SHAP explainer class that is used to generate the
                explanation.
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
        if not self.uses_shap_explainer and not self.use_comp_rep:
            raise NotImplementedError(
                "Experimental representation is not supported for non-Kernel SHAP "
                "explainer."
            )

        if background_data.empty:
            raise ValueError("The provided background data set is empty.")

        if self.use_comp_rep:

            def model(x):
                tensor = to_tensor(x)
                output = self.surrogate._posterior_comp(tensor).mean

                return output.detach().numpy()
        else:

            def model(x):
                df = pd.DataFrame(x, columns=background_data.columns)
                output = self.surrogate.posterior(df).mean

                return output.detach().numpy()

        # Handle special settings
        if "Lime" in explainer_cls.__name__:
            # Lime default mode is otherwise set to 'classification'
            kwargs["mode"] = "regression"

        try:
            shap_explainer = explainer_cls(model, background_data, **kwargs)

            # Explain first two data points to ensure that the explainer is working
            if self.uses_shap_explainer:
                shap_explainer(self.background_data.iloc[0:1])
        except shap.utils._exceptions.InvalidModelError:
            raise TypeError(
                f"The selected explainer class {explainer_cls} does not support the "
                f"provided surrogate model."
            )
        except TypeError as e:
            if "not supported for the input types" in str(e) and not self.use_comp_rep:
                raise NotImplementedError(
                    f"The selected explainer class {explainer_cls} does not support "
                    f"the experimental representation. Switch to computational "
                    f"representation or use a different explainer (e.g. the default "
                    f"shap.KernelExplainer)."
                )
            else:
                raise e
        return shap_explainer

    def _init_explanation(
        self,
        explained_data: pd.DataFrame | None = None,
    ) -> shap.Explanation:
        """Compute the Shapley values based on the chosen explainer and data set.

        Args:
            explained_data: The data set for which the Shapley values should be
                computed.

        Returns:
            shap.Explanation: The computed Shapley values.

        Raises:
            ValueError: If the provided data set does not have the same amount of
                parameters as the SHAP explainer background
        """
        if explained_data is None:
            explained_data = self.background_data
        elif not self.background_data.shape[1] == explained_data.shape[1]:
            raise ValueError(
                "The provided data does not have the same amount of "
                "parameters as the shap explainer background."
            )

        # Type checking for mypy
        assert self._explainer is not None

        if not self.uses_shap_explainer:
            # Return attributions for non-SHAP explainers
            if self._explainer.__module__.endswith("maple"):
                # Additional argument for maple to increase comparability to SHAP
                attributions = self._explainer.attributions(
                    explained_data, multiply_by_input=True
                )[0]
            else:
                attributions = self._explainer.attributions(explained_data)[0]

            explanations = shap.Explanation(
                values=attributions,
                base_values=self._explainer.model(self.background_data).mean(),
                data=explained_data,
                feature_names=explained_data.columns.values,
            )
            return explanations
        else:
            explanations = self._explainer(explained_data)

        """Ensure that the explanation object is of the correct dimensionality."""
        if len(explanations.shape) == 2:
            return explanations
        if len(explanations.shape) == 3:
            return explanations[:, :, 0]
        raise RuntimeError(
            f"The explanation obtained for {self.__class__.__name__} has an unexpected "
            f"invalid dimensionality of {len(explanations.shape)}."
        )

    @property
    def explanation(self) -> shap.Explanation:
        """Get the SHAP explanation object. Uses lazy evaluation."""
        if self._explanation is None:
            self._explanation = self._init_explanation()

        return self._explanation

    def plot(self, plot_type: str, **kwargs: Any) -> None:
        """Plot the Shapley values using the provided plot type.

        Args:
            plot_type: The type of plot to be created. Supported types are:
                "bar", "beeswarm", "force", "heatmap", "scatter".
            **kwargs: Additional keyword arguments to be passed to the plot function.

        Raises:
            ValueError: If the provided plot type is not supported.
        """
        if plot_type == "scatter":
            self._plot_shap_scatter(**kwargs)
            return None

        plot = getattr(shap.plots, plot_type, None)
        if (
            (plot_type not in SUPPORTED_SHAP_PLOTS)
            or (plot is None)
            or (not callable(plot))
        ):
            raise ValueError(
                f"Invalid plot type: '{plot_type}'. Available options: "
                f"{SUPPORTED_SHAP_PLOTS}."
            )

        plot(self.explanation, **kwargs)

    def _plot_shap_scatter(self, **kwargs: Any) -> None:
        """Plot the Shapley values as scatter plot while leaving out string values.

        Args:
            **kwargs: Additional keyword arguments to be passed to the plot function.
        """

        def is_not_numeric_column(col):
            return np.array([not isinstance(v, numbers.Number) for v in col]).any()

        if np.ndim(self.background_data) == 1:
            if is_not_numeric_column(self.background_data):
                warnings.warn(
                    "Cannot plot scatter plot for the provided "
                    "explanation as it contains non-numeric values."
                )
            else:
                shap.plots.scatter(self.explanation)
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
            shap.plots.scatter(self.explanation[:, number_enum], **kwargs)
