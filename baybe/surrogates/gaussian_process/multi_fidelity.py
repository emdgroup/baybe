"""Multi-fidelity Gaussian process surrogates."""

from __future__ import annotations

import gc
from functools import partial
from typing import TYPE_CHECKING, ClassVar

from attrs import define, field
from attrs.validators import is_callable
from typing_extensions import override

from baybe.exceptions import IncompatibleSurrogateError
from baybe.parameters.base import Parameter
from baybe.parameters.fidelity import (
    CategoricalFidelityParameter,
    NumericalDiscreteFidelityParameter,
)
from baybe.searchspace.core import SearchSpaceFidelityType
from baybe.surrogates.base import Surrogate
from baybe.surrogates.gaussian_process.components.fit_criterion import (
    FitCriterionFactoryProtocol,
)
from baybe.surrogates.gaussian_process.components.generic import (
    GPComponentType,
    to_component_factory,
)
from baybe.surrogates.gaussian_process.components.likelihood import (
    LikelihoodFactoryProtocol,
)
from baybe.surrogates.gaussian_process.presets.baybe import (
    BayBEFitCriterionFactory,
    BayBELikelihoodFactory,
)
from baybe.surrogates.gaussian_process.utils import (
    _ModelContext,
    _validate_searchspace_has_non_index_input,
)

if TYPE_CHECKING:
    import pandas as pd
    from botorch.models.gpytorch import GPyTorchModel
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from botorch.posteriors import Posterior
    from torch import Tensor

    from baybe.objectives.base import Objective
    from baybe.searchspace import SearchSpace


@define
class GaussianProcessSurrogateSTMF(Surrogate):
    """A GP surrogate for numerical discrete fidelity parameters.

    Wraps BoTorch's ``SingleTaskMultiFidelityGP``, which uses a downsampling kernel
    to model smooth degradation along a continuous fidelity axis.
    """

    supports_multi_fidelity: ClassVar[bool] = True
    # See base class.

    likelihood_factory: LikelihoodFactoryProtocol = field(
        alias="likelihood_or_factory",
        factory=BayBELikelihoodFactory,
        converter=partial(  # type: ignore[misc]
            to_component_factory, component_type=GPComponentType.LIKELIHOOD
        ),
        validator=is_callable(),
    )
    """The factory used to create the likelihood for the Gaussian process.

    Accepts:
        * :class:`.components.likelihood.LikelihoodFactory`
        * :class:`gpytorch.likelihoods.Likelihood`
    """

    fit_criterion_factory: FitCriterionFactoryProtocol = field(
        alias="fit_criterion_or_factory",
        factory=BayBEFitCriterionFactory,
        converter=partial(  # type: ignore[misc]
            to_component_factory, component_type=GPComponentType.CRITERION
        ),
        validator=is_callable(),
    )
    """The fitting criterion for Gaussian process hyperparameter optimization.

    Accepts:
        * :class:`.components.fit_criterion.FitCriterion`
        * :class:`.components.fit_criterion.FitCriterionFactoryProtocol`
    """

    _model = field(init=False, default=None, eq=False)
    """The actual model."""

    @override
    def to_botorch(self) -> GPyTorchModel:
        return self._model

    @override
    def _validate_fit_context(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:

        if searchspace.fidelity_type == SearchSpaceFidelityType.SINGLEFIDELITY:
            raise IncompatibleSurrogateError(
                f"'{self.__class__.__name__}' can only be fit on search spaces with a "
                f"'{NumericalDiscreteFidelityParameter.__name__}'."
            )

        if (
            searchspace.fidelity_type
            == SearchSpaceFidelityType.CATEGORICALMULTIFIDELITY
        ):
            from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate

            raise IncompatibleSurrogateError(
                f"'{self.__class__.__name__}' does not support "
                f"'{CategoricalFidelityParameter.__name__}'. "
                f"Use '{GaussianProcessSurrogate.__name__}' instead."
            )

        _validate_searchspace_has_non_index_input(searchspace, self.__class__.__name__)

    @override
    @staticmethod
    def _make_parameter_scaler_factory(
        parameter: Parameter,
    ) -> type[InputTransform] | None:
        return None

    @override
    @staticmethod
    def _make_target_scaler_factory() -> type[OutcomeTransform] | None:
        # For GPs, we let botorch handle the scaling. See Note [Scaling Workaround]
        # in gaussian_process/core.py.
        return None

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        return self._model.posterior(candidates_comp_scaled)

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        import botorch
        from botorch.models.transforms import Normalize, Standardize

        assert self._searchspace is not None  # provided by base class
        assert self._objective is not None  # provided by base class
        assert self._measurements is not None  # provided by base class

        context = _ModelContext(self._searchspace, self._objective, self._measurements)

        assert context.fidelity_idx is not None, (
            f"{self.__class__.__name__} can only be fit on multi fidelity searchspaces."
        )

        ### Input/output scaling
        input_transform = Normalize(
            train_x.shape[-1],
            bounds=context.parameter_bounds,
            indices=context.numerical_indices,
        )
        outcome_transform = Standardize(train_y.shape[-1])

        ### Likelihood
        likelihood = self.likelihood_factory(
            context.searchspace, context.objective, context.measurements
        )

        ### Criterion
        criterion = self.fit_criterion_factory(
            context.searchspace, context.objective, context.measurements
        )

        ### Model construction and fitting
        self._model = botorch.models.SingleTaskMultiFidelityGP(
            train_x,
            train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            likelihood=likelihood,
            data_fidelities=(context.fidelity_idx,),
        )
        mll = criterion.to_gpytorch(self._model.likelihood, self._model)
        botorch.fit.fit_gpytorch_mll(mll)

    @override
    def __str__(self) -> str:
        return (
            "Wrapper for a "
            ":class:`~botorch.models.gp_regression_fidelity.SingleTaskMultiFidelityGP`,"
            " used as the default GP for discrete numerical fidelity parameters,"
            " e.g., in multi fidelity knowledge gradient."
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
