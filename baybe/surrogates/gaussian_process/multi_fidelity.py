"""Multi-fidelity Gaussian process surrogates."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, ClassVar

from attrs import define, field
from typing_extensions import override

from baybe.parameters.base import Parameter
from baybe.surrogates.base import Surrogate
from baybe.surrogates.gaussian_process.utils import _ModelContext

if TYPE_CHECKING:
    from botorch.models.gpytorch import GPyTorchModel
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from botorch.posteriors import Posterior
    from torch import Tensor


@define
class GaussianProcessSurrogateSTMF(Surrogate):
    """Botorch's single task multi fidelity Gaussian process."""

    supports_transfer_learning: ClassVar[bool] = False
    # See base class.

    supports_multi_fidelity: ClassVar[bool] = True
    # See base class.

    _model = field(init=False, default=None, eq=False)
    """The actual model."""

    @override
    def to_botorch(self) -> GPyTorchModel:
        return self._model

    @override
    @staticmethod
    def _make_parameter_scaler_factory(
        parameter: Parameter,
    ) -> type[InputTransform] | None:
        return None

    @override
    @staticmethod
    def _make_target_scaler_factory() -> type[OutcomeTransform] | None:
        # For GPs, we let botorch handle the scaling. See [Scaling Workaround] above.
        return None

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        return self._model.posterior(candidates_comp_scaled)

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        import botorch
        import gpytorch

        assert self._searchspace is not None

        context = _ModelContext(self._searchspace)

        assert context.n_fidelity_dimensions > 0, (
            f"{self.__class__.__name__} can only be fit on multi fidelity searchspaces."
        )

        # For GPs, we let botorch handle the scaling. See [Scaling Workaround] above.
        input_transform = botorch.models.transforms.Normalize(  # type: ignore[attr-defined]
            train_x.shape[-1],
            bounds=context.parameter_bounds,
            indices=context.numerical_indices,
        )
        outcome_transform = botorch.models.transforms.Standardize(train_y.shape[-1])  # type: ignore[attr-defined]

        # construct and fit the Gaussian process
        self._model = botorch.models.SingleTaskMultiFidelityGP(
            train_x,
            train_y,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            data_fidelities=None
            if context.fidelity_idx is None
            else (context.fidelity_idx,),
        )

        mll = gpytorch.ExactMarginalLogLikelihood(self._model.likelihood, self._model)

        botorch.fit.fit_gpytorch_mll(mll)

    @override
    def __str__(self) -> str:
        return (
            "Wrapper for a"
            ":class:`~botorch.models.gp_regression_fidelity.SingleTaskMultiFidelityGP`,"
            "used as the default GP for discrete numerical fidelity parameters in,"
            "e.g., multi fidelity knowledge gradient."
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
