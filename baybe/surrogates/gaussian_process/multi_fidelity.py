"""Multi-fidelity Gaussian process surrogates."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, ClassVar

from attrs import define, field
from typing_extensions import override

from baybe.parameters.base import Parameter
from baybe.surrogates.base import Surrogate
from baybe.surrogates.gaussian_process.core import (
    _ModelContext,
)
from baybe.surrogates.gaussian_process.presets.core import (
    GaussianProcessPreset,
    make_gp_from_preset,
)

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

    # TODO: type should be Optional[botorch.models.SingleTaskGP] but is currently
    #   omitted due to: https://github.com/python-attrs/cattrs/issues/531
    _model = field(init=False, default=None, eq=False)
    """The actual model."""

    @staticmethod
    def from_preset(preset: GaussianProcessPreset) -> Surrogate:
        """Create a Gaussian process surrogate from one of the defined presets."""
        return make_gp_from_preset(preset)

    @override
    def to_botorch(self) -> GPyTorchModel:
        return self._model

    @override
    @staticmethod
    def _make_parameter_scaler_factory(
        parameter: Parameter,
    ) -> type[InputTransform] | None:
        # For GPs, we let botorch handle the scaling. See [Scaling Workaround] above.
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

        assert context.is_multi_fidelity, (
            "GaussianProcessSurrogateSTMF can only "
            "be fit on multi fidelity searchspaces."
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
        return "SingleTaskMultiFidelityGP with Botorch defaults."


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
