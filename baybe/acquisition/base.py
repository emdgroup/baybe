"""Base classes for all acquisition functions."""

from __future__ import annotations

import gc
from abc import ABC
from typing import TYPE_CHECKING, ClassVar, Literal, overload

import pandas as pd
from attrs import define

from baybe.exceptions import (
    IncompatibleAcquisitionFunctionError,
    UnidentifiedSubclassError,
)
from baybe.objectives.base import Objective
from baybe.searchspace.core import SearchSpace
from baybe.serialization.mixin import SerialMixin
from baybe.surrogates.base import SurrogateProtocol
from baybe.utils.basic import classproperty
from baybe.utils.boolean import is_abstract
from baybe.utils.dataframe import to_tensor

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction as BotorchAcquisitionFunction


@define(frozen=True)
class AcquisitionFunction(ABC, SerialMixin):
    """Abstract base class for all acquisition functions."""

    abbreviation: ClassVar[str]
    """An alternative name for type resolution."""

    supports_multi_output: ClassVar[bool] = False
    """Whether this acquisition function can handle models with multiple outputs."""

    @classproperty
    def is_analytic(cls) -> bool:
        """Flag indicating whether the acquisition function is analytic."""
        return not cls.abbreviation.startswith("q")

    @classproperty
    def supports_batching(cls) -> bool:
        """Flag indicating whether batch recommendation is supported."""
        return cls.abbreviation.startswith("q")

    @classproperty
    def supports_pending_experiments(cls) -> bool:
        """Flag indicating whether pending experiments are supported.

        This is based on the same mechanism underlying batched recommendations.
        """
        return cls.supports_batching

    @classproperty
    def _non_botorch_attrs(cls) -> tuple[str, ...]:
        """Names of attributes that are not passed to the BoTorch constructor."""
        return ()

    def to_botorch(
        self,
        surrogate: SurrogateProtocol,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None = None,
    ) -> BotorchAcquisitionFunction:
        """Create the botorch-ready representation of the function.

        The required structure of `measurements` is specified in
        :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`.
        """
        from baybe.acquisition._builder import BotorchAcquisitionFunctionBuilder

        if pending_experiments is not None and not self.supports_pending_experiments:
            raise IncompatibleAcquisitionFunctionError(
                f"The chosen acquisition function of type '{self.__class__.__name__}' "
                f"does not support pending experiments."
            )

        return BotorchAcquisitionFunctionBuilder(
            self, surrogate, searchspace, objective, measurements, pending_experiments
        ).build()

    @overload
    def evaluate(
        self,
        candidates: pd.DataFrame,
        surrogate: SurrogateProtocol,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None = None,
        *,
        jointly: Literal[True],
    ) -> float: ...

    @overload
    def evaluate(
        self,
        candidates: pd.DataFrame,
        surrogate: SurrogateProtocol,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None = None,
        *,
        jointly: Literal[False] = False,
    ) -> pd.Series: ...

    def evaluate(
        self,
        candidates: pd.DataFrame,
        surrogate: SurrogateProtocol,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None = None,
        *,
        jointly: bool = False,
    ) -> pd.Series | float:
        """Get the acquisition values for the given candidates.

        Args:
            candidates: The candidate points in experimental representation.
                For details, see :meth:`baybe.surrogates.base.Surrogate.posterior`.
            surrogate: The surrogate model to use for the acquisition function.
            searchspace: The search space.
                See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`.
            objective: The objective.
                See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`.
            measurements: Available experimentation data.
                See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`.
            pending_experiments: Optional pending experiments.
                See :meth:`baybe.recommenders.base.RecommenderProtocol.recommend`.
            jointly: If ``False``, the acquisition values are computed independently
                for each candidate. If ``True``, a single joint acquisition value is
                computed for the entire candidate set.

        Returns:
            Depending on the joint mode, either a single batch acquisition value or a
            series of individual acquisition values.
        """
        import torch

        # Assemble the Botorch acquisition function and its input
        botorch_acqf = self.to_botorch(
            surrogate, searchspace, objective, measurements, pending_experiments
        )
        comp = to_tensor(searchspace.transform(candidates))

        # Depending on joint mode, evaluate using t- or q-batching
        in_ = comp if jointly else comp.unsqueeze(-2)
        with torch.no_grad():
            out = botorch_acqf(in_)
        if jointly:
            return out.item()
        return pd.Series(out.numpy(), index=candidates.index)


def _get_botorch_acqf_class(
    baybe_acqf_cls: type[AcquisitionFunction], /
) -> type[BotorchAcquisitionFunction]:
    """Extract the BoTorch acquisition class for the given BayBE acquisition class."""
    import botorch

    for cls in baybe_acqf_cls.mro():
        if (
            acqf_cls := getattr(botorch.acquisition, cls.__name__, False)
            or getattr(botorch.acquisition.multi_objective, cls.__name__, False)
            or getattr(botorch.acquisition.multi_objective.parego, cls.__name__, False)
        ):
            if is_abstract(acqf_cls):
                continue
            return acqf_cls  # type: ignore

    raise UnidentifiedSubclassError(
        f"No BoTorch acquisition function class match found for "
        f"'{baybe_acqf_cls.__name__}'."
    )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
