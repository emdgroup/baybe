"""Composite surrogate models."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pandas as pd
from attrs import define, field
from typing_extensions import override

from baybe.objectives.base import Objective
from baybe.searchspace.core import SearchSpace
from baybe.serialization.mixin import SerialMixin
from baybe.surrogates.base import SurrogateProtocol
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate

if TYPE_CHECKING:
    from botorch.models.model import ModelList


@define
class BroadcastingSurrogate(SurrogateProtocol):
    """A class for broadcasting single-target surrogate logic to multiple targets."""

    _template: SurrogateProtocol = field(alias="surrogate")
    """The surrogate architecture used for broadcasting."""

    _models: list[SurrogateProtocol] = field(init=False, factory=list)
    """The independent model copies used for the individual targets."""

    @override
    def fit(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        for target in objective.targets:
            model = deepcopy(self._template)
            model.fit(searchspace, target.to_objective(), measurements)
            self._models.append(model)

    @override
    def to_botorch(self) -> ModelList:
        from botorch.models import ModelList
        from botorch.models.model_list_gp_regression import ModelListGP

        cls = (
            ModelListGP
            if isinstance(self._template, GaussianProcessSurrogate)
            else ModelList
        )
        return cls(*(m.to_botorch() for m in self._models))


@define
class CompositeSurrogate(SerialMixin, SurrogateProtocol):
    """A class for composing multi-target surrogates from single-target surrogates."""

    surrogates: dict[str, SurrogateProtocol] = field()
    """A dictionary mapping target names to single-target surrogates."""

    _target_names: tuple[str, ...] = field(init=False, eq=False)
    """The names of the targets modeled by the surrogate outputs."""

    @override
    def fit(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        for target in objective.targets:
            self.surrogates[target.name].fit(
                searchspace, target.to_objective(), measurements
            )
        self._target_names = tuple(t.name for t in objective.targets)

    @override
    def to_botorch(self) -> ModelList:
        from botorch.models import ModelList
        from botorch.models.model_list_gp_regression import ModelListGP

        cls = (
            ModelListGP
            if all(
                isinstance(s, GaussianProcessSurrogate)
                for s in self.surrogates.values()
            )
            else ModelList
        )
        return cls(*(self.surrogates[t].to_botorch() for t in self._target_names))
