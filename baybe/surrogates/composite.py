"""Composite surrogate models."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

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

_T = TypeVar("_T")


class _SurrogateGetter(Protocol):
    """A index-based mapping from strings to surrogates."""

    def __getitem__(self, key: str) -> SurrogateProtocol: ...


@define
class _BroadcastMapping(Generic[_T]):
    """A wrapper class providing copies of a given template object via indexing access.

    Essentially a serializable version of ``defaultdict(lambda: deepcopy(template))``.
    """

    template: _T = field()
    """The template object to be copied upon indexing access."""

    _data: dict[Any, _T] = field(init=False, factory=dict, eq=False)
    """An internal storage keeping track of already requested copies."""

    def __getitem__(self, key: Any, /) -> _T:
        """Create a new object copy upon first access."""
        if key not in self._data:
            self._data[key] = deepcopy(self.template)
        return self._data[key]


@define
class CompositeSurrogate(SerialMixin, SurrogateProtocol):
    """A class for composing multi-target surrogates from single-target surrogates."""

    surrogates: _SurrogateGetter = field()
    """An index-based mapping from target names to single-target surrogates."""

    _target_names: tuple[str, ...] = field(init=False, eq=False)
    """The names of the targets modeled by the surrogate outputs."""

    @classmethod
    def from_template(cls, surrogate: SurrogateProtocol) -> CompositeSurrogate:
        """Broadcast a given single-target surrogate logic to multiple targets."""
        return CompositeSurrogate(_BroadcastMapping(surrogate))

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
                isinstance(self.surrogates[t], GaussianProcessSurrogate)
                for t in self._target_names
            )
            else ModelList
        )
        return cls(*(self.surrogates[t].to_botorch() for t in self._target_names))
