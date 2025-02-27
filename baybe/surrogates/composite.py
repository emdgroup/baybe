"""Composite surrogate models."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

import pandas as pd
from attrs import define, field
from typing_extensions import override

from baybe.exceptions import IncompatibleSurrogateError
from baybe.objectives.base import Objective
from baybe.searchspace.core import SearchSpace
from baybe.serialization import converter
from baybe.serialization.mixin import SerialMixin
from baybe.surrogates.base import SurrogateProtocol
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.utils.basic import is_all_instance

if TYPE_CHECKING:
    from botorch.models.model import ModelList
    from botorch.posteriors import PosteriorList

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

    @property
    def _surrogates_flat(self) -> tuple[SurrogateProtocol, ...]:
        """The surrogates ordered according to the targets of the modeled objective."""
        return tuple(self.surrogates[t] for t in self._target_names)

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
            if is_all_instance(self._surrogates_flat, GaussianProcessSurrogate)
            else ModelList
        )
        return cls(*(s.to_botorch() for s in self._surrogates_flat))

    def posterior(self, candidates: pd.DataFrame, /) -> PosteriorList:
        """Compute the posterior for candidates in experimental representation.

        The (independent joint) posterior is represented as a collection of individual
        posterior models computed per target of the involved objective.
        For details, see :meth:`baybe.surrogates.base.Surrogate.posterior`.
        """
        if not all(hasattr(s, "posterior") for s in self._surrogates_flat):
            raise IncompatibleSurrogateError(
                "A posterior can only be computed if all involved surrogates offer "
                "posterior computation."
            )

        from botorch.posteriors import PosteriorList

        # TODO[typing]: a `has_all_attrs` typeguard similar to `is_all_instance` would
        #   be handy here but unclear if this is doable with the current typing system
        posteriors = [s.posterior(candidates) for s in self._surrogates_flat]  # type: ignore[attr-defined]
        return PosteriorList(*posteriors)


@converter.register_structure_hook
def structure_surrogate_getter(obj: dict, _) -> _SurrogateGetter:
    """Resolve the object type."""
    if (type_ := obj.pop("type")) == _BroadcastMapping.__name__:
        return converter.structure(obj, _BroadcastMapping[SurrogateProtocol])
    elif type_ == "dict":
        return converter.structure(obj, dict[str, SurrogateProtocol])
    return NotImplementedError(f"No structure hook implemented for '{type_}'.")


@converter.register_unstructure_hook
def unstructure_surrogate_getter(obj: _SurrogateGetter) -> dict:
    """Add the object type information."""
    return {"type": type(obj).__name__, **converter.unstructure(obj)}
