"""Composite surrogate models."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

import pandas as pd
from attrs import define, field
from typing_extensions import override

from baybe.exceptions import IncompatibleSurrogateError
from baybe.objectives.base import Objective
from baybe.searchspace.core import SearchSpace
from baybe.serialization import converter
from baybe.serialization.core import _TYPE_FIELD, add_type
from baybe.serialization.mixin import SerialMixin
from baybe.surrogates.base import PosteriorStatistic, SurrogateProtocol
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.utils.basic import is_all_instance
from baybe.utils.dataframe import handle_missing_values

if TYPE_CHECKING:
    from botorch.models.model import ModelList
    from botorch.posteriors import PosteriorList
    from torch import Tensor

_T = TypeVar("_T")


class _SurrogateGetter(Protocol):
    """An index-based mapping from strings to surrogates."""

    def __getitem__(self, key: str) -> SurrogateProtocol: ...


@define
class _ReplicationMapping(Generic[_T]):
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

    # IMPROVE: Currently, the class is implemented in the most vanilla way, using only
    #   BayBE's existing interfaces. There are several ways how it can be
    #   further optimized by integrating it more directly with the underlying gpytorch
    #   models. However, this probably requires some additional code adaptations to
    #   achieve a full integration. Some future directions:
    #   * Instead of fitting the models sequentially, a parallel optimization can
    #     be done via `SumMarginalLogLikelihood`. However, a full integration would
    #     also require supporting different fitting routines (e.g. LOO)
    #   * The manual construction of the `PosteriorList` can be avoided when
    #     the posterior computation is triggered directly on the `ModelList`. However,
    #     this requires a clean integration of the necessary pre-processing steps
    #     (transformation to computational representation + scaling)
    #   * There is currently a lot of redundancy because each of the surrogates
    #     internally stores a references to the fitting context (e.g. search space,
    #     objective, ...)

    surrogates: _SurrogateGetter = field()
    """An index-based mapping from target names to single-target surrogates."""

    _target_names: tuple[str, ...] | None = field(init=False, eq=False)
    """The names of the targets modeled by the surrogate outputs."""

    @classmethod
    def from_replication(cls, surrogate: SurrogateProtocol) -> CompositeSurrogate:
        """Replicate a given single-target surrogate logic for multiple targets."""
        return CompositeSurrogate(_ReplicationMapping(surrogate))

    @property
    def _surrogates_flat(self) -> tuple[SurrogateProtocol, ...]:
        """The surrogates ordered according to the targets of the modeled objective."""
        assert self._target_names is not None
        return tuple(self.surrogates[t] for t in self._target_names)

    @override
    def fit(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        for target in objective.targets:
            # Drop partial measurements for the respective target
            measurements_filtered = handle_missing_values(
                measurements, [target.name], drop=True
            )

            self.surrogates[target.name].fit(
                searchspace, target.to_objective(), measurements_filtered
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

    def posterior(self, candidates: pd.DataFrame) -> PosteriorList:
        """Compute the posterior for candidates in experimental representation.

        The (independent joint) posterior is represented as a collection of individual
        posterior models computed per target of the involved objective.
        For details, see :meth:`baybe.surrogates.base.Surrogate.posterior`.
        """
        if not all(hasattr(s, "posterior") for s in self._surrogates_flat):
            raise IncompatibleSurrogateError(
                f"'{self.__class__.__name__}' can only compute a posterior in "
                f"experimental representation if all involved surrogates offer "
                f"posteriors in experimental representation."
            )

        from botorch.posteriors import PosteriorList

        # TODO[typing]: a `has_all_attrs` typeguard similar to `is_all_instance` would
        #   be handy here but unclear if this is doable with the current typing system
        posteriors = [s.posterior(candidates) for s in self._surrogates_flat]  # type: ignore[attr-defined]
        return PosteriorList(*posteriors)

    def _posterior_comp(self, candidates_comp: Tensor, /) -> PosteriorList:
        """Compute the posterior for candidates in computational representation.

        The (independent joint) posterior is represented as a collection of individual
        posterior models computed per target of the involved objective.
        For details, see :meth:`baybe.surrogates.base.Surrogate._posterior_comp`.
        """
        if not all(hasattr(s, "_posterior_comp") for s in self._surrogates_flat):
            raise IncompatibleSurrogateError(
                f"'{self.__class__.__name__}' can only compute a posterior in "
                f"computational representation if all involved surrogates offer "
                f"posteriors in computational representation."
            )

        from botorch.posteriors import PosteriorList

        # TODO[typing]: a `has_all_attrs` typeguard similar to `is_all_instance` would
        #   be handy here but unclear if this is doable with the current typing system
        posteriors = [s._posterior_comp(candidates_comp) for s in self._surrogates_flat]  # type: ignore[attr-defined]
        return PosteriorList(*posteriors)

    def posterior_stats(
        self,
        candidates: pd.DataFrame,
        stats: Sequence[PosteriorStatistic] = ("mean", "std"),
    ) -> pd.DataFrame:
        """See :meth:`baybe.surrogates.base.Surrogate.posterior_stats`."""
        if not all(hasattr(s, "posterior_stats") for s in self._surrogates_flat):
            raise IncompatibleSurrogateError(
                "Posterior statistics can only be computed if all involved surrogates "
                "offer this computation."
            )

        dfs = [s.posterior_stats(candidates, stats) for s in self._surrogates_flat]  # type: ignore[attr-defined]
        return pd.concat(dfs, axis=1)


def _get_surrogate_getter_type(type: str) -> type[_SurrogateGetter]:
    """Retrieve the concrete {class}`SurrogateGetter` type from its serialized name."""
    if type == _ReplicationMapping.__name__:
        return _ReplicationMapping[SurrogateProtocol]
    elif type == "dict":
        return dict[str, SurrogateProtocol]
    else:
        raise NotImplementedError(
            f"No implementation of '{_SurrogateGetter.__name__}' found for '{type}'."
        )


def _structure_surrogate_getter(obj: dict, _) -> _SurrogateGetter:
    """Structure into the specified type."""
    container_type = _get_surrogate_getter_type(obj.pop(_TYPE_FIELD))
    return converter.structure(obj, container_type)


@add_type
def _unstructure_surrogate_getter(obj: _SurrogateGetter) -> dict:
    """Unstructure as the concrete type."""
    container_type = _get_surrogate_getter_type(type(obj).__name__)
    return converter.unstructure(obj, unstructure_as=container_type)


converter.register_structure_hook_func(
    lambda t: t is _SurrogateGetter, _structure_surrogate_getter
)
converter.register_unstructure_hook_func(
    lambda t: t is _SurrogateGetter, _unstructure_surrogate_getter
)
