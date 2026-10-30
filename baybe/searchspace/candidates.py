"""Candidates module for managing lazy candidate generation."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import narwhals.stable.v2 as nw
from attr.validators import deep_iterable, instance_of, min_len
from attrs import Attribute, cmp_using, define, evolve, field
from typing_extensions import override

from baybe.constraints import validate_constraints
from baybe.constraints.base import DiscreteFilteringConstraint
from baybe.exceptions import InfiniteSpaceError
from baybe.parameters.base import DiscreteParameter
from baybe.parameters.utils import sort_parameters
from baybe.searchspace.utils import build_constrained_product
from baybe.searchspace.validation import validate_parameters
from baybe.settings import active_settings
from baybe.utils._types import _NarwhalsFrame
from baybe.utils.basic import to_tuple
from baybe.utils.dataframe import Filter, _df_equals, _filter_dataframe, _is_filter
from baybe.utils.validation import validate_parameter_input

if TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoFrame


@runtime_checkable
class CandidatesProtocol(Protocol):
    """Type protocol specifying the interface candidate generators need to implement."""

    # Use slots so that derived classes also remain slotted
    # See also: https://www.attrs.org/en/stable/glossary.html#term-slotted-classes
    __slots__ = ()

    @property
    def parameters(self) -> tuple[DiscreteParameter, ...]:
        """The parameters spanning the space from which candidates are generated."""

    @property
    def is_finite(self) -> bool:
        """Indicates whether the candidate set is finite or infinite."""

    def filter(self, *filters: Filter, complement: bool = False) -> CandidatesProtocol:
        """Return a new candidate representation narrowed by the given filters."""

    def _to_lazy(self) -> nw.LazyFrame:
        """Generate all candidates."""


def _filter_candidates(
    candidates: CandidatesProtocol, *filters: Filter, complement: bool
) -> TableCandidates:
    """Filter a candidate set into a new lazy :class:`TableCandidates`.

    Generic fallback used by candidate representations that cannot express the
    given filters as part of their native representation.

    Args:
        candidates: The candidate representation to be filtered.
        *filters: One or more filter criteria.
        complement: If ``False``, the rows matching all given filters are kept.
            If ``True``, the complement set is kept instead.

    Returns:
        A new :class:`TableCandidates` object holding the filtered lazy frame.
    """
    filtered = _filter_dataframe(candidates._to_lazy(), *filters, complement=complement)
    return TableCandidates(candidates.parameters, filtered)


def _to_narwhals_frame(dataframe: IntoFrame | _NarwhalsFrame, /) -> _NarwhalsFrame:
    """Convert a native frame to a Narwhals frame (with narwhals frame passthrough)."""
    if isinstance(dataframe, (nw.DataFrame, nw.LazyFrame)):
        return dataframe
    return nw.maybe_reset_index(nw.from_native(dataframe))


@define(frozen=True)
class EmptyCandidates(CandidatesProtocol):
    """Sentinel class representing an empty candidate set with no parameters."""

    @override
    @property
    def parameters(self) -> tuple[DiscreteParameter, ...]:
        return ()

    @override
    @property
    def is_finite(self) -> bool:
        return True

    @override
    def filter(self, *filters: Filter, complement: bool = False) -> CandidatesProtocol:
        return self

    @override
    def _to_lazy(self) -> nw.LazyFrame:
        backend = active_settings.default_dataframe_backend
        return nw.from_dict({}, backend=backend).lazy()


@define(frozen=True)
class ProductCandidates(CandidatesProtocol):
    """Class for managing candidates from (filtered) Cartesian product spaces."""

    parameters: tuple[DiscreteParameter, ...] = field(
        converter=sort_parameters,
        validator=[
            min_len(1),
            deep_iterable(member_validator=instance_of(DiscreteParameter)),
            lambda _, __, x: validate_parameters(x),
        ],
    )
    """See :attr:`CandidatesProtocol.parameters`."""

    filters: tuple[Filter, ...] = field(default=(), converter=to_tuple)
    """Filters to apply to the Cartesian product."""

    @filters.validator
    def _validate_filters(
        self, attribute: Attribute, value: tuple[Filter, ...]
    ) -> None:  # noqa: DOC101, DOC103
        """Validate filters."""
        if invalid := [filter_ for filter_ in value if not _is_filter(filter_)]:
            raise TypeError(
                f"Invalid arguments provided as '{attribute.alias}': {invalid!r}."
            )
        constraints = [
            filter_
            for filter_ in value
            if isinstance(filter_, DiscreteFilteringConstraint)
        ]
        validate_constraints(constraints, self.parameters)

    @override
    @property
    def is_finite(self) -> bool:
        return all(p.is_finite for p in self.parameters)

    @override
    def filter(self, *filters: Filter, complement: bool = False) -> CandidatesProtocol:
        if not complement:
            return evolve(self, filters=self.filters + tuple(filters))

        if len(filters) == 1:
            # IMPROVE: Potentially it makes sense to introduce a `negate_filter`
            #    function or even turn the filter type union into an own type declaring
            #    the corresponding methods.
            match filter_ := filters[0]:
                case DiscreteFilteringConstraint():
                    inverted = evolve(filter_, exclude=not filter_.exclude)
                    return evolve(self, filters=self.filters + (inverted,))
                case nw.Expr():
                    return evolve(self, filters=self.filters + (~filter_,))

        # TODO: Complementing a conjunction of multiple filtering constraints amounts
        #   to a disjunction (De Morgan), which is not yet expressible as a single
        #   `DiscreteFilteringConstraint`. A `DiscreteOrConstraint` combinator or
        #   similar could close this gap; until then, fall back to a lazy table.
        return _filter_candidates(self, *filters, complement=complement)

    @override
    def _to_lazy(self) -> nw.LazyFrame:
        if not self.is_finite:
            raise InfiniteSpaceError(
                "Cannot generate all candidates from an infinite space."
            )

        constraints: list[DiscreteFilteringConstraint] = []
        non_constraint_filters: list[Filter] = []
        for filter_ in self.filters:
            if isinstance(filter_, DiscreteFilteringConstraint):
                constraints.append(filter_)
            else:
                non_constraint_filters.append(filter_)

        candidates_df = build_constrained_product(self.parameters, constraints)

        # TODO: Remove to lazy once build_constrained_product returns a nw.LazyFrame
        assert not isinstance(candidates_df, nw.LazyFrame)
        candidates = nw.from_native(candidates_df).lazy()
        return _filter_dataframe(candidates, *non_constraint_filters)


@define(frozen=True)
class TableCandidates(CandidatesProtocol):
    """Class for managing candidates provided in a tabular format."""

    parameters: tuple[DiscreteParameter, ...] = field(
        converter=sort_parameters,
        validator=[
            min_len(1),
            deep_iterable(member_validator=instance_of(DiscreteParameter)),
            lambda _, __, x: validate_parameters(x),
        ],
    )
    """See :attr:`CandidatesProtocol.parameters`."""

    _dataframe: _NarwhalsFrame = field(
        converter=_to_narwhals_frame,
        eq=cmp_using(eq=_df_equals),
    )
    """The dataframe containing the candidates."""

    @_dataframe.validator
    def _validate_dataframe(self, _: Attribute, value: _NarwhalsFrame) -> None:  # noqa: DOC101, DOC103
        if isinstance(value, nw.LazyFrame):
            columns = set(value.collect_schema().names())
            parameter_names = {parameter.name for parameter in self.parameters}
            if missing := parameter_names - columns:
                raise ValueError(
                    f"The input dataframe is missing columns for the following "
                    f"parameters: {missing}"
                )
            if extra := columns - parameter_names:
                raise ValueError(
                    f"The input dataframe contains columns that do not correspond to "
                    f"any parameter: {extra}"
                )
            return
        validate_parameter_input(
            value.to_pandas(),
            self.parameters,
            allow_extra=False,
            allow_empty=True,
        )

    @override
    @property
    def is_finite(self) -> bool:
        return True

    @override
    def filter(self, *filters: Filter, complement: bool = False) -> CandidatesProtocol:
        return _filter_candidates(self, *filters, complement=complement)

    @override
    def _to_lazy(self) -> nw.LazyFrame:
        if isinstance(self._dataframe, nw.LazyFrame):
            return self._dataframe
        return self._dataframe.lazy()


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
