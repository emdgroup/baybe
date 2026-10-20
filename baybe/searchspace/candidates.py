"""Candidates module for managing lazy candidate generation."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import narwhals.stable.v2 as nw
from attr.validators import deep_iterable, instance_of, min_len
from attrs import Attribute, cmp_using, define, evolve, field
from typing_extensions import override

from baybe.constraints import DISCRETE_CONSTRAINTS_FILTERING_ORDER, validate_constraints
from baybe.constraints.base import DiscreteConstraint, DiscreteFilteringConstraint
from baybe.exceptions import InfiniteSpaceError
from baybe.parameters.base import DiscreteParameter
from baybe.parameters.utils import sort_parameters
from baybe.searchspace.utils import build_constrained_product
from baybe.searchspace.validation import validate_parameters
from baybe.settings import active_settings
from baybe.utils.basic import is_all_instance, to_tuple
from baybe.utils.dataframe import _df_equals, _filter_dataframe
from baybe.utils.validation import validate_parameter_input

if TYPE_CHECKING:
    from baybe.utils.dataframe import Filter


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


def _materialize_filtered(
    candidates: CandidatesProtocol, *filters: Filter, complement: bool
) -> TableCandidates:
    """Materialize a filtered candidate set into a new :class:`TableCandidates`.

    Generic fallback used by candidate representations that cannot express the
    given filters symbolically.

    Args:
        candidates: The candidate representation to be filtered.
        *filters: One or more filter criteria.
        complement: If ``False``, the rows matching all given filters are kept.
            If ``True``, the complement set is kept instead.

    Returns:
        A new :class:`TableCandidates` object holding the filtered rows.
    """
    filtered = _filter_dataframe(
        candidates._to_lazy(), *filters, complement=complement
    ).collect()
    return TableCandidates(candidates.parameters, filtered.to_native())


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

    constraints: tuple[DiscreteConstraint, ...] = field(
        default=(),
        converter=lambda x: to_tuple(
            sorted(
                x, key=lambda c: DISCRETE_CONSTRAINTS_FILTERING_ORDER.index(c.__class__)
            )
        ),
        validator=deep_iterable(member_validator=instance_of(DiscreteConstraint)),
    )
    """Constraints to filter the Cartesian product of parameter values."""

    @constraints.validator
    def _validate_constraints(
        self, _: Attribute, value: tuple[DiscreteConstraint, ...]
    ):  # noqa: DOC101, DOC103
        validate_constraints(value, self.parameters)

    @override
    @property
    def is_finite(self) -> bool:
        return all(p.is_finite for p in self.parameters)

    @override
    def filter(self, *filters: Filter, complement: bool = False) -> CandidatesProtocol:
        # Filtering constraints can be appended to the existing constraint list
        # without ever materializing the candidate set, keeping it fully lazy
        if is_all_instance(filters, DiscreteFilteringConstraint):
            if not complement:
                return evolve(self, constraints=self.constraints + tuple(filters))
            if len(filters) == 1:
                (f,) = filters
                inverted = evolve(f, exclude=not f.exclude)
                return evolve(self, constraints=self.constraints + (inverted,))
            # Otherwise:
            # ----------
            # TODO: Complementing a conjunction of multiple constraints amounts to a
            #   disjunction (De Morgan), which is not yet expressible as a single
            #   `DiscreteFilteringConstraint`. A `DiscreteOrConstraint` combinator or
            #   similar could close this gap; until then, fall back to materialization.

        # TODO: Non-constraint filters (narwhals `Expr`, dataframe/semi-join) could
        #   in principle also stay lazy, e.g. via:
        #   * An adapter `DiscreteFilteringConstraint` wrapping the filter so it
        #     plugs into `build_constrained_product`'s incremental filtering.
        #   * An explicit escape hatch on `ProductCandidates` for symbolic
        #     non-constraint filters.
        #   Boolean masks remain inherently eager, since they reference fixed row
        #   positions that only exist once the candidate set is materialized.

        return _materialize_filtered(self, *filters, complement=complement)

    @override
    def _to_lazy(self) -> nw.LazyFrame:
        if not self.is_finite:
            raise InfiniteSpaceError(
                "Cannot generate all candidates from an infinite space."
            )

        candidates_df = build_constrained_product(self.parameters, self.constraints)

        # TODO: Remove to lazy once build_constrained_product returns a nw.LazyFrame
        assert not isinstance(candidates_df, nw.LazyFrame)
        return nw.from_native(candidates_df).lazy()


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

    _dataframe: nw.DataFrame = field(
        converter=lambda x: nw.from_native(x, eager_only=True),
        eq=cmp_using(eq=_df_equals),
    )
    """The dataframe containing the candidates."""

    @_dataframe.validator
    def _validate_dataframe(self, _: Attribute, value: nw.DataFrame) -> None:  # noqa: DOC101, DOC103
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
        # TODO: If `_dataframe` were allowed to hold a `nw.LazyFrame` instead of
        #   being restricted to `nw.DataFrame`, filtering could be applied lazily
        return _materialize_filtered(self, *filters, complement=complement)

    @override
    def _to_lazy(self) -> nw.LazyFrame:
        return self._dataframe.lazy()


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
