"""Candidates module for managing lazy candidate generation."""

import gc
from typing import Protocol

import narwhals.stable.v2 as nw
from attr.validators import deep_iterable, instance_of, min_len
from attrs import Attribute, define, field
from typing_extensions import override

from baybe.constraints import DISCRETE_CONSTRAINTS_FILTERING_ORDER, validate_constraints
from baybe.constraints.base import DiscreteConstraint
from baybe.exceptions import InfiniteSpaceError
from baybe.parameters.base import DiscreteParameter
from baybe.parameters.utils import sort_parameters
from baybe.searchspace.utils import build_constrained_product
from baybe.searchspace.validation import validate_parameter_names
from baybe.utils.basic import to_tuple
from baybe.utils.dataframe import to_lazy
from baybe.utils.validation import validate_parameter_input


class CandidatesProtocol(Protocol):
    """Type protocol specifying the interface candidate generators need to implement."""

    @property
    def parameters(self) -> tuple[DiscreteParameter, ...]:
        """The parameters spanning the space from which candidates are generated."""

    @property
    def is_finite(self) -> bool:
        """Indicates whether the candidate set is finite or infinite."""

    def to_lazy(self) -> nw.LazyFrame:
        """Generate all candidates."""


@define(frozen=True)
class ProductCandidates(CandidatesProtocol):
    """Class for managing candidates from (filtered) Cartesian product spaces."""

    parameters: tuple[DiscreteParameter, ...] = field(
        converter=sort_parameters,
        validator=[
            min_len(1),
            deep_iterable(member_validator=instance_of(DiscreteParameter)),
            lambda _, __, x: validate_parameter_names(x),
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
    def to_lazy(self) -> nw.LazyFrame:
        if not self.is_finite:
            raise InfiniteSpaceError(
                "Cannot generate all candidates from an infinite space."
            )

        candidates_df = build_constrained_product(self.parameters, self.constraints)

        # TODO: Remove to lazy once build_constrained_product returns a nw.LazyFrame
        assert not isinstance(candidates_df, nw.LazyFrame)
        return to_lazy(candidates_df)


@define(frozen=True)
class TableCandidates(CandidatesProtocol):
    """Class for managing candidates provided in a tabular format."""

    parameters: tuple[DiscreteParameter, ...] = field(
        converter=sort_parameters,
        validator=[
            min_len(1),
            deep_iterable(member_validator=instance_of(DiscreteParameter)),
            lambda _, __, x: validate_parameter_names(x),
        ],
    )
    """See :attr:`CandidatesProtocol.parameters`."""

    dataframe: nw.LazyFrame = field(converter=to_lazy)
    """The dataframe containing the candidates."""

    @dataframe.validator
    def _validate_dataframe(self, _: Attribute, value: nw.LazyFrame) -> None:  # noqa: DOC101, DOC103
        # TODO: Remove collect().to_pandas() once validation on lazy frames is supported
        validate_parameter_input(value.collect().to_pandas(), self.parameters)

    @override
    @property
    def is_finite(self) -> bool:
        return True

    @override
    def to_lazy(self) -> nw.LazyFrame:
        return self.dataframe


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
