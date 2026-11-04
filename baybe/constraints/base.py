"""Base classes for all constraints."""

from __future__ import annotations

import gc
from abc import ABC
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import narwhals.stable.v2 as nw
from attrs import define, field
from attrs.validators import deep_iterable, ge, instance_of, min_len

from baybe.serialization import SerialMixin
from baybe.utils.basic import classproperty

if TYPE_CHECKING:
    from narwhals.stable.v2.typing import FrameT, IntoDataFrameT, IntoLazyFrameT


@define
class Constraint(ABC, SerialMixin):
    """Abstract base class for all constraints."""

    # class variables
    numerical_only: ClassVar[bool] = False
    """Class variable encoding whether the constraint is valid only for numerical
    parameters."""

    # Object variables
    parameters: list[str] = field(
        validator=[min_len(1), deep_iterable(member_validator=instance_of(str))]
    )
    """The list of parameters used for the constraint."""

    @parameters.validator
    def _validate_params(  # noqa: DOC101, DOC103
        self, _: Any, params: list[str]
    ) -> None:
        """Validate the parameter list.

        Raises:
            ValueError: If ``params`` contains duplicate values.
        """
        if len(params) != len(set(params)):
            raise ValueError(
                f"The given 'parameters' list must have unique values "
                f"but was: {params}."
            )

    def summary(self) -> dict:
        """Return a custom summarization of the constraint."""
        constr_dict = dict(
            Type=self.__class__.__name__, Affected_Parameters=self.parameters
        )
        return constr_dict

    @property
    def is_continuous(self) -> bool:
        """Boolean indicating if this is a constraint over continuous parameters."""
        return isinstance(self, ContinuousConstraint)

    @property
    def is_discrete(self) -> bool:
        """Boolean indicating if this is a constraint over discrete parameters."""
        return isinstance(self, DiscreteConstraint)

    @property
    def _required_parameters(self) -> set[str]:
        """All parameter names needed for full constraint evaluation.

        For most constraints, this is simply the set of names from
        :attr:`~baybe.constraints.base.Constraint.parameters`.
        Constraints with additional parameter references (e.g., affected
        parameters in dependency constraints) override this to include those.
        """
        return set(self.parameters)


@define
class DiscreteConstraint(Constraint, ABC):
    """Abstract base class for discrete constraints."""


@define
class DiscreteFilteringConstraint(DiscreteConstraint, ABC):
    """Abstract base class for discrete constraints that filter the search space.

    A filtering constraint's specification defines which entries are **kept** in the
    search space. The keyword-only ``exclude`` flag inverts this: when ``True``, the
    specification instead defines which entries are **removed** and the complement is
    kept.

    Subclasses implement ``_to_matching_expr`` for row-wise constraints or
    ``_to_matching_mask`` for frame-wise constraints.
    """

    # object variables
    exclude: bool = field(default=False, kw_only=True, validator=instance_of(bool))
    """Whether to invert the selection (keep the complement of the specification)."""

    def __call__(
        self, df: IntoDataFrameT | IntoLazyFrameT, /
    ) -> IntoDataFrameT | IntoLazyFrameT:
        """Filter a native frame by the constraint.

        Args:
            df: A native frame where each row represents a parameter configuration.

        Returns:
            The filtered native frame. Lazy inputs remain lazy, although frame-wise
            constraints materialize them while evaluating their mask.
        """
        # TODO[typing]: https://github.com/narwhals-dev/narwhals/issues/3319
        return self._filter(  # type: ignore[type-var]
            nw.from_native(df)  # pyrefly: ignore[bad-argument-type]
        ).to_native()

    def _filter(self, df: FrameT, /) -> FrameT:
        """Filter a Narwhals frame by the constraint.

        Use this internally when the caller already has a Narwhals frame. Public
        callers should use :meth:`__call__` with a native frame.

        Args:
            df: A Narwhals frame where each row represents a parameter configuration.

        Raises:
            ValueError: If the frame does not contain all columns required for
                constraint evaluation.

        Returns:
            The filtered Narwhals frame. Lazy inputs remain lazy, although frame-wise
            constraints materialize them while evaluating their mask.
        """
        available = set(df.columns)
        if not self._can_evaluate(available):
            missing = self._required_parameters - available
            raise ValueError(
                f"'{self.__class__.__name__}' requires columns {missing}, "
                f"which are missing from the provided dataframe."
            )

        if self._has_expression_implementation:
            return df.filter(self._to_expr(df.collect_schema()))

        if isinstance(df, nw.LazyFrame):
            eager_df = df.collect()
            filtered = eager_df.filter(self._to_mask(eager_df))
            # TODO[typing]: https://github.com/facebook/pyrefly/issues/1575
            return filtered.lazy()  # pyrefly: ignore[bad-return]

        return df.filter(self._to_mask(df))

    def _can_evaluate(self, available: set[str], /) -> bool:
        """Indicate whether the constraint can be (partially) evaluated.

        Called to decide if the constraint logic should be invoked at all. The default
        implementation requires *all* parameters considered by the constraint to be
        present. Subclasses that support useful partial filtering override this,
        taking ``self.exclude`` into account (partial-evaluation soundness depends on
        both the constraint type and the ``exclude`` flag).

        Args:
            available: The set of column names present in the dataframe that
                is about to be evaluated.

        Returns:
            ``True`` if the constraint can apply a meaningful partial filtering
            given the *available* columns, ``False`` otherwise.
        """
        return self._required_parameters <= available

    @classproperty
    def _has_expression_implementation(cls) -> bool:
        """Whether this constraint can be represented by a row-wise expression."""
        return (
            cls._to_matching_expr is not DiscreteFilteringConstraint._to_matching_expr
        )

    def _to_expr(self, schema: Mapping[str, Any], /) -> nw.Expr:
        """Return a row-wise expression identifying rows to keep.

        Args:
            schema: The schema of the incoming dataframe.

        Returns:
            An expression that evaluates to ``True`` for rows kept by the constraint.
        """
        matching = self._to_matching_expr(schema)
        return ~matching if self.exclude else matching

    def _to_matching_expr(self, schema: Mapping[str, Any], /) -> nw.Expr:
        """Return a row-wise expression identifying matching rows.

        Args:
            schema: The schema of the incoming dataframe.

        Raises:
            NotImplementedError: If the constraint has no expression implementation.

        Returns:
            An expression that evaluates to ``True`` for rows matching the constraint
            specification before applying :attr:`exclude`.
        """
        raise NotImplementedError(
            f"'{self.__class__.__name__}' does not have an expression implementation."
        )

    def _to_mask(self, df: nw.DataFrame, /) -> nw.Series:
        """Return a Boolean Narwhals series identifying rows to keep.

        Args:
            df: The Narwhals dataframe to evaluate.

        Returns:
            A Boolean Narwhals series identifying rows kept by the constraint.
        """
        matching = self._to_matching_mask(df)
        return ~matching if self.exclude else matching

    def _to_matching_mask(self, df: nw.DataFrame, /) -> nw.Series:
        """Return a Boolean Narwhals series identifying matching rows.

        Args:
            df: The Narwhals dataframe to evaluate.

        Raises:
            NotImplementedError: If the constraint has no frame-wise implementation.

        Returns:
            A Boolean Narwhals series identifying rows matching the constraint
            specification before applying :attr:`exclude`.
        """
        raise NotImplementedError(
            f"'{self.__class__.__name__}' does not implement frame-wise evaluation."
        )


@define
class ContinuousConstraint(Constraint, ABC):
    """Abstract base class for continuous constraints."""

    # class variables
    numerical_only: ClassVar[bool] = True
    # See base class.


@define(slots=False)
class CardinalityConstraint(Constraint, ABC):
    r"""Abstract base class for cardinality constraints.

    Places a constraint on the set of nonzero (i.e. "active") values among the
    specified parameters, bounding it between the two given integers, i.e.

    .. math::
        \text{min_cardinality} \leq |\{p_i : p_i \neq 0\}| \leq \text{max_cardinality}

    where :math:`\{p_i\}` are the parameters specified for the constraint.

    Note that this can be equivalently regarded as L0-constraint on the vector
    containing the specified parameters.
    """

    # class variable
    numerical_only: ClassVar[bool] = True
    # See base class.

    # object variables
    min_cardinality: int = field(default=0, validator=[instance_of(int), ge(0)])
    "The minimum required cardinality."

    max_cardinality: int = field(validator=instance_of(int))
    "The maximum allowed cardinality."

    @max_cardinality.default
    def _default_max_cardinality(self):
        """Use the number of involved parameters as the upper limit by default."""
        return len(self.parameters)

    def __attrs_post_init__(self):
        """Validate the cardinality bounds.

        Raises:
            ValueError: If the provided cardinality bounds are invalid.
            ValueError: If the provided cardinality bounds impose no constraint.
        """
        if self.min_cardinality > self.max_cardinality:
            raise ValueError(
                f"The lower cardinality bound cannot be larger than the upper bound. "
                f"Provided values: {self.max_cardinality=}, {self.min_cardinality=}."
            )

        if self.max_cardinality > len(self.parameters):
            raise ValueError(
                f"The cardinality bound cannot exceed the number of parameters. "
                f"Provided values: {self.max_cardinality=}, {len(self.parameters)=}."
            )

        if self.min_cardinality == 0 and self.max_cardinality == len(self.parameters):
            raise ValueError(
                f"No constraint of type `{self.__class__.__name__}' is required "
                f"when the lower cardinality bound is zero and the upper bound equals "
                f"the number of parameters. Provided values: {self.min_cardinality=}, "
                f"{self.max_cardinality=}, {len(self.parameters)=}"
            )


class ContinuousNonlinearConstraint(ContinuousConstraint, ABC):
    """Abstract base class for continuous nonlinear constraints."""


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
