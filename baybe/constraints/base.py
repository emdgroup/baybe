"""Base classes for all constraints."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
from attrs import define, field
from attrs.validators import ge, instance_of, min_len

from baybe.serialization import SerialMixin
from baybe.utils.basic import classproperty

if TYPE_CHECKING:
    import polars as pl


@define
class Constraint(ABC, SerialMixin):
    """Abstract base class for all constraints."""

    # class variables
    numerical_only: ClassVar[bool] = False
    """Class variable encoding whether the constraint is valid only for numerical
    parameters."""

    # Object variables
    parameters: list[str] = field(validator=min_len(1))
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


@define(slots=False)
class DiscretePruningConstraint(DiscreteConstraint, ABC):
    """Abstract base class for discrete constraints that prune the search space.

    A pruning constraint's specification defines which entries are **kept** in the
    search space. The keyword-only ``exclude`` flag inverts this: when ``True``, the
    specification instead defines which entries are **pruned** and the complement is
    kept.

    Subclasses implement :meth:`_get_valid` (and optionally :meth:`_get_valid_polars`)
    to express positive "valid rows" logic. The base class derives the removal set and
    applies the ``exclude`` inversion.
    """

    # object variables
    exclude: bool = field(default=False, kw_only=True, validator=instance_of(bool))
    """Whether to invert the selection (keep the complement of the specification)."""

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

    def get_valid(
        self, df: pd.DataFrame, /, *, allow_missing: bool = False
    ) -> pd.Index:
        """Get the indices of dataframe entries that are valid under the constraint.

        Args:
            df: A dataframe where each row represents a parameter configuration.
            allow_missing: If ``False``, a :class:`ValueError` is raised when
                the dataframe is missing required parameter columns. If
                ``True``, the constraint performs partial filtering on the
                available columns.

        Returns:
            The dataframe indices of rows that fulfill the constraint.
        """
        invalid = self.get_invalid(df, allow_missing=allow_missing)
        return df.index.drop(invalid)

    def get_invalid(
        self, df: pd.DataFrame, /, *, allow_missing: bool = False
    ) -> pd.Index:
        """Get the indices of dataframe entries that are invalid under the constraint.

        Args:
            df: A dataframe where each row represents a parameter configuration.
            allow_missing: If ``False``, a :class:`ValueError` is raised when
                the dataframe is missing required parameter columns. If ``True``, the
                subclass is asked whether it can perform (partial) constraint
                evaluation; if not, an empty index is returned, signaling to the
                caller `there are no entries to be excluded *yet*`.

        Raises:
            ValueError: If ``allow_missing`` is ``False`` and the dataframe
                is missing required parameter columns.

        Returns:
            The dataframe indices of rows that violate the constraint.
        """
        # TODO: Should switch backends (pandas/polars/...) behind the scenes
        available = set(df.columns)

        if not allow_missing:
            if missing := self._required_parameters - available:
                raise ValueError(
                    f"'{self.__class__.__name__}' requires columns {missing} "
                    f"which are missing from the dataframe."
                )
        elif not self._can_evaluate(available):
            return pd.Index([])

        return self._get_invalid(df)

    def _get_invalid(self, df: pd.DataFrame, /) -> pd.Index:
        """Derive the invalid set from ``_get_valid``, applying ``exclude`` inversion.

        This method is concrete on the base class and should not be overridden.
        Subclasses implement :meth:`_get_valid` instead.
        """
        valid = self._get_valid(df)
        if self.exclude:
            return valid
        return df.index.drop(valid)

    @abstractmethod
    def _get_valid(self, df: pd.DataFrame, /) -> pd.Index:
        """Get the indices of valid entries (core logic for subclasses).

        This method is only called after it has been confirmed that the dataframe
        contains sufficient columns for (at least partial) evaluation. Implementations
        should contain only the constraint's core positive-selection logic without
        column-availability checks.

        The returned index represents the rows that the constraint's specification
        considers **valid** (to be kept when ``exclude=False``).

        Args:
            df: A dataframe where each row represents a parameter configuration.

        Returns:
            The dataframe indices of rows that fulfill the constraint's specification.
        """

    @classproperty
    def has_polars_implementation(cls) -> bool:
        """Whether this constraint class has a Polars implementation."""
        return cls._get_valid_polars is not DiscretePruningConstraint._get_valid_polars

    def get_invalid_polars(self) -> pl.Expr:
        """Translate the constraint to a Polars expression identifying rows to remove.

        Returns:
            The Polars expression to pass to :meth:`polars.LazyFrame.filter`.
        """
        valid_expr = self._get_valid_polars()
        if self.exclude:
            return valid_expr
        return ~valid_expr

    def _get_valid_polars(self) -> pl.Expr:
        """Translate the constraint to a Polars expression identifying valid rows.

        Subclasses with a Polars implementation override this method.

        Returns:
            A Polars expression that evaluates to ``True`` for rows that the
            constraint's specification considers valid.

        Raises:
            NotImplementedError: If the constraint class does not have a Polars
                implementation.
        """
        raise NotImplementedError(
            f"'{self.__class__.__name__}' does not have a Polars implementation."
        )


@define
class ContinuousConstraint(Constraint, ABC):
    """Abstract base class for continuous constraints."""

    # class variables
    numerical_only: ClassVar[bool] = True
    # See base class.


@define
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
