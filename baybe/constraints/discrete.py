"""Discrete constraints."""

from __future__ import annotations

import gc
from collections.abc import Callable, Sequence
from functools import reduce
from typing import TYPE_CHECKING, Any, ClassVar, cast

import cattrs
import numpy as np
import numpy.typing as npt
import pandas as pd
from attrs import define, field, fields
from attrs.validators import deep_iterable, ge, in_, instance_of, min_len
from typing_extensions import override

from baybe.constraints.base import (
    CardinalityConstraint,
    Constraint,
    DiscreteConstraint,
    DiscreteFilteringConstraint,
)
from baybe.constraints.conditions import (
    Condition,
    SubSelectionCondition,  # noqa: F401 (used in doctests)
    ThresholdCondition,
    _threshold_operators,
    _valid_logic_combiners,
    _valid_tolerance_operators,
)
from baybe.serialization import (
    block_deserialization_hook,
    block_serialization_hook,
    converter,
)
from baybe.serialization.core import _TYPE_FIELD, make_base_structure_hook
from baybe.utils.boolean import is_abstract
from baybe.utils.validation import finite_float

if TYPE_CHECKING:
    import polars as pl

    from baybe.symmetries.dependency import DependencySymmetry
    from baybe.symmetries.permutation import PermutationSymmetry


# >>>>>>>>>> Deprecation
def DiscreteExcludeConstraint(  # noqa: N802
    parameters: list[str],
    conditions: list[Condition],
    combiner: str = "AND",
) -> DiscreteSelectionConstraint:
    """A :class:`DiscreteSelectionConstraint` alias for backward compatibility."""  # noqa: D401
    import warnings

    warnings.warn(
        f"'{DiscreteExcludeConstraint.__name__}' is deprecated and will be removed "
        f"in a future version. Use '{DiscreteSelectionConstraint.__name__}' with "
        f"'{fields(DiscreteSelectionConstraint).exclude.alias}=True' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return DiscreteSelectionConstraint(
        parameters=parameters,
        conditions=conditions,
        combiner=combiner,
        exclude=True,
    )


# <<<<<<<<<< Deprecation


@define
class DiscreteSelectionConstraint(DiscreteFilteringConstraint):
    """Class for filtering search space entries based on conditions.

    Examples:
        >>> df = pd.DataFrame({
        ...     "Solvent": ["Water", "Water", "Hexane", "Hexane"],
        ...     "Temp": [80.0, 120.0, 80.0, 120.0],
        ... })
        >>> df
          Solvent   Temp
        0   Water   80.0
        1   Water  120.0
        2  Hexane   80.0
        3  Hexane  120.0
        >>> c = DiscreteSelectionConstraint(
        ...     parameters=["Solvent", "Temp"],
        ...     conditions=[
        ...         SubSelectionCondition(selection=["Hexane"]),
        ...         ThresholdCondition(threshold=100.0, operator=">="),
        ...     ],
        ...     exclude=True,
        ... )
        >>> list(c.get_invalid(df))
        [3]
    """

    # object variables
    conditions: list[Condition] = field(validator=min_len(1))
    """List of individual conditions."""

    combiner: str = field(default="AND", validator=in_(_valid_logic_combiners))
    """Operator encoding how to combine the individual conditions."""

    @override
    def _can_evaluate(self, available: set[str], /) -> bool:
        # A row can be dropped early during incremental construction only if a
        # later column can never change that decision. This holds exactly for:
        # - AND with exclude=False: once a present condition fails, the row is
        #   permanently dropped (an AND failure stays).
        # - OR with exclude=True: once a present condition holds, the row is
        #   permanently marked for removal (an OR match stays).
        # For XOR, the combined result can flip as further operands arrive, so
        # all parameters must be present first. All other cases must likewise
        # wait for every parameter.
        present = available & set(self.parameters)
        if not present:
            return False
        partial_ok = (self.combiner == "AND" and not self.exclude) or (
            self.combiner == "OR" and self.exclude
        )
        if not partial_ok and present != set(self.parameters):
            return False
        return True

    @override
    def _get_matching_rows(self, df: pd.DataFrame, /) -> pd.Index:
        pairs = [(p, c) for p, c in zip(self.parameters, self.conditions) if p in df]
        satisfied = [cond.evaluate(df[p]) for p, cond in pairs]
        res = reduce(_valid_logic_combiners[self.combiner], satisfied)

        return df.index[res]

    @override
    def _get_matching_rows_polars(self, schema: pl.Schema) -> pl.Expr:
        from baybe._optional.polars import polars as pl

        satisfied = []
        for k, cond in enumerate(self.conditions):
            satisfied.append(cond.to_polars(pl.col(self.parameters[k])))

        return pl.reduce(_valid_logic_combiners[self.combiner], satisfied)


@define
class DiscreteLinearConstraint(DiscreteFilteringConstraint):
    """Class for modelling linear (weighted-sum) constraints on discrete parameters.

    The constraint compares the (optionally weighted) sum of the specified parameters
    against ``rhs`` using the given ``operator``.

    Examples:
        >>> df = pd.DataFrame({"A": [1.0, 3.0, 5.0], "B": [2.0, 1.0, 3.0]})
        >>> df
             A    B
        0  1.0  2.0
        1  3.0  1.0
        2  5.0  3.0
        >>> c = DiscreteLinearConstraint(
        ...     parameters=["A", "B"],
        ...     operator="<=",
        ...     rhs=5.0,
        ... )
        >>> list(c.get_invalid(df))
        [2]

        With coefficients, the weighted sum is checked instead:

        >>> c = DiscreteLinearConstraint(
        ...     parameters=["A", "B"],
        ...     coefficients=(2.0, 1.0),
        ...     operator="<=",
        ...     rhs=5.0,
        ... )
        >>> list(c.get_invalid(df))
        [1, 2]
    """

    # IMPROVE: Look-ahead filtering would be possible if parameter
    # value ranges (min/max) were available to the constraint, allowing
    # bound-based pruning of partial sums before all parameters are
    # present. This could be expressed via a _can_evaluate override.

    # class variables
    numerical_only: ClassVar[bool] = True
    # See base class.

    # object variables
    operator: str = field(validator=in_(_threshold_operators))
    """The comparison operator (e.g. ``"="``, ``">="``, ``"<"``)."""

    coefficients: tuple[float, ...] = field(
        converter=lambda x: cattrs.structure(x, tuple[float, ...]),
        validator=deep_iterable(member_validator=finite_float),
    )
    """The coefficients for the weighted sum, one per entry in ``parameters``.

    Defaults to all-ones, i.e. an unweighted sum."""

    rhs: float = field(default=0.0, converter=float, validator=finite_float)
    """Right-hand side value of the comparison."""

    tolerance: float | None = field(
        default=None, converter=lambda x: float(x) if x is not None else None
    )
    """Numerical tolerance for equality/inequality operators that support it.

    Only applicable when ``operator`` is one of ``"="``, ``"=="``, ``"!="``.
    Set to a reasonable default when left as ``None``."""

    @coefficients.default
    def _default_coefficients(self) -> tuple[float, ...]:
        """Return equal weight coefficients as default."""
        return (1.0,) * len(self.parameters)

    @coefficients.validator
    def _validate_coefficients(  # noqa: DOC101, DOC103
        self, _: Any, coefficients: Sequence[float]
    ) -> None:
        """Validate the coefficients.

        Raises:
            ValueError: If the number of coefficients does not match the number of
                parameters.
        """
        if len(self.parameters) != len(coefficients):
            raise ValueError(
                "The given 'coefficients' list must have one floating point entry for "
                "each entry in 'parameters'."
            )
        if any(c == 0.0 for c in coefficients):
            raise ValueError("All entries in 'coefficients' must be non-zero.")

    @tolerance.validator
    def _validate_tolerance(  # noqa: DOC101, DOC103
        self, attribute: Any, value: float | None
    ) -> None:
        """Validate the tolerance.

        Raises:
            ValueError: If a tolerance is provided for a non-tolerance operator.
            ValueError: If the tolerance is not positive for a tolerance operator.
        """
        if self.operator not in _valid_tolerance_operators and value is not None:
            raise ValueError(
                f"Setting the '{attribute.alias}' is only valid with the following "
                f"operators: {_valid_tolerance_operators}, but got operator "
                f"'{self.operator}'."
            )
        if self.operator in _valid_tolerance_operators and value is not None:
            if value <= 0.0:
                raise ValueError(
                    f"'{attribute.alias}' must be positive, but got {value}."
                )

    def _build_condition(self) -> ThresholdCondition:
        """Build the internal threshold condition from the constraint fields."""
        kwargs: dict[str, Any] = {
            "threshold": self.rhs,
            "operator": self.operator,
        }
        if self.tolerance is not None:
            kwargs["tolerance"] = self.tolerance
        return ThresholdCondition(**kwargs)

    @override
    def _get_matching_rows(self, df: pd.DataFrame, /) -> pd.Index:
        evaluate_df = pd.Series(
            sum(
                df[p].to_numpy() * c for p, c in zip(self.parameters, self.coefficients)
            ),
            index=df.index,
        )
        condition = self._build_condition()
        mask_good = condition.evaluate(evaluate_df)

        return df.index[mask_good]

    @override
    def _get_matching_rows_polars(self, schema: pl.Schema) -> pl.Expr:
        from baybe._optional.polars import polars as pl

        weighted = [pl.col(p) * c for p, c in zip(self.parameters, self.coefficients)]
        condition = self._build_condition()
        return condition.to_polars(pl.sum_horizontal(weighted))


@define
class DiscreteProductConstraint(DiscreteFilteringConstraint):
    """Class for modelling product constraints on discrete parameters.

    The constraint compares the product of the specified parameters against ``rhs``
    using the given ``operator``.

    Examples:
        >>> df = pd.DataFrame({"A": [2.0, 3.0, 5.0], "B": [3.0, 2.0, 2.0]})
        >>> df
             A    B
        0  2.0  3.0
        1  3.0  2.0
        2  5.0  2.0
        >>> c = DiscreteProductConstraint(
        ...     parameters=["A", "B"],
        ...     operator="<=",
        ...     rhs=8.0,
        ... )
        >>> list(c.get_invalid(df))
        [2]
    """

    # IMPROVE: Look-ahead filtering would be possible if parameter
    # value ranges (min/max) were available to the constraint, allowing
    # bound-based pruning of partial products before all parameters are
    # present. This could be expressed via a _can_evaluate override.

    # class variables
    numerical_only: ClassVar[bool] = True
    # See base class.

    # object variables
    operator: str = field(default="", validator=instance_of(str))
    """The comparison operator (e.g. ``"="``, ``">="``, ``"<"``)."""

    rhs: float = field(default=0.0, converter=float, validator=finite_float)
    """Right-hand side value of the comparison."""

    tolerance: float | None = field(
        default=None, converter=lambda x: float(x) if x is not None else None
    )
    """Numerical tolerance for equality/inequality operators that support it.

    Only applicable when ``operator`` is one of ``"="``, ``"=="``, ``"!="``.
    Set to a reasonable default when left as ``None``."""

    # >>>>>>>>>> Deprecation
    condition: ThresholdCondition | None = field(default=None, kw_only=True)
    """Deprecated. Use ``operator``, ``rhs``, and ``tolerance`` instead."""

    # <<<<<<<<<< Deprecation

    def __attrs_post_init__(self):
        """Resolve the deprecated ``condition`` field and validate."""
        import warnings

        flds = fields(type(self))

        # >>>>>>>>>> Deprecation
        if self.condition is not None:
            if self.operator != "":
                raise ValueError(
                    f"Cannot specify both '{flds.condition.alias}' and "
                    f"'{flds.operator.alias}'. Use the new interface "
                    f"('{flds.operator.alias}', '{flds.rhs.alias}', "
                    f"'{flds.tolerance.alias}') instead."
                )
            warnings.warn(
                f"Passing '{flds.condition.alias}' to '{type(self).__name__}' is "
                f"deprecated and will be removed in a future version. Use "
                f"'{flds.operator.alias}' and '{flds.rhs.alias}' (and optionally "
                f"'{flds.tolerance.alias}') instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            object.__setattr__(self, "operator", self.condition.operator)
            object.__setattr__(self, "rhs", self.condition.threshold)
            object.__setattr__(self, "tolerance", self.condition.tolerance)
            object.__setattr__(self, "condition", None)
        # <<<<<<<<<< Deprecation

        # Validate operator
        if self.operator not in _threshold_operators:
            raise ValueError(
                f"'{flds.operator.alias}' must be one of "
                f"{list(_threshold_operators)}, but got '{self.operator}'."
            )

        # Validate tolerance
        if (
            self.operator not in _valid_tolerance_operators
            and self.tolerance is not None
        ):
            raise ValueError(
                f"Setting the '{flds.tolerance.alias}' is only valid with the "
                f"following operators: {_valid_tolerance_operators}, but got "
                f"operator '{self.operator}'."
            )
        if (
            self.operator in _valid_tolerance_operators
            and self.tolerance is not None
            and self.tolerance <= 0.0
        ):
            raise ValueError(
                f"'{flds.tolerance.alias}' must be positive, but got {self.tolerance}."
            )

    def _build_condition(self) -> ThresholdCondition:
        """Build the internal threshold condition from the constraint fields."""
        kwargs: dict[str, Any] = {
            "threshold": self.rhs,
            "operator": self.operator,
        }
        if self.tolerance is not None:
            kwargs["tolerance"] = self.tolerance
        return ThresholdCondition(**kwargs)

    @override
    def _get_matching_rows(self, df: pd.DataFrame, /) -> pd.Index:
        evaluate_df = df[self.parameters].prod(axis=1)
        condition = self._build_condition()
        mask_good = condition.evaluate(evaluate_df)

        return df.index[mask_good]

    @override
    def _get_matching_rows_polars(self, schema: pl.Schema) -> pl.Expr:
        from baybe._optional.polars import polars as pl

        condition = self._build_condition()
        expr = pl.reduce(lambda acc, x: acc * x, pl.col(self.parameters))
        return condition.to_polars(expr)


# >>>>>>>>>> Deprecation
def DiscreteSumConstraint(  # noqa: N802
    parameters, condition=None, coefficients=None, *, exclude=False
) -> DiscreteLinearConstraint:
    """A ``DiscreteLinearConstraint`` alias for backward compatibility."""  # noqa: D401
    import warnings

    warnings.warn(
        f"'DiscreteSumConstraint' is deprecated and will be removed in a future "
        f"version. Use '{DiscreteLinearConstraint.__name__}' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Translate the old ThresholdCondition-based interface
    if condition is not None:
        operator = condition.operator
        rhs = condition.threshold
        tolerance = condition.tolerance
    else:
        raise TypeError(
            f"Missing required argument 'condition'. Use "
            f"'{DiscreteLinearConstraint.__name__}' with 'operator' and 'rhs' instead."
        )

    new_kwargs: dict[str, Any] = {
        "operator": operator,
        "rhs": rhs,
        "exclude": exclude,
    }
    if tolerance is not None:
        new_kwargs["tolerance"] = tolerance
    if coefficients is not None:
        new_kwargs["coefficients"] = coefficients

    return DiscreteLinearConstraint(parameters, **new_kwargs)


# <<<<<<<<<< Deprecation


@define
class DiscreteRepetitionConstraint(DiscreteFilteringConstraint):
    """Class for constraining value repetition across parameters.

    Keeps only rows where no single value appears more than a specified number of
    times across the specified parameters.

    Examples:
        >>> df = pd.DataFrame({"A": ["x", "y", "x"], "B": ["y", "x", "x"]})
        >>> df
           A  B
        0  x  y
        1  y  x
        2  x  x

        Upper bound: row 2 has "x" twice, violating ``n_max_repetitions=1``:

        >>> c = DiscreteRepetitionConstraint(
        ...     parameters=["A", "B"], n_max_repetitions=1
        ... )
        >>> list(c.get_invalid(df))
        [2]

        With ``exclude=True``, the logic inverts and only repeated rows are kept:

        >>> c = DiscreteRepetitionConstraint(
        ...     parameters=["A", "B"], n_max_repetitions=1, exclude=True
        ... )
        >>> list(c.get_invalid(df))
        [0, 1]
    """

    # object variables
    n_max_repetitions: int = field(
        default=1, validator=[instance_of(int), ge(1)], kw_only=True
    )
    """Maximum number of times any single value may appear in a row."""

    def __attrs_post_init__(self) -> None:
        """Validate the maximum repetition count.

        Raises:
            ValueError: If the maximum repetition count imposes no meaningful
                constraint.
        """
        n_params = len(self.parameters)
        if self.n_max_repetitions >= n_params:
            raise ValueError(
                f"'{fields(type(self)).n_max_repetitions.alias}' must be less than "
                f"the number of parameters ({n_params}) to impose a meaningful "
                f"constraint, but got {self.n_max_repetitions}."
            )

    @override
    def _can_evaluate(self, available: set[str], /) -> bool:
        n_available = len(available & set(self.parameters))
        if self.exclude:
            # Once even assigning every missing parameter the same value cannot
            # exceed the maximum, the row is guaranteed to be excluded.
            return n_available >= len(self.parameters) - self.n_max_repetitions + 1
        # Exceeding the maximum requires at least one more available parameter.
        return n_available >= self.n_max_repetitions + 1

    @override
    def _get_matching_rows(self, df: pd.DataFrame, /) -> pd.Index:
        params = [p for p in self.parameters if p in df]

        # Encode all values to integer codes with a single global mapping so that
        # equality matches pandas semantics exactly (avoids false duplicates that a
        # naive string cast would introduce, e.g. int 1 vs. str "1"). Sorting the
        # integer codes per row groups equal values together.
        block = df[params].to_numpy()
        codes = pd.factorize(block.ravel())[0].reshape(block.shape)
        sorted_codes = np.sort(codes, axis=1)

        # Mark the start of each run of equal values along the sorted row, then
        # assign an increasing run id to every position via a cumulative sum.
        is_run_start = np.empty(sorted_codes.shape, dtype=bool)
        is_run_start[:, 0] = True
        is_run_start[:, 1:] = sorted_codes[:, 1:] != sorted_codes[:, :-1]
        run_ids = np.cumsum(is_run_start, axis=1)

        # The largest run (i.e. the highest per-value multiplicity) is found by
        # counting, for each possible run id, how many positions carry it. This
        # loop runs over the (small) number of parameters, not the dataframe rows.
        max_multiplicity = np.zeros(sorted_codes.shape[0], dtype=int)
        for run_id in range(1, sorted_codes.shape[1] + 1):
            max_multiplicity = np.maximum(
                max_multiplicity, (run_ids == run_id).sum(axis=1)
            )

        n_missing = len(self.parameters) - len(params)
        max_possible_multiplicity = (
            max_multiplicity + n_missing if self.exclude else max_multiplicity
        )
        mask_good = max_possible_multiplicity <= self.n_max_repetitions

        return df.index[mask_good]

    @override
    def _get_matching_rows_polars(self, schema: pl.Schema) -> pl.Expr:
        from baybe._optional.polars import polars as pl

        def _safe_eq(ci: str, cj: str) -> pl.Expr:
            """Compare two columns, returning ``False`` for incompatible dtypes."""
            di, dj = schema[ci], schema[cj]
            if di == dj or (di.is_numeric() and dj.is_numeric()):
                return pl.col(ci).eq_missing(pl.col(cj))
            return pl.lit(False)

        params = self.parameters
        counts = [pl.sum_horizontal(_safe_eq(ci, cj) for cj in params) for ci in params]
        max_count = pl.max_horizontal(counts)

        return max_count <= self.n_max_repetitions


# >>>>>>>>>> Deprecation
def DiscreteNoLabelDuplicatesConstraint(  # noqa: N802
    parameters: list[str],
) -> DiscreteRepetitionConstraint:
    """A :class:`DiscreteRepetitionConstraint` alias for backward compatibility."""  # noqa: D401
    import warnings

    flds = fields(DiscreteRepetitionConstraint)
    warnings.warn(
        f"'{DiscreteNoLabelDuplicatesConstraint.__name__}' is deprecated and will be "
        f"removed in a future version. Use '{DiscreteRepetitionConstraint.__name__}' "
        f"with '{flds.n_max_repetitions.alias}=1' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return DiscreteRepetitionConstraint(parameters=parameters, n_max_repetitions=1)


def DiscreteLinkedParametersConstraint(  # noqa: N802
    parameters: list[str],
) -> DiscreteRepetitionConstraint:
    """A :class:`DiscreteRepetitionConstraint` alias for backward compatibility."""  # noqa: D401
    import warnings

    flds = fields(DiscreteRepetitionConstraint)
    warnings.warn(
        f"'{DiscreteLinkedParametersConstraint.__name__}' is deprecated and will be "
        f"removed in a future version. Use '{DiscreteRepetitionConstraint.__name__}' "
        f"with '{flds.n_max_repetitions.alias}=len(parameters)-1' and "
        f"'{flds.exclude.alias}=True' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return DiscreteRepetitionConstraint(
        parameters=parameters,
        n_max_repetitions=len(parameters) - 1,
        exclude=True,
    )


# <<<<<<<<<< Deprecation


@define
class DiscreteDependenciesConstraint(DiscreteFilteringConstraint):
    """Constraint that specifies dependencies between parameters.

    For instance some parameters might only be relevant when another parameter has a
    certain value (e.g. parameter switch is 'on'). All dependencies must be declared in
    a single constraint.

    Examples:
        >>> df = pd.DataFrame({
        ...     "Switch": ["on", "off", "off"],
        ...     "Temp": [100, 200, 100],
        ... })
        >>> df
          Switch  Temp
        0     on   100
        1    off   200
        2    off   100
        >>> c = DiscreteDependenciesConstraint(
        ...     parameters=["Switch"],
        ...     conditions=[SubSelectionCondition(selection=["on"])],
        ...     affected_parameters=[["Temp"]],
        ... )
        >>> list(c.get_invalid(df))
        [2]
    """

    # object variables
    conditions: list[Condition] = field()
    """The list of individual conditions."""

    affected_parameters: list[list[str]] = field()
    """The parameters affected by the individual conditions."""

    # for internal use only
    permutation_invariant: bool = field(default=False, init=False)
    """Flag that indicates whether the affected parameters are permutation invariant.
    This should not be changed by the user but by other constraints using the class."""

    @affected_parameters.validator
    def _validate_affected_parameters(  # noqa: DOC101, DOC103
        self, _: Any, value: list[list[str]]
    ) -> None:
        """Validate the affected parameters.

        Raises:
            ValueError: If one set of affected parameters does not have exactly one
                condition.
        """
        if len(self.conditions) != len(value):
            raise ValueError(
                f"For the {self.__class__.__name__}, for each item in the "
                f"affected_parameters list you must provide exactly one condition in "
                f"the conditions list."
            )

    @property
    @override
    def _required_parameters(self) -> set[str]:
        """See base class."""
        params = set(self.parameters)
        for group in self.affected_parameters:
            params.update(group)
        return params

    @override
    def _get_matching_rows(self, df: pd.DataFrame, /) -> pd.Index:
        # Build an invariant indicator for each affected parameter: pair each value
        # with the value of the parameter it depends on. For rows where the dependency
        # condition is not met, use None as a sentinel so that all such rows with the
        # same dependency value appear identical, causing them to be detected as
        # duplicates. The indicator tuples are constructed directly without storing
        # any intermediate sentinel in the typed columns.
        censored_df = df.copy()
        for k, param in enumerate(self.parameters):
            invalid = ~self.conditions[k].evaluate(df[self.parameters[k]])
            for affected_param in self.affected_parameters[k]:
                censored_df[affected_param] = [
                    (None if inv else val, dep)
                    for val, dep, inv in zip(
                        censored_df[affected_param], censored_df[param], invalid
                    )
                ]

        # Merge the invariant indicator with all other parameters (i.e. neither the
        # affected nor the dependency-causing ones) and detect duplicates in that space.
        all_affected_params = [col for cols in self.affected_parameters for col in cols]
        other_params = (
            df.columns.drop(all_affected_params).drop(self.parameters).tolist()
        )
        invariant_indicator = censored_df[all_affected_params].apply(
            cast(Callable, frozenset)
            if self.permutation_invariant
            else cast(Callable, tuple),
            axis=1,
        )
        # Only include the other_params DataFrame if it is non-empty to avoid
        # pandas FutureWarning about concatenation with empty entries
        parts = [censored_df[other_params]] if other_params else []
        parts.append(invariant_indicator)
        df_eval = pd.concat(parts, axis=1)
        inds_good = df.index[~df_eval.duplicated(keep="first")]

        return inds_good

    def to_symmetries(self) -> tuple[DependencySymmetry, ...]:
        """Convert to :class:`~baybe.symmetries.dependency.DependencySymmetry` objects.

        Create one symmetry object per dependency relationship, i.e., per
        (parameter, condition, affected_parameters) triple.

        Returns:
            A tuple of dependency symmetries, one for each dependency in the
            constraint.
        """
        from baybe.symmetries.dependency import DependencySymmetry

        return tuple(
            DependencySymmetry(
                parameter_name=p,
                condition=c,
                affected_parameter_names=aps,
            )
            for p, c, aps in zip(
                self.parameters, self.conditions, self.affected_parameters, strict=True
            )
        )


@define
class DiscretePermutationInvarianceConstraint(DiscreteFilteringConstraint):
    """Constraint class for declaring that a set of parameters is permutation invariant.

    More precisely, this means that, ``(val_from_param1, val_from_param2)`` is
    equivalent to ``(val_from_param2, val_from_param1)``.

    *Note:* This constraint is evaluated during creation. In the future it might also be
    evaluated during modeling to make use of the invariance.

    Examples:
        >>> df = pd.DataFrame({"A": ["x", "y", "z"], "B": ["y", "x", "x"]})
        >>> df
           A  B
        0  x  y
        1  y  x
        2  z  x
        >>> c = DiscretePermutationInvarianceConstraint(parameters=["A", "B"])
        >>> list(c.get_invalid(df))
        [1]
    """

    # object variables
    dependencies: DiscreteDependenciesConstraint | None = field(default=None)
    """Dependencies connected with the invariant parameters."""

    @property
    @override
    def _required_parameters(self) -> set[str]:
        """See base class."""
        params = set(self.parameters)
        if self.dependencies:
            params.update(self.dependencies._required_parameters)
        return params

    @override
    def _can_evaluate(self, available: set[str], /) -> bool:
        # TODO: Potentially enable early filtering for all situations once
        #  sentinel/null values have been introduced instead of arbitrarily keeping one
        #  concrete row.
        # With exclude=True, all parameters must be present to identify the one
        # canonical kept row (due to keep='first' in the deduplication). This is to
        # make the result consistent with the kept row in the exclude=False case.
        if self.exclude:
            return self._required_parameters <= available
        # When dependencies are present, partial permutation dedup is unsafe:
        # the dependency logic changes which rows are permutation-equivalent
        # (inactive parameters become irrelevant), so removing permutation
        # duplicates before the dependency columns are available can discard
        # configurations that should have been kept as canonical representatives.
        if self.dependencies:
            return self._required_parameters <= available
        # Without dependencies, permutation dedup on a partial set is safe
        # during incremental construction: since new columns are added via
        # cross-product, rows that are permutation-equivalent on the available
        # subset will produce identical expansions.
        return len(available & set(self.parameters)) >= 2

    @override
    def _get_matching_rows(self, df: pd.DataFrame, /) -> pd.Index:
        cols = set(df.columns)
        params = [p for p in self.parameters if p in cols]

        # Merge a permutation invariant representation of all affected parameters with
        # the other parameters and indicate duplicates. This ensures that variation in
        # other parameters is also accounted for.
        other_params = df.columns.drop(params).tolist()
        frozen = df[params].apply(cast(Callable, frozenset), axis=1)
        parts = [df[other_params].copy(), frozen] if other_params else [frozen]
        df_eval = pd.concat(parts, axis=1)
        mask_canonical = ~df_eval.duplicated(keep="first")

        # Indices of canonical (non-duplicate) permutations
        inds_valid = df_eval.index[mask_canonical]

        # If there are dependencies connected to the invariant parameters evaluate them
        # here and remove resulting duplicates with a DependenciesConstraint
        if self.dependencies and self.dependencies._can_evaluate(set(df.columns)):
            self.dependencies.permutation_invariant = True
            inds_duplicate_independency_adjusted = self.dependencies.get_invalid(
                df.loc[inds_valid]
            )
            inds_valid = inds_valid.drop(inds_duplicate_independency_adjusted)

        return inds_valid

    def to_symmetry(self) -> PermutationSymmetry:
        """Convert to a :class:`~baybe.symmetries.permutation.PermutationSymmetry`.

        The constraint's parameters form the primary permutation group. If
        dependencies are attached, their parameters are added as an additional
        group that is permuted in lockstep.

        Returns:
            The corresponding permutation symmetry.
        """
        from baybe.symmetries.permutation import PermutationSymmetry

        groups = [self.parameters]
        if self.dependencies:
            groups.append(list(self.dependencies.parameters))
        return PermutationSymmetry(permutation_groups=groups)


@define
class DiscreteCustomConstraint(DiscreteFilteringConstraint):
    """Class for user-defined custom constraints."""

    # object variables
    validator: Callable[[pd.DataFrame], pd.Series] = field()
    """A user-defined function modeling the validation of the constraint. The expected
    return is a pandas series with Boolean entries True/False for search space elements
    you want to keep/remove."""

    @override
    def _get_matching_rows(self, df: pd.DataFrame, /) -> pd.Index:
        mask_good = self.validator(df[self.parameters])

        return df.index[mask_good]


@define
class DiscreteBatchConstraint(DiscreteConstraint):
    """Constraint ensuring recommendations in a batch share certain parameter values.

    When this constraint is active, the recommender internally subsets the
    candidate set (one subset for each unique value of the constrained
    parameter), obtains a full batch recommendation from each subset, and
    returns the batch with the highest joint acquisition value.

    This constraint is not supported by all recommenders. It is not applied during
    search space creation (all parameter values remain in the search space).

    Example:
        If parameter ``Temperature`` has values ``[50, 100, 150]`` and a batch of
        10 is requested, the recommender will generate three candidate batches
        (one all-50, one all-100, one all-150) and return the best one.

    Notes:
        This constraint can lead to overhead in the computation since optimization
        results in individual optimizations over several subsets. If there are
        multiple subset-generating constraints active, this can drastically increase
        the computational cost due to the combinatorial explosion.
    """

    def __attrs_post_init__(self):
        """Validate that exactly one parameter is specified."""
        if len(self.parameters) != 1:
            raise ValueError(
                f"'{self.__class__.__name__}' requires exactly one parameter, "
                f"but {len(self.parameters)} were provided: {self.parameters}."
            )

    def subset_masks(
        self, candidates_exp: pd.DataFrame, /
    ) -> list[npt.NDArray[np.bool_]]:
        """Return Boolean masks defining the subsets for this constraint.

        Each mask selects the rows in ``candidates_exp`` that belong to one
        subset, i.e. share the same value for the constrained parameter.

        Args:
            candidates_exp: The experimental representation of candidate points.

        Returns:
            A list of Boolean masks, one per unique value of the constrained
            parameter.
        """
        param = self.parameters[0]
        return [
            (candidates_exp[param] == v).values for v in candidates_exp[param].unique()
        ]


@define
class DiscreteCardinalityConstraint(CardinalityConstraint, DiscreteFilteringConstraint):
    """Class for discrete cardinality constraints.

    Examples:
        >>> df = pd.DataFrame({"A": [0.0, 1.0, 1.0], "B": [0.0, 0.0, 1.0]})
        >>> df
             A    B
        0  0.0  0.0
        1  1.0  0.0
        2  1.0  1.0
        >>> c = DiscreteCardinalityConstraint(
        ...     parameters=["A", "B"], max_cardinality=1
        ... )
        >>> list(c.get_invalid(df))
        [2]
    """

    # Class variables
    numerical_only: ClassVar[bool] = True
    # See base class.

    @override
    def _can_evaluate(self, available: set[str], /) -> bool:
        # With exclude=True we keep the rows violating the bounds. Whether a row
        # satisfies the bounds can still change as later columns are added (the
        # nonzero count only grows), so all parameters must be present first.
        if self.exclude:
            return self._required_parameters <= available
        # The max-cardinality check is safe on any non-empty subset: the
        # nonzero count can only increase as more parameters are added.
        return bool(available & set(self.parameters))

    @override
    def _get_matching_rows(self, df: pd.DataFrame, /) -> pd.Index:
        params = [p for p in self.parameters if p in set(df.columns)]
        all_present = len(params) == len(self.parameters)

        non_zeros = (df[params] != 0.0).sum(axis=1)
        # The max_cardinality check is safe on a partial subset: the nonzero
        # count can only increase as more parameters are added.
        mask_good = non_zeros <= self.max_cardinality
        # The min_cardinality check can only be applied when all parameters
        # are present, since missing parameters could still add nonzero values.
        if all_present:
            mask_good &= non_zeros >= self.min_cardinality
        return df.index[mask_good]


# Filtering constraints are approximately ordered according to increasing computational
# effort to minimize total time in their sequential application
DISCRETE_CONSTRAINTS_FILTERING_ORDER = (
    DiscreteSelectionConstraint,
    DiscreteRepetitionConstraint,
    DiscreteLinearConstraint,
    DiscreteProductConstraint,
    DiscreteCardinalityConstraint,
    DiscreteCustomConstraint,
    DiscretePermutationInvarianceConstraint,
    DiscreteDependenciesConstraint,
)

# Prevent (de-)serialization of custom constraints
converter.register_unstructure_hook(DiscreteCustomConstraint, block_serialization_hook)
converter.register_structure_hook(DiscreteCustomConstraint, block_deserialization_hook)


# >>>>>>>>>> Deprecation
def _unstructure_product_constraint(obj: DiscreteProductConstraint) -> dict:
    """Unstructure hook that excludes the deprecated ``condition`` field."""
    result = cattrs.gen.make_dict_unstructure_fn(DiscreteProductConstraint, converter)(
        obj
    )
    result.pop("condition", None)
    return result


converter.register_unstructure_hook(
    DiscreteProductConstraint, _unstructure_product_constraint
)


def _unpack_condition_payload(val: dict) -> None:
    """Unpack a legacy nested ``condition`` dict into top-level fields.

    Mutates *val* in place: extracts ``threshold`` → ``rhs``,
    ``operator`` → ``operator``, and (optionally) ``tolerance`` → ``tolerance``
    from the nested ``condition`` sub-dict, then removes the ``condition`` key.

    Args:
        val: The serialized constraint dict to transform.
    """
    cond = val.pop("condition", None)
    if cond is None:
        return
    if isinstance(cond, dict):
        cond = dict(cond)
        # Remove the type discriminator if present
        cond.pop("type", None)
        val["operator"] = cond["operator"]
        val["rhs"] = cond["threshold"]
        tol = cond.get("tolerance")
        if tol is not None:
            val["tolerance"] = tol


def _structure_constraint_compat(val: dict, cls: type) -> Constraint:
    """Structure hook that redirects legacy constraint type names."""
    val = dict(val)  # copy before mutating
    type_ = val.get(_TYPE_FIELD)
    if type_ == "DiscreteExcludeConstraint":
        val[_TYPE_FIELD] = "DiscreteSelectionConstraint"
        val["exclude"] = True
    elif type_ == "DiscreteNoLabelDuplicatesConstraint":
        val[_TYPE_FIELD] = "DiscreteRepetitionConstraint"
        val["n_max_repetitions"] = 1
    elif type_ == "DiscreteLinkedParametersConstraint":
        val[_TYPE_FIELD] = "DiscreteRepetitionConstraint"
        if (params := val.get("parameters")) is not None and len(params) >= 2:
            val["n_max_repetitions"] = len(params) - 1
        val["exclude"] = True
    elif type_ == "DiscreteSumConstraint":
        _unpack_condition_payload(val)
        val[_TYPE_FIELD] = "DiscreteLinearConstraint"
    elif type_ == "DiscreteProductConstraint" and "condition" in val:
        _unpack_condition_payload(val)
    return make_base_structure_hook(cls)(val, cls)


converter.register_structure_hook_func(
    lambda cls: issubclass(cls, Constraint) and is_abstract(cls),
    _structure_constraint_compat,
)
# <<<<<<<<<< Deprecation

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
