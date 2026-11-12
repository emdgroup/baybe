"""Discrete constraints."""

from __future__ import annotations

import gc
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from functools import reduce
from inspect import Parameter, Signature, signature
from typing import TYPE_CHECKING, Any, ClassVar, get_type_hints, overload

import cattrs
import narwhals.stable.v2 as nw
from attrs import define, field, fields
from attrs.converters import optional as optional_c
from attrs.validators import (
    deep_iterable,
    ge,
    gt,
    in_,
    instance_of,
    is_callable,
    min_len,
)
from attrs.validators import optional as optional_v
from narwhals.stable.v2.typing import IntoDataFrame, IntoSeries
from typing_extensions import override

from baybe.constraints.base import (
    CardinalityConstraint,
    Constraint,
    DiscreteConstraint,
    DiscreteFilteringConstraint,
)
from baybe.constraints.conditions import (
    _LOGIC_COMBINERS,
    _THRESHOLD_OPERATORS,
    _TOLERANCE_OPERATORS,
    Condition,
    SubSelectionCondition,  # noqa: F401 (used in doctests)
    ThresholdCondition,
    ThresholdOperator,
)
from baybe.serialization import (
    block_deserialization_hook,
    block_serialization_hook,
    converter,
)
from baybe.serialization.core import _TYPE_FIELD, make_base_structure_hook
from baybe.utils.basic import to_tuple, validate_all_instance
from baybe.utils.boolean import is_abstract
from baybe.utils.validation import finite_float

if TYPE_CHECKING:
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


@define(frozen=True)
class DiscreteSelectionConstraint(DiscreteFilteringConstraint):
    """Class for filtering search space entries based on conditions.

    Examples:
        >>> import pandas as pd
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
        >>> constraint = DiscreteSelectionConstraint(
        ...     parameters=["Solvent", "Temp"],
        ...     conditions=[
        ...         SubSelectionCondition(selection=["Hexane"]),
        ...         ThresholdCondition(threshold=100.0, operator=">="),
        ...     ],
        ...     exclude=True,
        ... )
        >>> constraint(df)
          Solvent   Temp
        0   Water   80.0
        1   Water  120.0
        2  Hexane   80.0
    """

    conditions: tuple[Condition, ...] = field(
        converter=to_tuple,
        validator=[
            min_len(1),
            deep_iterable(member_validator=instance_of(Condition)),
        ],
    )
    """The individual conditions."""

    combiner: str = field(default="AND", validator=in_(_LOGIC_COMBINERS))
    """Operator encoding how to combine the individual conditions."""

    @conditions.validator
    def _validate_conditions(self, _: Any, conditions: tuple[Condition, ...]) -> None:
        if len(conditions) != len(self.parameters):
            raise ValueError(
                f"'{fields(type(self)).conditions.alias}' must contain exactly one "
                f"condition per parameter, but got {len(conditions)} conditions for "
                f"{len(self.parameters)} parameters."
            )

    @override
    def _can_evaluate(self, available: set[str], /) -> bool:
        # A row can be dropped early during incremental construction only if a
        # later column can never change that decision. This holds exactly for:
        # - AND with exclude=False: once a present condition fails, the row is
        #   permanently dropped (an AND failure stays).
        # - OR with exclude=True: once a present condition holds, the row is
        #   permanently marked for removal (an OR match stays).
        # For XOR, the combined result can flip as further operands arrive, so
        # all parameters must be present before evaluating. All other cases must
        # likewise wait for every parameter.
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
    def _to_matching_expr(self, schema: Mapping[str, Any], /) -> nw.Expr:
        satisfied = [
            cond._to_expr(nw.col(param), schema[param])
            for param, cond in zip(self.parameters, self.conditions, strict=True)
            if param in schema
        ]
        return reduce(_LOGIC_COMBINERS[self.combiner], satisfied)


def _make_condition(
    operator: ThresholdOperator, rhs: float, tolerance: float | None
) -> ThresholdCondition:
    """Create a threshold condition, using its default when tolerance is omitted.

    Args:
        operator: The comparison operator.
        rhs: The comparison threshold.
        tolerance: The explicit tolerance, or ``None`` to use the default.

    Returns:
        The threshold condition.
    """
    kwargs = {} if tolerance is None else {"tolerance": tolerance}
    return ThresholdCondition(threshold=rhs, operator=operator, **kwargs)


@define(frozen=True)
class DiscreteLinearConstraint(DiscreteFilteringConstraint):
    """Class for modeling linear (weighted-sum) constraints on discrete parameters.

    The constraint compares the sum of the specified parameters, optionally weighted by
    :paramref:`DiscreteLinearConstraint.coefficients`, against
    :paramref:`DiscreteLinearConstraint.rhs` using
    :paramref:`DiscreteLinearConstraint.operator`.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"A": [1.0, 3.0, 5.0], "B": [2.0, 1.0, 3.0]})
        >>> df
             A    B
        0  1.0  2.0
        1  3.0  1.0
        2  5.0  3.0
        >>> constraint = DiscreteLinearConstraint(
        ...     parameters=["A", "B"],
        ...     operator="<=",
        ...     rhs=5.0,
        ... )
        >>> constraint(df)
             A    B
        0  1.0  2.0
        1  3.0  1.0

        With coefficients, the weighted sum is checked instead:

        >>> constraint = DiscreteLinearConstraint(
        ...     parameters=["A", "B"],
        ...     coefficients=(2.0, 1.0),
        ...     operator="<=",
        ...     rhs=5.0,
        ... )
        >>> constraint(df)
             A    B
        0  1.0  2.0
    """

    # IMPROVE: Look-ahead filtering would be possible if parameter
    # value ranges (min/max) were available to the constraint, allowing
    # bound-based pruning of partial sums before all parameters are
    # present. This could be expressed via a _can_evaluate override.

    numerical_only: ClassVar[bool] = True
    # See base class.

    operator: ThresholdOperator = field(validator=in_(_THRESHOLD_OPERATORS))
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
        default=None,
        converter=optional_c(float),
        validator=optional_v([finite_float, gt(0)]),
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
                f"There must be exactly one coefficient per parameter, but got "
                f"{len(coefficients)} coefficients for {len(self.parameters)} "
                f"parameters."
            )
        if any(c == 0.0 for c in coefficients):
            raise ValueError("All coefficients must be non-zero.")

    @tolerance.validator
    def _validate_tolerance(  # noqa: DOC101, DOC103
        self, attribute: Any, value: float | None
    ) -> None:
        """Validate the tolerance.

        Raises:
            ValueError: If a tolerance is provided for a non-tolerance operator.
        """
        if self.operator not in _TOLERANCE_OPERATORS and value is not None:
            raise ValueError(
                f"Setting the '{attribute.alias}' is only valid with the following "
                f"operators: {_TOLERANCE_OPERATORS}, but got operator "
                f"'{self.operator}'."
            )

    @override
    def _to_matching_expr(self, schema: Mapping[str, Any], /) -> nw.Expr:
        validate_all_instance(
            [schema[p] for p in self.parameters], nw.dtypes.NumericType
        )
        weighted = [
            nw.col(p) * c
            for p, c in zip(self.parameters, self.coefficients, strict=True)
        ]
        condition = _make_condition(self.operator, self.rhs, self.tolerance)
        return condition._to_expr(nw.sum_horizontal(weighted), nw.dtypes.NumericType())


@define(init=False, frozen=True)
class DiscreteProductConstraint(DiscreteFilteringConstraint):
    """Class for modeling product constraints on discrete parameters.

    The constraint compares the product of the specified parameters against
    :paramref:`DiscreteProductConstraint.rhs` using
    :paramref:`DiscreteProductConstraint.operator`.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"A": [2.0, 3.0, 5.0], "B": [3.0, 2.0, 2.0]})
        >>> df
             A    B
        0  2.0  3.0
        1  3.0  2.0
        2  5.0  2.0
        >>> constraint = DiscreteProductConstraint(
        ...     parameters=["A", "B"],
        ...     operator="<=",
        ...     rhs=8.0,
        ... )
        >>> constraint(df)
             A    B
        0  2.0  3.0
        1  3.0  2.0
    """

    # IMPROVE: Look-ahead filtering would be possible if parameter
    # value ranges (min/max) were available to the constraint, allowing
    # bound-based pruning of partial products before all parameters are
    # present. This could be expressed via a _can_evaluate override.

    numerical_only: ClassVar[bool] = True
    # See base class.

    __signature__: ClassVar[Signature]
    """The modern constructor signature exposed to introspection tools."""

    operator: ThresholdOperator = field(validator=in_(_THRESHOLD_OPERATORS))
    """The comparison operator (e.g. ``"="``, ``">="``, ``"<"``)."""

    rhs: float = field(default=0.0, converter=float, validator=finite_float)
    """Right-hand side value of the comparison."""

    tolerance: float | None = field(
        default=None,
        converter=optional_c(float),
        validator=optional_v([finite_float, gt(0)]),
    )
    """Numerical tolerance for equality/inequality operators that support it.

    Only applicable when ``operator`` is one of ``"="``, ``"=="``, ``"!="``.
    Set to a reasonable default when left as ``None``."""

    @overload
    def __init__(  # noqa: DOC101, DOC103 (overload; attributes document inputs)
        self,
        parameters: list[str],
        operator: ThresholdOperator,
        rhs: float = 0.0,
        tolerance: float | None = None,
        *,
        exclude: bool = False,
    ) -> None: ...

    @overload
    def __init__(  # noqa: DOC101, DOC103 (overload; attributes document inputs)
        self,
        parameters: list[str],
        condition: ThresholdCondition,
        *,
        exclude: bool = False,
    ) -> None: ...

    # The public overloads and attrs fields document the compatibility initializer.
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: DOC101, DOC103, DOC501
        # Normalize before attrs converters and validators see the arguments.
        import warnings

        flds = fields(type(self))
        supplied = dict(kwargs)
        condition = supplied.pop("condition", None)
        bound = _product_signature.bind_partial(*args, **supplied)
        if "operator" not in kwargs and isinstance(
            bound.arguments.get("operator"), ThresholdCondition
        ):
            condition = bound.arguments["operator"]

        if condition is not None:
            if {"operator", "rhs", "tolerance"} & kwargs.keys():
                raise ValueError(
                    "Cannot specify both 'condition' and modern comparison arguments."
                )
            legacy = _legacy_product_signature.bind(*args, **kwargs)
            values = dict(legacy.arguments)
            condition = values.pop("condition")
            values.update(
                operator=condition.operator,
                rhs=condition.threshold,
                tolerance=condition.tolerance,
            )
        else:
            values = dict(_product_signature.bind(*args, **supplied).arguments)

        self.__attrs_init__(**values)
        if condition is not None:
            warnings.warn(
                f"Passing 'condition' to '{self.__class__.__name__}' is "
                f"deprecated and will be removed in a future version. Use "
                f"'{flds.operator.alias}' and '{flds.rhs.alias}' (and optionally "
                f"'{flds.tolerance.alias}') instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    @tolerance.validator
    def _validate_tolerance(  # noqa: DOC101, DOC103
        self, attribute: Any, value: float | None
    ) -> None:
        """Validate compatibility between the operator and tolerance.

        Raises:
            ValueError: If a tolerance is provided for a non-tolerance operator.
        """
        if self.operator not in _TOLERANCE_OPERATORS and value is not None:
            raise ValueError(
                f"Setting the '{attribute.alias}' is only valid with the "
                f"following operators: {_TOLERANCE_OPERATORS}, but got "
                f"operator '{self.operator}'."
            )

    @override
    def _to_matching_expr(self, schema: Mapping[str, Any], /) -> nw.Expr:
        validate_all_instance(
            [schema[p] for p in self.parameters], nw.dtypes.NumericType
        )
        factors = [nw.col(p) for p in self.parameters]
        product = reduce(lambda left, right: left * right, factors)
        condition = _make_condition(self.operator, self.rhs, self.tolerance)
        return condition._to_expr(product, nw.dtypes.NumericType())


# >>>>>>>>>> Deprecation
# Derive the public signature from attrs while retaining the legacy input adapter.
_product_signature = signature(DiscreteProductConstraint.__attrs_init__).replace(
    parameters=[
        p
        for p in signature(DiscreteProductConstraint.__attrs_init__).parameters.values()
        if p.name != "self"
    ]
)
_legacy_product_signature = _product_signature.replace(
    parameters=[
        _product_signature.parameters["parameters"],
        Parameter("condition", Parameter.POSITIONAL_OR_KEYWORD),
        _product_signature.parameters["exclude"],
    ]
)
DiscreteProductConstraint.__signature__ = _product_signature


def DiscreteSumConstraint(  # noqa: N802
    parameters: list[str],
    condition: ThresholdCondition,
    coefficients: Sequence[float] | None = None,
) -> DiscreteLinearConstraint:
    """A :class:`DiscreteLinearConstraint` alias for backward compatibility."""  # noqa: D401
    import warnings

    warnings.warn(
        f"'{DiscreteSumConstraint.__name__}' is deprecated and will be removed "
        f"in a future version. Use '{DiscreteLinearConstraint.__name__}' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs: dict[str, Any] = (
        {} if coefficients is None else {"coefficients": coefficients}
    )
    return DiscreteLinearConstraint(
        parameters,
        operator=condition.operator,
        rhs=condition.threshold,
        tolerance=condition.tolerance,
        **kwargs,
    )


# <<<<<<<<<< Deprecation


@define(frozen=True)
class DiscreteRepetitionLimitConstraint(DiscreteFilteringConstraint):
    """Class for constraining value repetition across parameters.

    Keeps only rows where no single value appears more than a specified number of
    times across the specified parameters.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"A": ["x", "y", "x"], "B": ["y", "x", "x"]})
        >>> df
           A  B
        0  x  y
        1  y  x
        2  x  x

        Upper bound: row 2 has "x" twice, violating ``n_max_repetitions=1``:

        >>> constraint = DiscreteRepetitionLimitConstraint(
        ...     parameters=["A", "B"], n_max_repetitions=1
        ... )
        >>> constraint(df)
           A  B
        0  x  y
        1  y  x

        With ``exclude=True``, the logic inverts and only repeated rows are kept:

        >>> constraint = DiscreteRepetitionLimitConstraint(
        ...     parameters=["A", "B"], n_max_repetitions=1, exclude=True
        ... )
        >>> constraint(df)
           A  B
        2  x  x
    """

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
    def _to_matching_expr(self, schema: Mapping[str, Any], /) -> nw.Expr:
        params = [p for p in self.parameters if p in schema]

        def _safe_eq(left: str, right: str) -> nw.Expr:
            """Compare two columns while handling dtype differences safely."""
            left_dtype, right_dtype = schema[left], schema[right]
            if left_dtype != right_dtype and not (
                isinstance(left_dtype, nw.dtypes.NumericType)
                and isinstance(right_dtype, nw.dtypes.NumericType)
            ):
                return nw.lit(False)

            return nw.col(left) == nw.col(right)

        counts = [
            nw.sum_horizontal(_safe_eq(left, right).cast(nw.UInt16) for right in params)
            for left in params
        ]
        max_multiplicity = nw.max_horizontal(counts)
        if self.exclude:
            # Missing parameters can only increase the multiplicity. Add them
            # to obtain an upper bound, so that only rows guaranteed to satisfy the
            # limit can be removed by the constraint inversion.
            n_missing = len(self.parameters) - len(params)
            matching = max_multiplicity + n_missing <= self.n_max_repetitions
        else:
            # The partial count is a lower bound: rows already exceeding the limit are
            # definitely invalid; others are re-evaluated later.
            matching = max_multiplicity <= self.n_max_repetitions

        return matching


# >>>>>>>>>> Deprecation
def DiscreteNoLabelDuplicatesConstraint(  # noqa: N802
    parameters: list[str],
) -> DiscreteRepetitionLimitConstraint:
    """A :class:`DiscreteRepetitionLimitConstraint` alias for backward compatibility."""  # noqa: D401
    import warnings

    flds = fields(DiscreteRepetitionLimitConstraint)
    warnings.warn(
        f"'{DiscreteNoLabelDuplicatesConstraint.__name__}' is deprecated and will be "
        f"removed in a future version. Use "
        f"'{DiscreteRepetitionLimitConstraint.__name__}' "
        f"with '{flds.n_max_repetitions.alias}=1' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return DiscreteRepetitionLimitConstraint(parameters=parameters, n_max_repetitions=1)


def DiscreteLinkedParametersConstraint(  # noqa: N802
    parameters: list[str],
) -> DiscreteRepetitionLimitConstraint:
    """A :class:`DiscreteRepetitionLimitConstraint` alias for backward compatibility."""  # noqa: D401
    import warnings

    flds = fields(DiscreteRepetitionLimitConstraint)
    warnings.warn(
        f"'{DiscreteLinkedParametersConstraint.__name__}' is deprecated and will be "
        f"removed in a future version. Use "
        f"'{DiscreteRepetitionLimitConstraint.__name__}' "
        f"with '{flds.n_max_repetitions.alias}=len(parameters)-1' and "
        f"'{flds.exclude.alias}=True' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return DiscreteRepetitionLimitConstraint(
        parameters=parameters,
        n_max_repetitions=len(parameters) - 1,
        exclude=True,
    )


# <<<<<<<<<< Deprecation


@define(frozen=True)
class DiscreteDependenciesConstraint(DiscreteFilteringConstraint):
    """Constraint that specifies dependencies between parameters.

    For instance some parameters might only be relevant when another parameter has a
    certain value (e.g. parameter switch is 'on'). All dependencies must be declared in
    a single constraint.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "Switch": ["on", "off", "off"],
        ...     "Temp": [100, 200, 100],
        ... })
        >>> df
          Switch  Temp
        0     on   100
        1    off   200
        2    off   100
        >>> constraint = DiscreteDependenciesConstraint(
        ...     parameters=["Switch"],
        ...     conditions=[SubSelectionCondition(selection=["on"])],
        ...     affected_parameters=[["Temp"]],
        ... )
        >>> constraint(df)
          Switch  Temp
        0     on   100
        1    off   200
    """

    conditions: tuple[Condition, ...] = field()
    """The list of individual conditions."""

    affected_parameters: tuple[tuple[str, ...], ...] = field()
    """The parameters affected by the individual conditions."""

    # for internal use only
    permutation_invariant: bool = field(default=False, init=False)
    """Flag that indicates whether the affected parameters are permutation invariant.
    This should not be changed by the user but by other constraints using the class."""

    @affected_parameters.validator
    def _validate_affected_parameters(  # noqa: DOC101, DOC103
        self, _: Any, value: tuple[tuple[str, ...], ...]
    ) -> None:
        """Validate the affected parameters.

        Raises:
            ValueError: If one set of affected parameters does not have exactly one
                condition.
        """
        if len(self.conditions) != len(value):
            raise ValueError(
                f"Each group of affected parameters needs exactly one condition, "
                f"but got {len(value)} groups and {len(self.conditions)} conditions."
            )

    @property
    @override
    def _required_parameters(self) -> set[str]:
        """See base class."""
        params = set(self.parameters)
        for group in self.affected_parameters:
            params.update(group)
        return params

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


@define(frozen=True)
class DiscretePermutationInvarianceConstraint(DiscreteFilteringConstraint):
    """Constraint class for declaring that a set of parameters is permutation invariant.

    More precisely, this means that, ``(val_from_param1, val_from_param2)`` is
    equivalent to ``(val_from_param2, val_from_param1)``.

    *Note:* This constraint is evaluated during creation. In the future it might also be
    evaluated during modeling to make use of the invariance.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"A": ["x", "y", "z"], "B": ["y", "x", "x"]})
        >>> df
           A  B
        0  x  y
        1  y  x
        2  z  x
        >>> constraint = DiscretePermutationInvarianceConstraint(parameters=["A", "B"])
        >>> constraint(df)
           A  B
        0  x  y
        2  z  x
    """

    dependencies: DiscreteDependenciesConstraint | None = field(default=None)
    """Dependencies connected with the invariant parameters."""

    @dependencies.validator
    def _validate_dependencies(  # noqa: DOC101, DOC103
        self, _: Any, value: DiscreteDependenciesConstraint | None
    ) -> None:
        """Validate the dependencies constraint.

        Raises:
            ValueError: If the dependencies constraint uses ``exclude=True``.
        """
        if value is not None and value.exclude:
            raise ValueError(
                "The dependencies of a permutation invariance constraint cannot be "
                "inverted."
            )

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
            groups.append(self.dependencies.parameters)
        return PermutationSymmetry(permutation_groups=groups)


@define(frozen=True)
class DiscreteCustomConstraint(DiscreteFilteringConstraint):
    """Class for user-defined custom constraints."""

    validator: Callable[[IntoDataFrame], IntoSeries] = field(validator=is_callable())
    """A user-defined function that receives the selected parameters as a native
    dataframe and returns a Boolean series of the same backend."""

    @override
    def _to_matching_mask(self, df: nw.DataFrame, /) -> nw.Series:
        native_df = df.select(self.parameters).to_native()
        native_mask = self.validator(native_df)
        return nw.from_native(native_mask, series_only=True)


@define(frozen=True)
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


@define
class DiscreteCardinalityConstraint(  # pyrefly: ignore[invalid-inheritance]
    CardinalityConstraint, DiscreteFilteringConstraint
):
    """Class for discrete cardinality constraints.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"A": [0.0, 1.0, 1.0], "B": [0.0, 0.0, 1.0]})
        >>> df
             A    B
        0  0.0  0.0
        1  1.0  0.0
        2  1.0  1.0
        >>> constraint = DiscreteCardinalityConstraint(
        ...     parameters=["A", "B"], max_cardinality=1
        ... )
        >>> constraint(df)
             A    B
        0  0.0  0.0
        1  1.0  0.0
    """

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
    def _to_matching_expr(self, schema: Mapping[str, Any], /) -> nw.Expr:
        parameters = [p for p in self.parameters if p in schema]
        validate_all_instance([schema[p] for p in parameters], nw.dtypes.NumericType)
        n_active = nw.sum_horizontal(
            (nw.col(p) != 0.0).cast(nw.UInt16) for p in parameters
        )

        # The upper bound is safe on partial frames: the nonzero count can only grow
        # as missing parameters are added.
        matching = n_active <= self.max_cardinality

        # The lower bound can only be checked once all parameters are available;
        # a partial row may still reach the minimum cardinality later.
        if set(self.parameters) <= set(schema):
            matching = matching & (n_active >= self.min_cardinality)

        return matching


# Filtering constraints are approximately ordered according to increasing computational
# effort to minimize total time in their sequential application
DISCRETE_CONSTRAINTS_FILTERING_ORDER = (
    DiscreteSelectionConstraint,
    DiscreteRepetitionLimitConstraint,
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
_product_structure_hook = converter.get_structure_hook(DiscreteProductConstraint)


def _structure_product_constraint(val: dict, cls: type) -> DiscreteProductConstraint:
    """Route legacy Product input through its warning-emitting constructor.

    Args:
        val: The serialized constraint.
        cls: The requested concrete class.

    Returns:
        The deserialized Product constraint.
    """
    val = dict(val)
    if val.get("condition") is not None:
        # Let the normal hook reject mismatching type tags.
        if val.get(_TYPE_FIELD, cls.__name__) != cls.__name__:
            return _product_structure_hook(val, cls)
        val.pop(_TYPE_FIELD, None)
        val["condition"] = converter.structure(
            deepcopy(val["condition"]), ThresholdCondition
        )
        val["parameters"] = converter.structure(val["parameters"], list[str])
        return cls(**val)
    return _product_structure_hook(val, cls)


converter.register_structure_hook(
    DiscreteProductConstraint, _structure_product_constraint
)


def _structure_constraint_compat(val: dict, cls: type) -> Constraint:
    """Structure legacy constraints through their compatibility constructors.

    Args:
        val: The serialized constraint.
        cls: The requested abstract class.

    Returns:
        The deserialized constraint.

    Raises:
        TypeError: If the legacy replacement is incompatible with the requested class.
    """
    val = dict(val)  # copy before mutating
    type_ = val.get(_TYPE_FIELD)
    factories: dict[str, Callable[..., Constraint]] = {
        factory.__name__: factory
        for factory in (
            DiscreteExcludeConstraint,
            DiscreteNoLabelDuplicatesConstraint,
            DiscreteLinkedParametersConstraint,
            DiscreteSumConstraint,
        )
    }
    if isinstance(type_, str) and (factory := factories.get(type_)):
        hints = get_type_hints(factory)
        if not issubclass(hints["return"], cls):
            raise TypeError(f"'{type_}' is not compatible with '{cls.__name__}'.")
        val.pop(_TYPE_FIELD)
        bound = signature(factory).bind(**val)
        arguments = {
            name: converter.structure(deepcopy(value), hints[name])
            for name, value in bound.arguments.items()
        }
        return factory(**arguments)
    return make_base_structure_hook(cls)(val, cls)


converter.register_structure_hook_func(
    lambda cls: issubclass(cls, Constraint) and is_abstract(cls),
    _structure_constraint_compat,
)
# <<<<<<<<<< Deprecation

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
