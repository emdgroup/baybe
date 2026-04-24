"""Tests comparing naive vs incremental constrained Cartesian product construction."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from baybe.constraints import (
    DiscreteCardinalityConstraint,
    DiscreteDependenciesConstraint,
    DiscreteExcludeConstraint,
    DiscreteLinkedParametersConstraint,
    DiscreteNoLabelDuplicatesConstraint,
    DiscretePermutationInvarianceConstraint,
    DiscreteSumConstraint,
    SubSelectionCondition,
    ThresholdCondition,
)
from baybe.constraints.base import DiscreteConstraint
from baybe.parameters import CategoricalParameter, NumericalDiscreteParameter
from baybe.parameters.base import DiscreteParameter
from baybe.searchspace.utils import (
    _apply_constraint_filter_pandas,
    parameter_cartesian_prod_pandas,
    parameter_cartesian_prod_pandas_constrained,
)


def _no_constraints_scenario() -> tuple[
    Sequence[DiscreteParameter], Sequence[DiscreteConstraint]
]:
    params = [
        CategoricalParameter(name="A", values=["a1", "a2"]),
        CategoricalParameter(name="B", values=["b1", "b2", "b3"]),
    ]
    return params, []


def _no_label_duplicates_scenario() -> tuple[
    Sequence[DiscreteParameter], Sequence[DiscreteConstraint]
]:
    values = ["x", "y", "z", "w"]
    params = [CategoricalParameter(name=f"P{i}", values=values) for i in range(4)]
    constraints = [
        DiscreteNoLabelDuplicatesConstraint(parameters=[p.name for p in params])
    ]
    return params, constraints


def _linked_parameters_scenario() -> tuple[
    Sequence[DiscreteParameter], Sequence[DiscreteConstraint]
]:
    values = ["a", "b", "c"]
    params = [CategoricalParameter(name=f"P{i}", values=values) for i in range(3)]
    constraints = [
        DiscreteLinkedParametersConstraint(parameters=[p.name for p in params])
    ]
    return params, constraints


def _exclude_scenario(
    combiner: str,
) -> tuple[Sequence[DiscreteParameter], Sequence[DiscreteConstraint]]:
    params = [
        CategoricalParameter(name="A", values=["a1", "a2", "a3"]),
        CategoricalParameter(name="B", values=["b1", "b2", "b3"]),
        CategoricalParameter(name="C", values=["c1", "c2", "c3"]),
    ]
    constraints = [
        DiscreteExcludeConstraint(
            parameters=["A", "B"],
            conditions=[
                SubSelectionCondition(selection=["a1"]),
                SubSelectionCondition(selection=["b1"]),
            ],
            combiner=combiner,
        )
    ]
    return params, constraints


def _cardinality_scenario() -> tuple[
    Sequence[DiscreteParameter], Sequence[DiscreteConstraint]
]:
    params = [
        NumericalDiscreteParameter(name=f"P{i}", values=[0.0, 1.0, 2.0])
        for i in range(4)
    ]
    constraints = [
        DiscreteCardinalityConstraint(
            parameters=[p.name for p in params],
            min_cardinality=1,
            max_cardinality=2,
        )
    ]
    return params, constraints


def _sum_scenario() -> tuple[Sequence[DiscreteParameter], Sequence[DiscreteConstraint]]:
    params = [
        NumericalDiscreteParameter(name=f"P{i}", values=[0.0, 25.0, 50.0, 75.0, 100.0])
        for i in range(3)
    ]
    constraints = [
        DiscreteSumConstraint(
            parameters=[p.name for p in params],
            condition=ThresholdCondition(threshold=100, operator="=", tolerance=0.1),
        )
    ]
    return params, constraints


def _dependencies_scenario() -> tuple[
    Sequence[DiscreteParameter], Sequence[DiscreteConstraint]
]:
    params = [
        CategoricalParameter(name="Switch", values=["on", "off"]),
        CategoricalParameter(name="Label", values=["a", "b", "c"]),
        NumericalDiscreteParameter(name="Amount", values=[1.0, 2.0, 3.0]),
    ]
    constraints = [
        DiscreteDependenciesConstraint(
            parameters=["Switch"],
            conditions=[SubSelectionCondition(selection=["on"])],
            affected_parameters=[["Label"]],
        )
    ]
    return params, constraints


def _permutation_invariance_scenario() -> tuple[
    Sequence[DiscreteParameter], Sequence[DiscreteConstraint]
]:
    values = ["a", "b", "c", "d"]
    params = [CategoricalParameter(name=f"P{i}", values=values) for i in range(3)]
    constraints = [
        DiscretePermutationInvarianceConstraint(parameters=[p.name for p in params])
    ]
    return params, constraints


def _permutation_invariance_with_dependencies_scenario() -> tuple[
    Sequence[DiscreteParameter], Sequence[DiscreteConstraint]
]:
    solvents = ["water", "ethanol", "methanol", "acetone"]
    labels = [
        CategoricalParameter(name=f"Slot{i}_Label", values=solvents) for i in range(3)
    ]
    amounts = [
        NumericalDiscreteParameter(
            name=f"Slot{i}_Amount", values=list(np.linspace(0, 100, 5))
        )
        for i in range(3)
    ]
    params = labels + amounts
    label_names = [lbl.name for lbl in labels]
    amount_names = [amt.name for amt in amounts]

    constraints = [
        DiscretePermutationInvarianceConstraint(
            parameters=label_names,
            dependencies=DiscreteDependenciesConstraint(
                parameters=amount_names,
                conditions=[
                    ThresholdCondition(threshold=0.0, operator=">"),
                    ThresholdCondition(threshold=0.0, operator=">"),
                    ThresholdCondition(threshold=0.0, operator=">"),
                ],
                affected_parameters=[[n] for n in label_names],
            ),
        ),
        DiscreteSumConstraint(
            parameters=amount_names,
            condition=ThresholdCondition(threshold=100, operator="=", tolerance=0.1),
        ),
        DiscreteNoLabelDuplicatesConstraint(parameters=label_names),
    ]
    return params, constraints


def _mixed_scenario() -> tuple[
    Sequence[DiscreteParameter], Sequence[DiscreteConstraint]
]:
    params = [
        CategoricalParameter(name="Cat1", values=["a", "b", "c"]),
        CategoricalParameter(name="Cat2", values=["a", "b", "c"]),
        CategoricalParameter(name="Cat3", values=["a", "b", "c"]),
        NumericalDiscreteParameter(name="Num1", values=[0.0, 50.0, 100.0]),
        NumericalDiscreteParameter(name="Num2", values=[0.0, 50.0, 100.0]),
    ]
    constraints = [
        DiscreteNoLabelDuplicatesConstraint(parameters=["Cat1", "Cat2", "Cat3"]),
        DiscreteSumConstraint(
            parameters=["Num1", "Num2"],
            condition=ThresholdCondition(threshold=100, operator="<="),
        ),
    ]
    return params, constraints


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(_no_constraints_scenario, id="no_constraints"),
        pytest.param(_no_label_duplicates_scenario, id="no_label_duplicates"),
        pytest.param(_linked_parameters_scenario, id="linked_parameters"),
        pytest.param(partial(_exclude_scenario, "OR"), id="exclude_or"),
        pytest.param(partial(_exclude_scenario, "AND"), id="exclude_and"),
        pytest.param(_cardinality_scenario, id="cardinality"),
        pytest.param(_sum_scenario, id="sum"),
        pytest.param(_dependencies_scenario, id="dependencies"),
        pytest.param(_permutation_invariance_scenario, id="permutation_invariance"),
        pytest.param(
            _permutation_invariance_with_dependencies_scenario,
            id="permutation_invariance_with_deps",
        ),
        pytest.param(_mixed_scenario, id="mixed"),
    ],
)
def test_constrained_cartesian_product(scenario):
    """Verify incremental and naive product construction produce identical results."""
    parameters, constraints = scenario()

    # Naive approach: full product then filter
    df_naive = parameter_cartesian_prod_pandas(parameters)
    _apply_constraint_filter_pandas(df_naive, constraints)

    # Incremental approach
    df_incremental = parameter_cartesian_prod_pandas_constrained(
        parameters, constraints
    )

    # Column order must be identical
    assert list(df_incremental.columns) == list(df_naive.columns)

    # Content must be identical (row order may differ)
    cols = df_naive.columns.tolist()
    assert_frame_equal(
        df_incremental.sort_values(cols).reset_index(drop=True),
        df_naive.sort_values(cols).reset_index(drop=True),
    )
