"""Shared mock helpers for LLM recommender tests."""

import json
from types import SimpleNamespace

from baybe._optional.info import CHEM_INSTALLED
from baybe.objectives import SingleTargetObjective
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget

if CHEM_INSTALLED:
    from baybe.parameters import SubstanceParameter

PATCH_TARGET = "baybe._optional.llm.completion"
"""Patch target for mocking the LiteLLM ``completion`` function."""


def mock_response(content):
    """Build a fake LiteLLM response object from raw text content.

    Needs to use a SimpleNamespace to mimic the structure of a real LiteLLM response.
    """
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def make_valid_json(searchspace, batch_size):
    """Sample eligible candidates and return them as the expected JSON string."""
    params = searchspace.parameters
    discrete_params = [p for p in params if p.is_discrete]
    continuous_params = [p for p in params if not p.is_discrete]

    continuous_entry = {
        cp.name: sum(cp.bounds.to_tuple()) / 2 for cp in continuous_params
    }

    suggestions = []
    if discrete_params:
        candidates = searchspace.discrete.get_candidates()[0]
        for _, row in candidates.head(batch_size).iterrows():
            entry = {p.name: row[p.name] for p in discrete_params}
            entry.update(continuous_entry)
            suggestions.append({"explanation": "Mock suggestion.", "parameters": entry})
    else:
        for _ in range(batch_size):
            suggestions.append(
                {
                    "explanation": "Mock suggestion.",
                    "parameters": dict(continuous_entry),
                }
            )

    return json.dumps(
        suggestions,
    )


def make_suggestion(searchspace, **overrides):
    """Build a valid suggestion dict with optional parameter overrides.

    Overrides replace or add parameter values, e.g. ``Num="abc"`` to inject an
    invalid value or ``Unknown="x"`` to add an extra parameter.
    """
    params = {}
    discrete_params = [p for p in searchspace.parameters if p.is_discrete]
    continuous_params = [p for p in searchspace.parameters if not p.is_discrete]
    if discrete_params:
        row = searchspace.discrete.get_candidates()[0].iloc[0]
        params.update({p.name: row[p.name] for p in discrete_params})
    for cp in continuous_params:
        params[cp.name] = sum(cp.bounds.to_tuple()) / 2
    params.update(overrides)
    return {"explanation": "test", "parameters": params}


def make_response_json(searchspace, **overrides):
    """Build a JSON response string with one suggestion.

    Overrides are forwarded to :func:`make_suggestion`.
    """
    return json.dumps(
        [make_suggestion(searchspace, **overrides)],
    )


def make_discrete_searchspace():
    """A discrete search space with categorical, numerical, and substance params."""
    params: list = [
        CategoricalParameter("Cat", ("A", "B", "C")),
        NumericalDiscreteParameter("Num", (1.0, 2.0, 3.0)),
    ]
    if CHEM_INSTALLED:
        params.append(
            SubstanceParameter(
                "Solvent",
                data={"Water": "O", "Ethanol": "CCO", "Methanol": "CO"},
                encoding="MORDRED",
            )
        )
    return SearchSpace.from_product(params)


def make_continuous_searchspace():
    """A search space with a single continuous parameter."""
    return SearchSpace.from_product(
        [NumericalContinuousParameter("Cont", bounds=(0.0, 10.0))]
    )


def make_hybrid_searchspace():
    """A search space with both discrete and continuous parameters."""
    return SearchSpace.from_product(
        [
            CategoricalParameter("Cat", ("A", "B", "C")),
            NumericalContinuousParameter("Cont", bounds=(0.0, 10.0)),
        ]
    )


def make_objective():
    """An objective with metadata to avoid the missing-metadata warning."""
    return SingleTargetObjective(
        target=NumericalTarget(name="yield"),
        metadata={"description": "Maximize the reaction yield."},
    )
