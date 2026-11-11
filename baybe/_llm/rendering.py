"""Rendering helpers for language model prompts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from attrs import fields

from baybe.parameters.base import _DiscreteLabelLikeParameter
from baybe.parameters.categorical import CategoricalParameter
from baybe.parameters.custom import CustomDiscreteParameter
from baybe.parameters.substance import SubstanceParameter

if TYPE_CHECKING:
    from baybe.constraints.base import Constraint
    from baybe.objectives.base import Objective
    from baybe.parameters.base import Parameter


def _drop_empty(serialized: dict[str, Any], /) -> dict[str, Any]:
    """Remove empty fields from a serialized object, recursively.

    Empty means ``None`` values and empty dictionaries: ``None`` is how an unset
    optional attribute is serialized, and an empty dictionary is what a nested object
    reduces to once all of its fields are dropped, i.e. serialization residue rather
    than payload. Lists are traversed but never removed, not even when empty: since
    cleaning preserves list lengths, an empty list was explicitly empty in the input,
    so dropping it would discard declared, structurally expected data instead of
    residue.

    Args:
        serialized: The serialized object to clean up.

    Returns:
        The cleaned-up serialized object.
    """
    return {
        key: cleaned
        for key, value in serialized.items()
        if (cleaned := _clean_value(value)) is not None and cleaned != {}
    }


def _clean_value(value: Any, /) -> Any:
    """Recursively clean a value inside a serialized object.

    Args:
        value: The value to clean up.

    Returns:
        The cleaned-up value.
    """
    if isinstance(value, dict):
        return _drop_empty(value)
    if isinstance(value, list):
        return [_clean_value(item) for item in value]
    return value


def to_prompt_dict(obj: Parameter | Constraint | Objective, /) -> dict[str, Any]:
    """Serialize an object into the representation shown to the language model.

    Args:
        obj: The object to serialize.

    Returns:
        The serialized object.
    """
    serialized = obj.to_dict()
    if isinstance(obj, _DiscreteLabelLikeParameter):
        active = list(obj.active_values)
        serialized.pop(fields(_DiscreteLabelLikeParameter)._active_values.alias)
        # For label-like parameters, ``values`` holds the active labels only.
        serialized[fields(CategoricalParameter)._values.alias] = active
        if isinstance(obj, SubstanceParameter):
            # Substance parameters keep the SMILES of their active labels.
            serialized[fields(SubstanceParameter).data.alias] = {
                label: smiles for label, smiles in obj.data.items() if label in active
            }
        elif isinstance(obj, CustomDiscreteParameter):
            # The descriptor table of custom parameters is omitted since it is only
            # relevant for the surrogate model.
            del serialized[fields(CustomDiscreteParameter).data.alias]
    # ``None`` values and empty dictionaries are dropped since they carry no
    # information for the language model.
    return _drop_empty(serialized)


def to_json_lines(serialized_objects: Iterable[dict[str, Any]], /) -> str:
    """Render serialized objects as JSON, one object per line.

    Args:
        serialized_objects: The serialized objects to render.

    Returns:
        The rendered objects.
    """
    return "\n".join(json.dumps(obj, ensure_ascii=False) for obj in serialized_objects)
