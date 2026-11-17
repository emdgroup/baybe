"""Rendering helpers for language model prompts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import fields

from baybe.parameters.base import _DiscreteLabelLikeParameter
from baybe.parameters.categorical import CategoricalParameter
from baybe.parameters.custom import CustomDiscreteParameter
from baybe.parameters.substance import SubstanceParameter

if TYPE_CHECKING:
    from baybe.constraints.base import Constraint
    from baybe.objectives.base import Objective
    from baybe.parameters.base import Parameter


_T = TypeVar("_T", dict[str, Any], list[Any], object)
"""Serialized data, constrained so that dictionaries and lists keep their type."""


def _strip_empty(value: _T, /) -> _T:
    """Recursively remove ``None`` values and empty dictionaries from serialized data.

    Both are serialization residue, i.e. unset optional attributes and nested objects
    without remaining fields. Lists are kept even when empty, since they are content.

    Args:
        value: The serialized data to clean up.

    Returns:
        The cleaned-up data.
    """
    match value:
        case dict():
            cleaned = {k: _strip_empty(v) for k, v in value.items()}
            # pyrefly: ignore[bad-return]  # No per-constraint check
            return {k: v for k, v in cleaned.items() if v not in (None, {})}
        case list():
            # pyrefly: ignore[bad-return]  # No per-constraint check
            return [_strip_empty(item) for item in value]
        case _:
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
    return _strip_empty(serialized)


def to_json_lines(serialized_objects: Iterable[dict[str, Any]], /) -> str:
    """Render serialized objects as JSON, one object per line.

    Args:
        serialized_objects: The serialized objects to render.

    Returns:
        The rendered objects.
    """
    return "\n".join(json.dumps(obj, ensure_ascii=False) for obj in serialized_objects)


def render_template(template: str, /, context: Mapping[str, Any]) -> str:
    """Render a prompt template with the shared Jinja configuration.

    Block tags leave no blank lines in the rendered prompt because ``trim_blocks``
    is enabled, and an undefined variable raises instead of silently rendering an
    empty string because ``StrictUndefined`` is used.

    Args:
        template: The Jinja template to render.
        context: The variables passed to the template.

    Returns:
        The rendered template.
    """
    from baybe._optional.llm import StrictUndefined, Template

    return Template(template, trim_blocks=True, undefined=StrictUndefined).render(
        context
    )
