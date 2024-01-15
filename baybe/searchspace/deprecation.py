"""Temporary functionality for backward compatibility."""

from __future__ import annotations

from typing import List

from baybe.constraints.base import Constraint
from baybe.parameters.base import Parameter
from baybe.searchspace import SearchSpace
from baybe.utils import converter


def structure_searchspace_from_config(specs: dict, _) -> SearchSpace:
    """Assemble the search space from "config" format.

    It uses the alternative :func:`baybe.searchspace.core.SearchSpace.from_product`
    constructor, which allows to deserialize search space specifications that are
    provided in a user-friendly format (i.e. via parameters and constraints).
    """
    parameters = converter.structure(specs["parameters"], List[Parameter])
    constraints = specs.get("constraints", None)
    if constraints:
        constraints = converter.structure(specs["constraints"], List[Constraint])
    return SearchSpace.from_product(parameters, constraints)
