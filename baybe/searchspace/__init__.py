"""BayBE search spaces."""

from baybe.searchspace.candidates import (
    CandidatesProtocol,
    ProductCandidates,
    TableCandidates,
)
from baybe.searchspace.continuous import SubspaceContinuous
from baybe.searchspace.core import (
    SearchSpace,
    SearchSpaceType,
    validate_searchspace_from_config,
)
from baybe.searchspace.discrete import SubspaceDiscrete

__all__ = [
    # Search space
    "validate_searchspace_from_config",
    "SearchSpace",
    "SearchSpaceType",
    # Discrete
    "CandidatesProtocol",
    "ProductCandidates",
    "TableCandidates",
    "SubspaceDiscrete",
    # Continuous
    "SubspaceContinuous",
]
