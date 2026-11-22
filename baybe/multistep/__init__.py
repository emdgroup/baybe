"""Multi-step Bayesian optimization over a DAG of experimental steps.

Each node owns its search space and produces a single output consumed by its
children. The outputs of leaf nodes form the graph outputs; an objective evaluates
all of them. A single
:class:`MultiStepModel` fits one model per node and propagates Monte-Carlo
samples through the DAG, while a :class:`MultiStepCampaign` records node
measurements and recommends the next node action.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .acquisition import PartialKnowledgeGradientFN
from .campaign import MultiStepCampaign
from .dag import MultiStepDAG
from .data import MultiStepData, NodeMeasurement
from .node import FeasibilityCondition, MultiStepNode

if TYPE_CHECKING:
    from .model import MultiStepModel, validate_network_measurements

_LAZY_EXPORTS = {
    "MultiStepModel": ".model",
    "validate_network_measurements": ".model",
}


def __getattr__(name: str) -> Any:
    """Import heavy model exports only when they are actually requested."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "MultiStepCampaign",
    "MultiStepDAG",
    "MultiStepData",
    "MultiStepModel",
    "MultiStepNode",
    "FeasibilityCondition",
    "NodeMeasurement",
    "PartialKnowledgeGradientFN",
    "validate_network_measurements",
]
