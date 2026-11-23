"""Multi-step Bayesian optimization over a DAG of experimental steps.

Each node owns its search space and produces one or more named outputs, which its
children consume together. The outputs of leaf nodes form the graph outputs; the
objective evaluates them in its declared target order. A :class:`MultiStepModel`
fits one node model per DAG node, independently fitting multiple outputs within a
node, and propagates complete Monte-Carlo sample paths through the DAG. A
:class:`MultiStepCampaign` records each node execution and its output bundle as one
measurement, then recommends the next node action.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .acquisition import PartialKnowledgeGradientFN
from .campaign import MultiStepCampaign
from .context import ContextMode, MultiStepContext
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
    "MultiStepContext",
    "ContextMode",
    "MultiStepDAG",
    "MultiStepData",
    "MultiStepModel",
    "MultiStepNode",
    "FeasibilityCondition",
    "NodeMeasurement",
    "PartialKnowledgeGradientFN",
    "validate_network_measurements",
]
