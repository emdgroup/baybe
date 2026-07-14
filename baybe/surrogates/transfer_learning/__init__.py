"""Transfer learning surrogates.

This subpackage collects surrogates that build a target model from one or more source
models, dispatched via
:attr:`~baybe.parameters.categorical.TaskParameter.override_transfer_learning_mode`.
"""

from baybe.surrogates.transfer_learning.mean_transfer import MeanTransferSurrogate
from baybe.surrogates.transfer_learning.residual_transfer import (
    ResidualTransferSurrogate,
)

__all__ = [
    "MeanTransferSurrogate",
    "ResidualTransferSurrogate",
]
