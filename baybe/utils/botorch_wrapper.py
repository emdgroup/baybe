"""A wrapper class for synthetic BoTorch test functions.

They wrap synthetic BoTorch test function to simplify using these for testing purposes.
"""

import logging
from typing import Optional, Type

from botorch.test_functions import SyntheticTestFunction
from torch import Tensor

log = logging.getLogger(__name__)


class BayBEBotorchFunctionWrapper:
    """
    Wrapper class for handling BoTorch synthetic test functions.

    Enables to simply create a function which can be called for evaluation. It also
    ensures that it is safe to construct a test function that is only available for
    fixed dimensions even if a dimension keyword is used.

    Parameters
    ----------
    test_function: SyntheticTestFunction
        The BoTorch SyntheticTestFunction that should be wrapped
    dim: int, optional
        The dimension of the test function. Also used as a flag to check whether the
        test function has a fixed or variable dimension.
    """

    def __init__(
        self,
        test_function: Type[SyntheticTestFunction],
        dim: Optional[int] = None,
    ) -> None:
        # If the test_function already does not have a flexible dimension, then we
        # ignore the dim keyword and print a corresponding warning
        if hasattr(test_function, "dim"):
            if test_function.dim != dim:
                log.warning(
                    "You choose a dimension of %d for the test function"
                    "%s. However, this function can only be used in"
                    "%s dimension, so the provided dimension is ignored.",
                    dim,
                    test_function,
                    test_function.dim,
                )
            self.test_function = test_function()
        else:
            self.test_function = test_function(dim=dim)

    def __call__(self, *x: float) -> float:
        # Cast the provided list of floats to a tensor.
        x_tensor = Tensor(x)
        result = self.test_function.forward(x_tensor)
        # We do not need to return a tuple here.
        return float(result)

    def __getattr__(self, item):
        # Identify the attributes of this function with the original BoTorch function.
        return getattr(self.test_function, item)
