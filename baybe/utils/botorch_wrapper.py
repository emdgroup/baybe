"""A wrapper class for synthetic BoTorch test functions.

They wrap synthetic BoTorch test function to simplify using these for testing purposes.
"""

import logging
from typing import Optional, Type

from botorch.test_functions import SyntheticTestFunction
from torch import Tensor

log = logging.getLogger(__name__)


def botorch_function_wrapper(
    test_function: Type[SyntheticTestFunction], dim: Optional[int] = None
):
    """
    Wrapper for botorch analytical test functions. Turns them into a format that is
    accepted by lookup in simulations.

    Parameters
    ----------
    test_function: Type[SyntheticTestFunction]
        Class of the test function, e.g. Rastrigin from botorch.test_functions
    dim: int
        Dimensionality of the test

    Returns
    -------
    Callable of signature Callable[[float,...], float]
    """
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
        test_function = test_function()
    else:
        test_function = test_function(dim=dim)

    def wrapper(*x: float) -> float:
        # Cast the provided list of floats to a tensor.
        x_tensor = Tensor(x)
        result = test_function.forward(x_tensor)
        # We do not need to return a tuple here.
        return float(result)

    return wrapper
