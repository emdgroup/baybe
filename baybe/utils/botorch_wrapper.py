"""A wrapper class for synthetic BoTorch test functions.

They wrap synthetic BoTorch test function to simplify using these for testing purposes.
"""

import logging

from botorch.test_functions import SyntheticTestFunction
from torch import Tensor

log = logging.getLogger(__name__)


def botorch_function_wrapper(test_function: SyntheticTestFunction):
    """
    Wrapper for botorch analytical test functions. Turns them into a format that is
    accepted by lookup in simulations.

    Parameters
    ----------
    test_function: SyntheticTestFunction
        The synthetic test function, e.g. "Rastrigin()" from botorch.test_functions

    Returns
    -------
    Callable of signature Callable[[float,...], float]
    """

    def wrapper(*x: float) -> float:
        # Cast the provided list of floats to a tensor.
        x_tensor = Tensor(x)
        result = test_function.forward(x_tensor)
        # We do not need to return a tuple here.
        return float(result)

    return wrapper
