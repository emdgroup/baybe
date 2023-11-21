"""A wrapper class for synthetic BoTorch test functions."""

from botorch.test_functions import SyntheticTestFunction
from torch import Tensor


def botorch_function_wrapper(test_function: SyntheticTestFunction):
    """Turn a BoTorch test function into a format accepted by lookup in simulations.

    See :mod:`baybe.simulation` for details.

    Args:
        test_function: The synthetic test function from BoTorch. See
            https://botorch.org/api/test_functions.html.

    Returns:
        A wrapped version of the provided function.
    """

    def wrapper(*x: float) -> float:
        # Cast the provided list of floats to a tensor.
        x_tensor = Tensor(x)
        result = test_function.forward(x_tensor)
        # We do not need to return a tuple here.
        return float(result)

    return wrapper
