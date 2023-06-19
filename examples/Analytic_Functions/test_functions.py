"""A collection of test functions.

They wrap synthetic BoTorch test function to simplify using these for testing purposes.
"""

from abc import ABC
from typing import List, Optional, Tuple

from botorch.test_functions import (
    Ackley,
    Branin,
    Cosine8,
    Hartmann,
    Rastrigin,
    Rosenbrock,
    Shekel,
    SyntheticTestFunction,
)
from torch import Tensor


class AbstractTestFunction(ABC):
    """
    Wrapper class for handling BoTorch synthetic test functions.

    Enables to simply create a function which can be called for evaluation. It also
    ensures that it is safe to construct a test function that is only available for
    fixed dimensions even if a dimension keyword is used.

    Parameters
    ----------
    test_function: SyntheticTestFunction
        The BoTorch SyntheticTestFunction that should be wrapped
    bounds: List[Tuple[float, float]]
        The bounds of the input variables. Have to be set upon defining an inherited
        class.
    dim: int, optional
        The dimension of the test function. Also used as a flag to check whether the
        test function has a fixed or variable dimension.
    """

    def __init__(
        self,
        test_function: SyntheticTestFunction,
        bounds: List[Tuple[float, float]],
        dim: Optional[int] = None,
    ) -> None:
        # Only if a dim keyword is provided, we attempt to use it for the construction
        # of the actual test function
        if dim:
            self.test_function = test_function(dim=dim, bounds=bounds)
        else:
            self.test_function = test_function(bounds=bounds)

    def __call__(self, x: Tensor) -> float:
        # Make it easier to evaluate a function.
        return self.test_function.forward(x)

    def __getattr__(self, item):
        # Identify the attributes of this function with the original BoTorch function.
        return getattr(self.test_function, item)


class AckleyTestFunction(AbstractTestFunction):
    """The Ackley test function.

    It can be of arbitrary dimension dim and is bounded by [-32.768,32.768]^dim.
    It has one minimizer for its global minimum at x=(0,...,0) with f(x)=0.
    """

    def __init__(self, dim=2) -> None:
        super().__init__(
            test_function=Ackley,
            bounds=[
                (-32.768, 32.768),
            ]
            * dim,
            dim=dim,
        )


class BraninTestFunction(AbstractTestFunction):
    """The Branin test function.

    It is two dimensional and bounded by [-5, 10] x [0,10]
    It has three minimizer for its global minimum at x1=(-pi, 12.275), x2=(pi,2,275) and
    x3=(9,42478, 2,475) with f(xi)=0.397887.
    """

    def __init__(self, dim=None):
        super().__init__(test_function=Branin, bounds=[(-5, 10), (0, 15)])


class Cosine8TestFunction(AbstractTestFunction):
    """The Cosine8 test function.

    It is eight dimensional and bounded by [-1,1]^8.
    It has one minimizer for its global minimum at x=(0,...,0) with f(x)=0.
    """

    def __init__(self, dim=None):
        super().__init__(
            test_function=Cosine8,
            bounds=[
                (-1, 1),
            ]
            * 8,
        )


class HartmannTestFunction(AbstractTestFunction):
    """The Hartmann test function.

    It is six dimensional and bounded by [0,1]^6.
    It has 6 lcaol minima and one global minimum at
    x = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573) with f(x)=-3.32237
    """

    def __init__(self, dim=None) -> None:
        super().__init__(
            test_function=Hartmann,
            bounds=[
                (0, 1),
            ]
            * 6,
        )


class RastriginTestFunction(AbstractTestFunction):
    """The Rastrigin test function.

    It can be of arbitrary dimension dim and is bounded by [-5.12, 5.12]^dim.
    It has a lot of local minima but only one global minimum at x=(0,...,0) with f(x)=0.
    """

    def __init__(self, dim: int = 4):
        super().__init__(
            test_function=Rastrigin,
            bounds=[
                (-5.12, 5.12),
            ]
            * dim,
            dim=dim,
        )


class RosenbrockTestFunction(AbstractTestFunction):
    """The Rosenbrock test function.

    It can be of arbitrary dimension and is bounded by [-5,10]^dim.
    It has one local minimum at x=(1,...,1) with f(x)=0.
    """

    def __init__(self, dim=4):
        super().__init__(
            test_function=Rosenbrock,
            bounds=[
                (-5, 10),
            ]
            * dim,
            dim=dim,
        )


class ShekelTestFunction(AbstractTestFunction):
    """The Shekel test function.

    It is four dimensional and bounded by [0,10]^4.
    It has one global minimum at x=(4,4,4,4) with f(x)=-10.5363.
    """

    def __init__(self, dim=None):
        super().__init__(
            test_function=Shekel,
            bounds=[
                (0, 10),
            ]
            * 4,
        )
