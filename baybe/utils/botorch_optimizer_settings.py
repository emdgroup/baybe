"""Settings of initial conditions for botorch's optimization function."""

from collections.abc import Callable

import torch
from botorch.optim.initializers import gen_batch_initial_conditions
from torch import Tensor

from baybe.searchspace.continuous import SubspaceContinuous


def get_initial_condition_settings(
    subspace_continuous: SubspaceContinuous
) -> tuple[Callable | None, dict]:
    """Get settings related to initial condition generator in botorch.

    Args:
        subspace_continuous: the continuous subspace object.

    Returns:
        An initial condition generator.
        A dictionary of additional keyword args needed in the initial condition
        generator.
    """
    if subspace_continuous.constraints_cardinality is None:
        ic_generator = None
        ic_gen_kwargs = {}
    else:

        def get_initial_conditions_generator(
            subspace_continuous: SubspaceContinuous,
        ) -> Callable[[int, int, int], Tensor]:
            """Return a callable for generating initial values.

            Args:
                subspace_continuous: A continuous subspace object.

            Returns:
                Callable[[int, int, int], Tensor]: A callable that generates initial
                values satisfying all constraints.
            """

            def generator(n: int, q: int, seed: int) -> Tensor:
                """Generate initial condition for botorch optimizer."""
                # see generator in gen_batch_initial_conditions

                candidates = subspace_continuous.sample_uniform(n * q)
                return (
                    torch.tensor(candidates.values)
                    .reshape(n, q, candidates.shape[-1])
                    .contiguous()
                )

            return generator

        ic_generator = gen_batch_initial_conditions
        ic_gen_kwargs = {
            "generator": get_initial_conditions_generator(
                subspace_continuous=subspace_continuous,
            )
        }
    return ic_generator, ic_gen_kwargs
