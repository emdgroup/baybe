"""Continuous recommendation routines for BotorchRecommender."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Collection, Iterable
from typing import TYPE_CHECKING, Any

import pandas as pd
from attrs import evolve, fields

from baybe.constraints.continuous import ContinuousCardinalityConstraint
from baybe.constraints.utils import is_cardinality_fulfilled
from baybe.exceptions import (
    IncompatibilityError,
    MinimumCardinalityViolatedWarning,
)
from baybe.parameters.numerical import _FixedNumericalContinuousParameter
from baybe.searchspace import SubspaceContinuous
from baybe.utils.basic import flatten

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.recommenders.pure.bayesian.botorch.core import BotorchRecommender

_MAX_FEASIBLE_IC_ATTEMPTS = 5
"""Maximum number of attempts for drawing feasible initial conditions.

The sample count doubles after each unsuccessful attempt. In total, at most
``2 ** _MAX_FEASIBLE_IC_ATTEMPTS - 1`` times the initial sample count is drawn.
"""

_FEASIBLE_IC_SAMPLES_PER_CANDIDATE = 64
"""Number of samples drawn per required candidate in the first attempt.

The value corresponds to an assumed feasibility rate of roughly 1/64.
"""

_MIN_FEASIBLE_IC_SAMPLES = 256
"""Minimum number of samples drawn in the first attempt."""


def make_feasible_ic_generator(
    constraints: Collection[Callable[[Tensor], Tensor]],
    subspace_continuous: SubspaceContinuous,
    idx_offset: int = 0,
    preset_fixed_features: dict[int, float] | None = None,
) -> Callable[..., Tensor]:
    """Create an initial condition generator respecting nonlinear constraints.

    BoTorch requires the initial conditions of a constrained acquisition function
    optimization to satisfy all nonlinear inequality constraints. The returned
    generator draws configurations from the given continuous subspace, keeps only
    those satisfying the constraints and reshapes the result into the format expected
    by BoTorch. Feasible configurations found in earlier attempts are retained, and
    the number of drawn samples is doubled after each unsuccessful attempt.

    Args:
        constraints: The nonlinear constraints encoded as ``callable(x) >= 0``.
            The callables must support batched inputs of shape ``(n, d)``.
        subspace_continuous: The continuous subspace to draw samples from. It must not
            contain the constraints encoded in ``constraints`` because its sampling
            routine cannot handle general nonlinear constraints.
        idx_offset: The index offset of the continuous parameters within the full
            computational representation, i.e. the number of preceding columns
            occupied by discrete parameters.
        preset_fixed_features: Fixed features applied to every generated
            configuration, given as ``{index: value}`` pairs over the full
            computational representation. Used for parameters that are fixed by the
            search space itself, since their fixed values are not passed to the
            generator by BoTorch. Merged with the fixed features passed by BoTorch at
            runtime. Conflicting values for the same index raise a ``ValueError``.

    Returns:
        A generator compatible with the ``ic_generator`` argument of BoTorch's
        optimization routines.
    """
    # Strip the nonlinear constraints so that the standard sampling routine can be
    # used. Cardinality constraints are preserved since they are handled natively.
    sampling_subspace = evolve(
        subspace_continuous,
        constraints_nonlin=subspace_continuous.constraints_cardinality,
    )

    def generate(
        *,
        bounds: Tensor,
        q: int,
        num_restarts: int,
        raw_samples: int | None = None,
        fixed_features: dict[int, float] | None = None,
        acq_function: Any = None,
        options: dict[str, Any] | None = None,
        inequality_constraints: Any = None,
        equality_constraints: Any = None,
    ) -> Tensor:
        """Generate feasible initial conditions for BoTorch's optimizer.

        Args:
            bounds: The bounds of the optimization problem, used to determine the
                device, dtype and dimensionality of the generated candidates.
            q: The number of candidates per batch element.
            num_restarts: The number of optimizer restarts.
            raw_samples: The number of raw samples used to derive the initial sample
                budget. Ignored if smaller than the budget derived from the required
                number of candidates.
            fixed_features: Fixed features passed by BoTorch as ``{index: value}``
                pairs over the full computational representation. They must be
                consistent with ``preset_fixed_features``; conflicts raise a
                ``ValueError``.
            acq_function: The acquisition function. Unused, accepted for interface
                compatibility with BoTorch.
            options: BoTorch optimization options. Unused.
            inequality_constraints: BoTorch inequality constraints. Unused.
            equality_constraints: BoTorch equality constraints. Unused.

        Raises:
            CandidateGenerationError: If the sample budget is exhausted before
                enough feasible candidates have been found.
            ValueError: If the candidate layout is inconsistent or a constraint
                returns an unexpected number of values.

        Returns:
            A tensor of shape ``(num_restarts, q, d)`` containing feasible initial
            conditions.
        """
        import torch
        from botorch.exceptions.errors import CandidateGenerationError

        from baybe.utils.dataframe import to_tensor

        # Unused arguments accepted for interface compatibility with BoTorch.
        del acq_function, options
        del inequality_constraints, equality_constraints

        required = num_restarts * q

        # BoTorch passes the fixed features of the current optimization call. The
        # preset ones account for parameters that are fixed by the search space
        # itself, whose values are not forwarded by BoTorch.
        preset = preset_fixed_features or {}
        runtime = fixed_features or {}
        conflicts = [
            (index, preset[index], runtime[index])
            for index in preset.keys() & runtime.keys()
            if runtime[index] != preset[index]
        ]
        if conflicts:
            raise ValueError(
                f"Conflicting fixed feature values between preset and runtime "
                f"values: {conflicts}."
            )
        merged_fixed_features = {**preset, **runtime}
        missing = set(range(idx_offset)) - merged_fixed_features.keys()
        if missing:
            raise ValueError(
                f"Every discrete dimension requires a fixed feature, but no value "
                f"was provided for the indices {sorted(missing)}."
            )

        n_samples = max(
            raw_samples or 0,
            _FEASIBLE_IC_SAMPLES_PER_CANDIDATE * required,
            _MIN_FEASIBLE_IC_SAMPLES,
        )
        budget = n_samples * (2**_MAX_FEASIBLE_IC_ATTEMPTS - 1)
        feasible = torch.empty(
            (0, bounds.shape[-1]), dtype=bounds.dtype, device=bounds.device
        )
        drawn = 0

        for _attempt in range(_MAX_FEASIBLE_IC_ATTEMPTS):
            n = min(n_samples, budget - drawn)
            samples = to_tensor(sampling_subspace.sample_uniform(n)).to(
                dtype=bounds.dtype, device=bounds.device
            )
            if idx_offset + samples.shape[-1] != bounds.shape[-1]:
                raise ValueError(
                    f"The sampled subspace has {samples.shape[-1]} dimensions, but "
                    f"{bounds.shape[-1] - idx_offset} were expected."
                )

            candidates = torch.zeros(
                (n, bounds.shape[-1]),
                dtype=bounds.dtype,
                device=bounds.device,
            )
            candidates[:, idx_offset:] = samples
            for index, value in merged_fixed_features.items():
                candidates[:, index] = value

            is_feasible = torch.ones(n, dtype=torch.bool, device=bounds.device)
            # Gradients are not needed for the feasibility check, and tracking them
            # across thousands of samples would be wasteful
            with torch.no_grad():
                for constraint in constraints:
                    values = constraint(candidates)
                    if values.numel() != n:
                        raise ValueError(
                            f"A constraint returned {values.numel()} values for "
                            f"{n} candidates."
                        )
                    is_feasible &= values.reshape(n) >= 0

            feasible = torch.cat((feasible, candidates[is_feasible]))
            drawn += n
            if len(feasible) >= required:
                return feasible[:required].reshape(num_restarts, q, -1)
            n_samples *= 2

        raise CandidateGenerationError(
            f"Could not generate {required} initial conditions satisfying the "
            f"nonlinear constraints after drawing {drawn} configurations in "
            f"{_MAX_FEASIBLE_IC_ATTEMPTS} attempts. Only {len(feasible)} feasible "
            f"configurations were found. The feasible region defined by the "
            f"nonlinear constraints may be too small."
        )

    return generate


def recommend_continuous_torch(
    recommender: BotorchRecommender,
    subspace_continuous: SubspaceContinuous,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Dispatcher selecting the continuous optimization routine."""
    if subspace_continuous.n_subsets > 0:
        return recommend_continuous_with_cardinality_constraints(
            recommender, subspace_continuous, batch_size
        )
    else:
        return recommend_continuous_without_cardinality_constraints(
            recommender, subspace_continuous, batch_size
        )


def recommend_continuous_with_cardinality_constraints(
    recommender: BotorchRecommender,
    subspace_continuous: SubspaceContinuous,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Recommend from a continuous space with cardinality constraints.

    Optimizes the acquisition function across subsets defined by cardinality
    constraints and returns the best result.

    The specific collection of subsets considered by the recommender is obtained
    as either the full combinatorial set of possible parameter splits or a random
    selection thereof, depending on the upper bound specified by the corresponding
    recommender attribute.

    In each subset, the constraint-imposed configuration is fixed, so that the
    constraints can be removed and a regular optimization can be performed. The
    recommendation is then constructed from the combined optimization results of the
    unconstrained spaces.

    Args:
        recommender: The recommender instance.
        subspace_continuous: The continuous subspace from which to generate
            recommendations.
        batch_size: The size of the recommendation batch.

    Returns:
        The recommendations and corresponding acquisition values.

    Raises:
        ValueError: If the continuous search space has no cardinality
            constraints.
    """
    if subspace_continuous.n_subsets == 0:
        raise ValueError(
            f"'{recommend_continuous_with_cardinality_constraints.__name__}' "
            f"expects a subspace with cardinality constraints."
        )

    # Determine search scope based on number of subset configurations
    configs: Iterable[frozenset[str]]
    if subspace_continuous.n_subsets <= recommender.max_n_subsets:
        configs = subspace_continuous.inactive_parameter_combinations()
    else:
        configs = subspace_continuous._sample_inactive_parameters(
            recommender.max_n_subsets
        )

    # Create closures for each subset configuration
    def make_callable(
        inactive_params: Collection[str],
    ) -> Callable[[], tuple[Tensor, Tensor]]:
        def optimize() -> tuple[Tensor, Tensor]:
            import torch

            sub = subspace_continuous._enforce_cardinality_constraints(inactive_params)
            # Note: We explicitly evaluate the acqf function for the batch
            # because the object returned by the optimization routine may
            # contain joint or individual acquisition values, depending on
            # whether sequential or joint optimization is applied
            p, _ = recommend_continuous_torch(recommender, sub, batch_size)
            with torch.no_grad():
                acqf_value = recommender._botorch_acqf(p)
            return p, acqf_value

        return optimize

    callables = (make_callable(ip) for ip in configs)
    points, acqf_value = recommender._optimize_over_subsets(callables)

    # Check if any minimum cardinality constraints are violated
    if not is_cardinality_fulfilled(
        pd.DataFrame(points, columns=subspace_continuous.parameter_names),
        subspace_continuous,
        check_maximum=False,
    ):
        warnings.warn(
            "At least one minimum cardinality constraint has been violated. "
            "This may occur when parameter ranges extend beyond zero in both "
            "directions, making the feasible region non-convex. For such "
            "parameters, minimum cardinality constraints are currently not "
            "enforced due to the complexity of the resulting optimization problem.",
            MinimumCardinalityViolatedWarning,
        )

    return points, acqf_value


def recommend_continuous_without_cardinality_constraints(
    recommender: BotorchRecommender,
    subspace_continuous: SubspaceContinuous,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Recommend from a continuous search space without cardinality constraints.

    Args:
        recommender: The recommender instance.
        subspace_continuous: The continuous subspace from which to generate
            recommendations.
        batch_size: The size of the recommendation batch.

    Returns:
        The recommendations and corresponding acquisition values.

    Raises:
        ValueError: If the continuous search space has cardinality constraints.
    """
    import torch
    from botorch.optim import optimize_acqf

    if subspace_continuous.n_subsets > 0:
        raise ValueError(
            f"'{recommend_continuous_without_cardinality_constraints.__name__}' "
            f"expects a subspace without cardinality constraints."
        )

    fixed_parameters = {
        idx: p.value
        for (idx, p) in enumerate(subspace_continuous.parameters)
        if isinstance(p, _FixedNumericalContinuousParameter)
    }

    # TODO: Add option for automatic choice once the "settings" PR is merged,
    #   which ships the necessary machinery
    if (
        recommender.sequential_continuous
        and subspace_continuous.has_interpoint_constraints
    ):
        from baybe.recommenders.pure.bayesian.botorch.core import BotorchRecommender

        raise IncompatibilityError(
            f"Setting the "
            f"'{fields(BotorchRecommender).sequential_continuous.name}' "
            f"flag to ``True`` while interpoint constraints are present in the "
            f"continuous subspace is not supported. "
        )

    # Translate the nonlinear constraints, excluding cardinality constraints since
    # those are already handled by the subset-based optimization routines.
    nonlinear_inequality_constraints = flatten(
        c.to_botorch(subspace_continuous.parameters)
        for c in subspace_continuous.constraints_nonlin
        if not isinstance(c, ContinuousCardinalityConstraint)
    )

    equality_constraints = flatten(
        c.to_botorch(
            subspace_continuous.parameters,
            batch_size=batch_size if c.is_interpoint else None,
        )
        for c in subspace_continuous.constraints_lin_eq
    )

    if nonlinear_inequality_constraints:
        # BoTorch combines fixed features and nonlinear constraints incorrectly,
        # inserting the fixed values twice when assembling the constraint input
        # (https://github.com/meta-pytorch/botorch/issues/3332). The fixed parameters
        # are instead kept in place by their degenerate bounds.
        # TODO: Pass the fixed features again once the fix from
        #   https://github.com/meta-pytorch/botorch/pull/3333 is released.
        bo_fixed_features = None
    else:
        bo_fixed_features = fixed_parameters or None

    # NOTE: The explicit `or None` conversion is added as an additional safety net
    #   because it is unclear if the corresponding presence checks for these
    #   arguments is correctly implemented in all invoked BoTorch subroutines.
    #   For details: https://github.com/pytorch/botorch/issues/2042
    points, acqf_values = optimize_acqf(
        acq_function=recommender._botorch_acqf,
        bounds=torch.from_numpy(
            subspace_continuous.comp_rep_bounds.to_numpy(copy=True)
        ),
        q=batch_size,
        num_restarts=recommender.n_restarts,
        raw_samples=recommender.n_raw_samples,
        fixed_features=bo_fixed_features,
        equality_constraints=equality_constraints or None,
        inequality_constraints=flatten(
            c.to_botorch(
                subspace_continuous.parameters,
                batch_size=batch_size if c.is_interpoint else None,
            )
            for c in subspace_continuous.constraints_lin_ineq
        )
        or None,
        nonlinear_inequality_constraints=nonlinear_inequality_constraints or None,
        sequential=recommender.sequential_continuous,
        ic_generator=(
            make_feasible_ic_generator(
                [constraint for constraint, _ in nonlinear_inequality_constraints],
                subspace_continuous,
                preset_fixed_features=fixed_parameters or None,
            )
            if nonlinear_inequality_constraints
            else None
        ),
    )
    assert acqf_values is not None  # for mypy; guaranteed by optimize_acqf defaults
    return points, acqf_values
