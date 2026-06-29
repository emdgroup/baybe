"""Hybrid recommendation routines for BotorchRecommender."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from attrs import evolve

from baybe.constraints.utils import is_cardinality_fulfilled
from baybe.exceptions import (
    IncompatibilityError,
    IncompatibleAcquisitionFunctionError,
    MinimumCardinalityViolatedWarning,
)
from baybe.searchspace import SearchSpace
from baybe.utils.basic import flatten
from baybe.utils.dataframe import to_tensor
from baybe.utils.sampling_algorithms import sample_numerical_df

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.recommenders.pure.bayesian.botorch.core import BotorchRecommender


def recommend_hybrid_without_subsets(
    recommender: BotorchRecommender,
    searchspace: SearchSpace,
    batch_size: int,
) -> pd.DataFrame:
    """Recommend points using the ``optimize_acqf_mixed`` function of BoTorch.

    This functions samples points from the discrete subspace, performs optimization
    in the continuous subspace with these points being fixed and returns the best
    found solution.

    **Important**: This performs a brute-force calculation by fixing every possible
    assignment of discrete variables and optimizing the continuous subspace for
    each of them. It is thus computationally expensive.

    **Note**: This function implicitly assumes that discrete search space parts in
    the respective data frame come first and continuous parts come second.

    Args:
        recommender: The recommender instance.
        searchspace: The search space in which the recommendations should be made.
        batch_size: The size of the calculated batch.

    Raises:
        IncompatibleAcquisitionFunctionError: If a non-Monte Carlo acquisition
            function is used with a batch size > 1.

    Returns:
        The recommended points.
    """
    assert recommender._objective is not None

    # Interpoint constraints cannot be used with optimize_acqf_mixed, see
    # https://github.com/meta-pytorch/botorch/issues/2996
    if searchspace.continuous.has_interpoint_constraints:
        raise IncompatibilityError(
            "Interpoint constraints are not available in hybrid spaces."
        )
    if (
        batch_size > 1
        and not recommender._get_acquisition_function(
            recommender._objective
        ).supports_batching
    ):
        raise IncompatibleAcquisitionFunctionError(
            f"The '{recommender.__class__.__name__}' only works with Monte Carlo "
            f"acquisition functions for batch sizes > 1."
        )

    import torch
    from botorch.optim import optimize_acqf_mixed

    # Transform discrete candidates
    candidates = searchspace.discrete.get_candidates()
    candidates_comp = searchspace.discrete.transform(candidates)

    # Calculate the number of samples from the given percentage
    n_candidates = math.ceil(
        recommender.sampling_percentage * len(candidates_comp.index)
    )

    # Potential sampling of discrete candidates
    if recommender.hybrid_sampler is not None:
        candidates_comp = sample_numerical_df(
            candidates_comp, n_candidates, method=recommender.hybrid_sampler
        )

    # Prepare all considered discrete configurations in the
    # List[Dict[int, float]] format expected by BoTorch.
    num_comp_columns = len(candidates_comp.columns)
    candidates_comp.columns = list(range(num_comp_columns))
    fixed_features_list = candidates_comp.to_dict("records")

    # Actual call of the BoTorch optimization routine
    # NOTE: The explicit `or None` conversion is added as an additional safety net
    #   because it is unclear if the corresponding presence checks for these
    #   arguments is correctly implemented in all invoked BoTorch subroutines.
    #   For details: https://github.com/pytorch/botorch/issues/2042
    points, _ = optimize_acqf_mixed(
        acq_function=recommender._botorch_acqf,
        bounds=torch.from_numpy(searchspace.comp_rep_bounds.to_numpy(copy=True)),
        q=batch_size,
        num_restarts=recommender.n_restarts,
        raw_samples=recommender.n_raw_samples,
        fixed_features_list=fixed_features_list,  # type: ignore[arg-type]
        equality_constraints=flatten(
            c.to_botorch(
                searchspace.continuous.parameters,
                idx_offset=len(candidates_comp.columns),
                batch_size=batch_size if c.is_interpoint else None,
            )
            for c in searchspace.continuous.constraints_lin_eq
        )
        or None,
        inequality_constraints=flatten(
            c.to_botorch(
                searchspace.continuous.parameters,
                idx_offset=num_comp_columns,
                batch_size=batch_size if c.is_interpoint else None,
            )
            for c in searchspace.continuous.constraints_lin_ineq
        )
        or None,
    )

    # Align candidates with search space index. Done via including the search space
    # index during the merge, which is used later for back-translation into the
    # experimental representation
    merged = pd.merge(
        pd.DataFrame(points),
        candidates_comp.reset_index(),
        on=list(candidates_comp.columns),
        how="left",
    ).set_index("index")

    # Get experimental representation of discrete part
    rec_disc_exp = candidates.loc[merged.index]

    # Combine discrete and continuous parts
    rec_exp = pd.concat(
        [
            rec_disc_exp,
            merged.iloc[:, num_comp_columns:].set_axis(
                searchspace.continuous.parameter_names, axis=1
            ),
        ],
        axis=1,
    )

    return rec_exp


def recommend_hybrid_with_subsets(
    recommender: BotorchRecommender,
    searchspace: SearchSpace,
    batch_size: int,
) -> pd.DataFrame:
    """Recommend from a hybrid space with subset constraints.

    Uses ``SearchSpace.subsets()`` to enumerate the Cartesian
    product of discrete and continuous subset configurations, capped at
    ``max_n_subsets`` total. In purely discrete search spaces, subsets
    with fewer candidates than ``batch_size`` are pre-filtered.

    Args:
        recommender: The recommender instance.
        searchspace: The search space in which the recommendations should be made.
        batch_size: The size of the calculated batch.

    Returns:
        The recommended points.
    """
    subspace_c = searchspace.continuous

    # Get combined configurations, capped at max_n_subsets
    # NOTE: No min_discrete_candidates filtering in hybrid spaces because
    # optimize_acqf_mixed can produce multiple recommendations from a single
    # discrete candidate by varying continuous parameters.
    candidates = searchspace.discrete.get_candidates()
    combined_masks: Iterable[tuple[np.ndarray, frozenset[str]]]
    if searchspace.n_subsets <= recommender.max_n_subsets:
        combined_masks = searchspace.subsets()
    else:
        combined_masks = searchspace.sample_subsets(recommender.max_n_subsets)

    def make_callable(
        d_mask: np.ndarray,
        c_inactive_params: frozenset[str],
    ) -> Callable[[], tuple[pd.DataFrame, Tensor]]:
        def optimize() -> tuple[pd.DataFrame, Tensor]:
            import torch

            mod_disc = evolve(
                searchspace.discrete,
                exp_rep=candidates.loc[d_mask],
            )
            mod_cont = (
                subspace_c._enforce_cardinality_constraints(c_inactive_params)
                if c_inactive_params
                else subspace_c
            )
            mod_searchspace = evolve(
                searchspace, discrete=mod_disc, continuous=mod_cont
            )

            rec = recommend_hybrid_without_subsets(
                recommender, mod_searchspace, batch_size
            )

            comp = mod_searchspace.transform(rec)
            with torch.no_grad():
                acqf_value = recommender._botorch_acqf(
                    to_tensor(comp.values).unsqueeze(0)
                )
            return rec, acqf_value

        return optimize

    callables = (make_callable(d_mask, c_ip) for d_mask, c_ip in combined_masks)
    best_rec, _ = recommender._optimize_over_subsets(callables)

    # Post-check minimum cardinality on continuous columns
    if subspace_c.constraints_cardinality and not is_cardinality_fulfilled(
        best_rec[list(subspace_c.parameter_names)],
        subspace_c,
        check_maximum=False,
    ):
        warnings.warn(
            "At least one minimum cardinality constraint has been violated. "
            "This may occur when parameter ranges extend beyond zero in both "
            "directions, making the feasible region non-convex. For such "
            "parameters, minimum cardinality constraints are currently not "
            "enforced due to the complexity of the resulting optimization "
            "problem.",
            MinimumCardinalityViolatedWarning,
        )

    return best_rec
