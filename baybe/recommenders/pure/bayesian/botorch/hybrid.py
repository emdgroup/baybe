"""Hybrid recommendation routines for BotorchRecommender."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw
import numpy as np
from attrs import evolve

from baybe.constraints.utils import is_cardinality_fulfilled
from baybe.exceptions import (
    IncompatibilityError,
    IncompatibleAcquisitionFunctionError,
    MinimumCardinalityViolatedWarning,
)
from baybe.searchspace import SearchSpace
from baybe.searchspace.candidates import TableCandidates
from baybe.settings import active_settings
from baybe.utils.basic import flatten
from baybe.utils.dataframe import _df_with_backend, to_tensor
from baybe.utils.sampling_algorithms import sample_numerical_df

if TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame
    from torch import Tensor

    from baybe.recommenders.pure.bayesian.botorch.core import BotorchRecommender


def recommend_hybrid_without_subsets(
    recommender: BotorchRecommender,
    searchspace: SearchSpace,
    batch_size: int,
) -> IntoDataFrame:
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

    from botorch.optim import optimize_acqf_mixed

    # Transform discrete candidates
    candidates = nw.from_native(searchspace.discrete.get_candidates(), eager_only=True)
    candidates_comp = nw.from_native(searchspace.discrete.transform(candidates))

    # Calculate the number of samples from the given percentage
    n_candidates = math.ceil(recommender.sampling_percentage * len(candidates_comp))

    # Potential sampling of discrete candidates
    if recommender.hybrid_sampler is not None:
        candidates_comp = nw.from_native(
            sample_numerical_df(
                candidates_comp.to_pandas(),
                n_candidates,
                method=recommender.hybrid_sampler,
            ),
        )

    # Prepare all considered discrete configurations in the
    # List[Dict[int, float]] format expected by BoTorch.
    n_comp_columns = len(candidates_comp.columns)
    fixed_features_list = [
        dict(enumerate(row)) for row in candidates_comp.to_numpy().tolist()
    ]

    # Actual call of the BoTorch optimization routine
    # NOTE: The explicit `or None` conversion is added as an additional safety net
    #   because it is unclear if the corresponding presence checks for these
    #   arguments is correctly implemented in all invoked BoTorch subroutines.
    #   For details: https://github.com/pytorch/botorch/issues/2042
    points, _ = optimize_acqf_mixed(
        acq_function=recommender._botorch_acqf,
        bounds=to_tensor(searchspace.comp_rep_bounds),
        q=batch_size,
        num_restarts=recommender.n_restarts,
        raw_samples=recommender.n_raw_samples,
        fixed_features_list=fixed_features_list,
        equality_constraints=flatten(
            c.to_botorch(
                searchspace.continuous.parameters,
                idx_offset=n_comp_columns,
                batch_size=batch_size if c.is_interpoint else None,
            )
            for c in searchspace.continuous.constraints_lin_eq
        )
        or None,
        inequality_constraints=flatten(
            c.to_botorch(
                searchspace.continuous.parameters,
                idx_offset=n_comp_columns,
                batch_size=batch_size if c.is_interpoint else None,
            )
            for c in searchspace.continuous.constraints_lin_ineq
        )
        or None,
    )

    # Recover the positional index of the discrete part of each recommended point.
    # Operating directly on the BoTorch output avoids introducing any further
    # imprecision beyond what the optimizer itself produces.
    disc_choices = to_tensor(candidates_comp)
    disc_points = points[:, :n_comp_columns]
    row_idxs = (
        (disc_choices.unsqueeze(0) == disc_points.unsqueeze(1))
        .all(dim=-1)
        .int()
        .argmax(dim=1)
    )

    # Combine the discrete part in experimental representation with the
    # optimized continuous part from the BoTorch output
    rec_cont = nw.from_numpy(
        points[:, n_comp_columns:].numpy(),
        schema=searchspace.continuous.parameter_names,
        backend=active_settings.default_dataframe_backend,
    )
    rec_disc_exp = _df_with_backend(
        candidates[row_idxs.tolist()], active_settings.default_dataframe_backend
    )
    return nw.concat([rec_disc_exp, rec_cont], how="horizontal").to_native()


def recommend_hybrid_with_subsets(
    recommender: BotorchRecommender,
    searchspace: SearchSpace,
    batch_size: int,
) -> IntoDataFrame:
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
    candidates = nw.from_native(searchspace.discrete.get_candidates(), eager_only=True)
    combined_masks: Iterable[tuple[np.ndarray, frozenset[str]]]
    if searchspace.n_subsets <= recommender.max_n_subsets:
        combined_masks = searchspace.subsets()
    else:
        combined_masks = searchspace.sample_subsets(recommender.max_n_subsets)

    def make_callable(
        d_mask: np.ndarray,
        c_inactive_params: frozenset[str],
    ) -> Callable[[], tuple[IntoDataFrame, Tensor]]:
        def optimize() -> tuple[IntoDataFrame, Tensor]:
            import torch

            # TODO: Replace with .filter() method to avoid materialization
            mod_disc = evolve(
                searchspace.discrete,
                candidates=TableCandidates(
                    searchspace.discrete.parameters,
                    candidates.filter(d_mask.tolist()).to_native(),
                ),
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
                acqf_value = recommender._botorch_acqf(to_tensor(comp).unsqueeze(0))
            return rec, acqf_value

        return optimize

    callables = (make_callable(d_mask, c_ip) for d_mask, c_ip in combined_masks)
    best_rec, _ = recommender._optimize_over_subsets(callables)

    # Post-check minimum cardinality on continuous columns
    if subspace_c.constraints_cardinality and not is_cardinality_fulfilled(
        nw.from_native(best_rec, eager_only=True)
        .select(subspace_c.parameter_names)
        .to_pandas(),
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
