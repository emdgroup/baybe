"""Generality recommendation logic for BayesianRecommender (CurryBO algorithm)."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.parameters.categorical import GeneralityParameter
    from baybe.recommenders.pure.bayesian.core import BayesianRecommender
    from baybe.searchspace import SearchSpace


class _Context(NamedTuple):
    """Context values and column mapping for the generality parameter."""

    w_values_comp: Tensor
    """All context values in comp-rep."""

    x_col_indices: list[int]
    """Indices of X columns in the full comp-rep."""

    w_col_indices: list[int]
    """Indices of W columns in the full comp-rep."""

    gen_param: GeneralityParameter
    """The generality parameter object."""


def recommend_generality(
    recommender: BayesianRecommender,
    searchspace: SearchSpace,
    batch_size: int,
    measurements: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Two-step generality recommendation.

    Step 1: Optimize acquisition function over x-space using a
        model wrapper that aggregates predictions across all contexts.
    Step 2: Select which context to measure via posterior uncertainty.

    Args:
        recommender: The BayesianRecommender instance.
        searchspace: The full search space (possibly pre-filtered by campaign).
        batch_size: Number of points to recommend.
        measurements: The preprocessed measurements.

    Returns:
        DataFrame with batch_size rows of recommended experiments.
    """
    from botorch.acquisition.analytic import PosteriorStandardDeviation
    from botorch.acquisition.objective import IdentityMCObjective

    from baybe.acquisition.base import _get_botorch_acqf_class
    from baybe.surrogates.generality import _GeneralityModel
    from baybe.utils.basic import match_attributes

    gen_param = searchspace._generality_parameter

    objective = recommender._objective
    base_model = recommender._surrogate_model.to_botorch()

    x_subspace, x_col_indices, w_col_indices, w_values = (
        searchspace._split_by_generality()
    )
    w_ctx = _Context(
        w_values_comp=w_values,
        x_col_indices=x_col_indices,
        w_col_indices=w_col_indices,
        gen_param=gen_param,
    )

    target_transform = objective.to_botorch()

    gen_model = _GeneralityModel(
        base_model=base_model,
        w_values_comp=w_ctx.w_values_comp,
        x_col_indices=w_ctx.x_col_indices,
        w_col_indices=w_ctx.w_col_indices,
        aggregation=gen_param.aggregation,
        target_transform=target_transform,
    )

    acqf_obj = recommender._get_acquisition_function(objective)
    botorch_acqf_cls = _get_botorch_acqf_class(type(acqf_obj))

    acqf_kwargs: dict = dict(
        model=gen_model,
        objective=IdentityMCObjective(),
    )

    if not objective.is_multi_output:
        from botorch.acquisition.analytic import PosteriorMean

        _, best_f_scores = recommender.optimizer(
            1, PosteriorMean(model=gen_model), x_subspace
        )
        acqf_kwargs["best_f"] = best_f_scores.item()

    user_attrs, _ = match_attributes(
        acqf_obj,
        botorch_acqf_cls.__init__,
        strict=False,
        ignore=acqf_obj._non_botorch_attrs,
    )
    acqf_kwargs.update(user_attrs)

    sig = inspect.signature(botorch_acqf_cls).parameters
    botorch_acqf = botorch_acqf_cls(
        **{k: v for k, v in acqf_kwargs.items() if k in sig}
    )

    x_points, _ = recommender.optimizer(batch_size, botorch_acqf, x_subspace)

    w_acqf = PosteriorStandardDeviation(model=base_model)

    rows: list[dict] = []
    for i in range(batch_size):
        x_comp = x_points[i]
        x_exp_row = x_subspace._comp_rep_to_exp_rep(
            dict(zip(x_subspace.comp_rep_columns, x_comp.tolist()))
        )

        w_searchspace = searchspace._fix_parameters(**x_exp_row)
        w_point, _ = recommender.optimizer(1, w_acqf, w_searchspace)
        w_exp_row = w_searchspace._comp_rep_to_exp_rep(
            dict(zip(w_searchspace.comp_rep_columns, w_point[0].tolist()))
        )

        entry = dict(x_exp_row)
        entry[gen_param.name] = w_exp_row[gen_param.name]
        rows.append(entry)

    return pd.DataFrame(rows)
