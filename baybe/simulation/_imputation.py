"""Target value imputation functionality."""

from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd

from baybe.targets import NumericalTarget


def impute_target_values(
    targets: Sequence[NumericalTarget],
    lookup: pd.DataFrame,
    mode: Literal["best", "worst", "mean", "random"],
) -> pd.Series:
    """Compute imputation values for the given targets from a lookup dataframe.

    Args:
        targets: The :class:`~baybe.targets.numerical.NumericalTarget` objects for which
            to compute the imputation values.
        lookup: The lookup dataframe.
        mode: The used impute mode.
            See :func:`baybe.simulation.scenarios.simulate_scenarios` for details.

    Returns:
        A series containing imputation values for all targets.

    Raises:
        ValueError: If the given imputation mode is not supported.
        ValueError: If the lookup dataframe is missing columns for any of the targets.
    """
    names = [t.name for t in targets]

    if missing := set(names) - set(lookup.columns):
        raise ValueError(
            f"The lookup dataframe is missing columns for "
            f"the following targets: {missing}"
        )

    if mode == "random":
        return lookup[names].sample(1).iloc[0]

    if mode == "mean":
        return lookup[names].mean(axis=0)

    if mode in ("best", "worst"):
        values: dict[str, Any] = {}
        for target in targets:
            transformed = target.transform(lookup[target.name])

            # Shuffle for random tie-breaking
            # TODO: Add option to control how ties are handled (e.g. random, first, all)
            transformed = transformed.sample(frac=1, replace=False)

            opt_idx = transformed.idxmax() if mode == "best" else transformed.idxmin()
            values[target.name] = lookup.loc[opt_idx, target.name]
        return pd.Series(values)

    raise ValueError(f"Unsupported imputation mode: {mode}")
