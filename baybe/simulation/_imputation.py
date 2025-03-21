"""Target value imputation functionality."""

from typing import Literal

import numpy as np
import pandas as pd

from baybe.targets import NumericalTarget
from baybe.targets.enum import TargetMode


def _impute_lookup(
    row: pd.Series,
    lookup: pd.DataFrame,
    targets: list[NumericalTarget],
    mode: Literal["error", "best", "worst", "mean", "random"] = "error",
) -> np.ndarray:
    """Perform data imputation for missing lookup values.

    Depending on the chosen mode, this might raise errors instead.

    Args:
        row: The data that should be matched with the lookup dataframe.
        lookup: The lookup dataframe.
        targets: The campaign targets, providing the required mode information.
        mode: The used impute mode.
            See :func:`baybe.simulation.scenarios.simulate_scenarios` for details.

    Returns:
        The filled-in lookup results.

    Raises:
        IndexError: If the mode ``"error"`` is chosen and at least one of the targets
            could not be found.
    """
    # TODO: this function needs another code cleanup and refactoring

    target_names = [t.name for t in targets]
    if mode == "mean":
        match_vals = lookup.loc[:, target_names].mean(axis=0).values
    elif mode == "worst":
        worst_vals = []
        for target in targets:
            if target.mode is TargetMode.MAX:
                worst_vals.append(lookup.loc[:, target.name].min().flatten()[0])
            elif target.mode is TargetMode.MIN:
                worst_vals.append(lookup.loc[:, target.name].max().flatten()[0])
            if target.mode is TargetMode.MATCH:
                worst_vals.append(
                    lookup.loc[
                        lookup.loc[
                            (lookup[target.name] - target.bounds.center).abs().idxmax(),
                        ],
                        target.name,
                    ].flatten()[0]
                )
        match_vals = np.array(worst_vals)
    elif mode == "best":
        best_vals = []
        for target in targets:
            if target.mode is TargetMode.MAX:
                best_vals.append(lookup.loc[:, target.name].max().flatten()[0])
            elif target.mode is TargetMode.MIN:
                best_vals.append(lookup.loc[:, target.name].min().flatten()[0])
            if target.mode is TargetMode.MATCH:
                best_vals.append(
                    lookup.loc[
                        lookup.loc[
                            (lookup[target.name] - target.bounds.center).abs().idxmin(),
                        ],
                        target.name,
                    ].flatten()[0]
                )
        match_vals = np.array(best_vals)
    elif mode == "random":
        vals = []
        randindex = np.random.choice(lookup.index)
        for target in targets:
            vals.append(lookup.loc[randindex, target.name].flatten()[0])
        match_vals = np.array(vals)
    else:
        raise IndexError(
            f"Cannot match the recommended row {row} to any of "
            f"the rows in the lookup."
        )

    return match_vals
