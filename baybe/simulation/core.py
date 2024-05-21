"""Core simulation and backtesting functionality."""

from __future__ import annotations

import warnings
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from typing import Callable, Literal, Optional, Union

import numpy as np
import pandas as pd

from baybe.campaign import Campaign
from baybe.exceptions import NotEnoughPointsLeftError, NothingToSimulateError
from baybe.simulation.lookup import _look_up_target_values
from baybe.targets.enum import TargetMode
from baybe.utils.dataframe import add_parameter_noise
from baybe.utils.numerical import closer_element, closest_element
from baybe.utils.random import temporary_seed


def simulate_experiment(
    campaign: Campaign,
    lookup: Optional[Union[pd.DataFrame, Callable]] = None,
    /,
    *,
    batch_size: int = 1,
    n_doe_iterations: Optional[int] = None,
    initial_data: Optional[pd.DataFrame] = None,
    random_seed: Optional[int] = None,
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
    noise_percent: Optional[float] = None,
) -> pd.DataFrame:
    """Simulate a Bayesian optimization loop.

    The most basic type of simulation. Runs a single execution of the loop either
    for a specified number of steps or until there are no more configurations left
    to be tested.

    Args:
        campaign: The DOE setting to be simulated.
        lookup: The lookup used to close the loop, provided in the form of a dataframe
            or callable that define the targets for the queried parameter settings:
            First, a dataframe containing experimental settings and their target
            results can be chosen.
            Second, A callable, providing target values for the given parameter
            settings. can be chosen. The callable is assumed to return either a float
            or a tuple of floats and to accept an arbitrary number of floats as input.
            Finally,``None`` can be chosen, producing fake results.
        batch_size: The number of recommendations to be queried per iteration.
        n_doe_iterations:  The number of iterations to run the design-of-experiments
            loop. If not specified, the simulation proceeds until there are no more
            testable configurations left.
        initial_data: The initial measurement data to be ingested before starting the
            loop.
        random_seed: An optional random seed to be used for the simulation.
        impute_mode: Specifies how a missing lookup will be handled.
            There are six different options available.

            - ``"error"``: An error will be thrown.
            - ``"worst"``: Imputation uses the worst available value for each target.
            - ``"best"``: Imputation uses the best available value for each target.
            - ``"mean"``: Imputation uses the mean value for each target.
            - ``"random"``: A random row will be used as lookup.
            - ``"ignore"``: The search space is stripped before recommendations are made
              so that unmeasured experiments will not be recommended.
        noise_percent: If not ``None``, relative noise in percent of
            ``noise_percent`` will be applied to the parameter measurements.

    Returns:
        A dataframe ready for plotting, see the ``Note`` for details.

    Raises:
        TypeError: If a non-suitable lookup is chosen.
        ValueError: If the impute mode ``ignore`` is chosen for non-dataframe lookup.
        ValueError: If a setup is provided that would run indefinitely.

    Note:
        The returned dataframe contains the following columns:

        * ``Iteration``:
          Corresponds to the DOE iteration (starting at 0)
        * ``Num_Experiments``:
          Corresponds to the running number of experiments performed (usually x-axis)
        * for each target a column ``{targetname}_IterBest``:
          Corresponds to the best result for that target at the respective iteration
        * for each target a column ``{targetname}_CumBest``:
          Corresponds to the best result for that target up to including
          respective iteration
        * for each target a column ``{targetname}_Measurements``:
          The individual measurements obtained for the respective target and iteration
    """
    # TODO: Due to the "..." operator, sphinx does not render this properly. Might
    #   want to investigate in the future.
    # TODO: Use a `will_terminate` campaign property to decide if the campaign will
    #   run indefinitely or not, and allow omitting `n_doe_iterations` for the latter.

    context = temporary_seed(random_seed) if random_seed is not None else nullcontext()
    with context:
        #   Validate the lookup mechanism
        if not (isinstance(lookup, (pd.DataFrame, Callable)) or (lookup is None)):
            raise TypeError(
                "The lookup can either be 'None', a pandas dataframe or a callable."
            )

        # Validate the data imputation mode
        if (impute_mode == "ignore") and (not isinstance(lookup, pd.DataFrame)):
            raise ValueError(
                "Impute mode 'ignore' is only available for dataframe lookups."
            )

        # Clone the campaign to avoid mutating the original object
        # TODO: Reconsider if deepcopies are required once [16605] is resolved
        campaign = deepcopy(campaign)

        # Add the initial data
        if initial_data is not None:
            campaign.add_measurements(initial_data)

        # For impute_mode 'ignore', do not recommend space entries that are not
        # available in the lookup
        # TODO [16605]: Avoid direct manipulation of metadata
        if impute_mode == "ignore":
            exp_rep = campaign.searchspace.discrete.exp_rep
            lookup_configurations = lookup[
                [p.name for p in campaign.parameters]
            ].drop_duplicates()
            missing_inds = exp_rep.index[
                exp_rep.merge(lookup_configurations, how="left", indicator=True)[
                    "_merge"
                ]
                == "left_only"
            ]
            campaign.searchspace.discrete.metadata.loc[
                missing_inds, "dont_recommend"
            ] = True

        # Run the DOE loop
        limit = n_doe_iterations or np.inf
        k_iteration = 0
        n_experiments = 0
        dfs = []
        while k_iteration < limit:
            # Get the next recommendations and corresponding measurements
            try:
                measured = campaign.recommend(batch_size=batch_size)
            except NotEnoughPointsLeftError:
                # TODO: There can be still N < batch_quantity points left in the search
                #   space. Once the recommender/strategy refactoring is completed,
                #   find an elegant way to return those.
                warnings.warn(
                    "The simulation of the campaign ended because because not "
                    "sufficiently many points were left for recommendation",
                    UserWarning,
                )
                break

            # Temporary workaround to enable returning incomplete simulations
            except Exception as ex:
                warnings.warn(
                    f"An error has occurred during the simulation, "
                    f"therefore incomplete simulation results are returned. "
                    f"The error message was:\n{str(ex)}"
                )
                break

            n_experiments += len(measured)
            _look_up_target_values(measured, campaign, lookup, impute_mode)

            # Create the summary for the current iteration and store it
            result = pd.DataFrame(
                [  # <-- this ensures that the internal lists to not get expanded
                    {
                        "Iteration": k_iteration,
                        "Num_Experiments": n_experiments,
                        **{
                            f"{target.name}_Measurements": measured[
                                target.name
                            ].to_list()
                            for target in campaign.targets
                        },
                    }
                ]
            )
            dfs.append(result)

            # Apply optional noise to the parameter measurements
            if noise_percent:
                add_parameter_noise(
                    measured,
                    campaign.parameters,
                    noise_type="relative_percent",
                    noise_level=noise_percent,
                )

            # Update the campaign
            campaign.add_measurements(measured)

            # Update the iteration counter
            k_iteration += 1

        # Collect the iteration results
        if len(dfs) == 0:
            raise NothingToSimulateError()
        results = pd.concat(dfs, ignore_index=True)

        # Add the instantaneous and running best values for all targets
        for target in campaign.targets:
            # Define the summary functions for the current target
            if target.mode is TargetMode.MAX:
                agg_fun = np.max
                cum_fun = np.maximum.accumulate
            elif target.mode is TargetMode.MIN:
                agg_fun = np.min
                cum_fun = np.minimum.accumulate
            elif target.mode is TargetMode.MATCH:
                match_val = target.bounds.center
                agg_fun = partial(closest_element, target=match_val)
                cum_fun = lambda x: np.array(  # noqa: E731
                    np.frompyfunc(
                        partial(closer_element, target=match_val),
                        2,
                        1,
                    ).accumulate(x),
                    dtype=float,
                )

            # Add the summary columns
            measurement_col = f"{target.name}_Measurements"
            iterbest_col = f"{target.name}_IterBest"
            cumbest_cols = f"{target.name}_CumBest"
            results[iterbest_col] = results[measurement_col].apply(agg_fun)
            results[cumbest_cols] = cum_fun(results[iterbest_col])

        return results
