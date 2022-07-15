"""
Config functionality
"""

import logging
from typing import Tuple

from baybe import parameters, targets


log = logging.getLogger(__name__)

# Allowed options and their default values
allowed_config_options = {
    "project_name": "Unnamed Project",
    "random_seed": 1337,
    "allow_repeated_recommendations": True,
    "allow_recommending_already_measured": True,
    "numerical_measurements_must_be_within_tolerance": True,
}


def parse_config(config: dict) -> Tuple[list, list]:
    """
    Parses a BayBE config dictionary. Also sets default values for various flags and
    options directly in the config variable.

    Parameters
    ----------
    config : dict
        A dictionary containing parameter and target info as well as flags and options
        for the method.

    Returns
    -------
    2-Tuple with lists for parsed parameters and targets. The config parameter could
    also be altered because it is assured that all flags and options are set to default
    values
    """

    if ("objective" not in config.keys()) or ("parameters" not in config.keys()):
        raise AssertionError("Your config must define 'parameters' and 'objective'")

    # Parameters
    params = []
    for param in config["parameters"]:
        params.append(parameters.parse_parameter(param))

    # Objective
    objective = config["objective"]
    mode = objective.get("mode", None)
    if mode == "SINGLE":
        targs_dict = objective.get("targets", [])
        if len(targs_dict) != 1:
            raise ValueError(
                f"Config with objective mode SINGLE must specify exactly one target, "
                f"but specified several or none: {targs_dict}"
            )

        target_dict = targs_dict[0]
        targs = [targets.parse_single_target(target_dict)]
    elif mode == "MULTI_DESIRABILITY":
        raise NotImplementedError("This objective mode is not implemented yet")
    elif mode == "MULTI_PARETO":
        raise NotImplementedError("This objective mode is not implemented yet")
    elif mode == "MULTI_TASK":
        raise NotImplementedError("This objective mode is not implemented yet")
    else:
        raise ValueError(
            f"Objective mode is {mode}, but must be one of {targets.allowed_modes}"
        )

    # Options
    for option, value in allowed_config_options.items():
        config.setdefault(option, value)

    # Check for unknown options
    unrecognized_options = [
        key
        for key in config.keys()
        if key not in (list(allowed_config_options) + ["parameters", "objective"])
    ]
    if len(unrecognized_options) > 0:
        raise AssertionError(
            f"The provided config option(s) '{unrecognized_options}'"
            f" is/are not in the allowed "
            f"options {list(allowed_config_options)}"
        )

    return params, targs
