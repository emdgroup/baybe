"""
Config functionality
"""

import logging
from typing import Tuple

from baybe import parameters, targets


log = logging.getLogger(__name__)

# dictionary for storing the allowed options and their default values
allowed_config_options = {
    "Project_Name": "Unnamed Project",
    "Random_Seed": 1337,
    "Allow_repeated_recommendations": True,
    "Allow_recommending_already_measured": True,
    "Num_measurements_must_be_within_tolerance": True,
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
    2-Tuple with lists for parsed parameters and targets. The config parameter can also
    be altered because it is assured that all flags and options are set to default
    values
    """

    if ("Objective" not in config.keys()) or ("Parameters" not in config.keys()):
        raise AssertionError("Your config must define 'Parameters' and 'Objective'")

    # Parameters
    params = []
    for param in config["Parameters"]:
        params.append(parameters.parse_parameter(param))

    # Objective
    objective = config["Objective"]
    mode = objective.get("Mode", None)
    if mode == "SINGLE":
        targs_dict = objective.get("Targets", [])
        if len(targs_dict) != 1:
            raise ValueError(
                f"Config with objective mode SINGLE must specify exactly one target, "
                f"but specified several or none: {targs_dict}"
            )

        target_dict = targs_dict[0]
        targs = [targets.parse_single_target(target_dict)]
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
        if key not in (list(allowed_config_options) + ["Parameters", "Objective"])
    ]
    if len(unrecognized_options) > 0:
        raise AssertionError(
            f"The provided config option(s) '{unrecognized_options}'"
            f" is/are not in the allowed "
            f"options {list(allowed_config_options)}"
        )

    return params, targs
