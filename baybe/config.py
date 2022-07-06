"""
Config functionality
"""

import logging

from baybe import parameters, targets


log = logging.getLogger(__name__)


def parse_config(config: dict) -> tuple:
    """
    :param config_file: file string to a json file
    :return: tuple (list of parameters, list of targets)
    """

    # Parameters
    params = []
    for param in config.get("Parameters", []):
        params.append(parameters.parse_parameter(param))

    # Targets
    objective = config.get("Objective", {})
    mode = objective.get("Mode", None)
    if mode == "SINGLE":
        objectives = objective.get("Objectives", [])
        if len(objectives) != 1:
            raise ValueError(
                f"Config with objective mode SINGLE must specify exactly one target, "
                f"but specified several or none: {objectives}"
            )

        target_dict = objectives[0]
        targs = [targets.parse_single_target(target_dict)]
    else:
        raise ValueError(
            f"Objective mode is {mode}, but must be one of {targets.allowed_modes}"
        )

    return params, targs
