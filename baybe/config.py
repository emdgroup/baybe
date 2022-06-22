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
            log.error(
                "Config with objective mode SINGLE must specify exactly one target"
            )
            params = []
            targ = None
        else:
            target_dict = objectives[0]
            targ = targets.parse_single_target(target_dict)
    else:
        log.error(
            "Objective mode is %s, but must be one of %s", mode, targets.allowed_modes
        )
        params = []
        targ = None

    return params, targ
