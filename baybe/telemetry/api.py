"""The telemetry API accessible from within BayBE code."""

import os
from collections.abc import Sequence

import pandas as pd

from baybe.parameters.base import Parameter
from baybe.utils.boolean import strtobool
from baybe.utils.dataframe import fuzzy_row_match

# Telemetry labels for metrics
TELEM_LABELS = {
    "RECOMMENDED_MEASUREMENTS_PERCENTAGE": "value_recommended-measurements-percentage",
    "BATCH_SIZE": "value_batch-size",
    "COUNT_ADD_RESULTS": "count_add-results",
    "COUNT_RECOMMEND": "count_recommend",
    "NUM_PARAMETERS": "value_num-parameters",
    "NUM_CONSTRAINTS": "value_num-constraints",
    "COUNT_SEARCHSPACE_CREATION": "count_searchspace-created",
    "NAKED_INITIAL_MEASUREMENTS": "count_naked-initial-measurements-added",
}

# Telemetry environment variable names
VARNAME_TELEMETRY_ENABLED = "BAYBE_TELEMETRY_ENABLED"
VARNAME_TELEMETRY_HOSTNAME = "BAYBE_TELEMETRY_HOSTNAME"
VARNAME_TELEMETRY_USERNAME = "BAYBE_TELEMETRY_USERNAME"

# Telemetry settings defaults
DEFAULT_TELEMETRY_ENABLED = "true"


def is_enabled() -> bool:
    """Tell whether telemetry currently is enabled.

    Telemetry can be disabled by setting the respective environment variable.
    """
    return strtobool(
        os.environ.get(VARNAME_TELEMETRY_ENABLED, DEFAULT_TELEMETRY_ENABLED)
    )


def telemetry_record_value(instrument_name: str, value: int | float) -> None:
    """Transmit a given value under a given label to the telemetry backend.

    The values are recorded as histograms, i.e. the info about record time and sample
    size is also available. This can be used to count function calls (record the
    value 1) or statistics about any variable (record its value). Due to serialization
    limitations only certain data types of value are allowed.

    Args:
        instrument_name: The label under which this statistic is logged.
        value: The value of the statistic to be logged.
    """
    if is_enabled():
        from baybe.telemetry._telemetry import transmission_queue

        transmission_queue.put((instrument_name, value))


def telemetry_record_recommended_measurement_percentage(
    cached_recommendation: pd.DataFrame,
    measurements: pd.DataFrame,
    parameters: Sequence[Parameter],
) -> None:
    """Submit the percentage of added measurements.

    More precisely, submit the percentage of added measurements that correspond to
    previously recommended ones (called cached recommendations).

    The matching is performed via fuzzy row matching, using the utils function
    :func:`baybe.utils.dataframe.fuzzy_row_match`. The calculation is only performed
    if telemetry is enabled. If no cached recommendation exists the percentage is not
    calculated and instead a different event ('naked initial measurement added') is
    recorded.

    Args:
        cached_recommendation: The cached recommendations.
        measurements: The measurements which are supposed to be checked against cached
            recommendations.
        parameters: The list of parameters spanning the entire search space.
    """
    if is_enabled():
        from baybe.telemetry._telemetry import transmission_queue

        if cached_recommendation.empty:
            transmission_queue.put((TELEM_LABELS["NAKED_INITIAL_MEASUREMENTS"], 1))
        else:
            recommended_measurements_percentage = (
                len(fuzzy_row_match(cached_recommendation, measurements, parameters))
                / len(cached_recommendation)
                * 100.0
            )
            transmission_queue.put(
                (
                    TELEM_LABELS["RECOMMENDED_MEASUREMENTS_PERCENTAGE"],
                    recommended_measurements_percentage,
                )
            )
