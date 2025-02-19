"""Telemetry functionality for BayBE.

For more details, see https://emdgroup.github.io/baybe/stable/userguide/envvars.html#telemetry
"""

from baybe.telemetry.api import (
    TELEMETRY_LABELS,
    VARNAME_TELEMETRY_ENABLED,
    VARNAME_TELEMETRY_HOSTNAME,
    VARNAME_TELEMETRY_USERNAME,
    telemetry_record_recommended_measurement_percentage,
    telemetry_record_value,
)

__all__ = [
    "TELEMETRY_LABELS",
    "VARNAME_TELEMETRY_ENABLED",
    "VARNAME_TELEMETRY_HOSTNAME",
    "VARNAME_TELEMETRY_USERNAME",
    "telemetry_record_recommended_measurement_percentage",
    "telemetry_record_value",
]
