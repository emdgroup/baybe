"""Telemetry functionality for BayBE.

For more details, see https://emdgroup.github.io/baybe/stable/userguide/envvars.html#telemetry
"""

import getpass
import hashlib
import os
import socket
import warnings
from collections.abc import Sequence
from urllib.parse import urlparse

import pandas as pd

from baybe.parameters.base import Parameter
from baybe.utils.boolean import strtobool
from baybe.utils.dataframe import fuzzy_row_match

# Telemetry environment variable names
VARNAME_TELEMETRY_ENABLED = "BAYBE_TELEMETRY_ENABLED"
VARNAME_TELEMETRY_ENDPOINT = "BAYBE_TELEMETRY_ENDPOINT"
VARNAME_TELEMETRY_VPN_CHECK = "BAYBE_TELEMETRY_VPN_CHECK"
VARNAME_TELEMETRY_VPN_CHECK_TIMEOUT = "BAYBE_TELEMETRY_VPN_CHECK_TIMEOUT"
VARNAME_TELEMETRY_USERNAME = "BAYBE_TELEMETRY_USERNAME"
VARNAME_TELEMETRY_HOSTNAME = "BAYBE_TELEMETRY_HOSTNAME"

# Telemetry settings defaults
DEFAULT_TELEMETRY_ENABLED = "true"
DEFAULT_TELEMETRY_ENDPOINT = (
    "https://public.telemetry.baybe.p.uptimize.merckgroup.com:4317"
)
DEFAULT_TELEMETRY_VPN_CHECK = "true"
DEFAULT_TELEMETRY_VPN_CHECK_TIMEOUT = "0.5"


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

# Attempt telemetry import
try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.metrics import Histogram, get_meter, set_meter_provider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
except ImportError:
    # Failed telemetry install/import should not fail baybe, so telemetry is being
    # disabled in that case
    if strtobool(os.environ.get(VARNAME_TELEMETRY_ENABLED, DEFAULT_TELEMETRY_ENABLED)):
        warnings.warn(
            "Opentelemetry could not be imported, potentially it is not installed. "
            "Disabling baybe telemetry.",
            UserWarning,
        )
    os.environ[VARNAME_TELEMETRY_ENABLED] = "false"


def is_enabled() -> bool:
    """Tell whether telemetry currently is enabled.

    Telemetry can be disabled by setting the respective environment variable.
    """
    return strtobool(
        os.environ.get(VARNAME_TELEMETRY_ENABLED, DEFAULT_TELEMETRY_ENABLED)
    )


# Attempt telemetry initialization
if is_enabled():
    # Assign default user and machine name
    try:
        DEFAULT_TELEMETRY_USERNAME = (
            hashlib.sha256(getpass.getuser().upper().encode()).hexdigest().upper()[:10]
        )
    except ModuleNotFoundError:
        # getpass.getuser() does not work on Windows if all the environment variables
        # it checks are empty. Since then there is no way of inferring the username, we
        # use UNKNOWN as fallback.
        DEFAULT_TELEMETRY_USERNAME = "UNKNOWN"

    DEFAULT_TELEMETRY_HOSTNAME = (
        hashlib.sha256(socket.gethostname().encode()).hexdigest().upper()[:10]
    )

    _endpoint_url = os.environ.get(
        VARNAME_TELEMETRY_ENDPOINT, DEFAULT_TELEMETRY_ENDPOINT
    )

    # Test endpoint URL
    try:
        # Parse endpoint URL
        _endpoint_url_parsed = urlparse(_endpoint_url)
        _endpoint_hostname = _endpoint_url_parsed.hostname
        _endpoint_port = _endpoint_url_parsed.port if _endpoint_url_parsed.port else 80
        try:
            _TIMEOUT_S = float(
                os.environ.get(
                    VARNAME_TELEMETRY_VPN_CHECK_TIMEOUT,
                    DEFAULT_TELEMETRY_VPN_CHECK_TIMEOUT,
                )
            )
        except (ValueError, TypeError):
            warnings.warn(
                f"WARNING: Value passed for environment variable "
                f"{VARNAME_TELEMETRY_VPN_CHECK_TIMEOUT} is not a valid floating point "
                f"number. Using default of {DEFAULT_TELEMETRY_VPN_CHECK_TIMEOUT}.",
                UserWarning,
            )
            _TIMEOUT_S = float(DEFAULT_TELEMETRY_VPN_CHECK_TIMEOUT)

        # Send a test request. If there is no internet connection or a firewall is
        # present this will throw an error and telemetry will be deactivated.
        if strtobool(
            os.environ.get(VARNAME_TELEMETRY_VPN_CHECK, DEFAULT_TELEMETRY_VPN_CHECK)
        ):
            socket.gethostbyname("verkehrsnachrichten.merck.de")

        # User has connectivity to the telemetry endpoint, so we initialize
        _instruments: dict[str, Histogram] = {}
        _resource = Resource.create(
            {"service.namespace": "BayBE", "service.name": "SDK"}
        )
        _reader = PeriodicExportingMetricReader(
            exporter=OTLPMetricExporter(
                endpoint=_endpoint_url,
                insecure=True,
            )
        )
        _provider = MeterProvider(resource=_resource, metric_readers=[_reader])
        set_meter_provider(_provider)
        _meter = get_meter("aws-otel", "1.0")
    except Exception as ex:
        # Catching broad exception here and disabling telemetry in that case to avoid
        # any telemetry timeouts or interference for the user in case of unexpected
        # errors. Possible ones are for instance ``socket.gaierror`` in case the user
        # has no internet connection.
        if os.environ.get(VARNAME_TELEMETRY_USERNAME, "").startswith("DEV_"):
            # This warning is only printed for developers to make them aware of
            # potential issues
            warnings.warn(
                f"WARNING: BayBE Telemetry endpoint {_endpoint_url} cannot be reached. "
                "Disabling telemetry. The exception encountered was: "
                f"{type(ex).__name__}, {ex}",
                UserWarning,
            )
        os.environ[VARNAME_TELEMETRY_ENABLED] = "false"
else:
    DEFAULT_TELEMETRY_USERNAME = "UNKNOWN"
    DEFAULT_TELEMETRY_HOSTNAME = "UNKNOWN"


def get_user_details() -> dict[str, str]:
    """Generate user details.

    These are submitted as metadata with requested telemetry stats.

    Returns:
        The hostname and username in hashed format as well as the package version.
    """
    from baybe import __version__

    username_hash = os.environ.get(
        VARNAME_TELEMETRY_USERNAME, DEFAULT_TELEMETRY_USERNAME
    )
    hostname_hash = os.environ.get(
        VARNAME_TELEMETRY_HOSTNAME, DEFAULT_TELEMETRY_HOSTNAME
    )

    return {"host": hostname_hash, "user": username_hash, "version": __version__}


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
        _submit_scalar_value(instrument_name, value)


def _submit_scalar_value(instrument_name: str, value: int | float) -> None:
    """See :func:`baybe.telemetry.telemetry_record_value`."""
    if instrument_name in _instruments:
        histogram = _instruments[instrument_name]
    else:
        histogram = _meter.create_histogram(
            instrument_name,
            description=f"Histogram for instrument {instrument_name}",
        )
        _instruments[instrument_name] = histogram
    histogram.record(value, get_user_details())


def telemetry_record_recommended_measurement_percentage(
    cached_recommendation: pd.DataFrame,
    measurements: pd.DataFrame,
    parameters: Sequence[Parameter],
    numerical_measurements_must_be_within_tolerance: bool,
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
        numerical_measurements_must_be_within_tolerance: If ``True``, numerical
            parameter entries are matched with the reference elements only if there is
            a match within the parameter tolerance. If ``False``, the closest match
            is considered, irrespective of the distance.
    """
    if is_enabled():
        if len(cached_recommendation) > 0:
            recommended_measurements_percentage = (
                len(
                    fuzzy_row_match(
                        cached_recommendation,
                        measurements,
                        parameters,
                        numerical_measurements_must_be_within_tolerance,
                    )
                )
                / len(cached_recommendation)
                * 100.0
            )
            _submit_scalar_value(
                TELEM_LABELS["RECOMMENDED_MEASUREMENTS_PERCENTAGE"],
                recommended_measurements_percentage,
            )
        else:
            _submit_scalar_value(
                TELEM_LABELS["NAKED_INITIAL_MEASUREMENTS"],
                1,
            )
