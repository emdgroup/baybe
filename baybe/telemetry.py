"""Telemetry functionality for BayBE.

The following environment variables control the behavior of BayBE telemetry:

BAYBE_TELEMETRY_ENABLED
    Flag that can turn off telemetry entirely (default is `true`). To turn it off set it
    to `false`.

BAYBE_TELEMETRY_ENDPOINT
    The receiving endpoint URL for telemetry data.

BAYBE_TELEMETRY_TIMEOUT
    The timeout in seconds for the check whether the endpoint URL is reachable.

BAYBE_TELEMETRY_USERNAME
    The name of the user executing BayBE code. Defaults to an irreversible hash of
    the username according to the OS.

BAYBE_TELEMETRY_HOSTNAME
    The name of the machine executing BayBE code. Defaults to an irreversible hash of
    the machine name.
"""
import getpass
import hashlib
import logging
import os
import socket
from typing import Dict, List, Union

from urllib.parse import urlparse

import pandas as pd

from baybe.parameters.base import Parameter
from baybe.utils import fuzzy_row_match, strtobool

_logger = logging.getLogger(__name__)

try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.metrics import get_meter, set_meter_provider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
except ImportError:
    # Failed telemetry install/import should not fail baybe, so telemetry is being
    # disabled in that case
    if strtobool(os.environ.get("BAYBE_TELEMETRY_ENABLED", "true")):
        _logger.warning(
            "Opentelemetry could not be imported, potentially it is not "
            "installed. Disabling baybe telemetry."
        )
    os.environ["BAYBE_TELEMETRY_ENABLED"] = "false"


def is_enabled() -> bool:
    """Tell whether telemetry currently is enabled.

    Telemetry can be disabled by setting the respective environment variable.
    """
    return strtobool(os.environ.get("BAYBE_TELEMETRY_ENABLED", "true"))


# Global telemetry labels
TELEM_LABELS = {
    "RECOMMENDED_MEASUREMENTS_PERCENTAGE": "value_recommended-measurements-percentage",
    "BATCH_QUANTITY": "value_batch-quantity",
    "COUNT_ADD_RESULTS": "count_add-results",
    "COUNT_RECOMMEND": "count_recommend",
    "NUM_PARAMETERS": "value_num-parameters",
    "NUM_CONSTRAINTS": "value_num-constraints",
    "COUNT_SEARCHSPACE_CREATION": "count_searchspace-created",
    "NAKED_INITIAL_MEASUREMENTS": "count_naked-initial-measurements-added",
}

# Create resources only if telemetry is activated
if is_enabled():
    _endpoint_url = os.environ.get(
        "BAYBE_TELEMETRY_ENDPOINT",
        "https://public.telemetry.baybe.p.uptimize.merckgroup.com:4317",
    )

    # Test endpoint URL
    try:
        # Parse endpoint URL
        _endpoint_url_parsed = urlparse(_endpoint_url)
        _endpoint_hostname = _endpoint_url_parsed.hostname
        _endpoint_port = _endpoint_url_parsed.port if _endpoint_url_parsed.port else 80
        try:
            _TIMEOUT_S = float(os.environ.get("BAYBE_TELEMETRY_TIMEOUT", "0.5"))
        except (ValueError, TypeError):
            _logger.warning(
                "WARNING: Value passed for environment variable BAYBE_TELEMETRY_TIMEOUT"
                " is not a valid floating point number. Using default of 0.5."
            )
            _TIMEOUT_S = 0.5

        with socket.create_connection(
            (_endpoint_hostname, _endpoint_port), timeout=_TIMEOUT_S
        ) as sock:
            _instruments = {}
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

            # Setup Global Metric Provider
            _meter = get_meter("aws-otel", "1.0")
    except Exception:
        # Catching broad exception here and disabling telemetry in that case to avoid
        # any telemetry timeouts or interference for the user in case of unexpected
        # errors. Possible ones are for instance ``socket.gaierror`` in case the user
        # has no internet connection.
        _logger.warning(
            "WARNING: BayBE Telemetry endpoint %s cannot be reached. "
            "Disabling telemetry.",
            _endpoint_url,
        )
        os.environ["BAYBE_TELEMETRY_ENABLED"] = "false"


def get_user_details() -> Dict[str, str]:
    """Generate user details.

    These are submitted as metadata with requested telemetry stats.

    Returns:
        The hostname and username in hashed format as well as the package version.
    """
    from baybe import __version__  # pylint: disable=import-outside-toplevel

    username_hash = os.environ.get("BAYBE_TELEMETRY_USERNAME", None) or (
        hashlib.sha256(getpass.getuser().upper().encode())
        .hexdigest()
        .upper()[:10]  # take only first 10 digits to enhance readability
    )
    hostname_hash = os.environ.get("BAYBE_TELEMETRY_HOSTNAME", None) or (
        hashlib.sha256(socket.gethostname().encode()).hexdigest().upper()[:10]
    )

    return {"host": hostname_hash, "user": username_hash, "version": __version__}


def telemetry_record_value(
    instrument_name: str, value: Union[bool, int, float, str]
) -> None:
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


def _submit_scalar_value(
    instrument_name: str, value: Union[bool, int, float, str]
) -> None:
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
    parameters: List[Parameter],
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
        numerical_measurements_must_be_within_tolerance: If ```True```, numerical
            parameter entries are matched with the reference elements only if there is
            a match within the parameter tolerance. If ```False```, the closest match
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
