"""Telemetry functionality for BayBE.

Important:
    BayBE collects anonymous usage statistics **only** for employees of Merck KGaA,
    Darmstadt, Germany and/or its affiliates. The recording of metrics is turned off
    for all other users and impossible due to a VPN block. In any case, the usage
    statistics do **not** involve logging of recorded measurements, targets or any
    project information that would allow for reconstruction of details. The user and
    host machine names are irreversibly anonymized.

**Monitored quantities are:**
    * ``batch_quantity`` used when querying recommendations
    * Number of parameters in the search space
    * Number of constraints in the search space
    * How often ``recommend`` was called
    * How often ``add_measurements`` was called
    * How often a search space is newly created
    * How often initial measurements are added before recommendations were calculated
      ("naked initial measurements")
    * The fraction of measurements added that correspond to previous recommendations
    * Each measurement is associated with an irreversible hash of the user- and hostname

**The following environment variables control the behavior of BayBE telemetry:**

``BAYBE_TELEMETRY_ENABLED``
    Flag that can turn off telemetry entirely (default is `true`). To turn it off set it
    to `false`.

``BAYBE_TELEMETRY_ENDPOINT``
    The receiving endpoint URL for telemetry data.

``BAYBE_TELEMETRY_VPN_CHECK``
    Flag turning an initial telemetry connectivity check on/off (default is `true`).

``BAYBE_TELEMETRY_VPN_CHECK_TIMEOUT``
    The timeout in seconds for the check whether the endpoint URL is reachable.

``BAYBE_TELEMETRY_USERNAME``
    The name of the user executing BayBE code. Defaults to an irreversible hash of
    the username according to the OS.

``BAYBE_TELEMETRY_HOSTNAME``
    The name of the machine executing BayBE code. Defaults to an irreversible hash of
    the machine name.

If you wish to disable logging, you can set the following environment variable:

.. code-block:: console

    export BAYBE_TELEMETRY_ENABLED=false

or in Python:

.. code-block:: python

    import os
    os.environ["BAYBE_TELEMETRY_ENABLED"] = "false"

before calling any BayBE functionality.

Telemetry can be re-enabled by simply removing the variable:

.. code-block:: console

    unset BAYBE_TELEMETRY_ENABLED

or in Python:

.. code-block:: python

    os.environ.pop["BAYBE_TELEMETRY_ENABLED"]

Note, however, that (un-)setting the variable in the shell will not affect the running
Python session.
"""
import getpass
import hashlib
import logging
import os
import socket
from typing import Dict, List, Union
from urllib.parse import urlparse

import pandas as pd
import requests

from baybe.parameters.base import Parameter
from baybe.utils import fuzzy_row_match, strtobool

_logger = logging.getLogger(__name__)

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
DEFAULT_TELEMETRY_USERNAME = (
    hashlib.sha256(getpass.getuser().upper().encode()).hexdigest().upper()[:10]
)  # this hash is irreversible and cannot identify the user or their machine
DEFAULT_TELEMETRY_HOSTNAME = (
    hashlib.sha256(socket.gethostname().encode()).hexdigest().upper()[:10]
)  # this hash is irreversible and cannot identify the user or their machine

# Telemetry labels for metrics
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

# Attempt telemetry import
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
    if strtobool(os.environ.get(VARNAME_TELEMETRY_ENABLED, DEFAULT_TELEMETRY_ENABLED)):
        _logger.warning(
            "Opentelemetry could not be imported, potentially it is not "
            "installed. Disabling baybe telemetry."
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
            _logger.warning(
                "WARNING: Value passed for environment variable %s"
                " is not a valid floating point number. Using default of %s.",
                VARNAME_TELEMETRY_VPN_CHECK_TIMEOUT,
                DEFAULT_TELEMETRY_VPN_CHECK_TIMEOUT,
            )
            _TIMEOUT_S = float(DEFAULT_TELEMETRY_VPN_CHECK_TIMEOUT)

        # Send a test request. If there is no internet connection or a firewall is
        # present this will throw an error and telemetry will be deactivated.
        if strtobool(
            os.environ.get(VARNAME_TELEMETRY_VPN_CHECK, DEFAULT_TELEMETRY_VPN_CHECK)
        ):
            response = requests.get(
                "http://verkehrsnachrichten.merck.de/", timeout=_TIMEOUT_S
            )
            if response.status_code != 200:
                raise requests.RequestException("Cannot reach telemetry network.")

        # User has connectivity to the telemetry endpoint, so we initialize
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
        _meter = get_meter("aws-otel", "1.0")
    except Exception:
        # Catching broad exception here and disabling telemetry in that case to avoid
        # any telemetry timeouts or interference for the user in case of unexpected
        # errors. Possible ones are for instance ``socket.gaierror`` in case the user
        # has no internet connection.
        _logger.info(
            "WARNING: BayBE Telemetry endpoint %s cannot be reached. "
            "Disabling telemetry.",
            _endpoint_url,
        )
        os.environ[VARNAME_TELEMETRY_ENABLED] = "false"


def get_user_details() -> Dict[str, str]:
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
