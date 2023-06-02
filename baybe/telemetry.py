"""
Telemetry  functionality for BayBE.
"""
import getpass
import hashlib

import os
import socket
from typing import Dict, List, Union

import pandas as pd
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.metrics import get_meter, set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

from baybe import __version__

from .parameters import Parameter
from .utils import fuzzy_row_match, strtobool


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

_instruments = {}
_resource = Resource.create({"service.namespace": "BayBE", "service.name": "SDK"})
_reader = PeriodicExportingMetricReader(
    exporter=OTLPMetricExporter(
        endpoint="***REMOVED***.elb."
        "eu-central-1.amazonaws.com:4317",
        insecure=True,
    )
)
_provider = MeterProvider(resource=_resource, metric_readers=[_reader])
set_meter_provider(_provider)

# Setup Global Metric Provider
_meter = get_meter("aws-otel", "1.0")


def get_user_details() -> Dict[str, str]:
    """
    Generate user details that are submitted as metadata with requested telemetry stats.

    Returns
    -------
        dict: Contains the hostname and username in hashed format as well as the package
         version
    """
    username_hash = os.environ.get("BAYBE_DEBUG_FAKE_USERHASH", None) or (
        hashlib.sha256(getpass.getuser().upper().encode())
        .hexdigest()
        .upper()[:10]  # take only first 10 digits to enhance readability in dashboard
    )
    hostname_hash = os.environ.get("BAYBE_DEBUG_FAKE_HOSTHASH", None) or (
        hashlib.sha256(socket.gethostname().encode()).hexdigest().upper()[:10]
    )
    # Alternatively one could take the MAC address like hex(uuid.getnode())

    return {"host": hostname_hash, "user": username_hash, "version": __version__}


def is_enabled() -> bool:
    """
    Tells whether telemetry currently is enabled. Telemetry can be disabled by setting
    the respective environment variable.

    Returns
    -------
        bool
    """
    return strtobool(os.environ.get("BAYBE_TELEMETRY_ENABLED", "true"))


def telemetry_record_value(
    instrument_name: str, value: Union[bool, int, float, str]
) -> None:
    """
    Transmits a given value under a given label to the telemetry backend. The values are
     recorded as histograms, i.e. the info about record time and sample size is also
     available. This can be used to count function calls (record the value 1) or
     statistics about any variable (record its value). Due to serialization limitations
     only certain data types of value are allowed.

    Parameters
    ----------
    instrument_name: str
        The label under which this statistic is logged.
    value
        The value of the statistic to be logged.

    Returns
    -------
        None
    """
    if is_enabled():
        _submit_scalar_value(instrument_name, value)


def _submit_scalar_value(
    instrument_name: str, value: Union[bool, int, float, str]
) -> None:
    """
    See telemetry_record_value.
    """
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
    """
    Submits the percentage of added measurements that correspond to previously
    recommended ones (called cached recommendations). The matching is performed via
    fuzzy row matching. The calculation is only performed if telemetry is enabled. If
    no cached recommendation exists the percentage is not calculated and instead a
    different event ('naked initial measurement added') is recorded.

    Parameters
    ----------
    cached_recommendation: pd.DataFrame
        The cached recommendations.
    measurements: pd.DataFrame
        The measurements which are supposed to be checked against cached
        recommendations.
    parameters: List of BayBE parameters
        The list of parameters spanning the entire searchspace.
    numerical_measurements_must_be_within_tolerance: bool
        If True, numerical parameter entries are matched with the reference elements
        only if there is a match within the parameter tolerance. If False,
        the closest match is considered, irrespective of the distance.

    Returns
    -------
        None
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
