"""
Telemetry  functionality for BayBE.
"""
import getpass
import hashlib
import os
import socket

from opentelemetry._metrics import get_meter, set_meter_provider
from opentelemetry.exporter.otlp.proto.grpc._metric_exporter import OTLPMetricExporter
from opentelemetry.sdk._metrics import MeterProvider
from opentelemetry.sdk._metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

_resource = Resource.create({"service.name": "baybe.sdk", "service.namespace": "baybe"})
_reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(
        "***REMOVED***"
        ".elb.eu-central-1.amazonaws.com:4317",
        True,
    )
)
_provider = MeterProvider(resource=_resource, metric_readers=[_reader])
set_meter_provider(_provider)

# Setup Global Metric Provider
_meter = get_meter("aws-otel", "1.0")


def get_user_hash() -> str:
    """
    Generate a unique hash value for the current user based on the host name and
    uppercase username, e.g. hash of 'LTD1234M123132'.

    Returns
    -------
        str
    """
    return hashlib.sha256(
        (socket.gethostname() + getpass.getuser().upper()).encode()
    ).hexdigest()
    # Alternatively one could take the MAC address like hex(uuid.getnode())


def is_enabled() -> bool:
    """
    Tells whether telemetry currently is enabled. Telemetry can be disabled by setting
    the respective environment variable.

    Returns
    -------
        bool
    """
    return os.environ.get("BAYBE_TELEMETRY_ENABLED", "").lower() in [
        "false",
        "no",
        "off",
        "0",
    ]
