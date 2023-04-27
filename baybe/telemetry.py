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

resource = Resource.create({"service.name": "baybe.sdk", "service.namespace": "baybe"})
reader = PeriodicExportingMetricReader(
    OTLPMetricExporter(
        "***REMOVED***"
        ".elb.eu-central-1.amazonaws.com:4317",
        True,
    )
)
provider = MeterProvider(resource=resource, metric_readers=[reader])
set_meter_provider(provider)

# Setup Global Metric Provider
meter = get_meter("aws-otel", "1.0")
# Setup Metric Components
recommendation_counter = meter.create_counter(
    "recommendation.counter", description="Counts the number of recommendations jobs"
)


# Generate a unique hash value for the current user based on the host name and
# uppercase username, e.g.
CALLER_ID = hashlib.sha256(
    (socket.gethostname() + getpass.getuser().upper()).encode()
).hexdigest()
# alternative:

# Flag that tells whether telemetry is active or disabled through environment variables
# This is decided at the time the module is loaded
ENABLED = "BAYBE_TELEMETRY_ENABLED" in os.environ and os.environ[
    "TELEMETRY_ENABLED"
].lower() in ["false", "no", "off", "0"]
