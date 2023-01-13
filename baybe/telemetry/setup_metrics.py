# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from opentelemetry._metrics import set_meter_provider, get_meter
from opentelemetry.exporter.otlp.proto.grpc._metric_exporter import OTLPMetricExporter
from opentelemetry.sdk._metrics import MeterProvider

from opentelemetry.sdk._metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

resource = Resource.create({"service.name": "baybe.sdk",
                            "service.namespace": "baybe"})


reader = PeriodicExportingMetricReader(
    OTLPMetricExporter("***REMOVED***.elb.eu-central-1.amazonaws.com:4317", True))
provider = MeterProvider(resource=resource, metric_readers=[reader])
set_meter_provider(provider)

# Setup Global Metric Provider
meter = get_meter("aws-otel", "1.0")

# Setup Metric Components
recommendation_counter = meter.create_counter(
    "recommendation.counter", description="Counts the number of recommendations jobs")
