"""
Telemetry  functionality for BayBE.
"""
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


# Old Code temporary
# from opentelemetry import propagate, trace
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
#
# from opentelemetry.propagators.aws import AwsXRayPropagator
#
# from opentelemetry.sdk.extension.aws.trace import AwsXRayIdGenerator
# from opentelemetry.sdk.resources import Resource
# from opentelemetry.sdk.trace import TracerProvider
# from opentelemetry.sdk.trace.export import BatchSpanProcessor
#
# propagate.set_global_textmap(AwsXRayPropagator())
#
# # OTLP Exporter is configured through environment variables:
# # - OTEL_EXPORTER_OTLP_ENDPOINT
# # - OTEL_EXPORTER_OTLP_CERTIFICATE
# otlp_span_exporter = OTLPSpanExporter(
#     "***REMOVED***.elb.eu-central-1.amazonaws.com:4317",
#     True,
# )
# span_processor = BatchSpanProcessor(otlp_span_exporter)
#
# service_resource = Resource.create(
#     {"service.name": "baybe.sdk", "service.namespace": "baybe"}
# )
#
# trace.set_tracer_provider(
#     TracerProvider(
#         active_span_processor=span_processor,
#         id_generator=AwsXRayIdGenerator(),
#         resource=service_resource,
#     )
# )
#
# tracer = trace.get_tracer(__name__)

# "BayBE __init__", attributes={"client.mac": hex(uuid.getnode())}

# from .telemetry.setup_tracer import tracer
# @tracer.start_as_current_span("_simulate_experiment")
# with tracer.start_as_current_span(
#         "run_doe_iteration", attributes={"simulation.k_iteration": k_iteration}
# ):
