from opentelemetry import propagate, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


from opentelemetry.propagators.aws import AwsXRayPropagator


from opentelemetry.sdk.extension.aws.trace import AwsXRayIdGenerator
from opentelemetry.sdk.resources import Resource

propagate.set_global_textmap(AwsXRayPropagator())

# OTLP Exporter is configured through environment variables:
# - OTEL_EXPORTER_OTLP_ENDPOINT
# - OTEL_EXPORTER_OTLP_CERTIFICATE
otlp_span_exporter = OTLPSpanExporter(
    "***REMOVED***.elb.eu-central-1.amazonaws.com:4317", True)
span_processor = BatchSpanProcessor(otlp_span_exporter)

service_resource = Resource.create({"service.name": "baybe.sdk",
                                    "service.namespace": "baybe"})

trace.set_tracer_provider(
    TracerProvider(
        active_span_processor=span_processor,
        id_generator=AwsXRayIdGenerator(),
        resource=service_resource,
    )
)

tracer = trace.get_tracer(__name__)
