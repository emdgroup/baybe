"""Internal telemetry logic."""

from __future__ import annotations

import getpass
import hashlib
import os
import socket
import warnings
from queue import Queue
from threading import Thread
from typing import TYPE_CHECKING, Any

from attrs import define, field, fields
from typing_extensions import override

from baybe.telemetry.api import (
    VARNAME_TELEMETRY_ENABLED,
    VARNAME_TELEMETRY_HOSTNAME,
    VARNAME_TELEMETRY_USERNAME,
    is_enabled,
)
from baybe.utils.boolean import strtobool

if TYPE_CHECKING:
    from opentelemetry.metrics import Histogram, Meter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource


# Telemetry environment variable names
VARNAME_TELEMETRY_ENDPOINT = "BAYBE_TELEMETRY_ENDPOINT"
VARNAME_TELEMETRY_VPN_CHECK = "BAYBE_TELEMETRY_VPN_CHECK"

# Telemetry settings defaults
DEFAULT_TELEMETRY_ENDPOINT = (
    "https://public.telemetry.baybe.p.uptimize.merckgroup.com:4317"
)
DEFAULT_TELEMETRY_VPN_CHECK = "true"
try:
    DEFAULT_TELEMETRY_USERNAME = (
        hashlib.sha256(getpass.getuser().upper().encode()).hexdigest().upper()[:10]
    )
except ModuleNotFoundError:
    # `getpass.getuser()`` does not work on Windows if all the environment variables
    # it checks are empty. Since there is no way of inferring the username in this case,
    # we use a fallback.
    DEFAULT_TELEMETRY_USERNAME = "UNKNOWN"
DEFAULT_TELEMETRY_HOSTNAME = (
    hashlib.sha256(socket.gethostname().encode()).hexdigest().upper()[:10]
)

# Derived constants
ENDPOINT_URL = os.environ.get(VARNAME_TELEMETRY_ENDPOINT, DEFAULT_TELEMETRY_ENDPOINT)
TELEMETRY_VPN_CHECK = strtobool(
    os.environ.get(VARNAME_TELEMETRY_VPN_CHECK, DEFAULT_TELEMETRY_VPN_CHECK)
)


@define
class TelemetryTools:
    """Class for lazy-initialization of telemetry objects."""

    _is_initialized: bool = False
    """Boolean flag indicating if initialization is completed."""

    # Telemetry objects
    instruments: dict[str, Histogram] = field(factory=dict)
    resource: Resource | None = None
    reader: PeriodicExportingMetricReader | None = None
    provider: MeterProvider | None = None
    meter: Meter | None = None

    @override
    def __getattribute__(self, name: str, /) -> Any:
        """Lazily initialize telemetry objects upon first access."""
        if name not in [
            (fields(TelemetryTools)).instruments.name,
            "_initialize",
        ]:
            try:
                self._initialize()
            except Exception:
                if is_enabled() and user_is_developer():
                    warnings.warn(
                        "Opentelemetry could not be imported. Potentially it is "
                        "not installed. Disabling BayBE telemetry.",
                        UserWarning,
                    )
                    os.environ[VARNAME_TELEMETRY_ENABLED] = "false"

        return super().__getattribute__(name)

    def _initialize(self) -> None:
        """Initialize the telemetry objects."""
        if self._is_initialized:
            return

        # Lazy imports
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.metrics import get_meter, set_meter_provider
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource

        # Initialize instruments
        self.resource = Resource.create(
            {"service.namespace": "BayBE", "service.name": "SDK"}
        )
        self.reader = PeriodicExportingMetricReader(
            exporter=OTLPMetricExporter(
                endpoint=ENDPOINT_URL,
                insecure=True,
            )
        )
        self.provider = MeterProvider(
            resource=self.resource, metric_readers=[self.reader]
        )
        set_meter_provider(self.provider)
        self.meter = get_meter("aws-otel", "1.0")

        # Mark initialization as completed
        self._is_initialized = True


class CloseableQueue(Queue):
    """A queue that can be shut down, ignoring incoming items thereafter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._closed = False

    @property
    def is_closed(self) -> bool:
        """Boolean value indicating if the queue is closed."""
        return self._closed

    def close(self):
        """Remove all queue elements and prevent new ones from being added."""
        with self.mutex:
            self._closed = True
            self.queue.clear()

    @override
    def put(self, item, block=True, timeout=None):
        if self._closed:
            return
        super().put(item, block, timeout)


def transmit_events(queue: Queue) -> None:
    """Transmit the telemetry events waiting in the given queue."""
    while True:
        event = queue.get()
        submit_scalar_value(*event)
        queue.task_done()


def test_connection() -> Exception | None:
    """Check if the telemetry endpoint is reachable."""
    try:
        # Send a test request. If the request fails (e.g. no connection, outside
        # VPN, or firewall) this will throw an error.
        socket.gethostbyname("verkehrsnachrichten.merck.de")
        return None

    except Exception as ex:
        # Catching broad exception here to avoid interference for the user.
        # Possible errors are for instance ``socket.gaierror`` in case the user
        # has no internet connection.
        return ex


def daemon_task() -> None:
    """The telemetry logic to be executed in the daemon thread."""  # noqa
    # Telemetry is inactive
    if not is_enabled():
        transmission_queue.close()
        return

    # Telemetry is active but the endpoint is not reachable
    if TELEMETRY_VPN_CHECK and (ex := test_connection()) is not None:
        if user_is_developer():
            # Only printed for developers to make them aware of potential issues
            warnings.warn(
                f"WARNING: BayBE Telemetry endpoint '{ENDPOINT_URL}' cannot be "
                f"reached. Disabling telemetry. The exception encountered was: "
                f"{type(ex).__name__}, {ex}",
                UserWarning,
            )
        transmission_queue.close()
        return

    # If everything is ready for transmission, process the incoming events
    transmit_events(transmission_queue)


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


def user_is_developer() -> bool:
    """Determine if the user is a developer."""
    return os.environ.get(VARNAME_TELEMETRY_USERNAME, "").startswith("DEV_")


def submit_scalar_value(instrument_name: str, value: int | float) -> None:
    """See :func:`baybe.telemetry.api.telemetry_record_value`."""
    if instrument_name in tools.instruments:
        histogram = tools.instruments[instrument_name]
    else:
        histogram = tools.meter.create_histogram(  # type: ignore[union-attr]
            instrument_name,
            description=f"Histogram for instrument {instrument_name}",
        )
        tools.instruments[instrument_name] = histogram
    histogram.record(value, get_user_details())


tools = TelemetryTools()
transmission_queue = CloseableQueue()
Thread(target=daemon_task).start()
