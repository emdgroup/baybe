"""Internal telemetry logic."""

from __future__ import annotations

import getpass
import hashlib
import os
import socket
import warnings
from queue import Queue
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
VARNAME_TELEMETRY_VPN_CHECK_TIMEOUT = "BAYBE_TELEMETRY_VPN_CHECK_TIMEOUT"

# Telemetry settings defaults
DEFAULT_TELEMETRY_ENDPOINT = (
    "https://public.telemetry.baybe.p.uptimize.merckgroup.com:4317"
)
DEFAULT_TELEMETRY_VPN_CHECK = "true"
DEFAULT_TELEMETRY_VPN_CHECK_TIMEOUT = "0.5"
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
    _is_initialized: bool = False
    """Boolean flag for lazy initialization."""

    # Telemetry objects
    instruments: dict[str, Histogram] = field(factory=dict)
    resource: Resource | None = None
    reader: PeriodicExportingMetricReader | None = None
    provider: MeterProvider | None = None
    meter: Meter | None = None

    @override
    def __getattribute__(self, name: str, /) -> Any:
        if name not in [
            (fields(TelemetryTools)).instruments.name,
            self._lazy_initialize.__name__,
        ]:
            try:
                self._lazy_initialize()
            except Exception:
                if is_enabled():
                    warnings.warn(
                        "Opentelemetry could not be imported, potentially it is "
                        "not installed. Disabling BayBE telemetry.",
                        UserWarning,
                    )
                    os.environ[VARNAME_TELEMETRY_ENABLED] = "false"

        return super().__getattribute__(name)

    def _lazy_initialize(self) -> None:
        """Lazily initialize the telemetry objects upon first access."""
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


def test_connection() -> None:
    """Close the transmission queue if the telemetry endpoint is unreachable."""
    try:
        # Send a test request. If there is no internet connection or a firewall is
        # present this will throw an error and telemetry will be deactivated.
        socket.gethostbyname("verkehrsnachrichten.merck.de")

    except Exception as ex:
        # Catching broad exception here and disabling telemetry in that case to avoid
        # any telemetry timeouts or interference for the user in case of unexpected
        # errors. Possible ones are for instance ``socket.gaierror`` in case the user
        # has no internet connection.
        if os.environ.get(VARNAME_TELEMETRY_USERNAME, "").startswith("DEV_"):
            # This warning is only printed for developers to make them aware of
            # potential issues
            warnings.warn(
                f"WARNING: BayBE Telemetry endpoint {ENDPOINT_URL} cannot be reached. "
                "Disabling telemetry. The exception encountered was: "
                f"{type(ex).__name__}, {ex}",
                UserWarning,
            )
        transmission_queue.close()


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


def _submit_scalar_value(instrument_name: str, value: int | float) -> None:
    """See :func:`baybe.telemetry.telemetry_record_value`."""
    if instrument_name in tools.instruments:
        histogram = tools.instruments[instrument_name]
    else:
        histogram = tools.meter.create_histogram(
            instrument_name,
            description=f"Histogram for instrument {instrument_name}",
        )
        tools.instruments[instrument_name] = histogram
    histogram.record(value, get_user_details())


tools = TelemetryTools()
transmission_queue = CloseableQueue()

if is_enabled() and TELEMETRY_VPN_CHECK:
    test_connection()
