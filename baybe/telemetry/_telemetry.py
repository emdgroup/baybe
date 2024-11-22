"""Internal telemetry logic."""

import getpass
import hashlib
import os
import socket
import warnings

from baybe.telemetry.api import (
    VARNAME_TELEMETRY_ENABLED,
    VARNAME_TELEMETRY_HOSTNAME,
    VARNAME_TELEMETRY_USERNAME,
    is_enabled,
)
from baybe.utils.boolean import strtobool

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


# Attempt telemetry import
try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.metrics import Histogram, get_meter, set_meter_provider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
except ImportError:
    # Failed telemetry install/import should not fail baybe, so telemetry is being
    # disabled in that case
    if is_enabled():
        warnings.warn(
            "Opentelemetry could not be imported, potentially it is not installed. "
            "Disabling baybe telemetry.",
            UserWarning,
        )
    os.environ[VARNAME_TELEMETRY_ENABLED] = "false"


# Attempt telemetry initialization
if is_enabled():
    # Assign default user and machine name
    try:
        DEFAULT_TELEMETRY_USERNAME = (
            hashlib.sha256(getpass.getuser().upper().encode()).hexdigest().upper()[:10]
        )
    except ModuleNotFoundError:
        # getpass.getuser() does not work on Windows if all the environment variables
        # it checks are empty. Since then there is no way of inferring the username, we
        # use UNKNOWN as fallback.
        DEFAULT_TELEMETRY_USERNAME = "UNKNOWN"

    DEFAULT_TELEMETRY_HOSTNAME = (
        hashlib.sha256(socket.gethostname().encode()).hexdigest().upper()[:10]
    )

    _endpoint_url = os.environ.get(
        VARNAME_TELEMETRY_ENDPOINT, DEFAULT_TELEMETRY_ENDPOINT
    )

    # Test endpoint URL
    try:
        # Send a test request. If there is no internet connection or a firewall is
        # present this will throw an error and telemetry will be deactivated.
        if strtobool(
            os.environ.get(VARNAME_TELEMETRY_VPN_CHECK, DEFAULT_TELEMETRY_VPN_CHECK)
        ):
            socket.gethostbyname("verkehrsnachrichten.merck.de")

        # User has connectivity to the telemetry endpoint, so we initialize
        _instruments: dict[str, Histogram] = {}
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
    except Exception as ex:
        # Catching broad exception here and disabling telemetry in that case to avoid
        # any telemetry timeouts or interference for the user in case of unexpected
        # errors. Possible ones are for instance ``socket.gaierror`` in case the user
        # has no internet connection.
        if os.environ.get(VARNAME_TELEMETRY_USERNAME, "").startswith("DEV_"):
            # This warning is only printed for developers to make them aware of
            # potential issues
            warnings.warn(
                f"WARNING: BayBE Telemetry endpoint {_endpoint_url} cannot be reached. "
                "Disabling telemetry. The exception encountered was: "
                f"{type(ex).__name__}, {ex}",
                UserWarning,
            )
        os.environ[VARNAME_TELEMETRY_ENABLED] = "false"
else:
    DEFAULT_TELEMETRY_USERNAME = "UNKNOWN"
    DEFAULT_TELEMETRY_HOSTNAME = "UNKNOWN"


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
    if instrument_name in _instruments:
        histogram = _instruments[instrument_name]
    else:
        histogram = _meter.create_histogram(
            instrument_name,
            description=f"Histogram for instrument {instrument_name}",
        )
        _instruments[instrument_name] = histogram
    histogram.record(value, get_user_details())
