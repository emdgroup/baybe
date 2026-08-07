"""Sequence encoders."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import narwhals.stable.v2 as nw
from attrs import define, field
from attrs.validators import instance_of
from exceptiongroup import ExceptionGroup

from baybe.settings import active_settings
from baybe.utils.dataframe import _df_with_backend

if TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame, IntoSeries


@runtime_checkable
class EncoderProtocol(Protocol):
    """Type protocol specifying the interface encoders need to implement."""

    # Use slots so that derived classes also remain slotted
    # See also: https://www.attrs.org/en/stable/glossary.html#term-slotted-classes
    __slots__ = ()

    def __call__(self, series: IntoSeries, /) -> IntoDataFrame:
        """Encode a given series of values.

        Args:
            series: A series in an arbitrary backend, containing the values to encode.

        Returns:
            A dataframe containing the encoded representations of the input values, in
            the same backend as the input series and with the same row order.
        """


@define
class _Encoder:
    """A narwhals wrapper for user-specified encoders to hide their native backend.

    Wraps a user-provided instance of an :class:`EncoderProtocol` and automatically
    infers which native dataframe backend it expects via trial and error. The inferred
    backend is cached after the first successful call so that trial-and-error detection
    runs only once.
    """

    _encoder: EncoderProtocol = field(
        alias="encoder", validator=instance_of(EncoderProtocol)
    )
    """The user-provided encoder."""

    _implementation: nw.Implementation | None = field(
        default=None, init=False, eq=False
    )
    """The inferred native backend, cached after the first successful call."""

    def __call__(self, series: nw.Series, /) -> nw.DataFrame:
        """Encode a narwhals series, inferring the required backend if not yet known.

        On the first call, tries available backends in order (starting with
        :attr:`~baybe.settings.Settings.default_dataframe_backend`) until the wrapped
        callable succeeds. The successful backend is cached for all subsequent calls.

        Args:
            series: A narwhals series containing the values to encode.

        Returns:
            A narwhals dataframe containing the encoded representations, collected into
            the same backend used for the input series.
        """
        if self._implementation is None:
            self._implementation, result = self._infer_backend(series)
            return result

        return self._encode(series, self._implementation)

    def _infer_backend(
        self, series: nw.Series, /
    ) -> tuple[nw.Implementation, nw.DataFrame]:
        """Infer the encoder's backend by trial-and-error across available backends.

        Args:
            series: The series to use for probing.

        Returns:
            A tuple containing:

                * the first backend for which the encoder succeeds
                * the resulting encoded dataframe

        Raises:
            ExceptionGroup: If the encoder raises an exception for every tried backend.
        """
        preferred = active_settings.default_dataframe_backend
        ordered = [preferred] + [b for b in nw.Implementation if b is not preferred]
        backends = [b for b in ordered if _is_backend_imported(b)]

        exceptions: list[Exception] = []
        for backend in backends:
            try:
                result = self._encode(series, backend)
                return backend, result
            except Exception as ex:  # noqa: BLE001
                exceptions.append(ex)

        raise ExceptionGroup("The encoder failed for all tried backends", exceptions)

    def _encode(self, series: nw.Series, backend: nw.Implementation, /) -> nw.DataFrame:
        """Call the encoder with the given series using the specified backend.

        Args:
            series: The series to encode.
            backend: The native backend to convert the series to before encoding.

        Returns:
            A narwhals dataframe in the original series backend.
        """
        native_series = _df_with_backend(series, backend).to_native()
        result = nw.from_native(self._encoder(native_series), eager_only=True)
        return _df_with_backend(result, series.implementation)


def _is_backend_imported(backend: nw.Implementation) -> bool:
    """Check if the given native backend has already been imported."""
    getter = getattr(nw.dependencies, f"get_{backend.value}", None)
    return getter is not None and getter() is not None


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
