"""Serialization mixin class."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from baybe.serialization.core import _add_type_to_dict, converter

_T = TypeVar("_T", bound="SerialMixin")

if TYPE_CHECKING:
    from _typeshed import SupportsRead, SupportsWrite


class SerialMixin:
    """A mixin class providing serialization functionality."""

    # Use slots so that derived classes also remain slotted
    # See also: https://www.attrs.org/en/stable/glossary.html#term-slotted-classes
    __slots__ = ()

    def to_dict(self) -> dict:
        """Create an object's dictionary representation.

        Returns:
            The dictionary representation of the object.
        """
        dct = converter.unstructure(self)
        return _add_type_to_dict(dct, self.__class__.__name__)

    @classmethod
    def from_dict(cls: type[_T], dictionary: dict) -> _T:
        """Create an object from its dictionary representation.

        Args:
            dictionary: The dictionary representation.

        Returns:
            The reconstructed object.
        """
        return converter.structure(dictionary, cls)

    def to_json(
        self,
        sink: str | Path | SupportsWrite[str] | None = None,
        /,
        *,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> str:
        """Create an object's JSON representation.

        Args:
            sink: The JSON sink. Can be:

                - ``None`` (only returns the JSON string).
                - A file path or ``Path`` object pointing to a location where to write
                  the JSON content.
                - A file-like object with a ``write()`` method.

            overwrite: Boolean flag indicating if to overwrite the file if it already
                exists. Only relevant if ``sink`` is a file path or ``Path`` object.
            **kwargs: Additional keyword arguments to pass to :func:`json.dumps`.

        Raises:
            FileExistsError: If ``sink`` points to an already existing file but
                ``overwrite`` is ``False``.

        Returns:
            The JSON representation as a string.
        """
        string = json.dumps(self.to_dict(), **kwargs)

        if isinstance(sink, str):
            sink = Path(sink)

        if isinstance(sink, Path):
            if sink.is_file() and not overwrite:
                raise FileExistsError(
                    f"The file '{sink}' already exists. If you want to overwrite it, "
                    f"explicitly set the 'overwrite' flag to 'True'."
                )
            sink.write_text(string)
        elif sink is None:
            pass
        else:
            sink.write(string)

        return string

    @classmethod
    def from_json(cls: type[_T], source: str | Path | SupportsRead[str], /) -> _T:
        """Create an object from its JSON representation.

        Args:
            source: The JSON source. Can be:

                - A string containing JSON content.
                - A file path or ``Path`` object pointing to a JSON file.
                - A file-like object with a ``read()`` method.

        Raises:
            ValueError: If ``source`` is not one of the allowed types.

        Returns:
            The reconstructed object.
        """
        if isinstance(source, Path):
            string = source.read_text()
        elif isinstance(source, str):
            try:
                string = Path(source).read_text()
            except OSError:
                string = source
        else:
            try:
                string = source.read()
            except Exception:
                raise ValueError(
                    "The method argument must be a string containing valid JSON, "
                    "a string holding a file path to a JSON file / a corresponding "
                    "'Path' object, or a file-like object with a 'read()' method."
                )

        return cls.from_dict(json.loads(string))
