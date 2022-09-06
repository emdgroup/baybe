"""Provides the version number."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("baybe")
except PackageNotFoundError:
    # package is not installed
    pass

del version
del PackageNotFoundError
