"""Provides the version number."""

from importlib.metadata import PackageNotFoundError, version
from importlib.resources import path

from setuptools_scm import get_version

try:
    __version__ = version("baybe")
except PackageNotFoundError:
    pass

try:
    with path("baybe", "") as package_folder:
        __version__ = get_version(
            root=str(package_folder / ".."),
            version_scheme="post-release",
            local_scheme="dirty-tag",
        )
except LookupError:
    __version__ = "unknown"

del version
del PackageNotFoundError
del path
del get_version
