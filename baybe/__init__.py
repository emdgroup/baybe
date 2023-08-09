"""BayBE — A Bayesian Back End for Design of Experiments."""

from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
    version as _version,
)
from importlib.resources import path as _path

from setuptools_scm import get_version as _get_version


try:
    __version__ = _version(__name__)
except _PackageNotFoundError:
    pass

try:
    with _path(__name__, "") as package_folder:
        __version__ = _get_version(
            root=str(package_folder / ".."),
            version_scheme="post-release",
            local_scheme="dirty-tag",
        )
except LookupError:
    __version__ = "unknown"

########################################################################################
# Prepare namespace
########################################################################################

from baybe.core import BayBE  # pylint: disable=wrong-import-position

__all__ = [
    "__version__",
    "BayBE",
]
