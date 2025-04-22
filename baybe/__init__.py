"""BayBE — A Bayesian Back End for Design of Experiments."""

import warnings

from baybe.campaign import Campaign

# Show deprecation warnings
warnings.filterwarnings("default", category=DeprecationWarning, module="baybe")


def infer_version() -> str:  # pragma: no cover
    """Determine the package version for the different ways the code can be invoked."""
    # ----------------------------------------------------------------------------------
    # Attempt 1:
    # Try to dynamically infer the version on the fly from the git history using
    # setuptools_scm, provided the latter is installed. This relies on the same logic as
    # for writing the version metadata during the installation process (see Attempt 2
    # below) but has the advantage that the version info is always up-to-date, even when
    # using an "editable installation" using the "-e" flag, where the metadata file may
    # become outdated when the git HEAD changes.
    try:
        from pathlib import Path

        from setuptools_scm import get_version

        return get_version(
            root=str(Path(__path__[0]).parent),
            version_scheme="post-release",
            local_scheme="dirty-tag",
        )
    except (LookupError, ImportError):
        pass

    # ----------------------------------------------------------------------------------
    # Attempt 2:
    # If the package is installed, retrieve the version from the metadata generated
    # via setuptools_scm during the installation process. (If the install had been
    # attempted without the git folder in place, setuptools_scm would have complained,
    # causing the installation process to fail.)
    from importlib.metadata import version

    try:
        return version(__name__)
    except ModuleNotFoundError:
        pass

    # ----------------------------------------------------------------------------------
    # Fallback:
    # If neither the package is installed nor the git history is available, return
    # "unknown" since the version cannot be determined.
    return "unknown"


__version__ = infer_version()
__all__ = [
    "__version__",
    "Campaign",
]

del infer_version
