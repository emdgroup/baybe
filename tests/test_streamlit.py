"""Tests for the streamlit demos."""

import glob
import subprocess

import pytest

from baybe._optional.info import STREAMLIT_INSTALLED


@pytest.mark.skipif(
    not STREAMLIT_INSTALLED, reason="Optional dependency streamlit not installed."
)
@pytest.mark.parametrize("script", glob.glob("streamlit/*.py"))
def test_streamlit_scripts(script):
    """All streamlit demos run without errors."""
    result = subprocess.run(["python", script], stderr=subprocess.PIPE, check=False)
    assert result.returncode == 0, f"Error running {script}: {result.stderr.decode()}"
