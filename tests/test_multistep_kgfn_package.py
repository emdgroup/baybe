"""Structural tests for the consolidated KGFN package."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_gain_module_exposes_the_pkgfn_core():
    """The gain mathematics lives in exactly one importable module."""
    from baybe.multistep.kgfn.gain import (
        _GainContext,
        _make_gain_context,
        _validate_gain_models,
        compute_pkgfn_gain,
    )

    assert callable(_make_gain_context)
    assert callable(_validate_gain_models)
    assert _GainContext.__module__ == "baybe.multistep.kgfn.gain"
    assert compute_pkgfn_gain.__module__ == "baybe.multistep.kgfn.gain"


def test_reference_module_exposes_sampling_and_seeding():
    """Finite references and their deterministic seed live together."""
    from baybe.multistep.kgfn.references import (
        _derive_seed,
        _make_reference_points,
    )

    assert _derive_seed(0, "references") == _derive_seed(0, "references")
    assert _make_reference_points.__module__ == "baybe.multistep.kgfn.references"


def test_direct_module_lives_in_the_kgfn_package():
    """The direct policy has one location and the old module is gone."""
    import baybe.multistep
    from baybe.multistep.kgfn import direct as kgfn_direct

    assert kgfn_direct.recommend_direct.__module__ == "baybe.multistep.kgfn.direct"
    assert kgfn_direct._GainContext.__module__ == "baybe.multistep.kgfn.gain"
    assert kgfn_direct._make_reference_points.__module__ == (
        "baybe.multistep.kgfn.references"
    )
    package_dir = Path(baybe.multistep.__file__).parent
    assert not (package_dir / "direct.py").exists()
    assert (package_dir / "kgfn" / "direct.py").exists()


def test_kgfn_direct_imports_first_without_a_cycle():
    """A fresh interpreter can import the direct module without a partial package."""
    result = subprocess.run(
        [sys.executable, "-c", "import baybe.multistep.kgfn.direct"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
