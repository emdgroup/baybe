"""Finite global reference sampling for the direct pKGFN policy.

The reference set replaces the forbidden parameter product: each reference is
sampled parameter-wise from the factorized layout, and the same seed reproduces
the same set for baseline, fantasy, and search initialization.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from baybe.exceptions import IncompatibilityError
from baybe.parameters.numerical import NumericalContinuousParameter

if TYPE_CHECKING:
    from torch import Tensor

    from ..model import MultiStepModel
    from .direct import TerminalSearchProfile


def _derive_seed(base_seed: int, *parts: str) -> int:
    """Derive a stable nonnegative PCG64 seed from explicit components."""
    payload = ":".join((str(base_seed), *parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _make_reference_points(
    model: MultiStepModel, profile: TerminalSearchProfile
) -> Tensor:
    """Sample global reference configurations parameter-wise, without a product."""
    import torch

    if not model.dag.product_free:
        raise IncompatibilityError(
            "The direct pKGFN policy requires a product-free DAG."
        )
    layout = model.dag._configuration_space
    reference = next(model.parameters())
    rng = np.random.Generator(
        np.random.PCG64(_derive_seed(profile.sampling_seed, "references"))
    )
    rows: list[dict[str, Any]] = []
    for _ in range(profile.raw_samples):
        row: dict[str, Any] = {}
        for parameter in layout.parameters:
            if isinstance(parameter, NumericalContinuousParameter):
                row[parameter.name] = float(
                    rng.uniform(parameter.bounds.lower, parameter.bounds.upper)
                )
            else:
                values = parameter.active_values
                row[parameter.name] = values[int(rng.integers(len(values)))]
        rows.append(row)

    comp_rep = layout.transform(pd.DataFrame(rows))
    points = torch.as_tensor(
        comp_rep.to_numpy(dtype=float, copy=True),
        dtype=reference.dtype,
        device=reference.device,
    ).unsqueeze(-2)
    expected = (profile.raw_samples, 1, len(layout.comp_rep_columns))
    if points.shape != expected:
        raise IncompatibilityError(
            f"Reference sampling produced shape {tuple(points.shape)}, expected "
            f"{expected}."
        )
    if not torch.isfinite(points).all():
        raise IncompatibilityError("Reference sampling produced non-finite values.")
    return points
