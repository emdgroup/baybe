"""Scaling tests."""

import math
from unittest.mock import Mock

import pytest
import torch
from botorch.models.transforms.input import InputStandardize, InputTransform, Normalize

from baybe.utils.scaling import ColumnTransformer


def test_column_transformer():
    """Basic test that validates the transformation of a ColumnTransformer."""
    # Define test input
    base = torch.stack([-1 * torch.ones(5), torch.ones(5)])
    x_train = 20 * base
    x_test = 10 * base

    # Create and fit transformer
    mapping = {
        (0, 2): Normalize(2),
        (1, 3): InputStandardize(2),
    }
    transformer = ColumnTransformer(mapping)
    transformer.fit(x_train)

    # Transform data
    y_train = transformer.transform(x_train)
    y_test = transformer.transform(x_test)

    # Expected results
    s = 1 / math.sqrt(2)
    target_train = torch.tensor(
        [
            [0.0, -s, 0.0, -s, -20.0],
            [1.0, s, 1.0, s, 20.0],
        ]
    )
    target_test = torch.tensor(
        [
            [0.25, -s / 2, 0.25, -s / 2, -10.0],
            [0.75, s / 2, 0.75, s / 2, 10.0],
        ]
    )

    assert torch.allclose(y_train, target_train)
    assert torch.allclose(y_test, target_test)


def test_non_disjoint_column_mapping():
    """Creating a column transformer with non-disjoint columns raises an error."""
    t = Mock(InputTransform)
    mapping = {(0, 1): t, (1, 2): t}
    with pytest.raises(ValueError, match="are not disjoint"):
        ColumnTransformer(mapping)
