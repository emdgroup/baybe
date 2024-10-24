"""Tests for the NormalizedAreaUnderTheCurve class."""

import pandas as pd
import pytest

from benchmark.src.metric.auc import NormalizedAreaUnderTheCurve


@pytest.fixture
def sample_data():
    data = {
        "Scenario": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "Monte Carlo": [1, 2, 1, 2, 1, 2, 1, 2],
        "Num_Experiments": [1, 1, 2, 2, 1, 1, 2, 2],
        "Objective": [1, 2, 5, 9, 2, 11, 12, 13],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_lookup():
    return pd.DataFrame({"Objective": [1.0, 15.0]})


@pytest.fixture
def highest_value(sample_data, sample_lookup):
    return (sample_data["Objective"].max() - sample_lookup["Objective"].min()) / (
        sample_lookup["Objective"].max() - sample_lookup["Objective"].min()
    )


@pytest.fixture
def sample_metric(sample_lookup):
    return NormalizedAreaUnderTheCurve(lookup=sample_lookup, objective_name="Objective")


def test_normalize_data(sample_metric, sample_data, highest_value):
    normalized_data = sample_metric._normalize_data(sample_data, "Objective")
    assert normalized_data["Objective"].max() == highest_value
    assert normalized_data["Objective"].min() == 0.0
    expected_values = [
        0,
        1 / 14,
        4 / 14,
        8 / 14,
        1 / 14,
        10 / 14,
        11 / 14,
        12 / 14,
    ]
    assert normalized_data["Objective"].equals(pd.Series(expected_values))


def test_evaluate(sample_metric, sample_data):
    result = sample_metric.evaluate(sample_data)
    assert "A" in result
    assert "B" in result
    assert isinstance(result["A"], float)
    assert isinstance(result["B"], float)


def test_check_threshold(sample_metric):
    sample_metric.threshold = {"A": 0.5}
    with pytest.raises(ValueError):
        sample_metric._check_threshold({"A": 0.4})


def test_str(sample_metric):
    assert str(sample_metric) == "Normalized Area Under the Curve"


def test_results(sample_metric, sample_data):
    result = sample_metric.evaluate(sample_data)
    assert pytest.approx(result["A"], 0.0001) == pytest.approx(13 / 56, 0.0001)
    assert pytest.approx(result["B"], 0.0001) == pytest.approx(17 / 28, 0.0001)
