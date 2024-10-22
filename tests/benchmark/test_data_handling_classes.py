"""Tests for the data handling classes."""

import os
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from pandas import DataFrame

from benchmark.src.persistance.data_handling_classes import (
    LocalExperimentResultPersistence,
    S3ExperimentResultPersistence,
)
from benchmark.src.result import Result


@pytest.fixture
def mock_result():
    result = MagicMock(spec=Result)
    result.identifier = UUID("12345678123456781234567812345678")
    result.metadata = {"key": "value"}
    result.get_execution_time_ns.return_value = 123456789
    result.title = "Test Result"
    result.to_csv.return_value = "col1,col2\nval1,val2"
    return result


@pytest.fixture
def mock_date_time() -> datetime:
    return datetime.fromisoformat("2023-01-01T00:00:00")


@pytest.fixture
def mock_s3_persistence(mock_date_time):
    with patch.dict(
        os.environ,
        {
            "BAYBE_PERFORMANCE_PERSISTANCE_PATH": "test-bucket",
            "GITHUB_REF_NAME": "main",
            "GITHUB_SHA": "testsha",
            "GITHUB_ACTOR": "testactor",
            "GITHUB_RUN_ID": "testrunid",
        },
    ):
        return S3ExperimentResultPersistence(mock_date_time)


@pytest.fixture
def mock_local_persistence(mock_date_time):
    return LocalExperimentResultPersistence(mock_date_time)


def test_s3_persist_new_result(mock_s3_persistence, mock_result):
    with patch.object(mock_s3_persistence._object_session, "client") as mock_client:
        mock_s3_persistence.persist_new_result(mock_result)
        mock_client().put_object.assert_called_once()
        args, kwargs = mock_client().put_object.call_args
        assert kwargs["Bucket"] == "test-bucket"
        assert kwargs["Key"].startswith("12345678-1234-5678-1234-567812345678/main/")
        assert kwargs["Body"] == "col1,col2\nval1,val2"
        assert kwargs["ContentType"] == "text/csv"
        assert kwargs["Metadata"]["title"] == "Test Result"


def test_s3_load_compare_result(mock_s3_persistence):
    experiment_id = UUID("12345678123456781234567812345678")
    with patch(
        "benchmark.src.persistance.data_handling_classes.S3ExperimentResultPersistence._get_newest_dataset_from_last_release"
    ) as mock_get_newest:
        mock_get_newest.return_value = DataFrame({"col1": ["val1"], "col2": ["val2"]})
        result = mock_s3_persistence.load_compare_result(experiment_id)
        mock_get_newest.assert_called_once_with(experiment_id)
        assert isinstance(result, DataFrame)


def test_local_load_compare_result(mock_local_persistence):
    experiment_id = UUID("12345678123456781234567812345678")
    with patch(
        "os.listdir",
        return_value=[
            "12345678-1234-5678-1234-567812345678_2023-01-01_testversion.csv"
        ],
    ), patch(
        "pandas.read_csv", return_value=DataFrame({"col1": ["val1"], "col2": ["val2"]})
    ):
        result = mock_local_persistence.load_compare_result(experiment_id)
        assert isinstance(result, DataFrame)
