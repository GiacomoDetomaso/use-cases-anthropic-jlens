import importlib
import sys

import pandas as pd
import pytest


@pytest.fixture
def dataset_source(monkeypatch):
    source_dataframe = pd.DataFrame(
        {"instruction": ["Test"], "intent": ["cancel_order"]}
    )
    target_dataframe = pd.DataFrame(
        {"text": ["Test"], "category": ["adversarial"]}
    )
    monkeypatch.setattr(pd, "read_csv", lambda _: source_dataframe)
    monkeypatch.setattr(pd, "read_parquet", lambda _: target_dataframe)
    sys.modules.pop("dataset_agent.dataset_source", None)

    return importlib.import_module("dataset_agent.dataset_source")


def test_read_dataset_uses_csv_reader(dataset_source, monkeypatch):
    expected = pd.DataFrame({"instruction": ["CSV"]})
    monkeypatch.setattr(pd, "read_csv", lambda _: expected)

    actual = dataset_source._read_dataset("hf://datasets/example.csv")

    assert actual is expected


def test_read_dataset_uses_parquet_reader(dataset_source, monkeypatch):
    expected = pd.DataFrame({"instruction": ["Parquet"]})
    monkeypatch.setattr(pd, "read_parquet", lambda _: expected)

    actual = dataset_source._read_dataset("hf://datasets/example.parquet")

    assert actual is expected


def test_read_dataset_rejects_unsupported_formats(dataset_source):
    with pytest.raises(ValueError, match="Supported formats are CSV and Parquet"):
        dataset_source._read_dataset("hf://datasets/example.json")