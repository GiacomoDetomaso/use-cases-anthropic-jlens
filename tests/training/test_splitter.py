from pathlib import Path

import pandas as pd
import pytest

from training.dataset import splitter


@pytest.fixture
def source_dataset() -> pd.DataFrame:
    labels = ["benign"] * 80 + ["malicious"] * 20
    return pd.DataFrame({"text": [f"example-{index}" for index in range(100)], "label_1": labels})


@pytest.fixture
def split_manifest_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    manifest_path = tmp_path / "split_manifest.csv"
    monkeypatch.setattr(splitter, "SPLIT_MANIFEST_PATH", manifest_path)
    return manifest_path


@pytest.mark.parametrize(
    ("dataset_format", "reader_name"),
    [("csv", "read_csv"), ("jsonl", "read_json")],
)
def test_build_splits_reads_dataset_with_configured_format(
    monkeypatch: pytest.MonkeyPatch,
    source_dataset: pd.DataFrame,
    split_manifest_path: Path,
    dataset_format: str,
    reader_name: str,
) -> None:
    reader_calls: list[Path] = []

    def read_dataset(path: Path) -> pd.DataFrame:
        reader_calls.append(path)
        return source_dataset

    monkeypatch.setattr(splitter.settings.output_dataset, "format", dataset_format)
    monkeypatch.setattr(splitter.settings.training, "train_sampling", "no")
    monkeypatch.setattr(splitter.pd, reader_name, read_dataset)

    splitter.build_splits()

    assert reader_calls == [splitter.GENERATED_DATASET_PATH]


def test_build_splits_rejects_unknown_dataset_format(
    monkeypatch: pytest.MonkeyPatch,
    split_manifest_path: Path,
) -> None:
    monkeypatch.setattr(splitter.settings.output_dataset, "format", "parquet")

    with pytest.raises(RuntimeError, match="No available dataset format"):
        splitter.build_splits()


def test_build_splits_saves_complete_manifest(
    monkeypatch: pytest.MonkeyPatch,
    source_dataset: pd.DataFrame,
    split_manifest_path: Path,
) -> None:
    read_csv = pd.read_csv
    monkeypatch.setattr(splitter.settings.output_dataset, "format", "csv")
    monkeypatch.setattr(splitter.settings.training, "train_sampling", "no")
    monkeypatch.setattr(splitter.pd, "read_csv", lambda _: source_dataset)

    splitter.build_splits()

    manifest = read_csv(split_manifest_path)
    assert list(manifest.columns) == [
        "example_id",
        "text",
        "label",
        "split",
        "selected_for_training",
    ]
    assert manifest["example_id"].is_unique
    assert len(manifest) == len(source_dataset)
    assert manifest["label"].isin([0, 1]).all()
    assert manifest["split"].value_counts().to_dict() == {"train": 85, "test": 15}
    assert manifest.loc[manifest["split"] == "train", "selected_for_training"].all()
    assert not manifest.loc[manifest["split"] == "test", "selected_for_training"].any()


@pytest.mark.parametrize(
    ("sampling", "expected_counts"),
    [("no", {0: 68, 1: 17}), ("70/30", {0: 39, 1: 17}), ("50/50", {0: 17, 1: 17})],
)
def test_build_splits_applies_configured_training_balance(
    monkeypatch: pytest.MonkeyPatch,
    source_dataset: pd.DataFrame,
    split_manifest_path: Path,
    sampling: str,
    expected_counts: dict[int, int],
) -> None:
    read_csv = pd.read_csv
    monkeypatch.setattr(splitter.settings.output_dataset, "format", "csv")
    monkeypatch.setattr(splitter.settings.training, "train_sampling", sampling)
    monkeypatch.setattr(splitter.pd, "read_csv", lambda _: source_dataset)

    splitter.build_splits()

    manifest = read_csv(split_manifest_path)
    labels = manifest.loc[manifest["selected_for_training"], "label"]
    assert labels.value_counts().to_dict() == expected_counts
    assert len(manifest) == len(source_dataset)
