from pathlib import Path

import pandas as pd
import pytest

from training.dataset import build_dataset


@pytest.fixture
def feature_dataset_base_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    output_path = tmp_path / "features"
    monkeypatch.setattr(build_dataset, "FEATURE_EXTRACTED_BASE_FILE_PATH_NO_EXT", output_path)
    return output_path


def test_build_uses_existing_npz_feature_dataset(
    monkeypatch: pytest.MonkeyPatch, feature_dataset_base_path: Path
) -> None:
    feature_path = feature_dataset_base_path.with_suffix(".npz")
    feature_path.touch()
    dataset = pd.DataFrame({"example_id": [1], "feature_0": [0.5]})

    build_splits_calls = 0

    def build_splits() -> None:
        nonlocal build_splits_calls
        build_splits_calls += 1

    monkeypatch.setattr(build_dataset, "build_splits", build_splits)
    monkeypatch.setattr(
        build_dataset,
        "extract_jlens_features",
        lambda: pytest.fail("should not extract when an artifact already exists"),
    )
    monkeypatch.setattr(build_dataset, "load_feature_dataset", lambda path: dataset)

    result = build_dataset.build()

    assert build_splits_calls == 1
    assert result is dataset


def test_build_extracts_then_loads_feature_dataset(
    monkeypatch: pytest.MonkeyPatch, feature_dataset_base_path: Path
) -> None:
    feature_path = feature_dataset_base_path.with_suffix(".npz")
    dataset = pd.DataFrame({"example_id": [1], "feature_0": [0.5]})
    loaded_paths: list[Path] = []

    def extract_jlens_features() -> None:
        feature_path.touch()

    def load_feature_dataset(path: Path) -> pd.DataFrame:
        loaded_paths.append(path)
        return dataset

    monkeypatch.setattr(build_dataset, "build_splits", lambda: None)
    monkeypatch.setattr(build_dataset, "extract_jlens_features", extract_jlens_features)
    monkeypatch.setattr(build_dataset, "load_feature_dataset", load_feature_dataset)

    result = build_dataset.build()

    assert loaded_paths == [feature_path]
    assert result is dataset


def test_build_raises_when_extraction_creates_no_feature_dataset(
    monkeypatch: pytest.MonkeyPatch, feature_dataset_base_path: Path
) -> None:
    monkeypatch.setattr(build_dataset, "build_splits", lambda: None)
    monkeypatch.setattr(build_dataset, "extract_jlens_features", lambda: None)

    with pytest.raises(RuntimeError, match="Feature extraction did not create an output dataset"):
        build_dataset.build()


def test_build_removes_example_id_column_when_requested(
    monkeypatch: pytest.MonkeyPatch, feature_dataset_base_path: Path
) -> None:
    feature_dataset_base_path.with_suffix(".parquet").touch()
    dataset = pd.DataFrame({"example_id": [1], "feature_0": [0.5]})

    monkeypatch.setattr(build_dataset, "build_splits", lambda: None)
    monkeypatch.setattr(build_dataset, "load_feature_dataset", lambda path: dataset)

    result = build_dataset.build(remove_example_id_col=True)

    assert list(result.columns) == ["feature_0"]
    pd.testing.assert_frame_equal(dataset, pd.DataFrame({"example_id": [1], "feature_0": [0.5]}))
