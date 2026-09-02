from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.dataset import feature


@pytest.fixture
def split_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    output_directory = tmp_path / "splits"
    output_directory.mkdir()
    monkeypatch.setattr(feature, "SPLITTED_DATASET_FOLDER", output_directory)
    return output_directory


@pytest.fixture
def pre_processed_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    output_directory = tmp_path / "pre_processed"
    monkeypatch.setattr(feature, "PRE_PROCESSED_DATASET_FOLDER", output_directory)
    return output_directory


@pytest.mark.parametrize(
    ("embedding_pooling", "expected_features"),
    [
        ("mean", [1.5, 1.0]),
        ("max", [2.0, 1.0]),
        ("logit_weighted", [1.7310586, 1.0]),
        ("concat", [99.0, 99.0]),
    ],
)
def test_extract_jlens_from_splits_saves_pooled_token_embedding_features(
    monkeypatch: pytest.MonkeyPatch,
    split_directory: Path,
    pre_processed_directory: Path,
    embedding_pooling: str,
    expected_features: list[float],
) -> None:
    import torch

    pd.Series(["train text"]).to_csv(split_directory / "x_train.csv")
    pd.Series(["test text"]).to_csv(split_directory / "x_test.csv")

    class FakeLens:
        def apply(self, model, text, *, layers):
            logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            return {layers[0]: logits}, None, None

    class FakeTokenizer:
        def batch_decode(self, token_ids, **kwargs):
            return [f"token-{token_id}" for token_id in token_ids]

    class FakeEmbeddingModel:
        def encode(self, texts, *, convert_to_tensor=False):
            if len(texts) == 1 and " " in texts[0]:
                embeddings = [[99.0, 99.0]]
            else:
                embeddings = [[float(text.removeprefix("token-")), 1.0] for text in texts]
            return torch.tensor(embeddings) if convert_to_tensor else np.array(embeddings)

    class FakeModel:
        tokenizer = FakeTokenizer()

    monkeypatch.setattr(feature.settings.training, "k", 2)
    monkeypatch.setattr(feature.settings.training, "embedding_pooling", embedding_pooling)
    monkeypatch.setattr(feature.settings.training, "target_layer", 14)
    monkeypatch.setattr(feature.settings.training, "processing_device", "cpu")
    monkeypatch.setattr(feature.settings.training, "output_device", "cpu")
    monkeypatch.setattr(feature.settings.training, "pre_processed_dataset_format", "NumPy")
    monkeypatch.setattr(
        feature, "SentenceTransformer", lambda *args, **kwargs: FakeEmbeddingModel()
    )
    monkeypatch.setattr(feature, "_load_jlens_model", lambda: (FakeModel(), FakeLens()))

    feature.extract_jlens_from_splits()

    np.testing.assert_allclose(
        np.load(pre_processed_directory / "x_train.npy"), [expected_features]
    )
    np.testing.assert_allclose(np.load(pre_processed_directory / "x_test.npy"), [expected_features])
