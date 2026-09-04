from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training.dataset import feature


@pytest.fixture
def split_manifest_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    manifest_path = tmp_path / "split_manifest.csv"
    monkeypatch.setattr(feature, "SPLIT_MANIFEST_PATH", manifest_path)
    return manifest_path


@pytest.mark.parametrize(
    ("embedding_pooling", "expected_features"),
    [
        ("mean", [1.5, 1.0]),
        ("max", [2.0, 1.0]),
        ("concat", [99.0, 99.0]),
    ],
)
def test_extract_jlens_features_saves_id_keyed_token_embedding_features(
    monkeypatch: pytest.MonkeyPatch,
    split_manifest_path: Path,
    tmp_path: Path,
    embedding_pooling: str,
    expected_features: list[float],
) -> None:
    import torch

    pd.DataFrame(
        {
            "example_id": [0, 1],
            "text": ["first text", "second text"],
            "label": [0, 1],
            "split": ["train", "test"],
            "selected_for_training": [True, False],
        }
    ).to_csv(split_manifest_path, index=False)

    class FakeLens:
        def apply(self, model, text, *, layers):
            logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            return {layers[0]: logits}, None, None

    class FakeTokenizer:
        def decode(self, token_ids, **kwargs):
            return f"token-{token_ids[0]}"

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
    artifact_path = tmp_path / f"features_test-model_{embedding_pooling}.npz"
    monkeypatch.setattr(feature, "get_feature_dataset_path", lambda *args: artifact_path)
    monkeypatch.setattr(
        feature, "SentenceTransformer", lambda *args, **kwargs: FakeEmbeddingModel()
    )
    monkeypatch.setattr(feature, "_load_jlens_model", lambda: (FakeModel(), FakeLens()))

    feature.extract_jlens_features()

    artifact = np.load(artifact_path)
    np.testing.assert_array_equal(artifact["example_id"], [0, 1])
    np.testing.assert_allclose(artifact["features"], [expected_features, expected_features])

    feature_dataset = feature.load_feature_dataset(artifact_path)
    assert list(feature_dataset.columns) == ["example_id", "feature_0", "feature_1"]
    np.testing.assert_allclose(feature_dataset.iloc[:, 1:], [expected_features, expected_features])


def test_logit_weighted_pooling_is_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(feature.settings.training, "embedding_pooling", "logit_weighted")

    with pytest.raises(NotImplementedError, match="not implemented"):
        feature._pool_candidate_embeddings(
            torch.tensor([[1.0, 2.0]]),
            ["token"],
            embedding_model=None,  # type: ignore[arg-type]
        )
