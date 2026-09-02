from __future__ import annotations

from typing import Any

import jlens  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import torch
import transformers
from sentence_transformers import SentenceTransformer

from app.settings import settings
from training.dataset._paths import (
    JLENS_REPOSITORY,
    PRE_PROCESSED_DATASET_FOLDER,
    SPLITTED_DATASET_FOLDER,
)


def _normalise_model_name(model_name: str) -> str:
    return "".join(character for character in model_name.lower() if character.isalnum())


def _find_lens_file(model_name: str) -> str:
    """Return the Hub checkpoint whose path matches the configured base model."""
    from huggingface_hub import HfApi

    normalised_model_name = _normalise_model_name(model_name)
    files = HfApi().list_repo_files(JLENS_REPOSITORY, repo_type="model")
    matching_files = [
        file_name
        for file_name in files
        if file_name.endswith(".pt")
        and _normalise_model_name(file_name).find(normalised_model_name) >= 0
    ]
    if len(matching_files) != 1:
        raise ValueError(
            f"Could not find one J-lens checkpoint for {model_name!r} in {JLENS_REPOSITORY}. "
            "Set training.model_name to the Hugging Face base-model identifier for a supported lens."
        )
    return matching_files[0]


def _torch_device(device_name: str) -> torch.device:
    if device_name == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("A GPU was requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def _load_jlens_model() -> tuple[jlens.LensModel, jlens.JacobianLens]:
    model_name = settings.training.model_name
    if not model_name:
        raise ValueError("training.model_name must name a J-lens-supported Hugging Face model")

    processing_device = _torch_device(settings.training.processing_device)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if processing_device.type == "cuda" else torch.float32,
        device_map=processing_device.type,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(
        JLENS_REPOSITORY,
        filename=_find_lens_file(model_name),
    )
    return model, lens


def _pool_candidate_embeddings(
    candidate_embeddings: torch.Tensor,
    top_k_logits: torch.Tensor,
    decoded_candidates: list[str],
    embedding_model: SentenceTransformer,
) -> torch.Tensor:
    match settings.training.embedding_pooling:
        case "mean":
            return candidate_embeddings.mean(dim=(0, 1))
        case "max":
            return candidate_embeddings.amax(dim=(0, 1))
        case "logit_weighted":
            weights = torch.softmax(top_k_logits, dim=-1).to(candidate_embeddings.device)[
                ..., np.newaxis
            ]
            return (candidate_embeddings * weights).sum(dim=1).mean(dim=0)
        case "concat":
            return embedding_model.encode(
                [" ".join(decoded_candidates)], convert_to_tensor=True
            ).to(_torch_device(settings.training.processing_device))[0]

    raise ValueError(f"Unsupported embedding pooling: {settings.training.embedding_pooling}")


def _extract_embedding_features(
    logits: torch.Tensor, tokenizer: Any, embedding_model: SentenceTransformer
) -> np.ndarray:
    # Shape [n_position, k]: retain top k concepts per position
    top_k = logits.topk(settings.training.k, dim=-1)

    # Flatten decoded tokens
    decoded_candidates = tokenizer.batch_decode(
        top_k.indices.flatten().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Shape: [n_position, k, embedding_size]
    candidate_embeddings = embedding_model.encode(
        decoded_candidates, convert_to_tensor=True
    ).reshape(*top_k.indices.shape, -1)

    pooled_features = _pool_candidate_embeddings(
        candidate_embeddings, top_k.values, decoded_candidates, embedding_model
    )

    return pooled_features.to(_torch_device(settings.training.output_device)).cpu().numpy()


def _extract_features(
    texts: pd.Series[str],
    model: jlens.LensModel,
    lens: jlens.JacobianLens,
    embedding_model: SentenceTransformer,
) -> np.ndarray:
    features = []
    target_layer = settings.training.target_layer

    for text in texts.astype(str):
        # Calculate the logit lens at every token position.
        # These are J-lens readout of what the model’s internal state
        # at that position is predisposed to predict next
        lens_logits_dict, _, _ = lens.apply(model, text, layers=[target_layer])

        features.append(
            _extract_embedding_features(
                lens_logits_dict[settings.training.target_layer], model.tokenizer, embedding_model
            )
        )
    return np.stack(features)


def extract_jlens_from_splits() -> None:
    """Convert saved text splits to pooled top-K J-lens token embeddings."""
    x_train = pd.read_csv(SPLITTED_DATASET_FOLDER / "x_train.csv", index_col=0).iloc[:, 0]
    x_test = pd.read_csv(SPLITTED_DATASET_FOLDER / "x_test.csv", index_col=0).iloc[:, 0]

    model, lens = _load_jlens_model()

    embedding_model = SentenceTransformer(
        settings.training.embedding_model_name,
        device=str(_torch_device(settings.training.processing_device)),
    )

    PRE_PROCESSED_DATASET_FOLDER.mkdir(parents=True, exist_ok=True)
    split_features = {
        "x_train": _extract_features(x_train, model, lens, embedding_model),
        "x_test": _extract_features(x_test, model, lens, embedding_model),
    }
    match settings.training.pre_processed_dataset_format:
        case "NumPy":
            for split_name, features in split_features.items():
                np.save(PRE_PROCESSED_DATASET_FOLDER / f"{split_name}.npy", features)
        case "Parquet":
            for split_name, features in split_features.items():
                pd.DataFrame(features).to_parquet(
                    PRE_PROCESSED_DATASET_FOLDER / f"{split_name}.parquet", index=False
                )
