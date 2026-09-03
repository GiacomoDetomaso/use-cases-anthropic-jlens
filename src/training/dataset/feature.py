from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jlens  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import torch
import transformers
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.settings import settings
from training.dataset._paths import (
    FEATURE_EXTRACTED_BASE_FILE_PATH_NO_EXT,
    JLENS_REPOSITORY,
    SPLIT_MANIFEST_PATH,
)
from training.dataset.splitter import EXAMPLE_ID_COLUMN, TEXT_COLUMN

if TYPE_CHECKING:
    from pathlib import Path


def _normalise_model_name(model_name: str) -> str:
    """Normalise a model identifier for checkpoint-name comparisons.

    The normalisation converts the name to lowercase and removes every
    non-alphanumeric character. This makes identifiers such as
    ``"meta-llama/Llama-3.1"`` comparable with filenames that use a
    different separator convention.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier or checkpoint filename.

    Returns
    -------
    str
        Lowercase, alphanumeric-only representation of ``model_name``.
    """
    return "".join(character for character in model_name.lower() if character.isalnum())


def _find_lens_file(model_name: str) -> str:
    """Find the J-lens checkpoint associated with a base model.

    The function lists model files in the configured J-lens repository and
    selects the single ``.pt`` file whose normalised filename contains the
    normalised base-model name.

    Parameters
    ----------
    model_name : str
        Hugging Face identifier of the configured causal language model.

    Returns
    -------
    str
        Path of the matching J-lens checkpoint within the Hugging Face
        repository.

    Raises
    ------
    ValueError
        If no checkpoint, or more than one checkpoint, matches the supplied
        model name.
    """
    from huggingface_hub import HfApi

    normalised_model_names = {
        _normalise_model_name(model_name),
        _normalise_model_name(model_name.rsplit("/", maxsplit=1)[-1]),
    }
    files = HfApi().list_repo_files(JLENS_REPOSITORY, repo_type="model")
    matching_files = [
        file_name
        for file_name in files
        if file_name.endswith(".pt")
        and any(
            normalised_model_name in _normalise_model_name(file_name)
            for normalised_model_name in normalised_model_names
        )
    ]

    canonical_files = [
        file_name for file_name in matching_files if file_name.endswith("_jacobian_lens.pt")
    ]
    if len(canonical_files) == 1:
        return canonical_files[0]

    if len(matching_files) != 1:
        raise ValueError(
            f"Could not find one J-lens checkpoint for {model_name!r} in {JLENS_REPOSITORY}. "
            "Set training.model_name to the Hugging Face base-model identifier for a supported lens."
        )
    return matching_files[0]


def _torch_device(device_name: str) -> torch.device:
    """Resolve a configured device name to a PyTorch device.

    Parameters
    ----------
    device_name : str
        Requested device name. The value ``"gpu"`` maps to CUDA; every other
        value maps to CPU.

    Returns
    -------
    torch.device
        Resolved PyTorch device, either ``cuda`` or ``cpu``.

    Raises
    ------
    RuntimeError
        If GPU processing is requested but CUDA is unavailable.
    """
    if device_name == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("A GPU was requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def _load_jlens_model() -> tuple[jlens.LensModel, jlens.JacobianLens]:
    """Load the configured language model and its matching J-lens checkpoint.

    The Hugging Face causal language model is loaded on the configured
    processing device and converted to a J-lens-compatible model. The
    corresponding Jacobian lens checkpoint is then retrieved from the
    configured J-lens repository.

    Returns
    -------
    model : jlens.LensModel
        J-lens wrapper around the configured Hugging Face causal language
        model and tokenizer.
    lens : jlens.JacobianLens
        Pretrained Jacobian lens compatible with ``model``.

    Raises
    ------
    ValueError
        If no training model name is configured or a unique matching lens
        checkpoint cannot be found.
    RuntimeError
        If GPU processing is requested but CUDA is unavailable.
    """
    model_name = settings.training.model_name
    if not model_name:
        raise ValueError("training.model_name must name a J-lens-supported Hugging Face model")

    processing_device = _torch_device(settings.training.processing_device)

    logger.info("Loading language model {} on {}", model_name, processing_device)

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if processing_device.type == "cuda" else torch.float32,
        device_map=processing_device.type,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    model = jlens.from_hf(hf_model, tokenizer)

    logger.info("Loading matching J-lens checkpoint")
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
    """Pool token-candidate embeddings into one feature vector.

    The pooling strategy is controlled by
    ``settings.training.embedding_pooling``. Supported strategies are
    arithmetic mean, element-wise maximum, logit-weighted mean, and
    concatenation followed by sentence embedding.

    Parameters
    ----------
    candidate_embeddings : torch.Tensor
        Candidate-token embeddings with shape
        ``(n_positions, k, embedding_size)``.
    top_k_logits : torch.Tensor
        Top-k logits corresponding to candidate tokens, with shape
        ``(n_positions, k)``.
    decoded_candidates : list[str]
        Flattened list of decoded top-k token candidates.
    embedding_model : SentenceTransformer
        Sentence-transformer model used by the ``"concat"`` pooling method.

    Returns
    -------
    torch.Tensor
        Pooled embedding vector with shape ``(embedding_size,)``.

    Raises
    ------
    ValueError
        If the configured pooling strategy is unsupported.
    """
    match settings.training.embedding_pooling:
        case "mean":
            return candidate_embeddings.mean(dim=(0, 1))
        case "max":
            return candidate_embeddings.amax(dim=(0, 1))
        case "logit_weighted":
            weights = torch.softmax(top_k_logits, dim=-1).to(candidate_embeddings.device)[
                ..., torch.newaxis
            ]
            return (candidate_embeddings * weights).sum(dim=1).mean(dim=0)
        case "concat":
            return embedding_model.encode(
                [" ".join(decoded_candidates)], convert_to_tensor=True
            ).to(_torch_device(settings.training.processing_device))[0]

    raise ValueError(f"Unsupported embedding pooling: {settings.training.embedding_pooling}")


def _extract_embedding_features(
    logits: torch.Tensor,
    tokenizer: Any,
    embedding_model: SentenceTransformer,
) -> np.ndarray:
    """Convert J-lens logits into a pooled semantic feature vector.

    For every token position, the function retains the configured number of
    highest-logit candidate tokens, decodes them, embeds them using a
    sentence-transformer model, and pools the resulting embeddings.

    Parameters
    ----------
    logits : torch.Tensor
        J-lens vocabulary logits with shape
        ``(n_positions, vocabulary_size)``.
    tokenizer : Any
        Tokenizer implementing ``batch_decode`` for converting token IDs to
        text candidates.
    embedding_model : SentenceTransformer
        Sentence-transformer model used to embed decoded token candidates.

    Returns
    -------
    numpy.ndarray
        One-dimensional pooled feature vector with shape
        ``(embedding_size,)``.
    """
    # Shape [n_position, k]: retain top k concepts per position.
    top_k = logits.topk(settings.training.k, dim=-1)

    # Decode the flattened token candidates.
    decoded_candidates = tokenizer.batch_decode(
        top_k.indices.flatten().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Shape: [n_position, k, embedding_size].
    candidate_embeddings = embedding_model.encode(
        decoded_candidates,
        convert_to_tensor=True,
    ).reshape(*top_k.indices.shape, -1)

    pooled_features = _pool_candidate_embeddings(
        candidate_embeddings,
        top_k.values,
        decoded_candidates,
        embedding_model,
    )

    return pooled_features.to(_torch_device(settings.training.output_device)).cpu().numpy()


def _extract_features(
    texts: pd.Series[str],
    model: jlens.LensModel,
    lens: jlens.JacobianLens,
    embedding_model: SentenceTransformer,
) -> np.ndarray:
    """Extract pooled J-lens features for a collection of text samples.

    Each text is passed through the Jacobian lens at the configured target
    layer. The resulting logits are transformed into a single pooled semantic
    embedding using the configured top-k and pooling settings.

    Parameters
    ----------
    texts : pandas.Series[str]
        Series containing the input text for each dataset example.
    model : jlens.LensModel
        J-lens-compatible language model used for inference.
    lens : jlens.JacobianLens
        Pretrained Jacobian lens used to read out logits at the target layer.
    embedding_model : SentenceTransformer
        Model used to embed decoded top-k token candidates.

    Returns
    -------
    numpy.ndarray
        Feature matrix with shape ``(n_examples, embedding_size)``.
    """
    features = []
    target_layer = settings.training.target_layer

    for text in texts.astype(str):
        # Read the model's next-token predispositions at every token position.
        lens_logits_dict, _, _ = lens.apply(model, text, layers=[target_layer])

        features.append(
            _extract_embedding_features(
                lens_logits_dict[target_layer],
                model.tokenizer,
                embedding_model,
            )
        )

    return np.stack(features)


def _feature_frame(example_ids: np.ndarray, features: np.ndarray) -> pd.DataFrame:
    feature_columns = [f"feature_{index}" for index in range(features.shape[1])]
    feature_frame = pd.DataFrame(features, columns=feature_columns)
    feature_frame.insert(0, EXAMPLE_ID_COLUMN, example_ids)
    return feature_frame


def load_feature_dataset(feature_path: Path) -> pd.DataFrame:
    """Load a persisted feature dataset into its canonical tabular form."""
    match feature_path.suffix:
        case ".npz":
            with np.load(feature_path) as artifact:
                feature_frame = _feature_frame(artifact[EXAMPLE_ID_COLUMN], artifact["features"])
        case ".parquet":
            feature_frame = pd.read_parquet(feature_path)
        case _:
            raise ValueError(f"Unsupported feature dataset format: {feature_path.suffix}")

    if feature_frame.columns[0] != EXAMPLE_ID_COLUMN:
        raise ValueError("Feature dataset must have example_id as its first column")
    return feature_frame


def extract_jlens_features() -> None:
    """Extract and persist J-lens feature vectors for the generated dataset.

    The function validates alignment between the split manifest and generated
    dataset, loads the configured language model and J-lens checkpoint,
    extracts one pooled feature vector per text sample, and saves the output
    with stable example identifiers.

    Output format is controlled by
    ``settings.training.pre_processed_dataset_format``:

    - ``"NumPy"``: writes a compressed archive containing ``example_id`` and
      ``features`` arrays.
    - ``"Parquet"``: writes a table with one row per example, an ID column,
      and one column per embedding dimension.

    Raises
    ------
    ValueError
        If required columns are missing, example IDs are not aligned with the
        generated dataset, the model name is missing, or the J-lens checkpoint
        cannot be uniquely resolved.
    RuntimeError
        If an unavailable GPU is requested or the configured dataset format is
        unsupported.
    """
    manifest = pd.read_csv(SPLIT_MANIFEST_PATH)

    if EXAMPLE_ID_COLUMN not in manifest:
        raise ValueError(f"Split manifest is missing column: {EXAMPLE_ID_COLUMN}")

    ordered_manifest = manifest.sort_values(EXAMPLE_ID_COLUMN)

    example_ids = ordered_manifest[EXAMPLE_ID_COLUMN].to_numpy()

    model, lens = _load_jlens_model()

    logger.info("Loading embedding model {}", settings.training.embedding_model_name)

    embedding_model = SentenceTransformer(
        settings.training.embedding_model_name,
        device=str(_torch_device(settings.training.processing_device)),
    )

    logger.info("Extracting J-lens features for {} examples", len(ordered_manifest))

    features = _extract_features(ordered_manifest[TEXT_COLUMN], model, lens, embedding_model)
    feature_frame = _feature_frame(example_ids, features)

    match settings.training.pre_processed_dataset_format:
        case "NumPy":
            output_path = str(FEATURE_EXTRACTED_BASE_FILE_PATH_NO_EXT) + ".npz"

            np.savez(
                output_path,
                allow_pickle=True,
                **{
                    EXAMPLE_ID_COLUMN: feature_frame[EXAMPLE_ID_COLUMN].to_numpy(),
                    "features": feature_frame.iloc[:, 1:].to_numpy(),
                },
            )

            logger.info("Saved extracted features to {}", output_path)
        case "Parquet":
            output_path = str(FEATURE_EXTRACTED_BASE_FILE_PATH_NO_EXT) + ".parquet"

            feature_frame.to_parquet(
                output_path,
                index=False,
            )

            logger.info("Saved extracted features to {}", output_path)
