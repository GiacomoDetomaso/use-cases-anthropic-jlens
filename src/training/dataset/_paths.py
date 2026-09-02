"""Private module that holds some path constant definition"""

from pathlib import Path

from app.settings import settings

_GENERATED_DATASET_PATH_ROOT = Path(__file__).resolve().parents[2]

GENERATED_DATASET_PATH = (
    _GENERATED_DATASET_PATH_ROOT
    / f"{settings.output_dataset.name}.{settings.output_dataset.format}"
)

SPLITTED_DATASET_FOLDER = _GENERATED_DATASET_PATH_ROOT / "splits"

PRE_PROCESSED_DATASET_FOLDER = _GENERATED_DATASET_PATH_ROOT / "pre_processed"

JLENS_REPOSITORY = "neuronpedia/jacobian-lens"
