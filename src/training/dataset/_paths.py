"""Private module that holds some path constant definition"""

from pathlib import Path

from app.settings import settings

GENERATED_DATASET_PATH_ROOT = Path(__file__).resolve().parents[2]

GENERATED_DATASET_PATH = (
    GENERATED_DATASET_PATH_ROOT / f"{settings.output_dataset.name}.{settings.output_dataset.format}"
)

SPLIT_MANIFEST_PATH = GENERATED_DATASET_PATH_ROOT / "split_manifest.csv"

FEATURE_EXTRACTED_BASE_FILE_PATH_NO_EXT = GENERATED_DATASET_PATH_ROOT / "features"

JLENS_REPOSITORY = "neuronpedia/jacobian-lens"
